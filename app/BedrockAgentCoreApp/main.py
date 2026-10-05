import asyncio
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator

from agent_squad.agents import Agent as SquadAgent, AgentOptions
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions
from agent_squad.classifiers.classifier import ClassifierResult
from agent_squad.orchestrator import AgentSquad
from agent_squad.types import ConversationMessage, ParticipantRole
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp_client.client import get_streamable_http_mcp_client
from model.load import load_model
from strands import Agent, tool
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands_tools import calculator, current_time

from bylaws_retriever import search_company_bylaws
from stock_analyst import get_ast_press_releases, get_stock_quote

app = BedrockAgentCoreApp()
log = app.logger


# ==============================================================================
# 1. Custom Typed Business Tools
# ==============================================================================

@tool
def calculate_sla_deadline(days: int, start_date_iso: str = "") -> str:
    """
    Calculate an SLA resolution deadline excluding weekends (Saturday and Sunday).

    Args:
        days: Number of business days required for resolution (must be positive).
        start_date_iso: Optional ISO start date (YYYY-MM-DD). Defaults to today if empty.

    Returns:
        A string with the target completion date in 'YYYY-MM-DD (Day)' format.
    """
    if days <= 0:
        return "Error: SLA days must be a positive integer."

    try:
        curr = datetime.fromisoformat(start_date_iso) if start_date_iso else datetime.utcnow()
    except ValueError:
        return "Error: start_date_iso must be in YYYY-MM-DD format."

    added = 0
    while added < days:
        curr += timedelta(days=1)
        if curr.weekday() < 5:  # Monday to Friday
            added += 1

    return curr.strftime("%Y-%m-%d (%A)")


@tool
def count_keyword(text: str, keyword: str) -> int:
    """
    Count how many times a keyword appears in a chunk of text (case-insensitive).

    Args:
        text: The text to search.
        keyword: Case-insensitive keyword or phrase to count.

    Returns:
        Number of occurrences as an integer.
    """
    if not isinstance(text, str) or not isinstance(keyword, str):
        return 0

    clean_kw = keyword.strip().lower()
    if not clean_kw:
        return 0

    return text.lower().count(clean_kw)


# ==============================================================================
# 2. Specialized Sub-Agents (Strands SDK with Session Isolation)
# ==============================================================================

# Agent A: AST SpaceMobile Stock Analyst
stock_analyst_strands = Agent(
    model=load_model(),
    tools=[get_stock_quote, get_ast_press_releases, calculator],
    conversation_manager=NullConversationManager(),
    system_prompt=(
        "You are the AST SpaceMobile Stock Analyst agent. You specialize exclusively in "
        "AST SpaceMobile (NASDAQ: ASTS) equity research, market metrics, valuation, and official press releases. "
        "When asked about ASTS, always use get_stock_quote for live prices and get_ast_press_releases for official company updates. "
        "Structure your answers with an executive summary, live market data table, and key catalysts. "
        "Identify yourself as the AST SpaceMobile Stock Analyst."
    ),
)

# Agent B: Corporate Bylaws & Governance Specialist
bylaws_specialist_strands = Agent(
    model=load_model(),
    tools=[search_company_bylaws],
    conversation_manager=NullConversationManager(),
    system_prompt=(
        "You are the Corporate Bylaws & Document Specialist agent. You specialize in CFAS corporate governance, "
        "bylaws, board of directors rules, officer duties, emergency expenditures, quorum requirements, "
        "and any uploaded corporate documents and policies. "
        "Always use search_company_bylaws to retrieve verified excerpts from the knowledge base, "
        "and cite the exact Article, Section, or document headers. "
        "Identify yourself as the Corporate Bylaws & Document Specialist."
    ),
)

# Agent C: Operations, Compute & SLA Assistant
operations_assistant_strands = Agent(
    model=load_model(),
    tools=[calculator, current_time, count_keyword, calculate_sla_deadline],
    conversation_manager=NullConversationManager(),
    system_prompt=(
        "You are the Operations & General Assistant agent. You specialize in deterministic arithmetic math, "
        "current date and time queries, calculating business day SLAs, keyword frequency counts, and general conversation. "
        "Always use available tools for calculations, dates, and keyword counts rather than estimating. "
        "Identify yourself as the Operations & General Assistant."
    ),
)


# ==============================================================================
# 3. AgentSquad Adapter and Robust Router
# ==============================================================================

class StrandsAdapterAgent(SquadAgent):
    """Adapts a Strands SDK Agent to participate in AgentSquad routing."""

    def __init__(self, options: AgentOptions, strands_agent: Agent):
        super().__init__(options)
        self.strands_agent = strands_agent

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: Any,
        additional_params: Any = None,
    ) -> ConversationMessage:
        # Reconstruct session messages from AgentSquad's isolated chat history
        session_messages = []
        for m in (chat_history or []):
            role_str = "user" if m.role == ParticipantRole.USER.value else "assistant"
            blocks = []
            for b in (m.content or []):
                if isinstance(b, dict) and "text" in b:
                    blocks.append({"text": b["text"]})
                elif isinstance(b, str):
                    blocks.append({"text": b})
            if blocks:
                session_messages.append({"role": role_str, "content": blocks})

        self.strands_agent.messages = session_messages
        try:
            result = self.strands_agent(input_text)
        finally:
            self.strands_agent.messages = []

        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": str(result)}],
        )


def extract_text_from_message(msg: Any) -> str:
    """Safely extract human-readable text from a ConversationMessage."""
    if isinstance(msg, ConversationMessage):
        pieces = []
        for block in msg.content or []:
            if isinstance(block, dict) and "text" in block:
                pieces.append(block["text"])
        if pieces:
            return "".join(pieces)
        return str(msg)
    return str(msg)


class RobustBedrockClassifier(BedrockClassifier):
    """Classifier with forced tool choice and semantic keyword fallback."""

    async def process_request(self, input_text: str, chat_history: Any) -> ClassifierResult:
        user_message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": input_text}]
        )
        toolConfig = {
            "tools": self.tools,
            "toolChoice": {"tool": {"name": "analyzePrompt"}},
        }
        inference_config = {key: value for key, value in self.inference_config.items() if value is not None}
        converse_cmd = {
            "modelId": self.model_id,
            "messages": [{"role": user_message.role, "content": user_message.content}],
            "system": [{"text": self.system_prompt}],
            "toolConfig": toolConfig,
            "inferenceConfig": inference_config,
        }

        try:
            response = self.client.converse(**converse_cmd)
            for content_block in response.get("output", {}).get("message", {}).get("content", []):
                if "toolUse" in content_block:
                    tool_use = content_block["toolUse"]
                    agent_id = str(tool_use.get("input", {}).get("selected_agent", "")).lower()

                    selected = self.get_agent_by_id(agent_id)
                    if not selected:
                        for key, ag in self.agents.items():
                            if key in agent_id or ag.name.lower() in agent_id:
                                selected = ag
                                break

                    # Semantic keyword fallback if classifier chose 'unknown'
                    if not selected or agent_id == "unknown":
                        t = input_text.lower()
                        # Document / Uploaded file questions take precedence unless ASTS is explicitly mentioned
                        if any(w in t for w in ["rubric", "upload", "file", "document", "policy", "bylaw", "quorum", "board", "officer", "voting", "clause", "provision", "agreement", "stipend"]):
                            selected = self.agents.get("corporate-bylaws-specialist")
                        elif any(w in t for w in ["asts", "ast spacemobile", "stock", "share price", "52-week", "press release"]):
                            selected = self.agents.get("ast-spacemobile-stock-analyst")
                        else:
                            selected = self.agents.get("operations-general-assistant")

                    return ClassifierResult(
                        selected_agent=selected or list(self.agents.values())[-1],
                        confidence=float(tool_use.get("input", {}).get("confidence", 0.95)),
                    )

        except Exception as e:
            log.warning(f"Classifier error: {e}, falling back to semantic matcher.")

        # Fallback to general assistant
        t = input_text.lower()
        if any(w in t for w in ["rubric", "upload", "file", "document", "policy", "bylaw", "quorum", "board", "officer", "voting", "stipend"]):
            chosen = self.agents.get("corporate-bylaws-specialist")
        elif any(w in t for w in ["asts", "ast spacemobile", "stock", "share price", "press release"]):
            chosen = self.agents.get("ast-spacemobile-stock-analyst")
        else:
            chosen = self.agents.get("operations-general-assistant")

        return ClassifierResult(
            selected_agent=chosen or list(self.agents.values())[-1],
            confidence=0.85
        )


# Instantiate the Supervisor Router
classifier = RobustBedrockClassifier(
    BedrockClassifierOptions(
        model_id="amazon.nova-micro-v1:0",
        region="us-east-1",
        inference_config={
            "maxTokens": 300,
            "temperature": 0,
        },
    )
)

orchestrator = AgentSquad(classifier=classifier)

orchestrator.add_agent(
    StrandsAdapterAgent(
        AgentOptions(
            name="AST SpaceMobile Stock Analyst",
            description=(
                "Specializes in AST SpaceMobile (ASTS), stock market data, share prices, "
                "52-week pricing metrics, valuation, and official company press releases."
            ),
        ),
        strands_agent=stock_analyst_strands,
    )
)

orchestrator.add_agent(
    StrandsAdapterAgent(
        AgentOptions(
            name="Corporate Bylaws Specialist",
            description=(
                "Specializes in corporate governance, board of directors, board meetings, "
                "voting rules, quorum requirements, corporate bylaws, officer duties, and membership rules."
            ),
        ),
        strands_agent=bylaws_specialist_strands,
    )
)

orchestrator.add_agent(
    StrandsAdapterAgent(
        AgentOptions(
            name="Operations & General Assistant",
            description=(
                "Specializes in math, arithmetic calculations, calendar dates, business day SLA scheduling, "
                "keyword counting, general conversation, personal preferences, and general inquiries."
            ),
        ),
        strands_agent=operations_assistant_strands,
    )
)


# ==============================================================================
# 4. Message Parsing and Cloud Entrypoint
# ==============================================================================

def _extract_prompt(payload: dict) -> str:
    """Accepts validated harness messages, tool results, or a plain prompt string."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    if "messages" in payload:
        messages = payload["messages"]
        if isinstance(messages, list) and messages:
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and "text" in block:
                                return block["text"]
                    elif isinstance(content, str):
                        return content
        return str(messages)

    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return prompt


@app.entrypoint
async def invoke(payload: dict, context: Any) -> AsyncGenerator[dict, None]:
    """
    Asynchronous streaming entrypoint for Bedrock AgentCore Runtime.
    Routes queries to specialized agents via the AgentSquad Supervisor.
    """
    session_id = getattr(context, "session_id", "default-session")
    user_id = getattr(context, "user_id", "default-user")
    prompt = _extract_prompt(payload)

    log.info(f"AgentSquad routing prompt for user '{user_id}', session '{session_id}'...")

    response = await orchestrator.route_request(
        prompt,
        user_id=user_id,
        session_id=session_id,
    )

    chosen_agent = response.metadata.agent_name or "Operations & General Assistant"
    log.info(f"AgentSquad chose: {chosen_agent}")

    output_text = extract_text_from_message(response.output)
    full_output = f"[Routing: {chosen_agent}]\n\n{output_text}"

    yield {
        "event": {
            "contentBlockDelta": {
                "delta": {
                    "text": full_output
                }
            }
        }
    }


# Mount the FastAPI testing portal onto the BedrockAgentCore Starlette runtime
try:
    from server import app as portal_app
    app.mount("/portal", portal_app)
except Exception as _e:
    log.warning(f"Could not mount portal app: {_e}")


if __name__ == "__main__":
    app.run()
