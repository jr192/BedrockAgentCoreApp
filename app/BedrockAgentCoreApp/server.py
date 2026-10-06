"""
FastAPI Enterprise Upload & Testing Portal for Bedrock AgentCore Multi-Agent System
Enables recruiters, interviewers, and authorized users to upload custom documents to S3,
triggers serverless vector chunking and Titan v2 embedding, and queries the multi-agent squad.
Secured via private access token verification.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Optional
import urllib.parse
import uuid

import boto3
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import sys

# Ensure both local app dir and project root are in sys.path
_app_dir = str(Path(__file__).resolve().parent)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
for p in [_app_dir, _project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from bylaws_retriever import reload_index, search_company_documents
    from lambda_ingest.handler import process_file
    from main import evaluate_guardrails, extract_text_from_message, orchestrator
except ImportError:
    from app.BedrockAgentCoreApp.bylaws_retriever import reload_index, search_company_documents
    from lambda_ingest.handler import process_file
    from app.BedrockAgentCoreApp.main import evaluate_guardrails, extract_text_from_message, orchestrator

# AWS Environment & Configuration
REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("DOCS_S3_BUCKET", "cfas-corporate-docs-725079717969")
INTERVIEW_SECRET = os.environ.get("INTERVIEW_SECRET", "cfas-agent-demo-2026")

s3_client = boto3.client("s3", region_name=REGION)

app = FastAPI(
    title="CFAS Multi-Agent Evaluation & Ingestion Portal",
    description="Private executive testing endpoint for AWS Bedrock Multi-Agent architecture.",
    version="1.0.0",
)

# Enable CORS for external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Security & Token Verification Dependency
# ==============================================================================

def verify_token(
    token: Optional[str] = Query(None),
    key: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> str:
    """
    Enforces private access for authorized interviewers and recruiters.
    Accepts token via ?token=..., ?key=..., X-API-Key header, or Bearer auth.
    """
    provided_key = token or key or x_api_key

    if not provided_key and authorization:
        parts = authorization.split(" ")
        if len(parts) == 2 and parts[0].lower() == "bearer":
            provided_key = parts[1]

    if provided_key != INTERVIEW_SECRET:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Access to this private evaluation portal requires a valid access key.",
        )
    return provided_key


# ==============================================================================
# Request & Response Models
# ==============================================================================

class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "interviewer-rh"


class ChatResponse(BaseModel):
    status: str
    agent: str
    routing_badge: str
    response: str
    session_id: str


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/api/health")
def health_check():
    """Public healthcheck probe for container and load balancer monitoring."""
    return {
        "status": "HEALTHY",
        "service": "cfas-agent-ingestion-portal",
        "region": REGION,
        "s3_bucket": S3_BUCKET,
        "agents": [
            "João Rodrigues - Senior SE & GenAI Specialist",
            "AST SpaceMobile Stock Analyst",
            "Corporate Bylaws & Document Specialist",
            "Operations & General Assistant",
        ],
        "guardrail": {
            "id": "dxes1svttuw8",
            "name": "Enterprise-Compliance-Guardrail",
            "status": "ACTIVE",
            "policies": [
                "INSULTS (HIGH)",
                "HATE (HIGH)",
                "MISCONDUCT (HIGH)",
                "PROMPT_ATTACK (HIGH)",
                "CandidateDefamationAndAbuse (DENY)",
            ],
        },
    }


@app.get("/api/documents")
def list_documents(_: str = Depends(verify_token)):
    """List all indexed documents currently stored in the S3 vector store."""
    try:
        resp = s3_client.get_object(Bucket=S3_BUCKET, Key="bylaws/bylaws_index.json")
        chunks = json.loads(resp["Body"].read().decode("utf-8"))
    except Exception:
        chunks = []

    doc_map: dict[str, dict] = {}
    for c in chunks:
        doc_id = c.get("doc_id", "cfas-bylaws-rev-9")
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "doc_id": doc_id,
                "filename": c.get("filename", f"{doc_id}.docx"),
                "category": c.get("category", "governance"),
                "tenant_id": c.get("tenant_id", "cfas-corp"),
                "s3_uri": c.get("s3_uri", f"s3://{S3_BUCKET}/bylaws/cfas-bylaws-rev-9.docx"),
                "chunk_count": 0,
            }
        doc_map[doc_id]["chunk_count"] += 1

    return {
        "status": "SUCCESS",
        "total_documents": len(doc_map),
        "total_chunks": len(chunks),
        "documents": list(doc_map.values()),
    }


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    tenant_id: str = Form("cfas-corp"),
    category: str = Form("general"),
    _: str = Depends(verify_token),
):
    """
    POST Endpoint for Custom File Ingestion:
    1. Validates file format (.docx, .txt, .md).
    2. Uploads raw document directly to S3: s3://<bucket>/uploads/<tenant>/<filename>.
    3. Triggers automated semantic chunking and Titan v2 embedding (1024-dim).
    4. Updates the serverless S3 vector store atomically.
    5. Refreshes the in-memory retriever cache so the agent can immediately cite the file.
    """
    filename = file.filename or f"upload_{uuid.uuid4().hex[:8]}.txt"
    ext = Path(filename).suffix.lower()

    if ext not in [".docx", ".txt", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: .docx, .txt, .md",
        )

    # Save to temporary local file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # 1. Upload raw file to S3
        s3_key = f"uploads/{tenant_id}/{filename}"
        s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
        print(f"Uploading file to {s3_uri}...")

        s3_client.upload_file(
            tmp_path,
            S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": file.content_type or "application/octet-stream"},
        )

        # 2. Run the chunking and embedding pipeline
        doc_id = Path(filename).stem
        ingest_result = process_file(
            file_path=tmp_path,
            s3_uri=s3_uri,
            tenant_id=tenant_id,
            doc_id=doc_id,
            bucket=S3_BUCKET,
            category=category,
        )

        # 3. Invalidate retriever cache so new chunks are immediately searchable
        reload_index()

        suggested_questions = [
            f"Summarize the main provisions of {filename}.",
            f"What does {filename} state regarding key responsibilities or rules?",
            f"What are the specific requirements or numbers cited in {filename}?",
        ]

        return {
            "status": "SUCCESS",
            "message": f"Successfully uploaded and indexed '{filename}' ({ingest_result['chunks_indexed']} chunks).",
            "filename": filename,
            "doc_id": doc_id,
            "tenant_id": tenant_id,
            "s3_uri": s3_uri,
            "chunks_indexed": ingest_result["chunks_indexed"],
            "checksum": ingest_result["checksum"],
            "category": category,
            "suggested_questions": suggested_questions,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(
    payload: ChatRequest,
    _: str = Depends(verify_token),
):
    """
    POST Endpoint for Agent Interaction:
    Sends user prompts to the AgentSquad Supervisor.
    Routes to ASTS Stock Analyst, Corporate Bylaws/Document Specialist, or Operations Assistant.
    """
    session_id = payload.session_id or f"session-{uuid.uuid4().hex[:12]}"
    user_id = payload.user_id or "interviewer-rh"

    # 1. AWS Bedrock Guardrail Gate
    gr_result = evaluate_guardrails(payload.prompt)
    if gr_result["intervened"]:
        return ChatResponse(
            status="SUCCESS",
            agent="Enterprise Guardrail Interception",
            routing_badge="[Guardrail: Intervened]",
            response=f"[Guardrail: Intervened]\n\n🛡️ **Enterprise Safety Interception**\n\n{gr_result['message']}",
            session_id=session_id,
        )

    try:
        response = await orchestrator.route_request(
            payload.prompt,
            user_id=user_id,
            session_id=session_id,
        )

        chosen_agent = response.metadata.agent_name or "Operations & General Assistant"
        output_text = extract_text_from_message(response.output)

        return ChatResponse(
            status="SUCCESS",
            agent=chosen_agent,
            routing_badge=f"[Routing: {chosen_agent}]",
            response=output_text,
            session_id=session_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


# ==============================================================================
# Web UI Endpoint (Executive Dark-Mode Glassmorphism Interface)
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
def index_page():
    """
    Renders an executive interactive testing portal for recruiters & interviewers.
    Includes Drag-and-Drop file ingestion to S3, real-time chunking feedback,
    and live multi-agent chat with routing badges and citations.
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CFAS Multi-Agent System | Enterprise Evaluation Portal</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0d14;
      --card-bg: rgba(18, 24, 38, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-blue: #3b82f6;
      --accent-indigo: #6366f1;
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.12) 0px, transparent 50%);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* Auth Gate Overlay */
    #authModal {
      position: fixed;
      inset: 0;
      background: rgba(10, 13, 20, 0.85);
      backdrop-filter: blur(12px);
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .auth-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 36px;
      width: 90%;
      max-width: 440px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
      text-align: center;
    }

    .auth-card h2 {
      font-family: 'Outfit', sans-serif;
      font-size: 24px;
      margin-bottom: 8px;
    }

    .auth-card p {
      color: var(--text-muted);
      font-size: 14px;
      margin-bottom: 24px;
    }

    .auth-input {
      width: 100%;
      padding: 12px 16px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      color: #fff;
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      margin-bottom: 16px;
      outline: none;
      transition: border-color 0.2s;
    }

    .auth-input:focus { border-color: var(--accent-indigo); }

    .btn-primary {
      width: 100%;
      padding: 12px 20px;
      background: linear-gradient(135deg, var(--accent-indigo), var(--accent-blue));
      border: none;
      border-radius: 8px;
      color: #fff;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
    }

    .btn-primary:hover { opacity: 0.95; transform: translateY(-1px); }

    /* Top Navigation */
    header {
      padding: 18px 32px;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      backdrop-filter: blur(8px);
    }

    .logo-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .logo-badge {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: linear-gradient(135deg, #6366f1, #3b82f6);
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'Outfit', sans-serif;
      font-weight: 800;
      font-size: 18px;
    }

    .logo-title h1 {
      font-family: 'Outfit', sans-serif;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .logo-title span {
      font-size: 12px;
      color: var(--text-muted);
    }

    .header-pills {
      display: flex;
      gap: 10px;
    }

    .pill {
      font-size: 11px;
      font-weight: 600;
      padding: 6px 12px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .pill-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent-emerald);
      box-shadow: 0 0 8px var(--accent-emerald);
    }

    /* Main Grid Layout */
    main {
      flex: 1;
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 24px;
      padding: 24px 32px;
      max-width: 1600px;
      margin: 0 auto;
      width: 100%;
    }

    @media (max-width: 1024px) {
      main { grid-template-columns: 1fr; }
    }

    /* Left Sidebar: Ingestion Studio */
    .ingest-panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }

    .panel-title {
      font-family: 'Outfit', sans-serif;
      font-size: 16px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .dropzone {
      border: 2px dashed rgba(99, 102, 241, 0.4);
      border-radius: 12px;
      padding: 32px 16px;
      text-align: center;
      cursor: pointer;
      background: rgba(99, 102, 241, 0.03);
      transition: all 0.2s ease;
    }

    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent-cyan);
      background: rgba(6, 182, 212, 0.06);
    }

    .dropzone svg {
      width: 42px;
      height: 42px;
      margin-bottom: 12px;
      fill: none;
      stroke: var(--accent-indigo);
      stroke-width: 2;
    }

    .dropzone p {
      font-size: 13px;
      font-weight: 500;
      margin-bottom: 4px;
    }

    .dropzone span {
      font-size: 11px;
      color: var(--text-muted);
    }

    .file-input { display: none; }

    .selected-file-box {
      display: none;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px;
      font-size: 13px;
    }

    .progress-bar-container {
      display: none;
      height: 6px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 3px;
      overflow: hidden;
      margin-top: 8px;
    }

    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent-indigo), var(--accent-cyan));
      transition: width 0.3s;
    }

    .docs-list-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-top: 8px;
    }

    .docs-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 240px;
      overflow-y: auto;
    }

    .doc-item {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .doc-item span.name {
      font-family: 'JetBrains Mono', monospace;
      color: #e2e8f0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 240px;
    }

    .doc-item span.chunks {
      font-size: 11px;
      color: var(--accent-cyan);
      background: rgba(6, 182, 212, 0.1);
      padding: 2px 8px;
      border-radius: 10px;
    }

    /* Right Panel: Chat Console */
    .chat-panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }

    .chat-header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .chat-messages {
      flex: 1;
      padding: 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-height: calc(100vh - 300px);
    }

    .message {
      max-width: 85%;
      padding: 14px 18px;
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.5;
    }

    .message.user {
      align-self: flex-end;
      background: linear-gradient(135deg, #4f46e5, #3b82f6);
      color: #fff;
      border-bottom-right-radius: 2px;
    }

    .message.agent {
      align-self: flex-start;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      border-bottom-left-radius: 2px;
    }

    .badge-routing {
      display: inline-block;
      font-size: 11px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 12px;
      margin-bottom: 8px;
    }

    .badge-guardrail {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .badge-candidate {
      background: rgba(168, 85, 247, 0.15);
      color: #c084fc;
      border: 1px solid rgba(168, 85, 247, 0.3);
    }

    .badge-asts {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-bylaws {
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .badge-ops {
      background: rgba(245, 158, 11, 0.15);
      color: #fcd34d;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .quick-chips {
      padding: 8px 24px;
      display: flex;
      gap: 8px;
      overflow-x: auto;
      border-top: 1px solid var(--card-border);
    }

    .chip {
      font-size: 12px;
      padding: 6px 14px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s;
    }

    .chip:hover {
      color: #fff;
      border-color: var(--accent-indigo);
      background: rgba(99, 102, 241, 0.1);
    }

    .chat-input-bar {
      padding: 16px 24px;
      border-top: 1px solid var(--card-border);
      display: flex;
      gap: 12px;
    }

    .chat-input {
      flex: 1;
      padding: 12px 18px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      color: #fff;
      font-size: 14px;
      outline: none;
    }

    .chat-input:focus { border-color: var(--accent-indigo); }

    .btn-send {
      padding: 12px 24px;
      background: linear-gradient(135deg, var(--accent-indigo), var(--accent-blue));
      border: none;
      border-radius: 10px;
      color: #fff;
      font-weight: 600;
      cursor: pointer;
    }

    .btn-send:disabled { opacity: 0.5; cursor: not-allowed; }
  </style>
</head>
<body>

  <!-- Auth Key Gate Modal -->
  <div id="authModal">
    <div class="auth-card">
      <div class="logo-badge" style="margin: 0 auto 16px auto;">C</div>
      <h2>Private Testing Portal</h2>
      <p>Enter the private interview access token to test document uploads and the autonomous multi-agent flow.</p>
      <input type="password" id="keyInput" class="auth-input" placeholder="Enter Access Key..." value="cfas-agent-demo-2026">
      <button class="btn-primary" onclick="submitAuth()">Unlock Portal</button>
    </div>
  </div>

  <!-- Header -->
  <header>
    <div class="logo-group">
      <div class="logo-badge">C</div>
      <div class="logo-title">
        <h1>CFAS Autonomous Multi-Agent Squad</h1>
        <span>AWS Bedrock AgentCore • Serverless Vector DB • Strands SDK</span>
      </div>
    </div>
    <div class="header-pills">
      <div class="pill"><div class="pill-dot"></div> S3 Ingestion Active</div>
      <div class="pill">Region: us-east-1</div>
      <div class="pill" id="authPill">🔒 Authenticated</div>
    </div>
  </header>

  <!-- Main Body -->
  <main>
    <!-- Left: Ingestion Studio -->
    <div class="ingest-panel">
      <div class="panel-title">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        Upload File to S3 & Index
      </div>

      <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
        <svg viewBox="0 0 24 24"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M12 12v9"/><path d="m16 16-4-4-4 4"/></svg>
        <p>Drop file here or click to browse</p>
        <span>Supports .docx, .txt, .md (Auto-Titan v2 Embedded)</span>
        <input type="file" id="fileInput" class="file-input" accept=".docx,.txt,.md" onchange="handleFileSelected(event)">
      </div>

      <div id="fileInfoBox" class="selected-file-box">
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
          <strong id="fileNameDisplay">filename.docx</strong>
          <span id="fileSizeDisplay" style="color:var(--text-muted);">12 KB</span>
        </div>
        <button id="uploadBtn" class="btn-primary" onclick="performUpload()">Upload to S3 & Process</button>
        <div class="progress-bar-container" id="progressContainer">
          <div class="progress-bar" id="progressBar"></div>
        </div>
        <div id="uploadStatusText" style="font-size:11px; color:var(--accent-cyan); margin-top:6px;"></div>
      </div>

      <div class="docs-list-title">Currently Indexed Documents in S3</div>
      <div class="docs-list" id="docsList">
        <div style="color:var(--text-muted); font-size:12px;">Loading indexed documents...</div>
      </div>
    </div>

    <!-- Right: Chat Console -->
    <div class="chat-panel">
      <div class="chat-header">
        <div style="display:flex; align-items:center; gap:8px;">
          <div class="pill-dot"></div>
          <span style="font-weight:600; font-size:14px;">Autonomous Supervisor Stream</span>
        </div>
        <span style="font-size:12px; color:var(--text-muted);">Supervisor: Nova Micro | Agents: Strands 1.50</span>
      </div>

      <div class="chat-messages" id="chatMessages">
        <div class="message agent">
          <div class="badge-routing badge-candidate">[Routing: João Rodrigues - Senior SE & GenAI Specialist]</div>
          <div>Welcome! I am the multi-agent system. You can explore <strong>João Rodrigues's career profile & GenAI/SE experience</strong>, upload and index custom documents to S3 on the left, query corporate bylaws, or get live ASTS market intelligence!</div>
        </div>
      </div>

      <div class="quick-chips">
        <div class="chip" onclick="setPrompt('Tell me about João Rodrigues background and experience as a Senior Software & GenAI Engineer.')">👨‍💻 João's Bio & Experience</div>
        <div class="chip" onclick="setPrompt('What Generative AI projects, RAG pipelines, and model fine-tuning has João built at Euronext?')">🚀 Euronext GenAI Projects</div>
        <div class="chip" onclick="setPrompt('Describe João experience handling millions of requests per minute at FanDuel during the Super Bowl.')">🏈 FanDuel Super Bowl Scale</div>
        <div class="chip" onclick="setPrompt('Why is João Rodrigues the ideal candidate for a Senior Software Engineer and GenAI role?')">🎯 Why Hire João?</div>
        <div class="chip" onclick="setPrompt('What is ASTS current stock price and 52-week high?')">📈 ASTS Stock Quote</div>
        <div class="chip" onclick="setPrompt('What constitutes a quorum for the Board of Directors under the bylaws?')">🏛️ Quorum Rules</div>
      </div>

      <div class="chat-input-bar">
        <input type="text" id="promptInput" class="chat-input" placeholder="Ask a question about the uploaded document or market..." onkeydown="if(event.key==='Enter') sendMessage()">
        <button id="sendBtn" class="btn-send" onclick="sendMessage()">Send</button>
      </div>
    </div>
  </main>

  <script>
    let activeToken = "";
    let selectedFile = null;
    let sessionId = "session-" + Math.random().toString(36).substring(2, 10);

    // Read token from URL query or localStorage
    const urlParams = new URLSearchParams(window.location.search);
    const queryKey = urlParams.get('token') || urlParams.get('key');
    if (queryKey) {
      activeToken = queryKey;
      localStorage.setItem('cfas_portal_token', activeToken);
    } else {
      activeToken = localStorage.getItem('cfas_portal_token') || "";
    }

    if (activeToken) {
      document.getElementById('authModal').style.display = 'none';
      loadDocuments();
    }

    function submitAuth() {
      const val = document.getElementById('keyInput').value.trim();
      if (!val) return;
      activeToken = val;
      localStorage.setItem('cfas_portal_token', activeToken);
      document.getElementById('authModal').style.display = 'none';
      loadDocuments();
    }

    function handleFileSelected(e) {
      const file = e.target.files[0];
      if (!file) return;
      selectedFile = file;
      document.getElementById('fileNameDisplay').textContent = file.name;
      document.getElementById('fileSizeDisplay').textContent = (file.size / 1024).toFixed(1) + " KB";
      document.getElementById('fileInfoBox').style.display = 'block';
    }

    async function performUpload() {
      if (!selectedFile) return;
      const btn = document.getElementById('uploadBtn');
      const pContainer = document.getElementById('progressContainer');
      const pBar = document.getElementById('progressBar');
      const statusText = document.getElementById('uploadStatusText');

      btn.disabled = true;
      btn.textContent = "Processing...";
      pContainer.style.display = 'block';
      pBar.style.width = '30%';
      statusText.textContent = "Uploading to S3 bucket & generating Titan v2 embeddings...";

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("tenant_id", "cfas-corp");

      try {
        const resp = await fetch(`/api/upload?token=${encodeURIComponent(activeToken)}`, {
          method: "POST",
          body: formData,
        });

        pBar.style.width = '80%';
        const data = await resp.json();

        if (!resp.ok) {
          throw new Error(data.detail || "Upload failed");
        }

        pBar.style.width = '100%';
        statusText.textContent = `✓ Uploaded to S3 and indexed ${data.chunks_indexed} chunks!`;
        btn.textContent = "Upload Complete!";
        
        // Notify chat
        appendMessage('agent', `[Routing: Corporate Bylaws & Document Specialist]`, 
          `I have indexed **${data.filename}** into the S3 vector store (${data.chunks_indexed} chunks with Titan v2 embeddings). You can now ask me any questions about it!`
        );

        loadDocuments();
      } catch (err) {
        statusText.textContent = `Error: ${err.message}`;
        btn.disabled = false;
        btn.textContent = "Retry Upload";
      }
    }

    async function loadDocuments() {
      try {
        const resp = await fetch(`/api/documents?token=${encodeURIComponent(activeToken)}`);
        if (!resp.ok) return;
        const data = await resp.json();
        const listEl = document.getElementById('docsList');
        listEl.innerHTML = "";

        data.documents.forEach(doc => {
          const item = document.createElement('div');
          item.className = 'doc-item';
          item.innerHTML = `
            <span class="name" title="${doc.filename}">${doc.filename}</span>
            <span class="chunks">${doc.chunk_count} chunks</span>
          `;
          listEl.appendChild(item);
        });
      } catch (e) {
        console.error("Failed to load documents", e);
      }
    }

    function setPrompt(text) {
      document.getElementById('promptInput').value = text;
      sendMessage();
    }

    async function sendMessage() {
      const input = document.getElementById('promptInput');
      const text = input.value.trim();
      if (!text) return;

      appendMessage('user', '', text);
      input.value = "";

      const sendBtn = document.getElementById('sendBtn');
      sendBtn.disabled = true;
      sendBtn.textContent = "Thinking...";

      try {
        const resp = await fetch(`/api/chat?token=${encodeURIComponent(activeToken)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: text, session_id: sessionId })
        });

        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "Request failed");

        appendMessage('agent', data.routing_badge, data.response);
      } catch (err) {
        appendMessage('agent', '[Routing Error]', `Error executing request: ${err.message}`);
      } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = "Send";
      }
    }

    function appendMessage(role, badge, text) {
      const container = document.getElementById('chatMessages');
      const div = document.createElement('div');
      div.className = `message ${role}`;

      if (badge) {
        const badgeDiv = document.createElement('div');
        let badgeClass = 'badge-ops';
        if (badge.includes('Guardrail')) badgeClass = 'badge-guardrail';
        else if (badge.includes('João') || badge.includes('SE & GenAI') || badge.includes('Rodrigues')) badgeClass = 'badge-candidate';
        else if (badge.includes('Stock')) badgeClass = 'badge-asts';
        else if (badge.includes('Bylaw') || badge.includes('Document')) badgeClass = 'badge-bylaws';
        badgeDiv.className = `badge-routing ${badgeClass}`;
        badgeDiv.textContent = badge;
        div.appendChild(badgeDiv);
      }

      const bodyDiv = document.createElement('div');
      bodyDiv.innerHTML = text.replace(/\\n/g, '<br>').replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
      div.appendChild(bodyDiv);

      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting CFAS Private Upload & Evaluation Portal on port {port}...")
    print(f"Access URL: http://localhost:{port}/?token={INTERVIEW_SECRET}")
    uvicorn.run(app, host="0.0.0.0", port=port)
