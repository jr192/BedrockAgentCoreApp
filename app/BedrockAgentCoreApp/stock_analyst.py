import html
import requests
from strands import Agent, tool
from strands_tools import calculator, current_time


@tool
def get_ast_press_releases(limit: int = 5) -> str:
    """
    Fetch the latest official press releases from the AST SpaceMobile investor relations feed
    (https://investors.ast-science.com/press-releases).

    Args:
        limit: Number of recent press releases to fetch (default: 5, max: 15).

    Returns:
        Formatted string containing publication date, headline, summary, and direct links.
    """
    safe_limit = max(1, min(limit, 15))
    url = "https://www.accesswire.com/qm/data/getHeadlines.json"
    params = {
        "topics": "ASTS",
        "excludeTopics": "NONCOMPANY",
        "noSrc": "qmr",
        "src": "pzo,bayaw,prn,bwi,TheNewsWire,nfil,actw,irw,acn,cnw,nwd,glpr,nwmw",
        "summary": "true",
        "summLen": "500",
        "limit": safe_limit,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            return f"Error: Received HTTP {resp.status_code} while fetching press releases."

        data = resp.json()
        news_groups = data.get("results", {}).get("news", [])
        if not news_groups or not news_groups[0].get("newsitem"):
            return "No recent press releases found for AST SpaceMobile."

        releases = []
        for item in news_groups[0]["newsitem"][:safe_limit]:
            date_str = item.get("datetime", "")[:10]
            headline = html.unescape(item.get("headline", "Untitled"))
            raw_summary = item.get("qmsummary", "").strip()
            clean_summary = html.unescape(" ".join(raw_summary.split()))
            link = item.get("storyurl") or item.get("permalink") or "https://investors.ast-science.com/press-releases"

            releases.append(
                f"• Date: {date_str}\n"
                f"  Headline: {headline}\n"
                f"  Summary: {clean_summary}\n"
                f"  Link: {link}"
            )

        header = f"=== AST SpaceMobile Press Releases (Source: investors.ast-science.com) ===\n"
        return header + "\n\n".join(releases)

    except requests.RequestException as e:
        return f"Network error fetching press releases: {e}"
    except Exception as e:
        return f"Unexpected error processing press releases: {e}"


@tool
def get_stock_quote(ticker: str = "ASTS") -> str:
    """
    Fetch the latest live equity market data for a stock ticker (e.g. ASTS).

    Args:
        ticker: Stock ticker symbol (default: ASTS).

    Returns:
        Summary of regular market price, 52-week high/low, day high/low, and currency.
    """
    clean_ticker = ticker.strip().upper()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{clean_ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return f"Error: Could not retrieve market data for {clean_ticker} (HTTP {resp.status_code})."

        data = resp.json()
        result = data.get("chart", {}).get("result")
        if not result:
            return f"Error: No data available for symbol '{clean_ticker}'."

        meta = result[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        high_52 = meta.get("fiftyTwoWeekHigh")
        low_52 = meta.get("fiftyTwoWeekLow")
        day_high = meta.get("regularMarketDayHigh")
        day_low = meta.get("regularMarketDayLow")
        currency = meta.get("currency", "USD")

        return (
            f"=== Market Data for {clean_ticker} ===\n"
            f"Current Price: {price} {currency}\n"
            f"Day Range: {day_low} - {day_high} {currency}\n"
            f"52-Week Range: {low_52} - {high_52} {currency}"
        )
    except Exception as e:
        return f"Failed to retrieve quote for {clean_ticker}: {e}"


# Senior Equity Research Analyst System Persona
STOCK_ANALYST_SYSTEM_PROMPT = """
You are a Senior Equity Research Analyst specializing in space technology, telecommunications, and emerging aerospace growth companies, with primary focus on AST SpaceMobile (NASDAQ: ASTS).

Your Mandate:
1. Provide data-driven, objective, and professional investment analysis.
2. When asked about recent developments, launches, commercial contracts, regulatory filings, or partnerships, always use the `get_ast_press_releases` tool to retrieve live, official releases directly from the AST investor relations feed.
3. When discussing pricing, valuation, or range, use `get_stock_quote` for live prices and `calculator` for precise financial math (e.g. market caps, percentages, multiples).
4. Never speculate or fabricate news items. Always cite release dates and official headlines when discussing events.
5. Structure your research notes clearly with:
   - Executive Summary
   - Key Catalysts & Recent Press Release Insights
   - Valuation & Market Context
   - Key Risks & Considerations
"""

# Assemble Stock Analyst Agent
stock_analyst_agent = Agent(
    tools=[
        get_ast_press_releases,  # Reads latest AST press releases from investor relations
        get_stock_quote,         # Real-time stock market data
        calculator,              # Deterministic financial math
        current_time,            # Live date & time anchor
    ],
    system_prompt=STOCK_ANALYST_SYSTEM_PROMPT,
)


if __name__ == "__main__":
    print("=== AST SpaceMobile Stock Analyst Agent ===")
    prompt = (
        "Please read the latest press releases from AST SpaceMobile (ASTS), check the current stock price, "
        "and provide a concise briefing on recent company milestones, upcoming launches, and key partnerships."
    )
    print(f"\nAnalyzing: {prompt}\n")
    response = stock_analyst_agent(prompt)
    print("\n=== ANALYST BRIEFING ===")
    print(response)
