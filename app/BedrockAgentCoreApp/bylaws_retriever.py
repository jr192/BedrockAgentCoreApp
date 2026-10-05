"""
Corporate Bylaws Semantic Retrieval Tool (Bedrock AgentCore Application)
Uses lightweight, zero-idle-cost serverless semantic retrieval (Option 1).
Reads pre-computed Titan v2 embeddings from local bylaws_index.json or S3,
eliminating the need for an expensive 24/7 OpenSearch Serverless cluster.
"""

import json
import os
from pathlib import Path
import boto3
from strands import tool

REGION = os.environ.get("AWS_REGION", "us-east-1")
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)
s3_client = boto3.client("s3", region_name=REGION)

S3_BUCKET = "cfas-corporate-docs-725079717969"
S3_KEY = "bylaws/bylaws_index.json"
SOURCE_URI = "s3://cfas-corporate-docs-725079717969/bylaws/cfas-bylaws-rev-9.docx"

_INDEX_CACHE: list[dict] = []

try:
    from vector_db.db_manager import VectorDBManager
    _VECTOR_DB = VectorDBManager()
except Exception as e:
    _VECTOR_DB = None


def _load_bylaws_index() -> list[dict]:
    """Load the pre-computed bylaws semantic chunk index."""
    global _INDEX_CACHE
    if _INDEX_CACHE:
        return _INDEX_CACHE

    # 1. Try local file in current directory or parent
    candidates = [
        Path(__file__).resolve().parent / "bylaws_index.json",
        Path(__file__).resolve().parent.parent / "bylaws_index.json",
        Path("/Users/joaorodrigues/Desktop/Dev/Curso AI/AIP-Materials/BedrockAgentCoreApp/app/BedrockAgentCoreApp/bylaws_index.json"),
    ]
    for cand in candidates:
        if cand.exists():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    _INDEX_CACHE = json.load(f)
                    return _INDEX_CACHE
            except Exception as e:
                print(f"Warning: Failed to load local {cand}: {e}")

    # 2. Try fetching from S3
    try:
        resp = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        _INDEX_CACHE = json.loads(resp["Body"].read().decode("utf-8"))
        return _INDEX_CACHE
    except Exception as e:
        print(f"Warning: Could not fetch bylaws_index.json from S3: {e}")

    return []


def reload_index() -> list[dict]:
    """Force reloads the index cache from S3 or local files."""
    global _INDEX_CACHE
    _INDEX_CACHE = []
    return _load_bylaws_index()


@tool
def search_company_documents(query: str) -> list[dict]:
    """
    Search corporate bylaws and uploaded company documents.
    Retrieves verified excerpts, legal citations, policies, and governance
    rules from all indexed documents in the S3 vector store.

    Args:
        query: Semantic question (e.g. 'What constitutes a quorum?' or 'Summarize uploaded document').

    Returns:
        Verified excerpts with citations, confidence scores, and source links.
    """
    # 1. Primary: High-performance partitioned Vector Database
    if _VECTOR_DB is not None:
        try:
            resp = bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": query[:2000]}),
            )
            q_vec = json.loads(resp["body"].read())["embedding"]
            results = _VECTOR_DB.search(
                query_vector=q_vec,
                tenant_id="cfas-corp",
                category=None,  # Search across all document categories
                top_k=4,
                query_text=query,
            )
            if results:
                return results
        except Exception as e:
            print(f"Warning: VectorDB search error: {e}. Falling back to pre-computed index.")

    # 2. Fallback: Pre-computed index file
    chunks = _load_bylaws_index()
    if not chunks:
        return [{"error": "Document index could not be loaded."}]

    try:
        # Embed the query with Titan v2 (costs ~$0.0000002)
        resp = bedrock_runtime.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": query[:2000]}),
        )
        body = json.loads(resp["body"].read())
        q_vec = body["embedding"]

        # Dot product similarity (embeddings from Titan v2 are normalized)
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        scored = []
        for c in chunks:
            dot = sum(x * y for x, y in zip(q_vec, c["embedding"]))
            # Keyword relevance boost for exact matches in header/text/filename
            c_text_lower = c.get("text", "").lower()
            c_header_lower = c.get("header", "").lower()
            c_file_lower = c.get("filename", "").lower()
            kw_matches = sum(1 for w in query_words if w in c_text_lower or w in c_header_lower or w in c_file_lower)
            kw_bonus = min(0.10, kw_matches * 0.02)
            total_score = dot + kw_bonus

            scored.append({
                "citation": c.get("header") or c.get("citation", "General Section"),
                "article": c.get("article", "General"),
                "section": c.get("section", ""),
                "score": round(float(total_score), 4),
                "content": c.get("text", ""),
                "filename": c.get("filename", "cfas-bylaws-rev-9.docx"),
                "source": c.get("s3_uri") or SOURCE_URI,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:4]

    except Exception as e:
        return [{"error": f"Failed to retrieve document excerpts: {str(e)}"}]


# Alias for backward compatibility with existing agent definitions
search_company_bylaws = search_company_documents
