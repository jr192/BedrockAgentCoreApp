"""
Test Suite for FastAPI Document Ingestion & Multi-Agent Portal
Tests authentication gates, file uploads, S3 vector indexing, and live agent chat querying.
"""

import io
import json
import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure sys.path includes app and project root
_root = Path(__file__).resolve().parent.parent
_app_dir = _root / "app" / "BedrockAgentCoreApp"
for p in [str(_root), str(_app_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.BedrockAgentCoreApp.server import app, INTERVIEW_SECRET

client = TestClient(app)


def test_health_endpoint():
    """Verify public health endpoint returns 200 and system details."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "cfas-agent-ingestion-portal" in data["service"]
    assert len(data["agents"]) == 3


def test_authentication_gate():
    """Verify unauthorized requests are blocked with HTTP 401."""
    # 1. No token provided
    resp_no_token = client.get("/api/documents")
    assert resp_no_token.status_code == 401
    assert "Unauthorized" in resp_no_token.json()["detail"]

    # 2. Invalid token provided
    resp_bad_token = client.get("/api/documents?token=wrong-secret-key")
    assert resp_bad_token.status_code == 401

    # 3. Valid token provided via query parameter
    resp_valid = client.get(f"/api/documents?token={INTERVIEW_SECRET}")
    assert resp_valid.status_code == 200
    data = resp_valid.json()
    assert data["status"] == "SUCCESS"
    assert "documents" in data


def test_web_portal_ui():
    """Verify the executive HTML interface is served with 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert "CFAS Multi-Agent System" in response.text
    assert "Upload File to S3" in response.text


def test_file_upload_and_indexing():
    """Verify uploading a custom document to S3 and vector store indexing."""
    sample_text = (
        "CFAS REMOTE WORK AND EXPENSE REIMBURSEMENT POLICY (REV 2026)\n"
        "SECTION 1: EQUIPMENT ALLOWANCE\n"
        "All full-time engineering employees are eligible for a one-time equipment stipend of $2,500 USD.\n"
        "Reimbursements must be submitted within 30 calendar days of purchase.\n\n"
        "SECTION 2: TRAVEL AUTHORIZATION\n"
        "Travel exceeding $1,000 USD must receive written pre-approval from the Vice President of Engineering.\n"
        "Emergency travel approvals can be expedited within 4 hours by the Chief Operating Officer."
    )

    file_bytes = io.BytesIO(sample_text.encode("utf-8"))
    files = {"file": ("cfas-remote-policy-2026.txt", file_bytes, "text/plain")}
    data = {"tenant_id": "cfas-corp", "category": "policy"}

    # 1. Reject unauthorized upload
    resp_unauth = client.post("/api/upload", files=files, data=data)
    assert resp_unauth.status_code == 401

    # 2. Accept authorized upload
    file_bytes.seek(0)
    resp_auth = client.post(
        f"/api/upload?token={INTERVIEW_SECRET}",
        files=files,
        data=data,
    )
    assert resp_auth.status_code == 200
    result = resp_auth.json()
    assert result["status"] == "SUCCESS"
    assert result["filename"] == "cfas-remote-policy-2026.txt"
    assert result["chunks_indexed"] > 0
    assert "s3://" in result["s3_uri"]
    print(f"\n[Test] Indexed {result['chunks_indexed']} chunks to S3: {result['s3_uri']}")


def test_query_agent_about_uploaded_document():
    """Verify that asking the agent about the uploaded document routes and cites correctly."""
    query = "According to the remote policy, what is the equipment stipend amount for full-time engineering employees?"
    
    resp = client.post(
        f"/api/chat?token={INTERVIEW_SECRET}",
        json={"prompt": query, "session_id": "test-interviewer-session-01"},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["status"] == "SUCCESS"
    assert "Document Specialist" in result["agent"] or "Bylaw" in result["agent"]
    assert "2,500" in result["response"] or "stipend" in result["response"].lower()
    print(f"\n[Test] Routed to: {result['agent']}")
    print(f"[Test] Agent response: {result['response']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
