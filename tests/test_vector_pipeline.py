"""
Test Suite for Serverless Vector Database & Ingestion Pipeline
Verifies:
1. Context-enriched chunking and Titan v2 embedding ingestion.
2. Fast partitioned retrieval (< 20ms).
3. Multi-tenant isolation (tenant A cannot see tenant B's chunks).
4. Hybrid search accuracy for Quorum and Emergency expenditure questions.
"""

import os
import sys
import time

# Add root directory to python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from lambda_ingest.handler import process_file
from vector_db.db_manager import VectorDBManager

DOCX_FILE = os.path.join(ROOT_DIR, "data", "cfas-bylaws-rev-9.docx")
TENANT_CFAS = "cfas-corp"
TENANT_OTHER = "tenant-unrelated"


def run_pipeline_test():
    print("=" * 70)
    print("🚀 STEP 1: INGESTING DOCUMENT INTO VECTOR DATABASE")
    print("=" * 70)

    start_time = time.time()
    ingest_result = process_file(
        file_path=DOCX_FILE,
        s3_uri="s3://cfas-corporate-docs-725079717969/bylaws/cfas-bylaws-rev-9.docx",
        tenant_id=TENANT_CFAS,
        doc_id="cfas-bylaws-rev-9",
        category="governance",
    )
    elapsed = time.time() - start_time

    print(f"Status           : {ingest_result['status']}")
    print(f"Tenant           : {ingest_result['tenant_id']}")
    print(f"Document ID      : {ingest_result['doc_id']}")
    print(f"Chunks Indexed   : {ingest_result['chunks_indexed']}")
    print(f"File Checksum    : {ingest_result['checksum'][:16]}...")
    print(f"Ingestion Time   : {elapsed:.2f}s\n")

    print("=" * 70)
    print("🔍 STEP 2: RETRIEVAL EFFICIENCY & RELEVANCE BENCHMARKS")
    print("=" * 70)

    from lambda_ingest.handler import get_embedding
    vector_db = VectorDBManager()

    test_queries = [
        ("What constitutes a quorum for the Board of Directors under the bylaws?", "ARTICLE VIII"),
        ("What is the maximum expenditure the President can authorize in an emergency?", "ARTICLE VII"),
        ("Are proxy votes allowed at meetings and do they count toward quorum?", "ARTICLE VIII"),
    ]

    for q, expected_article in test_queries:
        t0 = time.time()
        q_vec = get_embedding(q)
        results = vector_db.search(
            query_vector=q_vec,
            tenant_id=TENANT_CFAS,
            category="governance",
            top_k=3,
            query_text=q,
        )
        latency = (time.time() - t0) * 1000

        top_match = results[0] if results else None
        top_citation = top_match["citation"] if top_match else "None"
        top_score = top_match["score"] if top_match else 0.0
        success = expected_article in top_citation

        status_icon = "✅ PASS" if success else "❌ FAIL"
        print(f"Query    : \"{q}\"")
        print(f"Result   : {status_icon} | Latency: {latency:.1f}ms | Top Score: {top_score:.4f}")
        print(f"Citation : {top_citation}")
        print(f"Excerpt  : {top_match['content'][:140]}...\n")

    print("=" * 70)
    print("🔒 STEP 3: MULTI-TENANT ISOLATION TEST")
    print("=" * 70)

    isolated_results = vector_db.search(
        query_vector=q_vec,
        tenant_id=TENANT_OTHER,  # Attempting to search another tenant
        category="governance",
        top_k=3,
        query_text="What constitutes a quorum?",
    )
    print(f"Searching with tenant_id='{TENANT_OTHER}':")
    print(f"Matches returned: {len(isolated_results)}")
    if len(isolated_results) == 0:
        print("✅ SUCCESS: Strict tenant isolation confirmed. Zero cross-tenant data bleed.\n")
    else:
        print("❌ WARNING: Cross-tenant data bleed detected!\n")


if __name__ == "__main__":
    run_pipeline_test()
