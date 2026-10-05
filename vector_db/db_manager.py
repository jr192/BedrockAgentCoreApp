"""
Vector Database Manager
Provides efficient storage, partitioning, upserting, and filtered hybrid retrieval.
Supports both local file-system storage (for development) and direct S3 URIs (for AWS Lambda).
"""

import os
from typing import Any, Optional
import lancedb
from vector_db.schema import DOCUMENT_CHUNKS_SCHEMA


class VectorDBManager:
    """Serverless Vector Database manager built on top of LanceDB / Arrow."""

    def __init__(self, db_uri: Optional[str] = None):
        # Default to local path or S3 URI from environment
        self.db_uri = db_uri or os.environ.get(
            "VECTOR_DB_URI",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store")
        )
        self.db = lancedb.connect(self.db_uri)
        self.table_name = "document_chunks"

    def get_or_create_table(self):
        """Get existing chunks table or initialize with standard schema."""
        existing = self.db.table_names()
        if self.table_name in existing:
            return self.db.open_table(self.table_name)
        return self.db.create_table(self.table_name, schema=DOCUMENT_CHUNKS_SCHEMA)

    def upsert_chunks(self, tenant_id: str, doc_id: str, records: list[dict]) -> int:
        """
        Idempotently insert chunks for a document.
        Removes any previous version of the document for this tenant first.
        """
        table = self.get_or_create_table()

        # Delete old chunks for this document if they exist
        try:
            filter_expr = f"tenant_id = '{tenant_id}' AND doc_id = '{doc_id}'"
            table.delete(filter_expr)
        except Exception:
            pass  # Table might be empty or delete not needed

        if not records:
            return 0

        table.add(records)
        return len(records)

    def delete_document(self, tenant_id: str, doc_id: str) -> None:
        """Prune all chunks belonging to a deleted document."""
        table = self.get_or_create_table()
        filter_expr = f"tenant_id = '{tenant_id}' AND doc_id = '{doc_id}'"
        table.delete(filter_expr)

    def search(
        self,
        query_vector: list[float],
        tenant_id: str,
        category: Optional[str] = None,
        doc_id: Optional[str] = None,
        top_k: int = 3,
        query_text: Optional[str] = None,
    ) -> list[dict]:
        """
        Search with partition filtering (tenant, category, doc_id) and optional hybrid reranking.

        Args:
            query_vector: 1024-dim embedding of the user's query.
            tenant_id: Strict isolation key (e.g. 'cfas-corp').
            category: Optional domain filter ('governance', 'finance', etc.).
            doc_id: Optional single-document scope.
            top_k: Number of results to return.
            query_text: Optional raw text for exact keyword scoring boost.
        """
        table = self.get_or_create_table()

        # Build SQL filter for partition pruning
        filters = [f"tenant_id = '{tenant_id}'"]
        if category:
            filters.append(f"category = '{category}'")
        if doc_id:
            filters.append(f"doc_id = '{doc_id}'")
        where_clause = " AND ".join(filters)

        # Vector search bounded by partition filter
        query = table.search(query_vector).where(where_clause).limit(top_k * 2)
        raw_results = query.to_list()

        if not raw_results:
            return []

        # Hybrid scoring boost: combine vector distance with keyword matches
        scored_results = []
        query_terms = [w.lower() for w in (query_text or "").split() if len(w) > 3]

        for item in raw_results:
            # LanceDB returns _distance (L2 or cosine distance; lower is closer, or dot product)
            # Normalize to similarity score
            distance = item.get("_distance", 1.0)
            similarity = 1.0 / (1.0 + distance)

            kw_matches = 0
            if query_terms:
                body_lower = (item.get("text") or "").lower()
                header_lower = (item.get("citation") or "").lower()
                kw_matches = sum(1 for term in query_terms if term in body_lower or term in header_lower)

            # 80% vector similarity + 20% exact keyword presence
            hybrid_score = round(similarity + (min(0.2, kw_matches * 0.05)), 4)

            scored_results.append({
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "tenant_id": item.get("tenant_id"),
                "category": item.get("category"),
                "citation": item.get("citation"),
                "article": item.get("article"),
                "section": item.get("section"),
                "score": hybrid_score,
                "content": item.get("text"),
                "source": item.get("s3_uri"),
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]
