"""
Vector Database Schema and Metadata Definition
Defines the columnar Apache Arrow schema for LanceDB / Serverless Vector Store.
"""

import pyarrow as pa

# 1024-dimensional embedding vector produced by amazon.titan-embed-text-v2:0
EMBEDDING_DIM = 1024

DOCUMENT_CHUNKS_SCHEMA = pa.schema([
    # Vector column for ANN (Approximate Nearest Neighbors) search
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM), nullable=False),

    # Primary & Multi-tenant Partitioning Keys
    pa.field("chunk_id", pa.string(), nullable=False),       # Unique ID: {doc_id}_{chunk_index:04d}
    pa.field("tenant_id", pa.string(), nullable=False),      # Multi-tenant isolation key (e.g. 'cfas-corp')
    pa.field("doc_id", pa.string(), nullable=False),         # Document identifier (e.g. 'cfas-bylaws-rev-9')
    pa.field("filename", pa.string(), nullable=False),       # Original uploaded file name
    pa.field("category", pa.string(), nullable=False),       # High-level domain: 'governance', 'finance', 'technical'

    # Hierarchical Document Metadata (for precise legal/corporate citations)
    pa.field("article", pa.string(), nullable=True),         # E.g. 'ARTICLE VIII - MEETINGS'
    pa.field("section", pa.string(), nullable=True),         # E.g. 'Overview' or 'SECTION B'
    pa.field("citation", pa.string(), nullable=False),       # E.g. 'ARTICLE VIII - MEETINGS - Overview'
    pa.field("chunk_index", pa.int32(), nullable=False),     # Position in document (0, 1, 2...)
    pa.field("char_count", pa.int32(), nullable=False),      # Character length of text
    pa.field("token_count", pa.int32(), nullable=False),     # Estimated token count (~chars / 4)

    # Content
    pa.field("text", pa.string(), nullable=False),           # Chunk body text
    pa.field("context_header", pa.string(), nullable=False), # Context-enriched prefix prepended during embedding

    # Provenance and Lifecycle Tracking
    pa.field("s3_uri", pa.string(), nullable=False),         # s3://bucket/key
    pa.field("file_checksum", pa.string(), nullable=False),  # SHA256 of source file for idempotency
    pa.field("created_at", pa.string(), nullable=False),     # ISO timestamp
])
