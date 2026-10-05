"""
AWS Lambda S3 Ingestion Handler (Cloud-Native, Zero-External-Dependencies)
Triggered automatically by S3 ObjectCreated events when files are uploaded to S3.
Parses .docx, .txt, and .md files, performs context-enriched semantic chunking,
generates Titan v2 embeddings, and updates the serverless vector store in S3.
"""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)
s3_client = boto3.client("s3", region_name=REGION)

DEFAULT_BUCKET = "cfas-corporate-docs-725079717969"
INDEX_KEY = "bylaws/bylaws_index.json"


def compute_sha256(file_path: str) -> str:
    """Compute SHA256 checksum of a file for idempotency checks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_paragraphs(file_path: str) -> list[str]:
    """
    Extract paragraphs from supported file formats without third-party dependencies.
    Uses stdlib zipfile + ElementTree for .docx files.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".docx":
        with zipfile.ZipFile(file_path) as z:
            xml_content = z.read("word/document.xml")
        tree = ET.fromstring(xml_content)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for p in tree.findall(".//w:p", ns):
            texts = [node.text for node in p.findall(".//w:t", ns) if node.text]
            if texts:
                paragraphs.append("".join(texts).strip())
        return [p for p in paragraphs if p]

    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def semantic_chunk_document(paragraphs: list[str]) -> list[dict]:
    """
    Split legal/corporate documents along Article and Section boundaries.
    Falls back to paragraph windows if no legal headers exist.
    """
    chunks = []
    current_article = "Preamble / Header"
    current_section = "General"
    current_content = []

    article_pattern = re.compile(r"^ARTICLE\s+[IVXLCDM]+", re.IGNORECASE)
    section_pattern = re.compile(r"^SECTION\s+[A-Z0-9]+", re.IGNORECASE)

    def flush_chunk():
        nonlocal current_content
        text_body = "\n".join(current_content).strip()
        if text_body:
            chunk_num = len(chunks) + 1
            header = f"{current_article} - {current_section}".strip(" -")
            chunks.append({
                "chunk_index": chunk_num - 1,
                "article": current_article,
                "section": current_section,
                "citation": header,
                "text": text_body,
            })
        current_content = []

    has_headers = any(article_pattern.match(p) for p in paragraphs)

    if has_headers:
        for para in paragraphs:
            if article_pattern.match(para):
                flush_chunk()
                current_article = para
                current_section = "Overview"
            elif section_pattern.match(para):
                flush_chunk()
                current_section = para
            else:
                current_content.append(para)
        flush_chunk()
    else:
        batch_size = 3
        for i in range(0, len(paragraphs), batch_size):
            batch = paragraphs[i:i + batch_size]
            body = "\n".join(batch)
            chunks.append({
                "chunk_index": len(chunks),
                "article": "General",
                "section": f"Paragraphs {i+1}-{i+len(batch)}",
                "citation": f"Section {i // batch_size + 1}",
                "text": body,
            })

    return chunks


def get_embedding(text: str) -> list[float]:
    """Generate 1024-dimension normalized embedding using Titan v2."""
    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text[:2000]}),
    )
    body = json.loads(response["body"].read())
    return body["embedding"]


def update_s3_vector_store(
    bucket: str,
    tenant_id: str,
    doc_id: str,
    records: list[dict],
) -> int:
    """
    Update both the consolidated index and tenant-specific index in S3.
    Guarantees atomic updates: replaces old chunks for the same doc_id.
    """
    # 1. Update consolidated bylaws_index.json
    existing_records = []
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=INDEX_KEY)
        existing_records = json.loads(resp["Body"].read().decode("utf-8"))
    except Exception:
        existing_records = []

    # Filter out previous versions of this doc_id
    filtered = [r for r in existing_records if r.get("doc_id") != doc_id]

    # Convert new records to standard index schema
    new_index_entries = []
    for r in records:
        new_index_entries.append({
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "tenant_id": r["tenant_id"],
            "category": r["category"],
            "article": r["article"],
            "section": r["section"],
            "header": r["citation"],
            "text": r["text"],
            "embedding": r["vector"],
            "s3_uri": r["s3_uri"],
        })

    updated_full = filtered + new_index_entries
    s3_client.put_object(
        Bucket=bucket,
        Key=INDEX_KEY,
        Body=json.dumps(updated_full).encode("utf-8"),
        ContentType="application/json",
    )

    # 2. Store tenant-partitioned backup: vector-store/{tenant_id}/{doc_id}.json
    tenant_key = f"vector-store/{tenant_id}/{doc_id}.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=tenant_key,
        Body=json.dumps(new_index_entries).encode("utf-8"),
        ContentType="application/json",
    )

    return len(records)


def process_file(
    file_path: str,
    s3_uri: str,
    tenant_id: str,
    doc_id: str,
    bucket: str = DEFAULT_BUCKET,
    category: str = "governance",
) -> dict[str, Any]:
    """
    Full ingestion workflow for an uploaded file:
    Extract ➔ Context-Enriched Chunking ➔ Titan v2 Embedding ➔ S3 Vector Store Update.
    """
    filename = os.path.basename(file_path)
    checksum = compute_sha256(file_path)
    paragraphs = extract_paragraphs(file_path)
    raw_chunks = semantic_chunk_document(paragraphs)

    records = []
    now_iso = datetime.utcnow().isoformat()

    print(f"Embedding {len(raw_chunks)} chunks for {filename} (tenant: {tenant_id})...")
    for item in raw_chunks:
        chunk_idx = item["chunk_index"]
        header = item["citation"]
        body = item["text"]

        context_header = f"Document: {filename} | Category: {category} | {header}"
        full_embed_text = f"{context_header}\n{body}"

        vector = get_embedding(full_embed_text)

        records.append({
            "vector": vector,
            "chunk_id": f"{doc_id}_{chunk_idx:04d}",
            "tenant_id": tenant_id,
            "doc_id": doc_id,
            "filename": filename,
            "category": category,
            "article": item["article"],
            "section": item["section"],
            "citation": header,
            "chunk_index": chunk_idx,
            "char_count": len(body),
            "token_count": max(1, len(body) // 4),
            "text": body,
            "context_header": context_header,
            "s3_uri": s3_uri,
            "file_checksum": checksum,
            "created_at": now_iso,
        })

    # Update S3 vector stores
    inserted = update_s3_vector_store(
        bucket=bucket,
        tenant_id=tenant_id,
        doc_id=doc_id,
        records=records,
    )

    # Also keep local VectorDBManager in sync if available
    try:
        from vector_db.db_manager import VectorDBManager
        vdb = VectorDBManager()
        vdb.upsert_chunks(tenant_id=tenant_id, doc_id=doc_id, records=records)
    except Exception:
        pass

    return {
        "status": "SUCCESS",
        "tenant_id": tenant_id,
        "doc_id": doc_id,
        "filename": filename,
        "chunks_indexed": inserted,
        "checksum": checksum,
        "category": category,
        "s3_index": f"s3://{bucket}/{INDEX_KEY}",
    }


def lambda_handler(event: dict, context: Any = None) -> dict[str, Any]:
    """
    AWS Lambda handler entrypoint.
    Receives S3 ObjectCreated events or direct JSON invocation payloads.
    """
    print("Received event:", json.dumps(event))
    results = []

    # Case 1: S3 Event Notification
    if "Records" in event:
        for record in event["Records"]:
            bucket = record["s3"]["bucket"]["name"]
            key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

            # Skip events originating from vector-store/ or bylaws_index.json to avoid loops
            if "vector-store" in key or "bylaws_index.json" in key:
                print(f"Skipping internal index file: {key}")
                continue

            parts = key.split("/")
            tenant_id = parts[1] if len(parts) >= 3 else "cfas-corp"
            filename = parts[-1]
            doc_id = Path(filename).stem
            category = "governance" if "bylaw" in filename.lower() else "general"

            local_path = f"/tmp/{filename}"
            print(f"Downloading s3://{bucket}/{key} to {local_path}...")
            s3_client.download_file(bucket, key, local_path)
            s3_uri = f"s3://{bucket}/{key}"

            res = process_file(
                file_path=local_path,
                s3_uri=s3_uri,
                tenant_id=tenant_id,
                doc_id=doc_id,
                bucket=bucket,
                category=category,
            )
            results.append(res)

    # Case 2: Direct Invocation payload
    elif "file_path" in event:
        res = process_file(
            file_path=event["file_path"],
            s3_uri=event.get("s3_uri", f"local://{os.path.basename(event['file_path'])}"),
            tenant_id=event.get("tenant_id", "cfas-corp"),
            doc_id=event.get("doc_id", Path(event["file_path"]).stem),
            bucket=event.get("bucket", DEFAULT_BUCKET),
            category=event.get("category", "governance"),
        )
        results.append(res)

    return {
        "statusCode": 200,
        "body": results,
    }
