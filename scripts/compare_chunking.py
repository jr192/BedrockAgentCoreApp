"""
Phase 4 - Task 4.1: Chunk Corporate Document Corpus
Comparison of Fixed-size Chunking vs Semantic Chunking on `cfas-bylaws-rev-9.docx`
"""

import os
import re
from pathlib import Path
import docx


def extract_text_from_docx(file_path: str) -> list[str]:
    """Extract non-empty paragraphs from a .docx file."""
    doc = docx.Document(file_path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def fixed_chunking(full_text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Splits text blindly into fixed windows of characters with a sliding overlap."""
    chunks = []
    start = 0
    total_len = len(full_text)
    step = chunk_size - overlap

    chunk_id = 1
    while start < total_len:
        end = min(start + chunk_size, total_len)
        chunk_content = full_text[start:end]
        chunks.append({
            "chunk_id": f"fixed_{chunk_id:03d}",
            "start_idx": start,
            "end_idx": end,
            "char_count": len(chunk_content),
            "text": chunk_content,
        })
        if end >= total_len:
            break
        start += step
        chunk_id += 1

    return chunks


def semantic_chunking(paragraphs: list[str]) -> list[dict]:
    """Segments legal text along logical Article & Section boundaries."""
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
            chunks.append({
                "chunk_id": f"semantic_{chunk_num:03d}",
                "article": current_article,
                "section": current_section,
                "header": f"{current_article} - {current_section}".strip(" -"),
                "char_count": len(text_body),
                "text": text_body,
            })
        current_content = []

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
    return chunks


def compare_strategies(docx_path: str):
    if not os.path.exists(docx_path):
        print(f"Error: Document not found at {docx_path}")
        return

    print("=" * 75)
    print(f"📄 DOCUMENT: {os.path.basename(docx_path)}")
    print("=" * 75)

    paragraphs = extract_text_from_docx(docx_path)
    full_text = "\n\n".join(paragraphs)
    print(f"Total Paragraphs: {len(paragraphs)}")
    print(f"Total Characters: {len(full_text):,}\n")

    fixed_chunks = fixed_chunking(full_text, chunk_size=500, overlap=50)
    sem_chunks = semantic_chunking(paragraphs)

    print("📊 1. METRICS COMPARISON")
    print("-" * 75)
    print(f"{'Metric':<30} | {'Fixed (500 chars, 50 ovlp)':<25} | {'Semantic (Article/Section)':<25}")
    print("-" * 75)
    print(f"{'Total Chunks Created':<30} | {len(fixed_chunks):<25} | {len(sem_chunks):<25}")

    fixed_lengths = [c["char_count"] for c in fixed_chunks]
    sem_lengths = [c["char_count"] for c in sem_chunks]

    print(f"{'Average Chunk Size (chars)':<30} | {sum(fixed_lengths)//len(fixed_lengths):<25} | {sum(sem_lengths)//len(sem_lengths):<25}")
    print(f"{'Min Chunk Size (chars)':<30} | {min(fixed_lengths):<25} | {min(sem_lengths):<25}")
    print(f"{'Max Chunk Size (chars)':<30} | {max(fixed_lengths):<25} | {max(sem_lengths):<25}")
    print("-" * 75)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    doc_candidates = [
        base_dir.parent / "cfas-bylaws-rev-9.docx",
        base_dir.parent.parent / "cfas-bylaws-rev-9.docx",
        Path("/Users/joaorodrigues/Desktop/Dev/Curso AI/AIP-Materials/cfas-bylaws-rev-9.docx"),
    ]
    target_file = None
    for cand in doc_candidates:
        if cand.exists():
            target_file = str(cand)
            break

    if target_file:
        compare_strategies(target_file)
    else:
        print("Could not find cfas-bylaws-rev-9.docx in candidate paths.")
