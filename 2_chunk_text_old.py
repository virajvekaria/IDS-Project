# 2_chunk_text.py

import json
import sys
import re
from typing import List, Dict

def combine_page_text_and_tables(page_data: Dict) -> str:
    """
    Combine the main text with table data so we don't lose table info.
    """
    base_text = page_data.get("text", "")
    tables_str_list = []

    # If the page_data has 'tables', flatten them into strings
    for tbl in page_data.get("tables", []):
        for row in tbl["rows"]:
            header_str = (row.get("header") or "").strip()
            content_list = row.get("content", [])
            # Combine bullet/line items with a semicolon or newline
            content_text = "; ".join(content_list)
            # Build a single string for this row
            if header_str:
                row_text = f"{header_str}: {content_text}"
            else:
                row_text = content_text
            tables_str_list.append(row_text)

    if tables_str_list:
        tables_combined = "\n".join(tables_str_list)
        # Append to base text
        base_text += "\n\n[TABLE DATA]\n" + tables_combined + "\n"

    return base_text

def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs based on blank lines or double newlines.
    This is a simple regex approach.
    """
    paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

def split_paragraph_into_chunks(paragraph: str, max_words: int) -> List[str]:
    """
    If a paragraph exceeds max_words, split it into sub-chunks.
    """
    words = paragraph.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        sub_chunk = " ".join(words[start:end])
        chunks.append(sub_chunk)
        start = end
    return chunks

def semantic_chunk_text(
    pages_content: List[Dict],
    chunk_size: int = 200
) -> List[Dict]:
    """
    Perform semantic (paragraph-based) chunking.

    1. Combine page text + table data
    2. Split into paragraphs
    3. If a paragraph is too big (> chunk_size words), split further
    4. Return a list of chunks with page_number + text
    """
    all_chunks = []
    chunk_id_counter = 0

    for page_data in pages_content:
        page_num = page_data["page_number"]

        # Combine text + table data
        combined_text = combine_page_text_and_tables(page_data)

        # Split into paragraphs
        paragraphs = split_into_paragraphs(combined_text)

        # For each paragraph, if it's too large, further split
        for paragraph in paragraphs:
            # Count words
            word_count = len(paragraph.split())
            if word_count <= chunk_size:
                # It's already small enough to be a chunk
                chunk_record = {
                    "chunk_id": f"chunk_{chunk_id_counter}",
                    "page_number": page_num,
                    "text": paragraph
                }
                all_chunks.append(chunk_record)
                chunk_id_counter += 1
            else:
                # Split big paragraph
                sub_chunks = split_paragraph_into_chunks(paragraph, chunk_size)
                for sub_chunk in sub_chunks:
                    chunk_record = {
                        "chunk_id": f"chunk_{chunk_id_counter}",
                        "page_number": page_num,
                        "text": sub_chunk
                    }
                    all_chunks.append(chunk_record)
                    chunk_id_counter += 1

    return all_chunks

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 2_chunk_text.py <input_json_pages> <output_json_chunks> [chunk_size]")
        sys.exit(1)

    input_json = sys.argv[1]
    output_json = sys.argv[2]

    # If user provides a chunk_size, use it
    chunk_size = 200
    if len(sys.argv) >= 4:
        chunk_size = int(sys.argv[3])

    with open(input_json, "r", encoding="utf-8") as f:
        pages_content = json.load(f)

    chunks = semantic_chunk_text(pages_content, chunk_size=chunk_size)
    print(f"Created {len(chunks)} chunks from {len(pages_content)} page entries.")

    # Save chunks to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunked data to {output_json}.")
