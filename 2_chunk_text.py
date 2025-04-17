import json
import sys
import os
import re
from typing import List, Dict

from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

# Load embedding model globally
model = SentenceTransformer("all-MiniLM-L6-v2")


def combine_page_text_and_tables(page_data: Dict) -> str:
    """
    Combine the main text with table data so we don't lose table info.
    """
    base_text = page_data.get("text", "")
    tables_str_list = []

    # Flatten table rows into strings
    for tbl in page_data.get("tables", []):
        for row in tbl["rows"]:
            header_str = (row.get("header") or "").strip()
            content_list = row.get("content", [])
            content_text = "; ".join(content_list)
            row_text = f"{header_str}: {content_text}" if header_str else content_text
            tables_str_list.append(row_text)

    if tables_str_list:
        tables_combined = "\n".join(tables_str_list)
        base_text += "\n\n[TABLE DATA]\n" + tables_combined + "\n"

    return base_text


def split_into_sentences(text: str) -> List[str]:
    return sent_tokenize(text.strip())


def semantic_chunk_sentences(sentences: List[str], chunk_size: int) -> List[str]:
    """
    Group sentences based on semantic similarity and word count.
    """
    if not sentences:
        return []

    embeddings = model.encode(sentences, convert_to_tensor=True)
    chunks = []
    current_chunk = []
    current_len = 0

    for i, sentence in enumerate(sentences):
        word_count = len(sentence.split())

        if not current_chunk:
            current_chunk.append(sentence)
            current_len += word_count
            continue

        similarity = util.cos_sim(embeddings[i], embeddings[i - 1]).item()

        if similarity > 0.6 or current_len < chunk_size:
            current_chunk.append(sentence)
            current_len += word_count
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def semantic_chunk_text(pages_content: List[Dict], source_pdf: str, chunk_size: int = 200) -> List[Dict]:
    """
    Perform semantic (embedding-based) chunking of page data.
    """
    all_chunks = []
    chunk_id_counter = 0

    for page_data in pages_content:
        page_num = page_data["page_number"]
        combined_text = combine_page_text_and_tables(page_data)

        sentences = split_into_sentences(combined_text)
        semantic_chunks = semantic_chunk_sentences(sentences, chunk_size)

        for chunk_text in semantic_chunks:
            chunk_record = {
                "chunk_id": f"chunk_{chunk_id_counter}",
                "source_pdf": source_pdf,
                "page_number": page_num,
                "text": chunk_text
            }
            all_chunks.append(chunk_record)
            chunk_id_counter += 1

    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 2_chunk_text_multi.py <input_json1> <input_json2> ... <output_json_chunks> [chunk_size]")
        sys.exit(1)

    # *input_jsons, output_json = sys.argv[1:-1], sys.argv[-1]
    # print(sys.argv)

    # # Optional chunk size
    # chunk_size = 200
    # if len(sys.argv) > len(input_jsons) + 2:
    #     chunk_size = int(sys.argv[-1])
    #     output_json = sys.argv[-2]

    try:
        chunk_size = int(sys.argv[-1])
        input_jsons, output_json = sys.argv[1:-2], sys.argv[-2]
    except:
        chunk_size = 200 # default
        input_jsons, output_json = sys.argv[1:-1], sys.argv[-1]

    all_chunks = []
    for input_json in input_jsons:
        print(input_json)
        if not os.path.exists(input_json):
            print(f"File not found: {input_json}")
            continue

        with open(input_json, "r", encoding="utf-8") as f:
            pages_content = json.load(f)

        source_pdf = os.path.basename(input_json).replace("_output.json", ".pdf")

        chunks = semantic_chunk_text(pages_content, source_pdf, chunk_size=chunk_size)
        print(f"Created {len(chunks)} chunks from {input_json}.")
        all_chunks.extend(chunks)

    print(f"Total combined chunks: {len(all_chunks)}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved all chunks to {output_json}.")
