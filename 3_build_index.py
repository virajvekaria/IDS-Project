# 3_build_index.py

import json
import sys
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

def load_chunks(chunks_json: str) -> List[Dict]:
    with open(chunks_json, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    return chunks_data

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python 3_build_index.py <chunks_json> <model_name> <output_index_prefix>")
        sys.exit(1)

    chunks_json = sys.argv[1]
    model_name = sys.argv[2]
    index_prefix = sys.argv[3]

    # 1) Load chunk data
    chunks_data = load_chunks(chunks_json)
    texts = [chunk["text"] for chunk in chunks_data]

    # 2) Load embedding model
    model = SentenceTransformer(model_name)
    # If using "all-MiniLM-L6-v2", dimension is 384
    # If using "multi-qa-mpnet-base-dot-v1", dimension is 768, etc.
    # We'll dynamically figure out dimension from 1 embedding:
    sample_emb = model.encode(["test"], convert_to_numpy=True)
    embedding_dim = sample_emb.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(embedding_dim)

    # 3) Embed chunks
    print("Embedding chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print("Finished embeddings.")

    # 4) Add to FAISS
    index.add(embeddings)

    # 5) Save index to disk
    faiss.write_index(index, f"{index_prefix}.index")
    print(f"FAISS index saved to {index_prefix}.index")

    # 6) Save chunk metadata separately
    with open(f"{index_prefix}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    print(f"Chunk metadata saved to {index_prefix}_metadata.json")
