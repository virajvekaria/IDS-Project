# 4_query_qa.py

import sys
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import ollama

def load_index(index_path: str):
    index = faiss.read_index(index_path)
    return index

def load_metadata(metadata_path: str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# def generate_llm_answer(question: str, retrieved_docs: list, model_name: str) -> str:
#     """
#     Example function that uses a local Hugging Face model to generate an answer
#     from the retrieved documents.

#     1. Loads the model + tokenizer from 'model_name'.
#     2. Creates a prompt that includes the retrieved docs as context.
#     3. Generates text.
#     4. Returns the model's answer.

#     Note: This can be resource-intensive for large models. Test with smaller ones first.
#     """

#     if not retrieved_docs:
#         return "No relevant information found in the document."

#     # Build the context from top retrieved docs
#     context_text = ""
#     for doc in retrieved_docs:
#         context_text += f"(Page {doc['page_number']}): {doc['text']}\n\n"

#     # Create a system-like instruction or a direct prompt
#     # Encourage the model to cite the pages in its answer.
#     prompt = (
#         f"Context:\n{context_text}"
#         f"Question: {question}\n\n"
#         f"Please provide a concise, direct answer using only the above context. "
#         f"Cite the page numbers when possible.\n\nAnswer:"
#     )

#     # Load model/tokenizer and create pipeline
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

#     # Generate the answer
#     output = generator(
#         prompt,
#         max_new_tokens=200,   # GOOD: allows generating up to 200 tokens after prompt
#         do_sample=True,
#         top_p=0.9,
#         temperature=0.3
#     )


#     # 'output' is a list of dicts [{'generated_text': "..."}]
#     generated_text = output[0]["generated_text"]

#     # Often the model will echo the prompt back, so you might want
#     # to strip out the prompt from the final answer if needed.
#     # A simple approach is to remove everything up to "Answer:" 
#     # and keep the rest, or parse carefully.
#     answer_part = generated_text.split("Answer:")[-1].strip()

#     return answer_part

def generate_llm_answer(question: str, retrieved_docs: list, model_name: str) -> str:
    if not retrieved_docs:
        return "No relevant information found in the document."

    # Build context from retrieved docs
    context_text = ""
    for doc in retrieved_docs:
        context_text += f"(Page {doc['page_number']}): {doc['text']}\n\n"

    # Final prompt
    prompt = (
        f"You are a helpful assistant answering questions using DOE documents.\n\n"
        f"Context (extracted from the DOE documents):\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Guidelines for your answer:\n"
        f"- Only use information from the context.\n"
        f"- Provide a concise and clear answer.\n"
        f"- Cite page numbers explicitly in parentheses like this: (Page X).\n"
        f"- If multiple ideas come from different pages, cite them separately.\n"
        f"- Do not invent information not in the context.\n\n"
        f"Answer:"
    )


    # Generate answer using Ollama
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}  ])

    return response['message']['content'].strip()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python 4_query_qa.py <index_file> <metadata_file> <model_name> <query_text>")
        sys.exit(1)

    index_file = sys.argv[1]
    metadata_file = sys.argv[2]
    model_name = sys.argv[3]
    query_text = sys.argv[4]

    # 1) Load FAISS index
    index = load_index(index_file)

    # 2) Load chunk metadata
    metadata = load_metadata(metadata_file)

    # 3) Load embedding model for query
    # embedder = SentenceTransformer(model_name) 
    embedding_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    embedder = SentenceTransformer(embedding_model)

    # (You could use a different model for embeddings vs. generation if you like.)

    # 4) Embed the query
    query_emb = embedder.encode([query_text], convert_to_numpy=True)

    # 5) Search in FAISS
    top_k = 3
    distances, indices = index.search(query_emb, top_k)

    retrieved_docs = []
    for dist, idx in zip(distances[0], indices[0]):
        doc_meta = metadata[idx].copy()
        doc_meta["distance"] = float(dist)
        retrieved_docs.append(doc_meta)

    # Debug info
    print(f"Query: {query_text}")
    print(f"Top {top_k} results:")
    for i, doc in enumerate(retrieved_docs):
        print(f"  {i+1}) Page {doc['page_number']} | Dist={doc['distance']:.4f} | {doc['text'][:75]}...")

    # Generate a real answer from the local model
    answer = generate_llm_answer(query_text, retrieved_docs, model_name)
    print("\nFinal Answer:\n", answer)
