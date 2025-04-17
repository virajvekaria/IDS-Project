import sys
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

def load_index(index_path: str):
    return faiss.read_index(index_path)

def load_metadata(metadata_path: str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def chat_loop(index, metadata, embedding_model: str, ollama_model: str, top_k: int = 3):
    """
    Interactive chat loop.
    - index, metadata: for retrieval
    - embedding_model: SentenceTransformer model for question embeddings
    - ollama_model: The model name used by Ollama
    - top_k: how many chunks to retrieve each turn
    """
    # Initialize the embedder
    embedder = SentenceTransformer(embedding_model)

    # We'll store the conversation as a list of messages
    # Example: [ {"role": "user", "content": "..."},
    #            {"role": "assistant", "content": "..."} ]
    conversation_history = []

    print("\nEntering chat mode. Type 'exit' or 'quit' to end.\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break

        # 1) Embed user question + retrieve from FAISS
        query_emb = embedder.encode([user_input], convert_to_numpy=True)
        distances, indices = index.search(query_emb, top_k)

        # 2) Build a "context" string from retrieved docs
        retrieved_docs = []
        for dist, idx in zip(distances[0], indices[0]):
            doc_meta = metadata[idx]
            retrieved_docs.append(doc_meta)

        # Add chunk references to the context so the model can see them
        context_text = ""
        for doc in retrieved_docs:
            context_text += f"(Page {doc['page_number']}) {doc['text']}\n\n"

        # 3) We'll create a "system" message to hold the context from retrieval,
        #    plus the entire conversation so far. We'll then add the new user message.
        #    Ollama doesn't strictly follow the "messages" format as OpenAI does,
        #    but we can replicate a conversation approach by passing them in order.

        # Let's build a single prompt that includes:
        # - Summaries or direct quotes from previous conversation
        # - The newly retrieved context
        # - The new user question

        # Convert conversation_history to a user/assistant style text:
        conversation_text = ""
        for msg in conversation_history:
            role = msg["role"]
            content = msg["content"]
            conversation_text += f"{role.capitalize()}: {content}\n\n"

        # Now add the new user input (like a chat)
        conversation_text += f"User: {user_input}\n\n"

        # Add instructions for the model to incorporate retrieved docs
        prompt = (
            "You are a helpful assistant. You have the following conversation history and context.\n\n"
            f"Conversation:\n{conversation_text}"
            f"Relevant PDF Chunks:\n{context_text}"
            "Please use ONLY the above conversation and relevant PDF chunks to answer the user's latest question.\n"
            "Cite the page numbers when providing information from the PDF. "
            "If the information is not in the PDF, say you don't have that data.\n\n"
            "Assistant:"
        )

        # 4) Call Ollama with the full prompt
        # We'll just keep the entire conversation in a single user message (which includes the retrieved docs).
        response = ollama.chat(
            model=ollama_model, 
            messages=[{"role": "user", "content": prompt}],
        )

        assistant_reply = response['message']['content'].strip()
        print(f"Assistant: {assistant_reply}\n")

        # 5) Append the user query and assistant's response to conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": assistant_reply})

def main():
    if len(sys.argv) < 4:
        print("Usage: python 4_query_qa_chat.py <index_file> <metadata_file> <ollama_model>")
        print("Example: python 4_query_qa_chat.py faiss_index.index faiss_index_metadata.json deepseek-r1:7b")
        sys.exit(1)

    index_file = sys.argv[1]
    metadata_file = sys.argv[2]
    ollama_model = sys.argv[3]

    # 1) Load FAISS index + metadata
    index = load_index(index_file)
    metadata = load_metadata(metadata_file)

    # 2) Hardcode or choose an embedding model for retrieval
    embedding_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    # 3) Start chat loop
    chat_loop(index, metadata, embedding_model, ollama_model, top_k=3)

if __name__ == "__main__":
    main()
