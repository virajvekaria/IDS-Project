"""
Retrieval module for searching and retrieving document chunks.
"""

import os
import json
import re
import pickle
from typing import List, Dict, Tuple, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import ollama

import config


class DocumentRetriever:
    """
    Retrieves document chunks based on queries and generates answers.
    """

    def __init__(
        self,
        embedding_model: str = config.DEFAULT_EMBEDDING_MODEL,
        llm_model: str = config.DEFAULT_LLM_MODEL,
        vector_weight: float = config.DEFAULT_VECTOR_WEIGHT,
    ):
        """
        Initialize the document retriever.

        Args:
            embedding_model: Name of the SentenceTransformer model to use for queries
            llm_model: Name of the LLM model to use for answer generation
            vector_weight: Weight for vector scores in hybrid search (0.0-1.0)
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_weight = vector_weight
        self.embedder = SentenceTransformer(embedding_model)

    def load_index(self, index_path: str) -> faiss.Index:
        """
        Load a FAISS index from disk.

        Args:
            index_path: Path to the FAISS index file

        Returns:
            FAISS index
        """
        index = faiss.read_index(index_path)
        return index

    def load_metadata(self, metadata_path: str) -> List[Dict]:
        """
        Load chunk metadata from JSON file.

        Args:
            metadata_path: Path to the metadata JSON file

        Returns:
            List of chunk dictionaries
        """
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def load_keyword_index(
        self, vectorizer_path: str, tfidf_path: str
    ) -> Tuple[TfidfVectorizer, Any]:
        """
        Load keyword search components (TF-IDF vectorizer and matrix).

        Args:
            vectorizer_path: Path to the pickled vectorizer
            tfidf_path: Path to the TF-IDF matrix

        Returns:
            Tuple of (vectorizer, tfidf_matrix)
        """
        # Load vectorizer
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        # Load TF-IDF matrix
        tfidf_matrix = load_npz(tfidf_path)

        return vectorizer, tfidf_matrix

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query text for keyword search.

        Args:
            query: Query text

        Returns:
            Preprocessed query
        """
        # Convert to lowercase
        query = query.lower()
        # Remove special characters but keep spaces between words
        query = re.sub(r"[^\w\s]", " ", query)
        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query).strip()
        return query

    def keyword_search(
        self, query: str, vectorizer: TfidfVectorizer, tfidf_matrix, top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Perform keyword-based search using TF-IDF.

        Args:
            query: Query text
            vectorizer: TF-IDF vectorizer
            tfidf_matrix: TF-IDF matrix
            top_k: Number of top results to return

        Returns:
            List of (index, score) tuples
        """
        # Preprocess query
        processed_query = self.preprocess_query(query)

        # Transform query to TF-IDF space
        query_vector = vectorizer.transform([processed_query])

        # Calculate cosine similarity between query and documents
        # Using dot product for sparse matrices (equivalent to cosine similarity for normalized vectors)
        similarities = (query_vector * tfidf_matrix.T).toarray()[0]

        # Get top-k results
        top_indices = np.argsort(-similarities)[:top_k]
        results = [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0
        ]

        return results

    def hybrid_search(
        self,
        query: str,
        vector_index: faiss.Index,
        vectorizer: Optional[TfidfVectorizer] = None,
        tfidf_matrix=None,
        vector_weight: Optional[float] = None,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Perform hybrid search combining vector and keyword search.

        Args:
            query: Query text
            vector_index: FAISS index
            vectorizer: TF-IDF vectorizer (optional)
            tfidf_matrix: TF-IDF matrix (optional)
            vector_weight: Weight for vector scores (0.0-1.0)
            top_k: Number of top results to return

        Returns:
            List of (index, score) tuples
        """
        # Use instance default if not provided
        vector_weight = vector_weight or self.vector_weight

        # Encode query for vector search
        query_emb = self.embedder.encode([query], convert_to_numpy=True)

        # Vector search
        vector_k = min(top_k * 2, vector_index.ntotal)  # Get more results for reranking
        vector_distances, vector_indices = vector_index.search(query_emb, vector_k)

        # Convert to list of (index, score) tuples
        # Note: FAISS returns L2 distances, so smaller is better
        # Convert to similarity score (1 / (1 + distance))
        vector_results = [
            (int(idx), float(1 / (1 + dist)))
            for dist, idx in zip(vector_distances[0], vector_indices[0])
        ]

        # If no keyword components, return vector results
        if vectorizer is None or tfidf_matrix is None:
            return vector_results[:top_k]

        # Keyword search
        keyword_results = self.keyword_search(
            query, vectorizer, tfidf_matrix, top_k * 2
        )

        # Combine results with weighted scores
        # Create dictionaries for easy lookup
        vector_dict = {idx: score for idx, score in vector_results}
        keyword_dict = {idx: score for idx, score in keyword_results}

        # Get all unique document IDs
        all_ids = set(vector_dict.keys()) | set(keyword_dict.keys())

        # Combine scores
        combined_scores = []
        for doc_id in all_ids:
            vector_score = vector_dict.get(doc_id, 0.0)
            keyword_score = keyword_dict.get(doc_id, 0.0)

            # Normalize scores (both should be in 0-1 range)
            combined_score = (vector_weight * vector_score) + (
                (1 - vector_weight) * keyword_score
            )
            combined_scores.append((doc_id, combined_score))

        # Sort by combined score (descending) and return top-k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:top_k]

    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Dict],
        model_name: Optional[str] = None,
        stream: bool = False,
    ):
        """
        Generate an answer using an LLM based on retrieved documents.

        Args:
            question: Question text
            retrieved_docs: List of retrieved document chunks
            model_name: Name of the LLM model to use (overrides instance default if provided)
            stream: Whether to stream the response

        Returns:
            If stream=False: Generated answer as a string
            If stream=True: Generator yielding response chunks
        """
        # Use instance default if not provided
        model_name = model_name or self.llm_model

        if not retrieved_docs:
            if stream:

                def empty_generator():
                    yield "No relevant information found in the document."

                return empty_generator()
            else:
                return "No relevant information found in the document."

        # Build context from retrieved docs
        context_text = ""
        for doc in retrieved_docs:
            context_text += f"(Page {doc['page_number']}): {doc['text']}\n\n"

        # Final prompt
        prompt = (
            f"You are a helpful assistant answering questions using document content.\n\n"
            f"Context (extracted from the documents):\n{context_text}\n\n"
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
        if stream:
            # Return a generator that yields response chunks
            def response_generator():
                for chunk in ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                ):
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]

            return response_generator()
        else:
            # Return the complete response
            response = ollama.chat(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"].strip()

    def chat(
        self,
        query: str,
        conversation_history: List[Dict],
        index_path: str,
        metadata_path: str,
        use_hybrid: bool = config.USE_HYBRID_SEARCH,
        top_k: int = 3,
        stream: bool = False,
    ):
        """
        Process a chat query and return an answer.

        Args:
            query: Query text
            conversation_history: List of conversation history dictionaries
            index_path: Path to the FAISS index file
            metadata_path: Path to the metadata JSON file
            use_hybrid: Whether to use hybrid search
            top_k: Number of top results to return
            stream: Whether to stream the response

        Returns:
            If stream=False: Tuple of (answer, retrieved_docs)
            If stream=True: Tuple of (generator yielding response chunks, retrieved_docs)
        """
        # 1) Load index and metadata
        vector_index = self.load_index(index_path)
        metadata = self.load_metadata(metadata_path)

        # 2) Determine search method and load necessary components
        vectorizer = None
        tfidf_matrix = None

        if use_hybrid:
            # Check if keyword index files exist
            index_prefix = index_path.replace(".index", "")
            vectorizer_path = f"{index_prefix}_vectorizer.pkl"
            tfidf_path = f"{index_prefix}_tfidf.npz"

            if os.path.exists(vectorizer_path) and os.path.exists(tfidf_path):
                try:
                    vectorizer, tfidf_matrix = self.load_keyword_index(
                        vectorizer_path, tfidf_path
                    )
                except Exception as e:
                    print(f"Error loading keyword index: {e}")
                    use_hybrid = False

        # 3) Perform search
        if use_hybrid and vectorizer is not None and tfidf_matrix is not None:
            # Hybrid search
            search_results = self.hybrid_search(
                query,
                vector_index,
                vectorizer,
                tfidf_matrix,
                vector_weight=self.vector_weight,
                top_k=top_k,
            )

            # Convert search results to document list
            retrieved_docs = []
            for idx, score in search_results:
                doc_meta = metadata[idx].copy()
                doc_meta["score"] = float(score)  # Hybrid score
                retrieved_docs.append(doc_meta)
        else:
            # Vector-only search
            query_emb = self.embedder.encode([query], convert_to_numpy=True)
            distances, indices = vector_index.search(query_emb, top_k)

            retrieved_docs = []
            for dist, idx in zip(distances[0], indices[0]):
                doc_meta = metadata[idx].copy()
                doc_meta["distance"] = float(dist)
                # Add a normalized score for consistency with hybrid search
                doc_meta["score"] = float(1 / (1 + dist))
                retrieved_docs.append(doc_meta)

        # 4) Build context from conversation history and retrieved docs
        context_text = ""
        for doc in retrieved_docs:
            context_text += f"(Page {doc['page_number']}): {doc['text']}\n\n"

        # Convert conversation_history to a user/assistant style text
        conversation_text = ""
        for msg in conversation_history:
            role = msg["role"]
            content = msg["content"]
            conversation_text += f"{role.capitalize()}: {content}\n\n"

        # Add the new user input
        conversation_text += f"User: {query}\n\n"

        # 5) Generate answer
        prompt = (
            "You are a helpful assistant. You have the following conversation history and context.\n\n"
            f"Conversation:\n{conversation_text}"
            f"Relevant Document Chunks:\n{context_text}"
            "Please use ONLY the above conversation and relevant document chunks to answer the user's latest question.\n"
            "Cite the page numbers when providing information from the documents. "
            "If the information is not in the documents, say you don't have that data.\n\n"
            "Assistant:"
        )

        if stream:
            # Return a generator that yields response chunks
            def response_generator():
                for chunk in ollama.chat(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                ):
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]

            return response_generator(), retrieved_docs
        else:
            # Return the complete response
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )

            answer = response["message"]["content"].strip()

            return answer, retrieved_docs

    def search(
        self,
        query: str,
        index_path: str,
        metadata_path: str,
        use_hybrid: bool = config.USE_HYBRID_SEARCH,
        top_k: int = 5,
        stream: bool = False,
    ):
        """
        Search for documents and generate an answer.

        Args:
            query: Query text
            index_path: Path to the FAISS index file
            metadata_path: Path to the metadata JSON file
            use_hybrid: Whether to use hybrid search
            top_k: Number of top results to return
            stream: Whether to stream the response

        Returns:
            If stream=False: Tuple of (retrieved_docs, answer)
            If stream=True: Tuple of (retrieved_docs, generator yielding response chunks)
        """
        # 1) Load index and metadata
        vector_index = self.load_index(index_path)
        metadata = self.load_metadata(metadata_path)

        # 2) Determine search method and load necessary components
        vectorizer = None
        tfidf_matrix = None

        if use_hybrid:
            # Check if keyword index files exist
            index_prefix = index_path.replace(".index", "")
            vectorizer_path = f"{index_prefix}_vectorizer.pkl"
            tfidf_path = f"{index_prefix}_tfidf.npz"

            if os.path.exists(vectorizer_path) and os.path.exists(tfidf_path):
                try:
                    vectorizer, tfidf_matrix = self.load_keyword_index(
                        vectorizer_path, tfidf_path
                    )
                except Exception as e:
                    print(f"Error loading keyword index: {e}")
                    use_hybrid = False

        # 3) Perform search
        if use_hybrid and vectorizer is not None and tfidf_matrix is not None:
            # Hybrid search
            search_results = self.hybrid_search(
                query,
                vector_index,
                vectorizer,
                tfidf_matrix,
                vector_weight=self.vector_weight,
                top_k=top_k,
            )

            # Convert search results to document list
            retrieved_docs = []
            for idx, score in search_results:
                doc_meta = metadata[idx].copy()
                doc_meta["score"] = float(score)  # Hybrid score
                retrieved_docs.append(doc_meta)
        else:
            # Vector-only search
            query_emb = self.embedder.encode([query], convert_to_numpy=True)
            distances, indices = vector_index.search(query_emb, top_k)

            retrieved_docs = []
            for dist, idx in zip(distances[0], indices[0]):
                doc_meta = metadata[idx].copy()
                doc_meta["distance"] = float(dist)
                # Add a normalized score for consistency with hybrid search
                doc_meta["score"] = float(1 / (1 + dist))
                retrieved_docs.append(doc_meta)

        # 4) Generate answer
        if stream:
            # Return a generator that yields response chunks
            answer_generator = self.generate_answer(query, retrieved_docs, stream=True)
            return retrieved_docs, answer_generator
        else:
            # Return the complete response
            answer = self.generate_answer(query, retrieved_docs)
            return retrieved_docs, answer
