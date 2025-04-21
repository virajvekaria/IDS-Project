"""
Retrieval module for searching and retrieving document chunks.
"""

import os
import json
import re
import pickle
import logging
from typing import List, Dict, Tuple, Any, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import ollama

import config

# Set up logging
logger = logging.getLogger(__name__)


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

        # Determine device based on available resources
        self.device = self._get_optimal_device()
        logger.info(f"Using device: {self.device} for embeddings")

        # Initialize the embedding model on the appropriate device
        try:
            self.embedder = SentenceTransformer(embedding_model, device=self.device)
            logger.info(
                f"Successfully loaded embedding model {embedding_model} on {self.device}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load model on {self.device}, falling back to CPU: {e}"
            )
            self.device = "cpu"
            self.embedder = SentenceTransformer(embedding_model, device=self.device)

    def _get_optimal_device(self) -> str:
        """
        Determine the optimal device (CUDA/CPU) based on available resources.

        Returns:
            String representing the device to use ('cuda:0', 'cpu', etc.)
        """
        if torch.cuda.is_available():
            # Check available GPU memory
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = total_memory - allocated_memory

                # If we have at least 500MB free, use GPU
                if free_memory > 500 * 1024 * 1024:  # 500MB in bytes
                    return "cuda:0"
                else:
                    logger.warning(
                        f"Insufficient GPU memory: {free_memory/1024/1024:.2f}MB free, falling back to CPU"
                    )
                    return "cpu"
            except Exception as e:
                logger.warning(f"Error checking GPU memory: {e}, falling back to CPU")
                return "cpu"
        else:
            return "cpu"

    def load_index(self, index_path: str) -> faiss.Index:
        """
        Load a FAISS index from disk.

        Args:
            index_path: Path to the FAISS index file

        Returns:
            FAISS index
        """
        index = faiss.read_index(index_path)

        # Optimize HNSW parameters if it's an HNSW index
        if isinstance(index, faiss.IndexHNSWFlat):
            # Set efSearch parameter for better search quality
            # Higher values give more accurate results but slower search
            index.hnsw.efSearch = 128
            logger.info(f"Loaded HNSW index and set efSearch to 128")

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
        try:
            query_emb = self.embedder.encode([query], convert_to_numpy=True)
        except Exception as e:
            logger.warning(
                f"Error during encoding with {self.device}, falling back to CPU: {e}"
            )
            # Fall back to CPU if there's an error
            old_device = self.device
            self.device = "cpu"
            self.embedder = SentenceTransformer(
                self.embedding_model, device=self.device
            )
            logger.info(f"Switched from {old_device} to {self.device} due to error")
            query_emb = self.embedder.encode([query], convert_to_numpy=True)

        # Vector search
        vector_k = min(top_k * 2, vector_index.ntotal)  # Get more results for reranking

        # If using HNSW index, temporarily increase efSearch for better recall
        if isinstance(vector_index, faiss.IndexHNSWFlat):
            original_ef_search = vector_index.hnsw.efSearch
            # Increase efSearch for this query to get better recall
            vector_index.hnsw.efSearch = max(original_ef_search, 256)
            logger.info(
                f"Temporarily increased HNSW efSearch to {vector_index.hnsw.efSearch} for better recall"
            )

            # Perform search
            vector_distances, vector_indices = vector_index.search(query_emb, vector_k)

            # Restore original efSearch
            vector_index.hnsw.efSearch = original_ef_search
        else:
            # Standard search for other index types
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
            f"- Do NOT invent information not in the context.\n"
            f"- Start your response directly with the answer.\n\n"
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
            try:
                query_emb = self.embedder.encode([query], convert_to_numpy=True)

                # If using HNSW index, temporarily increase efSearch for better recall
                if isinstance(vector_index, faiss.IndexHNSWFlat):
                    original_ef_search = vector_index.hnsw.efSearch
                    # Increase efSearch for this query to get better recall
                    vector_index.hnsw.efSearch = max(original_ef_search, 256)
                    logger.info(
                        f"Temporarily increased HNSW efSearch to {vector_index.hnsw.efSearch} for better recall"
                    )

                    # Perform search
                    distances, indices = vector_index.search(query_emb, top_k)

                    # Restore original efSearch
                    vector_index.hnsw.efSearch = original_ef_search
                else:
                    # Standard search for other index types
                    distances, indices = vector_index.search(query_emb, top_k)
            except Exception as e:
                logger.warning(
                    f"Error during vector search with {self.device}, falling back to CPU: {e}"
                )
                # Fall back to CPU if there's an error
                old_device = self.device
                self.device = "cpu"
                self.embedder = SentenceTransformer(
                    self.embedding_model, device=self.device
                )
                logger.info(f"Switched from {old_device} to {self.device} due to error")
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

        # Check if there's any conversation history
        has_conversation_history = len(conversation_history) > 0

        # Convert conversation_history to a user/assistant style text
        conversation_text = ""
        if has_conversation_history:
            for msg in conversation_history:
                role = msg["role"]
                content = msg["content"]
                conversation_text += f"{role.capitalize()}: {content}\n\n"

            # Add the new user input with conversation context
            conversation_text += f"User: {query}\n\n"
        else:
            # Just the current query without conversation context
            conversation_text = f"User: {query}\n\n"

        # 5) Generate answer
        # Check if there are any relevant documents
        has_relevant_docs = len(retrieved_docs) > 0 and any(
            doc.get("score", 0) > 0.5 for doc in retrieved_docs
        )

        # Determine if the query is likely document-related or just conversational
        # Simple conversational queries like greetings, thanks, etc.
        simple_conversational = len(query.split()) <= 3 and any(
            word in query.lower()
            for word in [
                "hi",
                "hello",
                "hey",
                "thanks",
                "thank",
                "bye",
                "goodbye",
                "ok",
                "okay",
                "yes",
                "no",
            ]
        )

        # For document-related queries or when we have relevant docs and it's not a simple greeting
        if has_relevant_docs and not simple_conversational:
            if has_conversation_history:
                prompt = (
                    "You are a helpful assistant. You have the following conversation history and document context.\n\n"
                    f"Conversation:\n{conversation_text}"
                    f"Relevant Document Chunks:\n{context_text}"
                    "Please answer the user's latest question according to the document context.\n\n"
                    "Guidelines:\n"
                    "1. IMPORTANT: You MUST cite page numbers when you are using information from the documents.\n"
                    "2. When citing, use the format (Page X) immediately after the information from that page.\n"
                    "3. Be very explicit with citations - every fact or piece of information from the documents needs a citation.\n"
                    "4. If the question is conversational and not about the documents, respond naturally without citations.\n"
                    "5. For simple greetings or conversational exchanges, respond naturally without any citations.\n"
                    "6. Keep your answers concise and to the point.\n"
                    "7. Start your response directly with the answer.\n\n"
                    "Assistant:"
                )
            else:
                # No conversation history, just the current query and document context
                prompt = (
                    "You are a helpful assistant answering a question using document content.\n\n"
                    f"User Question: {query}\n\n"
                    f"Relevant Document Chunks:\n{context_text}"
                    "Please answer the user's question according to the document context.\n\n"
                    "Guidelines:\n"
                    "1. IMPORTANT: You MUST cite page numbers when you are using information from the documents.\n"
                    "2. When citing, use the format (Page X) immediately after the information from that page.\n"
                    "3. Be very explicit with citations - every fact or piece of information from the documents needs a citation.\n"
                    "4. Keep your answers concise and to the point.\n"
                    "5. Start your response directly with the answer.\n\n"
                    "Assistant:"
                )
        else:
            # For conversational questions with no relevant document context
            if has_conversation_history:
                prompt = (
                    "You are a helpful assistant. You have the following conversation history.\n\n"
                    f"Conversation:\n{conversation_text}"
                    "Please answer the user's latest question based on the conversation history.\n\n"
                    "Guidelines:\n"
                    "1. Respond naturally as a helpful assistant.\n"
                    "2. If the question requires specific knowledge that you don't have, politely say so.\n"
                    "3. Keep your answers concise and to the point.\n"
                    "4. Start your response directly with the answer.\n\n"
                    "Assistant:"
                )
            else:
                # No conversation history, just a simple conversational query
                prompt = (
                    "You are a helpful assistant.\n\n"
                    f"User Question: {query}\n\n"
                    "Please answer the user's question.\n\n"
                    "Guidelines:\n"
                    "1. Respond naturally as a helpful assistant.\n"
                    "2. If the question requires specific knowledge that you don't have, politely say so.\n"
                    "3. Keep your answers concise and to the point.\n"
                    "4. Start your response directly with the answer.\n\n"
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
            try:
                query_emb = self.embedder.encode([query], convert_to_numpy=True)

                # If using HNSW index, temporarily increase efSearch for better recall
                if isinstance(vector_index, faiss.IndexHNSWFlat):
                    original_ef_search = vector_index.hnsw.efSearch
                    # Increase efSearch for this query to get better recall
                    vector_index.hnsw.efSearch = max(original_ef_search, 256)
                    logger.info(
                        f"Temporarily increased HNSW efSearch to {vector_index.hnsw.efSearch} for better recall"
                    )

                    # Perform search
                    distances, indices = vector_index.search(query_emb, top_k)

                    # Restore original efSearch
                    vector_index.hnsw.efSearch = original_ef_search
                else:
                    # Standard search for other index types
                    distances, indices = vector_index.search(query_emb, top_k)
            except Exception as e:
                logger.warning(
                    f"Error during vector search with {self.device}, falling back to CPU: {e}"
                )
                # Fall back to CPU if there's an error
                old_device = self.device
                self.device = "cpu"
                self.embedder = SentenceTransformer(
                    self.embedding_model, device=self.device
                )
                logger.info(f"Switched from {old_device} to {self.device} due to error")
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
