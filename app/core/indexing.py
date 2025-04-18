"""
Indexing module for creating and managing vector indexes.
"""

import json
import os
import re
import time
import pickle
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Any, Optional

import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz

import config

# Configure logging
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Creates and manages vector indexes for document chunks.
    """

    def __init__(
        self,
        embedding_model: str = config.DEFAULT_EMBEDDING_MODEL,
        index_type: str = config.INDEX_TYPE,
        use_hybrid: bool = config.USE_HYBRID_SEARCH,
    ):
        """
        Initialize the document indexer.

        Args:
            embedding_model: Name of the SentenceTransformer model to use
            index_type: Type of FAISS index to create ('flat' or 'ivf')
            use_hybrid: Whether to use hybrid search (vector + keyword)
        """
        self.embedding_model = embedding_model
        self.index_type = index_type
        self.use_hybrid = use_hybrid

        # Determine device based on available resources
        self.device = self._get_optimal_device()
        logger.info(f"Using device: {self.device} for document indexing")

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

    def load_chunks(self, chunks_json: str) -> List[Dict]:
        """
        Load chunk data from JSON file.

        Args:
            chunks_json: Path to the JSON file containing chunks

        Returns:
            List of chunk dictionaries
        """
        with open(chunks_json, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        return chunks_data

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for keyword indexing.

        Args:
            text: Text to preprocess

        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces between words
        text = re.sub(r"[^\w\s]", " ", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def batch_encode(
        self, model: SentenceTransformer, texts: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode texts in batches to avoid memory issues.

        Args:
            model: SentenceTransformer model
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            Array of embeddings
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = model.encode(
                batch, convert_to_numpy=True, show_progress_bar=False
            )
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def build_vector_index(
        self, embeddings: np.ndarray, index_type: str = "flat"
    ) -> faiss.Index:
        """
        Build a FAISS index from embeddings.

        Args:
            embeddings: Array of embeddings
            index_type: Type of FAISS index to create ('flat' or 'ivf')

        Returns:
            FAISS index
        """
        embedding_dim = embeddings.shape[1]

        if index_type == "flat":
            # Simple flat index - exact but slower for large datasets
            index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "ivf":
            # IVF index - faster but approximate
            # Number of clusters - rule of thumb: sqrt(n) where n is dataset size
            n_clusters = min(int(np.sqrt(embeddings.shape[0])), 256)
            n_clusters = max(n_clusters, 4)  # At least 4 clusters

            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, n_clusters)
            # Train the index
            index.train(embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Add vectors to the index
        index.add(embeddings)
        return index

    def build_keyword_index(self, texts: List[str]) -> Tuple[TfidfVectorizer, Any]:
        """
        Build a TF-IDF index for keyword search.

        Args:
            texts: List of texts to index

        Returns:
            Tuple of (vectorizer, tfidf_matrix)
        """
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]

        # Create and fit TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

        return vectorizer, tfidf_matrix

    def build_index(self, chunks_json: str, output_prefix: str) -> Dict[str, str]:
        """
        Build vector and optionally keyword indexes from chunks.

        Args:
            chunks_json: Path to the JSON file containing chunks
            output_prefix: Prefix for output files

        Returns:
            Dictionary of output file paths
        """
        start_time = time.time()
        print(f"Starting index build with model: {self.embedding_model}")
        print(f"Index type: {self.index_type}, Hybrid: {self.use_hybrid}")

        # 1) Load chunk data
        chunks_data = self.load_chunks(chunks_json)
        texts = [chunk["text"] for chunk in chunks_data]
        print(f"Loaded {len(texts)} chunks from {chunks_json}")

        # 3) Create vector embeddings
        print("Creating vector embeddings...")
        # Sequential embedding generation using the optimal device
        try:
            model = SentenceTransformer(self.embedding_model, device=self.device)
            logger.info(f"Using {self.device} for embedding generation")
            embeddings = self.batch_encode(model, texts)
        except Exception as e:
            logger.warning(
                f"Error using {self.device} for embeddings: {e}, falling back to CPU"
            )
            self.device = "cpu"
            model = SentenceTransformer(self.embedding_model, device=self.device)
            logger.info(f"Switched to {self.device} for embedding generation")
            embeddings = self.batch_encode(model, texts)

        print(f"Created embeddings with shape: {embeddings.shape}")

        # 4) Build and save vector index
        print(f"Building {self.index_type} vector index...")
        vector_index = self.build_vector_index(embeddings, self.index_type)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

        # Save vector index
        index_path = f"{output_prefix}.index"
        faiss.write_index(vector_index, index_path)
        print(f"Vector index saved to {index_path}")

        # 5) Build keyword index if requested
        vectorizer_path = None
        tfidf_path = None

        if self.use_hybrid:
            print("Building keyword index...")
            vectorizer, tfidf_matrix = self.build_keyword_index(texts)

            # Save keyword index components
            vectorizer_path = f"{output_prefix}_vectorizer.pkl"
            with open(vectorizer_path, "wb") as f:
                pickle.dump(vectorizer, f)

            tfidf_path = f"{output_prefix}_tfidf.npz"
            save_npz(tfidf_path, tfidf_matrix)
            print(f"Keyword index saved to {vectorizer_path} and {tfidf_path}")

        # 6) Save chunk metadata separately
        metadata_path = f"{output_prefix}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        print(f"Chunk metadata saved to {metadata_path}")

        # Report total time
        elapsed_time = time.time() - start_time
        print(f"\nIndex building completed in {elapsed_time:.2f} seconds")

        # Return paths to created files
        result = {
            "index_path": index_path,
            "metadata_path": metadata_path,
        }

        if self.use_hybrid:
            result["vectorizer_path"] = vectorizer_path
            result["tfidf_path"] = tfidf_path

        return result

    def update_index(
        self, index_path: str, metadata_path: str, new_chunks: List[Dict]
    ) -> Dict[str, str]:
        """
        Update an existing index with new chunks.

        Args:
            index_path: Path to the existing FAISS index
            metadata_path: Path to the existing metadata file
            new_chunks: List of new chunk dictionaries to add

        Returns:
            Dictionary of updated file paths
        """
        # 1) Load existing index and metadata
        vector_index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # 2) Get texts from new chunks
        new_texts = [chunk["text"] for chunk in new_chunks]

        if not new_texts:
            print("No new chunks to add.")
            return {
                "index_path": index_path,
                "metadata_path": metadata_path,
            }

        # 3) Create embeddings for new chunks
        print(f"Creating embeddings for {len(new_texts)} new chunks...")
        try:
            model = SentenceTransformer(self.embedding_model, device=self.device)
            logger.info(f"Using {self.device} for embedding generation")
            new_embeddings = self.batch_encode(model, new_texts)
        except Exception as e:
            logger.warning(
                f"Error using {self.device} for embeddings: {e}, falling back to CPU"
            )
            self.device = "cpu"
            model = SentenceTransformer(self.embedding_model, device=self.device)
            logger.info(f"Switched to {self.device} for embedding generation")
            new_embeddings = self.batch_encode(model, new_texts)

        # 4) Add new embeddings to the index
        vector_index.add(new_embeddings)

        # 5) Update metadata
        metadata.extend(new_chunks)

        # 6) Save updated index and metadata
        faiss.write_index(vector_index, index_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Index updated with {len(new_chunks)} new chunks.")

        # 7) Update keyword index if it exists
        result = {
            "index_path": index_path,
            "metadata_path": metadata_path,
        }

        if self.use_hybrid:
            # Check if keyword index files exist
            index_prefix = index_path.replace(".index", "")
            vectorizer_path = f"{index_prefix}_vectorizer.pkl"
            tfidf_path = f"{index_prefix}_tfidf.npz"

            if os.path.exists(vectorizer_path) and os.path.exists(tfidf_path):
                print("Updating keyword index...")

                # Load existing vectorizer
                with open(vectorizer_path, "rb") as f:
                    vectorizer = pickle.load(f)

                # Load existing TF-IDF matrix
                tfidf_matrix = load_npz(tfidf_path)

                # Get all texts (existing + new)
                all_texts = [chunk["text"] for chunk in metadata]

                # Rebuild keyword index
                preprocessed_texts = [self.preprocess_text(text) for text in all_texts]
                vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

                # Save updated keyword index
                with open(vectorizer_path, "wb") as f:
                    pickle.dump(vectorizer, f)

                save_npz(tfidf_path, tfidf_matrix)

                print(f"Keyword index updated.")

                result["vectorizer_path"] = vectorizer_path
                result["tfidf_path"] = tfidf_path

        return result
