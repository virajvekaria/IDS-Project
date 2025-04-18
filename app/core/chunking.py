"""
Text chunking module for semantic chunking of document content.
"""

import json
import os
import re
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional

from sentence_transformers import SentenceTransformer, util
import nltk

# Configure logging
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
try:
    nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize
except Exception as e:
    print(f"NLTK punkt download failed: {e}. Using fallback tokenizer.")

    # Fallback tokenizer if NLTK fails
    def sent_tokenize(text):
        # Simple sentence tokenizer based on common punctuation
        if not text or not isinstance(text, str):
            return []

        sentences = []
        # Split by newlines first
        for line in text.split("\n"):
            if not line.strip():
                continue
            # Split by common sentence-ending punctuation
            parts = re.split(r"(?<=[.!?])\s+", line)
            sentences.extend([p.strip() for p in parts if p.strip()])

        # If no sentences were found, return the original text as a single sentence
        if not sentences and text.strip():
            return [text.strip()]

        return sentences


import config


class DocumentChunker:
    """
    Chunks document content semantically using embeddings.
    """

    def __init__(
        self,
        embedding_model: str = config.DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Initialize the document chunker.

        Args:
            embedding_model: Name of the SentenceTransformer model to use
            chunk_size: Target chunk size in words
            overlap: Number of sentences to overlap between chunks
        """
        self.model_name = embedding_model
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Determine device based on available resources
        self.device = self._get_optimal_device()
        logger.info(f"Using device: {self.device} for document chunking")

        # Initialize the embedding model on the appropriate device
        try:
            self.model = SentenceTransformer(embedding_model, device=self.device)
            logger.info(
                f"Successfully loaded embedding model {embedding_model} on {self.device}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load model on {self.device}, falling back to CPU: {e}"
            )
            self.device = "cpu"
            self.model = SentenceTransformer(embedding_model, device=self.device)

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

    def combine_page_text_and_tables(self, page_data: Dict) -> str:
        """
        Combine the main text with table data so we don't lose table info.

        Args:
            page_data: Dictionary containing page content

        Returns:
            Combined text string
        """
        base_text = page_data.get("text", "")
        tables_str_list = []

        # Flatten table rows into strings
        for tbl in page_data.get("tables", []):
            for row in tbl["rows"]:
                header_str = (row.get("header") or "").strip()
                content_list = row.get("content", [])
                content_text = "; ".join(content_list)
                row_text = (
                    f"{header_str}: {content_text}" if header_str else content_text
                )
                tables_str_list.append(row_text)

        if tables_str_list:
            tables_combined = "\n".join(tables_str_list)
            base_text += "\n\n[TABLE DATA]\n" + tables_combined + "\n"

        return base_text

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        return sent_tokenize(text.strip())

    def calculate_adaptive_threshold(self, sentences: List[str], embeddings) -> float:
        """
        Calculate an adaptive similarity threshold based on the average similarity
        between consecutive sentences in the document.

        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings

        Returns:
            Adaptive similarity threshold
        """
        if len(sentences) < 3:
            return 0.6  # Default threshold for very short documents

        similarities = []
        for i in range(1, len(sentences)):
            sim = util.cos_sim(embeddings[i], embeddings[i - 1]).item()
            similarities.append(sim)

        # Calculate mean similarity
        mean_sim = sum(similarities) / len(similarities)

        # Adjust threshold based on mean similarity
        # Lower mean similarity -> lower threshold to avoid over-chunking
        # Higher mean similarity -> higher threshold for better chunking
        if mean_sim < 0.4:
            return max(0.4, mean_sim * 1.2)  # Lower bound of 0.4
        elif mean_sim > 0.8:
            return min(0.8, mean_sim * 0.9)  # Upper bound of 0.8
        else:
            return mean_sim

    def semantic_chunk_sentences(
        self, sentences: List[str], chunk_size: int, overlap: int = 1
    ) -> List[str]:
        """
        Group sentences based on semantic similarity and word count with overlap.

        Args:
            sentences: List of sentences to chunk
            chunk_size: Target chunk size in words
            overlap: Number of sentences to overlap between chunks

        Returns:
            List of chunked text
        """
        if not sentences:
            return []

        # Encode all sentences at once for efficiency
        try:
            embeddings = self.model.encode(sentences, convert_to_tensor=True)
        except Exception as e:
            logger.warning(
                f"Error during encoding with {self.device}, falling back to CPU: {e}"
            )
            # Fall back to CPU if there's an error
            old_device = self.device
            self.device = "cpu"
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Switched from {old_device} to {self.device} due to error")
            embeddings = self.model.encode(sentences, convert_to_tensor=True)

        # Calculate adaptive threshold based on document characteristics
        similarity_threshold = self.calculate_adaptive_threshold(sentences, embeddings)
        print(f"Using adaptive similarity threshold: {similarity_threshold:.2f}")

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

            if similarity > similarity_threshold or current_len < chunk_size:
                current_chunk.append(sentence)
                current_len += word_count
            else:
                # Save the current chunk
                chunks.append(" ".join(current_chunk))

                # Start a new chunk with overlap
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:] + [sentence]

                # Recalculate current length
                current_len = sum(len(s.split()) for s in current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_page(
        self,
        page_data: Dict,
        source_doc: str,
        chunk_size: int,
        overlap: int,
        start_id: int,
    ) -> List[Dict]:
        """
        Process a single page for parallel execution.

        Args:
            page_data: Dictionary containing page content
            source_doc: Source document filename
            chunk_size: Target chunk size in words
            overlap: Number of sentences to overlap between chunks
            start_id: Starting chunk ID

        Returns:
            List of chunk dictionaries
        """
        page_num = page_data["page_number"]
        combined_text = self.combine_page_text_and_tables(page_data)
        sentences = self.split_into_sentences(combined_text)
        semantic_chunks = self.semantic_chunk_sentences(sentences, chunk_size, overlap)

        page_chunks = []
        chunk_id_counter = start_id

        for chunk_text in semantic_chunks:
            chunk_record = {
                "chunk_id": f"chunk_{chunk_id_counter}",
                "source_doc": source_doc,
                "page_number": page_num,
                "text": chunk_text,
            }
            page_chunks.append(chunk_record)
            chunk_id_counter += 1

        return page_chunks

    def chunk_document(
        self,
        pages_content: List[Dict],
        source_doc: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[Dict]:
        """
        Perform semantic (embedding-based) chunking of page data with parallel processing.

        Args:
            pages_content: List of page data dictionaries
            source_doc: Source document filename
            chunk_size: Target chunk size in words (overrides instance default if provided)
            overlap: Number of sentences to overlap between chunks (overrides instance default if provided)

        Returns:
            List of chunk dictionaries
        """
        # Use instance defaults if not provided
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.overlap

        all_chunks = []

        # Sequential processing for all documents
        chunk_id_counter = 0
        for page_data in pages_content:
            page_num = page_data["page_number"]
            combined_text = self.combine_page_text_and_tables(page_data)
            sentences = self.split_into_sentences(combined_text)
            semantic_chunks = self.semantic_chunk_sentences(
                sentences, chunk_size, overlap
            )

            for chunk_text in semantic_chunks:
                chunk_record = {
                    "chunk_id": f"chunk_{chunk_id_counter}",
                    "source_doc": source_doc,
                    "page_number": page_num,
                    "text": chunk_text,
                }
                all_chunks.append(chunk_record)
                chunk_id_counter += 1

        # Sort chunks by ID to ensure consistent ordering
        all_chunks.sort(key=lambda x: int(x["chunk_id"].split("_")[1]))
        return all_chunks

    def save_chunks(self, chunks: List[Dict], output_path: str) -> None:
        """
        Save chunks to a JSON file.

        Args:
            chunks: List of chunk dictionaries
            output_path: Path to save the JSON file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
