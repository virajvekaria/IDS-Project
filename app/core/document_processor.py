"""
Document processor module for handling PDF documents.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from pypdf import PdfReader
import nltk
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
try:
    nltk.download("punkt", quiet=True)
except Exception as e:
    logger.warning(f"NLTK punkt download failed: {e}. This may affect text processing.")

import config
from nltk.tokenize import sent_tokenize


class DocumentProcessor:
    """
    Processes PDF documents and extracts their content.
    Supported formats:
    - PDF
    - TXT
    """

    def __init__(self):
        """
        Initialize the document processor.
        """
        pass

    def process_document(self, file_path: str) -> Tuple[List[Dict], Dict]:
        """
        Process a document and extract its content.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (pages_content, metadata)
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        # Extract metadata
        metadata = {
            "filename": file_path.name,
            "file_size": os.path.getsize(file_path),
            "file_type": file_ext[1:],  # Remove the dot
            "path": str(file_path),
        }

        # Currently only supporting PDF files
        if file_ext == ".pdf":
            pages_content = self._process_pdf(file_path)
            metadata["page_count"] = len(pages_content)
        elif file_ext == ".txt":
            pages_content = self._process_txt(file_path)
            metadata["page_count"] = 1
        else:
            raise ValueError(
                f"Unsupported file type: {file_ext}. Only PDF and TXT files are supported."
            )

        return pages_content, metadata

    def _process_pdf(self, file_path: str) -> List[Dict]:
        """
        Extract content from a PDF file using pypdf.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of page content dictionaries
        """
        pages_content = []

        try:
            # Open the PDF file
            pdf_reader = PdfReader(file_path)

            # Process each page
            for i, page in enumerate(pdf_reader.pages):
                page_num = i + 1
                page_text = page.extract_text() or ""

                # Create page data dictionary
                page_data = {
                    "page_number": page_num,
                    "text": page_text.strip(),
                    "tables": [],  # No table extraction in pypdf
                }

                pages_content.append(page_data)

            logger.info(f"Processed PDF with {len(pages_content)} pages")

            if not pages_content:
                logger.warning(f"No content extracted from {file_path}")
                # Add an empty page to avoid downstream errors
                pages_content.append(
                    {
                        "page_number": 1,
                        "text": "",
                        "tables": [],
                    }
                )

        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {str(e)}")
            raise

        return pages_content

    # Only supporting PDF and TXT files

    def _process_txt(self, file_path: str) -> List[Dict]:
        """
        Extract content from a text file.

        Args:
            file_path: Path to the text file

        Returns:
            List with a single page content dictionary
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            page_data = {"page_number": 1, "text": text.strip(), "tables": []}

            return [page_data]
        except Exception as e:
            raise Exception(f"Error processing text file: {str(e)}")

    def save_processed_content(
        self, pages_content: List[Dict], output_path: str
    ) -> None:
        """
        Save processed content to a JSON file.

        Args:
            pages_content: List of page content dictionaries
            output_path: Path to save the JSON file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pages_content, f, ensure_ascii=False, indent=2)
