"""
Configuration settings for the Document Intelligence Search System (DISS).
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DIR = DATA_DIR / "processed"
INDEXES_DIR = DATA_DIR / "indexes"

# Create directories if they don't exist
for dir_path in [DOCUMENTS_DIR, PROCESSED_DIR, INDEXES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/diss.db")

# Vector store settings
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
DEFAULT_CHUNK_SIZE = 1000  # Increased for better context
DEFAULT_CHUNK_OVERLAP = 200  # Increased for better context continuity
DEFAULT_VECTOR_WEIGHT = 0.7
USE_HYBRID_SEARCH = True  # Enable hybrid search for better results
INDEX_TYPE = "flat"  # "flat" or "ivf" - flat is better for smaller datasets

# LLM settings
DEFAULT_LLM_MODEL = "llama3:8b"  # Default Ollama model

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

# File upload settings
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "csv": "text/csv",
    "txt": "text/plain",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
}

# OCR settings
OCR_LANGUAGE = "eng"  # Default OCR language
