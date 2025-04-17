"""
Initialize documents from the PDFs folder.
"""

import os
import sys
import json
import nltk
import logging
import traceback
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("document_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger("init_documents")

# Ensure NLTK data is downloaded
try:
    nltk.download("punkt", quiet=True)
except Exception as e:
    logger.warning(f"NLTK punkt download failed: {e}. This may affect text processing.")

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.api.models.database import SessionLocal, engine, Base
from app.api.models.models import Document, ProcessingJob
from app.api.schemas import schemas
from app.api.services.document_service import DocumentService
import config


def init_documents():
    """
    Initialize documents from the PDFs folder.
    Process all PDF files and index them for search.
    """
    logger.info("Initializing documents from PDFs folder...")
    start_time = datetime.now()

    # Create database tables if they don't exist
    Base.metadata.create_all(bind=engine)

    # Get a database session
    db = SessionLocal()

    try:
        # Check if the PDFs folder exists
        pdfs_dir = Path("PDFs")
        if not pdfs_dir.exists() or not pdfs_dir.is_dir():
            logger.error(f"PDFs folder not found at {pdfs_dir.absolute()}")
            return

        # Get all PDF files in the PDFs folder
        pdf_files = list(pdfs_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in the PDFs folder")
            return

        logger.info(f"Found {len(pdf_files)} PDF files in the PDFs folder")

        # Ensure all required directories exist
        for directory in [
            config.DOCUMENTS_DIR,
            config.PROCESSED_DIR,
            config.INDEXES_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

        # Process each PDF file
        successful_docs = 0
        failed_docs = 0

        for pdf_file in pdf_files:
            try:
                # Check if the document already exists in the database
                existing_doc = (
                    db.query(Document)
                    .filter(Document.filename == pdf_file.name)
                    .first()
                )

                # Skip if already processed and indexed, unless forced
                if existing_doc and existing_doc.processed and existing_doc.indexed:
                    logger.info(
                        f"Document {pdf_file.name} already processed and indexed"
                    )
                    successful_docs += 1
                    continue

                logger.info(f"Processing {pdf_file.name}...")

                # Copy the file to the documents directory if it doesn't exist
                dest_path = os.path.join(config.DOCUMENTS_DIR, pdf_file.name)
                if not os.path.exists(dest_path):
                    with open(pdf_file, "rb") as src_file:
                        with open(dest_path, "wb") as dest_file:
                            dest_file.write(src_file.read())
                    logger.debug(f"Copied {pdf_file.name} to documents directory")

                # Create or update document record
                file_size = os.path.getsize(pdf_file)
                if existing_doc:
                    # Update existing document
                    existing_doc.file_path = dest_path
                    existing_doc.file_size = file_size
                    existing_doc.processed = False
                    existing_doc.indexed = False
                    db.commit()
                    document_id = existing_doc.id
                    logger.debug(
                        f"Updated existing document record for {pdf_file.name}"
                    )
                else:
                    # Create new document
                    document_data = schemas.DocumentCreate(
                        filename=pdf_file.name,
                        file_path=dest_path,
                        file_type="pdf",
                        file_size=file_size,
                    )
                    document = DocumentService.create_document(db, document_data)
                    document_id = document.id
                    logger.debug(f"Created new document record for {pdf_file.name}")

                # Process the document with improved parameters
                success, job_ids, error = DocumentService.process_document(
                    db,
                    document_id,
                    chunk_size=config.DEFAULT_CHUNK_SIZE,
                    chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
                    index_name="default_index",
                )

                if success:
                    logger.info(f"Successfully processed {pdf_file.name}")
                    successful_docs += 1
                else:
                    logger.error(f"Failed to process {pdf_file.name}: {error}")
                    failed_docs += 1

            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                logger.debug(traceback.format_exc())
                failed_docs += 1

        # Log summary
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(
            f"Document initialization complete in {processing_time:.2f} seconds"
        )
        logger.info(f"Successfully processed {successful_docs} documents")
        if failed_docs > 0:
            logger.warning(f"Failed to process {failed_docs} documents")

    except Exception as e:
        logger.error(f"Error during document initialization: {str(e)}")
        logger.debug(traceback.format_exc())
    finally:
        db.close()


if __name__ == "__main__":
    init_documents()
