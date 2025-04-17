"""
Document service for handling document operations.
"""

import os
import json
import nltk
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy.orm import Session

from app.api.models.models import Document, Chunk, ProcessingJob, IndexStore
from app.api.schemas import schemas
from app.core.document_processor import DocumentProcessor
from app.core.chunking import DocumentChunker
from app.core.indexing import DocumentIndexer

import config

# Ensure NLTK data is downloaded
try:
    nltk.download("punkt", quiet=True)
except Exception as e:
    print(f"NLTK punkt download failed: {e}. This may affect text processing.")


class DocumentService:
    """
    Service for handling document operations.
    """

    @staticmethod
    def get_documents(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
        """
        Get all documents.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of documents
        """
        return db.query(Document).offset(skip).limit(limit).all()

    @staticmethod
    def get_document(db: Session, document_id: int) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            db: Database session
            document_id: Document ID

        Returns:
            Document or None
        """
        return db.query(Document).filter(Document.id == document_id).first()

    @staticmethod
    def create_document(db: Session, document: schemas.DocumentCreate) -> Document:
        """
        Create a new document.

        Args:
            db: Database session
            document: Document creation schema

        Returns:
            Created document
        """
        db_document = Document(**document.model_dump())
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        return db_document

    @staticmethod
    def update_document(
        db: Session, document_id: int, document: schemas.DocumentBase
    ) -> Optional[Document]:
        """
        Update a document.

        Args:
            db: Database session
            document_id: Document ID
            document: Document update schema

        Returns:
            Updated document or None
        """
        db_document = db.query(Document).filter(Document.id == document_id).first()
        if db_document:
            for key, value in document.model_dump().items():
                setattr(db_document, key, value)
            db_document.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(db_document)
        return db_document

    @staticmethod
    def delete_document(db: Session, document_id: int) -> bool:
        """
        Delete a document.

        Args:
            db: Database session
            document_id: Document ID

        Returns:
            True if deleted, False otherwise
        """
        db_document = db.query(Document).filter(Document.id == document_id).first()
        if db_document:
            db.delete(db_document)
            db.commit()
            return True
        return False

    @staticmethod
    def get_document_chunks(
        db: Session, document_id: int, skip: int = 0, limit: int = 100
    ) -> List[Chunk]:
        """
        Get chunks for a document.

        Args:
            db: Database session
            document_id: Document ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of chunks
        """
        return (
            db.query(Chunk)
            .filter(Chunk.document_id == document_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_document_jobs(
        db: Session, document_id: int, skip: int = 0, limit: int = 100
    ) -> List[ProcessingJob]:
        """
        Get processing jobs for a document.

        Args:
            db: Database session
            document_id: Document ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of processing jobs
        """
        return (
            db.query(ProcessingJob)
            .filter(ProcessingJob.document_id == document_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def create_processing_job(
        db: Session, job: schemas.ProcessingJobCreate
    ) -> ProcessingJob:
        """
        Create a new processing job.

        Args:
            db: Database session
            job: Processing job creation schema

        Returns:
            Created processing job
        """
        db_job = ProcessingJob(**job.model_dump())
        db.add(db_job)
        db.commit()
        db.refresh(db_job)
        return db_job

    @staticmethod
    def update_processing_job(
        db: Session, job_id: int, status: str, error_message: Optional[str] = None
    ) -> Optional[ProcessingJob]:
        """
        Update a processing job.

        Args:
            db: Database session
            job_id: Processing job ID
            status: New status
            error_message: Error message (if any)

        Returns:
            Updated processing job or None
        """
        db_job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if db_job:
            db_job.status = status
            if status == "processing" and not db_job.started_at:
                db_job.started_at = datetime.utcnow()
            elif status in ["completed", "failed"]:
                db_job.completed_at = datetime.utcnow()
            if error_message:
                db_job.error_message = error_message
            db.commit()
            db.refresh(db_job)
        return db_job

    @staticmethod
    def create_chunks(db: Session, chunks: List[Dict], document_id: int) -> List[Chunk]:
        """
        Create chunks for a document.

        Args:
            db: Database session
            chunks: List of chunk dictionaries
            document_id: Document ID

        Returns:
            List of created chunks
        """
        db_chunks = []
        for chunk in chunks:
            db_chunk = Chunk(
                chunk_id=chunk["chunk_id"],
                document_id=document_id,
                page_number=chunk["page_number"],
                text=chunk["text"],
            )
            db.add(db_chunk)
            db_chunks.append(db_chunk)
        db.commit()
        for chunk in db_chunks:
            db.refresh(chunk)
        return db_chunks

    @staticmethod
    def get_or_create_index_store(
        db: Session, name: str, index_data: Dict
    ) -> IndexStore:
        """
        Get or create an index store.

        Args:
            db: Database session
            name: Index store name
            index_data: Index data dictionary

        Returns:
            Index store
        """
        db_index = db.query(IndexStore).filter(IndexStore.name == name).first()
        if db_index:
            # Update existing index
            for key, value in index_data.items():
                setattr(db_index, key, value)
            db_index.updated_at = datetime.utcnow()
        else:
            # Create new index
            db_index = IndexStore(name=name, **index_data)
            db.add(db_index)
        db.commit()
        db.refresh(db_index)
        return db_index

    @staticmethod
    def get_index_store(db: Session, name: str) -> Optional[IndexStore]:
        """
        Get an index store by name.

        Args:
            db: Database session
            name: Index store name

        Returns:
            Index store or None
        """
        return db.query(IndexStore).filter(IndexStore.name == name).first()

    @staticmethod
    def get_all_index_stores(db: Session) -> List[IndexStore]:
        """
        Get all index stores.

        Args:
            db: Database session

        Returns:
            List of index stores
        """
        return db.query(IndexStore).all()

    @staticmethod
    def process_document(
        db: Session,
        document_id: int,
        chunk_size: int = 200,
        chunk_overlap: int = 1,
        index_name: Optional[str] = None,
    ) -> Tuple[bool, List[int], Optional[str]]:
        """
        Process a document (extract, chunk, and index).

        Args:
            db: Database session
            document_id: Document ID
            chunk_size: Chunk size in words
            chunk_overlap: Chunk overlap in sentences
            index_name: Index store name (optional)

        Returns:
            Tuple of (success, job_ids, error_message)
        """
        # Get document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return False, [], "Document not found"

        # Create processing jobs
        extraction_job = ProcessingJob(
            document_id=document_id, job_type="extraction", status="pending"
        )
        chunking_job = ProcessingJob(
            document_id=document_id, job_type="chunking", status="pending"
        )
        indexing_job = ProcessingJob(
            document_id=document_id, job_type="indexing", status="pending"
        )

        db.add(extraction_job)
        db.add(chunking_job)
        db.add(indexing_job)
        db.commit()
        db.refresh(extraction_job)
        db.refresh(chunking_job)
        db.refresh(indexing_job)

        job_ids = [extraction_job.id, chunking_job.id, indexing_job.id]

        try:
            # 1. Extract document content
            DocumentService.update_processing_job(db, extraction_job.id, "processing")

            processor = DocumentProcessor()
            pages_content, metadata = processor.process_document(document.file_path)

            # Update document metadata
            document.page_count = metadata.get("page_count", 0)
            document.processed = True
            document.updated_at = datetime.utcnow()
            db.commit()

            # Save extracted content
            processed_dir = config.PROCESSED_DIR
            os.makedirs(processed_dir, exist_ok=True)
            processed_path = os.path.join(processed_dir, f"{document.id}_content.json")

            with open(processed_path, "w", encoding="utf-8") as f:
                json.dump(pages_content, f, ensure_ascii=False, indent=2)

            DocumentService.update_processing_job(db, extraction_job.id, "completed")

            # 2. Chunk document content
            DocumentService.update_processing_job(db, chunking_job.id, "processing")

            chunker = DocumentChunker()
            chunks = chunker.chunk_document(
                pages_content,
                source_doc=document.filename,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

            # Save chunks to database
            DocumentService.create_chunks(db, chunks, document.id)

            # Save chunks to file
            chunks_path = os.path.join(processed_dir, f"{document.id}_chunks.json")

            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            DocumentService.update_processing_job(db, chunking_job.id, "completed")

            # 3. Index document chunks
            DocumentService.update_processing_job(db, indexing_job.id, "processing")

            # Determine index name
            if not index_name:
                index_name = "default_index"

            # Check if index exists
            index_store = (
                db.query(IndexStore).filter(IndexStore.name == index_name).first()
            )

            indexes_dir = config.INDEXES_DIR
            os.makedirs(indexes_dir, exist_ok=True)

            indexer = DocumentIndexer()

            if index_store:
                # Update existing index
                index_files = indexer.update_index(
                    index_store.index_path, index_store.metadata_path, chunks
                )

                # Update index store
                index_store.chunk_count += len(chunks)
                index_store.updated_at = datetime.utcnow()
                db.commit()
            else:
                # Create new index
                index_prefix = os.path.join(indexes_dir, index_name)

                index_files = indexer.build_index(chunks_path, index_prefix)

                # Create index store
                index_data = {
                    "index_path": index_files["index_path"],
                    "metadata_path": index_files["metadata_path"],
                    "embedding_model": indexer.embedding_model,
                    "index_type": indexer.index_type,
                    "use_hybrid": indexer.use_hybrid,
                    "chunk_count": len(chunks),
                }

                if "vectorizer_path" in index_files:
                    index_data["vectorizer_path"] = index_files["vectorizer_path"]
                    index_data["tfidf_path"] = index_files["tfidf_path"]

                DocumentService.get_or_create_index_store(db, index_name, index_data)

            # Update document
            document.indexed = True
            document.updated_at = datetime.utcnow()
            db.commit()

            DocumentService.update_processing_job(db, indexing_job.id, "completed")

            return True, job_ids, None

        except Exception as e:
            error_message = str(e)

            # Update jobs with error
            for job_id in job_ids:
                job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
                if job and job.status == "processing":
                    DocumentService.update_processing_job(
                        db, job.id, "failed", error_message
                    )
                elif job and job.status == "pending":
                    DocumentService.update_processing_job(
                        db, job.id, "failed", "Previous step failed"
                    )

            return False, job_ids, error_message
