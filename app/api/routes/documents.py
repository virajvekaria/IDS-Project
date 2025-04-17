"""
Document routes for the API.
"""
import os
import shutil
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session

from app.api.models.database import get_db
from app.api.models.models import Document, ProcessingJob
from app.api.schemas import schemas
from app.api.services.document_service import DocumentService

import config

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", response_model=List[schemas.DocumentResponse])
def get_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get all documents.
    """
    documents = DocumentService.get_documents(db, skip=skip, limit=limit)
    return documents


@router.get("/{document_id}", response_model=schemas.DocumentResponse)
def get_document(document_id: int, db: Session = Depends(get_db)):
    """
    Get a document by ID.
    """
    document = DocumentService.get_document(db, document_id=document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.post("/", response_model=schemas.DocumentResponse)
async def create_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a new document.
    """
    # Check file size
    file_size = 0
    contents = await file.read()
    file_size = len(contents)
    await file.seek(0)
    
    if file_size > config.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    # Check file extension
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext[1:] not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not supported")
    
    # Save file
    file_path = os.path.join(config.DOCUMENTS_DIR, filename)
    
    # Check if file already exists
    if os.path.exists(file_path):
        # Add timestamp to filename
        import time
        timestamp = int(time.time())
        filename = f"{os.path.splitext(filename)[0]}_{timestamp}{file_ext}"
        file_path = os.path.join(config.DOCUMENTS_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(contents)
    
    # Create document record
    document_data = schemas.DocumentCreate(
        filename=filename,
        file_path=file_path,
        file_type=file_ext[1:],
        file_size=file_size,
        page_count=0
    )
    
    document = DocumentService.create_document(db, document_data)
    return document


@router.delete("/{document_id}", response_model=bool)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    Delete a document.
    """
    document = DocumentService.get_document(db, document_id=document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file if it exists
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    # Delete processed files if they exist
    processed_path = os.path.join(config.PROCESSED_DIR, f"{document_id}_content.json")
    if os.path.exists(processed_path):
        os.remove(processed_path)
        
    chunks_path = os.path.join(config.PROCESSED_DIR, f"{document_id}_chunks.json")
    if os.path.exists(chunks_path):
        os.remove(chunks_path)
    
    # Delete document record
    return DocumentService.delete_document(db, document_id=document_id)


@router.get("/{document_id}/chunks", response_model=List[schemas.ChunkResponse])
def get_document_chunks(document_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get chunks for a document.
    """
    document = DocumentService.get_document(db, document_id=document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunks = DocumentService.get_document_chunks(db, document_id=document_id, skip=skip, limit=limit)
    return chunks


@router.get("/{document_id}/jobs", response_model=List[schemas.ProcessingJobResponse])
def get_document_jobs(document_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get processing jobs for a document.
    """
    document = DocumentService.get_document(db, document_id=document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    jobs = DocumentService.get_document_jobs(db, document_id=document_id, skip=skip, limit=limit)
    return jobs


def process_document_background(document_id: int, chunk_size: int, chunk_overlap: int, index_name: Optional[str], db: Session):
    """
    Process a document in the background.
    """
    DocumentService.process_document(db, document_id, chunk_size, chunk_overlap, index_name)


@router.post("/process", response_model=schemas.ProcessDocumentResponse)
def process_document(
    request: schemas.ProcessDocumentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Process a document (extract, chunk, and index).
    """
    document = DocumentService.get_document(db, document_id=request.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Create processing jobs
    extraction_job = ProcessingJob(
        document_id=request.document_id,
        job_type="extraction",
        status="pending"
    )
    chunking_job = ProcessingJob(
        document_id=request.document_id,
        job_type="chunking",
        status="pending"
    )
    indexing_job = ProcessingJob(
        document_id=request.document_id,
        job_type="indexing",
        status="pending"
    )
    
    db.add(extraction_job)
    db.add(chunking_job)
    db.add(indexing_job)
    db.commit()
    db.refresh(extraction_job)
    db.refresh(chunking_job)
    db.refresh(indexing_job)
    
    job_ids = [extraction_job.id, chunking_job.id, indexing_job.id]
    
    # Process document in background
    background_tasks.add_task(
        process_document_background,
        request.document_id,
        request.chunk_size,
        request.chunk_overlap,
        request.index_name,
        db
    )
    
    return schemas.ProcessDocumentResponse(
        document_id=request.document_id,
        status="processing",
        job_ids=job_ids
    )
