"""
React API routes for the Document Intelligence Search System (DISS).
"""

import os
import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

import config

from app.api.models.database import get_db
from app.api.models import models
from app.api.schemas import schemas
from app.api.services.document_service import DocumentService
from app.api.services.conversation_service import ConversationService
from app.api.services.search_service import SearchService

router = APIRouter(
    prefix="/react-api",
    tags=["react-api"],
    responses={404: {"description": "Not found"}},
)


@router.get("/documents", response_model=List[schemas.DocumentResponse])
def get_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get all documents for React frontend.
    """
    documents = DocumentService.get_documents(db, skip=skip, limit=limit)
    return documents


@router.get("/all-documents", response_model=List[schemas.DocumentResponse])
def get_all_documents(db: Session = Depends(get_db)):
    """
    Get all documents directly from the database.
    """
    # Direct database query to get all documents
    documents = db.query(models.Document).all()
    return documents


@router.delete("/documents/{document_id}", response_model=schemas.DocumentResponse)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    Delete a document and its associated files.
    """
    # Get the document
    document = (
        db.query(models.Document).filter(models.Document.id == document_id).first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get the file path
    file_path = document.file_path

    # Delete the document from the database
    db.delete(document)
    db.commit()

    # Delete the file if it exists
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            # Log the error but don't fail the request
            print(f"Error deleting file {file_path}: {e}")

    # Delete any chunks associated with the document
    chunks = (
        db.query(models.Chunk).filter(models.Chunk.document_id == document_id).all()
    )
    for chunk in chunks:
        db.delete(chunk)
    db.commit()

    return document


@router.post("/documents/upload", response_model=schemas.DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
):
    """
    Upload a new document for React frontend.
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
    os.makedirs(config.DOCUMENTS_DIR, exist_ok=True)
    file_path = os.path.join(config.DOCUMENTS_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(contents)

    # Create document record
    document_data = schemas.DocumentCreate(
        filename=filename,
        file_path=file_path,
        file_type=file_ext[1:],
        file_size=file_size,
        page_count=0,  # Will be updated during processing
        processed=False,
        indexed=False,
    )

    document = DocumentService.create_document(db, document_data)

    # Always process document in background
    if background_tasks:
        background_tasks.add_task(
            DocumentService.process_document,
            db,
            document.id,
            chunk_size=config.DEFAULT_CHUNK_SIZE,
            chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
            index_name="default_index",
        )
    else:
        # If background_tasks is not available, process synchronously
        DocumentService.process_document(
            db,
            document.id,
            chunk_size=config.DEFAULT_CHUNK_SIZE,
            chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
            index_name="default_index",
        )

    return document


@router.get("/conversations", response_model=List[schemas.ConversationResponse])
def get_conversations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get all conversations for React frontend.
    """
    conversations = ConversationService.get_conversations(db, skip=skip, limit=limit)
    return conversations


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=List[schemas.MessageResponse],
)
def get_conversation_messages(
    conversation_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    """
    Get messages for a conversation for React frontend.
    """
    conversation = ConversationService.get_conversation(
        db, conversation_id=conversation_id
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = ConversationService.get_messages(
        db, conversation_id=conversation_id, skip=skip, limit=limit
    )
    return messages


@router.post("/chat")
def chat(
    request: schemas.ChatRequest,
    index_name: str = "default_index",
    stream: bool = False,
    db: Session = Depends(get_db),
):
    """
    Process a chat query for React frontend.
    """
    # Check if index exists
    index_store = (
        db.query(models.IndexStore).filter(models.IndexStore.name == index_name).first()
    )
    if not index_store:
        raise HTTPException(status_code=404, detail="Index not found")

    message = request.message
    conv_id = request.conversation_id

    # Process chat query
    conversation_id, answer, references = SearchService.chat(
        db,
        message,
        conversation_id=conv_id,
        index_name=index_name,
        stream=stream,
    )

    # Convert references to dict for serialization
    reference_dicts = []
    if references and len(references) > 0:
        reference_dicts = (
            [ref.dict() for ref in references]
            if hasattr(references[0], "dict")
            else references
        )

    if stream:
        # Return a streaming response
        def generate():
            # Yield the initial response with conversation ID and references
            initial_data = {
                "conversation_id": conversation_id,
                "references": reference_dicts,
                "message": "",
            }
            yield f"data: {json.dumps(initial_data)}\n\n"

            # Stream the message content
            for chunk in answer:
                yield f"data: {json.dumps({'content': chunk})}\n\n"

            # Signal the end of the stream
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Return a regular response
        return {
            "conversation_id": conversation_id,
            "message": answer,
            "references": reference_dicts,
        }
