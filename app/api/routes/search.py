"""
Search routes for the API.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.models.database import get_db
from app.api.models.models import IndexStore
from app.api.schemas import schemas
from app.api.services.search_service import SearchService

router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=schemas.SearchResponse)
def search(
    query: schemas.SearchQuery,
    index_name: str = "default_index",
    db: Session = Depends(get_db)
):
    """
    Search for documents.
    """
    # Check if index exists
    index_store = db.query(IndexStore).filter(IndexStore.name == index_name).first()
    if not index_store:
        raise HTTPException(status_code=404, detail="Index not found")
    
    # Perform search
    search_results, answer = SearchService.search(
        db,
        query.query,
        index_name=index_name,
        top_k=query.top_k,
        use_hybrid=query.use_hybrid
    )
    
    # Convert to response format
    results = []
    for result in search_results:
        results.append(schemas.SearchResult(
            chunk_id=result["chunk_id"],
            document_id=result["document_id"],
            page_number=result["page_number"],
            text=result["text"],
            score=result.get("score", 0.0)
        ))
    
    return schemas.SearchResponse(
        query=query.query,
        results=results,
        answer=answer
    )


@router.post("/chat", response_model=schemas.ChatResponse)
def chat(
    request: schemas.ChatRequest,
    index_name: str = "default_index",
    db: Session = Depends(get_db)
):
    """
    Process a chat query.
    """
    # Check if index exists
    index_store = db.query(IndexStore).filter(IndexStore.name == index_name).first()
    if not index_store:
        raise HTTPException(status_code=404, detail="Index not found")
    
    # Process chat query
    conversation_id, answer, references = SearchService.chat(
        db,
        request.message,
        conversation_id=request.conversation_id,
        index_name=index_name
    )
    
    # Convert to response format
    results = []
    for result in references:
        results.append(schemas.SearchResult(
            chunk_id=result["chunk_id"],
            document_id=result["document_id"],
            page_number=result["page_number"],
            text=result["text"],
            score=result.get("score", 0.0)
        ))
    
    return schemas.ChatResponse(
        conversation_id=conversation_id,
        message=answer,
        references=results
    )
