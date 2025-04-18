"""
Search routes for the API.
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse

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
    db: Session = Depends(get_db),
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
        use_hybrid=query.use_hybrid,
    )

    # Convert to response format
    results = []
    for result in search_results:
        results.append(
            schemas.SearchResult(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                page_number=result["page_number"],
                text=result["text"],
                score=result.get("score", 0.0),
            )
        )

    return schemas.SearchResponse(query=query.query, results=results, answer=answer)


@router.post("/chat")
def chat(
    request: schemas.ChatRequest,
    index_name: str = "default_index",
    stream: bool = False,
    db: Session = Depends(get_db),
):
    """
    Process a chat query.
    """
    # Check if index exists
    index_store = db.query(IndexStore).filter(IndexStore.name == index_name).first()
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

    # Convert to response format
    results = []
    for result in references:
        results.append(
            schemas.SearchResult(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                page_number=result["page_number"],
                text=result["text"],
                score=result.get("score", 0.0),
            )
        )

    if stream:
        # Return a streaming response
        from fastapi.responses import StreamingResponse

        def generate():
            # Yield the initial response with conversation ID and references
            initial_data = {
                "conversation_id": conversation_id,
                "references": [r.dict() for r in results],
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
        return schemas.ChatResponse(
            conversation_id=conversation_id, message=answer, references=results
        )
