"""
Conversation routes for the API.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.models.database import get_db
from app.api.schemas import schemas
from app.api.services.conversation_service import ConversationService

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", response_model=List[schemas.ConversationResponse])
def get_conversations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get all conversations.
    """
    conversations = ConversationService.get_conversations(db, skip=skip, limit=limit)
    return conversations


@router.get("/{conversation_id}", response_model=schemas.ConversationResponse)
def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """
    Get a conversation by ID.
    """
    conversation = ConversationService.get_conversation(db, conversation_id=conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.post("/", response_model=schemas.ConversationResponse)
def create_conversation(conversation: schemas.ConversationCreate, db: Session = Depends(get_db)):
    """
    Create a new conversation.
    """
    return ConversationService.create_conversation(db, conversation)


@router.put("/{conversation_id}", response_model=schemas.ConversationResponse)
def update_conversation(conversation_id: int, conversation: schemas.ConversationBase, db: Session = Depends(get_db)):
    """
    Update a conversation.
    """
    db_conversation = ConversationService.update_conversation(db, conversation_id, conversation)
    if db_conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return db_conversation


@router.delete("/{conversation_id}", response_model=bool)
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """
    Delete a conversation.
    """
    result = ConversationService.delete_conversation(db, conversation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return result


@router.get("/{conversation_id}/messages", response_model=List[schemas.MessageResponse])
def get_conversation_messages(conversation_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get messages for a conversation.
    """
    conversation = ConversationService.get_conversation(db, conversation_id=conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = ConversationService.get_messages(db, conversation_id=conversation_id, skip=skip, limit=limit)
    return messages
