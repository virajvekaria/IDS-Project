"""
Conversation service for handling conversation operations.
"""

from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session

from app.api.models.models import Conversation, Message, MessageReference
from app.api.schemas import schemas


class ConversationService:
    """
    Service for handling conversation operations.
    """

    @staticmethod
    def get_conversations(
        db: Session, skip: int = 0, limit: int = 100
    ) -> List[Conversation]:
        """
        Get all conversations.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of conversations
        """
        return db.query(Conversation).offset(skip).limit(limit).all()

    @staticmethod
    def get_conversation(db: Session, conversation_id: int) -> Optional[Conversation]:
        """
        Get a conversation by ID.

        Args:
            db: Database session
            conversation_id: Conversation ID

        Returns:
            Conversation or None
        """
        return db.query(Conversation).filter(Conversation.id == conversation_id).first()

    @staticmethod
    def create_conversation(
        db: Session, conversation: schemas.ConversationCreate
    ) -> Conversation:
        """
        Create a new conversation.

        Args:
            db: Database session
            conversation: Conversation creation schema

        Returns:
            Created conversation
        """
        db_conversation = Conversation(**conversation.model_dump())
        db.add(db_conversation)
        db.commit()
        db.refresh(db_conversation)
        return db_conversation

    @staticmethod
    def update_conversation(
        db: Session, conversation_id: int, conversation: schemas.ConversationBase
    ) -> Optional[Conversation]:
        """
        Update a conversation.

        Args:
            db: Database session
            conversation_id: Conversation ID
            conversation: Conversation update schema

        Returns:
            Updated conversation or None
        """
        db_conversation = (
            db.query(Conversation).filter(Conversation.id == conversation_id).first()
        )
        if db_conversation:
            for key, value in conversation.model_dump().items():
                setattr(db_conversation, key, value)
            db.commit()
            db.refresh(db_conversation)
        return db_conversation

    @staticmethod
    def delete_conversation(db: Session, conversation_id: int) -> bool:
        """
        Delete a conversation.

        Args:
            db: Database session
            conversation_id: Conversation ID

        Returns:
            True if deleted, False otherwise
        """
        db_conversation = (
            db.query(Conversation).filter(Conversation.id == conversation_id).first()
        )
        if db_conversation:
            db.delete(db_conversation)
            db.commit()
            return True
        return False

    @staticmethod
    def get_messages(
        db: Session, conversation_id: int, skip: int = 0, limit: int = 100
    ) -> List[Message]:
        """
        Get messages for a conversation.

        Args:
            db: Database session
            conversation_id: Conversation ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of messages
        """
        return (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def create_message(db: Session, message: schemas.MessageCreate) -> Message:
        """
        Create a new message.

        Args:
            db: Database session
            message: Message creation schema

        Returns:
            Created message
        """
        db_message = Message(**message.model_dump())
        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        return db_message

    @staticmethod
    def update_message_content(
        db: Session, message_id: int, content: str
    ) -> Optional[Message]:
        """
        Update a message's content.

        Args:
            db: Database session
            message_id: Message ID
            content: New message content

        Returns:
            Updated message or None
        """
        db_message = db.query(Message).filter(Message.id == message_id).first()
        if db_message:
            db_message.content = content
            db.commit()
            db.refresh(db_message)
        return db_message

    @staticmethod
    def get_message_references(db: Session, message_id: int) -> List[MessageReference]:
        """
        Get references for a message.

        Args:
            db: Database session
            message_id: Message ID

        Returns:
            List of message references
        """
        return (
            db.query(MessageReference)
            .filter(MessageReference.message_id == message_id)
            .all()
        )

    @staticmethod
    def create_message_reference(
        db: Session, reference: schemas.MessageReferenceCreate
    ) -> MessageReference:
        """
        Create a new message reference.

        Args:
            db: Database session
            reference: Message reference creation schema

        Returns:
            Created message reference
        """
        db_reference = MessageReference(**reference.model_dump())
        db.add(db_reference)
        db.commit()
        db.refresh(db_reference)
        return db_reference
