"""
Search service for handling search operations.
"""

from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy.orm import Session

from app.api.models.models import Document, Chunk, IndexStore
from app.api.schemas import schemas
from app.core.retrieval import DocumentRetriever

import config


class SearchService:
    """
    Service for handling search operations.
    """

    @staticmethod
    def search(
        db: Session,
        query: str,
        index_name: str = "default_index",
        top_k: int = 5,
        use_hybrid: Optional[bool] = None,
        stream: bool = False,
    ):
        """
        Search for documents.

        Args:
            db: Database session
            query: Search query
            index_name: Index store name
            top_k: Number of top results to return
            use_hybrid: Whether to use hybrid search (overrides index setting if provided)
            stream: Whether to stream the response

        Returns:
            If stream=False: Tuple of (search_results, answer)
            If stream=True: Tuple of (search_results, generator yielding response chunks)
        """
        # Get index store
        index_store = db.query(IndexStore).filter(IndexStore.name == index_name).first()
        if not index_store:
            return [], "Index not found"

        # Determine whether to use hybrid search
        if use_hybrid is None:
            use_hybrid = index_store.use_hybrid

        # Create retriever
        retriever = DocumentRetriever()

        # Perform search
        search_results, answer = retriever.search(
            query,
            index_store.index_path,
            index_store.metadata_path,
            use_hybrid=use_hybrid,
            top_k=top_k,
            stream=stream,
        )

        # Enhance search results with document information
        enhanced_results = []
        for result in search_results:
            # Get document ID from chunk
            document = (
                db.query(Document)
                .filter(Document.filename == result["source_doc"])
                .first()
            )
            if document:
                result["document_id"] = document.id
                enhanced_results.append(result)

        return enhanced_results, answer

    @staticmethod
    def chat(
        db: Session,
        query: str,
        conversation_id: Optional[int] = None,
        index_name: str = "default_index",
        top_k: int = 3,
        use_hybrid: Optional[bool] = None,
        stream: bool = False,
    ):
        """
        Process a chat query.

        Args:
            db: Database session
            query: Chat query
            conversation_id: Conversation ID (optional)
            index_name: Index store name
            top_k: Number of top results to return
            use_hybrid: Whether to use hybrid search (overrides index setting if provided)
            stream: Whether to stream the response

        Returns:
            If stream=False: Tuple of (conversation_id, answer, references)
            If stream=True: Tuple of (conversation_id, generator yielding response chunks, references)
        """
        from app.api.models.models import Conversation, Message
        from app.api.services.conversation_service import ConversationService

        # Get index store
        index_store = db.query(IndexStore).filter(IndexStore.name == index_name).first()
        if not index_store:
            return conversation_id or 0, "Index not found", []

        # Determine whether to use hybrid search
        if use_hybrid is None:
            use_hybrid = index_store.use_hybrid

        # Get or create conversation
        if conversation_id:
            conversation = (
                db.query(Conversation)
                .filter(Conversation.id == conversation_id)
                .first()
            )
            if not conversation:
                # Create new conversation if ID not found
                conversation = ConversationService.create_conversation(
                    db, schemas.ConversationCreate()
                )
                conversation_id = conversation.id
        else:
            # Create new conversation
            conversation = ConversationService.create_conversation(
                db, schemas.ConversationCreate()
            )
            conversation_id = conversation.id

        # Get conversation history
        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .all()
        )
        conversation_history = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # Create retriever
        retriever = DocumentRetriever()

        # Process chat query
        answer, retrieved_docs = retriever.chat(
            query,
            conversation_history,
            index_store.index_path,
            index_store.metadata_path,
            use_hybrid=use_hybrid,
            top_k=top_k,
            stream=stream,
        )

        # Add user message to conversation
        user_message = ConversationService.create_message(
            db,
            schemas.MessageCreate(
                conversation_id=conversation_id, role="user", content=query
            ),
        )

        # Add assistant message to conversation
        if not stream:
            # For non-streaming responses, save the complete answer
            assistant_message = ConversationService.create_message(
                db,
                schemas.MessageCreate(
                    conversation_id=conversation_id, role="assistant", content=answer
                ),
            )
        else:
            # For streaming responses, save a placeholder message
            # The actual content will be updated by the client
            assistant_message = ConversationService.create_message(
                db,
                schemas.MessageCreate(
                    conversation_id=conversation_id,
                    role="assistant",
                    content="[Streaming response]",
                ),
            )

        # Add references to assistant message
        for doc in retrieved_docs:
            # Get document ID from chunk
            document = (
                db.query(Document)
                .filter(Document.filename == doc["source_doc"])
                .first()
            )
            if document:
                # Get chunk ID
                chunk = (
                    db.query(Chunk)
                    .filter(
                        Chunk.document_id == document.id,
                        Chunk.page_number == doc["page_number"],
                        Chunk.text == doc["text"],
                    )
                    .first()
                )

                if chunk:
                    ConversationService.create_message_reference(
                        db,
                        schemas.MessageReferenceCreate(
                            message_id=assistant_message.id,
                            chunk_id=chunk.id,
                            relevance_score=doc.get("score", 0.0),
                        ),
                    )

        # Enhance retrieved docs with document information
        enhanced_results = []
        for result in retrieved_docs:
            # Get document ID from chunk
            document = (
                db.query(Document)
                .filter(Document.filename == result["source_doc"])
                .first()
            )
            if document:
                result["document_id"] = document.id
                enhanced_results.append(result)

        return conversation_id, answer, enhanced_results
