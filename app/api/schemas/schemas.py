"""
Pydantic schemas for API request and response validation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


# Document schemas
class DocumentBase(BaseModel):
    """Base document schema."""

    filename: str
    file_type: str
    file_size: int
    page_count: int = 0


class DocumentCreate(DocumentBase):
    """Document creation schema."""

    file_path: str


class DocumentResponse(DocumentBase):
    """Document response schema."""

    id: int
    processed: bool
    indexed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# Chunk schemas
class ChunkBase(BaseModel):
    """Base chunk schema."""

    chunk_id: str
    page_number: int
    text: str


class ChunkCreate(ChunkBase):
    """Chunk creation schema."""

    document_id: int


class ChunkResponse(ChunkBase):
    """Chunk response schema."""

    id: int
    document_id: int
    created_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# Processing job schemas
class ProcessingJobBase(BaseModel):
    """Base processing job schema."""

    job_type: str
    status: str
    error_message: Optional[str] = None


class ProcessingJobCreate(ProcessingJobBase):
    """Processing job creation schema."""

    document_id: int


class ProcessingJobResponse(ProcessingJobBase):
    """Processing job response schema."""

    id: int
    document_id: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# Conversation schemas
class ConversationBase(BaseModel):
    """Base conversation schema."""

    title: Optional[str] = None


class ConversationCreate(ConversationBase):
    """Conversation creation schema."""

    pass


class ConversationResponse(ConversationBase):
    """Conversation response schema."""

    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# Message schemas
class MessageBase(BaseModel):
    """Base message schema."""

    role: str
    content: str


class MessageCreate(MessageBase):
    """Message creation schema."""

    conversation_id: int


class MessageResponse(MessageBase):
    """Message response schema."""

    id: int
    conversation_id: int
    created_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# Message reference schemas
class MessageReferenceBase(BaseModel):
    """Base message reference schema."""

    relevance_score: Optional[float] = None


class MessageReferenceCreate(MessageReferenceBase):
    """Message reference creation schema."""

    message_id: int
    chunk_id: int


class MessageReferenceResponse(MessageReferenceBase):
    """Message reference response schema."""

    id: int
    message_id: int
    chunk_id: int
    created_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# Index store schemas
class IndexStoreBase(BaseModel):
    """Base index store schema."""

    name: str
    index_path: str
    metadata_path: str
    vectorizer_path: Optional[str] = None
    tfidf_path: Optional[str] = None
    embedding_model: str
    index_type: str
    use_hybrid: bool = False
    chunk_count: int = 0


class IndexStoreCreate(IndexStoreBase):
    """Index store creation schema."""

    pass


class IndexStoreResponse(IndexStoreBase):
    """Index store response schema."""

    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# Search schemas
class SearchQuery(BaseModel):
    """Search query schema."""

    query: str
    top_k: int = 5
    use_hybrid: bool = True


class SearchResult(BaseModel):
    """Search result schema."""

    chunk_id: str
    document_id: int
    page_number: int
    text: str
    score: float


class SearchResponse(BaseModel):
    """Search response schema."""

    query: str
    results: List[SearchResult]
    answer: str


# Chat schemas
class ChatMessage(BaseModel):
    """Chat message schema."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request schema."""

    conversation_id: Optional[int] = None
    message: str


class ChatResponse(BaseModel):
    """Chat response schema."""

    conversation_id: int
    message: str
    references: List[SearchResult]


# Document processing schemas
class ProcessDocumentRequest(BaseModel):
    """Process document request schema."""

    document_id: int
    chunk_size: int = 200
    chunk_overlap: int = 1
    index_name: Optional[str] = None


class ProcessDocumentResponse(BaseModel):
    """Process document response schema."""

    document_id: int
    status: str
    job_ids: List[int]
