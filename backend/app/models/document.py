"""
Database models for documents and chunks with pgvector support.
"""
from datetime import datetime
from typing import List, Dict, Any
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Column, String, Integer, DateTime, ForeignKey,
    Text, ARRAY, Boolean, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.db.session import Base


class Document(Base):
    """
    Stores metadata about uploaded documents.
    """
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Organization
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Relationship:
    organization = relationship("Organization", back_populates="documents")

    # File metadata
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)

    # Classification
    doc_type = Column(String(50), nullable=False, index=True)
    department = Column(String(100), index=True)

    # Document details
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    has_tables = Column(Boolean, default=False)
    has_images = Column(Boolean, default=False)

    # Processing status
    status = Column(String(20), nullable=False, default="pending", index=True)
    processing_error = Column(Text, nullable=True)

    # Timestamps
    upload_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    processed_date = Column(DateTime, nullable=True)
    last_accessed = Column(DateTime, nullable=True)

    # Uploader
    uploaded_by = Column(UUID(as_uuid=True), nullable=False)

    # JSONB metadata
    doc_metadata = Column("metadata", JSONB, nullable=False, default=dict)

    # Relationship
    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    __table_args__ = (
        Index('idx_doc_type_dept', 'doc_type', 'department'),
        Index('idx_status_upload', 'status', 'upload_date'),
        Index('idx_metadata_gin', 'metadata', postgresql_using='gin'),
        Index('idx_org_status', 'organization_id', 'status'),
    )

    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"

    def to_dict(self):
        return {
            "id": str(self.id),
            "filename": self.filename,
            "doc_type": self.doc_type,
            "department": self.department,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "status": self.status,
            "upload_date": self.upload_date.isoformat() if self.upload_date else None,
            "metadata": self.doc_metadata,
        }


class Chunk(Base):
    """
    Stores individual text chunks with embeddings.
    """
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_length = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)

    chunk_type = Column(String(50), nullable=False, index=True)

    # Context
    page_numbers = Column(ARRAY(Integer), nullable=False, default=list)
    section_title = Column(String(512), nullable=True)
    preceding_context = Column(Text, nullable=True)

    # AI-generated intelligent title
    title = Column(String(200), nullable=True, index=True)

    # Embedding vector
    embedding = Column(Vector(1536), nullable=False)

    # JSONB chunk metadata
    chunk_metadata = Column("metadata", JSONB, nullable=False, default=dict)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index('idx_doc_chunk', 'document_id', 'chunk_index'),
        Index(
            'idx_embedding_cosine',
            'embedding',
            postgresql_using='ivfflat',
            postgresql_with={'lists': 100},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
        Index('idx_chunk_metadata_gin', 'metadata', postgresql_using='gin'),
        Index(
            'idx_content_fts',
            'content',
            postgresql_using='gin',
            postgresql_ops={'content': 'gin_trgm_ops'}
        ),
        Index('idx_chunk_type_doc', 'chunk_type', 'document_id'),
    )

    # Editing fields:

    is_edited = Column(Boolean, nullable=False, default=False, index=True)
    edited_at = Column(DateTime, nullable=True)
    edited_by = Column(UUID(as_uuid=True), nullable=True)
    original_content = Column(Text, nullable=True)
    edit_count = Column(Integer, nullable=False, default=0)

    def __repr__(self):
        return f"<Chunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"

    def to_dict(self, include_embedding=False, include_edit_info=False):
        result = {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "chunk_index": self.chunk_index,
            "content": self.content,
            "chunk_type": self.chunk_type,
            "page_numbers": self.page_numbers,
            "section_title": self.section_title,
            "title": self.title,
            "token_count": self.token_count,
            "metadata": self.chunk_metadata,
        }
        if include_embedding:
            result["embedding"] = self.embedding
        if include_edit_info:
            result["is_edited"] = self.is_edited
            result["edited_at"] = self.edited_at.isoformat() if self.edited_at else None
            result["edited_by"] = str(self.edited_by) if self.edited_by else None
            result["edit_count"] = self.edit_count
        return result

# Add new model at the end of the file:
class ChunkEditHistory(Base):
    """Track chunk edit history."""
    __tablename__ = "chunk_edit_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    chunk_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    edited_by = Column(UUID(as_uuid=True), nullable=False)
    edited_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    old_content = Column(Text, nullable=False)
    new_content = Column(Text, nullable=False)
    change_summary = Column(String(500), nullable=True)
    
    # FIXED: Use edit_metadata as attribute name, "metadata" as DB column name
    edit_metadata = Column("metadata", JSONB, nullable=False, default=dict)
    
    __table_args__ = (
        Index('idx_chunk_edit_history_chunk', 'chunk_id'),
        Index('idx_chunk_edit_history_doc', 'document_id'),
        Index('idx_chunk_edit_history_user', 'edited_by'),
    )
    
    def __repr__(self):
        return f"<ChunkEditHistory(chunk_id={self.chunk_id}, edited_at={self.edited_at})>"


# Update __all__ at the end:
__all__ = ["Document", "Chunk", "ChunkEditHistory", "Base"]
