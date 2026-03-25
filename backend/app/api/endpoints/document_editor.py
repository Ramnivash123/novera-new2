"""
Document editor API endpoints.
Handles chunk viewing, editing, and document preview.
Admin-only for editing, read-only for regular users.
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from loguru import logger

from app.db.session import get_db
from app.models.user import User
from app.api.dependencies.auth import get_current_active_user, get_current_admin_user
from app.services.document_editing.chunk_editor import chunk_editor_service
from app.services.document_editing.document_viewer import document_viewer_service
from app.services.generation.title_generator import title_generator
from app.models.document import Chunk as ChunkModel

router = APIRouter()


# Request/Response Models
class ChunkEditRequest(BaseModel):
    """Request to edit a chunk."""
    chunk_id: str
    new_content: str = Field(..., min_length=10, max_length=10000)
    metadata: Optional[dict] = None


class BatchChunkEditRequest(BaseModel):
    """Request to edit multiple chunks."""
    edits: List[dict] = Field(..., min_items=1, max_items=50)


class ChunkResponse(BaseModel):
    """Single chunk response."""
    id: str
    document_id: str
    chunk_index: int
    content: str
    chunk_type: str
    page_numbers: List[int]
    section_title: Optional[str]
    token_count: int
    title: Optional[str] = None
    is_edited: bool
    edited_at: Optional[str]
    edited_by: Optional[str]
    edit_count: int
    metadata: dict


class DocumentInfoResponse(BaseModel):
    """Document information response."""
    id: str
    filename: str
    original_filename: str
    file_size_mb: float
    doc_type: str
    department: Optional[str]
    total_pages: int
    total_chunks: int
    has_tables: bool
    has_images: bool
    status: str
    upload_date: str
    processed_date: Optional[str]
    mime_type: str
    preview_type: str
    is_previewable: bool


class EditHistoryResponse(BaseModel):
    """Edit history item."""
    id: str
    edited_at: str
    edited_by: str
    old_content: str
    new_content: str
    change_summary: Optional[str]
    metadata: dict


class DocumentEditStatsResponse(BaseModel):
    """Document editing statistics."""
    total_chunks: int
    edited_chunks: int
    unedited_chunks: int
    total_edits: int
    edit_percentage: float


# Endpoints

@router.get("/documents/{document_id}/info", response_model=DocumentInfoResponse)
async def get_document_info(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get document information including preview capabilities.
    Available to all authenticated users.
    """
    doc_info = await document_viewer_service.get_document_info(document_id, db)
    
    if not doc_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return DocumentInfoResponse(
        **doc_info,
        preview_type=document_viewer_service.get_preview_type(doc_info['filename']),
        is_previewable=document_viewer_service.is_previewable(doc_info['filename'])
    )


@router.get("/documents/{document_id}/chunks", response_model=List[ChunkResponse])
async def get_document_chunks(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get all chunks for a document.
    Available to all authenticated users (read-only for non-admins).
    """
    chunks = await chunk_editor_service.get_document_chunks(
        document_id,
        db,
        include_edit_info=True
    )
    
    return [ChunkResponse(**chunk) for chunk in chunks]


@router.get("/documents/{document_id}/preview/text")
async def get_text_preview(
    document_id: UUID,
    max_chars: int = 5000,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get text preview for text-based documents.
    Available to all authenticated users.
    """
    preview = await document_viewer_service.get_text_preview(
        document_id,
        db,
        max_chars=max_chars
    )
    
    if not preview:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Preview not available for this document type"
        )
    
    return preview


@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Download the original document file.
    Available to all authenticated users.
    """
    from pathlib import Path
    
    doc_info = await document_viewer_service.get_document_info(document_id, db)
    
    if not doc_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    file_path = Path(doc_info['file_path'])
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found on disk"
        )
    
    return FileResponse(
        path=str(file_path),
        filename=doc_info['original_filename'],
        media_type=doc_info['mime_type']
    )


@router.put("/chunks/{chunk_id}/edit", response_model=ChunkResponse)
async def edit_chunk(
    chunk_id: UUID,
    request: ChunkEditRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Edit a chunk's content.
    ADMIN ONLY - automatically regenerates embedding.
    """
    success, chunk_data, error = await chunk_editor_service.edit_chunk(
        chunk_id=chunk_id,
        new_content=request.new_content,
        edited_by=admin.id,
        db=db,
        metadata=request.metadata
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    logger.info(f"Admin {admin.email} edited chunk {chunk_id}")
    
    return ChunkResponse(**chunk_data)


@router.post("/chunks/batch-edit")
async def batch_edit_chunks(
    request: BatchChunkEditRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Edit multiple chunks in batch.
    ADMIN ONLY.
    """
    results = await chunk_editor_service.batch_edit_chunks(
        edits=request.edits,
        edited_by=admin.id,
        db=db
    )
    
    logger.info(f"Admin {admin.email} batch edited {results['successful']} chunks")
    
    return results


@router.post("/chunks/{chunk_id}/revert")
async def revert_chunk(
    chunk_id: UUID,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Revert chunk to its original content.
    ADMIN ONLY.
    """
    success, error = await chunk_editor_service.revert_chunk(chunk_id, db)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    logger.info(f"Admin {admin.email} reverted chunk {chunk_id}")
    
    return {"message": "Chunk reverted successfully", "chunk_id": str(chunk_id)}


@router.delete("/chunks/{chunk_id}")
async def delete_chunk(
    chunk_id: UUID,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Delete a chunk.
    ADMIN ONLY - use with caution.
    """
    success, error = await chunk_editor_service.delete_chunk(chunk_id, db)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    logger.info(f"Admin {admin.email} deleted chunk {chunk_id}")
    
    return {"message": "Chunk deleted successfully", "chunk_id": str(chunk_id)}


@router.get("/chunks/{chunk_id}/history", response_model=List[EditHistoryResponse])
async def get_chunk_history(
    chunk_id: UUID,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get edit history for a chunk.
    Available to all authenticated users.
    """
    history = await chunk_editor_service.get_chunk_edit_history(
        chunk_id,
        db,
        limit=limit
    )
    
    return [EditHistoryResponse(**item) for item in history]


@router.get("/documents/{document_id}/edit-stats", response_model=DocumentEditStatsResponse)
async def get_document_edit_stats(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get editing statistics for a document.
    Available to all authenticated users.
    """
    stats = await chunk_editor_service.get_document_edit_stats(document_id, db)
    
    return DocumentEditStatsResponse(**stats)

@router.post("/chunks/{chunk_id}/generate-title")
async def generate_chunk_title(
    chunk_id: UUID,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Generate/regenerate AI title for a specific chunk.
    ADMIN ONLY.
    """
    from sqlalchemy import select, update
    
    # Get chunk
    result = await db.execute(
        select(ChunkModel).where(ChunkModel.id == chunk_id)
    )
    chunk = result.scalar_one_or_none()
    
    if not chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk not found"
        )
    
    try:
        # Generate new title
        title = await title_generator.generate_title(
            content=chunk.content,
            chunk_type=chunk.chunk_type,
            section_title=chunk.section_title,
            page_numbers=chunk.page_numbers,
            chunk_index=chunk.chunk_index
        )
        
        # Update chunk
        await db.execute(
            update(ChunkModel)
            .where(ChunkModel.id == chunk_id)
            .values(title=title)
        )
        await db.commit()
        
        logger.info(f"Admin {admin.email} generated title for chunk {chunk_id}: '{title}'")
        
        return {
            "chunk_id": str(chunk_id),
            "title": title,
            "message": "Title generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Title generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Title generation failed: {str(e)}"
        )


@router.post("/documents/{document_id}/generate-all-titles")
async def generate_all_titles(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Generate titles for all chunks in a document.
    ADMIN ONLY - runs in background.
    """
    from sqlalchemy import select, update
    
    # Get all chunks for document
    result = await db.execute(
        select(ChunkModel)
        .where(ChunkModel.document_id == document_id)
        .order_by(ChunkModel.chunk_index)
    )
    chunks = result.scalars().all()
    
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No chunks found for this document"
        )
    
    logger.info(f"Admin {admin.email} requested title generation for {len(chunks)} chunks")
    
    # Generate titles for all chunks
    success_count = 0
    failed_count = 0
    
    for chunk in chunks:
        try:
            title = await title_generator.generate_title(
                content=chunk.content,
                chunk_type=chunk.chunk_type,
                section_title=chunk.section_title,
                page_numbers=chunk.page_numbers,
                chunk_index=chunk.chunk_index
            )
            
            await db.execute(
                update(ChunkModel)
                .where(ChunkModel.id == chunk.id)
                .values(title=title)
            )
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to generate title for chunk {chunk.id}: {e}")
            failed_count += 1
            continue
    
    await db.commit()
    
    logger.info(f"Generated {success_count} titles, {failed_count} failed")
    
    return {
        "document_id": str(document_id),
        "total_chunks": len(chunks),
        "success": success_count,
        "failed": failed_count,
        "message": f"Generated titles for {success_count} chunks"
    }


@router.put("/chunks/{chunk_id}/title")
async def update_chunk_title(
    chunk_id: UUID,
    title: str,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Manually update chunk title.
    ADMIN ONLY.
    """
    from sqlalchemy import select, update
    
    if len(title) > 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Title too long (max 200 characters)"
        )
    
    # Update title
    result = await db.execute(
        update(ChunkModel)
        .where(ChunkModel.id == chunk_id)
        .values(title=title)
        .returning(ChunkModel.id)
    )
    
    updated = result.scalar_one_or_none()
    
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk not found"
        )
    
    await db.commit()
    
    logger.info(f"Admin {admin.email} manually updated title for chunk {chunk_id}")
    
    return {
        "chunk_id": str(chunk_id),
        "title": title,
        "message": "Title updated successfully"
    }


__all__ = ['router']