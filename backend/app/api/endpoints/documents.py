"""
Document management API endpoints with RBAC.
Handles upload, processing, listing, and deletion of documents.
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from pydantic import BaseModel, Field
from datetime import datetime
from loguru import logger

from app.db.session import get_db
from app.models.document import Document, Chunk
from app.models.user import User
from app.db.session import AsyncSessionLocal   
from app.models.organization import Organization
from app.utils.file_utils import FileValidator, file_handler
from app.services.document_processing.processor import document_processor
from app.api.dependencies.auth import get_current_active_user, get_current_admin_user


router = APIRouter()


# Pydantic Models for Request/Response
class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    filename: str
    status: str
    message: str
    file_size_mb: float


class DocumentListItem(BaseModel):
    """Individual document in list response."""
    id: str
    filename: str
    doc_type: str
    department: Optional[str]
    total_pages: int
    total_chunks: int
    status: str
    upload_date: str
    processed_date: Optional[str]


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    total: int
    documents: List[DocumentListItem]

class MultiUploadResult(BaseModel):
    filename: str
    status: str
    message: str
    document_id: Optional[str] = None
    file_size_mb: Optional[float] = None
    error: Optional[str] = None


class MultiUploadResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    duplicates: int
    results: List[MultiUploadResult]

class ProcessingStatusResponse(BaseModel):
    """Response model for processing status."""
    document_id: str
    filename: str
    status: str
    total_pages: Optional[int]
    total_chunks: Optional[int]
    upload_date: Optional[str]
    processed_date: Optional[str]
    error: Optional[str]


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Query(..., description="Document type: finance, hrms, policy, etc."),
    department: Optional[str] = Query(None, description="Department name"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Upload a new document for processing.
    
    ADMIN ONLY - Regular users have read-only access to documents.
    Documents are scoped to the admin's organization.
    """
    logger.info(f"Document upload request: {file.filename} (type: {doc_type}) by admin {current_user.email} (org: {current_user.organization_id})")
    
    # ORGANIZATION CHECK: Ensure user has an organization
    if not current_user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must belong to an organization to upload documents"
        )
    
    # Step 1: Validate file
    is_valid, error_message = FileValidator.validate_file(file)
    if not is_valid:
        logger.warning(f"File validation failed: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)
    
    # Step 2: Check for duplicate (by hash within organization)
    try:
        file_path, file_hash, file_size = await file_handler.save_upload_file(
            file, 
            str(current_user.id)
        )
        
        # ORGANIZATION SCOPING: Check for duplicates only within the same organization
        result = await db.execute(
            select(Document).where(
                Document.file_hash == file_hash,
                Document.organization_id == current_user.organization_id
            )
        )
        existing_doc = result.scalar_one_or_none()
        
        if existing_doc:
            logger.info(f"Duplicate document detected in organization {current_user.organization_id}: {file_hash}")
            file_handler.delete_file(file_path)
            
            return DocumentUploadResponse(
                document_id=str(existing_doc.id),
                filename=existing_doc.filename,
                status="duplicate",
                message="This document already exists in your organization",
                file_size_mb=round(file_size / (1024 * 1024), 2)
            )
        
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    # Step 3: Create database record with organization scoping
    try:
        document = Document(
            filename=file.filename,
            original_filename=file.filename,
            file_path=str(file_path),
            file_size_bytes=file_size,
            file_hash=file_hash,
            doc_type=doc_type,
            department=department,
            status="pending",
            uploaded_by=current_user.id,
            organization_id=current_user.organization_id,  # ORGANIZATION SCOPING
            metadata={
                "mime_type": FileValidator.get_mime_type(file.filename),
                "upload_source": "api",
                "uploaded_by_email": current_user.email,
                "organization_id": str(current_user.organization_id)
            }
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        logger.info(f"✅ Document record created: {document.id} for organization {current_user.organization_id}")
        
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        file_handler.delete_file(file_path)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    # Step 4: Queue background processing
    background_tasks.add_task(
        process_document_background,
        document.id,
        file_path,
    )
    
    return DocumentUploadResponse(
        document_id=str(document.id),
        filename=document.filename,
        status="processing",
        message="Document uploaded successfully and queued for processing",
        file_size_mb=round(file_size / (1024 * 1024), 2)
    )

@router.post("/documents/upload-multiple", response_model=MultiUploadResponse)
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    doc_type: str = Query(..., description="Document type: finance, hrms, policy, etc."),
    department: Optional[str] = Query(None, description="Department name"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Upload multiple documents in a single request.
    Each file is validated, deduplicated, and queued for processing independently.
    Returns a summary of successes, failures, and duplicates.
    """
    if not current_user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must belong to an organization to upload documents"
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )

    if len(files) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 20 files per batch upload"
        )

    results: List[MultiUploadResult] = []
    succeeded = 0
    failed = 0
    duplicates = 0

    for file in files:
        try:
            # Validate file type and size
            is_valid, error_message = FileValidator.validate_file(file)
            if not is_valid:
                results.append(MultiUploadResult(
                    filename=file.filename or "unknown",
                    status="failed",
                    message=error_message,
                    error=error_message
                ))
                failed += 1
                continue

            # Save file and check for duplicates
            file_path, file_hash, file_size = await file_handler.save_upload_file(
                file, str(current_user.id)
            )

            existing_result = await db.execute(
                select(Document).where(
                    Document.file_hash == file_hash,
                    Document.organization_id == current_user.organization_id
                )
            )
            existing_doc = existing_result.scalar_one_or_none()

            if existing_doc:
                file_handler.delete_file(file_path)
                results.append(MultiUploadResult(
                    filename=file.filename or "unknown",
                    status="duplicate",
                    message="This document already exists in your organization",
                    document_id=str(existing_doc.id),
                    file_size_mb=round(file_size / (1024 * 1024), 2)
                ))
                duplicates += 1
                continue

            # Create document record
            document = Document(
                filename=file.filename,
                original_filename=file.filename,
                file_path=str(file_path),
                file_size_bytes=file_size,
                file_hash=file_hash,
                doc_type=doc_type,
                department=department,
                status="pending",
                uploaded_by=current_user.id,
                organization_id=current_user.organization_id,
                metadata={
                    "mime_type": FileValidator.get_mime_type(file.filename),
                    "upload_source": "api_multi",
                    "uploaded_by_email": current_user.email,
                    "organization_id": str(current_user.organization_id)
                }
            )
            db.add(document)
            await db.commit()
            await db.refresh(document)

            # Queue processing
            background_tasks.add_task(
                process_document_background,
                document.id,
                file_path,
            )

            results.append(MultiUploadResult(
                filename=file.filename or "unknown",
                status="processing",
                message="Queued for processing",
                document_id=str(document.id),
                file_size_mb=round(file_size / (1024 * 1024), 2)
            ))
            succeeded += 1
            logger.info(f"Queued document: {document.id} ({file.filename})")

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            results.append(MultiUploadResult(
                filename=file.filename or "unknown",
                status="failed",
                message="Upload failed due to server error",
                error=str(e)
            ))
            failed += 1

    logger.info(
        f"Multi-upload complete: {succeeded} queued, "
        f"{duplicates} duplicates, {failed} failed "
        f"(org: {current_user.organization_id})"
    )

    return MultiUploadResponse(
        total=len(files),
        succeeded=succeeded,
        failed=failed,
        duplicates=duplicates,
        results=results
    )

async def process_document_background(
    document_id: UUID,
    file_path,
):
    """
    Background task — creates its own DB session.
    The request session is closed by the time this runs.
    """
    from app.db.session import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        try:
            logger.info(f"Starting background processing for {document_id}")
            stats = await document_processor.process_document(
                document_id, file_path, db
            )
            logger.info(f"Background processing completed: {stats}")
        except Exception as e:
            logger.error(f"Background processing failed for {document_id}: {str(e)}")
            try:
                from sqlalchemy import update
                await db.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(status="failed", processing_error=str(e)[:500])
                )
                await db.commit()
            except Exception as db_err:
                logger.error(f"Failed to update error status: {db_err}")

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    doc_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    List documents with optional filtering.
    
    ORGANIZATION SCOPED:
    - Users see only documents in their organization
    - Super admins see all documents across all organizations
    - Org admins see only their organization's documents
    """
    # Build query with ORGANIZATION SCOPING
    query = select(Document)
    
    # ORGANIZATION SCOPING: Users only see documents in their organization
    if not current_user.is_super_admin():
        if not current_user.organization_id:
            # User has no organization, return empty list
            return DocumentListResponse(total=0, documents=[])
        
        query = query.where(Document.organization_id == current_user.organization_id)
    
    # Apply filters
    if doc_type:
        query = query.where(Document.doc_type == doc_type)
    
    if status:
        query = query.where(Document.status == status)
    
    if department:
        query = query.where(Document.department == department)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Get documents
    query = query.order_by(Document.upload_date.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    documents = result.scalars().all()
    
    logger.info(f"User {current_user.email} (org: {current_user.organization_id}, role: {current_user.role}) retrieved {len(documents)} documents")
    
    # Convert to response model
    document_list = [
        DocumentListItem(
            id=str(doc.id),
            filename=doc.filename,
            doc_type=doc.doc_type,
            department=doc.department,
            total_pages=doc.total_pages or 0,
            total_chunks=doc.total_chunks or 0,
            status=doc.status,
            upload_date=doc.upload_date.isoformat(),
            processed_date=doc.processed_date.isoformat() if doc.processed_date else None
        )
        for doc in documents
    ]
    
    return DocumentListResponse(
        total=total,
        documents=document_list
    )


@router.get("/documents/{document_id}/status", response_model=ProcessingStatusResponse)
async def get_document_status(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get processing status of a specific document.
    
    ORGANIZATION SCOPED: Users can only view documents in their organization.
    Super admins can view any document.
    """
    # Build query with ORGANIZATION SCOPING
    query = select(Document).where(Document.id == document_id)
    
    # ORGANIZATION SCOPING: Users can only view documents in their org
    if not current_user.is_super_admin():
        if not current_user.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User must belong to an organization"
            )
        query = query.where(Document.organization_id == current_user.organization_id)
    
    result = await db.execute(query)
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or access denied"
        )
    
    return ProcessingStatusResponse(
        document_id=str(document.id),
        filename=document.filename,
        status=document.status,
        total_pages=document.total_pages,
        total_chunks=document.total_chunks,
        upload_date=document.upload_date.isoformat() if document.upload_date else None,
        processed_date=document.processed_date.isoformat() if document.processed_date else None,
        error=document.processing_error
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Delete a document and all its chunks.
    Also removes the file from disk.
    
    ADMIN ONLY with ORGANIZATION SCOPING:
    - Org admins can only delete documents in their organization
    - Super admins can delete any document
    """
    # Build query with ORGANIZATION SCOPING
    query = select(Document).where(Document.id == document_id)
    
    # ORGANIZATION SCOPING: Org admins can only delete docs in their org
    if current_user.is_org_admin() and not current_user.is_super_admin():
        if not current_user.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin must belong to an organization"
            )
        query = query.where(Document.organization_id == current_user.organization_id)
    
    result = await db.execute(query)
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or access denied"
        )
    
    # Delete file from disk
    from pathlib import Path
    file_path = Path(document.file_path)
    file_handler.delete_file(file_path)
    
    # Delete chunks (cascade should handle this, but explicit is better)
    await db.execute(
        delete(Chunk).where(Chunk.document_id == document_id)
    )
    
    # Delete document record
    await db.delete(document)
    await db.commit()
    
    logger.info(f"✅ Document deleted by admin {current_user.email} (org: {current_user.organization_id}): {document_id}")
    
    return {
        "message": "Document deleted successfully",
        "document_id": str(document_id)
    }


@router.post("/documents/{document_id}/reprocess")
async def reprocess_document(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Reprocess a document (useful if chunking/embedding parameters changed).
    
    ADMIN ONLY with ORGANIZATION SCOPING:
    - Org admins can only reprocess documents in their organization
    - Super admins can reprocess any document
    """
    # Build query with ORGANIZATION SCOPING
    query = select(Document).where(Document.id == document_id)
    
    # ORGANIZATION SCOPING: Org admins can only reprocess docs in their org
    if current_user.is_org_admin() and not current_user.is_super_admin():
        if not current_user.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin must belong to an organization"
            )
        query = query.where(Document.organization_id == current_user.organization_id)
    
    result = await db.execute(query)
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or access denied"
        )
    
    # Queue reprocessing
    background_tasks.add_task(
        document_processor.reprocess_document,
        document_id,
        db
    )
    
    logger.info(f"Document reprocessing queued by admin {current_user.email} (org: {current_user.organization_id}): {document_id}")
    
    return {
        "message": "Document reprocessing queued",
        "document_id": str(document_id)
    }

__all__ = ['router']