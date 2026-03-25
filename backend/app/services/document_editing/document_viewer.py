"""
Document viewer service for previewing uploaded documents.
Supports PDF, DOCX, TXT, and other formats.
"""
from pathlib import Path
from typing import Dict, Any, Optional, BinaryIO
from uuid import UUID
import mimetypes
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.models.document import Document


class DocumentViewerService:
    """Service for viewing and previewing documents."""
    
    async def get_document_info(
        self,
        document_id: UUID,
        db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """
        Get document information including file path and metadata.
        
        Args:
            document_id: Document UUID
            db: Database session
            
        Returns:
            Document info dictionary
        """
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if not document:
            return None
        
        file_path = Path(document.file_path)
        
        return {
            'id': str(document.id),
            'filename': document.filename,
            'original_filename': document.original_filename,
            'file_path': str(file_path),
            'file_exists': file_path.exists(),
            'file_size_bytes': document.file_size_bytes,
            'file_size_mb': round(document.file_size_bytes / (1024 * 1024), 2),
            'doc_type': document.doc_type,
            'department': document.department,
            'total_pages': document.total_pages,
            'total_chunks': document.total_chunks,
            'has_tables': document.has_tables,
            'has_images': document.has_images,
            'status': document.status,
            'upload_date': document.upload_date.isoformat() if document.upload_date else None,
            'processed_date': document.processed_date.isoformat() if document.processed_date else None,
            'metadata': document.doc_metadata,
            'mime_type': self._get_mime_type(document.filename)
        }
    
    def get_document_file_path(
        self,
        document: Document
    ) -> Path:
        """Get the file system path for a document."""
        return Path(document.file_path)
    
    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type from filename."""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'
    
    def is_previewable(self, filename: str) -> bool:
        """Check if document type supports preview."""
        ext = Path(filename).suffix.lower()
        previewable_types = ['.pdf', '.txt', '.md', '.json', '.csv']
        return ext in previewable_types
    
    def get_preview_type(self, filename: str) -> str:
        """Determine preview type for frontend."""
        ext = Path(filename).suffix.lower()
        
        if ext == '.pdf':
            return 'pdf'
        elif ext in ['.txt', '.md', '.csv', '.json']:
            return 'text'
        elif ext in ['.docx', '.doc']:
            return 'docx'
        elif ext in ['.xlsx', '.xls']:
            return 'excel'
        else:
            return 'download'
    
    async def get_text_preview(
        self,
        document_id: UUID,
        db: AsyncSession,
        max_chars: int = 5000
    ) -> Optional[Dict[str, Any]]:
        """
        Get text preview for text-based documents.
        
        Args:
            document_id: Document UUID
            db: Database session
            max_chars: Maximum characters to return
            
        Returns:
            Preview data
        """
        doc_info = await self.get_document_info(document_id, db)
        if not doc_info or not doc_info['file_exists']:
            return None
        
        file_path = Path(doc_info['file_path'])
        ext = file_path.suffix.lower()
        
        if ext not in ['.txt', '.md', '.csv', '.json']:
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(max_chars)
            
            is_truncated = len(content) >= max_chars
            
            return {
                'content': content,
                'is_truncated': is_truncated,
                'total_size': doc_info['file_size_bytes'],
                'preview_size': len(content)
            }
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
    async def get_download_url(
        self,
        document_id: UUID,
        db: AsyncSession
    ) -> Optional[str]:
        """
        Get download URL for a document.
        
        Args:
            document_id: Document UUID
            db: Database session
            
        Returns:
            Download URL path
        """
        doc_info = await self.get_document_info(document_id, db)
        if not doc_info:
            return None
        
        return f"/api/v1/documents/{document_id}/download"


# Global instance
document_viewer_service = DocumentViewerService()

__all__ = ['DocumentViewerService', 'document_viewer_service']