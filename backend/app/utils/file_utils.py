"""
File handling utilities for document upload and validation.
Handles file operations, validation, and hash generation.
"""
import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException
from loguru import logger

from app.core.config import settings


class FileValidator:
    """Validates uploaded files against security and size constraints."""
    
    ALLOWED_MIME_TYPES = {
        'application/pdf': ['.pdf'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
        'application/msword': ['.doc'],
        'text/plain': ['.txt'],
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
        'application/vnd.ms-excel': ['.xls'],
    }
    
    DANGEROUS_EXTENSIONS = {'.exe', '.bat', '.sh', '.cmd', '.com', '.scr'}
    
    @classmethod
    def validate_file(cls, file: UploadFile) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded file for security and compliance.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check filename
        if not file.filename:
            return False, "Filename is required"
        
        # Get file extension (with dot)
        file_ext = Path(file.filename).suffix.lower()
        
        # Check against dangerous extensions
        if file_ext in cls.DANGEROUS_EXTENSIONS:
            return False, f"File type {file_ext} is not allowed for security reasons"
        
        # Remove the dot for comparison with settings.allowed_extensions_list
        file_ext_without_dot = file_ext.lstrip('.')
        
        # Check against allowed extensions
        if file_ext_without_dot not in settings.allowed_extensions_list:
            return False, f"File type {file_ext} not supported. Allowed: {', '.join(settings.allowed_extensions_list)}"
        
        # Validate file size (if we can get it)
        if hasattr(file, 'size') and file.size:
            if file.size > settings.max_upload_size_bytes:
                return False, f"File size exceeds maximum allowed size of {settings.max_upload_size_mb}MB"
        
        return True, None
    
    @classmethod
    def get_mime_type(cls, filename: str) -> Optional[str]:
        """Get MIME type from filename."""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type


class FileHandler:
    """Handles file storage, retrieval, and cleanup operations."""
    
    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.processed_dir = Path(settings.processed_dir)
        
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """
        Calculate SHA256 hash of file for deduplication.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex string of file hash
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    async def save_upload_file(
        self, 
        file: UploadFile, 
        user_id: str,
        preserve_filename: bool = False
    ) -> Tuple[Path, str, int]:
        """
        Save uploaded file to disk.
        
        Args:
            file: FastAPI UploadFile
            user_id: ID of user uploading file
            preserve_filename: Whether to keep original filename
            
        Returns:
            Tuple of (file_path, file_hash, file_size)
        """
        try:
            # Generate unique filename
            original_filename = file.filename
            file_ext = Path(original_filename).suffix.lower()
            
            if preserve_filename:
                # Sanitize filename
                safe_filename = self._sanitize_filename(original_filename)
                final_filename = safe_filename
            else:
                # Generate unique filename with timestamp
                import uuid
                from datetime import datetime
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                final_filename = f"{timestamp}_{unique_id}{file_ext}"
            
            # Create user-specific subdirectory
            user_upload_dir = self.upload_dir / user_id
            user_upload_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = user_upload_dir / final_filename
            
            # Save file in chunks
            file_size = 0
            with open(file_path, "wb") as buffer:
                while chunk := await file.read(8192):  # 8KB chunks
                    buffer.write(chunk)
                    file_size += len(chunk)
            
            # Calculate file hash
            file_hash = self.calculate_file_hash(file_path)
            
            logger.info(f"File saved: {file_path} (hash: {file_hash[:16]}..., size: {file_size} bytes)")
            
            return file_path, file_hash, file_size
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal and other attacks.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = os.path.basename(filename)
        
        # Replace dangerous characters
        dangerous_chars = ['/', '\\', '..', '\x00']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename
    
    def delete_file(self, file_path: Path) -> bool:
        """
        Delete file from disk.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if deleted successfully
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        if file_path.exists():
            return file_path.stat().st_size / (1024 * 1024)
        return 0.0


# Global instance
file_handler = FileHandler()

__all__ = ['FileValidator', 'FileHandler', 'file_handler']