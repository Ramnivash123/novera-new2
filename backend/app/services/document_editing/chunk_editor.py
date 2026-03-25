"""
Chunk editing service for modifying extracted document chunks.
Handles editing, re-embedding, and version tracking.
"""
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from loguru import logger

from app.models.document import Chunk, ChunkEditHistory, Document
from app.services.embedding.embedding_service import embedding_service


class ChunkEditorService:
    """Service for editing document chunks with automatic re-embedding."""
    
    async def get_document_chunks(
        self,
        document_id: UUID,
        db: AsyncSession,
        include_edit_info: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document UUID
            db: Database session
            include_edit_info: Include editing metadata
            
        Returns:
            List of chunk dictionaries
        """
        result = await db.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        chunks = result.scalars().all()
        
        return [chunk.to_dict(include_edit_info=include_edit_info) for chunk in chunks]
    
    async def get_chunk_by_id(
        self,
        chunk_id: UUID,
        db: AsyncSession
    ) -> Optional[Chunk]:
        """Get a single chunk by ID."""
        result = await db.execute(
            select(Chunk).where(Chunk.id == chunk_id)
        )
        return result.scalar_one_or_none()
    
    async def edit_chunk(
        self,
        chunk_id: UUID,
        new_content: str,
        edited_by: UUID,
        db: AsyncSession,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Edit a chunk's content and regenerate its embedding.
        
        Args:
            chunk_id: Chunk UUID
            new_content: New content for the chunk
            edited_by: User ID making the edit
            db: Database session
            metadata: Optional metadata about the edit
            
        Returns:
            Tuple of (success, updated_chunk_dict, error_message)
        """
        try:
            # Get chunk
            chunk = await self.get_chunk_by_id(chunk_id, db)
            if not chunk:
                return False, None, "Chunk not found"
            
            # Validate content
            if not new_content or len(new_content.strip()) < 10:
                return False, None, "Content must be at least 10 characters"
            
            if len(new_content) > 10000:
                return False, None, "Content too long (max 10,000 characters)"
            
            old_content = chunk.content
            
            # Check if content actually changed
            if old_content.strip() == new_content.strip():
                return False, None, "No changes detected"
            
            # Store original content if this is the first edit
            if not chunk.is_edited:
                chunk.original_content = old_content
            
            # Generate new embedding
            logger.info(f"Generating new embedding for chunk {chunk_id}")
            new_embedding = await embedding_service.generate_embedding(new_content)
            
            # Update chunk
            chunk.content = new_content.strip()
            chunk.content_length = len(new_content.strip())
            chunk.embedding = new_embedding
            chunk.is_edited = True
            chunk.edited_at = datetime.utcnow()
            chunk.edited_by = edited_by
            chunk.edit_count += 1
            
            # Update token count (approximate)
            chunk.token_count = len(new_content.split())
            
            # Create edit history record
            edit_history = ChunkEditHistory(
                chunk_id=chunk_id,
                document_id=chunk.document_id,
                edited_by=edited_by,
                old_content=old_content,
                new_content=new_content,
                change_summary=self._generate_change_summary(old_content, new_content),
                edit_metadata=metadata or {} 
            )
            
            db.add(edit_history)
            await db.commit()
            await db.refresh(chunk)
            
            logger.info(f"✅ Chunk {chunk_id} edited successfully")
            
            return True, chunk.to_dict(include_edit_info=True), None
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error editing chunk {chunk_id}: {str(e)}")
            return False, None, f"Failed to edit chunk: {str(e)}"
    
    async def batch_edit_chunks(
        self,
        edits: List[Dict[str, Any]],
        edited_by: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Edit multiple chunks in a batch.
        
        Args:
            edits: List of {'chunk_id': UUID, 'new_content': str}
            edited_by: User ID
            db: Database session
            
        Returns:
            Summary of batch operation
        """
        results = {
            'total': len(edits),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        for edit in edits:
            chunk_id = edit.get('chunk_id')
            new_content = edit.get('new_content')
            
            success, _, error = await self.edit_chunk(
                chunk_id=chunk_id,
                new_content=new_content,
                edited_by=edited_by,
                db=db
            )
            
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1
                results['errors'].append({
                    'chunk_id': str(chunk_id),
                    'error': error
                })
        
        return results
    
    async def revert_chunk(
        self,
        chunk_id: UUID,
        db: AsyncSession
    ) -> Tuple[bool, Optional[str]]:
        """
        Revert a chunk to its original content.
        
        Args:
            chunk_id: Chunk UUID
            db: Database session
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            chunk = await self.get_chunk_by_id(chunk_id, db)
            if not chunk:
                return False, "Chunk not found"
            
            if not chunk.is_edited or not chunk.original_content:
                return False, "Chunk has not been edited"
            
            # Restore original content
            chunk.content = chunk.original_content
            chunk.content_length = len(chunk.original_content)
            
            # Regenerate embedding for original content
            new_embedding = await embedding_service.generate_embedding(chunk.original_content)
            chunk.embedding = new_embedding
            
            # Reset edit fields
            chunk.is_edited = False
            chunk.edited_at = None
            chunk.edited_by = None
            chunk.edit_count = 0
            chunk.original_content = None
            
            await db.commit()
            
            logger.info(f"✅ Chunk {chunk_id} reverted to original")
            return True, None
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error reverting chunk {chunk_id}: {str(e)}")
            return False, f"Failed to revert: {str(e)}"
    
    async def get_chunk_edit_history(
        self,
        chunk_id: UUID,
        db: AsyncSession,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get edit history for a chunk.
        
        Args:
            chunk_id: Chunk UUID
            db: Database session
            limit: Maximum number of history records
            
        Returns:
            List of edit history records
        """
        result = await db.execute(
            select(ChunkEditHistory)
            .where(ChunkEditHistory.chunk_id == chunk_id)
            .order_by(ChunkEditHistory.edited_at.desc())
            .limit(limit)
        )
        history = result.scalars().all()
        
        return [
            {
                'id': str(h.id),
                'edited_at': h.edited_at.isoformat(),
                'edited_by': str(h.edited_by),
                'old_content': h.old_content,
                'new_content': h.new_content,
                'change_summary': h.change_summary,
                'metadata': h.edit_metadata
            }
            for h in history
        ]
    
    async def get_document_edit_stats(
        self,
        document_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get editing statistics for a document.
        
        Args:
            document_id: Document UUID
            db: Database session
            
        Returns:
            Statistics dictionary
        """
        from sqlalchemy import func
        
        # Total chunks
        total_result = await db.execute(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id == document_id)
        )
        total_chunks = total_result.scalar()
        
        # Edited chunks
        edited_result = await db.execute(
            select(func.count())
            .select_from(Chunk)
            .where(
                and_(
                    Chunk.document_id == document_id,
                    Chunk.is_edited == True
                )
            )
        )
        edited_chunks = edited_result.scalar()
        
        # Total edits
        edits_result = await db.execute(
            select(func.count())
            .select_from(ChunkEditHistory)
            .where(ChunkEditHistory.document_id == document_id)
        )
        total_edits = edits_result.scalar()
        
        return {
            'total_chunks': total_chunks,
            'edited_chunks': edited_chunks,
            'unedited_chunks': total_chunks - edited_chunks,
            'total_edits': total_edits,
            'edit_percentage': round((edited_chunks / total_chunks * 100) if total_chunks > 0 else 0, 2)
        }
    
    def _generate_change_summary(self, old_content: str, new_content: str) -> str:
        """Generate a brief summary of what changed."""
        old_len = len(old_content)
        new_len = len(new_content)
        
        diff = new_len - old_len
        
        if abs(diff) < 10:
            return "Minor text changes"
        elif diff > 0:
            return f"Content expanded (+{diff} chars)"
        else:
            return f"Content shortened ({diff} chars)"
    
    async def delete_chunk(
        self,
        chunk_id: UUID,
        db: AsyncSession
    ) -> Tuple[bool, Optional[str]]:
        """
        Delete a chunk (admin only operation).
        
        Args:
            chunk_id: Chunk UUID
            db: Database session
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            chunk = await self.get_chunk_by_id(chunk_id, db)
            if not chunk:
                return False, "Chunk not found"
            
            document_id = chunk.document_id
            
            await db.delete(chunk)
            
            # Update document chunk count
            doc_result = await db.execute(
                select(Document).where(Document.id == document_id)
            )
            document = doc_result.scalar_one_or_none()
            if document and document.total_chunks > 0:
                document.total_chunks -= 1
            
            await db.commit()
            
            logger.info(f"✅ Chunk {chunk_id} deleted")
            return True, None
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting chunk {chunk_id}: {str(e)}")
            return False, f"Failed to delete: {str(e)}"


# Global instance
chunk_editor_service = ChunkEditorService()

__all__ = ['ChunkEditorService', 'chunk_editor_service']