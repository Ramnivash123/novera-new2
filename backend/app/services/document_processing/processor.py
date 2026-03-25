"""
Document processing orchestrator.
Coordinates extraction, chunking, embedding, and storage operations.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.models.document import Document, Chunk as ChunkModel
from app.services.document_processing.text_extractor import text_extractor
from app.services.document_processing.chunker import semantic_chunker
from app.services.embedding.embedding_service import embedding_service
from app.services.generation.title_generator import title_generator


class DocumentProcessor:
    """
    Orchestrates the complete document processing pipeline.
    Handles extraction → chunking → embedding → storage.
    """
    
    async def process_document(
        self,
        document_id: UUID,
        file_path: Path,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline.
        
        Args:
            document_id: UUID of document record
            file_path: Path to uploaded file
            db: Database session
            
        Returns:
            Processing statistics
        """
        start_time = datetime.utcnow()
        
        try:
            # Update status to processing
            document = await self._get_document(document_id, db)
            document.status = "processing"
            await db.commit()
            
            logger.info(f"Starting processing for document {document_id}")
            
            # Step 1: Extract text and structure
            logger.info("Step 1: Extracting document structure")
            doc_structure = text_extractor.extract_document(file_path)

            # Update document metadata
            document.total_pages = doc_structure.total_pages
            document.has_tables = doc_structure.has_tables
            document.has_images = doc_structure.has_images

            # Properly update JSONB metadata field
            current_metadata = document.doc_metadata or {}
            updated_metadata = {**current_metadata, **doc_structure.metadata}
            document.doc_metadata = updated_metadata
            
            # Step 2: Chunk the document
            logger.info("Step 2: Chunking document")
            chunks = semantic_chunker.chunk_document(
                doc_structure,
                document_title=document.filename
            )
            
            document.total_chunks = len(chunks)
            
            # Step 3: Generate embeddings
            logger.info("Step 3: Generating embeddings")
            chunks_with_embeddings = await embedding_service.embed_chunks_with_context(
                [chunk.to_dict() for chunk in chunks],
                document_title=document.filename
            )
            
            # Step 4: Store chunks in database WITH TITLE GENERATION
            logger.info("Step 4: Storing chunks with embeddings and generating titles")
            await self._store_chunks(
                document_id,
                chunks_with_embeddings,
                db
            )
            
            # Update document status
            document.status = "completed"
            document.processed_date = datetime.utcnow()
            await db.commit()
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            stats = {
                "document_id": str(document_id),
                "total_pages": doc_structure.total_pages,
                "total_chunks": len(chunks),
                "has_tables": doc_structure.has_tables,
                "processing_time_seconds": processing_time,
                "status": "success"
            }
            
            logger.info(f"✅ Document processing completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error processing document {document_id}: {str(e)}")
            
            # Update document with error
            try:
                document = await self._get_document(document_id, db)
                document.status = "failed"
                document.processing_error = str(e)
                await db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update error status: {str(db_error)}")
            
            raise
    
    async def _get_document(self, document_id: UUID, db: AsyncSession) -> Document:
        """Fetch document by ID."""
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        return document
    
    async def _store_chunks(
        self,
        document_id: UUID,
        chunks: list[Dict[str, Any]],
        db: AsyncSession
    ) -> None:
        """
        Store chunks with embeddings and AI-generated titles in database.
        
        Args:
            document_id: Parent document ID
            chunks: List of chunk dictionaries with embeddings
            db: Database session
        """
        chunk_models = []
        
        for chunk_data in chunks:
            # Generate AI title for this chunk
            title = None
            try:
                title = await title_generator.generate_title(
                    content=chunk_data['content'],
                    chunk_type=chunk_data['chunk_type'],
                    section_title=chunk_data.get('section_title'),
                    page_numbers=chunk_data.get('page_numbers'),
                    chunk_index=chunk_data['chunk_index']
                )
                logger.debug(f"Generated title for chunk {chunk_data['chunk_index']}: '{title}'")
            except Exception as e:
                logger.warning(f"Title generation failed for chunk {chunk_data['chunk_index']}: {e}")
                title = None  # Will use fallback in frontend
            
            chunk_model = ChunkModel(
                id=uuid4(),
                document_id=document_id,
                chunk_index=chunk_data['chunk_index'],
                content=chunk_data['content'],
                content_length=len(chunk_data['content']),
                token_count=chunk_data['token_count'],
                chunk_type=chunk_data['chunk_type'],
                page_numbers=chunk_data['page_numbers'],
                section_title=chunk_data.get('section_title'),
                title=title,  # NEW: AI-generated title
                embedding=chunk_data['embedding'],
                chunk_metadata=chunk_data.get('metadata', {})
            )
            
            chunk_models.append(chunk_model)
        
        # Bulk insert chunks
        db.add_all(chunk_models)
        await db.commit()
        
        titles_generated = sum(1 for c in chunk_models if c.title)
        logger.info(f"Stored {len(chunk_models)} chunks for document {document_id}, generated {titles_generated} titles")
    
    async def reprocess_document(
        self,
        document_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Reprocess an existing document.
        Deletes old chunks and processes fresh.
        
        Args:
            document_id: Document to reprocess
            db: Database session
            
        Returns:
            Processing statistics
        """
        # Get document
        document = await self._get_document(document_id, db)
        
        # Delete existing chunks
        logger.info(f"Deleting existing chunks for document {document_id}")
        from sqlalchemy import delete
        await db.execute(
            delete(ChunkModel).where(ChunkModel.document_id == document_id)
        )
        await db.commit()
        
        # Process document
        return await self.process_document(
            document_id,
            Path(document.file_path),
            db
        )
    
    async def get_processing_status(
        self,
        document_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get current processing status of a document.
        
        Args:
            document_id: Document ID
            db: Database session
            
        Returns:
            Status information
        """
        document = await self._get_document(document_id, db)
        
        return {
            "document_id": str(document.id),
            "filename": document.filename,
            "status": document.status,
            "total_pages": document.total_pages,
            "total_chunks": document.total_chunks,
            "upload_date": document.upload_date.isoformat() if document.upload_date else None,
            "processed_date": document.processed_date.isoformat() if document.processed_date else None,
            "error": document.processing_error
        }


# Global instance
document_processor = DocumentProcessor()

__all__ = ['DocumentProcessor', 'document_processor']