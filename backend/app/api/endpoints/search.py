"""
Search API endpoints for testing retrieval system.
Provides endpoints to test different search strategies.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from loguru import logger

from app.db.session import get_db
from app.services.retrieval.pipeline import retrieval_pipeline


router = APIRouter()


# Response Models
class ChunkResult(BaseModel):
    """Individual chunk in search results."""
    chunk_id: str
    document_id: str
    content: str
    chunk_type: str
    page_numbers: List[int]
    section_title: Optional[str]
    relevance_score: float = Field(..., description="Combined relevance score")
    metadata: dict


class SearchResponse(BaseModel):
    """Search results response."""
    query: str
    total_results: int
    chunks: List[ChunkResult]
    retrieval_strategy: str
    intent: str
    complexity: str
    total_context_tokens: int


class SourceInfo(BaseModel):
    """Source citation information."""
    document: str
    page: Optional[int]
    section: Optional[str]
    chunk_id: str


class ContextResponse(BaseModel):
    """Full context response with assembled text."""
    query: str
    context_text: str
    total_tokens: int
    chunks_used: int
    sources: List[SourceInfo]
    retrieval_metadata: dict


@router.post("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., description="Search query", min_length=1),
    top_k: int = Query(5, ge=1, le=20, description="Number of results"),
    doc_type: Optional[str] = Query(None, description="Filter by document type"),
    department: Optional[str] = Query(None, description="Filter by department"),
    db: AsyncSession = Depends(get_db)
):
    """
    Search for relevant chunks across all documents.
    
    This endpoint demonstrates the retrieval system:
    - Query processing and enhancement
    - Hybrid search (semantic + keyword)
    - Reranking with Cohere
    - Context assembly
    
    Example:
    ```
    POST /search?query=What is the PF contribution rate?&top_k=5
    ```
    """
    logger.info(f"Search request: query='{query}', top_k={top_k}")
    
    # Retrieve using full pipeline
    result = await retrieval_pipeline.retrieve(
        query=query,
        db=db,
        top_k=top_k,
        doc_type=doc_type,
        department=department,
        include_context=False  # Don't expand for search endpoint
    )
    
    # Format chunks for response
    chunks = []
    for chunk in result['chunks']:
        chunk_result = ChunkResult(
            chunk_id=chunk['chunk_id'],
            document_id=chunk['document_id'],
            content=chunk['content'][:500] + '...' if len(chunk['content']) > 500 else chunk['content'],
            chunk_type=chunk['chunk_type'],
            page_numbers=chunk['page_numbers'],
            section_title=chunk.get('section_title'),
            relevance_score=chunk.get('rerank_score', chunk.get('fused_score', 0)),
            metadata={
                'document_title': chunk['metadata'].get('document_title'),
                'doc_type': chunk['metadata'].get('doc_type'),
                'department': chunk['metadata'].get('department'),
            }
        )
        chunks.append(chunk_result)
    
    return SearchResponse(
        query=query,
        total_results=len(chunks),
        chunks=chunks,
        retrieval_strategy=result['retrieval_metadata']['search_strategy'],
        intent=result['retrieval_metadata']['intent'],
        complexity=result['retrieval_metadata']['complexity'],
        total_context_tokens=result['total_tokens']
    )


@router.post("/search/context", response_model=ContextResponse)
async def search_with_context(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20),
    doc_type: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Search and return assembled context ready for LLM.
    
    This endpoint provides the complete context that would be
    sent to the LLM for answer generation.
    
    Use this to:
    - Debug retrieval quality
    - Test context assembly
    - Verify source citations
    """
    logger.info(f"Context search: query='{query}'")
    
    # Retrieve with context expansion
    result = await retrieval_pipeline.retrieve(
        query=query,
        db=db,
        top_k=top_k,
        doc_type=doc_type,
        department=department,
        include_context=True
    )
    
    # Format sources
    sources = [
        SourceInfo(
            document=src['document'],
            page=src['page'],
            section=src['section'],
            chunk_id=src['chunk_id']
        )
        for src in result['sources']
    ]
    
    return ContextResponse(
        query=query,
        context_text=result['context_text'],
        total_tokens=result['total_tokens'],
        chunks_used=len(result['chunks']),
        sources=sources,
        retrieval_metadata=result['retrieval_metadata']
    )


@router.post("/search/document/{document_id}", response_model=SearchResponse)
async def search_within_document(
    document_id: str,
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db)
):
    """
    Search within a specific document only.
    
    Useful for:
    - Document-specific questions
    - Clarifying information from a known document
    - Focused retrieval
    """
    logger.info(f"Document search: doc_id={document_id}, query='{query}'")
    
    result = await retrieval_pipeline.retrieve_from_document(
        query=query,
        document_id=document_id,
        db=db,
        top_k=top_k
    )
    
    # Format response
    chunks = []
    for chunk in result['chunks']:
        chunk_result = ChunkResult(
            chunk_id=chunk['chunk_id'],
            document_id=chunk['document_id'],
            content=chunk['content'][:500] + '...' if len(chunk['content']) > 500 else chunk['content'],
            chunk_type=chunk['chunk_type'],
            page_numbers=chunk['page_numbers'],
            section_title=chunk.get('section_title'),
            relevance_score=chunk.get('rerank_score', chunk.get('fused_score', 0)),
            metadata=chunk.get('metadata', {})
        )
        chunks.append(chunk_result)
    
    return SearchResponse(
        query=query,
        total_results=len(chunks),
        chunks=chunks,
        retrieval_strategy="document_specific",
        intent="",
        complexity="",
        total_context_tokens=result['total_tokens']
    )


@router.get("/search/test")
async def test_search_endpoint():
    """
    Test endpoint to verify search API is working.
    """
    return {
        "status": "operational",
        "message": "Search API is ready",
        "endpoints": [
            "POST /search - General search",
            "POST /search/context - Search with context",
            "POST /search/document/{id} - Document-specific search"
        ]
    }


__all__ = ['router']