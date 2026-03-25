"""
Chat API endpoints for RAG system.
Provides conversational interface with full RAG pipeline.
"""
from typing import List, Dict, Any, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, Query, HTTPException, status
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from pydantic import BaseModel, Field
from datetime import datetime
from loguru import logger
import json
import time

from app.db.session import get_db
from app.services.generation.chat_service import chat_service
from app.api.dependencies.auth import get_current_user
from app.models.user import User
from app.models.document import Document, Chunk as ChunkModel


router = APIRouter()


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model."""
    query: str = Field(..., min_length=1, max_length=1000, description="User's question")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    doc_type: Optional[str] = Field(None, description="Filter by document type")
    department: Optional[str] = Field(None, description="Filter by department")
    stream: bool = Field(False, description="Enable streaming response")


class SourceInfo(BaseModel):
    """Source citation information."""
    document: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_id: str


class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str
    conversation_id: str
    sources: List[SourceInfo]
    citations: List[dict] = []
    confidence: str
    status: str
    suggestions: List[str] = []
    metadata: dict


class ConversationMessage(BaseModel):
    """Individual message in conversation."""
    id: str
    role: str
    content: str
    timestamp: str
    metadata: dict = {}


class ConversationDetail(BaseModel):
    """Complete conversation details."""
    id: str
    user_id: str
    created_at: str
    updated_at: str
    messages: List[ConversationMessage]
    metadata: dict = {}
    context: dict = {}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Chat with the AI assistant using RAG."""
    user_id = str(current_user.id)
    logger.info(f"Chat request from user {user_id}: '{request.query[:50]}...'")
    
    if request.stream:
        return StreamingResponse(
            _stream_chat(request, db, user_id),
            media_type="text/event-stream"
        )
    
    try:
        result = await chat_service.chat(
            query=request.query,
            conversation_id=request.conversation_id,
            user_id=user_id,
            db=db,
            doc_type=request.doc_type,
            department=request.department,
            stream=False
        )
        
        if result.get('error'):
            logger.error(f"Chat service returned error: {result['error']}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
        
        sources = []
        for src in result.get('sources', []):
            try:
                sources.append(SourceInfo(
                    document=src.get('document', 'Unknown'),
                    page=src.get('page'),
                    section=src.get('section'),
                    chunk_id=src.get('chunk_id', 'unknown')
                ))
            except Exception as e:
                logger.warning(f"Error formatting source: {str(e)}")
                continue
        
        logger.info(f"Chat service returned suggestions: {result.get('suggestions')}")
        
        response = ChatResponse(
            answer=result['answer'],
            conversation_id=result['conversation_id'],
            sources=sources,
            citations=result.get('citations', []),
            confidence=result.get('confidence', 'medium'),
            status=result.get('status', 'success'),
            suggestions=result.get('suggestions', []),
            metadata=result.get('metadata', {})
        )
        
        logger.info(f"Returning {len(response.suggestions)} suggestions to frontend")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )


async def _stream_chat(request: ChatRequest, db: AsyncSession, user_id: str):
    """Internal function to stream chat responses."""
    try:
        result_stream = await chat_service.chat(
            query=request.query,
            conversation_id=request.conversation_id,
            user_id=user_id,
            db=db,
            doc_type=request.doc_type,
            department=request.department,
            stream=True
        )
        
        async for chunk in result_stream:
            yield f"data: {json.dumps(chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        error_data = {'type': 'error', 'error': str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


@router.get("/chat/conversations")
async def list_conversations(
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user)
):
    """List all conversations for the current user."""
    user_id = str(current_user.id)
    
    try:
        conversations = await chat_service.list_conversations(
            user_id=user_id,
            limit=limit
        )
        return conversations
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/chat/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get complete conversation history."""
    user_id = str(current_user.id)
    
    try:
        conversation = await chat_service.get_conversation_history(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if 'error' in conversation:
            error_msg = conversation['error']
            if error_msg == 'Conversation not found':
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)
            else:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=error_msg)
        
        messages = []
        for msg in conversation.get('messages', []):
            try:
                messages.append(ConversationMessage(
                    id=msg.get('id', ''),
                    role=msg.get('role', 'user'),
                    content=msg.get('content', ''),
                    timestamp=msg.get('timestamp', ''),
                    metadata=msg.get('metadata', {})
                ))
            except Exception as e:
                logger.warning(f"Error formatting message: {str(e)}")
                continue
        
        return ConversationDetail(
            id=conversation['id'],
            user_id=conversation['user_id'],
            created_at=conversation['created_at'],
            updated_at=conversation['updated_at'],
            messages=messages,
            metadata=conversation.get('metadata', {}),
            context=conversation.get('context', {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a conversation and its history."""
    user_id = str(current_user.id)
    
    try:
        deleted = await chat_service.delete_conversation(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or unauthorized"
            )
        
        return {
            'message': 'Conversation deleted successfully',
            'conversation_id': conversation_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/chat/test")
async def test_chat():
    """Test endpoint to verify chat API is working."""
    return {
        'status': 'operational',
        'message': 'Chat API is ready',
        'features': [
            'RAG-based question answering',
            'Conversation history',
            'Source citations',
            'Input/output guardrails',
            'Streaming responses',
            'Natural conversational responses'
        ]
    }


@router.get("/chat/conversations/{conversation_id}/analytics")
async def get_conversation_analytics(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get analytics for a conversation."""
    user_id = str(current_user.id)
    
    try:
        from app.services.generation.context_manager import context_manager
        
        conversation = await chat_service.get_conversation_history(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if 'error' in conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=conversation['error']
            )
        
        context = await context_manager.get_or_create_context(conversation_id)
        messages = conversation.get('messages', [])
        user_messages = [m for m in messages if m['role'] == 'user']
        assistant_messages = [m for m in messages if m['role'] == 'assistant']
        
        documents_referenced = set()
        sources_count = 0
        
        for msg in assistant_messages:
            metadata = msg.get('metadata', {})
            sources = metadata.get('sources', [])
            sources_count += len(sources)
            for source in sources:
                documents_referenced.add(source.get('document', 'Unknown'))
        
        confidences = [
            msg.get('metadata', {}).get('confidence', 'medium')
            for msg in assistant_messages
        ]
        
        confidence_scores = {
            'high': confidences.count('high'),
            'medium': confidences.count('medium'),
            'low': confidences.count('low')
        }
        
        context_summary = context.get_context_summary() if context else {}
        
        analytics = {
            'conversation_id': conversation_id,
            'total_messages': len(messages),
            'user_queries': len(user_messages),
            'ai_responses': len(assistant_messages),
            'documents_referenced': list(documents_referenced),
            'total_documents': len(documents_referenced),
            'total_sources_cited': sources_count,
            'confidence_distribution': confidence_scores,
            'primary_document': context_summary.get('primary_document'),
            'active_documents': context_summary.get('active_documents', []),
            'topics': context_summary.get('topics', []),
            'time_periods_discussed': context_summary.get('time_periods', []),
            'created_at': conversation['created_at'],
            'duration_minutes': _calculate_duration(conversation),
        }
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _calculate_duration(conversation: Dict[str, Any]) -> float:
    """Calculate conversation duration in minutes."""
    from datetime import datetime
    created = datetime.fromisoformat(conversation['created_at'])
    updated = datetime.fromisoformat(conversation['updated_at'])
    duration = (updated - created).total_seconds() / 60
    return round(duration, 1)


def _export_to_markdown(conversation: Dict[str, Any]) -> str:
    """Convert conversation to markdown format."""
    lines = [
        f"# Conversation Export",
        f"",
        f"**Conversation ID:** {conversation['id']}",
        f"**Created:** {conversation['created_at']}",
        f"**Last Updated:** {conversation['updated_at']}",
        f"",
        f"---",
        f"",
    ]
    
    for msg in conversation['messages']:
        role = "**User**" if msg['role'] == 'user' else "**AI Assistant**"
        timestamp = msg.get('timestamp', '')
        
        lines.append(f"### {role} ({timestamp})")
        lines.append(f"")
        lines.append(msg['content'])
        lines.append(f"")
        
        if msg['role'] == 'assistant':
            metadata = msg.get('metadata', {})
            sources = metadata.get('sources', [])
            
            if sources:
                lines.append(f"**Sources:**")
                for source in sources:
                    lines.append(f"- {source.get('document')} (Page {source.get('page', 'N/A')})")
                lines.append(f"")
        
        lines.append(f"---")
        lines.append(f"")
    
    return "\n".join(lines)


@router.get("/chat/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: str,
    format: str = Query("markdown", pattern="^(markdown|json|pdf)$"),
    current_user: User = Depends(get_current_user)
):
    """
    Export conversation in various formats.
    
    Formats:
    - markdown: Formatted markdown with sources
    - json: Complete JSON with metadata
    - pdf: Professional PDF report
    """
    user_id = str(current_user.id)
    
    try:
        conversation = await chat_service.get_conversation_history(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if 'error' in conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=conversation['error']
            )
        
        if format == "json":
            return conversation
        
        elif format == "markdown":
            markdown = _export_to_markdown(conversation)
            return Response(
                content=markdown,
                media_type="text/markdown",
                headers={
                    "Content-Disposition": f"attachment; filename=conversation_{conversation_id[:8]}.md"
                }
            )
        
        elif format == "pdf":
            try:
                from app.services.export.pdf_generator import pdf_generator
                pdf_buffer = pdf_generator.generate_conversation_pdf(conversation)
                return Response(
                    content=pdf_buffer.read(),
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f"attachment; filename=conversation_{conversation_id[:8]}.pdf"
                    }
                )
            except ImportError:
                # reportlab not installed — fall back to markdown download
                logger.warning("reportlab not available, falling back to markdown export")
                markdown = _export_to_markdown(conversation)
                return Response(
                    content=markdown,
                    media_type="text/markdown",
                    headers={
                        "Content-Disposition": f"attachment; filename=conversation_{conversation_id[:8]}.md"
                    }
                )
            except Exception as e:
                logger.error(f"PDF generation failed: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"PDF generation failed: {str(e)}"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ====== DEBUG ENDPOINTS ======

@router.get("/chat/debug/database")
async def debug_database(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Debug endpoint: Check database contents."""
    doc_query = select(
        Document.status,
        func.count(Document.id).label('count')
    ).group_by(Document.status)
    
    doc_result = await db.execute(doc_query)
    doc_counts = {row.status: row.count for row in doc_result.all()}
    
    chunk_query = select(func.count(ChunkModel.id))
    chunk_result = await db.execute(chunk_query)
    total_chunks = chunk_result.scalar()
    
    embedding_check = await db.execute(
        text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
    )
    chunks_with_embeddings = embedding_check.scalar()
    
    sample_query = select(
        Document.filename,
        Document.status,
        Document.total_chunks,
        Document.upload_date
    ).limit(5)
    
    sample_result = await db.execute(sample_query)
    samples = [
        {
            'filename': row.filename,
            'status': row.status,
            'chunks': row.total_chunks,
            'uploaded': row.upload_date.isoformat() if row.upload_date else None
        }
        for row in sample_result.all()
    ]
    
    return {
        'database_status': 'connected',
        'documents_by_status': doc_counts,
        'total_chunks': total_chunks,
        'chunks_with_embeddings': chunks_with_embeddings,
        'sample_documents': samples,
        'diagnosis': {
            'has_documents': sum(doc_counts.values()) > 0,
            'has_chunks': total_chunks > 0,
            'has_embeddings': chunks_with_embeddings > 0,
            'ready_for_search': chunks_with_embeddings > 0
        }
    }


@router.get("/chat/debug/chunks")
async def debug_chunks(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Debug: Check chunk embeddings in database."""
    total_result = await db.execute(select(func.count(ChunkModel.id)))
    total_chunks = total_result.scalar()
    
    embedding_result = await db.execute(
        text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
    )
    chunks_with_embeddings = embedding_result.scalar()
    
    dimension_result = await db.execute(
        text("SELECT array_length(embedding, 1) as dim FROM chunks LIMIT 1")
    )
    dimension_row = dimension_result.first()
    actual_dimensions = dimension_row[0] if dimension_row else None
    
    sample_result = await db.execute(
        select(ChunkModel.id, ChunkModel.content, ChunkModel.chunk_type)
        .limit(1)
    )
    sample = sample_result.first()
    
    return {
        'total_chunks': total_chunks,
        'chunks_with_embeddings': chunks_with_embeddings,
        'expected_dimensions': 1536,
        'actual_dimensions': actual_dimensions,
        'dimension_match': actual_dimensions == 1536 if actual_dimensions else False,
        'sample_chunk': {
            'id': str(sample[0]),
            'content': sample[1][:100] + '...' if sample and sample[1] else None,
            'type': sample[2] if sample else None
        } if sample else None,
        'status': 'ready' if chunks_with_embeddings > 0 else 'no_data'
    }


@router.get("/chat/debug/test-embedding")
async def test_embedding(
    current_user: User = Depends(get_current_user)
):
    """Test if Gemini embedding API is working."""
    from app.services.embedding.embedding_service import embedding_service
    
    test_text = "This is a test sentence to verify Gemini embeddings are working."
    
    try:
        start_time = time.time()
        embedding = await embedding_service.generate_embedding(test_text)
        elapsed_time = time.time() - start_time
        
        return {
            'status': 'success',
            'model': embedding_service.model_name,
            'dimensions': len(embedding),
            'expected_dimensions': embedding_service.dimensions,
            'dimension_match': len(embedding) == embedding_service.dimensions,
            'elapsed_seconds': round(elapsed_time, 2),
            'sample_values': embedding[:5],
            'using_local_fallback': embedding_service.use_local_fallback
        }
    except Exception as e:
        logger.error(f"Embedding test failed: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'using_local_fallback': getattr(embedding_service, 'use_local_fallback', False)
        }


@router.get("/chat/debug/config")
async def debug_config(current_user: User = Depends(get_current_user)):
    """Debug: Check current configuration."""
    from app.core.config import settings
    
    return {
        'embedding_model': settings.gemini_embedding_model,
        'embedding_dimensions': settings.gemini_embedding_dimensions,
        'chat_model': settings.gemini_chat_model,
        'chunk_size': settings.chunk_size,
        'retrieval_top_k': settings.retrieval_top_k,
        'similarity_threshold': settings.similarity_threshold
    }

@router.get("/chat/conversations/{conversation_id}/token-stats")
async def get_conversation_token_stats(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed token usage and cost statistics for a conversation.
    """
    user_id = str(current_user.id)
    
    try:
        conversation = await chat_service.get_conversation_history(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if 'error' in conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=conversation['error']
            )
        
        messages = conversation.get('messages', [])
        assistant_messages = [m for m in messages if m['role'] == 'assistant']
        
        total_tokens = 0
        total_cached_tokens = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        
        message_stats = []
        
        for msg in assistant_messages:
            metadata = msg.get('metadata', {})
            tokens = metadata.get('tokens', {})
            
            msg_total = tokens.get('total_tokens', 0)
            msg_cached = tokens.get('cached_tokens', 0)
            msg_prompt = tokens.get('prompt_tokens', 0)
            msg_completion = tokens.get('completion_tokens', 0)
            
            msg_regular = msg_total - msg_cached
            msg_cost = (msg_cached * 0.00000003) + (msg_regular * 0.0000003)
            
            total_tokens += msg_total
            total_cached_tokens += msg_cached
            total_prompt_tokens += msg_prompt
            total_completion_tokens += msg_completion
            total_cost += msg_cost
            
            message_stats.append({
                'message_id': msg.get('id'),
                'timestamp': msg.get('timestamp'),
                'tokens': {
                    'total': msg_total,
                    'cached': msg_cached,
                    'prompt': msg_prompt,
                    'completion': msg_completion
                },
                'cost_usd': round(msg_cost, 6)
            })
        
        cache_efficiency = (total_cached_tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        return {
            'conversation_id': conversation_id,
            'summary': {
                'total_messages': len(assistant_messages),
                'total_tokens': total_tokens,
                'cached_tokens': total_cached_tokens,
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'cache_efficiency_percent': round(cache_efficiency, 2),
                'total_cost_usd': round(total_cost, 6),
                'avg_tokens_per_message': round(total_tokens / len(assistant_messages)) if assistant_messages else 0,
                'avg_cost_per_message': round(total_cost / len(assistant_messages), 6) if assistant_messages else 0
            },
            'messages': message_stats,
            'pricing_info': {
                'model': 'gemini-2.5-flash',
                'cached_token_price': 0.00000003,
                'regular_token_price': 0.0000003,
                'currency': 'USD'
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/chat/debug/list-models")
async def list_gemini_models():
    """List all available Gemini models."""
    import google.generativeai as genai
    from app.core.config import settings
    
    genai.configure(api_key=settings.gemini_api_key)
    
    try:
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append({
                    'name': m.name,
                    'display_name': m.display_name,
                    'description': m.description[:100] if m.description else ''
                })
        
        return {
            'available_models': models,
            'current_config': settings.gemini_chat_model
        }
    except Exception as e:
        return {'error': str(e)}
    
class SelectiveExportRequest(BaseModel):
    """Request model for selective export."""
    message_indices: List[int] = Field(..., description="Indices of messages to export")
    format: str = Field("markdown", pattern="^(markdown|json|pdf)$")


@router.post("/chat/conversations/{conversation_id}/export-selected")
async def export_selected_messages(
    conversation_id: str,
    request: SelectiveExportRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Export selected messages from conversation.
    
    Supports PDF, Markdown, and JSON formats with professional formatting.
    """
    user_id = str(current_user.id)
    
    try:
        # Get full conversation
        conversation = await chat_service.get_conversation_history(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if 'error' in conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=conversation['error']
            )
        
        # Filter messages by indices
        all_messages = conversation.get('messages', [])
        selected_messages = []
        
        for idx in sorted(request.message_indices):
            if 0 <= idx < len(all_messages):
                selected_messages.append(all_messages[idx])
        
        if not selected_messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid messages selected"
            )
        
        # Create filtered conversation object
        filtered_conversation = {
            **conversation,
            'messages': selected_messages,
            'metadata': {
                **conversation.get('metadata', {}),
                'is_selective_export': True,
                'total_messages_in_conversation': len(all_messages),
                'exported_message_count': len(selected_messages),
                'exported_indices': request.message_indices
            }
        }
        
        # Export based on format
        if request.format == "json":
            return filtered_conversation
        
        elif request.format == "markdown":
            markdown = _export_selected_to_markdown(filtered_conversation)
            return Response(
                content=markdown,
                media_type="text/markdown",
                headers={
                    "Content-Disposition": f"attachment; filename=conversation_selective_{conversation_id[:8]}.md"
                }
            )
        
        elif request.format == "pdf":
            try:
                from app.services.export.pdf_generator import pdf_generator
                
                # Generate PDF with selection info
                pdf_buffer = pdf_generator.generate_conversation_pdf(filtered_conversation)
                
                return Response(
                    content=pdf_buffer.read(),
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f"attachment; filename=conversation_selective_{conversation_id[:8]}.pdf"
                    }
                )
            except Exception as e:
                logger.error(f"PDF generation failed: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"PDF generation failed: {str(e)}"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in selective export: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _export_selected_to_markdown(conversation: Dict[str, Any]) -> str:
    """Convert selected messages to markdown format."""
    metadata = conversation.get('metadata', {})
    
    lines = [
        f"# Selective Conversation Export",
        f"",
        f"**Conversation ID:** {conversation['id']}",
        f"**Created:** {conversation['created_at']}",
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"**Selection:** {metadata.get('exported_message_count', 0)} of {metadata.get('total_messages_in_conversation', 0)} messages",
        f"",
        f"---",
        f"",
        f"## Selected Messages",
        f"",
    ]
    
    for idx, msg in enumerate(conversation['messages'], 1):
        role = "**User**" if msg['role'] == 'user' else "**AI Assistant**"
        timestamp = msg.get('timestamp', '')
        
        lines.append(f"### Message {idx}: {role}")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                lines.append(f"*{dt.strftime('%I:%M %p')}*")
            except Exception:
                lines.append(f"*{timestamp}*")
        lines.append(f"")
        lines.append(msg['content'])
        lines.append(f"")
        
        if msg['role'] == 'assistant':
            msg_metadata = msg.get('metadata', {})
            sources = msg_metadata.get('sources', [])
            
            if sources:
                lines.append(f"**Sources:**")
                for source in sources:
                    doc_name = source.get('document', 'Unknown')
                    page = source.get('page', 'N/A')
                    lines.append(f"- {doc_name} (Page {page})")
                lines.append(f"")
        
        lines.append(f"---")
        lines.append(f"")
    
    return "\n".join(lines)

__all__ = ['router']