"""
Admin API endpoints for user and system management.
Only accessible by users with admin role.
Includes context analytics and quality monitoring.
"""
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timedelta
from loguru import logger

from app.db.session import get_db
from app.models.user import User
from app.models.organization import Organization
from app.models.document import Document
from app.api.dependencies.auth import get_current_admin_user
from app.core.security import get_password_hash, validate_password_strength, validate_email


router = APIRouter()


# Request/Response Models
class CreateUserRequest(BaseModel):
    """Admin creates new user."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    role: str = Field(default="user", pattern="^(user|admin)$")
    is_active: bool = True


class UpdateUserRequest(BaseModel):
    """Admin updates user."""
    full_name: Optional[str] = None
    role: Optional[str] = Field(None, pattern="^(user|admin)$")
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None


class UserListItem(BaseModel):
    """User in admin list."""
    id: str
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    created_at: str
    last_login: Optional[str]
    document_count: int = 0


class UserStatsResponse(BaseModel):
    """User statistics."""
    total_users: int
    active_users: int
    admin_users: int
    regular_users: int
    verified_users: int


class SystemStatsResponse(BaseModel):
    """System-wide statistics."""
    total_users: int
    total_documents: int
    total_chunks: int
    active_sessions: int
    storage_used_mb: float


# User Management Endpoints

@router.get("/admin/users", response_model=dict)
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    role: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    List users within the admin's organization.
    Super admins see all users across all organizations.
    Org admins only see users in their organization.
    """
    # Base query
    query = select(User)
    
    # ORGANIZATION SCOPING: Org admins can only see their org users
    if admin.is_org_admin() and not admin.is_super_admin():
        query = query.where(User.organization_id == admin.organization_id)
    
    # Apply filters
    if role:
        query = query.where(User.role == role)
    
    if is_active is not None:
        query = query.where(User.is_active == is_active)
    
    if search:
        search_pattern = f"%{search}%"
        query = query.where(
            (User.email.ilike(search_pattern)) |
            (User.username.ilike(search_pattern)) |
            (User.full_name.ilike(search_pattern))
        )
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Get users
    query = query.order_by(User.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    users = result.scalars().all()
    
    # Get document counts for each user
    user_list = []
    for user in users:
        doc_count_query = select(func.count()).select_from(Document).where(
            Document.uploaded_by == user.id
        )
        doc_count_result = await db.execute(doc_count_query)
        doc_count = doc_count_result.scalar()
        
        user_list.append(
            UserListItem(
                id=str(user.id),
                email=user.email,
                username=user.username,
                full_name=user.full_name,
                role=user.role,
                is_active=user.is_active,
                is_verified=user.is_verified,
                created_at=user.created_at.isoformat(),
                last_login=user.last_login.isoformat() if user.last_login else None,
                document_count=doc_count
            )
        )
    
    return {
        "total": total,
        "users": user_list
    }


@router.get("/admin/users/stats", response_model=UserStatsResponse)
async def get_user_stats(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Get user statistics.
    Organization-scoped for org admins, global for super admins.
    """
    # Base queries
    user_query = select(func.count()).select_from(User)
    active_query = select(func.count()).select_from(User).where(User.is_active == True)
    admin_query = select(func.count()).select_from(User).where(User.role.in_(["org_admin", "super_admin"]))
    verified_query = select(func.count()).select_from(User).where(User.is_verified == True)
    
    # ORGANIZATION SCOPING: Apply org filter for org admins
    if admin.is_org_admin() and not admin.is_super_admin():
        org_filter = User.organization_id == admin.organization_id
        user_query = user_query.where(org_filter)
        active_query = active_query.where(org_filter)
        admin_query = admin_query.where(org_filter)
        verified_query = verified_query.where(org_filter)
    
    # Execute queries
    total_result = await db.execute(user_query)
    total = total_result.scalar()
    
    active_result = await db.execute(active_query)
    active = active_result.scalar()
    
    admin_result = await db.execute(admin_query)
    admins = admin_result.scalar()
    
    verified_result = await db.execute(verified_query)
    verified = verified_result.scalar()
    
    return UserStatsResponse(
        total_users=total,
        active_users=active,
        admin_users=admins,
        regular_users=total - admins,
        verified_users=verified
    )


@router.post("/admin/users", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: CreateUserRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Create a new user within the admin's organization.
    Super admins can create users in any organization (requires organization_id).
    Org admins can only create users in their own organization.
    """
    # Validate email
    if not validate_email(request.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )
    
    # Validate password
    is_valid, error_msg = validate_password_strength(request.password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    
    # ROLE VALIDATION: Prevent org admins from creating super admins
    if request.role == "super_admin" and not admin.is_super_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only super admins can create super admin users"
        )
    
    # ROLE VALIDATION: Prevent org admins from creating other org admins
    if request.role == "org_admin" and admin.is_org_admin() and not admin.is_super_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only super admins can create organization admins"
        )
    
    # Check if email exists
    result = await db.execute(select(User).where(User.email == request.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username exists
    result = await db.execute(select(User).where(User.username == request.username))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # ORGANIZATION ASSIGNMENT: Org admins can only create users in their org
    organization_id = admin.organization_id
    
    # Create user
    new_user = User(
        email=request.email,
        username=request.username,
        hashed_password=get_password_hash(request.password),
        full_name=request.full_name,
        role=request.role,
        is_active=request.is_active,
        is_verified=True,
        organization_id=organization_id,  # ORGANIZATION SCOPING
        user_metadata={"created_by": str(admin.id), "created_by_admin": True}
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    logger.info(f"Admin {admin.email} created new user: {new_user.email} in organization {organization_id}")
    
    return {
        "message": "User created successfully",
        "user": new_user.to_dict()
    }


@router.get("/admin/users/{user_id}", response_model=dict)
async def get_user_details(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Get detailed information about a specific user.
    Admin only.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get document stats
    doc_count_query = select(func.count()).select_from(Document).where(
        Document.uploaded_by == user_id
    )
    doc_count_result = await db.execute(doc_count_query)
    doc_count = doc_count_result.scalar()
    
    # Get recent documents
    recent_docs_query = select(Document).where(
        Document.uploaded_by == user_id
    ).order_by(Document.upload_date.desc()).limit(5)
    recent_docs_result = await db.execute(recent_docs_query)
    recent_docs = recent_docs_result.scalars().all()
    
    return {
        "user": user.to_dict(),
        "stats": {
            "total_documents": doc_count,
        },
        "recent_documents": [
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "upload_date": doc.upload_date.isoformat(),
                "status": doc.status
            }
            for doc in recent_docs
        ]
    }


@router.put("/admin/users/{user_id}", response_model=dict)
async def update_user(
    user_id: UUID,
    request: UpdateUserRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Update user information.
    Org admins can only update users in their organization.
    Super admins can update any user.
    """
    # Get user
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # ORGANIZATION SCOPING: Org admins can only update users in their org
    if admin.is_org_admin() and not admin.is_super_admin():
        if user.organization_id != admin.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only update users in your organization"
            )
    
    # ROLE PROTECTION: Cannot change super admin roles
    if user.role == "super_admin" and not admin.is_super_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot modify super admin users"
        )
    
    # ROLE PROTECTION: Org admins cannot create/modify org admins
    if request.role == "org_admin" and admin.is_org_admin() and not admin.is_super_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only super admins can assign organization admin role"
        )
    
    # SELF-PROTECTION: Prevent admin from deactivating themselves
    if user_id == admin.id and request.is_active is False:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    # SELF-PROTECTION: Prevent admin from demoting themselves
    if user_id == admin.id and request.role and request.role not in ["org_admin", "super_admin"]:
        # Check if there are other admins in the organization
        if admin.is_org_admin() and not admin.is_super_admin():
            admin_count_result = await db.execute(
                select(func.count()).select_from(User).where(
                    User.role == "org_admin",
                    User.is_active == True,
                    User.organization_id == admin.organization_id,
                    User.id != admin.id
                )
            )
            other_admins = admin_count_result.scalar()
            
            if other_admins == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot remove admin role. You are the only active admin in your organization."
                )
    
    # Update fields
    if request.full_name is not None:
        user.full_name = request.full_name
    if request.role is not None:
        user.role = request.role
    if request.is_active is not None:
        user.is_active = request.is_active
    if request.is_verified is not None:
        user.is_verified = request.is_verified
    
    user.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(user)
    
    logger.info(f"Admin {admin.email} updated user {user.email}")
    
    return {
        "message": "User updated successfully",
        "user": user.to_dict()
    }


@router.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Delete a user and all their data.
    Org admins can only delete users in their organization.
    Super admins can delete any user.
    """
    # Get user
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # ORGANIZATION SCOPING: Org admins can only delete users in their org
    if admin.is_org_admin() and not admin.is_super_admin():
        if user.organization_id != admin.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only delete users in your organization"
            )
    
    # SELF-PROTECTION: Prevent admin from deleting themselves
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    # ROLE PROTECTION: Cannot delete super admins
    if user.role == "super_admin" and not admin.is_super_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete super admin users"
        )
    
    # Delete user's documents and chunks (cascade should handle this)
    await db.execute(delete(Document).where(Document.uploaded_by == user_id))
    
    # Delete user
    await db.delete(user)
    await db.commit()
    
    logger.info(f"Admin {admin.email} deleted user {user.email}")
    
    return {
        "message": "User deleted successfully",
        "user_id": str(user_id)
    }


@router.post("/admin/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: UUID,
    new_password: str = Query(..., min_length=8),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Reset a user's password.
    Admin only.
    """
    # Validate password
    is_valid, error_msg = validate_password_strength(new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update password
    user.hashed_password = get_password_hash(new_password)
    user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    logger.info(f"Admin {admin.email} reset password for user {user.email}")
    
    return {
        "message": "Password reset successfully"
    }


# System Management Endpoints

@router.get("/admin/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Get system-wide statistics.
    Organization-scoped for org admins, global for super admins.
    """
    # Base queries
    user_query = select(func.count()).select_from(User)
    doc_query = select(func.count()).select_from(Document)
    storage_query = select(func.sum(Document.file_size_bytes)).select_from(Document)
    
    # ORGANIZATION SCOPING: Apply org filter for org admins
    if admin.is_org_admin() and not admin.is_super_admin():
        org_filter_user = User.organization_id == admin.organization_id
        org_filter_doc = Document.organization_id == admin.organization_id
        
        user_query = user_query.where(org_filter_user)
        doc_query = doc_query.where(org_filter_doc)
        storage_query = storage_query.where(org_filter_doc)
    
    # Total users
    user_count_result = await db.execute(user_query)
    total_users = user_count_result.scalar()
    
    # Total documents
    doc_count_result = await db.execute(doc_query)
    total_docs = doc_count_result.scalar()
    
    # Total chunks
    from app.models.document import Chunk
    chunk_query = select(func.count()).select_from(Chunk)
    
    # ORGANIZATION SCOPING for chunks (via document relationship)
    if admin.is_org_admin() and not admin.is_super_admin():
        chunk_query = chunk_query.join(Document).where(
            Document.organization_id == admin.organization_id
        )
    
    chunk_count_result = await db.execute(chunk_query)
    total_chunks = chunk_count_result.scalar()
    
    # Storage used
    storage_result = await db.execute(storage_query)
    storage_bytes = storage_result.scalar() or 0
    storage_mb = storage_bytes / (1024 * 1024)
    
    return SystemStatsResponse(
        total_users=total_users,
        total_documents=total_docs,
        total_chunks=total_chunks,
        active_sessions=0,
        storage_used_mb=round(storage_mb, 2)
    )


@router.get("/admin/documents", response_model=dict)
async def list_all_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    doc_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    user_id: Optional[UUID] = Query(None),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    List all documents.
    Organization-scoped for org admins, global for super admins.
    """
    # Build query
    query = select(Document)
    
    # ORGANIZATION SCOPING: Org admins only see their org's documents
    if admin.is_org_admin() and not admin.is_super_admin():
        query = query.where(Document.organization_id == admin.organization_id)
    
    if doc_type:
        query = query.where(Document.doc_type == doc_type)
    
    if status:
        query = query.where(Document.status == status)
    
    if user_id:
        query = query.where(Document.uploaded_by == user_id)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Get documents
    query = query.order_by(Document.upload_date.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    documents = result.scalars().all()
    
    # Get user info for each document
    doc_list = []
    for doc in documents:
        user_result = await db.execute(select(User).where(User.id == doc.uploaded_by))
        user = user_result.scalar_one_or_none()
        
        doc_list.append({
            "id": str(doc.id),
            "filename": doc.filename,
            "doc_type": doc.doc_type,
            "department": doc.department,
            "total_pages": doc.total_pages or 0,
            "total_chunks": doc.total_chunks or 0,
            "status": doc.status,
            "upload_date": doc.upload_date.isoformat(),
            "processed_date": doc.processed_date.isoformat() if doc.processed_date else None,
            "uploaded_by": {
                "id": str(user.id),
                "email": user.email,
                "username": user.username
            } if user else None
        })
    
    return {
        "total": total,
        "documents": doc_list
    }


# ============================================================================
# NEW: CONTEXT ANALYTICS & MONITORING ENDPOINTS
# ============================================================================

@router.get("/admin/analytics/context")
async def get_context_analytics(
    days: int = Query(7, ge=1, le=90),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get context usage analytics.
    
    Admin only. Shows:
    - Context hit rate
    - Most referenced documents
    - Query reformulation effectiveness
    - Average conversation length
    """
    try:
        from app.services.generation.context_manager import context_manager
        
        # Get all active contexts
        contexts = context_manager.contexts
        
        # Calculate statistics
        total_contexts = len(contexts)
        contexts_with_docs = sum(1 for c in contexts.values() if c.primary_document)
        
        # Document reference frequency
        doc_references = {}
        for context in contexts.values():
            for ref in context.document_references:
                doc = ref['document']
                doc_references[doc] = doc_references.get(doc, 0) + 1
        
        # Sort by frequency
        top_documents = sorted(
            doc_references.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Time period tracking
        time_periods = []
        for context in contexts.values():
            time_periods.extend(context.time_periods)
        
        time_period_count = len(set(time_periods))
        
        # Calculate averages
        avg_messages = sum(c.message_count for c in contexts.values()) / max(total_contexts, 1)
        
        # Entity tracking
        all_entities = {}
        for context in contexts.values():
            for entity_type, values in context.entities.items():
                if entity_type not in all_entities:
                    all_entities[entity_type] = []
                all_entities[entity_type].extend(values)
        
        entity_stats = {
            entity_type: len(set(values))
            for entity_type, values in all_entities.items()
        }
        
        analytics = {
            'total_active_contexts': total_contexts,
            'contexts_with_document_scope': contexts_with_docs,
            'document_scope_rate': round(contexts_with_docs / max(total_contexts, 1) * 100, 1),
            'top_referenced_documents': [
                {'document': doc, 'reference_count': count}
                for doc, count in top_documents
            ],
            'unique_time_periods_tracked': time_period_count,
            'average_messages_per_conversation': round(avg_messages, 1),
            'entity_tracking': entity_stats,
            'context_features': {
                'query_reformulation': True,
                'document_scoping': True,
                'time_period_tracking': True,
                'entity_extraction': True,
            },
            'performance': {
                'avg_context_age_minutes': round(
                    sum((datetime.utcnow() - c.created_at).seconds / 60 for c in contexts.values()) / max(total_contexts, 1),
                    1
                ) if total_contexts > 0 else 0
            }
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting context analytics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/admin/analytics/retrieval")
async def get_retrieval_analytics(
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get retrieval performance analytics.
    
    Shows:
    - Average retrieval time
    - Hybrid vs semantic vs keyword usage
    - Reranking impact
    - Context-aware retrieval effectiveness
    """
    # This would ideally pull from a metrics database
    # For now, return structure with placeholder data
    
    return {
        'retrieval_strategies': {
            'hybrid': {'count': 0, 'avg_time_ms': 150},
            'semantic': {'count': 0, 'avg_time_ms': 120},
            'keyword': {'count': 0, 'avg_time_ms': 80},
        },
        'context_aware_searches': {
            'scoped': 0,
            'global': 0,
            'fallback_triggered': 0,
        },
        'reranking': {
            'enabled': True,
            'avg_score_improvement': 0.15,
            'rerank_model': 'rerank-english-v3.0',
        },
        'performance': {
            'avg_retrieval_time_ms': 150,
            'p95_retrieval_time_ms': 300,
            'p99_retrieval_time_ms': 500,
        },
        'quality_metrics': {
            'avg_similarity_score': 0.75,
            'avg_rerank_score': 0.82,
            'documents_with_embeddings': 100,
        }
    }


@router.get("/admin/conversations/quality")
async def get_conversation_quality_metrics(
    limit: int = Query(50, ge=10, le=200),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get conversation quality metrics.
    
    Admin only. Shows:
    - Low confidence responses
    - Conversations with no sources
    - Error rates
    - User satisfaction indicators
    """
    try:
        from app.services.generation.conversation_manager import conversation_manager
        
        all_conversations = conversation_manager.conversations
        
        # Analyze quality
        low_confidence_count = 0
        no_sources_count = 0
        error_count = 0
        total_responses = 0
        
        quality_issues = []
        
        for conv_id, conv in list(all_conversations.items())[:limit]:
            messages = conv.get('messages', [])
            
            for msg in messages:
                if msg['role'] == 'assistant':
                    total_responses += 1
                    metadata = msg.get('metadata', {})
                    
                    # Check confidence
                    if metadata.get('confidence') == 'low':
                        low_confidence_count += 1
                        quality_issues.append({
                            'conversation_id': conv_id,
                            'issue': 'low_confidence',
                            'message_preview': msg['content'][:100] + '...',
                            'timestamp': msg.get('timestamp')
                        })
                    
                    # Check sources
                    if not metadata.get('sources'):
                        no_sources_count += 1
                    
                    # Check errors
                    if metadata.get('error'):
                        error_count += 1
                        quality_issues.append({
                            'conversation_id': conv_id,
                            'issue': 'error',
                            'message_preview': msg['content'][:100] + '...',
                            'timestamp': msg.get('timestamp')
                        })
        
        # Calculate rates
        low_conf_rate = (low_confidence_count / max(total_responses, 1)) * 100
        no_sources_rate = (no_sources_count / max(total_responses, 1)) * 100
        error_rate = (error_count / max(total_responses, 1)) * 100
        
        return {
            'total_conversations_analyzed': min(limit, len(all_conversations)),
            'total_responses': total_responses,
            'quality_metrics': {
                'low_confidence_responses': low_confidence_count,
                'low_confidence_rate': round(low_conf_rate, 1),
                'responses_without_sources': no_sources_count,
                'no_sources_rate': round(no_sources_rate, 1),
                'error_responses': error_count,
                'error_rate': round(error_rate, 1),
            },
            'quality_issues': quality_issues[:20],
            'recommendations': _generate_quality_recommendations(
                low_confidence_count,
                no_sources_count,
                error_count
            ),
            'health_status': _calculate_health_status(low_conf_rate, error_rate)
        }
        
    except Exception as e:
        logger.error(f"Error getting quality metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/admin/analytics/conversations")
async def get_conversation_analytics(
    days: int = Query(7, ge=1, le=90),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get overall conversation analytics.
    
    Shows:
    - Total conversations
    - Average conversation length
    - Popular topics
    - Usage patterns
    """
    try:
        from app.services.generation.conversation_manager import conversation_manager
        from app.services.generation.context_manager import context_manager
        
        all_conversations = conversation_manager.conversations
        
        # Calculate metrics
        total_conversations = len(all_conversations)
        total_messages = sum(len(c.get('messages', [])) for c in all_conversations.values())
        
        # Average length
        avg_length = total_messages / max(total_conversations, 1)
        
        # Collect topics from contexts
        all_topics = []
        for conv_id in all_conversations.keys():
            context = context_manager.get_context(conv_id)
            if context:
                all_topics.extend(context.topics)
        
        # Count topic frequency
        from collections import Counter
        topic_counts = Counter(all_topics)
        top_topics = topic_counts.most_common(10)
        
        # Time distribution (by hour)
        hour_distribution = {}
        for conv in all_conversations.values():
            created = datetime.fromisoformat(conv['created_at'])
            hour = created.hour
            hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
        
        return {
            'period_days': days,
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'average_conversation_length': round(avg_length, 1),
            'top_topics': [
                {'topic': topic, 'count': count}
                for topic, count in top_topics
            ],
            'usage_by_hour': [
                {'hour': hour, 'conversations': count}
                for hour, count in sorted(hour_distribution.items())
            ],
            'active_conversations': sum(
                1 for c in all_conversations.values()
                if (datetime.utcnow() - datetime.fromisoformat(c['updated_at'])).seconds < 3600
            )
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation analytics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Helper Functions

def _generate_quality_recommendations(
    low_conf: int,
    no_sources: int,
    errors: int
) -> List[str]:
    """Generate recommendations based on quality metrics."""
    recommendations = []
    
    if low_conf > 10:
        recommendations.append(
            "High number of low-confidence responses detected. "
            "Consider: 1) Improving document quality, 2) Adjusting similarity thresholds, "
            "3) Adding more training documents"
        )
    
    if no_sources > 20:
        recommendations.append(
            "Many responses without sources. This could indicate: "
            "1) Too many conversational queries, 2) Documents not covering user questions, "
            "3) Retrieval threshold too high"
        )
    
    if errors > 5:
        recommendations.append(
            "Error rate is elevated. Check: 1) API quotas, 2) Database connections, "
            "3) System logs for recurring issues"
        )
    
    if not recommendations:
        recommendations.append("System quality metrics are healthy!")
    
    return recommendations


def _calculate_health_status(low_conf_rate: float, error_rate: float) -> str:
    """Calculate overall system health status."""
    if error_rate > 10 or low_conf_rate > 30:
        return "critical"
    elif error_rate > 5 or low_conf_rate > 20:
        return "warning"
    else:
        return "healthy"


__all__ = ['router']