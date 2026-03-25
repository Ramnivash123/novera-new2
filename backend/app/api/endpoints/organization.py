"""
Organization management API endpoints.
Super admin only - manages organizations, assignments, and statistics.
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from pydantic import BaseModel, Field
from datetime import datetime
from loguru import logger

from app.db.session import get_db
from app.models.user import User
from app.models.organization import Organization
from app.models.document import Document
from app.models.customization import OrganizationCustomization
from pydantic import BaseModel, Field, validator
from app.api.dependencies.auth import get_current_user

router = APIRouter()


async def get_current_super_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    if not current_user.is_super_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin access required"
        )
    return current_user


class CreateOrganizationRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    display_name: str = Field(..., min_length=2, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    max_users: Optional[int] = Field(None, ge=1)
    max_documents: Optional[int] = Field(None, ge=1)
    max_storage_gb: Optional[int] = Field(None, ge=1)  # Changed from max_storage_mb
    settings: Optional[dict] = None

    @validator('name')
    def sanitize_name(cls, v):
        """
        Auto-sanitize organization name to create valid slug.
        - Convert to lowercase
        - Replace spaces with hyphens
        - Remove invalid characters
        - Strip leading/trailing hyphens
        """
        if not v:
            raise ValueError('Organization name is required')
        
        # Convert to lowercase and strip whitespace
        sanitized = v.lower().strip()
        
        # Replace spaces with hyphens
        sanitized = sanitized.replace(' ', '-')
        
        # Remove any characters that aren't lowercase letters, numbers, hyphens, or underscores
        import re
        sanitized = re.sub(r'[^a-z0-9_-]', '', sanitized)
        
        # Remove consecutive hyphens/underscores
        sanitized = re.sub(r'[-_]+', '-', sanitized)
        
        # Strip leading/trailing hyphens/underscores
        sanitized = sanitized.strip('-_')
        
        if not sanitized:
            raise ValueError('Organization name must contain at least one alphanumeric character')
        
        if len(sanitized) < 2:
            raise ValueError('Organization name must be at least 2 characters long')
        
        return sanitized

class UpdateOrganizationRequest(BaseModel):
    display_name: Optional[str] = Field(None, min_length=2, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    is_active: Optional[bool] = None
    max_users: Optional[int] = Field(None, ge=1)
    max_documents: Optional[int] = Field(None, ge=1)
    max_storage_gb: Optional[int] = Field(None, ge=1)
    settings: Optional[dict] = None


class OrganizationResponse(BaseModel):
    id: str
    name: str
    display_name: str
    description: Optional[str]
    is_active: bool
    max_users: Optional[int]
    max_documents: Optional[int]
    max_storage_gb: Optional[int]
    settings: dict
    created_at: str
    updated_at: str
    user_count: int = 0
    document_count: int = 0
    storage_used_mb: float = 0.0
    admin_count: int = 0


class OrganizationListResponse(BaseModel):
    total: int
    organizations: List[OrganizationResponse]


@router.post("/super-admin/organizations", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_organization(
    request: CreateOrganizationRequest,
    db: AsyncSession = Depends(get_db),
    super_admin: User = Depends(get_current_super_admin)
):
    """
    Create a new organization with auto-sanitized slug.
    """
    # Check if organization with sanitized name already exists
    result = await db.execute(
        select(Organization).where(Organization.name == request.name)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Organization with name '{request.name}' already exists"
        )
    
    try:
        organization = Organization(
            name=request.name,
            slug=request.name,
            display_name=request.display_name,
            description=request.description,
            max_users=request.max_users,
            max_documents=request.max_documents,
            max_storage_gb=request.max_storage_gb,
            settings=request.settings or {},
            created_by=super_admin.id
        )
        
        db.add(organization)
        await db.commit()
        await db.refresh(organization)
        
        # Create default customization for the organization
        customization = OrganizationCustomization(
            organization_id=organization.id,
            primary_color="#0ea5e9",
            secondary_color="#d946ef",
            accent_color="#8b5cf6"
        )
        db.add(customization)
        await db.commit()
        
        logger.info(f"Super admin {super_admin.email} created organization: {organization.name}")
        
        return {
            "message": "Organization created successfully",
            "organization": organization.to_dict()
        }
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Organization creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Organization creation failed"
        )

@router.get("/super-admin/organizations", response_model=OrganizationListResponse)
async def list_organizations(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    is_active: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    super_admin: User = Depends(get_current_super_admin)
):
    query = select(Organization)
    
    if is_active is not None:
        query = query.where(Organization.is_active == is_active)
    
    if search:
        search_pattern = f"%{search}%"
        query = query.where(
            (Organization.name.ilike(search_pattern)) |
            (Organization.display_name.ilike(search_pattern))
        )
    
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    query = query.order_by(Organization.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    organizations = result.scalars().all()
    
    org_list = []
    for org in organizations:
        user_count_result = await db.execute(
            select(func.count()).select_from(User).where(User.organization_id == org.id)
        )
        user_count = user_count_result.scalar()
        
        admin_count_result = await db.execute(
            select(func.count()).select_from(User).where(
                and_(User.organization_id == org.id, User.role == "org_admin")
            )
        )
        admin_count = admin_count_result.scalar()
        
        doc_count_result = await db.execute(
            select(func.count()).select_from(Document).where(Document.organization_id == org.id)
        )
        doc_count = doc_count_result.scalar()
        
        storage_result = await db.execute(
            select(func.sum(Document.file_size_bytes)).select_from(Document).where(
                Document.organization_id == org.id
            )
        )
        storage_bytes = storage_result.scalar() or 0
        storage_mb = storage_bytes / (1024 * 1024)
        
        org_list.append(
            OrganizationResponse(
                id=str(org.id),
                name=org.name,
                display_name=org.display_name,
                description=org.description,
                is_active=org.is_active,
                max_users=org.max_users,
                max_documents=org.max_documents,
                max_storage_gb=org.max_storage_gb,
                settings=org.settings,
                created_at=org.created_at.isoformat(),
                updated_at=org.updated_at.isoformat(),
                user_count=user_count,
                document_count=doc_count,
                storage_used_mb=round(storage_mb, 2),
                admin_count=admin_count
            )
        )
    
    return OrganizationListResponse(
        total=total,
        organizations=org_list
    )


@router.get("/super-admin/organizations/{organization_id}", response_model=dict)
async def get_organization_details(
    organization_id: UUID,
    db: AsyncSession = Depends(get_db),
    super_admin: User = Depends(get_current_super_admin)
):
    result = await db.execute(
        select(Organization).where(Organization.id == organization_id)
    )
    org = result.scalar_one_or_none()
    
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )
    
    user_count_result = await db.execute(
        select(func.count()).select_from(User).where(User.organization_id == org.id)
    )
    user_count = user_count_result.scalar()
    
    admin_count_result = await db.execute(
        select(func.count()).select_from(User).where(
            and_(User.organization_id == org.id, User.role == "org_admin")
        )
    )
    admin_count = admin_count_result.scalar()
    
    doc_count_result = await db.execute(
        select(func.count()).select_from(Document).where(Document.organization_id == org.id)
    )
    doc_count = doc_count_result.scalar()
    
    storage_result = await db.execute(
        select(func.sum(Document.file_size_bytes)).select_from(Document).where(
            Document.organization_id == org.id
        )
    )
    storage_bytes = storage_result.scalar() or 0
    storage_mb = storage_bytes / (1024 * 1024)
    
    recent_users_result = await db.execute(
        select(User).where(User.organization_id == org.id)
        .order_by(User.created_at.desc()).limit(5)
    )
    recent_users = recent_users_result.scalars().all()
    
    admins_result = await db.execute(
        select(User).where(
            and_(User.organization_id == org.id, User.role == "org_admin")
        ).order_by(User.created_at.desc())
    )
    admins = admins_result.scalars().all()
    
    return {
        "organization": org.to_dict(),
        "statistics": {
            "total_users": user_count,
            "total_admins": admin_count,
            "total_documents": doc_count,
            "storage_used_mb": round(storage_mb, 2),
            "storage_used_gb": round(storage_mb / 1024, 2),  
            "max_storage_gb": org.max_storage_gb,
            "storage_limit_mb": org.max_storage_gb * 1024 if org.max_storage_gb else None,
            "storage_percentage": round((storage_mb / (org.max_storage_gb * 1024) * 100), 1) if org.max_storage_gb else None
        },
        "recent_users": [
            {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "created_at": user.created_at.isoformat()
            }
            for user in recent_users
        ],
        "admins": [
            {
                "id": str(admin.id),
                "username": admin.username,
                "email": admin.email,
                "created_at": admin.created_at.isoformat()
            }
            for admin in admins
        ]
    }


@router.put("/super-admin/organizations/{organization_id}", response_model=dict)
async def update_organization(
    organization_id: UUID,
    request: UpdateOrganizationRequest,
    db: AsyncSession = Depends(get_db),
    super_admin: User = Depends(get_current_super_admin)
):
    result = await db.execute(
        select(Organization).where(Organization.id == organization_id)
    )
    org = result.scalar_one_or_none()
    
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )
    
    update_data = request.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        if value is not None:
            setattr(org, field, value)
    
    org.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(org)
    
    logger.info(f"Super admin {super_admin.email} updated organization {org.name}")
    
    return {
        "message": "Organization updated successfully",
        "organization": org.to_dict()
    }


@router.delete("/super-admin/organizations/{organization_id}")
async def delete_organization(
    organization_id: UUID,
    db: AsyncSession = Depends(get_db),
    super_admin: User = Depends(get_current_super_admin)
):
    result = await db.execute(
        select(Organization).where(Organization.id == organization_id)
    )
    org = result.scalar_one_or_none()
    
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )
    
    if org.name == "default":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete default organization"
        )
    
    user_count_result = await db.execute(
        select(func.count()).select_from(User).where(User.organization_id == org.id)
    )
    user_count = user_count_result.scalar()
    
    if user_count > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete organization with {user_count} users. Please reassign or delete users first."
        )
    
    await db.delete(org)
    await db.commit()
    
    logger.info(f"Super admin {super_admin.email} deleted organization {org.name}")
    
    return {
        "message": "Organization deleted successfully",
        "organization_id": str(organization_id)
    }


@router.get("/super-admin/organizations/{organization_id}/users")
async def get_organization_users(
    organization_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    role: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db),
    super_admin: User = Depends(get_current_super_admin)
):
    result = await db.execute(
        select(Organization).where(Organization.id == organization_id)
    )
    org = result.scalar_one_or_none()
    
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )
    
    query = select(User).where(User.organization_id == organization_id)
    
    if role:
        query = query.where(User.role == role)
    
    if is_active is not None:
        query = query.where(User.is_active == is_active)
    
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    query = query.order_by(User.created_at.desc()).offset(skip).limit(limit)
    users_result = await db.execute(query)
    users = users_result.scalars().all()
    
    return {
        "total": total,
        "organization": org.to_dict(),
        "users": [user.to_dict() for user in users]
    }


@router.post("/super-admin/organizations/{organization_id}/assign-admin")
async def assign_organization_admin(
    organization_id: UUID,
    user_id: UUID = Query(...),
    db: AsyncSession = Depends(get_db),
    super_admin: User = Depends(get_current_super_admin)
):
    org_result = await db.execute(
        select(Organization).where(Organization.id == organization_id)
    )
    org = org_result.scalar_one_or_none()
    
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )
    
    user_result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if user.organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User does not belong to this organization"
        )
    
    if user.role == "super_admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change role of super admin"
        )
    
    user.role = "org_admin"
    user.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(user)
    
    logger.info(f"Super admin {super_admin.email} assigned {user.email} as admin of {org.name}")
    
    return {
        "message": f"User {user.email} is now an admin of {org.display_name}",
        "user": user.to_dict()
    }


@router.post("/super-admin/organizations/{organization_id}/revoke-admin")
async def revoke_organization_admin(
    organization_id: UUID,
    user_id: UUID = Query(...),
    db: AsyncSession = Depends(get_db),
    super_admin: User = Depends(get_current_super_admin)
):
    user_result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if user.organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User does not belong to this organization"
        )
    
    if user.role != "org_admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not an organization admin"
        )
    
    user.role = "user"
    user.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(user)
    
    logger.info(f"Super admin {super_admin.email} revoked admin role from {user.email}")
    
    return {
        "message": f"Admin role revoked from {user.email}",
        "user": user.to_dict()
    }


@router.get("/super-admin/statistics")
async def get_super_admin_statistics(
    db: AsyncSession = Depends(get_db),
    super_admin: User = Depends(get_current_super_admin)
):
    org_count_result = await db.execute(select(func.count()).select_from(Organization))
    total_orgs = org_count_result.scalar()
    
    active_org_result = await db.execute(
        select(func.count()).select_from(Organization).where(Organization.is_active == True)
    )
    active_orgs = active_org_result.scalar()
    
    user_count_result = await db.execute(select(func.count()).select_from(User))
    total_users = user_count_result.scalar()
    
    doc_count_result = await db.execute(select(func.count()).select_from(Document))
    total_docs = doc_count_result.scalar()
    
    storage_result = await db.execute(select(func.sum(Document.file_size_bytes)).select_from(Document))
    total_storage_bytes = storage_result.scalar() or 0
    total_storage_mb = total_storage_bytes / (1024 * 1024)
    
    return {
        "total_organizations": total_orgs,
        "active_organizations": active_orgs,
        "total_users": total_users,
        "total_documents": total_docs,
        "total_storage_mb": round(total_storage_mb, 2)
    }


__all__ = ['router']