from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.session import Base


class Organization(Base):
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), unique=True, nullable=False, index=True)
    slug = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=True)
    
    is_active = Column(Boolean, default=True, nullable=False)
    
    max_users = Column(Integer, nullable=True)
    max_documents = Column(Integer, nullable=True)
    max_storage_gb = Column(Integer, nullable=True)
    
    settings = Column(JSONB, nullable=False, default=dict)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), nullable=True)
    
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="organization", cascade="all, delete-orphan")
    customization = relationship("OrganizationCustomization", back_populates="organization", uselist=False)
    
    __table_args__ = (
        Index('idx_org_name_active', 'name', 'is_active'),
    )
    
    def __repr__(self):
        return f"<Organization(id={self.id}, name={self.name})>"
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "slug": self.slug,
            "display_name": self.display_name,
            "description": self.description,
            "is_active": self.is_active,
            "max_users": self.max_users,
            "max_documents": self.max_documents,
            "max_storage_gb": self.max_storage_gb,  # Changed from max_storage_mb
            "settings": self.settings,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }