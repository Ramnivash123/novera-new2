"""
Enhanced user models with authentication support.
"""
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import Column, String, Boolean, DateTime, Index, ForeignKey, inspect as sa_inspect
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import relationship
from app.db.session import Base


class User(Base):
    """
    User model with authentication and profile information.
    """
    __tablename__ = "users"

    # Primary Key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)

    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)

    # Profile Information
    full_name = Column(String(255), nullable=True)
    avatar_url = Column(String(512), nullable=True)

    # Role & Status
    role = Column(String(20), nullable=False, default="user")
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    # Organization (for customization)
    organization = relationship("Organization", back_populates="users")
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # JSONB Columns
    preferences = Column(JSONB, nullable=False, default=dict)
    user_metadata = Column("metadata", JSONB, nullable=False, default=dict)

    __table_args__ = (
        Index('idx_email_active', 'email', 'is_active'),
        Index('idx_username_active', 'username', 'is_active'),
    )

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"

    def is_admin(self):
        return self.role in ["admin", "org_admin", "super_admin"]
    
    def is_super_admin(self):
        return self.role == "super_admin"

    def is_org_admin(self):
        return self.role in ["admin", "org_admin"]

    def to_dict(self, include_sensitive: bool = False, include_organization: bool = True):
        """
        Convert user to dictionary.
        
        IMPORTANT: This method is defensive against lazy loading in async contexts.
        It checks if the organization relationship is loaded before accessing it.
        """
        data = {
            "id": str(self.id),
            "email": self.email,
            "username": self.username,
            "full_name": self.full_name,
            "avatar_url": self.avatar_url,
            "role": self.role,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "preferences": self.preferences or {},
            "metadata": self.user_metadata or {},
            "organization_id": str(self.organization_id) if self.organization_id else None,
        }
        
        # Include organization details if requested and available
        # FIX: Check if organization is loaded before accessing to avoid lazy load
        if include_organization and self.organization_id:
            # Use SQLAlchemy's inspect to check if relationship is loaded
            insp = sa_inspect(self)
            
            # Only access self.organization if it's already loaded
            if 'organization' not in insp.unloaded:
                # Organization is loaded, safe to access
                if self.organization:
                    data["organization"] = {
                        "id": str(self.organization.id),
                        "name": self.organization.name,
                        "display_name": self.organization.display_name
                    }
            else:
                # Organization not loaded - don't trigger lazy load in async context
                # Just set to None, the organization_id is already in the data
                data["organization"] = None
        
        return data

class RefreshToken(Base):
    """
    Refresh token model for JWT authentication.
    """
    __tablename__ = "refresh_tokens"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    token = Column(String(512), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    revoked = Column(Boolean, default=False, nullable=False)

    user_agent = Column(String(512), nullable=True)
    ip_address = Column(String(45), nullable=True)

    __table_args__ = (
        Index('idx_token_active', 'token', 'revoked'),
        Index('idx_user_active', 'user_id', 'revoked'),
    )

    def __repr__(self):
        return f"<RefreshToken(id={self.id}, user_id={self.user_id}, revoked={self.revoked})>"
    
class PasswordResetToken(Base):
    """
    Password reset token model for secure password recovery.
    """
    __tablename__ = "password_reset_tokens"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    token = Column(String(512), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    used = Column(Boolean, default=False, nullable=False)
    ip_address = Column(String(45), nullable=True)

    __table_args__ = (
        Index('idx_reset_token_active', 'token', 'used'),
        Index('idx_reset_user_active', 'user_id', 'used'),
    )

    def __repr__(self):
        return f"<PasswordResetToken(id={self.id}, user_id={self.user_id}, used={self.used})>"

class EmailVerificationToken(Base):
    """
    Email verification token model for account activation.
    """
    __tablename__ = "email_verification_tokens"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    token = Column(String(512), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    used = Column(Boolean, default=False, nullable=False)
    ip_address = Column(String(45), nullable=True)

    __table_args__ = (
        Index('idx_verification_token_active', 'token', 'used'),
        Index('idx_verification_user_active', 'user_id', 'used'),
    )

    def __repr__(self):
        return f"<EmailVerificationToken(id={self.id}, user_id={self.user_id}, used={self.used})>"

__all__ = ['User', 'RefreshToken', 'PasswordResetToken', 'EmailVerificationToken']
