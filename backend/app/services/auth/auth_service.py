"""
Authentication service for user registration, login, and token management.
"""
from datetime import datetime, timedelta
from typing import Optional, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
import secrets
from app.models.user import User, RefreshToken, PasswordResetToken, EmailVerificationToken
from app.services.email.email_service import email_service
from sqlalchemy.orm import selectinload 

from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    validate_password_strength,
    validate_email
)
from app.core.config import settings


class AuthService:
    """Service for handling authentication operations."""

    async def register_user(
        self,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str],
        ip_address: Optional[str],
        db: AsyncSession,
        organization_id: Optional[UUID] = None
    ) -> Tuple[bool, Optional[User], Optional[str]]:
        """
        Register a new user.
        
        Args:
            email: User email
            username: Username
            password: Plain password
            full_name: Full name (optional)
            ip_address: IP address of requester
            db: Database session
            organization_id: Organization to assign user to (optional, defaults to 'default')
            
        Returns:
            Tuple of (success, user, error_message)
        """
        from app.models.organization import Organization
        
        # Validate email
        if not validate_email(email):
            return False, None, "Invalid email format"
        
        # Validate password strength
        is_strong, error = validate_password_strength(password)
        if not is_strong:
            return False, None, error
        
        # Check if email already exists
        result = await db.execute(
            select(User).where(User.email == email)
        )
        if result.scalar_one_or_none():
            return False, None, "Email already registered"
        
        # Check if username already exists
        result = await db.execute(
            select(User).where(User.username == username)
        )
        if result.scalar_one_or_none():
            return False, None, "Username already taken"
        
        # ORGANIZATION ASSIGNMENT: Get or create default organization
        if organization_id is None:
            # Get default organization
            org_result = await db.execute(
                select(Organization).where(Organization.name == "default")
            )
            default_org = org_result.scalar_one_or_none()
            
            if not default_org:
                return False, None, "Default organization not found. Please contact administrator."
            
            organization_id = default_org.id
        else:
            # Verify organization exists
            org_result = await db.execute(
                select(Organization).where(Organization.id == organization_id)
            )
            org = org_result.scalar_one_or_none()
            
            if not org:
                return False, None, "Invalid organization"
            
            if not org.is_active:
                return False, None, "Organization is not active"
        
        # Create user
        try:
            user = User(
                email=email.lower(),
                username=username,
                hashed_password=get_password_hash(password),
                full_name=full_name,
                role="user",
                is_active=True,
                is_verified=False,
                organization_id=organization_id,  # ORGANIZATION ASSIGNMENT
                preferences={
                    "theme": "light",
                    "language": "en",
                    "notifications": True
                }
            )
            
            db.add(user)
            await db.commit()
            await db.refresh(user, ['organization'])

            await self.send_verification_email(
                user_id=user.id,
                email=user.email,
                username=user.username,
                ip_address=ip_address,
                db=db
            )
            
            logger.info(f"New user registered: {user.email} (org: {organization_id})")
            
            return True, user, None
            
        except Exception as e:
            await db.rollback()
            logger.error(f"User registration failed: {str(e)}")
            return False, None, "Registration failed. Please try again."
    
    async def authenticate_user(
        self,
        email: str,
        password: str,
        db: AsyncSession
    ) -> Tuple[bool, Optional[User], Optional[str]]:
        """
        Authenticate user with email and password.
        
        Args:
            email: User email
            password: Plain password
            db: Database session
            
        Returns:
            Tuple of (success, user, error_message)
        """
        # Get user by email WITH organization eagerly loaded
        result = await db.execute(
            select(User)
            .options(selectinload(User.organization))
            .where(User.email == email.lower())
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return False, None, "Invalid email or password"
        
        # Verify password
        if not verify_password(password, user.hashed_password):
            return False, None, "Invalid email or password"
        
        # Check if account is active
        if not user.is_active:
            return False, None, "Account is deactivated"
        
        # Update last login
        user.last_login = datetime.utcnow()
        await db.commit()
        
        logger.info(f"User authenticated: {user.email}")
        
        return True, user, None
    
    async def create_tokens(
        self,
        user: User,
        db: AsyncSession,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> dict:
        """
        Create access and refresh tokens for user.
        Include role AND organization_id in JWT payload for RBAC and multi-tenancy.
        
        Args:
            user: User object
            db: Database session
            user_agent: User agent string
            ip_address: IP address
            
        Returns:
            Dictionary with access_token, refresh_token, and metadata
        """
        # Create access token with role AND organization_id
        access_token = create_access_token(
            data={
                "sub": str(user.id),
                "email": user.email,
                "role": user.role,
                "organization_id": str(user.organization_id) if user.organization_id else None
            }
        )
        
        # Create refresh token
        refresh_token_str = create_refresh_token(
            data={"sub": str(user.id)}
        )
        
        # Store refresh token in database
        refresh_token = RefreshToken(
            user_id=user.id,
            token=refresh_token_str,
            expires_at=datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days),
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        db.add(refresh_token)
        await db.commit()
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token_str,
            "token_type": "bearer",
            "expires_in": settings.access_token_expire_minutes * 60,
            "user": user.to_dict()
        }
    
    async def refresh_access_token(
        self,
        refresh_token_str: str,
        db: AsyncSession
    ) -> Tuple[bool, Optional[dict], Optional[str]]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token_str: Refresh token string
            db: Database session
            
        Returns:
            Tuple of (success, tokens_dict, error_message)
        """
        # Decode refresh token
        payload = decode_token(refresh_token_str)
        
        if not payload or payload.get("type") != "refresh":
            return False, None, "Invalid refresh token"
        
        # Get user_id from token
        user_id = payload.get("sub")
        if not user_id:
            return False, None, "Invalid refresh token"
        
        # Check if refresh token exists in database
        result = await db.execute(
            select(RefreshToken).where(
                RefreshToken.token == refresh_token_str,
                RefreshToken.revoked == False
            )
        )
        db_token = result.scalar_one_or_none()
        
        if not db_token:
            return False, None, "Refresh token not found or revoked"
        
        # Check if token is expired
        if db_token.expires_at < datetime.utcnow():
            return False, None, "Refresh token expired"
        
        # Get user
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            return False, None, "User not found or inactive"
        
        # Create new access token with role AND organization_id
        access_token = create_access_token(
            data={
                "sub": str(user.id),
                "email": user.email,
                "role": user.role,
                "organization_id": str(user.organization_id) if user.organization_id else None
            }
        )
        
        return True, {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.access_token_expire_minutes * 60
        }, None
    
    async def revoke_refresh_token(
        self,
        refresh_token_str: str,
        db: AsyncSession
    ) -> bool:
        """
        Revoke a refresh token (logout).
        
        Args:
            refresh_token_str: Refresh token to revoke
            db: Database session
            
        Returns:
            True if revoked successfully
        """
        result = await db.execute(
            select(RefreshToken).where(RefreshToken.token == refresh_token_str)
        )
        token = result.scalar_one_or_none()
        
        if token:
            token.revoked = True
            await db.commit()
            logger.info(f"Refresh token revoked for user {token.user_id}")
            return True
        
        return False
    
    async def update_user_profile(
        self,
        user_id: UUID,
        full_name: Optional[str],
        avatar_url: Optional[str],
        preferences: Optional[dict],
        metadata: Optional[dict],
        db: AsyncSession
    ) -> Tuple[bool, Optional[User], Optional[str]]:
        """
        Update user profile information.
        
        Args:
            user_id: User ID
            full_name: New full name
            avatar_url: New avatar URL
            preferences: User preferences
            metadata: User metadata
            db: Database session
            
        Returns:
            Tuple of (success, user, error_message)
        """
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return False, None, "User not found"
        
        try:
            if full_name is not None:
                user.full_name = full_name
            
            if avatar_url is not None:
                user.avatar_url = avatar_url
            
            if preferences is not None:
                user.preferences.update(preferences)
            
            if metadata is not None:
                user.user_metadata.update(metadata)
            
            user.updated_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(user)
            
            logger.info(f"Profile updated for user {user.email}")
            
            return True, user, None
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Profile update failed: {str(e)}")
            return False, None, "Profile update failed"
    
    async def change_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password: str,
        db: AsyncSession
    ) -> Tuple[bool, Optional[str]]:
        """
        Change user password.
        
        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password
            db: Database session
            
        Returns:
            Tuple of (success, error_message)
        """
        # Get user
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return False, "User not found"
        
        # Verify current password
        if not verify_password(current_password, user.hashed_password):
            return False, "Current password is incorrect"
        
        # Validate new password
        is_strong, error = validate_password_strength(new_password)
        if not is_strong:
            return False, error
        
        try:
            user.hashed_password = get_password_hash(new_password)
            user.updated_at = datetime.utcnow()
            
            await db.commit()
            
            logger.info(f"Password changed for user {user.email}")
            
            return True, None
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Password change failed: {str(e)}")
            return False, "Password change failed"

    async def request_password_reset(
        self,
        email: str,
        ip_address: Optional[str],
        db: AsyncSession
    ) -> Tuple[bool, Optional[str]]:
        """
        Request password reset - generates token and sends email.

        Args:
            email: User's email address
            ip_address: IP address of requester
            db: Database session

        Returns:
            Tuple of (success, error_message)
        """
        result = await db.execute(
            select(User).where(User.email == email.lower())
        )
        user = result.scalar_one_or_none()

        if not user:
            logger.warning(f"Password reset requested for non-existent email: {email}")
            return True, None

        if not user.is_active:
            return False, "Account is deactivated"

        try:
            reset_token = secrets.token_urlsafe(32)
            
            expires_at = datetime.utcnow() + timedelta(
                minutes=settings.password_reset_token_expire_minutes
            )

            token_entry = PasswordResetToken(
                user_id=user.id,
                token=reset_token,
                expires_at=expires_at,
                ip_address=ip_address
            )

            db.add(token_entry)
            await db.commit()

            email_sent = email_service.send_password_reset_email(
                to_email=user.email,
                reset_token=reset_token,
                username=user.username
            )

            if not email_sent:
                logger.error(f"Failed to send password reset email to {user.email}")
                return False, "Failed to send reset email. Please try again."

            logger.info(f"Password reset email sent to {user.email}")
            return True, None

        except Exception as e:
            await db.rollback()
            logger.error(f"Password reset request failed: {str(e)}")
            return False, "Failed to process password reset request"

    async def verify_reset_token(
        self,
        token: str,
        db: AsyncSession
    ) -> Tuple[bool, Optional[UUID], Optional[str]]:
        """
        Verify password reset token.

        Args:
            token: Reset token string
            db: Database session

        Returns:
            Tuple of (is_valid, user_id, error_message)
        """
        result = await db.execute(
            select(PasswordResetToken).where(
                PasswordResetToken.token == token,
                PasswordResetToken.used == False
            )
        )
        reset_token = result.scalar_one_or_none()

        if not reset_token:
            return False, None, "Invalid or expired reset token"

        if reset_token.expires_at < datetime.utcnow():
            return False, None, "Reset token has expired"

        result = await db.execute(
            select(User).where(User.id == reset_token.user_id)
        )
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            return False, None, "User not found or inactive"

        return True, user.id, None

    async def reset_password(
        self,
        token: str,
        new_password: str,
        db: AsyncSession
    ) -> Tuple[bool, Optional[str]]:
        """
        Reset password using reset token.

        Args:
            token: Reset token string
            new_password: New password
            db: Database session

        Returns:
            Tuple of (success, error_message)
        """
        is_valid, user_id, error = await self.verify_reset_token(token, db)

        if not is_valid:
            return False, error

        is_strong, password_error = validate_password_strength(new_password)
        if not is_strong:
            return False, password_error

        try:
            result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()

            if not user:
                return False, "User not found"

            user.hashed_password = get_password_hash(new_password)
            user.updated_at = datetime.utcnow()

            result = await db.execute(
                select(PasswordResetToken).where(
                    PasswordResetToken.token == token
                )
            )
            reset_token = result.scalar_one_or_none()

            if reset_token:
                reset_token.used = True

            await db.commit()

            logger.info(f"Password reset successful for user {user.email}")
            return True, None

        except Exception as e:
            await db.rollback()
            logger.error(f"Password reset failed: {str(e)}")
            return False, "Password reset failed. Please try again."

    async def send_verification_email(
        self,
        user_id: UUID,
        email: str,
        username: str,
        ip_address: Optional[str],
        db: AsyncSession
    ) -> Tuple[bool, Optional[str]]:
        """
        Send email verification link to user.

        Args:
            user_id: User ID
            email: User's email address
            username: User's username
            ip_address: IP address of requester
            db: Database session

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Generate verification token
            verification_token = secrets.token_urlsafe(32)
            
            # Token expires in 24 hours
            expires_at = datetime.utcnow() + timedelta(hours=24)

            # Store token in database
            token_entry = EmailVerificationToken(
                user_id=user_id,
                token=verification_token,
                expires_at=expires_at,
                ip_address=ip_address
            )

            db.add(token_entry)
            await db.commit()

            # Send verification email
            email_sent = email_service.send_verification_email(
                to_email=email,
                verification_token=verification_token,
                username=username
            )

            if not email_sent:
                logger.error(f"Failed to send verification email to {email}")
                return False, "Failed to send verification email. Please try again."

            logger.info(f"Verification email sent to {email}")
            return True, None

        except Exception as e:
            await db.rollback()
            logger.error(f"Send verification email failed: {str(e)}")
            return False, "Failed to send verification email"

    async def verify_email(
        self,
        token: str,
        db: AsyncSession
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify user's email using verification token.

        Args:
            token: Verification token
            db: Database session

        Returns:
            Tuple of (success, error_message)
        """
        # Get verification token from database
        result = await db.execute(
            select(EmailVerificationToken).where(
                EmailVerificationToken.token == token,
                EmailVerificationToken.used == False
            )
        )
        verification_token = result.scalar_one_or_none()

        if not verification_token:
            return False, "Invalid or expired verification token"

        # Check if token is expired
        if verification_token.expires_at < datetime.utcnow():
            return False, "Verification link has expired. Please request a new one."

        try:
            # Get user
            result = await db.execute(
                select(User).where(User.id == verification_token.user_id)
            )
            user = result.scalar_one_or_none()

            if not user:
                return False, "User not found"

            # Check if already verified
            if user.is_verified:
                return False, "Email already verified"

            # Mark user as verified
            user.is_verified = True
            user.updated_at = datetime.utcnow()

            # Mark token as used
            verification_token.used = True

            await db.commit()

            logger.info(f"Email verified for user {user.email}")
            return True, None

        except Exception as e:
            await db.rollback()
            logger.error(f"Email verification failed: {str(e)}")
            return False, "Email verification failed. Please try again."

    async def resend_verification_email(
        self,
        user_id: UUID,
        ip_address: Optional[str],
        db: AsyncSession
    ) -> Tuple[bool, Optional[str]]:
        """
        Resend verification email to user.

        Args:
            user_id: User ID
            ip_address: IP address of requester
            db: Database session

        Returns:
            Tuple of (success, error_message)
        """
        # Get user
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            return False, "User not found"

        if user.is_verified:
            return False, "Email already verified"

        # Check if a recent token was sent (rate limiting)
        result = await db.execute(
            select(EmailVerificationToken)
            .where(
                EmailVerificationToken.user_id == user_id,
                EmailVerificationToken.used == False,
                EmailVerificationToken.created_at > datetime.utcnow() - timedelta(minutes=5)
            )
        )
        recent_token = result.scalar_one_or_none()

        if recent_token:
            return False, "Verification email was recently sent. Please wait 5 minutes before requesting again."

        # Send new verification email
        return await self.send_verification_email(
            user_id=user.id,
            email=user.email,
            username=user.username,
            ip_address=ip_address,
            db=db
        )

# Global instance
auth_service = AuthService()

__all__ = ['AuthService', 'auth_service']