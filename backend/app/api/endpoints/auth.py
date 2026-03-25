"""
Authentication API endpoints.
Handles registration, login, logout, token refresh, and password management.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr, Field

from app.db.session import get_db
from app.services.auth.auth_service import auth_service
from app.api.dependencies.auth import get_current_user, get_current_active_user
from app.models.user import User


router = APIRouter()


# Request Models
class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class UpdateProfileRequest(BaseModel):
    """Profile update request."""
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: Optional[dict] = None
    metadata: Optional[dict] = None


# Response Models
class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str
    expires_in: int
    user: dict


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: Optional[str]
    avatar_url: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    created_at: str
    last_login: Optional[str]
    preferences: Optional[dict] = {}   
    metadata: Optional[dict] = {}     

class ForgotPasswordRequest(BaseModel):
    """Forgot password request."""
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Reset password request."""
    token: str
    new_password: str = Field(..., min_length=8)


class VerifyResetTokenRequest(BaseModel):
    """Verify reset token request."""
    token: str

class ResendVerificationRequest(BaseModel):
    """Resend verification email request."""
    pass


class VerifyEmailRequest(BaseModel):
    """Verify email request."""
    token: str

@router.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account.
    
    User will be assigned to the default organization.
    Only super admins can assign users to specific organizations via admin endpoints.
    
    Requirements:
    - Unique email and username
    - Password must be at least 8 characters
    - Password must contain uppercase, lowercase, digit, and special character
    
    Returns access and refresh tokens upon successful registration.
    Sends verification email to user.
    """
    # Get IP address
    ip_address = http_request.client.host if http_request.client else None
    
    # Register user (will be assigned to default organization)
    success, user, error = await auth_service.register_user(
        email=request.email,
        username=request.username,
        password=request.password,
        full_name=request.full_name,
        ip_address=ip_address,
        db=db,
        organization_id=None  # Will default to 'default' organization
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    # Create tokens for newly registered user
    tokens = await auth_service.create_tokens(user, db)
    
    return tokens

@router.post("/auth/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Login with email and password.
    
    Returns access and refresh tokens upon successful authentication.
    """
    success, user, error = await auth_service.authenticate_user(
        email=request.email,
        password=request.password,
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error,
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get client info
    user_agent = http_request.headers.get("user-agent")
    ip_address = http_request.client.host if http_request.client else None
    
    # Create tokens
    tokens = await auth_service.create_tokens(
        user,
        db,
        user_agent=user_agent,
        ip_address=ip_address
    )
    
    return tokens


@router.post("/auth/refresh", response_model=dict)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    
    Returns a new access token.
    """
    success, tokens, error = await auth_service.refresh_access_token(
        refresh_token_str=request.refresh_token,
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error,
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return tokens


@router.post("/auth/logout")
async def logout(
    request: RefreshTokenRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout user by revoking refresh token.
    """
    await auth_service.revoke_refresh_token(request.refresh_token, db)
    
    return {"message": "Logged out successfully"}


@router.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current authenticated user information including organization details.
    """
    return current_user.to_dict(include_organization=True)


@router.put("/auth/profile", response_model=UserResponse)
async def update_profile(
    request: UpdateProfileRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user profile information.
    """
    success, user, error = await auth_service.update_user_profile(
        user_id=current_user.id,
        full_name=request.full_name,
        avatar_url=request.avatar_url,
        preferences=request.preferences,
        metadata=request.metadata,
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return user.to_dict()


@router.post("/auth/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Change user password.
    
    Requires current password for verification.
    """
    success, error = await auth_service.change_password(
        user_id=current_user.id,
        current_password=request.current_password,
        new_password=request.new_password,
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return {"message": "Password changed successfully"}


@router.get("/auth/test")
async def test_auth():
    """
    Test endpoint to verify authentication API is working.
    """
    return {
        "status": "operational",
        "message": "Authentication API is ready",
        "endpoints": [
            "POST /auth/register",
            "POST /auth/login",
            "POST /auth/refresh",
            "POST /auth/logout",
            "GET /auth/me",
            "PUT /auth/profile",
            "POST /auth/change-password",
            "POST /auth/forgot-password",
            "POST /auth/verify-reset-token",
            "POST /auth/reset-password",
            "POST /auth/verify-email",
            "POST /auth/resend-verification"
        ]
    }

@router.post("/auth/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Request password reset.
    
    Sends an email with password reset link if email exists.
    Always returns success to prevent email enumeration.
    """
    ip_address = http_request.client.host if http_request.client else None
    
    success, error = await auth_service.request_password_reset(
        email=request.email,
        ip_address=ip_address,
        db=db
    )
    
    if not success and error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return {
        "message": "If the email exists, a password reset link has been sent",
        "email": request.email
    }


@router.post("/auth/verify-reset-token")
async def verify_reset_token(
    request: VerifyResetTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Verify if a password reset token is valid.
    
    Used to validate token before showing reset password form.
    """
    is_valid, user_id, error = await auth_service.verify_reset_token(
        token=request.token,
        db=db
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return {
        "valid": True,
        "message": "Token is valid"
    }


@router.post("/auth/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Reset password using reset token.
    
    Validates token and sets new password.
    """
    success, error = await auth_service.reset_password(
        token=request.token,
        new_password=request.new_password,
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return {"message": "Password reset successfully"}

@router.post("/auth/verify-email")
async def verify_email(
    request: VerifyEmailRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Verify user's email address using verification token.
    
    Token is sent via email after registration.
    """
    success, error = await auth_service.verify_email(
        token=request.token,
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return {
        "message": "Email verified successfully",
        "verified": True
    }


@router.post("/auth/resend-verification")
async def resend_verification(
    http_request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Resend verification email to current user.
    
    Can only be used if email is not yet verified.
    Rate limited to one request per 5 minutes.
    """
    if current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already verified"
        )
    
    ip_address = http_request.client.host if http_request.client else None
    
    success, error = await auth_service.resend_verification_email(
        user_id=current_user.id,
        ip_address=ip_address,
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return {
        "message": "Verification email sent",
        "email": current_user.email
    }

__all__ = ['router']