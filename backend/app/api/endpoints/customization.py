"""
Enhanced Customization API endpoints with comprehensive theming support.
"""
from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field, validator
from datetime import datetime
from loguru import logger
import os
import shutil
from pathlib import Path
import re

from app.db.session import get_db
from app.models.user import User
from app.models.customization import OrganizationCustomization
from app.api.dependencies.auth import get_current_admin_user, get_optional_user
from app.models.organization import Organization
from sqlalchemy import and_
from app.core.config import settings

router = APIRouter()

# Pydantic Models
class ColorValidator:
    """Validator for hex color codes (supports alpha channel)."""
    @classmethod
    def validate_hex_color(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        # Accept both #RRGGBB and #RRGGBBAA formats
        if not re.match(r'^#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$', value):
            raise ValueError(f'Invalid hex color: {value}. Must be in format #RRGGBB or #RRGGBBAA')
        return value.lower()

class CustomizationUpdate(BaseModel):
    """Comprehensive customization update model."""
     
    # Dark Mode
    dark_mode_enabled: Optional[bool] = None
    dark_mode_colors: Optional[dict] = None
    
    # Branding
    app_name: Optional[str] = Field(None, max_length=255)
    app_tagline: Optional[str] = Field(None, max_length=512)
    
    # Primary Colors
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    accent_color: Optional[str] = None
    
    # Semantic Colors
    success_color: Optional[str] = None
    warning_color: Optional[str] = None
    error_color: Optional[str] = None
    info_color: Optional[str] = None
    
    # Background Colors
    background_color: Optional[str] = None
    background_secondary: Optional[str] = None
    background_tertiary: Optional[str] = None
    sidebar_color: Optional[str] = None
    
    # Text Colors
    text_primary_color: Optional[str] = None
    text_secondary_color: Optional[str] = None
    
    # Border and Shadow
    border_color: Optional[str] = None
    shadow_color: Optional[str] = None
    
    # Button Styling
    button_primary_color: Optional[str] = None
    button_text_color: Optional[str] = None
    button_secondary_color: Optional[str] = None
    button_secondary_text: Optional[str] = None
    button_border_radius: Optional[str] = None
    
    # Input Styling
    input_border_color: Optional[str] = None
    input_focus_color: Optional[str] = None
    input_border_radius: Optional[str] = None
    
    # Card Styling
    card_background: Optional[str] = None
    card_border_color: Optional[str] = None
    card_border_radius: Optional[str] = None
    card_shadow: Optional[str] = None
    
    # Navigation Styling
    nav_background: Optional[str] = None
    nav_text_color: Optional[str] = None
    nav_active_color: Optional[str] = None
    nav_hover_color: Optional[str] = None
    
    # Typography
    font_family: Optional[str] = Field(None, max_length=100)
    font_size_base: Optional[str] = Field(None, max_length=10)
    font_size_heading: Optional[str] = Field(None, max_length=10)
    font_weight_normal: Optional[str] = Field(None, max_length=10)
    font_weight_medium: Optional[str] = Field(None, max_length=10)
    font_weight_bold: Optional[str] = Field(None, max_length=10)
    line_height_base: Optional[str] = Field(None, max_length=10)
    line_height_heading: Optional[str] = Field(None, max_length=10)
    letter_spacing: Optional[str] = Field(None, max_length=10)
    
    # Layout
    border_radius: Optional[str] = Field(None, max_length=10)
    spacing_unit: Optional[str] = Field(None, max_length=10)
    spacing_xs: Optional[str] = Field(None, max_length=10)
    spacing_sm: Optional[str] = Field(None, max_length=10)
    spacing_md: Optional[str] = Field(None, max_length=10)
    spacing_lg: Optional[str] = Field(None, max_length=10)
    spacing_xl: Optional[str] = Field(None, max_length=10)
    
    # Animation
    animation_speed: Optional[str] = Field(None, max_length=10)
    enable_animations: Optional[bool] = None
    
    # Dark Mode (duplicate declaration removed)
    # dark_mode_enabled: Optional[bool] = None  # Already declared above
    # dark_mode_colors: Optional[dict] = None   # Already declared above
    
    # Advanced
    custom_css: Optional[str] = None
    custom_settings: Optional[dict] = None
    
    # Metadata
    theme_name: Optional[str] = Field(None, max_length=100)
    theme_description: Optional[str] = Field(None, max_length=500)
    
    @validator('primary_color', 'secondary_color', 'accent_color', 'success_color', 
               'warning_color', 'error_color', 'info_color', 'background_color',
               'background_secondary', 'background_tertiary', 'sidebar_color',
               'text_primary_color', 'text_secondary_color', 'border_color', 'shadow_color',
               'button_primary_color', 'button_text_color', 'button_secondary_color',
               'button_secondary_text', 'input_border_color', 'input_focus_color',
               'card_background', 'card_border_color', 'nav_background',
               'nav_text_color', 'nav_active_color', 'nav_hover_color')
    def validate_color(cls, v):
        return ColorValidator.validate_hex_color(v)


class ThemePreset(BaseModel):
    """Theme preset definition."""
    name: str
    description: str
    colors: dict
    components: dict
    preview_image: Optional[str] = None


# File upload configuration
UPLOAD_DIR = Path(settings.upload_dir) / "branding"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg', '.ico'}
MAX_IMAGE_SIZE_MB = 5


def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file."""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    max_size_bytes = MAX_IMAGE_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {MAX_IMAGE_SIZE_MB}MB"
        )


async def save_uploaded_file(file: UploadFile, organization_name: str, file_type: str) -> str:
    """Save uploaded file and return the URL path."""
    validate_image_file(file)
    
    file_ext = Path(file.filename).suffix.lower()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{organization_name}_{file_type}_{timestamp}{file_ext}"
    file_path = UPLOAD_DIR / filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save file"
        )
    
    return f"/uploads/branding/{filename}"


async def get_or_create_customization(
    db: AsyncSession,
    organization_id: UUID
) -> OrganizationCustomization:
    """
    Get existing customization or create default one for an organization.
    
    Args:
        db: Database session
        organization_id: Organization UUID
        
    Returns:
        OrganizationCustomization object
    """
    # Verify organization exists
    org_result = await db.execute(
        select(Organization).where(Organization.id == organization_id)
    )
    organization = org_result.scalar_one_or_none()
    
    if not organization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )
    
    # Get or create customization
    result = await db.execute(
        select(OrganizationCustomization).where(
            OrganizationCustomization.organization_id == organization_id
        )
    )
    customization = result.scalar_one_or_none()
    
    if not customization:
        customization = OrganizationCustomization(
            organization_id=organization_id,
            primary_color="#0ea5e9",
            secondary_color="#d946ef",
            accent_color="#8b5cf6"
        )
        db.add(customization)
        await db.commit()
        await db.refresh(customization)
    
    return customization


# Theme Presets
THEME_PRESETS = [
    {
        "name": "Ocean Blue",
        "description": "Professional blue theme with clean aesthetics",
        "colors": {
            "primary": "#0284c7",
            "secondary": "#0ea5e9",
            "accent": "#06b6d4",
            "success": "#10b981",
            "warning": "#f59e0b",
            "error": "#ef4444",
            "info": "#3b82f6",
            "background": "#ffffff",
            "background_secondary": "#f0f9ff",
            "background_tertiary": "#e0f2fe",
            "sidebar": "#ffffff",
            "text_primary": "#0c4a6e",
            "text_secondary": "#475569",
            "border": "#cbd5e1",
            "shadow": "#00000015"
        },
        "components": {
            "button_border_radius": "8px",
            "input_border_radius": "6px",
            "card_border_radius": "12px"
        }
    },
    {
        "name": "Purple Tech",
        "description": "Modern purple gradient for tech companies",
        "colors": {
            "primary": "#7c3aed",
            "secondary": "#a855f7",
            "accent": "#c026d3",
            "success": "#10b981",
            "warning": "#f59e0b",
            "error": "#ef4444",
            "info": "#8b5cf6",
            "background": "#ffffff",
            "background_secondary": "#faf5ff",
            "background_tertiary": "#f3e8ff",
            "sidebar": "#fefefe",
            "text_primary": "#581c87",
            "text_secondary": "#6b7280",
            "border": "#e9d5ff",
            "shadow": "#00000012"
        },
        "components": {
            "button_border_radius": "10px",
            "input_border_radius": "8px",
            "card_border_radius": "14px"
        }
    },
    {
        "name": "Emerald Green",
        "description": "Fresh green theme for eco-friendly brands",
        "colors": {
            "primary": "#059669",
            "secondary": "#10b981",
            "accent": "#34d399",
            "success": "#22c55e",
            "warning": "#f59e0b",
            "error": "#ef4444",
            "info": "#06b6d4",
            "background": "#ffffff",
            "background_secondary": "#f0fdf4",
            "background_tertiary": "#dcfce7",
            "sidebar": "#ffffff",
            "text_primary": "#064e3b",
            "text_secondary": "#6b7280",
            "border": "#d1d5db",
            "shadow": "#00000010"
        },
        "components": {
            "button_border_radius": "8px",
            "input_border_radius": "8px",
            "card_border_radius": "10px"
        }
    },
    {
        "name": "Sunset Orange",
        "description": "Warm orange theme with energy and creativity",
        "colors": {
            "primary": "#ea580c",
            "secondary": "#f97316",
            "accent": "#fb923c",
            "success": "#10b981",
            "warning": "#fbbf24",
            "error": "#ef4444",
            "info": "#3b82f6",
            "background": "#ffffff",
            "background_secondary": "#fff7ed",
            "background_tertiary": "#ffedd5",
            "sidebar": "#ffffff",
            "text_primary": "#7c2d12",
            "text_secondary": "#78716c",
            "border": "#e7e5e4",
            "shadow": "#00000012"
        },
        "components": {
            "button_border_radius": "12px",
            "input_border_radius": "8px",
            "card_border_radius": "16px"
        }
    },
    {
        "name": "Corporate Gray",
        "description": "Sophisticated gray theme for corporate environments",
        "colors": {
            "primary": "#1f2937",
            "secondary": "#4b5563",
            "accent": "#6b7280",
            "success": "#10b981",
            "warning": "#f59e0b",
            "error": "#ef4444",
            "info": "#3b82f6",
            "background": "#ffffff",
            "background_secondary": "#f9fafb",
            "background_tertiary": "#f3f4f6",
            "sidebar": "#ffffff",
            "text_primary": "#111827",
            "text_secondary": "#6b7280",
            "border": "#e5e7eb",
            "shadow": "#00000008"
        },
        "components": {
            "button_border_radius": "6px",
            "input_border_radius": "6px",
            "card_border_radius": "8px"
        }
    },
    {
        "name": "Minimal Dark",
        "description": "Sleek dark theme with subtle accents",
        "colors": {
            "primary": "#3b82f6",
            "secondary": "#60a5fa",
            "accent": "#93c5fd",
            "success": "#10b981",
            "warning": "#f59e0b",
            "error": "#ef4444",
            "info": "#06b6d4",
            "background": "#111827",
            "background_secondary": "#1f2937",
            "background_tertiary": "#374151",
            "sidebar": "#1f2937",
            "text_primary": "#f9fafb",
            "text_secondary": "#d1d5db",
            "border": "#4b5563",
            "shadow": "#00000025"
        },
        "components": {
            "button_border_radius": "8px",
            "input_border_radius": "8px",
            "card_border_radius": "10px"
        }
    }
]


# ADMIN ENDPOINTS

@router.get("/admin/customization", response_model=dict)
async def get_admin_customization(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Get customization settings for the admin's organization.
    Super admins get their own organization's customization.
    Org admins get their organization's customization.
    """
    if not admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin must belong to an organization"
        )
    
    customization = await get_or_create_customization(db, admin.organization_id)
    return customization.to_dict()


@router.put("/admin/customization", response_model=dict)
async def update_customization(
    request: CustomizationUpdate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Update customization settings for the admin's organization.
    Org admins can only update their own organization's customization.
    """
    if not admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin must belong to an organization"
        )
    
    customization = await get_or_create_customization(db, admin.organization_id)
    
    update_data = request.dict(exclude_unset=True)
    
    logger.info(f"Admin {admin.email} updating customization for org {admin.organization_id}")
    
    # Handle dark_mode_colors separately since it's a JSONB field
    dark_mode_colors = update_data.pop('dark_mode_colors', None)
    if dark_mode_colors is not None:
        customization.dark_mode_colors = dark_mode_colors
        logger.info(f"✅ Updated dark_mode_colors: {dark_mode_colors}")
    
    # Update other fields
    for field, value in update_data.items():
        if hasattr(customization, field) and value is not None:
            setattr(customization, field, value)
            if field == 'dark_mode_enabled':
                logger.info(f"✅ Dark mode enabled set to: {value}")
    
    customization.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(customization)
    
    logger.info(f"✅ Customization updated for organization {admin.organization_id}")
    
    return customization.to_dict()

@router.post("/admin/customization/logo")
async def upload_logo(
    file: UploadFile = File(...),
    logo_type: str = "light",
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Upload organization logo for the admin's organization.
    """
    if not admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin must belong to an organization"
        )
    
    if logo_type not in ['light', 'dark', 'favicon']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid logo_type. Must be 'light', 'dark', or 'favicon'"
        )
    
    # Get organization name for file naming
    org_result = await db.execute(
        select(Organization).where(Organization.id == admin.organization_id)
    )
    organization = org_result.scalar_one()
    
    file_url = await save_uploaded_file(file, organization.name, logo_type)
    
    customization = await get_or_create_customization(db, admin.organization_id)
    
    if logo_type == "light":
        customization.logo_url = file_url
    elif logo_type == "dark":
        customization.logo_dark_url = file_url
    elif logo_type == "favicon":
        customization.favicon_url = file_url
    
    customization.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(customization)
    
    logger.info(f"Admin {admin.email} uploaded {logo_type} logo for organization {organization.name}")
    
    return {
        "message": f"{logo_type.capitalize()} logo uploaded successfully",
        "url": file_url,
        "customization": customization.to_dict()
    }


@router.delete("/admin/customization/logo")
async def delete_logo(
    logo_type: str = "light",
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Delete organization logo for the admin's organization.
    """
    if not admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin must belong to an organization"
        )
    
    if logo_type not in ['light', 'dark', 'favicon']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid logo_type"
        )
    
    customization = await get_or_create_customization(db, admin.organization_id)
    
    logo_url = None
    if logo_type == "light":
        logo_url = customization.logo_url
        customization.logo_url = None
    elif logo_type == "dark":
        logo_url = customization.logo_dark_url
        customization.logo_dark_url = None
    elif logo_type == "favicon":
        logo_url = customization.favicon_url
        customization.favicon_url = None
    
    if logo_url:
        file_path = UPLOAD_DIR / Path(logo_url).name
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Failed to delete file: {str(e)}")
    
    customization.updated_at = datetime.utcnow()
    await db.commit()
    
    # Get org name for logging
    org_result = await db.execute(
        select(Organization).where(Organization.id == admin.organization_id)
    )
    organization = org_result.scalar_one()
    
    logger.info(f"Admin {admin.email} deleted {logo_type} logo for organization {organization.name}")
    
    return {"message": f"{logo_type.capitalize()} logo deleted successfully"}


@router.post("/admin/customization/reset")
async def reset_customization(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Reset customization to default values for the admin's organization.
    """
    if not admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin must belong to an organization"
        )
    
    customization = await get_or_create_customization(db, admin.organization_id)
    
    # Delete logo files if they exist
    logo_files = [
        customization.logo_url,
        customization.logo_dark_url,
        customization.favicon_url
    ]
    
    for logo_url in logo_files:
        if logo_url:
            file_path = UPLOAD_DIR / Path(logo_url).name
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted logo file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete file {file_path}: {str(e)}")
    
    # Reset ALL fields to defaults
    customization.logo_url = None
    customization.logo_dark_url = None
    customization.favicon_url = None
    customization.app_name = None
    customization.app_tagline = None
    
    customization.primary_color = '#0ea5e9'
    customization.secondary_color = '#d946ef'
    customization.accent_color = '#8b5cf6'
    
    customization.success_color = '#10b981'
    customization.warning_color = '#f59e0b'
    customization.error_color = '#ef4444'
    customization.info_color = '#3b82f6'
    
    customization.background_color = '#ffffff'
    customization.background_secondary = '#f9fafb'
    customization.background_tertiary = '#f3f4f6'
    customization.sidebar_color = '#ffffff'
    
    customization.text_primary_color = '#111827'
    customization.text_secondary_color = '#6b7280'
    
    customization.border_color = '#e5e7eb'
    customization.shadow_color = '#00000010'
    
    customization.button_primary_color = None
    customization.button_text_color = '#ffffff'
    customization.button_secondary_color = None
    customization.button_secondary_text = '#374151'
    customization.button_border_radius = '8px'
    
    customization.input_border_color = '#d1d5db'
    customization.input_focus_color = None
    customization.input_border_radius = '8px'
    
    customization.card_background = '#ffffff'
    customization.card_border_color = '#e5e7eb'
    customization.card_border_radius = '12px'
    customization.card_shadow = '0 1px 3px rgba(0,0,0,0.1)'
    
    customization.nav_background = None
    customization.nav_text_color = None
    customization.nav_active_color = None
    customization.nav_hover_color = None
    
    customization.font_family = None
    customization.font_size_base = '14px'
    customization.font_size_heading = '24px'
    customization.font_weight_normal = '400'
    customization.font_weight_medium = '500'
    customization.font_weight_bold = '700'
    customization.line_height_base = '1.5'
    customization.line_height_heading = '1.2'
    customization.letter_spacing = '0'
    
    customization.border_radius = '8px'
    customization.spacing_unit = '16px'
    customization.spacing_xs = '4px'
    customization.spacing_sm = '8px'
    customization.spacing_md = '16px'
    customization.spacing_lg = '24px'
    customization.spacing_xl = '32px'
    
    customization.animation_speed = '300ms'
    customization.enable_animations = True
    
    customization.dark_mode_enabled = False
    customization.dark_mode_colors = None
    
    customization.custom_css = None
    customization.custom_settings = {}
    
    customization.theme_name = None
    customization.theme_description = None
    
    customization.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(customization)
    
    # Get org name for logging
    org_result = await db.execute(
        select(Organization).where(Organization.id == admin.organization_id)
    )
    organization = org_result.scalar_one()
    
    logger.info(f"Admin {admin.email} reset customization for organization {organization.name}")
    
    return {
        "message": "Customization reset to defaults successfully",
        "customization": customization.to_dict()
    }


@router.get("/admin/customization/presets", response_model=List[dict])
async def get_theme_presets(
    admin: User = Depends(get_current_admin_user)
):
    """Get available theme presets."""
    return THEME_PRESETS


@router.post("/admin/customization/apply-preset")
async def apply_theme_preset(
    preset_name: str,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Apply a theme preset to the admin's organization.
    """
    if not admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin must belong to an organization"
        )
    
    preset = next((p for p in THEME_PRESETS if p['name'] == preset_name), None)
    
    if not preset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Theme preset '{preset_name}' not found"
        )
    
    customization = await get_or_create_customization(db, admin.organization_id)
    
    # Color mapping from preset keys to database column names
    color_mapping = {
        'primary': 'primary_color',
        'secondary': 'secondary_color',
        'accent': 'accent_color',
        'success': 'success_color',
        'warning': 'warning_color',
        'error': 'error_color',
        'info': 'info_color',
        'background': 'background_color',
        'background_secondary': 'background_secondary',
        'background_tertiary': 'background_tertiary',
        'sidebar': 'sidebar_color',
        'text_primary': 'text_primary_color',
        'text_secondary': 'text_secondary_color',
        'border': 'border_color',
        'shadow': 'shadow_color',
    }

    # Apply preset colors with proper mapping
    for preset_key, color_value in preset['colors'].items():
        db_column = color_mapping.get(preset_key)
        if db_column and hasattr(customization, db_column):
            setattr(customization, db_column, color_value)
            logger.debug(f"Set {db_column} = {color_value}")

    # Apply preset components (border radius, etc.)
    for comp_key, comp_value in preset['components'].items():
        if hasattr(customization, comp_key):
            setattr(customization, comp_key, comp_value)
            logger.debug(f"Set {comp_key} = {comp_value}")
    
    customization.theme_name = preset['name']
    customization.theme_description = preset['description']
    customization.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(customization)
    
    # Get org name for logging
    org_result = await db.execute(
        select(Organization).where(Organization.id == admin.organization_id)
    )
    organization = org_result.scalar_one()
    
    logger.info(f"Admin {admin.email} applied preset '{preset_name}' to organization {organization.name}")
    
    return {
        "message": f"Theme preset '{preset_name}' applied successfully",
        "customization": customization.to_dict()
    }

@router.get("/admin/customization/export")
async def export_customization(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Export customization as JSON for the admin's organization.
    """
    if not admin.organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin must belong to an organization"
        )
    
    customization = await get_or_create_customization(db, admin.organization_id)
    return customization.to_dict()


# PUBLIC ENDPOINT

@router.get("/customization/current", response_model=dict)
async def get_current_customization(
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user)
):
    """
    Get customization settings for the current user's organization.
    If no user is authenticated, returns default organization's customization.
    """
    # Determine organization
    organization_id = None
    
    if user and user.organization_id:
        organization_id = user.organization_id
    else:
        # Get default organization
        org_result = await db.execute(
            select(Organization).where(Organization.name == "default")
        )
        default_org = org_result.scalar_one_or_none()
        
        if default_org:
            organization_id = default_org.id
        else:
            # Fallback: create minimal customization response
            return {
                "id": "default",
                "organization_name": "default",
                "branding": {
                    "app_name": "Novera",
                    "app_tagline": None,
                    "logo_url": None,
                    "logo_dark_url": None,
                    "favicon_url": None,
                },
                "colors": {
                    "primary": "#0ea5e9",
                    "secondary": "#d946ef",
                    "accent": "#8b5cf6",
                    "success": "#10b981",
                    "warning": "#f59e0b",
                    "error": "#ef4444",
                    "info": "#3b82f6",
                    "background": "#ffffff",
                    "background_secondary": "#f9fafb",
                    "background_tertiary": "#f3f4f6",
                    "sidebar": "#ffffff",
                    "text_primary": "#111827",
                    "text_secondary": "#6b7280",
                    "border": "#e5e7eb",
                    "shadow": "#00000010",
                }
            }
    
    customization = await get_or_create_customization(db, organization_id)
    
    return customization.to_dict()


__all__ = ['router']