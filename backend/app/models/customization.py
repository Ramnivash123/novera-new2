"""
Enhanced Organization customization model with comprehensive theming support.
"""
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import Column, String, Boolean, DateTime, Index, Text, ForeignKey, inspect as sa_inspect
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import relationship

from app.db.session import Base


class OrganizationCustomization(Base):
    """
    Enhanced organization customization settings for comprehensive white-labeling.
    Supports complete theme customization across all UI components.
    """
    __tablename__ = "organization_customizations"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), unique=True, nullable=False)

    # Relationship:
    organization = relationship("Organization", back_populates="customization")
    
    # Branding Assets
    logo_url = Column(String(512), nullable=True)
    logo_dark_url = Column(String(512), nullable=True)
    favicon_url = Column(String(512), nullable=True)
    app_name = Column(String(255), nullable=True)
    app_tagline = Column(String(512), nullable=True)
    
    # Primary Color System
    primary_color = Column(String(9), nullable=False, default="#0ea5e9")
    secondary_color = Column(String(9), nullable=False, default="#d946ef")
    accent_color = Column(String(9), nullable=False, default="#8b5cf6")

    # Semantic Colors
    success_color = Column(String(9), nullable=False, default="#10b981")
    warning_color = Column(String(9), nullable=False, default="#f59e0b")
    error_color = Column(String(9), nullable=False, default="#ef4444")
    info_color = Column(String(9), nullable=False, default="#3b82f6")

    # Background Colors
    background_color = Column(String(9), nullable=False, default="#ffffff")
    background_secondary = Column(String(9), nullable=False, default="#f9fafb")
    background_tertiary = Column(String(9), nullable=False, default="#f3f4f6")
    sidebar_color = Column(String(9), nullable=False, default="#ffffff")

    # Text Colors
    text_primary_color = Column(String(9), nullable=False, default="#111827")
    text_secondary_color = Column(String(9), nullable=False, default="#6b7280")

    # Border and Shadow
    border_color = Column(String(9), nullable=False, default="#e5e7eb")
    shadow_color = Column(String(9), nullable=False, default="#00000010")

    # Button Styling
    button_primary_color = Column(String(9), nullable=True)
    button_text_color = Column(String(9), nullable=False, default="#ffffff")
    button_secondary_color = Column(String(9), nullable=True)
    button_secondary_text = Column(String(9), nullable=False, default="#374151")

    button_border_radius = Column(String(10), nullable=False, default="8px")
    input_border_radius = Column(String(10), nullable=False, default="8px")
    card_border_radius = Column(String(10), nullable=False, default="12px")
    card_shadow = Column(String(50), nullable=False, default="0 1px 3px rgba(0,0,0,0.1)")

    # Input Styling
    input_border_color = Column(String(9), nullable=False, default="#d1d5db")
    input_focus_color = Column(String(9), nullable=True)

    # Card Styling
    card_background = Column(String(9), nullable=False, default="#ffffff")
    card_border_color = Column(String(9), nullable=False, default="#e5e7eb")

    # Navigation Styling
    nav_background = Column(String(9), nullable=True)
    nav_text_color = Column(String(9), nullable=True)
    nav_active_color = Column(String(9), nullable=True)
    nav_hover_color = Column(String(9), nullable=True)
    
    # Typography
    font_family = Column(String(100), nullable=True)
    font_size_base = Column(String(10), nullable=False, default="14px")
    font_size_heading = Column(String(10), nullable=False, default="24px")
    font_weight_normal = Column(String(10), nullable=False, default="400")
    font_weight_medium = Column(String(10), nullable=False, default="500")
    font_weight_bold = Column(String(10), nullable=False, default="700")
    line_height_base = Column(String(10), nullable=False, default="1.5")
    line_height_heading = Column(String(10), nullable=False, default="1.2")
    letter_spacing = Column(String(10), nullable=False, default="0")
    
    # Layout and Spacing
    border_radius = Column(String(10), nullable=False, default="8px")
    spacing_unit = Column(String(10), nullable=False, default="16px")
    spacing_xs = Column(String(10), nullable=False, default="4px")
    spacing_sm = Column(String(10), nullable=False, default="8px")
    spacing_md = Column(String(10), nullable=False, default="16px")
    spacing_lg = Column(String(10), nullable=False, default="24px")
    spacing_xl = Column(String(10), nullable=False, default="32px")
    
    # Animation
    animation_speed = Column(String(10), nullable=False, default="300ms")
    enable_animations = Column(Boolean, default=True, nullable=False)
    
    # Dark Mode
    dark_mode_enabled = Column(Boolean, default=False, nullable=False)
    dark_mode_colors = Column(JSONB, nullable=True)
    
    # Advanced Customization
    custom_css = Column(Text, nullable=True)
    custom_settings = Column(JSONB, nullable=False, default=dict)
    
    # Theme Metadata
    theme_name = Column(String(100), nullable=True)
    theme_description = Column(String(500), nullable=True)
    is_preset = Column(Boolean, default=False, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(PGUUID(as_uuid=True), nullable=True)
    
    __table_args__ = (
        Index('idx_org_id_active', 'organization_id', 'is_active'),
    )

    def __repr__(self):
        """Fixed: Use organization_id instead of organization_name."""
        return f"<OrganizationCustomization(org_id={self.organization_id}, theme={self.theme_name})>"

    def to_dict(self):
        """
        Convert to dictionary for API responses with comprehensive structure.
        
        IMPORTANT: Defensive against lazy loading in async contexts.
        Only accesses organization relationship if already loaded.
        """
        data = {
            "id": str(self.id),
            "organization_id": str(self.organization_id),
            "branding": {
                "app_name": self.app_name,
                "app_tagline": self.app_tagline,
                "logo_url": self.logo_url,
                "logo_dark_url": self.logo_dark_url,
                "favicon_url": self.favicon_url,
            },
            "colors": {
                "primary": self.primary_color,
                "secondary": self.secondary_color,
                "accent": self.accent_color,
                "success": self.success_color,
                "warning": self.warning_color,
                "error": self.error_color,
                "info": self.info_color,
                "background": self.background_color,
                "background_secondary": self.background_secondary,
                "background_tertiary": self.background_tertiary,
                "sidebar": self.sidebar_color,
                "text_primary": self.text_primary_color,
                "text_secondary": self.text_secondary_color,
                "border": self.border_color,
                "shadow": self.shadow_color,
            },
            "buttons": {
                "primary_color": self.button_primary_color or self.primary_color,
                "primary_text": self.button_text_color,
                "secondary_color": self.button_secondary_color or self.background_secondary,
                "secondary_text": self.button_secondary_text,
                "border_radius": self.button_border_radius,
            },
            "inputs": {
                "border_color": self.input_border_color,
                "focus_color": self.input_focus_color or self.primary_color,
                "border_radius": self.input_border_radius,
            },
            "cards": {
                "background": self.card_background,
                "border_color": self.card_border_color,
                "border_radius": self.card_border_radius,
                "shadow": self.card_shadow,
            },
            "navigation": {
                "background": self.nav_background or self.sidebar_color,
                "text_color": self.nav_text_color or self.text_secondary_color,
                "active_color": self.nav_active_color or self.primary_color,
                "hover_color": self.nav_hover_color or f"{self.primary_color}10",
            },
            "typography": {
                "font_family": self.font_family,
                "font_size_base": self.font_size_base,
                "font_size_heading": self.font_size_heading,
                "font_weight_normal": self.font_weight_normal,
                "font_weight_medium": self.font_weight_medium,
                "font_weight_bold": self.font_weight_bold,
                "line_height_base": self.line_height_base,
                "line_height_heading": self.line_height_heading,
                "letter_spacing": self.letter_spacing,
            },
            "layout": {
                "border_radius": self.border_radius,
                "spacing_unit": self.spacing_unit,
                "spacing_xs": self.spacing_xs,
                "spacing_sm": self.spacing_sm,
                "spacing_md": self.spacing_md,
                "spacing_lg": self.spacing_lg,
                "spacing_xl": self.spacing_xl,
            },
            "animations": {
                "speed": self.animation_speed,
                "enabled": self.enable_animations,
            },
            "dark_mode": {
                "enabled": self.dark_mode_enabled,
                "colors": self.dark_mode_colors or {},
            },
            "advanced": {
                "custom_css": self.custom_css,
                "custom_settings": self.custom_settings,
            },
            "metadata": {
                "theme_name": self.theme_name,
                "theme_description": self.theme_description,
                "is_preset": self.is_preset,
                "is_active": self.is_active,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            }
        }
        
        insp = sa_inspect(self)
        if 'organization' not in insp.unloaded and self.organization:
            data["organization"] = {
                "id": str(self.organization.id),
                "name": self.organization.name,
                "display_name": self.organization.display_name
            }
        
        return data


__all__ = ['OrganizationCustomization']