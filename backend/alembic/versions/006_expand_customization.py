"""expand customization schema for comprehensive theming

Revision ID: 006_expand_customization
Revises: 005_add_title_to_chunks
Create Date: 2024-12-27 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '006_expand_customization'
down_revision = '005_add_title_to_chunks'
branch_labels = None
depends_on = None


def upgrade():
    # Add semantic color fields
    op.add_column('organization_customizations', 
        sa.Column('success_color', sa.String(length=7), nullable=False, server_default='#10b981'))
    op.add_column('organization_customizations', 
        sa.Column('warning_color', sa.String(length=7), nullable=False, server_default='#f59e0b'))
    op.add_column('organization_customizations', 
        sa.Column('error_color', sa.String(length=7), nullable=False, server_default='#ef4444'))
    op.add_column('organization_customizations', 
        sa.Column('info_color', sa.String(length=7), nullable=False, server_default='#3b82f6'))
    
    # Add background variants
    op.add_column('organization_customizations', 
        sa.Column('background_secondary', sa.String(length=7), nullable=False, server_default='#f9fafb'))
    op.add_column('organization_customizations', 
        sa.Column('background_tertiary', sa.String(length=7), nullable=False, server_default='#f3f4f6'))
    
    # Add border and shadow colors
    op.add_column('organization_customizations', 
        sa.Column('border_color', sa.String(length=7), nullable=False, server_default='#e5e7eb'))
    op.add_column('organization_customizations', 
        sa.Column('shadow_color', sa.String(length=9), server_default='#00000010', nullable=False))
    
    # Add component styling fields
    op.add_column('organization_customizations', 
        sa.Column('button_border_radius', sa.String(length=10), nullable=False, server_default='8px'))
    op.add_column('organization_customizations', 
        sa.Column('button_secondary_color', sa.String(length=7), nullable=True))
    op.add_column('organization_customizations', 
        sa.Column('button_secondary_text', sa.String(length=7), nullable=False, server_default='#374151'))
    
    op.add_column('organization_customizations', 
        sa.Column('input_border_color', sa.String(length=7), nullable=False, server_default='#d1d5db'))
    op.add_column('organization_customizations', 
        sa.Column('input_focus_color', sa.String(length=7), nullable=True))
    op.add_column('organization_customizations', 
        sa.Column('input_border_radius', sa.String(length=10), nullable=False, server_default='8px'))
    
    op.add_column('organization_customizations', 
        sa.Column('card_background', sa.String(length=7), nullable=False, server_default='#ffffff'))
    op.add_column('organization_customizations', 
        sa.Column('card_border_color', sa.String(length=7), nullable=False, server_default='#e5e7eb'))
    op.add_column('organization_customizations', 
        sa.Column('card_border_radius', sa.String(length=10), nullable=False, server_default='12px'))
    op.add_column('organization_customizations', 
        sa.Column('card_shadow', sa.String(length=50), nullable=False, server_default='0 1px 3px rgba(0,0,0,0.1)'))
    
    # Add navigation styling
    op.add_column('organization_customizations', 
        sa.Column('nav_background', sa.String(length=7), nullable=True))
    op.add_column('organization_customizations', 
        sa.Column('nav_text_color', sa.String(length=7), nullable=True))
    op.add_column('organization_customizations', 
        sa.Column('nav_active_color', sa.String(length=7), nullable=True))
    op.add_column('organization_customizations', 
        sa.Column('nav_hover_color', sa.String(length=7), nullable=True))
    
    # Add typography enhancements
    op.add_column('organization_customizations', 
        sa.Column('font_weight_normal', sa.String(length=10), nullable=False, server_default='400'))
    op.add_column('organization_customizations', 
        sa.Column('font_weight_medium', sa.String(length=10), nullable=False, server_default='500'))
    op.add_column('organization_customizations', 
        sa.Column('font_weight_bold', sa.String(length=10), nullable=False, server_default='700'))
    op.add_column('organization_customizations', 
        sa.Column('line_height_base', sa.String(length=10), nullable=False, server_default='1.5'))
    op.add_column('organization_customizations', 
        sa.Column('line_height_heading', sa.String(length=10), nullable=False, server_default='1.2'))
    op.add_column('organization_customizations', 
        sa.Column('letter_spacing', sa.String(length=10), nullable=False, server_default='0'))
    
    # Add spacing scale
    op.add_column('organization_customizations', 
        sa.Column('spacing_xs', sa.String(length=10), nullable=False, server_default='4px'))
    op.add_column('organization_customizations', 
        sa.Column('spacing_sm', sa.String(length=10), nullable=False, server_default='8px'))
    op.add_column('organization_customizations', 
        sa.Column('spacing_md', sa.String(length=10), nullable=False, server_default='16px'))
    op.add_column('organization_customizations', 
        sa.Column('spacing_lg', sa.String(length=10), nullable=False, server_default='24px'))
    op.add_column('organization_customizations', 
        sa.Column('spacing_xl', sa.String(length=10), nullable=False, server_default='32px'))
    
    # Add animation preferences
    op.add_column('organization_customizations', 
        sa.Column('animation_speed', sa.String(length=10), nullable=False, server_default='300ms'))
    op.add_column('organization_customizations', 
        sa.Column('enable_animations', sa.Boolean(), nullable=False, server_default='true'))
    
    # Add dark mode support
    op.add_column('organization_customizations', 
        sa.Column('dark_mode_enabled', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('organization_customizations', 
        sa.Column('dark_mode_colors', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    
    # Add custom CSS support
    op.add_column('organization_customizations', 
        sa.Column('custom_css', sa.Text(), nullable=True))
    
    # Add theme metadata
    op.add_column('organization_customizations', 
        sa.Column('theme_name', sa.String(length=100), nullable=True))
    op.add_column('organization_customizations', 
        sa.Column('theme_description', sa.String(length=500), nullable=True))
    op.add_column('organization_customizations', 
        sa.Column('is_preset', sa.Boolean(), nullable=False, server_default='false'))


def downgrade():
    # Remove all new columns
    columns_to_drop = [
        'success_color', 'warning_color', 'error_color', 'info_color',
        'background_secondary', 'background_tertiary', 'border_color', 'shadow_color',
        'button_border_radius', 'button_secondary_color', 'button_secondary_text',
        'input_border_color', 'input_focus_color', 'input_border_radius',
        'card_background', 'card_border_color', 'card_border_radius', 'card_shadow',
        'nav_background', 'nav_text_color', 'nav_active_color', 'nav_hover_color',
        'font_weight_normal', 'font_weight_medium', 'font_weight_bold',
        'line_height_base', 'line_height_heading', 'letter_spacing',
        'spacing_xs', 'spacing_sm', 'spacing_md', 'spacing_lg', 'spacing_xl',
        'animation_speed', 'enable_animations',
        'dark_mode_enabled', 'dark_mode_colors',
        'custom_css', 'theme_name', 'theme_description', 'is_preset'
    ]
    
    for column in columns_to_drop:
        op.drop_column('organization_customizations', column)