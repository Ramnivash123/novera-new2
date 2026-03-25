"""add typography and layout customization

Revision ID: 004_add_typography
Revises: 001_add_customization
Create Date: 2024-12-23 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '004_add_typography'
down_revision = '001_add_customization'
branch_labels = None
depends_on = None


def upgrade():
    # Add typography columns
    op.add_column('organization_customizations', 
        sa.Column('font_family', sa.String(length=100), nullable=True))
    op.add_column('organization_customizations', 
        sa.Column('font_size_base', sa.String(length=10), nullable=False, server_default='14px'))
    op.add_column('organization_customizations', 
        sa.Column('font_size_heading', sa.String(length=10), nullable=False, server_default='24px'))
    
    # Add layout columns
    op.add_column('organization_customizations', 
        sa.Column('border_radius', sa.String(length=10), nullable=False, server_default='8px'))
    op.add_column('organization_customizations', 
        sa.Column('spacing_unit', sa.String(length=10), nullable=False, server_default='16px'))


def downgrade():
    op.drop_column('organization_customizations', 'spacing_unit')
    op.drop_column('organization_customizations', 'border_radius')
    op.drop_column('organization_customizations', 'font_size_heading')
    op.drop_column('organization_customizations', 'font_size_base')
    op.drop_column('organization_customizations', 'font_family')