"""add organization customization

Revision ID: 001_add_customization
Revises: 
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from uuid import uuid4

# revision identifiers, used by Alembic.
revision = '001_add_customization'
down_revision = '70a355a1d4ba'
branch_labels = None
depends_on = None


def upgrade():
    # Create organization_customizations table
    op.create_table(
        'organization_customizations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid4),
        sa.Column('organization_name', sa.String(length=255), nullable=False),
        sa.Column('logo_url', sa.String(length=512), nullable=True),
        sa.Column('logo_dark_url', sa.String(length=512), nullable=True),
        sa.Column('favicon_url', sa.String(length=512), nullable=True),
        sa.Column('primary_color', sa.String(length=7), nullable=False, server_default='#0ea5e9'),
        sa.Column('secondary_color', sa.String(length=7), nullable=False, server_default='#d946ef'),
        sa.Column('accent_color', sa.String(length=7), nullable=False, server_default='#8b5cf6'),
        sa.Column('background_color', sa.String(length=7), nullable=False, server_default='#ffffff'),
        sa.Column('sidebar_color', sa.String(length=7), nullable=False, server_default='#ffffff'),
        sa.Column('text_primary_color', sa.String(length=7), nullable=False, server_default='#111827'),
        sa.Column('text_secondary_color', sa.String(length=7), nullable=False, server_default='#6b7280'),
        sa.Column('button_primary_color', sa.String(length=7), nullable=True),
        sa.Column('button_text_color', sa.String(length=7), nullable=False, server_default='#ffffff'),
        sa.Column('custom_settings', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('app_name', sa.String(length=255), nullable=True),
        sa.Column('app_tagline', sa.String(length=512), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('organization_name')
    )
    
    # Create indexes
    op.create_index('idx_org_name_active', 'organization_customizations', ['organization_name', 'is_active'])
    op.create_index(op.f('ix_organization_customizations_organization_name'), 'organization_customizations', ['organization_name'])
    
    # Add organization_name column to users table
    op.add_column('users', sa.Column('organization_name', sa.String(length=255), nullable=True, server_default='default'))
    op.create_index('ix_users_organization_name', 'users', ['organization_name'])
    
    # Insert default organization
    op.execute("""
        INSERT INTO organization_customizations (
            id, organization_name, primary_color, secondary_color, accent_color,
            background_color, sidebar_color, text_primary_color, text_secondary_color,
            button_text_color, custom_settings, is_active, created_at, updated_at
        ) VALUES (
            gen_random_uuid(), 'default', '#0ea5e9', '#d946ef', '#8b5cf6',
            '#ffffff', '#ffffff', '#111827', '#6b7280',
            '#ffffff', '{}', true, now(), now()
        )
    """)


def downgrade():
    # Remove index and column from users
    op.drop_index('ix_users_organization_name', table_name='users')
    op.drop_column('users', 'organization_name')
    
    # Drop indexes
    op.drop_index(op.f('ix_organization_customizations_organization_name'), table_name='organization_customizations')
    op.drop_index('idx_org_name_active', table_name='organization_customizations')
    
    # Drop table
    op.drop_table('organization_customizations')