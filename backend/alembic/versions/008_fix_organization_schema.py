"""Fix organization schema - add missing columns for production.

Revision ID: 008_fix_org_schema
Revises: b9e94cec8243
Create Date: 2025-02-23
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = '008_fix_org_schema'
down_revision: Union[str, None] = '007_multi_org'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _safe_add_column(table: str, column: str, col_type: str):
    """Add column — skips silently if column already exists OR table doesn't exist."""
    op.execute(f"""
        DO $$ BEGIN
            ALTER TABLE {table} ADD COLUMN {column} {col_type};
        EXCEPTION
            WHEN duplicate_column THEN NULL;
            WHEN undefined_table  THEN NULL;
        END $$
    """)


def _safe_create_index(name: str, table: str, columns: str, unique: bool = False):
    """Create index — uses IF NOT EXISTS so it never fails on duplicates."""
    unique_str = "UNIQUE" if unique else ""
    op.execute(
        f"CREATE {unique_str} INDEX IF NOT EXISTS {name} ON {table}({columns});"
    )


def upgrade() -> None:
    """Add all missing columns to organizations and related tables."""

    # ============================================
    # ORGANIZATIONS — Missing columns
    # ============================================
    _safe_add_column('organizations', 'slug', 'VARCHAR(255)')
    _safe_add_column('organizations', 'display_name', 'VARCHAR(255)')
    _safe_add_column('organizations', 'description', 'VARCHAR(1000)')
    _safe_add_column('organizations', 'max_users', 'INTEGER')
    _safe_add_column('organizations', 'max_documents', 'INTEGER')
    _safe_add_column('organizations', 'max_storage_gb', 'INTEGER')
    _safe_add_column('organizations', 'settings', "JSONB DEFAULT '{}'")
    _safe_add_column('organizations', 'created_by', 'UUID')
    _safe_add_column('organizations', 'updated_at', 'TIMESTAMP DEFAULT NOW()')

    # Populate slug from name where NULL
    op.execute("""
        UPDATE organizations 
        SET slug = LOWER(REPLACE(REPLACE(REPLACE(name, ' ', '-'), '_', '-'), '.', '-'))
        WHERE slug IS NULL
    """)

    # Populate display_name where NULL
    op.execute("""
        UPDATE organizations 
        SET display_name = INITCAP(REPLACE(REPLACE(name, '-', ' '), '_', ' '))
        WHERE display_name IS NULL
    """)

    # Make slug NOT NULL after populating
    op.execute("""
        DO $$ BEGIN
            ALTER TABLE organizations ALTER COLUMN slug SET NOT NULL;
        EXCEPTION WHEN others THEN
            NULL;
        END $$
    """)

    # Make display_name NOT NULL after populating
    op.execute("""
        DO $$ BEGIN
            ALTER TABLE organizations ALTER COLUMN display_name SET NOT NULL;
        EXCEPTION WHEN others THEN
            NULL;
        END $$
    """)

    # Unique index on slug
    _safe_create_index('ix_organizations_slug', 'organizations', 'slug', unique=True)
    _safe_create_index('idx_org_name_active', 'organizations', 'name, is_active')

    # ============================================
    # USERS — Missing columns
    # ============================================
    _safe_add_column('users', 'organization_id', 
                     'UUID REFERENCES organizations(id) ON DELETE CASCADE')
    _safe_add_column('users', 'preferences', "JSONB NOT NULL DEFAULT '{}'")
    _safe_add_column('users', 'metadata', "JSONB NOT NULL DEFAULT '{}'")
    _safe_add_column('users', 'avatar_url', 'VARCHAR(512)')

    # Index on organization_id
    _safe_create_index('ix_users_organization_id', 'users', 'organization_id')

    # ============================================
    # REFRESH TOKENS — Missing columns
    # ============================================
    _safe_add_column('refresh_tokens', 'user_agent', 'VARCHAR(512)')
    _safe_add_column('refresh_tokens', 'ip_address', 'VARCHAR(45)')

    # ============================================
    # PASSWORD RESET TOKENS — Missing columns
    # ============================================
    _safe_add_column('password_reset_tokens', 'ip_address', 'VARCHAR(45)')

    # ============================================
    # EMAIL VERIFICATION TOKENS — Missing columns
    # ============================================
    _safe_add_column('email_verification_tokens', 'ip_address', 'VARCHAR(45)')

    # ============================================
    # ORGANIZATION CUSTOMIZATIONS — Missing columns
    # ============================================
    _safe_add_column('organization_customizations', 'dark_mode_enabled', 
                     'BOOLEAN DEFAULT FALSE')
    _safe_add_column('organization_customizations', 'dark_mode_colors', 'JSONB')
    _safe_add_column('organization_customizations', 'app_name', 'VARCHAR(255)')
    _safe_add_column('organization_customizations', 'app_tagline', 'VARCHAR(512)')
    _safe_add_column('organization_customizations', 'theme_name', 'VARCHAR(100)')
    _safe_add_column('organization_customizations', 'theme_description', 'VARCHAR(500)')

    # ============================================
    # SEED DEFAULT ORGANIZATION
    # ============================================
    op.execute("""
        INSERT INTO organizations (
            id, name, slug, display_name, is_active, 
            settings, created_at, updated_at
        )
        VALUES (
            gen_random_uuid(), 'default', 'default', 
            'Default Organization', TRUE, 
            '{}', NOW(), NOW()
        )
        ON CONFLICT (name) DO UPDATE SET
            slug = COALESCE(organizations.slug, EXCLUDED.slug),
            display_name = COALESCE(organizations.display_name, EXCLUDED.display_name),
            updated_at = NOW()
    """)

    # ============================================
    # ASSIGN EXISTING USERS TO DEFAULT ORG
    # ============================================
    op.execute("""
        UPDATE users 
        SET organization_id = (
            SELECT id FROM organizations WHERE name = 'default' LIMIT 1
        )
        WHERE organization_id IS NULL
    """)


def downgrade() -> None:
    """Remove added columns (careful in production)."""
    # Organizations
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS slug")
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS display_name")
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS description")
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS max_users")
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS max_documents")
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS max_storage_gb")
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS settings")
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS created_by")
