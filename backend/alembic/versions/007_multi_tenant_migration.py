"""Multi-tenant migration

Revision ID: 007_multi_tenant
Revises: 006_expand_customization
Create Date: 2025-01-XX XX:XX:XX
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime

revision = '007_multi_org'
down_revision = 'b9e94cec8243'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Create organizations table
    op.create_table(
        'organizations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('display_name', sa.String(255), nullable=False),
        sa.Column('description', sa.String(1000), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('max_users', sa.Integer(), nullable=True),
        sa.Column('max_documents', sa.Integer(), nullable=True),
        sa.Column('max_storage_gb', sa.Integer(), nullable=True),
        sa.Column('settings', postgresql.JSONB(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
    )
    
    # 2. Create default organization
    from uuid import uuid4
    default_org_id = uuid4()
    
    op.execute(f"""
        INSERT INTO organizations (id, name, display_name, description, is_active, created_at, updated_at)
        VALUES (
            '{default_org_id}',
            'default',
            'Default Organization',
            'Default organization for existing users',
            true,
            NOW(),
            NOW()
        )
    """)
    
    # 3. Add organization_id to users table
    op.add_column('users', sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=True))
    
    # 4. Migrate existing users to default organization
    op.execute(f"""
        UPDATE users
        SET organization_id = '{default_org_id}'
        WHERE organization_id IS NULL
    """)
    
    # 5. Make organization_id NOT NULL
    op.alter_column('users', 'organization_id', nullable=False)
    
    # 6. Add FK constraint
    op.create_foreign_key(
        'fk_users_organization',
        'users', 'organizations',
        ['organization_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # 7. Update role column to support super_admin
    op.execute("""
        UPDATE users
        SET role = 'org_admin'
        WHERE role = 'admin'
    """)
    
    # 8. Drop old organization_name column
    op.drop_column('users', 'organization_name')
    
    # 9. Add organization_id to documents
    op.add_column('documents', sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=True))
    
    # 10. Migrate documents to default organization
    op.execute(f"""
        UPDATE documents
        SET organization_id = '{default_org_id}'
        WHERE organization_id IS NULL
    """)
    
    # 11. Make documents.organization_id NOT NULL
    op.alter_column('documents', 'organization_id', nullable=False)
    
    # 12. Add FK for documents
    op.create_foreign_key(
        'fk_documents_organization',
        'documents', 'organizations',
        ['organization_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # 13. Update customizations
    op.add_column('organization_customizations', 
                  sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=True))
    
    # 14. Migrate customizations
    op.execute(f"""
        UPDATE organization_customizations
        SET organization_id = '{default_org_id}'
        WHERE organization_name = 'default'
    """)
    
    op.alter_column('organization_customizations', 'organization_id', nullable=False)
    
    op.create_foreign_key(
        'fk_customizations_organization',
        'organization_customizations', 'organizations',
        ['organization_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # 15. Drop organization_name from customizations
    op.drop_column('organization_customizations', 'organization_name')
    
    # 16. Add indexes
    op.create_index('idx_org_name_active', 'organizations', ['name', 'is_active'])
    op.create_index('idx_users_org', 'users', ['organization_id'])
    op.create_index('idx_docs_org_status', 'documents', ['organization_id', 'status'])


def downgrade():
    # Reverse migration if needed
    op.drop_index('idx_docs_org_status', 'documents')
    op.drop_index('idx_users_org', 'users')
    op.drop_index('idx_org_name_active', 'organizations')
    
    # ... (reverse all operations)