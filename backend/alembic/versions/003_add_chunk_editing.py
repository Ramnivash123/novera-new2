"""add chunk editing support

Revision ID: 003_add_chunk_editing
Revises: 002_create_default_admin
Create Date: 2024-12-03 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = '003_add_chunk_editing'
down_revision = '002_create_default_admin'
branch_labels = None
depends_on = None


def upgrade():
    """Add chunk editing fields and edit history table."""
    
    # Add editing fields to chunks table
    op.add_column('chunks', sa.Column('is_edited', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('chunks', sa.Column('edited_at', sa.DateTime(), nullable=True))
    op.add_column('chunks', sa.Column('edited_by', UUID(as_uuid=True), nullable=True))
    op.add_column('chunks', sa.Column('original_content', sa.Text(), nullable=True))
    op.add_column('chunks', sa.Column('edit_count', sa.Integer(), nullable=False, server_default='0'))
    
    # Create chunk_edit_history table
    op.create_table(
        'chunk_edit_history',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('chunk_id', UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('document_id', UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('edited_by', UUID(as_uuid=True), nullable=False),
        sa.Column('edited_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('old_content', sa.Text(), nullable=False),
        sa.Column('new_content', sa.Text(), nullable=False),
        sa.Column('change_summary', sa.String(500), nullable=True),
        sa.Column('metadata', JSONB(), nullable=False, server_default='{}'),
    )
    
    # Add indexes
    op.create_index('idx_chunk_edit_history_chunk', 'chunk_edit_history', ['chunk_id'])
    op.create_index('idx_chunk_edit_history_doc', 'chunk_edit_history', ['document_id'])
    op.create_index('idx_chunk_edit_history_user', 'chunk_edit_history', ['edited_by'])
    op.create_index('idx_chunks_edited', 'chunks', ['is_edited', 'document_id'])
    
    print("\n✅ Chunk editing support added")


def downgrade():
    """Remove chunk editing support."""
    
    # Drop edit history table
    op.drop_index('idx_chunk_edit_history_user', table_name='chunk_edit_history')
    op.drop_index('idx_chunk_edit_history_doc', table_name='chunk_edit_history')
    op.drop_index('idx_chunk_edit_history_chunk', table_name='chunk_edit_history')
    op.drop_table('chunk_edit_history')
    
    # Remove columns from chunks
    op.drop_index('idx_chunks_edited', table_name='chunks')
    op.drop_column('chunks', 'edit_count')
    op.drop_column('chunks', 'original_content')
    op.drop_column('chunks', 'edited_by')
    op.drop_column('chunks', 'edited_at')
    op.drop_column('chunks', 'is_edited')
    
    print("\n✅ Chunk editing support removed")