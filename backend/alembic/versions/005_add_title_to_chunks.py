"""add title column to chunks

Revision ID: 003_add_title_to_chunks
Revises: 004_add_typography
Create Date: 2025-12-26 13:15:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '005_add_title_to_chunks'
down_revision = '004_add_typography'  # Change this to your latest migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add title column to chunks table."""
    # Add title column
    op.add_column('chunks', sa.Column('title', sa.String(), nullable=True))
    
    # Optional: Add index for better query performance
    op.create_index(
        'ix_chunks_title',
        'chunks',
        ['title'],
        unique=False
    )


def downgrade() -> None:
    """Remove title column from chunks table."""
    # Drop index
    op.drop_index('ix_chunks_title', table_name='chunks')
    
    # Drop column
    op.drop_column('chunks', 'title')