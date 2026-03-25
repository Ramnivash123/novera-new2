"""
API endpoints module.
"""
from app.api.endpoints import (
    health,
    auth,
    documents,
    chat,
    search,
    admin,
    document_editor,
    customization,
    organization,
)

__all__ = [
    'health',
    'auth',
    'documents',
    'chat',
    'search',
    'admin',
    'document_editor',
    'customization',
    'organization',
]