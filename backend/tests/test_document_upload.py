"""
Test script for document processing pipeline.
Tests upload, processing, and retrieval operations.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import httpx
from loguru import logger
from backend.app.core.config import settings



async def test_document_upload():
    """Test document upload via API."""
    
    logger.info("="*60)
    logger.info("Testing Document Upload API")
    logger.info("="*60)
    
    # Create a test PDF file
    test_file_path = Path("test_data/sample_document.txt")
    test_file_path.parent.mkdir(exist_ok=True)
    
    if not test_file_path.exists():
        with open(test_file_path, "w") as f:
            f.write("""
Sample Financial Document

Company: Example Corp
Department: Finance
Date: 2024-01-15

Section 1: Revenue Summary
Our total revenue for Q4 2024 was $1.5 million, representing a 15% increase 
from the previous quarter. Key drivers included:
- Product sales: $900,000
- Service revenue: $400,000
- Consulting: $200,000

Section 2: Expense Breakdown
Total expenses: $1.2 million
- Salaries: $600,000
- Operations: $300,000
- Marketing: $200,000
- Other: $100,000

Table: Quarterly Comparison
Quarter | Revenue | Expenses | Profit
Q1 2024 | $1.2M   | $1.0M    | $0.2M
Q2 2024 | $1.3M   | $1.1M    | $0.2M
Q3 2024 | $1.4M   | $1.15M   | $0.25M
Q4 2024 | $1.5M   | $1.2M    | $0.3M

Section 3: Outlook
We project continued growth in 2025 with expected revenue of $7 million.
""")
        logger.info(f"Created test file: {test_file_path}")
    
    # Upload document via API
    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(test_file_path, "rb") as f:
            files = {"file": (test_file_path.name, f, "text/plain")}
            params = {
                "doc_type": "finance",
                "department": "Finance Department"
            }
            
            logger.info("Uploading document...")
            response = await client.post(
                f"http://localhost:8000{settings.api_v1_prefix}/documents/upload",
                files=files,
                params=params
            )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Upload successful!")
            logger.info(f"   Document ID: {result['document_id']}")
            logger.info(f"   Status: {result['status']}")
            logger.info(f"   Message: {result['message']}")
            
            document_id = result['document_id']
            
            # Wait for processing to complete
            logger.info("\nWaiting for document processing...")
            await asyncio.sleep(10)  # Give it time to process
            
            # Check status
            status_response = await client.get(
                f"http://localhost:8000{settings.api_v1_prefix}/documents/{document_id}/status"
            )
            
            if status_response.status_code == 200:
                status = status_response.json()
                logger.info(f"\n✅ Processing Status:")
                logger.info(f"   Status: {status['status']}")
                logger.info(f"   Total Pages: {status['total_pages']}")
                logger.info(f"   Total Chunks: {status['total_chunks']}")
                logger.info(f"   Processed Date: {status['processed_date']}")
                
                if status['error']:
                    logger.error(f"   Error: {status['error']}")
            
            return document_id
        else:
            logger.error(f"❌ Upload failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return None


async def test_document_listing():
    """Test document listing API."""
    
    logger.info("\n" + "="*60)
    logger.info("Testing Document Listing API")
    logger.info("="*60)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8000{settings.api_v1_prefix}/documents",
            params={"limit": 10}
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"\n✅ Found {result['total']} documents")
            
            for doc in result['documents']:
                logger.info(f"\n   Document: {doc['filename']}")
                logger.info(f"   ID: {doc['id']}")
                logger.info(f"   Type: {doc['doc_type']}")
                logger.info(f"   Status: {doc['status']}")
                logger.info(f"   Chunks: {doc['total_chunks']}")
        else:
            logger.error(f"❌ Listing failed: {response.status_code}")


async def test_chunk_retrieval():
    """Test chunk retrieval from database."""
    
    logger.info("\n" + "="*60)
    logger.info("Testing Chunk Retrieval")
    logger.info("="*60)
    
    from app.models.document import Document, Chunk
    from sqlalchemy import select
    
    # Create database session
    engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Get first document
        result = await session.execute(
            select(Document).limit(1)
        )
        document = result.scalar_one_or_none()
        
        if document:
            logger.info(f"\nDocument: {document.filename}")
            logger.info(f"Status: {document.status}")
            logger.info(f"Total Chunks: {document.total_chunks}")
            
            # Get chunks for this document
            chunks_result = await session.execute(
                select(Chunk)
                .where(Chunk.document_id == document.id)
                .limit(3)
            )
            chunks = chunks_result.scalars().all()
            
            logger.info(f"\n✅ Retrieved {len(chunks)} chunks:")
            for idx, chunk in enumerate(chunks, 1):
                logger.info(f"\n   Chunk {idx}:")
                logger.info(f"   Type: {chunk.chunk_type}")
                logger.info(f"   Tokens: {chunk.token_count}")
                logger.info(f"   Pages: {chunk.page_numbers}")
                logger.info(f"   Content preview: {chunk.content[:150]}...")
                logger.info(f"   Embedding dimensions: {len(chunk.embedding) if chunk.embedding else 0}")
        else:
            logger.warning("No documents found in database")


async def main():
    """Run all tests."""
    
    logger.info("\n" + "="*60)
    logger.info("🧪 Phase 2: Document Processing Tests")
    logger.info("="*60 + "\n")
    
    try:
        # Test 1: Upload document
        document_id = await test_document_upload()
        
        # Test 2: List documents
        await test_document_listing()
        
        # Test 3: Retrieve chunks
        await test_chunk_retrieval()
        
        logger.info("\n" + "="*60)
        logger.info("✅ All tests completed!")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())