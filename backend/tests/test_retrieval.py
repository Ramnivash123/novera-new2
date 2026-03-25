"""
Test script for Phase 3: Retrieval System.
Tests search, reranking, and context assembly.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from loguru import logger
import json

from app.core.config import settings


async def test_basic_search():
    """Test basic search functionality."""
    
    logger.info("="*60)
    logger.info("Test 1: Basic Search")
    logger.info("="*60)
    
    test_queries = [
        "What is the revenue for Q4 2024?",
        "salary benefits compensation",
        "expense breakdown",
        "quarterly comparison table"
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for query in test_queries:
            logger.info(f"\n🔍 Query: '{query}'")
            
            response = await client.post(
                f"http://localhost:8000{settings.api_v1_prefix}/search",
                params={"query": query, "top_k": 3}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info(f"✅ Results: {result['total_results']} chunks")
                logger.info(f"   Strategy: {result['retrieval_strategy']}")
                logger.info(f"   Intent: {result['intent']}")
                logger.info(f"   Complexity: {result['complexity']}")
                
                for idx, chunk in enumerate(result['chunks'], 1):
                    logger.info(f"\n   Chunk {idx}:")
                    logger.info(f"   Score: {chunk['relevance_score']:.3f}")
                    logger.info(f"   Type: {chunk['chunk_type']}")
                    logger.info(f"   Pages: {chunk['page_numbers']}")
                    logger.info(f"   Content: {chunk['content'][:150]}...")
            else:
                logger.error(f"❌ Search failed: {response.status_code}")
                logger.error(f"   Response: {response.text}")


async def test_context_assembly():
    """Test context assembly for LLM."""
    
    logger.info("\n" + "="*60)
    logger.info("Test 2: Context Assembly")
    logger.info("="*60)
    
    query = "What was the net profit and what were the main revenue sources?"
    
    logger.info(f"\n🔍 Query: '{query}'")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"http://localhost:8000{settings.api_v1_prefix}/search/context",
            params={"query": query, "top_k": 5}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            logger.info(f"\n✅ Context assembled:")
            logger.info(f"   Chunks used: {result['chunks_used']}")
            logger.info(f"   Total tokens: {result['total_tokens']}")
            logger.info(f"   Sources: {len(result['sources'])}")
            
            logger.info(f"\n📄 Assembled Context:")
            logger.info("="*60)
            logger.info(result['context_text'])
            logger.info("="*60)
            
            logger.info(f"\n📚 Sources:")
            for idx, source in enumerate(result['sources'], 1):
                logger.info(f"   {idx}. {source['document']} - Page {source['page']}")
                if source['section']:
                    logger.info(f"      Section: {source['section']}")
        else:
            logger.error(f"❌ Context assembly failed: {response.status_code}")


async def test_query_processing():
    """Test different query types."""
    
    logger.info("\n" + "="*60)
    logger.info("Test 3: Query Processing")
    logger.info("="*60)
    
    test_cases = [
        {
            "query": "How to calculate PF contribution?",
            "expected_intent": "procedural"
        },
        {
            "query": "What is the difference between Q3 and Q4 revenue?",
            "expected_intent": "analytical"
        },
        {
            "query": "expense policy guidelines",
            "expected_intent": "compliance"
        },
        {
            "query": "salary structure breakdown",
            "expected_intent": "financial"
        }
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for test in test_cases:
            query = test['query']
            logger.info(f"\n🔍 Query: '{query}'")
            logger.info(f"   Expected intent: {test['expected_intent']}")
            
            response = await client.post(
                f"http://localhost:8000{settings.api_v1_prefix}/search",
                params={"query": query, "top_k": 3}
            )
            
            if response.status_code == 200:
                result = response.json()
                detected_intent = result['intent']
                
                if detected_intent == test['expected_intent']:
                    logger.info(f"   ✅ Correct intent: {detected_intent}")
                else:
                    logger.warning(f"   ⚠️  Detected: {detected_intent} (expected: {test['expected_intent']})")
                
                logger.info(f"   Strategy: {result['retrieval_strategy']}")
                logger.info(f"   Results: {result['total_results']}")


async def test_filtering():
    """Test search with filters."""
    
    logger.info("\n" + "="*60)
    logger.info("Test 4: Filtered Search")
    logger.info("="*60)
    
    # Test doc_type filter
    logger.info("\n📋 Testing doc_type filter...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"http://localhost:8000{settings.api_v1_prefix}/search",
            params={
                "query": "revenue expenses",
                "top_k": 5,
                "doc_type": "finance"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Finance documents: {result['total_results']} results")
            
            # Verify all results are from finance documents
            all_finance = all(
                chunk['metadata'].get('doc_type') == 'finance'
                for chunk in result['chunks']
            )
            
            if all_finance:
                logger.info("   ✅ All results are from finance documents")
            else:
                logger.warning("   ⚠️  Some results are not from finance documents")
        else:
            logger.error(f"❌ Filtered search failed")


async def test_performance():
    """Test search performance."""
    
    logger.info("\n" + "="*60)
    logger.info("Test 5: Performance Benchmarks")
    logger.info("="*60)
    
    import time
    
    queries = [
        "revenue breakdown",
        "salary policy",
        "expense table Q4",
        "What is the net profit margin?",
        "How much did we spend on marketing?"
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        total_time = 0
        
        for query in queries:
            start = time.time()
            
            response = await client.post(
                f"http://localhost:8000{settings.api_v1_prefix}/search",
                params={"query": query, "top_k": 5}
            )
            
            elapsed = time.time() - start
            total_time += elapsed
            
            if response.status_code == 200:
                logger.info(f"✅ '{query}' - {elapsed:.2f}s")
            else:
                logger.error(f"❌ '{query}' - Failed")
        
        avg_time = total_time / len(queries)
        logger.info(f"\n📊 Performance Summary:")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Average time: {avg_time:.2f}s per query")
        logger.info(f"   Queries tested: {len(queries)}")


async def main():
    """Run all retrieval tests."""
    
    logger.info("\n" + "="*60)
    logger.info("🧪 Phase 3: Retrieval System Tests")
    logger.info("="*60 + "\n")
    
    try:
        await test_basic_search()
        await test_context_assembly()
        await test_query_processing()
        await test_filtering()
        await test_performance()
        
        logger.info("\n" + "="*60)
        logger.info("✅ All retrieval tests completed!")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())