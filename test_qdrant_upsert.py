#!/usr/bin/env python3
"""
Test script untuk test upsert points ke Qdrant.
Ini akan membantu identify apakah error terjadi pada read operation atau write operation.
"""

import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Test Qdrant upsert operation
    """
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    logger.info("=" * 60)
    logger.info("QDRANT UPSERT TEST")
    logger.info("=" * 60)
    logger.info(f"📍 URL: {qdrant_url}")
    logger.info("")
    
    try:
        # Initialize client
        logger.info("⏳ Connecting to Qdrant...")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, verify=False)
        logger.info("✓ Connected\n")
        
        # Create test collection
        collection_name = "test_collection"
        logger.info(f"📋 Creating test collection '{collection_name}'...")
        
        if client.collection_exists(collection_name):
            logger.info(f"⚠️  Collection already exists, deleting...")
            client.delete_collection(collection_name)
        
        # Create collection with 1536 dimensions (OpenAI embedding size)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        logger.info(f"✓ Collection created\n")
        
        # Create dummy vectors for testing
        logger.info("📐 Creating dummy vectors...")
        points = [
            PointStruct(
                id=1,
                vector=[0.1] * 1536,  # Dummy vector
                payload={"text": "Test document 1", "source": "test"}
            ),
            PointStruct(
                id=2,
                vector=[0.2] * 1536,  # Different dummy vector
                payload={"text": "Test document 2", "source": "test"}
            ),
        ]
        logger.info(f"✓ Created {len(points)} dummy points\n")
        
        # Upsert points
        logger.info("⏳ Upserting points to collection...")
        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        logger.info("✓ Upsert successful\n")
        
        # Get collection info
        logger.info("📊 Getting collection info...")
        info = client.get_collection(collection_name)
        logger.info(f"✓ Collection points: {info.points_count}\n")
        
        # Search test
        logger.info("🔍 Testing search...")
        results = client.search(
            collection_name=collection_name,
            query_vector=[0.15] * 1536,
            limit=2,
        )
        logger.info(f"✓ Search returned {len(results)} results\n")
        
        # Cleanup - delete test collection
        logger.info("🗑️  Cleaning up test collection...")
        client.delete_collection(collection_name)
        logger.info("✓ Test collection deleted\n")
        
        logger.info("=" * 60)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("=" * 60)
        logger.info("Qdrant is fully operational for embedding operations.")
        logger.info("Ready to proceed with embed.py")
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ TEST FAILED: {type(e).__name__}")
        logger.error(f"Detail: {str(e)}")
        logger.error("=" * 60)
        logger.error("\nTroubleshooting:")
        logger.error("1. Check network connectivity: ping qdrant.staytech.xyz")
        logger.error("2. Check Qdrant logs: kubectl logs -n qdrant deployment/qdrant")
        logger.error("3. Check firewall rules in Kubernetes")
        logger.error("4. Verify QDRANT_URL and QDRANT_API_KEY in .env")


if __name__ == "__main__":
    main()
