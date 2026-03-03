#!/usr/bin/env python3
"""
Script untuk mengecek status Qdrant, collections, dan vector information.

Fitur:
- Cek koneksi ke Qdrant
- List semua collections yang ada
- Lihat detail setiap collection (points count, vector size, distance metric)
- Cek health status Qdrant
- Delete collection jika diperlukan

Usage:
    python check_qdrant.py [--delete COLLECTION_NAME]
    
Examples:
    python check_qdrant.py
    python check_qdrant.py --delete knowledge_base_ml
"""

import os
import sys
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_qdrant_client():
    """
    Initialize Qdrant client dengan configuration dari .env
    """
    qdrant_host = os.getenv('QDRANT_HOST', 'qdrant.staytech.xyz')
    qdrant_port = int(os.getenv('QDRANT_PORT', '443'))
    qdrant_https = os.getenv('QDRANT_HTTPS', 'true').lower() == 'true'
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    if not qdrant_host:
        logger.error("❌ QDRANT_HOST tidak ditemukan di .env")
        return None
    
    try:
        logger.info(f"📍 Connecting to Qdrant: {qdrant_host}:{qdrant_port} (HTTPS: {qdrant_https})")
        client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            https=qdrant_https,
            api_key=qdrant_api_key
        )
        logger.info("✓ Connection successful\n")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to connect: {str(e)}")
        return None


def check_health(client):
    """
    Check Qdrant server health
    """
    logger.info("=" * 60)
    logger.info("QDRANT SERVER HEALTH")
    logger.info("=" * 60)
    
    try:
        # Try to get collections as a health check
        collections = client.get_collections()
        logger.info("✓ Server is responsive")
        logger.info(f"📊 Total collections: {len(collections.collections)}\n")
        return True
    except Exception as e:
        logger.error(f"❌ Server health check failed: {str(e)}\n")
        return False


def list_collections(client):
    """
    List semua collections yang ada
    """
    logger.info("=" * 60)
    logger.info("EXISTING COLLECTIONS")
    logger.info("=" * 60)
    
    try:
        collections = client.get_collections()
        
        if not collections.collections:
            logger.info("📭 No collections found\n")
            return []
        
        logger.info(f"Found {len(collections.collections)} collection(s):\n")
        
        for collection in collections.collections:
            logger.info(f"  📦 Collection: {collection.name}")
            
            # Get detailed info
            try:
                info = client.get_collection(collection.name)
                logger.info(f"     - Points: {info.points_count}")
                logger.info(f"     - Vector size: {info.vectors_config.size if info.vectors_config else 'N/A'}")
                logger.info(f"     - Distance: {info.vectors_config.distance if info.vectors_config else 'N/A'}")
                logger.info("")
            except Exception as e:
                logger.warning(f"     - Error getting details: {str(e)}\n")
        
        return [c.name for c in collections.collections]
    
    except Exception as e:
        logger.error(f"❌ Failed to list collections: {str(e)}\n")
        return []


def get_collection_info(client, collection_name):
    """
    Get detailed information tentang sebuah collection
    """
    logger.info("=" * 60)
    logger.info(f"COLLECTION DETAILS: {collection_name}")
    logger.info("=" * 60)
    
    try:
        info = client.get_collection(collection_name)
        
        logger.info(f"✓ Collection '{collection_name}' exists")
        logger.info(f"  📊 Total points: {info.points_count}")
        
        if info.vectors_config:
            logger.info(f"  📐 Vector size: {info.vectors_config.size}")
            logger.info(f"  📏 Distance metric: {info.vectors_config.distance}")
        
        if info.status:
            logger.info(f"  ✓ Status: {info.status}")
        
        logger.info("")
        return True
        
    except Exception as e:
        error_msg = str(e).lower()
        # Check if the error is just a missing collection rather than a fatal crash
        if "not found" in error_msg or "doesn't exist" in error_msg:
            logger.warning(f"⚠️  Collection not found: {str(e)}\n")
        else:
            logger.error(f"❌ Error: {str(e)}\n")
        return False

def delete_collection(client, collection_name):
    """
    Delete sebuah collection
    """
    logger.info("=" * 60)
    logger.info(f"DELETE COLLECTION: {collection_name}")
    logger.info("=" * 60)
    
    try:
        # Check jika collection ada
        if not client.collection_exists(collection_name):
            logger.warning(f"⚠️  Collection '{collection_name}' does not exist\n")
            return False
        
        # Confirm deletion
        response = input(f"⚠️  Are you sure you want to delete collection '{collection_name}'? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("❌ Deletion cancelled\n")
            return False
        
        # Delete
        client.delete_collection(collection_name)
        logger.info(f"✓ Collection '{collection_name}' successfully deleted\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to delete collection: {str(e)}\n")
        return False


def search_vectors(client, collection_name, query_vector, limit=5):
    """
    Search vectors dalam collection (untuk testing)
    
    Note: Ini hanya example, membutuhkan actual query vector
    """
    logger.info("=" * 60)
    logger.info(f"SEARCH VECTORS: {collection_name}")
    logger.info("=" * 60)
    
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
        )
        
        logger.info(f"✓ Found {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. Score: {result.score:.4f}")
            if result.payload:
                logger.info(f"     Payload: {result.payload}\n")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Search failed: {str(e)}\n")
        return []


def main():
    """
    Main function
    """
    logger.info("=" * 60)
    logger.info("QDRANT CHECKER")
    logger.info("=" * 60)
    logger.info("")
    
    # Get client
    client = get_qdrant_client()
    if not client:
        sys.exit(1)
    
    # Check health
    if not check_health(client):
        logger.warning("⚠️  Server might not be fully responsive\n")
    
    # List collections
    collections = list_collections(client)
    
    # Check untuk specific collection dari .env
    target_collection = os.getenv('QDRANT_COLLECTION_NAME', 'knowledge_base_ml')
    logger.info("=" * 60)
    logger.info(f"CHECKING TARGET COLLECTION: {target_collection}")
    logger.info("=" * 60)
    get_collection_info(client, target_collection)
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✓ Server: Accessible")
    logger.info(f"✓ Collections found: {len(collections)}")
    
    if target_collection in collections:
        info = client.get_collection(target_collection)
        logger.info(f"✓ Target collection '{target_collection}': EXISTS with {info.points_count} points")
    else:
        logger.info(f"⚠️  Target collection '{target_collection}': NOT FOUND (will be created during embedding)")
    
    logger.info("")


if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--delete' and len(sys.argv) > 2:
            collection_to_delete = sys.argv[2]
            client = get_qdrant_client()
            if client:
                delete_collection(client, collection_to_delete)
        else:
            print("Usage: python check_qdrant.py [--delete COLLECTION_NAME]")
            sys.exit(1)
    else:
        main()
