#!/usr/bin/env python3
"""
Test script untuk verifikasi koneksi ke Qdrant dan collection management.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables dari .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_qdrant_connection():
    """
    Test koneksi ke Qdrant dan verifikasi collection management.
    """
    # Get configuration dari environment variables
    qdrant_host = os.getenv('QDRANT_HOST', 'qdrant.staytech.xyz')
    qdrant_port = int(os.getenv('QDRANT_PORT', '443'))
    qdrant_https = os.getenv('QDRANT_HTTPS', 'true').lower() == 'true'
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'knowledge_base_ml')
    
    # Validate required configurations
    if not qdrant_host:
        logger.error("❌ QDRANT_HOST tidak ditemukan di .env file")
        return False
    if not qdrant_port:
        logger.error("❌ QDRANT_PORT tidak ditemukan di .env file")
        return False
    
    logger.info("=" * 60)
    logger.info("TEST QDRANT CONNECTION")
    logger.info("=" * 60)
    
    try:
        logger.info(f"📍 Qdrant: {qdrant_host}:{qdrant_port} (HTTPS: {qdrant_https})")
        logger.info(f"🔐 API Key: {'****' if qdrant_api_key else 'NONE (unauthenticated)'}")
        logger.info(f"📚 Collection: {collection_name}")
        logger.info("\n⏳ Menghubungkan ke Qdrant API...")
        
        # Initialize Qdrant client
        client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            https=qdrant_https,
            api_key=qdrant_api_key
        )
        
        logger.info("✓ Qdrant client berhasil diinisialisasi")
        
        # Test basic connectivity by getting server info
        logger.info("🔍 Mengecek server health...")
        try:
            # Try to get collections (basic health check)
            collections = client.get_collections()
            logger.info(f"✓ Server responsive. Total collections: {len(collections.collections)}")
        except Exception as e:
            logger.warning(f"⚠️  Tidak bisa retrieve collections: {str(e)}")
        
        # Check if collection exists
        logger.info(f"\n📋 Checking collection '{collection_name}'...")
        collection_exists = client.collection_exists(collection_name)
        
        if collection_exists:
            logger.info(f"✓ Collection '{collection_name}' sudah ada")
            
            # Get collection info
            collection_info = client.get_collection(collection_name)
            logger.info(f"   - Vector size: {collection_info.config.params.vectors.size}")
            logger.info(f"   - Distance metric: {collection_info.config.params.vectors.distance}")
            logger.info(f"   - Point count: {collection_info.points_count}")
            
        else:
            logger.info(f"⚠️  Collection '{collection_name}' belum ada")
            logger.info(f"\n🔧 Mencoba membuat collection '{collection_name}'...")
            
            # Try to create collection
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                logger.info(f"✓ Collection '{collection_name}' berhasil dibuat")
                
                # Verify creation
                collection_info = client.get_collection(collection_name)
                logger.info(f"   - Vector size: {collection_info.config.params.vectors.size}")
                logger.info(f"   - Distance metric: {collection_info.config.params.vectors.distance}")
                
                logger.info(f"\n🗑️  Menghapus collection test (untuk cleanup)...")
                client.delete_collection(collection_name)
                logger.info(f"✓ Collection dihapus (collection akan dibuat ulang saat embedding)")
                
            except Exception as e:
                logger.error(f"❌ Gagal membuat collection: {str(e)}")
                return False
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ TEST QDRANT BERHASIL!")
        logger.info("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("❌ TEST QDRANT GAGAL!")
        logger.error("=" * 60)
        logger.error(f"Error: {type(e).__name__}")
        logger.error(f"Detail: {str(e)}")
        logger.error("\nTroubleshooting tips:")
        logger.error("1. Periksa QDRANT_URL - pastikan URL benar dan accessible")
        logger.error("2. Periksa QDRANT_API_KEY - pastikan key benar (jika memerlukan auth)")
        logger.error("3. Coba ping URL: curl -H 'api-key: {QDRANT_API_KEY}' {QDRANT_URL}/health")
        logger.error("4. Periksa firewall/ingress rules di Kubernetes cluster")
        logger.error("5. Cek logs Qdrant pod: kubectl logs -n qdrant deployment/qdrant")
        logger.error("=" * 60 + "\n")
        
        return False


if __name__ == "__main__":
    # Check if .env file exists
    if not os.path.exists('.env'):
        logger.warning("⚠️  File .env tidak ditemukan. Menggunakan .env.example sebagai template.")
        logger.warning("   Silakan copy .env.example ke .env dan sesuaikan konfigurasi:")
        logger.warning("   $ cp .env.example .env")
        logger.warning("   Kemudian edit file .env dengan nilai yang sesuai\n")
    
    # Run test
    success = test_qdrant_connection()
    
    # Exit dengan appropriate exit code
    sys.exit(0 if success else 1)
