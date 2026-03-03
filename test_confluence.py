#!/usr/bin/env python3
"""
Test script untuk verifikasi koneksi ke Confluence dan mengambil data dari space tertentu.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import ConfluenceLoader

# Load environment variables dari .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_confluence_connection():
    """
    Test koneksi ke Confluence dan coba load documents dari space yang ditentukan.
    """
    # Get configuration dari environment variables
    confluence_url = os.getenv('CONFLUENCE_URL')
    confluence_username = os.getenv('CONFLUENCE_USERNAME')
    confluence_credential = os.getenv('CONFLUENCE_CREDENTIAL')
    space_key = os.getenv('CONFLUENCE_SPACE_KEY', 'MKB')
    
    # Validate required configurations
    if not all([confluence_url, confluence_username, confluence_credential]):
        logger.error("❌ Konfigurasi Confluence tidak lengkap. Periksa .env file:")
        logger.error(f"   - CONFLUENCE_URL: {confluence_url}")
        logger.error(f"   - CONFLUENCE_USERNAME: {confluence_username}")
        logger.error(f"   - CONFLUENCE_CREDENTIAL: {'****' if confluence_credential else 'MISSING'}")
        return False
    
    logger.info("=" * 60)
    logger.info("TEST CONFLUENCE CONNECTION")
    logger.info("=" * 60)
    
    try:
        logger.info(f"📍 Confluence URL: {confluence_url}")
        logger.info(f"👤 Username: {confluence_username}")
        logger.info(f"📚 Space Key: {space_key}")
        logger.info("\n⏳ Menghubungkan ke Confluence API...")
        
        # Initialize ConfluenceLoader
        loader = ConfluenceLoader(
            url=confluence_url,
            username=confluence_username,
            api_key=confluence_credential,
            space_key=space_key,
            limit=50,
        )
        
        logger.info("✓ ConfluenceLoader berhasil diinisialisasi")
        
        # Try to load documents
        logger.info("📄 Mengambil documents dari space...")
        documents = loader.load()
        
        if not documents:
            logger.warning(f"⚠️  Tidak ada documents ditemukan di space '{space_key}'")
            logger.warning("   Periksa apakah space_key sudah benar dan space memiliki content")
            return True  # Koneksi berhasil, tapi space kosong
        
        logger.info(f"✓ Berhasil mengambil {len(documents)} halaman dari space '{space_key}'")
        
        # Display document details
        logger.info("\n" + "=" * 60)
        logger.info("DETAIL DOCUMENTS YANG BERHASIL DIAMBIL:")
        logger.info("=" * 60)
        
        for idx, doc in enumerate(documents[:5], 1):  # Show first 5 documents
            logger.info(f"\n[{idx}] Halaman: {doc.metadata.get('title', 'Unknown')}")
            logger.info(f"    Source: {doc.metadata.get('source', 'Unknown')}")
            logger.info(f"    Content preview (100 chars): {doc.page_content[:100]}...")
        
        if len(documents) > 5:
            logger.info(f"\n... dan {len(documents) - 5} dokumen lainnya")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✅ TEST CONFLUENCE BERHASIL!")
        logger.info(f"   Total documents: {len(documents)}")
        logger.info("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("❌ TEST CONFLUENCE GAGAL!")
        logger.error("=" * 60)
        logger.error(f"Error: {type(e).__name__}")
        logger.error(f"Detail: {str(e)}")
        logger.error("\nTroubleshooting tips:")
        logger.error("1. Periksa CONFLUENCE_URL - pastikan URL benar dan accessible")
        logger.error("2. Periksa CONFLUENCE_USERNAME - pastikan username benar")
        logger.error("3. Periksa CONFLUENCE_CREDENTIAL - pastikan password/token benar")
        logger.error("4. Periksa CONFLUENCE_SPACE_KEY - pastikan space key benar (case-sensitive)")
        logger.error("5. Coba ping URL: curl -u {username}:{password} {url}/rest/api/space/{space_key}")
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
    success = test_confluence_connection()
    
    # Exit dengan appropriate exit code
    sys.exit(0 if success else 1)
