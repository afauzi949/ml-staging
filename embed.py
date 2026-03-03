#!/usr/bin/env python3
"""
Script untuk embedding data dari Confluence ke Qdrant menggunakan OpenAI embeddings.

Fitur:
- Load documents dari Confluence space
- Chunk documents dengan RecursiveCharacterTextSplitter
- Generate embeddings menggunakan OpenAI text-embedding-3-small
- Store vectors ke Qdrant vector database
- Comprehensive error handling dan logging
- Configuration via .env file

Usage:
    python embed.py
    
Requirements:
    - .env file dengan CONFLUENCE_URL, CONFLUENCE_USERNAME, dll (see .env.example)
    - Credentials untuk Confluence, OpenAI, dan Qdrant
    - Network access ke ketiga services tersebut
"""

import os
import sys
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import ConfluenceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables dari .env file
load_dotenv()

# ==========================================
# SETUP LOGGING
# ==========================================
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# LOAD KONFIGURASI DARI ENVIRONMENT
# ==========================================
CONFLUENCE_URL = os.getenv('CONFLUENCE_URL')
CONFLUENCE_USERNAME = os.getenv('CONFLUENCE_USERNAME')
CONFLUENCE_CREDENTIAL = os.getenv('CONFLUENCE_CREDENTIAL')
SPACE_KEY = os.getenv('CONFLUENCE_SPACE_KEY', 'MKB')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant.staytech.xyz')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '443'))
QDRANT_HTTPS = os.getenv('QDRANT_HTTPS', 'true').lower() == 'true'
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'knowledge_base_ml')

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))



def validate_configuration():
    """
    Validasi bahwa semua konfigurasi yang diperlukan sudah tersedia.
    """
    logger.info("=" * 60)
    logger.info("VALIDASI KONFIGURASI")
    logger.info("=" * 60)
    
    missing = []
    
    configs = {
        'CONFLUENCE_URL': CONFLUENCE_URL,
        'CONFLUENCE_USERNAME': CONFLUENCE_USERNAME,
        'CONFLUENCE_CREDENTIAL': CONFLUENCE_CREDENTIAL,
        'OPENAI_API_KEY': OPENAI_API_KEY,
        'QDRANT_HOST': QDRANT_HOST,
        'QDRANT_API_KEY': QDRANT_API_KEY,
    }
    
    for key, value in configs.items():
        if not value:
            missing.append(key)
            logger.error(f"❌ {key}: MISSING")
        else:
            # Mask sensitive values
            if 'API_KEY' in key or 'CREDENTIAL' in key:
                display_value = '****' + value[-4:] if len(value) > 4 else '****'
            else:
                display_value = value
            logger.info(f"✓ {key}: {display_value}")
    
    if missing:
        logger.error(f"\n❌ Konfigurasi tidak lengkap. Missing: {', '.join(missing)}")
        logger.error("\nPastikan file .env sudah ada dan lengkap:")
        logger.error("  1. Copy .env.example ke .env: cp .env.example .env")
        logger.error("  2. Edit .env dengan nilai yang sesuai")
        logger.error("  3. Jalankan script ini lagi")
        return False
    
    logger.info("✓ Semua konfigurasi valid\n")
    return True


def main():
    """
    Main pipeline untuk embedding Confluence documents ke Qdrant.
    """
    try:
        # Validate configuration
        if not validate_configuration():
            return False
        
        # ==========================================
        # STEP 1: Load documents dari Confluence
        # ==========================================
        logger.info("=" * 60)
        logger.info("STEP 1: MENGHUBUNGKAN KE CONFLUENCE")
        logger.info("=" * 60)
        
        try:
            logger.info(f"📍 URL: {CONFLUENCE_URL}")
            logger.info(f"👤 Username: {CONFLUENCE_USERNAME}")
            logger.info(f"📚 Space Key: {SPACE_KEY}")
            logger.info("\n⏳ Menghubungkan ke API Confluence...")
            
            loader = ConfluenceLoader(
                url=CONFLUENCE_URL,
                username=CONFLUENCE_USERNAME,
                api_key=CONFLUENCE_CREDENTIAL,
                space_key=SPACE_KEY,
                limit=50,
            )
            
            documents = loader.load()
            
            if not documents:
                logger.error(f"❌ Tidak ada documents ditemukan di space '{SPACE_KEY}'")
                logger.error("   Periksa apakah space_key sudah benar dan memiliki content")
                return False
            
            logger.info(f"✓ Berhasil menarik {len(documents)} halaman dari Confluence\n")
            
        except Exception as e:
            logger.error(f"❌ Gagal menghubungkan ke Confluence: {type(e).__name__}")
            logger.error(f"   Detail: {str(e)}")
            logger.error("\nTroubleshooting:")
            logger.error("  - Periksa CONFLUENCE_URL - pastikan URL benar dan accessible")
            logger.error("  - Periksa CONFLUENCE_USERNAME dan CONFLUENCE_CREDENTIAL")
            logger.error("  - Periksa CONFLUENCE_SPACE_KEY - case-sensitive")
            return False
        
        # ==========================================
        # STEP 2: Chunk documents
        # ==========================================
        logger.info("=" * 60)
        logger.info("STEP 2: MEMECAH TEKS (CHUNKING)")
        logger.info("=" * 60)
        
        try:
            logger.info(f"⚙️  Chunk size: {CHUNK_SIZE} tokens")
            logger.info(f"⚙️  Chunk overlap: {CHUNK_OVERLAP} tokens")
            logger.info("\n⏳ Memecah teks...")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                logger.error("❌ Tidak ada chunks yang dihasilkan dari chunking")
                return False
            
            logger.info(f"✓ Teks berhasil dipecah menjadi {len(chunks)} chunks\n")
            
        except Exception as e:
            logger.error(f"❌ Gagal melakukan chunking: {type(e).__name__}")
            logger.error(f"   Detail: {str(e)}")
            return False
        
        # ==========================================
        # STEP 3: Initialize OpenAI & Qdrant
        # ==========================================
        logger.info("=" * 60)
        logger.info("STEP 3: INISIALISASI OPENAI & QDRANT")
        logger.info("=" * 60)
        
        try:
            logger.info(f"🤖 Embedding model: {EMBEDDING_MODEL}")
            logger.info("⏳ Inisialisasi OpenAI embeddings...")
            
            embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base="https://ai.sumopod.com/v1"
            )
            logger.info("✓ OpenAI embeddings berhasil diinisialisasi")
            
            logger.info(f"\n📍 Qdrant: {QDRANT_HOST}:{QDRANT_PORT} (HTTPS: {QDRANT_HTTPS})")
            logger.info("⏳ Inisialisasi Qdrant client...")
            
            client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                https=QDRANT_HTTPS,
                api_key=QDRANT_API_KEY
            )
            logger.info("✓ Qdrant client berhasil diinisialisasi")
            
            # Check dan create collection jika belum ada
            logger.info(f"\n📋 Checking collection '{COLLECTION_NAME}'...")
            
            if not client.collection_exists(COLLECTION_NAME):
                logger.info(f"⏳ Collection '{COLLECTION_NAME}' belum ada. Membuat collection...")
                
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                logger.info(f"✓ Collection '{COLLECTION_NAME}' berhasil dibuat\n")
            else:
                collection_info = client.get_collection(COLLECTION_NAME)
                logger.info(f"✓ Collection '{COLLECTION_NAME}' sudah ada")
                logger.info(f"  - Current points: {collection_info.points_count}\n")
            
        except Exception as e:
            logger.error(f"❌ Gagal inisialisasi OpenAI/Qdrant: {type(e).__name__}")
            logger.error(f"   Detail: {str(e)}")
            logger.error("\nTroubleshooting:")
            logger.error("  - Periksa OPENAI_API_KEY - pastikan valid dan memiliki quota")
            logger.error("  - Periksa QDRANT_URL dan QDRANT_API_KEY")
            logger.error("  - Pastikan network accessible ke OpenAI dan Qdrant")
            return False
        
        # ==========================================
        # STEP 4: Embed dan upsert ke Qdrant
        # ==========================================
        logger.info("=" * 60)
        logger.info("STEP 4: EMBEDDING DAN UPSERT KE QDRANT")
        logger.info("=" * 60)
        
        try:
            logger.info(f"⏳ Mengirim {len(chunks)} chunks untuk embedding...")
            logger.info("  (Ini mungkin memakan waktu, tergantung jumlah chunks)")
            
            vector_store = QdrantVectorStore.from_documents(
                chunks,
                embeddings,
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                https=QDRANT_HTTPS,
                api_key=QDRANT_API_KEY,
                collection_name=COLLECTION_NAME,
            )
            
            logger.info("✓ Semua chunks berhasil di-embed dan di-upsert ke Qdrant\n")
            
        except Exception as e:
            logger.error(f"❌ Gagal embedding/upsert: {type(e).__name__}")
            logger.error(f"   Detail: {str(e)}")
            logger.error("\nTroubleshooting:")
            logger.error("  - Periksa OpenAI API rate limits")
            logger.error("  - Periksa Qdrant storage capacity")
            logger.error("  - Coba batch processing dengan lebih sedikit chunks")
            return False
        
        # ==========================================
        # SUMMARY
        # ==========================================
        logger.info("=" * 60)
        logger.info("✅ EMBEDDING SELESAI!")
        logger.info("=" * 60)
        logger.info(f"📊 Summary:")
        logger.info(f"   - Documents dari Confluence: {len(documents)}")
        logger.info(f"   - Chunks setelah splitting: {len(chunks)}")
        logger.info(f"   - Embedding model: {EMBEDDING_MODEL}")
        logger.info(f"   - Vector database: {COLLECTION_NAME} @ {QDRANT_HOST}:{QDRANT_PORT}")
        logger.info(f"\n✓ Data siap digunakan untuk RAG!")
        logger.info("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ PIPELINE GAGAL!")
        logger.error("=" * 60)
        logger.error(f"Unexpected error: {type(e).__name__}")
        logger.error(f"Detail: {str(e)}")
        logger.error("=" * 60 + "\n")
        
        import traceback
        logger.debug(traceback.format_exc())
        
        return False



if __name__ == "__main__":
    # Check if .env file exists
    if not os.path.exists('.env'):
        logger.warning("⚠️  File .env tidak ditemukan!")
        logger.warning("   Silakan buat file .env dengan copy dari .env.example:")
        logger.warning("   $ cp .env.example .env")
        logger.warning("   Kemudian edit file .env dengan nilai konfigurasi yang sesuai\n")
        sys.exit(1)
    
    # Run main pipeline
    logger.info("\n" + "🚀 " * 20)
    success = main()
    
    # Exit dengan appropriate exit code
    sys.exit(0 if success else 1)