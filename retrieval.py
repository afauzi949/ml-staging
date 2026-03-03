#!/usr/bin/env python3
"""
Script untuk melakukan RAG (Retrieval-Augmented Generation) queries menggunakan Qdrant dan OpenAI.

Fitur:
- Retrieve relevant documents dari Qdrant vector store
- Generate answers menggunakan OpenAI GPT model
- Interactive query mode
- Comprehensive error handling dan logging

Usage:
    python retrieval.py "Your question here"
    python retrieval.py  # Interactive mode
    
Requirements:
    - .env file dengan OPENAI_API_KEY, QDRANT_URL, dll (see .env.example)
    - Data yang sudah di-embed ke Qdrant (run embed.py first)
"""

import os
import sys
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from qdrant_client import QdrantClient

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
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
LLM_MODEL = os.getenv('LLM_MODEL', 'seed-2-0-mini-free')

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'knowledge_base_ml')


def validate_configuration():
    """
    Validasi bahwa konfigurasi yang diperlukan untuk retrieval sudah tersedia.
    """
    missing = []
    
    configs = {
        'OPENAI_API_KEY': OPENAI_API_KEY,
        'QDRANT_URL': QDRANT_URL,
    }
    
    for key, value in configs.items():
        if not value:
            missing.append(key)
    
    if missing:
        logger.error(f"❌ Konfigurasi tidak lengkap. Missing: {', '.join(missing)}")
        logger.error("\nPastikan file .env sudah ada dan lengkap")
        return False
    
    return True


def initialize_rag_chain():
    """
    Initialize RAG chain dengan Qdrant vector store dan OpenAI LLM.
    """
    try:
        logger.info("\n⏳ Inisialisasi RAG chain...")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Initialize vector store
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        # Initialize LLM
        llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.7,
        )
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Define prompt template
        template = """Anda adalah helpbot yang membantu menjawab pertanyaan berdasarkan knowledge base ML.

Gunakan informasi berikut dari knowledge base untuk menjawab pertanyaan:
{context}

Pertanyaan: {question}

Instruksi:
1. Berikan jawaban yang akurat dan singkat berdasarkan konteks yang diberikan
2. Jika informasi tidak tersedia di knowledge base, katakan "Informasi tidak tersedia di knowledge base"
3. Gunakan bahasa Indonesia yang jelas dan mudah dipahami
4. Jika relevan, tambahkan referensi ke sumber dokumen

Jawaban:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain
        rag_chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join([
                    f"[{doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"
                    for doc in docs
                ])),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        
        logger.info("✓ RAG chain berhasil diinisialisasi")
        logger.info(f"  - Embedding model: {EMBEDDING_MODEL}")
        logger.info(f"  - LLM model: {LLM_MODEL}")
        logger.info(f"  - Collection: {COLLECTION_NAME}")
        logger.info(f"  - Qdrant URL: {QDRANT_URL}")
        
        return rag_chain, retriever
        
    except Exception as e:
        logger.error(f"❌ Gagal inisialisasi RAG chain: {type(e).__name__}")
        logger.error(f"   Detail: {str(e)}")
        logger.error("\nTroubleshooting:")
        logger.error("  - Pastikan data sudah di-embed ke Qdrant (run embed.py terlebih dahulu)")
        logger.error("  - Periksa OPENAI_API_KEY - pastikan valid dan memiliki quota")
        logger.error("  - Periksa QDRANT_URL dan QDRANT_API_KEY")
        
        return None, None


def query_rag(rag_chain, retriever, question):
    """
    Query RAG chain dengan pertanyaan dan return jawaban + retrieved documents.
    """
    try:
        logger.info(f"\n🔍 Query: {question}")
        logger.info("⏳ Retrieving documents dan generating answer...")
        
        # Get answer from RAG chain
        answer = rag_chain.invoke(question)
        
        # Get retrieved documents
        retrieved_docs = retriever.invoke(question)
        
        # Display answer
        logger.info("\n" + "=" * 60)
        logger.info("📝 JAWABAN:")
        logger.info("=" * 60)
        
        if hasattr(answer, 'content'):
            print(answer.content)
            answer_text = answer.content
        else:
            print(str(answer))
            answer_text = str(answer)
        
        # Display retrieved documents
        logger.info("\n" + "=" * 60)
        logger.info(f"📚 DOKUMEN RETRIEVE ({len(retrieved_docs)} dokumen):")
        logger.info("=" * 60)
        
        for idx, doc in enumerate(retrieved_docs, 1):
            title = doc.metadata.get('title', 'Unknown')
            source = doc.metadata.get('source', 'Unknown')
            logger.info(f"\n[{idx}] {title}")
            logger.info(f"    Source: {source}")
            logger.info(f"    Preview: {doc.page_content[:150]}...")
        
        logger.info("\n" + "=" * 60 + "\n")
        
        return answer_text, retrieved_docs
        
    except Exception as e:
        logger.error(f"❌ Query gagal: {type(e).__name__}")
        logger.error(f"   Detail: {str(e)}")
        
        import traceback
        logger.debug(traceback.format_exc())
        
        return None, None


def interactive_mode(rag_chain, retriever):
    """
    Mode interaktif untuk query RAG chain multiple times.
    """
    logger.info("\n" + "=" * 60)
    logger.info("INTERACTIVE RAG MODE")
    logger.info("=" * 60)
    logger.info("Ketik pertanyaan Anda (atau 'exit' untuk keluar):")
    logger.info("=" * 60 + "\n")
    
    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()
            
            if not question:
                logger.warning("⚠️  Pertanyaan tidak boleh kosong")
                continue
            
            if question.lower() in ['exit', 'quit', 'keluar']:
                logger.info("\n👋 Terima kasih! Goodbye!")
                break
            
            # Query RAG
            answer, docs = query_rag(rag_chain, retriever, question)
            
            if not answer:
                logger.error("❌ Tidak ada jawaban yang dihasilkan")
                continue
            
        except KeyboardInterrupt:
            logger.info("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Error: {str(e)}")
            continue


def main():
    """
    Main function untuk RAG retrieval.
    """
    # Validate configuration
    if not validate_configuration():
        return False
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        logger.error("❌ File .env tidak ditemukan!")
        logger.error("   Copy dari .env.example dan edit sesuai konfigurasi:")
        logger.error("   $ cp .env.example .env")
        return False
    
    logger.info("=" * 60)
    logger.info("RAG RETRIEVAL - ML Knowledge Base")
    logger.info("=" * 60)
    
    # Initialize RAG chain
    rag_chain, retriever = initialize_rag_chain()
    
    if not rag_chain or not retriever:
        logger.error("❌ Gagal inisialisasi RAG chain")
        return False
    
    # Check if we have command-line arguments or interactive mode
    if len(sys.argv) > 1:
        # Query mode: python retrieval.py "Your question"
        question = " ".join(sys.argv[1:])
        answer, docs = query_rag(rag_chain, retriever, question)
        return answer is not None
    else:
        # Interactive mode: python retrieval.py
        interactive_mode(rag_chain, retriever)
        return True


if __name__ == "__main__":
    logger.info("\n" + "🚀 " * 20)
    success = main()
    sys.exit(0 if success else 1)
