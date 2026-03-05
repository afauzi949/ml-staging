#!/usr/bin/env python3
"""
Flask web server untuk RAG Chatbot interface.
Menggunakan Qdrant vector store dan OpenAI-compatible LLM melalui sumopod proxy.

Usage:
    python app.py
    # Akses di http://<server-ip>:5000
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load .env dari parent directory (ml-staging/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ==========================================
# SETUP LOGGING
# ==========================================
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==========================================
# KONFIGURASI DARI ENVIRONMENT
# ==========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "seed-2-0-mini-free")

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant.staytech.xyz")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "443"))
QDRANT_HTTPS = os.getenv("QDRANT_HTTPS", "true").lower() == "true"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base_ml")

CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "https://confluence.staytech.xyz")

app = Flask(__name__)

# Global RAG chain dan retriever (diinisialisasi sekali saat startup)
rag_chain = None
retriever = None
init_error = None


def initialize_rag():
    """Inisialisasi RAG chain saat startup Flask."""
    global rag_chain, retriever, init_error

    try:
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_qdrant import QdrantVectorStore
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough

        logger.info("Inisialisasi RAG chain...")

        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base="https://ai.sumopod.com/v1",
        )

        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            https=QDRANT_HTTPS,
            api_key=QDRANT_API_KEY,
        )

        llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.7,
            openai_api_base="https://ai.sumopod.com/v1",
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

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

        rag_chain = (
            {
                "context": retriever
                | (
                    lambda docs: "\n\n".join(
                        [
                            f"[{doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"
                            for doc in docs
                        ]
                    )
                ),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        logger.info("RAG chain berhasil diinisialisasi")
        logger.info(f"  - Embedding model : {EMBEDDING_MODEL}")
        logger.info(f"  - LLM model       : {LLM_MODEL}")
        logger.info(f"  - Collection      : {COLLECTION_NAME}")
        logger.info(f"  - Qdrant          : {QDRANT_HOST}:{QDRANT_PORT}")

    except Exception as e:
        init_error = str(e)
        logger.error(f"Gagal inisialisasi RAG chain: {e}")


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def index():
    return render_template(
        "index.html",
        llm_model=LLM_MODEL,
        collection=COLLECTION_NAME,
        ready=(rag_chain is not None),
        init_error=init_error,
    )


@app.route("/chat", methods=["POST"])
def chat():
    if rag_chain is None:
        return jsonify(
            {
                "error": f"RAG chain belum siap. {init_error or 'Periksa log server.'}",
            }
        ), 503

    data = request.get_json(silent=True)
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "Pesan tidak boleh kosong."}), 400

    question = data["message"].strip()
    logger.info(f"Query: {question}")

    try:
        answer_obj = rag_chain.invoke(question)
        answer_text = (
            answer_obj.content if hasattr(answer_obj, "content") else str(answer_obj)
        )

        retrieved_docs = retriever.invoke(question)
        sources = []
        seen = set()
        for doc in retrieved_docs:
            title = doc.metadata.get("title", "Unknown")
            source = doc.metadata.get("source", "")
            # Bangun URL Confluence jika source berisi page ID
            if source and source not in seen:
                seen.add(source)
                sources.append({"title": title, "url": source})
            elif title not in seen:
                seen.add(title)
                sources.append({"title": title, "url": ""})

        return jsonify({"answer": answer_text, "sources": sources})

    except Exception as e:
        logger.error(f"Error saat query: {e}")
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok" if rag_chain is not None else "initializing",
            "rag_ready": rag_chain is not None,
            "model": LLM_MODEL,
            "collection": COLLECTION_NAME,
        }
    )


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    initialize_rag()
    port = int(os.getenv("CHATBOT_PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Memulai Flask server di port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug)
