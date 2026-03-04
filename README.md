# ML Knowledge Base - Confluence to Qdrant Embedding

Script untuk melakukan embedding data dari Confluence ke Qdrant Vector Database menggunakan OpenAI embeddings model.

## 📋 Daftar Isi

1. [Prerequisites](#prerequisites)
2. [Instalasi Dependencies](#instalasi-dependencies)
3. [Setup Konfigurasi](#setup-konfigurasi)
4. [Troubleshooting Confluence Authentication](#troubleshooting-confluence-authentication)
5. [Cara Menjalankan Scripts](#cara-menjalankan-scripts)
6. [Struktur Project](#struktur-project)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Sebelum menjalankan script, pastikan sudah memiliki:

### 1. **Confluence Instance yang Accessible**
- Confluence server yang sudah di-deploy di Kubernetes
- Space dengan key "MKB" (atau sesuai CONFLUENCE_SPACE_KEY di .env)
- Data/Pages yang sudah ada di space tersebut
- Credentials untuk akses Confluence (username + password atau Personal Access Token)

### 2. **Qdrant Vector Database**
- Qdrant instance yang sudah di-deploy di Kubernetes
- Accessible via domain (contoh: `qdrant.staytech.xyz`)
- API Key (jika authentication diperlukan)

### 3. **OpenAI API**
- OpenAI API Key dengan akses ke embedding models (`text-embedding-3-small`, `text-embedding-3-large`)
- Quota/balance yang cukup untuk embedding

### 4. **Python Environment**
- Python 3.8+ installed
- pip atau conda untuk install dependencies

### 5. **Network Access**
- Akses ke ketiga services: Confluence, Qdrant, dan OpenAI API
- DNS resolution untuk domains yang dipakai

---

## Instalasi Dependencies

### Opsi 1: Menggunakan requirements.txt (Recommended)

```bash
# Install semua dependencies sekaligus
pip install -r requirements.txt
```

### Opsi 2: Install manual

```bash
# Core dependencies untuk LangChain & Qdrant
pip install langchain==0.1.10
pip install 'langchain-community>=0.0.25'
pip install langchain-openai==0.0.8
pip install langchain-qdrant==0.1.3
pip install 'qdrant-client>=1.10.1'
pip install openai>=1.3.0

# Confluence integration
pip install atlassian-python-api==3.41.0

# Environment variable management
pip install python-dotenv==1.0.0
```

### Cek instalasi

```bash
python3 -c "import langchain; import qdrant_client; import openai; print('✓ All packages installed')"
```

---

## Setup Konfigurasi

### 1. Copy template .env

```bash
cp .env.example .env
```

### 2. Edit file .env dengan konfigurasi sesuai

```bash
nano .env
# atau editor lain sesuai preferensi
```

### 3. Konfigurasi yang perlu diisi:

#### **Confluence Configuration**
```env
CONFLUENCE_URL=https://confluence.staytech.xyz  # Ganti ke https jika mendapat 308 redirect
CONFLUENCE_USERNAME=admin
CONFLUENCE_CREDENTIAL=****  # Password atau Personal Access Token
CONFLUENCE_SPACE_KEY=MKB
```

#### **OpenAI Configuration**
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

#### **Qdrant Configuration**
```env
QDRANT_URL=http://qdrant.staytech.xyz
QDRANT_API_KEY=****
QDRANT_COLLECTION_NAME=knowledge_base_ml
```

#### **Application Configuration** (optional, sudah ada default)
```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
LOG_LEVEL=INFO
```

---

## Troubleshooting Confluence Authentication

### Error: "Basic Authentication has been disabled on this instance"

Jika mendapat error ini saat test_confluence.py, berarti Confluence instance Anda menggunakan authentication method lain (biasanya Personal Access Token).

#### **Solusi: Gunakan Personal Access Token (PAT)**

##### Di Confluence Cloud atau Server dengan PAT enabled:

1. Login ke Confluence dengan akun admin
2. Pergi ke Settings → Personal Access Tokens (atau Manage Account)
3. Generate token baru
4. Copy token tersebut
5. Di file `.env`, update:
   ```env
   CONFLUENCE_USERNAME=admin@your-org.com  # Email atau username
   CONFLUENCE_CREDENTIAL=your-generated-token-here  # Token dari step 3
   ```

##### Jika Confluence Self-Hosted dengan SSO (SAML/OAuth):

Jika menggunakan Single Sign-On, basic auth tidak akan bekerja. Opsi:

1. **Disable SSO temporarily** untuk akun service account (not recommended untuk production)
2. **Gunakan App Password** (jika tersedia di instance Anda)
3. **Hubungi Confluence admin** untuk setup Personal Access Token

##### Alternatif: Gunakan Confluence REST API dengan OAuth

Jika token approach tidak bekerja, bisa setup OAuth 2.0:

```bash
# Pastikan confluence loader bisa akses API
# Dokumentasi: https://docs.atlassian.com/atlassian-python-api/
```

---

## Cara Menjalankan Scripts

### 1. **Test Confluence Connection** (Recommended dimulai dari sini)

```bash
python3 test_confluence.py
```

**Expected output:**
```
════════════════════════════════════════════════════════════
TEST CONFLUENCE CONNECTION
════════════════════════════════════════════════════════════
📍 Confluence URL: http://confluence.staytech.xyz
👤 Username: admin
📚 Space Key: MKB

⏳ Menghubungkan ke Confluence API...
✓ ConfluenceLoader berhasil diinisialisasi
📄 Mengambil documents dari space...
✓ Berhasil mengambil 5 halaman dari space 'MKB'

════════════════════════════════════════════════════════════
✅ TEST CONFLUENCE BERHASIL!
   Total documents: 5
════════════════════════════════════════════════════════════
```

**Jika gagal:** Lihat error message, cross-check credentials di .env

---

### 2. **Test Qdrant Connection**

```bash
python3 test_qdrant.py
```

**Expected output:**
```
════════════════════════════════════════════════════════════
TEST QDRANT CONNECTION
════════════════════════════════════════════════════════════
📍 Qdrant URL: http://qdrant.staytech.xyz
🔐 API Key: ****
📚 Collection: knowledge_base_ml

⏳ Menghubungkan ke Qdrant API...
✓ Qdrant client berhasil diinisialisasi
🔍 Mengecek server health...
✓ Server responsive. Total collections: 2

📋 Checking collection 'knowledge_base_ml'...
⚠️  Collection 'knowledge_base_ml' belum ada
🔧 Mencoba membuat collection 'knowledge_base_ml'...
✓ Collection 'knowledge_base_ml' berhasil dibuat
🗑️  Menghapus collection test (untuk cleanup)...
✓ Collection dihapus (collection akan dibuat ulang saat embedding)

════════════════════════════════════════════════════════════
✅ TEST QDRANT BERHASIL!
════════════════════════════════════════════════════════════
```

**Jika gagal:** Periksa QDRANT_URL dan QDRANT_API_KEY di .env

---

### 3. **Jalankan Full Embedding Pipeline**

```bash
python3 embed.py
```

**Proses:**
1. ✅ Validasi konfigurasi
2. ✅ Load documents dari Confluence
3. ✅ Chunk documents
4. ✅ Inisialisasi OpenAI & Qdrant
5. ✅ Embed documents dan upsert ke Qdrant

**Expected output:**
```
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 

════════════════════════════════════════════════════════════
VALIDASI KONFIGURASI
════════════════════════════════════════════════════════════
✓ CONFLUENCE_URL: http://confluence.staytech.xyz
✓ CONFLUENCE_USERNAME: admin
✓ CONFLUENCE_CREDENTIAL: ****
✓ OPENAI_API_KEY: ****
✓ QDRANT_URL: http://qdrant.staytech.xyz
✓ QDRANT_API_KEY: ****
✓ Semua konfigurasi valid

════════════════════════════════════════════════════════════
STEP 1: MENGHUBUNGKAN KE CONFLUENCE
════════════════════════════════════════════════════════════
...
✓ Berhasil menarik 5 halaman dari Confluence

════════════════════════════════════════════════════════════
✅ EMBEDDING SELESAI!
════════════════════════════════════════════════════════════
📊 Summary:
   - Documents dari Confluence: 5
   - Chunks setelah splitting: 42
   - Embedding model: text-embedding-3-small
   - Vector database: knowledge_base_ml @ http://qdrant.staytech.xyz

✓ Data siap digunakan untuk RAG!
════════════════════════════════════════════════════════════
```

**⚠️ Note:** 
- Proses ini akan memanggil OpenAI API, jadi akan ada biaya
- Waktu tergantung jumlah documents dan chunks
- Pastikan OpenAI API key memiliki quota yang cukup

---

### 4. **Query dengan RAG - Mode Interaktif**

Setelah embedding selesai, bisa query documents dan dapatkan answer dari OpenAI:

```bash
python3 retrieval.py
```

**Mode interaktif:**
```
════════════════════════════════════════════════════════════
INTERACTIVE RAG MODE
════════════════════════════════════════════════════════════
Ketik pertanyaan Anda (atau 'exit' untuk keluar):
════════════════════════════════════════════════════════════

❓ Pertanyaan Anda: Apa itu machine learning?

🔍 Query: Apa itu machine learning?
⏳ Retrieving documents dan generating answer...

════════════════════════════════════════════════════════════
📝 JAWABAN:
════════════════════════════════════════════════════════════
Machine learning adalah cabang dari artificial intelligence yang fokus 
pada pengembangan algoritma yang dapat belajar dari data...

════════════════════════════════════════════════════════════
📚 DOKUMEN RETRIEVE (4 dokumen):
════════════════════════════════════════════════════════════

[1] Introduction to ML
    Source: http://confluence.staytech.xyz/pages/...
    Preview: Machine learning is a method of data analysis that automates...

[2] ML Fundamentals
    Source: http://confluence.staytech.xyz/pages/...
    Preview: The fundamentals of machine learning include supervised learning...
```

---

### 5. **Query dengan RAG - Command Line Mode**

Atau jalankan single query langsung:

```bash
python3 retrieval.py "Apa itu deep learning?"
```

---

## Struktur Project

```
/root/ml-staging/
├── .env                      # Konfigurasi (jangan push ke repo)
├── .env.example              # Template konfigurasi
├── requirements.txt          # Python dependencies
├── embed.py                  # Main script untuk embedding
├── retrieval.py              # Script untuk RAG query
├── test_confluence.py        # Test script untuk Confluence
├── test_qdrant.py            # Test script untuk Qdrant
├── qdrant.yml                # Kubernetes deployment config untuk Qdrant
└── README.md                 # File ini
```

---

## Troubleshooting

### Issue 1: "HTTPError: Basic Authentication has been disabled"

**Penyebab:** Confluence tidak menggunakan basic auth

**Solusi:** 
- Lihat section [Troubleshooting Confluence Authentication](#troubleshooting-confluence-authentication)
- Gunakan Personal Access Token (PAT) di CONFLUENCE_CREDENTIAL

---

### Issue 2: "Unable to connect to Qdrant"

**Penyebab:** Qdrant server tidak accessible atau tidak running

**Verifikasi:**

```bash
# Test DNS resolution
nslookup qdrant.staytech.xyz

# Test HTTP connectivity
curl -v http://qdrant.staytech.xyz/health

# Check dari Kubernetes (jika deploy di K8s)
kubectl get pods -n qdrant  # Check if pod running
kubectl logs -n qdrant deployment/qdrant  # Check logs
kubectl port-forward -n qdrant svc/qdrant 6333:6333  # Port forward
```

---

### Issue 3: "OpenAI API Error: Rate limit exceeded"

**Penyebab:** OpenAI API rate limit tercapai

**Solusi:**
- Tunggu beberapa menit dan coba lagi
- Reduce CHUNK_SIZE di .env untuk batch size lebih kecil
- Upgrade OpenAI plan untuk higher limits

---

### Issue 4: "No documents found in space"

**Penyebab:** 
- Space tidak memiliki pages
- CONFLUENCE_SPACE_KEY tidak sesuai
- Pages ada tapi tidak bisa di-read

**Verifikasi:**

```bash
# Test manual connection dengan atlassian-python-api
python3 -c "
from atlassian import Confluence
confluence = Confluence(url='http://confluence.staytech.xyz', username='admin', password='****')
pages = confluence.get_all_pages_from_space('MKB')
print(f'Found {len(pages)} pages')
"
```

---

### Issue 5: "CHUNK_SIZE too large" atau memory error

**Penyebab:** Documents terlalu besar, menyebbkan chunks terlalu besar

**Solusi:**

Edit `.env`:
```env
CHUNK_SIZE=500  # Reduce dari 1000
CHUNK_OVERLAP=100  # Reduce dari 200
```

Kemudian jalankan ulang `embed.py`

---

### Issue 6: Dependency conflict - "langchain-community<0.1,>=0.0.25" atau "qdrant-client<2.0.0,>=1.10.1"

**Penyebab:** Versi paket tidak kompatibel dengan dependency requirements

**Solusi:**

Install ulang dependencies dengan versi yang benar:
```bash
# Jika menggunakan requirements.txt
pip install --upgrade -r requirements.txt

# Atau install manual dengan versi yang benar
pip install 'langchain-community>=0.0.25'
pip install 'qdrant-client>=1.10.1'
```

Verifikasi tidak ada konflik:
```bash
pip check
```

---

### Issue 7: "308 Permanent Redirect" saat connect ke Confluence

**Penyebab:** Confluence server mungkin redirect HTTP ke HTTPS, atau URL tidak benar

**Solusi - Option A: Gunakan HTTPS**

Update `.env`:
```env
CONFLUENCE_URL=https://confluence.staytech.xyz  # Ganti http ke https
```

**Solusi - Option B: Check actual URL Confluence**

Tanya ke Confluence admin atau cek:
```bash
# Test dengan follow redirect
curl -L -u admin:password https://confluence.staytech.xyz/rest/api/space/MKB

# Atau test HTTPS version
curl -u admin:password https://confluence.staytech.xyz/rest/api/
```

**Solusi - Option C: Disable SSL verification (dev only)**

Jika certificate issue, edit script atau set environment:
```bash
export PYTHONHTTPSVERIFY=0  # Tidak recommended untuk production
```

---

## Performance Optimization

### 1. Reduce Embedding Costs

```env
# Gunakan lebih kecil model (lebih murah)
EMBEDDING_MODEL=text-embedding-3-small  # Cheapest

# vs
EMBEDDING_MODEL=text-embedding-3-large  # More expensive, better quality
```

### 2. Optimize Chunking

```env
CHUNK_SIZE=800   # Balance antara too small vs too large
CHUNK_OVERLAP=100  # Small overlap untuk efficiency
```

### 3. Batch Processing (untuk large spaces)

Edit `embed.py` untuk implement batch processing jika ada timeout:

```python
# Split chunks into batches untuk avoid timeout
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    QdrantVectorStore.from_documents(...)
```

---

## Next Steps

1. ✅ Setup .env file dengan credentials
2. ✅ Jalankan test_confluence.py untuk verify koneksi
3. ✅ Jalankan test_qdrant.py untuk verify Qdrant
4. ✅ Jalankan embed.py untuk embedding documents
5. ✅ Query dengan retrieval.py

---

## Support & References

### Dokumentasi

- [LangChain Documentation](https://python.langchain.com/)
- [Qdrant Vector Database](https://qdrant.tech/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Atlassian Confluence Loader](https://docs.atlassian.com/atlassian-python-api/)

### Useful Commands

```bash
# Check .env file
cat .env

# Run tests dengan verbose logging
LOG_LEVEL=DEBUG python3 test_confluence.py

# Check Qdrant collections
python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='http://qdrant.staytech.xyz', api_key='****')
collections = client.get_collections()
for c in collections.collections:
    print(f'{c.name}: {client.get_collection(c.name).points_count} points')
"
```

---

**Last Updated:** March 2026
