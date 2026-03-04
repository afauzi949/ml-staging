# AGENTS.md

## Cursor Cloud specific instructions

This is a Python-based ML Knowledge Base RAG pipeline that embeds Confluence wiki documents into a Qdrant vector database using OpenAI embeddings, then provides RAG-based Q&A via `retrieval.py`. See `README.md` for full documentation.

### Key scripts

| Script | Purpose |
|---|---|
| `embed.py` | Full embedding pipeline: Confluence -> chunk -> embed -> Qdrant |
| `retrieval.py` | RAG query interface (interactive or CLI) |
| `test_confluence.py` | Test Confluence connectivity |
| `test_qdrant.py` | Test Qdrant connectivity |
| `test_qdrant_upsert.py` | Test Qdrant write operations |
| `check_qdrant.py` | Qdrant admin utility (list/inspect/delete collections) |

### Running scripts

Activate the virtualenv before running any script:

```bash
source /workspace/.venv/bin/activate
python3 <script>.py
```

### Required environment variables

All scripts expect the following env vars (provided via secrets or `.env`): `CONFLUENCE_URL`, `CONFLUENCE_USERNAME`, `CONFLUENCE_CREDENTIAL`, `CONFLUENCE_SPACE_KEY`, `OPENAI_API_KEY`, `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_HTTPS`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`.

### Known gotchas

- `beautifulsoup4` and `lxml` are required at runtime by ConfluenceLoader but are not listed in `requirements.txt`. The update script installs them.
- Scripts that use `QDRANT_HOST`/`QDRANT_PORT`/`QDRANT_HTTPS` params (`embed.py`, `test_qdrant.py`, `check_qdrant.py`) work correctly with the remote Qdrant. Scripts that use `QDRANT_URL` (`retrieval.py`, `test_qdrant_upsert.py`) may fail with TLS connection reset against the remote HTTPS Qdrant endpoint; use a local Docker Qdrant (`docker run -d -p 6333:6333 qdrant/qdrant:latest`) for those.
- Some test scripts (`test_qdrant_upsert.py`, `check_qdrant.py`) use deprecated qdrant-client APIs (`client.search`, `info.vectors_config`). These are pre-existing compatibility issues with newer qdrant-client versions; the core LangChain-based pipeline handles this internally.
- There is no linting or automated test framework configured in this repo. Validation is done by running the scripts directly.
- The file `=1.3.0` in the repo root is an accidental artifact from a bad pip install command; it is not a project file.
