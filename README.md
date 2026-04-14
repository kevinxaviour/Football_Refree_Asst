# FIFA Referee Assistant API

Production-oriented RAG API for IFAB Laws of the Game with:
- LangChain orchestration
- Contextual embeddings + MMR retrieval
- Structured outputs (Pydantic schema)
- LLM fallback chain: OpenAI (primary) → Groq → OpenRouter
- Trace metadata (`trace_id`, provider used) in each response

## Architecture & Approaches

### 1. Document Ingestion Pipeline
```
PDF Download → Layout-Aware Extraction → Semantic Chunking → Embedding → Vector Storage
```
- **PDF Extraction**: Uses `pdfplumber` for layout-aware text extraction from IFAB Laws PDF
- **Law Detection**: Regex-based detection of Law numbers (1-17) to tag pages with metadata
- **Text Cleaning**: Removes PDF artifacts (page numbers, headers/footers, excessive whitespace)

### 2. Semantic Chunking Strategy
Unlike traditional character-count chunking, this approach:
- Splits text into sentences using regex (handles abbreviations like "e.g.", "Law 12.")
- Embeds each sentence with the same model used for retrieval
- Computes cosine similarity between consecutive sentences
- Inserts chunk breaks where similarity drops below threshold (0.70)
- Caps chunks at 12 sentences max to avoid overly long segments
- **Result**: Logically coherent chunks that keep related rules together

### 3. Embedding & Vector Storage
- **Primary**: `intfloat/e5-small-v2` (local, fast) or OpenAI `text-embedding-3-small`
- **Vector Store**: ChromaDB with PersistentClient for local persistence
- **Indexing**: Batch processing (100 chunks/batch) with cosine similarity space
- **Query Prefixing**: Uses `passage:` and `query:` prefixes per E5 model convention

### 4. Retrieval with MMR Reranking
```
Dense Search (top_k * 6 candidates) → MMR Reranking → Final top_k results
```
- **MMR (Maximum Marginal Relevance)**: Balances relevance vs. diversity
- **Lambda Param**: 0.72 (favors relevance slightly over diversity)
- **Law Filtering**: Optional filter to restrict search to specific Law

### 5. LLM Fallback Chain
```
OpenAI GPT-4o-mini → Groq Llama-3.3-70B → OpenRouter (any model)
```
- Sequential fallback with error handling
- Each provider uses `temperature=0.1` for consistent outputs
- OpenRouter supports custom headers for site identification
- LangSmith tracing captures all invocations with metadata

### 6. Structured Outputs
Pydantic schemas ensure consistent, typed responses:
- `situation`: Description of the refereeing scenario
- `applicable_laws`: List of relevant Laws
- `ruling`: The decision
- `explanation`: Detailed reasoning
- `key_exceptions`: Important caveats
- `citations`: Law, page number, and supporting quote
- `confidence`: high/medium/low assessment

## API Response Shape

`POST /query` returns:
- `answer` (readable text)
- `structured_output` with:
  - `situation`
  - `applicable_laws`
  - `ruling`
  - `explanation`
  - `key_exceptions`
  - `citations[]` (`law`, `page_num`, `quote`)
  - `confidence`
- `sources[]` retrieved chunks
- `provider_used` (`openai`, `groq`, `openrouter`)
- `trace_id`

## Environment Variables

Required (at least one provider key):
- `OPENAI_API_KEY` (primary)
- `GROQ_API_KEY` (fallback)
- `OPENROUTER_API_KEY` (fallback)

Optional:
- `OPENAI_MODEL` (default `gpt-4o-mini`)
- `GROQ_MODEL` (default `llama-3.3-70b-versatile`)
- `OPENROUTER_MODEL` (default `openai/gpt-4o-mini`)
- `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`)
- `OPENROUTER_SITE_URL`
- `OPENROUTER_APP_NAME`
- `EMBEDDING_PROVIDER` (`openai` or `local`, default `openai`)
- `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`)
- `EMBEDDING_MODEL` (default `intfloat/e5-small-v2`)
- `CHROMA_PERSIST_DIR` (default `./vectorstore`)
- `CHROMA_COLLECTION_NAME` (default `fifa_laws`)
- `CORS_ALLOW_ORIGINS` (comma-separated)

LangChain tracing (optional but recommended):
- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_PROJECT=refreeassisant`

## Local Run

```bash
pip install -r requirements.txt
python scripts/build_index.py
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Render Deploy (API)

1. Push this repo to GitHub.
2. In Render, create a new **Web Service** from repo.
3. Build command:
   - `pip install -r requirements.txt`
4. Start command:
   - `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Set environment variables (provider keys and optional LangSmith keys).
6. Run one indexing job by calling `POST /ingest` or run `python scripts/build_index.py` in Render shell.

## Query Workflow

When a user submits a question via `POST /query`:
1. **Embedding**: Query is embedded using the same model
2. **Retrieval**: ChromaDB returns top 48 candidates (8 × 6)
3. **MMR Reranking**: Diversity penalty applied, final 8 selected
4. **Context Building**: Chunks formatted with Law/page/score metadata
5. **LLM Invocation**: LangChain routes to first available provider
6. **Structured Parsing**: Response parsed into Pydantic schema
7. **Response**: Combined answer + structured_output + sources + trace metadata

## Example Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can a goalkeeper handle a deliberate back-pass?",
    "top_k": 8
  }'
```
