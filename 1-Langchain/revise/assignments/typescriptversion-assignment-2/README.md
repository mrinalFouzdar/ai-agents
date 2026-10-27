# Assignment 2 - Conversational RAG with Multi-Source Ingestion & Re-Ranking (TypeScript / Node.js)

A 1:1 TypeScript / Node.js port of `assignment-2/conversational_rag.py`.

## Features
- **Multi-Source Ingestion**:
  - **Source 1 (Web)**: TechFerry Careers page ingested via `CheerioWebBaseLoader` (filtered by `.uk-card`), split with `chunkSize: 600, chunkOverlap: 100`.
  - **Source 2 (PDF)**: `Mrinal_Fouzdar.pdf` loaded via `PDFLoader`, split with `chunkSize: 400, chunkOverlap: 80`.
- **Embeddings**: `OllamaEmbeddings` using `mxbai-embed-large`.
- **Vector Store**: `FaissStore` with local persistence (`save` & `load` to/from `./faiss_index`).
- **Two-Stage Retrieval & Re-ranking**: Top-10 candidates retrieved from FAISS, re-ranked down to top-3 using Groq LLM (`openai/gpt-oss-20b`).
- **Memory & Trimming**: Multi-session conversation memory with `trimMessages` (500-token cap).
- **Interactive CLI**: Dynamic session switching with `/session <id>` and exit with `quit` or `exit`.

## Prerequisites
- Node.js (v18+)
- Ollama running locally with `mxbai-embed-large` pulled:
  ```bash
  ollama pull mxbai-embed-large
  ```
- `GROQ_API_KEY` set in `.env`

## Running the Application
```bash
# Install dependencies
npm install

# Start the interactive CLI
npm start
# or
npx tsx conversational_rag.ts
```

## CLI Commands
- `/session <name>` - Switch to or create a named conversation session (e.g. `/session techferry`)
- `exit` or `quit` - Terminate the session
