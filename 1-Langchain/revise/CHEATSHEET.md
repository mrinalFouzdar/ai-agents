# LangChain — One-Page Revision Sheet

Everything from `1-Langchain/` on one page: concept + why it matters + minimal code shape.
Read this top to bottom in ~30–40 min as your primary study pass; use
[`PLAN.md`](./PLAN.md) for the timed hands-on tweaks per topic.

🔥 = what you'll actually see in real production codebases, not just tutorials.

---

## What's actually most-used in production (skim this first)

- **`prompt | model | parser` (LCEL pipe)** — the #1 pattern. Nearly every production
  LangChain chain is built this way, regardless of what's inside each piece.
- **`ChatPromptTemplate`** over plain `PromptTemplate` — production prompts almost always need
  a system message, so this is the default.
- **`RecursiveCharacterTextSplitter`** — the default splitter everywhere; the others are
  reached for only for a specific format (HTML, JSON).
- **`.as_retriever()`** — the standard adapter that lets *any* vector store (FAISS, Pinecone,
  Qdrant, pgvector...) plug into the same LCEL chain. The vector store swaps, this call doesn't.
- **`RunnableWithMessageHistory` + `session_id`** — the standard chatbot-memory pattern, but in
  production the in-memory `dict` store gets swapped for a persistent one (Redis/Postgres-backed).
- **`trim_messages`** — used in essentially every production chatbot; unbounded history = runaway token cost.
- **LangSmith** — the standard for tracing/debugging/evals once an app leaves prototyping.
- **PDF loaders (`PyPDFLoader`/`PyMuPDFLoader`)** — the most common ingestion source in
  real RAG apps (internal docs, reports, contracts).

---

## 1. Models, Prompts, Output Parsers
*Files: `1.1-openai/`, `1.2-ollama/`*

- **Model wrappers** are interchangeable — same `.invoke()` interface, different backend:
  `ChatOpenAI`, `ChatGroq`, `ChatOllama`.
- **Prompt templates** turn a template string + variables into the real prompt:
  `PromptTemplate` (plain text) vs `ChatPromptTemplate` (structured system/human/ai messages).
- **Output parsers** convert the raw model response object into something usable:
  `StrOutputParser()` → plain string.
- **LangSmith** auto-traces every call for debugging (set via env vars, no code change needed).

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])
chain = prompt | model | StrOutputParser()
chain.invoke({"question": "..."})
```

---

## 2. Data Ingestion (Document Loaders)
*Files: `3.2-DataIngestion/`*

- Every loader outputs a list of `Document` objects: `.page_content` (text) + `.metadata` (source info).
- Pick the loader by **source type**, not by what you're trying to do with it:

| Source | Loader |
|---|---|
| Plain text file | `TextLoader` |
| PDF | `PyPDFLoader` / `PyMuPDFLoader` |
| Webpage | `WebBaseLoader` |
| Wikipedia | `WikipediaLoader` |
| Arxiv paper | `ArxivLoader` |

```python
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

docs = PyPDFLoader("attention.pdf").load()
web_docs = WebBaseLoader("https://example.com/article").load()
all_docs = docs + web_docs          # multiple sources → one list → one retriever
```

---

## 3. Data Transformation (Text Splitters)
*Files: `3.3-Data Transformer/`*

- LLMs/embedding models have limited context → split long docs into chunks first.
- `RecursiveCharacterTextSplitter` is the default: tries separators in order (paragraph →
  sentence → word) so chunks rarely cut mid-word. Plain `CharacterTextSplitter` just splits on
  one fixed separator, less careful.
- `chunk_size` = max characters per chunk. `chunk_overlap` = characters shared between
  consecutive chunks, so context isn't lost right at a boundary.
- Specialized splitters exist too: `HTMLHeaderTextSplitter` (splits by HTML heading structure),
  `RecursiveJsonSplitter` (splits nested JSON).

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
```

---

## 4. Embeddings
*Files: `4-Embeddings/`*

- Turns text into a fixed-length numeric vector that captures meaning. Similar meaning → vectors
  that are close together (distance/cosine similarity).
- Options used in this repo:

| Embedding | Runs where | Needs key? |
|---|---|---|
| `OpenAIEmbeddings` | API | Yes (OpenAI) |
| `OllamaEmbeddings` | Local (via Ollama) | No |
| `HuggingFaceEmbeddings` | Local (sentence-transformers) | No |

```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🔥 API alternative — same interface, just swap the class:
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector = embeddings.embed_query("some text")   # -> list[float]
```

⚠️ Use the **same** embedding model for indexing and querying — mixing them makes the vectors
incomparable and retrieval returns garbage.

---

## 5. Vector Store — FAISS
*Files: `5-VectorStore/`*

- Stores embedded chunks; lets you search for the most semantically relevant ones to a query.
- `similarity_search(query, k=4)` → top-k `Document`s.
- `similarity_search_with_score(query)` → same, plus L2 distance (**lower = more similar** for FAISS).
- `.as_retriever()` → wraps the store as a `Retriever` so it plugs straight into an LCEL chain.

```python
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

results = db.similarity_search_with_score("my question")   # [(Document, score), ...]

# Persist so you don't re-embed every run (this is what 5-VectorStore/faiss_index/ is):
db.save_local("faiss_index")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
```

---

## 6. LCEL (LangChain Expression Language)
*Files: `6-LCEL/`*

- The `|` operator chains "Runnables" together — output of the left becomes input of the right:
  `prompt | model | parser`.
- Composable and swappable: change one piece without rewriting the rest.
- **LangServe** (`serve.py` + `client.py`) takes an LCEL chain and exposes it as a REST API —
  same chain, now callable over HTTP.

```python
chain = prompt | model | StrOutputParser()
chain.invoke({"question": "..."})   # same pattern as section 1, this IS an LCEL chain
```

### 🔥 Putting it together — a minimal RAG chain
This is the assembly step the assignments ask for: retriever output becomes the `context`
variable inside the prompt. The part worth memorizing.

```python
from langchain_core.runnables import RunnablePassthrough

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the context below. If it's not in the context, say you don't know.\n\n{context}"),
    ("user", "{question}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)
rag_chain.invoke("What is multi-head attention?")
```

---

## 7. Conversation History / Memory
*Files: `7-ConversationHistory/`*

- Chat models are stateless by default — every `.invoke()` call has no memory of the last one.
- `RunnableWithMessageHistory` wraps a chain + a history store, keyed by `session_id`, so each
  session's messages are tracked and replayed into future calls automatically.
- `trim_messages` caps how many past messages get sent, so history doesn't grow unbounded and
  blow past the model's context window.

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(chain, get_session_history)
chain_with_history.invoke(
    {"question": "..."},
    config={"configurable": {"session_id": "user-1"}}
)
```

---

## The full pipeline (how 1–7 fit together)

```
Load doc (2) → Split (3) → Embed (4) → Store in FAISS (5) → Retriever
                                                                 │
System prompt + retrieved context + chat history ──► LCEL chain (1, 6)
                                                                 │
                                          RunnableWithMessageHistory (7)
                                                                 │
                                                    Multi-turn RAG chatbot
```

This is what all 3 assignments in `PLAN.md` have you rebuild, with decreasing scaffolding.

## Concept-only (not yet coded)
- **Agents vs Agentic AI** (`AI agent vs Agentic AI/`, PDFs only): an "agent" is a single
  LLM-driven decision-maker that can call tools; "agentic AI" describes systems where multiple
  such steps/agents coordinate toward a goal. No hands-on practice yet — just be able to state
  the distinction in one sentence.
