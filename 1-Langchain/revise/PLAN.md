# LangChain Revision Plan

A condensed, practice-first revision of everything covered in `1-Langchain/` so far,
ending in 3 progressively independent assignments.

**Two phases, in order:**
- **Phase 1 (2 hours total)** — real revision, topic by topic. You watched videos and typed a
  little, so this isn't a "quick recall" skim — each topic below re-explains the concept in
  plain English so you don't need to re-watch anything, then has you run the real notebook and
  make one small hands-on tweak.
- **Phase 2** — the 3 assignments. Don't start these until Phase 1 is done — you'll build much
  faster and with less frustration once the concepts are fresh again.

## 1. What you've already covered (source of truth for revision)

| # | Topic | Where in repo | Core APIs / ideas to recall |
|---|-------|---------------|------------------------------|
| 1 | Models, Prompts, Output Parsers | `1.1-openai/`, `1.2-ollama/` | `ChatOpenAI`/`ChatGroq`, `ChatOllama`, `PromptTemplate`/`ChatPromptTemplate`, `StrOutputParser`, LangSmith tracing |
| 2 | Data Ingestion | `3.2-DataIngestion/` | `TextLoader`, `PyPDFLoader`/`PyMuPDFLoader`, `WebBaseLoader`, `ArxivLoader`, `WikipediaLoader` |
| 3 | Data Transformation (splitters) | `3.3-Data Transformer/` | `RecursiveCharacterTextSplitter`, `CharacterTextSplitter`, `HTMLHeaderTextSplitter`, `RecursiveJsonSplitter` — chunk_size / chunk_overlap tradeoffs |
| 4 | Embeddings | `4-Embeddings/` | `OpenAIEmbeddings`, `OllamaEmbeddings`, `HuggingFaceEmbeddings` (sentence-transformers) |
| 5 | Vector Stores | `5-VectorStore/` | `FAISS.from_documents`, `similarity_search`, `similarity_search_with_score`, `.as_retriever()` |
| 6 | LCEL (LangChain Expression Language) | `6-LCEL/` | `prompt \| model \| parser` pipe syntax, Runnables, LangServe (`serve.py` + `client.py`) |
| 7 | Conversation History / Memory | `7-ConversationHistory/` | `ChatMessageHistory`, `RunnableWithMessageHistory`, session-keyed history, `trim_messages` |
| 8 | Concepts: Agents vs Agentic AI | `AI agent vs Agentic AI/` (PDFs, no code yet) | Difference between a single AI agent and an agentic system — recall only, not coded yet |

These 7 coded topics form one pipeline end to end:

```
Load doc → Split into chunks → Embed chunks → Store in vector DB → Retrieve →
LCEL chain (prompt | model | parser) → wrap with message history for multi-turn chat
```

That's exactly what the 3 assignments below make you build, three times, with decreasing help.

## 1.5. Before you start (5 min — do this once, not inside the 2-hour budget)

Nothing kills a time-boxed revision session faster than a missing API key or an import error.
Check these first:

- **API keys** in the repo-root `.env`: whichever provider you'll use (`OPENAI_API_KEY` /
  `GROQ_API_KEY`), plus `LANGCHAIN_API_KEY` + `LANGCHAIN_TRACING_V2=true` if you want LangSmith traces.
- **Packages installed**: `pip install -r 1-Langchain/6-LCEL/requirements.txt` covers everything
  used across these topics (langchain, faiss-cpu, pypdf, sentence_transformers, langchain-groq, …).
- **Ollama** (only if you plan to use `ChatOllama`/`OllamaEmbeddings`): make sure the Ollama app
  is running and the model you reference is pulled.

If a provider isn't set up, don't burn revision time fixing it — switch to one that works
(Groq and HuggingFace embeddings need the least setup) and move on.

## 2. Phase 1 — Deep revision (2 hours total, do this before Phase 2)

A one-page companion to this section exists at
[`revise/CHEATSHEET.md`](./CHEATSHEET.md) — open it side by side, it has all 7 recaps + code
snippets on a single page so you're not hunting through notebooks to remember syntax.

For each topic: read the **Recap** (this replaces re-watching the video), **run** the real
notebook once top to bottom, do the **one tweak** (this is the practice part — don't skip it),
then honestly fill in the **confidence check**.

| # | Topic | Time |
|---|-------|------|
| 1 | Models / Prompts / Output Parsers | 15 min |
| 2 | Data Ingestion | 12 min |
| 3 | Data Transformation / Splitters | 12 min |
| 4 | Embeddings | 12 min |
| 5 | Vector Store (FAISS) | 15 min |
| 6 | LCEL | 20 min |
| 7 | Conversation History | 25 min |
| — | Buffer | 9 min |
| | **Total** | **120 min** |

---

### 1. Models / Prompts / Output Parsers (15 min)
**Recap:** `ChatOpenAI` / `ChatGroq` / `ChatOllama` are interchangeable model wrappers — same
interface, different provider. `PromptTemplate`/`ChatPromptTemplate` turn a template string +
variables into the actual prompt sent to the model. `StrOutputParser` converts the raw model
response object into plain text. LangSmith traces every call so you can debug what a chain
actually sent/received.
**Run:** `1.1-openai/1.1.1-GettingStarted.ipynb`, `1.1-openai/1.1.2-Simpleapp.ipynb`
*(skim if time: `1.2-ollama/1.2.1-Simpleapp.ipynb` — same app, local model)*
**Tweak:** In `1.1.2-Simpleapp.ipynb`, swap the model class for `ChatGroq` or `ChatOllama` and
compare the output to the original.
**Confidence (1–5):** ______ — if ≤2, note it as an Assignment-1 "lean on repo" topic.

### 2. Data Ingestion (12 min)
**Recap:** Loaders normalize any source into `Document` objects (`page_content` + `metadata`).
`TextLoader` for plain text, `PyPDFLoader`/`PyMuPDFLoader` for PDFs, `WebBaseLoader` for
webpages, `WikipediaLoader`/`ArxivLoader` for those APIs specifically.
**Run:** `3.2-DataIngestion/3.2-DataIngestion.ipynb`
**Tweak:** Load `speech.txt` with `TextLoader` and `attention.pdf` with `PyPDFLoader` side by
side; print `len(docs)` and one `.metadata` for each — notice how metadata differs by source type.
**Confidence (1–5):** ______

### 3. Data Transformation / Splitters (12 min)
**Recap:** LLMs and embedding models have context/input limits, so long docs get chunked first.
`RecursiveCharacterTextSplitter` tries separators hierarchically (paragraph → sentence → word)
so it rarely cuts mid-word — the default choice over plain `CharacterTextSplitter`.
`chunk_size` = max chars per chunk; `chunk_overlap` = shared text between neighboring chunks so
context isn't lost at boundaries.
**Run:** `3.3-Data Transformer/3.3-textsplitter.ipynb`, `3.4-CharacterTextsplitter.ipynb`
*(skim if time: `3.5-HTMLtextssplitter.ipynb`, `3.6-RecursiveJsonSplitter.ipynb` — format-specific splitters)*
**Tweak:** Split the same document at `chunk_size=500` vs `chunk_size=1000` (same overlap),
compare `len(chunks)` for each.
**Confidence (1–5):** ______

### 4. Embeddings (12 min)
**Recap:** Embeddings turn text into a fixed-length vector capturing meaning — similar meaning
→ vectors that land close together. `OpenAIEmbeddings`/`HuggingFaceEmbeddings` are API/local
options; `OllamaEmbeddings` runs locally via Ollama.
**Run:** `4-Embeddings/4.1-embedding.ipynb`, skim `4.3-huggingface.ipynb`
*(skim if time: `4.2-ollamaemnedding.ipynb` — local embeddings via Ollama)*
**Tweak:** Embed 2 similar sentences + 1 unrelated one; use FAISS (or manual cosine similarity)
to confirm the similar pair scores closer than the unrelated one.
**Confidence (1–5):** ______

### 5. Vector Store — FAISS (15 min)
**Recap:** FAISS stores embedded chunks for fast similarity search. `similarity_search(query)`
returns top-k relevant chunks; `similarity_search_with_score` also gives the distance (lower =
more similar for FAISS's L2 distance). `.as_retriever()` wraps the store so it plugs directly
into an LCEL chain.
**Run:** `5-VectorStore/5.1-Faiss.ipynb`
**Tweak:** Query the FAISS index with 2 different questions, compare the returned chunks + scores.
**Confidence (1–5):** ______

### 6. LCEL (20 min)
**Recap:** The `|` operator chains Runnables so input flows left to right:
`prompt | model | parser`. This gives you a composable, swappable pipeline instead of manual
glue code. LangServe (`serve.py` + `client.py`) takes that same LCEL chain and exposes it as a
REST endpoint.
**Run:** `6-LCEL/simplellmLCEL.ipynb`; skim `6-LCEL/serve.py` and `6-LCEL/client.py`
**Tweak:** In a scratch cell, build your own 3-step LCEL chain (new prompt topic, e.g. summarize
instead of translate) from scratch — don't copy the existing translation one verbatim.
**Confidence (1–5):** ______

### 7. Conversation History (25 min)
**Recap:** Chat models are stateless by default — `RunnableWithMessageHistory` plus a history
store keyed by `session_id` is what adds memory across turns. `trim_messages` keeps history
bounded so it doesn't overflow the model's context window as a conversation grows.
**Run:** `7-ConversationHistory/1-chatbots.ipynb`
**Tweak:** Ask a follow-up question that only makes sense with prior context (e.g. "what did I
just ask you?"), confirm it works. Then start a **second** `session_id` and confirm its history
is separate from the first.
**Confidence (1–5):** ______

---

**Phase 1 → Phase 2 gate:** Don't start Assignment 1 until all 7 rows above have a confidence
rating. Any topic you rated ≤2 isn't a reason to redo revision (protect the 2-hour budget) —
just make a mental note to lean on the repo more for that specific piece during Assignment 1.

## 3. The 3 assignments (Phase 2)

All three build the **same kind of thing** — a document-grounded conversational Q&A system —
so you feel the repetition, but the amount of help you're allowed decreases each time.

Create one subfolder per assignment as you start it:
`1-Langchain/revise/assignment-1/`, `assignment-2/`, `assignment-3/`.

---

### Assignment 1 — Guided (reference this codebase freely)

**Goal:** Build a RAG-based Q&A chatbot over `attention.pdf` (already in `3.2-DataIngestion/` and
`3.3-Data Transformer/`).

**Must include:**
1. Load the PDF (`PyPDFLoader` or `PyMuPDFLoader`)
2. Split it (`RecursiveCharacterTextSplitter`)
3. Embed chunks (your choice: OpenAI, Ollama, or HuggingFace)
4. Store + retrieve with FAISS
5. LCEL chain: `prompt | model | StrOutputParser()`
6. Wrap the chain with `RunnableWithMessageHistory` so it remembers earlier turns in the same session

**Help allowed:** Freely open and adapt code from `3.2`, `3.3`, `4`, `5`, `6`, `7`. Copying and
adjusting is fine — the goal here is re-assembling the pieces correctly, not memorization.

**Deliverable:** `revise/assignment-1/rag_chatbot.ipynb`

---

### Assignment 2 — Semi-independent (help only when truly stuck)

**Goal:** Build a conversational RAG system that's a step harder than Assignment 1:

1. Ingest from **two different sources** combined (e.g. `WebBaseLoader` on a blog/article **plus**
   a PDF) into one retriever
2. Use a **different embedding model** than you used in Assignment 1
3. Add `trim_messages` so history doesn't grow unbounded
4. Support **multiple concurrent sessions** (two different `session_id`s with separate histories, prove they don't leak into each other)

**Help allowed:** Try from memory/LangChain docs first. Only open this repo's notebooks if
stuck for more than ~10–15 minutes on one specific piece — and when you do, leave a one-line
comment in your notebook noting what you needed help with (this tells you what to drill later).

**Deliverable:** `revise/assignment-2/conversational_rag.ipynb` (+ a short note of what you needed help with)

---

### Assignment 3 — Final, independent (no repo reference)

**Goal:** From a blank file, no peeking at any notebook in this repo, build a clean
**Conversational Document Q&A app as a `.py` script** (not a notebook — treat it like a real
deliverable):

1. Config via `.env` (reuse the pattern from `6-LCEL`, don't copy the code)
2. Ingest a document of your choice
3. Split, embed, store in FAISS, retrieve
4. LCEL RAG chain with a proper system prompt (instruct the model to answer only from context)
5. `RunnableWithMessageHistory` keyed by `session_id`, with trimming
6. A simple CLI loop: user types a question, gets an answer, history persists until they quit
7. **Stretch goal:** serve it with LangServe (`serve.py` + `client.py`), mirroring `6-LCEL/`

**Constraint:** Time-box yourself to 60–90 minutes, like an interview/exam. If you truly can't
proceed, stop, write down exactly where you got stuck, and only then look it up — that gap is
your real revision signal, more useful than the assignment itself.

**Deliverable:** `revise/assignment-3/app.py` (+ `serve.py`/`client.py` if you attempt the stretch goal)

---

## 4. Self-check rubric (use for all 3 assignments)

| Check | A1 | A2 | A3 |
|---|---|---|---|
| Loads document(s) correctly | ☐ | ☐ | ☐ |
| Splits with sensible chunk_size/overlap | ☐ | ☐ | ☐ |
| Embeds + stores in FAISS without error | ☐ | ☐ | ☐ |
| Retriever returns relevant chunks (manually check 1 query) | ☐ | ☐ | ☐ |
| LCEL chain runs end to end | ☐ | ☐ | ☐ |
| Multi-turn memory actually works (ask a follow-up that needs prior context) | ☐ | ☐ | ☐ |
| Runs cleanly top to bottom with no leftover errors | ☐ | ☐ | ☐ |

## 5. Suggested schedule (compressed)

| Session | Length | Focus |
|---|---|---|
| 0 | **2 hrs** | **Phase 1 — deep revision of all 7 topics** (Section 2 + `CHEATSHEET.md`). Can split into 2×1hr back to back if needed, but finish it before Session 1. |
| 1 | ~1.5–2 hrs | Assignment 1 (guided) |
| 2 | ~1.5–2 hrs | Finish Assignment 1 if needed, diff your solution against the original notebooks |
| 3 | ~1.5–2 hrs | Assignment 2 (semi-independent) |
| 4 | ~1–1.5 hrs | Assignment 3 (timed, fully independent) + fill in the self-check rubric honestly |

Session 0 (Phase 1) should ideally happen in one sitting so the concepts are still fresh when
Assignment 1 starts right after. Sessions 1–4 (Phase 2) can then spread across the following
days — just don't let more than a day gap between sessions or things decay again.
