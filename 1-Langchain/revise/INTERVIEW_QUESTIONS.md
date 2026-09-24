# LangChain — Likely Interview Questions

Q&A format, grouped by the same 7 topics as [`CHEATSHEET.md`](./CHEATSHEET.md). Answers are kept
short and plain — enough to answer confidently, not a full lecture. If you can answer these
without looking, you're in good shape.

---

## General / Conceptual (almost always asked first)

**Q: What is LangChain, and why not just call the LLM API directly?**
A: LangChain is a framework that standardizes common LLM-app pieces — prompts, model calls,
parsing output, retrieval, memory — behind consistent interfaces, so you can swap providers or
add pieces (retrieval, memory) without rewriting glue code each time.

**Q: What is RAG (Retrieval-Augmented Generation) and why use it?**
A: Instead of relying only on what the model learned during training, RAG retrieves relevant
chunks from your own data at query time and feeds them into the prompt, so answers are grounded
in current, specific, private information the model was never trained on.

**Q: Walk me through a full RAG pipeline end to end.**
A: Load documents → split into chunks → embed chunks → store in a vector store → at query time,
embed the question and retrieve the most similar chunks → pass question + retrieved chunks into
an LCEL chain (`prompt | model | parser`) → optionally wrap with message history for multi-turn chat.

**Q: What's the difference between a Chain and an Agent?**
A: A chain runs a fixed sequence of steps you defined — same path every time. An agent uses the
LLM to *decide* at runtime which tool to call and in what order, so the path varies by input.
Chains are predictable and cheap; agents are flexible but slower and harder to debug.

**Q: What's the difference between an "AI agent" and "agentic AI"?**
A: An agent is a single LLM-driven component that can decide which tool to call and act on it.
"Agentic AI" describes a system where multiple such decision-making steps (or agents) chain
together and coordinate toward a larger goal, rather than one single call-and-respond.

---

## 1. Models, Prompts, Output Parsers

**Q: Difference between `PromptTemplate` and `ChatPromptTemplate`?**
A: `PromptTemplate` builds a single plain-text prompt. `ChatPromptTemplate` builds a structured
list of role-tagged messages (system/human/ai) — what chat-based models actually expect.

**Q: Why use an output parser instead of just using the raw model response?**
A: The raw response is a message object with metadata; a parser (e.g. `StrOutputParser`)
extracts just the piece you need (plain text, or a structured object), so downstream code
doesn't have to know about the model's response format.

**Q: What's the point of a system message?**
A: It sets persistent instructions/behavior for the model (role, tone, constraints) that apply
to the whole conversation, separate from the user's actual question.

**Q: What is LangSmith used for?**
A: Tracing and debugging — it logs every step of a chain's execution (prompts sent, responses
received, latency, token usage) so you can inspect what actually happened, not just the final output.

---

## 2. Data Ingestion

**Q: What does a Document Loader actually return?**
A: A list of `Document` objects, each with `page_content` (the text) and `metadata`
(source info like filename, page number, or URL).

**Q: How do you choose which loader to use?**
A: By source type, not by intent — PDF → `PyPDFLoader`/`PyMuPDFLoader`, webpage →
`WebBaseLoader`, plain text → `TextLoader`, and so on.

**Q: Why does metadata matter?**
A: It lets you trace an answer back to its source (e.g. "page 4 of report.pdf"), and can be used
to filter retrieval later (e.g. only search a specific document).

---

## 3. Text Splitters

**Q: Why split documents before embedding them?**
A: Embedding models and LLM context windows have size limits, and retrieval is more precise
over small, focused chunks than one giant document.

**Q: Why is `RecursiveCharacterTextSplitter` usually preferred?**
A: It tries splitting on a hierarchy of separators (paragraph, then sentence, then word), so it
tends to keep semantically meaningful text together instead of cutting mid-sentence like a
naive fixed-character split.

**Q: What's the tradeoff between a large and a small `chunk_size`?**
A: Small chunks → more precise retrieval but less surrounding context per chunk. Large chunks →
more context per chunk but retrieval gets noisier (a chunk may contain the answer buried among
irrelevant text).

**Q: What does `chunk_overlap` solve?**
A: It repeats a bit of text between consecutive chunks so information near a chunk boundary
isn't lost or split awkwardly across two chunks.

---

## 4. Embeddings

**Q: What is an embedding, in plain terms?**
A: A list of numbers (a vector) that represents the meaning of a piece of text, positioned so
that texts with similar meaning end up close together in that vector space.

**Q: How do you compare two embeddings for similarity?**
A: With a distance/similarity metric — commonly cosine similarity or L2 (Euclidean) distance;
closer/higher-similarity = more semantically related.

**Q: API-based (OpenAI) vs local (HuggingFace) embeddings — tradeoffs?**
A: API-based: usually higher quality, no local compute needed, but costs money and sends data
externally. Local: free and private, but needs local compute and setup, and may be lower quality
depending on the model.

**Q: Can you use one embedding model to index documents and a different one to embed queries?**
A: No. Each model produces vectors in its own space, so distances between them are meaningless
and retrieval returns junk. If you change the embedding model, you must re-embed and rebuild the
whole index.

---

## 5. Vector Store (FAISS)

**Q: What is a vector store, and why not just use a normal SQL database?**
A: A vector store is built to do fast similarity search over high-dimensional vectors — finding
the "nearest" vectors to a query — which relational databases aren't optimized for.

**Q: What does `similarity_search_with_score` give you that plain `similarity_search` doesn't?**
A: The distance/similarity score for each result, so you can judge how relevant a match actually
is, or filter out weak matches below a threshold.

**Q: What does `.as_retriever()` do?**
A: Wraps the vector store in a standard `Retriever` interface so it plugs directly into an LCEL
chain — the rest of the chain doesn't need to know which vector store is behind it.

**Q: How do you avoid re-embedding your documents every time the app restarts?**
A: Persist the index to disk with `db.save_local("faiss_index")` and reload it with
`FAISS.load_local(...)` on startup — embedding is the slow, expensive step, so you do it once
and reuse the saved index.

**Q: FAISS vs a managed vector DB (Pinecone, Qdrant, pgvector)?**
A: FAISS is a local, in-process library — fast and simple for prototyping or moderate scale, but
you handle persistence/scaling yourself. Managed vector DBs add persistence, horizontal scaling,
and metadata filtering out of the box, at the cost of an external service dependency.

---

## 6. LCEL

**Q: What does the `|` operator do in LangChain?**
A: It composes Runnables into a pipeline — the output of the step on the left becomes the input
to the step on the right (`prompt | model | parser`).

**Q: What is a Runnable?**
A: The common interface every LCEL piece implements — prompts, models, parsers, retrievers, even
plain functions. Because they all share methods like `.invoke()`, `.stream()`, and `.batch()`,
any of them can be piped into any other.

**Q: How is LCEL different from the older `LLMChain` style?**
A: `LLMChain` and friends were pre-built classes you configured; LCEL is composition — you build
the chain yourself from Runnables with `|`. LCEL gives streaming, batching, and async for free,
and is the current standard, with the legacy chain classes deprecated.

**Q: Why use LCEL instead of writing your own manual pipeline code?**
A: It's declarative and composable — steps are easy to swap, chains support streaming/batching/
async automatically, and it integrates directly with tracing (LangSmith) and serving (LangServe).

**Q: What problem does LangServe solve?**
A: It takes an existing LCEL chain and exposes it as a REST API with minimal extra code, so the
same chain you built and tested locally becomes callable over HTTP.

---

## 7. Conversation History / Memory

**Q: Why are LLMs stateless, and how do you add memory?**
A: Each API call is independent by default — the model has no memory of past calls.
`RunnableWithMessageHistory` adds memory by storing and replaying prior messages for a given
`session_id` on each new call.

**Q: What is `session_id` for?**
A: It's the key that separates one conversation's history from another's, so multiple users (or
chat sessions) don't see each other's messages.

**Q: Why trim conversation history instead of keeping all of it?**
A: Context windows and cost are both proportional to how much history you send — unbounded
history eventually overflows the model's limit and gets expensive; `trim_messages` caps it.

**Q: How would you persist chat history across server restarts?**
A: Swap the in-memory dict-based history store for a persistent backend-backed one (e.g.
Redis- or database-backed chat message history), keyed the same way by `session_id`.

---

## RAG scenario questions (common follow-ups)

**Q: How do you stop a RAG chatbot from hallucinating answers not in the retrieved context?**
A: Instruct it explicitly in the system prompt to only answer from the provided context and say
it doesn't know otherwise; optionally check retrieval scores and refuse to answer if nothing
relevant was retrieved.

**Q: What would you do if retrieval keeps returning irrelevant chunks?**
A: Check chunk_size/overlap (too big/small), try a different embedding model, inspect scores
via `similarity_search_with_score`, or improve the underlying document quality/structure.

**Q: How do you evaluate whether a RAG pipeline's retrieval is any good?**
A: Build a small set of test questions with known correct source chunks, then check whether
retrieval actually surfaces those chunks (precision/recall-style manual or automated eval) —
this is also a core use case for LangSmith's eval tooling.
