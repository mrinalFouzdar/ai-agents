import os
import json
import bs4
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import trim_messages
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")

current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, "faiss_index")

# ==============================================================================
# 1. Embeddings Model (Assignment 2 Req 3: Different model than Assignment 1)
# ==============================================================================
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# ==============================================================================
# 2. Vector Store Setup (Assignment 2 Req 4: save_local & load_local)
# ==============================================================================
if os.path.exists(index_path):
    print(f"Loading existing FAISS index from: {index_path}")
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
else:
    print("No existing index found. Starting multi-source ingestion...")

    # --- Source 1: Web Ingestion & Source-Specific Chunking ---
    print("1. Loading Web content (TechFerry Careers)...")
    web_loader = WebBaseLoader(
        web_path=('https://www.techferry.com/careers.html',),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer('div', attrs={'class': lambda c: c and 'uk-card' in c}))
    )
    web_docs = web_loader.load()

    # Web chunking strategy (tuned for HTML paragraph length)
    web_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    web_chunks = web_splitter.split_documents(web_docs)
    for chunk in web_chunks:
        chunk.metadata["source_type"] = "web"
        chunk.metadata["source_title"] = "TechFerry Careers"

    # --- Source 2: PDF Ingestion & Source-Specific Chunking ---
    print("2. Loading PDF content (Mrinal Fouzdar Resume)...")
    pdf_file = os.path.join(current_dir, "Mrinal_Fouzdar.pdf")
    pdf_loader = PyPDFLoader(pdf_file)
    pdf_docs = pdf_loader.load()

    # PDF chunking strategy (tuned for dense document text)
    pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    pdf_chunks = pdf_splitter.split_documents(pdf_docs)
    for chunk in pdf_chunks:
        chunk.metadata["source_type"] = "pdf"
        chunk.metadata["source_title"] = "Mrinal Resume"

    # --- Combine Enriched Chunks & Build Index ---
    all_chunks = web_chunks + pdf_chunks
    print(f"Total chunks unified: {len(all_chunks)} (Web: {len(web_chunks)}, PDF: {len(pdf_chunks)})")

    print("3. Generating embeddings & building FAISS vector index...")
    db = FAISS.from_documents(all_chunks, embeddings)

    # Save locally to avoid re-embedding on future runs
    db.save_local(index_path)
    print(f"FAISS index successfully saved to: {index_path}")

# ==============================================================================
# 3. Unified Retriever with LLM Re-Ranking
# ==============================================================================
# Step 1: Retrieve top-10 candidate chunks from the unified FAISS vector store
retriever = db.as_retriever(search_kwargs={"k": 10})
print("Retriever ready for querying!")

llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=groq_api_key)

# Re-ranker Prompt: Evaluates and selects the top-3 most relevant documents
reranker_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert document relevance ranker.\n"
        "Given a user question and a numbered list of retrieved documents, select the top 3 "
        "most relevant document IDs.\n"
        "Return ONLY a valid JSON array of strings containing the selected IDs, e.g. [\"doc_0\", \"doc_2\"]. "
        "Do not include explanations or markdown fences."
    )),
    ("human", "Question: {question}\n\nDocuments:\n{docs}"),
])

reranker_chain = reranker_prompt | llm | StrOutputParser()

def rerank_and_format(data):
    """Retrieves top candidates, re-ranks with LLM, and formats the best chunks into text."""
    query = data["question"]
    docs = retriever.invoke(query)

    # Map candidate docs
    doc_map = {f"doc_{i}": doc for i, doc in enumerate(docs)}
    doc_text = "\n\n".join(f"[{doc_id}]: {doc.page_content}" for doc_id, doc in doc_map.items())

    try:
        raw_result = reranker_chain.invoke({"docs": doc_text, "question": query})
        cleaned_json = raw_result.strip().strip("`").replace("json", "").strip()
        selected_ids = json.loads(cleaned_json)
        top_docs = [doc_map[did] for did in selected_ids if did in doc_map]
        if not top_docs:
            top_docs = docs[:3]
    except Exception:
        # Fallback to top-3 closest embeddings if re-ranker fails
        top_docs = docs[:3]

    return "\n\n".join(f"[Source: {doc.metadata.get('source_title', 'Document')}]\n{doc.page_content}" for doc in top_docs)


# ==============================================================================
# 4. Prompt, Trimmer & LCEL Chain Assembly
# ==============================================================================
system_prompt = (
    "You are a helpful assistant for question-answering tasks.\n"
    "Use ONLY the following retrieved context (which includes candidate background "
    "and company career details) to answer the user's question.\n"
    "If the answer is not found in the context, strictly say 'I don't know'. "
    "Do not extrapolate or assume.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# Trimmer: Caps chat history at 500 tokens to prevent unbounded growth
trimmer = trim_messages(
    max_tokens=500,
    strategy="last",
    token_counter=llm,
    include_system=False,
    start_on="human",
)

rag_chain = (
    {
        "context": rerank_and_format,
        "chat_history": (lambda x: x.get("chat_history", [])) | trimmer,
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ==============================================================================
# 5. Multi-Session Memory Management
# ==============================================================================
session_store = {}

def get_session_history(session_id: str):
    """Returns or creates a ChatMessageHistory for the specified session_id."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


# ==============================================================================
# 6. Interactive CLI Chat Loop with Session Switching
# ==============================================================================
if __name__ == "__main__":
    print("\n=== Conversational RAG with Re-Ranking CLI Started ===")
    print("Commands: '/session <id>' to switch sessions | 'quit' to exit")

    current_session = "default"

    while True:
        try:
            user_input = input(f"\n[{current_session}] You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            # Handle dynamic session switching (/session <id>)
            if user_input.startswith("/session "):
                new_session = user_input.split(" ", 1)[1].strip()
                if new_session:
                    current_session = new_session
                    print(f"Switched to session: '{current_session}'")
                    continue

            # Query with the active session
            config = {"configurable": {"session_id": current_session}}
            response = conversational_rag.invoke({"question": user_input}, config=config)
            print(f"\n[{current_session}] Assistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")