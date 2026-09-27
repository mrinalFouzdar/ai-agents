"""
Conversational RAG Chatbot with History
---------------------------------------
An end-to-end Retrieval-Augmented Generation (RAG) system with multi-turn memory:
1. Document Ingestion: Loads PDF content.
2. Chunking: Splits large text into semantically coherent overlapping chunks.
3. Vector Storage: Generates embeddings with Ollama and indexes chunks with FAISS.
4. Retrieval: Queries top-k relevant chunks based on user questions.
5. Prompt Engineering: Enforces strict context grounding and integrates chat history.
6. LCEL Chain: Connects retriever, prompt, LLM (OpenAI GPT-4o), and output parser.
7. Memory Management: Stores and retrieves conversation history per session.
8. CLI Loop: Interactive terminal chat interface with graceful exit options.
"""

import os
from dotenv import load_dotenv

# LangChain - Document loaders & Splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain - Embeddings & Vector Store
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# LangChain - Models
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# LangChain - Prompts, Runnables & Parsers
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# ==============================================================================
# STEP 1: Environment & API Keys Configuration
# ==============================================================================
# Load environment variables (OPENAI_API_KEY, GROQ_API_KEY, etc.) from .env file
load_dotenv()


# ==============================================================================
# STEP 2: Document Ingestion (Load PDF)
# ==============================================================================
# Load the target PDF document into LangChain Document objects
pdf_file = "Mrinal_Fouzdar.pdf"
loader = PyPDFLoader(pdf_file)
pdf_docs = loader.load()

# Optional debug check:
# print(f"Loaded {len(pdf_docs)} page(s) from {pdf_file}")


# ==============================================================================
# STEP 3: Text Splitting & Chunking
# ==============================================================================
# Break down full pages into smaller chunks to fit LLM context limits and improve retrieval precision
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Maximum character length per chunk
    chunk_overlap=100      # Overlapping characters between adjacent chunks to preserve context
)
texts = text_splitter.split_documents(pdf_docs)

# Optional debug check:
# print(f"Split document into {len(texts)} chunks.")


# ==============================================================================
# STEP 4: Embeddings & Vector Store Setup
# ==============================================================================
# 4.1. Initialize local embedding model via Ollama
embeddings = OllamaEmbeddings(model="all-minilm")

# 4.2. Index chunked documents in an in-memory FAISS vector database
db = FAISS.from_documents(texts, embeddings)

# 4.3. Convert vector store into a retriever to fetch the top 2 closest matching chunks (k=2)
retriever = db.as_retriever(search_kwargs={"k": 2})
retriver = retriever  # Alias for backward compatibility


# ==============================================================================
# STEP 5: Language Model Configuration
# ==============================================================================
# Primary LLM: OpenAI GPT-4o
llm = ChatOpenAI(model="gpt-4o")

# Alternative: Fast inference via Groq
# llm = ChatGroq(model="llama-3.1-8b-instant")


# ==============================================================================
# STEP 6: Prompt Template (Context + Chat History)
# ==============================================================================
# System prompt instructs the model to answer ONLY from the retrieved context
system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use ONLY the following pieces of retrieved context to answer the question. "
    "If the answer is not contained in the context, strictly say 'I don't know'. "
    "Do not make up facts or use outside knowledge.\n\n"
    "Context:\n{context}"
)

# Multi-message prompt structure:
# 1. System instructions with grounded context
# 2. Placeholder for prior conversation messages
# 3. New question from the user
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
])


# ==============================================================================
# STEP 7: Helper Functions & LCEL Chain Assembly
# ==============================================================================
def format_docs(docs):
    """Formats retrieved document chunks into a single concatenated text string."""
    return "\n\n".join(document.page_content for document in docs)


# Build RAG Chain using LangChain Expression Language (LCEL):
# Input dict -> Extract question & history -> Retrieve context -> Prompt -> LLM -> String Output
rag_chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": lambda x: x["question"],
        "chat_history": lambda x: x.get("chat_history", []),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ==============================================================================
# STEP 8: Conversational Memory & Session Management
# ==============================================================================
# In-memory dictionary for storing chat histories keyed by session_id
store = {}

def get_session_history(session_id: str):
    """Fetches the ChatMessageHistory for a given session, or creates a new one if it doesn't exist."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Wrap the base RAG chain with session-aware history management
conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Optional: Inspect Runnable pipeline topology
# print(conversational_rag)


# ==============================================================================
# STEP 9: Interactive CLI Chat Loop
# ==============================================================================
print("\n=== RAG Chatbot CLI Started (Type 'exit', 'quit', or 'q' to quit) ===")
config = {"configurable": {"session_id": "user_session_1"}}

try:
    while True:
        user_input = input("\nYou: ").strip()

        # Check for user exit commands
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting chat. Goodbye!")
            break

        # Ignore accidental empty returns
        if not user_input:
            continue

        # Invoke the conversational pipeline with session configuration
        response = conversational_rag.invoke(
            {"question": user_input},
            config=config,
        )
        print(f"\nAssistant: {response}")

except (KeyboardInterrupt, EOFError):
    # Handle Ctrl+C or Ctrl+D cleanly without traceback
    print("\nExiting chat. Goodbye!")
