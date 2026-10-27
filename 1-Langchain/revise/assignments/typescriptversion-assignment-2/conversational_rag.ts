import * as fs from "node:fs";
import * as path from "node:path";
import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import * as dotenv from "dotenv";

import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import {
  trimMessages,
  type BaseMessage,
} from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";
import { RunnableSequence } from "@langchain/core/runnables";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings } from "@langchain/ollama";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { ChatGroq } from "@langchain/groq";

// Load environment variables (.env in current dir, fallback to parent dirs)
dotenv.config();
if (!process.env.GROQ_API_KEY) {
  dotenv.config({ path: path.resolve(__dirname, "../../../../.env") });
}

const groqApiKey = process.env.GROQ_API_KEY;
if (!groqApiKey) {
  console.warn("WARNING: GROQ_API_KEY is not set in environment or .env file!");
}

const currentDir = __dirname;
const indexPath = path.join(currentDir, "faiss_index");

async function main() {
  // ==============================================================================
  // 1. Embeddings Model (Assignment 2 Req 3: Different model than Assignment 1)
  // ==============================================================================
  const embeddings = new OllamaEmbeddings({
    model: "mxbai-embed-large",
  });

  // ==============================================================================
  // 2. Vector Store Setup (Assignment 2 Req 4: save_local & load_local)
  // ==============================================================================
  let db: FaissStore;

  if (fs.existsSync(indexPath)) {
    console.log(`Loading existing FAISS index from: ${indexPath}`);
    db = await FaissStore.load(indexPath, embeddings);
  } else {
    console.log("No existing index found. Starting multi-source ingestion...");

    // --- Source 1: Web Ingestion & Source-Specific Chunking ---
    console.log("1. Loading Web content (TechFerry Careers)...");
    const webLoader = new CheerioWebBaseLoader(
      "https://www.techferry.com/careers.html",
      {
        selector: 'div[class*="uk-card"]',
      }
    );
    const webDocs = await webLoader.load();

    // Web chunking strategy (tuned for HTML paragraph length)
    const webSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 600,
      chunkOverlap: 100,
    });
    const webChunks = await webSplitter.splitDocuments(webDocs);
    for (const chunk of webChunks) {
      chunk.metadata = {
        ...chunk.metadata,
        source_type: "web",
        source_title: "TechFerry Careers",
      };
    }

    // --- Source 2: PDF Ingestion & Source-Specific Chunking ---
    console.log("2. Loading PDF content (Mrinal Fouzdar Resume)...");
    const pdfFile = path.join(currentDir, "Mrinal_Fouzdar.pdf");
    const pdfLoader = new PDFLoader(pdfFile);
    const pdfDocs = await pdfLoader.load();

    // PDF chunking strategy (tuned for dense document text)
    const pdfSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 400,
      chunkOverlap: 80,
    });
    const pdfChunks = await pdfSplitter.splitDocuments(pdfDocs);
    for (const chunk of pdfChunks) {
      chunk.metadata = {
        ...chunk.metadata,
        source_type: "pdf",
        source_title: "Mrinal Resume",
      };
    }

    // --- Combine Enriched Chunks & Build Index ---
    const allChunks = [...webChunks, ...pdfChunks];
    console.log(
      `Total chunks unified: ${allChunks.length} (Web: ${webChunks.length}, PDF: ${pdfChunks.length})`
    );

    console.log("3. Generating embeddings & building FAISS vector index...");
    db = await FaissStore.fromDocuments(allChunks, embeddings);

    // Save locally to avoid re-embedding on future runs
    await db.save(indexPath);
    console.log(`FAISS index successfully saved to: ${indexPath}`);
  }

  // ==============================================================================
  // 3. Unified Retriever with LLM Re-Ranking
  // ==============================================================================
  // Step 1: Retrieve top-10 candidate chunks from the unified FAISS vector store
  const retriever = db.asRetriever({ k: 10 });
  console.log("Retriever ready for querying!");

  const llm = new ChatGroq({
    model: "openai/gpt-oss-20b",
    apiKey: groqApiKey,
  });

  // Re-ranker Prompt: Evaluates and selects the top-3 most relevant documents
  const rerankerPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are an expert document relevance ranker.\n" +
        "Given a user question and a numbered list of retrieved documents, select the top 3 " +
        "most relevant document IDs.\n" +
        'Return ONLY a valid JSON array of strings containing the selected IDs, e.g. ["doc_0", "doc_2"]. ' +
        "Do not include explanations or markdown fences.",
    ],
    ["human", "Question: {question}\n\nDocuments:\n{docs}"],
  ]);

  const rerankerChain = rerankerPrompt.pipe(llm).pipe(new StringOutputParser());

  async function rerankAndFormat(data: { question: string }): Promise<string> {
    const query = data.question;
    const docs = await retriever.invoke(query);

    // Map candidate docs
    const docMap = new Map<string, Document>();
    docs.forEach((doc, idx) => {
      docMap.set(`doc_${idx}`, doc);
    });

    const docText = Array.from(docMap.entries())
      .map(([docId, doc]) => `[${docId}]: ${doc.pageContent}`)
      .join("\n\n");

    let topDocs: Document[] = [];
    try {
      const rawResult = (await rerankerChain.invoke({
        docs: docText,
        question: query,
      })) as string;

      const cleanedJson = rawResult
        .trim()
        .replace(/^```json/i, "")
        .replace(/^```/, "")
        .replace(/```$/, "")
        .trim();

      const selectedIds: string[] = JSON.parse(cleanedJson);
      topDocs = selectedIds
        .map((did) => docMap.get(did))
        .filter((d): d is Document => d !== undefined);

      if (topDocs.length === 0) {
        topDocs = docs.slice(0, 3);
      }
    } catch {
      // Fallback to top-3 closest embeddings if re-ranker fails
      topDocs = docs.slice(0, 3);
    }

    return topDocs
      .map(
        (doc) =>
          `[Source: ${doc.metadata?.source_title || "Document"}]\n${doc.pageContent}`
      )
      .join("\n\n");
  }

  // ==============================================================================
  // 4. Prompt, Trimmer & LCEL Chain Assembly
  // ==============================================================================
  const systemPrompt =
    "You are a helpful assistant for question-answering tasks.\n" +
    "Use ONLY the following retrieved context (which includes candidate background " +
    "and company career details) to answer the user's question.\n" +
    "If the answer is not found in the context, strictly say 'I don't know'. " +
    "Do not extrapolate or assume.\n\n" +
    "Context:\n{context}";

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{question}"],
  ]);

  const ragChain = RunnableSequence.from([
    {
      context: async (input: { question: string; chat_history?: BaseMessage[] }) => {
        return await rerankAndFormat(input);
      },
      chat_history: async (input: { question: string; chat_history?: BaseMessage[] }) => {
        const history = input.chat_history || [];
        // Trimmer: Caps chat history at 500 tokens to prevent unbounded growth
        return await trimMessages(history, {
          maxTokens: 500,
          strategy: "last",
          tokenCounter: (messages: BaseMessage[]) =>
            messages.reduce(
              (acc, m) =>
                acc +
                (typeof m.content === "string"
                  ? Math.ceil(m.content.length / 4)
                  : 10),
              0
            ),
          includeSystem: false,
          startOn: "human",
        });
      },
      question: (input: { question: string }) => input.question,
    },
    prompt,
    llm,
    new StringOutputParser(),
  ]);

  // ==============================================================================
  // 5. Multi-Session Memory Management
  // ==============================================================================
  const sessionStore = new Map<string, InMemoryChatMessageHistory>();

  function getSessionHistory(sessionId: string): InMemoryChatMessageHistory {
    if (!sessionStore.has(sessionId)) {
      sessionStore.set(sessionId, new InMemoryChatMessageHistory());
    }
    return sessionStore.get(sessionId)!;
  }

  const conversationalRag = new RunnableWithMessageHistory({
    runnable: ragChain,
    getMessageHistory: (sessionId: string) => getSessionHistory(sessionId),
    inputMessagesKey: "question",
    historyMessagesKey: "chat_history",
  });

  // ==============================================================================
  // 6. Interactive CLI Chat Loop with Session Switching
  // ==============================================================================
  console.log("\n=== Conversational RAG with Re-Ranking CLI Started (TypeScript) ===");
  console.log("Commands: '/session <id>' to switch sessions | 'quit' to exit");

  let currentSession = "default";
  const rl = readline.createInterface({ input, output });

  try {
    while (true) {
      const rawInput = await rl.question(`\n[${currentSession}] You: `);
      const userInput = rawInput.trim();

      if (!userInput) {
        continue;
      }

      if (["exit", "quit", "q"].includes(userInput.toLowerCase())) {
        console.log("Goodbye!");
        break;
      }

      // Handle dynamic session switching (/session <id>)
      if (userInput.startsWith("/session ")) {
        const newSession = userInput.slice(9).trim();
        if (newSession) {
          currentSession = newSession;
          console.log(`Switched to session: '${currentSession}'`);
          continue;
        }
      }

      // Query with the active session
      const config = { configurable: { sessionId: currentSession } };
      const response = await conversationalRag.invoke(
        { question: userInput },
        config
      );
      console.log(`\n[${currentSession}] Assistant: ${response}`);
    }
  } catch (err: any) {
    if (err?.code !== "ERR_USE_AFTER_CLOSE") {
      console.error(`\nAn error occurred: ${err?.message || err}`);
    }
  } finally {
    rl.close();
  }
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
