import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";


class DeepSeek {
  Model: string
  llm: ChatOllama
  documentPath: string = ""
  documents: any[] = []
  chunkSize: number = 250
  chunkOverlap: number = 0
  vector_db: MemoryVectorStore
  selectEmbedding: any
  searchType: "similarity" | "mmr" = "similarity"
  kDocuments: number
  retriever: any
  ragChain: any

  constructor({ model, path, chunkSize, chunkOverlap, searchType, kDocuments }: { model: string, path: string, chunkSize: number, chunkOverlap: number, searchType: "similarity" | "mmr", kDocuments: number }) {
    this.Model = model
    this.documentPath = path
    this.chunkSize = chunkSize
    this.chunkOverlap = chunkOverlap,
      this.kDocuments = kDocuments,
      this.searchType = searchType
  }

  async init() {
    this.initChatModel();
    await this.loadDocuments()
    this.selectEmbedding = new OllamaEmbeddings({ model: "all-minilm:33m" })
    await this.createVectorStore()
    this.createRetriever()
    this.ragChain = await this.createChain()
    return this
  }

  async loadDocuments() {
    console.log("Loading Documents : invoking loadDocuments()")
    const directoryLoader = new DirectoryLoader(this.documentPath, {
      ".pdf": (path: string) => new PDFLoader(path)
    })
    const directoryDocs = await directoryLoader.load();

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
    });

    const splitDocs = await textSplitter.splitDocuments(directoryDocs);
    this.documents = splitDocs;
  }

  async createVectorStore() {
    console.log("Loading Vector Store : invoking createVectorStore()")
    this.vector_db = await MemoryVectorStore.fromDocuments(this.documents, this.selectEmbedding)
  }

  createRetriever() {
    console.log("Loading Retriever : invoking createRetriever()")
    this.retriever = this.vector_db.asRetriever({
      k: this.kDocuments,
      searchType: this.searchType
    });
  }

  async createChain() {
    console.log("Loading Chain : invoking createChain()")
    // creating history aware retriever
    const contextualizeQSystemPrompt =
      "Given a chat history and the latest user question " +
      "which might reference context in the chat history, " +
      "formulate a standalone question which can be understood " +
      "without the chat history. Do NOT answer the question, " +
      "just reformulate it if needed and otherwise return it as is.";

    const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
      ["system", contextualizeQSystemPrompt],
      new MessagesPlaceholder("chat_history"),
      ["human", "{input}"],
    ]);

    const historyAwareRetriever = await createHistoryAwareRetriever({
      llm: this.llm,
      retriever: this.retriever,
      rephrasePrompt: contextualizeQPrompt,
    });
    console.log(" created history aware retriever", historyAwareRetriever.toString()[0])

    const SYSTEM_TEMPLATE = `Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
`;
    const questionAnsweringPrompt = ChatPromptTemplate.fromMessages([
      ["system", SYSTEM_TEMPLATE],
      new MessagesPlaceholder("chat_history"),
      ["human", "{input}"],
    ]);

    const questionAnswerChain = await createStuffDocumentsChain({
      llm: this.llm,
      prompt: questionAnsweringPrompt,
    });

    console.log(" created questionAnswerChain", questionAnswerChain.toString()[0])

    const ragChain = await createRetrievalChain({
      retriever: historyAwareRetriever,
      combineDocsChain: questionAnswerChain,
    });

    console.log(" created ragChain", ragChain.toString()[0])
    return ragChain
  }

  async initChatModel() {
    console.log("Loading Model : invoking initChatModel()")
    this.llm = new ChatOllama({
      model: this.Model,
    });
  }


}


export default async function runDeepSeek() {
  const deepseek = new DeepSeek({
    model: "deepseek-r1:1.5b",
    path: "app/mydata/",
    chunkSize: 250,
    chunkOverlap: 0,
    searchType: "similarity",
    kDocuments: 10
  });
  await deepseek.init();
  const chain = deepseek.ragChain
  const chat = []
  const response = await chain.invoke({
    input: "What is the capital of France?",
    chat_history: chat
  })
  console.log(response)
}

