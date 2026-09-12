import os
import shutil
import tempfile
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Document Loaders & Splitters
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

# Load environment variables from .env
load_dotenv()

app = FastAPI(title="Multi-Document RAG API")

# Enable CORS for React/Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize free embedding model on CPU

embeddings = None


def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return embeddings


# In-Memory Stores
# vectorstores: { session_id (str): Chroma_instance }
vectorstores = {}
# chat_histories: { session_id (str): InMemoryChatMessageHistory_instance }
chat_histories = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieves or initializes the chat history for a given session."""
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]


class QueryRequest(BaseModel):
    session_id: str
    question: str


def load_file_content(file_path: str, filename: str):
    """Parses text dynamically based on document file extension."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
        return loader.load()
    elif ext == ".txt":
        loader = TextLoader(file_path)
        return loader.load()
    elif ext in [".xlsx", ".xls", ".csv"]:
        import pandas as pd

        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        text_data = df.to_string()
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(text_data)
            tmp_txt_path = tmp.name

        loader = TextLoader(tmp_txt_path)
        docs = loader.load()
        os.remove(tmp_txt_path)
        return docs
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


@app.get("/health")
def health_check():
    """Keep-alive ping endpoint for cron jobs."""
    return {"status": "ok", "message": "Server active"}


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
):
    """Receives document upload.

    If session_id exists, appends vectors to that session's store. If
    session_id is None, generates a new session_id. Enforces size limits,
    chunk caps, and batch vectorization for Render memory stability.
    """
    allowed_exts = [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".csv"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Please upload one of: {', '.join(allowed_exts)}",
        )

    # 1. File Size Guard (Prevents processing large files into RAM)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds the 10 MB limit for free tier deployment.",
        )

    # Use existing session_id or create a new UUID
    current_session_id = session_id if session_id else str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        documents = load_file_content(tmp_path, file.filename)

        # 2. Text Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)

        # 3. Safety Guard: Hard cap maximum chunks to protect 512 MB RAM limit
        if len(splits) > 300:
            splits = splits[:300]

        # 4. Batch Vector Embedding (32 chunks per batch)
        embedding_function = get_embeddings()

        if current_session_id in vectorstores:
            vectorstore = vectorstores[current_session_id]
        else:
            # Initialize vectorstore using the first batch to avoid empty initialization errors
            first_batch = splits[:32]
            vectorstore = Chroma.from_documents(
                documents=first_batch,
                embedding=embedding_function,
                collection_name=f"session_{current_session_id}",
            )
            vectorstores[current_session_id] = vectorstore
            splits = splits[32:]  # Exclude first batch since it's already indexed

        # Process remaining splits in mini-batches of 32
        batch_size = 32
        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            vectorstore.add_documents(documents=batch)

        return {
            "status": "success",
            "session_id": current_session_id,
            "filename": file.filename,
            "chunks_processed": (
                len(splits) + 32
                if current_session_id in vectorstores
                and len(splits) < len(text_splitter.split_documents(documents))
                else len(splits)
            ),
            "message": "Document indexed successfully in memory-safe batches.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/chat")
async def chat_with_document(request: QueryRequest):
    """
    Answers user queries with conversational history awareness
    based on the document context retrieved for the session_id.
    """
    session_id = request.session_id

    if session_id not in vectorstores:
        raise HTTPException(
            status_code=400,
            detail="No documents found for this session. Please upload a file first.",
        )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=500, detail="GROQ_API_KEY environment variable missing."
        )

    llm = ChatGroq(
        groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b", temperature=0
    )

    retriever = vectorstores[session_id].as_retriever(search_kwargs={"k": 3})

    # Step A: Contextualize Question Chain
    # Rephrases follow-up questions using past conversation history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = (
        contextualize_q_prompt | llm | StrOutputParser() | retriever
    )

    # Step B: Main QA Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_system_prompt = (
        "You are an AI assistant answering questions strictly based on the provided document context. "
        "If the answer is not contained in the context, explicitly state that it is unavailable.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # FIXED SETUP
    rag_chain = (
        {
            # We use a lambda to map both variables into the history_aware_retriever
            "context": (
                (lambda x: {"input": x["input"], "chat_history": x["chat_history"]})
                | history_aware_retriever
                | format_docs
            ),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # Step C: Wrap chain with automatic session message persistence
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Execute chain for the current session
    answer = conversational_rag_chain.invoke(
        {"input": request.question},
        config={"configurable": {"session_id": session_id}},
    )

    return {
        "session_id": session_id,
        "question": request.question,
        "answer": answer,
    }
