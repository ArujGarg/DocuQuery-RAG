import os
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Document Loaders & Splitters
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Global reference for Chroma vector store
vectorstore = None


class QueryRequest(BaseModel):
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


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receives document upload, chunks text, generates embeddings, stores in ChromaDB."""
    global vectorstore

    allowed_exts = [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".csv"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Please upload one of: {', '.join(allowed_exts)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        documents = load_file_content(tmp_path, file.filename)

        # Chunk text (700 chars, 100 char overlap)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)

        # Vectorize and index in ChromaDB
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_processed": len(splits),
            "message": "Document indexed successfully.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/chat")
async def chat_with_document(request: QueryRequest):
    """Answers user queries based on context retrieved from the index."""
    global vectorstore

    if vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a document first.",
        )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=500, detail="GROQ_API_KEY environment variable missing."
        )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b", temperature=0
    )

    system_prompt = (
        "You are an AI assistant answering questions strictly based on the provided document context. "
        "If the answer is not contained in the context, explicitly state that it is unavailable.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(request.question)

    return {"question": request.question, "answer": answer}
