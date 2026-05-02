import os
import shutil
import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import config
from src.vector_store import ChromaVectorStore
from src.llm import create_llm
from src.ingestion import IngestionPipeline
from src.query_engine import QueryEngine


# ── App state — loaded once, lives for the entire server lifetime ──────────────
class AppState:
    vector_store: ChromaVectorStore = None
    engine: QueryEngine = None
    pipeline: IngestionPipeline = None

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\nLoading models and connecting to vector store...")
    state.vector_store = ChromaVectorStore(config.vector_store, config.embedding)
    llm = create_llm(config.llm)
    state.pipeline = IngestionPipeline(config, state.vector_store)
    state.engine = QueryEngine(state.vector_store, llm, config.llm)
    print("Ready. Models loaded once — all queries served from memory.\n")
    yield
    print("Shutting down.")


app = FastAPI(title="Contract Analyzer", lifespan=lifespan)
app.mount("/ui", StaticFiles(directory="ui"), name="ui")


# ── Request / Response models ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    question: str

class StatusResponse(BaseModel):
    chunks: int
    llm_ready: bool
    provider: str
    model: str


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("ui/index.html")


@app.get("/api/status", response_model=StatusResponse)
def status():
    return StatusResponse(
        chunks=state.vector_store.document_count(),
        llm_ready=create_llm(config.llm).is_available(),
        provider=config.llm.provider,
        model=config.llm.model_name,
    )


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = state.engine.query(req.question)

    seen = set()
    sources = []
    for doc in result.source_documents:
        key = (doc.source, doc.page)
        if key not in seen:
            seen.add(key)
            label = f"{doc.source} (page {doc.page})" if doc.page > 0 else doc.source
            sources.append(label)

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        question=result.query,
    )


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    allowed = (".pdf", ".txt", ".docx")
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    os.makedirs(config.contracts_dir, exist_ok=True)
    save_path = os.path.join(config.contracts_dir, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = state.pipeline.ingest_file(save_path)
    return {"filename": file.filename, "chunks": chunks}


@app.post("/api/ingest/directory")
def ingest_directory():
    os.makedirs(config.contracts_dir, exist_ok=True)
    total, files = state.pipeline.ingest_directory(config.contracts_dir)
    return {"files": files, "total_chunks": total}


@app.delete("/api/clear")
def clear():
    state.vector_store.clear()
    return {"message": "Vector store cleared."}


if __name__ == "__main__":
    os.makedirs("ui", exist_ok=True)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
