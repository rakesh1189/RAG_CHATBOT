import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from typing import List

from models import UploadResponse, AskRequest, AskResponse, Source
from rag import upload_pdf, answer_question

# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings, SettingsConfigDict  # <-- add this

class Settings(BaseSettings):
    # required/used here:
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str | None = None
    VECTOR_STORE_DIR: str = "./storage/chroma"

    # optional (lets pydantic accept them if present)
    OPENAI_CHAT_MODEL: str | None = None
    OPENAI_EMBED_MODEL: str | None = None
    CHUNK_SIZE: int | None = None
    CHUNK_OVERLAP: int | None = None
    TOP_K: int | None = None

    # pydantic v2 config
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")  # <-- key line

settings = Settings()


app = FastAPI(title="RAG PDF Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/upload", response_model=UploadResponse)
async def api_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    file_bytes = await file.read()
    try:
        doc_id, pages = upload_pdf(file_bytes)
        return UploadResponse(doc_id=doc_id, pages=pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask", response_model=AskResponse)
async def api_ask(req: AskRequest):
    try:
        result = answer_question(req.doc_id, req.question, req.history or [])
        # Pydantic model conversion
        sources = [Source(**s) for s in result["sources"]]
        return AskResponse(answer=result["answer"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
