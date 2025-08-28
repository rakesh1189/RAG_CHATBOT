from pydantic import BaseModel, Field
from typing import List, Optional

class UploadResponse(BaseModel):
    doc_id: str
    pages: int

class AskRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID returned from /api/upload")
    question: str
    history: Optional[List[dict]] = Field(default_factory=list)

class Source(BaseModel):
    page_start: int
    page_end: int
    score: float
    preview: str

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
