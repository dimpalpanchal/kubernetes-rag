from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from .document import DocumentChunkResponse

class ChatMessageCreate(BaseModel):
    session_id: str
    message: str

class ChatMessageResponse(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True

class ChatResponse(BaseModel):
    response: str
    sources: List[DocumentChunkResponse]

class ChatSessionResponse(BaseModel):
    session_id: str
    title: str
    started_at: datetime
    last_active: datetime

