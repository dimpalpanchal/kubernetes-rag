from pydantic import BaseModel
from typing import Optional, Dict, Any

class DocumentChunkResponse(BaseModel):
    id: int
    content: str
    metadata_: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
