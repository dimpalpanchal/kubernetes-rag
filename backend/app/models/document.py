from sqlalchemy import Column, Integer, Text, Computed
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from pgvector.sqlalchemy import Vector
from app.core.database import Base

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB, nullable=True)
    embedding = Column(Vector(1536)) # text-embedding-3-small dimension
    ts_vector = Column(TSVECTOR, Computed("to_tsvector('english', content)", persisted=True))
