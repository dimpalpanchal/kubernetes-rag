from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.models.chat import ChatMessage
from app.schemas.chat import ChatMessageCreate, ChatResponse, ChatMessageResponse
from app.schemas.document import DocumentChunkResponse

from app.services.query_rewriter import QueryRewriterService
from app.services.hybrid_search import HybridSearchService
from app.services.reranker import RerankerService
from app.services.generation import GenerationService

router = APIRouter()

query_rewriter = QueryRewriterService()
hybrid_search = HybridSearchService()
reranker = RerankerService()
generation = GenerationService()

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatMessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # 1. Load chat history (last 4 turns)
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == request.session_id)
        .where(ChatMessage.user_id == current_user.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(8) # 4 turns = 8 messages
    )
    res = await db.execute(stmt)
    history_messages = res.scalars().all()[::-1] # Reverse to get chronological order
    
    chat_history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history_messages])
    
    # 2. Rewrite Query
    rewritten_query = await query_rewriter.rewrite(request.message, chat_history_str)
    
    # 3. Hybrid Search
    top_chunks = await hybrid_search.search(db, rewritten_query, top_k=20)
    
    # 4. Rerank
    reranked_chunks = reranker.rerank(rewritten_query, top_chunks, top_k=4)
    
    # 5. Generation
    final_answer = await generation.generate(rewritten_query, reranked_chunks, chat_history_str)
    
    # 6. Save to DB
    user_msg = ChatMessage(
        user_id=current_user.id,
        session_id=request.session_id,
        role="user",
        content=request.message
    )
    ai_msg = ChatMessage(
        user_id=current_user.id,
        session_id=request.session_id,
        role="assistant",
        content=final_answer
    )
    db.add(user_msg)
    db.add(ai_msg)
    await db.commit()
    
    # Format sources for response
    sources = [
        DocumentChunkResponse(
            id=chunk.id,
            content=chunk.content,
            metadata_=chunk.metadata_
        ) for chunk in reranked_chunks
    ]
    
    return ChatResponse(response=final_answer, sources=sources)

@router.get("/history/{session_id}", response_model=List[ChatMessageResponse])
async def get_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .where(ChatMessage.user_id == current_user.id)
        .order_by(ChatMessage.created_at.asc())
    )
    res = await db.execute(stmt)
    return res.scalars().all()
