import asyncio
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from typing import List

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.models.chat import ChatMessage
from app.schemas.chat import ChatMessageCreate, ChatResponse, ChatMessageResponse, ChatSessionResponse
from app.schemas.document import DocumentChunkResponse

from app.services.intent_classifier import IntentClassifierService
from app.services.query_rewriter import QueryRewriterService
from app.services.hybrid_search import HybridSearchService
from app.services.reranker import RerankerService
from app.services.generation import GenerationService

router = APIRouter()

intent_classifier = IntentClassifierService()
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
    
    # 2. Classify Intent
    intent = await intent_classifier.classify(request.message)
    
    if intent == "CONVERSATIONAL":
        final_answer = await generation.generate_conversational(request.message, chat_history_str)
        reranked_chunks = []
    else:
        # 3. Rewrite Query
        rewritten_query = await query_rewriter.rewrite(request.message, chat_history_str)
        
        # 4. Hybrid Search (top_k=20 candidate pool)
        top_chunks = await hybrid_search.search(db, rewritten_query, top_k=20)
        
        # 5. Groq Rerank & Deduplicate
        reranked_chunks = await reranker.rerank_async(rewritten_query, top_chunks, top_k=4)
        
        # 6. Generation
        final_answer = await generation.generate(rewritten_query, reranked_chunks, chat_history_str)
    
    # 7. Save to DB
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

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    subquery = (
        select(
            ChatMessage.session_id,
            func.min(ChatMessage.created_at).label("started_at"),
            func.max(ChatMessage.created_at).label("last_active")
        )
        .where(ChatMessage.user_id == current_user.id)
        .group_by(ChatMessage.session_id)
        .subquery()
    )
    
    stmt = (
        select(subquery.c.session_id, subquery.c.started_at, subquery.c.last_active)
        .order_by(subquery.c.last_active.desc())
    )
    res = await db.execute(stmt)
    rows = res.all()
    
    sessions = []
    for session_id, started_at, last_active in rows:
        first_msg_stmt = (
            select(ChatMessage.content)
            .where(ChatMessage.session_id == session_id)
            .where(ChatMessage.user_id == current_user.id)
            .where(ChatMessage.role == "user")
            .order_by(ChatMessage.created_at.asc())
            .limit(1)
        )
        first_msg_res = await db.execute(first_msg_stmt)
        first_msg = first_msg_res.scalar_one_or_none()
        
        title = first_msg if first_msg else "New Session"
        if len(title) > 35:
            title = title[:35] + "..."
            
        sessions.append(
            ChatSessionResponse(
                session_id=session_id,
                title=title,
                started_at=started_at,
                last_active=last_active
            )
        )
        
    return sessions

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    stmt = (
        delete(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .where(ChatMessage.user_id == current_user.id)
    )
    await db.execute(stmt)
    await db.commit()
    return {"message": "Session deleted successfully"}

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
