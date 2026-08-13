from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from app.core.config import settings
from app.models.document import DocumentChunk
from typing import List

class GenerationService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
        self.prompt = PromptTemplate(
            input_variables=["context", "chat_history", "query"],
            template="""You are an expert Kubernetes AI assistant. Use the following retrieved documentation chunks 
and chat history to answer the user's query accurately, clearly, and concisely.
Incorporate relevant details from the provided documentation context. Provide a complete and direct answer to the user's query.

Documentation Context:
{context}

Chat History:
{chat_history}

User Query: {query}
Answer:"""
        )
        self.convo_prompt = PromptTemplate(
            input_variables=["chat_history", "query"],
            template="""You are an expert Kubernetes AI assistant. Respond warmly, concisely, and helpfully to the user's conversational message.
If they are greeting you or asking who you are, introduce yourself as the Kubernetes Docs Assistant and let them know you're ready to answer any questions about Kubernetes, pods, deployments, services, clusters, and configuration.

Chat History:
{chat_history}

User Query: {query}
Answer:"""
        )

    async def generate(self, query: str, docs: List[DocumentChunk], chat_history: str) -> str:
        context_str = ""
        for i, doc in enumerate(docs):
            source = doc.metadata_.get("source", "Unknown") if doc.metadata_ else "Unknown"
            context_str += f"[Chunk {i+1}] (Source: {source}):\n{doc.content}\n\n"
            
        formatted_prompt = self.prompt.format(
            context=context_str,
            chat_history=chat_history,
            query=query
        )
        response = await self.llm.ainvoke(formatted_prompt)
        return response.content.strip()

    async def generate_conversational(self, query: str, chat_history: str) -> str:
        formatted_prompt = self.convo_prompt.format(
            chat_history=chat_history,
            query=query
        )
        response = await self.llm.ainvoke(formatted_prompt)
        return response.content.strip()

