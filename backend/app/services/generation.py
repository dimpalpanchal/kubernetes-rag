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
and the chat history to answer the user's query accurately.
If the retrieved context does not contain the answer, say that you don't know based on the provided documentation, but 
you can provide a general answer if helpful. Always cite your sources when possible using the provided metadata.

Documentation Context:
{context}

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
