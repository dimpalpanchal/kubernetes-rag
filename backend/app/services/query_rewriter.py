from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from app.core.config import settings

class QueryRewriterService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "query"],
            template="""You are a helpful assistant. Given the following chat history and a new user query, 
rewrite the user query to be a standalone query that captures the full context of the conversation. 
If the query is already standalone or no chat history is provided, just return the original query.
Do not answer the query, just rewrite it.

Chat History:
{chat_history}

User Query: {query}
Standalone Query:"""
        )

    async def rewrite(self, query: str, chat_history: str) -> str:
        if not chat_history.strip():
            return query
        
        formatted_prompt = self.prompt.format(chat_history=chat_history, query=query)
        response = await self.llm.ainvoke(formatted_prompt)
        return response.content.strip()
