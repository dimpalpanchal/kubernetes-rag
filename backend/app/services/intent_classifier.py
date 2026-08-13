import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from app.core.config import settings

class IntentClassifierService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY, temperature=0)
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""Classify the following user message into one of two categories:
- CONVERSATIONAL: Greetings, pleasantries, small talk, gratitude, farewells, general questions about the AI assistant's identity ("who are you?", "what can you do?"), or non-technical chatter.
- KUBERNETES_QUERY: Technical questions, documentation lookups, conceptual explanations, CLI commands, troubleshooting, or code related to Kubernetes, containers, networking, storage, or cloud native architecture.

Output EXACTLY one word: CONVERSATIONAL or KUBERNETES_QUERY.

User Message: {query}
Category:"""
        )
        
        self.convo_regex = re.compile(
            r"^(hello|hi|hey|greetings|howdy|good morning|good afternoon|good evening|thanks|thank you|thank you so much|thanks!|who are you\??|what can you do\??|bye|goodbye|help)$",
            re.IGNORECASE
        )
        
        self.k8s_regex = re.compile(
            r"\b(pod|pods|deployment|deployments|service|services|ingress|kubectl|hpa|rbac|cluster|clusters|crashloopbackoff|namespace|namespaces|k8s|kubernetes|configmap|configmaps|secret|secrets|node|nodes|container|containers|pv|pvc|daemonset|statefulset|helm)\b",
            re.IGNORECASE
        )

    async def classify(self, query: str) -> str:
        clean_query = query.strip()
        if not clean_query:
            return "CONVERSATIONAL"
            
        # Fast path check for common short conversational expressions
        if self.convo_regex.match(clean_query.lower()):
            return "CONVERSATIONAL"
            
        # Fast path check for explicit Kubernetes queries
        if self.k8s_regex.search(clean_query):
            return "KUBERNETES_QUERY"
            
        try:
            formatted_prompt = self.prompt.format(query=clean_query)
            response = await self.llm.ainvoke(formatted_prompt)
            res_text = response.content.strip().upper()
            
            if "CONVERSATIONAL" in res_text:
                return "CONVERSATIONAL"
            return "KUBERNETES_QUERY"
        except Exception:
            return "KUBERNETES_QUERY"
