from app.database.database import conn, model
from app.core.config import settings

from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_core.caches import InMemoryCache
from fastapi import Depends
from typing import Annotated 

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.1, 
    api_key=settings.openai_api_key
)

set_llm_cache(InMemoryCache())

class QuestionService():
    def search_portfolio(self, query, top_k=3):
        query_embedding = model.encode(query) 
        query_embedding = query_embedding.tolist() 

        if isinstance(query_embedding[0], list):  
            query_embedding = query_embedding[0]

        # Search for similar content
        results = conn.execute("""
            SELECT content
            FROM portfolio_content
            ORDER BY embedding <-> ?
            LIMIT ?;
        """, [query_embedding, top_k]).fetchall()
        
        return [content[0] for content in results]

    def answer(self, user_query: str):
        relevant_contexts = self.search_portfolio(user_query)
        context_text = "\n\n".join(relevant_contexts)

        messages = [
        {"role": "system", "content": """You are an AI assistant that provides detailed and well-structured answers about my portfolio.
        
            - If the query is related to my experience, skills, or projects, extract relevant information from the provided context and elaborate on it.
            - If multiple topics are mentioned, try to address each one clearly.
            - If a direct answer is not available, make a logical connection based on the given context rather than saying "I cannot answer."
            - If needed, provide insights about my contributions, challenges faced, and the impact of my work.
            - Don't mention about from "context provided", or "from the portfolio".
            - Act as you are my assistant answering someone who asks about my related experience, skills, or projects. Answer the question asked as the third party and calling my name "Zain".
            """},
            {"role": "user", "content": f"Here is relevant information from my portfolio:\n\n{context_text}\n\nBased on this, please provide a detailed response to the following query:\n\n{user_query}"}
        ]

        output = llm.invoke(messages)
        return output.content

        
question_service = QuestionService

QuestionService = Annotated[QuestionService, Depends(question_service)]