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

    def answer(self, relevant_contexts, user_query: str):
        if not relevant_contexts:
            return "I don't have information on that topic."
    
        context_text = "\n\n".join(relevant_contexts)

        messages = [
        {"role": "system", "content": """You are an AI assistant helping answer questions about Zain's portfolio, experience, skills, and projects.  

            - **Only** answer questions that are directly related to Zain's portfolio, work experience, technical skills, or projects.  
            - If a question is **not** relevant (e.g., personal life, hobbies, pet names), respond with: **"I only provide information about Zain's professional experience, skills, and projects."**  
            - If multiple topics are mentioned, answer only the ones relevant to Zain's work and experience. Ignore unrelated topics.  
            - If no useful information is found in the context, say: **"I don't have information on that topic."** Do **not** attempt to infer or generate unrelated details.  
            - If the provided context contains **partially relevant** information, use only the parts that are useful and ignore the rest.  
            - Never mention "from the context provided" or "from the portfolio." Instead, answer naturally as Zain's assistant.  
            - Structure responses clearly, making them easy to understand.  

            **Reminder:** Strictly avoid answering anything unrelated to Zain's professional background. If unsure, default to: **"I only provide information about Zain's professional experience, skills, and projects."**  

        """},
        {"role": "user", "content": f"Here is relevant information from my portfolio:\n\n{context_text}\n\nBased on this, please provide a detailed response to the following query:\n\n{user_query}"}
        ]

        output = llm.invoke(messages)
        return output.content

        
question_service = QuestionService

QuestionService = Annotated[QuestionService, Depends(question_service)]