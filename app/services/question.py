from app.database.database import conn
from app.core.config import settings

from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.caches import InMemoryCache
from fastapi import Depends
from typing import Annotated 

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.1, 
    api_key=settings.openai_api_key
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=settings.openai_api_key
)

prompt = open('prompt.txt', 'r')

set_llm_cache(InMemoryCache())

class QuestionService():
    def search_portfolio(self, query):
        
        query_embedding = embedding_model.embed_query(query)

        if isinstance(query_embedding[0], list):  
            query_embedding = query_embedding[0]

        results = conn.execute("""
            SELECT content, embedding <=> ? AS similarity
            FROM portfolio_content
            WHERE similarity <= 0.6
            ORDER BY similarity ASC
            LIMIT 3;
        """, [query_embedding]).fetchall()

        print(results)
        
        return [content[0] for content in results]

    def answer(self, relevant_contexts, user_query: str):
        if not relevant_contexts:
            return "I don't have information on that topic."
    
        context_text = "\n\n".join(relevant_contexts)

        messages = [
        {"role": "system", "content": prompt.read()},
        {"role": "user", "content": f"Here is relevant information from my portfolio:\n\n{context_text}\n\nBased on this, please provide a detailed response to the following query:\n\n{user_query}"}
        ]

        output = llm.invoke(messages)
        return output.content

        
question_service = QuestionService

QuestionService = Annotated[QuestionService, Depends(question_service)]