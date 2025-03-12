from app.database.database import conn, model

from fastapi import Depends
from typing import Annotated  


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
        
        prompt = f"""You are a helpful assistant for my portfolio website.
        
        Here is relevant information from my portfolio:
        {context_text}
        
        Based on this information, please answer: {user_query}"""

        return user_query
        
question_service = QuestionService

QuestionService = Annotated[QuestionService, Depends(question_service)]