from app.database.database import conn
from app.core.config import settings

from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.caches import InMemoryCache
from fastapi import Depends
from typing import Annotated, List
import os 

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.1, 
    api_key=settings.openai_api_key
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=settings.openai_api_key
)

prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'prompt.txt')
with open(prompt_path, 'r') as f:
    prompt_content = f.read()

set_llm_cache(InMemoryCache())

class QuestionService():
    def search_portfolio(self, query: str) -> List[str]:
        """Search for relevant portfolio content based on query similarity."""
        try:
            query_embedding = embedding_model.embed_query(query)

            if isinstance(query_embedding[0], list):  
                query_embedding = query_embedding[0]

            results = conn.execute("""
                SELECT content, embedding <=> ? AS similarity, category, company
                FROM portfolio_content
                WHERE embedding <=> ? <= 0.6
                ORDER BY embedding <=> ? ASC
                LIMIT 5;
            """, [query_embedding, query_embedding, query_embedding]).fetchall()

            if not results:
                results = conn.execute("""
                    SELECT content, embedding <=> ? AS similarity, category, company
                    FROM portfolio_content
                    ORDER BY embedding <=> ? ASC
                    LIMIT 4;
                """, [query_embedding, query_embedding]).fetchall()

            return [content[0] for content in results]
        except Exception as e:
            print(f"Error searching portfolio: {e}")
            return []

    def answer(self, relevant_contexts: List[str], user_query: str) -> str:
        """Generate an answer based on relevant contexts and user query."""
        if not relevant_contexts:
            return "I don't have information on that topic."
        
        if self._is_irrelevant_query(user_query):
            return "I only provide information about Zain's professional experience, skills, and projects."
    
        context_text = "\n\n".join(relevant_contexts)

        messages = [
            {"role": "system", "content": prompt_content},
            {"role": "user", "content": f"Here is relevant information from my portfolio:\n\n{context_text}\n\nBased on this, please provide a detailed response to the following query:\n\n{user_query}"}
        ]

        try:
            output = llm.invoke(messages)
            return output.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I'm sorry, I encountered an error while processing your request."

    def search_by_category(self, query: str, category: str = None) -> List[str]:
        """Search for content within a specific category."""
        try:
            query_embedding = embedding_model.embed_query(query)

            if isinstance(query_embedding[0], list):  
                query_embedding = query_embedding[0]

            if category:
                results = conn.execute("""
                    SELECT content, embedding <=> ? AS similarity
                    FROM portfolio_content
                    WHERE category = ? AND embedding <=> ? <= 0.7
                    ORDER BY embedding <=> ? ASC
                    LIMIT 4;
                """, [query_embedding, category, query_embedding, query_embedding]).fetchall()
            else:
                return self.search_portfolio(query)

            return [content[0] for content in results]
        except Exception as e:
            print(f"Error searching by category: {e}")
            return []

    def _is_irrelevant_query(self, query: str) -> bool:
        """Check if query is about non-professional topics."""
        irrelevant_keywords = [
            'personal life', 'hobby', 'hobbies', 'family', 'relationship', 'dating',
            'politics', 'religion', 'opinion on', 'what do you think about',
            'favorite color', 'favorite food', 'age', 'birthday', 'address',
            'phone number', 'private', 'personal opinion', 'where do you live',
            'how old are you', 'are you single', 'do you have kids'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in irrelevant_keywords)

    def _detect_query_category(self, query: str) -> str:
        """Detect the most likely category for the query."""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['skill', 'technology', 'programming', 'language', 'tech stack']):
            return 'Technical Skills'
        elif any(keyword in query_lower for keyword in ['project', 'built', 'developed', 'created']):
            return 'Projects'
        elif any(keyword in query_lower for keyword in ['mentor', 'teach', 'student', 'guide']):
            return 'Mentoring'
        elif any(keyword in query_lower for keyword in ['work', 'job', 'company', 'experience', 'role']):
            return 'Work Experience'
        elif any(keyword in query_lower for keyword in ['education', 'degree', 'university', 'study']):
            return 'Education'
        elif any(keyword in query_lower for keyword in ['language', 'english', 'speak']):
            return 'Languages'
        
        return None

    def get_comprehensive_answer(self, user_query: str) -> str:
        """Main method to get answer for user query with enhanced search."""
        detected_category = self._detect_query_category(user_query)
        
        if detected_category:
            relevant_contexts = self.search_by_category(user_query, detected_category)
            if not relevant_contexts:
                relevant_contexts = self.search_portfolio(user_query)
        else:
            relevant_contexts = self.search_portfolio(user_query)
        
        return self.answer(relevant_contexts, user_query)


def get_question_service() -> QuestionService:
    """Dependency injection for QuestionService."""
    return QuestionService()

QuestionServiceDep = Annotated[QuestionService, Depends(get_question_service)]