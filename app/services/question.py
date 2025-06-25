from app.database.database import engine, get_sync_session
from app.core.config import settings
from sqlalchemy import text
import numpy as np
from datetime import datetime

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
    model="text-embedding-3-small",
    api_key=settings.openai_api_key
)

prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'prompt.txt')
with open(prompt_path, 'r') as f:
    prompt_content = f.read()

set_llm_cache(InMemoryCache())

class QuestionService:
    def __init__(self):
        pass
    def search_portfolio(self, query: str) -> List[str]:
        """Search for relevant portfolio content based on query similarity."""
        try:
            query_embedding = embedding_model.embed_query(query)

            if isinstance(query_embedding[0], list):  
                query_embedding = query_embedding[0]

            query_vector = np.array(query_embedding)
            
            with engine.connect() as connection:
                results = connection.execute(
                    text("""
                        SELECT content, 1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity, category, company
                        FROM portfolio_content
                        WHERE 1 - (embedding <=> CAST(:query_embedding AS vector)) >= 0.4
                        ORDER BY embedding <=> CAST(:query_embedding AS vector) ASC
                        LIMIT 8;
                    """), 
                    {"query_embedding": query_vector.tolist()}
                ).fetchall()

                if not results:
                    results = connection.execute(
                        text("""
                            SELECT content, 1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity, category, company
                            FROM portfolio_content
                            ORDER BY embedding <=> CAST(:query_embedding AS vector) ASC
                            LIMIT 8;
                        """), 
                        {"query_embedding": query_vector.tolist()}
                    ).fetchall()

            return [content[0] for content in results]
        except Exception as e:
            print(f"Error searching portfolio: {e}")
            return []

    def answer(self, relevant_contexts: List[str], user_query: str) -> str:
        """Generate answer based on relevant contexts and user query."""
        try:
            if self._is_irrelevant_query(user_query):
                return "I can only provide information about Zain's professional background, skills, and experience. Please ask questions related to his portfolio."
            
            if not relevant_contexts:
                return "I couldn't find relevant information to answer your question. Please try rephrasing or ask about Zain's skills, projects, or work experience."
            
            context_text = "\n\n".join(relevant_contexts)
            current_year = datetime.now().year
            
            # Enhanced prompt for experience queries
            if self._is_experience_query(user_query):
                prompt = f"""Based on the following comprehensive information about Zain's professional experience, please provide a detailed answer about his work history. Current year is {current_year}.

                Context (includes work experience, projects, and technical skills):
                {context_text}

                User Question: {user_query}

                Please structure your response to include:
                1. Job titles, companies, and dates
                2. Specific accomplishments, projects, and technologies used at each role
                3. Key contributions and achievements
                4. Technical skills developed or utilized

                Provide a comprehensive overview that shows both the timeline and the detailed work done at each position."""
            else:
                prompt = f"""Based on the following information about Zain, please answer the user's question comprehensively and professionally. Current year is {current_year}.

                Context:
                {context_text}

                User Question: {user_query}

                Please provide a detailed and informative answer based only on the provided context. If the context doesn't contain enough information to fully answer the question, mention what information is available."""
            
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I encountered an error while generating the answer. Please try again."

    def search_by_category(self, query: str, category: str = None) -> List[str]:
        """Search for content within a specific category."""
        try:
            query_embedding = embedding_model.embed_query(query)

            if isinstance(query_embedding[0], list):  
                query_embedding = query_embedding[0]

            if category:
                query_vector = np.array(query_embedding)
                
                with engine.connect() as connection:
                    results = connection.execute(
                        text("""
                            SELECT content, 1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
                            FROM portfolio_content
                            WHERE category = :category AND 1 - (embedding <=> CAST(:query_embedding AS vector)) >= 0.3
                            ORDER BY embedding <=> CAST(:query_embedding AS vector) ASC
                            LIMIT 8;
                        """), 
                        {"query_embedding": query_vector.tolist(), "category": category}
                    ).fetchall()
            else:
                return self.search_portfolio(query)

            return [content[0] for content in results]
        except Exception as e:
            print(f"Error searching by category: {e}")
            return []

    def search_work_experience(self, query: str) -> List[str]:
        """Enhanced search for work experience with detailed accomplishments."""
        try:
            query_embedding = embedding_model.embed_query(query)

            if isinstance(query_embedding[0], list):  
                query_embedding = query_embedding[0]

            query_vector = np.array(query_embedding)
            
            with engine.connect() as connection:
                # Get work experience entries with company details
                results = connection.execute(
                    text("""
                        SELECT content, 1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity, category, company
                        FROM portfolio_content
                        WHERE category = 'Work Experience' 
                        ORDER BY embedding <=> CAST(:query_embedding AS vector) ASC
                        LIMIT 10;
                    """), 
                    {"query_embedding": query_vector.tolist()}
                ).fetchall()
                
                # Also get related technical skills and projects for each company
                companies = list(set([result[3] for result in results if result[3]]))
                additional_context = []
                
                for company in companies:
                    company_details = connection.execute(
                        text("""
                            SELECT content, category
                            FROM portfolio_content
                            WHERE company = :company AND category IN ('Technical Skills', 'Projects')
                            LIMIT 5;
                        """), 
                        {"company": company}
                    ).fetchall()
                    additional_context.extend([detail[0] for detail in company_details])

            # Combine work experience with related accomplishments
            all_content = [result[0] for result in results] + additional_context
            return all_content[:12]  # Limit total results
            
        except Exception as e:
            print(f"Error searching work experience: {e}")
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
        """Detect the category of the user's query."""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['skill', 'technology', 'programming', 'language', 'framework', 'tool']):
            return 'Technical Skills'
        elif any(keyword in query_lower for keyword in ['project', 'built', 'developed', 'created', 'application', 'system']):
            return 'Projects'
        elif any(keyword in query_lower for keyword in ['work', 'job', 'role', 'position', 'company', 'employer', 'career', 'experience', 'past', 'years']):
            return 'Work Experience'
        elif any(keyword in query_lower for keyword in ['education', 'degree', 'university', 'college', 'study', 'academic']):
            return 'Education'
        elif any(keyword in query_lower for keyword in ['certificate', 'certification', 'course', 'training']):
            return 'Certifications'
        
        return None
    
    def _is_experience_query(self, query: str) -> bool:
        """Check if query is specifically about work experience with timeline."""
        query_lower = query.lower()
        experience_keywords = ['experience', 'past', 'years', 'worked', 'roles', 'positions']
        timeline_keywords = ['3 years', 'past 3', 'last 3', 'recent', 'timeline']
        
        has_experience = any(keyword in query_lower for keyword in experience_keywords)
        has_timeline = any(keyword in query_lower for keyword in timeline_keywords)
        
        return has_experience and (has_timeline or 'years' in query_lower)

    def get_comprehensive_answer(self, user_query: str) -> str:
        """Get a comprehensive answer based on the user's query."""
        try:
            if self._is_irrelevant_query(user_query):
                return "I can only provide information about Zain's professional background, skills, and experience. Please ask questions related to his portfolio."
            
            # Check if this is a detailed experience query
            if self._is_experience_query(user_query):
                relevant_contexts = self.search_work_experience(user_query)
            else:
                category = self._detect_query_category(user_query)
                
                if category:
                    relevant_contexts = self.search_by_category(user_query, category)
                    if not relevant_contexts:
                        relevant_contexts = self.search_portfolio(user_query)
                else:
                    relevant_contexts = self.search_portfolio(user_query)
            
            if not relevant_contexts:
                return "I couldn't find relevant information to answer your question. Please try rephrasing or ask about Zain's skills, projects, or work experience."
            
            return self.answer(relevant_contexts, user_query)
        except Exception as e:
            print(f"Error getting comprehensive answer: {e}")
            return "I encountered an error while processing your question. Please try again."


def get_question_service() -> QuestionService:
    """Dependency injection for QuestionService."""
    return QuestionService()

QuestionServiceDep = Annotated[QuestionService, Depends(get_question_service)]