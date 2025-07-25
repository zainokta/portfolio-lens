import json
from langchain_openai import OpenAIEmbeddings
import os
from typing import List, Dict
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings

def load_portfolio_data() -> List[str]:
    """Load and structure portfolio data from JSON file."""
    with open('backfill.json') as json_data:
        return json.load(json_data)

def create_database_schema(conn):
    """Create the database schema with proper indexing."""
    
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    
    conn.execute(text("DROP TABLE IF EXISTS portfolio_content"))
    
    conn.execute(text("""
    CREATE TABLE portfolio_content (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        embedding vector(1536),
        category VARCHAR(50),
        company VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """))
    
    conn.commit()

def categorize_content(content: str) -> Dict[str, str]:
    """Categorize content and extract company information."""
    content_lower = content.lower()
    
    company = "General"
    if "accelbyte" in content_lower:
        company = "AccelByte"
    elif "efishery" in content_lower:
        company = "eFishery"
    elif "dibimbing" in content_lower:
        company = "Dibimbing.id"
    elif "sakoo" in content_lower:
        company = "Sakoo"
    elif "alterra" in content_lower:
        company = "Alterra"
    elif "ruangguru" in content_lower:
        company = "Ruangguru"
    
    category = "General"
    if any(keyword in content_lower for keyword in ["tech stack", "programming", "technologies"]):
        category = "Technical Skills"
    elif any(keyword in content_lower for keyword in ["project", "developed", "implemented", "architected"]):
        category = "Projects"
    elif any(keyword in content_lower for keyword in ["mentored", "guided", "taught", "students"]):
        category = "Mentoring"
    elif any(keyword in content_lower for keyword in ["worked", "engineer", "intern", "present"]):
        category = "Work Experience"
    elif any(keyword in content_lower for keyword in ["education", "degree", "university", "politeknik"]):
        category = "Education"
    elif any(keyword in content_lower for keyword in ["language", "english", "indonesian"]):
        category = "Languages"
    
    return {"category": category, "company": company}

def main():
    """Main function to populate the portfolio database."""
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file.")
    
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )
    
    engine = create_engine(settings.database_url)
    
    try:
        with engine.connect() as conn:
            create_database_schema(conn)
            
            portfolio_chunks = load_portfolio_data()
            
            print(f"Processing {len(portfolio_chunks)} portfolio entries...")
            
            embeddings = embedding_model.embed_documents(portfolio_chunks)
            
            for i, (content, embedding) in enumerate(zip(portfolio_chunks, embeddings)):
                metadata = categorize_content(content)
                
                embedding_vector = np.array(embedding).tolist()
                
                conn.execute(
                    text("INSERT INTO portfolio_content (content, embedding, category, company) VALUES (:content, :embedding, :category, :company)"),
                    {
                        "content": content,
                        "embedding": embedding_vector,
                        "category": metadata["category"],
                        "company": metadata["company"]
                    }
                )
            
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_category ON portfolio_content(category)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_company ON portfolio_content(company)"))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS portfolio_content_embedding_idx 
                ON portfolio_content USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
            
            conn.commit()
            
            print(f"Successfully populated database with {len(portfolio_chunks)} entries")
            
            result = conn.execute(text("""
                SELECT category, company, COUNT(*) as count 
                FROM portfolio_content 
                GROUP BY category, company 
                ORDER BY category, company
            """)).fetchall()
            
            print("\nData Summary:")
            for row in result:
                print(f"  {row[0]} - {row[1]}: {row[2]} entries")
                
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()