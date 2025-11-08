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

def load_portfolio_data() -> List[Dict[str, str]]:
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
            
            portfolio_data = load_portfolio_data()

            print(f"Processing {len(portfolio_data)} portfolio entries...")

            # Extract content for embedding generation
            contents = [item["content"] for item in portfolio_data]

            # Generate embeddings for all content
            embeddings = embedding_model.embed_documents(contents)

            # Insert data with explicit category and company from JSON
            for i, (item, embedding) in enumerate(zip(portfolio_data, embeddings)):
                embedding_vector = np.array(embedding).tolist()

                conn.execute(
                    text("INSERT INTO portfolio_content (content, embedding, category, company) VALUES (:content, :embedding, :category, :company)"),
                    {
                        "content": item["content"],
                        "embedding": embedding_vector,
                        "category": item["category"],
                        "company": item["company"]
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

            print(f"Successfully populated database with {len(portfolio_data)} entries")
            
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