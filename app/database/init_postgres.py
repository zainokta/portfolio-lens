"""PostgreSQL database initialization script."""

from sqlalchemy import text
from app.database.database import engine
from app.core.config import settings


def create_database_schema():
    """Create the database schema and required extensions."""
    with engine.connect() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        
        # Create the portfolio_content table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS portfolio_content (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(1536), 
                category VARCHAR(100),
                company VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        
        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS portfolio_content_embedding_idx 
            ON portfolio_content USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """))
        
        connection.commit()
        print("Database schema created successfully!")

if __name__ == "__main__":
    create_database_schema()