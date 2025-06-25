import duckdb
import json
from langchain_openai import OpenAIEmbeddings
import getpass
import os
from typing import List, Dict

def load_portfolio_data() -> List[str]:
    """Load and structure portfolio data from JSON file."""
    with open('backfill.json') as json_data:
        return json.load(json_data)

def create_database_schema(conn):
    """Create the database schema with proper indexing."""
    conn.execute("DROP TABLE IF EXISTS portfolio_content")
    
    conn.execute("""
    CREATE TABLE portfolio_content (
        id INTEGER PRIMARY KEY,
        content TEXT NOT NULL,
        embedding FLOAT[] NOT NULL,
        category VARCHAR(50),
        company VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

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
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )
    
    conn = duckdb.connect(database="portfolio.db")
    
    try:
        conn.execute("INSTALL vss; LOAD vss;")
        
        create_database_schema(conn)
        
        portfolio_chunks = load_portfolio_data()
        
        print(f"Processing {len(portfolio_chunks)} portfolio entries...")
        
        embeddings = embedding_model.embed_documents(portfolio_chunks)
        
        for i, (content, embedding) in enumerate(zip(portfolio_chunks, embeddings)):
            metadata = categorize_content(content)
            
            conn.execute(
                "INSERT INTO portfolio_content (id, content, embedding, category, company) VALUES (?, ?, ?, ?, ?)",
                [i, content, embedding, metadata["category"], metadata["company"]]
            )
        
        conn.execute("CREATE INDEX idx_category ON portfolio_content(category)")
        conn.execute("CREATE INDEX idx_company ON portfolio_content(company)")
        
        print(f"Successfully populated database with {len(portfolio_chunks)} entries")
        
        result = conn.execute("""
            SELECT category, company, COUNT(*) as count 
            FROM portfolio_content 
            GROUP BY category, company 
            ORDER BY category, company
        """).fetchall()
        
        print("\nData Summary:")
        for row in result:
            print(f"  {row[0]} - {row[1]}: {row[2]} entries")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()