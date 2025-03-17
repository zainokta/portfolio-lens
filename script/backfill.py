import duckdb
import json
from langchain_openai import OpenAIEmbeddings
import getpass
import os

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

conn = duckdb.connect(database="portfolio.db")

conn.execute("INSTALL vss; LOAD vss;")

portfolio_chunks = []

with open('backfill.json') as json_data:
    portfolio_chunks = json.load(json_data)

embeddings = embedding_model.embed_documents(portfolio_chunks)

conn.execute("""
CREATE TABLE portfolio_content (
    id INTEGER PRIMARY KEY,
    content TEXT,
    embedding FLOAT[]
)
""")

for i, (content, embedding) in enumerate(zip(portfolio_chunks, embeddings)):
    conn.execute(
        "INSERT INTO portfolio_content VALUES (?, ?, ?)",
        [i, content, embedding]
    )
conn.close()