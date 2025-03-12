import duckdb
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

conn = duckdb.connect(database="portfolio.db")

conn.execute("INSTALL vss; LOAD vss;")

portfolio_chunks = []

with open('backfill.json') as json_data:
    portfolio_chunks = json.load(json_data)

embeddings = model.encode(portfolio_chunks)

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
        [i, content, embedding.tolist()]
    )