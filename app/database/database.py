import duckdb
from app.core.config import settings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

conn = duckdb.connect(database=settings.db_name, read_only=True)
