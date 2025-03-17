import duckdb
from app.core.config import settings

conn = duckdb.connect(database=settings.db_name, read_only=True)
