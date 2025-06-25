from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
from app.core.config import settings
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

Base = declarative_base()

engine = create_engine(
    settings.database_url,
    poolclass=NullPool,
    echo=settings.environment == "development"
)

async_engine = create_async_engine(
    settings.async_database_url,
    poolclass=NullPool,
    echo=settings.environment == "development"
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

def get_sync_session():
    """Get sync database session for compatibility."""
    session = SessionLocal()
    try:
        return session
    finally:
        session.close()

conn = engine
