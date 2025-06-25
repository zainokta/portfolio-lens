from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime

from app.core.config import settings
from app.api.routes.question import question_router
from app.database.database import conn
from app.api.middleware import limiter
from dotenv import load_dotenv

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield {
    }

    conn.close()

def create_application() -> FastAPI:
    # Conditionally set docs URLs - disable in production
    docs_url = None if settings.is_production else "/docs"
    redoc_url = None if settings.is_production else "/redoc"
    
    application = FastAPI(
        title=settings.project_name,
        description=settings.project_description,
        version=settings.version,
        docs_url=docs_url,
        redoc_url=redoc_url,
        lifespan=lifespan
    )
    
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    application.state.limiter = limiter
    
    # Health check endpoint
    @application.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": settings.project_name,
            "version": settings.version
        }
    
    # Include routers
    application.include_router(question_router)
    
    return application

app = create_application()

if __name__ == "__main__":
    import uvicorn
    load_dotenv()
    uvicorn.run("main:app", host="0.0.0.0", port=settings.port)