from fastapi import APIRouter, HTTPException, Request

from app.schemas.question import Question
from app.services.question import QuestionServiceDep
from app.api.middleware import limiter

question_router = APIRouter(prefix="/api/question", tags=["question"])


@question_router.post("/query")
@limiter.limit("10 per minute", error_message="Rate limit exceeded. Try again later.")
async def query_question(request: Request, question: Question, service: QuestionServiceDep):
    if len(question.query) > 300:
        raise HTTPException(status_code=400, detail="Query too long")
    
    if not question.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
     
    try:
        answer = service.get_comprehensive_answer(question.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")