from fastapi import APIRouter, HTTPException, Request

from app.schemas.question import Question
from app.services.question import QuestionService
from app.api.middleware import limiter

question_router = APIRouter(prefix="/api/question", tags=["question"])


@question_router.post("/query")
@limiter.limit("2 per hour", error_message="Rate limit exceeded. Try again later.")
async def query_question(request: Request, question: Question, service: QuestionService):
    if len(question.query) > 300:
        raise HTTPException(status_code=400, detail="Query too long")
     
    relevant_contexts = service.search_portfolio(question.query)

    # return service.answer(relevant_contexts, question.query)
    return relevant_contexts