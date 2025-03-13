from fastapi import APIRouter

from app.schemas.question import Question
from app.services.question import QuestionService

question_router = APIRouter(prefix="/api/question", tags=["question"])


@question_router.post("/query")
def query_question(request: Question, service: QuestionService):
    relevant_contexts = service.search_portfolio(request.query)

    return service.answer(relevant_contexts, request.query)