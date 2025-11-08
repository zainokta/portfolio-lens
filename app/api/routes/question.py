from fastapi import APIRouter, HTTPException, Request

from app.schemas.question import Question
from app.services.question import QuestionServiceDep
from app.api.middleware import limiter

question_router = APIRouter(prefix="/api/question", tags=["question"])


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request, handling proxy headers."""
    # Check for X-Forwarded-For header (common with reverse proxies like nginx)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs, use the first one (original client)
        return forwarded_for.split(",")[0].strip()

    # Check for X-Real-IP header (alternative proxy header)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fallback to direct client IP
    if request.client:
        return request.client.host

    # Last resort fallback
    return "unknown"


@question_router.post("/query")
@limiter.limit("10 per minute", error_message="Rate limit exceeded. Try again later.")
async def query_question(request: Request, question: Question, service: QuestionServiceDep):
    if len(question.query) > 300:
        raise HTTPException(status_code=400, detail="Query too long")

    if not question.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Extract client IP for rate limiting
    client_ip = get_client_ip(request)

    try:
        answer = service.get_comprehensive_answer(question.query, client_id=client_ip)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")