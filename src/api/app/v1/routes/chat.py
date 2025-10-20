# src/api/app/v1/routes/chat.py
from fastapi import APIRouter, Depends, HTTPException
from ..schemas.chat import ChatRequest, ChatResponse
from ..services.chat_service import ChatService

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def handle_chat_request(
    request: ChatRequest,
    chat_service: ChatService = Depends(ChatService)
):
    """
    API endpoint for handling chat requests.
    Delegates all business logic to the ChatService.
    """
    try:
        return await chat_service.create_itinerary(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))