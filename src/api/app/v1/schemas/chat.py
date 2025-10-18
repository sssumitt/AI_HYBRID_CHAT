# src/api/app/v1/schemas/chat.py
from pydantic import BaseModel
from typing import List, Optional,Dict

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    answer: str
    source_ids: List[str]
    conversation_id: str