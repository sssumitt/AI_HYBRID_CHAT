import uuid
from packages.core_logic.graph import app
from ..schemas.chat import ChatRequest, ChatResponse

class ChatService:
    async def create_itinerary(self, request: ChatRequest) -> ChatResponse:
        """
        Orchestrates the hybrid RAG pipeline using LangGraph.
        """
        from packages.core_logic.state import AgentState
        initial_state: AgentState = {
            "user_query": request.query,
            "history": request.history or [],
            "matches": [],
            "graph_facts": [],
            "summary": "",
            "draft_itinerary": "",
            "validation_feedback": None,
            "iteration_count": 0,
            "answer": "",
            "source_ids": []
        }

        # Execute LangGraph workflow with a recursion limit protecting against infinite loops
        result = await app.ainvoke(initial_state, config={"recursion_limit": 15})

        answer = result.get("answer", "")
        source_ids = result.get("source_ids", [])
        
        convo_id = request.conversation_id or str(uuid.uuid4())
        
        # Construct updated history
        updated_history = (request.history or []) + [
            {"role": "user", "content": request.query},
            {"role": "assistant", "content": answer},
        ]
        # Keep only the last 10 messages (5 user-assistant pairs)
        updated_history = updated_history[-10:]

        return ChatResponse(
            answer=answer,
            source_ids=source_ids,
            conversation_id=convo_id,
            history=updated_history
        )
