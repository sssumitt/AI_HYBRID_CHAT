import uuid
from packages.core_logic.rag_pipeline import (
    pinecone_query,
    fetch_graph_context,
    search_summary,
    call_chat,
)
# Update the import to the new prompt function
from packages.core_logic.llm_prompts import build_prompt_with_history
from packages.core_logic.config import TOP_K
from ..schemas.chat import ChatRequest, ChatResponse

class ChatService:
    async def create_itinerary(self, request: ChatRequest) -> ChatResponse:
        """
        Orchestrates the RAG pipeline, now with conversation history.
        """
        # Step 1: Retrieve
        matches = await pinecone_query(request.query, top_k=TOP_K)
        match_ids = [m.get("id") for m in matches if m.get("id")]
        graph_facts = await fetch_graph_context(match_ids)

        # Step 2: Augment
        summary = await search_summary(request.query, matches, graph_facts)

        # Step 3: Generate, now passing the history to the prompt builder
        prompt = build_prompt_with_history(request.query, summary, request.history)
        answer = await call_chat(prompt)

        # Step 4: Prepare response and update history
        convo_id = request.conversation_id or str(uuid.uuid4())
        
        # The new history includes the user's last message and the AI's new response
        updated_history = (request.history or []) + [
            {"role": "user", "content": request.query},
            {"role": "assistant", "content": answer},
        ]
        # Limit the history to last (5 most recent messages)
        updated_history = updated_history[-10:]

        return ChatResponse(
            answer=answer,
            source_ids=match_ids,
            conversation_id=convo_id,
            history=updated_history
        )
