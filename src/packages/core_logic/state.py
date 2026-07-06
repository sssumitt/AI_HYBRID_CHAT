# src/packages/core_logic/state.py
from typing import List, Dict, Any, Optional, TypedDict

class AgentState(TypedDict):
    user_query: str
    history: List[Dict[str, str]]
    matches: List[Dict[str, Any]]
    graph_facts: List[Dict[str, Any]]
    summary: str
    draft_itinerary: str
    validation_feedback: Optional[str]
    iteration_count: int
    answer: str
    source_ids: List[str]
