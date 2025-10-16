# src/packages/core_logic/llm_prompts.py
from typing import List, Dict, Any
from packages.core_logic.utils import truncate
from packages.core_logic.config import TOP_K

def create_summary_prompt_content(user_query: str, matches: List[Dict[str, Any]], facts: List[Dict[str, Any]]) -> str:
    """Creates the text content for the summary prompt."""
    vec_context_str = "\n".join(
        [
            f"- Name: {truncate(m.get('metadata', {}).get('name', ''), 80)}, Description: {truncate(m.get('metadata', {}).get('description', ''), 300)} (id: {m.get('id')})"
            for m in (matches or [])[:TOP_K]
        ]
    )
    graph_context_str = "\n".join(
        [
            f"- {truncate(f.get('source_name', 'N/A'), 80)} {f.get('rel', 'related to')} {truncate(f.get('target_name', 'N/A'), 120)}"
            for f in (facts or [])[:120]
        ]
    )
    return (
        "You are a data synthesizer for a travel assistant. Your task is to process raw data and create a clean summary. Follow these steps:\n"
        "1. If a result has a generic name (e.g., 'Attraction 123'), you may invent a plausible descriptive name using its description (e.g., 'Golden Hand Bridge'). Mark invented names with [inferred].\n"
        "2. Based on the user's query and ALL provided data, create a concise summary of the most relevant places and their relationships.\n"
        "3. Format the summary as a few bullet points. Always include the original node IDs in parentheses.\n\n"
        f"User Query: \"{truncate(user_query, 800)}\"\n\n"
        f"Top Search Results:\n{vec_context_str}\n\n"
        f"Knowledge Graph Facts:\n{graph_context_str}\n\n"
        "Concise Summary (in bullet points, using improved names where needed):"
    )

def build_prompt(user_query: str, summary: str) -> List[Dict[str, str]]:
    """Builds the final chat prompt using a Chain-of-Thought approach."""
    system = (
        "You are an expert travel agent. Your goal is to create a clear, actionable, and LOGISTICALLY REALISTIC travel itinerary. "
        "Prioritize the user's enjoyment and avoid excessive travel time on short trips. The final output must be clean and user-facing."
    )
    user_prompt = (
        f"User query: {truncate(user_query, 800)}\n\n"
        f"I have found a summary of relevant information:\n{truncate(summary, 3000)}\n\n"
        "Please follow these two steps to answer the user's query:\n\n"
        "**Step 1: Reasoning & Feasibility Check (Internal Thought).** "
        "First, think step-by-step (2-4 sentences). Analyze the summary and evaluate logistics. If the places are in distant cities, recommend focusing on one region for short trips.\n\n"
        "**Step 2: Final Itinerary (User-Facing Answer).** "
        "Based ONLY on your reasoning in Step 1, create a concise 2-3 step travel itinerary. Write in a friendly and helpful tone. State place names clearly. DO NOT include node IDs or internal reasoning."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]