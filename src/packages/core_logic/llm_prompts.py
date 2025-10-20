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

def build_prompt_with_history(user_query: str, summary: str, history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """
    Builds the final chat prompt, incorporating conversation history for context.
    """
    history = history or []

    system_prompt = (
        "You are an expert travel agent AI. Your personality is helpful, knowledgeable, and friendly. "
        "Your primary goal is to create clear, actionable, and **logistically realistic** travel itineraries. "
        "You must prioritize the user's enjoyment by avoiding excessive travel time, especially on short trips. "
        "The final output must be clean, user-facing, and ready to be displayed."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    # Add the current user query with the RAG summary
    user_prompt_content = (
        f"Based on the user's query and the information summary below, please generate a response by following these instructions precisely.\n\n"
        f"**User query:** {truncate(user_query, 800)}\n\n"
        f"**Information Summary:**\n{truncate(summary, 3000)}\n\n"
        "---"
        "\n\n"
        "**Instructions:**\n"
        "Your entire response MUST be structured in two parts, inside `<reasoning>` and `<itinerary>` tags.\n\n"
        "**Part 1: Your Reasoning (Internal Thought Process)**\n"
        "Inside the `<reasoning>` tag, think step-by-step.Each step should be seperated by a new line character.Analyze the user's query, chat history, and the provided summary. "
        "**Crucially, you must evaluate the logistics and travel time between locations.** For trips of 5 days or less, if the main locations are more than a 4-hour drive apart, you MUST recommend focusing on a single region to create a more enjoyable experience. State this conclusion clearly.\n\n"
        "**Part 2: The Final Itinerary (User-Facing Answer)**\n"
        "Inside the `<itinerary>` tag, write the final, user-facing response. This output should be friendly, clear, and based **only** on your conclusions from the `<reasoning>` part. \n"
        "- **Format:** For multi-day plans, use a `Day 1:`, `Day 2:` format. Within each day, use `Morning:`, `Afternoon:`, and `Evening:` when appropriate.\n"
        "- **Pro-Tips:** As an expert, include 1-2 helpful 'pro-tips' (e.g., 'Pro-Tip: Book your tickets online to avoid queues.' or suggesting a specific local dish).\n"
        "- **DO NOT** include node IDs, your internal reasoning, or the tags themselves in this final answer.\n\n"
        "**Example Output Structure:**\n"
        "<reasoning>\n"
        "The user wants a 5-day trip covering both Hoi An and Da Lat. A quick check reveals that travel between these two cities takes over 12 hours by road and there are no direct flights. This is not feasible for a 5-day trip. I will advise the user to focus on Hoi An and the surrounding area to have a more relaxed and enjoyable trip.\n"
        "</reasoning>\n"
        "<itinerary>\n"
        "A 5-day trip covering both Hoi An and Da Lat would be quite rushed due to the long travel time between them. For a more relaxed and romantic experience, I recommend focusing on the beautiful city of Hoi An and its surroundings!\n\n"
        "Here is a suggested 5-day itinerary:\n\n"
        "**Day 1: Arrival and Ancient Town Charm**\n"
        "**Morning:** Arrive in Hoi An, check into your hotel, and take a leisurely stroll to get your bearings.\n"
        "**Afternoon:** Explore the heart of the Ancient Town. Don't miss the iconic Japanese Covered Bridge and the vibrant local markets.\n"
        "**Evening:** Enjoy a romantic dinner. Afterwards, take a scenic lantern boat ride on the Thu Bon River for breathtaking views.\n"
        "**Day 2: ...**\n"
        "</itinerary>"
    )
    
    messages.append({"role": "user", "content": user_prompt_content})

    #  adding fallback to prevent unrelated queries 
    fallback_instruction = (
        "\n\n"
        "**Fallback Rule:**\n"
        "If the user's query is unrelated to travel, trips, or Vietnam, do NOT attempt to answer it. "
        "Instead, politely reply with the following message exactly:\n"
        "\"I'm your Vietnam Travel Assistant 🇻🇳 — I can help you plan trips, itineraries, and travel tips for Vietnam. "
        "Could you please ask me something related to your Vietnam travel plans?\""
    )

    messages[-1]["content"] += fallback_instruction

    return messages