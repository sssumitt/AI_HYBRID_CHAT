# src/packages/core_logic/nodes.py
import json
from packages.core_logic.state import AgentState
from packages.core_logic.model_factory import get_chat_model
from packages.core_logic.prompts import SUMMARY_PROMPT, GENERATION_PROMPT, CRITIQUE_PROMPT
from packages.core_logic.rag_pipeline import pinecone_query, fetch_graph_context
from packages.core_logic.utils import truncate, log
from packages.core_logic.config import TOP_K
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

async def retrieve_node(state: AgentState) -> dict:
    log.info("Executing retrieve_node...")
    user_query = state.get("user_query")
    
    # 1. Vector Search
    matches = await pinecone_query(user_query, top_k=TOP_K)
    match_ids = [str(m.get("id")) for m in matches if m.get("id")]
    
    # 2. Graph Database Search
    graph_facts = await fetch_graph_context(match_ids)
    
    # 3. Context Formatting
    vec_context_str = "\n".join(
        [
            f"- Name: {truncate(m.get('metadata', {}).get('name', ''), 80)}, Description: {truncate(m.get('metadata', {}).get('description', ''), 300)} (id: {m.get('id')})"
            for m in (matches or [])[:TOP_K]
        ]
    )
    graph_context_str = "\n".join(
        [
            f"- {truncate(f.get('source_name', 'N/A'), 80)} {f.get('rel', 'related to')} {truncate(f.get('target_name', 'N/A'), 120)}"
            for f in (graph_facts or [])[:120]
        ]
    )
    
    # 4. Context Synthesis (Summary generation)
    model = get_chat_model(temperature=0.1)
    summary_chain = SUMMARY_PROMPT | model | StrOutputParser()
    
    summary = await summary_chain.ainvoke({
        "user_query": truncate(user_query, 800),
        "vec_context": vec_context_str,
        "graph_context": graph_context_str
    })
    
    log.info("Context retrieval and synthesis completed.")
    return {
        "matches": matches,
        "graph_facts": graph_facts,
        "summary": summary,
        "source_ids": match_ids
    }

async def generate_node(state: AgentState) -> dict:
    log.info("Executing generate_node (iteration: %d)...", state.get("iteration_count", 0) + 1)
    user_query = state.get("user_query")
    summary = state.get("summary")
    history = state.get("history") or []
    validation_feedback = state.get("validation_feedback")
    
    # Format chat history to LangChain Message objects
    history_msgs = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            history_msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            history_msgs.append(AIMessage(content=content))
        elif role == "system":
            history_msgs.append(SystemMessage(content=content))
            
    # Include validation feedback if the previous draft failed
    if validation_feedback:
        feedback_instruction = (
            f"\n\n**CRITICAL FEEDBACK FROM PREVIOUS DRAFT:**\n"
            f"Your previous draft was REJECTED because: {validation_feedback}\n"
            f"Please address this feedback and fix the itinerary in your new response."
        )
    else:
        feedback_instruction = ""
        
    model = get_chat_model(temperature=0.2)
    generation_chain = GENERATION_PROMPT | model | StrOutputParser()
    
    draft_itinerary = await generation_chain.ainvoke({
        "user_query": truncate(user_query, 800),
        "summary": truncate(summary, 3000),
        "history": history_msgs,
        "feedback_instruction": feedback_instruction
    })
    
    iteration_count = state.get("iteration_count", 0) + 1
    log.info("Itinerary drafting completed.")
    return {
        "draft_itinerary": draft_itinerary,
        "iteration_count": iteration_count
    }

async def critique_node(state: AgentState) -> dict:
    log.info("Executing critique_node...")
    user_query = state.get("user_query")
    draft_itinerary = state.get("draft_itinerary")
    iteration_count = state.get("iteration_count", 0)
    
    model = get_chat_model(temperature=0.1)
    critique_chain = CRITIQUE_PROMPT | model | StrOutputParser()
    
    critique_res = await critique_chain.ainvoke({
        "user_query": user_query,
        "draft_itinerary": draft_itinerary
    })
    
    # Parse critique result (expecting JSON format)
    try:
        # Strip potential markdown syntax wrapping
        cleaned_res = critique_res.strip()
        if cleaned_res.startswith("```"):
            cleaned_res = cleaned_res.split("\n", 1)[1]
        if cleaned_res.endswith("```"):
            cleaned_res = cleaned_res.rsplit("\n", 1)[0]
        cleaned_res = cleaned_res.strip()
        if cleaned_res.startswith("json"):
            cleaned_res = cleaned_res[4:].strip()
            
        data = json.loads(cleaned_res)
        passed = data.get("passed", True)
        feedback = data.get("feedback", "")
    except Exception as e:
        log.error("Failed to parse critique output: %s. Raw output: %s", e, critique_res)
        # Fallback to passing if the parser fails to prevent blocking the pipeline
        passed = True
        feedback = ""
        
    log.info("Critique result: passed=%s, feedback=%s", passed, feedback)
    
    # Set final answer if critique passes OR if we've exhausted our maximum iterations (limit: 3)
    if passed or iteration_count >= 3:
        if not passed:
            log.warning("Critique failed but maximum iteration count (%d) reached. Proceeding with best-effort draft.", iteration_count)
        answer = draft_itinerary
        validation_feedback = None
    else:
        answer = ""
        validation_feedback = feedback
        
    return {
        "validation_feedback": validation_feedback,
        "answer": answer
    }
