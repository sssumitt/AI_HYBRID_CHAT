import asyncio
from packages.core_logic.clients import setup_clients, shutdown_clients
from packages.core_logic.rag_pipeline import (
    pinecone_query,
    fetch_graph_context,
    search_summary,
    call_chat,
)
from packages.core_logic.llm_prompts import build_prompt_with_history
from packages.core_logic.config import log, TOP_K

MAX_HISTORY_MESSAGES = 10

async def interactive_chat():
    """Main interactive command-line interface loop with conversation history."""
    await setup_clients()
    log.info("Hybrid travel assistant is ready. Type 'exit' to quit.")

    # Initialize an empty list to store the conversation history
    conversation_history = []

    try:
        while True:
            query = await asyncio.to_thread(input, "\nEnter your travel question: ")
            if not query or query.lower() in ("exit", "quit"):
                break

            try:
                # --- The RAG pipeline is now a single conceptual block ---
                
                # Retrieve
                matches = await pinecone_query(query, top_k=TOP_K)
                match_ids = [m.get("id") for m in matches if m.get("id")]
                graph_facts = await fetch_graph_context(match_ids)
                
                # Augment
                summary = await search_summary(query, matches, graph_facts)
                
                # Generate (now with history)
                prompt = build_prompt_with_history(query, summary, conversation_history)

                print(f"\n=== Summary ===\n{summary}\n========================\n")

                answer = await call_chat(prompt)

                # --- Update and Display ---

                print(f"\n=== Assistant Answer ===\n{answer}\n========================")

                
                # Add the user's query and the assistant's answer to the history
                conversation_history.append({"role": "user", "content": query})
                conversation_history.append({"role": "assistant", "content": answer})
                
                # Add a limit to history (cost effective)
                conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]

            except Exception as e:
                log.exception("An error occurred during the RAG pipeline: %s", e)
                print("Sorry, an error occurred. Please try again.")

    finally:
        await shutdown_clients()

if __name__ == "__main__":
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        log.info("\nExiting gracefully.")
