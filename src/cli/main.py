# src/cli/main.py
import asyncio
from packages.core_logic.clients import setup_clients, shutdown_clients
from packages.core_logic.rag_pipeline import (
    pinecone_query,
    fetch_graph_context,
    search_summary,
    call_chat,
)
from packages.core_logic.llm_prompts import build_prompt
from packages.core_logic.config import log, TOP_K

async def interactive_chat():
    """Main interactive command-line interface loop."""
    await setup_clients()
    log.info("Hybrid travel assistant is ready. Type 'exit' to quit.")

    try:
        while True:
            query = await asyncio.to_thread(input, "\nEnter your travel question: ")
            if not query or query.lower() in ("exit", "quit"):
                break

            try:
                matches = await pinecone_query(query, top_k=TOP_K)
                match_ids = [m.get("id") for m in matches if m.get("id")]
                graph_facts = await fetch_graph_context(match_ids)
                summary = await search_summary(query, matches, graph_facts)
                prompt = build_prompt(query, summary)
                answer = await call_chat(prompt)

                print("\n=== Assistant Answer ===\n")
                print(answer)
                print("\n========================\n")

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