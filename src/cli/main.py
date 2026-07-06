import asyncio
from packages.core_logic.clients import setup_clients, shutdown_clients
from packages.core_logic.graph import app
from packages.core_logic.config import log

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
                # Prepare initial state for the LangGraph flow
                from packages.core_logic.state import AgentState
                state: AgentState = {
                    "user_query": query,
                    "history": conversation_history,
                    "matches": [],
                    "graph_facts": [],
                    "summary": "",
                    "draft_itinerary": "",
                    "validation_feedback": None,
                    "iteration_count": 0,
                    "answer": "",
                    "source_ids": []
                }

                # Invoke the LangGraph workflow
                result = await app.ainvoke(state, config={"recursion_limit": 15})

                # Display RAG summary generated inside the retrieval stage
                print(f"\n=== Summary ===\n{result.get('summary')}\n========================\n")

                # Display the final validated response
                answer = result.get("answer", "")
                print(f"\n=== Assistant Answer ===\n{answer}\n========================")

                # Add user query and assistant response to history
                conversation_history.append({"role": "user", "content": query})
                conversation_history.append({"role": "assistant", "content": answer})
                
                # Limit history to prevent excessive context size
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
