# Implementation Plan: Upgrading to Agentic Hybrid RAG

This document outlines the step-by-step migration of the `AI_HYBRID_CHAT` project. The goal is twofold: 
1. Refactor direct OpenAI API calls to use LangChain, making the application LLM-agnostic.
2. Implement LangGraph to orchestrate a self-correcting feedback loop (Agentic Workflow).

---

## Phase 1: Codebase Audit & Structural Alignment

Before writing new code, map out where the current hardcoded logic lives.

* **Locate LLM Calls:** Search through `src/packages/core_logic/` and identify every instance of `openai.ChatCompletion.create` or `openai.AsyncOpenAI()`.
* **Identify Prompts:** Extract all hardcoded string prompts into a dedicated `prompts.py` file to cleanly separate logic from text instructions.
* **Map the Data Flow:** Trace the exact input/output structure of your current linear pipeline: `User Query -> Pinecone IDs -> Neo4j Context -> LLM Prompt -> JSON/Text Output`.
* **Install New Dependencies:** Run `uv pip install langchain langchain-openai langgraph` to ensure the required packages are available.

---

## Phase 2: LangChain Refactoring (LLM Agnostic)

Replace native OpenAI calls with LangChain's unified interfaces. This allows you to swap OpenAI for Anthropic, Llama, or local models by simply changing one configuration variable.

* **Create a Model Factory:** In your core logic, build a function to initialize the LLM using `langchain_openai.ChatOpenAI`. Expose configuration parameters (temperature, model name) via `.env`.
* **Refactor Prompts:** Wrap your extracted string prompts in `ChatPromptTemplate.from_messages()`. This standardizes how system and user instructions are injected.
* **Implement LCEL (LangChain Expression Language):** Rewrite the generation function using the `prompt | llm | output_parser` syntax. 
* **Test Linear Parity:** Run the application via the CLI (`python src/cli/main.py`) to confirm the app still works identically to the original version, but is now fully powered by LangChain components.

---

## Phase 3: LangGraph Orchestration (The Feedback Loop)

Transform the linear LCEL chain into a stateful, cyclic graph. This introduces the "guardrails and feedback loops" required by the job description.

* **Define the State (`state.py`):** Create a `TypedDict` that will act as the shared memory for your graph. It should track variables like `user_query`, `retrieved_context`, `draft_itinerary`, `validation_feedback`, and `iteration_count`.
* **Build the Nodes (`nodes.py`):** Wrap your existing logic into distinct functions (nodes) that accept and return the State.
    * `retrieve_node`: Fetches data from Pinecone and Neo4j.
    * `generate_node`: Uses the LangChain LLM to draft the itinerary.
    * `critique_node`: Uses the LLM (or a lighter model) to evaluate the draft against constraints (e.g., budget limits, geographical feasibility).
* **Configure the Edges & Routing (`graph.py`):** * Connect `retrieve` to `generate`.
    * Connect `generate` to `critique`.
    * Create a **conditional edge** after `critique`. If the critique passes, route to `END`. If it fails, route back to `generate` with the critique feedback appended to the state.
* **Compile the Graph:** Use `StateGraph` to compile these nodes and edges into an executable application.

---

## Phase 4: Integration & Edge Case Handling

Connect the new LangGraph engine back to your FastAPI backend and secure the loop against infinite cycles.

* **Set a Recursion Limit:** When executing the LangGraph application, set `{"recursion_limit": 3}` to ensure the agent doesn't get stuck in an infinite loop if the critique keeps failing.
* **Update FastAPI Endpoints:** Modify `src/api/app/main.py` to call the compiled LangGraph app (`app.invoke(state)`) instead of the old linear function.
* **Format Frontend Output:** Ensure the final API response correctly extracts the finalized itinerary from the LangGraph State object to pass back to the Next.js frontend.
* **Clean Up:** Remove the old OpenAI-specific code and update the `README.md` to reflect the new Agentic architecture.
