# src/packages/core_logic/graph.py
from langgraph.graph import StateGraph, END
from packages.core_logic.state import AgentState
from packages.core_logic.nodes import retrieve_node, generate_node, critique_node

def route_critique(state: AgentState):
    # Check if we should loop back or terminate
    if state.get("validation_feedback") is not None:
        return "generate"
    return END

# Initialize the workflow graph
workflow = StateGraph(AgentState)  # type: ignore


# Add node definitions
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("critique", critique_node)

# Setup edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "critique")

# Setup routing edge
workflow.add_conditional_edges(
    "critique",
    route_critique,
    {
        "generate": "generate",
        END: END
    }
)

# Compile into an executable application
app = workflow.compile()
