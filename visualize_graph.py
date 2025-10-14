# visualize_graph.py
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver, exceptions
from pyvis.network import Network

load_dotenv()

# -------------
# Configuration
# -------------
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO_BATCH_LIMIT = 500  # Max number of relationships to visualize

# -------------
# Core Functions
# -------------
def fetch_subgraph(tx, limit):
    """Fetches a subgraph from Neo4j up to a specified limit."""
    print(f"Fetching up to {limit} relationships from the graph...")
    query = """
    MATCH (a:Entity)-[r]->(b:Entity)
    RETURN a.id AS a_id, labels(a) AS a_labels, a.name AS a_name,
           b.id AS b_id, labels(b) AS b_labels, b.name AS b_name,
           type(r) AS rel
    LIMIT $limit
    """
    result = tx.run(query, limit=limit)
    return list(result) # Return results as a list of records

def build_pyvis_graph(records, output_html="graph_visualization.html"):
    """Builds and saves an interactive pyvis graph from Neo4j records."""
    print("Building interactive graph visualization...")
    net = Network(height="900px", width="100%", notebook=False, directed=True, bgcolor="#222222", font_color="white")

    # Define colors for different node types for better visualization
    color_map = {
        "City": "#FF5733",
        "Attraction": "#33C4FF",
        "Hotel": "#33FF57",
        "Activity": "#F1C40F",
        "Entity": "#99A3A4" # Fallback color
    }

    added_nodes = set() # Use a set for efficient tracking of added nodes

    for rec in records:
        source_id, source_name, source_labels = rec["a_id"], rec["a_name"], rec["a_labels"]
        target_id, target_name, target_labels = rec["b_id"], rec["b_name"], rec["b_labels"]
        rel_type = rec["rel"]

        # Add source node if not already added
        if source_id not in added_nodes:
            primary_label = next((label for label in source_labels if label != 'Entity'), 'Entity')
            color = color_map.get(primary_label, color_map["Entity"])
            label = f"{source_name or source_id}\n({primary_label})"
            net.add_node(source_id, label=label, title=source_name, color=color)
            added_nodes.add(source_id)

        # Add target node if not already added
        if target_id not in added_nodes:
            primary_label = next((label for label in target_labels if label != 'Entity'), 'Entity')
            color = color_map.get(primary_label, color_map["Entity"])
            label = f"{target_name or target_id}\n({primary_label})"
            net.add_node(target_id, label=label, title=target_name, color=color)
            added_nodes.add(target_id)

        # Add the edge connecting the source and target nodes.
        net.add_edge(source_id, target_id, title=rel_type, label=rel_type)

    # Add physics controls to the UI for better interaction
    net.show_buttons(filter_=['physics'])
    net.save_graph(output_html)
    print(f"Saved visualization to '{output_html}'. Open this file in your browser.")

# -------------
# Main Execution
# -------------
def main():
    """Main function to connect to Neo4j and generate the visualization."""
    driver = None
    try:
        driver: Driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")

        with driver.session(database="neo4j") as session:
            records = session.execute_read(fetch_subgraph, limit=NEO_BATCH_LIMIT)

        if not records:
            print("No data returned from Neo4j. Cannot generate visualization.")
            return

        build_pyvis_graph(records)

    except exceptions.AuthError as e:
        print(f"Authentication failed: {e}. Please check your .env file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if driver:
            driver.close()
            print("\nConnection to Neo4j closed.")

# ---------------------------
if __name__ == "__main__":
    main()