# load_to_neo4j.py
import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver, exceptions
from pathlib import Path

load_dotenv()

# -------------
# Configuration
# -------------
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "vietnam_travel_dataset.json")

# -------------
# Cypher Functions
# -------------
def create_constraints(tx):
    """Ensure uniqueness constraint on Entity(id)."""
    print("Ensuring uniqueness constraint on :Entity(id) exists...")
    query = "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE"
    tx.run(query)

def load_nodes_batch(tx, nodes_batch):
    """Loads all nodes into Neo4j using a single, efficient UNWIND batch."""
    print(f"Importing {len(nodes_batch)} nodes in a single transaction...")
    query = """
    UNWIND $nodes AS node_props
    MERGE (e:Entity {id: node_props.id})
    SET e += apoc.map.clean(node_props, ['connections'], [])
    WITH e, node_props
    CALL apoc.create.addLabels(e, [node_props.type]) YIELD node
    RETURN count(node) AS imported_count
    """
    result = tx.run(query, nodes=nodes_batch)
    summary = result.consume()
    print(f" Created {summary.counters.nodes_created} nodes.")

def load_relationships_batch(tx, relationships_batch):
    """
    Loads all relationships into Neo4j using a single UNWIND batch.
    """
    print(f"Importing {len(relationships_batch)} relationships in a single transaction...")
    query = """
    UNWIND $rels AS rel_data
    MATCH (source:Entity {id: rel_data.source_id})
    MATCH (target:Entity {id: rel_data.target_id})
    CALL apoc.create.relationship(source, rel_data.type, {}, target) YIELD rel
    RETURN count(rel) AS imported_count
    """
    result = tx.run(query, rels=relationships_batch)
    summary = result.consume()
    print(f"Created {summary.counters.relationships_created} relationships.")

# -------------
# Main Execution
# -------------

def main():
    """Main function to connect to Neo4j and run the import."""
    driver = None
    try:
        # Establish connection to the database
        driver: Driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")

        # Load the dataset from the JSON file
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading or parsing data file: {e}")
            return

        with driver.session(database="neo4j") as session:
            # 1. Ensure constraints are set
            session.execute_write(create_constraints)

            # 2. Load all nodes in ONE efficient transaction
            session.execute_write(load_nodes_batch, all_data)

            # 3. Prepare a flat list of all relationships
            all_relationships = []
            for node in all_data:
                source_id = node.get("id")
                for conn in node.get("connections", []):
                    target_id = conn.get("target")
                    rel_type = conn.get("relation", "RELATED_TO")
                    if source_id and target_id and rel_type:
                        all_relationships.append({
                            "source_id": source_id,
                            "target_id": target_id,
                            "type": rel_type
                        })

            # 4. Load all relationships in ONE efficient transaction
            if all_relationships:
                session.execute_write(load_relationships_batch, all_relationships)

        print("\n Data import complete!")

    except exceptions.AuthError as e:
        print(f"Authentication failed: {e}. Please check your .env file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if driver:
            driver.close()
            print("\nConnection to Neo4j closed.")
            
# ----------------------------
if __name__ == "__main__":
    main()