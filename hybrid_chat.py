# hybrid_chat.py
import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase, Driver

# Load environment variables
load_dotenv()

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

# Get config from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
VECTOR_DIM = 1536 # text-embedding-3-small

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to Pinecone index using modern syntax
if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    print(f"Creating managed index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1" # Corrected region for AWS
        )
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Connect to Neo4j
driver: Driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str) -> List[float]:
    """Get embedding for a text string."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def pinecone_query(query_text: str, top_k=TOP_K) -> List[Dict[str, Any]]:
    """Query Pinecone index using embedding."""
    vec = embed_text(query_text)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    print(f"DEBUG: Pinecone found {len(res['matches'])} matches.")
    return res["matches"]

def fetch_graph_context(node_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch neighboring nodes from Neo4j using a read transaction."""
    facts = []

    def get_neighbors_tx(tx, nid: str) -> List[Dict[str, Any]]:
        q = (
            "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
            "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
            "m.name AS name, m.type AS type, m.description AS description "
            "LIMIT 10"
        )
        result = tx.run(q, nid=nid)
        return [
            {
                "source": nid,
                "rel": r["rel"],
                "target_id": r["id"],
                "target_name": r["name"],
                "target_desc": (r["description"] or "")[:400],
                "labels": r["labels"]
            }
            for r in result
        ]

    with driver.session() as session:
        for node_id in node_ids:
            # Use session.execute_read for safe, transactional reads
            records = session.execute_read(get_neighbors_tx, nid=node_id)
            facts.extend(records)

    print(f"DEBUG: Graph query returned {len(facts)} facts.")
    return facts

def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a chat prompt combining vector DB matches and graph facts."""
    system = (
        "You are a helpful travel assistant. Use the provided semantic search results "
        "and graph facts to answer the user's query briefly and concisely. "
        "Cite node ids when referencing specific places or attractions."
    )

    vec_context = []
    for m in pinecone_matches:
        meta = m["metadata"]
        score = m.get("score", 0)
        snippet = f"- id: {m['id']}, name: {meta.get('name','')}, type: {meta.get('type','')}, score: {score:.4f}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"User query: {user_query}\n\n"
         "Top semantic matches (from vector DB):\n" + "\n".join(vec_context) + "\n\n"
         "Graph facts (neighboring relations):\n" + "\n".join(graph_context) + "\n\n"
         "Based on the above, answer the user's question. If helpful, suggest 2–3 concrete itinerary steps or tips and mention node ids for references."}
    ]
    return prompt

def call_chat(prompt_messages):
    """Call OpenAI ChatCompletion."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=600,
        temperature=0.2
    )
    return resp.choices[0].message.content

# -----------------------------
# Interactive chat
# -----------------------------
def interactive_chat():
    print("Hybrid travel assistant. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower() in ("exit","quit"):
            driver.close() # Close the driver connection when exiting
            break

        matches = pinecone_query(query, top_k=TOP_K)
        match_ids = [m["id"] for m in matches]
        graph_facts = fetch_graph_context(match_ids)
        prompt = build_prompt(query, matches, graph_facts)
        answer = call_chat(prompt)
        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")
        
# -----------------------------------
if __name__ == "__main__":
    interactive_chat()