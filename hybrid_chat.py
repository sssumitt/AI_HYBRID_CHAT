# hybrid_chat.py
import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Callable
import hashlib

from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import AsyncGraphDatabase, AsyncDriver
from upstash_redis.asyncio import Redis

# -----------------------------
# Basic logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hybrid_chat")

# Load environment variables
load_dotenv()

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
VECTOR_DIM = 1536
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
UPSTASH_REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN")
CACHE_EXPIRATION_SECONDS = 2592000  # 30 days

# -----------------------------
# Validate required env vars (fail fast)
# -----------------------------
_required_env = {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
    "NEO4J_URI": NEO4J_URI,
    "UPSTASH_REDIS_URL": UPSTASH_REDIS_URL,
    "UPSTASH_REDIS_TOKEN": UPSTASH_REDIS_TOKEN,
}
_missing = [k for k, v in _required_env.items() if not v]
if _missing:
    raise RuntimeError(f"Missing required environment variables: {_missing}")

# -----------------------------
# Global Client Variables (initialized in setup)
# -----------------------------
aclient: AsyncOpenAI = None
pc: Pinecone = None
aredis: Redis = None
index = None  # pinecone index wrapper
driver: AsyncDriver = None

# -----------------------------
# Utility helpers
# -----------------------------
def _cache_key_for_text(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"embed:v1:{EMBED_MODEL}:{h}"

def truncate(s: str, n: int = 600) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

async def _close_if_callable(obj: object):
    """
    Safely close/aclosed objects which may expose either close() or aclose().
    """
    if obj is None:
        return
    for name in ("aclose", "close"):
        fn = getattr(obj, name, None)
        if callable(fn):
            res = fn()
            if asyncio.iscoroutine(res):
                await res
            return

# Callable[[ArgTypes...], ReturnType]-> fn that is callable and takes ArgTypes and return ReturnType -> can be awaited
async def with_retries(fn: Callable[..., asyncio.Future], *args, retries: int = 3, base_delay: float = 0.5, backoff: float = 2.0, **kwargs):
    """
    Generic retry helper for async callables.
    fn should be an async callable or a function returning an awaitable (like asyncio.to_thread calls).
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            wait = base_delay * (backoff ** attempt)
            log.warning("Transient error on attempt %d/%d: %s — retrying in %.2fs", attempt + 1, retries, exc, wait)
            await asyncio.sleep(wait)
    log.error("All retries failed: %s", last_exc)
    raise last_exc

# -----------------------------
# Async Initialization & Shutdown
# -----------------------------
async def setup_clients():
    """Initializes clients and ensures the Pinecone index exists."""
    global aclient, pc, aredis, index, driver
    log.info("Initializing clients...")
    # OpenAI client
    aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Pinecone client (sync-style SDK wrapped with asyncio.to_thread where needed)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Redis (Upstash asyncio client)
    aredis = Redis(url=UPSTASH_REDIS_URL, token=UPSTASH_REDIS_TOKEN)

    # Pinecone index existence - tolerant to SDK return shapes
    try:
        existing_indexes_obj = await asyncio.to_thread(pc.list_indexes)
    except Exception as e:
        # Fallback: try again without awaiting (shouldn't normally happen)
        log.warning("pc.list_indexes() failed: %s", e)
        existing_indexes_obj = []
    existing = []
    # Normalize different shapes
    if isinstance(existing_indexes_obj, dict):
        existing = [i.get("name") for i in existing_indexes_obj.get("indexes", []) if isinstance(i, dict)]
    elif hasattr(existing_indexes_obj, "indexes"):
        try:
            existing = [getattr(i, "name", i) for i in existing_indexes_obj.indexes]
        except Exception:
            existing = list(existing_indexes_obj.indexes)
    elif isinstance(existing_indexes_obj, (list, tuple)):
        existing = list(existing_indexes_obj)
    else:
        try:
            existing = list(existing_indexes_obj)
        except Exception:
            existing = []

    if PINECONE_INDEX_NAME not in existing:
        log.info("Creating managed index: %s", PINECONE_INDEX_NAME)
        # Wrap create_index in to_thread to avoid blocking
        try:
            await asyncio.to_thread(
                pc.create_index,
                name=PINECONE_INDEX_NAME,
                dimension=VECTOR_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as e:
            # If a race caused the index to be created concurrently, tolerate it
            if "already exists" in str(e).lower():
                log.info("Index already exists (race tolerated).")
            else:
                log.exception("Failed to create index: %s", e)
                raise

    # Create an index handle
    index = pc.Index(PINECONE_INDEX_NAME)

    # Neo4j async driver
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        keep_alive=True,
        max_connection_lifetime=300,
    )

    log.info("Clients initialized successfully.")

async def shutdown_clients():
    """Close all clients gracefully."""
    log.info("Shutting down clients...")
    try:
        if driver:
            await driver.close()
    except Exception as e:
        log.warning("Error closing Neo4j driver: %s", e)

    await _close_if_callable(aredis)
    await _close_if_callable(aclient)
    # Pinecone client typically doesn't require explicit close in many SDK versions
    log.info("All connections closed. Goodbye!")

# -----------------------------
# Core functions
# -----------------------------
async def embed_text(text: str) -> List[float]:
    """Generates an embedding, using Redis for caching and retries."""
    key = _cache_key_for_text(text)
    cached = await aredis.get(key)
    if cached:
        if isinstance(cached, (bytes, bytearray)):
            cached = cached.decode("utf-8")
        try:
            return json.loads(cached)
        except Exception:
            log.warning("Failed to parse cached embedding; will regenerate.")

    async def _call_embed():
        return await aclient.embeddings.create(model=EMBED_MODEL, input=[text])

    resp = await with_retries(_call_embed, retries=3)
    embedding = resp.data[0].embedding
    if len(embedding) != VECTOR_DIM:
        raise RuntimeError(f"Embedding dimension mismatch: {len(embedding)} != expected {VECTOR_DIM}")
    await aredis.set(key, json.dumps(embedding), ex=CACHE_EXPIRATION_SECONDS)
    return embedding

async def pinecone_query(query_text: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Query Pinecone index using embedding (safe access + retry)."""
    vec = await embed_text(query_text)

    async def _query():
        # index.query is often sync; run in thread
        return await asyncio.to_thread(index.query, vector=vec, top_k=top_k, include_metadata=True)

    res = await with_retries(_query, retries=3)

    # Normalize response shapes
    matches = []
    try:
        matches = getattr(res, "matches", None) or (res.get("matches") if isinstance(res, dict) else None) or []
    except Exception:
        try:
            matches = list(res.matches)
        except Exception:
            matches = []

    # Defensive: ensure each match is a dict-ish
    normalized = []
    for m in matches:
        if isinstance(m, dict):
            normalized.append(m)
        else:
            # try attribute access
            try:
                d = {
                    "id": getattr(m, "id", None) or m.get("id") if isinstance(m, dict) else None,
                    "score": getattr(m, "score", None),
                    "metadata": getattr(m, "metadata", None) or (m.metadata if hasattr(m, "metadata") else None) or (m.get("metadata") if isinstance(m, dict) else {}),
                }
                normalized.append(d)
            except Exception:
                continue
    return normalized

async def fetch_graph_context(node_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch neighbor data from Neo4j for multiple nodes in a single query."""
    if not node_ids:
        return []

    q = (
        "UNWIND $node_ids AS nid "
        "MATCH (n:Entity {id:nid})-[r]-(m:Entity) "
        "WITH DISTINCT n, r, m "
        "RETURN n.id AS source_id, n.name AS source_name, type(r) AS rel, "
        "m.id AS target_id, m.name AS target_name, m.description AS target_desc "
        "LIMIT 200"
    )
    facts = []
    async with driver.session() as session:
        result = await session.run(q, node_ids=node_ids)
        async for record in result:
            facts.append(
                {
                    "source_id": record.get("source_id"),
                    "source_name": record.get("source_name"),
                    "rel": record.get("rel"),
                    "target_id": record.get("target_id"),
                    "target_name": record.get("target_name"),
                    "target_desc": record.get("target_desc") or "",
                }
            )
    log.debug("Graph query returned %d facts", len(facts))
    return facts

# -----------------------------
# Prompting / LLM wrappers
# -----------------------------
async def search_summary(user_query: str, matches: List[Dict[str, Any]], facts: List[Dict[str, Any]]) -> str:
    """Uses an LLM to summarize the combined search results."""
    # Build compact context with truncation & limits to avoid token explosion
    vec_context_str = "\n".join(
        [
            f"- Name: {truncate(m.get('metadata', {}).get('name', ''), 80)}, Description: {truncate(m.get('metadata', {}).get('description', ''), 300)} (id: {m.get('id')})"
            for m in (matches or [])[: TOP_K]
        ]
    )
    graph_context_str = "\n".join(
        [
            f"- {truncate(f.get('source_name', 'N/A'),80)} {f.get('rel','related to')} {truncate(f.get('target_name','N/A'),120)}"
            for f in (facts or [])[: 120]
        ]
    )

    summary_prompt = (
        "You are a data synthesizer for a travel assistant. Your task is to process raw data and create a clean summary. Follow these steps:\n"
        "1. If a result has a generic name (e.g., 'Attraction 123'), you may invent a plausible descriptive name using its description (e.g., 'Golden Hand Bridge'). Mark invented names with [inferred].\n"
        "2. Based on the user's query and ALL provided data, create a concise summary of the most relevant places and their relationships.\n"
        "3. Format the summary as a few bullet points. Always include the original node IDs in parentheses.\n\n"
        f"User Query: \"{truncate(user_query, 800)}\"\n\n"
        f"Top Search Results:\n{vec_context_str}\n\n"
        f"Knowledge Graph Facts:\n{graph_context_str}\n\n"
        "Concise Summary (in bullet points, using improved names where needed):"
    )

    async def _call():
        return await aclient.chat.completions.create(
            model=CHAT_MODEL, messages=[{"role": "user", "content": summary_prompt}], max_tokens=350, temperature=0.1
        )

    resp = await with_retries(_call, retries=3)
    try:
        return resp.choices[0].message.content
    except Exception:
        # best-effort fallback
        text = getattr(resp, "text", None) or json.dumps(resp)
        return str(text)

def build_prompt(user_query: str, summary: str) -> List[Dict[str, str]]:
    """Builds the final chat prompt using a Chain-of-Thought approach."""
    system = (
        "You are an expert travel agent. Your goal is to create a clear, actionable, and LOGISTICALLY REALISTIC travel itinerary. "
        "Prioritize the user's enjoyment and avoid excessive travel time on short trips. The final output must be clean and user-facing."
    )

    user_prompt = (
        f"User query: {truncate(user_query, 800)}\n\n"
        f"I have found a summary of relevant information:\n{truncate(summary, 3000)}\n\n"
        "Please follow these two steps to answer the user's query:\n\n"
        "**Step 1: Reasoning & Feasibility Check (Internal Thought).** "
        "First, think step-by-step (2-4 sentences). Analyze the summary and evaluate logistics. If the places are in distant cities, recommend focusing on one region for short trips.\n\n"
        "**Step 2: Final Itinerary (User-Facing Answer).** "
        "Based ONLY on your reasoning in Step 1, create a concise 2-3 step travel itinerary. Write in a friendly and helpful tone. State place names clearly. DO NOT include node IDs or internal reasoning."
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]

async def call_chat(prompt_messages: List[Dict[str, str]]) -> str:
    """Call OpenAI Chat Completion with retries."""
    async def _call():
        return await aclient.chat.completions.create(
            model=CHAT_MODEL, messages=prompt_messages, max_tokens=600, temperature=0.2
        )

    resp = await with_retries(_call, retries=3)
    try:
        return resp.choices[0].message.content
    except Exception:
        return getattr(resp, "text", str(resp))

# -----------------------------
# Interactive chat
# -----------------------------
async def interactive_chat():
    """Main interactive command-line interface loop with graceful startup/shutdown."""
    await setup_clients()

    log.info("Hybrid travel assistant. Type 'exit' to quit.")
    try:
        while True:
            query = await asyncio.to_thread(input, "\nEnter your travel question: ")
            if not query or query.lower() in ("exit", "quit"):
                break

            try:
                matches = await pinecone_query(query, top_k=TOP_K)
            except Exception as e:
                log.exception("Error querying Pinecone: %s", e)
                print("Sorry — an error occurred while searching. Try again.")
                continue

            match_ids = [m.get("id") for m in matches if m.get("id")]

            graph_facts = []
            if match_ids:
                try:
                    graph_facts = await fetch_graph_context(match_ids)
                except Exception as e:
                    log.exception("Error fetching graph context: %s", e)
                    graph_facts = []

            try:
                summary = await search_summary(query, matches, graph_facts)
            except Exception as e:
                log.exception("Error generating summary: %s", e)
                print("Sorry — an error occurred while summarizing results.")
                continue

            prompt = build_prompt(query, summary)
            try:
                answer = await call_chat(prompt)
            except Exception as e:
                log.exception("Error calling chat model: %s", e)
                print("Sorry — an error occurred while generating the itinerary.")
                continue

            print("\n=== Assistant Answer ===\n")
            print(answer)
            print("\n========================\n")
    finally:
        await shutdown_clients()

# -----------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        log.info("Exiting gracefully (keyboard interrupt).")
