# src/packages/core_logic/rag_pipeline.py
import json
import asyncio
from typing import List, Dict, Any

# Import the clients module directly to access its live state
from packages.core_logic import clients
from packages.core_logic.config import *
from packages.core_logic.utils import _cache_key_for_text, with_retries
from packages.core_logic.llm_prompts import create_summary_prompt_content

async def embed_text(text: str) -> List[float]:
    key = _cache_key_for_text(text)
    if cached := await clients.aredis.get(key):
        if isinstance(cached, (bytes, bytearray)):
            cached = cached.decode("utf-8")
        try:
            return json.loads(cached)
        except Exception:
            log.warning("Failed to parse cached embedding; will regenerate.")

    resp = await with_retries(clients.aclient.embeddings.create, model=EMBED_MODEL, input=[text])
    embedding = resp.data[0].embedding
    if len(embedding) != VECTOR_DIM:
        raise RuntimeError(f"Embedding dimension mismatch: {len(embedding)} != expected {VECTOR_DIM}")
    await clients.aredis.set(key, json.dumps(embedding), ex=CACHE_EXPIRATION_SECONDS)
    return embedding

async def pinecone_query(query_text: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    vec = await embed_text(query_text)
    res = await with_retries(asyncio.to_thread, clients.index.query, vector=vec, top_k=top_k, include_metadata=True)
    
    matches = []
    try:
        matches = getattr(res, "matches", None) or (res.get("matches") if isinstance(res, dict) else None) or []
    except Exception:
        try:
            matches = list(res.matches)
        except Exception:
            matches = []
    
    normalized = []
    for m in matches:
        if isinstance(m, dict):
            normalized.append(m)
        else:
            try:
                d = {
                    "id": getattr(m, "id", None),
                    "score": getattr(m, "score", None),
                    "metadata": getattr(m, "metadata", {}),
                }
                normalized.append(d)
            except Exception:
                continue
    return normalized

async def fetch_graph_context(node_ids: List[str]) -> List[Dict[str, Any]]:
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
    async with clients.driver.session() as session:
        result = await session.run(q, node_ids=node_ids)
        facts = [record.data() async for record in result]
    log.debug("Graph query returned %d facts", len(facts))
    return facts

async def search_summary(user_query: str, matches: List[Dict[str, Any]], facts: List[Dict[str, Any]]) -> str:
    summary_prompt_content = create_summary_prompt_content(user_query, matches, facts)
    resp = await with_retries(
        clients.aclient.chat.completions.create,
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": summary_prompt_content}],
        max_tokens=350,
        temperature=0.1
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return getattr(resp, "text", str(resp))

async def call_chat(prompt_messages: List[Dict[str, str]]) -> str:
    resp = await with_retries(
        clients.aclient.chat.completions.create,
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=600,
        temperature=0.2
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return getattr(resp, "text", str(resp))