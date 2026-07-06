# src/packages/core_logic/rag_pipeline.py
import json
import asyncio
from typing import List, Dict, Any

# Import the clients module directly to access its live state
from packages.core_logic import clients
from packages.core_logic.config import *
from packages.core_logic.utils import _cache_key_for_text, with_retries, truncate
from packages.core_logic.llm_prompts import create_summary_prompt_content
from packages.core_logic.model_factory import get_chat_model, get_embeddings_model
from packages.core_logic.prompts import SUMMARY_PROMPT
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore

async def embed_text(text: str) -> List[float]:
    assert clients.aredis is not None
    key = _cache_key_for_text(text)
    if cached := await clients.aredis.get(key):
        if isinstance(cached, (bytes, bytearray)):
            cached = cached.decode("utf-8")
        try:
            return json.loads(cached)
        except Exception:
            log.warning("Failed to parse cached embedding; will regenerate.")

    embeddings_model = get_embeddings_model()
    embedding = await with_retries(embeddings_model.aembed_query, text)
    if len(embedding) != VECTOR_DIM:
        raise RuntimeError(f"Embedding dimension mismatch: {len(embedding)} != expected {VECTOR_DIM}")
    await clients.aredis.set(key, json.dumps(embedding), ex=CACHE_EXPIRATION_SECONDS)
    return embedding

async def pinecone_query(query_text: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    assert clients.index is not None
    vec = await embed_text(query_text)
    
    embeddings_model = get_embeddings_model()
    vectorstore = PineconeVectorStore(
        index=clients.index,
        embedding=embeddings_model
    )
    
    res = await with_retries(vectorstore.asimilarity_search_by_vector_with_score, embedding=vec, k=top_k)
    
    normalized = []
    for doc, score in res:
        meta = dict(doc.metadata)
        # Preserve backwards compatibility for matches description field
        desc = doc.page_content or meta.get("description") or ""
        meta["description"] = desc
        
        normalized.append({
            "id": doc.metadata.get("id") or getattr(doc, "id", None),
            "score": score,
            "metadata": meta
        })
    log.info(f"Pinecone query returned {len(normalized)} matches for query: {query_text!r}")
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
    assert clients.driver is not None
    async with clients.driver.session() as session:
        result = await session.run(q, node_ids=node_ids)
        facts = [record.data() async for record in result]
    log.info("Graph query returned %d facts", len(facts))
    return facts

async def search_summary(user_query: str, matches: List[Dict[str, Any]], facts: List[Dict[str, Any]]) -> str:
    vec_context_str = "\n".join(
        [
            f"- Name: {truncate(m.get('metadata', {}).get('name', ''), 80)}, Description: {truncate(m.get('metadata', {}).get('description', ''), 300)} (id: {m.get('id')})"
            for m in (matches or [])[:TOP_K]
        ]
    )
    graph_context_str = "\n".join(
        [
            f"- {truncate(f.get('source_name', 'N/A'), 80)} {f.get('rel', 'related to')} {truncate(f.get('target_name', 'N/A'), 120)}"
            for f in (facts or [])[:120]
        ]
    )

    model = get_chat_model(temperature=0.1)
    summary_chain = SUMMARY_PROMPT | model | StrOutputParser()
    return await summary_chain.ainvoke({
        "user_query": truncate(user_query, 800),
        "vec_context": vec_context_str,
        "graph_context": graph_context_str
    })

async def call_chat(prompt_messages: List[Dict[str, str]]) -> str:
    messages = []
    for msg in prompt_messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
            
    model = get_chat_model(temperature=0.2)
    resp = await model.ainvoke(messages)
    return str(resp.content)