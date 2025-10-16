# src/packages/core_logic/clients.py
import asyncio
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import AsyncGraphDatabase, AsyncDriver
from upstash_redis.asyncio import Redis

# Import all config variables
from packages.core_logic.config import *
from packages.core_logic.utils import _close_if_callable

# --- Global Client Variables ---
aclient: AsyncOpenAI = None
pc: Pinecone = None
aredis: Redis = None
index = None
driver: AsyncDriver = None

async def setup_clients():
    global aclient, pc, aredis, index, driver
    log.info("Initializing clients...")
    aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    aredis = Redis(url=UPSTASH_REDIS_URL, token=UPSTASH_REDIS_TOKEN)

    try:
        existing_indexes_obj = await asyncio.to_thread(pc.list_indexes)
    except Exception as e:
        log.warning("pc.list_indexes() failed: %s", e)
        existing_indexes_obj = []
    
    existing = []
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
        try:
            await asyncio.to_thread(
                pc.create_index,
                name=PINECONE_INDEX_NAME,
                dimension=VECTOR_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                log.info("Index already exists (race tolerated).")
            else:
                log.exception("Failed to create index: %s", e)
                raise
    
    index = pc.Index(PINECONE_INDEX_NAME)
    
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        keep_alive=True,
        max_connection_lifetime=300,
    )
    log.info("Clients initialized successfully.")

async def shutdown_clients():
    log.info("Shutting down clients...")
    await _close_if_callable(driver)
    await _close_if_callable(aredis)
    await _close_if_callable(aclient)
    log.info("All connections closed. Goodbye!")