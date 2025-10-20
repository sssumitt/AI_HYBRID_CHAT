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