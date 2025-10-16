# src/packages/core_logic/config.py
import os
import logging
from dotenv import load_dotenv

# --- Basic Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hybrid_chat")

# --- Load Environment Variables ---
load_dotenv()

# --- Model and Search Config ---
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
VECTOR_DIM = 1536
TOP_K = 5
CACHE_EXPIRATION_SECONDS = 2592000  # 30 days

# --- API Keys and Connection Strings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
UPSTASH_REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN")

# --- Validate Required Environment Variables (Fail Fast) ---
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