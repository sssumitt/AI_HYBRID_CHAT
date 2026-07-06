# src/packages/core_logic/config.py
import os
import logging
from dotenv import load_dotenv

# --- Basic Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hybrid_chat")

# --- Load Environment Variables ---
load_dotenv()

# --- LangSmith Tracing ---
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
if LANGCHAIN_TRACING_V2.lower() == "true":
    log.info("LangSmith tracing is enabled. Project: %s", os.getenv("LANGCHAIN_PROJECT", "ai-hybrid-chat"))

# --- Model and Search Config ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1536"))
TOP_K = int(os.getenv("TOP_K", "5"))
CACHE_EXPIRATION_SECONDS = int(os.getenv("CACHE_EXPIRATION_SECONDS", "2592000"))  # 30 days

# --- API Keys and Connection Strings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "")
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL", "")
UPSTASH_REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN", "")

# --- Validate Required Environment Variables (Fail Fast) ---
_required_env = {
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
    "NEO4J_URI": NEO4J_URI,
    "UPSTASH_REDIS_URL": UPSTASH_REDIS_URL,
    "UPSTASH_REDIS_TOKEN": UPSTASH_REDIS_TOKEN,
}

_missing = [k for k, v in _required_env.items() if not v]

# Require keys based on configured models
is_openai_chat = "gpt" in CHAT_MODEL.lower() or "o1" in CHAT_MODEL.lower() or "o3" in CHAT_MODEL.lower()
is_openai_embed = "text-embedding" in EMBED_MODEL.lower() and "google" not in EMBED_MODEL.lower()

if (is_openai_chat or is_openai_embed) and not OPENAI_API_KEY:
    _missing.append("OPENAI_API_KEY")
if (not is_openai_chat or not is_openai_embed) and not GEMINI_API_KEY:
    _missing.append("GEMINI_API_KEY")

if _missing:
    raise RuntimeError(f"Missing required environment variables: {_missing}")