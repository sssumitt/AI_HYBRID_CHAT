# src/packages/core_logic/utils.py
import asyncio
import hashlib
from typing import Callable

# Use absolute imports for clarity and robustness
from packages.core_logic.config import log, EMBED_MODEL

def _cache_key_for_text(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"embed:v1:{EMBED_MODEL}:{h}"

def truncate(s: str, n: int = 600) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

async def _close_if_callable(obj: object):
    if obj is None:
        return
    for name in ("aclose", "close"):
        fn = getattr(obj, name, None)
        if callable(fn):
            res = fn()
            if asyncio.iscoroutine(res):
                await res
            return

async def with_retries(fn: Callable[..., asyncio.Future], *args, retries: int = 3, base_delay: float = 0.5, backoff: float = 2.0, **kwargs):
    last_exc = None
    for attempt in range(retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            wait = base_delay * (backoff**attempt)
            log.warning("Transient error on attempt %d/%d: %s — retrying in %.2fs", attempt + 1, retries, exc, wait)
            await asyncio.sleep(wait)
    log.error("All retries failed: %s", last_exc)
    raise last_exc