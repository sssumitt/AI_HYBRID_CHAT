# src/packages/core_logic/model_factory.py
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from packages.core_logic.config import CHAT_MODEL, EMBED_MODEL, GEMINI_API_KEY, OPENAI_API_KEY

def get_chat_model(temperature: float = 0.2) -> BaseChatModel:
    """
    Returns a model-agnostic chat model instance (Gemini or OpenAI) based on CHAT_MODEL.
    """
    if "gpt" in CHAT_MODEL.lower() or "o1" in CHAT_MODEL.lower() or "o3" in CHAT_MODEL.lower():
        return ChatOpenAI(
            model=CHAT_MODEL,
            temperature=temperature,
            api_key=OPENAI_API_KEY  # type: ignore
        )
    else:
        return ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            temperature=temperature,
            google_api_key=GEMINI_API_KEY
        )

def get_embeddings_model() -> Embeddings:
    """
    Returns a model-agnostic embeddings model instance (Gemini or OpenAI) based on EMBED_MODEL.
    """
    if "text-embedding" in EMBED_MODEL.lower() and "google" not in EMBED_MODEL.lower():
        return OpenAIEmbeddings(
            model=EMBED_MODEL,
            api_key=OPENAI_API_KEY  # type: ignore
        )
    else:
        return GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=GEMINI_API_KEY  # type: ignore
        )
