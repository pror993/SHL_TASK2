import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_KEY = os.getenv("QDRANT_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "").strip()
RERANKER_ENABLED = _env_bool("RERANKER_ENABLED", True)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "").strip()
