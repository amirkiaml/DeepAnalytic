"""
config.py

Central config loaded from environment variables / .env file.
No secrets live in code — only here as os.getenv() lookups.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory


class Settings:
    # --- API keys (set these in your .env file, never hardcode) ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY")

    # --- Pinecone ---
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "sep-rag-index")

    # Which namespace within the index to query. Namespaces are hard
    # partitions: a query against one cannot see vectors in another. The
    # full 1,803-article corpus lives in "articles-full"; the original
    # 100-article test slice is in the default namespace (empty string),
    # kept intact so the two remain comparable. Everything that queries
    # the index should route through this setting rather than defaulting
    # to the old test slice by accident.
    PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "articles-full")

    # --- Models ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "rerank-english-v3.0")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

    # Decomposition (query-generation) model, deliberately separate from
    # LLM_MODEL above. Splitting a question into sub-questions is a much
    # simpler task than actually answering one grounded in retrieved
    # context, so it doesn't need the same model doing final generation —
    # a smaller/cheaper model is a reasonable default here, and this stays
    # a distinct setting so the two can be tuned independently as pricing
    # or quality needs change.
    DECOMPOSITION_MODEL: str = os.getenv("DECOMPOSITION_MODEL", "gpt-4o-mini")

    # --- Retrieval params ---
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "10"))
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))


settings = Settings()

# Fail fast if required secrets are missing — better to crash on startup
# than fail confusingly mid-request.
_required = {
    "OPENAI_API_KEY": settings.OPENAI_API_KEY,
    "PINECONE_API_KEY": settings.PINECONE_API_KEY,
    "COHERE_API_KEY": settings.COHERE_API_KEY,
}
_missing = [k for k, v in _required.items() if not v]
if _missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(_missing)}. "
        f"Set them in a .env file or your environment before running."
    )
