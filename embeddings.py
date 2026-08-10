"""
embeddings.py

Single source of truth for the embedding model. Both ingest.py (writes
vectors) and rag_pipeline.py (reads vectors) should import from here —
if indexing and querying ever use different embedding models or
dimensions, retrieval breaks (usually silently, or with a confusing
Pinecone dimension-mismatch error).
"""

from langchain_openai import OpenAIEmbeddings
from config import settings

# Known output dimensions per OpenAI embedding model — needed when
# creating a new Pinecone index (it must be told the vector size upfront).
EMBED_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def get_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )


def get_embedding_dimension() -> int:
    dim = EMBED_DIMENSIONS.get(settings.EMBEDDING_MODEL)
    if dim is None:
        raise ValueError(
            f"Unknown embedding dimension for model '{settings.EMBEDDING_MODEL}'. "
            f"Add it to EMBED_DIMENSIONS in embeddings.py."
        )
    return dim
