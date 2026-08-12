"""
rag_pipeline.py

Deployable RAG pipeline: retrieval (Pinecone) -> rerank (Cohere) -> generation (OpenAI).
No interactive input() calls — everything is driven by config/environment so this
can run inside a web service (FastAPI, etc.) with no human at a terminal.

Embedding model and Pinecone connection now go through the shared embeddings.py
and vectorstore.py modules (the same ones ingest.py uses) instead of building
separate clients here. This guarantees indexing and querying always agree on
embedding model/dimension — a mismatch between the two silently breaks
retrieval, so there should only ever be one place in the codebase that
decides what "the embedding model" is.

Note: conversational memory (multi-turn follow-ups) was tried and pulled out
for now — each query() / query_naive() call is fully independent, no history.
"""

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
import cohere

from config import settings
from embeddings import get_embedder
from vectorstore import VectorDB


class RerankRAG:
    def __init__(self):
        # Shared embedding model — same source ingest.py uses, so query-time
        # embeddings always match what was used to build the index.
        self.embed = get_embedder()

        # Shared Pinecone connection logic. connect_to_index() assumes the
        # index already exists (built by ingest.py) — it does not create one.
        vector_db = VectorDB()
        index = vector_db.connect_to_index(settings.PINECONE_INDEX_NAME)
        self.vectorstore = PineconeVectorStore(index=index, embedding=self.embed)

        self.co = cohere.Client(api_key=settings.COHERE_API_KEY)

        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
        )

    def _generate(self, question: str, docs: list) -> dict:
        """
        Shared generation step: stuff retrieved/reranked docs into a
        prompt and get an answer from the LLM. Used by both query()
        and query_naive() so the prompt logic stays in one place.
        """
        context = "\n\n".join(d.page_content for d in docs)
        prompt = (
            "Answer the question using only the context below. "
            "If the context doesn't contain the answer, say so — don't make things up.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        response = self.llm.invoke(prompt)
        return {
            "answer": response.content,
            # PineconeVectorStore pulls "text" out of metadata and into
            # page_content when building the Document, so it's no longer
            # present in d.metadata by this point. Re-attach it explicitly
            # so downstream consumers (like run_eval.py's chunk logging)
            # can see the actual retrieved text, not just title/section.
            "sources": [{**d.metadata, "text": d.page_content} for d in docs],
        }

    def query_naive(self, question: str, k: int = None) -> dict:
        """
        Naive RAG: retrieve -> generate. No reranking, no dedup.
        Plain vector similarity search only. Useful as a baseline
        comparison against query() (rerank-enhanced) results.
        """
        k = k or settings.RETRIEVAL_K
        docs = self.vectorstore.similarity_search(question, k=k)

        if not docs:
            return {"answer": "I couldn't find any relevant documents.", "sources": []}

        return self._generate(question, docs)

    def query(self, question: str, k: int = None, top_n: int = None) -> dict:
        """
        Run the full retrieve -> rerank -> generate pipeline for a single question.

        Args:
            question: the user's question
            k: number of docs to retrieve before reranking (defaults to config)
            top_n: number of docs to keep after reranking (defaults to config)

        Returns:
            dict with 'answer' (str) and 'sources' (list of metadata dicts,
            each including 'title' — the article title — and 'section' —
            the sub-heading the chunk came from, per section_parser.py)
        """
        k = k or settings.RETRIEVAL_K
        top_n = top_n or settings.RERANK_TOP_N

        # 1. Retrieve — wide candidate pool via vector similarity
        docs = self.vectorstore.similarity_search(question, k=k)

        if not docs:
            return {"answer": "I couldn't find any relevant documents.", "sources": []}

        # 2. Rerank — Cohere expects plain text per document, not a dict
        # bundled with unrelated metadata fields. Pulling out just the
        # text avoids relying on Cohere silently ignoring extra keys.
        # Ask Cohere to rank ALL candidates (not just top_n) so we have
        # enough ranked results left to dedupe from below.
        texts = [d.page_content for d in docs]
        reranked = self.co.rerank(
            query=question,
            documents=texts,
            top_n=len(texts),
            model=settings.RERANK_MODEL,
        )

        # Walk the full reranked list (best score first) and keep up to
        # max_per_title chunks per unique article (metadata["title"] —
        # the article title, not to be confused with metadata["section"],
        # the sub-heading within it). A hard cap of 1 can silently drop
        # genuinely complementary content from long survey articles that
        # cover multiple angles — both relevant, from the same article.
        # A cap of 2 balances that against one article flooding the results.
        max_per_title = 2
        title_counts = {}
        top_docs = []
        for hit in reranked.results:
            doc = docs[hit.index]
            title = doc.metadata.get("title")
            if title_counts.get(title, 0) >= max_per_title:
                continue
            title_counts[title] = title_counts.get(title, 0) + 1
            top_docs.append(doc)
            if len(top_docs) >= top_n:
                break

        # 3. Generate — stuff reranked docs into a prompt as context
        return self._generate(question, top_docs)


# Simple manual test — run `python rag_pipeline.py` locally (never in production)
if __name__ == "__main__":
    rag = RerankRAG()
    result = rag.query("What does Socrates think about death?")
    print("\nANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(" -", s.get("title", "unknown"), "|", s.get("section", ""), "|", s.get("source", ""))