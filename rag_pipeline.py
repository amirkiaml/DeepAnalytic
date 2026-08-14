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
from multiquery import generate_subqueries


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
        prompt and get an answer from the LLM. Used by query(),
        query_naive(), query_multi(), and query_multi_concat() so the
        prompt logic stays in one place.

        Prompt explicitly invites synthesis across separately-stated
        facts (v2, see multiquery_dev_log.md) -- the original wording
        ("if the context doesn't contain the answer, say so") turned out
        to be ambiguous between two very different situations: the
        context genuinely lacking the needed facts, versus the context
        containing every needed fact but never explicitly stating the
        connection between them. Testing found the model collapsing
        those two cases together and declining to answer even when both
        halves of a comparison question were present, clean, and
        uncapped in context -- this rewording is meant to separate them.
        """
        context = "\n\n".join(d.page_content for d in docs)
        prompt = (
            "Answer the question using only the facts in the context below.\n\n"
            "If the question asks you to compare, connect, or relate two or more "
            "things, and the context contains factual information about each of "
            "them separately, construct that comparison yourself using only those "
            "facts -- the source text does not need to have already stated the "
            "comparison explicitly for you to make it. Only say the context "
            "doesn't contain the answer if it is missing the underlying facts "
            "needed, not merely because it doesn't spell out the connection in "
            "so many words.\n\n"
            "Never invent facts that aren't in the context.\n\n"
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

    def _rerank_and_dedup(self, question: str, docs: list, top_n: int) -> list:
        """
        Shared rerank + dedup step, used by both query() and
        query_multi(). Pulled out so multiquery gets the exact same
        max_per_title logic as single-query rerank rather than a second,
        divergent copy of it — this is the same lesson learned earlier
        with embeddings.py/vectorstore.py: one place decides how
        reranking + dedup works, not one copy per call site.

        Args:
            question: the (original) user question, used for reranking relevance
            docs: candidate documents to rerank (may include duplicates
                  across sub-queries when called from query_multi())
            top_n: number of docs to keep after reranking

        Returns:
            list of Document objects, reranked and deduped by max_per_title
        """
        if not docs:
            return []

        texts = [d.page_content for d in docs]
        reranked = self.co.rerank(
            query=question,
            documents=texts,
            top_n=len(texts),
            model=settings.RERANK_MODEL,
        )

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

        return top_docs

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

        # 2. Rerank + dedup — shared with query_multi()
        top_docs = self._rerank_and_dedup(question, docs, top_n)

        # 3. Generate — stuff reranked docs into a prompt as context
        return self._generate(question, top_docs)

    def _rerank_multi_with_floor(self, question: str, subquery_docs: list, top_n: int, min_per_subquery: int, max_per_title: int) -> list:
        """
        Reranked selection for query_multi(), with a guaranteed floor:
        every subquery gets at least min_per_subquery chunks represented
        in the final result, before the remaining budget is filled by
        rerank quality alone. Without this, a global top_n cut across a
        pooled multi-subquery candidate set can easily end up with zero
        chunks from some subqueries — mathematically impossible to
        represent every sub-topic in a 10+-part composite question if
        top_n stays fixed at a small default regardless of how many
        subqueries were generated.

        Args:
            question: original user question (reranked against this, not any subquery)
            subquery_docs: list of (subquery_index, Document) tuples
            top_n: total chunk budget for the final answer
            min_per_subquery: minimum chunks guaranteed per subquery
            max_per_title: cap on chunks from any one article title. Must
                            be at least the number of subqueries, or the
                            floor guarantee above can be silently blocked
                            when multiple subqueries' best content happens
                            to share the same article (a real failure mode
                            found in testing: two genuinely different
                            subquery topics both living in the same long
                            survey article hit a cap meant for single-query
                            redundancy control, not cross-subquery fairness).
        """
        docs = [d for _, d in subquery_docs]
        origins = [i for i, _ in subquery_docs]
        if not docs:
            return []

        texts = [d.page_content for d in docs]
        reranked = self.co.rerank(
            query=question,
            documents=texts,
            top_n=len(texts),
            model=settings.RERANK_MODEL,
        )
        rank_order = [hit.index for hit in reranked.results]  # best first

        title_counts = {}
        taken = set()
        top_docs = []
        n_subqueries = max(origins) + 1 if origins else 0

        # Phase 1 — floor: each subquery's single best-ranked chunk
        # (subject to max_per_title) gets in first, regardless of how it
        # ranks against chunks from other subqueries.
        for _ in range(min_per_subquery):
            for sq_i in range(n_subqueries):
                for idx in rank_order:
                    if idx in taken or origins[idx] != sq_i:
                        continue
                    title = docs[idx].metadata.get("title")
                    if title_counts.get(title, 0) >= max_per_title:
                        continue
                    top_docs.append(docs[idx])
                    taken.add(idx)
                    title_counts[title] = title_counts.get(title, 0) + 1
                    break

        # Phase 2 — fill remaining budget by global rerank order
        for idx in rank_order:
            if len(top_docs) >= top_n:
                break
            if idx in taken:
                continue
            title = docs[idx].metadata.get("title")
            if title_counts.get(title, 0) >= max_per_title:
                continue
            top_docs.append(docs[idx])
            taken.add(idx)
            title_counts[title] = title_counts.get(title, 0) + 1

        return top_docs

    def _naive_multi_with_floor(self, subquery_docs: list, top_n: int, min_per_subquery: int, max_per_title: int) -> list:
        """
        Non-reranked equivalent of _rerank_multi_with_floor() — no Cohere
        call, just similarity order (already how each subquery's results
        arrive from Pinecone). Same floor guarantee: every subquery
        contributes at least min_per_subquery chunks before the rest of
        the budget fills round-robin across subqueries. See
        _rerank_multi_with_floor() for why max_per_title must scale with
        subquery count here, not stay fixed at the single-query default.
        """
        by_subquery = {}
        for i, d in subquery_docs:
            by_subquery.setdefault(i, []).append(d)

        title_counts = {}
        taken = set()
        top_docs = []

        def doc_key(d):
            return (d.metadata.get("title"), d.metadata.get("section"), d.metadata.get("chunk"))

        # Phase 1 — floor
        for _ in range(min_per_subquery):
            for i in sorted(by_subquery):
                for d in by_subquery[i]:
                    key = doc_key(d)
                    if key in taken:
                        continue
                    title = d.metadata.get("title")
                    if title_counts.get(title, 0) >= max_per_title:
                        continue
                    top_docs.append(d)
                    taken.add(key)
                    title_counts[title] = title_counts.get(title, 0) + 1
                    break

        # Phase 2 — fill remaining budget, round-robin across subqueries
        progress = True
        idx_per_sq = {i: 0 for i in by_subquery}
        while len(top_docs) < top_n and progress:
            progress = False
            for i in sorted(by_subquery):
                if len(top_docs) >= top_n:
                    break
                lst = by_subquery[i]
                j = idx_per_sq[i]
                while j < len(lst):
                    d = lst[j]
                    j += 1
                    key = doc_key(d)
                    title = d.metadata.get("title")
                    if key in taken or title_counts.get(title, 0) >= max_per_title:
                        continue
                    top_docs.append(d)
                    taken.add(key)
                    title_counts[title] = title_counts.get(title, 0) + 1
                    progress = True
                    break
                idx_per_sq[i] = j

        return top_docs

    def query_multi(
        self,
        question: str,
        max_subqueries: int = 5,
        k: int = None,
        top_n: int = None,
        use_rerank: bool = True,
        min_per_subquery: int = 1,
        max_per_title: int = None,
    ) -> dict:
        """
        Multiquery RAG: decompose the question into a model-decided
        number of reformulated versions, retrieve for each separately,
        then combine into a final answer — with or without reranking,
        your choice, same as query() vs. query_naive().

        Built for composite questions where a single embedding pulls
        toward whichever half of the question is semantically "louder"
        and starves the other half (see blog Part 1/3, the Cicero vs.
        Socrates trial case) — retrieving each sub-topic separately, with
        its own clean embedding target, is the direct fix for that
        specific failure mode.

        Args:
            question: the user's question
            max_subqueries: upper bound on how many reformulated queries
                             to generate. The actual number is decided by
                             the model based on how many distinct
                             sub-topics it judges the question to have —
                             a simple question typically gets 1, a
                             genuinely composite one gets more, up to
                             this cap. Not a fixed target.
            k: number of docs to retrieve per sub-query (defaults to config)
            top_n: total chunk budget for the final answer. Auto-scales
                   up if the model generated more subqueries than this
                   would otherwise be able to represent — see
                   min_per_subquery.
            use_rerank: if True (default), runs the pooled candidates
                        through Cohere reranking. If False, skips
                        reranking entirely and uses plain similarity
                        order — same trade-off as query() vs. query_naive().
            min_per_subquery: minimum chunks guaranteed to survive per
                               subquery, regardless of how they rank
                               globally. Without this, a fixed top_n can
                               make it mathematically impossible to
                               represent every sub-topic in a many-part
                               composite question.
            max_per_title: cap on chunks from any one article title.
                            Defaults to max(2, number of subqueries) if
                            not given — must be at least the subquery
                            count, or the floor guarantee above can be
                            silently blocked when several subqueries'
                            best content happens to live in the same
                            article (found in testing: Cicero and
                            Socrates both live in "Ancient Political
                            Philosophy", so a cap of 2 starved a 3-subquery
                            question of its third floor pick).
        """
        k = k or settings.RETRIEVAL_K
        requested_top_n = top_n or settings.RERANK_TOP_N

        subqueries = generate_subqueries(question, max_n=max_subqueries)

        # top_n floor: guarantee it's at least large enough for every
        # subquery to contribute min_per_subquery chunks. A fixed
        # top_n smaller than len(subqueries) silently drops entire
        # sub-topics no matter how well retrieval/reranking performs.
        effective_top_n = max(requested_top_n, len(subqueries) * min_per_subquery)

        # max_per_title must be at least the subquery count -- otherwise
        # the per-subquery floor can be blocked by a cap designed for a
        # different problem (single-query redundancy, not cross-subquery
        # fairness). See docstring above.
        effective_max_per_title = max_per_title if max_per_title is not None else max(2, len(subqueries))

        # Retrieve for each sub-query, remembering which subquery
        # produced each doc (needed for the per-subquery floor below),
        # deduping by (title, section, chunk) across subqueries.
        seen = set()
        subquery_docs = []  # list of (subquery_index, Document)
        for i, sq in enumerate(subqueries):
            docs = self.vectorstore.similarity_search(sq, k=k)
            for d in docs:
                key = (d.metadata.get("title"), d.metadata.get("section"), d.metadata.get("chunk"))
                if key in seen:
                    continue
                seen.add(key)
                subquery_docs.append((i, d))

        if not subquery_docs:
            return {"answer": "I couldn't find any relevant documents.", "sources": [], "subqueries": subqueries}

        if use_rerank:
            # Reranked against the ORIGINAL question, not any sub-query —
            # the sub-queries are only for widening retrieval, relevance
            # should still be judged against what the user actually asked.
            top_docs = self._rerank_multi_with_floor(question, subquery_docs, effective_top_n, min_per_subquery, effective_max_per_title)
        else:
            top_docs = self._naive_multi_with_floor(subquery_docs, effective_top_n, min_per_subquery, effective_max_per_title)

        result = self._generate(question, top_docs)
        result["subqueries"] = subqueries  # useful for debugging/logging, not required downstream
        return result


    def query_multi_concat(self, question: str, max_subqueries: int = 5, k_per_subquery: int = 3) -> dict:
        """
        Simpler alternative to query_multi(): no rerank, no max_per_title
        cap, no floor/fill phases. Each subquery just gets its own fixed
        slice of context (k_per_subquery chunks), all slices are deduped
        and concatenated, and the whole pool goes straight to generation
        against the original question.

        This sidesteps an entire class of bug found in query_multi(): a
        max_per_title cap designed for single-query redundancy control
        can silently fight against the per-subquery floor guarantee when
        multiple subqueries' relevant content happens to share one
        article (found in testing — Cicero and Socrates both live in
        "Ancient Political Philosophy", and a cap of 2 blocked the third
        subquery's floor pick entirely). No cap here means no such
        conflict is possible by construction.

        Trade-off: context size grows linearly with subquery count and
        isn't bounded the way query_multi()'s top_n is. Keep
        k_per_subquery low (2-3) for many-part questions — there's no
        reranking step here to clean up noise afterward, so a wide
        k_per_subquery multiplied across many subqueries can flood the
        context with lower-quality matches.

        Args:
            question: the user's question
            max_subqueries: upper bound on how many sub-questions to
                             generate (see generate_subqueries())
            k_per_subquery: how many chunks each individual subquery
                             contributes, before dedup
        """
        subqueries = generate_subqueries(question, max_n=max_subqueries)

        seen = set()
        all_docs = []
        for sq in subqueries:
            docs = self.vectorstore.similarity_search(sq, k=k_per_subquery)
            for d in docs:
                key = (d.metadata.get("title"), d.metadata.get("section"), d.metadata.get("chunk"))
                if key in seen:
                    continue
                seen.add(key)
                all_docs.append(d)

        if not all_docs:
            return {"answer": "I couldn't find any relevant documents.", "sources": [], "subqueries": subqueries}

        result = self._generate(question, all_docs)
        result["subqueries"] = subqueries
        return result


# Simple manual test — run `python rag_pipeline.py` locally (never in production)
if __name__ == "__main__":
    rag = RerankRAG()
    result = rag.query("What does Socrates think about death?")
    print("\nANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(" -", s.get("title", "unknown"), "|", s.get("section", ""), "|", s.get("source", ""))
