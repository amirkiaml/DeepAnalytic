# Scaling and Pilot Strategy Log

Separate from the build-focused dev logs (`multiquery_dev_log.md`, `oruka_retrieval_debug_log.md`, `chunking_conclusions.md`). Those record what was built and what broke. This one records decisions about what to measure, what to ship, and in what order, plus the reasoning and external research behind those decisions.

## Where things stand: tests already completed

All three of the following were run against the 100-article test index, not the full SEP corpus. That scope limitation is the reason this log exists.

**1. Naive vs. reranked retrieval, human-scored** (`tests/eval_scored.xlsx`) 10 questions across 5 articles, scored on Accuracy, Source Match, and Completeness. Naive scored 4.11 overall, reranking 3.74. Two reproducible failures were traced with direct chunk-level evidence: a retrieval-narrowing failure (reranking discarded the chunks containing the answer) and a generation-layer failure (reranking kept the right chunk and the model still declined to synthesize). **Reranking lost.**

**2. Multiquery decomposition** (`notebooks/5_Multiquery_Production.ipynb`, `multiquery_dev_log.md`) Multiquery beat both naive and reranked retrieval on composite questions, decisively so at 10 and 18 topics where single-query retrieval could only ever answer one topic. Within multiquery, the non-reranked concat path beat the reranked path, which failed completely at 18 subqueries. **Multiquery won; reranking lost again.**

**3. Lexical phrase-overlap blend** (`notebooks/10_Oruka_Retrieval_Fix.ipynb`, `oruka_retrieval_debug_log.md`) TF-IDF cosine similarity blended with vector similarity at weight 0.5 moved a buried correct chunk from rank 22 to rank 8, with zero regression on two control questions. **Works, but notebook-only, never wired into `rag_pipeline.py`.**

**Through-line across all three:** reranking consistently underperforms, multiquery wins on complex questions, lexical matching fixes a specific known failure. All established at 100-article scale only.

## Decision: drop reranking as the default

Three independent tests pointed the same direction. Reranking is being dropped as the default path, with `query_multi_concat()` becoming the primary pipeline.

Important scoping note: this means changing defaults, **not deleting the reranking code**. Corpus scale changes retrieval dynamics, so reranking may earn its place back at full corpus size. Keeping the code makes that retestable.

Also worth noting: the known 18-subquery reranking collapse becomes largely moot under this decision, since it lives in a code path that would no longer be the default. Not fixed, but no longer blocking.

## The research that shaped the scaling decision

Before deciding whether to index the full SEP corpus ahead of the pilot, pulled external research on how RAG systems behave as corpus size grows. Findings, with sources:

**Corpus scaling measurably degrades vector-RAG accuracy on complex reasoning.** Xiang et al., "When to use Graphs in RAG" (arxiv.org/pdf/2506.05690), ran controlled experiments at three corpus sizes and found standard vector RAG accuracy on complex reasoning dropped from 58.6% to 43.2% -- a 26% relative decline -- as the corpus grew 20x. The SEP expansion here is roughly 17x, close to the same scale factor.

**The mechanism they name is exactly the Oruka bug.** Their explanation: vector retrieval "is prone to capturing high-similarity but irrelevant noise as the search space expands." That is a textbook description of what was diagnosed by hand in `oruka_retrieval_debug_log.md` -- the correct chunk buried at rank 22 under thematically-adjacent chunks from other articles. An independently documented failure mode, found independently.

**Similarity scores rise while recall falls.** EnterpriseRAG-Bench (arxiv.org/pdf/2605.05253) evaluated at five corpus sizes and found that as the corpus grows, top-10 cosine similarity *rises* while Recall@10 *declines*, for both BM25 and vector search. **Consequence: watching similarity scores would produce the wrong conclusion.** Recall@k is the metric that matters.

**Degradation is silent.** A Towards Data Science analysis of HNSW (the indexing algorithm Pinecone uses) found retrieval quality degrades silently as the vector database grows, even when latency remains stable and the embedding model and distance metric are unchanged. Nothing errors, nothing slows down, quality just quietly drops. This is the core argument for measuring rather than eyeballing.

**One honest counterweight:** "Less LLM, More Documents" (arxiv.org/html/2510.02657) found corpus scaling *consistently strengthens* RAG and can match the gains of a larger model tier, with diminishing returns. Bigger corpora do help -- mostly on open-domain factual QA. The degradation shows up specifically on complex reasoning, which is what philosophy questions are. Both findings are real; they apply to different task types.

**Enterprise mitigations documented in the same literature:** hybrid retrieval (BM25 + vector, fused), metadata filtering (narrow the search space before retrieval runs), GraphRAG (structural rather than pure similarity matching -- notably stayed flat across all corpus sizes in the Xiang et al. experiments where vector RAG dropped 26%), HNSW parameter tuning, and continuous eval as ongoing monitoring rather than a one-time gate.

## Decision: measure the expansion, don't assume it

Full corpus gets indexed into a **separate** Pinecone index, leaving the 100-article index intact and queryable. Without both, there's no before-state to compare against and no way to distinguish "the corpus is too big" from "the pipeline was always mediocre."

**What gets measured: Recall@k on retrieval only, no generation.** For each existing eval question, the known-correct chunk is already identified. The measurement is simply whether that chunk still appears in top-k after expansion, and at what rank. Cheap (no LLM calls, no human scoring), fast, and directly targets the documented failure mode.

**Which pipeline to measure with: naive single-query.** Not because the questions are simple, but because it's the cleanest instrument. Multiquery fires several searches and pools results, which partially *masks* crowding damage -- arguably its whole benefit. Naive gives the unmasked number.

**Secondary signals worth capturing in the same run:** rank position, not just hit/miss (still top-10 but moved from rank 2 to rank 9 is a warning), and cross-article contamination (how many of the top-10 come from articles other than the expected one -- the crowding mechanism showing up directly).

**Explicitly not measured here:** similarity scores (they mislead, per the research above), and pipeline comparison (naive vs. multiquery is a query-strategy question already answered, not a corpus-scale question).

## Decision: keep the variables separate

The lexical blend does **not** get turned on at the same time as the corpus expansion. If both change together, a good result is uninterpretable -- did expansion not hurt, or did it hurt and the blend compensate?

There's also a specific technical reason the blend can't be assumed to transfer: TF-IDF term weights are computed relative to the collection they're fitted on. A word distinctive across 100 philosophy articles may be far less distinctive across 1,700. The 0.5 weight was tuned at small scale and may behave differently at large scale.

**Sequence:** (1) full corpus, pure vector, eval -- gives the true cost of expansion. (2) then turn on the lexical blend, eval again -- gives whether it helps *at this scale*. Two runs, one variable each.

## The decision tree for what ships to testers

- **Retrieval holds up at full corpus** -> ship full corpus, `query_multi_concat()`, no lexical blend. Simplest thing that works.
- **Retrieval degrades, lexical blend recovers it** -> ship full corpus with the blend on. Measured evidence for both the problem and the fix.
- **Retrieval degrades, lexical doesn't recover it** -> ship the 100-article index, tell testers plainly it's a subset. A subset pilot is completely honest and testers care about answer quality, not corpus size.

In every branch: **one pipeline, no mode-switching.** Testers get a single experience; the only variables are which index and whether the blend is on.

## What gets measured with testers, and why it's different

Every eval to this point has been self-judged against self-authored expectations. That has a ceiling: it cannot establish whether the output is useful to a philosopher doing philosophy. That is the only thing testers can answer that internal evaluation cannot.

**1. Answer quality on their own real questions** -- not the test questions, theirs, from actual work or teaching. A 1-5 rating plus free-text "what was wrong or missing." The rating aggregates; the free text carries the actual signal.

**2. Source correctness** -- show retrieved passages alongside the answer, ask whether those were the right passages. This separates retrieval failure from generation failure, an ambiguity that has come up repeatedly in the build logs.

**3. Would you use this, and for what** -- one blunt question. Cheapest signal available and the most quotable for a portfolio.

**Deliberately not measured:** pipeline comparison or anything asking testers to evaluate architecture. They aren't equipped for it and it burns goodwill on questions internal evals answer better.

**Honest framing:** with a handful of testers this is qualitative evidence, not statistically meaningful ratings. A few specific reactions from qualified people is genuinely valuable and citable ("three philosophers tested it; one caught a case where..."), but no elaborate scoring apparatus is warranted at this sample size. Direct DM or email is enough.

## After the pilot: branches

**When to move on rather than add testers:** if feedback from the first few converges on the same points, the version has taught what it can and more testers on it mostly repeat. If feedback diverges wildly, the picture isn't stable yet and more testers on the *same* version is worthwhile.

**If the core experience works** -> go straight to building and testing the referee feature. More testers on the same chatbot version answers an already-answered question. The genuinely open question is whether anyone wants a referee tool at all -- that underpins the whole strategic pivot and is completely untested. Even a rough version (paste a paragraph, get back which claims SEP supports, contradicts, or doesn't address) tells you whether the concept lands before building two full stages of it.

**Important caveat:** "works" must mean more than "nobody complained." Polite, vague, or low-engagement feedback is absence of evidence, not validation. What counts is people using it on their own real questions and returning with specifics, positive or negative.

**If quality problems surface** -> fix those plus ship the critique function, and return to a similar-sized group rather than expanding. Widening the audience for a version with known problems mostly generates duplicate complaints. Note that real philosophers hitting real failures on real questions is better bug-finding data than anything generatable internally -- a large part of why the pilot is worth running.

**If feedback is lukewarm** ("interesting, but I wouldn't use it") -> most valuable outcome and the one to take most seriously. It means the referee direction needs rethinking before two more stages get built on top of it. Better learned from three people this month than after months of building.

**Regardless of branch:** the pilot itself is a blog post (what shipped, what philosophers said, what changed as a result) -- and it's stronger with honest negative findings than a glowing one. And get one concrete quotable sentence if any tester offers one; a single specific reaction does more for a portfolio than a page of self-reported metrics.

**Keep continuity:** whoever tests round two should include at least a couple of round-one testers. Someone who saw v1 can tell you "this is noticeably better at X" -- a fresh tester structurally cannot.

**On scaling to a larger audience:** not worth it until there's a version with consistently positive basics *and* the referee feature in some form. Before that, it spends network goodwill on a version already known not to be the actual product direction.

## Status of outstanding items, precisely

Not all "backlog" -- these are different kinds of thing:

- **Oruka lexical fix**: *contingent*, not deferred. Whether it ships is gated on the full-index eval results, per the decision tree above.
- **18-subquery reranking collapse**: *largely moot* if reranking is no longer the default path. Worth understanding only if reranking is reconsidered at full corpus scale.
- **Critique function**: *deferred*. Parked by choice. Note it was already mentioned publicly in the tester-recruitment post as coming soon, so it carries a small commitment.
- **Full-corpus indexing**: *in progress*, gated on the eval described above.
