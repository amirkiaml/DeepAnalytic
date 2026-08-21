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
- **Critique function**: *genuinely deferred*. Parked by choice. Note it was already mentioned publicly in the tester-recruitment post as coming soon, so it carries a small commitment.
- **Full-corpus indexing**: *in progress*, gated on the eval described above.

## Implementation step 1: namespace support in ingest.py

**Decision changed from the original plan.** The plan called for indexing the full corpus into a *separate* Pinecone index. Research on Pinecone's current free-tier limits made that unworkable and pointed to a better approach anyway.

**Why a separate index doesn't work:** the free Starter tier caps serverless indexes (sources conflict between 1 and 5, and this project already hit the ceiling once during the chunking-strategy experiments, when creating a fifth index failed outright). Pinecone's own guidance is explicit that additional indexes each consume separate storage while namespaces are free and carry no performance overhead.

**What namespaces give instead:** a namespace is a hard partition within a single index. A query against one namespace cannot see vectors in another under any circumstances — they're not filtered out, they're simply outside the search space. This delivers exactly the isolation the before/after comparison needs, with no index-count problem. The pattern is already proven in this codebase, since `9_Chunking_Strategy_Comparison.ipynb` used five namespaces successfully.

**Storage checked and fine:** ~100K vectors at 1,536 dimensions is roughly 570MB against a 2GB free-tier allowance.

**Cost checked and negligible:** at text-embedding-3-small's $0.02 per million tokens, the full-corpus embedding run is roughly $0.80 standard, or $0.40 via the Batch API. Embedding cost is not a real consideration here. The genuine cost risk at query time is Pinecone read units, billed at 1 per 1,000 vectors scanned, which is a further argument for metadata filtering — narrowing what gets scanned cuts cost as well as improving quality.

**Changes made to `ingest.py`:**

- Added a `NAMESPACE` constant at the top, defaulting to `""` (the default namespace, where the existing 100-article data lives). Set to something like `"articles-full"` for the full-corpus run.
- `embed_and_upsert()` now passes `namespace=` through to Pinecone when one is set.
- Added retry with exponential backoff (4 attempts, doubling delay) around embed-and-upsert. At full scale this makes roughly 1,000 API calls, so a single transient rate-limit or network blip partway through would otherwise waste the entire run.
- Added per-article error handling, so a malformed article gets skipped and reported rather than crashing a long run. At 100 articles a bad row is obvious; at ~1,800 it's a real risk.
- Added a failure summary at the end: count of permanently failed batches (with an estimate of missing chunks) and a list of skipped articles, so a partially-successful run is visible rather than silently incomplete.
- Startup now prints the target index and namespace, so it's impossible to accidentally write into the wrong partition without noticing.

**Not yet done:** the actual full-corpus run, and the retrieval comparison that follows it.

## Implementation step 2: the Recall@k comparison notebook, and a measurement bug caught early

Built `11_Corpus_Scaling_Recall.ipynb` to measure retrieval quality before and after the corpus expansion. Three parts: baseline against the existing 100-article data in the default namespace, the same measurement against the full corpus in `articles-full`, then a direct diff.

**Design choices worth recording.** Search runs at k=50 while reporting Recall@5, @10, and @20, so one query yields every threshold and, more usefully, shows the actual rank of correct chunks that miss the top 10. That distinction matters for the next decision: a chunk sitting at rank 12 is a very different situation from one at rank 40, and only the first is plausibly recoverable by a lexical blend. Alongside rank, each question records how many of its top 10 results came from an article other than the expected one, which measures cross-article crowding directly rather than as a pass/fail threshold.

**Baseline run produced results that contradicted what was already known, which turned out to be the useful part.** First run reported Recall@10 of 0.40, with 6 of 10 questions never finding the correct chunk anywhere in the top 50. That could not be reconciled with the existing human-scored eval, which gave naive retrieval 4.11/5 overall and 4.00 on Completeness on these same questions. A system genuinely failing 6 of 10 retrievals could not produce those scores.

**The tell was in the crowding column.** Both Animalism questions showed `OffArticle@10 = 0`, meaning all ten top results came from the Animalism article, retrieval had worked perfectly, and yet both were recorded as misses. Identical Recall at k=5, 10, and 20 was a second tell: real degradation produces a gradient, with some chunks at rank 12 and some at 18, whereas identical numbers across every threshold pointed to never-founds rather than near-misses.

**Root cause: the eval file describes sections the system doesn't track.** `Section_Expected` values were written from article tables of contents and include subsections, for example "3.1 Thinking Animal Argument". But `section_parser.py` splits only at top-level headings, so the metadata actually stored in Pinecone carries only "Arguments for and Objections to Animalism". There is no subsection field in the index to compare against, so section-level matching produced false misses on questions where retrieval had worked correctly.

**Fix: match on article rather than section.** This is looser, and the trade-off is worth stating plainly. Article-level recall cannot detect the Oruka-style failure, where the right article is retrieved but a wrong section within it outranks the correct one. That failure mode is real and documented, it simply isn't measurable with the metadata as currently indexed. What article-level recall does measure directly is cross-article crowding, other articles displacing the correct one, which is exactly the risk the corpus-scaling research identified and exactly the question this notebook exists to answer.

**Three options were considered before settling on this.** Accept the looser metric for now, since it answers the scaling question adequately. Rewrite the eval CSV's expected sections to top-level titles, roughly twenty minutes of manual work, which would restore section-level measurement and also fix a real inconsistency in the eval file. Or index subsection metadata properly, which would require changing `section_parser.py` and re-indexing everything, and which is genuinely worth doing eventually since subsection titles would improve retrieval and enable the cross-reference work sketched in blog Part 3. Chose the first for now on the grounds that the scaling decision shouldn't be blocked on a measurement refinement that doesn't bear on it. The second is worth doing soon. The third belongs on the roadmap with the other structural improvements.

**Worth noting for its own sake:** this bug was caught only because the result contradicted an existing measurement. Without the earlier human-scored eval to check against, a Recall@10 of 0.40 would have looked like a plausible finding and might well have derailed the whole scaling decision.

**Still not done:** the baseline re-run with the corrected function, the full-corpus indexing, and the comparison itself.

## Baseline established: Recall@10 = 1.00 on the 100-article corpus

Re-ran Part 1 with the corrected article-level matching. Results:

| Metric | Value |
|---|---|
| Recall@5 | 0.90 |
| Recall@10 | 1.00 |
| Recall@20 | 1.00 |
| Mean rank where found | 1.7 |
| Never found in top 50 | 0 of 10 |
| Mean off-article chunks in top 10 | 3.2 |

Every question found its expected article, and in eight of ten cases the correct article was the single top result. This is consistent with the earlier human-scored eval that gave naive retrieval 4.11/5 overall, which is the reconciliation the earlier buggy run failed. The baseline is solid.

**The number most worth watching is the crowding metric, not the recall number.** At 100 articles, an average of 3.2 out of every 10 top results already come from an article other than the expected one. Recall@10 is at ceiling and cannot improve, so it can only stay flat or drop after expansion. Crowding, by contrast, has plenty of room to worsen, and it is the mechanism the scaling research identified as the actual cause of degradation. If that 3.2 climbs sharply on the full corpus while recall holds, that is an early warning rather than a clean result.

**Two questions are already showing strain and are worth watching individually.** Both al-Farabi questions returned the correct article at ranks 6 and 3 rather than rank 1, each with 8 of 10 top results coming from other articles. These are the most crowding-sensitive questions in the set, and they are the ones most likely to drop out of the top 10 first if expansion hurts.

**A note on how the measurement bug was caught, worth keeping.** The corrected result only looks obviously right in hindsight. What made the earlier bug detectable was having a prior, independent measurement to check against: a Recall@10 of 0.40 could not be reconciled with a human-scored 4.11/5 on the same questions. Without that earlier eval, 0.40 would have looked like a plausible finding about retrieval quality and might well have derailed the scaling decision entirely. Keeping independent measurements around, even ones that seem superseded, has direct diagnostic value.

## Implementation step 3: 300-article validation run

Ran `ingest.py` at 300 articles into a throwaway `articles-300-test` namespace before committing to the full corpus, to confirm the namespace parameter works end to end and to get a real throughput number rather than an extrapolation from the original 100-article run.

**Result: clean.** 18,143 chunks indexed from 300 articles in roughly 6 minutes, no failed batches, no skipped articles.

**Parsing held up at 3x scale, marginally better than before.** 96.3% clean TOC matches, 3.0% partial, 0.3% regex fallback, 0.3% failed, against 95% clean on the original 100-article run. Worth noting because the parser encountering more varied article structures could plausibly have degraded, and it didn't.

**Namespace isolation confirmed working.** `describe_index_stats()` afterwards showed `articles-300-test` at 18,143 vectors and the default namespace unchanged at 5,687. The partition holds, and the 100-article baseline is safe from contamination by the full-corpus run.

**Revised projections from real data.** Chunk density came in at roughly 60 per article, higher than the ~57 implied by the first 100-article run, which projects to about 109,000 chunks for the full corpus rather than the 100,000 estimated earlier. Still comfortably inside Pinecone's 2GB free tier at roughly 620MB, though tighter than the original estimate. Timing at 1.04 seconds per article projects to roughly 36 minutes for the full run.

**One fix made as a result of this run.** The end-of-run warning about TOC parsing quality was firing on a single failed article out of 300, because its trigger condition was `failed > 0`. That threshold made sense when a test run was 100 articles and one failure meant 1%, but at full corpus scale a handful of genuinely malformed articles will fire it every time, which trains you to ignore a warning that should mean something. Changed to `failed > parsed_total * 0.02`, so it only fires when more than 2% of articles fail outright, which at 1,800 articles means 36 failures rather than one.

## Full corpus indexed, and the scaling question answered

**The run.** 111,048 chunks from 1,803 articles into the `articles-full` namespace, in roughly 25 minutes, with no failed batches and no skipped articles. Placement verified afterwards: `articles-full` at 111,048 vectors, default namespace unchanged at 5,687, and the throwaway 300-article test namespace cleaned up. Namespace isolation held throughout, which is what makes the before/after comparison trustworthy.

**Parsing quality improved with scale, which was not expected.** TOC matching came in at 97.5% clean across the full corpus, against 96.3% at 300 articles and 95% at 100. Only one article out of 1,803 failed outright. The reasonable prior was that a parser tuned on 100 articles would degrade as it met more varied structures across a much larger corpus, and the opposite happened. The corrected warning threshold correctly stayed silent at one failure.

**Recall did not degrade.** Recall@5 held at 0.90, Recall@10 and @20 held at 1.00, zero questions dropped out of the top 10, and nothing fell out of the top 50. A 19.5x corpus expansion produced no measured recall loss.

**Crowding moved exactly as the research predicted.** Mean off-article chunks in the top 10 rose from 3.2 to 4.7, a 47% increase, so nearly half of every top-10 result set now comes from an article other than the one being asked about. Mean rank of the correct article slipped from 1.7 to 2.2. The mechanism identified in the Xiang et al. work, thematically adjacent content from other documents displacing the correct one as the search space expands, is real and measurable here. It simply has not crossed the threshold where it costs an answer yet.

**Two questions sit closest to that threshold.** Both al-Farabi questions returned the correct article at ranks 6 and 4, with 8 and 9 of their top 10 results coming from other articles. These were already the weakest cases at baseline and remain the most likely to slip first if the corpus grows further or question phrasing shifts.

**Two limits worth stating on what this result can claim.** Recall@10 was already at ceiling on the baseline at 1.00, so the test could only detect degradation and never improvement. And ten questions is a small sample: no drop here is meaningful evidence, but it is not proof that no question anywhere in the corpus degraded. The crowding number is the more informative measurement precisely because it had room to move in both directions and did.

**Also worth noting against the research.** The Xiang et al. finding of a 26% relative accuracy decline was measured at roughly a 20x scale factor, very close to this expansion's 19.5x, and specifically on complex reasoning tasks. That decline did not appear here at the retrieval level. One plausible explanation is that their measurement was end-to-end accuracy on complex reasoning rather than article-level retrieval recall, which is a looser target and, per the earlier note in this log, cannot detect within-article failures at all. The two results are not necessarily in conflict, they measure different things.

## Decision: ship the full corpus

Per the decision tree recorded earlier in this log, this outcome lands in the first branch. **Ship the full corpus, using `query_multi_concat()`, without the lexical phrase-overlap blend.** The blend was explicitly contingent on degradation that did not materialize, and adding it now would mean introducing a variable to solve a problem the measurement says does not currently exist.

The blend remains available and tested in `10_Oruka_Retrieval_Fix.ipynb`. The crowding trend is the thing to watch: if it continues climbing as the corpus grows or as more question types are tried, and eventually starts costing recall, the blend is the first mitigation to reach for, followed by metadata filtering.

**Remaining before shipping to testers:** flip the reranking defaults in `rag_pipeline.py` (`use_rerank=False`, `query_multi_concat()` as the primary path), then build the Streamlit app with keys in Streamlit secrets, a usage cap, one pipeline with no mode switching, and a plain-language intro.

## Research bibliography

Consolidated reference list for the scaling and evaluation work, kept in one place because these sources will accompany the blog posts written about these tests. The scaling sources were pulled before the corpus expansion and are discussed in context earlier in this log; the evaluation-methodology sources were pulled afterwards, when deciding whether the current test set was actually sufficient to justify shipping.

### Corpus scaling

**Xiang et al., "When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation"** — https://arxiv.org/pdf/2506.05690
Ran controlled experiments at three corpus sizes and found standard vector RAG accuracy on complex reasoning dropped from 58.6% to 43.2%, a 26% relative decline, as the corpus grew 20x. Names the mechanism explicitly: vector retrieval "is prone to capturing high-similarity but irrelevant noise as the search space expands," which is a textbook description of the crowding measured in this project. Also found GraphRAG stayed flat at roughly 60% across all corpus sizes, making structural retrieval the most robust option they tested. **Their proposed solution:** move from pure similarity matching to graph-based retrieval that matches on explicit entities and relationships, since that structure does not degrade as the search space grows. Relevant here because SEP's own internal cross-reference links are effectively a ready-made graph, and this is the same structural direction the Part 3 cross-reference work points toward.

**EnterpriseRAG-Bench** — https://arxiv.org/pdf/2605.05253
Evaluated retrieval across five corpus sizes and found that as the corpus grows, top-10 cosine similarity *rises* while Recall@10 *declines*, for both BM25 and dense vector search. This is the single most decision-relevant finding for this project's measurement design: watching similarity scores would have produced exactly the wrong conclusion about whether expansion hurt. **Their proposed solution is primarily diagnostic rather than architectural:** measure recall directly at realistic corpus sizes rather than trusting similarity as a proxy, and evaluate at multiple corpus scales rather than only the one you happen to be testing on.

**HNSW silent degradation analysis** — Towards Data Science
Found that retrieval quality degrades silently as a vector database grows, even when latency remains stable and the embedding model and distance metric are unchanged. Nothing errors and nothing slows down, which is the core argument for measuring rather than eyeballing after any expansion. **Their proposed solution:** tune the index's own parameters, particularly `ef_search`, which trades recall against latency and which most teams never touch, and treat evaluation as ongoing monitoring rather than a one-time gate, since the degradation is gradual and invisible.

**"Less LLM, More Documents: Searching for Improved RAG"** — https://arxiv.org/html/2510.02657
The honest counterweight to the sources above: found corpus scaling *consistently strengthens* RAG and can match the gains of moving to a larger model tier, with diminishing returns. The degradation findings elsewhere concentrate on complex reasoning tasks, while this one measures mostly open-domain factual QA. Both are real; they apply to different task types, and philosophy questions sit closer to the former. **Their proposed direction, if anything, is the inverse of a mitigation:** where budget allows a choice between a larger model and a larger corpus, expanding the corpus is often the better investment, which is a useful counterpoint to hold against the instinct to treat every scaling problem as something to defend against.

### Evaluation methodology

**RAG Evaluation Checklist** — https://hiro.solutions/rag-evaluation-checklist-retrieval-quality-answer-accuracy
Three points directly applicable to this project's current gap. First, check head and tail queries separately, because "improvements on common queries can hide regressions on niche but important questions" — and the inverse applies here, since this project's test set is entirely tail queries about distinctive entities. Second, "review lost documents. Which previously retrievable sources disappeared from top-k? This is often more informative than average score changes," which is precisely why the crowding metric carried more signal than the flat recall number. Third, watch for near-duplicate retrieval, since "a retriever that returns five similar chunks may score acceptably on relevance but still starve the generator of useful context diversity." **Their proposed solutions are diagnostic:** segment evaluation by query type rather than reporting a single average, track which specific documents dropped out of top-k rather than watching aggregate scores, inspect chunk boundaries for splits that separate definitions from their explanations, and explicitly test metadata filtering rather than assuming it works.

**Latenode, "RAG Evaluation: Complete Guide"** — https://latenode.com/blog/rag-evaluation-complete-guide-to-testing-retrieval-augmented-generation-systems
Argues golden question sets should include a mix of factual inquiries, multi-step reasoning challenges, and ambiguous edge cases, and that query diversity means testing variations in language, complexity, and context, since systems that handle structured queries well often falter on conversational phrasing or unfamiliar terminology. Supports the paraphrase-stability test idea. **Their proposed solution:** build golden sets deliberately covering multiple query types and phrasings from the start, rather than assembling a set from whatever questions happened to be convenient, which is exactly the bias this project's current ten questions exhibit.

**"Unanswerability Evaluation for Retrieval Augmented Generation"** — https://arxiv.org/pdf/2412.12300
Notes that existing evaluation methods including RAGAS, ARES, RGB, and MultiHop-RAG all focus on answerable queries and "overlook a critical aspect: the ability of RAG systems to appropriately handle unanswerable requests," arguing that rejecting unanswerable queries is essential for reliability and safety. Particularly relevant to the referee-tool direction, where confidently answering a question the corpus cannot support is worse than admitting the gap. **Their proposed solution:** build unanswerable queries into the evaluation set as a first-class category and measure rejection behaviour explicitly, rather than only measuring accuracy on questions the corpus can support.

**Braintrust, "RAG evaluation metrics"** — https://www.braintrust.dev/articles/rag-evaluation-metrics
Recommends separating retrieval and generation into distinct evaluation spans so that "when debugging low scores, you can pinpoint whether retrieval surfaced wrong documents or generation misused correct context" — an ambiguity this project has hit repeatedly. Also argues the best evaluation data comes from real user queries, since production reveals question patterns developers never anticipate, which is a direct argument for the pilot. **Their proposed solution:** instrument retrieval and generation as separate traced spans so failures can be attributed to one or the other without guesswork, and collect production queries continuously into a growing golden dataset that doubles as a regression suite.

**Toloka, "RAG evaluation: a technical guide"** — https://toloka.ai/blog/rag-evaluation-a-technical-guide-to-measuring-retrieval-augmented-generation
Recommends automated scoring for bulk coverage with expert review reserved selectively for ambiguous or high-impact queries, matching the two-rater approach already used in this project's eval. Also notes that offline metrics degrade over time as user behaviour and knowledge sources change, supporting continuous evaluation rather than one-time gating. **Their proposed solution:** a hybrid evaluation strategy, using automated metrics for broad coverage while reserving human judgement for the ambiguous and high-stakes cases where automated scoring misses subtle reasoning problems, plus instrumenting the live system to log retrieval performance and answer grounding rather than relying only on offline test sets.

## Planned additional tests before shipping

The full-corpus result showed recall holding while crowding rose 47%. That is a pass, but the test set has a specific and now-identified weakness: **all ten questions are tail queries**, targeting distinctive entities like Oruka, al-Farabi, and Abelard that have few competitors even at 1,803 articles. The crowding risk is worst for concepts that many articles discuss, and the current set contains none of those. Per the head/tail point in the RAG Evaluation Checklist above, this is exactly the case where a clean result can hide a real regression.

Three tests planned, in priority order:

**1. Head queries: generic concepts.** Questions like "what is supervenience," "what is the difference between necessary and sufficient conditions," or "what is the is-ought gap" have few competitors at 100 articles and dozens at 1,803. This targets the crowding mechanism directly. No ground truth exists for these, but that is not required for the comparison: measuring how much the top-10 result set changes between the two corpora, and whether full-corpus results remain coherent and on-topic, is enough to reveal a problem.

**2. Paraphrase stability.** Ask the same question three ways and check whether the correct article still surfaces. The two al-Farabi questions currently sit at ranks 6 and 4 with 8 and 9 of their top 10 off-article, fragile enough that different phrasing might tip them out. This tests whether the existing passes are robust or merely lucky.

**3. Unanswerability.** A few questions the corpus genuinely cannot answer, checking whether the system declines rather than confabulating. Lower priority for the scaling question specifically, but directly relevant to the referee-tool direction, where a confident wrong answer is worse than an admitted gap.

A fourth possibility, worth considering but not prioritised: pick five articles at random from the full corpus and write a question for each. The current five were selected partly because they had produced interesting failures, which is a biased sample in a way that could cut either direction.

## Correction and expansion: the crowding reading was wrong

The entry above recorded the full-corpus result as "recall held, crowding rose 47%," treating the off-article increase as the mechanism the research predicted, held in check but visibly present. Three additional tests showed that reading was wrong, and the correction matters enough to record rather than quietly amend.

**The flaw in the metric.** `OffArticle@10` counts chunks from *other* articles, not from *wrong* ones. The 100-article baseline was an alphabetical slice, so it excluded most of the corpus by construction, including entries genuinely relevant to the test questions. A question about al-Farabi's metaphysics competing with an actual article on Arabic and Islamic metaphysics is not noise displacing signal; it is the corpus finally containing material that should always have been there. Crowding by irrelevant content and enrichment by relevant content produce identical numbers. The metric cannot distinguish them, and I had read it as though it could.

**Head queries settled it.** Five generic-concept questions, run against both namespaces. Overlap between the two result sets averaged 0.4 articles out of 10, with three of five showing zero overlap, which on the earlier reading would have looked catastrophic. Reading the actual titles showed the opposite. Four of five head queries returned the entry named after the concept on the full corpus where the 100-article corpus had returned articles that merely mention it. Supervenience went from nine chunks of *Anomalous Monism* to six of *Supervenience*, plus *Ontological Dependence* and *Scientific Reduction*. Natural kinds went from a scatter including *Peter Abelard*, *Anaxagoras*, and *The Concept of the Aesthetic* to seven chunks of *Natural Kinds* plus *Natural Properties*. Necessary and sufficient conditions went from *Abstract Objects* and *Abilities* to nine chunks of *Necessary and Sufficient Conditions*. Near-total turnover was what improvement looked like.

**Chunk-level inspection resolved the two ambiguous cases.** The is-ought question had looked like the one genuine crowding case, with *Combining Logics* and *Bernard Bolzano* flagged as probable noise from their titles. Reading the passages reversed that. The 100-article corpus returned nothing about the is-ought gap at all, matching only on the word "ought" in unrelated deontic and aesthetic contexts, topping out at 0.511 similarity. The full corpus led with *Thick Ethical Concepts* on exactly the right passage, "the intuitive contrast between is and ought marks an important gap between distinct domains," at 0.632, with every full-corpus result scoring above the best 100-corpus result. The two flagged entries turned out to be about the logic of ought-propositions, not incidental keyword matches. My title-based judgement was too harsh.

The a priori question was the more decision-relevant case, since it was the one where both corpora had the correct article available. The 100-corpus returned it eight times of ten; the full corpus three times plus four adjacent entries, which looked like possible dilution. Reading the chunks showed otherwise. *Epistemology* took the top slot at 0.709, above anything the 100-corpus produced, with a passage directly defining what counts as experience for a priori justification. *Kant's Theory of Judgment* covered the synthetic a priori, central to the distinction and absent from the 100-corpus results entirely. Meanwhile six of the 100-corpus chunks came from a single section, several on peripheral debates: the Lottery Paradox, Evans on contingent a priori propositions, Turri's unlikely-event example. That was depth on the margins rather than on the question asked.

**Paraphrase stability held.** Two question sets, three phrasings each, both corpora, twelve runs total. Nothing dropped out of the top ten anywhere. The al-Farabi set, flagged as weakest, returned ranks 3, 4, 6 on the 100-article corpus and 4, 4, 6 on the full corpus, so expansion moved rank by at most one position on one phrasing. Rewording moved rank more than expanding the corpus did, which locates the difficulty in the question rather than in corpus size. The Cicero set held rank 1 across all six runs.

**Against Xiang et al.** Their controlled experiments found a 26% relative accuracy decline on complex reasoning at a 20x scale factor, nearly identical to this expansion's 19.5x, attributed to vector retrieval "capturing high-similarity but irrelevant noise as the search space expands." That did not happen here, and the honest response is to explain rather than to claim refutation. Their metric was end-to-end accuracy on complex reasoning; this one is article-level retrieval recall, a looser target that cannot detect within-article failures at all. SEP may also be unusually favourable, being a curated encyclopedia where most concepts have a dedicated entry, so expansion adds canonical articles rather than marginally-relevant documents. And the starting point here was an alphabetical slice rather than a representative sample, so part of what expansion did was correct an artificial deficiency rather than dilute a healthy corpus. One result, one corpus, a looser metric.

**Decision unchanged, reasoning strengthened.** Ship the full corpus with `query_multi_concat()`, no lexical blend. The blend was contingent on degradation that did not appear, and now appears less likely to appear than the first reading suggested.

**Methodological note worth carrying forward.** Both times a number in this project looked alarming, it was the measurement rather than the system. The first was section-level matching producing a false Recall@10 of 0.40. The second was `OffArticle@10` conflating enrichment with crowding. In both cases the resolution came from reading actual content rather than refining the metric. Aggregate numbers were useful for flagging where to look and consistently wrong about what was found there.

## Future direction: a possible research contribution

Parked deliberately, not abandoned. Worth writing down while the reasoning is fresh, because the specific gap identified here is not something anyone in the sources reviewed for this project appears to have isolated.

**The idea.** The corpus-scaling literature treats corpus *size* as the independent variable. Xiang et al. measured a 26% relative accuracy decline on complex reasoning across a twentyfold expansion; "Less LLM, More Documents" found expansion consistently strengthening RAG. Both are real results, measured on different task types, and the field tends to cite whichever suits the argument being made. But neither isolates corpus *composition*, and that seems to be doing unacknowledged work. Adding documents that are canonically relevant to a query is a different operation from adding documents that merely resemble it, and both count as "scaling" in existing designs.

This project's own result is a data point for that. A nineteen and a half fold expansion of a curated encyclopedia, where most concepts have a dedicated entry, produced no measured degradation and, on head queries, clear improvement. That is exactly what the composition hypothesis would predict, and it sits awkwardly against a size-only account.

**What a testable version would look like.** Take one base corpus. Expand it two ways to the same final size: once with documents canonically relevant to the query set, once with documents that are topically adjacent but not authoritative. Measure both. If the curves diverge, size alone does not explain the degradation reported in the literature, and existing results contain a confound worth naming.

**What is missing before this is publishable, stated plainly.** The current result is n=1: one corpus, one embedding model, one chunking strategy, ten tail questions and five head questions written by two people. The metric is article-level retrieval recall, which cannot detect the within-article failures this project has separately documented occurring. A reviewer's first question would be whether the effect holds on a second corpus, and there is no answer to that yet. Minimum additions: a second and ideally third corpus with different composition characteristics, a metric that catches within-article failures, a substantially larger and less self-authored question set, and probably a collaborator with empirical ML evaluation experience.

**Why it is parked rather than pursued.** A workshop paper is months of work for a credential that matters primarily in academia. The immediate priorities are shipping the pilot, the referee tool, and the job search, and the findings already do more as blog posts and a documented repo than they would as a preprint. The more promising route: if the referee tool works and philosophers use it, that generates a paper with real users and real data behind it, which is a stronger contribution than a scaling ablation on one corpus. Revisit after the pilot.

## Implementation step 4: the pilot app, and a namespace bug caught in the process

**A wiring gap found while building the app.** `rag_pipeline.py` connects to the index but never specified a namespace, so it was querying the default namespace, which still holds the original 100-article test slice. The full 1,803-article corpus sits in `articles-full`. This affected everything routed through `RerankRAG`, not just the app: `chat.py`, `check_chunks.py`, and `run_eval.py` were all silently querying the small corpus. Only `ingest.py` had namespace awareness, since that was the file the namespace work was done in.

Worth noting this would have been invisible without looking for it. Every one of those tools would have run without error and returned plausible answers, just from the wrong corpus. The recall comparison in `11_Corpus_Scaling_Recall.ipynb` was unaffected because it queries Pinecone directly rather than through `RerankRAG`, which is why the full-corpus results were still valid.

Fixed by adding `PINECONE_NAMESPACE` to `config.py`, defaulting to `articles-full`, and passing it through to `PineconeVectorStore` in `rag_pipeline.py`. Everything downstream now routes through one setting rather than defaulting to the old slice by accident.

**Decisions made for the pilot app.**

*Which pipeline.* `query_multi_concat()`, with no mode switching exposed to testers. Reranking is not disabled anywhere in the codebase; the app simply calls a method that has no rerank path. That was the minimal option: no changes to `rag_pipeline.py`, and `query_multi()` keeps its `use_rerank` default intact so existing eval comparisons remain reproducible.

*Memory.* Three turns, implemented entirely in the app layer. Previous questions and truncated previous answers are prepended before the question reaches decomposition, so a follow-up like "how does that relate to physicalism" can resolve its reference. The pipeline itself stays stateless. Testers are told not to lean on it, for two reasons: self-contained questions retrieve better, and self-contained claims are the unit the referee tool will eventually operate on, so getting testers into that habit is useful beyond this pilot.

*Sources.* Collapsible under each answer, showing article, section, a 400-character excerpt, and a link to the real entry. Also shows the generated subqueries when a question was split. The feedback plan asks testers whether the retrieved passages were right, which requires them to be visible; collapsing keeps the interface readable while making that judgement possible.

*Cost control, two layers.* A hard budget cap on the OpenAI project holding the API key, plus a 15-query per-session limit in the app. The budget cap alone protects the wallet but produces raw API errors mid-session once exhausted; the session cap gives a clear message instead, and stops one enthusiastic tester consuming the entire budget in an afternoon.

*Access.* Email plus a per-person password, stored as a `[passwords]` table in Streamlit secrets. One password each rather than a shared one, so access can be revoked individually.

*Logging.* Google Sheets rather than local files, because Streamlit Community Cloud's filesystem is ephemeral and anything written to disk vanishes on restart or redeploy. Toggleable via a `LOGGING_ENABLED` secret without redeploying. Emails are hashed with a salt into short pseudonymous IDs, so one tester's questions can be grouped for analysis without storing who they are alongside their interactions. Logging failures are caught and never break a session. Testers are told in the intro that interactions are logged anonymously.

**Still outstanding before shipping:** deploy to Streamlit Community Cloud, set up the Google Sheet and service account, generate per-tester passwords, and resolve the SEP terms-of-use question for a publicly-accessible app serving retrieved passages.

## Verifying the namespace fix, and a smaller issue found along the way

Ran `rag_pipeline.py`'s smoke test after wiring the namespace through. The fix worked, confirmed by the sources rather than by any assertion: results came back from *Socrates*, *Thomas More*, and *Religion and Morality*, all of which sit well past "Philosophy of Architecture" alphabetically and therefore could not have existed in the original 100-article slice.

The answer quality also improved noticeably on the same question the smoke test has always used. Earlier runs drew mostly on *Afterlife* and *Ancient Political Philosophy*, which are relevant but oblique. The full corpus leads with the dedicated *Socrates* entry, and picks up the specific detail that Socrates read his guiding spirit's failure to intervene at the trial as an invitation from the gods, which is the kind of textual specificity a philosopher would expect.

**The smoke test itself was outdated and has been updated.** It was still calling `query()`, the single-query reranked path, which is the path three separate evaluations in this project found flat-to-harmful. It now calls `query_multi_concat()`, matching what `app.py` actually serves, and prints the generated subqueries before the answer so the decomposition step is visible rather than silent.

**A smaller issue surfaced immediately once subqueries were visible.** On the deliberately single-topic question "What does Socrates think about death?", decomposition returned:

```
- What are Socrates' views on death?
- What does Socrates think about death?
```

That is a paraphrase plus the original, not a decomposition. The topic-first structured output was supposed to return exactly one subquery for a single-topic question, which was verified when that fix was built. What appears to be happening is that the model reformulates rather than passing the question through unchanged, and the safety-net logic then appends the original because it is not an exact string match, producing two near-identical searches.

Not harmful. Two nearly-identical retrievals waste an API call but do not degrade the answer, which in this case was better than the reranked version, correctly catching the tension between the *Apology*'s agnosticism about the soul and the *Phaedo*'s arguments for its immortality. But it does mean the "one topic, one subquery" behaviour is not holding as cleanly in practice as it did when tested in isolation. Worth checking whether the pattern is consistent across other single-topic questions; if so, the decomposition prompt likely needs an explicit line stating that returning the question unchanged is a valid output. Parked as a low-priority item, not a blocker for the pilot.

## Changes made in this session, summarised

- `config.py`: added `PINECONE_NAMESPACE`, defaulting to `articles-full`, so everything that queries the index routes through one setting rather than silently defaulting to the old 100-article slice.
- `rag_pipeline.py`: passes that namespace through to `PineconeVectorStore`; smoke test switched from `query()` to `query_multi_concat()` and now prints subqueries.
- `ingest.py`: namespace support, retry with exponential backoff, per-article error handling, end-of-run failure summary, and a corrected TOC-warning threshold that no longer fires on a single failed article.
- `app.py`: new Streamlit pilot interface. Per-person password auth, 15-query session cap, three-turn app-layer memory, collapsible sources with excerpts and links, anonymised Google Sheets logging with a toggle.
- `secrets_template.toml`: template for Streamlit secrets, covering API keys, namespace, logging config, and the per-tester password table.
