# Oruka Retrieval Bug: Investigation Log

Follow-up investigation into the still-open Oruka drift case first flagged in multiquery_dev_log.md section 12. Separate log because this is a focused diagnostic thread, not part of the multiquery build itself.

## The bug, restated

Question: "What three negative claims about African philosophy was Oruka trying to counter?" Both naive and multiquery retrieval consistently answer with a real but wrong-but-adjacent list, claims from a different section of the same article ("Ethnophilosophy, Unanimity and African Critical Thought"), not the actual three claims stated in "Oruka's Project."

## Theory 1: hybrid search would fix a ranking competition between two sections

Working assumption going in: the wrong section wins a head-to-head ranking fight against the correct one, and keyword-aware hybrid search would tip the balance back.

**Disproven.** Pulled the actual text of both sections. The wrong section ("Ethnophilosophy...") opens with "The third negative claim Oruka aimed to challenge..." -- language that deliberately echoes the correct section's own phrasing ("aimed to counter three negative claims"), because it's explicitly referencing back to a list the correct section already established. Both sections would score similarly well on any keyword-overlap metric. A pure lexical signal can't cleanly distinguish "states the three claims" from "references back to them." Hybrid search, as originally scoped, wouldn't reliably fix this.

## Theory 2: this is the mention-vs-explanation pointer pattern from Part 3

The wrong section's opening ("the third negative claim") is an implicit backward reference, the same shape as the animalism case's "as we saw in section 1.2," except with no explicit citation to follow. Reasonable theory, not yet tested against real evidence at the time.

## Checking the actual chunk boundaries

Before testing theory 2, checked whether the correct section's answer-bearing content was even chunked in a retrievable way.

Pulled section 1 ("Oruka's Project") through the current `section_parser.py` + `chunker.py` directly. Confirmed `parsed_how: toc_match` -- all 5 sections parsed cleanly, no TOC issue here.

Section split into exactly 2 chunks. Chunk 0: the Akoko communalism quote. Chunk 1: the tail of the Chaungo truth quote, its analysis, *and* the actual "Oruka's survey of sages aimed to counter three negative claims..." sentence with the full numbered list, plus a closing transition paragraph.

**This chunk is a strong, not heavily diluted, match.** Roughly half of chunk 1 is directly and specifically about the three claims, in near-verbatim phrasing to the actual question. Not the "buried behind an unrelated quote" dilution originally suspected.

## Checking for index staleness

Given how much the chunking code changed over the course of this project, worth checking whether the live Pinecone index actually reflects current chunking logic, or whether it's stale from an earlier `ingest.py` run.

Queried the production index directly by metadata filter (not similarity) for every chunk under "African Sage Philosophy." Confirmed `Oruka's Project | chunk 1.0` is present in the live index, matching exactly what local re-chunking produced. **Staleness ruled out.** The correct, well-formed, strongly-matching chunk exists in the index and simply isn't surfacing in retrieval.

## Where this leaves the diagnosis

The chunk is real, present, well-formed, and phrased close to the question. It still doesn't make top-10 retrieval. That rules out staleness and weakens (without fully killing) the dilution theory.

**Refined theory 3, not yet tested:** this could be a crowding-out problem rather than a single head-to-head ranking loss. The metadata dump showed "What counts as Sage Philosophy?" has 11 chunks and "Ethnophilosophy, Unanimity and African Critical Thought" has 6, both far more than "Oruka's Project"'s 2. If several chunks from those larger sections are each individually decent (not great, but decent) matches to the question, thanks to shared high-level vocabulary (unanimity, philosophical status, Oruka, critique), they could collectively crowd the one genuinely correct chunk out of a small top-k window, even if the correct chunk's raw similarity score is respectable on its own.

## Rank-position check: the real answer

Ran a full similarity search (k=50) against production, target question, and located exactly where the correct chunk ranks.

**Result: rank 22, score 0.5682.** Top result (wrong section) scores 0.7989 -- a large, substantial 0.23-point gap, not a narrow miss.

**Two compounding causes identified, not one:**

1. **Content purity.** The wrong chunk is nearly 100% about "false claims Oruka countered" -- dense, single-topic, echoing the question's framing throughout ("negative claim," "false view," "unanimity"). The correct chunk is roughly half Chaungo's truth-quote analysis, half the actual three-claims content -- diluting its score even though the matching sentence is present. The earlier dilution theory wasn't wrong, just incomplete without this comparison.

2. **Cross-article crowding.** Ranks 9, 10, 12, 14-16 in the top-50 are from "Africana Philosophy" and "Contemporary Africana Philosophy" -- different articles entirely, broadly on-topic, each contributing several moderately-scoring chunks. The correct chunk isn't just losing to one wrong chunk, it's buried under a wider cluster of thematically-adjacent-but-different-topic competition from articles outside the one actually being asked about.

**Promising lead worth testing before building full hybrid infrastructure:** the correct chunk's exact phrase is "aimed to counter three negative claims" -- close to verbatim overlap with the question's "three negative claims." The wrong chunk says "the third negative claim" -- an ordinal reference, not the same multi-word phrase. A lexical/phrase-match signal targeting exact n-gram overlap, not just individual term frequency, could plausibly favor the correct chunk in a way pure semantic similarity currently doesn't.

**Status: real, well-understood diagnosis reached. Fix tested, partial success at initial weight, further tuning in progress.** Built `notebooks/10_Oruka_Retrieval_Fix.ipynb`, reproducing the full diagnostic chain as runnable cells and testing the phrase-overlap fix directly.

## Fix test results, LEXICAL_WEIGHT = 0.3

Confirmed via TF-IDF cosine similarity blended with vector similarity (0.7 vector weight, 0.3 lexical weight) across the same top-50 candidates from the rank-position check above.

**Target chunk rank moved from 22 to 13.** Real, measurable improvement, not yet enough to land inside a typical top-10 retrieval window.

**Notable confirming detail:** the target chunk's lexical score, 0.1537, is the *highest* of any chunk in the resulting top 15, higher than the wrong chunk's 0.1119. This directly confirms the phrase-overlap hypothesis was correct in direction -- the target chunk actually does have stronger literal phrase overlap with the question than everything around it. The reason it still doesn't crack the top 10 at this weight is that the underlying vector-score gap (0.7988 vs. 0.5682, a 0.23 point difference) is large enough that a 0.3 lexical weight, while it helped, wasn't sufficient to fully close it.

**Not yet done:** testing a higher lexical weight (0.5-0.6 range) to see whether the target chunk can be pushed into single digits, and running the Part 3 regression check (Cicero and thinking-animal control questions) to confirm this approach doesn't cost accuracy somewhere that was already working under pure vector search. Both results needed before deciding whether this fix is worth adopting.

## Fix test results, LEXICAL_WEIGHT = 0.5

**Target chunk rank moved from 22 to 8**, a meaningful result -- this now lands inside a realistic top-10 retrieval window, not just a directional improvement. At this weight the target chunk's combined score (0.3610) sits closely behind several other African Sage Philosophy chunks, no longer buried under the wide field of cross-article competition that dominated the unweighted ranking.

## Regression check results

Both control questions pulled from `SEP/tests/eval_systematic.csv` were re-run with the LEXICAL_WEIGHT = 0.5 blend and compared against plain vector search.

- "What is the thinking animal argument?" -- top result identical before and after (Animalism | Arguments for and Objections to Animalism).
- "Why does Cicero say Rome under the Republic satisfies the definition of a res publica?" -- top result identical before and after (Ancient Political Philosophy | The Roman Republic and Cicero).

**Zero regression on either control question.** The blend changed nothing about the top result on cases that were already working correctly under pure vector search.

## Verdict

The lexical phrase-overlap fix works as hypothesized, moves the target chunk from rank 22 to rank 8 with no observed cost on two independent, already-validated test cases. This is a promising result for a fix built from a lightweight TF-IDF signal rather than full BM25/hybrid infrastructure, and it directly confirms the phrase-overlap theory from earlier in this log: the target chunk's lexical score was already the highest in its neighborhood even at the lower weight, this was about giving that signal enough influence to matter against a real vector-score gap, not about the signal being weak or absent.

**Not yet done, worth doing before calling this settled:**
- Testing on a broader set of questions beyond these three (the original bug case plus two regression controls) to see whether LEXICAL_WEIGHT = 0.5 is a good default or happens to be tuned to this specific handful of cases.
- Deciding whether to bake this into `rag_pipeline.py`'s retrieval path directly, or keep it as an optional mode, given it changes scoring behavior for every query, not just ones like the Oruka case.
- Considering whether rank 8, while inside a typical top-10 window, is actually reliable enough given how close the target's score sits to its neighbors -- worth checking whether a slightly different phrasing of the same question holds the improvement or loses it.


## Status: NOT yet implemented in the codebase

Everything above is tested and confirmed inside `notebooks/10_Oruka_Retrieval_Fix.ipynb` only. None of `rag_pipeline.py`'s real query methods (`query()`, `query_naive()`, `query_multi()`, `query_multi_concat()`) currently do any lexical blending -- production retrieval is still pure vector search, unchanged. Wiring this in is a deliberate future decision, not an oversight, given the three open questions above (generalization, default-on vs. optional, rank-8 stability) are worth answering on a wider test set before it becomes the silent default for every query.
