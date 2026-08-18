# Multiquery Pipeline: Development Log

Running record of building `query_multi()` and `query_multi_concat()` out of the 2023 notebook prototype. Kept in one place because the debugging arc itself turned out to be the more interesting story than any single fix -- each bug found pointed at a sharper version of the same underlying question the blog series has been circling since Part 1: is the bottleneck retrieval, or is it what happens after?

## 1. Starting point: the 2023 notebook

`4. RAG with Multiquerying.ipynb` used LangChain's built-in `MultiQueryRetriever`, `LLMChain`, and `gpt-3.5-turbo` -- all dated, and the built-in retriever's generic document dedup had no awareness of this project's `max_per_title` cap or section metadata. Decided to rebuild from scratch as `multiquery.py` (LCEL-based sub-query generation) plus a `query_multi()` method on `RerankRAG`, rather than wire the old retriever in directly.

## 2. Bug: paraphrases, not decomposition

First working version of `generate_subqueries()` produced technically correct output -- a list of N distinct strings -- but every single one was still a paraphrase of the full compound question:

> "How does Cicero's res publica compare to Socrates' relationship to > the Laws of Athens?" produced five reworded versions of that exact > question, each still asking about both Cicero and Socrates at once.

This defeats the entire purpose. A paraphrase still forces one embedding to represent two topics simultaneously -- the exact problem multiquery exists to fix. Caught by actually reading the subquery output before trusting it, not by assuming the mechanism worked because it returned the right *shape* of data.

**Fix:** rewrote the prompt with explicit good/bad examples (paraphrase vs. decomposition side by side), since the abstract instruction ("focused on each distinct part") wasn't landing. After the fix, the same question correctly produced two atomic sub-questions plus the original as a safety net -- three total, not six padded rewordings.

## 3. Bug: max_per_title cap fighting the per-subquery floor

Built a guaranteed floor (`min_per_subquery`) so every subquery contributes at least one chunk to the final context, regardless of how it ranks globally -- otherwise a fixed `top_n` can make it mathematically impossible to represent every sub-topic in a many-part question. Reused the existing `max_per_title` cap (originally built to stop one article flooding single-query results) without questioning whether it still made sense in this new context.

It didn't. On the first real test, Cicero and Socrates both live in the same article ("Ancient Political Philosophy"), so their two floor picks used up the entire cap of 2 before a third subquery could get its own pick in. The fill phase, with nowhere else to go, grabbed an unrelated Anaxagoras chunk instead. Confirmed directly by printing actual chunk text, not just title/section labels: the two real, correctly-retrieved chunks looked exactly right, and the system still padded out the remaining budget with noise because of an unrelated cap built for a different problem.

**Fix:** `max_per_title` now defaults to `max(2, number of subqueries)` rather than a fixed constant -- a cap designed for single-query redundancy control shouldn't silently override a guarantee built specifically for cross-subquery fairness.

## 4. Simplification: query_multi_concat()

Rather than keep patching the rerank+floor+cap machinery, built a second, much simpler method: no rerank, no cap, no phase-1/phase-2 logic. Each subquery gets a small fixed `k_per_subquery` (2-3 chunks), all slices dedup and concatenate, straight to generation. This sidesteps the entire bug class from step 3 by construction -- there's no cap to fight a floor that doesn't exist, because there's no competition between subqueries for a shared budget at all.

Trade-off, stated plainly: context size grows linearly with subquery count and isn't bounded the way `query_multi()`'s `top_n` is. Fine for a handful of subqueries; worth watching on the 10- and 18-part demo questions, where an uncapped, unranked pool could get noisy.

## 5. The clean finding: retrieval solved, generation still refused

Ran both `query_multi()` (post-fix) and `query_multi_concat()` on the same Cicero/Socrates question. Printed the actual retrieved chunk text for both. Both methods retrieved clean, correct, on-topic content for *both* halves of the question -- Cicero's exact definition, Socrates' Laws of Athens passage, `query_multi_concat()` even pulled a third genuinely relevant chunk with zero noise.

Both still answered: "the context does not provide a direct comparison."

This is the cleanest version of the Part 3 finding produced so far. Earlier versions of "reranking discards the answer" always had a live confound -- maybe the pool was noisy, maybe recall was imperfect, maybe reranking's narrowing lost something. This time there's no confound left standing: no rerank step, no cap, no noise, both facts sitting directly in front of the model in clean form, and it still declined to connect them. The answer even correctly restated both facts individually before declining -- it wasn't confused about content, it specifically declined the act of synthesis.

## 6. Prompt fix, tested

Original generation prompt: *"Answer the question using only the context below. If the context doesn't contain the answer, say so -- don't make things up."*

Hypothesis: this collapses two different situations into one instruction. "The context lacks the needed facts" and "the context has the facts but never explicitly states the connection between them" are not the same failure mode, but the prompt treats a missing *explicit comparison* the same as a missing *fact*, which is a much more conservative (and here, wrong) standard than the question actually needs.

New prompt explicitly separates the two: still forbids inventing facts not present in context, but explicitly instructs the model to construct a comparison itself from separately-stated facts when the question asks for one, rather than requiring the source text to have already spelled it out. Tested against the exact same retrieved context from step 5 -- same chunks, only the instruction changed.

**Result: fixed.** Both `query_multi()` and `query_multi_concat()` produced genuine, well-reasoned comparisons on the identical question that had failed identically across naive, rerank, and every chunking strategy in Parts 1-3 of the blog series. Not paraphrased restatements sitting side by side -- actual synthesis, with a real closing sentence connecting Cicero's collective-agreement framework to Socrates' individual-conscience-versus-civic-duty tension.

`query_multi_concat()`'s answer was meaningfully richer than `query_multi()`'s -- it incorporated the "corrupt regime is not truly a res publica" point and Socrates' "better to suffer injustice than commit it" principle, both from the third chunk (the Crito/Republic justice discussion) that only the uncapped concat method had room for. The simpler method won this comparison, and not by accident: it had strictly more of the right context available, because there was no cap fighting to keep it out.

**What this closes out:** the prompt change alone took this specific question from a clean, evidence-backed refusal to a correct answer, using identical retrieved content. That's about as isolated a causal result as this kind of testing produces -- worth treating as the headline finding of whichever post covers this arc. The bottleneck named across Parts 1 and 3 (generation declining to synthesize across separately-stated facts) had a real, cheap fix once precisely diagnosed: not a bigger model, not better retrieval, one paragraph of instruction distinguishing "missing facts" from "missing an already-written conclusion."

**Still open:** whether this prompt change holds up across the other composite questions (Anselm, Abelard, Animalism, and especially the 10- and 18-part demo questions), or whether it was well-suited to this one case specifically. Next step before generalizing the finding.

## 7. Head-to-head: cap-fixed rerank vs. concat, same prompt fix applied

Both of these runs used the fixed prompt from step 6 -- earlier attempts, before that fix, produced nothing usable at all: the model refused to answer on this exact question no matter which retrieval method fed it, since the old prompt treated an unstated comparison as equivalent to a missing fact. The comparison below is only meaningful because that ground was already fixed; it's isolating the retrieval-strategy variable, not re-litigating the prompt one.

Both methods now produce genuine synthesized comparisons on the Cicero/ Socrates question -- confirms the prompt fix (step 6) generalizes across both retrieval strategies, not just one.

**`query_multi()` (rerank, cap fixed to 3):** correct, well-structured, but thinner -- built entirely from the two floor-guaranteed chunks (Cicero's definition, Socrates' Laws of Athens passage). The cap fix stopped total starvation of a subquery, but a third genuinely relevant chunk still had to compete for a fill-phase slot against Cohere's relevance judgment across the whole pool, and lost.

**`query_multi_concat()` (no rerank, no cap):** same two facts, plus a third -- the Crito/Republic justice passage -- pulled in because each subquery's top-k arrives unconditionally, with no competitive re-ranking step deciding whether it's "worth" a spot. Answer is accordingly richer: cites Cicero's "corrupt regime is not truly a res publica" and Socrates' "better to suffer injustice than commit it," neither of which appear in the rerank version.

**Takeaway:** the cap fix solved the catastrophic failure (a subquery losing all representation) but not a subtler one underneath it -- useful supplementary context still has to win a relevance contest to survive, and can lose that contest even when it's genuinely good context. `query_multi_concat()` sidesteps this by not running that contest at all. Worth treating as a real point in favor of the simpler method as the default going forward, not just a debugging convenience.

## 8. Control test regression: simple questions get over-decomposed

Ran the deliberately simple, single-topic control question -- "What is the thinking animal argument?" -- specifically to check that multiquery doesn't hurt what naive already handles well. It failed the check.

`query()` alone produced a correct, complete answer: the four premises, the actual philosophers involved (Olson, Snowdon, Carter, McDowell, Ayers), and the real criticism (no principled reason the person is the whole animal rather than just its thinking part).

`generate_subqueries()` split this single-topic question into 5 subqueries anyway -- premises, proposer, context, criticisms, plus the original -- despite the prompt already saying "if the question genuinely has only one topic, a single reformulation is enough." The resulting `query_multi()` answer had the same premises, but dropped the philosopher names and the specific criticism, replaced with a vaguer "supports animalism, biological organisms" line. Splitting an already-well-covered single topic into five subqueries didn't add information -- it diluted the one chunk that already had everything.

**Root cause, on reflection:** the earlier fix (step 2) addressed paraphrasing vs. decomposition -- the sub-questions were now genuinely atomic and single-topic each, which was real progress. But "how many sub-questions to write" was still an implicit side effect of free-text generation length, never an explicit, separately-verified decision. A more detailed prompt telling the model "don't over-split" didn't fully fix this, because the model had no structural checkpoint forcing it to commit to a topic count before writing subqueries -- it could still default to a habitual "give me 3-5 items" pattern regardless of actual question structure.

## 9. Structural fix: topic-first, schema-enforced decomposition

Rather than iterate further on prompt wording (already tried once, partial fix only), moved to a structurally different approach: force the model to explicitly list the distinct TOPICS a question is asking about first, via a Pydantic schema (`with_structured_output`), then generate exactly one subquery per topic -- `len(subqueries) == len(topics)` by construction, not by convention.

This removes the specific degree of freedom that caused the regression in step 8: there's no longer an implicit "how many lines do I feel like writing" decision separate from the topic-identification decision. A single-topic question can only produce one subquery, because there's only one topic to enumerate.

Also added a `return_topics=True` option purely for debugging -- logging the model's identified topics directly shows whether it's calibrating correctly (a real, inspectable signal) rather than only being able to infer miscalibration indirectly from a bad final answer, as happened in step 8.

The system prompt now names the exact step-8 failure as its canonical bad example ("thinking animal argument" premises/proposer/criticisms mis-split into separate topics), rather than a generic warning against over-splitting -- worth testing whether anchoring on the *actual* failure case generalizes better than the more abstract instruction from step 2 did.

**Result: fixed.** Ran `python multiquery.py` directly against both the composite question and the exact simple question that regressed in step 8.

Composite question: `TOPICS IDENTIFIED (2): ["Cicero's res publica", "Socrates' relationship to the Laws of Athens"]` -- correctly identifies two genuinely distinct topics, same as before.

Simple question: `TOPICS IDENTIFIED (1): ['the thinking animal argument']`, exactly one subquery produced, no padding. This is the case that broke under the free-text version -- it previously produced 5 subqueries by splitting the same single topic into facets. The structural fix (explicit topic count enforced by schema, rather than implicit line-count from free text) held where a more detailed prompt alone hadn't.

**What this closes out:** the calibration problem from step 8 wasn't really a wording problem, even though the first attempt at fixing it (step 2) was a wording fix and did work for its specific target (paraphrasing vs. decomposition). "How many subqueries" needed its own explicit decision point, separate from "what do the subqueries say" -- once that was structurally enforced rather than requested, the model stopped drifting into habitual over-splitting on questions that didn't call for it.

**Still open:** whether this holds up under the full notebook run (tests 3-9, especially the 10- and 18-part demo questions, where the model has to correctly identify many real topics without either under- or over-splitting) -- one clean before/after pair is encouraging, not yet a generalized result.

## 10. Full notebook re-run: control test confirmed fixed

Test 3 (the thinking-animal control question) in the actual notebook, not just the isolated `python multiquery.py` check, now produces exactly one subquery, `['What is the thinking animal argument?']`, matching the original question. Answer quality is back to parity with `query()` alone: correct premises, all philosopher names (Snowdon, Carter, McDowell, Ayers, Olson), and the precise criticism (infinite regress of nonidentical thinking parts). The step 8 regression is fully resolved, and the fix generalizes beyond the single isolated test case it was verified against in step 9.

## 11. New finding: single-query retrieval can miss a whole side outright

Re-running test 2 (Cicero/Socrates) turned up something the earlier runs hadn't shown as starkly: `query()` alone, this time, retrieved **zero** Cicero content. Both returned sources were from "Socrates and Plato" -- none from "The Roman Republic and Cicero" at all. Earlier runs on this same question at least surfaced Cicero content and lost it during reranking (step 5's finding). This run, single-query retrieval didn't even find it in the first place.

`query_multi()` and `query_multi_concat()` both found and correctly synthesized both halves, again, reliably.

This sharpens the case for multiquery beyond what step 5 established. The earlier framing was "reranking sometimes discards a chunk that retrieval found." This run shows a second, more basic failure mode underneath that one: single-query retrieval itself can miss one whole side of a composite question, not just lose it during a later step. Multiquery, by construction, doesn't have this failure mode -- each topic gets its own targeted retrieval pass, so there's no single embedding that has to represent both halves and risk favoring one over the other to the point of total omission.

Worth being honest about a limitation here too: this makes single-query retrieval look somewhat non-deterministic run to run on this specific question (found Cicero content in some runs, missed it entirely in this one), which is itself worth a note in whichever post covers this -- either genuine variability in how Cohere's reranker or the vector search handles near-tie relevance scores, or a reminder that "it worked when I tested it once" is a weak standard to trust for a production claim.

## 12. Test 4 (Oruka): multiquery reproduces the same drift, unchanged

Both `query()` and `query_multi()` gave essentially the same answer, close to word-for-word in places, and it's the same substantive drift already documented earlier in the project's eval work on this exact question. The real three claims Oruka set out to counter (Africans don't reason like Greeks, oral tradition can't produce philosophy, African traditions enforce unanimity) live in the article's own "Oruka's Project" section. What both methods actually retrieved and answered from was a different, adjacent list from the ethnophilosophy discussion elsewhere in the same article -- unanimity, anonymity, and the supposed need for a mental leap from myth. Real content, genuinely in the source, not hallucinated -- just not what the question asked about.

Subqueries generated: two, both close paraphrases of the original, not a real decomposition. That's the topic-first fix working correctly, not failing -- this question genuinely has one topic, not several. The failure here was never a single embedding having to represent two competing topics at once, which is what multiquery exists to fix. It's that the correct section isn't ranking highly enough against a topically similar but substantively different one, regardless of how many times or in what phrasing the question gets searched. Multiquery has no mechanism to touch this failure mode, since there's no second topic to split off and target separately.

## REMINDER: pending action item from section 12

The Oruka drift (section 12) still needs a fix. Brainstormed direction, not yet implemented:

**Likely right fix: hybrid retrieval (BM25 + vector + RRF fusion).** This is the specific mechanism suggested in the r/Rag thread from Part 1 -- notably, that fix didn't apply to either of the earlier Cicero or animalism failures, since recall was fine in both of those. This Oruka case is the first one in the project where hybrid retrieval plausibly would help: the correct section ("Oruka's Project") almost certainly shares strong lexical overlap with the question ("Oruka," "three," "negative," "claims") that a keyword signal would catch even where semantic similarity currently favors the wrong-but-adjacent ethnophilosophy section instead.

**Why not multiquery or cross-reference following:** neither applies. There's no second topic to decompose (confirmed -- section 12 already shows the topic-first fix correctly generating just close paraphrases, not a real split, since this genuinely is one topic). And the wrong section isn't a stub pointing at the right one, it's a complete, self-contained wrong answer competing on equal footing -- not the "mention vs explanation" pointer failure from Part 3 either.

**Suggested order of attack, cheapest first:**
1. Check whether "Oruka's Project"'s chunk text has strong literal lexical overlap with the question (e.g. "three" near "claims") -- if so, even a crude keyword boost might fix this specific case before building full hybrid infrastructure.
2. If not enough, build real hybrid search -- now has a concrete, motivating failure case behind it rather than a theoretical one.

**Not yet done.** Revisit this before treating the multiquery arc as closed.

## 13. Test 5 (Anselm): both methods succeed, near-identical quality

Composite question spanning two sections of the same article -- God's mercy/justice (section 3) and freedom/sin (section 4). Both `query()` and `query_multi()` produced correct, complete, well-reasoned answers connecting the two halves: God's supreme goodness requiring mercy alongside justice, tied to the angels' possession of both a will for happiness and a will for justice, giving them genuine self-initiated choice.

Subqueries generated: four -- "how does Anselm reconcile mercy and justice," "what is Anselm's view of freedom," "what is Anselm's perspective on sin," plus the original question. This is a real decomposition, not a paraphrase repeat like test 4 -- three genuinely different sub-topics identified, matching the question's actual structure (a definition/reconciliation half plus two related but distinct doctrinal topics, freedom and sin, that the source treats in adjacent but separate subsections).

Difference between the two answers is minor and mostly in framing: `query()` frames the connecting concept as "self-initiated action"; `query_multi()` leads with "rectitude of will" (Anselm's own term for what freedom actually preserves) before getting to self-initiated action, arguably the more precise framing since rectitude of will is what Anselm's own definition of freedom is built on. Neither answer is wrong; multiquery's is marginally closer to the source's own terminology.

**Read on this one:** unlike test 4, this composite question did benefit from real decomposition, and the outcome was a wash-to-slight- win rather than a fix for a documented failure -- `query()` already handled this one well on its own, since both halves apparently rank adequately even under a single embedding here. Useful data point for the emerging picture: multiquery's win is concentrated in cases where single-query retrieval specifically struggles (Cicero/Socrates, repeatedly), not a blanket improvement across every composite question.

## 14. Test 6 (Abelard): a clean multiquery win, and a genuinely interesting answer

`query()` alone failed outright on this one -- retrieval apparently surfaced only the ethics/intentions content and nothing on universals, so the single-query answer explicitly states it can't make the connection because half the needed material simply isn't in context. Same shape of failure as the earlier Cicero case: one embedding, representing two topics, effectively starved one side.

`query_multi()` retrieved both halves cleanly and did something more interesting than a simple "here's fact A, here's fact B, here's how they relate" -- it identified a real structural parallel between Abelard's metaphysics and his ethics that neither retrieved chunk states explicitly: both positions privilege the particular, individual case over abstract, generalized categories. Universals are "merely words," not real entities standing over concrete individuals; moral worth lives in the individual agent's particular intention, not in some general category of "the deed." That's a genuine philosophical synthesis, not just concatenation of two facts side by side -- worth flagging as one of the stronger pieces of evidence in this whole log that the generation-layer prompt fix (step 6 in this log) is doing what it was built to do: constructing a real connection from separately verified facts, not just restating them next to each other.

**Read:** consistent with the pattern emerging from tests 4-5 -- multiquery's real advantage shows up specifically where single-query retrieval concretely fails to represent both halves of a composite question, which happened here just as clearly as it did with Cicero.

## 15. Test 7 (Animalism): another clean multiquery win, though thinner than test 6

`query()` alone failed again in the same honest way as test 6 -- retrieved the thinking animal argument content but not the organic/ somatic distinction, and correctly declined to connect them rather than fabricating a link. This is the same article that produced the confirmed rerank failure earlier in the project's main eval (naive succeeded, rerank failed on this exact organic/somatic question) -- worth noting this article specifically seems to be a recurring source of single-path retrieval trouble across different pipeline configurations, not a one-off.

`query_multi()` retrieved both halves and did connect them, but the connection is real rather than a genuine insight the way test 6's was -- it correctly states that both the thinking-animal argument and the organic/somatic debate concern "the conditions of animal continuity and the nature of human animals," which is true and relevant, but doesn't actually explain the substantive relationship the question was asking about (does the thinking animal argument, on its own, favor one side of the organic/somatic split over the other, or is it neutral between them). The answer is accurate and non-fabricated, but somewhat more descriptive-of-both-topics than genuinely synthetic -- a real answer, just a thinner one than test 6's.

**Read:** consistent with the pattern from tests 4-6 -- another case where single-query retrieval concretely failed and multiquery fixed the retrieval side of the problem. But this one is a useful reminder that fixing retrieval doesn't automatically guarantee the generation step produces a deep synthesis every time -- sometimes what comes out is closer to "both topics discussed, here's how they're thematically related" rather than a real substantive connection. Worth checking whether this is prompt-sensitive (would a stronger synthesis instruction help here specifically) or whether this particular question just doesn't have as tight a substantive link between its two halves as the Abelard case did.

## 16. Tests 8-9, rewritten as natural prose: the real stress test

Both extreme demo questions were rewritten from numbered lists into genuinely flowing prose before this run, specifically so the decomposition step would have to find topic boundaries from sentence structure rather than from pre-marked "(1)... (2)..." delimiters -- a much closer approximation of how a real user would actually type a long, rambling question.

### Topic count: held up well

### Real headline: multiquery decomposition is what makes these questions answerable at all

Single-query retrieval, working from one embedding trying to represent 10 or 18 genuinely distinct topics at once, could only ever land on one of them (Abelard, both times) -- everything else had no path to an answer regardless of what happened downstream. On the 10-point question it at least admitted the rest was missing; on the 18-point question it did something worse, answering only Abelard and not flagging any of the other 17 as unaddressed at all, a silent gap rather than an honest one.

Multiquery decomposition is what made the other 9-17 topics retrievable in the first place. Topic count held up at both scales: 10 subqueries for the 10-point prose question, 18 for the 18-point one, correctly identified from natural language transitions ("Switching over to," "Turning to," "And finally") with no explicit numbering to lean on. That's the real, load-bearing finding of this test -- the structural fix generalizes well beyond the small 1-2 topic cases it was originally verified against, and without it, most of both questions simply couldn't be answered by this pipeline at all.

### Second-order question: what to do with the subqueries once decomposition succeeds

Given decomposition worked in every version of both tests, the remaining question is how best to use the resulting subqueries -- reranked, or simply concatenated per-subquery. This is a real, worth -logging finding, but secondary to the decomposition result above.

Both the 10- and 18-point rerank runs performed poorly, and the 18-point one failed completely -- flatly declined to answer anything at all, despite 18 correctly-targeted retrieval passes behind it. This is a real, currently-unexplained regression: reranking a pool built from 18 separate subqueries, then applying the same floor/cap logic tuned and validated on 2-3-subquery cases, may simply not scale the way it was designed to. Not yet diagnosed -- likely candidates are the effective_top_n scaling becoming unworkable at this subquery count, or Cohere's rerank call itself struggling to meaningfully differentiate relevance across a much larger, more topically diverse candidate pool than it was ever tested against.

On both questions, the no-rerank concat version was the method that actually delivered on what decomposition made possible -- covering the full breadth of what was asked, with real, mostly accurate content for each topic, some genuinely well-paired (induction/abduction with the bad lot argument, Bakke with SFFA v. Harvard). Every method tested so far in this log that skipped rerank in favor of simple per-subquery slices has matched or outperformed its reranked counterpart; here, at 18 subqueries, reranking didn't just underperform, it produced a complete non-answer where the simpler method, working from the exact same correctly-decomposed subqueries, produced something genuinely useful.

**Not yet verified:** how factually accurate the no-rerank concat answers actually are across all 10-18 points -- this log has confirmed breadth of coverage, not yet checked each individual claim against source text the way earlier single-topic tests were checked. Worth a real accuracy pass before treating this as a finished result, not just a promising one.

## 17. Brainstorm: implications for the paper-referee tool

Raised while reviewing tests 8-9: these extreme multi-part questions are a strong proxy for what a referee tool would actually need to do -- check many distinct claims in a submitted paper against SEP in a single pass, the same "many topics, one question" shape just stress-tested here.

**The head-to-head against frontier models is now concrete, not hypothetical.** The same 10- and 18-part questions, run against plain ChatGPT/Claude/Gemini with no corpus access, would very likely either hallucinate confidently on unfamiliar SEP-specific specifics (Powell's exact Bakke reasoning, Gaunilo's Lost Island, Oruka's specific claims) or hedge vagely across the board. This system, even with the rerank regression unresolved, gave real citations and mostly-correct specifics for 11+ distinct claims in one pass using the no-rerank path. That comparison is a real demo, not a claim needing further justification before it can be shown.

**The rerank failure at scale is a referee-relevant finding, not just an engineering footnote.** Finding this collapse now, before building the referee tool's core loop, is exactly the kind of thing worth catching early -- a referee tool built on the reranked path would have inherited a broken mechanism for its single most common real use case (checking many claims at once).

**Working conclusion:** the referee tool's core retrieval loop should likely be built around `query_multi_concat()`'s approach (decompose, retrieve per-topic, skip the rerank competition), not `query_multi()`'s current reranked path, given the evidence so far.

**Caution before treating this as settled:** breadth of coverage is not the same as accuracy. Before committing to this architecture, worth spot-checking specific claims in the 18-point concat answer against real source text, the same discipline applied to every earlier single-topic test in this log. A referee tool that is wrong with confidence is worse than one that is incomplete but honest about it.


## REMINDER: To Do:

1. **Run the frontier-model head-to-head.** Paste the exact 10- and 18-part demo questions into plain ChatGPT/Claude/Gemini with no corpus access. Expect confident hallucination on unfamiliar SEP-specific details (Powell's exact Bakke reasoning, Gaunilo's Lost Island, Oruka's specific claims) or vague hedging across the board. This is the actual demo for the referee-tool pitch, not a hypothetical -- the grounded system already gave real citations and mostly-correct specifics for 11+ distinct claims in one shot; the comparison just needs to be run and captured.

2. **Treat the rerank failure at 18 subqueries as a referee-relevant finding, not just an engineering footnote.** A referee tool checking a real paper against SEP will routinely need to verify many distinct claims in one pass -- the same shape just stress-tested here. Catching this collapse now, before the referee tool's core loop gets built on it, is the win -- worth stating that plainly wherever this gets written up.

3. **Architectural decision: build the referee tool's core retrieval loop around `query_multi_concat()`'s approach, not `query_multi()`'s current reranked path.** Concrete, evidence-backed, not a vague intuition -- decompose, retrieve per-topic, skip the rerank competition.

4. **Caution to hold onto before treating any of this as settled:** breadth of coverage is not accuracy. Spot-check a handful of the 18-point concat answer's specific claims against real source text before treating "the simple method wins" as a finished result worth building a product decision on -- same discipline as every other finding in this log.

**Then: prep the blog.** Once the above is done, move to pulling this full multiquery arc (sections 1-17 plus the recent additions) into blog-ready material.

## 18. Correction to item 1: frontier models with search DON'T hallucinate here

Ran the frontier-model comparison, but not quite the test condition planned. The output reviewed was a frontier model's "highest effort, research enabled" answer to the exact 10-part demo question -- not a no-corpus-access baseline. The prediction in the reminder was that a plain frontier model would hallucinate confidently or hedge vaguely on unfamiliar SEP-specific details. That prediction was wrong for this test condition.

The actual output was strong: correct, precisely-sourced answers across all 10 topics, real quotes with correct attribution, correct section numbers, correct legal citations with real dates, and honest flagging of genuine scholarly disputes (e.g. noting translators disagree on how to render Cicero's *iuris consensu*) rather than false confidence. Quality was comparable to, arguably exceeding in citation precision, this project's own best output on the same question.

**Why this makes sense in hindsight:** SEP is a public, well-indexed website. A frontier model with real search access can effectively build its own on-the-fly retrieval over the same source material this project draws from. "No grounding" was the wrong assumption for a research-enabled model -- the hallucination risk this project has been solving for applies specifically to models *without* search access, a narrower and less interesting comparison than the one originally planned.

**Real implication for the referee-app pitch:** the differentiator can't be "we prevent hallucination frontier models can't avoid," because this result shows a well-equipped frontier model can avoid it, at least on well-indexed public content like SEP. The honest differentiators are elsewhere:
- **Cost and latency at volume.** This was explicitly a slow, expensive "highest effort" invocation for one question. A referee tool needs to run this shape of check hundreds of times cheaply, not once expensively.
- **Reproducibility and auditability.** A frontier model's live web search is a black box, not fully controllable or reproducible run to run. This pipeline's retrieval is deterministic and inspectable -- every claim traces to a specific, loggable chunk.
- **Corpus control.** This output pulled in non-SEP legal sources alongside SEP content, reasonable for a broad research question, but a liability for a referee tool that specifically needs "does this match what SEP says," not "does this match what's generally true across sources."

**Not yet done:** the actual planned comparison (frontier models without search/research access) still hasn't been run. Worth doing that too, since it's a fairer test of the narrower claim this project can actually defend -- but the broader "we beat frontier models generally" framing from the plan needs to be retired, not just delayed.

## 19. Refined referee-app pitch, after the frontier-model correction

Decided not to run the remaining planned frontier-model comparison (item from the reminder) -- section 18's finding already forces a rethink of the pitch itself, running more comparisons on the old framing wouldn't add much.

**Sharper, more honest version of the differentiator**, replacing "we prevent hallucination frontier models can't avoid" (disproven in section 18): a referee tool's actual job isn't answering one research question well, it's taking an arbitrary paper, decomposing it into its many individual checkable claims, and running each one against a corpus-restricted database independently -- structured, per-claim verdicts (supported / contradicted / no evidence, with citation), not a flowing research essay.

A sufficiently scaffolded frontier agent could in principle be prompted to attempt this. The honest claim isn't "frontier models cannot do this" -- it's that doing it well requires exactly the engineering this project has spent this session building and debugging, which a generic deep-research invocation doesn't provide out of the box:

- **Structured, per-claim output** rather than prose -- a referee needs "claim 4: supported by SEP §2.3, here's the passage," not an essay that happens to touch the claim somewhere.
- **Genuine corpus restriction.** Section 18's frontier output pulled in law reviews and legal opinions alongside SEP -- reasonable for a broad research question, wrong for a tool whose job is specifically "does this match what SEP says," not "what's generally true anywhere on the web."
- **The floor guarantee, specifically.** A paper might have 50-100 checkable claims. Without the per-unit floor built and debugged in sections 3-4 of this log, exactly the kind of silent starvation seen in the reranked 18-part test (section 16) would recur -- except now it's "claim 40 of a real paper got silently dropped," a much higher- stakes failure than a dropped demo topic.
- **Cost and determinism at real scale.** Running an expensive "highest effort, research enabled" pass per claim across 50-100 claims in an actual paper is a structurally different cost problem than running it once for a demo question.

**Reframe worth keeping:** tonight's 10- and 18-part demo questions weren't only a multiquery stress test. In retrospect they were an unintentional working prototype of the referee tool's actual core mechanism -- decompose into atomic units, retrieve targeted evidence per unit without starving any of them, combine into a structured answer. The referee tool isn't a new build from scratch; it's this same pipeline, pointed at a paper's claims instead of a question's topics.

**Honest caveat to carry forward:** none of this has been tested against a real paper yet. The claim above is a reasoned extrapolation from the obtained evidence, not itself a tested result -- worth flagging clearly wherever this framing gets used until an actual paper has been run through the pipeline.

## 20. Strategic pivot: referee tool becomes the primary project

Decided to redirect the project's primary framing. The RAG chatbot, previously the whole product, becomes a companion feature of a paper-referee tool -- once a paper gets structured feedback, the chatbot lets you discuss the specific pieces of that feedback. The referee tool becomes the main product.

**Market check, done before committing to this:** searched for existing tools that decompose a paper into individual claims and check each one against a single curated authoritative source, philosophy-specific or otherwise. Found active general "AI peer review" research (a recent arxiv paper specifically on reliability risks of AI referees for STEM venues, hybrid AI/human desk-review pilots at AAAI 2026) and strong literature-grounding tools (Elicit, Paperguide, Scite) built for finding and citing existing broad research. Found nothing doing the specific thing described here: claim-by-claim decomposition checked against one curated, authoritative corpus. Held with appropriate humility -- absence from search results isn't proof of absence -- but a genuinely positive signal for the pivot.

**Why the pivot is the right call:** the actual debugged mechanism (decompose into atomic units, retrieve targeted evidence per unit without starving any of them, per section 19) already is the referee tool's core loop, just pointed at a question's topics instead of a paper's claims. The chatbot alone was a much more commoditized pitch; this is sharper and better matches what actually got built.

**Scoping caution, worth holding onto:** SEP-grounding catches a real and valuable class of error -- misattributed views, misstated facts, claims that contradict established scholarly consensus -- but cannot evaluate whether a genuinely novel argument is any good, since novel contributions by definition won't have a match in SEP. "Referee" as a name risks overclaiming full argument-quality judgment. Working name going forward should reflect the actual scope: a scholarly grounding and consistency checker, not a full peer-review replacement -- same honesty discipline this project has run on throughout.

## 21. Staged roadmap toward an actual referee, and priority

Full staged roadmap, ordered by how much reuses infrastructure already built versus what needs genuinely new capability:

1. **Objection-and-response checking.** SEP entries routinely have explicit "objections" or "criticisms" sections built into the source structure. Given a paper's thesis, retrieve documented objections to that specific position and check whether the paper addresses them. Reuses the exact decompose/retrieve/floor-guarantee machinery already built and debugged (sections 2-9) -- just retrieving "known objections to X" instead of "definition of X." Nearly free given what already exists.
2. **Argument reconstruction.** New capability -- extract a paper's stated argument into explicit premises and a conclusion as a structured object, not prose. Once extracted, each premise can be checked individually against the corpus (sharper than checking the paper's general topic), and basic validity checking becomes possible (does the conclusion follow, is there a fallacy pattern in the inference).
3. **Novelty and divergence, not binary match/no-match.** Current grounding is essentially pass/fail. A referee-relevant version asks how much a position overlaps with an existing canonical view and specifically where it diverges -- "differs from the standard formulation in claiming X" is a real referee comment; "doesn't match anything in the corpus" is not, since original contributions are supposed to not match.
4. **Internal consistency checking**, no external corpus at all -- decompose the paper itself into claims and compare them against each other across sections, catching self-contradiction independent of any database.
5. **Corpus expansion** (more sources beyond SEP), deliberately last, not first -- expanding the database before the argument-evaluation layer above exists just produces a bigger grounding-only tool, not a smarter one.

**Deliberately excluded from the roadmap, stated honestly:** actual judgment of whether an argument is good, independent of validity and grounding -- whether the premises are the right ones to focus on, whether the project is interesting or worth pursuing -- is not being promised as automatable. That stays genuinely human for the foreseeable future. Honest positioning: "flags what a human referee would want to look at closely," not "renders a verdict."

**Decided priority for the project going forward: stages 1-2 first.** These are the most useful and immediate steps, and both build directly on infrastructure already debugged.

**New feature, logged for the roadmap:** user-uploadable document corpus, on top of SEP, so a draft can be checked against SEP plus a philosopher's own relevant citation list or recent papers they're responding to, not just the fixed encyclopedia corpus. Architecturally reasonable to build on the existing ingestion pipeline -- `chunker.py`, `embeddings.py`, and `vectorstore.py` are already generic, not SEP-specific, and `section_parser.py` already has a regex-fallback and whole-document-fallback path (section 12-era `_regex_fallback` logic) for documents without a usable TOC, which is exactly the shape most user-uploaded PDFs would need. The real new work is namespace/user-scoping (so one user's uploaded corpus doesn't leak into another's queries) and an upload/ingest entry point, not a new retrieval architecture.

**Closing this session's log here.** Full night's arc: multiquery build and three real bugs found and fixed (paraphrase-not-decomposition, max_per_title vs. floor conflict, free-text vs. structured decomposition), a clean isolated prompt fix, a genuine scale failure at 18 subqueries still not fully diagnosed, the Oruka drift case still open (hybrid retrieval flagged as the likely fix, not yet built), the frontier-model correction, and the resulting strategic pivot toward the referee tool with an honest staged roadmap and scoping discipline carried through to the end.
