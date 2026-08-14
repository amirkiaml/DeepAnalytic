## Wrapping The Notebook: What I Actually Learned

I went into this wanting a clean answer: which of five chunking strategies retrieves best? Section-aware, fixed-window, paragraph, sentence-window, semantic. Ten articles, a handful of single-topic questions, then some harder composite ones, then one formal/symbolic article just to see if anything broke.

I didn't get a clean winner. For a while that felt like a wasted afternoon. It wasn't, and here's why.

### What didn't happen

No strategy pulled decisively ahead on straightforward, single-topic questions. Section-aware, fixed-window, and paragraph-based chunking retrieved nearly identical chunks on most of the questions I tried, word for word in places. When I ran those through generation too, section-aware's answers came out slightly thinner than the others on the same underlying evidence, small enough to be noise but worth flagging rather than smoothing over. That's not what I expected. Section-aware chunking was built on the assumption that respecting an article's real structure (via its table of contents) would beat blindly splitting by token count. On this corpus, that assumption mostly didn't pay off. The paragraph breaks in a well-written SEP article already line up with the topic breaks closely enough that a dumb splitter gets almost the same result as a structure-aware one.

### What did happen, decisively

**Semantic chunking is out.** It was the only strategy that ever clearly won on a question (catching a specific detail, like Abelard's sleeping-monk example, that nothing else found) and the only strategy that ever clearly lost (drifting into an unrelated passage about heresy when asked about universals). That volatility is itself the finding. I don't want my retrieval quality depending on a coin flip, so semantic chunking is disqualified as a default even though it has a real, occasionally impressive ceiling.

**The chunking layer isn't where the real problem lives.** This is the one that actually matters. When I pushed into harder composite questions (comparing something from one part of an article to something from another part entirely), nearly every strategy failed the same way: not by failing to retrieve the relevant content, but by retrieving it and then refusing to synthesize it into an actual comparison. The evidence for both halves of the question was sitting right there in the context window, and the model still said "the context doesn't provide a direct comparison." That's not a retrieval problem. That's a generation problem, and no amount of re-chunking the source text was going to fix it.

Put together with what I found yesterday (the reranker sometimes discarding the one chunk that actually contained the answer, even when a wider net had caught it), a pattern is forming. The bottleneck in this pipeline isn't how the text gets cut up. It's what happens after retrieval: which chunks survive the narrowing step, and whether the model is willing to reason across what it's given rather than just paraphrasing whichever chunk looks most directly on-topic.

### The decision

I'm keeping section-aware chunking as the default. Not because it won the comparison (it didn't, it mostly tied), but because nothing beat it, it matches the real structure of the source articles (which matters for how I explain this project later, even where it didn't move the numbers), and switching to a roughly-tied alternative isn't worth the churn. Semantic chunking is ruled out for anything that needs to run unattended, given how unpredictable it was. One thing worth a second look later: semantic was also the strategy most willing to attempt real synthesis on the composite questions, when the other four mostly just refused. That doesn't undo the volatility problem, but it's a data point in its favor I don't want to lose track of.

### Where the effort goes next

Not into a sixth chunking strategy. Into the generation layer. The plan was always to build a multiquery and self-critique step eventually, and this experiment gave me a real reason to move that up rather than treat it as a someday-project: it's the actual bottleneck, demonstrated with real questions and real failures, not a guess.

One cheap thing worth trying before building all of that: a small prompt tweak that explicitly invites the model to connect evidence across chunks rather than requiring the source to state the comparison outright. If that alone fixes some of these refusals, it tells me part of this is a prompt problem, not purely an architecture problem, and it's worth ruling that out before investing real time in the bigger agentic build.

## Final Words

A little deflated that there's no single "winner" chunker to point to. But looking back at it straight: I ran a real test, on real data, with a hypothesis stated in advance, and the result was "this variable matters less than I thought, here's the one that matters more." That's a better outcome than confirming what I already believed going in would have been. It's just less satisfying to write up.

## Idea for later: chunks that mention vs. chunks that explain

Sometimes a retrieved chunk matches on similarity because it *mentions* a concept or phrase, not because it *explains* it. The real explanation may live in a different section that the matched chunk only references or gestures at. A generator working from the mention-only chunk produces a thin or vague answer even though a genuinely good answer exists elsewhere in the same article.

Worth exploring: using SEP's own cross-reference structure (already flagged in the portfolio strategy doc as a disambiguation asset) to detect when a matched chunk points to a fuller explanation elsewhere, and pull that section in too rather than trusting the first match alone. This is a form of multi-hop or graph-aware retrieval, not a chunking-strategy fix. Comes after multiquery/self-critique in priority, but same root cause: retrieval finding *a* relevant chunk isn't the same as retrieval finding *the* explanatory chunk.

### Refinement, found while scoring the second full run

**Source note:** the original idea (mention vs. explanation, above) came out of manually reading a small number of hand-picked questions directly in `9_Chunking_Strategy_Comparison.ipynb` -- side-by-side chunk comparisons across the five chunking strategies, no scoring rubric involved, just eyeballing output. The refinement below comes from a different source: the systematic naive-vs-rerank eval (`run_eval.py` against `tests/eval_systematic.csv`), scored against `scoring_rubric.csv`, with a second independent human pass spot-checking the highest-stakes rows once `Chunks_Consulted` actually had real text in it. Different tool, different method, same underlying pattern showing up twice -- which is part of why it's worth trusting.

Once `Chunks_Consulted` actually had real text in it, the same pattern turned up in both of our two known rerank failures, confirmed directly rather than inferred from source titles.

**Animalism (organic vs. somatic):** the reranked context kept a chunk that names both terms and points at where they're actually explained: *"As we already saw in section 1.2, for instance, the source of animalist opposition to the psychological criterion may stem from either organicist or somaticist commitments."* That's an explicit backward reference to the real content. The two chunks that naive kept and rerank dropped are exactly the ones section 1.2 is pointing at. The pipeline had no mechanism to follow that pointer.

**Ancient Political Philosophy (Cicero's res publica):** here retrieval actually did keep the chunk with the real definition (res populi, agreement on law). But the definition passage doesn't itself state *why Rome specifically* satisfies it -- that connection is made across the surrounding institutional-structure discussion (Polybius, the mixed constitution) rather than stated as one self-contained claim. So this isn't quite the same failure as animalism: retrieval succeeded, but the specific causal link the question asked for was implicit, assembled across the section rather than sitting in one chunk. Scored this one lower on accuracy than the first pass did, since it's a genuinely harder case than "the model just refused to use content it had" -- it's closer to "the model had the material but the causal claim required synthesis the material doesn't spell out directly."

**African Sage Philosophy (Oruka's three claims)** shows a third, harder variant: the retrieved chunk opens with *"The third negative claim Oruka aimed to challenge..."* -- an implicit backward reference (ordinal counting) with no explicit citation to follow, unlike the animalism case. Section 1, which actually lists all three claims together, never got retrieved at all. There's no link here to detect and follow; the model would need to recognize that "the third" presupposes content it doesn't have, rather than silently reconstructing a plausible three-item list to fill the gap.

**Working distinction for later:** at least two different sub-cases live inside the same broader idea --
- **explicit pointer** (animalism's "as we saw in section 1.2") -- theoretically followable if the pipeline looked for cross-references
- **implicit pointer via discourse structure** (Oruka's "the third," or the Cicero case's causal claim spread across a section) -- harder, since there's no explicit link to detect, just a presupposition the model should notice and flag rather than paper over

Both are downstream of the same root cause named above, but the fix is not the same fix. The first is closer to citation-following. The second is closer to teaching the model (or a self-critique step) to recognize when it's filling a gap rather than reporting what's actually there.

