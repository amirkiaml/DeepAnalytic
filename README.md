# DeepAnalytic

A retrieval-augmented QA system over the Stanford Encyclopedia of Philosophy, built on a custom-scraped philosophy corpus. Started as a research project on applying LLMs to analytic philosophy texts; current focus is a section-aware RAG pipeline with reranking and a working query interface.

## What's here

- **`ingest.py`** -- indexing pipeline: loads the SEP corpus, splits each article into sections using its Table of Contents, chunks within section boundaries, embeds with OpenAI, upserts to Pinecone.
- **`rag_pipeline.py`** -- query-time pipeline: retrieve -> rerank (Cohere) -> generate (OpenAI). Includes a naive-retrieval-only path for baseline comparison.
- **`multiquery.py`** -- topic-first, schema-enforced question decomposition for composite questions, used by `rag_pipeline.py`'s `query_multi()` and `query_multi_concat()`.
- **`section_parser.py`** -- TOC-aware chunking logic, with a regex fallback and per-article parsing-method logging.
- **`chunker.py`**, **`embeddings.py`**, **`vectorstore.py`** -- supporting modules for text splitting, embedding config, and Pinecone index management.
- **`chat.py`** -- terminal chat interface for querying the indexed corpus.
- **`check_chunks.py`** -- diagnostic tool for inspecting retrieved chunks for a given query.
- **`run_eval.py`** -- automation script that runs the systematic eval set through naive and reranked modes and logs results; see `tests/` and Evaluation below.
- **`config.py`** -- centralized, environment-driven settings. No secrets in code.
- **`tests/`** -- eval question set, raw results, scoring, and rubric (`eval_systematic.csv`, `eval_systematic_results.csv`, `eval_scored.xlsx`, `scoring_rubric.csv`).
- **`notebooks/`** -- original research notebooks this project grew out of, plus `9_Chunking_Strategy_Comparison.ipynb`, `chunking_conclusions.md`, `5_Multiquery_Production.ipynb`, and `multiquery_dev_log.md` -- later comparisons and findings notes.

## Running it

All commands assume the `deepanalytic` conda environment is active and a `.env` file is present (see `config.py` for required keys).

```
python ingest.py          # build the Pinecone index (TEST_MODE/TEST_ROWS control full-corpus vs. slice)
python chat.py             # interactive terminal chat, naive or rerank mode
python check_chunks.py     # show raw retrieved chunks for a hardcoded query, before reranking
python rag_pipeline.py     # single hardcoded smoke-test query against RerankRAG
python run_eval.py         # run the eval set through both modes, logs to tests/eval_systematic_results.csv (real API calls)
```

## Environment setup

```
conda create -n deepanalytic python=3.11 -y
conda activate deepanalytic
pip install -r requirements.txt
```
Create a `.env` file with `OPENAI_API_KEY`, `PINECONE_API_KEY`, `COHERE_API_KEY`, `PINECONE_INDEX_NAME` -- see `config.py` for the full list.

## Data

**SEP** -- full text, TOCs, bibliographies, and metadata for the entire Stanford Encyclopedia of Philosophy (2024 edition). The corpus this pipeline runs on.

**Springer** -- metadata for 28 Springer Nature philosophy journals (~800k pages). Scraping code included; full-text indexing is a future direction.

**Other** -- ~10k pages from *Analysis* (Oxford University Press), scraped, pending licensing review before release.

## Why section-aware chunking

Standard RAG chunks by fixed token windows regardless of content structure, which can straddle two unrelated subsections in one chunk. This pipeline instead uses each SEP article's own Table of Contents to chunk within real section boundaries, attaching the section title as metadata.

On a 100-article test, TOC-based parsing succeeded fully on 95% of articles, partially on the rest, with no outright failures.

**Update:** a later, dedicated comparison against four alternative chunking strategies found section-aware chunking does *not* clearly outperform the two simplest alternatives on this corpus -- natural paragraph breaks already track topic breaks closely enough that a structure-blind splitter gets a similar result. It remains the default, but the honest finding is that chunking strategy matters less than expected, and the real bottleneck sits downstream (reranking's narrowing step, and generation's willingness to synthesize). Full comparison and findings in `notebooks/chunking_conclusions.md`.

## Evaluation

A systematic naive-vs-reranked comparison, 5 articles chosen for known interesting cases (two confirmed rerank failures, one accuracy-drift case, one cross-article retrieval case, one clean baseline), 2 questions each, scored on Accuracy, Source Match, and Completeness.

| | Accuracy | Source Match | Completeness | Overall |
|---|---|---|---|---|
| Naive retrieval | 4.78 / 5 | 3.56 / 5 | 4.00 / 5 | 4.11 |
| With reranking | 4.67 / 5 | 3.56 / 5 | 3.00 / 5 | 3.74 |

Reranking was flat-to-worse across the board, with two fully reproducible failures backed by direct evidence in the retrieved chunk text:

- **Retrieval-narrowing failure** (organic vs. somatic animalism): reranking discarded the two chunks containing the actual definitions, kept an unrelated one instead.
- **Generation-layer failure** (Cicero's *res publica*): reranking kept the relevant chunk, and the model still declined to answer -- the needed causal claim required synthesis across a section rather than restating one sentence.

Both point to the same underlying pattern: a retrieved chunk can mention or presuppose content it doesn't itself contain, with no mechanism in the pipeline to notice and follow that gap. Full writeup in `notebooks/chunking_conclusions.md`.

Full test set, answers, retrieved chunk text, and per-question scoring with notes are in `tests/eval_scored.xlsx`.

**Write-ups**
- [Building a RAG System That Knows What Section It's In (Part 1)](https://vmachines.substack.com/p/building-a-rag-system-that-knows)
- Five Chunking Strategies, No Winner, One Real Finding (Part 2) -- coming soon
- Chunks That Mention vs. Chunks That Explain (Part 3) -- coming soon
- From Multiquery to a Referee (Part 4) -- coming soon

**Scoring methodology:** this follows standard current practice for RAG evaluation: LLM-assisted test-set generation, paired with LLM-as-judge scoring and human verification. Claude generated the article selection, the questions, and the expected-key-points rubric (a common synthetic-eval-set-generation pattern, e.g. RAGAS's testset generator does the same thing), and built the automation script and first-pass scoring. I did an independent second scoring pass with my own comments, spot-checking the highest-stakes rows directly against the retrieved chunk text. Both sets of scores and notes are recorded side by side in `tests/eval_scored.xlsx`, including at least one case where my score corrected Claude's first pass.

## Roadmap

- Full-corpus indexing (currently tested on a 100-article slice)
- FastAPI service wrapping `rag_pipeline.py`
- Generation-layer work first: multiquery decomposition, a self-critique step, and a prompt-tweak test before further chunking/reranking tuning
- Cross-reference/multi-hop retrieval: follow pointers inside a retrieved chunk to the section that actually explains a concept
- Deployment (AWS)
- Open-source model option alongside the OpenAI-backed pipeline