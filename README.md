# DeepAnalytic

A retrieval-augmented QA system over the Stanford Encyclopedia of Philosophy,
built on a custom-scraped philosophy corpus. Started as a research project on
applying LLMs to analytic philosophy texts; the current focus is a
section-aware RAG pipeline — chunking articles along their actual heading
structure rather than blind token windows — with reranking and a working
query interface.

## What's here

- **`ingest.py`** — indexing pipeline: loads the SEP corpus, splits each
  article into sections using its Table of Contents (with a regex fallback
  for articles where TOC parsing doesn't fully resolve), chunks within
  section boundaries, embeds with OpenAI, and upserts to Pinecone.
- **`rag_pipeline.py`** — query-time pipeline: retrieve → rerank (Cohere) →
  generate (OpenAI). Includes both a reranked path and a naive-retrieval-only
  path for baseline comparison.
- **`section_parser.py`** — the TOC-aware chunking logic. Falls back cleanly
  from exact TOC matching to regex heading detection to whole-article
  indexing, and records which method was used per article for auditing.
- **`chunker.py`**, **`embeddings.py`**, **`vectorstore.py`** — supporting
  modules for token-based text splitting, embedding model config, and
  Pinecone index management.
- **`chat.py`** — terminal chat interface for querying the indexed corpus.
- **`check_chunks.py`** — diagnostic tool for inspecting which chunks get
  retrieved for a given query, useful for debugging retrieval quality.
- **`run_eval.py`** — automation script that runs a systematic question
  set through both naive and reranked retrieval modes and logs the
  results for scoring; see `tests/` and Evaluation below.
- **`config.py`** — centralized, environment-driven settings (API keys,
  model names, retrieval parameters). No secrets in code.
- **`tests/`** — the evaluation question set, raw results, and per-question
  scoring (`eval_systematic.csv`, `eval_systematic_results.csv`,
  `eval_scored.csv`).
- **`notebooks/`** — the original research notebooks this project grew out
  of: corpus scraping, naive RAG, reranking, multiquery retrieval, and
  early parameter-testing experiments. Kept as a record of how the
  production pipeline evolved.

## Running it

All commands below assume you're in the project root with the
`deepanalytic` conda environment activated (`conda activate deepanalytic`)
and a `.env` file present with the required API keys (see `config.py`
for the full list).

```
python ingest.py
```
Builds the Pinecone index: loads the SEP corpus, section-parses each
article, chunks it, embeds the chunks, and upserts them. `TEST_MODE` and
`TEST_ROWS` at the top of the file control whether this runs against the
full corpus or a small slice — leave `TEST_MODE = True` for a cheap,
fast sanity check before committing to a full (paid) indexing run.

```
python chat.py
```
Starts an interactive terminal chat against the indexed corpus. Type a
question, then choose `naive` or `rerank` (default) mode when prompted.
Prints the answer plus the source articles/sections it was drawn from.
Type `exit` or `quit` to stop.

```
python check_chunks.py
```
Diagnostic tool: shows the raw chunks retrieved for a single hardcoded
query/article pair (edit `TARGET_TITLE` and `QUERY` at the top of the
file) before reranking happens. Useful for confirming retrieval is
actually finding the right passage when an answer looks off.

```
python rag_pipeline.py
```
Runs a single hardcoded test query directly against `RerankRAG`, useful
as a quick smoke test after changing the pipeline — confirms indexing,
retrieval, reranking, and generation are all wired correctly without
needing the interactive chat loop.

```
python run_eval.py
```
Runs the full systematic evaluation: every question in
`tests/eval_systematic.csv`, in both naive and rerank mode, logging real
answers and sources to `tests/eval_systematic_results.csv`. Skips rows
that already have an answer, so a failed run can be safely re-launched
without re-paying for completed rows. This makes real API calls — don't
run it casually.

## Environment setup

```
conda create -n deepanalytic python=3.11 -y
conda activate deepanalytic
pip install -r requirements.txt
```
Then create a `.env` file in the project root with `OPENAI_API_KEY`,
`PINECONE_API_KEY`, `COHERE_API_KEY`, and `PINECONE_INDEX_NAME` — see
`config.py` for the complete list of settings and their defaults.

## Data

### Stanford Encyclopedia of Philosophy (SEP)
The entirety of the SEP, 2024 edition, has been scraped — full article
text, tables of contents, bibliographies, and metadata (authors, dates,
citations). This is the corpus the current RAG pipeline runs on.

### Springer
Metadata for 28 Springer Nature philosophy journals (~800k pages) has been
scraped. Scraping code is included; full-text indexing of this corpus is a
future direction.

### Other
Text and metadata for ~10k pages from *Analysis* (Oxford University Press)
have also been scraped. Code will be shared and the dataset released after
preprocessing, pending licensing review.

## Why section-aware chunking

Standard RAG chunks articles into fixed-size token windows regardless of
content structure, which means a single chunk can straddle two unrelated
subsections — hurting both retrieval precision and the coherence of what
gets handed to the LLM as context. This pipeline instead uses each SEP
article's own Table of Contents to locate real section boundaries in the
body text, then chunks within those boundaries. Each chunk's section title
becomes part of its metadata, giving the reranker and the LLM real
structural context (e.g. "Avicenna and the Aristotelian Tradition" rather
than an anonymous span of text).

On an initial test run (100 articles), TOC-based section parsing succeeded
fully on 95% of articles and partially on the remaining 5%, with no
articles falling back to plain regex heading detection or failing outright.

## Evaluation

A systematic naive-vs-reranked retrieval comparison was run against the
100-article test index: 10 articles chosen for topical and structural
variety, 2 questions each (one narrow/section-specific, one
broad/synthesizing), 40 total runs, each scored on Accuracy, Source
Match, and Completeness against a key written before any answers were
seen.

| | Accuracy | Source Match | Completeness | Overall |
|---|---|---|---|---|
| Naive retrieval | 4.84 / 5 | 3.63 / 5 | 4.11 / 5 | 4.19 |
| With reranking | 4.79 / 5 | 3.68 / 5 | 3.68 / 5 | 4.05 |

The headline finding wasn't "reranking wins" — on this sample, reranking
was roughly flat on Accuracy and only marginally better on Source Match,
while actually scoring *lower* on Completeness. Two runs surfaced a
concrete failure mode: on broad, synthesizing questions, the reranker
sometimes deprioritized the one chunk containing the answer even though
naive retrieval's wider candidate pool caught it, causing the pipeline
to falsely report the context as insufficient.

Full test set, answers, sources returned, and per-question scoring with
notes are in `tests/eval_scored.csv`. Write-up: [Building a RAG System
That Knows What Section It's In (Part 1)](#) — link once published.

**AI-assist disclosure for this evaluation:** Claude drafted the article
selection, the questions, the expected-key-points rubric, and the
automation script (`run_eval.py`) that ran all 40 tests, and did the
first scoring pass. A second, independent human scoring pass is
recorded in the same file's `Your_*` columns.

## Roadmap

- Full-corpus indexing (currently tested on a 100-article slice)
- FastAPI service wrapping `rag_pipeline.py`
- Investigate the reranker's Completeness gap found in evaluation above,
  before treating reranking as a strict improvement over naive retrieval
- Deployment (Azure)
- Open-source model option alongside the OpenAI-backed pipeline
