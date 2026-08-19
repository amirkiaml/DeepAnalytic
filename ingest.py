"""
ingest.py

Full indexing pipeline: load SEP corpus -> split into sections (TOC-aware,
regex fallback) -> chunk each section -> embed -> upsert to Pinecone.

Run with: python ingest.py

TEST_MODE limits the run to a small slice of the corpus so you can verify
everything works — parsing, chunking, embedding, upserting — before
spending the time/cost to index the full dataset. Flip it off for the
real run.

NAMESPACE controls which partition of the index gets written to. Pinecone
namespaces are hard partitions: a query against one namespace cannot see
vectors in another, at all. They're free and carry no performance
overhead, unlike additional indexes which each consume separate storage
against the free tier's limits. This is what lets the 100-article test
data and the full corpus coexist in one index, so a before/after
retrieval comparison is possible without deleting anything.
"""

import time

import pandas as pd
from tqdm import tqdm
from uuid import uuid4

from config import settings
from section_parser import split_into_sections
from chunker import TextSplitter
from embeddings import get_embedder, get_embedding_dimension
from vectorstore import VectorDB

# --- Tune these ---
TEST_MODE = False     # <-- set to False for the full corpus run
TEST_ROWS = 300
NAMESPACE = "articles-full"        # "" = default namespace; use e.g. "articles-full" for the full corpus
BATCH_SIZE = 100      # chunks embedded + upserted per API call
CHUNK_SIZE = 400
CHUNK_OVERLAP = 20

# Retry settings — matter much more at full-corpus scale, where a single
# transient rate-limit or network blip partway through a long run would
# otherwise lose everything done so far.
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2  # seconds; doubles each retry


def build_metadata(row, section_title: str, chunk_idx: int, chunk_text: str) -> dict:
    return {
        "article_id": str(row["ID"]),
        "source": row["Url"],
        "title": row["Title"],
        "section": section_title,
        "authors": str(row["Authors"]),
        "citation": row["BibURL"],
        "date": str(row["Date"]),
        "chunk": chunk_idx,
        "text": chunk_text,  # Pinecone needs the raw text stored as metadata to return it on retrieval
    }


def embed_and_upsert(embed, index, texts: list, metadatas: list, namespace: str):
    """
    Embed a batch and upsert it, retrying with exponential backoff on
    failure. A full-corpus run makes ~1000 of these calls; without
    retries, one transient error partway through wastes the whole run.
    """
    ids = [str(uuid4()) for _ in texts]

    for attempt in range(MAX_RETRIES):
        try:
            vectors = embed.embed_documents(texts)
            if namespace:
                index.upsert(vectors=list(zip(ids, vectors, metadatas)), namespace=namespace)
            else:
                index.upsert(vectors=list(zip(ids, vectors, metadatas)))
            return True
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"\n  FAILED after {MAX_RETRIES} attempts: {e}")
                return False
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"\n  Retry {attempt + 1}/{MAX_RETRIES - 1} in {delay}s ({type(e).__name__})")
            time.sleep(delay)

    return False


def main():
    print("Loading SEP corpus...")
    df = pd.read_parquet("data/SEP.parquet")  # relative path — run this script from the project root

    if TEST_MODE:
        df = df[:TEST_ROWS]
        print(f"TEST_MODE is ON — indexing only {TEST_ROWS} rows. "
              f"Set TEST_MODE = False in ingest.py for the full run.")
    else:
        print(f"TEST_MODE is OFF — indexing the full corpus ({len(df)} articles).")

    print(f"Target index: {settings.PINECONE_INDEX_NAME}")
    print(f"Target namespace: {NAMESPACE or '(default)'}")

    df["Title"] = df["Title"].astype(str)
    df["Text"] = df["Text"].astype(str)
    df["TOC"] = df["TOC"].astype(str)

    splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embed = get_embedder()

    vector_db = VectorDB()
    vector_db.create_index(
        settings.PINECONE_INDEX_NAME,
        dimension=get_embedding_dimension(),
        metric="cosine",
    )
    index = vector_db.connect_to_index(settings.PINECONE_INDEX_NAME)

    texts_batch, metadatas_batch = [], []
    parsed_how_counts = {"toc_match": 0, "toc_partial": 0, "regex_fallback": 0, "failed": 0}
    total_chunks = 0
    failed_batches = 0
    skipped_articles = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing articles"):
        # Skip malformed articles rather than crashing a long run partway
        # through. At 100 articles a bad row is obvious; at ~1800 it's a
        # real risk worth handling.
        try:
            sections, parsed_how = split_into_sections(row["TOC"], row["Text"])
        except Exception as e:
            skipped_articles.append((row.get("Title", "unknown"), str(e)))
            continue

        parsed_how_counts[parsed_how] += 1

        for section_title, section_text in sections:
            chunks = splitter.split_text(section_text)
            for i, chunk_text in enumerate(chunks):
                texts_batch.append(chunk_text)
                metadatas_batch.append(build_metadata(row, section_title, i, chunk_text))
                total_chunks += 1

        if len(texts_batch) >= BATCH_SIZE:
            ok = embed_and_upsert(embed, index, texts_batch, metadatas_batch, NAMESPACE)
            if not ok:
                failed_batches += 1
            texts_batch, metadatas_batch = [], []

    if texts_batch:
        ok = embed_and_upsert(embed, index, texts_batch, metadatas_batch, NAMESPACE)
        if not ok:
            failed_batches += 1

    parsed_total = sum(parsed_how_counts.values())

    print(f"\nDone. Indexed {total_chunks} chunks from {parsed_total} articles "
          f"into namespace '{NAMESPACE or '(default)'}'.")
    print("Section-parsing method breakdown:")
    for method, count in parsed_how_counts.items():
        pct = 100 * count / parsed_total if parsed_total else 0
        print(f"  {method}: {count} ({pct:.1f}%)")

    if failed_batches:
        print(f"\nWARNING: {failed_batches} batch(es) failed permanently after retries. "
              f"Roughly {failed_batches * BATCH_SIZE} chunks may be missing from the index.")

    if skipped_articles:
        print(f"\nWARNING: {len(skipped_articles)} article(s) skipped due to parsing errors:")
        for title, err in skipped_articles[:10]:
            print(f"  - {title}: {err}")
        if len(skipped_articles) > 10:
            print(f"  ... and {len(skipped_articles) - 10} more")

    if parsed_how_counts["failed"] > parsed_total * 0.02 or parsed_how_counts["regex_fallback"] > parsed_total * 0.3:
        print(
            "\nNOTE: A significant share of articles didn't parse cleanly via TOC. "
            "Worth spot-checking a few with check_chunks.py before trusting retrieval "
            "quality on this index — the heading regex in section_parser.py may need tuning."
        )


if __name__ == "__main__":
    main()
