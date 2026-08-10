"""
ingest.py

Full indexing pipeline: load SEP corpus -> split into sections (TOC-aware,
regex fallback) -> chunk each section -> embed -> upsert to Pinecone.

Run with: python ingest.py

TEST_MODE limits the run to a small slice of the corpus so you can verify
everything works — parsing, chunking, embedding, upserting — before
spending the time/cost to index the full dataset. Flip it off for the
real run.
"""

import pandas as pd
from tqdm import tqdm
from uuid import uuid4

from config import settings
from section_parser import split_into_sections
from chunker import TextSplitter
from embeddings import get_embedder, get_embedding_dimension
from vectorstore import VectorDB

# --- Tune these ---
TEST_MODE = True     # <-- set to False for the full corpus run
TEST_ROWS = 100
BATCH_SIZE = 100      # chunks embedded + upserted per API call
CHUNK_SIZE = 400
CHUNK_OVERLAP = 20


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


def embed_and_upsert(embed, index, texts: list, metadatas: list):
    ids = [str(uuid4()) for _ in texts]
    vectors = embed.embed_documents(texts)
    index.upsert(vectors=zip(ids, vectors, metadatas))


def main():
    print("Loading SEP corpus...")
    df = pd.read_parquet("data/SEP.parquet")  # relative path — run this script from the project root

    if TEST_MODE:
        df = df[:TEST_ROWS]
        print(f"TEST_MODE is ON — indexing only {TEST_ROWS} rows. "
              f"Set TEST_MODE = False in ingest.py for the full run.")

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

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing articles"):
        sections, parsed_how = split_into_sections(row["TOC"], row["Text"])
        parsed_how_counts[parsed_how] += 1

        for section_title, section_text in sections:
            chunks = splitter.split_text(section_text)
            for i, chunk_text in enumerate(chunks):
                texts_batch.append(chunk_text)
                metadatas_batch.append(build_metadata(row, section_title, i, chunk_text))
                total_chunks += 1

        if len(texts_batch) >= BATCH_SIZE:
            embed_and_upsert(embed, index, texts_batch, metadatas_batch)
            texts_batch, metadatas_batch = [], []

    if texts_batch:
        embed_and_upsert(embed, index, texts_batch, metadatas_batch)

    print(f"\nDone. Indexed {total_chunks} chunks from {len(df)} articles.")
    print("Section-parsing method breakdown:")
    for method, count in parsed_how_counts.items():
        pct = 100 * count / len(df) if len(df) else 0
        print(f"  {method}: {count} ({pct:.1f}%)")

    if parsed_how_counts["failed"] > 0 or parsed_how_counts["regex_fallback"] > len(df) * 0.3:
        print(
            "\nNOTE: A significant share of articles didn't parse cleanly via TOC. "
            "Worth spot-checking a few with check_chunks.py before trusting retrieval "
            "quality on this index — the heading regex in section_parser.py may need tuning."
        )


if __name__ == "__main__":
    main()
