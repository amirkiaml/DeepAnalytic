"""
run_eval.py

Reads eval_systematic.csv, runs every (Question, Mode) row through the
live RAG pipeline, and fills in Answer + Sources_Returned automatically.

Scoring columns (Claude_*, Your_*, Notes) are left untouched — those
still require human/LLM judgment and are filled in afterward.

Run with: python run_eval.py
Output: eval_systematic_results.csv (does not overwrite the original,
so you can re-run safely without losing manual scores you've already
entered — see the merge note at the bottom of this file).
"""

import csv
import time

from rag_pipeline import RerankRAG

INPUT_FILE = "tests/eval_systematic.csv"
OUTPUT_FILE = "tests/eval_systematic_results.csv"


def format_sources(sources: list) -> str:
    """Turn the list of metadata dicts into a compact 'Title | Section' string."""
    seen = set()
    parts = []
    for s in sources:
        title = s.get("title", "unknown")
        section = s.get("section", "")
        key = (title, section)
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{title} | {section}")
    return "; ".join(parts)


def format_chunks(sources: list, max_chars: int = 400) -> str:
    """
    Log the actual chunk text that was retrieved/reranked and handed to
    the LLM as context — not just which article/section it came from.
    This is what makes an eval genuinely auditable: you can see exactly
    what the model saw, not just infer it from the article title.

    Each chunk is truncated to max_chars to keep the CSV readable; the
    full untruncated text is still in Pinecone if you need to go deeper
    on a specific row.
    """
    blocks = []
    for i, s in enumerate(sources, 1):
        title = s.get("title", "unknown")
        section = s.get("section", "")
        text = s.get("text", "")
        snippet = text[:max_chars] + ("..." if len(text) > max_chars else "")
        blocks.append(f"[{i}] {title} | {section}\n{snippet}")
    return "\n---\n".join(blocks)


def main():
    print("Loading RAG pipeline...")
    rag = RerankRAG()

    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    total = len(rows)
    for i, row in enumerate(rows, 1):
        question = row["Question"]
        mode = row["Mode"].strip().lower()

        # Skip rows that already have an answer, so a partial re-run
        # doesn't waste API calls re-doing work.
        if row.get("Answer", "").strip():
            print(f"[{i}/{total}] Skipping (already answered): {question[:60]}...")
            continue

        print(f"[{i}/{total}] ({mode}) {question[:70]}...")

        try:
            if mode == "naive":
                result = rag.query_naive(question)
            else:
                result = rag.query(question)

            row["Answer"] = result["answer"]
            row["Sources_Returned"] = format_sources(result["sources"])
            row["Chunks_Consulted"] = format_chunks(result["sources"])

        except Exception as e:
            row["Answer"] = f"ERROR: {e}"
            row["Sources_Returned"] = ""
            row["Chunks_Consulted"] = ""
            print(f"    ERROR on this row: {e}")

        # Small delay to stay comfortably under rate limits across
        # OpenAI/Pinecone/Cohere calls.
        time.sleep(1)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results written to {OUTPUT_FILE}")
    print("Scoring columns (Claude_*, Your_*, Notes) are still blank — "
          "fill those in by hand or paste this file back to Claude for scoring.")


if __name__ == "__main__":
    main()
