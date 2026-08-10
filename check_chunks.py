"""
check_chunks.py

Diagnostic script: shows which chunks of a given article got retrieved
for a specific query, so you can see whether the passage you expect
is actually in the candidate pool before reranking/generation happens.

Run with: python check_chunks.py
"""

from rag_pipeline import RerankRAG

TARGET_TITLE = "Arabic and Islamic Philosophy of Language and Logic"
QUERY = "How does action provide perceptual information about the environment?"
K = 10  # how many raw chunks to pull before filtering by title


def main():
    rag = RerankRAG()

    raw_docs = rag.vectorstore.similarity_search(QUERY, k=K)

    found_any = False
    for d in raw_docs:
        if d.metadata.get("title") == TARGET_TITLE:
            found_any = True
            print("CHUNK", d.metadata.get("chunk"), ":")
            print(d.page_content[:200])
            print("---")

    if not found_any:
        print(f"No chunks from '{TARGET_TITLE}' were retrieved in the top {K}.")


if __name__ == "__main__":
    main()
