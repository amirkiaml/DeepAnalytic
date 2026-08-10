"""
chat.py

Simple terminal chatbot for the SEP RAG pipeline.
Each question is answered independently — no conversational memory yet.
Run with: python chat.py
Type 'exit' or 'quit' to stop.
"""

from rag_pipeline import RerankRAG


def main():
    print("Loading RAG pipeline (Pinecone + OpenAI + Cohere)...")
    rag = RerankRAG()
    print("Ready. Ask a question about the SEP articles (type 'exit' to quit).\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        if not question:
            continue

        mode = input("Mode - naive/rerank (Enter for rerank): ").strip().lower()

        if mode in ("naive", "n"):
            result = rag.query_naive(question)
        else:
            result = rag.query(question)

        print(f"\nBot: {result['answer']}\n")

        if result["sources"]:
            print("Sources:")
            seen = set()
            for s in result["sources"]:
                title = s.get("title", "unknown")
                if title in seen:
                    continue
                seen.add(title)
                print(f"  - {title} ({s.get('source', '')})")
        print()  # blank line before next prompt


if __name__ == "__main__":
    main()
