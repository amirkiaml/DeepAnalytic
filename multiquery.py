"""
multiquery.py

Generates reformulated versions of a question to widen retrieval before
similarity search, for questions where a single embedding struggles to
find both halves of a composite question (see chunking_conclusions.md
and blog Part 3 for the evidence this was built to address).

v2 (structured output): the first version generated free text and split
on newlines, letting "how many lines were written" implicitly decide
the subquery count. That turned out not to be robust -- on a simple,
single-topic control question ("What is the thinking animal argument?"),
the model split it into 5 subqueries by FACET (premises, proposer,
context, criticisms) instead of recognizing it as one topic, and the
resulting answer was measurably worse than a single query would have
been. A more explicit prompt with good/bad examples didn't fully fix
this -- the underlying issue wasn't wording, it was that "how many
subqueries" was an implicit side effect of free-text generation length,
never an explicit, auditable decision.

This version forces that decision to be explicit: the model must first
list the distinct TOPICS the question is actually asking about, then
generate exactly one subquery per topic, enforced by a schema
(topics and subqueries are the same length by construction). A
single-topic question can only ever produce one subquery this way,
because there's no separate "how many lines do I feel like writing"
degree of freedom left to drift on. The `topics` field is also a real
debugging signal -- log it to see directly when/if the model starts
treating facets of one topic as if they were separate topics.

Built with LCEL (prompt | llm | structured output), not the deprecated
LLMChain pattern from the original 2023 notebook. Also deliberately
does NOT use LangChain's built-in MultiQueryRetriever -- its generic
document dedup has no awareness of this project's max_per_title cap or
section metadata, so plugging it in directly would regress the
source-noise fixes already validated in rag_pipeline.py.
generate_subqueries() here only generates the reformulated questions;
retrieval, dedup, and rerank stay in rag_pipeline.py where the existing
logic already handles them.
"""

from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings


class QueryDecomposition(BaseModel):
    """Structured decomposition of a question into atomic, independently-searchable sub-questions."""

    topics: List[str] = Field(
        description=(
            "The distinct topics or claims this question is actually asking about, "
            "named briefly. A single-topic question (e.g. 'what is X', 'explain Y') "
            "has EXACTLY ONE entry here, even if X has multiple facets (premises, "
            "proposers, criticisms, context) -- those are all part of the same one "
            "topic, not separate topics. Only list more than one topic if the "
            "question genuinely requires separate information about multiple "
            "distinct things, such as a comparison between two named things, a "
            "multi-part list of unrelated claims, or a question spanning unrelated "
            "subjects."
        )
    )
    subqueries: List[str] = Field(
        description=(
            "One standalone, atomic retrieval question per topic listed above, in "
            "the same order. Must be the same length as topics."
        )
    )


DECOMPOSITION_SYSTEM_PROMPT = """You are an AI assistant helping to improve document retrieval against a vector database via similarity search.

First identify the distinct topics this question is actually asking about. Then generate exactly one standalone retrieval question per topic.

Critical -- do not split ONE topic into multiple facets. "What is the thinking animal argument?" is ONE topic (the argument itself). It should produce topics=["the thinking animal argument"] and exactly one subquery, NOT separate subqueries for its premises, its proposer, its context, and its criticisms. Those are all facets of a single topic, answerable by one good retrieval -- splitting them apart adds nothing and can actively dilute a good answer.

By contrast, "How does Cicero's res publica compare to Socrates' relationship to the Laws of Athens?" has TWO distinct topics -- Cicero's res publica, and Socrates' relationship to the Laws of Athens -- so it should produce two subqueries, one per topic, each answerable using information about only that one topic, not both at once.

Never produce more than {max_n} topics/subqueries, regardless of how complex the question seems. When in doubt about whether something is a separate topic or just a facet of one topic, default to treating it as a facet -- under-splitting costs less than over-splitting."""


def get_subquery_chain():
    # Deliberately DECOMPOSITION_MODEL, not LLM_MODEL — splitting a
    # question apart is a much simpler task than answering one grounded
    # in retrieved context, so it doesn't need the same (more expensive)
    # model that does final generation. See config.py for the reasoning.
    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.DECOMPOSITION_MODEL,
        temperature=settings.TEMPERATURE,
    ).with_structured_output(QueryDecomposition)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DECOMPOSITION_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    return prompt | llm


def generate_subqueries(question: str, max_n: int = 5, return_topics: bool = False):
    """
    Returns a list of DECOMPOSED sub-questions (atomic, single-topic
    each), plus the original question itself as a safety net.

    The number of sub-questions is now determined by an explicit,
    schema-enforced topic count, not by however many lines a free-text
    generation happened to produce -- see module docstring for why the
    earlier free-text version wasn't robust enough.

    Args:
        question: the question to decompose
        max_n: cost-control ceiling on topics/subqueries, not a target
        return_topics: if True, returns (subqueries, topics) instead of
                        just subqueries -- useful for logging/debugging
                        to directly see what the model judged as the
                        distinct topics, e.g. to catch over-decomposition
                        (facets misidentified as separate topics)

    Returns:
        list of subquery strings, or (list, list) if return_topics=True
    """
    chain = get_subquery_chain()
    result = chain.invoke({"question": question, "max_n": max_n})

    topics = result.topics[:max_n]
    subqueries = result.subqueries[: len(topics)]

    if question not in subqueries:
        subqueries.append(question)

    if return_topics:
        return subqueries, topics
    return subqueries


if __name__ == "__main__":
    # Quick manual check -- run `python multiquery.py`
    # Prints the identified topics alongside subqueries, so you can see
    # directly whether the model is calibrating correctly.
    test_questions = [
        "How does Cicero's res publica compare to Socrates' relationship to the Laws of Athens at his trial?",
        "What is the thinking animal argument?",
    ]
    for q in test_questions:
        subqueries, topics = generate_subqueries(q, return_topics=True)
        print(f"\nQUESTION: {q}")
        print(f"TOPICS IDENTIFIED ({len(topics)}): {topics}")
        print("SUBQUERIES:")
        for sq in subqueries:
            print(" -", sq)
