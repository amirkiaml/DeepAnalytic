"""
chunker.py

Token-aware text splitting, shared by ingest.py.
"""

import re
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

_encoding = tiktoken.get_encoding("cl100k_base")  # matches gpt-3.5/gpt-4 family tokenization


def tiktoken_len(text: str) -> int:
    return len(_encoding.encode(text, disallowed_special=()))


class TextSplitter:
    def __init__(self, chunk_size=400, chunk_overlap=20, separators=None):
        separators = separators or ["\n\n", "\n", " ", ""]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=tiktoken_len,
            separators=separators,
        )

    def split_text(self, text: str) -> list:
        return self.text_splitter.split_text(text)
