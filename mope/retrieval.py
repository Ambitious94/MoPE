"""Lightweight retrieval and reading helpers for pipeline experts.

The goal is to give the MoPE scaffold a concrete, dependency-free retrieval
stack that can ingest documents, score them against a query, and produce a
human-readable summary for downstream vectorization.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .pipeline import PipelineExpert


def _tokenize(text: str) -> List[str]:
    return [tok for tok in text.lower().replace("\n", " ").split() if tok]


@dataclass
class Document:
    doc_id: str
    text: str


class DocumentStore:
    """In-memory document store with simple BM25-style scoring."""

    def __init__(self) -> None:
        self._documents: Dict[str, Document] = {}
        self._doc_freqs: Counter[str] = Counter()
        self._tokenized: Dict[str, List[str]] = {}

    def add(self, doc_id: str, text: str) -> None:
        tokens = _tokenize(text)
        self._documents[doc_id] = Document(doc_id=doc_id, text=text)
        self._tokenized[doc_id] = tokens
        self._doc_freqs.update(set(tokens))

    def add_many(self, docs: Iterable[Tuple[str, str]]) -> None:
        for doc_id, text in docs:
            self.add(doc_id, text)

    def _idf(self, term: str) -> float:
        doc_freq = self._doc_freqs.get(term, 0) + 1
        return math.log((1 + len(self._documents)) / doc_freq)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        query_tokens = _tokenize(query)
        scores: Dict[str, float] = defaultdict(float)
        for doc_id, tokens in self._tokenized.items():
            token_counts = Counter(tokens)
            doc_len = len(tokens) or 1
            for term in query_tokens:
                tf = token_counts.get(term, 0) / doc_len
                if tf == 0:
                    continue
                scores[doc_id] += tf * self._idf(term)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        results = []
        for doc_id, score in ranked[:top_k]:
            snippet = self._build_snippet(doc_id, query_tokens)
            results.append({"doc_id": doc_id, "score": score, "snippet": snippet})
        return results

    def _build_snippet(self, doc_id: str, query_tokens: List[str]) -> str:
        text = self._documents[doc_id].text
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        for sentence in sentences:
            lower = sentence.lower()
            if any(term in lower for term in query_tokens):
                return sentence
        return sentences[0] if sentences else text


class EvidenceReader:
    """Aggregates retrieved snippets into a concise textual answer."""

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def read(self, query: str, hits: List[Dict[str, object]]) -> str:
        if not hits:
            return f"no supporting evidence found for '{query}'"
        parts = []
        for hit in hits:
            parts.append(f"{hit['doc_id']}: {hit['snippet']}")
        return " | ".join(parts)


def make_retrieval_search(store: DocumentStore, top_k: int = 3):
    def _search(prompt: str) -> str:
        hits = store.search(prompt, top_k=top_k)
        formatted = "; ".join(f"{h['doc_id']} (score={h['score']:.2f})" for h in hits)
        return f"retrieval hits for '{prompt}': {formatted}" if formatted else f"no hits for '{prompt}'"

    return _search


def make_reader(store: DocumentStore, top_k: int = 3):
    reader = EvidenceReader(store)

    def _read(prompt: str) -> str:
        hits = store.search(prompt, top_k=top_k)
        summary = reader.read(prompt, hits)
        return f"read evidence for '{prompt}': {summary}"

    return _read


def build_retrieval_pipelines(store: DocumentStore) -> Dict[str, PipelineExpert]:
    """Constructs retrieval-enhanced pipeline experts for the MoPE layer."""

    search_step = make_retrieval_search(store)
    read_step = make_reader(store)

    return {
        "retrieval-search-reader": PipelineExpert(
            name="retrieval-search-reader",
            steps=[search_step, read_step],
            description="Single-pass retrieval and reading.",
        ),
        "planner-retrieval-reader": PipelineExpert(
            name="planner-retrieval-reader",
            steps=[
                lambda prompt: f"plan retrieval for '{prompt}'",
                search_step,
                read_step,
            ],
            description="Plan then retrieve and read evidence.",
        ),
    }

