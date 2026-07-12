"""Local BM25 retrieval over EvidenceChunk objects.

Optimized version:
1. Uses an inverted index to avoid full-corpus scans per query.
2. Uses source_type partitions for source-filtered retrieval.
3. Uses exact identifier lookup for case numbers, patent numbers, and trademark-like IDs.
"""
from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from crossborder_agentic_rag.schemas.evidence import EvidenceChunk


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

# Examples:
# 3:90-cv-04129
# 5:20-cv-08009
CASE_RE = re.compile(r"\b\d{1,2}:\d{2}-[a-z]+-\d{3,6}\b", re.IGNORECASE)

# Patent numbers / serial-like identifiers.
# Keep broad, but exact-index lookup only returns if the ID exists in metadata/doc ids.
LONG_NUM_RE = re.compile(r"\b\d{7,8}\b")

DOC_ID_RE = re.compile(
    r"\b(?:patent|trademark|litigation):[A-Za-z0-9:_\-\.]+\b",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _norm_id(x: object) -> str:
    return str(x or "").strip().lower()


def _copy(c: EvidenceChunk, score: float) -> EvidenceChunk:
    return EvidenceChunk(
        c.chunk_id,
        c.doc_id,
        c.source_type,
        c.source_subtype,
        c.title,
        c.content,
        dict(c.metadata),
        score,
    )


def _matches(
    c: EvidenceChunk,
    filters: dict | None,
    source_types: list[str] | None,
) -> bool:
    if source_types and c.source_type not in source_types:
        return False

    for k, v in (filters or {}).items():
        actual = getattr(c, k, None) if hasattr(c, k) else c.metadata.get(k, "")

        if isinstance(v, str):
            if actual != v:
                return False
        elif isinstance(v, list):
            if actual not in v:
                return False
        else:
            raise ValueError(f"Unsupported filter value for {k}: {v!r}")

    return True


def _extract_exact_ids(query: str) -> list[str]:
    q = query or ""
    ids: list[str] = []

    for pat in (DOC_ID_RE, CASE_RE, LONG_NUM_RE):
        for m in pat.findall(q):
            val = _norm_id(m)
            if val and val not in ids:
                ids.append(val)

    return ids


class LocalBM25Retriever:
    def __init__(self, chunks: list[EvidenceChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b

        self.docs: list[list[str]] = []
        self.term_freqs: list[Counter] = []
        self.df: dict[str, int] = defaultdict(int)

        # term -> doc indices containing this term
        self.inverted: dict[str, list[int]] = defaultdict(list)

        # source_type -> doc indices
        self.by_source_type: dict[str, list[int]] = defaultdict(list)
        self.by_source_type_set: dict[str, set[int]] = defaultdict(set)

        # exact identifier -> doc indices
        self.exact_index: dict[str, set[int]] = defaultdict(set)

        total_len = 0

        for idx, c in enumerate(self.chunks):
            text = " ".join(
                [
                    c.title or "",
                    c.content or "",
                    c.source_type or "",
                    c.source_subtype or "",
                    " ".join(str(v) for v in (c.metadata or {}).values()),
                ]
            )

            toks = tokenize(text)
            tf = Counter(toks)

            self.docs.append(toks)
            self.term_freqs.append(tf)

            total_len += len(toks)

            for t in tf:
                self.df[t] += 1
                self.inverted[t].append(idx)

            self.by_source_type[c.source_type].append(idx)
            self.by_source_type_set[c.source_type].add(idx)

            self._add_exact_ids(idx, c)

        self.avgdl = (total_len / len(self.docs)) if self.docs else 0.0

    def _add_exact_ids(self, idx: int, c: EvidenceChunk) -> None:
        values: list[object] = [
            c.chunk_id,
            c.doc_id,
            c.title,
            c.metadata.get("case_number") if c.metadata else None,
            c.metadata.get("patent_number") if c.metadata else None,
            c.metadata.get("patent_id") if c.metadata else None,
            c.metadata.get("application_number") if c.metadata else None,
            c.metadata.get("serial_number") if c.metadata else None,
            c.metadata.get("registration_number") if c.metadata else None,
            c.metadata.get("parent_id") if c.metadata else None,
        ]

        for v in values:
            if not v:
                continue

            s = _norm_id(v)
            if not s:
                continue

            self.exact_index[s].add(idx)

            # Also index identifiers found inside the string.
            for pat in (DOC_ID_RE, CASE_RE, LONG_NUM_RE):
                for m in pat.findall(s):
                    self.exact_index[_norm_id(m)].add(idx)

    def _source_candidate_set(self, source_types: list[str] | None) -> set[int] | None:
        if not source_types:
            return None

        out: set[int] = set()
        for st in source_types:
            out.update(self.by_source_type_set.get(st, set()))
        return out

    def _exact_search(
        self,
        query: str,
        filters: dict | None,
        source_types: list[str] | None,
        top_k: int,
    ) -> list[EvidenceChunk]:
        ids = _extract_exact_ids(query)
        if not ids:
            return []

        candidate_ids: set[int] = set()
        for x in ids:
            candidate_ids.update(self.exact_index.get(_norm_id(x), set()))

        if not candidate_ids:
            return []

        source_set = self._source_candidate_set(source_types)
        if source_set is not None:
            candidate_ids &= source_set

        scored: list[tuple[float, int, EvidenceChunk]] = []

        for idx in candidate_ids:
            c = self.chunks[idx]
            if not _matches(c, filters, source_types):
                continue

            score = 1000.0

            cid = _norm_id(c.chunk_id)
            did = _norm_id(c.doc_id)
            title = _norm_id(c.title)
            meta = c.metadata or {}

            # Prefer direct exact metadata/doc matches.
            for x in ids:
                nx = _norm_id(x)
                if nx == did or nx == cid:
                    score += 100.0
                if nx == _norm_id(meta.get("case_number")):
                    score += 80.0
                if nx in title:
                    score += 10.0

            # Prefer case summary over docket when asking summary.
            if "summary" in _norm_id(c.source_subtype):
                score += 20.0
            if "case_summary" in _norm_id(c.source_subtype):
                score += 20.0

            scored.append((score, -idx, c))

        scored.sort(reverse=True)
        return [_copy(c, score) for score, _, c in scored[:top_k]]

    def _candidate_indices(
        self,
        query_terms: Iterable[str],
        source_types: list[str] | None,
    ) -> set[int]:
        candidate_ids: set[int] = set()

        for term in set(query_terms):
            ids = self.inverted.get(term)
            if ids:
                candidate_ids.update(ids)

        source_set = self._source_candidate_set(source_types)
        if source_set is not None:
            if candidate_ids:
                candidate_ids &= source_set
            else:
                candidate_ids = set(source_set)

        return candidate_ids

    def search(
        self,
        query: str,
        filters: dict | None = None,
        source_types: list[str] | None = None,
        top_k: int = 20,
    ) -> list[EvidenceChunk]:
        if top_k <= 0 or not self.chunks:
            return []

        q = tokenize(query)
        if not q:
            return []

        # Fast path for exact identifiers: case number, patent number, serial/doc id.
        exact_hits = self._exact_search(query, filters, source_types, top_k)
        if exact_hits:
            return exact_hits[:top_k]

        candidate_ids = self._candidate_indices(q, source_types)
        if not candidate_ids:
            return []

        N = len(self.chunks)
        heap: list[tuple[float, int, EvidenceChunk]] = []

        for idx in candidate_ids:
            c = self.chunks[idx]
            if not _matches(c, filters, source_types):
                continue

            toks = self.docs[idx]
            tf = self.term_freqs[idx]
            score = 0.0
            dl = len(toks) or 1

            for term in q:
                freq = tf.get(term, 0)
                if not freq:
                    continue

                df = self.df.get(term, 0)
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * (freq * (self.k1 + 1) / denom)

            if score <= 0:
                continue

            item = (float(score), -idx, c)

            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

        hits = sorted(heap, reverse=True)
        return [_copy(c, score) for score, _, c in hits]
