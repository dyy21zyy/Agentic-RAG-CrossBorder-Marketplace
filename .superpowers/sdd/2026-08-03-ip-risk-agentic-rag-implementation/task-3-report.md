# Task 3 Report: Image Contract Compatibility in Ingestion

## Status

DONE_WITH_CONCERNS

## Scope

Task 3 preserves the image contract through ingestion and chunking. Current XML, TSV, and CSV parser paths continue to construct `NormalizedDocument` with the schema default `images=[]`. `make_chunk` now copies `doc.images` into every `EvidenceChunk` using a new list, so future enrichment does not share the document list by reference.

No parser-specific image dictionaries were added. No chain-of-thought data or output was introduced.

## TDD Evidence

### Specified RED test

The brief specified:

```text
pytest -q tests/test_v1_schema_contracts.py::test_read_fixture_chunks_have_images_field
```

Observed output:

```text
.                                                                        [100%]
1 passed in 0.03s
```

This RED was pre-satisfied by Task 1. The existing `EvidenceChunk.images` default and `EvidenceChunk.from_dict()` fallback already make legacy JSONL chunks expose `images=[]`. The existing `NormalizedDocument` default and `from_dict()` fallback similarly satisfy the legacy document contract. Good Task 1 schema behavior was retained.

### Genuine RED test

Added `test_chunker_preserves_document_images` in `tests/test_stage3_chunkers.py`, then ran:

```text
pytest -q tests/test_stage3_chunkers.py::test_chunker_preserves_document_images
```

Observed RED output before implementation:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
___________________ test_chunker_preserves_document_images ____________________
...
>       assert all(chunk.images == [image] for chunk in chunks)
E       assert False
E        +  where False = all(<generator object test_chunker_preserves_document_images.<locals>.<genexpr> at 0x0000020FC59C9F20>)
1 failed in 0.22s
```

The failure was caused by `make_chunk` not passing document images to `EvidenceChunk`.

### Focused GREEN

```text
uv run pytest -q tests/test_v1_schema_contracts.py tests/test_stage2_parsers.py tests/test_stage3_chunkers.py
```

Observed output:

```text
............................................................             [100%]
60 passed in 1.98s
```

## Required Gates

### Ruff

Command:

```text
uv run ruff check .
```

Observed output:

```text
All checks passed!
```

### Full pytest

Command:

```text
uv run pytest -q
```

Observed output:

```text
........................................................................ [ 21%]
........................................................................ [ 43%]
........................................................................ [ 65%]
........................................................................ [ 87%]
.........................................                                [100%]
329 passed in 27.25s
```

## Files Changed

- `src/crossborder_agentic_rag/ingestion/chunkers.py`
- `tests/test_v1_schema_contracts.py`
- `tests/test_stage3_chunkers.py`
- `.superpowers/sdd/2026-08-03-ip-risk-agentic-rag-implementation/task-3-report.md`

## Commit

Commit message: `feat: preserve image contract through ingestion`

Implementation commit SHA: `6843053`.

The report was subsequently included in amended commit `7ea8937`. The implementation commit SHA above is retained as the stable reference for the code change.

## Concerns

1. The exact RED test prescribed by the brief could not fail because Task 1 had already implemented the required schema defaults and legacy deserialization behavior. This was recorded rather than forcing a regression.
2. The focused propagation test uses a non-empty `ImageAsset` to prove the future enrichment path. Existing XML/TSV/CSV parsers remain text-only and emit `images=[]` through schema defaults; no parser-specific test assertion was added beyond the legacy JSONL contract and the existing parser suite.
