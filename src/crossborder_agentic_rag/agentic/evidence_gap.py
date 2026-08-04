"""Evidence coverage helpers for the v1 screening runtime."""

from __future__ import annotations


def find_missing_evidence(scope: list[str], hits) -> list[str]:
    present = {hit.source_type for hit in hits}
    return [item for item in scope if item not in present]
