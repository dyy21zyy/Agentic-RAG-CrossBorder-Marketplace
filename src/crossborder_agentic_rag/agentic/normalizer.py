from typing import Any


def normalize_user_query(query: str, target_markets: list[str] | None) -> dict[str, Any]:
    cleaned = " ".join((query or "").split())
    if not cleaned:
        raise ValueError("query must be non-empty")
    markets = target_markets or ["US"]
    return {"query": cleaned, "target_markets": markets}
