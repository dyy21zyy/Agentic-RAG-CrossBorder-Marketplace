"""Structured preliminary IP risk screening report schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from crossborder_agentic_rag.constants import SOURCE_TYPES
from crossborder_agentic_rag.schemas._json import json_safe
from crossborder_agentic_rag.schemas.evidence import EvidenceHit


class RiskVerdict(str, Enum):
    NO_RISK_FOUND = "no_risk_found"
    CAUTION = "caution"
    NOT_RECOMMENDED = "not_recommended"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(slots=True)
class RiskScreeningReport:
    report_id: str
    trace_id: str
    created_at: str
    product_profile: dict[str, Any]
    target_markets: list[str]
    screening_scope: list[str]
    overall_verdict: RiskVerdict
    country_summaries: list[dict[str, Any]]
    risk_cards: dict[str, int]
    module_results: list[dict[str, Any]]
    evidence_items: list[EvidenceHit]
    action_recommendations: list[str]
    missing_evidence: list[str]
    limitations: list[str]
    langfuse_url: str | None = None

    def __post_init__(self) -> None:
        try:
            self.overall_verdict = RiskVerdict(self.overall_verdict)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"overall_verdict must be one of {[item.value for item in RiskVerdict]}") from exc
        if not isinstance(self.screening_scope, list):
            raise TypeError("screening_scope must be a list")
        invalid_scope = [scope for scope in self.screening_scope if scope not in SOURCE_TYPES]
        if invalid_scope:
            raise ValueError(f"screening_scope contains invalid values: {invalid_scope}")
        if not isinstance(self.product_profile, dict):
            raise TypeError("product_profile must be a dict")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "report_id": self.report_id,
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "product_profile": json_safe(self.product_profile, "product_profile"),
            "target_markets": list(self.target_markets),
            "screening_scope": list(self.screening_scope),
            "overall_verdict": self.overall_verdict.value,
            "country_summaries": json_safe(self.country_summaries, "country_summaries"),
            "risk_cards": json_safe(self.risk_cards, "risk_cards"),
            "module_results": json_safe(self.module_results, "module_results"),
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "action_recommendations": list(self.action_recommendations),
            "missing_evidence": list(self.missing_evidence),
            "limitations": list(self.limitations),
            "langfuse_url": self.langfuse_url,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskScreeningReport":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        values = dict(data)
        values["overall_verdict"] = RiskVerdict(values["overall_verdict"])
        values["evidence_items"] = [EvidenceHit.from_dict(item) for item in values["evidence_items"]]
        return cls(**values)
