"""
Evidence Graph Loader and Validator.

Loads evidence graphs from JSON with validation against the ontology.
Supports missing claims (increases epistemic uncertainty rather than failing).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from .ontology import ClaimType, get_claim_type, get_all_claim_types, CLAIM_TYPE_METADATA
from .schema import EvidenceGraph, Claim, Source, Polarity, ClaimSummary


logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A validation issue found during loading."""
    severity: str  # "error", "warning", "info"
    message: str
    claim_index: Optional[int] = None
    field: Optional[str] = None

    def __str__(self) -> str:
        loc = ""
        if self.claim_index is not None:
            loc = f"[claim {self.claim_index}] "
        if self.field:
            loc += f"({self.field}) "
        return f"{self.severity.upper()}: {loc}{self.message}"


@dataclass
class LoadResult:
    """Result of loading an evidence graph."""
    success: bool
    evidence_graph: Optional[EvidenceGraph] = None
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)


class EvidenceGraphLoader:
    """
    Loads and validates evidence graphs from JSON.

    Features:
    - Validates claim types against ontology
    - Allows missing claims (adds warnings, not errors)
    - Validates confidence ranges
    - Tracks validation issues without failing on non-critical problems
    """

    def __init__(
        self,
        strict_mode: bool = False,
        warn_on_missing_types: bool = True,
        warn_on_low_confidence: bool = True,
        low_confidence_threshold: float = 0.3,
    ):
        """
        Initialize loader.

        Args:
            strict_mode: If True, treat warnings as errors
            warn_on_missing_types: Warn if claim types from ontology are missing
            warn_on_low_confidence: Warn on claims with very low confidence
            low_confidence_threshold: Threshold for low confidence warnings
        """
        self.strict_mode = strict_mode
        self.warn_on_missing_types = warn_on_missing_types
        self.warn_on_low_confidence = warn_on_low_confidence
        self.low_confidence_threshold = low_confidence_threshold

    def load_from_file(self, path: Union[str, Path]) -> LoadResult:
        """
        Load evidence graph from a JSON file.

        Args:
            path: Path to JSON file

        Returns:
            LoadResult with evidence graph and any validation issues
        """
        path = Path(path)
        issues: List[ValidationIssue] = []

        if not path.exists():
            return LoadResult(
                success=False,
                issues=[ValidationIssue("error", f"File not found: {path}")],
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return LoadResult(
                success=False,
                issues=[ValidationIssue("error", f"Invalid JSON: {e}")],
            )
        except Exception as e:
            return LoadResult(
                success=False,
                issues=[ValidationIssue("error", f"Failed to read file: {e}")],
            )

        return self.load_from_dict(data)

    def load_from_json(self, json_str: str) -> LoadResult:
        """
        Load evidence graph from a JSON string.

        Args:
            json_str: JSON string

        Returns:
            LoadResult with evidence graph and any validation issues
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return LoadResult(
                success=False,
                issues=[ValidationIssue("error", f"Invalid JSON: {e}")],
            )

        return self.load_from_dict(data)

    def load_from_dict(self, data: Dict[str, Any]) -> LoadResult:
        """
        Load evidence graph from a dictionary.

        Args:
            data: Dictionary with evidence graph data

        Returns:
            LoadResult with evidence graph and any validation issues
        """
        issues: List[ValidationIssue] = []

        # Validate required fields
        if "company_id" not in data:
            issues.append(ValidationIssue("error", "Missing required field: company_id"))

        if issues and any(i.severity == "error" for i in issues):
            return LoadResult(success=False, issues=issues)

        # Validate claims
        claims_data = data.get("claims", [])
        valid_claims: List[Claim] = []

        for i, claim_data in enumerate(claims_data):
            claim_issues, claim = self._validate_claim(claim_data, i)
            issues.extend(claim_issues)

            if claim is not None:
                valid_claims.append(claim)

        # Check for missing claim types
        if self.warn_on_missing_types:
            covered_types = {c.claim_type for c in valid_claims}
            missing_types = set(get_all_claim_types()) - covered_types
            for ct in missing_types:
                issues.append(ValidationIssue(
                    "warning" if not self.strict_mode else "error",
                    f"No claims for type '{ct.value}' - increases epistemic uncertainty",
                ))

        # Check for low confidence claims
        if self.warn_on_low_confidence:
            for i, claim in enumerate(valid_claims):
                if claim.confidence < self.low_confidence_threshold:
                    issues.append(ValidationIssue(
                        "warning",
                        f"Very low confidence ({claim.confidence:.2f})",
                        claim_index=i,
                        field=claim.field,
                    ))

        # Determine success
        has_errors = any(i.severity == "error" for i in issues)
        if self.strict_mode:
            has_errors = has_errors or any(i.severity == "warning" for i in issues)

        if has_errors:
            return LoadResult(success=False, issues=issues)

        # Build evidence graph
        try:
            evidence_graph = EvidenceGraph(
                company_id=data["company_id"],
                claims=valid_claims,
                analyst_id=data.get("analyst_id"),
                version=data.get("version", "1.0"),
            )
        except Exception as e:
            return LoadResult(
                success=False,
                issues=[ValidationIssue("error", f"Failed to create evidence graph: {e}")],
            )

        return LoadResult(
            success=True,
            evidence_graph=evidence_graph,
            issues=issues,
        )

    def _validate_claim(
        self,
        data: Dict[str, Any],
        index: int,
    ) -> tuple[List[ValidationIssue], Optional[Claim]]:
        """
        Validate a single claim.

        Returns tuple of (issues, claim) where claim is None if invalid.
        """
        issues: List[ValidationIssue] = []

        # Required fields
        if "claim_type" not in data:
            issues.append(ValidationIssue(
                "error", "Missing claim_type", claim_index=index
            ))
            return issues, None

        if "field" not in data:
            issues.append(ValidationIssue(
                "error", "Missing field", claim_index=index
            ))
            return issues, None

        if "value" not in data:
            issues.append(ValidationIssue(
                "error", "Missing value", claim_index=index
            ))
            return issues, None

        # Validate claim type
        claim_type = get_claim_type(data["claim_type"])
        if claim_type is None:
            issues.append(ValidationIssue(
                "error",
                f"Unknown claim_type: {data['claim_type']}",
                claim_index=index,
            ))
            return issues, None

        # Validate confidence
        confidence = data.get("confidence", 1.0)
        if not isinstance(confidence, (int, float)):
            issues.append(ValidationIssue(
                "error", f"Confidence must be a number, got {type(confidence).__name__}",
                claim_index=index,
            ))
            return issues, None

        if not 0.0 <= confidence <= 1.0:
            issues.append(ValidationIssue(
                "warning",
                f"Confidence {confidence} out of range [0, 1], clamping",
                claim_index=index,
            ))
            confidence = max(0.0, min(1.0, confidence))

        # Validate polarity
        polarity_str = data.get("polarity", "neutral")
        try:
            polarity = Polarity(polarity_str)
        except ValueError:
            issues.append(ValidationIssue(
                "warning",
                f"Unknown polarity '{polarity_str}', defaulting to neutral",
                claim_index=index,
            ))
            polarity = Polarity.NEUTRAL

        # Validate source
        source = None
        if data.get("source"):
            source_data = data["source"]
            if "doc_id" not in source_data:
                issues.append(ValidationIssue(
                    "warning",
                    "Source missing doc_id, source will be ignored",
                    claim_index=index,
                ))
            else:
                try:
                    source = Source.from_dict(source_data)
                except Exception as e:
                    issues.append(ValidationIssue(
                        "warning",
                        f"Failed to parse source: {e}",
                        claim_index=index,
                    ))

        # Build claim
        try:
            claim = Claim(
                claim_type=claim_type,
                field=data["field"],
                value=data["value"],
                confidence=confidence,
                polarity=polarity,
                source=source,
                unit=data.get("unit"),
                notes=data.get("notes"),
                claim_id=data.get("claim_id"),
            )
        except Exception as e:
            issues.append(ValidationIssue(
                "error", f"Failed to create claim: {e}", claim_index=index
            ))
            return issues, None

        return issues, claim


def load_evidence_graph(
    source: Union[str, Path, Dict[str, Any]],
    strict: bool = False,
) -> LoadResult:
    """
    Convenience function to load an evidence graph.

    Args:
        source: File path, JSON string, or dictionary
        strict: If True, treat warnings as errors

    Returns:
        LoadResult with evidence graph and validation issues
    """
    loader = EvidenceGraphLoader(strict_mode=strict)

    if isinstance(source, dict):
        return loader.load_from_dict(source)
    elif isinstance(source, Path) or (isinstance(source, str) and not source.strip().startswith("{")):
        return loader.load_from_file(source)
    else:
        return loader.load_from_json(source)


def validate_evidence_graph(graph: EvidenceGraph) -> List[ValidationIssue]:
    """
    Validate an existing evidence graph.

    Returns list of validation issues found.
    """
    issues: List[ValidationIssue] = []

    # Check coverage
    missing_types = graph.missing_types
    if missing_types:
        for ct in missing_types:
            issues.append(ValidationIssue(
                "warning",
                f"No claims for type '{ct.value}'",
            ))

    # Check for conflicting claims
    claims_by_field: Dict[tuple, List[Claim]] = {}
    for claim in graph.claims:
        key = (claim.claim_type, claim.field)
        if key not in claims_by_field:
            claims_by_field[key] = []
        claims_by_field[key].append(claim)

    for key, claims in claims_by_field.items():
        if len(claims) > 1:
            # Check for conflicting values with high confidence
            high_conf_claims = [c for c in claims if c.confidence >= 0.8]
            if len(high_conf_claims) > 1:
                values = [c.value for c in high_conf_claims]
                if len(set(str(v) for v in values)) > 1:
                    issues.append(ValidationIssue(
                        "warning",
                        f"Conflicting high-confidence claims for {key[0].value}.{key[1]}",
                    ))

    # Check overall epistemic uncertainty
    if graph.overall_epistemic_uncertainty > 0.7:
        issues.append(ValidationIssue(
            "warning",
            f"High overall epistemic uncertainty ({graph.overall_epistemic_uncertainty:.2f})",
        ))

    return issues


def summarize_evidence_graph(graph: EvidenceGraph) -> Dict[str, Any]:
    """
    Generate a summary of an evidence graph.

    Returns dictionary with summary statistics.
    """
    from .ontology import get_all_claim_types

    summaries: Dict[str, ClaimSummary] = {}
    for ct in get_all_claim_types():
        claims = graph.get_claims_by_type(ct)
        summaries[ct.value] = ClaimSummary.from_claims(ct, claims)

    return {
        "company_id": graph.company_id,
        "total_claims": graph.claim_count,
        "coverage_ratio": graph.coverage_ratio,
        "average_confidence": graph.average_confidence,
        "epistemic_uncertainty": graph.overall_epistemic_uncertainty,
        "risk_claims": len(graph.get_risk_claims()),
        "supportive_claims": len(graph.get_supportive_claims()),
        "missing_types": [ct.value for ct in graph.missing_types],
        "type_summaries": {
            ct: {
                "count": s.count,
                "fields": s.fields,
                "avg_confidence": s.avg_confidence,
                "risk_count": s.risk_count,
                "supportive_count": s.supportive_count,
            }
            for ct, s in summaries.items()
        },
    }
