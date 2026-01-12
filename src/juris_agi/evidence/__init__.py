"""
Evidence Graph module for VC Investment Decisions.

This module provides data structures and utilities for representing,
loading, validating, and converting structured evidence claims for
venture capital investment analysis.

Key components:
- ontology: 15 claim types for VC due diligence
- schema: Evidence graph and claim data structures
- loader: JSON loading and validation
- to_juris_state: Conversion to JURIS-AGI MultiViewState
"""

from .ontology import (
    ClaimType,
    ClaimTypeInfo,
    CLAIM_TYPE_DESCRIPTIONS,
    CLAIM_TYPE_METADATA,
    get_claim_type,
    get_all_claim_types,
    get_risk_claim_types,
    get_claim_type_info,
)

from .schema import (
    Polarity,
    Source,
    Claim,
    EvidenceGraph,
    ClaimSummary,
    ClaimValue,
)

from .loader import (
    ValidationIssue,
    LoadResult,
    EvidenceGraphLoader,
    load_evidence_graph,
    validate_evidence_graph,
    summarize_evidence_graph,
)

from .to_juris_state import (
    SymbolicToken,
    EvidenceFeatures,
    EvidenceMultiViewState,
    EvidenceStateBuilder,
    build_evidence_state,
    evidence_to_wme_features,
    evidence_to_cre_tokens,
)

__all__ = [
    # Ontology
    "ClaimType",
    "ClaimTypeInfo",
    "CLAIM_TYPE_DESCRIPTIONS",
    "CLAIM_TYPE_METADATA",
    "get_claim_type",
    "get_all_claim_types",
    "get_risk_claim_types",
    "get_claim_type_info",
    # Schema
    "Polarity",
    "Source",
    "Claim",
    "EvidenceGraph",
    "ClaimSummary",
    "ClaimValue",
    # Loader
    "ValidationIssue",
    "LoadResult",
    "EvidenceGraphLoader",
    "load_evidence_graph",
    "validate_evidence_graph",
    "summarize_evidence_graph",
    # State conversion
    "SymbolicToken",
    "EvidenceFeatures",
    "EvidenceMultiViewState",
    "EvidenceStateBuilder",
    "build_evidence_state",
    "evidence_to_wme_features",
    "evidence_to_cre_tokens",
]
