"""
Evidence Graph to JURIS State Conversion.

Maps Evidence Graph → MultiViewState for integration with CRE/WME.
Exposes claims as symbolic tokens and features.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
import numpy as np

from .schema import EvidenceGraph, Claim, Polarity
from .ontology import ClaimType, get_all_claim_types, CLAIM_TYPE_METADATA


# Token vocabulary for claim types and polarities
CLAIM_TYPE_TOKENS = {ct: f"CT_{ct.value.upper()}" for ct in ClaimType}
POLARITY_TOKENS = {
    Polarity.SUPPORTIVE: "POL_SUPPORTIVE",
    Polarity.RISK: "POL_RISK",
    Polarity.NEUTRAL: "POL_NEUTRAL",
}

# Special tokens
TOKEN_MISSING_CLAIM = "MISSING_CLAIM"
TOKEN_LOW_CONFIDENCE = "LOW_CONF"
TOKEN_HIGH_CONFIDENCE = "HIGH_CONF"
TOKEN_UNKNOWN = "UNK"


@dataclass
class SymbolicToken:
    """
    A symbolic token representing evidence information.

    Used for CRE (Critic-Refiner-Executor) processing.
    """
    token: str
    """The token string."""

    token_type: str
    """Type of token: 'claim_type', 'polarity', 'field', 'value', 'special'."""

    source_claim_id: Optional[str] = None
    """Reference to source claim if applicable."""

    confidence: float = 1.0
    """Confidence associated with this token."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""


@dataclass
class EvidenceFeatures:
    """
    Numeric feature representation of evidence for WME.

    Provides structured features for world model processing.
    """
    # Coverage features (15-dim, one per claim type)
    type_coverage: np.ndarray
    """Binary vector: 1 if claim type has at least one claim."""

    type_claim_counts: np.ndarray
    """Count of claims per type."""

    type_avg_confidence: np.ndarray
    """Average confidence per claim type (0 if no claims)."""

    # Polarity features (3-dim)
    polarity_counts: np.ndarray
    """[supportive_count, risk_count, neutral_count]."""

    polarity_ratios: np.ndarray
    """[supportive_ratio, risk_ratio, neutral_ratio]."""

    # Overall features
    total_claims: int
    coverage_ratio: float
    average_confidence: float
    epistemic_uncertainty: float

    # Risk profile
    risk_score: float
    """Weighted risk score based on risk claims and their confidence."""

    support_score: float
    """Weighted support score based on supportive claims and their confidence."""

    def to_vector(self) -> np.ndarray:
        """Flatten all features to a single vector."""
        return np.concatenate([
            self.type_coverage,
            self.type_claim_counts / max(self.total_claims, 1),  # Normalized
            self.type_avg_confidence,
            self.polarity_counts / max(self.total_claims, 1),
            self.polarity_ratios,
            np.array([
                self.total_claims / 100,  # Normalized
                self.coverage_ratio,
                self.average_confidence,
                self.epistemic_uncertainty,
                self.risk_score,
                self.support_score,
            ]),
        ])

    @property
    def feature_dim(self) -> int:
        """Total dimension of feature vector."""
        return len(self.to_vector())


@dataclass
class EvidenceMultiViewState:
    """
    Multi-view state representation of evidence graph.

    Analogous to MultiViewState for grids, but for VC evidence.
    """
    evidence_graph: EvidenceGraph
    """Original evidence graph."""

    symbolic_tokens: List[SymbolicToken]
    """Tokenized representation for CRE."""

    features: EvidenceFeatures
    """Numeric features for WME."""

    claim_map: Dict[str, Claim]
    """Map from claim_id to Claim for lookup."""

    type_claims: Dict[ClaimType, List[Claim]]
    """Claims organized by type."""

    @property
    def company_id(self) -> str:
        return self.evidence_graph.company_id

    @property
    def claim_count(self) -> int:
        return self.evidence_graph.claim_count

    @property
    def epistemic_uncertainty(self) -> float:
        return self.features.epistemic_uncertainty


class EvidenceStateBuilder:
    """
    Builds EvidenceMultiViewState from an EvidenceGraph.

    Performs tokenization and feature extraction.
    """

    def __init__(
        self,
        include_values_as_tokens: bool = True,
        max_value_token_length: int = 50,
        confidence_threshold_low: float = 0.3,
        confidence_threshold_high: float = 0.8,
    ):
        """
        Initialize builder.

        Args:
            include_values_as_tokens: Whether to tokenize claim values
            max_value_token_length: Max length for value tokens
            confidence_threshold_low: Below this is "low confidence"
            confidence_threshold_high: Above this is "high confidence"
        """
        self.include_values_as_tokens = include_values_as_tokens
        self.max_value_token_length = max_value_token_length
        self.confidence_threshold_low = confidence_threshold_low
        self.confidence_threshold_high = confidence_threshold_high

    def build(self, evidence_graph: EvidenceGraph) -> EvidenceMultiViewState:
        """
        Build complete multi-view state from evidence graph.

        Args:
            evidence_graph: The evidence graph to convert

        Returns:
            EvidenceMultiViewState with tokens and features
        """
        # Organize claims by type
        type_claims = self._organize_by_type(evidence_graph)

        # Build claim map
        claim_map = self._build_claim_map(evidence_graph)

        # Generate symbolic tokens
        symbolic_tokens = self._tokenize(evidence_graph, type_claims)

        # Extract features
        features = self._extract_features(evidence_graph, type_claims)

        return EvidenceMultiViewState(
            evidence_graph=evidence_graph,
            symbolic_tokens=symbolic_tokens,
            features=features,
            claim_map=claim_map,
            type_claims=type_claims,
        )

    def _organize_by_type(
        self,
        graph: EvidenceGraph,
    ) -> Dict[ClaimType, List[Claim]]:
        """Organize claims by their type."""
        result: Dict[ClaimType, List[Claim]] = {ct: [] for ct in ClaimType}
        for claim in graph.claims:
            result[claim.claim_type].append(claim)
        return result

    def _build_claim_map(self, graph: EvidenceGraph) -> Dict[str, Claim]:
        """Build map from claim_id to claim."""
        claim_map = {}
        for i, claim in enumerate(graph.claims):
            claim_id = claim.claim_id or f"claim_{i}"
            claim_map[claim_id] = claim
        return claim_map

    def _tokenize(
        self,
        graph: EvidenceGraph,
        type_claims: Dict[ClaimType, List[Claim]],
    ) -> List[SymbolicToken]:
        """
        Generate symbolic tokens from evidence graph.

        Token sequence structure:
        [type_token, polarity_token, field_token, value_token?, confidence_token]
        repeated for each claim, plus MISSING_CLAIM tokens for uncovered types.
        """
        tokens: List[SymbolicToken] = []

        # Add tokens for each claim type
        for claim_type in get_all_claim_types():
            claims = type_claims.get(claim_type, [])

            if not claims:
                # Missing claim type - add uncertainty token
                tokens.append(SymbolicToken(
                    token=TOKEN_MISSING_CLAIM,
                    token_type="special",
                    confidence=0.0,
                    metadata={"missing_type": claim_type.value},
                ))
                continue

            # Add tokens for each claim
            for claim in claims:
                claim_id = claim.claim_id or f"claim_{len(tokens)}"

                # Claim type token
                tokens.append(SymbolicToken(
                    token=CLAIM_TYPE_TOKENS[claim_type],
                    token_type="claim_type",
                    source_claim_id=claim_id,
                    confidence=claim.confidence,
                ))

                # Polarity token
                tokens.append(SymbolicToken(
                    token=POLARITY_TOKENS[claim.polarity],
                    token_type="polarity",
                    source_claim_id=claim_id,
                    confidence=claim.confidence,
                ))

                # Field token
                tokens.append(SymbolicToken(
                    token=f"FIELD_{claim.field.upper()}",
                    token_type="field",
                    source_claim_id=claim_id,
                    confidence=claim.confidence,
                ))

                # Value token (if enabled)
                if self.include_values_as_tokens:
                    value_token = self._value_to_token(claim.value)
                    tokens.append(SymbolicToken(
                        token=value_token,
                        token_type="value",
                        source_claim_id=claim_id,
                        confidence=claim.confidence,
                        metadata={"raw_value": claim.value, "unit": claim.unit},
                    ))

                # Confidence marker token
                if claim.confidence < self.confidence_threshold_low:
                    tokens.append(SymbolicToken(
                        token=TOKEN_LOW_CONFIDENCE,
                        token_type="special",
                        source_claim_id=claim_id,
                        confidence=claim.confidence,
                    ))
                elif claim.confidence >= self.confidence_threshold_high:
                    tokens.append(SymbolicToken(
                        token=TOKEN_HIGH_CONFIDENCE,
                        token_type="special",
                        source_claim_id=claim_id,
                        confidence=claim.confidence,
                    ))

        return tokens

    def _value_to_token(self, value: Any) -> str:
        """Convert a claim value to a token string."""
        if value is None:
            return TOKEN_UNKNOWN

        if isinstance(value, bool):
            return "VAL_TRUE" if value else "VAL_FALSE"

        if isinstance(value, (int, float)):
            # Bucket numeric values
            if value <= 0:
                return "VAL_ZERO_OR_NEG"
            elif value < 100:
                return "VAL_SMALL"
            elif value < 10000:
                return "VAL_MEDIUM"
            elif value < 1000000:
                return "VAL_LARGE"
            else:
                return "VAL_VERY_LARGE"

        if isinstance(value, str):
            # Truncate and normalize
            val_str = value[:self.max_value_token_length].upper().replace(" ", "_")
            return f"VAL_{val_str}"

        if isinstance(value, list):
            return f"VAL_LIST_{len(value)}"

        if isinstance(value, dict):
            return f"VAL_DICT_{len(value)}"

        return TOKEN_UNKNOWN

    def _extract_features(
        self,
        graph: EvidenceGraph,
        type_claims: Dict[ClaimType, List[Claim]],
    ) -> EvidenceFeatures:
        """Extract numeric features from evidence graph."""
        all_types = get_all_claim_types()
        n_types = len(all_types)

        # Coverage features
        type_coverage = np.zeros(n_types)
        type_claim_counts = np.zeros(n_types)
        type_avg_confidence = np.zeros(n_types)

        for i, ct in enumerate(all_types):
            claims = type_claims.get(ct, [])
            if claims:
                type_coverage[i] = 1.0
                type_claim_counts[i] = len(claims)
                type_avg_confidence[i] = sum(c.confidence for c in claims) / len(claims)

        # Polarity features
        supportive_claims = graph.get_supportive_claims()
        risk_claims = graph.get_risk_claims()
        neutral_claims = [c for c in graph.claims if c.polarity == Polarity.NEUTRAL]

        polarity_counts = np.array([
            len(supportive_claims),
            len(risk_claims),
            len(neutral_claims),
        ], dtype=float)

        total = max(graph.claim_count, 1)
        polarity_ratios = polarity_counts / total

        # Risk and support scores (confidence-weighted)
        risk_score = 0.0
        if risk_claims:
            risk_score = sum(c.confidence for c in risk_claims) / len(risk_claims)
            # Scale by proportion of risk claims
            risk_score *= len(risk_claims) / total

        support_score = 0.0
        if supportive_claims:
            support_score = sum(c.confidence for c in supportive_claims) / len(supportive_claims)
            support_score *= len(supportive_claims) / total

        return EvidenceFeatures(
            type_coverage=type_coverage,
            type_claim_counts=type_claim_counts,
            type_avg_confidence=type_avg_confidence,
            polarity_counts=polarity_counts,
            polarity_ratios=polarity_ratios,
            total_claims=graph.claim_count,
            coverage_ratio=graph.coverage_ratio,
            average_confidence=graph.average_confidence,
            epistemic_uncertainty=graph.overall_epistemic_uncertainty,
            risk_score=risk_score,
            support_score=support_score,
        )


def build_evidence_state(evidence_graph: EvidenceGraph, **kwargs) -> EvidenceMultiViewState:
    """
    Convenience function to build evidence state.

    Args:
        evidence_graph: The evidence graph to convert
        **kwargs: Additional arguments for EvidenceStateBuilder

    Returns:
        EvidenceMultiViewState
    """
    builder = EvidenceStateBuilder(**kwargs)
    return builder.build(evidence_graph)


def evidence_to_wme_features(evidence_graph: EvidenceGraph) -> Dict[str, Any]:
    """
    Convert evidence graph to features for World Model Expert.

    Returns dictionary compatible with WME feature format.
    """
    state = build_evidence_state(evidence_graph)

    return {
        "company_id": evidence_graph.company_id,
        "claim_count": state.claim_count,
        "coverage_ratio": state.features.coverage_ratio,
        "average_confidence": state.features.average_confidence,
        "epistemic_uncertainty": state.features.epistemic_uncertainty,
        "risk_score": state.features.risk_score,
        "support_score": state.features.support_score,
        "type_coverage": state.features.type_coverage.tolist(),
        "polarity_ratios": state.features.polarity_ratios.tolist(),
        "feature_vector": state.features.to_vector().tolist(),
        "missing_types": [ct.value for ct in evidence_graph.missing_types],
    }


def evidence_to_cre_tokens(evidence_graph: EvidenceGraph) -> List[str]:
    """
    Convert evidence graph to token sequence for CRE.

    Returns list of token strings.
    """
    state = build_evidence_state(evidence_graph)
    return [t.token for t in state.symbolic_tokens]
