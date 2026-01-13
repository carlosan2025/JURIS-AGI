"""
Decision Report Schema for JURIS-AGI.

Defines the structure of a formal decision report with all required sections.
All data in the report must be traceable to the evidence graph or reasoning trace.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional


class DecisionOutcome(Enum):
    """Investment decision outcomes."""
    INVEST = "invest"
    PASS = "pass"
    DEFER = "defer"


class Polarity(Enum):
    """Evidence polarity."""
    SUPPORTIVE = "supportive"
    RISK = "risk"
    NEUTRAL = "neutral"


class ConfidenceLevel(Enum):
    """Qualitative confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Convert numeric score to confidence level."""
        if score >= 0.8:
            return cls.HIGH
        elif score >= 0.6:
            return cls.MEDIUM
        else:
            return cls.LOW


# =============================================================================
# Evidence Basis Components
# =============================================================================

@dataclass
class EvidenceItem:
    """
    A single piece of evidence from the evidence graph.

    All fields are derived directly from the Claim object.
    """
    claim_type: str
    """Type of claim (e.g., 'traction', 'team_quality')."""

    field: str
    """Field name (e.g., 'arr', 'founder_background')."""

    value: Any
    """The claim value."""

    confidence: float
    """Confidence score (0-1)."""

    polarity: Polarity
    """Evidence polarity."""

    source_document: Optional[str] = None
    """Source document ID if available."""

    source_locator: Optional[str] = None
    """Location within source document."""

    quote: Optional[str] = None
    """Direct quote from source if available."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_type": self.claim_type,
            "field": self.field,
            "value": self.value,
            "confidence": self.confidence,
            "polarity": self.polarity.value,
            "source_document": self.source_document,
            "source_locator": self.source_locator,
            "quote": self.quote,
        }


@dataclass
class EvidenceBasis:
    """
    Section: Evidence Basis

    Documents all evidence that contributed to the decision.
    Grouped by polarity and claim type.
    """
    total_claims: int
    """Total number of claims in evidence graph."""

    supportive_claims: List[EvidenceItem]
    """Claims supporting investment."""

    risk_claims: List[EvidenceItem]
    """Claims indicating risk."""

    neutral_claims: List[EvidenceItem]
    """Neutral claims providing context."""

    coverage_by_type: Dict[str, int]
    """Number of claims by claim type."""

    average_confidence: float
    """Average confidence across all claims."""

    low_confidence_claims: List[EvidenceItem]
    """Claims with confidence below threshold (potential weaknesses)."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "supportive_claims": [c.to_dict() for c in self.supportive_claims],
            "risk_claims": [c.to_dict() for c in self.risk_claims],
            "neutral_claims": [c.to_dict() for c in self.neutral_claims],
            "coverage_by_type": self.coverage_by_type,
            "average_confidence": self.average_confidence,
            "low_confidence_claims": [c.to_dict() for c in self.low_confidence_claims],
        }


# =============================================================================
# Decision Logic Components
# =============================================================================

@dataclass
class RuleEvaluation:
    """
    Evaluation of a single decision rule.

    Derived from the reasoning trace.
    """
    rule_name: str
    """Name of the rule."""

    rule_description: str
    """Human-readable description of what the rule evaluates."""

    result: str
    """Outcome of rule evaluation (pass/fail/partial)."""

    contributing_evidence: List[str]
    """List of evidence items that contributed to this rule."""

    weight: float
    """Weight of this rule in final decision (0-1)."""

    notes: Optional[str] = None
    """Additional notes on rule evaluation."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "rule_description": self.rule_description,
            "result": self.result,
            "contributing_evidence": self.contributing_evidence,
            "weight": self.weight,
            "notes": self.notes,
        }


@dataclass
class InferredDecisionLogic:
    """
    Section: Inferred Decision Logic

    Explains the rules and logic that led to the decision.
    All logic must be traceable to the reasoning trace.
    """
    primary_factors: List[str]
    """Main factors that drove the decision."""

    decision_threshold_description: str
    """Description of how decision threshold was applied."""

    rule_evaluations: List[RuleEvaluation]
    """Individual rule evaluations."""

    net_signal_score: float
    """Net signal score (supportive - risk, weighted by confidence)."""

    decision_rationale: str
    """Natural language explanation of why the decision was reached."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_factors": self.primary_factors,
            "decision_threshold_description": self.decision_threshold_description,
            "rule_evaluations": [r.to_dict() for r in self.rule_evaluations],
            "net_signal_score": self.net_signal_score,
            "decision_rationale": self.decision_rationale,
        }


# =============================================================================
# Deal Evaluation Components
# =============================================================================

@dataclass
class DealEvaluation:
    """
    Section: Deal Evaluation Against Rules

    Structured evaluation of the deal against investment criteria.
    """
    company_id: str
    """Company identifier."""

    company_name: Optional[str] = None
    """Company name if available."""

    sector: Optional[str] = None
    """Sector/industry."""

    stage: Optional[str] = None
    """Investment stage (Seed, Series A, etc.)."""

    # Key metrics (derived from evidence)
    traction_summary: Optional[str] = None
    """Summary of traction metrics."""

    team_summary: Optional[str] = None
    """Summary of team assessment."""

    market_summary: Optional[str] = None
    """Summary of market opportunity."""

    risk_summary: Optional[str] = None
    """Summary of key risks."""

    # Scores
    traction_score: Optional[float] = None
    team_score: Optional[float] = None
    market_score: Optional[float] = None
    risk_score: Optional[float] = None

    overall_score: Optional[float] = None
    """Overall deal score if computed."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "sector": self.sector,
            "stage": self.stage,
            "traction_summary": self.traction_summary,
            "team_summary": self.team_summary,
            "market_summary": self.market_summary,
            "risk_summary": self.risk_summary,
            "traction_score": self.traction_score,
            "team_score": self.team_score,
            "market_score": self.market_score,
            "risk_score": self.risk_score,
            "overall_score": self.overall_score,
        }


# =============================================================================
# Counterfactual Analysis Components
# =============================================================================

@dataclass
class SensitivityItem:
    """
    A single sensitivity/counterfactual finding.

    Derived from counterfactual analysis in the trace.
    """
    claim_type: str
    """Type of claim that was perturbed."""

    claim_field: str
    """Field that was perturbed."""

    original_value: Any
    """Original value before perturbation."""

    perturbed_value: Any
    """Value after perturbation."""

    original_decision: DecisionOutcome
    """Decision before perturbation."""

    new_decision: DecisionOutcome
    """Decision after perturbation."""

    perturbation_magnitude: float
    """Size of perturbation (0-1)."""

    criticality_score: float
    """How critical this claim is to the decision (0-1)."""

    explanation: str
    """Natural language explanation of the sensitivity."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_type": self.claim_type,
            "claim_field": self.claim_field,
            "original_value": self.original_value,
            "perturbed_value": self.perturbed_value,
            "original_decision": self.original_decision.value,
            "new_decision": self.new_decision.value,
            "perturbation_magnitude": self.perturbation_magnitude,
            "criticality_score": self.criticality_score,
            "explanation": self.explanation,
        }


@dataclass
class CounterfactualAnalysis:
    """
    Section: Counterfactual & Sensitivity Analysis

    Documents how the decision would change under different evidence scenarios.
    All derived from counterfactual testing in the reasoning trace.
    """
    robustness_score: float
    """Overall robustness score (0-1). Higher = more robust."""

    stability_margin: float
    """Average perturbation magnitude needed to flip decision."""

    total_counterfactuals_tested: int
    """Number of counterfactuals evaluated."""

    flips_found: int
    """Number of counterfactuals that flipped the decision."""

    critical_claims: List[SensitivityItem]
    """Claims most likely to flip the decision if changed."""

    decision_flip_scenarios: List[SensitivityItem]
    """Specific scenarios where decision flips."""

    robustness_interpretation: str
    """Natural language interpretation of robustness."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robustness_score": self.robustness_score,
            "stability_margin": self.stability_margin,
            "total_counterfactuals_tested": self.total_counterfactuals_tested,
            "flips_found": self.flips_found,
            "critical_claims": [c.to_dict() for c in self.critical_claims],
            "decision_flip_scenarios": [s.to_dict() for s in self.decision_flip_scenarios],
            "robustness_interpretation": self.robustness_interpretation,
        }


# =============================================================================
# Uncertainty & Limitations
# =============================================================================

@dataclass
class UncertaintyLimitations:
    """
    Section: Uncertainty & Limitations

    Documents known uncertainties, data gaps, and limitations.
    """
    epistemic_uncertainty: float
    """Uncertainty due to lack of knowledge (reducible)."""

    aleatoric_uncertainty: float
    """Inherent randomness (irreducible)."""

    total_uncertainty: float
    """Combined uncertainty measure."""

    confidence_level: ConfidenceLevel
    """Qualitative confidence level."""

    data_gaps: List[str]
    """Missing or incomplete data areas."""

    low_confidence_areas: List[str]
    """Areas where confidence is below threshold."""

    assumptions_made: List[str]
    """Key assumptions underlying the analysis."""

    limitations: List[str]
    """Known limitations of this analysis."""

    recommendations_for_diligence: List[str]
    """Recommended areas for further due diligence."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "total_uncertainty": self.total_uncertainty,
            "confidence_level": self.confidence_level.value,
            "data_gaps": self.data_gaps,
            "low_confidence_areas": self.low_confidence_areas,
            "assumptions_made": self.assumptions_made,
            "limitations": self.limitations,
            "recommendations_for_diligence": self.recommendations_for_diligence,
        }


# =============================================================================
# Executive Summary
# =============================================================================

@dataclass
class ExecutiveSummary:
    """
    Section: Executive Summary

    High-level summary of the decision and key findings.
    """
    decision: DecisionOutcome
    """Final investment decision."""

    confidence: float
    """Confidence in the decision (0-1)."""

    confidence_level: ConfidenceLevel
    """Qualitative confidence level."""

    one_line_summary: str
    """Single sentence summary of the decision."""

    key_strengths: List[str]
    """Top 3-5 strengths supporting investment."""

    key_risks: List[str]
    """Top 3-5 risks against investment."""

    critical_uncertainties: List[str]
    """Top uncertainties that could change the decision."""

    recommendation_text: str
    """Full recommendation paragraph."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "one_line_summary": self.one_line_summary,
            "key_strengths": self.key_strengths,
            "key_risks": self.key_risks,
            "critical_uncertainties": self.critical_uncertainties,
            "recommendation_text": self.recommendation_text,
        }


# =============================================================================
# Audit Metadata
# =============================================================================

@dataclass
class AuditMetadata:
    """
    Section: Audit Metadata

    Provenance and audit information for the report.
    """
    report_id: str
    """Unique report identifier."""

    generated_at: datetime
    """When the report was generated."""

    model_version: str
    """Version of the decision model used."""

    trace_id: Optional[str] = None
    """ID of the reasoning trace this report is based on."""

    analyst_id: Optional[str] = None
    """Analyst who created the evidence graph."""

    evidence_graph_version: Optional[str] = None
    """Version of the evidence graph."""

    num_claims_analyzed: int = 0
    """Total claims in evidence graph."""

    num_counterfactuals_tested: int = 0
    """Total counterfactuals tested."""

    analysis_runtime_seconds: Optional[float] = None
    """Time taken for analysis."""

    reproducibility_seed: Optional[int] = None
    """Random seed used for reproducibility."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "model_version": self.model_version,
            "trace_id": self.trace_id,
            "analyst_id": self.analyst_id,
            "evidence_graph_version": self.evidence_graph_version,
            "num_claims_analyzed": self.num_claims_analyzed,
            "num_counterfactuals_tested": self.num_counterfactuals_tested,
            "analysis_runtime_seconds": self.analysis_runtime_seconds,
            "reproducibility_seed": self.reproducibility_seed,
        }


# =============================================================================
# Complete Decision Report
# =============================================================================

@dataclass
class DecisionReport:
    """
    Complete Decision Report for JURIS-AGI.

    Contains all sections required for a formal investment decision report.
    All content is derived from the evidence graph and reasoning trace.
    """
    # Core sections
    executive_summary: ExecutiveSummary
    """High-level decision summary."""

    decision_logic: InferredDecisionLogic
    """Explanation of decision rules and logic."""

    evidence_basis: EvidenceBasis
    """All evidence supporting the decision."""

    deal_evaluation: DealEvaluation
    """Structured deal evaluation."""

    counterfactual_analysis: CounterfactualAnalysis
    """Sensitivity and counterfactual findings."""

    uncertainty_limitations: UncertaintyLimitations
    """Known uncertainties and limitations."""

    audit_metadata: AuditMetadata
    """Provenance and audit information."""

    # Optional additional content
    appendix: Optional[Dict[str, Any]] = None
    """Additional data for appendix."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire report to dictionary."""
        result = {
            "executive_summary": self.executive_summary.to_dict(),
            "decision_logic": self.decision_logic.to_dict(),
            "evidence_basis": self.evidence_basis.to_dict(),
            "deal_evaluation": self.deal_evaluation.to_dict(),
            "counterfactual_analysis": self.counterfactual_analysis.to_dict(),
            "uncertainty_limitations": self.uncertainty_limitations.to_dict(),
            "audit_metadata": self.audit_metadata.to_dict(),
        }
        if self.appendix:
            result["appendix"] = self.appendix
        return result

    @property
    def title(self) -> str:
        """Report title."""
        company = self.deal_evaluation.company_name or self.deal_evaluation.company_id
        return f"Investment Decision Report: {company}"

    @property
    def decision(self) -> DecisionOutcome:
        """The final decision."""
        return self.executive_summary.decision

    @property
    def confidence(self) -> float:
        """Decision confidence."""
        return self.executive_summary.confidence
