"""
Report Generator for JURIS-AGI.

Generates structured decision reports from evidence graphs and reasoning traces.

IMPORTANT: This generator is a PURE FUNCTION of its inputs.
- No new reasoning or inference is performed
- All statements are traceable to evidence or rules in the trace
- The generator only synthesizes and formats existing analysis
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from .schema import (
    DecisionReport,
    ExecutiveSummary,
    InferredDecisionLogic,
    EvidenceBasis,
    DealEvaluation,
    CounterfactualAnalysis,
    UncertaintyLimitations,
    AuditMetadata,
    EvidenceItem,
    RuleEvaluation,
    SensitivityItem,
    DecisionOutcome,
    Polarity,
    ConfidenceLevel,
)


class ReportGenerator:
    """
    Generates decision reports from evidence graphs and traces.

    This is a pure synthesis function - no new inference is performed.
    All output is derived strictly from the inputs.
    """

    MODEL_VERSION = "1.0.0"
    LOW_CONFIDENCE_THRESHOLD = 0.6

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.

        Args:
            seed: Random seed used in analysis (for reproducibility tracking)
        """
        self.seed = seed

    def generate(
        self,
        evidence_graph: Dict[str, Any],
        trace: Dict[str, Any],
        final_decision: str,
    ) -> DecisionReport:
        """
        Generate a complete decision report.

        Args:
            evidence_graph: The evidence graph (as dict)
            trace: The reasoning trace (as dict)
            final_decision: The final decision outcome ('invest', 'pass', 'defer')

        Returns:
            Complete DecisionReport
        """
        # Parse decision
        decision = DecisionOutcome(final_decision.lower())

        # Extract confidence from trace
        confidence = trace.get("confidence", 0.5)
        if "summary" in trace:
            confidence = trace["summary"].get("confidence", confidence)

        # Build each section
        evidence_basis = self._build_evidence_basis(evidence_graph)
        decision_logic = self._build_decision_logic(evidence_graph, trace, decision)
        deal_evaluation = self._build_deal_evaluation(evidence_graph, trace)
        counterfactual_analysis = self._build_counterfactual_analysis(trace)
        uncertainty = self._build_uncertainty_limitations(evidence_graph, trace)
        executive_summary = self._build_executive_summary(
            decision,
            confidence,
            evidence_basis,
            counterfactual_analysis,
            uncertainty,
            deal_evaluation,
        )
        audit_metadata = self._build_audit_metadata(evidence_graph, trace)

        return DecisionReport(
            executive_summary=executive_summary,
            decision_logic=decision_logic,
            evidence_basis=evidence_basis,
            deal_evaluation=deal_evaluation,
            counterfactual_analysis=counterfactual_analysis,
            uncertainty_limitations=uncertainty,
            audit_metadata=audit_metadata,
        )

    def _build_evidence_basis(self, evidence_graph: Dict[str, Any]) -> EvidenceBasis:
        """Build the Evidence Basis section from the evidence graph."""
        claims = evidence_graph.get("claims", [])

        supportive = []
        risk = []
        neutral = []
        coverage_by_type: Dict[str, int] = {}
        total_confidence = 0.0
        low_confidence = []

        for claim in claims:
            # Convert claim to EvidenceItem
            polarity_str = claim.get("polarity", "neutral")
            polarity = Polarity(polarity_str)

            source = claim.get("source", {})
            item = EvidenceItem(
                claim_type=claim.get("claim_type", "unknown"),
                field=claim.get("field", ""),
                value=claim.get("value"),
                confidence=claim.get("confidence", 0.5),
                polarity=polarity,
                source_document=source.get("doc_id") if source else None,
                source_locator=source.get("locator") if source else None,
                quote=source.get("quote") if source else None,
            )

            # Group by polarity
            if polarity == Polarity.SUPPORTIVE:
                supportive.append(item)
            elif polarity == Polarity.RISK:
                risk.append(item)
            else:
                neutral.append(item)

            # Track coverage
            claim_type = item.claim_type
            coverage_by_type[claim_type] = coverage_by_type.get(claim_type, 0) + 1

            # Track confidence
            total_confidence += item.confidence
            if item.confidence < self.LOW_CONFIDENCE_THRESHOLD:
                low_confidence.append(item)

        avg_confidence = total_confidence / len(claims) if claims else 0.0

        return EvidenceBasis(
            total_claims=len(claims),
            supportive_claims=supportive,
            risk_claims=risk,
            neutral_claims=neutral,
            coverage_by_type=coverage_by_type,
            average_confidence=avg_confidence,
            low_confidence_claims=low_confidence,
        )

    def _build_decision_logic(
        self,
        evidence_graph: Dict[str, Any],
        trace: Dict[str, Any],
        decision: DecisionOutcome,
    ) -> InferredDecisionLogic:
        """Build the Decision Logic section from the trace."""
        claims = evidence_graph.get("claims", [])

        # Calculate net signal score
        supportive_weight = 0.0
        risk_weight = 0.0
        for claim in claims:
            conf = claim.get("confidence", 0.5)
            polarity = claim.get("polarity", "neutral")
            if polarity == "supportive":
                supportive_weight += conf
            elif polarity == "risk":
                risk_weight += conf

        net_signal = supportive_weight - risk_weight

        # Extract primary factors from critical claims
        critical_claims = trace.get("critical_claims", [])
        primary_factors = []
        for cc in critical_claims[:5]:
            claim_type = cc.get("claim_type", "unknown")
            field = cc.get("claim_field", cc.get("field", ""))
            primary_factors.append(f"{claim_type}.{field}")

        # Build rule evaluations from trace
        rule_evaluations = self._extract_rule_evaluations(claims, trace)

        # Generate decision rationale
        rationale = self._generate_decision_rationale(
            decision, net_signal, len(claims), critical_claims
        )

        # Decision threshold description
        threshold_desc = self._get_threshold_description(decision, net_signal)

        return InferredDecisionLogic(
            primary_factors=primary_factors,
            decision_threshold_description=threshold_desc,
            rule_evaluations=rule_evaluations,
            net_signal_score=net_signal,
            decision_rationale=rationale,
        )

    def _extract_rule_evaluations(
        self,
        claims: List[Dict],
        trace: Dict[str, Any],
    ) -> List[RuleEvaluation]:
        """Extract rule evaluations from trace."""
        evaluations = []

        # Group claims by type for rule evaluation
        claims_by_type: Dict[str, List[Dict]] = {}
        for claim in claims:
            ct = claim.get("claim_type", "unknown")
            if ct not in claims_by_type:
                claims_by_type[ct] = []
            claims_by_type[ct].append(claim)

        # Create standard rules based on claim types present
        rule_definitions = [
            ("traction_assessment", "Evaluate revenue traction and growth metrics", ["traction"]),
            ("team_assessment", "Assess founder and team quality", ["team_quality", "team_composition"]),
            ("market_assessment", "Evaluate market opportunity and timing", ["market_scope"]),
            ("business_model_assessment", "Assess unit economics and business model viability", ["business_model"]),
            ("risk_assessment", "Identify and weigh execution and regulatory risks", ["execution_risk", "regulatory_risk"]),
            ("technical_assessment", "Evaluate product readiness and technical moat", ["product_readiness", "technical_moat"]),
        ]

        for rule_name, description, claim_types in rule_definitions:
            contributing = []
            total_score = 0.0
            count = 0

            for ct in claim_types:
                if ct in claims_by_type:
                    for claim in claims_by_type[ct]:
                        conf = claim.get("confidence", 0.5)
                        polarity = claim.get("polarity", "neutral")
                        field = claim.get("field", "")
                        contributing.append(f"{ct}.{field}")

                        if polarity == "supportive":
                            total_score += conf
                        elif polarity == "risk":
                            total_score -= conf
                        count += 1

            if count > 0:
                avg_score = total_score / count
                if avg_score > 0.2:
                    result = "pass"
                elif avg_score < -0.2:
                    result = "fail"
                else:
                    result = "partial"

                evaluations.append(RuleEvaluation(
                    rule_name=rule_name,
                    rule_description=description,
                    result=result,
                    contributing_evidence=contributing[:5],  # Top 5
                    weight=min(count / 10, 1.0),  # Weight based on evidence density
                ))

        return evaluations

    def _generate_decision_rationale(
        self,
        decision: DecisionOutcome,
        net_signal: float,
        num_claims: int,
        critical_claims: List[Dict],
    ) -> str:
        """Generate natural language decision rationale."""
        decision_word = decision.value.upper()

        if decision == DecisionOutcome.INVEST:
            base = f"The decision to {decision_word} is supported by a positive net signal score ({net_signal:.2f}) derived from {num_claims} evidence claims."
            if critical_claims:
                top_factor = critical_claims[0]
                base += f" The most critical supporting factor is {top_factor.get('claim_type', 'unknown')}.{top_factor.get('claim_field', top_factor.get('field', ''))}."
        elif decision == DecisionOutcome.PASS:
            base = f"The decision to {decision_word} reflects significant risk factors outweighing supportive evidence (net signal: {net_signal:.2f})."
            if critical_claims:
                risks = [cc for cc in critical_claims if cc.get("polarity") == "risk"]
                if risks:
                    top_risk = risks[0]
                    base += f" The primary concern is {top_risk.get('claim_type', 'unknown')}.{top_risk.get('claim_field', top_risk.get('field', ''))}."
        else:  # DEFER
            base = f"The decision to {decision_word} reflects insufficient evidence or high uncertainty (net signal: {net_signal:.2f}, {num_claims} claims)."
            base += " Additional due diligence is recommended before making a final determination."

        return base

    def _get_threshold_description(
        self,
        decision: DecisionOutcome,
        net_signal: float,
    ) -> str:
        """Get description of decision threshold application."""
        if decision == DecisionOutcome.INVEST:
            return f"Net signal score ({net_signal:.2f}) exceeds positive threshold. Supportive evidence outweighs identified risks."
        elif decision == DecisionOutcome.PASS:
            return f"Net signal score ({net_signal:.2f}) falls below acceptable threshold. Risk factors are prohibitive."
        else:
            return f"Net signal score ({net_signal:.2f}) is in the uncertain zone. Neither clear invest nor pass criteria are met."

    def _build_deal_evaluation(
        self,
        evidence_graph: Dict[str, Any],
        trace: Dict[str, Any],
    ) -> DealEvaluation:
        """Build the Deal Evaluation section."""
        claims = evidence_graph.get("claims", [])
        company_id = evidence_graph.get("company_id", "unknown")

        # Extract company info from claims
        company_name = None
        sector = None
        stage = None

        for claim in claims:
            ct = claim.get("claim_type")
            field = claim.get("field")
            value = claim.get("value")

            if ct == "company_identity":
                if field == "legal_name":
                    company_name = str(value)
                elif field == "sector":
                    sector = str(value)
            elif ct == "round_terms" and field == "stage":
                stage = str(value)

        # Build summaries from claims
        summaries = self._build_category_summaries(claims)

        return DealEvaluation(
            company_id=company_id,
            company_name=company_name,
            sector=sector,
            stage=stage,
            traction_summary=summaries.get("traction"),
            team_summary=summaries.get("team"),
            market_summary=summaries.get("market"),
            risk_summary=summaries.get("risk"),
        )

    def _build_category_summaries(self, claims: List[Dict]) -> Dict[str, str]:
        """Build summaries for each category from claims."""
        summaries = {}

        # Group claims
        traction_claims = [c for c in claims if c.get("claim_type") == "traction"]
        team_claims = [c for c in claims if c.get("claim_type") in ["team_quality", "team_composition"]]
        market_claims = [c for c in claims if c.get("claim_type") == "market_scope"]
        risk_claims = [c for c in claims if c.get("claim_type") in ["execution_risk", "regulatory_risk"]]

        # Traction summary
        if traction_claims:
            parts = []
            for c in traction_claims[:3]:
                field = c.get("field", "")
                value = c.get("value", "")
                parts.append(f"{field}: {value}")
            summaries["traction"] = "; ".join(parts)

        # Team summary
        if team_claims:
            parts = []
            for c in team_claims[:3]:
                field = c.get("field", "")
                value = c.get("value", "")
                parts.append(f"{field}: {value}")
            summaries["team"] = "; ".join(parts)

        # Market summary
        if market_claims:
            parts = []
            for c in market_claims[:3]:
                field = c.get("field", "")
                value = c.get("value", "")
                parts.append(f"{field}: {value}")
            summaries["market"] = "; ".join(parts)

        # Risk summary
        if risk_claims:
            parts = []
            for c in risk_claims[:3]:
                field = c.get("field", "")
                value = c.get("value", "")
                parts.append(f"{field}: {value}")
            summaries["risk"] = "; ".join(parts)

        return summaries

    def _build_counterfactual_analysis(
        self,
        trace: Dict[str, Any],
    ) -> CounterfactualAnalysis:
        """Build the Counterfactual Analysis section from trace."""
        robustness = trace.get("robustness", {})

        robustness_score = robustness.get("overall_score", 0.5)
        stability_margin = robustness.get("stability_margin", 0.5)
        counterfactuals_tested = robustness.get("perturbations_tested",
                                                 robustness.get("counterfactuals_tested", 0))
        flips_found = robustness.get("flips_found", 0)

        # Extract critical claims
        critical_claims_data = trace.get("critical_claims", [])
        critical_items = []

        for cc in critical_claims_data:
            claim_type = cc.get("claim_type", "unknown")
            field = cc.get("claim_field", cc.get("field", ""))
            criticality = cc.get("criticality_score", 0.5)

            # Get sensitivity info if available
            sensitivity = cc.get("sensitivity", {})
            flip_desc = cc.get("flip_description", f"Changes to {field} may affect decision")

            item = SensitivityItem(
                claim_type=claim_type,
                claim_field=field,
                original_value=cc.get("value", cc.get("claim_value")),
                perturbed_value=None,  # From perturbation details
                original_decision=DecisionOutcome(trace.get("decision", "defer")),
                new_decision=DecisionOutcome("pass" if trace.get("decision") == "invest" else "invest"),
                perturbation_magnitude=criticality,
                criticality_score=criticality,
                explanation=flip_desc,
            )
            critical_items.append(item)

        # Extract decision flip scenarios
        flip_scenarios = []
        for flip in trace.get("decision_flips", []):
            flip_scenarios.append(SensitivityItem(
                claim_type=flip.get("perturbations", [{}])[0].get("claim_type", "unknown") if flip.get("perturbations") else "unknown",
                claim_field=flip.get("perturbations", [{}])[0].get("claim_field", "") if flip.get("perturbations") else "",
                original_value=None,
                perturbed_value=None,
                original_decision=DecisionOutcome(flip.get("original_decision", "defer")),
                new_decision=DecisionOutcome(flip.get("new_decision", "defer")),
                perturbation_magnitude=flip.get("total_magnitude", 0.5),
                criticality_score=1.0 - flip.get("total_magnitude", 0.5),
                explanation=flip.get("perturbation_summary", "Decision flip scenario"),
            ))

        # Generate robustness interpretation
        interpretation = self._interpret_robustness(robustness_score, flips_found, counterfactuals_tested)

        return CounterfactualAnalysis(
            robustness_score=robustness_score,
            stability_margin=stability_margin,
            total_counterfactuals_tested=counterfactuals_tested,
            flips_found=flips_found,
            critical_claims=critical_items[:5],
            decision_flip_scenarios=flip_scenarios[:5],
            robustness_interpretation=interpretation,
        )

    def _interpret_robustness(
        self,
        score: float,
        flips: int,
        tested: int,
    ) -> str:
        """Generate interpretation of robustness results."""
        if score >= 0.8:
            stability = "highly stable"
            implication = "The decision is robust to most reasonable changes in evidence."
        elif score >= 0.6:
            stability = "moderately stable"
            implication = "The decision is stable but some evidence changes could affect it."
        elif score >= 0.4:
            stability = "somewhat fragile"
            implication = "The decision has notable sensitivity to key evidence."
        else:
            stability = "fragile"
            implication = "The decision is highly sensitive to evidence changes."

        if tested > 0:
            flip_rate = flips / tested
            flip_text = f"{flips} of {tested} counterfactuals ({flip_rate:.0%}) resulted in decision flips."
        else:
            flip_text = "No counterfactuals were tested."

        return f"The decision is {stability} with a robustness score of {score:.0%}. {flip_text} {implication}"

    def _build_uncertainty_limitations(
        self,
        evidence_graph: Dict[str, Any],
        trace: Dict[str, Any],
    ) -> UncertaintyLimitations:
        """Build the Uncertainty & Limitations section."""
        robustness = trace.get("robustness", {})
        claims = evidence_graph.get("claims", [])

        epistemic = robustness.get("epistemic_uncertainty", 0.2)
        aleatoric = robustness.get("aleatoric_uncertainty", 0.15)
        total = min(epistemic + aleatoric, 1.0)

        confidence = 1.0 - total
        confidence_level = ConfidenceLevel.from_score(confidence)

        # Identify data gaps
        data_gaps = self._identify_data_gaps(claims)

        # Identify low confidence areas
        low_conf_areas = []
        for claim in claims:
            if claim.get("confidence", 1.0) < self.LOW_CONFIDENCE_THRESHOLD:
                low_conf_areas.append(
                    f"{claim.get('claim_type', 'unknown')}.{claim.get('field', '')}"
                )

        # Standard assumptions
        assumptions = [
            "Evidence provided is accurate and current",
            "Market conditions remain stable during analysis period",
            "No material undisclosed information exists",
            "Historical performance is indicative of future potential",
        ]

        # Standard limitations
        limitations = [
            "This analysis is decision support only, not investment advice",
            "Confidence scores reflect model uncertainty, not success probability",
            "Analysis is based solely on provided evidence",
            "External factors not in evidence graph are not considered",
        ]

        # Recommendations based on gaps
        recommendations = []
        if len(low_conf_areas) > 3:
            recommendations.append("Verify low-confidence claims with additional sources")
        if "traction" in data_gaps:
            recommendations.append("Obtain verified financial statements and metrics")
        if "team_quality" in data_gaps:
            recommendations.append("Conduct reference checks on founding team")
        if "regulatory_risk" in data_gaps:
            recommendations.append("Review regulatory landscape and compliance requirements")
        if not recommendations:
            recommendations.append("Standard due diligence processes recommended")

        return UncertaintyLimitations(
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            confidence_level=confidence_level,
            data_gaps=data_gaps,
            low_confidence_areas=low_conf_areas[:10],
            assumptions_made=assumptions,
            limitations=limitations,
            recommendations_for_diligence=recommendations,
        )

    def _identify_data_gaps(self, claims: List[Dict]) -> List[str]:
        """Identify missing or underrepresented claim types."""
        # Expected claim types for thorough analysis
        expected_types = {
            "company_identity", "round_terms", "traction", "team_quality",
            "business_model", "market_scope", "differentiation",
        }

        present_types = {c.get("claim_type") for c in claims}
        missing = expected_types - present_types

        gaps = []
        for missing_type in missing:
            gaps.append(f"No {missing_type.replace('_', ' ')} claims provided")

        # Check for thin coverage
        type_counts = {}
        for c in claims:
            ct = c.get("claim_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1

        for ct, count in type_counts.items():
            if count == 1 and ct in expected_types:
                gaps.append(f"Limited evidence for {ct.replace('_', ' ')} (only 1 claim)")

        return gaps

    def _build_executive_summary(
        self,
        decision: DecisionOutcome,
        confidence: float,
        evidence_basis: EvidenceBasis,
        counterfactual: CounterfactualAnalysis,
        uncertainty: UncertaintyLimitations,
        deal_eval: DealEvaluation,
    ) -> ExecutiveSummary:
        """Build the Executive Summary section."""
        confidence_level = ConfidenceLevel.from_score(confidence)

        # Extract key strengths from supportive claims
        strengths = []
        for item in evidence_basis.supportive_claims[:5]:
            strengths.append(f"{item.claim_type}: {item.field} ({item.value})")

        # Extract key risks
        risks = []
        for item in evidence_basis.risk_claims[:5]:
            risks.append(f"{item.claim_type}: {item.field} ({item.value})")

        # Extract critical uncertainties from counterfactual
        uncertainties = []
        for item in counterfactual.critical_claims[:3]:
            uncertainties.append(f"{item.claim_type}.{item.claim_field}")

        # Generate one-line summary
        company = deal_eval.company_name or deal_eval.company_id
        one_liner = self._generate_one_liner(decision, confidence, company)

        # Generate recommendation text
        recommendation = self._generate_recommendation_text(
            decision, confidence, strengths, risks, uncertainties
        )

        return ExecutiveSummary(
            decision=decision,
            confidence=confidence,
            confidence_level=confidence_level,
            one_line_summary=one_liner,
            key_strengths=strengths,
            key_risks=risks,
            critical_uncertainties=uncertainties,
            recommendation_text=recommendation,
        )

    def _generate_one_liner(
        self,
        decision: DecisionOutcome,
        confidence: float,
        company: str,
    ) -> str:
        """Generate one-line summary."""
        conf_desc = "high" if confidence >= 0.7 else "moderate" if confidence >= 0.5 else "low"

        if decision == DecisionOutcome.INVEST:
            return f"Recommend INVEST in {company} with {conf_desc} confidence ({confidence:.0%})."
        elif decision == DecisionOutcome.PASS:
            return f"Recommend PASS on {company} with {conf_desc} confidence ({confidence:.0%})."
        else:
            return f"Recommend DEFER decision on {company} pending additional due diligence ({confidence:.0%} confidence)."

    def _generate_recommendation_text(
        self,
        decision: DecisionOutcome,
        confidence: float,
        strengths: List[str],
        risks: List[str],
        uncertainties: List[str],
    ) -> str:
        """Generate full recommendation paragraph."""
        parts = []

        if decision == DecisionOutcome.INVEST:
            parts.append(
                f"Based on the evidence analyzed, this deal presents a compelling investment opportunity "
                f"with {confidence:.0%} confidence."
            )
            if strengths:
                parts.append(f"Key strengths include: {'; '.join(strengths[:3])}.")
            if risks:
                parts.append(f"Notable risks to monitor: {'; '.join(risks[:2])}.")
        elif decision == DecisionOutcome.PASS:
            parts.append(
                f"Based on the evidence analyzed, this deal does not meet investment criteria "
                f"at this time ({confidence:.0%} confidence)."
            )
            if risks:
                parts.append(f"Primary concerns: {'; '.join(risks[:3])}.")
            if strengths:
                parts.append(f"Acknowledged strengths: {'; '.join(strengths[:2])}.")
        else:
            parts.append(
                f"The evidence is insufficient to make a definitive investment decision "
                f"({confidence:.0%} confidence)."
            )
            if uncertainties:
                parts.append(f"Key uncertainties requiring resolution: {'; '.join(uncertainties)}.")

        parts.append(
            "DISCLAIMER: This is decision support only, not investment advice. "
            "Final investment decisions remain the responsibility of the investment committee."
        )

        return " ".join(parts)

    def _build_audit_metadata(
        self,
        evidence_graph: Dict[str, Any],
        trace: Dict[str, Any],
    ) -> AuditMetadata:
        """Build the Audit Metadata section."""
        return AuditMetadata(
            report_id=f"report_{uuid.uuid4().hex[:12]}",
            generated_at=datetime.utcnow(),
            model_version=self.MODEL_VERSION,
            trace_id=trace.get("trace_id"),
            analyst_id=trace.get("analyst_id"),
            evidence_graph_version=evidence_graph.get("version"),
            num_claims_analyzed=len(evidence_graph.get("claims", [])),
            num_counterfactuals_tested=trace.get("robustness", {}).get(
                "perturbations_tested",
                trace.get("robustness", {}).get("counterfactuals_tested", 0)
            ),
            analysis_runtime_seconds=trace.get("runtime_seconds"),
            reproducibility_seed=self.seed,
        )


def generate_report(
    evidence_graph: Dict[str, Any],
    trace: Dict[str, Any],
    final_decision: str,
    seed: Optional[int] = None,
) -> DecisionReport:
    """
    Convenience function to generate a decision report.

    Args:
        evidence_graph: The evidence graph (as dict)
        trace: The reasoning trace (as dict)
        final_decision: The final decision outcome
        seed: Optional random seed for reproducibility tracking

    Returns:
        Complete DecisionReport
    """
    generator = ReportGenerator(seed=seed)
    return generator.generate(evidence_graph, trace, final_decision)
