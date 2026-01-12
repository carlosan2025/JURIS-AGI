"""Tests for VC counterfactual and robustness analysis."""

import pytest
from typing import Tuple

from juris_agi.evidence import (
    EvidenceGraph,
    Claim,
    ClaimType,
    Polarity,
)
from juris_agi.vc import (
    # Counterfactuals
    PerturbationType,
    ClaimPerturbation,
    CounterfactualEvidenceGraph,
    EvidenceCounterfactualGenerator,
    generate_counterfactuals,
    # Decision Analysis
    DecisionOutcome,
    DecisionCriticalClaim,
    DecisionRobustness,
    CounterfactualExplanation,
    DecisionAnalysisResult,
    DecisionAnalyzer,
    analyze_decision,
    # Trace
    VCDecisionTrace,
    VCDecisionTracer,
    create_decision_trace,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_graph():
    """Create a sample evidence graph for testing."""
    graph = EvidenceGraph(company_id="test-startup")

    # Add supportive claims
    graph.add_claim(Claim(
        ClaimType.TRACTION,
        "mrr",
        100000,
        confidence=0.9,
        polarity=Polarity.SUPPORTIVE,
        unit="USD",
    ))
    graph.add_claim(Claim(
        ClaimType.TEAM_QUALITY,
        "experience",
        "15 years",
        confidence=0.85,
        polarity=Polarity.SUPPORTIVE,
    ))
    graph.add_claim(Claim(
        ClaimType.MARKET_SCOPE,
        "tam",
        10000000000,
        confidence=0.7,
        polarity=Polarity.SUPPORTIVE,
        unit="USD",
    ))

    # Add risk claims
    graph.add_claim(Claim(
        ClaimType.EXECUTION_RISK,
        "key_hire",
        "CTO position unfilled",
        confidence=0.8,
        polarity=Polarity.RISK,
    ))
    graph.add_claim(Claim(
        ClaimType.CAPITAL_INTENSITY,
        "burn_rate",
        200000,
        confidence=0.9,
        polarity=Polarity.RISK,
        unit="USD/month",
    ))

    # Add neutral claims
    graph.add_claim(Claim(
        ClaimType.COMPANY_IDENTITY,
        "legal_name",
        "Test Startup Inc",
        confidence=1.0,
        polarity=Polarity.NEUTRAL,
    ))

    return graph


@pytest.fixture
def simple_decision_fn():
    """Create a simple decision function based on risk ratio."""
    def decision_fn(graph: EvidenceGraph) -> Tuple[DecisionOutcome, float]:
        if not graph.claims:
            return DecisionOutcome.DEFER, 0.5

        supportive = len(graph.get_supportive_claims())
        risk = len(graph.get_risk_claims())
        total = len(graph.claims)

        risk_ratio = risk / total if total > 0 else 0.5
        support_ratio = supportive / total if total > 0 else 0.5

        # Simple decision logic
        if support_ratio > 0.5 and risk_ratio < 0.3:
            return DecisionOutcome.INVEST, 0.8
        elif risk_ratio > 0.5:
            return DecisionOutcome.PASS, 0.7
        else:
            return DecisionOutcome.DEFER, 0.5

    return decision_fn


# =============================================================================
# Counterfactual Generation Tests
# =============================================================================


class TestCounterfactualGeneration:
    """Tests for counterfactual evidence graph generation."""

    def test_generate_counterfactuals(self, sample_graph):
        """Test basic counterfactual generation."""
        counterfactuals = generate_counterfactuals(sample_graph, num_counterfactuals=5, seed=42)

        assert len(counterfactuals) > 0
        assert len(counterfactuals) <= 5

        for cf in counterfactuals:
            assert cf.original_graph == sample_graph
            assert cf.modified_graph is not None
            assert len(cf.perturbations) > 0

    def test_counterfactual_has_perturbation(self, sample_graph):
        """Test that counterfactuals have tracked perturbations."""
        counterfactuals = generate_counterfactuals(sample_graph, num_counterfactuals=3, seed=42)

        for cf in counterfactuals:
            assert cf.num_perturbations > 0
            assert cf.total_perturbation_magnitude > 0

    def test_perturbation_types(self, sample_graph):
        """Test that different perturbation types are generated."""
        generator = EvidenceCounterfactualGenerator(seed=42)
        counterfactuals = generator.generate(sample_graph, num_counterfactuals=10)

        types_seen = {cf.perturbations[0].perturbation_type for cf in counterfactuals if cf.perturbations}

        # Should see multiple perturbation types
        assert len(types_seen) > 1

    def test_value_change_perturbation(self, sample_graph):
        """Test value change perturbation."""
        generator = EvidenceCounterfactualGenerator(seed=42)
        cf = generator._perturb_claim_value(sample_graph, 0, 0.5)

        assert cf is not None
        assert cf.perturbations[0].perturbation_type == PerturbationType.VALUE_CHANGE
        assert cf.perturbations[0].original_value != cf.perturbations[0].new_value

    def test_polarity_flip_perturbation(self, sample_graph):
        """Test polarity flip perturbation."""
        generator = EvidenceCounterfactualGenerator(seed=42)
        cf = generator._create_polarity_flip(sample_graph, 0)

        assert cf is not None
        assert cf.perturbations[0].perturbation_type == PerturbationType.POLARITY_FLIP

        original_polarity = sample_graph.claims[0].polarity
        modified_claim = cf.modified_graph.claims[0]
        assert modified_claim.polarity != original_polarity

    def test_confidence_change_perturbation(self, sample_graph):
        """Test confidence change perturbation."""
        generator = EvidenceCounterfactualGenerator(seed=42)
        cf = generator._create_confidence_change(sample_graph, 0, 0.3)

        assert cf is not None
        assert cf.perturbations[0].perturbation_type == PerturbationType.CONFIDENCE_CHANGE
        assert cf.modified_graph.claims[0].confidence == pytest.approx(0.3)

    def test_claim_removal_perturbation(self, sample_graph):
        """Test claim removal perturbation."""
        generator = EvidenceCounterfactualGenerator(seed=42)
        original_count = len(sample_graph.claims)

        cf = generator._create_claim_removal(sample_graph, 0)

        assert cf is not None
        assert cf.perturbations[0].perturbation_type == PerturbationType.CLAIM_REMOVAL
        assert len(cf.modified_graph.claims) == original_count - 1

    def test_perturbation_summary(self, sample_graph):
        """Test perturbation summary generation."""
        counterfactuals = generate_counterfactuals(sample_graph, num_counterfactuals=3, seed=42)

        for cf in counterfactuals:
            summary = cf.perturbation_summary
            assert isinstance(summary, str)
            assert len(summary) > 0

    def test_empty_graph_handling(self):
        """Test handling of empty evidence graph."""
        empty_graph = EvidenceGraph(company_id="empty")
        counterfactuals = generate_counterfactuals(empty_graph, num_counterfactuals=5)

        # Should handle gracefully (may return empty list)
        assert isinstance(counterfactuals, list)

    def test_minimal_flip_generation(self, sample_graph, simple_decision_fn):
        """Test minimal flip counterfactual generation."""
        generator = EvidenceCounterfactualGenerator(seed=42)

        cf = generator.generate_minimal_flip(
            sample_graph,
            lambda g: simple_decision_fn(g)[0] == DecisionOutcome.INVEST,
            max_attempts=50,
        )

        # May or may not find a flip depending on decision logic
        if cf is not None:
            assert cf.num_perturbations > 0


# =============================================================================
# Decision Analysis Tests
# =============================================================================


class TestDecisionAnalysis:
    """Tests for decision analysis with counterfactuals."""

    def test_analyze_decision(self, sample_graph, simple_decision_fn):
        """Test basic decision analysis."""
        result = analyze_decision(sample_graph, simple_decision_fn, seed=42)

        assert isinstance(result, DecisionAnalysisResult)
        assert result.decision in DecisionOutcome
        assert 0 <= result.confidence <= 1

    def test_critical_claims_identified(self, sample_graph, simple_decision_fn):
        """Test that critical claims are identified."""
        result = analyze_decision(sample_graph, simple_decision_fn, seed=42)

        # Should identify some critical claims
        assert isinstance(result.critical_claims, list)

        for cc in result.critical_claims:
            assert isinstance(cc, DecisionCriticalClaim)
            assert 0 <= cc.criticality_score <= 1

    def test_robustness_computed(self, sample_graph, simple_decision_fn):
        """Test that robustness is computed."""
        result = analyze_decision(sample_graph, simple_decision_fn, seed=42)

        assert result.robustness is not None
        assert isinstance(result.robustness, DecisionRobustness)
        assert 0 <= result.robustness.overall_score <= 1
        assert result.robustness.perturbations_tested > 0

    def test_counterfactual_explanations_generated(self, sample_graph, simple_decision_fn):
        """Test that counterfactual explanations are generated."""
        result = analyze_decision(sample_graph, simple_decision_fn, seed=42)

        assert isinstance(result.counterfactual_explanations, list)

        for exp in result.counterfactual_explanations:
            assert isinstance(exp, CounterfactualExplanation)
            assert isinstance(exp.explanation, str)
            assert len(exp.explanation) > 0
            assert isinstance(exp.key_changes, list)

    def test_analysis_result_to_dict(self, sample_graph, simple_decision_fn):
        """Test analysis result serialization."""
        result = analyze_decision(sample_graph, simple_decision_fn, seed=42)

        d = result.to_dict()

        assert "decision" in d
        assert "confidence" in d
        assert "critical_claims" in d
        assert "robustness" in d
        assert "counterfactual_explanations" in d

    def test_critical_claim_has_flip_description(self, sample_graph, simple_decision_fn):
        """Test that critical claims have flip descriptions."""
        result = analyze_decision(sample_graph, simple_decision_fn, seed=42)

        for cc in result.critical_claims:
            if cc.criticality_score > 0:
                assert cc.flip_description is not None
                assert isinstance(cc.flip_description, str)

    def test_robustness_by_claim_type(self, sample_graph, simple_decision_fn):
        """Test robustness breakdown by claim type."""
        result = analyze_decision(sample_graph, simple_decision_fn, seed=42)

        if result.robustness.robustness_by_claim_type:
            for ct, score in result.robustness.robustness_by_claim_type.items():
                assert isinstance(ct, str)
                assert 0 <= score <= 1


# =============================================================================
# Trace Tests
# =============================================================================


class TestVCDecisionTrace:
    """Tests for VC decision trace."""

    def test_create_decision_trace(self, sample_graph, simple_decision_fn):
        """Test creating a decision trace."""
        trace = create_decision_trace(
            sample_graph,
            simple_decision_fn,
            analyst_id="test-analyst",
            seed=42,
        )

        assert isinstance(trace, VCDecisionTrace)
        assert trace.company_id == "test-startup"
        assert trace.decision in DecisionOutcome
        assert trace.analyst_id == "test-analyst"

    def test_trace_has_entries(self, sample_graph, simple_decision_fn):
        """Test that trace has entries."""
        trace = create_decision_trace(sample_graph, simple_decision_fn, seed=42)

        assert len(trace.entries) > 0

        for entry in trace.entries:
            assert entry.timestamp is not None
            assert entry.entry_type is not None
            assert entry.content is not None

    def test_trace_has_critical_claims(self, sample_graph, simple_decision_fn):
        """Test that trace includes critical claims."""
        trace = create_decision_trace(sample_graph, simple_decision_fn, seed=42)

        assert isinstance(trace.critical_claims, list)

    def test_trace_has_robustness(self, sample_graph, simple_decision_fn):
        """Test that trace includes robustness analysis."""
        trace = create_decision_trace(sample_graph, simple_decision_fn, seed=42)

        assert trace.robustness is not None

    def test_trace_has_explanations(self, sample_graph, simple_decision_fn):
        """Test that trace includes counterfactual explanations."""
        trace = create_decision_trace(sample_graph, simple_decision_fn, seed=42)

        assert isinstance(trace.counterfactual_explanations, list)

    def test_trace_to_dict(self, sample_graph, simple_decision_fn):
        """Test trace serialization."""
        trace = create_decision_trace(sample_graph, simple_decision_fn, seed=42)

        d = trace.to_dict()

        assert "company_id" in d
        assert "decision" in d
        assert "confidence" in d
        assert "summary" in d
        assert "critical_claims" in d
        assert "robustness" in d
        assert "decision_flips" in d
        assert "trace_entries" in d

    def test_trace_to_json(self, sample_graph, simple_decision_fn):
        """Test trace JSON serialization."""
        trace = create_decision_trace(sample_graph, simple_decision_fn, seed=42)

        json_str = trace.to_json()

        assert isinstance(json_str, str)
        assert "test-startup" in json_str

    def test_trace_get_summary(self, sample_graph, simple_decision_fn):
        """Test trace summary generation."""
        trace = create_decision_trace(sample_graph, simple_decision_fn, seed=42)

        summary = trace.get_summary()

        assert "company_id" in summary
        assert "decision" in summary
        assert "confidence" in summary
        assert "robustness_score" in summary

    def test_trace_get_flip_summary(self, sample_graph, simple_decision_fn):
        """Test flip summary generation."""
        trace = create_decision_trace(sample_graph, simple_decision_fn, seed=42)

        flip_summary = trace.get_flip_summary()

        assert isinstance(flip_summary, list)
        for s in flip_summary:
            assert isinstance(s, str)
            assert "flips" in s.lower()

    def test_trace_from_analysis_result(self, sample_graph, simple_decision_fn):
        """Test creating trace from analysis result."""
        result = analyze_decision(sample_graph, simple_decision_fn, seed=42)

        trace = VCDecisionTrace.from_analysis_result(
            company_id="test-startup",
            result=result,
        )

        assert trace.decision == result.decision
        assert trace.confidence == result.confidence
        assert trace.critical_claims == result.critical_claims


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the complete VC decision pipeline."""

    def test_full_pipeline(self, sample_graph, simple_decision_fn):
        """Test complete analysis pipeline."""
        # Generate counterfactuals
        counterfactuals = generate_counterfactuals(sample_graph, num_counterfactuals=10, seed=42)
        assert len(counterfactuals) > 0

        # Analyze decision
        result = analyze_decision(sample_graph, simple_decision_fn, seed=42)
        assert result.decision in DecisionOutcome

        # Create trace
        trace = create_decision_trace(sample_graph, simple_decision_fn, seed=42)
        assert trace.company_id == sample_graph.company_id

        # Verify trace is complete
        trace_dict = trace.to_dict()
        assert all(key in trace_dict for key in [
            "decision", "confidence", "critical_claims",
            "robustness", "decision_flips", "trace_entries"
        ])

    def test_decision_sensitivity(self):
        """Test that decisions are sensitive to evidence changes."""
        # Create minimal graph
        graph = EvidenceGraph(company_id="sensitive-test")
        graph.add_claim(Claim(
            ClaimType.TRACTION,
            "mrr",
            100000,
            confidence=0.9,
            polarity=Polarity.SUPPORTIVE,
        ))

        # Decision based on MRR
        def mrr_decision(g):
            mrr_claims = [c for c in g.claims if c.field == "mrr"]
            if mrr_claims:
                mrr = mrr_claims[0].value
                if isinstance(mrr, (int, float)) and mrr > 50000:
                    return DecisionOutcome.INVEST, 0.8
            return DecisionOutcome.PASS, 0.7

        result = analyze_decision(graph, mrr_decision, seed=42)

        # The MRR claim should be identified as critical
        assert len(result.critical_claims) > 0

    def test_robustness_with_high_confidence(self):
        """Test robustness with high-confidence claims."""
        graph = EvidenceGraph(company_id="robust-test")

        # Add many high-confidence supportive claims
        for i in range(5):
            graph.add_claim(Claim(
                ClaimType.TRACTION,
                f"metric_{i}",
                10000 * (i + 1),
                confidence=0.95,
                polarity=Polarity.SUPPORTIVE,
            ))

        def majority_decision(g):
            supportive = len(g.get_supportive_claims())
            risk = len(g.get_risk_claims())
            if supportive > risk:
                return DecisionOutcome.INVEST, 0.9
            return DecisionOutcome.PASS, 0.7

        result = analyze_decision(graph, majority_decision, seed=42)

        # With many supportive claims, should be robust
        assert result.robustness.overall_score > 0.5

    def test_counterfactual_preserves_ontology(self, sample_graph):
        """Test that counterfactuals respect ontology constraints."""
        counterfactuals = generate_counterfactuals(sample_graph, num_counterfactuals=10, seed=42)

        for cf in counterfactuals:
            # All claims in modified graph should have valid claim types
            for claim in cf.modified_graph.claims:
                assert claim.claim_type in ClaimType
                assert 0 <= claim.confidence <= 1
                assert claim.polarity in Polarity

    def test_explanation_quality(self, sample_graph, simple_decision_fn):
        """Test that explanations are meaningful."""
        result = analyze_decision(sample_graph, simple_decision_fn, seed=42)

        for exp in result.counterfactual_explanations:
            # Explanation should mention the decision outcomes
            assert exp.original_decision.value in exp.explanation or "flips" in exp.explanation.lower()

            # Should have key changes
            assert len(exp.key_changes) > 0
