"""Tests for the Evidence Graph system for VC investment decisions."""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from juris_agi.evidence import (
    # Ontology
    ClaimType,
    get_claim_type,
    get_all_claim_types,
    get_risk_claim_types,
    get_claim_type_info,
    CLAIM_TYPE_METADATA,
    # Schema
    Polarity,
    Source,
    Claim,
    EvidenceGraph,
    ClaimSummary,
    # Loader
    load_evidence_graph,
    validate_evidence_graph,
    summarize_evidence_graph,
    EvidenceGraphLoader,
    ValidationIssue,
    # State conversion
    build_evidence_state,
    evidence_to_wme_features,
    evidence_to_cre_tokens,
    EvidenceStateBuilder,
)


# =============================================================================
# Ontology Tests
# =============================================================================


class TestOntology:
    """Tests for VC ontology."""

    def test_claim_type_count(self):
        """Test that we have exactly 15 claim types."""
        all_types = get_all_claim_types()
        assert len(all_types) == 15

    def test_all_claim_types_in_enum(self):
        """Test all expected claim types exist."""
        expected = [
            "company_identity", "round_terms", "use_of_proceeds",
            "team_quality", "team_composition",
            "product_readiness", "technical_moat", "differentiation",
            "market_scope", "business_model", "traction",
            "execution_risk", "regulatory_risk", "capital_intensity",
            "exit_logic",
        ]
        for name in expected:
            ct = get_claim_type(name)
            assert ct is not None, f"Missing claim type: {name}"

    def test_get_claim_type_case_insensitive(self):
        """Test claim type lookup is case insensitive."""
        assert get_claim_type("COMPANY_IDENTITY") == ClaimType.COMPANY_IDENTITY
        assert get_claim_type("Company_Identity") == ClaimType.COMPANY_IDENTITY
        assert get_claim_type("company_identity") == ClaimType.COMPANY_IDENTITY

    def test_get_claim_type_hyphen_support(self):
        """Test claim type lookup supports hyphens."""
        assert get_claim_type("company-identity") == ClaimType.COMPANY_IDENTITY
        assert get_claim_type("round-terms") == ClaimType.ROUND_TERMS

    def test_get_claim_type_unknown(self):
        """Test unknown claim type returns None."""
        assert get_claim_type("unknown_type") is None
        assert get_claim_type("") is None

    def test_risk_claim_types(self):
        """Test risk claim types are correctly identified."""
        risk_types = get_risk_claim_types()
        assert ClaimType.EXECUTION_RISK in risk_types
        assert ClaimType.REGULATORY_RISK in risk_types
        assert ClaimType.CAPITAL_INTENSITY in risk_types
        assert ClaimType.TEAM_QUALITY not in risk_types

    def test_claim_type_metadata(self):
        """Test all claim types have metadata."""
        for ct in get_all_claim_types():
            info = get_claim_type_info(ct)
            assert info is not None
            assert info.claim_type == ct
            assert len(info.description) > 0
            assert len(info.typical_fields) > 0


# =============================================================================
# Schema Tests
# =============================================================================


class TestSource:
    """Tests for Source dataclass."""

    def test_source_creation(self):
        """Test basic source creation."""
        source = Source(
            doc_id="pitch_deck_v1",
            locator="slide 5",
            quote="We have 100 paying customers",
            doc_type="pitch_deck",
        )
        assert source.doc_id == "pitch_deck_v1"
        assert source.locator == "slide 5"

    def test_source_to_dict(self):
        """Test source serialization."""
        source = Source(doc_id="doc1", locator="p1")
        d = source.to_dict()
        assert d["doc_id"] == "doc1"
        assert d["locator"] == "p1"

    def test_source_from_dict(self):
        """Test source deserialization."""
        data = {"doc_id": "doc1", "locator": "p1", "quote": "test"}
        source = Source.from_dict(data)
        assert source.doc_id == "doc1"
        assert source.quote == "test"


class TestClaim:
    """Tests for Claim dataclass."""

    def test_claim_creation(self):
        """Test basic claim creation."""
        claim = Claim(
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=50000,
            confidence=0.9,
            polarity=Polarity.SUPPORTIVE,
            unit="USD",
        )
        assert claim.claim_type == ClaimType.TRACTION
        assert claim.field == "mrr"
        assert claim.value == 50000
        assert claim.confidence == 0.9
        assert claim.unit == "USD"

    def test_claim_confidence_validation(self):
        """Test confidence must be 0-1."""
        with pytest.raises(ValueError):
            Claim(
                claim_type=ClaimType.TRACTION,
                field="mrr",
                value=1000,
                confidence=1.5,  # Invalid
            )

    def test_claim_is_high_confidence(self):
        """Test high confidence detection."""
        high = Claim(claim_type=ClaimType.TRACTION, field="mrr", value=1000, confidence=0.9)
        low = Claim(claim_type=ClaimType.TRACTION, field="mrr", value=1000, confidence=0.5)
        assert high.is_high_confidence is True
        assert low.is_high_confidence is False

    def test_claim_is_risk(self):
        """Test risk detection."""
        risk = Claim(
            claim_type=ClaimType.EXECUTION_RISK,
            field="hiring",
            value="critical role unfilled",
            polarity=Polarity.RISK,
        )
        neutral = Claim(
            claim_type=ClaimType.COMPANY_IDENTITY,
            field="name",
            value="Acme Corp",
        )
        assert risk.is_risk is True
        assert neutral.is_risk is False

    def test_claim_epistemic_uncertainty(self):
        """Test epistemic uncertainty calculation."""
        claim = Claim(
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=1000,
            confidence=0.7,
        )
        assert claim.epistemic_uncertainty == pytest.approx(0.3)

    def test_claim_to_dict(self):
        """Test claim serialization."""
        claim = Claim(
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=50000,
            confidence=0.9,
            polarity=Polarity.SUPPORTIVE,
        )
        d = claim.to_dict()
        assert d["claim_type"] == "traction"
        assert d["field"] == "mrr"
        assert d["value"] == 50000
        assert d["polarity"] == "supportive"

    def test_claim_from_dict(self):
        """Test claim deserialization."""
        data = {
            "claim_type": "traction",
            "field": "mrr",
            "value": 50000,
            "confidence": 0.9,
            "polarity": "supportive",
        }
        claim = Claim.from_dict(data)
        assert claim.claim_type == ClaimType.TRACTION
        assert claim.value == 50000


class TestEvidenceGraph:
    """Tests for EvidenceGraph dataclass."""

    def test_empty_graph(self):
        """Test empty evidence graph."""
        graph = EvidenceGraph(company_id="test-company")
        assert graph.claim_count == 0
        assert graph.coverage_ratio == 0.0
        assert graph.overall_epistemic_uncertainty == 1.0

    def test_add_claim(self):
        """Test adding claims."""
        graph = EvidenceGraph(company_id="test-company")
        claim = Claim(
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=50000,
        )
        graph.add_claim(claim)
        assert graph.claim_count == 1

    def test_get_claims_by_type(self):
        """Test filtering claims by type."""
        graph = EvidenceGraph(company_id="test")
        graph.add_claim(Claim(ClaimType.TRACTION, "mrr", 50000))
        graph.add_claim(Claim(ClaimType.TRACTION, "users", 1000))
        graph.add_claim(Claim(ClaimType.TEAM_QUALITY, "experience", "10 years"))

        traction_claims = graph.get_claims_by_type(ClaimType.TRACTION)
        assert len(traction_claims) == 2

    def test_get_risk_claims(self):
        """Test getting risk claims."""
        graph = EvidenceGraph(company_id="test")
        graph.add_claim(Claim(ClaimType.TRACTION, "mrr", 50000, polarity=Polarity.SUPPORTIVE))
        graph.add_claim(Claim(ClaimType.EXECUTION_RISK, "hiring", "gaps", polarity=Polarity.RISK))

        risk_claims = graph.get_risk_claims()
        assert len(risk_claims) == 1
        assert risk_claims[0].claim_type == ClaimType.EXECUTION_RISK

    def test_coverage_ratio(self):
        """Test coverage ratio calculation."""
        graph = EvidenceGraph(company_id="test")
        # Add claims for 3 of 15 types
        graph.add_claim(Claim(ClaimType.TRACTION, "mrr", 50000))
        graph.add_claim(Claim(ClaimType.TEAM_QUALITY, "exp", "good"))
        graph.add_claim(Claim(ClaimType.MARKET_SCOPE, "tam", "10B"))

        assert graph.coverage_ratio == pytest.approx(3 / 15)

    def test_missing_types(self):
        """Test missing types detection."""
        graph = EvidenceGraph(company_id="test")
        graph.add_claim(Claim(ClaimType.TRACTION, "mrr", 50000))

        missing = graph.missing_types
        assert len(missing) == 14  # 15 - 1
        assert ClaimType.TRACTION not in missing

    def test_to_dict(self):
        """Test graph serialization."""
        graph = EvidenceGraph(company_id="test")
        graph.add_claim(Claim(ClaimType.TRACTION, "mrr", 50000))

        d = graph.to_dict()
        assert d["company_id"] == "test"
        assert len(d["claims"]) == 1
        assert "metadata" in d
        assert d["metadata"]["claim_count"] == 1

    def test_from_dict(self):
        """Test graph deserialization."""
        data = {
            "company_id": "test",
            "claims": [
                {"claim_type": "traction", "field": "mrr", "value": 50000},
            ],
        }
        graph = EvidenceGraph.from_dict(data)
        assert graph.company_id == "test"
        assert graph.claim_count == 1


# =============================================================================
# Loader Tests
# =============================================================================


class TestLoader:
    """Tests for evidence graph loader."""

    def test_load_from_dict(self):
        """Test loading from dictionary."""
        data = {
            "company_id": "acme-corp",
            "claims": [
                {"claim_type": "traction", "field": "mrr", "value": 50000},
                {"claim_type": "team_quality", "field": "experience", "value": "10 years"},
            ],
        }
        result = load_evidence_graph(data)
        assert result.success is True
        assert result.evidence_graph.claim_count == 2

    def test_load_from_json_string(self):
        """Test loading from JSON string."""
        json_str = json.dumps({
            "company_id": "acme-corp",
            "claims": [
                {"claim_type": "traction", "field": "mrr", "value": 50000},
            ],
        })
        result = load_evidence_graph(json_str)
        assert result.success is True

    def test_load_from_file(self):
        """Test loading from file."""
        data = {
            "company_id": "acme-corp",
            "claims": [
                {"claim_type": "traction", "field": "mrr", "value": 50000},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = load_evidence_graph(Path(f.name))

        assert result.success is True

    def test_load_missing_company_id(self):
        """Test loading fails without company_id."""
        data = {"claims": []}
        result = load_evidence_graph(data)
        assert result.success is False
        assert result.has_errors is True

    def test_load_invalid_claim_type(self):
        """Test loading with invalid claim type fails."""
        data = {
            "company_id": "test",
            "claims": [
                {"claim_type": "invalid_type", "field": "test", "value": "test"},
            ],
        }
        result = load_evidence_graph(data)
        # Should fail due to invalid claim type (which is an error)
        assert result.success is False
        assert result.has_errors is True
        assert any("Unknown claim_type" in str(i) for i in result.errors)

    def test_load_missing_claim_types_warning(self):
        """Test warnings for missing claim types."""
        data = {
            "company_id": "test",
            "claims": [
                {"claim_type": "traction", "field": "mrr", "value": 50000},
            ],
        }
        result = load_evidence_graph(data)
        assert result.success is True
        # Should have warnings for missing types
        assert len(result.warnings) > 0

    def test_load_strict_mode(self):
        """Test strict mode treats warnings as errors."""
        data = {
            "company_id": "test",
            "claims": [
                {"claim_type": "traction", "field": "mrr", "value": 50000},
            ],
        }
        result = load_evidence_graph(data, strict=True)
        # Should fail due to missing types
        assert result.success is False

    def test_validate_existing_graph(self):
        """Test validation of existing graph."""
        graph = EvidenceGraph(company_id="test")
        graph.add_claim(Claim(ClaimType.TRACTION, "mrr", 50000))

        issues = validate_evidence_graph(graph)
        # Should have warnings for missing types
        assert len(issues) > 0

    def test_summarize_graph(self):
        """Test graph summarization."""
        graph = EvidenceGraph(company_id="test")
        graph.add_claim(Claim(ClaimType.TRACTION, "mrr", 50000, polarity=Polarity.SUPPORTIVE))
        graph.add_claim(Claim(ClaimType.EXECUTION_RISK, "hiring", "gaps", polarity=Polarity.RISK))

        summary = summarize_evidence_graph(graph)
        assert summary["company_id"] == "test"
        assert summary["total_claims"] == 2
        assert summary["risk_claims"] == 1
        assert summary["supportive_claims"] == 1


# =============================================================================
# State Conversion Tests
# =============================================================================


class TestStateConversion:
    """Tests for evidence to JURIS state conversion."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample evidence graph."""
        graph = EvidenceGraph(company_id="test-company")
        graph.add_claim(Claim(
            ClaimType.TRACTION,
            "mrr",
            50000,
            confidence=0.9,
            polarity=Polarity.SUPPORTIVE,
            unit="USD",
        ))
        graph.add_claim(Claim(
            ClaimType.TEAM_QUALITY,
            "experience",
            "10 years",
            confidence=0.8,
            polarity=Polarity.SUPPORTIVE,
        ))
        graph.add_claim(Claim(
            ClaimType.EXECUTION_RISK,
            "hiring",
            "CTO position unfilled",
            confidence=0.7,
            polarity=Polarity.RISK,
        ))
        return graph

    def test_build_evidence_state(self, sample_graph):
        """Test building evidence multi-view state."""
        state = build_evidence_state(sample_graph)

        assert state.company_id == "test-company"
        assert state.claim_count == 3
        assert len(state.symbolic_tokens) > 0
        assert state.features is not None

    def test_symbolic_tokens(self, sample_graph):
        """Test symbolic token generation."""
        state = build_evidence_state(sample_graph)
        tokens = state.symbolic_tokens

        # Should have tokens for each claim
        token_strs = [t.token for t in tokens]

        # Check claim type tokens
        assert "CT_TRACTION" in token_strs
        assert "CT_TEAM_QUALITY" in token_strs
        assert "CT_EXECUTION_RISK" in token_strs

        # Check polarity tokens
        assert "POL_SUPPORTIVE" in token_strs
        assert "POL_RISK" in token_strs

        # Check missing claim tokens
        assert "MISSING_CLAIM" in token_strs

    def test_evidence_features(self, sample_graph):
        """Test feature extraction."""
        state = build_evidence_state(sample_graph)
        features = state.features

        assert features.total_claims == 3
        assert 0 < features.coverage_ratio < 1
        assert 0 < features.average_confidence <= 1
        assert 0 < features.epistemic_uncertainty < 1

        # Check risk and support scores
        assert features.risk_score > 0
        assert features.support_score > 0

    def test_feature_vector(self, sample_graph):
        """Test feature vector generation."""
        state = build_evidence_state(sample_graph)
        vector = state.features.to_vector()

        assert len(vector) > 0
        assert vector.ndim == 1
        # Should be finite
        assert all(not (v != v) for v in vector)  # No NaNs

    def test_evidence_to_wme_features(self, sample_graph):
        """Test WME feature conversion."""
        features = evidence_to_wme_features(sample_graph)

        assert features["company_id"] == "test-company"
        assert features["claim_count"] == 3
        assert "feature_vector" in features
        assert len(features["feature_vector"]) > 0

    def test_evidence_to_cre_tokens(self, sample_graph):
        """Test CRE token conversion."""
        tokens = evidence_to_cre_tokens(sample_graph)

        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_empty_graph_handling(self):
        """Test handling of empty graph."""
        graph = EvidenceGraph(company_id="empty")
        state = build_evidence_state(graph)

        assert state.claim_count == 0
        assert state.features.epistemic_uncertainty == 1.0
        # Should have MISSING_CLAIM tokens
        assert any(t.token == "MISSING_CLAIM" for t in state.symbolic_tokens)

    def test_type_claims_mapping(self, sample_graph):
        """Test claims organized by type."""
        state = build_evidence_state(sample_graph)

        assert ClaimType.TRACTION in state.type_claims
        assert len(state.type_claims[ClaimType.TRACTION]) == 1

    def test_claim_map(self, sample_graph):
        """Test claim ID mapping."""
        state = build_evidence_state(sample_graph)

        # Should have 3 claims in map
        assert len(state.claim_map) == 3


class TestClaimSummary:
    """Tests for ClaimSummary."""

    def test_summary_from_claims(self):
        """Test summary generation."""
        claims = [
            Claim(ClaimType.TRACTION, "mrr", 50000, confidence=0.9, polarity=Polarity.SUPPORTIVE),
            Claim(ClaimType.TRACTION, "users", 1000, confidence=0.8, polarity=Polarity.NEUTRAL),
        ]
        summary = ClaimSummary.from_claims(ClaimType.TRACTION, claims)

        assert summary.count == 2
        assert summary.avg_confidence == pytest.approx(0.85)
        assert summary.supportive_count == 1
        assert summary.neutral_count == 1
        assert "mrr" in summary.fields
        assert "users" in summary.fields

    def test_empty_summary(self):
        """Test summary with no claims."""
        summary = ClaimSummary.from_claims(ClaimType.TRACTION, [])
        assert summary.count == 0
        assert summary.avg_confidence == 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full evidence graph pipeline."""

    def test_full_pipeline(self):
        """Test complete pipeline: load -> validate -> convert."""
        # Create sample evidence
        data = {
            "company_id": "startup-xyz",
            "analyst_id": "analyst-1",
            "claims": [
                {
                    "claim_type": "company_identity",
                    "field": "legal_name",
                    "value": "Startup XYZ Inc",
                    "confidence": 1.0,
                    "source": {"doc_id": "incorporation_docs"},
                },
                {
                    "claim_type": "round_terms",
                    "field": "raise_amount",
                    "value": 5000000,
                    "unit": "USD",
                    "confidence": 0.95,
                    "polarity": "neutral",
                },
                {
                    "claim_type": "traction",
                    "field": "mrr",
                    "value": 100000,
                    "unit": "USD",
                    "confidence": 0.9,
                    "polarity": "supportive",
                    "source": {
                        "doc_id": "financial_statements",
                        "locator": "page 3",
                        "quote": "Monthly recurring revenue: $100,000",
                    },
                },
                {
                    "claim_type": "execution_risk",
                    "field": "key_hire_gap",
                    "value": "No CTO",
                    "confidence": 0.85,
                    "polarity": "risk",
                    "notes": "Critical role for technical roadmap",
                },
            ],
        }

        # Load
        result = load_evidence_graph(data)
        assert result.success is True
        graph = result.evidence_graph

        # Validate
        issues = validate_evidence_graph(graph)
        # Should have warnings for missing types
        assert len(issues) > 0

        # Get summary
        summary = summarize_evidence_graph(graph)
        assert summary["total_claims"] == 4

        # Convert to JURIS state
        state = build_evidence_state(graph)
        assert state.claim_count == 4
        assert len(state.symbolic_tokens) > 0

        # Get WME features
        wme_features = evidence_to_wme_features(graph)
        assert wme_features["claim_count"] == 4

        # Get CRE tokens
        tokens = evidence_to_cre_tokens(graph)
        assert len(tokens) > 0

    def test_missing_data_increases_uncertainty(self):
        """Test that missing claims increase epistemic uncertainty."""
        # Full coverage graph (unrealistic but for testing)
        full_graph = EvidenceGraph(company_id="full")
        for ct in get_all_claim_types():
            full_graph.add_claim(Claim(ct, "test_field", "test_value", confidence=1.0))

        # Partial coverage graph
        partial_graph = EvidenceGraph(company_id="partial")
        partial_graph.add_claim(Claim(ClaimType.TRACTION, "mrr", 50000, confidence=1.0))

        # Compare uncertainties
        full_state = build_evidence_state(full_graph)
        partial_state = build_evidence_state(partial_graph)

        assert partial_state.epistemic_uncertainty > full_state.epistemic_uncertainty

    def test_variable_claim_count(self):
        """Test handling of variable number of claims."""
        # One claim
        graph1 = EvidenceGraph(company_id="test1")
        graph1.add_claim(Claim(ClaimType.TRACTION, "mrr", 50000))

        # Many claims
        graph10 = EvidenceGraph(company_id="test10")
        for i in range(10):
            graph10.add_claim(Claim(ClaimType.TRACTION, f"metric_{i}", i * 1000))

        # Both should work
        state1 = build_evidence_state(graph1)
        state10 = build_evidence_state(graph10)

        assert state1.claim_count == 1
        assert state10.claim_count == 10
