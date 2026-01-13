"""
Tests for LLM-based document extractors.

Tests extraction functionality without requiring actual LLM calls.
"""

import json
import pytest
from datetime import datetime

from juris_agi.evidence.extractors import (
    DocumentExtractor,
    ExtractionResult,
    ProposedClaim,
    ExtractionConfig,
    ExtractionStatus,
    PitchDeckExtractor,
    FinancialModelExtractor,
    TechDescriptionExtractor,
    ICMemoExtractor,
    ExtractorRegistry,
    get_extractor,
    extract_from_document,
)
from juris_agi.evidence.extractors.registry import (
    detect_document_type,
    create_mock_llm_fn,
)
from juris_agi.evidence.ontology import ClaimType
from juris_agi.evidence.schema import Polarity, Claim


# =============================================================================
# ProposedClaim Tests
# =============================================================================

class TestProposedClaim:
    """Tests for ProposedClaim dataclass."""

    def test_create_proposed_claim(self):
        """Test basic creation of a proposed claim."""
        from juris_agi.evidence.schema import Source

        source = Source(
            doc_id="test_doc",
            locator="Page 5",
            quote="$100K MRR",
            doc_type="pitch_deck",
        )

        claim = ProposedClaim(
            proposal_id="prop_123",
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=100000,
            confidence=0.85,
            polarity=Polarity.SUPPORTIVE,
            source=source,
            rationale="Clear revenue metric from traction slide",
        )

        assert claim.proposal_id == "prop_123"
        assert claim.claim_type == ClaimType.TRACTION
        assert claim.field == "mrr"
        assert claim.value == 100000
        assert claim.confidence == 0.85
        assert claim.polarity == Polarity.SUPPORTIVE
        assert claim.status == ExtractionStatus.PENDING
        assert claim.rationale == "Clear revenue metric from traction slide"

    def test_approve_claim(self):
        """Test approving a proposed claim."""
        claim = ProposedClaim(
            proposal_id="prop_123",
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=100000,
            confidence=0.85,
            polarity=Polarity.SUPPORTIVE,
            source=None,
            rationale="Test",
        )

        claim.approve(reviewer_notes="Looks correct")

        assert claim.status == ExtractionStatus.APPROVED
        assert claim.reviewer_notes == "Looks correct"
        assert claim.reviewed_at is not None

    def test_reject_claim(self):
        """Test rejecting a proposed claim."""
        claim = ProposedClaim(
            proposal_id="prop_123",
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=100000,
            confidence=0.85,
            polarity=Polarity.SUPPORTIVE,
            source=None,
            rationale="Test",
        )

        claim.reject(reviewer_notes="Value seems incorrect")

        assert claim.status == ExtractionStatus.REJECTED
        assert claim.reviewer_notes == "Value seems incorrect"

    def test_modify_claim(self):
        """Test modifying a proposed claim."""
        claim = ProposedClaim(
            proposal_id="prop_123",
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=100000,
            confidence=0.85,
            polarity=Polarity.SUPPORTIVE,
            source=None,
            rationale="Test",
        )

        claim.modify(
            value=150000,
            confidence=0.90,
            polarity=Polarity.NEUTRAL,
            reviewer_notes="Corrected value",
        )

        assert claim.status == ExtractionStatus.MODIFIED
        assert claim.modified_value == 150000
        assert claim.modified_confidence == 0.90
        assert claim.modified_polarity == Polarity.NEUTRAL

    def test_to_claim_approved(self):
        """Test converting approved proposal to Claim."""
        from juris_agi.evidence.schema import Source

        source = Source(doc_id="test_doc", locator="Page 5")

        proposal = ProposedClaim(
            proposal_id="prop_123",
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=100000,
            confidence=0.85,
            polarity=Polarity.SUPPORTIVE,
            source=source,
            rationale="Test rationale",
        )

        proposal.approve()
        claim = proposal.to_claim()

        assert isinstance(claim, Claim)
        assert claim.claim_type == ClaimType.TRACTION
        assert claim.field == "mrr"
        assert claim.value == 100000
        assert claim.confidence == 0.85
        assert claim.polarity == Polarity.SUPPORTIVE
        assert claim.claim_id == "prop_123"

    def test_to_claim_modified(self):
        """Test converting modified proposal uses modified values."""
        proposal = ProposedClaim(
            proposal_id="prop_123",
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=100000,
            confidence=0.85,
            polarity=Polarity.SUPPORTIVE,
            source=None,
            rationale="Test",
        )

        proposal.modify(value=150000, confidence=0.95)
        claim = proposal.to_claim()

        assert claim.value == 150000
        assert claim.confidence == 0.95
        # Polarity should remain original since not modified
        assert claim.polarity == Polarity.SUPPORTIVE

    def test_to_claim_pending_fails(self):
        """Test that pending proposals cannot be converted to claims."""
        proposal = ProposedClaim(
            proposal_id="prop_123",
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=100000,
            confidence=0.85,
            polarity=Polarity.SUPPORTIVE,
            source=None,
            rationale="Test",
        )

        with pytest.raises(ValueError, match="pending"):
            proposal.to_claim()

    def test_to_claim_rejected_fails(self):
        """Test that rejected proposals cannot be converted to claims."""
        proposal = ProposedClaim(
            proposal_id="prop_123",
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=100000,
            confidence=0.85,
            polarity=Polarity.SUPPORTIVE,
            source=None,
            rationale="Test",
        )

        proposal.reject()

        with pytest.raises(ValueError, match="rejected"):
            proposal.to_claim()

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        from juris_agi.evidence.schema import Source

        source = Source(doc_id="test_doc", locator="Page 5")

        original = ProposedClaim(
            proposal_id="prop_123",
            claim_type=ClaimType.TRACTION,
            field="mrr",
            value=100000,
            confidence=0.85,
            polarity=Polarity.SUPPORTIVE,
            source=source,
            rationale="Test rationale",
        )

        # Serialize
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data["proposal_id"] == "prop_123"
        assert data["claim_type"] == "traction"

        # Deserialize
        restored = ProposedClaim.from_dict(data)
        assert restored.proposal_id == original.proposal_id
        assert restored.claim_type == original.claim_type
        assert restored.value == original.value


# =============================================================================
# ExtractionConfig Tests
# =============================================================================

class TestExtractionConfig:
    """Tests for ExtractionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExtractionConfig()

        assert config.enabled is True
        assert config.model == "gpt-4o"
        assert config.temperature == 0.1
        assert config.max_claims_per_type == 10
        assert config.min_confidence == 0.3

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExtractionConfig(
            enabled=False,
            model="gpt-4-turbo",
            temperature=0.2,
            min_confidence=0.5,
        )

        assert config.enabled is False
        assert config.model == "gpt-4-turbo"
        assert config.temperature == 0.2
        assert config.min_confidence == 0.5

    def test_disabled_config(self):
        """Test that disabled config works without LLM."""
        config = ExtractionConfig(enabled=False)

        extractor = PitchDeckExtractor(config=config, llm_fn=None)
        assert extractor.enabled is False


# =============================================================================
# DocumentExtractor Tests
# =============================================================================

class TestDocumentExtractor:
    """Tests for base DocumentExtractor functionality."""

    def test_extractor_without_llm_is_disabled(self):
        """Test that extractor without LLM function is disabled."""
        extractor = PitchDeckExtractor(llm_fn=None)
        assert extractor.enabled is False

    def test_extractor_with_llm_is_enabled(self):
        """Test that extractor with LLM function is enabled."""
        llm_fn = create_mock_llm_fn()
        extractor = PitchDeckExtractor(llm_fn=llm_fn)
        assert extractor.enabled is True

    def test_extract_disabled_returns_error(self):
        """Test that extraction when disabled returns error result."""
        extractor = PitchDeckExtractor(llm_fn=None)
        result = extractor.extract("Some document content", "doc_123")

        assert isinstance(result, ExtractionResult)
        assert result.success is False
        assert len(result.errors) > 0
        assert "disabled" in result.errors[0].lower()

    def test_system_prompt_generation(self):
        """Test that system prompt is generated correctly."""
        extractor = PitchDeckExtractor()
        prompt = extractor.get_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "claim" in prompt.lower()
        assert "confidence" in prompt.lower()

    def test_extraction_prompt_generation(self):
        """Test that extraction prompt includes document content."""
        extractor = PitchDeckExtractor()
        prompt = extractor.get_extraction_prompt("Test content here", "doc_123")

        assert "Test content here" in prompt
        assert "doc_123" in prompt

    def test_parse_llm_response_valid_json(self):
        """Test parsing valid LLM JSON response."""
        extractor = PitchDeckExtractor()

        response = json.dumps([
            {
                "claim_type": "traction",
                "field": "mrr",
                "value": 100000,
                "confidence": 0.85,
                "polarity": "supportive",
                "locator": "Page 5",
                "rationale": "Clear metric",
            }
        ])

        proposals = extractor.parse_llm_response(response, "doc_123")

        assert len(proposals) == 1
        assert proposals[0].claim_type == ClaimType.TRACTION
        assert proposals[0].value == 100000

    def test_parse_llm_response_with_text_around_json(self):
        """Test parsing response with text around JSON."""
        extractor = PitchDeckExtractor()

        response = """Here are the extracted claims:
        [
            {
                "claim_type": "company_identity",
                "field": "legal_name",
                "value": "Test Corp",
                "confidence": 0.95,
                "polarity": "neutral",
                "rationale": "Title slide"
            }
        ]
        Note: Only one claim found."""

        proposals = extractor.parse_llm_response(response, "doc_123")

        assert len(proposals) == 1
        assert proposals[0].claim_type == ClaimType.COMPANY_IDENTITY
        assert proposals[0].value == "Test Corp"

    def test_parse_llm_response_filters_low_confidence(self):
        """Test that low confidence claims are filtered out."""
        config = ExtractionConfig(min_confidence=0.5)
        extractor = PitchDeckExtractor(config=config)

        response = json.dumps([
            {
                "claim_type": "traction",
                "field": "mrr",
                "value": 100000,
                "confidence": 0.3,  # Below threshold
                "polarity": "supportive",
                "rationale": "Unclear",
            },
            {
                "claim_type": "traction",
                "field": "arr",
                "value": 1200000,
                "confidence": 0.8,  # Above threshold
                "polarity": "supportive",
                "rationale": "Clear",
            },
        ])

        proposals = extractor.parse_llm_response(response, "doc_123")

        assert len(proposals) == 1
        assert proposals[0].field == "arr"

    def test_chunk_document_small(self):
        """Test that small documents are not chunked."""
        extractor = PitchDeckExtractor()
        text = "Short document content"

        chunks = extractor.chunk_document(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_document_large(self):
        """Test that large documents are chunked with overlap."""
        config = ExtractionConfig(chunk_size=100, overlap=20)
        extractor = PitchDeckExtractor(config=config)
        text = "x" * 250

        chunks = extractor.chunk_document(text)

        assert len(chunks) > 1
        # Check overlap exists
        assert len(chunks[0]) >= config.chunk_size - config.overlap


# =============================================================================
# Specific Extractor Tests
# =============================================================================

class TestPitchDeckExtractor:
    """Tests for PitchDeckExtractor."""

    def test_supported_claim_types(self):
        """Test that pitch deck extractor supports correct claim types."""
        extractor = PitchDeckExtractor()

        assert ClaimType.COMPANY_IDENTITY in extractor.SUPPORTED_CLAIM_TYPES
        assert ClaimType.TRACTION in extractor.SUPPORTED_CLAIM_TYPES
        assert ClaimType.TEAM_QUALITY in extractor.SUPPORTED_CLAIM_TYPES
        assert ClaimType.MARKET_SCOPE in extractor.SUPPORTED_CLAIM_TYPES
        # Should not include risk-specific types
        assert ClaimType.REGULATORY_RISK not in extractor.SUPPORTED_CLAIM_TYPES

    def test_doc_type(self):
        """Test document type identifier."""
        extractor = PitchDeckExtractor()
        assert extractor.DOC_TYPE == "pitch_deck"

    def test_extract_with_mock_llm(self):
        """Test full extraction with mock LLM."""
        llm_fn = create_mock_llm_fn()
        extractor = PitchDeckExtractor(llm_fn=llm_fn)

        result = extractor.extract("Sample pitch deck content", "pitch_v1")

        assert isinstance(result, ExtractionResult)
        assert result.doc_id == "pitch_v1"
        assert result.doc_type == "pitch_deck"
        assert result.success is True
        assert len(result.proposed_claims) > 0


class TestFinancialModelExtractor:
    """Tests for FinancialModelExtractor."""

    def test_supported_claim_types(self):
        """Test that financial model extractor supports correct claim types."""
        extractor = FinancialModelExtractor()

        assert ClaimType.TRACTION in extractor.SUPPORTED_CLAIM_TYPES
        assert ClaimType.CAPITAL_INTENSITY in extractor.SUPPORTED_CLAIM_TYPES
        assert ClaimType.BUSINESS_MODEL in extractor.SUPPORTED_CLAIM_TYPES
        # Should not include team types
        assert ClaimType.TEAM_QUALITY not in extractor.SUPPORTED_CLAIM_TYPES

    def test_doc_type(self):
        """Test document type identifier."""
        extractor = FinancialModelExtractor()
        assert extractor.DOC_TYPE == "financial_model"


class TestTechDescriptionExtractor:
    """Tests for TechDescriptionExtractor."""

    def test_supported_claim_types(self):
        """Test that tech extractor supports correct claim types."""
        extractor = TechDescriptionExtractor()

        assert ClaimType.PRODUCT_READINESS in extractor.SUPPORTED_CLAIM_TYPES
        assert ClaimType.TECHNICAL_MOAT in extractor.SUPPORTED_CLAIM_TYPES
        assert ClaimType.EXECUTION_RISK in extractor.SUPPORTED_CLAIM_TYPES
        # Should not include market types
        assert ClaimType.MARKET_SCOPE not in extractor.SUPPORTED_CLAIM_TYPES

    def test_doc_type(self):
        """Test document type identifier."""
        extractor = TechDescriptionExtractor()
        assert extractor.DOC_TYPE == "tech_description"


class TestICMemoExtractor:
    """Tests for ICMemoExtractor."""

    def test_supported_claim_types(self):
        """Test that IC memo extractor supports all claim types."""
        extractor = ICMemoExtractor()

        # IC memos should support all types
        assert ClaimType.COMPANY_IDENTITY in extractor.SUPPORTED_CLAIM_TYPES
        assert ClaimType.ROUND_TERMS in extractor.SUPPORTED_CLAIM_TYPES
        assert ClaimType.TRACTION in extractor.SUPPORTED_CLAIM_TYPES
        assert ClaimType.EXECUTION_RISK in extractor.SUPPORTED_CLAIM_TYPES
        assert ClaimType.EXIT_LOGIC in extractor.SUPPORTED_CLAIM_TYPES

    def test_doc_type(self):
        """Test document type identifier."""
        extractor = ICMemoExtractor()
        assert extractor.DOC_TYPE == "ic_memo"


# =============================================================================
# ExtractorRegistry Tests
# =============================================================================

class TestExtractorRegistry:
    """Tests for ExtractorRegistry."""

    def test_list_supported_types(self):
        """Test listing supported document types."""
        types = ExtractorRegistry.list_supported_types()

        assert "pitch_deck" in types
        assert "financial_model" in types
        assert "tech_description" in types
        assert "ic_memo" in types

    def test_get_extractor_class_by_type(self):
        """Test getting extractor class by document type."""
        cls = ExtractorRegistry.get_extractor_class("pitch_deck")
        assert cls is not None

        cls = ExtractorRegistry.get_extractor_class("financial_model")
        assert cls is not None

    def test_get_extractor_class_by_alias(self):
        """Test getting extractor class by alias."""
        cls = ExtractorRegistry.get_extractor_class("deck")
        assert cls is not None

        cls = ExtractorRegistry.get_extractor_class("financials")
        assert cls is not None

    def test_get_extractor_class_unknown_returns_none(self):
        """Test that unknown type returns None."""
        cls = ExtractorRegistry.get_extractor_class("unknown_type")
        assert cls is None

    def test_create_extractor(self):
        """Test creating extractor instance."""
        extractor = ExtractorRegistry.create_extractor("pitch_deck")
        assert extractor is not None
        assert isinstance(extractor, PitchDeckExtractor)

    def test_create_extractor_with_config(self):
        """Test creating extractor with custom config."""
        config = ExtractionConfig(min_confidence=0.7)
        extractor = ExtractorRegistry.create_extractor("pitch_deck", config=config)

        assert extractor is not None
        assert extractor.config.min_confidence == 0.7

    def test_get_type_aliases(self):
        """Test getting type aliases."""
        aliases = ExtractorRegistry.get_type_aliases()

        assert "pitch_deck" in aliases
        assert "deck" in aliases["pitch_deck"]


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_extractor(self):
        """Test get_extractor function."""
        extractor = get_extractor("pitch_deck")
        assert extractor is not None
        assert isinstance(extractor, DocumentExtractor)

    def test_extract_from_document(self):
        """Test extract_from_document function."""
        llm_fn = create_mock_llm_fn()
        config = ExtractionConfig()

        result = extract_from_document(
            document_text="Test content",
            doc_id="test_doc",
            doc_type="pitch_deck",
            config=config,
            llm_fn=llm_fn,
        )

        assert isinstance(result, ExtractionResult)
        assert result.doc_id == "test_doc"

    def test_extract_from_document_unknown_type(self):
        """Test extraction with unknown document type."""
        result = extract_from_document(
            document_text="Test content",
            doc_id="test_doc",
            doc_type="unknown_type",
        )

        assert result.success is False
        assert "unknown" in result.errors[0].lower()


class TestDetectDocumentType:
    """Tests for document type detection."""

    def test_detect_from_filename_pitch(self):
        """Test detecting pitch deck from filename."""
        assert detect_document_type("company_pitch_v2.pdf") == "pitch_deck"
        assert detect_document_type("investor_deck.pptx") == "pitch_deck"

    def test_detect_from_filename_financial(self):
        """Test detecting financial model from filename."""
        assert detect_document_type("financial_model.xlsx") == "financial_model"
        assert detect_document_type("budget_projections.xlsx") == "financial_model"

    def test_detect_from_filename_tech(self):
        """Test detecting tech doc from filename."""
        assert detect_document_type("technical_architecture.md") == "tech_description"
        assert detect_document_type("engineering_overview.pdf") == "tech_description"

    def test_detect_from_filename_ic_memo(self):
        """Test detecting IC memo from filename."""
        assert detect_document_type("ic_memo_acme.docx") == "ic_memo"
        assert detect_document_type("investment_recommendation.pdf") == "ic_memo"

    def test_detect_from_content_financial(self):
        """Test detecting from content."""
        content = "Monthly burn rate: $50K. Runway: 18 months. LTV/CAC ratio: 3.2"
        assert detect_document_type("document.txt", content) == "financial_model"

    def test_detect_from_content_tech(self):
        """Test detecting tech doc from content."""
        content = "Our microservices architecture uses Kubernetes for deployment."
        assert detect_document_type("document.txt", content) == "tech_description"

    def test_detect_from_content_ic_memo(self):
        """Test detecting IC memo from content."""
        content = "Investment Committee Memo\n\nRecommendation: Invest\n\nKey Risks:"
        assert detect_document_type("document.txt", content) == "ic_memo"

    def test_detect_default_to_pitch_deck(self):
        """Test default detection is pitch deck."""
        assert detect_document_type("random_file.txt") == "pitch_deck"


# =============================================================================
# ExtractionResult Tests
# =============================================================================

class TestExtractionResult:
    """Tests for ExtractionResult."""

    def test_empty_result(self):
        """Test empty extraction result."""
        result = ExtractionResult(
            doc_id="test",
            doc_type="pitch_deck",
        )

        assert result.success is True
        assert result.pending_count == 0
        assert result.approved_count == 0
        assert result.rejected_count == 0

    def test_result_with_claims(self):
        """Test result with proposed claims."""
        claims = [
            ProposedClaim(
                proposal_id="p1",
                claim_type=ClaimType.TRACTION,
                field="mrr",
                value=100000,
                confidence=0.85,
                polarity=Polarity.SUPPORTIVE,
                source=None,
                rationale="Test",
            ),
            ProposedClaim(
                proposal_id="p2",
                claim_type=ClaimType.TEAM_QUALITY,
                field="experience",
                value="10 years",
                confidence=0.75,
                polarity=Polarity.SUPPORTIVE,
                source=None,
                rationale="Test",
            ),
        ]

        result = ExtractionResult(
            doc_id="test",
            doc_type="pitch_deck",
            proposed_claims=claims,
        )

        assert result.pending_count == 2
        assert result.approved_count == 0

        # Approve one claim
        claims[0].approve()

        assert result.pending_count == 1
        assert result.approved_count == 1

    def test_get_approved_claims(self):
        """Test getting approved claims as Claim objects."""
        claims = [
            ProposedClaim(
                proposal_id="p1",
                claim_type=ClaimType.TRACTION,
                field="mrr",
                value=100000,
                confidence=0.85,
                polarity=Polarity.SUPPORTIVE,
                source=None,
                rationale="Test",
            ),
            ProposedClaim(
                proposal_id="p2",
                claim_type=ClaimType.TEAM_QUALITY,
                field="experience",
                value="10 years",
                confidence=0.75,
                polarity=Polarity.SUPPORTIVE,
                source=None,
                rationale="Test",
            ),
        ]

        claims[0].approve()
        claims[1].reject()

        result = ExtractionResult(
            doc_id="test",
            doc_type="pitch_deck",
            proposed_claims=claims,
        )

        approved = result.get_approved_claims()
        assert len(approved) == 1
        assert isinstance(approved[0], Claim)
        assert approved[0].field == "mrr"

    def test_serialization(self):
        """Test result serialization."""
        result = ExtractionResult(
            doc_id="test",
            doc_type="pitch_deck",
            extraction_time_seconds=1.5,
            model_used="gpt-4o",
        )

        data = result.to_dict()

        assert data["doc_id"] == "test"
        assert data["doc_type"] == "pitch_deck"
        assert data["extraction_time_seconds"] == 1.5
        assert "summary" in data
        assert data["summary"]["total"] == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestExtractionWorkflow:
    """Integration tests for complete extraction workflow."""

    def test_full_extraction_workflow(self):
        """Test complete extraction → review → merge workflow."""
        # 1. Extract
        llm_fn = create_mock_llm_fn()
        extractor = PitchDeckExtractor(llm_fn=llm_fn)

        result = extractor.extract(
            "Sample pitch deck with $100K MRR and strong team",
            "pitch_v1",
        )

        assert result.success
        assert len(result.proposed_claims) > 0

        # 2. Review
        for claim in result.proposed_claims:
            # Approve claims with high confidence
            if claim.confidence >= 0.8:
                claim.approve(reviewer_notes="Verified")
            elif claim.confidence >= 0.5:
                # Modify medium confidence claims
                claim.modify(
                    confidence=claim.confidence + 0.1,
                    reviewer_notes="Adjusted confidence",
                )
            else:
                claim.reject(reviewer_notes="Too uncertain")

        # 3. Get approved claims
        approved = result.get_approved_claims()

        # Should have at least one approved claim
        assert len(approved) > 0

        # All approved claims should be Claim instances
        for claim in approved:
            assert isinstance(claim, Claim)
            assert claim.confidence > 0
            assert claim.claim_type in ClaimType

    def test_extraction_preserves_source(self):
        """Test that extraction preserves source information."""
        # Create LLM function that returns claims with source info
        def llm_with_source(system: str, user: str) -> str:
            return json.dumps([{
                "claim_type": "traction",
                "field": "mrr",
                "value": 100000,
                "confidence": 0.85,
                "polarity": "supportive",
                "locator": "Page 8, Traction slide",
                "quote": "$100K MRR as of Q4",
                "rationale": "Clear metric",
            }])

        extractor = PitchDeckExtractor(llm_fn=llm_with_source)
        result = extractor.extract("Content", "doc_123")

        assert len(result.proposed_claims) == 1
        claim = result.proposed_claims[0]

        assert claim.source is not None
        assert claim.source.doc_id == "doc_123"
        assert claim.source.locator == "Page 8, Traction slide"
        assert claim.source.quote == "$100K MRR as of Q4"
        assert claim.source.doc_type == "pitch_deck"

    def test_works_without_llm(self):
        """Test that system works fully without LLM extraction enabled."""
        # Create extractor without LLM
        config = ExtractionConfig(enabled=False)
        extractor = PitchDeckExtractor(config=config, llm_fn=None)

        # Should not crash
        result = extractor.extract("Content", "doc_123")

        # Should return result with error
        assert not result.success
        assert len(result.errors) > 0
        assert len(result.proposed_claims) == 0

        # System should continue to function
        assert extractor.DOC_TYPE == "pitch_deck"
        assert len(extractor.SUPPORTED_CLAIM_TYPES) > 0
