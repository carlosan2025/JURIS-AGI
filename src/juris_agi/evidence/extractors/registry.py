"""
Extractor Registry.

Provides factory functions and registration for document extractors.
"""

from typing import Dict, Type, Optional, Callable, List, Tuple
from .base import (
    DocumentExtractor,
    ExtractionConfig,
    ExtractionResult,
    ProposedClaim,
    LLMCallFn,
)
from .pitch_deck import PitchDeckExtractor, PitchDeckExtractorV2
from .financial_model import FinancialModelExtractor, FinancialModelExtractorV2
from .tech_description import TechDescriptionExtractor, TechDescriptionExtractorV2
from .ic_memo import ICMemoExtractor, ICMemoExtractorV2


class ExtractorRegistry:
    """
    Registry of document extractors.

    Provides factory methods for creating extractors based on document type.
    """

    # Default extractors by document type
    _extractors: Dict[str, Type[DocumentExtractor]] = {
        "pitch_deck": PitchDeckExtractorV2,
        "financial_model": FinancialModelExtractorV2,
        "tech_description": TechDescriptionExtractorV2,
        "ic_memo": ICMemoExtractorV2,
        # Aliases
        "deck": PitchDeckExtractorV2,
        "pitch": PitchDeckExtractorV2,
        "financials": FinancialModelExtractorV2,
        "model": FinancialModelExtractorV2,
        "spreadsheet": FinancialModelExtractorV2,
        "tech": TechDescriptionExtractorV2,
        "architecture": TechDescriptionExtractorV2,
        "technical": TechDescriptionExtractorV2,
        "memo": ICMemoExtractorV2,
        "ic": ICMemoExtractorV2,
        "deal_memo": ICMemoExtractorV2,
    }

    # V1 extractors (simpler prompts)
    _v1_extractors: Dict[str, Type[DocumentExtractor]] = {
        "pitch_deck": PitchDeckExtractor,
        "financial_model": FinancialModelExtractor,
        "tech_description": TechDescriptionExtractor,
        "ic_memo": ICMemoExtractor,
    }

    @classmethod
    def register(
        cls,
        doc_type: str,
        extractor_class: Type[DocumentExtractor],
    ) -> None:
        """
        Register a new extractor for a document type.

        Args:
            doc_type: Document type identifier
            extractor_class: Extractor class to use
        """
        cls._extractors[doc_type.lower()] = extractor_class

    @classmethod
    def get_extractor_class(
        cls,
        doc_type: str,
        use_v1: bool = False,
    ) -> Optional[Type[DocumentExtractor]]:
        """
        Get extractor class for a document type.

        Args:
            doc_type: Document type identifier
            use_v1: Whether to use simpler V1 extractors

        Returns:
            Extractor class or None if not found
        """
        doc_type_lower = doc_type.lower()
        if use_v1:
            return cls._v1_extractors.get(doc_type_lower)
        return cls._extractors.get(doc_type_lower)

    @classmethod
    def create_extractor(
        cls,
        doc_type: str,
        config: Optional[ExtractionConfig] = None,
        llm_fn: Optional[LLMCallFn] = None,
        use_v1: bool = False,
    ) -> Optional[DocumentExtractor]:
        """
        Create an extractor instance for a document type.

        Args:
            doc_type: Document type identifier
            config: Extraction configuration
            llm_fn: LLM call function
            use_v1: Whether to use simpler V1 extractors

        Returns:
            Extractor instance or None if type not found
        """
        extractor_class = cls.get_extractor_class(doc_type, use_v1)
        if extractor_class is None:
            return None
        return extractor_class(config=config, llm_fn=llm_fn)

    @classmethod
    def list_supported_types(cls) -> List[str]:
        """List all supported document types (primary names only)."""
        return ["pitch_deck", "financial_model", "tech_description", "ic_memo"]

    @classmethod
    def get_type_aliases(cls) -> Dict[str, List[str]]:
        """Get aliases for each document type."""
        return {
            "pitch_deck": ["deck", "pitch"],
            "financial_model": ["financials", "model", "spreadsheet"],
            "tech_description": ["tech", "architecture", "technical"],
            "ic_memo": ["memo", "ic", "deal_memo"],
        }


def get_extractor(
    doc_type: str,
    config: Optional[ExtractionConfig] = None,
    llm_fn: Optional[LLMCallFn] = None,
) -> Optional[DocumentExtractor]:
    """
    Convenience function to get an extractor.

    Args:
        doc_type: Document type identifier
        config: Extraction configuration
        llm_fn: LLM call function

    Returns:
        Extractor instance or None
    """
    return ExtractorRegistry.create_extractor(doc_type, config, llm_fn)


def extract_from_document(
    document_text: str,
    doc_id: str,
    doc_type: str,
    config: Optional[ExtractionConfig] = None,
    llm_fn: Optional[LLMCallFn] = None,
) -> ExtractionResult:
    """
    Extract claims from a document.

    Args:
        document_text: Text content of the document
        doc_id: Document identifier
        doc_type: Type of document
        config: Extraction configuration
        llm_fn: LLM call function

    Returns:
        ExtractionResult with proposed claims
    """
    extractor = get_extractor(doc_type, config, llm_fn)

    if extractor is None:
        result = ExtractionResult(doc_id=doc_id, doc_type=doc_type)
        result.errors.append(f"Unknown document type: {doc_type}")
        return result

    return extractor.extract(document_text, doc_id)


def detect_document_type(
    filename: str,
    content_preview: Optional[str] = None,
) -> str:
    """
    Attempt to detect document type from filename and content.

    Args:
        filename: Name of the file
        content_preview: First ~1000 chars of content

    Returns:
        Best guess at document type
    """
    filename_lower = filename.lower()

    # Check filename patterns
    if any(kw in filename_lower for kw in ["pitch", "deck", "presentation"]):
        return "pitch_deck"
    if any(kw in filename_lower for kw in ["model", "financial", "projection", "budget"]):
        return "financial_model"
    if any(kw in filename_lower for kw in ["tech", "architecture", "technical", "engineering"]):
        return "tech_description"
    if any(kw in filename_lower for kw in ["memo", "ic_", "investment", "recommendation"]):
        return "ic_memo"

    # Check content if available
    if content_preview:
        preview_lower = content_preview.lower()

        # IC memo signals
        if any(kw in preview_lower for kw in [
            "investment committee",
            "deal memo",
            "recommendation:",
            "key risks:",
            "investment thesis",
        ]):
            return "ic_memo"

        # Financial model signals
        if any(kw in preview_lower for kw in [
            "monthly burn",
            "runway",
            "p&l",
            "balance sheet",
            "cash flow",
            "ltv/cac",
        ]):
            return "financial_model"

        # Technical doc signals
        if any(kw in preview_lower for kw in [
            "architecture",
            "tech stack",
            "api",
            "database",
            "microservices",
            "deployment",
        ]):
            return "tech_description"

    # Default to pitch deck (most common)
    return "pitch_deck"


def create_mock_llm_fn() -> LLMCallFn:
    """
    Create a mock LLM function for testing.

    Returns sample claims without calling any API.
    """
    import json

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        # Return sample claims based on prompt content
        sample_claims = [
            {
                "claim_type": "company_identity",
                "field": "legal_name",
                "value": "Sample Company Inc.",
                "confidence": 0.95,
                "polarity": "neutral",
                "locator": "Page 1",
                "quote": "Sample Company Inc.",
                "rationale": "Company name from title (mock extraction)",
            },
            {
                "claim_type": "traction",
                "field": "mrr",
                "value": 50000,
                "confidence": 0.75,
                "polarity": "supportive",
                "locator": "Page 5",
                "quote": "$50K MRR",
                "rationale": "Revenue metric extracted (mock extraction)",
            },
        ]
        return json.dumps(sample_claims)

    return mock_llm
