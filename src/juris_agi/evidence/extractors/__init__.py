"""
LLM-based document extractors for Evidence Graphs.

This module provides extractors that use LLMs to propose claims from
various document types (pitch decks, financial models, tech descriptions, IC memos).

Key principles:
- Extraction produces PROPOSED claims only
- All claims include confidence and source locators
- Human-approved claims are NEVER overwritten
- System works fully without LLM extraction enabled
"""

from .base import (
    DocumentExtractor,
    ExtractionResult,
    ProposedClaim,
    ExtractionConfig,
    ExtractionStatus,
)
from .pitch_deck import PitchDeckExtractor
from .financial_model import FinancialModelExtractor
from .tech_description import TechDescriptionExtractor
from .ic_memo import ICMemoExtractor
from .registry import ExtractorRegistry, get_extractor, extract_from_document

__all__ = [
    # Base classes
    "DocumentExtractor",
    "ExtractionResult",
    "ProposedClaim",
    "ExtractionConfig",
    "ExtractionStatus",
    # Extractors
    "PitchDeckExtractor",
    "FinancialModelExtractor",
    "TechDescriptionExtractor",
    "ICMemoExtractor",
    # Registry
    "ExtractorRegistry",
    "get_extractor",
    "extract_from_document",
]
