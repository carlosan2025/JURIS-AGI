"""
Decision Report generation for JURIS-AGI.

Provides structured, human-readable reports explaining investment decisions,
their basis, uncertainties, and counterfactuals.
"""

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
)
from .generator import ReportGenerator, generate_report
from .renderers import (
    render_html,
    render_markdown,
    render_pdf,
    ReportRenderer,
    HTMLRenderer,
    MarkdownRenderer,
    PDFRenderer,
)

__all__ = [
    # Schema
    "DecisionReport",
    "ExecutiveSummary",
    "InferredDecisionLogic",
    "EvidenceBasis",
    "DealEvaluation",
    "CounterfactualAnalysis",
    "UncertaintyLimitations",
    "AuditMetadata",
    "EvidenceItem",
    "RuleEvaluation",
    "SensitivityItem",
    # Generator
    "ReportGenerator",
    "generate_report",
    # Renderers
    "render_html",
    "render_markdown",
    "render_pdf",
    "ReportRenderer",
    "HTMLRenderer",
    "MarkdownRenderer",
    "PDFRenderer",
]
