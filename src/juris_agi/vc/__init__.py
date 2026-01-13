"""
VC Decision Reasoning Module.

Provides counterfactual and robustness analysis for VC investment decisions.

Key components:
- orchestrator: End-to-end VC decision workflow
- counterfactuals: Generate minimal perturbations to evidence graphs
- decision_analysis: Identify critical claims and compute robustness
- trace: Complete audit trail with counterfactual explanations
"""

from .counterfactuals import (
    PerturbationType,
    ClaimPerturbation,
    CounterfactualEvidenceGraph,
    EvidenceCounterfactualGenerator,
    generate_counterfactuals,
)

from .decision_analysis import (
    DecisionOutcome,
    DecisionCriticalClaim,
    DecisionRobustness,
    CounterfactualExplanation,
    DecisionAnalysisResult,
    DecisionAnalyzer,
    analyze_decision,
)

from .trace import (
    VCDecisionTraceEntry,
    VCDecisionTrace,
    VCDecisionTracer,
    create_decision_trace,
)

from .orchestrator import (
    OrchestratorConfig,
    OrchestratorTrace,
    VCOrchestrator,
    solve_vc_decision,
)

__all__ = [
    # Orchestrator
    "OrchestratorConfig",
    "OrchestratorTrace",
    "VCOrchestrator",
    "solve_vc_decision",
    # Counterfactuals
    "PerturbationType",
    "ClaimPerturbation",
    "CounterfactualEvidenceGraph",
    "EvidenceCounterfactualGenerator",
    "generate_counterfactuals",
    # Decision Analysis
    "DecisionOutcome",
    "DecisionCriticalClaim",
    "DecisionRobustness",
    "CounterfactualExplanation",
    "DecisionAnalysisResult",
    "DecisionAnalyzer",
    "analyze_decision",
    # Trace
    "VCDecisionTraceEntry",
    "VCDecisionTrace",
    "VCDecisionTracer",
    "create_decision_trace",
]
