"""
JURIS VC Decision-Rule DSL.

A domain-specific language for expressing VC investment decision rules
with support for:
- Real-world predicates (has, eq, in, ge, le, between, trend)
- Confidence gates (conf_ge)
- Source filtering (source_in)
- Three-valued logic (TRUE, FALSE, UNKNOWN)
- Safe missing field handling
- Value type normalization

Example usage:

    from juris_agi.vc_dsl import (
        parse,
        Ge, Le, Between, Has, And, ConfGe,
        Rule, RuleEngine, Decision,
        build_context_from_claims,
    )

    # Parse from DSL string
    pred = parse('and(conf_ge(traction.arr, 0.8), ge(traction.arr, 1000000))')

    # Or build programmatically
    pred = And([
        ConfGe("traction.arr", 0.8),
        Ge("traction.arr", 1000000),
    ])

    # Create a rule
    rule = Rule(
        rule_id="high_arr",
        name="High ARR",
        predicate=pred,
        decision=Decision.INVEST,
        priority=7,
    )

    # Evaluate against claims
    ctx = build_context_from_claims(claims)
    engine = RuleEngine([rule])
    trace = engine.evaluate(ctx)

    print(f"Decision: {trace.final_decision}")
"""

# Predicates
from .predicates_v2 import (
    # Result type
    EvalResult,
    # Context types
    FieldValue,
    EvalContext,
    # Base class
    Predicate,
    # Basic predicates
    Has,
    Eq,
    In,
    # Numeric predicates
    Ge,
    Le,
    Between,
    # Trend predicates
    Trend,
    # Confidence/source gates
    ConfGe,
    SourceIn,
    # Composite predicates
    And,
    Or,
    Not,
    Implies,
    # Utilities
    require_confidence,
    require_source,
    PREDICATE_REGISTRY,
)

# Typing
from .typing import (
    ValueType,
    TrendKind,
    UnitType,
    TypedValue,
    normalize_value,
    normalize_numeric,
    normalize_boolean,
    normalize_enum,
    infer_value_type,
    get_field_unit,
)

# Evaluation
from .evaluation import (
    Decision,
    Rule,
    RuleOutcome,
    EvaluationTrace,
    RuleEngine,
    build_context_from_claims,
    build_context_from_dict,
    create_threshold_rule,
    create_enum_rule,
    create_existence_rule,
)

# Parser
from .parser import (
    parse,
    pretty_print,
    parse_rule,
    format_rule,
    ParseError,
)

# Search Space
from .search_space import (
    TemplateType,
    PredicateTemplate,
    RuleTemplate,
    SearchSpaceConfig,
    TEMPLATE_BY_ID,
    ALL_TEMPLATES,
    TRACTION_TEMPLATES,
    TEAM_TEMPLATES,
    MARKET_TEMPLATES,
    FINANCIAL_TEMPLATES,
    RISK_TEMPLATES,
    DEAL_TEMPLATES,
    TREND_TEMPLATES,
    get_template_by_field,
    get_templates_for_claim_type,
    # Synthesizer
    SynthesizerConfig,
    SynthesisTrace,
    PredicateSynthesizer,
    synthesize_predicates,
)

# Threshold Proposer
from .thresholds import (
    ThresholdReason,
    ThresholdCandidate,
    ThresholdTrace,
    ThresholdPrior,
    ThresholdProposerConfig,
    propose_thresholds,
    propose_thresholds_for_template,
    propose_all_thresholds,
    get_thresholds_only,
    default_thresholds_for_field,
    FieldObservations,
    merge_threshold_traces,
    DOMAIN_HEURISTICS,
)

# Time Series
from .timeseries import (
    TimeGranularity,
    TimePoint,
    TimeSeriesPoint,
    TimeSeries,
    TrendResult,
    TrendConfig,
    TimeSeriesFeatures,
    parse_time_point,
    classify_trend,
    extract_features,
    add_timeseries_features_to_context,
    extract_all_timeseries_features,
    interpolate_missing,
    filter_outliers,
)

# Multi-Hypothesis
from .hypothesis import (
    HistoricalDecision,
    DecisionDataset,
    CoverageResult,
    CoverageStats,
    ExceptionCase,
    MDLScoreBreakdown,
    MDLScorer,
    PolicyHypothesis,
    HypothesisSetConfig,
    HypothesisSet,
    MultiHypothesisEngine,
)

# Hierarchical Reasoning
from .hierarchy import (
    PartitionKey,
    Partition,
    PolicyOverride,
    HierarchicalPolicy,
    HierarchicalEvaluationResult,
    HierarchicalLearningConfig,
    HierarchicalLearningEngine,
    learn_hierarchical_policy,
    evaluate_with_hierarchy,
    summarize_policy,
)

# Uncertainty Quantification
from .uncertainty import (
    UncertaintyLevel,
    UncertaintyReason,
    EpistemicUncertainty,
    AleatoricUncertainty,
    InformationRequest,
    UncertaintyReport,
    UncertaintyConfig,
    UncertaintyAnalyzer,
    analyze_uncertainty,
    should_request_more_info,
    get_top_information_requests,
)

__all__ = [
    # Results
    "EvalResult",
    # Context
    "FieldValue",
    "EvalContext",
    # Predicates
    "Predicate",
    "Has",
    "Eq",
    "In",
    "Ge",
    "Le",
    "Between",
    "Trend",
    "ConfGe",
    "SourceIn",
    "And",
    "Or",
    "Not",
    "Implies",
    "require_confidence",
    "require_source",
    "PREDICATE_REGISTRY",
    # Typing
    "ValueType",
    "TrendKind",
    "UnitType",
    "TypedValue",
    "normalize_value",
    "normalize_numeric",
    "normalize_boolean",
    "normalize_enum",
    "infer_value_type",
    "get_field_unit",
    # Evaluation
    "Decision",
    "Rule",
    "RuleOutcome",
    "EvaluationTrace",
    "RuleEngine",
    "build_context_from_claims",
    "build_context_from_dict",
    "create_threshold_rule",
    "create_enum_rule",
    "create_existence_rule",
    # Parser
    "parse",
    "pretty_print",
    "parse_rule",
    "format_rule",
    "ParseError",
    # Search Space
    "TemplateType",
    "PredicateTemplate",
    "RuleTemplate",
    "SearchSpaceConfig",
    "TEMPLATE_BY_ID",
    "ALL_TEMPLATES",
    "TRACTION_TEMPLATES",
    "TEAM_TEMPLATES",
    "MARKET_TEMPLATES",
    "FINANCIAL_TEMPLATES",
    "RISK_TEMPLATES",
    "DEAL_TEMPLATES",
    "TREND_TEMPLATES",
    "get_template_by_field",
    "get_templates_for_claim_type",
    # Synthesizer
    "SynthesizerConfig",
    "SynthesisTrace",
    "PredicateSynthesizer",
    "synthesize_predicates",
    # Threshold Proposer
    "ThresholdReason",
    "ThresholdCandidate",
    "ThresholdTrace",
    "ThresholdPrior",
    "ThresholdProposerConfig",
    "propose_thresholds",
    "propose_thresholds_for_template",
    "propose_all_thresholds",
    "get_thresholds_only",
    "default_thresholds_for_field",
    "FieldObservations",
    "merge_threshold_traces",
    "DOMAIN_HEURISTICS",
    # Time Series
    "TimeGranularity",
    "TimePoint",
    "TimeSeriesPoint",
    "TimeSeries",
    "TrendResult",
    "TrendConfig",
    "TimeSeriesFeatures",
    "parse_time_point",
    "classify_trend",
    "extract_features",
    "add_timeseries_features_to_context",
    "extract_all_timeseries_features",
    "interpolate_missing",
    "filter_outliers",
    # Multi-Hypothesis
    "HistoricalDecision",
    "DecisionDataset",
    "CoverageResult",
    "CoverageStats",
    "ExceptionCase",
    "MDLScoreBreakdown",
    "MDLScorer",
    "PolicyHypothesis",
    "HypothesisSetConfig",
    "HypothesisSet",
    "MultiHypothesisEngine",
    # Hierarchical Reasoning
    "PartitionKey",
    "Partition",
    "PolicyOverride",
    "HierarchicalPolicy",
    "HierarchicalEvaluationResult",
    "HierarchicalLearningConfig",
    "HierarchicalLearningEngine",
    "learn_hierarchical_policy",
    "evaluate_with_hierarchy",
    "summarize_policy",
    # Uncertainty Quantification
    "UncertaintyLevel",
    "UncertaintyReason",
    "EpistemicUncertainty",
    "AleatoricUncertainty",
    "InformationRequest",
    "UncertaintyReport",
    "UncertaintyConfig",
    "UncertaintyAnalyzer",
    "analyze_uncertainty",
    "should_request_more_info",
    "get_top_information_requests",
]
