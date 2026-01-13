"""
VC Reasoning Orchestrator.

Coordinates the end-to-end VC decision workflow:
1. Fetch evidence context (from Evidence API or direct claims)
2. Build evaluation context with time series features
3. Propose thresholds from data
4. Learn policies (optionally hierarchical)
5. Evaluate and analyze uncertainty
6. Produce final decision with audit trail
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from juris_agi.api.vc_models import (
    DecisionOutput,
    DirectClaim,
    PolicyOutput,
    PolicyRule,
    UncertaintyOutput,
    VCConstraints,
    VCJobEvent,
    VCJobResult,
    VCJobStatus,
    WorkingSetSummary,
)
from juris_agi.evidence_client.client import EvidenceApiClient
from juris_agi.evidence_client.types import (
    Claim,
    ClaimPolarity,
    ContextConstraints,
    EvidenceContext,
)
from juris_agi.vc_dsl import (
    # Evaluation
    Decision,
    EvalContext,
    EvaluationTrace,
    FieldValue,
    Rule,
    RuleEngine,
    build_context_from_claims,
    # Hypothesis
    CoverageStats,
    DecisionDataset,
    ExceptionCase,
    HistoricalDecision,
    HypothesisSet,
    HypothesisSetConfig,
    MDLScoreBreakdown,
    MultiHypothesisEngine,
    PolicyHypothesis,
    # Hierarchy
    HierarchicalLearningConfig,
    HierarchicalLearningEngine,
    HierarchicalPolicy,
    PartitionKey,
    evaluate_with_hierarchy,
    learn_hierarchical_policy,
    # Uncertainty
    UncertaintyAnalyzer,
    UncertaintyConfig,
    UncertaintyReport,
    analyze_uncertainty,
    # Thresholds
    ThresholdProposerConfig,
    propose_all_thresholds,
    # Time series
    TimeSeries,
    add_timeseries_features_to_context,
    extract_all_timeseries_features,
    # Parser
    pretty_print,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the VC orchestrator."""

    # Threshold proposer
    propose_thresholds: bool = True
    threshold_config: Optional[ThresholdProposerConfig] = None

    # Policy learning
    learn_hierarchical: bool = True
    hierarchical_config: Optional[HierarchicalLearningConfig] = None
    hypothesis_config: Optional[HypothesisSetConfig] = None

    # Uncertainty analysis
    analyze_uncertainty: bool = True
    uncertainty_config: Optional[UncertaintyConfig] = None

    # Output
    max_policies: int = 5
    include_trace: bool = True


@dataclass
class OrchestratorTrace:
    """Trace of orchestrator execution for audit/debugging."""

    events: list[VCJobEvent] = field(default_factory=list)
    context: Optional[EvidenceContext] = None
    eval_context: Optional[EvalContext] = None
    threshold_traces: dict[str, Any] = field(default_factory=dict)
    policies: list[PolicyHypothesis] = field(default_factory=list)
    hierarchical_policy: Optional[HierarchicalPolicy] = None
    evaluation_trace: Optional[EvaluationTrace] = None
    uncertainty_report: Optional[UncertaintyReport] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def add_event(
        self,
        event_type: str,
        data: dict[str, Any],
        message: Optional[str] = None,
    ) -> VCJobEvent:
        """Add an event to the trace."""
        event = VCJobEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            timestamp=datetime.utcnow(),
            data=data,
            message=message,
        )
        self.events.append(event)
        logger.info(f"[{event_type}] {message or ''} {data}")
        return event

    @property
    def runtime_seconds(self) -> float:
        """Get runtime in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time


class VCOrchestrator:
    """
    Orchestrates the VC decision workflow.

    Given a deal and question (or direct claims), produces a decision
    with supporting policies, uncertainty analysis, and full audit trail.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        evidence_client: Optional[EvidenceApiClient] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Orchestrator configuration
            evidence_client: Evidence API client (optional, for remote mode)
        """
        self.config = config or OrchestratorConfig()
        self.evidence_client = evidence_client

    async def solve(
        self,
        deal_id: Optional[str] = None,
        question: Optional[str] = None,
        claims: Optional[list[DirectClaim]] = None,
        constraints: Optional[VCConstraints] = None,
        historical_decisions: Optional[list[dict[str, Any]]] = None,
    ) -> VCJobResult:
        """
        Execute the full VC reasoning workflow.

        Args:
            deal_id: Deal identifier (for Evidence API lookup)
            question: Investment question
            claims: Direct claims (alternative to deal_id)
            constraints: Processing constraints
            historical_decisions: Historical decisions for policy learning

        Returns:
            VCJobResult with decision, policies, uncertainty, and trace
        """
        job_id = f"vcjob_{uuid.uuid4().hex[:12]}"
        trace = OrchestratorTrace()
        constraints = constraints or VCConstraints()
        created_at = datetime.utcnow()

        try:
            # Step 1: Fetch or build evidence context
            context = await self._fetch_context(
                deal_id=deal_id,
                question=question,
                claims=claims,
                constraints=constraints,
                trace=trace,
            )
            trace.context = context

            # Step 2: Build evaluation context
            eval_ctx = self._build_eval_context(context, trace)
            trace.eval_context = eval_ctx

            # Step 3: Propose thresholds (if enabled)
            if self.config.propose_thresholds:
                self._propose_thresholds(eval_ctx, trace)

            # Step 4: Learn policies
            policies, hierarchical_policy = self._learn_policies(
                eval_ctx=eval_ctx,
                historical_decisions=historical_decisions,
                constraints=constraints,
                trace=trace,
            )
            trace.policies = policies
            trace.hierarchical_policy = hierarchical_policy

            # Step 5: Evaluate current deal
            eval_trace, decision = self._evaluate(
                eval_ctx=eval_ctx,
                policies=policies,
                hierarchical_policy=hierarchical_policy,
                trace=trace,
            )
            trace.evaluation_trace = eval_trace

            # Step 6: Analyze uncertainty
            uncertainty_output = None
            if self.config.analyze_uncertainty and policies:
                uncertainty_output = self._analyze_uncertainty(
                    eval_ctx=eval_ctx,
                    policies=policies,
                    trace=trace,
                )
                # Adjust decision based on uncertainty
                if uncertainty_output and uncertainty_output.should_defer:
                    decision = DecisionOutput(
                        decision="defer",
                        confidence=1.0 - uncertainty_output.total_uncertainty,
                        explanation=f"Deferring due to high uncertainty: {uncertainty_output.top_reasons[0] if uncertainty_output.top_reasons else 'insufficient information'}",
                        supporting_rules=[],
                        blocking_rules=[],
                    )

            # Build result
            trace.end_time = time.time()

            return VCJobResult(
                job_id=job_id,
                status=VCJobStatus.COMPLETED,
                deal_id=deal_id,
                question=question,
                created_at=created_at,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                runtime_seconds=trace.runtime_seconds,
                context_id=context.context_id if context else None,
                working_set=self._build_working_set_summary(context),
                decision=decision,
                policies=self._format_policies(policies[:constraints.max_policies]),
                uncertainty=uncertainty_output,
                events=trace.events,
                trace_url=f"/vc/jobs/{job_id}/trace",
                events_url=f"/vc/jobs/{job_id}/events",
                trace_data=self._build_trace_data(trace) if self.config.include_trace else None,
            )

        except Exception as e:
            trace.end_time = time.time()
            trace.add_event(
                "error",
                {"error_type": type(e).__name__, "error_message": str(e)},
                f"Job failed: {e}",
            )
            logger.exception(f"VC solve job {job_id} failed")

            return VCJobResult(
                job_id=job_id,
                status=VCJobStatus.FAILED,
                deal_id=deal_id,
                question=question,
                created_at=created_at,
                completed_at=datetime.utcnow(),
                runtime_seconds=trace.runtime_seconds,
                error_message=str(e),
                events=trace.events,
            )

    async def _fetch_context(
        self,
        deal_id: Optional[str],
        question: Optional[str],
        claims: Optional[list[DirectClaim]],
        constraints: VCConstraints,
        trace: OrchestratorTrace,
    ) -> EvidenceContext:
        """Fetch evidence context from API or build from direct claims."""
        if claims:
            # Demo mode: build context from direct claims
            trace.add_event(
                "context_built",
                {"mode": "direct", "claim_count": len(claims)},
                f"Built context from {len(claims)} direct claims",
            )

            # Convert DirectClaim to Claim format
            claim_dicts = []
            for c in claims:
                claim_dict = {
                    "claim_type": c.claim_type,
                    "field": c.field,
                    "value": c.value,
                    "confidence": c.confidence,
                    "polarity": c.polarity,
                }
                if c.timeseries:
                    claim_dict["timeseries"] = c.timeseries
                claim_dicts.append(claim_dict)

            return EvidenceApiClient.from_direct_claims(
                deal_id=deal_id or "demo_deal",
                claims=claim_dicts,
                question=question,
            )

        elif deal_id and self.evidence_client and self.evidence_client.is_configured:
            # Remote mode: fetch from Evidence API
            trace.add_event(
                "context_fetching",
                {"deal_id": deal_id, "question": question},
                f"Fetching context for deal {deal_id}",
            )

            api_constraints = ContextConstraints(
                max_claims=constraints.max_claims,
                min_confidence=constraints.min_confidence,
                claim_types=constraints.focus_claim_types,
            )

            response = await self.evidence_client.create_context(
                deal_id=deal_id,
                question=question,
                constraints=api_constraints,
            )

            trace.add_event(
                "context_fetched",
                {
                    "context_id": response.context_id,
                    "total_claims": response.summary.total_claims,
                    "conflict_count": response.summary.conflict_count,
                },
                f"Fetched {response.summary.total_claims} claims, {response.summary.conflict_count} conflicts",
            )

            return EvidenceContext(
                context_id=response.context_id,
                deal_id=deal_id,
                question=question,
                claims=response.claims,
                conflicts=response.conflicts,
                citations=response.citations,
                summary=response.summary,
            )

        else:
            raise ValueError(
                "Either claims must be provided or Evidence API must be configured with a deal_id"
            )

    def _build_eval_context(
        self,
        context: EvidenceContext,
        trace: OrchestratorTrace,
    ) -> EvalContext:
        """Build evaluation context from evidence context."""
        # Build base context from claims
        eval_ctx = build_context_from_claims(context.claims)

        # Extract and add time series features
        timeseries_fields = []
        for claim in context.claims:
            if claim.has_timeseries:
                field_name = f"{claim.claim_type}.{claim.field}"
                timeseries_fields.append(field_name)

                # Create TimeSeries object
                ts_data = [{"t": p.t, "value": p.value} for p in claim.timeseries]
                ts = TimeSeries.from_list(field_name, ts_data)

                # Add features to context
                add_timeseries_features_to_context(eval_ctx, ts, field_name)

        trace.add_event(
            "working_set_built",
            {
                "total_fields": len(eval_ctx.fields),
                "timeseries_fields": timeseries_fields,
            },
            f"Built working set with {len(eval_ctx.fields)} fields, {len(timeseries_fields)} time series",
        )

        return eval_ctx

    def _propose_thresholds(
        self,
        eval_ctx: EvalContext,
        trace: OrchestratorTrace,
    ) -> None:
        """Propose thresholds based on observed data."""
        from juris_agi.vc_dsl import FieldObservations

        # Extract observations from context
        observations_list = []
        for field_name, field_value in eval_ctx.fields.items():
            if field_value.exists and isinstance(field_value.value, (int, float)):
                observations_list.append(
                    FieldObservations(field=field_name, values=[field_value.value])
                )

        if not observations_list:
            return

        # Propose thresholds
        config = self.config.threshold_config or ThresholdProposerConfig()
        threshold_traces = propose_all_thresholds(observations_list, config=config)

        # threshold_traces is dict[str, tuple[list[float], Optional[ThresholdTrace]]]
        trace.threshold_traces = {
            field: {
                "thresholds": thresholds,
                "trace": {
                    "observed_count": t.observed_count,
                    "observed_min": t.observed_min,
                    "observed_max": t.observed_max,
                    "final_thresholds": t.final_thresholds,
                } if t else None,
            }
            for field, (thresholds, t) in threshold_traces.items()
        }

        trace.add_event(
            "thresholds_proposed",
            {
                "fields_analyzed": len(threshold_traces),
                "sample_thresholds": {
                    k: v[0][0] if v[0] else None
                    for k, v in list(threshold_traces.items())[:3]
                },
            },
            f"Proposed thresholds for {len(threshold_traces)} fields",
        )

    def _learn_policies(
        self,
        eval_ctx: EvalContext,
        historical_decisions: Optional[list[dict[str, Any]]],
        constraints: VCConstraints,
        trace: OrchestratorTrace,
    ) -> tuple[list[PolicyHypothesis], Optional[HierarchicalPolicy]]:
        """Learn policies from historical decisions."""
        # Build dataset from historical decisions
        if not historical_decisions:
            # No historical data - use default policies
            trace.add_event(
                "policies_default",
                {"reason": "no_historical_data"},
                "Using default policies (no historical decisions provided)",
            )
            return self._create_default_policies(eval_ctx, constraints), None

        # Convert to HistoricalDecision objects
        decisions = []
        for hd in historical_decisions:
            # Build context from claims
            claims_data = hd.get("claims", [])
            claims = [
                Claim(
                    claim_id=f"hclaim_{i}",
                    claim_type=c["claim_type"],
                    field=c["field"],
                    value=c["value"],
                    confidence=c.get("confidence", 0.8),
                    polarity=ClaimPolarity(c.get("polarity", "neutral")),
                )
                for i, c in enumerate(claims_data)
            ]
            ctx = build_context_from_claims(claims)

            # Parse decision
            decision_str = hd.get("decision", "defer").upper()
            decision = Decision[decision_str] if decision_str in Decision.__members__ else Decision.DEFER

            decisions.append(
                HistoricalDecision(
                    decision_id=hd.get("id", f"hd_{len(decisions)}"),
                    context=ctx,
                    decision=decision,
                    metadata=hd.get("metadata", {}),
                )
            )

        dataset = DecisionDataset(decisions=decisions)

        # Learn hierarchical policy if enabled
        hierarchical_policy = None
        if self.config.learn_hierarchical:
            hierarchical_policy = self._learn_hierarchical(dataset, constraints, trace)

        # Learn multi-hypothesis policies
        policies = self._learn_multi_hypothesis(dataset, constraints, trace)

        return policies, hierarchical_policy

    def _learn_hierarchical(
        self,
        dataset: DecisionDataset,
        constraints: VCConstraints,
        trace: OrchestratorTrace,
    ) -> Optional[HierarchicalPolicy]:
        """Learn hierarchical policy with partition overrides."""
        config = self.config.hierarchical_config or HierarchicalLearningConfig(
            partition_key=PartitionKey.SECTOR,
            min_partition_size=2,
        )

        try:
            policy = learn_hierarchical_policy(dataset, config)

            trace.add_event(
                "hierarchical_policy_learned",
                {
                    "global_rules": len(policy.global_rules),
                    "overrides": list(policy.overrides.keys()),
                },
                f"Learned hierarchical policy: {len(policy.global_rules)} global rules, {len(policy.overrides)} overrides",
            )

            return policy
        except Exception as e:
            logger.warning(f"Failed to learn hierarchical policy: {e}")
            return None

    def _learn_multi_hypothesis(
        self,
        dataset: DecisionDataset,
        constraints: VCConstraints,
        trace: OrchestratorTrace,
    ) -> list[PolicyHypothesis]:
        """Learn multiple policy hypotheses."""
        config = self.config.hypothesis_config or HypothesisSetConfig(
            max_hypotheses=constraints.max_policies,
            min_coverage=constraints.min_coverage,
        )

        engine = MultiHypothesisEngine(config)
        hypothesis_set = engine.learn(dataset)

        trace.add_event(
            "policies_learned",
            {
                "num_policies": len(hypothesis_set.hypotheses),
                "best_mdl": hypothesis_set.best.mdl_score if hypothesis_set.best else None,
                "best_coverage": hypothesis_set.best.coverage.coverage_rate if hypothesis_set.best else None,
            },
            f"Learned {len(hypothesis_set.hypotheses)} policy hypotheses",
        )

        return hypothesis_set.hypotheses

    def _create_default_policies(
        self,
        eval_ctx: EvalContext,
        constraints: VCConstraints,
    ) -> list[PolicyHypothesis]:
        """Create default policies when no historical data is available."""
        from juris_agi.vc_dsl import (
            And,
            ConfGe,
            Ge,
            Has,
            Le,
        )

        # Create a simple default policy based on common VC criteria
        default_rules = [
            Rule(
                rule_id="high_arr",
                name="High ARR",
                predicate=And([ConfGe("traction.arr", 0.7), Ge("traction.arr", 1_000_000)]),
                decision=Decision.INVEST,
                priority=7,
            ),
            Rule(
                rule_id="strong_growth",
                name="Strong Growth",
                predicate=And([ConfGe("traction.growth_rate", 0.7), Ge("traction.growth_rate", 0.5)]),
                decision=Decision.INVEST,
                priority=6,
            ),
            Rule(
                rule_id="no_revenue",
                name="No Revenue",
                predicate=And([Has("traction.arr"), Le("traction.arr", 0)]),
                decision=Decision.PASS,
                priority=8,
            ),
        ]

        # Create a single default hypothesis
        return [
            PolicyHypothesis(
                hypothesis_id="default_policy",
                name="Default Policy",
                rules=default_rules,
                coverage_stats=CoverageStats(),
                exceptions=[],
                mdl_breakdown=MDLScoreBreakdown(
                    rule_complexity=5.0,
                    num_rules=3,
                    exception_cost=0.0,
                    num_exceptions=0,
                ),
            )
        ]

    def _evaluate(
        self,
        eval_ctx: EvalContext,
        policies: list[PolicyHypothesis],
        hierarchical_policy: Optional[HierarchicalPolicy],
        trace: OrchestratorTrace,
    ) -> tuple[EvaluationTrace, DecisionOutput]:
        """Evaluate the current deal against learned policies."""
        # Use hierarchical policy if available
        if hierarchical_policy:
            # Get sector from context if available
            sector = None
            if eval_ctx.has_field("deal.sector"):
                sector = eval_ctx.get_field("deal.sector").value

            result = evaluate_with_hierarchy(
                hierarchical_policy,
                eval_ctx,
                sector=sector,
            )

            eval_trace = result.trace
            decision = result.decision
        elif policies:
            # Use best policy
            best_policy = policies[0]
            engine = RuleEngine(best_policy.rules)
            eval_trace = engine.evaluate(eval_ctx)
            decision = eval_trace.final_decision
        else:
            # No policies - defer
            eval_trace = EvaluationTrace()
            decision = Decision.DEFER

        trace.add_event(
            "evaluation_complete",
            {
                "decision": decision.value if decision else "defer",
                "rules_fired": len(eval_trace.rules_fired) if eval_trace else 0,
                "rules_unknown": len(eval_trace.rules_unknown) if eval_trace else 0,
            },
            f"Evaluation complete: {decision.value if decision else 'defer'}",
        )

        # Build explanation
        explanation_parts = []
        supporting_rules = []
        blocking_rules = []

        if eval_trace:
            for outcome in eval_trace.rules_fired:
                if outcome.decision == decision:
                    supporting_rules.append(outcome.rule_name)
                    explanation_parts.append(outcome.explanation or outcome.rule_name)

            for outcome in eval_trace.rule_outcomes:
                if outcome.result.name == "TRUE" and outcome.decision != decision:
                    blocking_rules.append(outcome.rule_name)

        return eval_trace, DecisionOutput(
            decision=decision.value if decision else "defer",
            confidence=eval_trace.decision_confidence if eval_trace else 0.5,
            explanation="; ".join(explanation_parts) if explanation_parts else "No rules matched",
            supporting_rules=supporting_rules,
            blocking_rules=blocking_rules,
        )

    def _analyze_uncertainty(
        self,
        eval_ctx: EvalContext,
        policies: list[PolicyHypothesis],
        trace: OrchestratorTrace,
    ) -> Optional[UncertaintyOutput]:
        """Analyze uncertainty in the decision."""
        if not policies:
            return None

        # Build hypothesis set for uncertainty analysis
        config = HypothesisSetConfig(max_hypotheses=len(policies))
        hypothesis_set = HypothesisSet(config)
        # Directly set the hypotheses (they were already created)
        hypothesis_set.hypotheses = list(policies)

        # Analyze uncertainty
        analyzer = UncertaintyAnalyzer(self.config.uncertainty_config or UncertaintyConfig())
        report = analyzer.analyze(eval_ctx, hypothesis_set)

        trace.uncertainty_report = report

        trace.add_event(
            "uncertainty_analyzed",
            {
                "epistemic_score": report.epistemic.score,
                "aleatoric_score": report.aleatoric.score,
                "total_uncertainty": report.total_uncertainty,
                "should_defer": report.should_defer,
            },
            f"Uncertainty: epistemic={report.epistemic.score:.2f}, aleatoric={report.aleatoric.score:.2f}",
        )

        return UncertaintyOutput(
            epistemic_score=report.epistemic.score,
            aleatoric_score=report.aleatoric.score,
            total_uncertainty=report.total_uncertainty,
            uncertainty_level=report.level.value,
            top_reasons=[r.description for r in report.top_reasons[:5]],
            information_requests=[
                {
                    "field": req.field,
                    "reason": req.reason,
                    "importance": req.importance,
                }
                for req in report.information_requests[:5]
            ],
            should_defer=report.should_defer,
        )

    def _build_working_set_summary(self, context: EvidenceContext) -> Optional[WorkingSetSummary]:
        """Build working set summary from evidence context."""
        if not context:
            return None

        timeseries_fields = [
            f"{c.claim_type}.{c.field}" for c in context.claims if c.has_timeseries
        ]

        return WorkingSetSummary(
            total_claims=context.summary.total_claims,
            claims_by_type=context.summary.claims_by_type,
            claims_by_polarity=context.summary.claims_by_polarity,
            avg_confidence=context.summary.avg_confidence,
            timeseries_fields=timeseries_fields,
            conflict_count=context.summary.conflict_count,
        )

    def _format_policies(self, policies: list[PolicyHypothesis]) -> list[PolicyOutput]:
        """Format policies for API output."""
        return [
            PolicyOutput(
                policy_id=p.hypothesis_id,
                rules=[
                    PolicyRule(
                        rule_id=r.rule_id,
                        name=r.name,
                        predicate_dsl=pretty_print(r.predicate),
                        decision=r.decision.value,
                        priority=r.priority,
                        fields_used=r.predicate.get_fields(),
                    )
                    for r in p.rules
                ],
                mdl_score=p.score,  # .score property returns mdl_breakdown.total_score
                coverage=p.coverage_rate,
                exception_count=len(p.exceptions),
            )
            for p in policies
        ]

    def _build_trace_data(self, trace: OrchestratorTrace) -> dict[str, Any]:
        """Build trace data for API response."""
        return {
            "runtime_seconds": trace.runtime_seconds,
            "events": [e.model_dump() for e in trace.events],
            "context_id": trace.context.context_id if trace.context else None,
            "threshold_traces": trace.threshold_traces,
            "policy_count": len(trace.policies),
            "evaluation": {
                "rules_fired": [
                    {
                        "rule_id": o.rule_id,
                        "rule_name": o.rule_name,
                        "decision": o.decision.value if o.decision else None,
                    }
                    for o in (trace.evaluation_trace.rules_fired if trace.evaluation_trace else [])
                ],
                "final_decision": (
                    trace.evaluation_trace.final_decision.value
                    if trace.evaluation_trace and trace.evaluation_trace.final_decision
                    else None
                ),
            },
            "uncertainty": (
                {
                    "epistemic_score": trace.uncertainty_report.epistemic.score,
                    "aleatoric_score": trace.uncertainty_report.aleatoric.score,
                    "should_defer": trace.uncertainty_report.should_defer,
                }
                if trace.uncertainty_report
                else None
            ),
        }


# Convenience function for one-off calls
async def solve_vc_decision(
    deal_id: Optional[str] = None,
    question: Optional[str] = None,
    claims: Optional[list[DirectClaim]] = None,
    constraints: Optional[VCConstraints] = None,
    historical_decisions: Optional[list[dict[str, Any]]] = None,
    config: Optional[OrchestratorConfig] = None,
) -> VCJobResult:
    """
    Convenience function to solve a VC decision.

    Args:
        deal_id: Deal identifier (for Evidence API lookup)
        question: Investment question
        claims: Direct claims (alternative to deal_id)
        constraints: Processing constraints
        historical_decisions: Historical decisions for policy learning
        config: Orchestrator configuration

    Returns:
        VCJobResult with decision, policies, and uncertainty analysis
    """
    orchestrator = VCOrchestrator(config=config)
    return await orchestrator.solve(
        deal_id=deal_id,
        question=question,
        claims=claims,
        constraints=constraints,
        historical_decisions=historical_decisions,
    )
