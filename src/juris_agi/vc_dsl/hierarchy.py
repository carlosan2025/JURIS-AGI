"""
Hierarchical policy reasoning for JURIS VC DSL.

Reduces search complexity by partitioning training examples by sector/stage
and learning:
- Global policy (applies to all deals)
- Per-partition policy overrides (sector-specific, stage-specific rules)

This reflects real investment policies that vary by sector (e.g., biotech
requires different metrics than SaaS) and stage (seed vs growth).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .evaluation import Decision, Rule, RuleEngine, EvalContext
from .hypothesis import (
    HistoricalDecision,
    DecisionDataset,
    CoverageStats,
    ExceptionCase,
    MDLScorer,
    MDLScoreBreakdown,
    PolicyHypothesis,
    HypothesisSet,
    HypothesisSetConfig,
)
from .predicates_v2 import EvalResult

logger = logging.getLogger(__name__)


# =============================================================================
# Partition Types
# =============================================================================


class PartitionKey(Enum):
    """Types of partitions for hierarchical policies."""

    SECTOR = "sector"
    STAGE = "stage"
    SECTOR_STAGE = "sector_stage"  # Combined partition


@dataclass
class Partition:
    """A partition of the data for hierarchical learning."""

    key: PartitionKey
    value: str  # e.g., "saas", "biotech", "seed", "series_a"
    decisions: list[HistoricalDecision] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.key.value}:{self.value}"

    def __len__(self) -> int:
        return len(self.decisions)

    def to_dataset(self) -> DecisionDataset:
        """Convert to DecisionDataset."""
        return DecisionDataset(
            decisions=self.decisions,
            name=self.name,
        )


# =============================================================================
# Policy Override
# =============================================================================


@dataclass
class PolicyOverride:
    """
    A policy override for a specific partition.

    Overrides can:
    - Add new rules (partition-specific requirements)
    - Replace global rules (different thresholds)
    - Disable global rules (not applicable to this partition)
    """

    partition_name: str
    partition_key: PartitionKey
    partition_value: str

    # Override rules
    add_rules: list[Rule] = field(default_factory=list)
    replace_rules: dict[str, Rule] = field(default_factory=dict)  # rule_id -> replacement
    disable_rules: list[str] = field(default_factory=list)  # rule_ids to disable

    # Coverage stats for this override
    coverage_stats: CoverageStats = field(default_factory=CoverageStats)
    exceptions: list[ExceptionCase] = field(default_factory=list)
    mdl_score: float = 0.0

    # Learning metadata
    sample_count: int = 0
    improvement_over_global: float = 0.0  # How much better than global policy

    def is_empty(self) -> bool:
        """Check if override is empty (no changes)."""
        return (
            len(self.add_rules) == 0
            and len(self.replace_rules) == 0
            and len(self.disable_rules) == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "partition_name": self.partition_name,
            "partition_key": self.partition_key.value,
            "partition_value": self.partition_value,
            "add_rules": [
                {"rule_id": r.rule_id, "name": r.name, "predicate": r.predicate.to_dsl()}
                for r in self.add_rules
            ],
            "replace_rules": {
                rule_id: {"name": r.name, "predicate": r.predicate.to_dsl()}
                for rule_id, r in self.replace_rules.items()
            },
            "disable_rules": self.disable_rules,
            "coverage_rate": self.coverage_stats.coverage_rate,
            "accuracy": self.coverage_stats.accuracy,
            "sample_count": self.sample_count,
            "improvement_over_global": self.improvement_over_global,
        }


# =============================================================================
# Hierarchical Policy
# =============================================================================


@dataclass
class HierarchicalPolicy:
    """
    A hierarchical policy with global rules and partition overrides.

    Evaluation:
    1. Determine the deal's partition (sector/stage)
    2. Start with global rules
    3. Apply partition overrides (add/replace/disable)
    4. Evaluate against the modified rule set
    """

    name: str = "Hierarchical Policy"

    # Global policy (applies to all deals)
    global_rules: list[Rule] = field(default_factory=list)
    global_coverage_stats: CoverageStats = field(default_factory=CoverageStats)
    global_mdl_score: float = 0.0

    # Partition overrides
    overrides: dict[str, PolicyOverride] = field(default_factory=dict)  # partition_name -> override

    # Configuration
    sector_field: str = "deal.sector"
    stage_field: str = "deal.stage"

    def get_override(self, partition_name: str) -> Optional[PolicyOverride]:
        """Get override for a partition."""
        return self.overrides.get(partition_name)

    def get_rules_for_partition(self, partition_name: Optional[str]) -> list[Rule]:
        """
        Get effective rules for a partition.

        Applies overrides to global rules.
        """
        if partition_name is None or partition_name not in self.overrides:
            return list(self.global_rules)

        override = self.overrides[partition_name]

        # Start with global rules, excluding disabled ones
        effective_rules = [
            r for r in self.global_rules
            if r.rule_id not in override.disable_rules
        ]

        # Apply replacements
        effective_rules = [
            override.replace_rules.get(r.rule_id, r)
            for r in effective_rules
        ]

        # Add new rules
        effective_rules.extend(override.add_rules)

        return effective_rules

    def infer_partition(self, context: EvalContext) -> Optional[str]:
        """
        Infer the partition for a deal from its context.

        Returns partition name or None if partition cannot be determined.
        """
        sector_fv = context.get_field(self.sector_field)
        stage_fv = context.get_field(self.stage_field)

        sector = sector_fv.value if sector_fv.exists else None
        stage = stage_fv.value if stage_fv.exists else None

        # Normalize to lowercase
        if sector and isinstance(sector, str):
            sector = sector.lower()
        if stage and isinstance(stage, str):
            stage = stage.lower()

        # Try combined partition first
        if sector and stage:
            combined = f"{PartitionKey.SECTOR_STAGE.value}:{sector}_{stage}"
            if combined in self.overrides:
                return combined

        # Try sector partition
        if sector:
            sector_key = f"{PartitionKey.SECTOR.value}:{sector}"
            if sector_key in self.overrides:
                return sector_key

        # Try stage partition
        if stage:
            stage_key = f"{PartitionKey.STAGE.value}:{stage}"
            if stage_key in self.overrides:
                return stage_key

        return None

    def evaluate(
        self,
        context: EvalContext,
        default_decision: Decision = Decision.DEFER,
    ) -> "HierarchicalEvaluationResult":
        """
        Evaluate a deal under the hierarchical policy.

        Returns detailed result showing global and override contributions.
        """
        # Infer partition
        partition_name = self.infer_partition(context)

        # Get effective rules
        effective_rules = self.get_rules_for_partition(partition_name)

        # Evaluate
        engine = RuleEngine(effective_rules, default_decision=default_decision)
        trace = engine.evaluate(context)

        # Determine which override applied
        override_applied = self.overrides.get(partition_name) if partition_name else None

        return HierarchicalEvaluationResult(
            final_decision=trace.final_decision,
            confidence=trace.decision_confidence,
            partition_name=partition_name,
            override_applied=override_applied,
            global_rules_count=len(self.global_rules),
            effective_rules_count=len(effective_rules),
            rules_fired=[o.rule_id for o in trace.rules_fired],
            trace=trace,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "global_rules": [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "predicate": r.predicate.to_dsl(),
                    "decision": r.decision.value,
                    "priority": r.priority,
                }
                for r in self.global_rules
            ],
            "global_coverage_rate": self.global_coverage_stats.coverage_rate,
            "global_accuracy": self.global_coverage_stats.accuracy,
            "global_mdl_score": self.global_mdl_score,
            "overrides": {
                name: override.to_dict()
                for name, override in self.overrides.items()
            },
            "partition_config": {
                "sector_field": self.sector_field,
                "stage_field": self.stage_field,
            },
        }


@dataclass
class HierarchicalEvaluationResult:
    """Result of evaluating a deal under hierarchical policy."""

    final_decision: Decision
    confidence: float

    # Partition info
    partition_name: Optional[str] = None
    override_applied: Optional[PolicyOverride] = None

    # Rule stats
    global_rules_count: int = 0
    effective_rules_count: int = 0
    rules_fired: list[str] = field(default_factory=list)

    # Full trace
    trace: Any = None  # EvaluationTrace

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.final_decision.value,
            "confidence": self.confidence,
            "partition_name": self.partition_name,
            "override_applied": self.override_applied.partition_name if self.override_applied else None,
            "global_rules_count": self.global_rules_count,
            "effective_rules_count": self.effective_rules_count,
            "rules_fired": self.rules_fired,
        }


# =============================================================================
# Hierarchical Learning Engine
# =============================================================================


@dataclass
class HierarchicalLearningConfig:
    """Configuration for hierarchical policy learning."""

    # Partition settings
    min_partition_size: int = 5  # Minimum decisions to learn partition override
    partition_keys: list[PartitionKey] = field(
        default_factory=lambda: [PartitionKey.SECTOR, PartitionKey.STAGE]
    )

    # Global policy settings
    global_min_coverage: float = 0.5
    global_min_accuracy: float = 0.6

    # Override settings
    override_min_improvement: float = 0.1  # Min accuracy improvement to keep override
    override_budget: int = 3  # Max rules in an override (keep deltas small)

    # MDL settings
    mdl_rule_base_cost: float = 1.0
    mdl_predicate_cost: float = 0.5
    mdl_exception_cost: float = 2.0

    # Sector/stage field names
    sector_field: str = "deal.sector"
    stage_field: str = "deal.stage"


class HierarchicalLearningEngine:
    """
    Engine for learning hierarchical policies from historical decisions.

    Process:
    1. Learn global policy from all data
    2. Partition data by sector/stage
    3. For each partition with enough data:
       - Evaluate global policy on partition
       - Learn partition-specific override (limited budget)
       - Keep override only if significant improvement
    """

    def __init__(self, config: Optional[HierarchicalLearningConfig] = None):
        """Initialize the engine."""
        self.config = config or HierarchicalLearningConfig()
        self.scorer = MDLScorer(
            rule_base_cost=self.config.mdl_rule_base_cost,
            predicate_cost=self.config.mdl_predicate_cost,
            exception_cost=self.config.mdl_exception_cost,
        )

    def partition_dataset(
        self,
        dataset: DecisionDataset,
    ) -> dict[str, Partition]:
        """
        Partition dataset by sector and stage.

        Returns dict of partition_name -> Partition.
        """
        partitions: dict[str, Partition] = {}

        for hist in dataset.decisions:
            # Extract sector and stage from context
            sector_fv = hist.context.get_field(self.config.sector_field)
            stage_fv = hist.context.get_field(self.config.stage_field)

            sector = sector_fv.value if sector_fv.exists else None
            stage = stage_fv.value if stage_fv.exists else None

            # Normalize
            if sector and isinstance(sector, str):
                sector = sector.lower()
            if stage and isinstance(stage, str):
                stage = stage.lower()

            # Add to sector partition
            if sector and PartitionKey.SECTOR in self.config.partition_keys:
                key = f"{PartitionKey.SECTOR.value}:{sector}"
                if key not in partitions:
                    partitions[key] = Partition(
                        key=PartitionKey.SECTOR,
                        value=sector,
                    )
                partitions[key].decisions.append(hist)

            # Add to stage partition
            if stage and PartitionKey.STAGE in self.config.partition_keys:
                key = f"{PartitionKey.STAGE.value}:{stage}"
                if key not in partitions:
                    partitions[key] = Partition(
                        key=PartitionKey.STAGE,
                        value=stage,
                    )
                partitions[key].decisions.append(hist)

            # Add to combined partition
            if (
                sector and stage
                and PartitionKey.SECTOR_STAGE in self.config.partition_keys
            ):
                key = f"{PartitionKey.SECTOR_STAGE.value}:{sector}_{stage}"
                if key not in partitions:
                    partitions[key] = Partition(
                        key=PartitionKey.SECTOR_STAGE,
                        value=f"{sector}_{stage}",
                    )
                partitions[key].decisions.append(hist)

        # Filter out small partitions
        partitions = {
            k: v for k, v in partitions.items()
            if len(v) >= self.config.min_partition_size
        }

        logger.info(
            f"Created {len(partitions)} partitions from {len(dataset)} decisions"
        )
        for name, part in partitions.items():
            logger.debug(f"  {name}: {len(part)} decisions")

        return partitions

    def learn_global_policy(
        self,
        dataset: DecisionDataset,
        candidate_rules: list[list[Rule]],
    ) -> tuple[list[Rule], CoverageStats, MDLScoreBreakdown]:
        """
        Learn global policy from all data.

        Uses the best candidate rule set (lowest MDL score).
        """
        best_rules: list[Rule] = []
        best_stats = CoverageStats()
        best_breakdown = MDLScoreBreakdown()
        best_score = float("inf")

        for rules in candidate_rules:
            stats, exceptions, breakdown = self.scorer.score_coverage(rules, dataset)

            # Check minimum requirements
            if stats.coverage_rate < self.config.global_min_coverage:
                continue
            if stats.accuracy < self.config.global_min_accuracy:
                continue

            if breakdown.total_score < best_score:
                best_score = breakdown.total_score
                best_rules = rules
                best_stats = stats
                best_breakdown = breakdown

        logger.info(
            f"Global policy: {len(best_rules)} rules, "
            f"coverage={best_stats.coverage_rate:.2%}, "
            f"accuracy={best_stats.accuracy:.2%}"
        )

        return best_rules, best_stats, best_breakdown

    def learn_override(
        self,
        partition: Partition,
        global_rules: list[Rule],
        candidate_rules: list[Rule],
    ) -> Optional[PolicyOverride]:
        """
        Learn a policy override for a partition.

        Tries to improve on global policy with limited budget.
        """
        dataset = partition.to_dataset()

        # Evaluate global policy on this partition
        global_stats, _, _ = self.scorer.score_coverage(global_rules, dataset)
        global_accuracy = global_stats.accuracy

        logger.debug(
            f"Partition {partition.name}: global accuracy = {global_accuracy:.2%}"
        )

        # Try different override strategies
        best_override: Optional[PolicyOverride] = None
        best_improvement = 0.0

        # Strategy 1: Add rules
        for rule in candidate_rules[:self.config.override_budget]:
            test_rules = list(global_rules) + [rule]
            stats, exceptions, breakdown = self.scorer.score_coverage(test_rules, dataset)

            improvement = stats.accuracy - global_accuracy
            if improvement > best_improvement:
                best_improvement = improvement
                best_override = PolicyOverride(
                    partition_name=partition.name,
                    partition_key=partition.key,
                    partition_value=partition.value,
                    add_rules=[rule],
                    coverage_stats=stats,
                    exceptions=exceptions,
                    mdl_score=breakdown.total_score,
                    sample_count=len(partition),
                    improvement_over_global=improvement,
                )

        # Strategy 2: Replace rules with different thresholds
        for i, global_rule in enumerate(global_rules):
            for candidate in candidate_rules:
                # Check if candidate targets same field
                global_fields = set(global_rule.predicate.get_fields())
                candidate_fields = set(candidate.predicate.get_fields())

                if global_fields != candidate_fields:
                    continue

                # Try replacement
                test_rules = list(global_rules)
                test_rules[i] = candidate

                stats, exceptions, breakdown = self.scorer.score_coverage(test_rules, dataset)

                improvement = stats.accuracy - global_accuracy
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_override = PolicyOverride(
                        partition_name=partition.name,
                        partition_key=partition.key,
                        partition_value=partition.value,
                        replace_rules={global_rule.rule_id: candidate},
                        coverage_stats=stats,
                        exceptions=exceptions,
                        mdl_score=breakdown.total_score,
                        sample_count=len(partition),
                        improvement_over_global=improvement,
                    )

        # Strategy 3: Disable rules that hurt this partition
        for i, global_rule in enumerate(global_rules):
            test_rules = [r for j, r in enumerate(global_rules) if j != i]

            if not test_rules:
                continue

            stats, exceptions, breakdown = self.scorer.score_coverage(test_rules, dataset)

            improvement = stats.accuracy - global_accuracy
            if improvement > best_improvement:
                best_improvement = improvement
                best_override = PolicyOverride(
                    partition_name=partition.name,
                    partition_key=partition.key,
                    partition_value=partition.value,
                    disable_rules=[global_rule.rule_id],
                    coverage_stats=stats,
                    exceptions=exceptions,
                    mdl_score=breakdown.total_score,
                    sample_count=len(partition),
                    improvement_over_global=improvement,
                )

        # Only keep override if improvement is significant
        if best_override and best_improvement >= self.config.override_min_improvement:
            logger.info(
                f"Learned override for {partition.name}: "
                f"+{best_improvement:.2%} accuracy"
            )
            return best_override

        logger.debug(
            f"No significant override for {partition.name} "
            f"(best improvement: {best_improvement:.2%})"
        )
        return None

    def learn_hierarchical_policy(
        self,
        dataset: DecisionDataset,
        global_candidate_rules: list[list[Rule]],
        override_candidate_rules: list[Rule],
    ) -> HierarchicalPolicy:
        """
        Learn a complete hierarchical policy.

        Args:
            dataset: Historical decisions
            global_candidate_rules: Candidate rule sets for global policy
            override_candidate_rules: Individual rules for overrides

        Returns:
            HierarchicalPolicy with global rules and partition overrides
        """
        logger.info(f"Learning hierarchical policy from {len(dataset)} decisions")

        # Step 1: Learn global policy
        global_rules, global_stats, global_breakdown = self.learn_global_policy(
            dataset, global_candidate_rules
        )

        if not global_rules:
            logger.warning("Could not learn global policy")
            return HierarchicalPolicy(
                name="Empty Policy",
                sector_field=self.config.sector_field,
                stage_field=self.config.stage_field,
            )

        # Step 2: Partition data
        partitions = self.partition_dataset(dataset)

        # Step 3: Learn overrides for each partition
        overrides: dict[str, PolicyOverride] = {}

        for name, partition in partitions.items():
            override = self.learn_override(
                partition=partition,
                global_rules=global_rules,
                candidate_rules=override_candidate_rules,
            )
            if override:
                overrides[name] = override

        # Build policy
        policy = HierarchicalPolicy(
            name="Learned Hierarchical Policy",
            global_rules=global_rules,
            global_coverage_stats=global_stats,
            global_mdl_score=global_breakdown.total_score,
            overrides=overrides,
            sector_field=self.config.sector_field,
            stage_field=self.config.stage_field,
        )

        logger.info(
            f"Hierarchical policy: {len(global_rules)} global rules, "
            f"{len(overrides)} partition overrides"
        )

        return policy


# =============================================================================
# Convenience Functions
# =============================================================================


def learn_hierarchical_policy(
    dataset: DecisionDataset,
    global_candidate_rules: list[list[Rule]],
    override_candidate_rules: list[Rule],
    config: Optional[HierarchicalLearningConfig] = None,
) -> HierarchicalPolicy:
    """
    Convenience function to learn a hierarchical policy.

    Args:
        dataset: Historical decisions
        global_candidate_rules: Candidate rule sets for global policy
        override_candidate_rules: Individual rules for overrides
        config: Learning configuration

    Returns:
        HierarchicalPolicy
    """
    engine = HierarchicalLearningEngine(config=config)
    return engine.learn_hierarchical_policy(
        dataset=dataset,
        global_candidate_rules=global_candidate_rules,
        override_candidate_rules=override_candidate_rules,
    )


def evaluate_with_hierarchy(
    policy: HierarchicalPolicy,
    context: EvalContext,
    default_decision: Decision = Decision.DEFER,
) -> HierarchicalEvaluationResult:
    """
    Evaluate a deal under a hierarchical policy.

    Args:
        policy: Hierarchical policy
        context: Deal context
        default_decision: Default decision if no rules match

    Returns:
        HierarchicalEvaluationResult with decision and override info
    """
    return policy.evaluate(context, default_decision=default_decision)


def summarize_policy(policy: HierarchicalPolicy) -> dict[str, Any]:
    """
    Generate a human-readable summary of a hierarchical policy.

    Returns:
        Summary dict with global rules, overrides, and statistics
    """
    summary = {
        "name": policy.name,
        "global_policy": {
            "num_rules": len(policy.global_rules),
            "rules": [
                f"{r.name}: {r.predicate.to_dsl()} -> {r.decision.value}"
                for r in policy.global_rules
            ],
            "coverage": policy.global_coverage_stats.coverage_rate,
            "accuracy": policy.global_coverage_stats.accuracy,
        },
        "overrides": {},
    }

    for name, override in policy.overrides.items():
        override_summary = {
            "type": override.partition_key.value,
            "value": override.partition_value,
            "sample_count": override.sample_count,
            "improvement": override.improvement_over_global,
            "changes": [],
        }

        for rule in override.add_rules:
            override_summary["changes"].append(
                f"ADD: {rule.name}: {rule.predicate.to_dsl()}"
            )

        for rule_id, rule in override.replace_rules.items():
            override_summary["changes"].append(
                f"REPLACE {rule_id}: {rule.predicate.to_dsl()}"
            )

        for rule_id in override.disable_rules:
            override_summary["changes"].append(f"DISABLE: {rule_id}")

        summary["overrides"][name] = override_summary

    return summary
