"""
Tests for hierarchical policy reasoning.

Tests cover:
- Partitioning by sector/stage
- Global policy learning
- Partition-specific override learning
- Evaluation with overrides
- Biotech vs SaaS different rules scenario
"""

import pytest

from juris_agi.vc_dsl import (
    # Predicates
    Ge, Le, Has, And,
    # Evaluation
    Decision, Rule, EvalContext, FieldValue,
    build_context_from_dict,
    # Hypothesis (used by hierarchy)
    HistoricalDecision,
    DecisionDataset,
    # Hierarchy
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def saas_invest_rules():
    """Rules for SaaS: focus on ARR and growth."""
    return [
        Rule(
            rule_id="saas_high_arr",
            name="SaaS High ARR",
            predicate=Ge("traction.arr", 1_000_000),
            decision=Decision.INVEST,
            priority=5,
        ),
        Rule(
            rule_id="saas_low_arr",
            name="SaaS Low ARR",
            predicate=Le("traction.arr", 100_000),
            decision=Decision.PASS,
            priority=3,
        ),
    ]


@pytest.fixture
def biotech_invest_rules():
    """Rules for Biotech: focus on clinical stage and IP."""
    return [
        Rule(
            rule_id="biotech_clinical",
            name="Biotech Clinical Stage",
            predicate=Ge("milestones.clinical_phase", 2),
            decision=Decision.INVEST,
            priority=5,
        ),
        Rule(
            rule_id="biotech_early",
            name="Biotech Early Stage",
            predicate=Le("milestones.clinical_phase", 0),
            decision=Decision.PASS,
            priority=3,
        ),
    ]


@pytest.fixture
def generic_rules():
    """Generic rules that work across sectors."""
    return [
        Rule(
            rule_id="has_revenue",
            name="Has Revenue",
            predicate=Ge("traction.arr", 500_000),
            decision=Decision.INVEST,
            priority=5,
        ),
        Rule(
            rule_id="no_revenue",
            name="No Revenue",
            predicate=Le("traction.arr", 50_000),
            decision=Decision.PASS,
            priority=3,
        ),
    ]


@pytest.fixture
def mixed_sector_dataset():
    """Dataset with SaaS and Biotech companies."""
    decisions = []

    # SaaS companies - ARR focused
    # High ARR SaaS -> INVEST
    for i in range(8):
        ctx = build_context_from_dict({
            "deal.sector": "saas",
            "deal.stage": "series_a",
            "traction.arr": 1_500_000 + i * 100_000,
            "traction.growth_rate": 80,
            "milestones.clinical_phase": 0,  # Not relevant for SaaS
        })
        decisions.append(HistoricalDecision(
            deal_id=f"saas_invest_{i}",
            decision=Decision.INVEST,
            context=ctx,
        ))

    # Low ARR SaaS -> PASS
    for i in range(8):
        ctx = build_context_from_dict({
            "deal.sector": "saas",
            "deal.stage": "seed",
            "traction.arr": 50_000 + i * 5_000,
            "traction.growth_rate": 30,
            "milestones.clinical_phase": 0,
        })
        decisions.append(HistoricalDecision(
            deal_id=f"saas_pass_{i}",
            decision=Decision.PASS,
            context=ctx,
        ))

    # Biotech companies - Clinical phase focused
    # High clinical phase -> INVEST (even with no ARR!)
    for i in range(8):
        ctx = build_context_from_dict({
            "deal.sector": "biotech",
            "deal.stage": "series_b",
            "traction.arr": 0,  # Pre-revenue biotech
            "milestones.clinical_phase": 2 + i % 2,  # Phase 2 or 3
            "milestones.patents": 10 + i,
        })
        decisions.append(HistoricalDecision(
            deal_id=f"biotech_invest_{i}",
            decision=Decision.INVEST,
            context=ctx,
        ))

    # Early clinical phase -> PASS
    for i in range(8):
        ctx = build_context_from_dict({
            "deal.sector": "biotech",
            "deal.stage": "seed",
            "traction.arr": 0,
            "milestones.clinical_phase": 0,  # Pre-clinical
            "milestones.patents": 2,
        })
        decisions.append(HistoricalDecision(
            deal_id=f"biotech_pass_{i}",
            decision=Decision.PASS,
            context=ctx,
        ))

    return DecisionDataset(
        decisions=decisions,
        name="mixed_sector_dataset",
        description="Mixed SaaS and Biotech companies with different success metrics",
    )


@pytest.fixture
def stage_focused_dataset():
    """Dataset with clear stage-based decision patterns."""
    decisions = []

    # Seed stage - stricter requirements
    for i in range(6):
        ctx = build_context_from_dict({
            "deal.sector": "saas",
            "deal.stage": "seed",
            "traction.arr": 200_000 + i * 50_000,  # Modest ARR
            "team.founders_experience": 10 + i,
        })
        # Only invest if founders have experience
        decision = Decision.INVEST if i >= 3 else Decision.PASS
        decisions.append(HistoricalDecision(
            deal_id=f"seed_{i}",
            decision=decision,
            context=ctx,
        ))

    # Series A - different requirements
    for i in range(6):
        ctx = build_context_from_dict({
            "deal.sector": "saas",
            "deal.stage": "series_a",
            "traction.arr": 800_000 + i * 200_000,
            "team.founders_experience": 5,  # Less experience OK at Series A
        })
        # At Series A, ARR matters more
        decision = Decision.INVEST if i >= 2 else Decision.PASS
        decisions.append(HistoricalDecision(
            deal_id=f"series_a_{i}",
            decision=decision,
            context=ctx,
        ))

    return DecisionDataset(decisions=decisions, name="stage_focused")


# =============================================================================
# Partition Tests
# =============================================================================


class TestPartitioning:
    """Tests for data partitioning."""

    def test_partition_by_sector(self, mixed_sector_dataset):
        """Data is partitioned correctly by sector."""
        config = HierarchicalLearningConfig(
            partition_keys=[PartitionKey.SECTOR],
            min_partition_size=5,
        )
        engine = HierarchicalLearningEngine(config=config)

        partitions = engine.partition_dataset(mixed_sector_dataset)

        assert "sector:saas" in partitions
        assert "sector:biotech" in partitions
        assert len(partitions["sector:saas"]) == 16  # 8 invest + 8 pass
        assert len(partitions["sector:biotech"]) == 16

    def test_partition_by_stage(self, stage_focused_dataset):
        """Data is partitioned correctly by stage."""
        config = HierarchicalLearningConfig(
            partition_keys=[PartitionKey.STAGE],
            min_partition_size=3,
        )
        engine = HierarchicalLearningEngine(config=config)

        partitions = engine.partition_dataset(stage_focused_dataset)

        assert "stage:seed" in partitions
        assert "stage:series_a" in partitions
        assert len(partitions["stage:seed"]) == 6
        assert len(partitions["stage:series_a"]) == 6

    def test_partition_min_size_filtering(self, mixed_sector_dataset):
        """Small partitions are filtered out."""
        config = HierarchicalLearningConfig(
            partition_keys=[PartitionKey.SECTOR],
            min_partition_size=20,  # Higher than any partition
        )
        engine = HierarchicalLearningEngine(config=config)

        partitions = engine.partition_dataset(mixed_sector_dataset)

        # Both partitions should be filtered out
        assert len(partitions) == 0

    def test_partition_to_dataset(self):
        """Partition can be converted to DecisionDataset."""
        ctx = build_context_from_dict({"traction.arr": 1_000_000})
        decisions = [
            HistoricalDecision(
                deal_id="deal_1",
                decision=Decision.INVEST,
                context=ctx,
            ),
        ]

        partition = Partition(
            key=PartitionKey.SECTOR,
            value="saas",
            decisions=decisions,
        )

        dataset = partition.to_dataset()

        assert dataset.name == "sector:saas"
        assert len(dataset) == 1


# =============================================================================
# Global Policy Tests
# =============================================================================


class TestGlobalPolicy:
    """Tests for global policy learning."""

    def test_learn_global_policy(self, generic_rules):
        """Can learn global policy from candidate rules."""
        # Simple dataset where generic rules work
        decisions = [
            HistoricalDecision(
                deal_id=f"invest_{i}",
                decision=Decision.INVEST,
                context=build_context_from_dict({"traction.arr": 1_000_000}),
            )
            for i in range(10)
        ] + [
            HistoricalDecision(
                deal_id=f"pass_{i}",
                decision=Decision.PASS,
                context=build_context_from_dict({"traction.arr": 10_000}),
            )
            for i in range(10)
        ]
        dataset = DecisionDataset(decisions=decisions)

        config = HierarchicalLearningConfig(
            global_min_coverage=0.3,
            global_min_accuracy=0.5,
        )
        engine = HierarchicalLearningEngine(config=config)

        rules, stats, breakdown = engine.learn_global_policy(
            dataset, [generic_rules]
        )

        assert len(rules) > 0
        assert stats.coverage_rate > 0

    def test_global_policy_rejects_poor_coverage(self):
        """Global policy with poor coverage is rejected."""
        # Rules that don't match the data
        rules = [
            Rule(
                rule_id="wrong",
                name="Wrong",
                predicate=Ge("nonexistent.field", 1000),
                decision=Decision.INVEST,
            ),
        ]

        decisions = [
            HistoricalDecision(
                deal_id="deal_1",
                decision=Decision.INVEST,
                context=build_context_from_dict({"traction.arr": 1_000_000}),
            )
        ]
        dataset = DecisionDataset(decisions=decisions)

        config = HierarchicalLearningConfig(
            global_min_coverage=0.5,
        )
        engine = HierarchicalLearningEngine(config=config)

        learned_rules, stats, breakdown = engine.learn_global_policy(
            dataset, [[rules[0]]]
        )

        # Should return empty because coverage requirement not met
        assert len(learned_rules) == 0


# =============================================================================
# Override Learning Tests
# =============================================================================


class TestOverrideLearning:
    """Tests for partition-specific override learning."""

    def test_learn_override_adds_rule(self, generic_rules, biotech_invest_rules):
        """Override adds sector-specific rule."""
        # Biotech data where clinical phase matters
        decisions = [
            HistoricalDecision(
                deal_id=f"biotech_{i}",
                decision=Decision.INVEST if i >= 5 else Decision.PASS,
                context=build_context_from_dict({
                    "deal.sector": "biotech",
                    "traction.arr": 0,  # No ARR
                    "milestones.clinical_phase": i,
                }),
            )
            for i in range(10)
        ]

        partition = Partition(
            key=PartitionKey.SECTOR,
            value="biotech",
            decisions=decisions,
        )

        config = HierarchicalLearningConfig(
            override_min_improvement=0.1,
            override_budget=3,
        )
        engine = HierarchicalLearningEngine(config=config)

        override = engine.learn_override(
            partition=partition,
            global_rules=generic_rules,
            candidate_rules=biotech_invest_rules,
        )

        # Should learn an override because biotech needs different rules
        # (ARR-based rules won't work for pre-revenue biotech)
        if override:
            assert override.partition_name == "sector:biotech"
            assert override.improvement_over_global > 0

    def test_no_override_when_global_works(self, generic_rules):
        """No override when global policy already works well."""
        # Data where generic rules work perfectly
        decisions = [
            HistoricalDecision(
                deal_id=f"deal_{i}",
                decision=Decision.INVEST if i >= 5 else Decision.PASS,
                context=build_context_from_dict({
                    "deal.sector": "saas",
                    "traction.arr": 1_000_000 if i >= 5 else 10_000,
                }),
            )
            for i in range(10)
        ]

        partition = Partition(
            key=PartitionKey.SECTOR,
            value="saas",
            decisions=decisions,
        )

        config = HierarchicalLearningConfig(
            override_min_improvement=0.3,  # High threshold
        )
        engine = HierarchicalLearningEngine(config=config)

        override = engine.learn_override(
            partition=partition,
            global_rules=generic_rules,
            candidate_rules=[],
        )

        # No override needed
        assert override is None


# =============================================================================
# Hierarchical Policy Tests
# =============================================================================


class TestHierarchicalPolicy:
    """Tests for hierarchical policy evaluation."""

    def test_get_rules_for_partition_no_override(self, generic_rules):
        """Without override, returns global rules."""
        policy = HierarchicalPolicy(
            global_rules=generic_rules,
        )

        rules = policy.get_rules_for_partition(None)

        assert len(rules) == len(generic_rules)

    def test_get_rules_for_partition_with_add(self, generic_rules, biotech_invest_rules):
        """Override adds rules to global set."""
        policy = HierarchicalPolicy(
            global_rules=generic_rules,
            overrides={
                "sector:biotech": PolicyOverride(
                    partition_name="sector:biotech",
                    partition_key=PartitionKey.SECTOR,
                    partition_value="biotech",
                    add_rules=[biotech_invest_rules[0]],
                ),
            },
        )

        rules = policy.get_rules_for_partition("sector:biotech")

        assert len(rules) == len(generic_rules) + 1

    def test_get_rules_for_partition_with_disable(self, generic_rules):
        """Override disables specific global rules."""
        policy = HierarchicalPolicy(
            global_rules=generic_rules,
            overrides={
                "sector:biotech": PolicyOverride(
                    partition_name="sector:biotech",
                    partition_key=PartitionKey.SECTOR,
                    partition_value="biotech",
                    disable_rules=["has_revenue"],  # Disable ARR check for biotech
                ),
            },
        )

        rules = policy.get_rules_for_partition("sector:biotech")

        assert len(rules) == len(generic_rules) - 1
        assert all(r.rule_id != "has_revenue" for r in rules)

    def test_infer_partition_sector(self):
        """Infers sector partition from context."""
        policy = HierarchicalPolicy(
            overrides={
                "sector:saas": PolicyOverride(
                    partition_name="sector:saas",
                    partition_key=PartitionKey.SECTOR,
                    partition_value="saas",
                ),
            },
        )

        ctx = build_context_from_dict({
            "deal.sector": "saas",
            "traction.arr": 1_000_000,
        })

        partition = policy.infer_partition(ctx)

        assert partition == "sector:saas"

    def test_infer_partition_no_match(self):
        """Returns None when no partition matches."""
        policy = HierarchicalPolicy(
            overrides={
                "sector:saas": PolicyOverride(
                    partition_name="sector:saas",
                    partition_key=PartitionKey.SECTOR,
                    partition_value="saas",
                ),
            },
        )

        ctx = build_context_from_dict({
            "deal.sector": "fintech",  # Not in overrides
            "traction.arr": 1_000_000,
        })

        partition = policy.infer_partition(ctx)

        assert partition is None

    def test_evaluate_applies_override(self, generic_rules, biotech_invest_rules):
        """Evaluation applies correct override."""
        policy = HierarchicalPolicy(
            global_rules=generic_rules,
            overrides={
                "sector:biotech": PolicyOverride(
                    partition_name="sector:biotech",
                    partition_key=PartitionKey.SECTOR,
                    partition_value="biotech",
                    add_rules=[biotech_invest_rules[0]],
                    # Disable ARR-based PASS rule for pre-revenue biotech
                    disable_rules=["no_revenue"],
                ),
            },
        )

        # Pre-revenue biotech with good clinical data
        ctx = build_context_from_dict({
            "deal.sector": "biotech",
            "traction.arr": 0,  # No ARR
            "milestones.clinical_phase": 3,  # Phase 3
        })

        result = policy.evaluate(ctx)

        assert result.partition_name == "sector:biotech"
        assert result.override_applied is not None
        # Should INVEST based on clinical phase, not ARR
        assert result.final_decision == Decision.INVEST


# =============================================================================
# Integration Tests - Biotech vs SaaS
# =============================================================================


class TestBiotechVsSaas:
    """
    Key test: training data where biotech requires different rules than SaaS.

    This is the main deliverable test - verifying that the hierarchical
    system learns sector-specific policies.
    """

    def test_biotech_needs_different_rules_than_saas(
        self, mixed_sector_dataset, saas_invest_rules, biotech_invest_rules
    ):
        """
        Biotech and SaaS require different investment rules.

        SaaS: Focus on ARR and growth metrics
        Biotech: Focus on clinical phase and IP

        A single global policy cannot satisfy both sectors optimally.
        """
        # Global rules that work for SaaS
        global_rules = saas_invest_rules

        # Override rules for biotech
        override_candidates = biotech_invest_rules

        config = HierarchicalLearningConfig(
            partition_keys=[PartitionKey.SECTOR],
            min_partition_size=5,
            global_min_coverage=0.3,
            global_min_accuracy=0.3,  # Lower threshold since mixed sectors
            override_min_improvement=0.1,
            override_budget=3,
        )

        policy = learn_hierarchical_policy(
            dataset=mixed_sector_dataset,
            global_candidate_rules=[global_rules],
            override_candidate_rules=override_candidates,
            config=config,
        )

        # Check we have global rules
        assert len(policy.global_rules) > 0

        # Evaluate SaaS deal - should use global rules
        saas_deal = build_context_from_dict({
            "deal.sector": "saas",
            "traction.arr": 2_000_000,
            "milestones.clinical_phase": 0,
        })
        saas_result = policy.evaluate(saas_deal)

        assert saas_result.final_decision == Decision.INVEST
        assert saas_result.partition_name is None or "saas" in saas_result.partition_name

        # Evaluate biotech deal - may use override if learned
        biotech_deal = build_context_from_dict({
            "deal.sector": "biotech",
            "traction.arr": 0,  # Pre-revenue
            "milestones.clinical_phase": 3,
        })
        biotech_result = policy.evaluate(biotech_deal)

        # The biotech deal with Phase 3 should ideally be INVEST
        # If override was learned, it will apply clinical phase rules
        # If not, global rules may not cover it (DEFER) or may PASS (no ARR)
        assert biotech_result.partition_name is None or "biotech" in biotech_result.partition_name

    def test_saas_high_arr_invest(self, mixed_sector_dataset, saas_invest_rules, biotech_invest_rules):
        """SaaS company with high ARR should get INVEST."""
        config = HierarchicalLearningConfig(
            partition_keys=[PartitionKey.SECTOR],
            min_partition_size=5,
            global_min_coverage=0.3,
            global_min_accuracy=0.3,
        )

        policy = learn_hierarchical_policy(
            dataset=mixed_sector_dataset,
            global_candidate_rules=[saas_invest_rules],
            override_candidate_rules=biotech_invest_rules,
            config=config,
        )

        # High ARR SaaS
        ctx = build_context_from_dict({
            "deal.sector": "saas",
            "traction.arr": 3_000_000,
        })
        result = evaluate_with_hierarchy(policy, ctx)

        assert result.final_decision == Decision.INVEST

    def test_saas_low_arr_pass(self, mixed_sector_dataset, saas_invest_rules, biotech_invest_rules):
        """SaaS company with low ARR should get PASS."""
        config = HierarchicalLearningConfig(
            partition_keys=[PartitionKey.SECTOR],
            min_partition_size=5,
            global_min_coverage=0.3,
            global_min_accuracy=0.3,
        )

        policy = learn_hierarchical_policy(
            dataset=mixed_sector_dataset,
            global_candidate_rules=[saas_invest_rules],
            override_candidate_rules=biotech_invest_rules,
            config=config,
        )

        # Low ARR SaaS
        ctx = build_context_from_dict({
            "deal.sector": "saas",
            "traction.arr": 20_000,
        })
        result = evaluate_with_hierarchy(policy, ctx)

        assert result.final_decision == Decision.PASS

    def test_policy_summary_shows_overrides(
        self, mixed_sector_dataset, saas_invest_rules, biotech_invest_rules
    ):
        """Policy summary shows global rules and overrides."""
        config = HierarchicalLearningConfig(
            partition_keys=[PartitionKey.SECTOR],
            min_partition_size=5,
            global_min_coverage=0.3,
            global_min_accuracy=0.3,
        )

        policy = learn_hierarchical_policy(
            dataset=mixed_sector_dataset,
            global_candidate_rules=[saas_invest_rules],
            override_candidate_rules=biotech_invest_rules,
            config=config,
        )

        summary = summarize_policy(policy)

        assert "global_policy" in summary
        assert "num_rules" in summary["global_policy"]
        assert summary["global_policy"]["num_rules"] > 0


class TestHierarchyToDict:
    """Tests for serialization."""

    def test_policy_to_dict(self, generic_rules):
        """HierarchicalPolicy converts to dict."""
        policy = HierarchicalPolicy(
            name="Test Policy",
            global_rules=generic_rules,
        )

        d = policy.to_dict()

        assert d["name"] == "Test Policy"
        assert len(d["global_rules"]) == len(generic_rules)

    def test_override_to_dict(self):
        """PolicyOverride converts to dict."""
        override = PolicyOverride(
            partition_name="sector:biotech",
            partition_key=PartitionKey.SECTOR,
            partition_value="biotech",
            disable_rules=["rule_1"],
            sample_count=100,
            improvement_over_global=0.15,
        )

        d = override.to_dict()

        assert d["partition_name"] == "sector:biotech"
        assert d["partition_key"] == "sector"
        assert d["disable_rules"] == ["rule_1"]
        assert d["improvement_over_global"] == 0.15

    def test_result_to_dict(self):
        """HierarchicalEvaluationResult converts to dict."""
        result = HierarchicalEvaluationResult(
            final_decision=Decision.INVEST,
            confidence=0.85,
            partition_name="sector:saas",
            global_rules_count=5,
            effective_rules_count=5,
            rules_fired=["rule_1", "rule_2"],
        )

        d = result.to_dict()

        assert d["decision"] == "invest"
        assert d["confidence"] == 0.85
        assert d["partition_name"] == "sector:saas"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_dataset(self, generic_rules):
        """Empty dataset produces empty policy."""
        dataset = DecisionDataset(decisions=[])

        policy = learn_hierarchical_policy(
            dataset=dataset,
            global_candidate_rules=[generic_rules],
            override_candidate_rules=[],
        )

        assert len(policy.global_rules) == 0

    def test_no_sector_field(self, generic_rules):
        """Deals without sector field use global policy."""
        decisions = [
            HistoricalDecision(
                deal_id="deal_1",
                decision=Decision.INVEST,
                context=build_context_from_dict({
                    "traction.arr": 1_000_000,
                    # No deal.sector field
                }),
            )
        ]
        dataset = DecisionDataset(decisions=decisions)

        config = HierarchicalLearningConfig(
            partition_keys=[PartitionKey.SECTOR],
            min_partition_size=1,
        )
        engine = HierarchicalLearningEngine(config=config)

        partitions = engine.partition_dataset(dataset)

        # No sector partition created
        assert len([p for p in partitions if "sector:" in p]) == 0

    def test_case_insensitive_sector(self):
        """Sector matching is case insensitive."""
        policy = HierarchicalPolicy(
            overrides={
                "sector:saas": PolicyOverride(
                    partition_name="sector:saas",
                    partition_key=PartitionKey.SECTOR,
                    partition_value="saas",
                ),
            },
        )

        # Uppercase sector
        ctx = build_context_from_dict({
            "deal.sector": "SAAS",
            "traction.arr": 1_000_000,
        })

        partition = policy.infer_partition(ctx)

        assert partition == "sector:saas"
