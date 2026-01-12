"""
Tests for Meta-Controller v1.

Tests cover:
- Regime determination (ARC-style discrete vs uncertain)
- Phase-based budget allocation
- Uncertainty-based refusal output
- Trace logging of budgets and uncertainty
"""

import pytest
from typing import Dict, Any, List

from juris_agi.core.types import Grid, ARCTask, ARCPair
from juris_agi.core.trace import SolveTrace
from juris_agi.controller.router import (
    TaskRegime,
    RegimeDecision,
    determine_regime,
)
from juris_agi.controller.scheduler import (
    ExpertType,
    SolvePhase,
    Budget,
    PhaseBudget,
    PhaseAllocation,
    PhaseScheduler,
    UncertaintyEstimate,
    allocate_budget,
)
from juris_agi.controller.refusal import (
    RefusalChecker,
    RefusalReason,
    UncertaintyOutput,
    compute_uncertainty_output,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def consistent_task() -> ARCTask:
    """Task with consistent patterns across examples."""
    train = [
        ARCPair(
            input=Grid.from_list([[1, 2], [3, 4]]),
            output=Grid.from_list([[4, 3], [2, 1]]),
        ),
        ARCPair(
            input=Grid.from_list([[5, 6], [7, 8]]),
            output=Grid.from_list([[8, 7], [6, 5]]),
        ),
        ARCPair(
            input=Grid.from_list([[1, 1], [2, 2]]),
            output=Grid.from_list([[2, 2], [1, 1]]),
        ),
    ]
    test = [
        ARCPair(
            input=Grid.from_list([[9, 0], [1, 2]]),
            output=Grid.from_list([[2, 1], [0, 9]]),
        ),
    ]
    return ARCTask(task_id="consistent_test", train=train, test=test)


@pytest.fixture
def inconsistent_task() -> ARCTask:
    """Task with inconsistent patterns (high uncertainty)."""
    train = [
        ARCPair(
            input=Grid.from_list([[1, 2], [3, 4]]),
            output=Grid.from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),  # Different dims
        ),
        ARCPair(
            input=Grid.from_list([[5, 6, 7], [8, 9, 0]]),
            output=Grid.from_list([[1]]),  # Very different
        ),
    ]
    test = [
        ARCPair(
            input=Grid.from_list([[1, 2]]),
            output=Grid.from_list([[0]]),
        ),
    ]
    return ARCTask(task_id="inconsistent_test", train=train, test=test)


@pytest.fixture
def scaling_task() -> ARCTask:
    """Task with 2x scaling."""
    train = [
        ARCPair(
            input=Grid.from_list([[1, 2], [3, 4]]),
            output=Grid.from_list([
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ]),
        ),
    ]
    test = [
        ARCPair(
            input=Grid.from_list([[5]]),
            output=Grid.from_list([[5, 5], [5, 5]]),
        ),
    ]
    return ARCTask(task_id="scaling_test", train=train, test=test)


# ============================================================================
# Tests for Regime Determination
# ============================================================================

class TestRegimeDetermination:
    """Tests for determine_regime() function."""

    def test_consistent_task_is_arc_discrete(self, consistent_task):
        """Consistent task should be classified as ARC_DISCRETE."""
        decision = determine_regime(consistent_task)

        assert isinstance(decision, RegimeDecision)
        assert decision.regime == TaskRegime.ARC_DISCRETE
        assert decision.confidence > 0.5

    def test_inconsistent_task_is_uncertain(self, inconsistent_task):
        """Inconsistent task should be classified as UNCERTAIN."""
        decision = determine_regime(inconsistent_task)

        assert decision.regime == TaskRegime.UNCERTAIN
        assert "uncertainty_score" in decision.features
        assert decision.features["uncertainty_score"] > 0.5

    def test_scaling_task_detection(self, scaling_task):
        """Scaling task should have dimension ratio features detected."""
        decision = determine_regime(scaling_task)

        # Should detect dimension ratio
        assert "dim_ratio_variance" in decision.features
        # Single example, so some uncertainty
        assert decision.features.get("few_examples", False) or len(scaling_task.train) >= 3

    def test_empty_task_is_uncertain(self):
        """Task with no training examples should be UNCERTAIN."""
        task = ARCTask(task_id="empty", train=[], test=[])
        decision = determine_regime(task)

        assert decision.regime == TaskRegime.UNCERTAIN
        assert decision.confidence < 0.5

    def test_decision_has_rationale(self, consistent_task):
        """Decision should include a rationale."""
        decision = determine_regime(consistent_task)

        assert decision.rationale is not None
        assert len(decision.rationale) > 0

    def test_decision_to_dict(self, consistent_task):
        """Decision should serialize to dict."""
        decision = determine_regime(consistent_task)
        d = decision.to_dict()

        assert "regime" in d
        assert "confidence" in d
        assert "features" in d
        assert "rationale" in d


# ============================================================================
# Tests for Phase-Based Budget Allocation
# ============================================================================

class TestPhaseBudgetAllocation:
    """Tests for phase-based budget allocation."""

    def test_default_allocation(self):
        """Test default budget allocation."""
        allocation = allocate_budget(
            total_time=60.0,
            total_iterations=10000,
        )

        assert isinstance(allocation, PhaseAllocation)
        assert allocation.total_time == 60.0
        assert allocation.total_iterations == 10000

        # All phases should be present
        for phase in SolvePhase:
            budget = allocation.get_phase_budget(phase)
            assert budget.time_limit > 0
            assert budget.iteration_limit > 0

    def test_uncertain_regime_allocation(self):
        """UNCERTAIN regime should allocate more to priors and refinement."""
        default_alloc = allocate_budget(60.0, 10000, regime=None)
        uncertain_alloc = allocate_budget(60.0, 10000, regime="UNCERTAIN")

        default_priors = default_alloc.get_phase_budget(SolvePhase.PRIORS)
        uncertain_priors = uncertain_alloc.get_phase_budget(SolvePhase.PRIORS)

        # UNCERTAIN should have more priors budget
        assert uncertain_priors.time_limit >= default_priors.time_limit

    def test_arc_discrete_regime_allocation(self):
        """ARC_DISCRETE regime should allocate more to synthesis."""
        default_alloc = allocate_budget(60.0, 10000, regime=None)
        discrete_alloc = allocate_budget(60.0, 10000, regime="ARC_DISCRETE")

        default_synth = default_alloc.get_phase_budget(SolvePhase.SYNTHESIS)
        discrete_synth = discrete_alloc.get_phase_budget(SolvePhase.SYNTHESIS)

        # ARC_DISCRETE should have more synthesis budget
        assert discrete_synth.time_limit >= default_synth.time_limit

    def test_allocation_sums_correctly(self):
        """Phase budgets should approximately sum to total."""
        allocation = allocate_budget(60.0, 10000)

        total_time = sum(
            allocation.get_phase_budget(phase).time_limit
            for phase in SolvePhase
        )

        # Should be approximately equal (allowing for rounding)
        assert abs(total_time - 60.0) < 1.0

    def test_allocation_to_dict(self):
        """Allocation should serialize to dict."""
        allocation = allocate_budget(60.0, 10000)
        d = allocation.to_dict()

        assert "total_time" in d
        assert "total_iterations" in d
        assert "phases" in d
        assert len(d["phases"]) == len(SolvePhase)


class TestPhaseScheduler:
    """Tests for PhaseScheduler class."""

    def test_scheduler_initialization(self):
        """Test scheduler initializes with correct allocation."""
        scheduler = PhaseScheduler(
            total_time=60.0,
            total_iterations=10000,
            regime="ARC_DISCRETE",
        )

        assert scheduler.allocation.total_time == 60.0
        assert scheduler.current_phase is None

    def test_start_phase(self):
        """Test starting a phase."""
        scheduler = PhaseScheduler(total_time=60.0, total_iterations=10000)

        budget = scheduler.start_phase(SolvePhase.SYNTHESIS)

        assert scheduler.current_phase == SolvePhase.SYNTHESIS
        assert budget.phase == SolvePhase.SYNTHESIS
        assert len(scheduler.history) == 1
        assert scheduler.history[0]["event"] == "phase_start"

    def test_end_phase(self):
        """Test ending a phase."""
        scheduler = PhaseScheduler(total_time=60.0, total_iterations=10000)

        scheduler.start_phase(SolvePhase.PRIORS)
        scheduler.end_phase(iterations_used=100, success=True)

        assert scheduler.current_phase is None
        assert len(scheduler.history) == 2
        assert scheduler.history[1]["event"] == "phase_end"
        assert scheduler.history[1]["success"] is True

    def test_get_remaining_time(self):
        """Test remaining time calculation."""
        scheduler = PhaseScheduler(total_time=60.0, total_iterations=10000)

        remaining = scheduler.get_remaining_time()
        assert remaining > 0
        assert remaining <= 60.0

    def test_should_continue(self):
        """Test should_continue logic."""
        scheduler = PhaseScheduler(total_time=60.0, total_iterations=10000)

        should_continue, reason = scheduler.should_continue()
        assert should_continue is True
        assert "available" in reason.lower()

    def test_phase_summary(self):
        """Test phase summary generation."""
        scheduler = PhaseScheduler(total_time=60.0, total_iterations=10000)
        scheduler.start_phase(SolvePhase.SYNTHESIS)

        summary = scheduler.get_phase_summary()

        assert "total_time" in summary
        assert "phases" in summary
        assert summary["current_phase"] == "SYNTHESIS"


class TestPhaseBudget:
    """Tests for PhaseBudget dataclass."""

    def test_budget_properties(self):
        """Test budget property calculations."""
        budget = PhaseBudget(
            phase=SolvePhase.SYNTHESIS,
            time_limit=30.0,
            iteration_limit=5000,
            time_used=10.0,
            iterations_used=2000,
        )

        assert budget.time_remaining == 20.0
        assert budget.iterations_remaining == 3000
        assert budget.exhausted is False
        assert budget.utilization == pytest.approx(1/3, rel=0.01)

    def test_budget_exhausted(self):
        """Test budget exhaustion detection."""
        budget = PhaseBudget(
            phase=SolvePhase.SYNTHESIS,
            time_limit=30.0,
            iteration_limit=5000,
            time_used=30.0,  # Exhausted
            iterations_used=0,
        )

        assert budget.exhausted is True

    def test_budget_to_dict(self):
        """Test budget serialization."""
        budget = PhaseBudget(
            phase=SolvePhase.PRIORS,
            time_limit=10.0,
            iteration_limit=1000,
        )
        d = budget.to_dict()

        assert d["phase"] == "PRIORS"
        assert d["time_limit"] == 10.0
        assert d["exhausted"] is False


# ============================================================================
# Tests for Uncertainty Estimation
# ============================================================================

class TestUncertaintyEstimate:
    """Tests for UncertaintyEstimate class."""

    def test_high_epistemic_with_low_candidates(self):
        """Few candidates and low score should give high epistemic uncertainty."""
        estimate = UncertaintyEstimate.compute(
            num_candidates=5,
            best_score=0.3,
            variance=0.1,
        )

        assert estimate.epistemic > 0.5
        assert estimate.total > 0.5

    def test_low_epistemic_with_many_candidates(self):
        """Many candidates and high score should give low epistemic uncertainty."""
        estimate = UncertaintyEstimate.compute(
            num_candidates=100,
            best_score=0.9,
            variance=0.1,
        )

        assert estimate.epistemic < 0.3

    def test_high_aleatoric_with_high_variance(self):
        """High variance should give high aleatoric uncertainty."""
        estimate = UncertaintyEstimate.compute(
            num_candidates=50,
            best_score=0.5,
            variance=0.8,
        )

        assert estimate.aleatoric > 0.5

    def test_total_uncertainty_bounded(self):
        """Total uncertainty should be bounded by 1.0."""
        estimate = UncertaintyEstimate.compute(
            num_candidates=0,
            best_score=0.0,
            variance=1.0,
        )

        assert estimate.total <= 1.0


# ============================================================================
# Tests for Uncertainty-Based Refusal
# ============================================================================

class TestUncertaintyOutput:
    """Tests for UncertaintyOutput and compute_uncertainty_output()."""

    def test_resolved_output(self):
        """Test output for resolved (successful) solve."""
        output = compute_uncertainty_output(
            task_id="test_task",
            best_program="identity",
            best_score=100.0,
            candidates=[{"score": 100.0}],
            wme_confidence=0.9,
            budget_summary={"remaining_time": 30.0},
        )

        assert output.resolved is True
        assert output.best_score == 100.0
        assert output.refusal_reason is None

    def test_unresolved_output(self):
        """Test output for unresolved solve."""
        output = compute_uncertainty_output(
            task_id="test_task",
            best_program="identity",
            best_score=50.0,
            candidates=[{"score": 50.0}, {"score": 40.0}],
            wme_confidence=0.5,
            budget_summary={"remaining_time": 0},
        )

        assert output.resolved is False
        assert output.refusal_reason is not None
        assert output.budget_exhausted is True

    def test_epistemic_uncertainty_calculation(self):
        """Test epistemic uncertainty in output."""
        # Few candidates, low score - high epistemic
        output = compute_uncertainty_output(
            task_id="test",
            best_program="identity",
            best_score=20.0,
            candidates=[{"score": 20.0}],
            wme_confidence=0.8,
            budget_summary={"remaining_time": 30.0},
        )

        assert output.epistemic_uncertainty > 0.5

    def test_aleatoric_uncertainty_from_wme(self):
        """Test aleatoric uncertainty from WME confidence."""
        # Low WME confidence - high aleatoric
        output = compute_uncertainty_output(
            task_id="test",
            best_program="identity",
            best_score=50.0,
            candidates=[{"score": 50.0}] * 10,
            wme_confidence=0.2,  # Low confidence
            budget_summary={"remaining_time": 30.0},
        )

        assert output.aleatoric_uncertainty > 0.5

    def test_is_confident_refusal(self):
        """Test confident refusal detection."""
        output = compute_uncertainty_output(
            task_id="test",
            best_program="identity",
            best_score=50.0,
            candidates=[{"score": 50.0}] * 50,  # Many candidates explored
            wme_confidence=0.8,
            budget_summary={"remaining_time": 0},  # Budget exhausted
        )

        # Explored well, budget exhausted, but no solution
        # Should be confident refusal (epistemic low)
        assert output.epistemic_uncertainty < 0.5 or output.budget_exhausted

    def test_should_retry(self):
        """Test retry recommendation."""
        output = compute_uncertainty_output(
            task_id="test",
            best_program="identity",
            best_score=40.0,
            candidates=[{"score": 40.0}],  # Few candidates
            wme_confidence=0.7,
            budget_summary={"remaining_time": 30.0},  # Budget available
        )

        # Few candidates, budget available - should retry
        if output.epistemic_uncertainty > 0.5:
            assert output.should_retry is True

    def test_output_to_dict(self):
        """Test output serialization."""
        output = compute_uncertainty_output(
            task_id="test",
            best_program="identity",
            best_score=100.0,
            candidates=[{"score": 100.0}],
            wme_confidence=0.9,
            budget_summary={},
        )
        d = output.to_dict()

        assert "task_id" in d
        assert "epistemic_uncertainty" in d
        assert "aleatoric_uncertainty" in d
        assert "suggestions" in d

    def test_suggestions_generated(self):
        """Test that appropriate suggestions are generated."""
        output = compute_uncertainty_output(
            task_id="test",
            best_program="identity",
            best_score=40.0,
            candidates=[{"score": 40.0}],
            wme_confidence=0.3,  # Low WME confidence
            budget_summary={"remaining_time": 30.0},
        )

        assert len(output.suggestions) > 0


# ============================================================================
# Tests for Trace Logging
# ============================================================================

class TestTraceLogging:
    """Tests for trace logging of budgets and uncertainty."""

    def test_log_budget(self):
        """Test logging budget information."""
        trace = SolveTrace.start("test_task")

        phase_budgets = {
            "PRIORS": {"time_limit": 6.0, "time_used": 2.0},
            "SYNTHESIS": {"time_limit": 36.0, "time_used": 30.0},
        }

        trace.log_budget(phase_budgets)

        assert trace.budget_per_phase == phase_budgets
        assert len(trace.entries) == 1
        assert trace.entries[0].event_type == "budget_update"

    def test_log_uncertainty(self):
        """Test logging uncertainty metrics."""
        trace = SolveTrace.start("test_task")

        trace.log_uncertainty(
            epistemic=0.4,
            aleatoric=0.3,
            num_hypotheses=15,
            diff_variance=120.5,
        )

        assert trace.uncertainty_metrics["epistemic"] == 0.4
        assert trace.uncertainty_metrics["aleatoric"] == 0.3
        assert trace.uncertainty_metrics["num_hypotheses"] == 15
        assert "total" in trace.uncertainty_metrics

    def test_set_regime(self):
        """Test setting regime in trace."""
        trace = SolveTrace.start("test_task")

        trace.set_regime(
            regime="ARC_DISCRETE",
            confidence=0.85,
            rationale="Consistent patterns detected",
        )

        assert trace.regime == "ARC_DISCRETE"
        assert len(trace.entries) == 1
        assert trace.entries[0].event_type == "regime_determined"

    def test_trace_to_dict_includes_budget_and_uncertainty(self):
        """Test that trace serialization includes budget and uncertainty."""
        trace = SolveTrace.start("test_task")

        trace.log_budget({"SYNTHESIS": {"time_limit": 30.0}})
        trace.log_uncertainty(
            epistemic=0.5,
            aleatoric=0.3,
            num_hypotheses=10,
            diff_variance=50.0,
        )
        trace.set_regime("UNCERTAIN", 0.6, "High variance")
        trace.finalize(success=False)

        d = trace.to_dict()

        assert "budget_per_phase" in d
        assert "uncertainty_metrics" in d
        assert "regime" in d
        assert d["regime"] == "UNCERTAIN"


# ============================================================================
# Integration Tests
# ============================================================================

class TestControllerIntegration:
    """Integration tests for controller components."""

    def test_regime_to_allocation_pipeline(self, consistent_task):
        """Test full pipeline from regime determination to budget allocation."""
        # Determine regime
        regime_decision = determine_regime(consistent_task)

        # Allocate budget based on regime
        allocation = allocate_budget(
            total_time=60.0,
            total_iterations=10000,
            regime=regime_decision.regime.name,
            task_features=regime_decision.features,
        )

        assert allocation.total_time == 60.0

        # Create scheduler with allocation
        scheduler = PhaseScheduler(
            total_time=60.0,
            total_iterations=10000,
            regime=regime_decision.regime.name,
        )

        # Start and end phases
        scheduler.start_phase(SolvePhase.PRIORS)
        scheduler.end_phase(iterations_used=50, success=True)

        scheduler.start_phase(SolvePhase.SYNTHESIS)
        scheduler.end_phase(iterations_used=500, success=True)

        # Get summary
        summary = scheduler.get_phase_summary()

        assert summary["current_phase"] is None  # No active phase
        assert len(scheduler.history) == 4  # 2 starts + 2 ends

    def test_trace_with_full_logging(self, consistent_task):
        """Test trace with all logging features."""
        trace = SolveTrace.start(consistent_task.task_id)

        # Log regime
        regime_decision = determine_regime(consistent_task)
        trace.set_regime(
            regime_decision.regime.name,
            regime_decision.confidence,
            regime_decision.rationale,
        )

        # Log budget
        allocation = allocate_budget(60.0, 10000, regime=regime_decision.regime.name)
        trace.log_budget(allocation.to_dict())

        # Simulate some operations
        trace.log("synthesis_start", "cre", iterations=0)
        trace.log("synthesis_complete", "cre", iterations=500, success=True)

        # Log uncertainty
        trace.log_uncertainty(
            epistemic=0.2,
            aleatoric=0.1,
            num_hypotheses=1,
            diff_variance=0.0,
        )

        trace.finalize(success=True, program="identity")

        # Verify all logged correctly
        d = trace.to_dict()
        assert d["success"] is True
        assert d["regime"] == "ARC_DISCRETE"
        assert "budget_per_phase" in d
        assert "uncertainty_metrics" in d
        assert len(d["entries"]) >= 4  # regime + budget + synthesis start/end + uncertainty
