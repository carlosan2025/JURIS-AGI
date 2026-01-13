"""Tests for Predicate Synthesizer with Threshold Proposer integration."""

import pytest

from juris_agi.vc_dsl.search_space import (
    PredicateSynthesizer,
    SynthesizerConfig,
    SynthesisTrace,
    synthesize_predicates,
    PredicateTemplate,
    TemplateType,
    SearchSpaceConfig,
    TRACTION_TEMPLATES,
    FINANCIAL_TEMPLATES,
)
from juris_agi.vc_dsl.predicates_v2 import (
    Predicate,
    Ge,
    Le,
    Between,
    And,
    Has,
)
from juris_agi.vc_dsl.typing import ValueType


class TestPredicateSynthesizer:
    """Tests for PredicateSynthesizer class."""

    def test_synthesizer_creation(self):
        synth = PredicateSynthesizer()
        assert synth.config is not None
        assert synth.observed_data == {}

    def test_synthesizer_with_config(self):
        config = SynthesizerConfig(max_predicates_per_template=5)
        synth = PredicateSynthesizer(config=config)
        assert synth.config.max_predicates_per_template == 5

    def test_synthesizer_with_observed_data(self):
        observed = {
            "traction.arr": [1000000, 2000000, 5000000],
        }
        synth = PredicateSynthesizer(observed_data=observed)
        assert "traction.arr" in synth.observed_data


class TestSynthesizeFromTemplate:
    """Tests for synthesizing predicates from templates."""

    def test_synthesize_existence_template(self):
        template = PredicateTemplate(
            template_id="test_existence",
            template_type=TemplateType.EXISTENCE,
            field="some.field",
        )
        synth = PredicateSynthesizer()
        predicates = synth.synthesize_from_template(template)

        assert len(predicates) == 1
        assert isinstance(predicates[0], Has)

    def test_synthesize_ge_template_with_observed_data(self):
        template = PredicateTemplate(
            template_id="arr_min",
            template_type=TemplateType.THRESHOLD_GE,
            field="traction.arr",
        )
        observed = {"traction.arr": [1000000, 2000000, 5000000]}
        synth = PredicateSynthesizer(observed_data=observed)

        predicates = synth.synthesize_from_template(template)

        assert len(predicates) > 0
        # All predicates should be And with ConfGe + Ge
        for pred in predicates:
            assert isinstance(pred, And)

    def test_synthesize_ge_template_with_domain_heuristics(self):
        template = PredicateTemplate(
            template_id="arr_min",
            template_type=TemplateType.THRESHOLD_GE,
            field="traction.arr",
        )
        synth = PredicateSynthesizer()  # No observed data

        predicates = synth.synthesize_from_template(template)

        # Should still get predicates from domain heuristics
        assert len(predicates) > 0

    def test_synthesize_le_template(self):
        template = PredicateTemplate(
            template_id="burn_max",
            template_type=TemplateType.THRESHOLD_LE,
            field="capital_intensity.monthly_burn",
        )
        synth = PredicateSynthesizer()

        predicates = synth.synthesize_from_template(template)

        assert len(predicates) > 0

    def test_synthesize_range_template(self):
        template = PredicateTemplate(
            template_id="valuation_range",
            template_type=TemplateType.RANGE,
            field="round_terms.pre_money_valuation",
        )
        observed = {"round_terms.pre_money_valuation": [5000000, 10000000, 25000000]}
        synth = PredicateSynthesizer(observed_data=observed)

        predicates = synth.synthesize_from_template(template)

        # Range predicates generate pairs
        assert len(predicates) > 0

    def test_traces_recorded(self):
        template = PredicateTemplate(
            template_id="test_ge",
            template_type=TemplateType.THRESHOLD_GE,
            field="traction.arr",
        )
        synth = PredicateSynthesizer()
        synth.synthesize_from_template(template)

        traces = synth.get_traces()
        assert len(traces) == 1
        assert traces[0].template_id == "test_ge"
        assert traces[0].field == "traction.arr"


class TestSynthesizerThresholdCaching:
    """Tests for threshold caching in synthesizer."""

    def test_thresholds_are_cached(self):
        template1 = PredicateTemplate(
            template_id="arr_min1",
            template_type=TemplateType.THRESHOLD_GE,
            field="traction.arr",
        )
        template2 = PredicateTemplate(
            template_id="arr_min2",
            template_type=TemplateType.THRESHOLD_GE,
            field="traction.arr",
        )

        synth = PredicateSynthesizer()
        synth.synthesize_from_template(template1)
        synth.synthesize_from_template(template2)

        # Same field should use cached thresholds
        assert "traction.arr" in synth._threshold_cache

    def test_cache_clear(self):
        template = PredicateTemplate(
            template_id="arr_min",
            template_type=TemplateType.THRESHOLD_GE,
            field="traction.arr",
        )
        synth = PredicateSynthesizer()
        synth.synthesize_from_template(template)

        assert len(synth._threshold_cache) > 0

        synth.clear_cache()
        assert len(synth._threshold_cache) == 0


class TestSynthesizeAll:
    """Tests for synthesizing from all templates."""

    def test_synthesize_all_default(self):
        synth = PredicateSynthesizer()
        predicates = synth.synthesize_all()

        # Should get predicates from default templates
        assert len(predicates) > 0

    def test_synthesize_all_with_config(self):
        config = SearchSpaceConfig(
            include_thresholds=True,
            include_ranges=False,
            include_trends=False,
        )
        synth = PredicateSynthesizer()
        predicates = synth.synthesize_all(config=config)

        assert len(predicates) > 0

    def test_synthesize_specific_templates(self):
        synth = PredicateSynthesizer()
        predicates = synth.synthesize_all(templates=TRACTION_TEMPLATES[:2])

        traces = synth.get_traces()
        assert len(traces) == 2


class TestSynthesizerConfig:
    """Tests for SynthesizerConfig options."""

    def test_disable_ge(self):
        config = SynthesizerConfig(synthesize_ge=False)
        template = PredicateTemplate(
            template_id="test",
            template_type=TemplateType.THRESHOLD_GE,
            field="field",
        )
        synth = PredicateSynthesizer(config=config)
        predicates = synth.synthesize_from_template(template)

        assert len(predicates) == 0

    def test_disable_le(self):
        config = SynthesizerConfig(synthesize_le=False)
        template = PredicateTemplate(
            template_id="test",
            template_type=TemplateType.THRESHOLD_LE,
            field="field",
        )
        synth = PredicateSynthesizer(config=config)
        predicates = synth.synthesize_from_template(template)

        assert len(predicates) == 0

    def test_disable_between(self):
        config = SynthesizerConfig(synthesize_between=False)
        template = PredicateTemplate(
            template_id="test",
            template_type=TemplateType.RANGE,
            field="field",
        )
        observed = {"field": [100, 200, 300]}
        synth = PredicateSynthesizer(config=config, observed_data=observed)
        predicates = synth.synthesize_from_template(template)

        assert len(predicates) == 0


class TestTraceSummary:
    """Tests for trace summary generation."""

    def test_get_trace_summary(self):
        synth = PredicateSynthesizer()
        synth.synthesize_all(templates=TRACTION_TEMPLATES[:3])

        summary = synth.get_trace_summary()

        assert "templates_processed" in summary
        assert "total_predicates" in summary
        assert "total_thresholds" in summary
        assert "by_template" in summary
        assert summary["templates_processed"] == 3


class TestSynthesizePredicatesFunction:
    """Tests for the convenience function."""

    def test_synthesize_predicates_basic(self):
        predicates, summary = synthesize_predicates()

        assert len(predicates) > 0
        assert "templates_processed" in summary

    def test_synthesize_predicates_with_observed_data(self):
        observed = {
            "traction.arr": [1000000, 2000000, 5000000],
            "business_model.gross_margin": [0.5, 0.6, 0.7],
        }
        predicates, summary = synthesize_predicates(observed_data=observed)

        assert len(predicates) > 0

    def test_synthesize_predicates_with_config(self):
        config = SynthesizerConfig(max_predicates_per_template=5)
        search_config = SearchSpaceConfig(include_trends=False)

        predicates, summary = synthesize_predicates(
            config=config,
            search_config=search_config,
        )

        assert len(predicates) > 0


class TestThresholdProposerIntegration:
    """Integration tests ensuring synthesizer uses Threshold Proposer correctly."""

    def test_thresholds_come_from_proposer(self):
        """Ensure ge/le predicates use thresholds from proposer, not infinite."""
        template = PredicateTemplate(
            template_id="arr_min",
            template_type=TemplateType.THRESHOLD_GE,
            field="traction.arr",
        )
        observed = {"traction.arr": [1000000, 2000000, 5000000, 10000000]}

        config = SynthesizerConfig(max_predicates_per_template=12)
        synth = PredicateSynthesizer(config=config, observed_data=observed)
        predicates = synth.synthesize_from_template(template)

        # Should get bounded number of predicates
        assert len(predicates) <= 12

        # Verify thresholds were logged
        traces = synth.get_traces()
        assert len(traces) == 1
        assert traces[0].threshold_count > 0
        assert traces[0].predicates_generated == len(predicates)

    def test_no_infinite_threshold_search(self):
        """Ensure we don't search infinite threshold space."""
        # Even with many observed values, we should be bounded
        observed = {"traction.arr": list(range(100000, 10000001, 100000))}  # 100 values

        config = SynthesizerConfig(max_predicates_per_template=12)
        synth = PredicateSynthesizer(config=config, observed_data=observed)

        template = PredicateTemplate(
            template_id="arr_min",
            template_type=TemplateType.THRESHOLD_GE,
            field="traction.arr",
        )
        predicates = synth.synthesize_from_template(template)

        # Must be bounded
        assert len(predicates) <= 12

    def test_range_predicates_from_threshold_pairs(self):
        """Ensure range predicates use threshold pairs, not infinite combinations."""
        template = PredicateTemplate(
            template_id="valuation_range",
            template_type=TemplateType.RANGE,
            field="round_terms.pre_money_valuation",
        )
        observed = {"round_terms.pre_money_valuation": [5000000, 10000000, 25000000, 50000000, 100000000]}

        synth = PredicateSynthesizer(observed_data=observed)
        predicates = synth.synthesize_from_template(template)

        # Number of range predicates should be bounded
        # Adjacent pairs: n-1, wider ranges: n-2
        # So roughly 2n-3 maximum
        traces = synth.get_traces()
        n_thresholds = traces[0].threshold_count

        # Range predicates = (n-1) adjacent + (n-2) wider = 2n-3
        max_expected = 2 * n_thresholds - 3 if n_thresholds >= 2 else 0
        assert len(predicates) <= max(max_expected, 0)
