"""
Unit tests for JURIS VC DSL predicates.

Tests:
- Each predicate type
- Missing field handling
- Low confidence handling
- Three-valued logic
"""

import pytest

from juris_agi.vc_dsl import (
    # Predicates
    Has,
    Eq,
    In,
    Ge,
    Le,
    Between,
    Trend,
    ConfGe,
    SourceIn,
    And,
    Or,
    Not,
    Implies,
    # Context
    EvalContext,
    FieldValue,
    EvalResult,
    # Types
    ValueType,
    TrendKind,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_context():
    """Context with basic field values."""
    return EvalContext(
        fields={
            "traction.arr": FieldValue(value=1000000, confidence=0.9),
            "traction.mrr": FieldValue(value=85000, confidence=0.85),
            "team.size": FieldValue(value=12, confidence=0.95),
            "stage": FieldValue(value="series_a", confidence=0.9),
            "gross_margin": FieldValue(value=0.75, confidence=0.8),
        }
    )


@pytest.fixture
def context_with_sources():
    """Context with source information."""
    return EvalContext(
        fields={
            "traction.arr": FieldValue(
                value=1000000, confidence=0.9, source_type="ic_memo"
            ),
            "projections.arr": FieldValue(
                value=5000000, confidence=0.6, source_type="pitch_deck"
            ),
        }
    )


@pytest.fixture
def empty_context():
    """Empty context with no fields."""
    return EvalContext(fields={})


# =============================================================================
# Has Predicate Tests
# =============================================================================


class TestHasPredicate:
    """Tests for Has predicate."""

    def test_has_existing_field(self, basic_context):
        """Should return TRUE for existing field."""
        pred = Has("traction.arr")
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_has_missing_field(self, basic_context):
        """Should return FALSE for missing field."""
        pred = Has("missing.field")
        assert pred.evaluate(basic_context) == EvalResult.FALSE

    def test_has_empty_context(self, empty_context):
        """Should return FALSE for any field in empty context."""
        pred = Has("any.field")
        assert pred.evaluate(empty_context) == EvalResult.FALSE

    def test_has_to_dsl(self):
        """Should produce correct DSL string."""
        pred = Has("traction.arr")
        assert pred.to_dsl() == "has(traction.arr)"


# =============================================================================
# Eq Predicate Tests
# =============================================================================


class TestEqPredicate:
    """Tests for Eq predicate."""

    def test_eq_matching_value(self, basic_context):
        """Should return TRUE for matching value."""
        pred = Eq("stage", "series_a")
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_eq_non_matching_value(self, basic_context):
        """Should return FALSE for non-matching value."""
        pred = Eq("stage", "seed")
        assert pred.evaluate(basic_context) == EvalResult.FALSE

    def test_eq_missing_field(self, basic_context):
        """Should return UNKNOWN for missing field."""
        pred = Eq("missing.field", "value")
        assert pred.evaluate(basic_context) == EvalResult.UNKNOWN

    def test_eq_numeric_value(self, basic_context):
        """Should work with numeric values."""
        pred = Eq("team.size", 12, ValueType.NUMERIC)
        assert pred.evaluate(basic_context) == EvalResult.TRUE


# =============================================================================
# In Predicate Tests
# =============================================================================


class TestInPredicate:
    """Tests for In predicate."""

    def test_in_matching_value(self, basic_context):
        """Should return TRUE when value in list."""
        pred = In("stage", ["seed", "series_a", "series_b"])
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_in_non_matching_value(self, basic_context):
        """Should return FALSE when value not in list."""
        pred = In("stage", ["seed", "series_b"])
        assert pred.evaluate(basic_context) == EvalResult.FALSE

    def test_in_missing_field(self, basic_context):
        """Should return UNKNOWN for missing field."""
        pred = In("missing", ["a", "b"])
        assert pred.evaluate(basic_context) == EvalResult.UNKNOWN


# =============================================================================
# Ge/Le Predicate Tests
# =============================================================================


class TestGePredicate:
    """Tests for Ge (>=) predicate."""

    def test_ge_above_threshold(self, basic_context):
        """Should return TRUE when value >= threshold."""
        pred = Ge("traction.arr", 500000)
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_ge_at_threshold(self, basic_context):
        """Should return TRUE when value == threshold."""
        pred = Ge("traction.arr", 1000000)
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_ge_below_threshold(self, basic_context):
        """Should return FALSE when value < threshold."""
        pred = Ge("traction.arr", 2000000)
        assert pred.evaluate(basic_context) == EvalResult.FALSE

    def test_ge_missing_field(self, basic_context):
        """Should return UNKNOWN for missing field."""
        pred = Ge("missing.field", 100)
        assert pred.evaluate(basic_context) == EvalResult.UNKNOWN


class TestLePredicate:
    """Tests for Le (<=) predicate."""

    def test_le_below_threshold(self, basic_context):
        """Should return TRUE when value <= threshold."""
        pred = Le("traction.arr", 2000000)
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_le_at_threshold(self, basic_context):
        """Should return TRUE when value == threshold."""
        pred = Le("traction.arr", 1000000)
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_le_above_threshold(self, basic_context):
        """Should return FALSE when value > threshold."""
        pred = Le("traction.arr", 500000)
        assert pred.evaluate(basic_context) == EvalResult.FALSE


# =============================================================================
# Between Predicate Tests
# =============================================================================


class TestBetweenPredicate:
    """Tests for Between predicate."""

    def test_between_in_range(self, basic_context):
        """Should return TRUE when value in range."""
        pred = Between("traction.arr", 500000, 2000000)
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_between_at_bounds(self, basic_context):
        """Should return TRUE at bounds (inclusive)."""
        pred = Between("traction.arr", 1000000, 1000000)
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_between_outside_range(self, basic_context):
        """Should return FALSE when value outside range."""
        pred = Between("traction.arr", 2000000, 5000000)
        assert pred.evaluate(basic_context) == EvalResult.FALSE

    def test_between_missing_field(self, basic_context):
        """Should return UNKNOWN for missing field."""
        pred = Between("missing", 0, 100)
        assert pred.evaluate(basic_context) == EvalResult.UNKNOWN


# =============================================================================
# Confidence Gate Tests
# =============================================================================


class TestConfGePredicate:
    """Tests for ConfGe (confidence gate) predicate."""

    def test_conf_ge_high_confidence(self, basic_context):
        """Should return TRUE when confidence >= threshold."""
        pred = ConfGe("traction.arr", 0.8)
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_conf_ge_low_confidence(self, basic_context):
        """Should return UNKNOWN when confidence < threshold."""
        pred = ConfGe("gross_margin", 0.9)  # Has confidence 0.8
        assert pred.evaluate(basic_context) == EvalResult.UNKNOWN

    def test_conf_ge_missing_field(self, basic_context):
        """Should return UNKNOWN for missing field."""
        pred = ConfGe("missing", 0.5)
        assert pred.evaluate(basic_context) == EvalResult.UNKNOWN


# =============================================================================
# Source Filter Tests
# =============================================================================


class TestSourceInPredicate:
    """Tests for SourceIn predicate."""

    def test_source_in_approved(self, context_with_sources):
        """Should return TRUE when source is approved."""
        pred = SourceIn("traction.arr", ["ic_memo", "audited_financials"])
        assert pred.evaluate(context_with_sources) == EvalResult.TRUE

    def test_source_in_not_approved(self, context_with_sources):
        """Should return UNKNOWN when source not approved."""
        pred = SourceIn("projections.arr", ["ic_memo", "audited_financials"])
        assert pred.evaluate(context_with_sources) == EvalResult.UNKNOWN


# =============================================================================
# Composite Predicate Tests
# =============================================================================


class TestAndPredicate:
    """Tests for And predicate."""

    def test_and_all_true(self, basic_context):
        """Should return TRUE when all predicates TRUE."""
        pred = And([
            Has("traction.arr"),
            Ge("traction.arr", 500000),
        ])
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_and_one_false(self, basic_context):
        """Should return FALSE when any predicate FALSE."""
        pred = And([
            Has("traction.arr"),
            Ge("traction.arr", 5000000),  # FALSE
        ])
        assert pred.evaluate(basic_context) == EvalResult.FALSE

    def test_and_one_unknown(self, basic_context):
        """Should return UNKNOWN when one UNKNOWN and rest TRUE."""
        pred = And([
            Has("traction.arr"),
            Ge("missing.field", 100),  # UNKNOWN
        ])
        assert pred.evaluate(basic_context) == EvalResult.UNKNOWN

    def test_and_short_circuit(self, basic_context):
        """Should short-circuit on FALSE."""
        pred = And([
            Ge("traction.arr", 5000000),  # FALSE
            Has("missing.field"),  # Would be FALSE but not evaluated
        ])
        assert pred.evaluate(basic_context) == EvalResult.FALSE


class TestOrPredicate:
    """Tests for Or predicate."""

    def test_or_one_true(self, basic_context):
        """Should return TRUE when any predicate TRUE."""
        pred = Or([
            Ge("traction.arr", 5000000),  # FALSE
            Has("traction.arr"),  # TRUE
        ])
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_or_all_false(self, basic_context):
        """Should return FALSE when all predicates FALSE."""
        pred = Or([
            Has("missing1"),
            Has("missing2"),
        ])
        assert pred.evaluate(basic_context) == EvalResult.FALSE

    def test_or_unknown_and_false(self, basic_context):
        """Should return UNKNOWN when one UNKNOWN and rest FALSE."""
        pred = Or([
            Has("missing"),  # FALSE
            Ge("missing", 100),  # UNKNOWN
        ])
        assert pred.evaluate(basic_context) == EvalResult.UNKNOWN


class TestNotPredicate:
    """Tests for Not predicate."""

    def test_not_true(self, basic_context):
        """Should return FALSE for NOT TRUE."""
        pred = Not(Has("traction.arr"))
        assert pred.evaluate(basic_context) == EvalResult.FALSE

    def test_not_false(self, basic_context):
        """Should return TRUE for NOT FALSE."""
        pred = Not(Has("missing"))
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_not_unknown(self, basic_context):
        """Should return UNKNOWN for NOT UNKNOWN."""
        pred = Not(Ge("missing", 100))
        assert pred.evaluate(basic_context) == EvalResult.UNKNOWN


class TestImpliesPredicate:
    """Tests for Implies predicate."""

    def test_implies_false_antecedent(self, basic_context):
        """Should return TRUE when antecedent is FALSE (vacuous truth)."""
        pred = Implies(
            Has("missing"),  # FALSE
            Ge("traction.arr", 5000000),  # Would be FALSE
        )
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_implies_true_antecedent_true_consequent(self, basic_context):
        """Should return TRUE when both TRUE."""
        pred = Implies(
            Has("traction.arr"),  # TRUE
            Ge("traction.arr", 500000),  # TRUE
        )
        assert pred.evaluate(basic_context) == EvalResult.TRUE

    def test_implies_true_antecedent_false_consequent(self, basic_context):
        """Should return FALSE when antecedent TRUE, consequent FALSE."""
        pred = Implies(
            Has("traction.arr"),  # TRUE
            Ge("traction.arr", 5000000),  # FALSE
        )
        assert pred.evaluate(basic_context) == EvalResult.FALSE


# =============================================================================
# Three-Valued Logic Tests
# =============================================================================


class TestThreeValuedLogic:
    """Tests for three-valued (Kleene) logic."""

    def test_unknown_and_true(self):
        """UNKNOWN AND TRUE = UNKNOWN."""
        assert (EvalResult.UNKNOWN & EvalResult.TRUE) == EvalResult.UNKNOWN

    def test_unknown_and_false(self):
        """UNKNOWN AND FALSE = FALSE."""
        assert (EvalResult.UNKNOWN & EvalResult.FALSE) == EvalResult.FALSE

    def test_unknown_or_true(self):
        """UNKNOWN OR TRUE = TRUE."""
        assert (EvalResult.UNKNOWN | EvalResult.TRUE) == EvalResult.TRUE

    def test_unknown_or_false(self):
        """UNKNOWN OR FALSE = UNKNOWN."""
        assert (EvalResult.UNKNOWN | EvalResult.FALSE) == EvalResult.UNKNOWN

    def test_not_unknown(self):
        """NOT UNKNOWN = UNKNOWN."""
        assert ~EvalResult.UNKNOWN == EvalResult.UNKNOWN

    def test_result_to_bool(self):
        """UNKNOWN should convert to False in boolean context."""
        assert bool(EvalResult.TRUE) is True
        assert bool(EvalResult.FALSE) is False
        assert bool(EvalResult.UNKNOWN) is False


# =============================================================================
# DSL Output Tests
# =============================================================================


class TestDSLOutput:
    """Tests for to_dsl() method."""

    def test_has_dsl(self):
        assert Has("field").to_dsl() == "has(field)"

    def test_eq_string_dsl(self):
        assert Eq("field", "value").to_dsl() == 'eq(field, "value")'

    def test_eq_numeric_dsl(self):
        assert Eq("field", 42).to_dsl() == "eq(field, 42)"

    def test_ge_dsl(self):
        assert Ge("field", 100.5).to_dsl() == "ge(field, 100.5)"

    def test_between_dsl(self):
        assert Between("field", 10, 20).to_dsl() == "between(field, 10, 20)"

    def test_and_dsl(self):
        pred = And([Has("a"), Has("b")])
        assert pred.to_dsl() == "and(has(a), has(b))"

    def test_nested_dsl(self):
        pred = And([Has("a"), Or([Has("b"), Has("c")])])
        assert pred.to_dsl() == "and(has(a), or(has(b), has(c)))"
