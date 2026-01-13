"""
Unit tests for JURIS VC DSL parser.

Tests:
- Parsing predicates from DSL strings
- Pretty printing
- Round-trip parsing
"""

import pytest

from juris_agi.vc_dsl import (
    parse,
    pretty_print,
    ParseError,
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
    # Types
    TrendKind,
)


# =============================================================================
# Basic Parsing Tests
# =============================================================================


class TestBasicParsing:
    """Tests for parsing basic predicates."""

    def test_parse_has(self):
        """Should parse has() predicate."""
        pred = parse("has(traction.arr)")

        assert isinstance(pred, Has)
        assert pred.field == "traction.arr"

    def test_parse_eq_string(self):
        """Should parse eq() with string value."""
        pred = parse('eq(stage, "series_a")')

        assert isinstance(pred, Eq)
        assert pred.field == "stage"
        assert pred.value == "series_a"

    def test_parse_eq_number(self):
        """Should parse eq() with numeric value."""
        pred = parse("eq(team.size, 12)")

        assert isinstance(pred, Eq)
        assert pred.value == 12

    def test_parse_ge(self):
        """Should parse ge() predicate."""
        pred = parse("ge(traction.arr, 1000000)")

        assert isinstance(pred, Ge)
        assert pred.field == "traction.arr"
        assert pred.threshold == 1000000

    def test_parse_ge_float(self):
        """Should parse ge() with float threshold."""
        pred = parse("ge(gross_margin, 0.75)")

        assert isinstance(pred, Ge)
        assert pred.threshold == 0.75

    def test_parse_le(self):
        """Should parse le() predicate."""
        pred = parse("le(burn_rate, 150000)")

        assert isinstance(pred, Le)
        assert pred.threshold == 150000

    def test_parse_between(self):
        """Should parse between() predicate."""
        pred = parse("between(valuation, 10000000, 50000000)")

        assert isinstance(pred, Between)
        assert pred.lo == 10000000
        assert pred.hi == 50000000

    def test_parse_in(self):
        """Should parse in() predicate."""
        pred = parse('in(stage, ["seed", "series_a", "series_b"])')

        assert isinstance(pred, In)
        assert pred.values == ["seed", "series_a", "series_b"]

    def test_parse_conf_ge(self):
        """Should parse conf_ge() predicate."""
        pred = parse("conf_ge(traction.arr, 0.8)")

        assert isinstance(pred, ConfGe)
        assert pred.min_confidence == 0.8

    def test_parse_source_in(self):
        """Should parse source_in() predicate."""
        pred = parse('source_in(arr, ["ic_memo", "audited_financials"])')

        assert isinstance(pred, SourceIn)
        assert pred.sources == ["ic_memo", "audited_financials"]

    def test_parse_trend(self):
        """Should parse trend() predicate."""
        pred = parse('trend(arr, 6, "up")')

        assert isinstance(pred, Trend)
        assert pred.window == 6
        assert pred.kind == TrendKind.UP


# =============================================================================
# Composite Parsing Tests
# =============================================================================


class TestCompositeParsing:
    """Tests for parsing composite predicates."""

    def test_parse_and(self):
        """Should parse and() predicate."""
        pred = parse("and(has(a), has(b))")

        assert isinstance(pred, And)
        assert len(pred.predicates) == 2
        assert all(isinstance(p, Has) for p in pred.predicates)

    def test_parse_or(self):
        """Should parse or() predicate."""
        pred = parse("or(has(a), has(b))")

        assert isinstance(pred, Or)
        assert len(pred.predicates) == 2

    def test_parse_not(self):
        """Should parse not() predicate."""
        pred = parse("not(has(field))")

        assert isinstance(pred, Not)
        assert isinstance(pred.predicate, Has)

    def test_parse_implies(self):
        """Should parse implies() predicate."""
        pred = parse("implies(has(a), ge(b, 100))")

        assert isinstance(pred, Implies)
        assert isinstance(pred.antecedent, Has)
        assert isinstance(pred.consequent, Ge)

    def test_parse_nested_and_or(self):
        """Should parse nested and/or predicates."""
        pred = parse("and(has(a), or(has(b), has(c)))")

        assert isinstance(pred, And)
        assert len(pred.predicates) == 2
        assert isinstance(pred.predicates[0], Has)
        assert isinstance(pred.predicates[1], Or)

    def test_parse_deeply_nested(self):
        """Should parse deeply nested predicates."""
        pred = parse("and(has(a), and(has(b), or(ge(c, 1), le(d, 2))))")

        assert isinstance(pred, And)
        inner = pred.predicates[1]
        assert isinstance(inner, And)
        innermost = inner.predicates[1]
        assert isinstance(innermost, Or)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestParseErrors:
    """Tests for parse error handling."""

    def test_unknown_predicate(self):
        """Should raise error for unknown predicate."""
        with pytest.raises(ParseError) as exc_info:
            parse("unknown(field)")

        assert "Unknown predicate" in str(exc_info.value)

    def test_unterminated_string(self):
        """Should raise error for unterminated string."""
        with pytest.raises(ParseError) as exc_info:
            parse('eq(field, "value)')

        assert "Unterminated string" in str(exc_info.value)

    def test_missing_argument(self):
        """Should raise error for missing argument."""
        with pytest.raises(ParseError):
            parse("ge(field)")  # Missing threshold

    def test_invalid_syntax(self):
        """Should raise error for invalid syntax."""
        with pytest.raises(ParseError):
            parse("has field")  # Missing parentheses


# =============================================================================
# Pretty Print Tests
# =============================================================================


class TestPrettyPrint:
    """Tests for pretty printing."""

    def test_pretty_print_simple(self):
        """Should pretty print simple predicate."""
        pred = Has("field")
        output = pretty_print(pred)

        assert output.strip() == "has(field)"

    def test_pretty_print_and(self):
        """Should pretty print and() with indentation."""
        pred = And([Has("a"), Has("b")])
        output = pretty_print(pred)

        assert "and(" in output
        assert "has(a)" in output
        assert "has(b)" in output

    def test_pretty_print_nested(self):
        """Should pretty print nested predicates."""
        pred = And([Has("a"), Or([Has("b"), Has("c")])])
        output = pretty_print(pred)

        assert "and(" in output
        assert "or(" in output


# =============================================================================
# Round-Trip Tests
# =============================================================================


class TestRoundTrip:
    """Tests for parse -> to_dsl round-trip."""

    def test_roundtrip_has(self):
        """Has should round-trip correctly."""
        original = "has(traction.arr)"
        pred = parse(original)
        assert pred.to_dsl() == original

    def test_roundtrip_ge(self):
        """Ge should round-trip correctly."""
        original = "ge(arr, 1000000)"
        pred = parse(original)
        assert pred.to_dsl() == original

    def test_roundtrip_eq_string(self):
        """Eq with string should round-trip correctly."""
        original = 'eq(stage, "series_a")'
        pred = parse(original)
        assert pred.to_dsl() == original

    def test_roundtrip_and(self):
        """And should round-trip correctly."""
        original = "and(has(a), has(b))"
        pred = parse(original)
        assert pred.to_dsl() == original

    def test_roundtrip_complex(self):
        """Complex predicate should round-trip correctly."""
        original = "and(conf_ge(arr, 0.8), ge(arr, 1000000))"
        pred = parse(original)
        assert pred.to_dsl() == original


# =============================================================================
# Whitespace Handling Tests
# =============================================================================


class TestWhitespaceHandling:
    """Tests for whitespace handling."""

    def test_whitespace_around_args(self):
        """Should handle whitespace around arguments."""
        pred = parse("ge( traction.arr , 1000000 )")

        assert isinstance(pred, Ge)
        assert pred.field == "traction.arr"

    def test_whitespace_in_list(self):
        """Should handle whitespace in lists."""
        pred = parse('in(stage, [ "a" , "b" , "c" ])')

        assert isinstance(pred, In)
        assert pred.values == ["a", "b", "c"]

    def test_multiline_input(self):
        """Should handle multiline input."""
        dsl = """
        and(
            has(a),
            ge(b, 100)
        )
        """
        pred = parse(dsl)

        assert isinstance(pred, And)
        assert len(pred.predicates) == 2
