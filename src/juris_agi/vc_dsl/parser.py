"""
Parser and pretty-printer for JURIS VC DSL.

Converts between DSL string representation and Predicate objects.

DSL Syntax:
    has(field)
    eq(field, value)
    in(field, [v1, v2, ...])
    ge(field, number)
    le(field, number)
    between(field, lo, hi)
    trend(field, window, kind)
    conf_ge(field, confidence)
    source_in(field, [s1, s2, ...])
    and(pred1, pred2, ...)
    or(pred1, pred2, ...)
    not(pred)
    implies(antecedent, consequent)
"""

import re
from typing import Any, Optional, Union

from .predicates_v2 import (
    Predicate,
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
    PREDICATE_REGISTRY,
)
from .typing import ValueType, TrendKind


class ParseError(Exception):
    """Error during DSL parsing."""

    def __init__(self, message: str, position: int = 0, context: str = ""):
        super().__init__(message)
        self.position = position
        self.context = context


class Token:
    """A token in the DSL."""

    def __init__(self, type_: str, value: Any, position: int = 0):
        self.type = type_
        self.value = value
        self.position = position

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


class Lexer:
    """Tokenizer for DSL strings."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def peek(self) -> Optional[str]:
        """Peek at current character."""
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def advance(self) -> Optional[str]:
        """Advance and return current character."""
        if self.pos >= len(self.text):
            return None
        char = self.text[self.pos]
        self.pos += 1
        return char

    def skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.peek() and self.peek().isspace():
            self.advance()

    def read_string(self) -> str:
        """Read a quoted string."""
        quote = self.advance()  # Consume opening quote
        result = []
        while True:
            char = self.peek()
            if char is None:
                raise ParseError("Unterminated string", self.pos)
            if char == quote:
                self.advance()  # Consume closing quote
                break
            if char == "\\":
                self.advance()
                escaped = self.advance()
                if escaped == "n":
                    result.append("\n")
                elif escaped == "t":
                    result.append("\t")
                else:
                    result.append(escaped)
            else:
                result.append(self.advance())
        return "".join(result)

    def read_number(self) -> Union[int, float]:
        """Read a numeric literal."""
        start = self.pos
        has_dot = False
        has_exp = False

        # Handle negative
        if self.peek() == "-":
            self.advance()

        while True:
            char = self.peek()
            if char is None:
                break
            if char.isdigit():
                self.advance()
            elif char == "." and not has_dot:
                has_dot = True
                self.advance()
            elif char in "eE" and not has_exp:
                has_exp = True
                self.advance()
                if self.peek() in "+-":
                    self.advance()
            else:
                break

        num_str = self.text[start : self.pos]
        if has_dot or has_exp:
            return float(num_str)
        return int(num_str)

    def read_identifier(self) -> str:
        """Read an identifier (alphanumeric + underscore + dot)."""
        start = self.pos
        while True:
            char = self.peek()
            if char is None:
                break
            if char.isalnum() or char in "_.":
                self.advance()
            else:
                break
        return self.text[start : self.pos]

    def tokenize(self) -> list[Token]:
        """Tokenize the entire input."""
        tokens = []

        while True:
            self.skip_whitespace()
            if self.pos >= len(self.text):
                break

            char = self.peek()
            pos = self.pos

            if char in "\"'":
                tokens.append(Token("STRING", self.read_string(), pos))
            elif char.isdigit() or (char == "-" and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()):
                tokens.append(Token("NUMBER", self.read_number(), pos))
            elif char.isalpha() or char == "_":
                ident = self.read_identifier()
                # Check for keywords
                if ident.lower() in {"true", "false"}:
                    tokens.append(Token("BOOLEAN", ident.lower() == "true", pos))
                else:
                    tokens.append(Token("IDENT", ident, pos))
            elif char == "(":
                tokens.append(Token("LPAREN", "(", pos))
                self.advance()
            elif char == ")":
                tokens.append(Token("RPAREN", ")", pos))
                self.advance()
            elif char == "[":
                tokens.append(Token("LBRACKET", "[", pos))
                self.advance()
            elif char == "]":
                tokens.append(Token("RBRACKET", "]", pos))
                self.advance()
            elif char == ",":
                tokens.append(Token("COMMA", ",", pos))
                self.advance()
            else:
                raise ParseError(f"Unexpected character: {char}", pos)

        tokens.append(Token("EOF", None, self.pos))
        return tokens


class Parser:
    """Parser for DSL strings."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        """Get current token."""
        return self.tokens[self.pos]

    def advance(self) -> Token:
        """Advance and return current token."""
        token = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def expect(self, type_: str) -> Token:
        """Expect a specific token type."""
        token = self.current()
        if token.type != type_:
            raise ParseError(
                f"Expected {type_}, got {token.type}",
                token.position,
            )
        return self.advance()

    def parse_predicate(self) -> Predicate:
        """Parse a predicate expression."""
        token = self.expect("IDENT")
        name = token.value.lower()

        if name not in PREDICATE_REGISTRY and name != "in":
            raise ParseError(f"Unknown predicate: {name}", token.position)

        self.expect("LPAREN")
        args = self.parse_args()
        self.expect("RPAREN")

        return self.build_predicate(name, args, token.position)

    def parse_args(self) -> list[Any]:
        """Parse argument list."""
        args = []

        while self.current().type != "RPAREN":
            if args:
                self.expect("COMMA")

            arg = self.parse_arg()
            args.append(arg)

        return args

    def parse_arg(self) -> Any:
        """Parse a single argument."""
        token = self.current()

        if token.type == "STRING":
            self.advance()
            return token.value
        elif token.type == "NUMBER":
            self.advance()
            return token.value
        elif token.type == "BOOLEAN":
            self.advance()
            return token.value
        elif token.type == "LBRACKET":
            return self.parse_list()
        elif token.type == "IDENT":
            # Could be a nested predicate or a field name
            # Look ahead to see if it's followed by (
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == "LPAREN":
                return self.parse_predicate()
            else:
                self.advance()
                return token.value
        else:
            raise ParseError(f"Unexpected token: {token.type}", token.position)

    def parse_list(self) -> list[Any]:
        """Parse a list literal."""
        self.expect("LBRACKET")
        items = []

        while self.current().type != "RBRACKET":
            if items:
                self.expect("COMMA")
            items.append(self.parse_arg())

        self.expect("RBRACKET")
        return items

    def build_predicate(self, name: str, args: list[Any], position: int) -> Predicate:
        """Build a predicate from parsed name and arguments."""
        try:
            if name == "has":
                return Has(field=args[0])

            elif name == "eq":
                return Eq(field=args[0], value=args[1])

            elif name == "in":
                return In(field=args[0], values=args[1])

            elif name == "ge":
                return Ge(field=args[0], threshold=float(args[1]))

            elif name == "le":
                return Le(field=args[0], threshold=float(args[1]))

            elif name == "between":
                return Between(field=args[0], lo=float(args[1]), hi=float(args[2]))

            elif name == "trend":
                kind = TrendKind(args[2]) if isinstance(args[2], str) else args[2]
                return Trend(field=args[0], window=int(args[1]), kind=kind)

            elif name == "conf_ge":
                return ConfGe(field=args[0], min_confidence=float(args[1]))

            elif name == "source_in":
                return SourceIn(field=args[0], sources=args[1])

            elif name == "and":
                return And(predicates=[p for p in args if isinstance(p, Predicate)])

            elif name == "or":
                return Or(predicates=[p for p in args if isinstance(p, Predicate)])

            elif name == "not":
                return Not(predicate=args[0])

            elif name == "implies":
                return Implies(antecedent=args[0], consequent=args[1])

            else:
                raise ParseError(f"Unknown predicate: {name}", position)

        except (IndexError, TypeError, ValueError) as e:
            raise ParseError(f"Invalid arguments for {name}: {e}", position)


def parse(dsl_string: str) -> Predicate:
    """
    Parse a DSL string into a Predicate.

    Args:
        dsl_string: DSL string representation

    Returns:
        Parsed Predicate

    Raises:
        ParseError: If parsing fails
    """
    lexer = Lexer(dsl_string)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse_predicate()


def pretty_print(predicate: Predicate, indent: int = 0) -> str:
    """
    Pretty-print a predicate with indentation.

    Args:
        predicate: Predicate to print
        indent: Current indentation level

    Returns:
        Formatted string representation
    """
    prefix = "  " * indent

    if isinstance(predicate, (And, Or)):
        name = "and" if isinstance(predicate, And) else "or"
        inner = ",\n".join(
            pretty_print(p, indent + 1) for p in predicate.predicates
        )
        return f"{prefix}{name}(\n{inner}\n{prefix})"

    elif isinstance(predicate, Not):
        inner = pretty_print(predicate.predicate, indent + 1)
        return f"{prefix}not(\n{inner}\n{prefix})"

    elif isinstance(predicate, Implies):
        ant = pretty_print(predicate.antecedent, indent + 1)
        con = pretty_print(predicate.consequent, indent + 1)
        return f"{prefix}implies(\n{ant},\n{con}\n{prefix})"

    else:
        return prefix + predicate.to_dsl()


# =============================================================================
# Rule DSL
# =============================================================================


def parse_rule(dsl_string: str) -> dict[str, Any]:
    """
    Parse a rule definition.

    Rule syntax:
        rule "name" priority=N decision=INVEST|PASS|DEFER when PREDICATE

    Returns:
        Dictionary with rule components
    """
    # Match rule header
    header_pattern = r'rule\s+"([^"]+)"\s+(?:priority=(\d+)\s+)?decision=(INVEST|PASS|DEFER)\s+when\s+'
    match = re.match(header_pattern, dsl_string, re.IGNORECASE)

    if not match:
        raise ParseError("Invalid rule syntax", 0, dsl_string[:50])

    name = match.group(1)
    priority = int(match.group(2)) if match.group(2) else 5
    decision = match.group(3).upper()

    # Parse the predicate
    predicate_str = dsl_string[match.end():]
    predicate = parse(predicate_str)

    return {
        "name": name,
        "priority": priority,
        "decision": decision,
        "predicate": predicate,
    }


def format_rule(
    name: str,
    decision: str,
    predicate: Predicate,
    priority: int = 5,
) -> str:
    """
    Format a rule as DSL string.

    Args:
        name: Rule name
        decision: Decision outcome
        predicate: Rule predicate
        priority: Rule priority

    Returns:
        Formatted rule string
    """
    pred_str = pretty_print(predicate, indent=1)
    return f'rule "{name}" priority={priority} decision={decision} when\n{pred_str}'
