"""DSL module: primitives, AST, interpreter, type system."""

from .ast import (
    ASTNode,
    PrimitiveNode,
    ComposeNode,
    LiteralNode,
    VariableNode,
    LambdaNode,
    ApplyNode,
    LetNode,
)
from .type_system import (
    DSLType,
    GridType,
    IntType,
    BoolType,
    ColorType,
    ListType,
    FunctionType,
    ObjectType,
    type_check,
)
from .primitives import (
    PRIMITIVES,
    get_primitive,
    list_primitives,
    PrimitiveSpec,
)
from .interpreter import DSLInterpreter, interpret
from .prettyprint import pretty_print, ast_to_source

__all__ = [
    # AST
    "ASTNode",
    "PrimitiveNode",
    "ComposeNode",
    "LiteralNode",
    "VariableNode",
    "LambdaNode",
    "ApplyNode",
    "LetNode",
    # Types
    "DSLType",
    "GridType",
    "IntType",
    "BoolType",
    "ColorType",
    "ListType",
    "FunctionType",
    "ObjectType",
    "type_check",
    # Primitives
    "PRIMITIVES",
    "get_primitive",
    "list_primitives",
    "PrimitiveSpec",
    # Interpreter
    "DSLInterpreter",
    "interpret",
    # Pretty print
    "pretty_print",
    "ast_to_source",
]
