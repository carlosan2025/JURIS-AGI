"""
AST (Abstract Syntax Tree) for the DSL.

Provides a typed representation of programs for manipulation and analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from .type_system import DSLType


class ASTNode(ABC):
    """Base class for all AST nodes."""

    @abstractmethod
    def children(self) -> List["ASTNode"]:
        """Return child nodes."""
        pass

    @abstractmethod
    def node_type(self) -> Optional[DSLType]:
        """Return the type of this node (if known)."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def depth(self) -> int:
        """Compute the depth of this AST."""
        children = self.children()
        if not children:
            return 1
        return 1 + max(c.depth() for c in children)

    def size(self) -> int:
        """Compute the number of nodes in this AST."""
        return 1 + sum(c.size() for c in self.children())


@dataclass
class LiteralNode(ASTNode):
    """A literal value (int, bool, color, list, etc.)."""
    value: Any
    inferred_type: Optional[DSLType] = None

    def children(self) -> List[ASTNode]:
        return []

    def node_type(self) -> Optional[DSLType]:
        return self.inferred_type

    def __str__(self) -> str:
        if isinstance(self.value, dict):
            return "{" + ", ".join(f"{k}: {v}" for k, v in self.value.items()) + "}"
        return repr(self.value)


@dataclass
class VariableNode(ASTNode):
    """A variable reference."""
    name: str
    inferred_type: Optional[DSLType] = None

    def children(self) -> List[ASTNode]:
        return []

    def node_type(self) -> Optional[DSLType]:
        return self.inferred_type

    def __str__(self) -> str:
        return self.name


@dataclass
class PrimitiveNode(ASTNode):
    """
    A primitive operation with arguments.

    This is the main building block for DSL programs.
    """
    name: str
    args: List[ASTNode] = field(default_factory=list)
    inferred_type: Optional[DSLType] = None

    def children(self) -> List[ASTNode]:
        return self.args

    def node_type(self) -> Optional[DSLType]:
        return self.inferred_type

    def __str__(self) -> str:
        if not self.args:
            return self.name
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"


@dataclass
class ComposeNode(ASTNode):
    """
    Composition of multiple operations (pipeline).

    Applies operations left-to-right: compose(f, g)(x) = g(f(x))
    """
    operations: List[ASTNode]
    inferred_type: Optional[DSLType] = None

    def children(self) -> List[ASTNode]:
        return self.operations

    def node_type(self) -> Optional[DSLType]:
        return self.inferred_type

    def __str__(self) -> str:
        return " >> ".join(str(op) for op in self.operations)


@dataclass
class LambdaNode(ASTNode):
    """A lambda expression."""
    params: List[Tuple[str, Optional[DSLType]]]  # (name, type) pairs
    body: ASTNode
    inferred_type: Optional[DSLType] = None

    def children(self) -> List[ASTNode]:
        return [self.body]

    def node_type(self) -> Optional[DSLType]:
        return self.inferred_type

    def __str__(self) -> str:
        params_str = ", ".join(
            f"{name}: {typ}" if typ else name
            for name, typ in self.params
        )
        return f"λ({params_str}). {self.body}"


@dataclass
class ApplyNode(ASTNode):
    """Function application."""
    function: ASTNode
    args: List[ASTNode]
    inferred_type: Optional[DSLType] = None

    def children(self) -> List[ASTNode]:
        return [self.function] + self.args

    def node_type(self) -> Optional[DSLType]:
        return self.inferred_type

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"({self.function})({args_str})"


@dataclass
class LetNode(ASTNode):
    """Let binding."""
    name: str
    value: ASTNode
    body: ASTNode
    inferred_type: Optional[DSLType] = None

    def children(self) -> List[ASTNode]:
        return [self.value, self.body]

    def node_type(self) -> Optional[DSLType]:
        return self.inferred_type

    def __str__(self) -> str:
        return f"let {self.name} = {self.value} in {self.body}"


@dataclass
class CondNode(ASTNode):
    """Conditional expression."""
    condition: ASTNode
    then_branch: ASTNode
    else_branch: ASTNode
    inferred_type: Optional[DSLType] = None

    def children(self) -> List[ASTNode]:
        return [self.condition, self.then_branch, self.else_branch]

    def node_type(self) -> Optional[DSLType]:
        return self.inferred_type

    def __str__(self) -> str:
        return f"if {self.condition} then {self.then_branch} else {self.else_branch}"


@dataclass
class MapNode(ASTNode):
    """Map over a list."""
    function: ASTNode
    list_expr: ASTNode
    inferred_type: Optional[DSLType] = None

    def children(self) -> List[ASTNode]:
        return [self.function, self.list_expr]

    def node_type(self) -> Optional[DSLType]:
        return self.inferred_type

    def __str__(self) -> str:
        return f"map({self.function}, {self.list_expr})"


@dataclass
class FilterNode(ASTNode):
    """Filter a list."""
    predicate: ASTNode
    list_expr: ASTNode
    inferred_type: Optional[DSLType] = None

    def children(self) -> List[ASTNode]:
        return [self.predicate, self.list_expr]

    def node_type(self) -> Optional[DSLType]:
        return self.inferred_type

    def __str__(self) -> str:
        return f"filter({self.predicate}, {self.list_expr})"


def walk_ast(node: ASTNode) -> List[ASTNode]:
    """Walk the AST and return all nodes in pre-order."""
    result = [node]
    for child in node.children():
        result.extend(walk_ast(child))
    return result


def transform_ast(
    node: ASTNode,
    transform_fn,  # Callable[[ASTNode], Optional[ASTNode]]
) -> ASTNode:
    """
    Transform an AST by applying a function to each node.

    If transform_fn returns None, the original node is kept.
    """
    result = transform_fn(node)
    if result is not None:
        return result

    # Recursively transform children
    if isinstance(node, PrimitiveNode):
        new_args = [transform_ast(arg, transform_fn) for arg in node.args]
        return PrimitiveNode(name=node.name, args=new_args, inferred_type=node.inferred_type)
    elif isinstance(node, ComposeNode):
        new_ops = [transform_ast(op, transform_fn) for op in node.operations]
        return ComposeNode(operations=new_ops, inferred_type=node.inferred_type)
    elif isinstance(node, LambdaNode):
        new_body = transform_ast(node.body, transform_fn)
        return LambdaNode(params=node.params, body=new_body, inferred_type=node.inferred_type)
    elif isinstance(node, ApplyNode):
        new_fn = transform_ast(node.function, transform_fn)
        new_args = [transform_ast(arg, transform_fn) for arg in node.args]
        return ApplyNode(function=new_fn, args=new_args, inferred_type=node.inferred_type)
    elif isinstance(node, LetNode):
        new_value = transform_ast(node.value, transform_fn)
        new_body = transform_ast(node.body, transform_fn)
        return LetNode(name=node.name, value=new_value, body=new_body, inferred_type=node.inferred_type)

    return node
