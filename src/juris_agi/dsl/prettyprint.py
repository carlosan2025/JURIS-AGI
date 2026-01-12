"""
Pretty printing for DSL ASTs.
"""

from typing import Optional

from .ast import (
    ASTNode,
    LiteralNode,
    VariableNode,
    PrimitiveNode,
    ComposeNode,
    LambdaNode,
    ApplyNode,
    LetNode,
    CondNode,
    MapNode,
    FilterNode,
)


def ast_to_source(ast: ASTNode, indent: int = 0) -> str:
    """
    Convert AST to readable source code.

    Args:
        ast: The AST node to convert
        indent: Current indentation level

    Returns:
        Source code string
    """
    prefix = "  " * indent

    if isinstance(ast, LiteralNode):
        if isinstance(ast.value, dict):
            items = ", ".join(f"{k}: {v}" for k, v in ast.value.items())
            return f"{{{items}}}"
        elif isinstance(ast.value, list):
            items = ", ".join(repr(v) for v in ast.value)
            return f"[{items}]"
        return repr(ast.value)

    elif isinstance(ast, VariableNode):
        return ast.name

    elif isinstance(ast, PrimitiveNode):
        if not ast.args:
            return ast.name
        args = ", ".join(ast_to_source(a, indent) for a in ast.args)
        return f"{ast.name}({args})"

    elif isinstance(ast, ComposeNode):
        ops = " >> ".join(ast_to_source(op, indent) for op in ast.operations)
        return ops

    elif isinstance(ast, LambdaNode):
        params = ", ".join(
            f"{name}: {typ}" if typ else name
            for name, typ in ast.params
        )
        body = ast_to_source(ast.body, indent)
        return f"λ({params}). {body}"

    elif isinstance(ast, ApplyNode):
        fn = ast_to_source(ast.function, indent)
        args = ", ".join(ast_to_source(a, indent) for a in ast.args)
        return f"({fn})({args})"

    elif isinstance(ast, LetNode):
        value = ast_to_source(ast.value, indent)
        body = ast_to_source(ast.body, indent)
        return f"let {ast.name} = {value} in\n{prefix}  {body}"

    elif isinstance(ast, CondNode):
        cond = ast_to_source(ast.condition, indent)
        then_b = ast_to_source(ast.then_branch, indent + 1)
        else_b = ast_to_source(ast.else_branch, indent + 1)
        return f"if {cond} then\n{prefix}  {then_b}\n{prefix}else\n{prefix}  {else_b}"

    elif isinstance(ast, MapNode):
        fn = ast_to_source(ast.function, indent)
        lst = ast_to_source(ast.list_expr, indent)
        return f"map({fn}, {lst})"

    elif isinstance(ast, FilterNode):
        pred = ast_to_source(ast.predicate, indent)
        lst = ast_to_source(ast.list_expr, indent)
        return f"filter({pred}, {lst})"

    else:
        return str(ast)


def pretty_print(ast: ASTNode) -> str:
    """
    Pretty print an AST with formatting.

    Returns a human-readable string representation.
    """
    return ast_to_source(ast)


def ast_to_tree(ast: ASTNode, indent: int = 0) -> str:
    """
    Convert AST to tree visualization.

    Args:
        ast: The AST node
        indent: Current indentation level

    Returns:
        Tree string representation
    """
    prefix = "  " * indent
    lines = []

    if isinstance(ast, LiteralNode):
        lines.append(f"{prefix}Literal({ast.value})")

    elif isinstance(ast, VariableNode):
        lines.append(f"{prefix}Var({ast.name})")

    elif isinstance(ast, PrimitiveNode):
        lines.append(f"{prefix}Prim({ast.name})")
        for arg in ast.args:
            lines.append(ast_to_tree(arg, indent + 1))

    elif isinstance(ast, ComposeNode):
        lines.append(f"{prefix}Compose")
        for op in ast.operations:
            lines.append(ast_to_tree(op, indent + 1))

    elif isinstance(ast, LambdaNode):
        params = ", ".join(name for name, _ in ast.params)
        lines.append(f"{prefix}Lambda({params})")
        lines.append(ast_to_tree(ast.body, indent + 1))

    elif isinstance(ast, ApplyNode):
        lines.append(f"{prefix}Apply")
        lines.append(ast_to_tree(ast.function, indent + 1))
        for arg in ast.args:
            lines.append(ast_to_tree(arg, indent + 1))

    elif isinstance(ast, LetNode):
        lines.append(f"{prefix}Let({ast.name})")
        lines.append(f"{prefix}  value:")
        lines.append(ast_to_tree(ast.value, indent + 2))
        lines.append(f"{prefix}  body:")
        lines.append(ast_to_tree(ast.body, indent + 2))

    else:
        lines.append(f"{prefix}{type(ast).__name__}")

    return "\n".join(lines)


def format_program_with_types(ast: ASTNode) -> str:
    """Format program showing inferred types."""
    def format_node(node: ASTNode, indent: int = 0) -> str:
        prefix = "  " * indent
        type_str = f" : {node.node_type()}" if node.node_type() else ""

        if isinstance(node, LiteralNode):
            return f"{prefix}{node.value}{type_str}"

        elif isinstance(node, VariableNode):
            return f"{prefix}{node.name}{type_str}"

        elif isinstance(node, PrimitiveNode):
            if not node.args:
                return f"{prefix}{node.name}{type_str}"
            args = ", ".join(format_node(a, 0) for a in node.args)
            return f"{prefix}{node.name}({args}){type_str}"

        elif isinstance(node, ComposeNode):
            ops = " >> ".join(format_node(op, 0) for op in node.operations)
            return f"{prefix}({ops}){type_str}"

        else:
            return f"{prefix}{node}{type_str}"

    return format_node(ast)
