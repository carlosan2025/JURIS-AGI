"""
DSL Interpreter - executes AST programs.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List

from ..core.types import Grid
from .ast import (
    ASTNode,
    LiteralNode,
    VariableNode,
    PrimitiveNode,
    ComposeNode,
    LambdaNode,
    ApplyNode,
    LetNode,
)
from .primitives import get_primitive, PRIMITIVES


class InterpreterError(Exception):
    """Error during interpretation."""
    pass


@dataclass
class Closure:
    """A closure capturing a lambda and its environment."""
    params: List[str]
    body: ASTNode
    env: Dict[str, Any]


class DSLInterpreter:
    """
    Interpreter for the DSL.

    Executes AST programs in a given environment.
    """

    def __init__(self, trace: bool = False):
        """
        Initialize interpreter.

        Args:
            trace: If True, record execution trace for debugging
        """
        self.trace = trace
        self.execution_trace: List[Dict[str, Any]] = []

    def interpret(
        self,
        ast: ASTNode,
        env: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Interpret an AST node in the given environment.

        Args:
            ast: The AST to interpret
            env: Variable bindings (default: empty)

        Returns:
            The result of evaluation
        """
        if env is None:
            env = {}

        if self.trace:
            self.execution_trace.append({
                "node": str(ast),
                "type": type(ast).__name__,
                "env_keys": list(env.keys()),
            })

        if isinstance(ast, LiteralNode):
            return ast.value

        elif isinstance(ast, VariableNode):
            if ast.name not in env:
                raise InterpreterError(f"Unbound variable: {ast.name}")
            return env[ast.name]

        elif isinstance(ast, PrimitiveNode):
            return self._eval_primitive(ast, env)

        elif isinstance(ast, ComposeNode):
            return self._eval_compose(ast, env)

        elif isinstance(ast, LambdaNode):
            # Create closure
            param_names = [name for name, _ in ast.params]
            return Closure(params=param_names, body=ast.body, env=env.copy())

        elif isinstance(ast, ApplyNode):
            return self._eval_apply(ast, env)

        elif isinstance(ast, LetNode):
            return self._eval_let(ast, env)

        else:
            raise InterpreterError(f"Unknown AST node type: {type(ast)}")

    def _eval_primitive(self, node: PrimitiveNode, env: Dict[str, Any]) -> Any:
        """Evaluate a primitive operation."""
        spec = get_primitive(node.name)
        if spec is None:
            raise InterpreterError(f"Unknown primitive: {node.name}")

        # Evaluate arguments from the AST
        args = [self.interpret(arg, env) for arg in node.args]

        # Check if we need to prepend the input grid
        # This handles Grid -> X primitives when called in a program context
        if "input" in env:
            # Check the signature: if first arg is Grid and we don't have a Grid, prepend input
            sig = spec.signature
            if sig.arg_types and len(sig.arg_types) > len(args):
                # Need more args than provided, prepend input
                args = [env["input"]] + args
            elif not args:
                # No args provided, use input as first arg
                args = [env["input"]]

        # Call the primitive implementation
        try:
            result = spec.implementation(*args)
            return result
        except Exception as e:
            raise InterpreterError(
                f"Error in primitive {node.name}: {e}"
            ) from e

    def _eval_compose(self, node: ComposeNode, env: Dict[str, Any]) -> Any:
        """
        Evaluate a composition of operations.

        The input is expected to be in env["input"] or as the first operation.
        Composition is left-to-right pipeline.
        """
        if not node.operations:
            raise InterpreterError("Empty composition")

        # Start with input from environment
        if "input" not in env:
            raise InterpreterError("Composition requires 'input' in environment")

        result = env["input"]

        for op in node.operations:
            # Create new env with current result as input
            op_env = env.copy()
            op_env["input"] = result

            if isinstance(op, PrimitiveNode):
                # For primitives, pass result as first argument if no args
                if not op.args:
                    spec = get_primitive(op.name)
                    if spec:
                        result = spec.implementation(result)
                    else:
                        raise InterpreterError(f"Unknown primitive: {op.name}")
                else:
                    # Evaluate with result available as input
                    result = self._eval_primitive(op, op_env)
            else:
                # For other nodes, interpret and assume callable
                fn_result = self.interpret(op, op_env)
                if callable(fn_result):
                    result = fn_result(result)
                elif isinstance(fn_result, Closure):
                    # Apply closure
                    if len(fn_result.params) != 1:
                        raise InterpreterError(
                            "Closure in composition must have exactly 1 parameter"
                        )
                    closure_env = fn_result.env.copy()
                    closure_env[fn_result.params[0]] = result
                    result = self.interpret(fn_result.body, closure_env)
                else:
                    result = fn_result

        return result

    def _eval_apply(self, node: ApplyNode, env: Dict[str, Any]) -> Any:
        """Evaluate function application."""
        fn = self.interpret(node.function, env)
        args = [self.interpret(arg, env) for arg in node.args]

        if isinstance(fn, Closure):
            if len(args) != len(fn.params):
                raise InterpreterError(
                    f"Arity mismatch: expected {len(fn.params)}, got {len(args)}"
                )
            # Extend closure environment with arguments
            new_env = fn.env.copy()
            for param, arg in zip(fn.params, args):
                new_env[param] = arg
            return self.interpret(fn.body, new_env)

        elif callable(fn):
            return fn(*args)

        else:
            raise InterpreterError(f"Cannot apply non-function: {type(fn)}")

    def _eval_let(self, node: LetNode, env: Dict[str, Any]) -> Any:
        """Evaluate let binding."""
        value = self.interpret(node.value, env)
        new_env = env.copy()
        new_env[node.name] = value
        return self.interpret(node.body, new_env)

    def create_program_function(self, ast: ASTNode) -> Callable[[Grid], Grid]:
        """
        Create a callable function from an AST.

        Returns a function that takes a Grid and returns a Grid.
        """
        def program(input_grid: Grid) -> Grid:
            env = {"input": input_grid}
            result = self.interpret(ast, env)
            if not isinstance(result, Grid):
                raise InterpreterError(
                    f"Program did not return Grid, got {type(result)}"
                )
            return result
        return program


def interpret(ast: ASTNode, env: Optional[Dict[str, Any]] = None) -> Any:
    """Convenience function to interpret an AST."""
    interpreter = DSLInterpreter()
    return interpreter.interpret(ast, env)


def make_program(ast: ASTNode) -> Callable[[Grid], Grid]:
    """Create a Grid -> Grid function from an AST."""
    interpreter = DSLInterpreter()
    return interpreter.create_program_function(ast)


def run_on_grid(ast: ASTNode, grid: Grid) -> Grid:
    """Run an AST program on a grid and return result."""
    program = make_program(ast)
    return program(grid)
