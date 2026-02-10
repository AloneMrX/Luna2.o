"""Safe arithmetic evaluator built on Python AST."""

from __future__ import annotations

import ast
import operator

_ALLOWED_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class MathEvaluationError(ValueError):
    """Raised when an expression contains disallowed syntax."""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_BINARY[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        value = _eval_node(node.operand)
        return _ALLOWED_UNARY[type(node.op)](value)

    raise MathEvaluationError("Expression contains unsupported operations.")


def evaluate_expression(expression: str) -> str:
    """Safely evaluate a plain arithmetic expression."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_node(parsed.body)
        if result.is_integer():
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Math error: division by zero."
    except (SyntaxError, MathEvaluationError, TypeError, ValueError):
        return "Math error: unsupported or invalid expression."
