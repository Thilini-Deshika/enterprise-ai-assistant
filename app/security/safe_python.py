"""AST allow-list validator for LLM-generated Python (RLM search plans and analysis snippets).

Allow-list, not deny-list: only node types known to be harmless are accepted; imports, class
definitions, `while`, dunder access, attribute escapes and huge constants are rejected before any
code runs. Execution adds a second layer (restricted builtins; analysis runs in an isolated
subprocess with a hard timeout). For production use a real sandbox (gVisor / Firecracker / e2b).
"""
from __future__ import annotations

import ast

from app.errors import GuardrailViolation

_COMMON_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.If, ast.Break, ast.Continue,
    ast.Pass, ast.Try, ast.ExceptHandler, ast.Call, ast.Name, ast.Constant, ast.Attribute, ast.Subscript,
    ast.Slice, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.comprehension, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.JoinedStr, ast.FormattedValue, ast.keyword, ast.Lambda, ast.arguments, ast.arg, ast.Starred,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context,
)
_RLM_ONLY = (ast.AsyncFunctionDef, ast.Await, ast.Return)
_FORBIDDEN_OPERATORS = (ast.Pow, ast.LShift, ast.MatMult)  # cheap DoS (9**9**9)

BANNED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "dir", "type", "object", "breakpoint", "exit", "quit", "help", "memoryview", "bytearray", "super",
    "classmethod", "staticmethod", "property", "os", "sys", "subprocess", "importlib", "builtins",
})
BANNED_ATTRS = frozenset({
    "format", "format_map", "mro", "system", "popen", "eval", "exec", "query", "io", "pickle", "load", "loads",
    "dump", "dumps", "to_pickle", "to_json", "to_csv", "to_sql", "to_excel", "to_parquet", "to_clipboard",
    "style", "plot", "values_to_file",
})
_ALLOWED_TO_ATTRS = frozenset({"to_dict", "to_list", "to_frame", "to_numpy", "to_string"})
_MAX_INT = 100_000


def validate_python(code: str, *, mode: str) -> ast.Module:
    """mode='analysis' (pandas snippet) or 'rlm' (single `async def run(env)` plan)."""
    if len(code) > 6000:
        raise GuardrailViolation("Generated code is too long")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise GuardrailViolation(f"Syntax error in generated code: {exc.msg} (line {exc.lineno})") from exc

    allowed = _COMMON_NODES + (_RLM_ONLY if mode == "rlm" else ())
    if mode == "rlm":
        funcs = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)]
        others = [n for n in tree.body if not isinstance(n, ast.AsyncFunctionDef) and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        if len(funcs) != 1 or funcs[0].name != "run" or others:
            raise GuardrailViolation("Plan must consist of exactly one `async def run(env):`")

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_OPERATORS):
            raise GuardrailViolation(f"Operator not allowed: {type(node).__name__}")
        if not isinstance(node, allowed):
            raise GuardrailViolation(f"Construct not allowed in generated code: {type(node).__name__}")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in BANNED_NAMES):
            raise GuardrailViolation(f"Name not allowed: {node.id}")
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_") or attr in BANNED_ATTRS or attr.startswith("read_"):
                raise GuardrailViolation(f"Attribute not allowed: .{attr}")
            if attr.startswith("to_") and attr not in _ALLOWED_TO_ATTRS:
                raise GuardrailViolation(f"Attribute not allowed: .{attr}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and "__" in node.value:
                raise GuardrailViolation("Dunder strings are not allowed")
            if isinstance(node.value, int) and not isinstance(node.value, bool) and abs(node.value) > _MAX_INT:
                raise GuardrailViolation("Integer constant too large")
    return tree
