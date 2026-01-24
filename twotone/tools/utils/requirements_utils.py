from __future__ import annotations

from collections.abc import Callable
import ast
import inspect
import textwrap

_REQUIRED_TOOLS_ATTR = "_twotone_required_tools"
_REQUIRED_DEPS_ATTR = "_twotone_required_tool_deps"


def _normalize_callable(func: Callable) -> Callable:
    return func.__func__ if hasattr(func, "__func__") else func


def collect_required_tools(*roots: Callable) -> set[str]:
    required: set[str] = set()
    seen: set[Callable] = set()
    stack = [_normalize_callable(root) for root in roots if root is not None]

    while stack:
        func = stack.pop()
        if func in seen:
            continue
        seen.add(func)

        required.update(getattr(func, _REQUIRED_TOOLS_ATTR, set()))
        required.update(_infer_tools_from_start_process_calls(func))
        deps = set(getattr(func, _REQUIRED_DEPS_ATTR, set()))
        deps.update(_infer_deps_from_calls(func))
        for dep in deps:
            if dep not in seen:
                stack.append(dep)

    return required


def _infer_deps_from_calls(func: Callable) -> set[Callable]:
    base = _normalize_callable(func)
    func_globals = getattr(base, "__globals__", {})
    owner_class = _resolve_owner_class(base)

    try:
        source = inspect.getsource(base)
    except (OSError, TypeError):
        return set()

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return set()

    local_classes = _collect_local_class_assignments(tree, func_globals)
    calls = _collect_calls(tree)
    deps: set[Callable] = set()

    for call in calls:
        target = _resolve_call_target(call.func, func_globals, owner_class, local_classes)
        target_func = _as_function(target)
        if target_func is None or target_func is base:
            continue
        if _is_start_process(target_func):
            continue
        deps.add(target_func)

    return deps


def _infer_tools_from_start_process_calls(func: Callable) -> set[str]:
    base = _normalize_callable(func)
    func_globals = getattr(base, "__globals__", {})
    owner_class = _resolve_owner_class(base)

    try:
        source = inspect.getsource(base)
    except (OSError, TypeError):
        return set()

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return set()

    local_classes = _collect_local_class_assignments(tree, func_globals)
    local_strings = _collect_local_string_assignments(tree, func_globals)
    calls = _collect_calls(tree)
    tools: set[str] = set()

    for call in calls:
        target = _resolve_call_target(call.func, func_globals, owner_class, local_classes)
        target_func = _as_function(target)
        if not _is_start_process(target_func):
            continue
        tool = _resolve_start_process_tool(call, local_strings, func_globals)
        if tool:
            tools.add(tool)

    return tools


def _is_start_process(func: Callable | None) -> bool:
    if func is None:
        return False
    if getattr(func, "__name__", "") != "start_process":
        return False
    module_name = getattr(func, "__module__", "")
    return module_name.endswith(".process_utils")


def _resolve_start_process_tool(call: ast.Call, local_strings: dict[str, str], func_globals: dict) -> str | None:
    if call.args:
        return _resolve_string_value(call.args[0], local_strings, func_globals)
    for keyword in call.keywords:
        if keyword.arg == "process":
            return _resolve_string_value(keyword.value, local_strings, func_globals)
    return None


def _resolve_string_value(expr: ast.expr, local_strings: dict[str, str], func_globals: dict) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.Str):
        return expr.s
    if isinstance(expr, ast.Name):
        if expr.id in local_strings:
            return local_strings[expr.id]
        value = func_globals.get(expr.id)
        if isinstance(value, str):
            return value
    return None


def _resolve_owner_class(func: Callable) -> type | None:
    qualname = getattr(func, "__qualname__", "")
    if "." not in qualname:
        return None
    class_name = qualname.split(".")[0]
    module = inspect.getmodule(func)
    if not module:
        return None
    owner = getattr(module, class_name, None)
    return owner if isinstance(owner, type) else None


def _collect_calls(tree: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []

    class CallCollector(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for stmt in node.body:
                self.visit(stmt)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            for stmt in node.body:
                self.visit(stmt)

        def visit_Call(self, node: ast.Call) -> None:
            calls.append(node)
            self.generic_visit(node)

    CallCollector().visit(tree)
    return calls


def _collect_local_class_assignments(tree: ast.AST, func_globals: dict) -> dict[str, type]:
    assignments: dict[str, type] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target_name = node.targets[0].id
        if not isinstance(node.value, ast.Call):
            continue
        cls = _resolve_name_or_attr(node.value.func, func_globals)
        if isinstance(cls, type):
            assignments[target_name] = cls
    return assignments


def _collect_local_string_assignments(tree: ast.AST, func_globals: dict) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        value = _resolve_string_value(node.value, {}, func_globals)
        if value is None:
            continue
        assignments[node.targets[0].id] = value
    return assignments


def _resolve_call_target(
    call_func: ast.expr,
    func_globals: dict,
    owner_class: type | None,
    local_classes: dict[str, type],
) -> Callable | None:
    if isinstance(call_func, ast.Name):
        return _as_function(func_globals.get(call_func.id))

    if isinstance(call_func, ast.Attribute):
        base = None
        if isinstance(call_func.value, ast.Name):
            base_name = call_func.value.id
            if base_name in ("self", "cls") and owner_class is not None:
                base = owner_class
            elif base_name in local_classes:
                base = local_classes[base_name]
            else:
                base = func_globals.get(base_name)
        else:
            base = _resolve_name_or_attr(call_func.value, func_globals)

        if base is not None:
            return _as_function(getattr(base, call_func.attr, None))

    return None


def _resolve_name_or_attr(expr: ast.expr, func_globals: dict) -> object | None:
    if isinstance(expr, ast.Name):
        return func_globals.get(expr.id)
    if isinstance(expr, ast.Attribute):
        base = _resolve_name_or_attr(expr.value, func_globals)
        if base is None:
            return None
        return getattr(base, expr.attr, None)
    return None


def _as_function(obj: object | None) -> Callable | None:
    if obj is None:
        return None
    if inspect.ismethod(obj):
        return obj.__func__
    if inspect.isfunction(obj):
        return obj
    return None
