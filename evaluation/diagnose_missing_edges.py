#!/usr/bin/env python3
"""Deep-diagnose missing call-graph edges by inspecting constraint engine internals.

For each ground-truth edge absent from the constraint output, this tool
inspects the *internal* builder state (class hierarchy, MRO chains, scope
registrations, parameter bindings, dependency records) to produce a
fine-grained root-cause classification.

Usage:
    python evaluation/diagnose_missing_edges.py \\
        --corpus evaluation/repo_level --project data_pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pyflow.analysis.callgraph.constraint_based.engine import ConstraintCallGraphBuilder
from pyflow.analysis.callgraph.constraint_based.model import (
    AnalysisOptions,
    GLOBAL_CONTEXT,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _last_segment(name: str) -> str:
    return name.rsplit(".", 1)[-1] if "." in name else name


def _module_of(name: str) -> str:
    """`a.b.C.m` → `a.b.C` (strip last dotted segment)"""
    return name.rsplit(".", 1)[0] if "." in name else ""


def _class_of_method(method_name: str) -> Optional[str]:
    """`a.b.C.m` → `a.b.C` if `m` looks like a method name (lowercase)."""
    parts = method_name.split(".")
    if len(parts) < 2:
        return None
    return ".".join(parts[:-1])


def _method_name(method_qualname: str) -> str:
    return method_qualname.rsplit(".", 1)[-1]


# ═══════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EdgeDiagnosis:
    caller: str
    callee: str
    category: str
    evidence: List[str] = field(default_factory=list)

    # Snapshot from engine internals
    caller_in_scopes: bool = False
    caller_in_functions: bool = False
    callee_in_scopes: bool = False
    callee_in_functions: bool = False
    caller_owner_class: Optional[str] = None
    callee_owner_class: Optional[str] = None
    caller_mro: List[str] = field(default_factory=list)
    callee_mro: List[str] = field(default_factory=list)
    caller_has_incoming: bool = False
    caller_incoming_from: List[str] = field(default_factory=list)
    self_value_kinds: List[str] = field(default_factory=list)
    self_value_details: List[str] = field(default_factory=list)
    caller_param_bindings: Dict[str, List[str]] = field(default_factory=dict)
    super_mro_next: Optional[str] = None
    callee_class_methods: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Diagnostic engine
# ═══════════════════════════════════════════════════════════════════════

class MissingEdgeDiagnoser:
    def __init__(self, builder: ConstraintCallGraphBuilder, gt: Dict[str, List[str]], cg):
        self._builder = builder
        self._gt = gt
        self._cg_graph = cg
        self._cg_edges: Set[Tuple[str, str]] = set()
        self._cg_dynamic: Set[str] = set()

        # Pre-compute edges from the built graph
        for caller, callee in cg.edges():
            self._cg_edges.add((caller, callee))

        # Which callers have dynamic summaries
        for caller, callees in cg.get().items():
            if any(c.startswith("<dynamic") for c in callees):
                self._cg_dynamic.add(caller)

    # ── lookup helpers ──

    def _func_info(self, name: str):
        return self._builder.functions.get(name)

    def _scope_info(self, name: str):
        return self._builder.scopes.get(name)

    def _class_info(self, name: str):
        return self._builder.classes.get(name)

    def _mro(self, class_name: str) -> List[str]:
        return self._builder._mro(class_name)

    def _scope_input(self, scope: str, param: str) -> List[str]:
        key = (scope, GLOBAL_CONTEXT)
        bindings = self._builder.scope_inputs.get(key, {})
        vals = bindings.get(param, set())
        return list(vals)

    def _callers_of(self, callee: str) -> List[str]:
        sources = []
        for src, tgt in self._cg_edges:
            if tgt == callee:
                sources.append(src)
        return sources

    # ── main diagnosis ──

    def diagnose_all(self) -> List[EdgeDiagnosis]:
        gt_edges: Set[Tuple[str, str]] = set()
        for caller, callees in self._gt.items():
            for callee in callees:
                gt_edges.add((caller, callee))

        missing = sorted(gt_edges - self._cg_edges)
        results = []
        for caller, callee in missing:
            d = self._diagnose_one(caller, callee)
            results.append(d)
        return results

    def _diagnose_one(self, caller: str, callee: str) -> EdgeDiagnosis:
        d = EdgeDiagnosis(caller=caller, callee=callee, category="UNKNOWN")

        # ── Basic existence checks ──
        d.caller_in_scopes = caller in self._builder.scopes
        d.caller_in_functions = caller in self._builder.functions
        d.callee_in_scopes = callee in self._builder.scopes
        d.callee_in_functions = callee in self._builder.functions

        # ── Class / MRO context ──
        caller_cls = _class_of_method(caller)
        callee_cls = _class_of_method(callee)
        d.caller_owner_class = caller_cls
        d.callee_owner_class = callee_cls

        if caller_cls and caller_cls in self._builder.classes:
            d.caller_mro = self._mro(caller_cls)
        if callee_cls and callee_cls in self._builder.classes:
            d.callee_mro = self._mro(callee_cls)
            cinfo = self._class_info(callee_cls)
            if cinfo:
                d.callee_class_methods = sorted(cinfo.methods.keys())

        # ── Incoming edges ──
        incoming = self._callers_of(caller)
        d.caller_has_incoming = len(incoming) > 0
        d.caller_incoming_from = incoming

        # ── Self parameter ──
        func_info = self._func_info(caller)
        scope_info = self._scope_info(caller)
        self_param = None
        if scope_info and scope_info.method_self_param:
            self_param = scope_info.method_self_param
        elif func_info and func_info.is_method and func_info.params:
            self_param = func_info.params[0]  # first positional param is self

        if self_param:
            self_vals = self._scope_input(caller, self_param)
            d.self_value_kinds = list({v.kind for v in self_vals})
            d.self_value_details = [f"{v.kind}:{v.name}" for v in self_vals]

            # Also check flow bindings (values discovered during analysis)
            flow_key = (caller, GLOBAL_CONTEXT)
            flow_bindings = self._builder.scope_flow_bindings.get(flow_key, {})
            flow_self = flow_bindings.get(self_param, set())
            if flow_self:
                for v in flow_self:
                    s = f"{v.kind}:{v.name}"
                    if s not in d.self_value_details:
                        d.self_value_details.append(f"[flow] {s}")

        # ── All parameter bindings ──
        inp_key = (caller, GLOBAL_CONTEXT)
        inp_bindings = self._builder.scope_inputs.get(inp_key, {})
        for pname, pvals in inp_bindings.items():
            if pvals:
                d.caller_param_bindings[pname] = sorted(f"{v.kind}:{v.name}" for v in pvals)

        # ── super().__init__() special case ──
        if caller.endswith(".__init__") and callee.endswith(".__init__"):
            # Find the super target
            caller_cls = _class_of_method(caller)
            if caller_cls and caller_cls in self._builder.classes:
                mro = self._mro(caller_cls)
                # The next __init__ in MRO after caller_cls
                for klass in mro[1:]:  # skip self
                    init_name = f"{klass}.__init__"
                    if init_name in self._builder.scopes or init_name in self._builder.functions:
                        d.super_mro_next = klass
                        break
                    # Also check if the class itself has __init__ defined
                    cinfo = self._class_info(klass)
                    if cinfo and "__init__" in cinfo.methods:
                        d.super_mro_next = klass
                        break
                    # If klass has no __init__ but inherits from something with __init__
                    # Check MRO of klass for __init__
                    for super_klass in self._mro(klass)[1:]:
                        sinfo = self._class_info(super_klass)
                        if sinfo and "__init__" in sinfo.methods:
                            d.super_mro_next = super_klass
                            break
                    if d.super_mro_next:
                        break

        # ── Classify ──
        d.category = self._classify(d)
        return d

    def _classify(self, d: EdgeDiagnosis) -> str:
        """Assign fine-grained root cause category."""

        # ── CASE A: Caller doesnʼt exist at all ──
        if not d.caller_in_scopes and not d.caller_in_functions:
            d.evidence.append(
                f"Caller `{d.caller}` is not registered as a scope or function. "
                "It was never collected — likely not reachable from the entry point's "
                "import graph."
            )
            return "CALLER_NOT_COLLECTED"

        # ── CASE B: Callee doesnʼt exist at all ──
        if not d.callee_in_scopes and not d.callee_in_functions:
            d.evidence.append(
                f"Callee `{d.callee}` is not registered as a scope or function. "
            )
            if d.callee_owner_class and d.callee_owner_class in self._builder.classes:
                cinfo = self._class_info(d.callee_owner_class)
                if cinfo:
                    method_name = _method_name(d.callee)
                    if method_name not in cinfo.methods:
                        d.evidence.append(
                            f"Class `{d.callee_owner_class}` exists but method "
                            f"`{method_name}` is not in its `methods` dict. "
                            f"Registered methods: {sorted(cinfo.methods.keys())[:10]}"
                        )
                    else:
                        d.evidence.append(
                            f"Class `{d.callee_owner_class}` exists and `{method_name}` "
                            "is registered as a method, but no corresponding scope was created. "
                            "This may indicate a symbol-collection gap."
                        )
                else:
                    d.evidence.append(
                        f"Class `{d.callee_owner_class}` not found in builder.classes."
                    )
            else:
                d.evidence.append(
                    "Callee class not registered; likely from an unloaded module or stdlib."
                )
            return "CALLEE_NOT_COLLECTED"

        # ── CASE C: super().__init__() pattern ──
        if d.caller.endswith(".__init__") and d.callee.endswith(".__init__"):
            return self._classify_super_init(d)

        # ── CASE D: Method call on self ──
        func_info = self._func_info(d.caller)
        scope_info = self._scope_info(d.caller)
        is_method = (func_info and func_info.is_method) or (
            scope_info and scope_info.method_self_param is not None
        )

        if is_method and d.self_value_kinds:
            # self has values — attribute lookup should have worked
            return self._classify_attr_lookup(d)
        elif is_method and not d.self_value_kinds:
            # self is empty — why?
            return self._classify_empty_self(d)
        else:
            # Plain function call
            return self._classify_plain_call(d)

    def _classify_super_init(self, d: EdgeDiagnosis) -> str:
        """Diagnose super().__init__() resolution failure."""
        caller_cls = _class_of_method(d.caller)
        callee_cls = _class_of_method(d.callee)

        d.evidence.append(
            f"Caller class: `{caller_cls}`  (bases: "
            f"{self._builder.classes[caller_cls].bases if caller_cls in self._builder.classes else '?'})"
        )
        d.evidence.append(f"Callee class: `{callee_cls}`")
        d.evidence.append(f"Caller MRO: {d.caller_mro}")
        d.evidence.append(f"Callee MRO: {d.callee_mro}")

        # ── MRO fracture check ──
        mro_broken = False
        if callee_cls and d.caller_mro and callee_cls not in d.caller_mro:
            cinfo = self._class_info(caller_cls) if caller_cls else None
            if cinfo:
                # Strategy 1: Check resolved bases
                for base in cinfo.bases:
                    base_mro = self._mro(base)
                    if callee_cls in base_mro:
                        mro_broken = True
                        d.evidence.append(
                            f"⚠ MRO FRACTURE: callee `{callee_cls}` IS in the MRO of "
                            f"base `{base}`, but `{base}` itself is NOT in caller's MRO."
                        )
                        break

                # Strategy 2: Check raw AST bases for Subscript nodes that weren't resolved
                if not mro_broken and cinfo.node:
                    import ast as ast_module
                    for raw_base in cinfo.node.bases:
                        if isinstance(raw_base, ast_module.Subscript):
                            # Extract the base name from e.g. Sink[Any] → Sink
                            base_name = None
                            if isinstance(raw_base.value, ast_module.Name):
                                base_name = raw_base.value.id
                            elif isinstance(raw_base.value, ast_module.Attribute):
                                base_name = ast_module.unparse(raw_base.value)
                            if base_name:
                                # Search ALL registered classes for this base name
                                found_cand = None
                                for cls_name in self._builder.classes:
                                    if cls_name.endswith(f".{base_name}") or cls_name == base_name:
                                        cand_mro = self._mro(cls_name)
                                        if callee_cls in cand_mro:
                                            found_cand = cls_name
                                            break
                                if found_cand:
                                    mro_broken = True
                                    d.evidence.append(
                                        f"⚠ MRO FRACTURE (generic subscript): the raw AST base "
                                        f"`{ast_module.unparse(raw_base)}` resolves to class "
                                        f"`{found_cand}` whose MRO includes callee `{callee_cls}`, "
                                        f"but `{found_cand}` was NOT added to the caller's MRO."
                                    )
                                    d.evidence.append(
                                        f"→ Likely cause: `_resolve_class_bases()` does not "
                                        f"handle `ast.Subscript` nodes (generic type parameters). "
                                        f"The base class `{base_name}` was silently dropped."
                                    )
                                if mro_broken:
                                    break
                        elif isinstance(raw_base, ast_module.Call):
                            # e.g., Generic[T, U]
                            base_name = None
                            if isinstance(raw_base.func, ast_module.Name):
                                base_name = raw_base.func.id
                            if base_name:
                                for cand in [f"{cinfo.module}.{base_name}", base_name]:
                                    if cand in self._builder.classes:
                                        cand_mro = self._mro(cand)
                                        if callee_cls in cand_mro:
                                            mro_broken = True
                                            d.evidence.append(
                                                f"⚠ MRO FRACTURE (generic subscript): raw base "
                                                f"`{ast_module.unparse(raw_base)}` → class "
                                                f"`{cand}` (MRO includes callee), but `{cand}` "
                                                f"was not added to caller's MRO."
                                            )
                                            break
                                if mro_broken:
                                    break

            if not mro_broken:
                d.evidence.append(
                    f"Callee class is NOT in caller MRO — this edge may be invalid "
                    "in the ground truth, or the caller's class hierarchy is incomplete."
                )
        elif callee_cls and d.caller_mro and callee_cls in d.caller_mro:
            idx = d.caller_mro.index(callee_cls)
            d.evidence.append(
                f"Callee class IS in caller MRO at position {idx} — "
                "MRO traversal should find it."
            )
            # Why didn't super() resolve to it?
            if d.super_mro_next:
                d.evidence.append(
                    f"However, super() would resolve to `{d.super_mro_next}` first, "
                    f"which is before `{callee_cls}` in the MRO."
                )
            else:
                d.evidence.append(
                    "Could not determine what super() would resolve to. "
                    "The super() protocol model may not be emitting synthetic "
                    "invocations correctly."
                )

        # Check callee registration
        if d.callee_in_scopes:
            d.evidence.append("Callee scope EXISTS.")
        elif d.callee_in_functions:
            d.evidence.append("Callee registered as FunctionInfo but NO scope created.")
        else:
            d.evidence.append(
                "Callee NOT registered as scope or function — it was never collected."
            )

        # Super() resolution: what would happen?
        if caller_cls and caller_cls in self._builder.classes:
            cinfo = self._class_info(caller_cls)
            d.evidence.append(
                f"Caller class direct bases: {cinfo.bases}"
            )
            mro = self._mro(caller_cls)
            # Find first class in MRO (after caller_cls) that has __init__
            found = []
            for klass in mro[1:]:
                k_info = self._class_info(klass)
                if k_info and "__init__" in k_info.methods:
                    found.append(klass)
            if found:
                d.evidence.append(
                    f"Classes in MRO with __init__ defined: {found} — "
                    f"super().__init__() should dispatch to `{found[0]}.__init__`"
                )
            else:
                d.evidence.append(
                    "No class in MRO (after self) has __init__ registered."
                )

        if not d.caller_has_incoming:
            d.evidence.append(
                "Caller has NO incoming edges — the method body is never "
                "analysed with real parameter values (self is empty)."
            )
            return "SUPER_INIT_UNREACHABLE"
        elif d.self_value_kinds:
            d.evidence.append(
                f"Caller HAS incoming edges and self={d.self_value_details[:3]}. "
                "super() resolution failed despite available context."
            )
            return "SUPER_INIT_RESOLUTION_FAILED"
        else:
            d.evidence.append(
                "Caller has incoming edges but self is still empty — "
                "callers may pass ⊤ for the receiver."
            )
            return "SUPER_INIT_EMPTY_SELF"

    def _classify_attr_lookup(self, d: EdgeDiagnosis) -> str:
        """self.xxx() where self has a value but lookup produced nothing."""
        d.evidence.append(
            f"self parameter values: {d.self_value_details[:5]}"
        )

        # What class does self point to?
        for val in d.self_value_details:
            if val.startswith("instance:"):
                instance_of = val.split(":", 1)[1]
                d.evidence.append(f"  → self is instance of `{instance_of}`")
                if instance_of in self._builder.classes:
                    mro = self._mro(instance_of)
                    d.evidence.append(f"  → MRO: {mro}")
                    callee_meth = _method_name(d.callee)
                    found_in = []
                    for klass in mro:
                        cinfo = self._class_info(klass)
                        if cinfo and callee_meth in cinfo.methods:
                            found_in.append(klass)
                    if found_in:
                        d.evidence.append(
                            f"  → Method `{callee_meth}` IS declared in: {found_in} "
                            f"— MRO traversal should find it."
                        )
                    else:
                        d.evidence.append(
                            f"  → Method `{callee_meth}` NOT found in any class in MRO. "
                            f"Check class declarations."
                        )
                break

        # Check callee exists
        if not d.callee_in_scopes and not d.callee_in_functions:
            d.evidence.append(
                f"Callee `{d.callee}` is not registered — attribute lookup "
                "may produce a value that wasn't collected as a scope."
            )
            return "ATTR_LOOKUP_CALLEE_NOT_REGISTERED"

        d.evidence.append(
            "self has a value AND callee exists, but the edge was not produced. "
            "This suggests a gap in _resolve_attribute's INSTANCE_KIND branch or "
            "in the interprocedural binding of the resolved bound method."
        )
        return "ATTR_LOOKUP_FAILED"

    def _classify_empty_self(self, d: EdgeDiagnosis) -> str:
        """Method call but self is empty."""
        func_info = self._func_info(d.caller)
        scope_info = self._scope_info(d.caller)

        if d.caller_has_incoming:
            d.evidence.append(
                f"Caller HAS {len(d.caller_incoming_from)} incoming edge(s): "
                f"{d.caller_incoming_from[:5]}"
            )
            # Check what callers pass
            for src in d.caller_incoming_from[:3]:
                src_returns = self._builder.scope_returns.get((src, GLOBAL_CONTEXT), set())
                if src_returns:
                    d.evidence.append(
                        f"  Caller `{src}` returns: {[str(v) for v in src_returns][:3]}"
                    )
            if d.caller_param_bindings:
                d.evidence.append(
                    f"Other parameters have values: {dict(list(d.caller_param_bindings.items())[:5])}"
                )
            d.evidence.append(
                "self is empty despite the caller having incoming edges. "
                "The callers may pass ⊤ or an incorrectly typed value for the receiver."
            )
            return "EMPTY_SELF_DESPITE_REACHABLE"
        else:
            d.evidence.append(
                "Caller has NO incoming edges — no code invokes this method from the "
                "entry point. self is empty because the analysis never sees a concrete "
                "receiver."
            )
            if d.caller_owner_class and d.caller_owner_class in self._builder.classes:
                # Check if the class itself is instantiated anywhere
                cinfo = self._class_info(d.caller_owner_class)
                if cinfo:
                    # Is there any edge TO this class (e.g., from a constructor)?
                    cls_incoming = self._callers_of(d.caller_owner_class)
                    if cls_incoming:
                        d.evidence.append(
                            f"However, class `{d.caller_owner_class}` IS referenced by: "
                            f"{cls_incoming[:5]}"
                        )
                    else:
                        d.evidence.append(
                            f"Class `{d.caller_owner_class}` also has no incoming "
                            "references — it is entirely unused by reachable code."
                        )
            d.evidence.append(
                "Root cause: reachability gap. The ground truth defines edges based on "
                "class hierarchy regardless of whether any code actually invokes the method."
            )
            return "UNREACHABLE_CALLER"

    def _classify_plain_call(self, d: EdgeDiagnosis) -> str:
        """Plain function call (non-method)."""
        if not d.caller_has_incoming:
            d.evidence.append(
                "Caller has NO incoming edges — plain function is never called."
            )
            return "UNREACHABLE_CALLER"

        d.evidence.append(
            f"Caller has {len(d.caller_incoming_from)} incoming edge(s). "
            f"Parameters: {dict(list(d.caller_param_bindings.items())[:5])}"
        )

        # ── Sub-pattern: recursive call ──
        if d.caller == d.callee:
            d.evidence.append(
                "This is a RECURSIVE call — the function calls itself. "
                "The fact that this edge is missing means the function's own "
                "qualified name cannot be resolved as a call target inside its body. "
                "This typically happens for nested/local functions whose internal "
                "name differs from the scope-registered qualified name."
            )
            return "RECURSIVE_UNRESOLVED"

        # ── Sub-pattern: receiver-like parameter lost to ⊤ ──
        # Check if any parameter named like a receiver (not self/cls) has ⊤ value
        has_top_param = False
        for pname, pvals in d.caller_param_bindings.items():
            if any('unknown' in v for v in pvals):
                has_top_param = True
                if d.callee_owner_class:
                    d.evidence.append(
                        f"Parameter `{pname}` has ⊤ value — this parameter likely "
                        f"serves as the receiver for method calls like "
                        f"`{pname}.{d.callee.split('.')[-1]}()`. "
                        f"Callee class `{d.callee_owner_class}` exists but the "
                        f"receiver identity was lost, so the edge cannot be produced."
                    )
                    return "RECEIVER_LOST"
                else:
                    d.evidence.append(
                        f"Parameter `{pname}` has ⊤ value — this likely causes "
                        f"the call resolution to fail."
                    )
        
        # ── Sub-pattern: callee not in scopes/functions ──
        if not d.callee_in_scopes and not d.callee_in_functions:
            d.evidence.append("Callee not collected.")
            return "CALLEE_NOT_COLLECTED"

        # ── Last resort: inspect the AST to find the call pattern ──
        scope = self._builder.scopes.get(d.caller)
        func_info = self._func_info(d.caller)
        ast_node = None
        if scope:
            ast_node = scope
        elif func_info:
            ast_node = func_info

        if ast_node and hasattr(ast_node, 'body'):
            import ast as ast_module
            callee_short = d.callee.rsplit('.', 1)[-1]
            for stmt in ast_node.body:
                for node in ast_module.walk(stmt):
                    if isinstance(node, ast_module.Call):
                        call_str = ast_module.unparse(node) if hasattr(ast_module, 'unparse') else ''
                        # Check if this call could produce the callee
                        if callee_short in call_str or '__str__' in call_str or callee_short == '__str__':
                            d.evidence.append(f"AST call: `{call_str[:120]}`")
                            # Check if it's str(x) → x.__str__()
                            if isinstance(node.func, ast_module.Name) and node.func.id == 'str':
                                d.evidence.append(
                                    f"`str()` builtin implicitly calls `__str__()` — "
                                    "this is a builtin-protocol gap: PyFlow does not model "
                                    "`str(x)` as an invocation of `x.__str__()`."
                                )
                                return "BUILTIN_PROTOCOL_GAP"
                            # Check receiver: is it a captured variable (not in params)?
                            if isinstance(node.func, ast_module.Attribute):
                                if isinstance(node.func.value, ast_module.Name):
                                    rcvr = node.func.value.id
                                    func_params = (func_info.params if func_info else []) or (scope.params if scope else [])
                                    if rcvr not in func_params:
                                        d.evidence.append(
                                            f"Receiver `{rcvr}` is a CAPTURED variable (not in params "
                                            f"{func_params}). It likely resolves to ⊤ in the closure."
                                        )
                                        return "RECEIVER_LOST"

        d.evidence.append(
            f"caller_in_scopes={d.caller_in_scopes} callee_in_scopes={d.callee_in_scopes} "
            f"self={d.self_value_kinds}"
        )
        return "UNKNOWN"


# ═══════════════════════════════════════════════════════════════════════
# Report formatting
# ═══════════════════════════════════════════════════════════════════════

ROOT_CAUSE_CATALOG: Dict[str, Tuple[str, str]] = {
    # ── Reachability gap (no concrete call path exists) ──
    "UNREACHABLE_CALLER": (
        "Caller is not invoked by any reachable code; self/cls params are ∅. "
        "The method body is never analysed with real receiver values.",
        "Add a synthetic driver that instantiates classes and exercises methods, "
        "or inject conservative self values during class-body analysis.",
    ),
    # ── super().__init__() variants ──
    "SUPER_INIT_UNREACHABLE": (
        "super().__init__() in an unreachable method — compound failure: "
        "lack of reachability PLUS super() protocol model gaps.",
        "Both (a) reachability injection AND (b) super() model fixes are needed.",
    ),
    "SUPER_INIT_EMPTY_SELF": (
        "super().__init__() in a reachable method, but self is still ∅ — "
        "callers pass ⊤ or imprecise receiver values.",
        "Check why callers can't propagate concrete instance types to self.",
    ),
    "SUPER_INIT_RESOLUTION_FAILED": (
        "super().__init__() with non-empty self, but the next __init__ in MRO "
        "was not resolved. The super() protocol model is incomplete or the "
        "class's MRO is broken (e.g. generic subscript not parsed).",
        "Fix generic subscript parsing in _resolve_class_bases(), or improve "
        "the super() protocol reduction to handle indirect __init__ inheritance.",
    ),
    # ── Method calls on self/cls ──
    "EMPTY_SELF_DESPITE_REACHABLE": (
        "Method has incoming calls but self/cls parameter receives no values — "
        "receiver identity was lost during interprocedural propagation.",
        "Trace upstream: which callers invoke this method, and what values do they "
        "pass for the receiver argument? The receiver may be ⊤ or lost in "
        "container/return/closure forwarding.",
    ),
    "ATTR_LOOKUP_FAILED": (
        "self has a concrete instance type AND callee scope exists, but "
        "_resolve_attribute's INSTANCE_KIND branch produced nothing. "
        "This is a gap in MRO traversal or method resolution.",
        "Check: does the callee method appear in the class's MRO methods dict? "
        "Are staticmethod/classmethod decorators handled correctly? "
        "Is the callee registered under the qualified name the lookup expects?",
    ),
    "ATTR_LOOKUP_CALLEE_NOT_REGISTERED": (
        "self has a concrete type and attr lookup may have produced a value, "
        "but that value was never registered as a scope/function.",
        "The callee function was not collected — check symbol collection for "
        "the class containing it (maybe an abstract method or property?).",
    ),
    # ── Callee/caller not collected ──
    "CALLER_NOT_COLLECTED": (
        "The caller function was never collected as a scope. It may be in an "
        "unloaded module, a native extension, or a dynamically generated function.",
        "Check module loading: is the caller's source file resolved and parsed?",
    ),
    "CALLEE_NOT_COLLECTED": (
        "The callee function was never collected as a scope. It may be inherited "
        "from an unloaded stdlib module, a native function, or missed during "
        "symbol collection.",
        "Check: is the callee's containing class/module loaded? Is it a "
        "dynamically generated function (e.g. namedtuple, dataclass __init__)?",
    ),
    # ── Model boundary ──
    "MODEL_BOUNDARY": (
        "Caller HAS a dynamic summary — this is an expected boundary crossing "
        "under the paper's abstraction (Theorem clause 3). The edge exists in "
        "concrete execution but the abstract domain cannot name the target.",
        "Consider extending the protocol model (descriptors, metaclasses, "
        "reflective calls, external library summaries) to recover a named edge.",
    ),
    # ── Implementation gap ──
    "IMPLEMENTATION_GAP": (
        "Caller has NO dynamic summary and NO named edge — the paper's "
        "soundness theorem (clause 2) guarantees an edge should exist for "
        "modeled semantics. This is an implementation bug.",
        "Debug the specific constraint rule that should have produced this edge "
        "([InstMethod], [BoundCall], [FunCall], etc.) and trace why it didn't fire.",
    ),
    # ── Plain function call failures ──
    "RECURSIVE_UNRESOLVED": (
        "A function that calls itself (recursive) cannot resolve its own name. "
        "Typically happens for nested/local functions whose internal name "
        "differs from the scope-registered qualified name.",
        "Check how nested function names are registered vs how they are "
        "looked up in the enclosing scope. The name binding may not propagate.",
    ),
    "RECEIVER_LOST": (
        "A reachable closure/function has a parameter that serves as receiver "
        "for method calls, but that parameter's value is ⊤ (unknown). "
        "Without a concrete receiver type, attribute lookup cannot resolve "
        "the method.",
        "Trace upstream: where does the ⊤-valued parameter come from? "
        "It may originate from a container load, a return value merge, "
        "or a dynamic/reflective operation.",
    ),
    "BUILTIN_PROTOCOL_GAP": (
        "A Python builtin (e.g. `str(x)`, `len(x)`, `iter(x)`) implicitly "
        "invokes a dunder method (`__str__`, `__len__`, `__iter__`), but "
        "PyFlow does not model this connection. The GT records the dunder "
        "method as the callee.",
        "Model the builtin-to-dunder protocol: when `str(x)` is called and "
        "x has a concrete type, emit a synthetic edge to `x.__str__()`.",
    ),
    # ── Fallback ──
    "UNKNOWN": (
        "Could not classify this missing edge into a known category.",
        "Manual investigation needed. Run with -vv to see full evidence.",
    ),
}

def print_report(diagnoses: List[EdgeDiagnosis], project_name: str, verbose: int = 0, file=sys.stdout):
    by_cat: Dict[str, List[EdgeDiagnosis]] = defaultdict(list)
    for d in diagnoses:
        by_cat[d.category].append(d)

    total = len(diagnoses)

    BOLD = "\033[1m"
    RESET = "\033[0m"
    print(f"\n{'='*72}")
    print(f"  {BOLD}DEEP DIAGNOSTIC: {project_name}{RESET}")
    print(f"  {total} missing edges analysed")
    print(f"{'='*72}")

    print(f"\n{'─'*72}")
    print(f"  ROOT CAUSE CATALOG")
    print(f"{'─'*72}")
    for cat in sorted(by_cat, key=lambda c: len(by_cat[c]), reverse=True):
        count = len(by_cat[cat])
        pct = count / total * 100
        desc = ROOT_CAUSE_CATALOG.get(cat, ("(no description)", ""))
        print(f"  {BOLD}{cat}{RESET}  ({count:3d}, {pct:5.1f}%)")
        print(f"    {desc[0]}")

    if verbose == 0:
        # Just summary — done
        return

    # -v & -vv: per-category details
    for cat in sorted(by_cat, key=lambda c: len(by_cat[c]), reverse=True):
        items = by_cat[cat]
        print(f"{'─'*72}")
        print(f"  {cat}  ({len(items)} edges)")
        print(f"{'─'*72}")

        if verbose >= 2:
            # -vv: show ALL edges with full evidence
            show = items
        else:
            # -v: show first 3 examples per category + 1 line per remaining
            show = items[:3]

        for diag in show:
            print(f"\n  ✗  {diag.caller}")
            print(f"     → {diag.callee}")
            for ev in diag.evidence:
                for line in textwrap.wrap(ev, width=66, initial_indent="     • ", subsequent_indent="       "):
                    print(line)

        if verbose < 2 and len(items) > 3:
            print(f"\n  ... and {len(items) - 3} more edges of this type")
            # List remaining edge pairs compactly
            for diag in items[3:]:
                print(f"      {diag.caller}  →  {diag.callee}")
        print()


# ═══════════════════════════════════════════════════════════════════════
# Project discovery (same as bench_repo_callgraph.py)
# ═══════════════════════════════════════════════════════════════════════

def _load_gt(gt_path: Path) -> Dict[str, List[str]]:
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    # Remove metadata keys from GT edge dict
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _resolve_entry(root: Path, manifest_entry: Optional[Dict] = None) -> Optional[Path]:
    gt_path = root / "callgraph.json"
    if not gt_path.exists():
        return None
    try:
        raw = json.loads(gt_path.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    entry_name = raw.get("_entry_point", "main.py")
    candidate = root / str(entry_name)
    if candidate.is_file():
        return candidate
    if manifest_entry:
        mf = manifest_entry.get("_entry_file")
        if mf:
            candidate = root / str(mf)
            if candidate.is_file():
                return candidate
    for fb in ("main.py", "__init__.py", f"{root.name}.py"):
        candidate = root / fb
        if candidate.is_file():
            return candidate
    return None


def diagnose_project(builder: ConstraintCallGraphBuilder, cg, project_root: Path) -> List[EdgeDiagnosis]:
    gt_path = project_root / "callgraph.json"
    gt = _load_gt(gt_path)
    diagnoser = MissingEdgeDiagnoser(builder, gt, cg)
    return diagnoser.diagnose_all()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Deep-diagnose missing call-graph edges.")
    parser.add_argument("--corpus", type=Path, default=Path("evaluation/repo_level"))
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v: show per-category examples; -vv: show all edges with full evidence")
    args = parser.parse_args()

    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"[ERROR] Corpus not found: {corpus}", file=sys.stderr)
        sys.exit(1)

    manifest_path = corpus / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] No manifest.json in {corpus}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets: List[Tuple[str, Path, Optional[str]]] = []
    for entry in manifest.get("projects", []):
        name = entry["name"]
        if args.project and name != args.project:
            continue
        proj_dir = corpus / entry["path"]
        if not proj_dir.is_dir():
            continue
        gt_path = proj_dir / "callgraph.json"
        if not gt_path.exists():
            continue
        entry_file = _resolve_entry(proj_dir, entry)
        if not entry_file:
            print(f"  [WARN] {name}: no entry file found", file=sys.stderr)
            continue
        targets.append((name, proj_dir, entry_file))

    if not targets:
        print("[ERROR] No projects found", file=sys.stderr)
        sys.exit(1)

    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("timeout")

    for name, proj_dir, entry_file in targets:
        BOLD = "\033[1m"
        RESET = "\033[0m"
        print(f"\n{'─'*72}\n  {BOLD}{name}{RESET}\n{'─'*72}", file=sys.stderr)
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(120)

            print(f"  Building constraint engine ...", file=sys.stderr)
            options = AnalysisOptions(
                warn_on_fixpoint_truncation=False,
                skip_stdlib_modules=True,
                emit_solver_stats=False,
            )
            builder = ConstraintCallGraphBuilder(
                entry_file.read_text(encoding="utf-8"),
                entry_path=str(entry_file),
                options=options,
            )
            cg = builder.build()

            signal.alarm(0)

            print(f"  Diagnosing missing edges ...", file=sys.stderr)
            diagnoses = diagnose_project(builder, cg, proj_dir)

            out = open(args.output, "w") if args.output else sys.stdout
            try:
                print_report(diagnoses, name, verbose=args.verbose, file=out)
            finally:
                if args.output:
                    out.close()

        except TimeoutError:
            print(f"  [SKIP] Timed out", file=sys.stderr)
        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
