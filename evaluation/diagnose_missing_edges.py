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

# Import normalization helper from the sibling bench module (same directory).
import sys as _sys
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in _sys.path:
    _sys.path.insert(0, _this_dir)
from bench_repo_callgraph import (
    normalize_callgraph_name,
    _compute_analysed_callers,
    _normalize_gt,
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
    def __init__(
        self, builder: ConstraintCallGraphBuilder, gt: Dict[str, List[str]],
        cg, project_root: Path, *,
        project_name: str = "",
        entry_file: Optional[Path] = None,
        whole_program: bool = False,
    ):
        self._builder = builder
        self._gt = gt
        self._cg_graph = cg
        self._project_root = project_root
        self._project_name = project_name
        self._entry_file = entry_file
        self._whole_program = whole_program
        self._cg_edges: Set[Tuple[str, str]] = set()
        self._cg_dynamic: Set[str] = set()

        # Helper to normalize a name using the same logic as the bench runner
        self._norm = lambda n: normalize_callgraph_name(
            n, project_name, entry_file=entry_file,
        )

        # Pre-compute edges from the built graph (normalized)
        for caller, callee in cg.edges():
            nc = self._norm(caller)
            nt = self._norm(callee)
            self._cg_edges.add((nc, nt))

        # Use the shared definition of "analysed" from the bench module.
        raw_graph = cg.get()
        minimal_norm: Dict[str, List[str]] = {}
        for caller, callees in raw_graph.items():
            nc = self._norm(caller)
            if nc.startswith("<"):
                continue
            kept = [self._norm(c) for c in callees if not c.startswith("<")]
            minimal_norm[nc] = kept
        self._reachable_callers = _compute_analysed_callers(minimal_norm)

        # Which callers have dynamic summaries (normalized)
        for caller, callees in cg.get().items():
            if any(c.startswith("<dynamic") for c in callees):
                self._cg_dynamic.add(self._norm(caller))

        # Build normalized scope/function sets for existence checks,
        # plus reverse mappings (normalized → raw) for deep inspection.
        self._normalized_scopes: Set[str] = set()
        self._norm_to_raw_scope: Dict[str, str] = {}
        for s in builder.scopes:
            ns = self._norm(s)
            self._normalized_scopes.add(ns)
            self._norm_to_raw_scope[ns] = s

        self._normalized_functions: Set[str] = set()
        self._norm_to_raw_func: Dict[str, str] = {}
        for f in builder.functions:
            nf = self._norm(f)
            self._normalized_functions.add(nf)
            self._norm_to_raw_func[nf] = f

        # Also for classes used in MRO / hierarchy checks
        self._norm_to_raw_class: Dict[str, str] = {}
        for c in builder.classes:
            nc = self._norm(c)
            self._norm_to_raw_class[nc] = c

        # Build reverse index: unqualified function name → list of qualified names
        # (uses normalized names so lookup matches GT convention)
        self._unqualified_index: Dict[str, List[str]] = defaultdict(list)
        for scope_name in builder.scopes:
            ns = self._norm(scope_name)
            unq = ns.rsplit('.', 1)[-1]
            self._unqualified_index[unq].append(ns)
        for func_name in builder.functions:
            nf = self._norm(func_name)
            unq = nf.rsplit('.', 1)[-1]
            if nf not in self._unqualified_index.get(unq, []):
                self._unqualified_index[unq].append(nf)

    # ── lookup helpers (operate on normalized names, resolve to raw for builder) ──

    def _raw_scope(self, norm_name: str) -> Optional[str]:
        """Resolve a normalized scope name back to the raw builder key."""
        if norm_name in self._builder.scopes:
            return norm_name
        return self._norm_to_raw_scope.get(norm_name)

    def _raw_func(self, norm_name: str) -> Optional[str]:
        if norm_name in self._builder.functions:
            return norm_name
        return self._norm_to_raw_func.get(norm_name)

    def _raw_class(self, norm_name: str) -> Optional[str]:
        if norm_name in self._builder.classes:
            return norm_name
        return self._norm_to_raw_class.get(norm_name)

    def _raw_name(self, norm_name: str) -> str:
        """Best-effort reverse-normalize: return the raw builder name."""
        return (
            self._raw_scope(norm_name)
            or self._raw_func(norm_name)
            or self._raw_class(norm_name)
            or norm_name
        )

    def _func_info(self, name: str):
        raw = self._raw_func(name) or self._raw_scope(name)
        if raw:
            return self._builder.functions.get(raw)
        return None

    def _scope_info(self, name: str):
        raw = self._raw_scope(name)
        if raw:
            return self._builder.scopes.get(raw)
        return None

    def _class_info(self, name: str):
        raw = self._raw_class(name)
        if raw:
            return self._builder.classes.get(raw)
        return None

    def _mro(self, class_name: str) -> List[str]:
        raw = self._raw_class(class_name)
        if raw:
            return self._builder._mro(raw)
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

    def _scope_exists(self, name: str) -> bool:
        """Check if a scope exists under either raw or normalized name."""
        if name in self._builder.scopes:
            return True
        return name in self._normalized_scopes

    def _func_exists(self, name: str) -> bool:
        """Check if a function exists under either raw or normalized name."""
        if name in self._builder.functions:
            return True
        return name in self._normalized_functions

    # ── main diagnosis ──

    def diagnose_all(self) -> List[EdgeDiagnosis]:
        # Apply the same GT normalisation as the bench (strip <builtin> /
        # <str> / <list> / ... callees).
        gt_edges: Set[Tuple[str, str]] = set()
        for caller, callees in _normalize_gt(self._gt).items():
            nc = self._norm(caller)
            for callee in callees:
                nt = self._norm(callee)
                gt_edges.add((nc, nt))

        # DDA: suppress edges from callers PyFlow never reached
        if not self._whole_program:
            gt_edges = {
                (c, t) for c, t in gt_edges if c in self._reachable_callers
            }

        missing = sorted(gt_edges - self._cg_edges)
        results = []
        for caller, callee in missing:
            d = self._diagnose_one(caller, callee)
            results.append(d)
        return results

    def _diagnose_one(self, caller: str, callee: str) -> EdgeDiagnosis:
        # Normalized names for display; raw names for builder lookups
        d = EdgeDiagnosis(caller=caller, callee=callee, category="UNKNOWN")
        raw_caller = self._raw_name(caller)
        raw_callee = self._raw_name(callee)

        # ── Basic existence checks (using normalized-aware lookup) ──
        d.caller_in_scopes = self._scope_exists(caller)
        d.caller_in_functions = self._func_exists(caller)
        d.callee_in_scopes = self._scope_exists(callee)
        d.callee_in_functions = self._func_exists(callee)

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
            return self._classify_not_collected(d, is_caller=True)

        # ── CASE B: Callee doesnʼt exist at all ──
        if not d.callee_in_scopes and not d.callee_in_functions:
            return self._classify_not_collected(d, is_caller=False)

        # ── CASE C: MRO broken by generic subscript? ──
        mro_result = self._classify_mro(d)
        if mro_result:
            return mro_result

        # ── CASE D: Method call ──
        func_info = self._func_info(d.caller)
        scope_info = self._scope_info(d.caller)
        is_method = (func_info and func_info.is_method) or (
            scope_info and scope_info.method_self_param is not None
        )

        if is_method and d.self_value_kinds:
            return self._classify_attr_lookup(d)
        elif is_method and not d.self_value_kinds:
            return self._classify_empty_self(d)
        else:
            return self._classify_plain_call(d)

    def _classify_mro(self, d: EdgeDiagnosis) -> Optional[str]:
        """Check if the missing edge is caused by MRO breakage from generic subscript.
        
        Returns a category string if MRO breakage is detected, None otherwise.
        """
        caller_cls = _class_of_method(d.caller)
        callee_cls = _class_of_method(d.callee)
        if not caller_cls or not callee_cls:
            return None

        cinfo = self._class_info(caller_cls)
        if not cinfo or not cinfo.node:
            return None

        # Check: are there raw AST bases with Subscript that were dropped?
        import ast as ast_module
        has_subscript_base = False
        for raw_base in cinfo.node.bases:
            if isinstance(raw_base, ast_module.Subscript):
                has_subscript_base = True
                break
            if isinstance(raw_base, ast_module.Call):
                has_subscript_base = True
                break

        if not has_subscript_base:
            return None

        # Check: is callee class reachable through one of the DROPPED bases?
        mro = self._mro(caller_cls)
        if callee_cls in mro:
            return None  # callee IS in MRO — not an MRO problem

        # The callee class is NOT in the caller's MRO.
        # Is it in the MRO of any DROPPED base?
        for raw_base in cinfo.node.bases:
            if not isinstance(raw_base, (ast_module.Subscript, ast_module.Call)):
                continue
            # Extract base class name from the Subscript/Call
            base_value = raw_base.value if isinstance(raw_base, ast_module.Subscript) else raw_base.func
            if isinstance(base_value, ast_module.Name):
                base_name = base_value.id
            elif isinstance(base_value, ast_module.Attribute):
                base_name = ast_module.unparse(base_value)
            else:
                continue

            # Find the actual qualified class for this base name
            for cls_name in self._builder.classes:
                if cls_name.endswith(f".{base_name}") or cls_name == base_name:
                    base_mro = self._mro(cls_name)
                    if callee_cls in base_mro:
                        d.evidence.append(
                            f"⚠ MRO BROKEN: generic subscript `{ast_module.unparse(raw_base)}` "
                            f"was dropped by _resolve_class_bases()."
                        )
                        d.evidence.append(
                            f"  Caller class: `{caller_cls}`  resolved bases: {cinfo.bases}"
                        )
                        d.evidence.append(
                            f"  Dropped base `{cls_name}` has MRO {base_mro} which "
                            f"includes callee class `{callee_cls}`."
                        )
                        d.evidence.append(
                            f"  Caller's actual MRO: {mro}  ← callee missing from this chain."
                        )
                        d.evidence.append(
                            f"→ Root cause: _resolve_class_bases() does not handle "
                            f"ast.Subscript (generic type parameters). "
                            f"The base class `{base_name}` was silently dropped, "
                            f"breaking the entire MRO chain."
                        )
                        return "MRO_BROKEN"

        return None

    def _classify_not_collected(self, d: EdgeDiagnosis, is_caller: bool) -> str:
        """Sub-classify a NOT_COLLECTED edge."""
        name = d.caller if is_caller else d.callee
        role = "Caller" if is_caller else "Callee"
        unq = name.rsplit('.', 1)[-1]

        # ── Check 1: Naming mismatch ──
        candidates = self._unqualified_index.get(unq, [])
        if candidates:
            d.evidence.append(
                f"{role} `{name}` not found, but {len(candidates)} scope(s) with "
                f"same unqualified name `{unq}` EXIST: {candidates[:5]}"
            )
            d.evidence.append(
                f"→ NAMING MISMATCH: function WAS collected, but under a different "
                f"qualified name. GT expects `{name}`, PyFlow uses `{candidates[0]}`."
            )
            d.evidence.append(
                "→ Cause: entry file loaded without package context "
                "(e.g. `sshtunnel.py` → module `main`, not `sshtunnel`)."
            )
            return "NAMING_MISMATCH"

        # ── Check 2: Is the module loaded? ──
        parts = name.split('.')
        module_candidate = None
        for i in range(len(parts), 0, -1):
            candidate = '.'.join(parts[:i])
            if candidate in self._builder.modules:
                module_candidate = candidate
                break

        if module_candidate:
            mod_info = self._builder.modules[module_candidate]
            import ast as ast_module
            for node in ast_module.walk(mod_info.tree):
                if isinstance(node, (ast_module.FunctionDef, ast_module.AsyncFunctionDef)):
                    if node.name == unq:
                        d.evidence.append(
                            f"{role} `{name}` DEFINED in AST of loaded module "
                            f"`{module_candidate}` (line {node.lineno}), but NOT "
                            f"collected as scope → SYMBOL-COLLECTION BUG."
                        )
                        return "SYMBOL_NOT_COLLECTED"
            d.evidence.append(
                f"Module `{module_candidate}` loaded, but `{unq}` not in its AST. "
                f"May be inherited or from another module."
            )

        # ── Check 3: File on disk? ──
        found_file = None
        for py_file in sorted(self._project_root.rglob('*.py')):
            try:
                if f'def {unq}' in py_file.read_text() or f'class {unq}' in py_file.read_text():
                    found_file = py_file
                    break
            except Exception:
                pass

        if found_file:
            d.evidence.append(
                f"`{unq}` found in `{found_file.relative_to(self._project_root)}`, "
                f"but file NOT loaded → IMPORT-RESOLUTION GAP."
            )
            return "MODULE_NOT_LOADED"
        else:
            d.evidence.append(
                f"`{name}` not in any loaded module or project file → "
                f"external/third-party code."
            )
            return "EXTERNAL_NOT_AVAILABLE"

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
        # ── Priority: RECURSIVE > BUILTIN > ATTR ──
        if d.caller == d.callee:
            d.evidence.append(
                f"self calls itself: `{d.caller}` → `{d.callee}`. "
                "This is a recursive self-call that should have been resolved."
            )
            return "RECURSIVE_UNRESOLVED"

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

        # ── AST-level pattern check ──
        callee_method = _method_name(d.callee)
        scope = self._builder.scopes.get(d.caller)
        if scope:
            import ast as ast_module
            found_self_call = False
            for stmt in scope.body:
                for node in ast_module.walk(stmt):
                    # Pattern: `x in y` → `y.__contains__(x)`
                    if isinstance(node, ast_module.Compare):
                        for op in node.ops:
                            if isinstance(op, (ast_module.In, ast_module.NotIn)):
                                if callee_method == '__contains__':
                                    d.evidence.append(
                                        f"Call pattern: `x in y` → GT expects `y.__contains__()`. "
                                        "This is a BUILTIN_PROTOCOL_GAP (operator protocol)."
                                    )
                                    return "BUILTIN_PROTOCOL_GAP"
                    if not isinstance(node, ast_module.Call):
                        continue
                    # Pattern: str(x) / len(x) / iter(x) → builtin protocol gap
                    if isinstance(node.func, ast_module.Name):
                        if node.func.id in ('str', 'len', 'iter', 'repr', 'bool', 'int'):
                            if callee_method == '__str__' and node.func.id == 'str':
                                d.evidence.append(
                                    f"Call pattern is `str(x)` → GT expects edge to `x.__str__()`. "
                                    f"This is a BUILTIN_PROTOCOL_GAP."
                                )
                                return "BUILTIN_PROTOCOL_GAP"
                    # Track self.xxx() calls
                    if isinstance(node.func, ast_module.Attribute):
                        if isinstance(node.func.value, ast_module.Name):
                            if node.func.value.id == 'self' and node.func.attr == callee_method:
                                found_self_call = True
                                call_str = ast_module.unparse(node)[:120]
                                d.evidence.append(f"Call pattern: `{call_str}`")
            
            # If callee method is never called via self.xxx(), this is NOT an attr-lookup issue
            if not found_self_call:
                d.evidence.append(
                    f"Callee `{callee_method}` is NOT called via self.xxx() in the caller body. "
                    "The receiver is a parameter, local variable, or container element — "
                    "this is a RECEIVER_LOST issue, not an attr-lookup failure on self."
                )
                return "RECEIVER_LOST"

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
    "UNREACHABLE_CALLER": (
        "self/cls empty because caller is never invoked from entry point.",
        "Add synthetic driver, or inject conservative self values in class-body analysis.",
    ),
    "MRO_BROKEN": (
        "Generic subscript (e.g. Sink[Any]) dropped by _resolve_class_bases() → MRO missing.",
        "Fix _resolve_class_bases() to handle ast.Subscript: extract the base class from Subscript.value.",
    ),
    "EMPTY_SELF_DESPITE_REACHABLE": (
        "Method has incoming edges but self is ∅ (caller is also unreachable).",
        "Transitive reachability gap — not a bug in the method itself.",
    ),
    "ATTR_LOOKUP_FAILED": (
        "self has type, callee exists, but _resolve_attribute produced nothing.",
        "Check MRO traversal, method registration names, staticmethod/classmethod handling.",
    ),
    "NAMING_MISMATCH": (
        "Function collected but under a different qualified name prefix.",
        "Set _entry_file in manifest.json, or normalize names in benchmark runner.",
    ),
    "SYMBOL_NOT_COLLECTED": (
        "Function defined in loaded module AST but not collected → _collect_symbols() bug.",
        "Debug why symbol collector skipped this definition.",
    ),
    "SCOPE_NOT_CREATED": (
        "Method in ClassInfo.methods but no scope → _initialize_scopes() gap.",
        "Check if scope initializer filters out certain methods.",
    ),
    "MODULE_NOT_LOADED": (
        "Source file exists on disk but was not loaded — entry point doesn't reach it.",
        "Module-level reachability gap: entry file doesn't import this module.",
    ),
    "EXTERNAL_NOT_AVAILABLE": (
        "Not defined in any project file — builtin or third-party code.",
        "Legitimate gap: PyFlow cannot analyze code it doesn't have.",
    ),
    "RECURSIVE_UNRESOLVED": (
        "Function calls itself but can't resolve its own qualified name.",
        "Scope name resolution issue — check _eval_expr for ast.Name lookup.",
    ),
    "RECEIVER_LOST": (
        "Closure/function has ⊤-valued parameter used as receiver — conservative may-analysis.",
        "Expected behavior: closure-captured types can be lost to ⊤. Not a bug.",
    ),
    "BUILTIN_PROTOCOL_GAP": (
        "Builtin (e.g. str(x)) implicitly calls dunder (__str__) but PyFlow doesn't model it.",
        "Model builtin-to-dunder protocol: str(x) → x.__str__(), etc.",
    ),
    "UNKNOWN": (
        "Could not classify — manual investigation needed.",
        "Run with -vv to see full evidence.",
    ),
}

# Tier classification (used by print_report and CLI help)
FIXABLE_TIER = {
    "MRO_BROKEN", "ATTR_LOOKUP_FAILED", "RECURSIVE_UNRESOLVED",
    "BUILTIN_PROTOCOL_GAP", "SYMBOL_NOT_COLLECTED", "SCOPE_NOT_CREATED",
}
REACHABILITY_TIER = {
    "UNREACHABLE_CALLER", "RECEIVER_LOST", "EMPTY_SELF_DESPITE_REACHABLE",
    "EXTERNAL_NOT_AVAILABLE", "MODULE_NOT_LOADED",
}
INFRA_TIER = {
    "NAMING_MISMATCH",
}

# ANSI colors used by both print_report and CLI help/errors
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_BLUE = "\033[34m"
C_RESET = "\033[0m"

def print_report(diagnoses: List[EdgeDiagnosis], project_name: str, verbose: int = 1,
                 show_filter: Optional[Set[str]] = None, file=sys.stdout):
    """Print per-project diagnostic report.
    
    verbose=1: catalog only (per-repo summary)
    verbose=2: catalog + 3 examples per category + compact list (-v)
    verbose=3: catalog + all edges with full evidence (-vv)
    """
    by_cat: Dict[str, List[EdgeDiagnosis]] = defaultdict(list)
    for d in diagnoses:
        by_cat[d.category].append(d)

    total = len(diagnoses)

    FIXABLE = FIXABLE_TIER
    REACHABILITY = REACHABILITY_TIER
    INFRA = INFRA_TIER

    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

    # Tight tags for summary line
    T_FIX = f"{RED}[fix]{RESET}"
    T_REACH = f"{GREEN}[reach]{RESET}"
    T_NAME = f"{BLUE}[name]{RESET}"

    # Fixed-width tags (8 visual chars including trailing space) for catalog alignment
    TAG_FIX  = f"{RED}[fix]   {RESET}"
    TAG_REACH = f"{GREEN}[reach] {RESET}"
    TAG_NAME  = f"{BLUE}[name]  {RESET}"

    def _tag(cat: str) -> str:
        if cat in FIXABLE:
            return TAG_FIX
        elif cat in REACHABILITY:
            return TAG_REACH
        elif cat in INFRA:
            return TAG_NAME
        return "         "

    # ── Header ──
    print(f"\n  {total} missing edges")

    # ── Catalog ──
    fixable_count = sum(len(by_cat.get(c, [])) for c in FIXABLE)
    infra_count = sum(len(by_cat.get(c, [])) for c in INFRA)
    reach_count = sum(len(by_cat.get(c, [])) for c in REACHABILITY)
    other_count = total - fixable_count - infra_count - reach_count

    for cat in sorted(by_cat, key=lambda c: len(by_cat[c]), reverse=True):
        count = len(by_cat[cat])
        pct = count / total * 100
        desc = ROOT_CAUSE_CATALOG.get(cat, ("", ""))[0]
        tag = _tag(cat)
        # All tags have same visible width (8 chars including trailing space).
        # ANSI codes in tag don't affect visual positioning of subsequent chars.
        pad = 54 - len(cat)
        if pad < 1:
            pad = 1
        print(f"  {tag}{cat}{' ' * pad}{count:>4d} ({pct:5.1f}%)")
        if desc:
            print(f"       {desc}")

    print(f"\n  {T_FIX}={fixable_count}  {T_NAME}={infra_count}  {T_REACH}={reach_count}  other={other_count}")

    if verbose <= 1:
        return

    if verbose >= 3:
        print(f"\n  ── details (-vvv) ──")
    else:
        print(f"\n  ── details (-vv) ──")

    for cat in sorted(by_cat, key=lambda c: len(by_cat[c]), reverse=True):
        if show_filter and cat not in show_filter:
            continue
        items = by_cat[cat]
        tag = _tag(cat)
        desc = ROOT_CAUSE_CATALOG.get(cat, ("", ""))
        print(f"\n  {tag} {cat}  ({len(items)} edges)")
        if desc[1]:
            print(f"  → {desc[1]}")

        show = items if verbose >= 3 else items[:3]
        for diag in show:
            print(f"\n    ✗ {diag.caller}")
            print(f"      → {diag.callee}")
            for ev in diag.evidence:
                for line in textwrap.wrap(ev, width=64, initial_indent="      • ", subsequent_indent="        "):
                    print(line)

        if verbose < 3 and len(items) > 3:
            print(f"\n    ... and {len(items) - 3} more")
            for diag in items[3:]:
                print(f"      {diag.caller}  →  {diag.callee}")


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


def diagnose_project(
    builder: ConstraintCallGraphBuilder, cg, project_root: Path, *,
    project_name: str = "",
    entry_file: Optional[Path] = None,
    whole_program: bool = False,
) -> List[EdgeDiagnosis]:
    gt_path = project_root / "callgraph.json"
    gt = _load_gt(gt_path)
    diagnoser = MissingEdgeDiagnoser(
        builder, gt, cg, project_root,
        project_name=project_name,
        entry_file=entry_file,
        whole_program=whole_program,
    )
    return diagnoser.diagnose_all()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def _print_all_repo_summary(
    all_results: List[Tuple[str, List[EdgeDiagnosis]]],
    show_filter: Optional[Set[str]] = None,
    file=sys.stdout,
):
    """Print combined overview across all repos."""
    total = 0
    combined: Dict[str, int] = defaultdict(int)
    for name, diagnoses in all_results:
        total += len(diagnoses)
        for d in diagnoses:
            combined[d.category] += 1

    if total == 0:
        print("No missing edges across all projects.", file=file)
        return

    print(f"\n  {total} missing edges across {len(all_results)} projects", file=file)

    cat_order = sorted(combined.keys(), key=lambda c: combined[c], reverse=True)
    for cat in cat_order:
        count = combined[cat]
        pct = count / total * 100
        # Build tag: color-wrapped fixed-width label
        if cat in FIXABLE_TIER:
            tag = f"{C_RED}[fix]   {C_RESET}"
        elif cat in REACHABILITY_TIER:
            tag = f"{C_GREEN}[reach] {C_RESET}"
        elif cat in INFRA_TIER:
            tag = f"{C_BLUE}[name]  {C_RESET}"
        else:
            tag = "         "
        pad = 54 - len(cat)
        if pad < 1:
            pad = 1
        print(f"  {tag}{cat}{' ' * pad}{count:>4d} ({pct:5.1f}%)", file=file)

    fix_total = sum(combined.get(c, 0) for c in FIXABLE_TIER)
    reach_total = sum(combined.get(c, 0) for c in REACHABILITY_TIER)
    name_total = sum(combined.get(c, 0) for c in INFRA_TIER)
    print(f"\n  {C_RED}[fix]{C_RESET}={fix_total}  {C_BLUE}[name]{C_RESET}={name_total}  {C_GREEN}[reach]{C_RESET}={reach_total}", file=file)


def main():
    # Build a color-coded root cause listing for help/errors
    _fix_cats = sorted([c for c in ROOT_CAUSE_CATALOG if c in FIXABLE_TIER])
    _name_cats = sorted([c for c in ROOT_CAUSE_CATALOG if c in INFRA_TIER])
    _reach_cats = sorted([c for c in ROOT_CAUSE_CATALOG if c in REACHABILITY_TIER])
    _other_cats = sorted([c for c in ROOT_CAUSE_CATALOG if c not in FIXABLE_TIER and c not in INFRA_TIER and c not in REACHABILITY_TIER])
    _cat_list = (
        f"{C_RED}[fix]{C_RESET} " + ", ".join(_fix_cats) + "\n"
        f"{C_BLUE}[name]{C_RESET} " + ", ".join(_name_cats) + "\n"
        f"{C_GREEN}[reach]{C_RESET} " + ", ".join(_reach_cats)
    )
    if _other_cats:
        _cat_list += "\nother: " + ", ".join(_other_cats)

    parser = argparse.ArgumentParser(
        description="Deep-diagnose missing call-graph edges.",
        epilog=f"Root cause categories:\n{_cat_list}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", type=Path, default=Path("evaluation/repo_level"))
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v: per-repo catalog; -vv: +3 examples each; -vvv: all edges")
    parser.add_argument("--show", action="append", default=None,
                        help="Only show verbose details for this root cause (repeatable)")
    parser.add_argument(
        "--whole-program",
        action="store_true",
        default=False,
        help="Whole-program analysis: diagnose all ground-truth edges. "
             "When off (default), only diagnose edges whose caller was "
             "reached by the engine (demand-driven evaluation).",
    )
    args = parser.parse_args()
    
    # Validate --show values
    if args.show:
        unknown = set(args.show) - set(ROOT_CAUSE_CATALOG.keys())
        if unknown:
            print(f"[ERROR] Unknown root cause(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"        {C_RED}[fix]{C_RESET} " + ", ".join(sorted(c for c in ROOT_CAUSE_CATALOG if c in FIXABLE_TIER)), file=sys.stderr)
            print(f"        {C_BLUE}[name]{C_RESET} " + ", ".join(sorted(c for c in ROOT_CAUSE_CATALOG if c in INFRA_TIER)), file=sys.stderr)
            print(f"        {C_GREEN}[reach]{C_RESET} " + ", ".join(sorted(c for c in ROOT_CAUSE_CATALOG if c in REACHABILITY_TIER)), file=sys.stderr)
            other_c = sorted(c for c in ROOT_CAUSE_CATALOG if c not in FIXABLE_TIER and c not in INFRA_TIER and c not in REACHABILITY_TIER)
            if other_c:
                print(f"        other: " + ", ".join(other_c), file=sys.stderr)
            sys.exit(1)
    show_filter = set(args.show) if args.show else None

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

    all_results: List[Tuple[str, List[EdgeDiagnosis]]] = []
    out = open(args.output, "w") if args.output else sys.stdout

    mode_tag = "[WPA]" if args.whole_program else "[DDA]"
    for name, proj_dir, entry_file in targets:
        YELLOW = "\033[33;1m"
        RESET = "\033[0m"
        print(f"\n{YELLOW}{name}{RESET} {mode_tag}", file=sys.stderr)
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
            diagnoses = diagnose_project(
                builder, cg, proj_dir,
                project_name=name,
                entry_file=entry_file,
                whole_program=args.whole_program,
            )
            all_results.append((name, diagnoses))

            # Print per-project report immediately when -v/-vv/-vvv
            if args.verbose > 0:
                print_report(diagnoses, name, verbose=args.verbose,
                            show_filter=show_filter, file=out)
                if args.output:
                    out.flush()
                else:
                    sys.stdout.flush()

        except TimeoutError:
            print(f"  [SKIP] Timed out", file=sys.stderr)
        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    if not all_results:
        print("[ERROR] All projects failed", file=sys.stderr)
        if args.output:
            out.close()
        sys.exit(1)

    try:
        if args.verbose == 0:
            # Default: all-repo summary (printed once at end)
            _print_all_repo_summary(all_results, show_filter=show_filter, file=out)
    finally:
        if args.output:
            out.close()


if __name__ == "__main__":
    main()
