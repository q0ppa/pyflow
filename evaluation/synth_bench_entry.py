#!/usr/bin/env python3
"""Auto-synthesise benchmark entry-point files to improve DDA coverage.

For each project in the corpus this tool scans the projectʼs Python
sources, collects public classes and functions, and generates a driver
file (``__pyflow_bench_synth__.py``) that imports and exercises every
discovered symbol.

Usage:
    # Generate synth entry files for all projects
    python evaluation/synth_bench_entry.py --corpus evaluation/repo_level

    # Run the benchmark with synth entries
    python evaluation/bench_repo_callgraph.py --corpus evaluation/repo_level

    # Clean up — remove synth files, restore original manifest
    python evaluation/synth_bench_entry.py --corpus evaluation/repo_level --clean
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════
# Synthesis engine
# ═══════════════════════════════════════════════════════════════════════

SYNTH_FILENAME = "__pyflow_bench_synth__.py"


def _resolve_pkg_info(project_root: Path, entry_file: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    """Return (pkg_root, pkg_name) for the project.

    *pkg_root* is the directory that serves as the Python package root
    (the directory whose *name* is the top-level package and whose
    subdirectories are subpackages).  *pkg_name* is the dotted import
    prefix.  Returns ``(None, None)`` for single-file projects.
    """
    if entry_file:
        entry = project_root / entry_file
    else:
        for candidate in ["__init__.py", "main.py", f"{project_root.name}.py"]:
            entry = project_root / candidate
            if entry.is_file():
                break
        else:
            return None, None

    if not entry.is_file():
        return None, None

    # Single .py file at project root → no package (import as bare module)
    if entry.suffix == ".py" and entry.parent == project_root and entry.name != "__init__.py":
        return None, None

    # Walk the entryʼs parent chain outward until the parent directory
    # no longer has __init__.py.  The innermost directory WITH
    # __init__.py is the package root.
    pkg_root = entry.parent
    while (pkg_root.parent / "__init__.py").exists():
        pkg_root = pkg_root.parent

    # Package name: the top-level import name is simply the directory
    # name of pkg_root (e.g. ``rich_cli``, ``cli_tool``).  Subpackages
    # are discovered during symbol collection.
    pkg_name = pkg_root.name

    return pkg_root, pkg_name

# Conservative sentinel values for unknown parameter types.
_SENTINELS: Dict[str, str] = {
    "str": '""',
    "int": "0",
    "float": "0.0",
    "bool": "False",
    "list": "[]",
    "dict": "{}",
    "tuple": "()",
    "set": "set()",
    "bytes": 'b""',
}


def _guess_sentinel(param_name: str, type_annotation: Optional[str]) -> str:
    """Return a plausible default value for a function parameter."""
    if type_annotation:
        ann = type_annotation.strip()
        # Strip generic parameters: list[int] → list, Optional[str] → str
        base = ann.split("[")[0].strip()
        # Handle Optional[X] / Union[X, None] — just use the first type
        if base in ("Optional", "Union"):
            inner = ann[ann.index("[") + 1:ann.rindex("]")]
            base = inner.split(",")[0].strip()
        if base in _SENTINELS:
            return _SENTINELS[base]
    # Heuristic: parameter name hints
    if param_name in ("name", "path", "filename", "url", "text", "message"):
        return '""'
    if param_name.startswith("is_") or param_name.startswith("has_") or param_name.startswith("enable"):
        return "False"
    return "None"


def _collect_symbols(
    source_files: List[Path], pkg_root: Path
) -> Tuple[
    List[Tuple[str, str, List[str], List[str]]],  # classes: (mod, name, init_params, method_names)
    List[Tuple[str, str, List[str]]],              # functions: (mod, name, params)
]:
    """Scan Python files and return (classes, functions).

    Each class entry: (module, name, [init_params_without_self], [public_method_names])
    Each function entry: (module, name, [params_without_self])
    """
    classes: List[Tuple[str, str, List[str], List[str]]] = []
    functions: List[Tuple[str, str, List[str]]] = []

    for py_file in sorted(source_files):
        if py_file.name == SYNTH_FILENAME:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Compute dotted module name relative to pkg_root
        rel = py_file.relative_to(pkg_root)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].replace(".py", "")
        module = ".".join(parts) if parts else ""

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.startswith("_"):
                    continue
                init_params: List[str] = []
                methods: List[str] = []
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        if sub.name == "__init__":
                            for arg in sub.args.args:
                                if arg.arg != "self":
                                    init_params.append(arg.arg)
                        elif not sub.name.startswith("_"):
                            # Public method — collect its params too
                            params = [
                                arg.arg for arg in sub.args.args
                                if arg.arg not in ("self", "cls")
                            ]
                            methods.append(sub.name)
                classes.append((module, node.name, init_params, methods))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                params = [
                    arg.arg for arg in node.args.args
                    if arg.arg not in ("self", "cls")
                ]
                functions.append((module, node.name, params))

    return classes, functions


def _generate_driver(
    classes: List[Tuple[str, str, List[str], List[str]]],
    functions: List[Tuple[str, str, List[str]]],
    project_name: str,
    pkg_name: str,
    pkg_parent: Path,
    project_root: Path,
) -> str:
    """Generate the source code of the synthesised entry file.

    Strategy for classes: create an empty subclass that bypasses
    ``__init__``, instantiate it, and call every public method.  PyFlow
    does not execute code, so the subclass trick gives ``self`` a
    concrete type without needing correct constructor arguments.

    Strategy for functions: call with ``None`` for every parameter.
    Even if this would crash at runtime, PyFlow still analyses the
    function body and discovers its outgoing call edges.
    """
    try:
        syspath_dir = pkg_parent.relative_to(project_root)
    except ValueError:
        syspath_dir = Path(os.path.relpath(pkg_parent, project_root))

    lines: List[str] = []
    lines.append('"""Auto-generated benchmark entry for project: {}"""'.format(project_name))
    lines.append("")
    lines.append("import sys")
    lines.append("from pathlib import Path")
    lines.append("")
    lines.append("# Make the package importable.")
    lines.append(f"_pkg_parent = Path(__file__).resolve().parent / {str(syspath_dir)!r}")
    lines.append("if str(_pkg_parent) not in sys.path:")
    lines.append("    sys.path.insert(0, str(_pkg_parent))")
    lines.append("")
    lines.append("# fmt: off")
    lines.append("# Generated by synth_bench_entry.py.")
    lines.append("")

    # ── imports ──
    imports: Set[str] = set()
    for module, name, _p1, _p2 in classes:
        full = f"{pkg_name}.{module}" if module else pkg_name
        imports.add(f"from {full} import {name}")
    for module, name, _params in functions:
        full = f"{pkg_name}.{module}" if module else pkg_name
        imports.add(f"from {full} import {name}")
    for imp in sorted(imports):
        lines.append(imp)
    lines.append("")

    # ── exercise functions ──
    if functions:
        lines.append("# ── functions ──")
        for module, name, params in functions:
            args = ", ".join(_guess_sentinel(p, None) for p in params)
            lines.append(f"{name}({args})")
        lines.append("")

    # ── exercise classes via empty subclass ──
    if classes:
        lines.append("# ── classes (empty subclass bypasses __init__ args) ──")
        for module, name, _init_params, methods in classes:
            synth_cls = f"_Synth{name}"
            inst = f"_synth_{name}"
            lines.append("")
            lines.append(f"class {synth_cls}({name}):")
            lines.append(f"    def __init__(self):")
            lines.append(f"        pass")
            lines.append(f"{inst} = {synth_cls}()")
            for meth in methods:
                # Call with None for each parameter — PyFlow still
                # analyses the body and finds outgoing edges.
                lines.append(f"{inst}.{meth}()")

    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════
# File-system operations
# ═══════════════════════════════════════════════════════════════════════

def _collect_py_files(project_root: Path) -> List[Path]:
    """Return all .py files in a project, excluding common noise dirs."""
    ignore = {"tests", "test", "__pycache__", ".git", ".github", "docs",
              "examples", "venv", ".venv", "node_modules", "build", "dist"}
    files: List[Path] = []
    for py_file in sorted(project_root.rglob("*.py")):
        parts = set(py_file.relative_to(project_root).parts)
        if parts & ignore:
            continue
        files.append(py_file)
    return files


def _backup_manifest(corpus: Path) -> Path:
    """Copy manifest.json → manifest.json.bak.  Returns the backup path."""
    manifest = corpus / "manifest.json"
    bak = corpus / "manifest.json.bak"
    if manifest.exists() and not bak.exists():
        shutil.copy2(manifest, bak)
    return bak


def _restore_manifest(corpus: Path) -> None:
    """Restore manifest.json from backup, remove the backup."""
    manifest = corpus / "manifest.json"
    bak = corpus / "manifest.json.bak"
    if bak.exists():
        shutil.copy2(bak, manifest)
        bak.unlink()


def _update_manifest(corpus: Path, synth_projects: Set[str]) -> None:
    """Set _entry_file in manifest for synthesised projects."""
    manifest = corpus / "manifest.json"
    if not manifest.exists():
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in data.get("projects", []):
        name = entry["name"]
        if name in synth_projects:
            entry["_entry_file"] = SYNTH_FILENAME
    manifest.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _cleanup_synth_files(corpus: Path) -> int:
    """Remove all synthesised entry files.  Return count removed."""
    count = 0
    for synth_file in corpus.rglob(SYNTH_FILENAME):
        synth_file.unlink()
        count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthesise benchmark entry files for DDA coverage.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("evaluation/repo_level"),
        help="Path to corpus root (default: evaluation/repo_level)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Remove all synthesised files and restore original manifest.",
    )
    args = parser.parse_args()

    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"[ERROR] Corpus not found: {corpus}", file=sys.stderr)
        return 1

    manifest_path = corpus / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] No manifest.json in {corpus}", file=sys.stderr)
        return 1

    if args.clean:
        removed = _cleanup_synth_files(corpus)
        _restore_manifest(corpus)
        print(f"Cleaned up {removed} synth file(s), restored manifest.json")
        return 0

    # ── synthesis ──
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    synth_projects: Set[str] = set()
    total_classes = 0
    total_funcs = 0

    for entry in manifest.get("projects", []):
        name = entry["name"]
        proj_dir = corpus / entry["path"]
        if not proj_dir.is_dir():
            continue

        entry_file_rel = entry.get("_entry_file")
        pkg_root, pkg_name = _resolve_pkg_info(proj_dir, entry_file_rel)

        if pkg_root is None:
            # Single-file project — synth won't help (all code already in
            # the entry file).  Skip.
            print(f"  {name}: single-file project, skipping")
            continue

        py_files = _collect_py_files(pkg_root)
        classes, functions = _collect_symbols(py_files, pkg_root)

        if not classes and not functions:
            print(f"  {name}: no public symbols found, skipping")
            continue

        total_methods = sum(len(methods) for _, _, _, methods in classes)

        driver = _generate_driver(
            classes, functions, name, pkg_name,
            pkg_parent=pkg_root.parent,
            project_root=proj_dir,
        )
        out_path = proj_dir / SYNTH_FILENAME
        out_path.write_text(driver, encoding="utf-8")

        synth_projects.add(name)
        total_classes += len(classes)
        total_funcs += len(functions)
        print(f"  {name}: {len(classes)} cls ({total_methods} methods), {len(functions)} func → {SYNTH_FILENAME}")

    # ── update manifest ──
    _backup_manifest(corpus)
    _update_manifest(corpus, synth_projects)

    print(f"\nSynthesised entry files for {len(synth_projects)} project(s)")
    print(f"Total: {total_classes} classes, {total_funcs} functions")
    print(f"Manifest backed up to manifest.json.bak")
    print(f"\nRun the benchmark with:")
    print(f"  python evaluation/bench_repo_callgraph.py --corpus {args.corpus}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
