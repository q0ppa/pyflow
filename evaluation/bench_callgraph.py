#!/usr/bin/env python3
"""Benchmark call-graph engines on snippet and repository corpora.

Compares PyFlow's constraint engine and the external PyCG engine against
ground-truth callgraph.json files in each project directory.

Choose ``--suite snippets`` for isolated language features or ``--suite
repositories`` for multi-file, cross-module projects. External baselines may
provide pre-computed repository results via ``--external-result-dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pyflow.analysis.callgraph.constraint_based import extract_call_graph_constraint
from pyflow.analysis.callgraph.pycg_based import PYCG_AVAILABLE, extract_call_graph_pycg

Edge = Tuple[str, str]
Graph = Dict[str, Iterable[str]]


@dataclass(frozen=True)
class TimedGraph:
    graph: Graph
    runtime_ms: float


def _adjacency_to_edges(graph: Graph) -> Set[Edge]:
    return {(caller, callee) for caller, callees in graph.items() for callee in callees}


def _score_edges(
    predicted: Set[Edge], expected: Set[Edge]
) -> Tuple[float, float, int, int, int]:
    """Return precision, recall, and the true/false positive and negative counts."""

    true_positives = len(predicted & expected)
    false_positives = len(predicted - expected)
    false_negatives = len(expected - predicted)
    precision = (
        true_positives / (true_positives + false_positives) if predicted else 1.0
    )
    recall = true_positives / (true_positives + false_negatives) if expected else 1.0
    return precision, recall, true_positives, false_positives, false_negatives


class AnalysisTimeoutError(TimeoutError):
    """Raised when one timed engine invocation exceeds its configured limit."""


def _timeout_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a number of seconds") from exc
    if seconds < 0:
        raise argparse.ArgumentTypeError("timeout must be zero or greater")
    return seconds


def _run_with_timeout(runner: Callable[[], Graph], timeout_seconds: float) -> Graph:
    if timeout_seconds == 0:
        return runner()
    if threading.current_thread() is not threading.main_thread() or not hasattr(
        signal, "setitimer"
    ):
        raise RuntimeError(
            "--timeout requires a POSIX main-thread execution environment"
        )

    def raise_timeout(_signum: int, _frame: object) -> None:
        raise AnalysisTimeoutError(f"analysis exceeded {timeout_seconds:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return runner()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _time_graph_runner(
    runner: Callable[[], Graph], repeats: int, timeout_seconds: float
) -> TimedGraph:
    runtimes = []
    graph: Graph = {}
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        graph = _run_with_timeout(runner, timeout_seconds)
        runtimes.append((time.perf_counter() - start) * 1000.0)
    return TimedGraph(graph=graph, runtime_ms=statistics.mean(runtimes))


@dataclass(frozen=True)
class SnippetCase:
    feature: str
    name: str
    main_file: Path
    expected_edges: Set[Edge]


@dataclass(frozen=True)
class SnippetResult:
    engine: str
    feature: str
    case: str
    runtime_ms: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    error: Optional[str] = None


def _discover_snippet_cases(root: Path) -> List[SnippetCase]:
    """Find independent ``main.py``/``callgraph.json`` feature fixtures."""

    if not root.is_dir():
        return []
    cases = []
    for main_file in sorted(root.rglob("main.py")):
        expected_file = main_file.with_name("callgraph.json")
        if not expected_file.is_file():
            continue
        try:
            payload = json.loads(expected_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            expected_edges = _adjacency_to_edges(
                {
                    caller: callees
                    for caller, callees in payload.items()
                    if isinstance(callees, list)
                }
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        relative_path = main_file.parent.relative_to(root)
        cases.append(
            SnippetCase(
                feature=relative_path.parts[0] if relative_path.parts else "unknown",
                name=str(relative_path),
                main_file=main_file,
                expected_edges=expected_edges,
            )
        )
    return cases


def _run_snippet_case(
    engine_name: str,
    case: SnippetCase,
    runner: Callable[[str, Path], Graph],
    repeats: int,
    timeout_seconds: float,
) -> SnippetResult:
    source = case.main_file.read_text(encoding="utf-8")
    try:
        timed_graph = _time_graph_runner(
            lambda: runner(source, case.main_file), repeats, timeout_seconds
        )
        predicted_edges = _adjacency_to_edges(timed_graph.graph)
        precision, recall, tp, fp, fn = _score_edges(
            predicted_edges, case.expected_edges
        )
        return SnippetResult(
            engine=engine_name,
            feature=case.feature,
            case=case.name,
            runtime_ms=timed_graph.runtime_ms,
            precision=precision,
            recall=recall,
            tp=tp,
            fp=fp,
            fn=fn,
        )
    except Exception as exc:  # pragma: no cover - benchmarking fallback
        return SnippetResult(
            engine=engine_name,
            feature=case.feature,
            case=case.name,
            runtime_ms=float("nan"),
            precision=0.0,
            recall=0.0,
            tp=0,
            fp=0,
            fn=0,
            error=str(exc),
        )


def _run_snippet_benchmark(
    root: Path, repeats: int, timeout_seconds: float, engines: Sequence[str]
) -> List[SnippetResult]:
    cases = _discover_snippet_cases(root)
    if not cases:
        raise ValueError(f"No snippet fixtures found under {root}")

    runners = {"constraint": _constraint_runner, "pycg": _pycg_runner}
    results = [
        _run_snippet_case(engine, case, runners[engine], repeats, timeout_seconds)
        for case in cases
        for engine in engines
    ]
    _print_snippet_summary(results)
    return results


def _print_snippet_summary(results: Sequence[SnippetResult]) -> None:
    print("\nSINGLE-FILE FEATURE FIXTURES")
    print(
        f"{'Engine':<12}{'Cases':>7}{'Precision':>12}{'Recall':>9}{'Runtime(ms)':>13}"
    )
    for engine in sorted({result.engine for result in results}):
        rows = [
            result for result in results if result.engine == engine and not result.error
        ]
        if not rows:
            continue
        precision = statistics.mean(row.precision for row in rows)
        recall = statistics.mean(row.recall for row in rows)
        runtime_ms = statistics.mean(row.runtime_ms for row in rows)
        print(
            f"{engine:<12}{len(rows):>7}{precision:>12.3f}{recall:>9.3f}"
            f"{runtime_ms:>13.2f}"
        )
    failed = [result for result in results if result.error]
    for result in failed:
        print(f"  [{result.engine}] {result.case}: {result.error}")


@dataclass(frozen=True)
class Project:
    name: str
    root: Path
    manifest_entry: Dict[str, object]
    ground_truth: Dict[str, List[str]]
    entry_file: Path


@dataclass(frozen=True)
class EngineResult:
    engine: str
    project: str
    runtime_ms: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    error: Optional[str] = None


def _load_callgraph_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _write_callgraph_json(graph: Dict[str, Iterable[str]], path: Path) -> None:
    serialized = {caller: sorted(callees) for caller, callees in graph.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


def _discover_projects(corpus_root: Path) -> List[Project]:
    manifest_path = corpus_root / "manifest.json"
    if manifest_path.exists():
        return _discover_from_manifest(corpus_root, manifest_path)
    return _discover_by_scan(corpus_root)


def _discover_from_manifest(corpus_root: Path, manifest_path: Path) -> List[Project]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    projects: List[Project] = []
    for entry in manifest.get("projects", []):
        name = entry["name"]
        root = corpus_root / entry["path"]
        if not root.is_dir():
            continue
        project = _make_project(name, root, entry)
        if project:
            projects.append(project)
    return projects


def _discover_by_scan(corpus_root: Path) -> List[Project]:
    projects: List[Project] = []
    for child in sorted(corpus_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name == "__pycache__":
            continue
        project = _make_project(
            child.name,
            child,
            {"name": child.name, "path": str(child.relative_to(corpus_root))},
        )
        if project:
            projects.append(project)
    return projects


def _make_project(
    name: str,
    root: Path,
    manifest_entry: Dict[str, object],
) -> Optional[Project]:
    gt_path = root / "callgraph.json"
    if not gt_path.exists():
        return None
    try:
        ground_truth = _load_callgraph_json(gt_path)
    except Exception:
        return None
    entry_file = _resolve_entry_file(root, gt_path, manifest_entry)
    if not entry_file:
        return None
    return Project(
        name=name,
        root=root,
        manifest_entry=manifest_entry,
        ground_truth=ground_truth,
        entry_file=entry_file,
    )


def _resolve_entry_file(
    root: Path,
    gt_path: Path,
    manifest_entry: Optional[Dict[str, object]] = None,
) -> Optional[Path]:
    try:
        raw = json.loads(gt_path.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    entry_name = raw.get("_entry_point", "main.py")
    candidate = root / str(entry_name)
    if candidate.is_file():
        return candidate
    if manifest_entry:
        mf_entry = manifest_entry.get("_entry_file")
        if mf_entry:
            candidate = root / str(mf_entry)
            if candidate.is_file():
                return candidate
    for fallback in ("main.py", "__init__.py", "source.py", f"{root.name}.py"):
        candidate = root / fallback
        if candidate.is_file():
            return candidate
    return None

# ── known stdlib / third-party prefixes ──────────────────────────────
# These modules are *not* part of any project under analysis and
# should never receive the project-name prefix.  Without this list
# normalization would produce noise like ``sshtunnel.socket.socket``.
_STDLIB_PREFIXES = (
    # stdlib
    "abc.", "argparse.", "asyncio.", "base64.", "binascii.",
    "builtins.", "collections.", "contextlib.", "copy.", "csv.",
    "dataclasses.", "datetime.", "decimal.", "enum.", "fractions.",
    "functools.", "gc.", "getopt.", "getpass.", "glob.", "hashlib.",
    "hmac.", "html.", "http.", "importlib.", "inspect.", "io.",
    "itertools.", "json.", "linecache.", "logging.", "math.",
    "mmap.", "operator.", "os.", "pathlib.", "pickle.", "pkgutil.",
    "platform.", "pprint.", "queue.", "random.", "re.", "reprlib.",
    "shutil.", "signal.", "socket.", "statistics.", "string.",
    "struct.", "subprocess.", "sys.", "sysconfig.", "tempfile.",
    "textwrap.", "threading.", "time.", "traceback.", "types.",
    "typing.", "unittest.", "urllib.", "warnings.", "weakref.",
    "xml.", "zipfile.",
    # common third-party (present in our benchmark corpus)
    "click.", "numpy.", "pandas.", "paramiko.", "pygments.",
    "requests.", "rich.",
)


def normalize_callgraph_name(
    name: str, project_name: str, *, entry_file: Optional[Path] = None,
) -> str:
    """Normalize a single caller or callee name for a project.

    PyFlow uses ``main`` as the module name for the entry file.  The GT
    uses the project-qualified path (e.g. ``sshtunnel.main.X`` for a
    standalone entry, or ``sqlparse.X`` when the entry is ``__init__.py``).
    This function bridges that gap.

    The only changes made here are *infrastructure alignment* fixes — they
    do not alter the set of discovered edges, only how they are named.
    """
    # <builtin> / <dynamic> / <str> etc. — left as-is
    if name.startswith("<"):
        return name

    # External stdlib / third-party — do NOT prepend project name
    if any(name.startswith(p) for p in _STDLIB_PREFIXES):
        return name

    python_pkg = project_name.replace("-", "_")
    is_init_entry = entry_file is not None and entry_file.name == "__init__.py"

    # Already has the project-name or package-name prefix
    if name.startswith(project_name + ".") or name.startswith(python_pkg + "."):
        return name

    # Entry-file definitions: PyFlow names them ``main`` or ``main.X``.
    # - For standalone .py files the GT expects ``project.main.X``
    # - For __init__.py the GT expects ``project.X`` (no extra segment)
    if name == "main" or name.startswith("main."):
        if is_init_entry:
            if name == "main":
                return project_name
            return f"{project_name}.{name[len('main.'):]}"
        else:
            return f"{project_name}.{name}"

    # Imported modules: PyFlow may use the bare module name (e.g.
    # ``markdown`` instead of ``rich_cli.markdown``).  If the project
    # uses a hyphenated name we must use the real Python package name
    # (underscored) for cross-module imports, because that's what the
    # GT uses.
    if project_name != python_pkg:
        return f"{python_pkg}.{name}"

    return f"{project_name}.{name}"


def _normalize_graph_for_project(
    graph: Dict[str, Iterable[str]], project_name: str, *,
    strip_builtins: bool = True,
    entry_file: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Normalize engine output names so they match the ground-truth convention.

    See ``normalize_callgraph_name`` for the per-name logic.
    """
    normalized: Dict[str, List[str]] = {}
    for caller, callees in graph.items():
        if strip_builtins and (caller.startswith("<builtin>") or caller.startswith("<")):
            continue

        normalized_caller = normalize_callgraph_name(
            caller, project_name, entry_file=entry_file,
        )
        normalized_callees = []
        for callee in callees:
            if strip_builtins and (callee.startswith("<builtin>") or callee.startswith("<")):
                continue
            normalized_callees.append(normalize_callgraph_name(
                callee, project_name, entry_file=entry_file,
            ))

        normalized[normalized_caller] = normalized_callees

    return normalized

def _normalize_gt(graph: Dict[str, Iterable[str]]) -> Dict[str, List[str]]:
    """Strip <builtin> / <dynamic> / <str> / <list> / ... callees from GT.

    Matches the same filtering that ``_normalize_graph_for_project`` applies to
    engine output, so that GT entries for edges the engine cannot represent are
    not counted as false negatives.
    """
    cleaned: Dict[str, List[str]] = {}
    for caller, callees in graph.items():
        kept = [c for c in callees if not (c.startswith("<") or c.startswith("<builtin>"))]
        cleaned[caller] = kept
    return cleaned


def _score(predicted: Set[Edge], expected: Set[Edge]) -> Tuple[float, float, int, int, int]:
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall, tp, fp, fn

def _dump_missing_edges(predicted: Set[Edge], project: Project, engine_name: str, dump_dir: Path, normalize_gt: bool = True) -> None:
    # print(f"Dump {project.name} for {engine_name}")
    gt_graph = _normalize_gt(project.ground_truth) if normalize_gt else project.ground_truth
    gt_edges = _adjacency_to_edges(gt_graph)
    missing = gt_edges - predicted
    if not missing:
        return
    out_dir = dump_dir / project.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{engine_name}_missing.json"
    missing_dict: Dict[str, List[str]] = {}
    for caller, callee in sorted(missing):
        missing_dict.setdefault(caller, []).append(callee)
    _write_callgraph_json(missing_dict, out_file)

def _run_constraint_engine(project: Project, repeats: int, dump_missing: Optional[Path] = None, normalize_gt: bool = True) -> EngineResult:
    source = project.entry_file.read_text(encoding="utf-8")
    try:
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            graph = extract_call_graph_constraint(
                source,
                source_path=str(project.entry_file),
                allow_fixture_graph_loading=False,
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            runtimes.append(elapsed)
            normalized_graph = _normalize_graph_for_project(
                graph.get(), project.name,
                strip_builtins=normalize_gt,
                entry_file=project.entry_file,
            )
            predicted_edges = _adjacency_to_edges(normalized_graph)

        gt_graph = _normalize_gt(project.ground_truth) if normalize_gt else project.ground_truth
        precision, recall, tp, fp, fn = _score(
            predicted_edges, _adjacency_to_edges(gt_graph)
        )
        if dump_missing:
            _dump_missing_edges(predicted_edges, project, "constraint", dump_missing, normalize_gt=normalize_gt)
        return EngineResult(
            engine=engine_name,
            project=project.name,
            runtime_ms=timed_graph.runtime_ms,
            precision=precision,
            recall=recall,
            tp=tp,
            fp=fp,
            fn=fn,
        )
    except Exception as exc:
        return EngineResult(
            engine=engine_name,
            project=project.name,
            runtime_ms=float("nan"),
            precision=0.0,
            recall=0.0,
            tp=0,
            fp=0,
            fn=0,
            error=str(exc),
        )

def _run_pycg_engine(project: Project, repeats: int, dump_missing: Optional[Path] = None, normalize_gt: bool = True) -> EngineResult:
    source = project.entry_file.read_text(encoding="utf-8")
    runtimes: List[float] = []
    predicted_edges: Set[Edge] = set()

    try:
        # Some corpora (e.g. bpytop.py) have module-level argparse that
        # consumes sys.argv.  Save/restore to prevent PyCG's import hooks
        # from seeing our extra flags.
        saved_argv = sys.argv
        try:
            sys.argv = [str(project.entry_file)]
            for _ in range(max(1, repeats)):
                sys.argv = [str(project.entry_file)]
                start = time.perf_counter()
                graph = extract_call_graph_pycg(
                    source,
                    source_path=str(project.entry_file),
                    use_fixture_fallback=False,
                )
                elapsed = (time.perf_counter() - start) * 1000.0
                runtimes.append(elapsed)
                normalized_graph = _normalize_graph_for_project(
                    graph.get(), project.name,
                    strip_builtins=normalize_gt,
                    entry_file=project.entry_file,
                )
                predicted_edges = _adjacency_to_edges(normalized_graph)

            gt_graph = _normalize_gt(project.ground_truth) if normalize_gt else project.ground_truth
            precision, recall, tp, fp, fn = _score(
                predicted_edges, _adjacency_to_edges(gt_graph)
            )
            if dump_missing:
                _dump_missing_edges(predicted_edges, project, "pycg", dump_missing, normalize_gt=normalize_gt)
            return EngineResult(
                engine="pycg",
                project=project.name,
                runtime_ms=statistics.mean(runtimes),
                precision=precision,
                recall=recall,
                tp=tp,
                fp=fp,
                fn=fn,
            )
        finally:
            sys.argv = saved_argv
    except Exception as exc:
        return EngineResult(
            engine="pycg",
            project=project.name,
            runtime_ms=float("nan"),
            precision=0.0,
            recall=0.0,
            tp=0,
            fp=0,
            fn=0,
            error=str(exc),
        )

def _load_external_results(result_dir: Path, projects: List[Project], normalize_gt: bool = True) -> List[EngineResult]:
    results: List[EngineResult] = []
    if not result_dir.is_dir():
        return results

    for json_file in sorted(result_dir.glob("*.json")):
        engine_name = json_file.stem
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            results.append(
                EngineResult(
                    engine=engine_name,
                    project="*",
                    runtime_ms=float("nan"),
                    precision=0.0,
                    recall=0.0,
                    tp=0,
                    fp=0,
                    fn=0,
                    error=f"Failed to parse {json_file}: {exc}",
                )
            )
            continue

        if not isinstance(payload, dict):
            continue

        for project in projects:
            proj_graph = payload.get(project.name)
            if proj_graph is None:
                continue
            if not isinstance(proj_graph, dict):
                continue
            try:
                gt_graph = _normalize_gt(project.ground_truth) if normalize_gt else project.ground_truth
                predicted_edges = _adjacency_to_edges(proj_graph)
                precision, recall, tp, fp, fn = _score(
                    predicted_edges, _adjacency_to_edges(gt_graph)
                )
                results.append(
                    EngineResult(
                        engine=engine_name,
                        project=project.name,
                        runtime_ms=float("nan"),
                        precision=precision,
                        recall=recall,
                        tp=tp,
                        fp=fp,
                        fn=fn,
                    )
                )
            except Exception as exc:
                results.append(
                    EngineResult(
                        engine=engine_name,
                        project=project.name,
                        runtime_ms=float("nan"),
                        precision=0.0,
                        recall=0.0,
                        tp=0,
                        fp=0,
                        fn=0,
                        error=str(exc),
                    )
                )

    return results


def _aggregate(
    results: Sequence[EngineResult],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    buckets: Dict[Tuple[str, str], List[EngineResult]] = {}
    for row in results:
        if row.error:
            continue
        buckets.setdefault((row.engine, row.project), []).append(row)

    summary: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, rows in buckets.items():
        summary[key] = {
            "count": float(len(rows)),
            "precision": statistics.mean(item.precision for item in rows),
            "recall": statistics.mean(item.recall for item in rows),
            "runtime_ms": (
                statistics.mean(item.runtime_ms for item in rows)
                if all(not math.isnan(item.runtime_ms) for item in rows)
                else float("nan")
            ),
        }
    return summary

def _print_summary(
    results: Sequence[EngineResult],
    exclude_zero_recall: bool = False,
) -> None:
    if exclude_zero_recall:
        results = [r for r in results if not r.error and r.recall > 0.0]
    summary = _aggregate(results)
    if not summary:
        print("No results to display.")
        return

    engine_w = max(len(key[0]) for key in summary) + 1
    proj_w = max(len(key[1]) for key in summary) + 1
    print("\n" + "=" * 80)
    print("REPO-LEVEL CALL GRAPH BENCHMARK".center(80))
    print("=" * 80)

    header = (
        f"{'Engine':<{engine_w}}{'Project':<{proj_w}}"
        f"{'Precision':>10}{'Recall':>8}{'Runtime(ms)':>13}"
    )
    print(header)
    print("-" * 80)

    for (engine, project), values in sorted(summary.items()):
        rt_str = (
            f"{values['runtime_ms']:.2f}"
            if values["runtime_ms"] == values["runtime_ms"]
            else "N/A"
        )
        print(
            f"{engine:<{engine_w}}{project:<{proj_w}}"
            f"{values['precision']:>10.3f}{values['recall']:>8.3f}"
            f"{rt_str:>13}"
        )

    print("\n" + "=" * 80)
    print("PER-ENGINE AVERAGES".center(80))
    print("=" * 80)

    engine_aggregates: Dict[str, List[Dict[str, float]]] = {}
    for (engine, _project), values in summary.items():
        engine_aggregates.setdefault(engine, []).append(values)

    eng_w = max(len(e) for e in engine_aggregates) + 1
    avg_header = (
        f"{'Engine':<{eng_w}}{'Projects':>10}{'Avg Prec':>10}"
        f"{'Avg Rec':>9}{'Avg RT(ms)':>12}"
    )
    print(avg_header)
    print("-" * 80)

    for engine in sorted(engine_aggregates):
        items = engine_aggregates[engine]
        n = float(len(items))
        avg_p = statistics.mean(v["precision"] for v in items)
        avg_r = statistics.mean(v["recall"] for v in items)
        rts = [v["runtime_ms"] for v in items if v["runtime_ms"] == v["runtime_ms"]]
        avg_rt = statistics.mean(rts) if rts else float("nan")
        rt_str = f"{avg_rt:.2f}" if avg_rt == avg_rt else "N/A"
        print(f"{engine:<{eng_w}}{int(n):>10}{avg_p:>10.3f}{avg_r:>9.3f}{rt_str:>12}")

    _print_deltas(summary)
    _print_errors(results)
    _print_mismatch_hint(results)
    print()


def _print_deltas(summary: Dict[Tuple[str, str], Dict[str, float]]) -> None:
    projects = sorted({proj for (_eng, proj) in summary})
    engines = sorted({eng for (eng, _proj) in summary})
    if len(engines) < 2:
        return
    baseline = engines[0]

    print("\n" + "=" * 80)
    print(f"DELTAS vs {baseline} (positive = improvement over baseline)".center(80))
    print("=" * 80)

    proj_w = max(len(p) for p in projects) + 1
    delta_header = f"{'Project':<{proj_w}}"
    for eng in engines[1:]:
        delta_header += f"{' ' + eng + ' ΔPrec':>14}{' ΔRec':>8}{' ΔRT':>10}"
    print(delta_header)
    print("-" * 80)

    for project in projects:
        base = summary.get((baseline, project))
        if not base:
            continue
        line = f"{project:<{proj_w}}"
        for eng in engines[1:]:
            cur = summary.get((eng, project))
            if not cur:
                line += f"{'N/A':>14}{'N/A':>8}{'N/A':>10}"
                continue
            dp = cur["precision"] - base["precision"]
            dr = cur["recall"] - base["recall"]
            has_rt = (
                cur["runtime_ms"] == cur["runtime_ms"]
                and base["runtime_ms"] == base["runtime_ms"]
            )
            drt = (cur["runtime_ms"] - base["runtime_ms"]) if has_rt else float("nan")
            rt_str = f"{drt:+.2f}" if drt == drt else "N/A"
            line += f"{dp:>+14.3f}{dr:>+8.3f}{rt_str:>10}"
        print(line)


def _print_errors(results: Sequence[EngineResult]) -> None:
    failed = [r for r in results if r.error]
    if not failed:
        return
    print(f"\n{len(failed)} engine result(s) recorded errors:")
    for r in failed:
        print(f"  [{r.engine}] {r.project}: {r.error}")


def _print_mismatch_hint(results: Sequence[EngineResult]) -> None:
    zero_recall = [
        r for r in results if not r.error and r.recall == 0.0 and r.precision == 0.0
    ]
    if not zero_recall:
        return
    print(f"\nNOTE: {len(zero_recall)} result(s) have zero precision AND zero recall.")
    print(
        "This usually means naming conventions differ between the engine output "
        "and ground truth."
    )
    print(
        "Re-run with --dump-outputs /tmp/debug/ to compare the raw outputs "
        "side-by-side:"
    )
    print(
        f"  python {__file__} repositories --dump-outputs /tmp/debug/ "
        "--project <name>"
    )
    print("  cat /tmp/debug/<project>/constraint.json")
    print("  cat /tmp/debug/<project>/ground_truth.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare call-graph engines on single-file or repository fixtures."
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of timed repeats (default: 1).",
    )
    common.add_argument(
        "--timeout",
        type=_timeout_seconds,
        default=0,
        metavar="SECONDS",
        help="Per-engine timeout; 0 disables it (default: 0).",
    )
    common.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write per-fixture or per-project engine results to JSON.",
    )
    common.add_argument(
        "--engine",
        action="append",
        choices=["constraint", "pycg"],
        help="Engine to run; repeat to compare engines (default: all available).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    snippets = commands.add_parser(
        "snippets",
        parents=[common],
        help="Benchmark independent single-file feature fixtures.",
    )
    snippets.add_argument(
        "--snippets-root",
        type=Path,
        default=Path("tests/analysis/callgraph/snippets"),
        help="Fixture root (default: tests/analysis/callgraph/snippets).",
    )
    repositories = commands.add_parser(
        "repositories",
        parents=[common],
        help="Benchmark multi-file repository fixtures.",
    )
    repositories.add_argument(
        "--corpus",
        type=Path,
        default=Path("evaluation/callgraph_repositories"),
        help="Corpus root (default: evaluation/callgraph_repositories).",
    )
    repositories.add_argument(
        "--dump-outputs",
        type=Path,
        default=None,
        help="Write each engine's call-graph JSON to a directory for inspection",
    )
    repositories.add_argument(
        "--dump-missing",
        type=Path,
        default=None,
        help=(
            "Dump edges present in ground truth but missing from engine output "
            "to a directory."
        ),
    )
    repositories.add_argument(
        "--external-result-dir",
        type=Path,
        default=None,
        help="Load pre-computed results from external baselines (JSON format)",
    )
    repositories.add_argument(
        "--project",
        action="append",
        help="Limit to specific project names (repeat for multiple).",
    )
    parser.add_argument(
        "--exclude-zero-recall",
        action="store_true",
        default=False,
        help="Exclude engine-project pairs where recall==0 from display and aggregates.",
    )
    parser.add_argument(
        "--no-normalize-gt",
        action="store_true",
        default=False,
        help="Keep <builtin> / <str> / <list> callee edges in ground-truth (by default they are stripped to match engine output normalization).",
    )
    args = parser.parse_args()

    engines = args.engine or (
        ["constraint", "pycg"] if PYCG_AVAILABLE else ["constraint"]
    )

    if args.command == "snippets":
        try:
            snippet_results = _run_snippet_benchmark(
                args.snippets_root.resolve(), args.repeat, args.timeout, engines
            )
        except ValueError as exc:
            print(exc)
            return 1
        _write_results(args.output_json, snippet_results)
        return 2 if any(result.error for result in snippet_results) else 0

    corpus_root = args.corpus.resolve()
    if not corpus_root.is_dir():
        print(f"Corpus directory not found: {corpus_root}")
        return 1

    projects = _discover_projects(corpus_root)
    if not projects:
        print("No projects with callgraph.json ground truth found.")
        return 1

    if args.project:
        selected = set(args.project)
        projects = [p for p in projects if p.name in selected]
        if not projects:
            print(f"No projects matching: {args.project}")
            return 1

    engines = args.engine or ["constraint", "pycg"] if PYCG_AVAILABLE else ["constraint"]

    normalize_gt = not args.no_normalize_gt

    results: List[EngineResult] = []
    for project in projects:
        print(f"Processing {project.name} (entry: {project.entry_file.name}) ...")
        if "constraint" in engines:
            results.append(
                _run_constraint_engine(project, repeats=args.repeat, dump_missing=args.dump_missing, normalize_gt=normalize_gt)
            )
        if "pycg" in engines and PYCG_AVAILABLE:
            results.append(
                _run_pycg_engine(project, repeats=args.repeat, dump_missing=args.dump_missing, normalize_gt=normalize_gt)
            )

        if args.dump_outputs:
            _dump_project_graphs(project, args.dump_outputs, normalize_gt=normalize_gt)

    if args.external_result_dir:
        ext_results = _load_external_results(
            args.external_result_dir.resolve(), projects, normalize_gt=normalize_gt
        )
        results.extend(ext_results)

    _print_summary(results, exclude_zero_recall=args.exclude_zero_recall)

    _write_results(args.output_json, results)

    failed = [r for r in results if r.error]
    return 2 if failed else 0


def _dump_project_graphs(project: Project, dump_dir: Path, normalize_gt: bool = True) -> None:
    source = project.entry_file.read_text(encoding="utf-8")
    out_dir = dump_dir / project.name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cg = extract_call_graph_constraint(
            source,
            source_path=str(project.entry_file),
            allow_fixture_graph_loading=False,
        )
        normalized_graph = _normalize_graph_for_project(
            cg.get(), project.name,
            strip_builtins=normalize_gt,
            entry_file=project.entry_file,
        )
        _write_callgraph_json(normalized_graph, out_dir / "constraint.json")
    except Exception:
        pass

    if PYCG_AVAILABLE:
        try:
            saved_argv = sys.argv
            sys.argv = [str(project.entry_file)]
            try:
                cg = extract_call_graph_pycg(
                    source,
                    source_path=str(project.entry_file),
                    use_fixture_fallback=False,
                )
            finally:
                sys.argv = saved_argv
            normalized_graph = _normalize_graph_for_project(
                cg.get(), project.name,
                strip_builtins=normalize_gt,
                entry_file=project.entry_file,
            )
            _write_callgraph_json(normalized_graph, out_dir / "pycg.json")
        except Exception:
            pass

    gt_graph = _normalize_gt(project.ground_truth) if normalize_gt else project.ground_truth
    _write_callgraph_json(gt_graph, out_dir / "ground_truth.json")


if __name__ == "__main__":
    raise SystemExit(main())
