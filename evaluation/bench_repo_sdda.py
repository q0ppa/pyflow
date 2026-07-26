#!/usr/bin/env python3
"""Strict-DDA benchmark wrapper.

Runs the constraint engine, classifies every missing edge via the
diagnosis tool, and then removes all ``[reach]``-tier edges from the
ground truth before scoring.  Only ``[fix]`` and ``[name]`` edges remain
— these are the *real* analysis gaps.

Usage:
    python evaluation/bench_repo_sdda.py --corpus evaluation/repo_level
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Reuse the benchʼs project discovery, normalisation, and scoring.
from bench_repo_callgraph import (
    Project,
    EngineResult,
    _adjacency_to_edges,
    _aggregate,
    _compute_analysed_callers,
    _discover_projects,
    _normalize_gt,
    _normalize_graph_for_project,
    _print_summary,
    _score,
)
from pyflow.analysis.callgraph.constraint_based import extract_call_graph_constraint
from pyflow.analysis.callgraph.constraint_based.engine import ConstraintCallGraphBuilder
from pyflow.analysis.callgraph.constraint_based.model import AnalysisOptions


# ═══════════════════════════════════════════════════════════════════════
# Per-project SDDA scoring
# ═══════════════════════════════════════════════════════════════════════

REACH_TIER: Set[str] = {
    "UNREACHABLE_CALLER",
    "RECEIVER_LOST",
    "EMPTY_SELF_DESPITE_REACHABLE",
    "EXTERNAL_NOT_AVAILABLE",
    "MODULE_NOT_LOADED",
}


def _classify_reach_edges(
    project: Project,
    normalized_graph: Dict[str, List[str]],
) -> Set[Tuple[str, str]]:
    """Run diagnosis and return the set of edges classified as [reach]."""
    from diagnose_missing_edges import MissingEdgeDiagnoser

    source = project.entry_file.read_text(encoding="utf-8")
    options = AnalysisOptions(
        warn_on_fixpoint_truncation=False,
        skip_stdlib_modules=True,
        emit_solver_stats=False,
    )
    builder = ConstraintCallGraphBuilder(
        source,
        entry_path=str(project.entry_file),
        options=options,
    )
    cg = builder.build()

    # Load raw GT (diagnosis applies _normalize_gt internally now)
    gt_raw = {
        k: v
        for k, v in json.loads(
            (project.root / "callgraph.json").read_text(encoding="utf-8")
        ).items()
        if not k.startswith("_")
    }

    diagnoser = MissingEdgeDiagnoser(
        builder, gt_raw, cg, project.root,
        project_name=project.name,
        entry_file=project.entry_file,
        whole_program=True,  # WPA: diagnose ALL GT edges
    )
    diagnoses = diagnoser.diagnose_all()

    reach_edges: Set[Tuple[str, str]] = set()
    for d in diagnoses:
        if d.category in REACH_TIER:
            reach_edges.add((d.caller, d.callee))
    return reach_edges


def _score_sdda(project: Project) -> EngineResult:
    """Run constraint engine, classify, filter GT, and score."""
    source = project.entry_file.read_text(encoding="utf-8")

    start = time.perf_counter()
    graph = extract_call_graph_constraint(
        source,
        source_path=str(project.entry_file),
        allow_fixture_graph_loading=False,
    )
    runtime_ms = (time.perf_counter() - start) * 1000.0

    raw_graph = graph.get()
    normalized_graph = _normalize_graph_for_project(
        raw_graph, project.name,
        entry_file=project.entry_file,
    )
    predicted_edges = _adjacency_to_edges(normalized_graph)

    # ── DDA caller-side filter ──
    gt_graph = _normalize_gt(project.ground_truth)
    analysed = _compute_analysed_callers(normalized_graph)
    gt_graph = {
        c: callees
        for c, callees in gt_graph.items()
        if c in analysed
    }

    # ── SDDA: also remove [reach] edges ──
    reach_edges = _classify_reach_edges(project, normalized_graph)
    gt_graph = {
        c: [t for t in callees if (c, t) not in reach_edges]
        for c, callees in gt_graph.items()
    }
    gt_graph = {c: callees for c, callees in gt_graph.items() if callees}

    # ── coverage of remaining GT ──
    all_gt_callers = set(_normalize_gt(project.ground_truth).keys())
    remaining_callers = len(gt_graph)
    coverage = remaining_callers / len(all_gt_callers) if all_gt_callers else 1.0

    precision, recall, tp, fp, fn = _score(
        predicted_edges, _adjacency_to_edges(gt_graph),
    )

    return EngineResult(
        engine="constraint-sdda",
        project=project.name,
        runtime_ms=runtime_ms,
        precision=precision,
        recall=recall,
        coverage=coverage,
        tp=tp,
        fp=fp,
        fn=fn,
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strict-DDA benchmark: remove all [reach]-tier edges from GT.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("evaluation/repo_level"),
        help="Path to corpus root (default: evaluation/repo_level)",
    )
    parser.add_argument(
        "--project",
        action="append",
        help="Limit to specific project names.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write results to JSON.",
    )
    args = parser.parse_args()

    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"[ERROR] Corpus not found: {corpus}", file=sys.stderr)
        return 1

    projects = _discover_projects(corpus)
    if args.project:
        selected = set(args.project)
        projects = [p for p in projects if p.name in selected]

    if not projects:
        print("No projects found.")
        return 1

    results: List[EngineResult] = []
    for project in projects:
        print(f"Processing {project.name} [SDDA] ...", file=sys.stderr)
        try:
            result = _score_sdda(project)
            results.append(result)
        except Exception as exc:
            results.append(EngineResult(
                engine="constraint-sdda",
                project=project.name,
                runtime_ms=float("nan"),
                precision=0.0,
                recall=0.0,
                tp=0,
                fp=0,
                fn=0,
                error=str(exc),
            ))

    _print_summary(results)

    if args.output_json:
        serializable = []
        for r in results:
            d = r.__dict__.copy()
            if d.get("runtime_ms", 0.0) != d.get("runtime_ms", 0.0):
                d["runtime_ms"] = None
            if d.get("coverage", 0.0) != d.get("coverage", 0.0):
                d["coverage"] = None
            serializable.append(d)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(serializable, indent=2), encoding="utf-8",
        )
        print(f"Wrote results to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
