#!/usr/bin/env python3
"""
Run JURIS-AGI on ARC tasks.

Usage:
    python -m juris_agi.eval.run_arc <task_or_dir> [--output <dir>]

Examples:
    python -m juris_agi.eval.run_arc data/arc_public/training/task.json
    python -m juris_agi.eval.run_arc data/arc_public/training/ --output results/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.types import ARCTask, Grid
from ..core.trace import TraceWriter
from ..controller.router import MetaController, ControllerConfig, determine_regime
from ..controller.refusal import RefusalChecker
from ..controller.scheduler import PhaseScheduler, SolvePhase


def load_task(filepath: Path) -> ARCTask:
    """Load an ARC task from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    task_id = filepath.stem
    return ARCTask.from_dict(task_id, data)


def load_tasks_from_dir(dirpath: Path) -> List[ARCTask]:
    """Load all ARC tasks from a directory."""
    tasks = []
    for filepath in sorted(dirpath.glob("*.json")):
        try:
            task = load_task(filepath)
            tasks.append(task)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    return tasks


def grid_to_list(grid: Grid) -> List[List[int]]:
    """Convert Grid to nested list for JSON serialization."""
    return grid.data.tolist()


def run_single_task(
    task: ARCTask,
    controller: MetaController,
    verbose: bool = True,
    show_uncertainty: bool = True,
) -> Dict[str, Any]:
    """Run solver on a single task."""
    if verbose:
        print(f"Solving task: {task.task_id}")
        print(f"  Training pairs: {len(task.train)}")
        print(f"  Test pairs: {len(task.test)}")

    # Determine regime
    regime_decision = determine_regime(task)
    if verbose and show_uncertainty:
        print(f"  Regime: {regime_decision.regime.name} (confidence: {regime_decision.confidence:.2f})")

    # Check for refusal
    checker = RefusalChecker()
    refusal = checker.check(task)

    if refusal.should_refuse:
        if verbose:
            print(f"  REFUSED: {refusal.explanation}")
        return {
            "task_id": task.task_id,
            "success": False,
            "refused": True,
            "refusal_reason": refusal.reason.name if refusal.reason else "unknown",
            "refusal_explanation": refusal.explanation,
            "regime": regime_decision.regime.name,
        }

    # Run solver
    result = controller.solve(task)

    if verbose:
        status = "SUCCESS" if result.success else "FAILED"
        print(f"  Result: {status}")
        if result.success:
            print(f"  Program: {result.audit_trace.program_source}")
            print(f"  Robustness: {result.audit_trace.robustness_score:.2f}")
        else:
            print(f"  Error: {result.error_message}")

    # Build result dict
    result_dict = {
        "task_id": task.task_id,
        "success": result.success,
        "refused": False,
        "predictions": [grid_to_list(p) for p in result.predictions],
        "program": result.audit_trace.program_source,
        "robustness_score": result.audit_trace.robustness_score,
        "constraints_satisfied": result.audit_trace.constraints_satisfied,
        "constraints_violated": result.audit_trace.constraints_violated,
        "synthesis_iterations": result.audit_trace.synthesis_iterations,
        "nodes_explored": result.audit_trace.search_nodes_explored,
        # Regime and uncertainty info
        "regime": regime_decision.regime.name,
        "regime_confidence": regime_decision.confidence,
        "runtime_seconds": result.audit_trace.runtime_seconds,
    }

    if not result.success:
        result_dict["error"] = result.error_message

    return result_dict


def run_evaluation(
    tasks: List[ARCTask],
    controller: MetaController,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run evaluation on multiple tasks."""
    results = []
    successes = 0
    failures = 0
    refusals = 0

    for i, task in enumerate(tasks):
        if verbose:
            print(f"\n[{i+1}/{len(tasks)}] ", end="")

        result = run_single_task(task, controller, verbose)
        results.append(result)

        if result.get("refused"):
            refusals += 1
        elif result["success"]:
            successes += 1
        else:
            failures += 1

    # Compute summary statistics
    total = len(tasks)
    attempted = total - refusals

    summary = {
        "total_tasks": total,
        "attempted": attempted,
        "successes": successes,
        "failures": failures,
        "refusals": refusals,
        "success_rate": successes / attempted if attempted > 0 else 0.0,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    # Compute regime statistics
    regime_counts: Dict[str, int] = {}
    for r in results:
        regime = r.get("regime", "unknown")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    # Compute average runtime
    runtimes = [r.get("runtime_seconds", 0) for r in results if not r.get("refused")]
    avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0.0

    summary["regime_distribution"] = regime_counts
    summary["average_runtime_seconds"] = avg_runtime

    if verbose:
        print(f"\n{'='*50}")
        print(f"SUMMARY")
        print(f"{'='*50}")
        print(f"Total tasks: {total}")
        print(f"Attempted: {attempted}")
        print(f"Successes: {successes}")
        print(f"Failures: {failures}")
        print(f"Refusals: {refusals}")
        print(f"Success rate: {summary['success_rate']:.1%}")

        # Print regime and uncertainty stats
        print(f"\n{'='*50}")
        print(f"REGIME & UNCERTAINTY STATS")
        print(f"{'='*50}")
        print(f"Average runtime: {avg_runtime:.2f}s")
        print(f"Regime distribution:")
        for regime, count in sorted(regime_counts.items()):
            pct = count / total * 100 if total > 0 else 0
            print(f"  {regime}: {count} ({pct:.1f}%)")

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {output_file}")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run JURIS-AGI solver on ARC tasks"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to task JSON file or directory of tasks",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum synthesis depth (default: 4)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=50,
        help="Beam search width (default: 50)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum synthesis iterations (default: 1000)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--no-refinement",
        action="store_true",
        help="Disable refinement loop",
    )
    parser.add_argument(
        "--no-robustness",
        action="store_true",
        help="Skip robustness checking",
    )

    args = parser.parse_args()

    # Load tasks
    if args.input.is_file():
        tasks = [load_task(args.input)]
    elif args.input.is_dir():
        tasks = load_tasks_from_dir(args.input)
        if not tasks:
            print(f"No tasks found in {args.input}")
            sys.exit(1)
    else:
        print(f"Invalid input path: {args.input}")
        sys.exit(1)

    # Create controller
    config = ControllerConfig(
        max_synthesis_depth=args.max_depth,
        beam_width=args.beam_width,
        max_synthesis_iterations=args.max_iterations,
        enable_refinement=not args.no_refinement,
        compute_robustness=not args.no_robustness,
    )
    controller = MetaController(config)

    # Run evaluation
    verbose = not args.quiet
    summary = run_evaluation(tasks, controller, args.output, verbose)

    # Exit code based on success
    sys.exit(0 if summary["successes"] > 0 else 1)


if __name__ == "__main__":
    main()
