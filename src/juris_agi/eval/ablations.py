#!/usr/bin/env python3
"""
Ablation studies for JURIS-AGI components.

Tests the contribution of different components to overall performance.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from ..core.types import ARCTask
from ..controller.router import MetaController, ControllerConfig


@dataclass
class AblationConfig:
    """Configuration for an ablation study."""
    name: str
    description: str
    controller_config: ControllerConfig


def get_ablation_configs() -> List[AblationConfig]:
    """Get standard ablation configurations."""
    return [
        # Full system
        AblationConfig(
            name="full",
            description="Full JURIS-AGI system",
            controller_config=ControllerConfig(
                enable_wme=True,
                enable_mal=True,
                enable_refinement=True,
                compute_robustness=True,
            ),
        ),
        # No WME
        AblationConfig(
            name="no_wme",
            description="Without World Model Expert",
            controller_config=ControllerConfig(
                enable_wme=False,
                enable_mal=True,
                enable_refinement=True,
                compute_robustness=True,
            ),
        ),
        # No MAL
        AblationConfig(
            name="no_mal",
            description="Without Memory & Abstraction Library",
            controller_config=ControllerConfig(
                enable_wme=True,
                enable_mal=False,
                enable_refinement=True,
                compute_robustness=True,
            ),
        ),
        # No Refinement
        AblationConfig(
            name="no_refinement",
            description="Without refinement loop",
            controller_config=ControllerConfig(
                enable_wme=True,
                enable_mal=True,
                enable_refinement=False,
                compute_robustness=True,
            ),
        ),
        # Synthesis only
        AblationConfig(
            name="synthesis_only",
            description="Pure beam search synthesis",
            controller_config=ControllerConfig(
                enable_wme=False,
                enable_mal=False,
                enable_refinement=False,
                compute_robustness=False,
            ),
        ),
        # Shallow synthesis
        AblationConfig(
            name="shallow_synthesis",
            description="Limited synthesis depth (2)",
            controller_config=ControllerConfig(
                max_synthesis_depth=2,
                enable_wme=True,
                enable_mal=True,
                enable_refinement=True,
            ),
        ),
        # Deep synthesis
        AblationConfig(
            name="deep_synthesis",
            description="Deep synthesis (6 levels)",
            controller_config=ControllerConfig(
                max_synthesis_depth=6,
                enable_wme=True,
                enable_mal=True,
                enable_refinement=True,
            ),
        ),
        # Narrow beam
        AblationConfig(
            name="narrow_beam",
            description="Narrow beam width (10)",
            controller_config=ControllerConfig(
                beam_width=10,
                enable_wme=True,
                enable_mal=True,
                enable_refinement=True,
            ),
        ),
        # Wide beam
        AblationConfig(
            name="wide_beam",
            description="Wide beam width (100)",
            controller_config=ControllerConfig(
                beam_width=100,
                enable_wme=True,
                enable_mal=True,
                enable_refinement=True,
            ),
        ),
    ]


def run_ablation_study(
    tasks: List[ARCTask],
    configs: Optional[List[AblationConfig]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run ablation study on a set of tasks.

    Args:
        tasks: Tasks to evaluate
        configs: Ablation configurations (default: standard set)
        verbose: Print progress

    Returns:
        Results dictionary with per-config metrics
    """
    if configs is None:
        configs = get_ablation_configs()

    results: Dict[str, Any] = {
        "num_tasks": len(tasks),
        "ablations": {},
    }

    for config in configs:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Running ablation: {config.name}")
            print(f"Description: {config.description}")
            print(f"{'='*50}")

        controller = MetaController(config.controller_config)

        successes = 0
        total_iterations = 0
        total_nodes = 0

        for i, task in enumerate(tasks):
            if verbose:
                print(f"  [{i+1}/{len(tasks)}] {task.task_id}...", end=" ")

            result = controller.solve(task)

            if result.success:
                successes += 1
                if verbose:
                    print("PASS")
            else:
                if verbose:
                    print("FAIL")

            total_iterations += result.audit_trace.synthesis_iterations
            total_nodes += result.audit_trace.search_nodes_explored

        ablation_results = {
            "name": config.name,
            "description": config.description,
            "successes": successes,
            "failures": len(tasks) - successes,
            "success_rate": successes / len(tasks) if tasks else 0.0,
            "avg_iterations": total_iterations / len(tasks) if tasks else 0,
            "avg_nodes_explored": total_nodes / len(tasks) if tasks else 0,
        }

        results["ablations"][config.name] = ablation_results

        if verbose:
            print(f"\nAblation '{config.name}' results:")
            print(f"  Success rate: {ablation_results['success_rate']:.1%}")
            print(f"  Avg iterations: {ablation_results['avg_iterations']:.1f}")

    # Compute relative performance
    if "full" in results["ablations"]:
        baseline = results["ablations"]["full"]["success_rate"]
        for name, ablation in results["ablations"].items():
            ablation["relative_performance"] = (
                ablation["success_rate"] / baseline if baseline > 0 else 0.0
            )

    return results


def compare_ablations(results: Dict[str, Any]) -> str:
    """Generate a comparison report from ablation results."""
    lines = []
    lines.append("=" * 60)
    lines.append("ABLATION STUDY RESULTS")
    lines.append("=" * 60)
    lines.append(f"Tasks evaluated: {results['num_tasks']}")
    lines.append("")

    # Sort by success rate
    sorted_ablations = sorted(
        results["ablations"].items(),
        key=lambda x: x[1]["success_rate"],
        reverse=True,
    )

    lines.append(f"{'Ablation':<20} {'Success Rate':>12} {'Relative':>10} {'Avg Iter':>10}")
    lines.append("-" * 60)

    for name, data in sorted_ablations:
        rel = data.get("relative_performance", 1.0)
        lines.append(
            f"{name:<20} {data['success_rate']:>11.1%} {rel:>10.2f} {data['avg_iterations']:>10.1f}"
        )

    lines.append("")
    lines.append("KEY FINDINGS:")

    # Identify important findings
    if "full" in results["ablations"] and "no_refinement" in results["ablations"]:
        full_rate = results["ablations"]["full"]["success_rate"]
        no_ref_rate = results["ablations"]["no_refinement"]["success_rate"]
        diff = full_rate - no_ref_rate
        if diff > 0.01:
            lines.append(f"  - Refinement adds {diff:.1%} to success rate")

    if "full" in results["ablations"] and "no_mal" in results["ablations"]:
        full_rate = results["ablations"]["full"]["success_rate"]
        no_mal_rate = results["ablations"]["no_mal"]["success_rate"]
        diff = full_rate - no_mal_rate
        if diff > 0.01:
            lines.append(f"  - Memory/Abstraction adds {diff:.1%} to success rate")

    return "\n".join(lines)


def main():
    """Run ablation study on synthetic tasks."""
    from .run_synthetic import get_synthetic_tasks

    tasks = get_synthetic_tasks()
    results = run_ablation_study(tasks, verbose=True)

    print("\n" + compare_ablations(results))


if __name__ == "__main__":
    main()
