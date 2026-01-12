#!/usr/bin/env python3
"""
Run JURIS-AGI on synthetic tasks for testing.

Creates simple synthetic tasks to verify the system works.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from ..core.types import ARCTask, ARCPair, Grid
from ..controller.router import MetaController, ControllerConfig


def create_identity_task() -> ARCTask:
    """Create a simple identity task."""
    grid1 = Grid.from_list([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    grid2 = Grid.from_list([
        [2, 2],
        [2, 2],
    ])

    return ARCTask(
        task_id="synthetic_identity",
        train=[
            ARCPair(input=grid1, output=grid1),
            ARCPair(input=grid2, output=grid2),
        ],
        test=[
            ARCPair(input=Grid.from_list([[3, 3, 3]]), output=Grid.from_list([[3, 3, 3]])),
        ],
    )


def create_rotate_task() -> ARCTask:
    """Create a rotation task (90 degrees clockwise)."""
    inp1 = Grid.from_list([
        [1, 0],
        [0, 0],
    ])
    out1 = Grid.from_list([
        [0, 1],
        [0, 0],
    ])

    inp2 = Grid.from_list([
        [2, 2, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    out2 = Grid.from_list([
        [0, 0, 2],
        [0, 0, 2],
        [0, 0, 0],
    ])

    test_inp = Grid.from_list([
        [3, 0],
        [3, 0],
    ])
    test_out = Grid.from_list([
        [3, 3],
        [0, 0],
    ])

    return ARCTask(
        task_id="synthetic_rotate90",
        train=[
            ARCPair(input=inp1, output=out1),
            ARCPair(input=inp2, output=out2),
        ],
        test=[
            ARCPair(input=test_inp, output=test_out),
        ],
    )


def create_reflect_h_task() -> ARCTask:
    """Create horizontal reflection task."""
    inp1 = Grid.from_list([
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    out1 = Grid.from_list([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0],
    ])

    inp2 = Grid.from_list([
        [2, 3],
        [0, 0],
    ])
    out2 = Grid.from_list([
        [3, 2],
        [0, 0],
    ])

    test_inp = Grid.from_list([
        [4, 0],
    ])
    test_out = Grid.from_list([
        [0, 4],
    ])

    return ARCTask(
        task_id="synthetic_reflect_h",
        train=[
            ARCPair(input=inp1, output=out1),
            ARCPair(input=inp2, output=out2),
        ],
        test=[
            ARCPair(input=test_inp, output=test_out),
        ],
    )


def create_crop_task() -> ARCTask:
    """Create a crop-to-content task."""
    inp1 = Grid.from_list([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ])
    out1 = Grid.from_list([
        [1, 1],
        [1, 1],
    ])

    inp2 = Grid.from_list([
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0],
    ])
    out2 = Grid.from_list([
        [2],
    ])

    test_inp = Grid.from_list([
        [0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 3, 3, 3, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    test_out = Grid.from_list([
        [0, 3, 0],
        [3, 3, 3],
        [0, 3, 0],
    ])

    return ARCTask(
        task_id="synthetic_crop",
        train=[
            ARCPair(input=inp1, output=out1),
            ARCPair(input=inp2, output=out2),
        ],
        test=[
            ARCPair(input=test_inp, output=test_out),
        ],
    )


def create_scale_task() -> ARCTask:
    """Create a 2x scaling task."""
    inp1 = Grid.from_list([
        [1],
    ])
    out1 = Grid.from_list([
        [1, 1],
        [1, 1],
    ])

    inp2 = Grid.from_list([
        [1, 2],
    ])
    out2 = Grid.from_list([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
    ])

    test_inp = Grid.from_list([
        [3, 0],
        [0, 3],
    ])
    test_out = Grid.from_list([
        [3, 3, 0, 0],
        [3, 3, 0, 0],
        [0, 0, 3, 3],
        [0, 0, 3, 3],
    ])

    return ARCTask(
        task_id="synthetic_scale2x",
        train=[
            ARCPair(input=inp1, output=out1),
            ARCPair(input=inp2, output=out2),
        ],
        test=[
            ARCPair(input=test_inp, output=test_out),
        ],
    )


def get_synthetic_tasks() -> List[ARCTask]:
    """Get all synthetic tasks."""
    return [
        create_identity_task(),
        create_rotate_task(),
        create_reflect_h_task(),
        create_crop_task(),
        create_scale_task(),
    ]


def run_synthetic_evaluation(
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run evaluation on synthetic tasks."""
    tasks = get_synthetic_tasks()

    config = ControllerConfig(
        max_synthesis_depth=3,
        beam_width=30,
        max_synthesis_iterations=500,
        compute_robustness=False,  # Skip for speed
    )
    controller = MetaController(config)

    results = []
    successes = 0

    for task in tasks:
        if verbose:
            print(f"\nTask: {task.task_id}")

        result = controller.solve(task)

        if verbose:
            status = "PASS" if result.success else "FAIL"
            print(f"  Result: {status}")
            if result.success:
                print(f"  Program: {result.audit_trace.program_source}")
            else:
                print(f"  Error: {result.error_message}")

        results.append({
            "task_id": task.task_id,
            "success": result.success,
            "program": result.audit_trace.program_source if result.success else None,
        })

        if result.success:
            successes += 1

    summary = {
        "total": len(tasks),
        "successes": successes,
        "failures": len(tasks) - successes,
        "success_rate": successes / len(tasks),
        "results": results,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"SYNTHETIC EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total: {summary['total']}")
        print(f"Successes: {summary['successes']}")
        print(f"Failures: {summary['failures']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run JURIS-AGI on synthetic tasks"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file for results",
    )

    args = parser.parse_args()

    summary = run_synthetic_evaluation(verbose=not args.quiet)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
