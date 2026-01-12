"""
Trace schema and writer for audit trails.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class TraceEntry:
    """A single entry in the reasoning trace."""
    timestamp: str
    event_type: str  # "synthesis", "evaluation", "refinement", "critic", etc.
    component: str   # "cre", "wme", "mal", "controller"
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(cls, event_type: str, component: str, **details: Any) -> "TraceEntry":
        return cls(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            component=component,
            details=details,
        )


@dataclass
class SolveTrace:
    """Complete trace of a solve attempt."""
    task_id: str
    start_time: str
    end_time: Optional[str] = None
    success: bool = False
    entries: List[TraceEntry] = field(default_factory=list)
    final_program: Optional[str] = None
    final_metrics: Dict[str, Any] = field(default_factory=dict)

    # Budget and uncertainty tracking
    budget_per_phase: Dict[str, Any] = field(default_factory=dict)
    uncertainty_metrics: Dict[str, Any] = field(default_factory=dict)
    regime: Optional[str] = None

    def add_entry(self, entry: TraceEntry) -> None:
        """Add an entry to the trace."""
        self.entries.append(entry)

    def log(
        self,
        event_type: str,
        component: str,
        **details: Any
    ) -> None:
        """Convenience method to log an event."""
        self.add_entry(TraceEntry.now(event_type, component, **details))

    def finalize(self, success: bool, program: Optional[str] = None) -> None:
        """Mark the trace as complete."""
        self.end_time = datetime.now().isoformat()
        self.success = success
        self.final_program = program

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "success": self.success,
            "final_program": self.final_program,
            "final_metrics": self.final_metrics,
            "entries": [asdict(e) for e in self.entries],
            # Budget and uncertainty
            "budget_per_phase": self.budget_per_phase,
            "uncertainty_metrics": self.uncertainty_metrics,
            "regime": self.regime,
        }

    def log_budget(self, phase_budgets: Dict[str, Any]) -> None:
        """Log budget allocation and usage per phase."""
        self.budget_per_phase = phase_budgets
        self.log("budget_update", "controller", phases=phase_budgets)

    def log_uncertainty(
        self,
        epistemic: float,
        aleatoric: float,
        num_hypotheses: int,
        diff_variance: float,
    ) -> None:
        """
        Log uncertainty metrics.

        Args:
            epistemic: Epistemic uncertainty (reducible through more computation)
            aleatoric: Aleatoric uncertainty (from WME confidence)
            num_hypotheses: Number of competing consistent hypotheses
            diff_variance: Variance in diff scores across training pairs
        """
        self.uncertainty_metrics = {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total": min(1.0, (epistemic + aleatoric) / 2),
            "num_hypotheses": num_hypotheses,
            "diff_variance": diff_variance,
        }
        self.log(
            "uncertainty_update",
            "controller",
            **self.uncertainty_metrics,
        )

    def set_regime(self, regime: str, confidence: float, rationale: str) -> None:
        """Set task regime classification."""
        self.regime = regime
        self.log(
            "regime_determined",
            "controller",
            regime=regime,
            confidence=confidence,
            rationale=rationale,
        )

    @classmethod
    def start(cls, task_id: str) -> "SolveTrace":
        """Create a new trace for a task."""
        return cls(
            task_id=task_id,
            start_time=datetime.now().isoformat(),
        )


class TraceWriter:
    """Writes traces to disk."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, trace: SolveTrace) -> Path:
        """Write a trace to a JSON file."""
        filename = f"{trace.task_id}_{trace.start_time.replace(':', '-')}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)

        return filepath

    def write_summary(
        self,
        traces: List[SolveTrace],
        filename: str = "summary.json"
    ) -> Path:
        """Write a summary of multiple traces."""
        summary = {
            "total_tasks": len(traces),
            "successful": sum(1 for t in traces if t.success),
            "failed": sum(1 for t in traces if not t.success),
            "success_rate": (
                sum(1 for t in traces if t.success) / len(traces)
                if traces else 0.0
            ),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "success": t.success,
                    "program": t.final_program,
                }
                for t in traces
            ],
        }

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        return filepath


class TraceContext:
    """Context manager for tracing a solve attempt."""

    def __init__(self, task_id: str, writer: Optional[TraceWriter] = None):
        self.trace = SolveTrace.start(task_id)
        self.writer = writer

    def __enter__(self) -> SolveTrace:
        return self.trace

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self.trace.log(
                "error",
                "system",
                error_type=str(exc_type),
                error_message=str(exc_val),
            )
            self.trace.finalize(success=False)
        if self.writer:
            self.writer.write(self.trace)


# ============================================================================
# JSONL Trace Writer for MAL integration
# ============================================================================

class JSONLTraceWriter:
    """
    JSONL writer for solve traces.

    Appends traces to a single file, one JSON object per line.
    More efficient for batch processing and streaming reads.
    """

    def __init__(self, trace_dir: str = "traces"):
        """
        Initialize JSONL trace writer.

        Args:
            trace_dir: Directory to write traces to
        """
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._current_session: Optional[str] = None

    def get_session_path(self, session_id: Optional[str] = None) -> Path:
        """Get path for current session's trace file."""
        if session_id is None:
            if self._current_session is None:
                self._current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = self._current_session
        return self.trace_dir / f"traces_{session_id}.jsonl"

    def write_trace(
        self,
        trace: SolveTrace,
        session_id: Optional[str] = None,
    ) -> Path:
        """
        Append a single trace to the JSONL file.

        Args:
            trace: The trace to write
            session_id: Optional session identifier

        Returns:
            Path to the trace file
        """
        trace_path = self.get_session_path(session_id)

        with open(trace_path, "a") as f:
            json.dump(trace.to_dict(), f, separators=(",", ":"))
            f.write("\n")

        return trace_path

    def write_traces(
        self,
        traces: List[SolveTrace],
        session_id: Optional[str] = None,
    ) -> Path:
        """
        Write multiple traces to the JSONL file.

        Args:
            traces: List of traces to write
            session_id: Optional session identifier

        Returns:
            Path to the trace file
        """
        trace_path = self.get_session_path(session_id)

        with open(trace_path, "a") as f:
            for trace in traces:
                json.dump(trace.to_dict(), f, separators=(",", ":"))
                f.write("\n")

        return trace_path


class JSONLTraceReader:
    """
    Reader for JSONL trace files.

    Supports streaming and filtering traces.
    """

    def __init__(self, trace_dir: str = "traces"):
        """Initialize trace reader."""
        self.trace_dir = Path(trace_dir)

    def list_trace_files(self) -> List[Path]:
        """List all JSONL trace files in the directory."""
        if not self.trace_dir.exists():
            return []
        return sorted(self.trace_dir.glob("traces_*.jsonl"))

    def read_traces(
        self,
        trace_path: Optional[Path] = None,
        success_only: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read traces from a file.

        Args:
            trace_path: Path to trace file (if None, reads most recent)
            success_only: If True, only return successful solves
            limit: Maximum number of traces to return

        Returns:
            List of trace dictionaries
        """
        if trace_path is None:
            files = self.list_trace_files()
            if not files:
                return []
            trace_path = files[-1]  # Most recent

        if not trace_path.exists():
            return []

        traces = []
        with open(trace_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    trace = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if success_only and not trace.get("success", False):
                    continue

                traces.append(trace)

                if limit is not None and len(traces) >= limit:
                    break

        return traces

    def read_all_traces(
        self,
        success_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Read all traces from all JSONL files.

        Args:
            success_only: If True, only return successful solves

        Returns:
            List of all trace dictionaries
        """
        all_traces = []
        for trace_path in self.list_trace_files():
            traces = self.read_traces(trace_path, success_only=success_only)
            all_traces.extend(traces)
        return all_traces

    def get_successful_programs(self) -> List[Dict[str, Any]]:
        """
        Get all successful programs with their features.

        Returns:
            List of dicts with program, task_id, and metrics
        """
        successful = []
        for trace in self.read_all_traces(success_only=True):
            if trace.get("final_program"):
                successful.append({
                    "task_id": trace["task_id"],
                    "program": trace["final_program"],
                    "metrics": trace.get("final_metrics", {}),
                    "start_time": trace.get("start_time"),
                })
        return successful

    def iter_traces(
        self,
        trace_path: Optional[Path] = None,
        success_only: bool = False,
    ):
        """
        Iterate over traces without loading all into memory.

        Args:
            trace_path: Path to trace file
            success_only: If True, only yield successful solves

        Yields:
            Trace dictionaries
        """
        if trace_path is None:
            files = self.list_trace_files()
            if not files:
                return
            trace_path = files[-1]

        if not trace_path.exists():
            return

        with open(trace_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    trace = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if success_only and not trace.get("success", False):
                    continue

                yield trace


def create_trace_from_task(task, task_id: str = "unknown") -> SolveTrace:
    """
    Create a SolveTrace initialized with task features.

    Args:
        task: ARCTask instance
        task_id: Task identifier

    Returns:
        Initialized SolveTrace
    """
    trace = SolveTrace.start(task_id)

    # Store task feature info in initial log entry
    input_dims = [pair.input.shape for pair in task.train]
    output_dims = [pair.output.shape for pair in task.train]

    all_input_colors = set()
    all_output_colors = set()
    for pair in task.train:
        all_input_colors.update(pair.input.palette)
        all_output_colors.update(pair.output.palette)

    trace.log(
        "task_loaded",
        "cre",
        num_train_pairs=len(task.train),
        num_test_pairs=len(task.test),
        input_dims=[list(d) for d in input_dims],
        output_dims=[list(d) for d in output_dims],
        input_palette=sorted(list(all_input_colors)),
        output_palette=sorted(list(all_output_colors)),
    )

    return trace
