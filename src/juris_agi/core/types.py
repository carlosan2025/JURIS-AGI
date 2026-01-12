"""
Core type definitions for JURIS-AGI.

All data structures are typed using dataclasses for clarity and immutability where appropriate.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Any, Set, FrozenSet
import numpy as np


# Type aliases
Color = int  # 0-9 in ARC (0 = black/background)
Point = Tuple[int, int]  # (row, col)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box."""
    min_row: int
    min_col: int
    max_row: int
    max_col: int

    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    def contains(self, row: int, col: int) -> bool:
        return (self.min_row <= row <= self.max_row and
                self.min_col <= col <= self.max_col)


@dataclass
class Grid:
    """
    A 2D grid of colors. Core representation for ARC tasks.

    Stored as numpy array internally for efficient operations.
    """
    data: np.ndarray  # Shape: (height, width), dtype: int

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.int32)
        if self.data.ndim != 2:
            raise ValueError(f"Grid must be 2D, got shape {self.data.shape}")

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def palette(self) -> Set[Color]:
        """Returns the set of colors used in this grid."""
        return set(np.unique(self.data).tolist())

    def __getitem__(self, key) -> int:
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def copy(self) -> "Grid":
        return Grid(self.data.copy())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.data, other.data)

    def __hash__(self) -> int:
        return hash(self.data.tobytes())

    @classmethod
    def from_list(cls, data: List[List[int]]) -> "Grid":
        return cls(np.array(data, dtype=np.int32))

    def to_list(self) -> List[List[int]]:
        return self.data.tolist()

    @classmethod
    def zeros(cls, height: int, width: int) -> "Grid":
        return cls(np.zeros((height, width), dtype=np.int32))

    @classmethod
    def full(cls, height: int, width: int, fill_value: Color) -> "Grid":
        return cls(np.full((height, width), fill_value, dtype=np.int32))


@dataclass(frozen=True)
class GridObject:
    """
    A discrete object extracted from a grid.

    Represented as a set of (row, col, color) tuples relative to the object's
    bounding box origin, plus the bounding box in the original grid.
    """
    pixels: FrozenSet[Tuple[int, int, Color]]  # (local_row, local_col, color)
    bbox: BoundingBox
    object_id: int = 0

    @property
    def colors(self) -> Set[Color]:
        """Returns the set of colors in this object (excluding background)."""
        return {c for _, _, c in self.pixels if c != 0}

    @property
    def primary_color(self) -> Optional[Color]:
        """Returns the most common non-background color."""
        color_counts: Dict[Color, int] = {}
        for _, _, c in self.pixels:
            if c != 0:
                color_counts[c] = color_counts.get(c, 0) + 1
        if not color_counts:
            return None
        return max(color_counts, key=lambda c: color_counts[c])

    @property
    def pixel_count(self) -> int:
        """Number of non-background pixels."""
        return sum(1 for _, _, c in self.pixels if c != 0)

    def to_grid(self) -> Grid:
        """Convert object to a grid (bbox-sized)."""
        grid = Grid.zeros(self.bbox.height, self.bbox.width)
        for r, c, color in self.pixels:
            grid[r, c] = color
        return grid


@dataclass
class ARCPair:
    """A single input-output pair from an ARC task."""
    input: Grid
    output: Grid


@dataclass
class ARCTask:
    """
    A complete ARC task with training and test pairs.
    """
    task_id: str
    train: List[ARCPair]
    test: List[ARCPair]  # For evaluation; output may be None during solving

    @classmethod
    def from_dict(cls, task_id: str, data: Dict[str, Any]) -> "ARCTask":
        """Load from ARC JSON format."""
        train = [
            ARCPair(
                input=Grid.from_list(pair["input"]),
                output=Grid.from_list(pair["output"])
            )
            for pair in data["train"]
        ]
        test = [
            ARCPair(
                input=Grid.from_list(pair["input"]),
                output=Grid.from_list(pair.get("output", [[0]]))
            )
            for pair in data["test"]
        ]
        return cls(task_id=task_id, train=train, test=test)


class ConstraintType(Enum):
    """Types of constraints for synthesis."""
    DIMENSION_MATCH = auto()  # Output dimensions must match
    PALETTE_SUBSET = auto()   # Output colors subset of allowed
    EXACT_MATCH = auto()      # Exact grid match on training
    OBJECT_COUNT = auto()     # Number of objects preserved/changed
    SYMMETRY = auto()         # Symmetry constraints


@dataclass
class Constraint:
    """A constraint used during synthesis."""
    constraint_type: ConstraintType
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # For soft constraints
    is_hard: bool = True  # Hard constraints must be satisfied


@dataclass
class SymbolicDiffSummary:
    """Summary of a symbolic diff for audit trace."""
    pair_index: int
    pair_type: str  # "train" or "test"
    dimension_match: bool
    exact_match: bool
    pixel_accuracy: float
    num_errors: int
    severity: float
    extra_colors: List[int] = field(default_factory=list)
    missing_colors: List[int] = field(default_factory=list)
    error_pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_index": self.pair_index,
            "pair_type": self.pair_type,
            "dimension_match": self.dimension_match,
            "exact_match": self.exact_match,
            "pixel_accuracy": self.pixel_accuracy,
            "num_errors": self.num_errors,
            "severity": self.severity,
            "extra_colors": self.extra_colors,
            "missing_colors": self.missing_colors,
            "error_pattern": self.error_pattern,
        }


@dataclass
class AuditTrace:
    """
    Complete audit trail of the solving process.

    Required for interpretability and debugging.
    """
    task_id: str
    program_source: str  # DSL program that solved the task
    program_ast: Optional[Any] = None  # AST representation
    constraints_satisfied: List[str] = field(default_factory=list)
    constraints_violated: List[str] = field(default_factory=list)
    diffs_from_expected: List[Dict[str, Any]] = field(default_factory=list)
    robustness_score: float = 0.0
    synthesis_iterations: int = 0
    refinement_steps: int = 0
    search_nodes_explored: int = 0
    wme_suggestions_used: List[str] = field(default_factory=list)
    mal_retrievals: List[str] = field(default_factory=list)

    # Extended fields for critic/refinement info
    program_depth: int = 0
    program_size: int = 0
    expansions_generated: int = 0
    candidates_pruned: int = 0
    runtime_seconds: float = 0.0
    symbolic_diffs: List[SymbolicDiffSummary] = field(default_factory=list)
    near_miss_count: int = 0
    refinement_applied: bool = False
    refinement_improved: bool = False
    refinement_edits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "program_source": self.program_source,
            "constraints_satisfied": self.constraints_satisfied,
            "constraints_violated": self.constraints_violated,
            "diffs_from_expected": self.diffs_from_expected,
            "robustness_score": self.robustness_score,
            "synthesis_iterations": self.synthesis_iterations,
            "refinement_steps": self.refinement_steps,
            "search_nodes_explored": self.search_nodes_explored,
            "wme_suggestions_used": self.wme_suggestions_used,
            "mal_retrievals": self.mal_retrievals,
            # Extended fields
            "program_depth": self.program_depth,
            "program_size": self.program_size,
            "expansions_generated": self.expansions_generated,
            "candidates_pruned": self.candidates_pruned,
            "runtime_seconds": self.runtime_seconds,
            "symbolic_diffs": [d.to_dict() for d in self.symbolic_diffs],
            "near_miss_count": self.near_miss_count,
            "refinement_applied": self.refinement_applied,
            "refinement_improved": self.refinement_improved,
            "refinement_edits": self.refinement_edits,
        }


@dataclass
class SolverResult:
    """Result from the JURIS-AGI solver."""
    task_id: str
    success: bool
    predictions: List[Grid]  # One per test input
    audit_trace: AuditTrace
    error_message: Optional[str] = None

    @property
    def is_certified(self) -> bool:
        """Whether the solution passed symbolic verification."""
        return self.success and len(self.audit_trace.constraints_violated) == 0
