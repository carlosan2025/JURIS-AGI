"""
Memory retrieval for similar solutions.

Stores and retrieves solutions based on task similarity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json

from ..core.types import Grid, ARCTask
from ..dsl.ast import ASTNode
from ..dsl.prettyprint import ast_to_source


@dataclass
class SolutionMemory:
    """A stored solution in memory."""
    task_id: str
    program: ASTNode
    program_source: str
    task_features: Dict[str, Any]
    success: bool
    robustness_score: float = 0.0
    usage_count: int = 0
    timestamp: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from memory retrieval."""
    memory: SolutionMemory
    similarity: float
    relevance_score: float


class MemoryStore(ABC):
    """
    Abstract base for memory storage.

    Stores solutions and retrieves similar ones.
    """

    @abstractmethod
    def store(self, memory: SolutionMemory) -> None:
        """Store a solution in memory."""
        pass

    @abstractmethod
    def retrieve(
        self,
        task: ARCTask,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve similar solutions.

        Args:
            task: Query task
            top_k: Number of results to return

        Returns:
            List of similar solutions with similarity scores
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored memories."""
        pass


class InMemoryStore(MemoryStore):
    """
    Simple in-memory storage.

    Uses feature-based similarity for retrieval.
    """

    def __init__(self):
        self.memories: Dict[str, SolutionMemory] = {}

    def store(self, memory: SolutionMemory) -> None:
        """Store a solution."""
        key = self._compute_key(memory)
        self.memories[key] = memory

    def retrieve(
        self,
        task: ARCTask,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve similar solutions."""
        if not self.memories:
            return []

        query_features = self._extract_features(task)
        results: List[RetrievalResult] = []

        for memory in self.memories.values():
            similarity = self._compute_similarity(query_features, memory.task_features)
            relevance = similarity * (1.0 if memory.success else 0.5)

            results.append(RetrievalResult(
                memory=memory,
                similarity=similarity,
                relevance_score=relevance,
            ))

        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()

    def _compute_key(self, memory: SolutionMemory) -> str:
        """Compute unique key for memory."""
        content = f"{memory.task_id}:{memory.program_source}"
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_features(self, task: ARCTask) -> Dict[str, Any]:
        """Extract features from a task for similarity computation."""
        features: Dict[str, Any] = {}

        if not task.train:
            return features

        # Aggregate features from training pairs
        for i, pair in enumerate(task.train):
            inp, out = pair.input, pair.output
            features[f"pair_{i}_input_shape"] = inp.shape
            features[f"pair_{i}_output_shape"] = out.shape
            features[f"pair_{i}_same_dims"] = inp.shape == out.shape
            features[f"pair_{i}_input_palette_size"] = len(inp.palette)
            features[f"pair_{i}_output_palette_size"] = len(out.palette)

        # Global features
        features["num_train_pairs"] = len(task.train)
        features["num_test_pairs"] = len(task.test)

        # Check consistency across pairs
        all_same_dims = all(
            task.train[0].input.shape == p.input.shape and
            task.train[0].output.shape == p.output.shape
            for p in task.train
        )
        features["consistent_dimensions"] = all_same_dims

        return features

    def _compute_similarity(
        self,
        query_features: Dict[str, Any],
        memory_features: Dict[str, Any],
    ) -> float:
        """Compute similarity between feature sets."""
        if not query_features or not memory_features:
            return 0.0

        common_keys = set(query_features.keys()) & set(memory_features.keys())
        if not common_keys:
            return 0.0

        matches = 0
        for key in common_keys:
            if query_features[key] == memory_features[key]:
                matches += 1

        return matches / len(common_keys)


class PersistentMemoryStore(MemoryStore):
    """
    Persistent memory storage with disk backing.

    Stub implementation - can be extended with actual persistence.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._in_memory = InMemoryStore()

    def store(self, memory: SolutionMemory) -> None:
        """Store a solution."""
        self._in_memory.store(memory)
        # TODO: Persist to disk

    def retrieve(
        self,
        task: ARCTask,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve similar solutions."""
        return self._in_memory.retrieve(task, top_k)

    def clear(self) -> None:
        """Clear all memories."""
        self._in_memory.clear()


def retrieve_similar(
    task: ARCTask,
    memory_store: MemoryStore,
    top_k: int = 5,
) -> List[RetrievalResult]:
    """Convenience function to retrieve similar solutions."""
    return memory_store.retrieve(task, top_k)


def create_memory_from_solution(
    task: ARCTask,
    program: ASTNode,
    success: bool,
    robustness_score: float = 0.0,
) -> SolutionMemory:
    """Create a memory entry from a solution."""
    # Extract task features
    features: Dict[str, Any] = {}
    if task.train:
        pair = task.train[0]
        features["input_shape"] = pair.input.shape
        features["output_shape"] = pair.output.shape
        features["same_dims"] = pair.input.shape == pair.output.shape

    return SolutionMemory(
        task_id=task.task_id,
        program=program,
        program_source=ast_to_source(program),
        task_features=features,
        success=success,
        robustness_score=robustness_score,
    )


# ============================================================================
# JSON-based Macro Storage for MAL
# ============================================================================

@dataclass
class StoredMacro:
    """A macro stored as JSON with tags for retrieval."""
    name: str
    code: str  # DSL source code
    tags: List[str]  # Heuristic tags for similarity matching
    mdl_cost: int = 1  # MDL cost of the macro
    usage_count: int = 0
    success_count: int = 0
    created_from_task: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "code": self.code,
            "tags": self.tags,
            "mdl_cost": self.mdl_cost,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "created_from_task": self.created_from_task,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredMacro":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            code=data["code"],
            tags=data.get("tags", []),
            mdl_cost=data.get("mdl_cost", 1),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            created_from_task=data.get("created_from_task"),
        )


class MacroStore:
    """
    JSON-based macro storage with heuristic retrieval.

    Stores macros as simple JSON objects and retrieves
    by tag-based similarity (no vectors).
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize macro store.

        Args:
            storage_path: Path to JSON file for persistence
        """
        from pathlib import Path

        self.storage_path = Path(storage_path) if storage_path else None
        self.macros: Dict[str, StoredMacro] = {}
        self._load()

    def _load(self) -> None:
        """Load macros from disk if path exists."""
        if self.storage_path and self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    for macro_data in data.get("macros", []):
                        macro = StoredMacro.from_dict(macro_data)
                        self.macros[macro.name] = macro
            except (json.JSONDecodeError, KeyError):
                pass  # Start fresh on error

    def _save(self) -> None:
        """Save macros to disk if path is set."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(
                    {"macros": [m.to_dict() for m in self.macros.values()]},
                    f,
                    indent=2,
                )

    def store_macro(self, macro: StoredMacro) -> None:
        """Store a macro."""
        self.macros[macro.name] = macro
        self._save()

    def get_macro(self, name: str) -> Optional[StoredMacro]:
        """Get a macro by name."""
        return self.macros.get(name)

    def retrieve_by_tags(
        self,
        query_tags: List[str],
        top_k: int = 5,
    ) -> List[Tuple[StoredMacro, float]]:
        """
        Retrieve macros by tag similarity.

        Uses Jaccard similarity between tag sets.

        Args:
            query_tags: Tags describing the current task
            top_k: Number of macros to return

        Returns:
            List of (macro, similarity_score) tuples
        """
        if not query_tags or not self.macros:
            return []

        query_set = set(query_tags)
        results: List[Tuple[StoredMacro, float]] = []

        for macro in self.macros.values():
            macro_set = set(macro.tags)
            if not macro_set:
                continue

            # Jaccard similarity
            intersection = len(query_set & macro_set)
            union = len(query_set | macro_set)
            similarity = intersection / union if union > 0 else 0.0

            # Boost by success rate
            if macro.usage_count > 0:
                success_rate = macro.success_count / macro.usage_count
                similarity *= (0.5 + 0.5 * success_rate)

            results.append((macro, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def retrieve_for_task(
        self,
        task: ARCTask,
        top_k: int = 5,
    ) -> List[Tuple[StoredMacro, float]]:
        """
        Retrieve macros relevant to a task using heuristic features.

        Extracts task tags and retrieves by similarity.
        """
        tags = extract_task_tags(task)
        return self.retrieve_by_tags(tags, top_k)

    def record_usage(self, macro_name: str, success: bool) -> None:
        """Record usage of a macro."""
        if macro_name in self.macros:
            self.macros[macro_name].usage_count += 1
            if success:
                self.macros[macro_name].success_count += 1
            self._save()

    def list_all(self) -> List[StoredMacro]:
        """List all stored macros."""
        return list(self.macros.values())

    def clear(self) -> None:
        """Clear all macros."""
        self.macros.clear()
        self._save()


def extract_task_tags(task: ARCTask) -> List[str]:
    """
    Extract heuristic tags from a task for macro retrieval.

    Tags capture task characteristics useful for matching.
    """
    tags: List[str] = []

    if not task.train:
        return tags

    # Dimension-based tags
    same_dims = all(
        p.input.shape == p.output.shape
        for p in task.train
    )
    if same_dims:
        tags.append("same_dims")
    else:
        tags.append("different_dims")

    # Check for scaling
    first_pair = task.train[0]
    ih, iw = first_pair.input.shape
    oh, ow = first_pair.output.shape

    if oh > ih or ow > iw:
        tags.append("enlarging")
        if oh == 2 * ih and ow == 2 * iw:
            tags.append("scale_2x")
        elif oh == 3 * ih and ow == 3 * iw:
            tags.append("scale_3x")
    elif oh < ih or ow < iw:
        tags.append("shrinking")
        tags.append("cropping")

    # Palette tags
    all_input_colors = set()
    all_output_colors = set()
    for pair in task.train:
        all_input_colors.update(pair.input.palette)
        all_output_colors.update(pair.output.palette)

    if all_output_colors.issubset(all_input_colors):
        tags.append("palette_preserved")
    if len(all_output_colors) < len(all_input_colors):
        tags.append("color_reduction")
    if len(all_output_colors) > len(all_input_colors):
        tags.append("color_addition")

    # Symmetry tags
    if _is_symmetric_h(first_pair.output):
        tags.append("output_h_symmetric")
    if _is_symmetric_v(first_pair.output):
        tags.append("output_v_symmetric")

    # Object count tags
    from ..representation.objects import extract_connected_objects
    try:
        input_objs = extract_connected_objects(first_pair.input)
        output_objs = extract_connected_objects(first_pair.output)
        if len(output_objs) == len(input_objs):
            tags.append("object_count_preserved")
        elif len(output_objs) < len(input_objs):
            tags.append("object_count_reduced")
        elif len(output_objs) > len(input_objs):
            tags.append("object_count_increased")
        if len(input_objs) == 1:
            tags.append("single_input_object")
        if len(output_objs) == 1:
            tags.append("single_output_object")
    except Exception:
        pass

    return tags


def _is_symmetric_h(grid: Grid) -> bool:
    """Check horizontal symmetry."""
    import numpy as np
    return np.array_equal(grid.data, np.fliplr(grid.data))


def _is_symmetric_v(grid: Grid) -> bool:
    """Check vertical symmetry."""
    import numpy as np
    return np.array_equal(grid.data, np.flipud(grid.data))


def retrieve_macros(
    task: ARCTask,
    macro_store: MacroStore,
    top_k: int = 5,
) -> List[Tuple[StoredMacro, float]]:
    """
    Convenience function to retrieve macros for a task.

    Args:
        task: The task to find macros for
        macro_store: The macro store to search
        top_k: Number of macros to return

    Returns:
        List of (macro, similarity) tuples
    """
    return macro_store.retrieve_for_task(task, top_k)
