"""
MultiViewState builder - combines grid tokens, objects, and relational graphs.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

from .types import Grid, GridObject, BoundingBox

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class MultiViewState:
    """
    Multi-view representation of a grid combining:
    - Raw grid tokens (pixel-level)
    - Extracted objects (object-level)
    - Relational graph (structure-level)
    """
    grid: Grid
    grid_tokens: np.ndarray  # Tokenized representation
    objects: List[GridObject] = field(default_factory=list)
    relational_graph: Optional[Any] = None  # networkx.DiGraph when available
    features: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.grid.height

    @property
    def width(self) -> int:
        return self.grid.width

    @property
    def num_objects(self) -> int:
        return len(self.objects)


class MultiViewStateBuilder:
    """
    Builds MultiViewState from a grid.

    Integrates tokenization, object extraction, and relation building.
    """

    def __init__(
        self,
        extract_objects: bool = True,
        build_relations: bool = True,
        compute_features: bool = True,
    ):
        self.extract_objects = extract_objects
        self.build_relations = build_relations
        self.compute_features = compute_features

    def build(self, grid: Grid) -> MultiViewState:
        """Build complete MultiViewState from a grid."""
        # Tokenize grid (simple: just the raw values for now)
        grid_tokens = grid.data.copy()

        # Extract objects if requested
        objects: List[GridObject] = []
        if self.extract_objects:
            from ..representation.objects import extract_connected_objects
            objects = extract_connected_objects(grid)

        # Build relational graph if requested
        relational_graph = None
        if self.build_relations and HAS_NETWORKX and objects:
            from ..representation.relations import build_relational_graph
            relational_graph = build_relational_graph(objects, grid)

        # Compute features if requested
        features: Dict[str, Any] = {}
        if self.compute_features:
            features = self._compute_features(grid, objects)

        return MultiViewState(
            grid=grid,
            grid_tokens=grid_tokens,
            objects=objects,
            relational_graph=relational_graph,
            features=features,
        )

    def _compute_features(
        self, grid: Grid, objects: List[GridObject]
    ) -> Dict[str, Any]:
        """Compute grid and object features."""
        features = {
            "grid_height": grid.height,
            "grid_width": grid.width,
            "palette": list(grid.palette),
            "num_colors": len(grid.palette),
            "num_objects": len(objects),
            "background_color": self._detect_background(grid),
        }

        if objects:
            features["object_sizes"] = [obj.pixel_count for obj in objects]
            features["object_colors"] = [
                list(obj.colors) for obj in objects
            ]

        return features

    def _detect_background(self, grid: Grid) -> int:
        """Detect the most likely background color (most common)."""
        unique, counts = np.unique(grid.data, return_counts=True)
        return int(unique[np.argmax(counts)])


def build_multiview_state(grid: Grid, **kwargs: Any) -> MultiViewState:
    """Convenience function to build MultiViewState."""
    builder = MultiViewStateBuilder(**kwargs)
    return builder.build(grid)
