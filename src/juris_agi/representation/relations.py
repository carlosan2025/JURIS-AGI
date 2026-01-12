"""
Build relational graphs from extracted objects.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple

from ..core.types import Grid, GridObject, BoundingBox

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class RelationType(Enum):
    """Types of spatial relations between objects."""
    ABOVE = auto()
    BELOW = auto()
    LEFT_OF = auto()
    RIGHT_OF = auto()
    OVERLAPS = auto()
    CONTAINS = auto()
    CONTAINED_BY = auto()
    ADJACENT = auto()
    SAME_COLOR = auto()
    SAME_SIZE = auto()
    SAME_SHAPE = auto()


@dataclass
class Relation:
    """A relation between two objects."""
    source_id: int
    target_id: int
    relation_type: RelationType
    strength: float = 1.0  # Confidence or degree of relation


def compute_bbox_relation(bbox1: BoundingBox, bbox2: BoundingBox) -> List[RelationType]:
    """Compute spatial relations between two bounding boxes."""
    relations = []

    # Vertical relations
    if bbox1.max_row < bbox2.min_row:
        relations.append(RelationType.ABOVE)
    elif bbox1.min_row > bbox2.max_row:
        relations.append(RelationType.BELOW)

    # Horizontal relations
    if bbox1.max_col < bbox2.min_col:
        relations.append(RelationType.LEFT_OF)
    elif bbox1.min_col > bbox2.max_col:
        relations.append(RelationType.RIGHT_OF)

    # Overlap/containment
    h_overlap = not (bbox1.max_col < bbox2.min_col or bbox1.min_col > bbox2.max_col)
    v_overlap = not (bbox1.max_row < bbox2.min_row or bbox1.min_row > bbox2.max_row)

    if h_overlap and v_overlap:
        # Check containment
        if (bbox1.min_row <= bbox2.min_row and bbox1.max_row >= bbox2.max_row and
            bbox1.min_col <= bbox2.min_col and bbox1.max_col >= bbox2.max_col):
            relations.append(RelationType.CONTAINS)
        elif (bbox2.min_row <= bbox1.min_row and bbox2.max_row >= bbox1.max_row and
              bbox2.min_col <= bbox1.min_col and bbox2.max_col >= bbox1.max_col):
            relations.append(RelationType.CONTAINED_BY)
        else:
            relations.append(RelationType.OVERLAPS)

    # Adjacency (within 1 pixel)
    adjacent = False
    if bbox1.max_row + 1 == bbox2.min_row or bbox2.max_row + 1 == bbox1.min_row:
        if h_overlap:
            adjacent = True
    if bbox1.max_col + 1 == bbox2.min_col or bbox2.max_col + 1 == bbox1.min_col:
        if v_overlap:
            adjacent = True
    if adjacent:
        relations.append(RelationType.ADJACENT)

    return relations


def compute_object_relations(obj1: GridObject, obj2: GridObject) -> List[Relation]:
    """Compute all relations between two objects."""
    relations = []

    # Spatial relations from bounding boxes
    spatial_rels = compute_bbox_relation(obj1.bbox, obj2.bbox)
    for rel_type in spatial_rels:
        relations.append(Relation(
            source_id=obj1.object_id,
            target_id=obj2.object_id,
            relation_type=rel_type,
        ))

    # Color-based relations
    if obj1.colors and obj2.colors:
        if obj1.primary_color == obj2.primary_color:
            relations.append(Relation(
                source_id=obj1.object_id,
                target_id=obj2.object_id,
                relation_type=RelationType.SAME_COLOR,
            ))

    # Size-based relations
    if obj1.pixel_count == obj2.pixel_count:
        relations.append(Relation(
            source_id=obj1.object_id,
            target_id=obj2.object_id,
            relation_type=RelationType.SAME_SIZE,
        ))

    return relations


def build_relational_graph(
    objects: List[GridObject],
    grid: Optional[Grid] = None,
) -> Any:
    """
    Build a directed graph of relations between objects.

    Returns a networkx DiGraph if available, otherwise a dict representation.
    """
    if not HAS_NETWORKX:
        # Fallback to dict representation
        graph: Dict[str, Any] = {
            "nodes": [],
            "edges": [],
        }

        for obj in objects:
            graph["nodes"].append({
                "id": obj.object_id,
                "bbox": (obj.bbox.min_row, obj.bbox.min_col,
                         obj.bbox.max_row, obj.bbox.max_col),
                "colors": list(obj.colors),
                "pixel_count": obj.pixel_count,
            })

        for i, obj1 in enumerate(objects):
            for obj2 in objects[i + 1:]:
                relations = compute_object_relations(obj1, obj2)
                for rel in relations:
                    graph["edges"].append({
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "type": rel.relation_type.name,
                        "strength": rel.strength,
                    })

        return graph

    # NetworkX graph
    G = nx.DiGraph()

    # Add nodes
    for obj in objects:
        G.add_node(
            obj.object_id,
            bbox=obj.bbox,
            colors=list(obj.colors),
            pixel_count=obj.pixel_count,
            primary_color=obj.primary_color,
        )

    # Add edges (relations)
    for i, obj1 in enumerate(objects):
        for obj2 in objects[i + 1:]:
            relations = compute_object_relations(obj1, obj2)
            for rel in relations:
                G.add_edge(
                    rel.source_id,
                    rel.target_id,
                    relation_type=rel.relation_type,
                    strength=rel.strength,
                )
                # Add reverse edge for symmetric relations
                if rel.relation_type in {
                    RelationType.ADJACENT,
                    RelationType.SAME_COLOR,
                    RelationType.SAME_SIZE,
                    RelationType.SAME_SHAPE,
                    RelationType.OVERLAPS,
                }:
                    G.add_edge(
                        rel.target_id,
                        rel.source_id,
                        relation_type=rel.relation_type,
                        strength=rel.strength,
                    )

    return G
