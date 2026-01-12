"""
Prior knowledge for ARC tasks.

Encodes domain knowledge about typical ARC transformations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum, auto


class TransformationCategory(Enum):
    """Categories of ARC transformations."""
    GEOMETRIC = auto()  # Rotations, reflections, translations
    COLOR = auto()       # Recoloring, palette changes
    STRUCTURAL = auto()  # Object manipulation, cropping
    PATTERN = auto()     # Repetition, tiling
    COUNTING = auto()    # Based on object counts
    CONDITIONAL = auto() # Based on properties


@dataclass
class TransformationPrior:
    """Prior probability for a transformation type."""
    name: str
    category: TransformationCategory
    base_probability: float  # Prior probability of this transform
    features: Dict[str, float] = field(default_factory=dict)  # Feature weights

    def compute_likelihood(self, features: Dict[str, Any]) -> float:
        """Compute likelihood given observed features."""
        likelihood = self.base_probability

        for feat_name, weight in self.features.items():
            if feat_name in features:
                feat_val = features[feat_name]
                if isinstance(feat_val, bool):
                    likelihood *= (weight if feat_val else (1 - weight))
                elif isinstance(feat_val, (int, float)):
                    likelihood *= max(0.01, 1 - abs(feat_val - weight) * 0.1)

        return likelihood


@dataclass
class PriorKnowledge:
    """Collection of prior knowledge about transformations."""
    transformation_priors: List[TransformationPrior]
    feature_weights: Dict[str, float] = field(default_factory=dict)

    def rank_transformations(
        self,
        features: Dict[str, Any],
    ) -> List[tuple[str, float]]:
        """
        Rank transformations by likelihood given features.

        Returns list of (transform_name, likelihood) sorted descending.
        """
        rankings = []
        for prior in self.transformation_priors:
            likelihood = prior.compute_likelihood(features)
            rankings.append((prior.name, likelihood))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


class ARCPriors(PriorKnowledge):
    """
    Default prior knowledge for ARC tasks.

    Based on analysis of common ARC patterns.
    """

    def __init__(self):
        priors = [
            # Geometric transforms are common
            TransformationPrior(
                name="identity",
                category=TransformationCategory.GEOMETRIC,
                base_probability=0.1,
                features={"same_dims": 1.0, "same_palette": 1.0},
            ),
            TransformationPrior(
                name="rotate90",
                category=TransformationCategory.GEOMETRIC,
                base_probability=0.15,
                features={"is_square": 0.8, "same_palette": 0.9},
            ),
            TransformationPrior(
                name="reflect_h",
                category=TransformationCategory.GEOMETRIC,
                base_probability=0.12,
                features={"same_dims": 0.9, "same_palette": 0.9},
            ),
            TransformationPrior(
                name="reflect_v",
                category=TransformationCategory.GEOMETRIC,
                base_probability=0.12,
                features={"same_dims": 0.9, "same_palette": 0.9},
            ),
            TransformationPrior(
                name="transpose",
                category=TransformationCategory.GEOMETRIC,
                base_probability=0.08,
                features={"dims_swapped": 1.0, "same_palette": 0.9},
            ),

            # Structural transforms
            TransformationPrior(
                name="crop_to_content",
                category=TransformationCategory.STRUCTURAL,
                base_probability=0.15,
                features={"smaller_output": 0.9, "same_palette": 0.9},
            ),
            TransformationPrior(
                name="extract_object",
                category=TransformationCategory.STRUCTURAL,
                base_probability=0.10,
                features={"smaller_output": 0.8, "has_objects": 0.9},
            ),

            # Pattern transforms
            TransformationPrior(
                name="tile",
                category=TransformationCategory.PATTERN,
                base_probability=0.08,
                features={"larger_output": 0.9, "dim_multiple": 0.9},
            ),
            TransformationPrior(
                name="scale",
                category=TransformationCategory.PATTERN,
                base_probability=0.10,
                features={"larger_output": 0.8, "dim_multiple": 0.9},
            ),

            # Color transforms
            TransformationPrior(
                name="recolor",
                category=TransformationCategory.COLOR,
                base_probability=0.12,
                features={"same_dims": 0.9, "palette_changed": 0.9},
            ),
            TransformationPrior(
                name="fill",
                category=TransformationCategory.COLOR,
                base_probability=0.05,
                features={"same_dims": 0.8},
            ),
        ]

        super().__init__(
            transformation_priors=priors,
            feature_weights={
                "same_dims": 0.3,
                "same_palette": 0.2,
                "has_objects": 0.15,
                "is_square": 0.1,
            },
        )

    def compute_features(
        self,
        input_grid: Any,  # Grid type
        output_grid: Any,
    ) -> Dict[str, Any]:
        """Compute features for prior ranking."""
        features = {}

        # Dimension features
        features["same_dims"] = input_grid.shape == output_grid.shape
        features["is_square"] = input_grid.height == input_grid.width
        features["dims_swapped"] = (
            input_grid.height == output_grid.width and
            input_grid.width == output_grid.height
        )
        features["smaller_output"] = (
            output_grid.height <= input_grid.height and
            output_grid.width <= input_grid.width and
            not features["same_dims"]
        )
        features["larger_output"] = (
            output_grid.height >= input_grid.height and
            output_grid.width >= input_grid.width and
            not features["same_dims"]
        )

        # Check dimension multiples
        if input_grid.height > 0 and input_grid.width > 0:
            h_ratio = output_grid.height / input_grid.height
            w_ratio = output_grid.width / input_grid.width
            features["dim_multiple"] = (
                h_ratio == int(h_ratio) and w_ratio == int(w_ratio)
            )
        else:
            features["dim_multiple"] = False

        # Palette features
        features["same_palette"] = input_grid.palette == output_grid.palette
        features["palette_changed"] = not features["same_palette"]

        return features


def get_default_priors() -> PriorKnowledge:
    """Get default ARC priors."""
    return ARCPriors()


# ============================================================================
# Proposed Priors and Invariants
# ============================================================================

@dataclass
class ProposedInvariant:
    """An invariant that should be preserved by the transformation."""
    name: str
    confidence: float
    value: Any  # The invariant value (e.g., dims tuple, palette set)
    is_hard: bool = True  # Hard invariants must be satisfied


@dataclass
class TransformFamily:
    """A family of related transformations."""
    name: str
    transforms: List[str]  # List of primitive names
    confidence: float
    evidence: List[str]


@dataclass
class ProposedPriors:
    """Result of propose_priors() analysis."""
    transform_families: List[TransformFamily]
    invariants: List[ProposedInvariant]
    task_features: Dict[str, Any]
    confidence: float

    def get_hard_invariants(self) -> List[ProposedInvariant]:
        """Get invariants that must be satisfied."""
        return [inv for inv in self.invariants if inv.is_hard]

    def get_soft_invariants(self) -> List[ProposedInvariant]:
        """Get invariants that are preferred but not required."""
        return [inv for inv in self.invariants if not inv.is_hard]

    def get_suggested_primitives(self) -> List[str]:
        """Get all suggested primitives from transform families."""
        all_prims: List[str] = []
        for family in self.transform_families:
            all_prims.extend(family.transforms)
        # Deduplicate while preserving order
        seen: Set[str] = set()
        result: List[str] = []
        for p in all_prims:
            if p not in seen:
                seen.add(p)
                result.append(p)
        return result


def propose_priors(
    state: "WorldModelState",
    task: Optional[Any] = None,
) -> ProposedPriors:
    """
    Propose likely transform families and invariants from state features.

    This heuristic analyzes task features to suggest:
    - Transform families: groups of related primitives likely to solve the task
    - Invariants: properties that should be preserved (dims, palette, etc.)

    Args:
        state: WorldModelState with task_features populated
        task: Optional ARCTask for additional analysis

    Returns:
        ProposedPriors with suggested families and invariants
    """
    features = state.task_features
    transform_families: List[TransformFamily] = []
    invariants: List[ProposedInvariant] = []

    # Analyze dimension relationships
    same_dims = features.get("same_dims", False)
    h_ratio = features.get("h_ratio", 1.0)
    w_ratio = features.get("w_ratio", 1.0)

    # Invariant: dimension preservation
    if same_dims:
        invariants.append(ProposedInvariant(
            name="dims_preserved",
            confidence=0.9,
            value=(features.get("input_height", 0), features.get("input_width", 0)),
            is_hard=True,
        ))
        # Suggest geometric transforms that preserve dimensions
        transform_families.append(TransformFamily(
            name="geometric_same_size",
            transforms=["rotate90", "reflect_h", "reflect_v", "transpose", "identity"],
            confidence=0.8,
            evidence=["same_dims detected"],
        ))

    # Analyze scaling/tiling
    if h_ratio == w_ratio and h_ratio > 1 and h_ratio == int(h_ratio):
        scale_factor = int(h_ratio)
        transform_families.append(TransformFamily(
            name="uniform_scaling",
            transforms=["scale", "tile_repeat"],
            confidence=0.85,
            evidence=[f"uniform scale factor {scale_factor}"],
        ))

    # Check for tiling pattern
    if h_ratio > 1 or w_ratio > 1:
        if h_ratio == int(h_ratio) and w_ratio == int(w_ratio):
            transform_families.append(TransformFamily(
                name="tiling",
                transforms=["tile_h", "tile_v", "tile_repeat"],
                confidence=0.7,
                evidence=[f"tiling ratio {int(h_ratio)}x{int(w_ratio)}"],
            ))

    # Check for cropping
    if h_ratio < 1 or w_ratio < 1:
        transform_families.append(TransformFamily(
            name="cropping",
            transforms=["crop_to_content", "crop_to_bbox"],
            confidence=0.75,
            evidence=["output smaller than input"],
        ))

    # Analyze palette relationships
    same_palette = features.get("same_palette", False)

    if same_palette:
        # Palette is a hard invariant if consistently preserved
        invariants.append(ProposedInvariant(
            name="palette_preserved",
            confidence=0.85,
            value=None,  # Will be extracted from actual grids
            is_hard=True,
        ))
    else:
        # Suggest color transforms
        transform_families.append(TransformFamily(
            name="recoloring",
            transforms=["recolor", "recolor_map", "fill_background"],
            confidence=0.6,
            evidence=["palette changed"],
        ))
        # Palette not preserved is a soft invariant (we note it but don't enforce)
        invariants.append(ProposedInvariant(
            name="palette_may_change",
            confidence=0.7,
            value=None,
            is_hard=False,
        ))

    # Check for symmetry patterns in features
    if features.get("is_square", False):
        transform_families.append(TransformFamily(
            name="symmetry",
            transforms=["rotate90", "reflect_h", "reflect_v", "transpose"],
            confidence=0.65,
            evidence=["square input grid"],
        ))

    # Default fallback if no clear pattern
    if not transform_families:
        transform_families.append(TransformFamily(
            name="general",
            transforms=["identity", "crop_to_content", "rotate90", "reflect_h"],
            confidence=0.3,
            evidence=["no clear pattern detected"],
        ))

    # Compute overall confidence
    if transform_families:
        overall_confidence = max(f.confidence for f in transform_families)
    else:
        overall_confidence = 0.3

    return ProposedPriors(
        transform_families=transform_families,
        invariants=invariants,
        task_features=features,
        confidence=overall_confidence,
    )


def extract_invariants_from_task(task: Any) -> List[ProposedInvariant]:
    """
    Extract invariants directly from an ARCTask.

    Analyzes all training pairs to find consistent properties.

    Args:
        task: ARCTask to analyze

    Returns:
        List of proposed invariants
    """
    if not hasattr(task, "train") or not task.train:
        return []

    invariants: List[ProposedInvariant] = []

    # Collect features from all pairs
    all_dims_same = True
    all_palettes_same = True
    output_dims: List[tuple] = []
    output_palettes: List[Set] = []

    for pair in task.train:
        inp, out = pair.input, pair.output

        # Check dimension consistency
        if inp.shape != out.shape:
            all_dims_same = False
        output_dims.append(out.shape)

        # Check palette consistency
        if inp.palette != out.palette:
            all_palettes_same = False
        output_palettes.append(out.palette)

    # Dimension invariant
    if all_dims_same:
        invariants.append(ProposedInvariant(
            name="dims_preserved",
            confidence=0.95,
            value="same_as_input",
            is_hard=True,
        ))
    else:
        # Check if all outputs have same dims (different from input)
        if len(set(output_dims)) == 1:
            invariants.append(ProposedInvariant(
                name="consistent_output_dims",
                confidence=0.9,
                value=output_dims[0],
                is_hard=True,
            ))

    # Palette invariant
    if all_palettes_same:
        invariants.append(ProposedInvariant(
            name="palette_preserved",
            confidence=0.9,
            value="same_as_input",
            is_hard=True,
        ))
    else:
        # Check if output palettes are subsets of combined input palette
        all_palettes_subset = all(
            pair.output.palette.issubset(pair.input.palette | {0})
            for pair in task.train
        )
        if all_palettes_subset:
            invariants.append(ProposedInvariant(
                name="palette_subset",
                confidence=0.8,
                value="subset_of_input",
                is_hard=False,  # Soft constraint
            ))

    return invariants
