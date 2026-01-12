"""
World Model Expert - provides world understanding and predictions.

This is a stub that can be extended with more sophisticated models.
The WME provides ADVISORY signals only - it has NO JURISDICTION.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ..core.types import Grid, ARCTask, ARCPair
from ..dsl.ast import ASTNode


@dataclass
class WorldModelState:
    """State maintained by the world model."""
    task_features: Dict[str, Any] = field(default_factory=dict)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    predictions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformationHypothesis:
    """A hypothesis about the transformation."""
    name: str
    confidence: float
    evidence: List[str]
    predicted_primitives: List[str]


class WorldModel(ABC):
    """
    Abstract world model interface.

    The world model provides:
    - Feature extraction from tasks
    - Transformation hypotheses
    - Predictions about likely solutions

    IMPORTANT: The world model is ADVISORY ONLY.
    It cannot veto solutions - only the symbolic critic can.
    """

    @abstractmethod
    def analyze_task(self, task: ARCTask) -> WorldModelState:
        """
        Analyze a task and build world model state.

        Returns state with hypotheses and predictions.
        """
        pass

    @abstractmethod
    def generate_hypotheses(
        self,
        task: ARCTask,
        state: Optional[WorldModelState] = None,
    ) -> List[TransformationHypothesis]:
        """
        Generate hypotheses about the transformation.

        Returns ranked list of hypotheses.
        """
        pass

    @abstractmethod
    def score_program(
        self,
        program: ASTNode,
        task: ARCTask,
        state: Optional[WorldModelState] = None,
    ) -> float:
        """
        Score a program based on world model predictions.

        Returns a soft score (0-1), not a hard veto.
        """
        pass


class HeuristicWorldModel(WorldModel):
    """
    Heuristic-based world model.

    Uses simple analysis to generate hypotheses.
    """

    def analyze_task(self, task: ARCTask) -> WorldModelState:
        """Analyze task using heuristics."""
        state = WorldModelState()

        if not task.train:
            return state

        # Extract features from all training pairs
        all_features: List[Dict[str, Any]] = []
        for pair in task.train:
            features = self._extract_pair_features(pair)
            all_features.append(features)

        # Find consistent features
        state.task_features = self._find_consistent_features(all_features)

        # Generate initial hypotheses
        state.hypotheses = [
            {"name": h.name, "confidence": h.confidence}
            for h in self.generate_hypotheses(task, state)
        ]

        # Compute overall confidence
        if state.hypotheses:
            state.confidence = max(h["confidence"] for h in state.hypotheses)
        else:
            state.confidence = 0.1

        return state

    def generate_hypotheses(
        self,
        task: ARCTask,
        state: Optional[WorldModelState] = None,
    ) -> List[TransformationHypothesis]:
        """Generate hypotheses based on task analysis."""
        hypotheses = []

        if not task.train:
            return hypotheses

        # Analyze first pair for quick hypotheses
        pair = task.train[0]
        inp, out = pair.input, pair.output

        # Identity hypothesis
        if inp == out:
            hypotheses.append(TransformationHypothesis(
                name="identity",
                confidence=0.95,
                evidence=["input equals output"],
                predicted_primitives=["identity"],
            ))
            return hypotheses  # Strong evidence, return early

        # Dimension-based hypotheses
        if inp.shape == out.shape:
            hypotheses.append(TransformationHypothesis(
                name="same_size_transform",
                confidence=0.6,
                evidence=["dimensions preserved"],
                predicted_primitives=["rotate90", "reflect_h", "reflect_v", "recolor"],
            ))

        if inp.height == out.width and inp.width == out.height:
            hypotheses.append(TransformationHypothesis(
                name="transpose_or_rotate",
                confidence=0.7,
                evidence=["dimensions swapped"],
                predicted_primitives=["transpose", "rotate90"],
            ))

        if out.height < inp.height or out.width < inp.width:
            hypotheses.append(TransformationHypothesis(
                name="cropping",
                confidence=0.6,
                evidence=["output smaller than input"],
                predicted_primitives=["crop_to_content", "crop_to_bbox"],
            ))

        # Scaling hypothesis
        if inp.height > 0 and inp.width > 0:
            h_ratio = out.height / inp.height
            w_ratio = out.width / inp.width
            if h_ratio == w_ratio and h_ratio == int(h_ratio) and h_ratio > 1:
                hypotheses.append(TransformationHypothesis(
                    name="scaling",
                    confidence=0.8,
                    evidence=[f"uniform scale factor {int(h_ratio)}"],
                    predicted_primitives=["scale"],
                ))

        # Tiling hypothesis
        if out.height % inp.height == 0 and out.width % inp.width == 0:
            h_tiles = out.height // inp.height
            w_tiles = out.width // inp.width
            if h_tiles > 1 or w_tiles > 1:
                hypotheses.append(TransformationHypothesis(
                    name="tiling",
                    confidence=0.7,
                    evidence=[f"tiling factor {h_tiles}x{w_tiles}"],
                    predicted_primitives=["tile_h", "tile_v"],
                ))

        # Color transformation hypothesis
        if inp.shape == out.shape and inp.palette != out.palette:
            hypotheses.append(TransformationHypothesis(
                name="recoloring",
                confidence=0.5,
                evidence=["palette changed"],
                predicted_primitives=["recolor", "recolor_map"],
            ))

        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses

    def score_program(
        self,
        program: ASTNode,
        task: ARCTask,
        state: Optional[WorldModelState] = None,
    ) -> float:
        """Score program based on alignment with hypotheses."""
        if state is None:
            state = self.analyze_task(task)

        if not state.hypotheses:
            return 0.5  # Neutral

        # Check if program primitives align with hypotheses
        program_str = str(program).lower()
        max_alignment = 0.0

        for hyp_dict in state.hypotheses:
            hyp = TransformationHypothesis(
                name=hyp_dict["name"],
                confidence=hyp_dict["confidence"],
                evidence=[],
                predicted_primitives=[],
            )
            # Simple string matching for alignment
            if hyp.name in program_str:
                max_alignment = max(max_alignment, hyp.confidence)

        return max_alignment if max_alignment > 0 else 0.3

    def _extract_pair_features(self, pair: ARCPair) -> Dict[str, Any]:
        """Extract features from an input-output pair."""
        inp, out = pair.input, pair.output
        return {
            "same_dims": inp.shape == out.shape,
            "same_palette": inp.palette == out.palette,
            "input_height": inp.height,
            "input_width": inp.width,
            "output_height": out.height,
            "output_width": out.width,
            "h_ratio": out.height / inp.height if inp.height > 0 else 0,
            "w_ratio": out.width / inp.width if inp.width > 0 else 0,
        }

    def _find_consistent_features(
        self,
        all_features: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Find features that are consistent across all pairs."""
        if not all_features:
            return {}

        consistent = {}
        first = all_features[0]

        for key, value in first.items():
            if all(f.get(key) == value for f in all_features):
                consistent[key] = value

        return consistent


def create_world_model() -> WorldModel:
    """Create default world model instance."""
    return HeuristicWorldModel()
