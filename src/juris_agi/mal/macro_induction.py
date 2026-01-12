"""
Macro induction from successful solutions.

Learns reusable patterns (macros) from solved tasks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter

from ..dsl.ast import (
    ASTNode,
    PrimitiveNode,
    ComposeNode,
    walk_ast,
)
from ..dsl.prettyprint import ast_to_source


@dataclass
class Macro:
    """A learned macro (reusable pattern)."""
    name: str
    pattern: ASTNode
    pattern_source: str
    frequency: int  # How often this pattern appeared
    contexts: List[str]  # Task types where this was useful
    success_rate: float = 1.0

    def __hash__(self) -> int:
        return hash(self.pattern_source)


@dataclass
class MacroMatch:
    """A match of a macro in a program."""
    macro: Macro
    location: int  # Index in AST
    match_score: float


class MacroLibrary:
    """
    Library of learned macros.

    Induces and stores reusable patterns from successful solutions.
    """

    def __init__(self, min_frequency: int = 2):
        self.macros: Dict[str, Macro] = {}
        self.min_frequency = min_frequency

    def add_program(
        self,
        program: ASTNode,
        task_context: str = "",
        success: bool = True,
    ) -> None:
        """
        Add a program to learn patterns from.

        Args:
            program: The successful program
            task_context: Description of task type
            success: Whether this program succeeded
        """
        # Extract sub-patterns
        patterns = self._extract_patterns(program)

        for pattern, source in patterns:
            if source in self.macros:
                # Update existing macro
                macro = self.macros[source]
                macro.frequency += 1
                if task_context and task_context not in macro.contexts:
                    macro.contexts.append(task_context)
                if success:
                    # Update success rate (running average)
                    macro.success_rate = (
                        macro.success_rate * (macro.frequency - 1) + 1.0
                    ) / macro.frequency
            else:
                # Create new macro
                name = f"macro_{len(self.macros)}"
                self.macros[source] = Macro(
                    name=name,
                    pattern=pattern,
                    pattern_source=source,
                    frequency=1,
                    contexts=[task_context] if task_context else [],
                    success_rate=1.0 if success else 0.0,
                )

    def get_frequent_macros(self, top_k: int = 10) -> List[Macro]:
        """Get most frequent macros."""
        macros = [m for m in self.macros.values() if m.frequency >= self.min_frequency]
        macros.sort(key=lambda m: m.frequency * m.success_rate, reverse=True)
        return macros[:top_k]

    def find_matches(
        self,
        program: ASTNode,
    ) -> List[MacroMatch]:
        """Find macro matches in a program."""
        matches = []
        program_source = ast_to_source(program)

        for source, macro in self.macros.items():
            if source in program_source:
                matches.append(MacroMatch(
                    macro=macro,
                    location=program_source.find(source),
                    match_score=macro.frequency * macro.success_rate,
                ))

        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches

    def suggest_macros(
        self,
        context: str = "",
        top_k: int = 5,
    ) -> List[Macro]:
        """
        Suggest macros for a given context.

        Args:
            context: Task context description
            top_k: Number of suggestions

        Returns:
            List of suggested macros
        """
        candidates = []

        for macro in self.macros.values():
            if macro.frequency < self.min_frequency:
                continue

            # Score based on context match and frequency
            context_score = 1.0 if context in macro.contexts else 0.5
            score = macro.frequency * macro.success_rate * context_score
            candidates.append((macro, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in candidates[:top_k]]

    def _extract_patterns(
        self,
        program: ASTNode,
    ) -> List[Tuple[ASTNode, str]]:
        """Extract sub-patterns from a program."""
        patterns = []

        # The full program
        patterns.append((program, ast_to_source(program)))

        # Extract composition subsequences
        if isinstance(program, ComposeNode) and len(program.operations) >= 2:
            for i in range(len(program.operations)):
                for j in range(i + 2, len(program.operations) + 1):
                    subseq = program.operations[i:j]
                    if len(subseq) == 1:
                        sub_pattern = subseq[0]
                    else:
                        sub_pattern = ComposeNode(list(subseq))
                    patterns.append((sub_pattern, ast_to_source(sub_pattern)))

        # Extract individual primitives with their arguments
        for node in walk_ast(program):
            if isinstance(node, PrimitiveNode) and node.args:
                patterns.append((node, ast_to_source(node)))

        return patterns

    def clear(self) -> None:
        """Clear all macros."""
        self.macros.clear()

    def export(self) -> List[Dict[str, Any]]:
        """Export macros as serializable format."""
        return [
            {
                "name": m.name,
                "pattern_source": m.pattern_source,
                "frequency": m.frequency,
                "contexts": m.contexts,
                "success_rate": m.success_rate,
            }
            for m in self.macros.values()
        ]


def induce_macros(
    programs: List[ASTNode],
    min_frequency: int = 2,
) -> MacroLibrary:
    """
    Induce macros from a list of programs.

    Args:
        programs: List of successful programs
        min_frequency: Minimum frequency to keep a macro

    Returns:
        MacroLibrary with induced patterns
    """
    library = MacroLibrary(min_frequency=min_frequency)

    for program in programs:
        library.add_program(program)

    return library


def suggest_program_from_macros(
    macros: List[Macro],
    max_length: int = 3,
) -> List[ASTNode]:
    """
    Generate program candidates from macros.

    Combines macros to form candidate programs.
    """
    candidates = []

    # Single macros as programs
    for macro in macros:
        candidates.append(macro.pattern)

    # Pairs of macros
    if len(macros) >= 2:
        for i, m1 in enumerate(macros[:5]):  # Limit combinations
            for m2 in macros[i+1:5]:
                # Compose macros
                composed = ComposeNode([m1.pattern, m2.pattern])
                candidates.append(composed)

    return candidates


# ============================================================================
# Trace-based Macro Induction
# ============================================================================

@dataclass
class CandidateMacro:
    """A candidate macro extracted from traces."""
    name: str
    code: str
    tags: List[str]
    source_task_ids: List[str]
    frequency: int
    mdl_cost: int
    score: float  # Combined score for ranking

    def to_stored_macro(self):
        """Convert to StoredMacro for storage."""
        from .retrieval import StoredMacro
        return StoredMacro(
            name=self.name,
            code=self.code,
            tags=self.tags,
            mdl_cost=self.mdl_cost,
            usage_count=self.frequency,
            success_count=self.frequency,  # Assume all from successful traces
            created_from_task=self.source_task_ids[0] if self.source_task_ids else None,
        )


class MacroInducer:
    """
    Induces candidate macros from solved traces.

    Extracts common subprograms and proposes them as macros.
    """

    def __init__(self, min_frequency: int = 2, min_length: int = 1, max_length: int = 4):
        """
        Initialize macro inducer.

        Args:
            min_frequency: Minimum occurrences to become a macro
            min_length: Minimum subprogram length (in primitives)
            max_length: Maximum subprogram length
        """
        self.min_frequency = min_frequency
        self.min_length = min_length
        self.max_length = max_length
        self._pattern_counts: Dict[str, List[Dict[str, Any]]] = {}  # code -> [{task_id, tags}]

    def process_trace(self, trace_dict: Dict[str, Any]) -> None:
        """
        Process a single trace to extract patterns.

        Args:
            trace_dict: A solve trace dictionary
        """
        if not trace_dict.get("success"):
            return

        program = trace_dict.get("final_program")
        if not program:
            return

        task_id = trace_dict.get("task_id", "unknown")

        # Extract tags from trace features
        tags = self._extract_tags_from_trace(trace_dict)

        # Extract subprograms
        subprograms = self._extract_subprograms(program)

        for code in subprograms:
            if code not in self._pattern_counts:
                self._pattern_counts[code] = []
            self._pattern_counts[code].append({
                "task_id": task_id,
                "tags": tags,
            })

    def process_traces(self, traces: List[Dict[str, Any]]) -> None:
        """Process multiple traces."""
        for trace in traces:
            self.process_trace(trace)

    def extract_candidates(self) -> List[CandidateMacro]:
        """
        Extract candidate macros from processed traces.

        Returns:
            List of candidate macros sorted by score
        """
        candidates = []
        macro_id = 0

        for code, occurrences in self._pattern_counts.items():
            if len(occurrences) < self.min_frequency:
                continue

            # Aggregate tags and task IDs
            all_tags: List[str] = []
            task_ids: List[str] = []
            for occ in occurrences:
                all_tags.extend(occ["tags"])
                if occ["task_id"] not in task_ids:
                    task_ids.append(occ["task_id"])

            # Deduplicate and sort tags by frequency
            tag_counts = Counter(all_tags)
            top_tags = [t for t, _ in tag_counts.most_common(5)]

            # Compute MDL cost (rough estimate based on code length)
            mdl_cost = self._estimate_mdl(code)

            # Score: frequency * diversity / mdl_cost
            frequency = len(occurrences)
            diversity = len(task_ids)
            score = (frequency * diversity) / max(mdl_cost, 1)

            candidates.append(CandidateMacro(
                name=f"induced_macro_{macro_id}",
                code=code,
                tags=top_tags,
                source_task_ids=task_ids,
                frequency=frequency,
                mdl_cost=mdl_cost,
                score=score,
            ))
            macro_id += 1

        # Sort by score
        candidates.sort(key=lambda m: m.score, reverse=True)
        return candidates

    def _extract_tags_from_trace(self, trace: Dict[str, Any]) -> List[str]:
        """Extract task characteristic tags from trace."""
        tags = []

        # From trace entries
        for entry in trace.get("entries", []):
            if entry.get("event_type") == "task_loaded":
                details = entry.get("details", {})
                input_dims = details.get("input_dims", [])
                output_dims = details.get("output_dims", [])

                if input_dims and output_dims:
                    if input_dims == output_dims:
                        tags.append("same_dims")
                    else:
                        tags.append("different_dims")

                    # Check scaling
                    if len(input_dims) > 0 and len(output_dims) > 0:
                        ih, iw = input_dims[0]
                        oh, ow = output_dims[0]
                        if oh == 2 * ih and ow == 2 * iw:
                            tags.append("scale_2x")
                        if oh < ih or ow < iw:
                            tags.append("cropping")

        # From final metrics
        metrics = trace.get("final_metrics", {})
        if metrics.get("exact_match"):
            tags.append("exact_match")

        return tags

    def _extract_subprograms(self, program_source: str) -> List[str]:
        """
        Extract subprogram patterns from a program source.

        Returns list of subprogram code strings.
        """
        subprograms = []

        # The full program
        subprograms.append(program_source)

        # Try to parse and extract subsequences
        # Simple heuristic: split on composition operator and extract windows
        if " >> " in program_source:
            parts = program_source.split(" >> ")
            for window_size in range(self.min_length, min(self.max_length + 1, len(parts) + 1)):
                for i in range(len(parts) - window_size + 1):
                    subseq = " >> ".join(parts[i:i + window_size])
                    subprograms.append(subseq)

        # Individual primitives with arguments
        import re
        prim_pattern = r'\b(\w+)\s*\([^)]*\)'
        for match in re.finditer(prim_pattern, program_source):
            subprograms.append(match.group(0))

        return subprograms

    def _estimate_mdl(self, code: str) -> int:
        """Estimate MDL cost of a code snippet."""
        # Simple heuristic: count operations
        cost = 1

        # Count compositions
        cost += code.count(" >> ")

        # Count function calls
        cost += code.count("(")

        return cost

    def clear(self) -> None:
        """Clear accumulated patterns."""
        self._pattern_counts.clear()


def extract_candidate_macros(
    traces: List[Dict[str, Any]],
    min_frequency: int = 2,
) -> List[CandidateMacro]:
    """
    Convenience function to extract candidate macros from traces.

    Args:
        traces: List of successful solve traces
        min_frequency: Minimum occurrences for a macro

    Returns:
        List of candidate macros
    """
    inducer = MacroInducer(min_frequency=min_frequency)
    inducer.process_traces(traces)
    return inducer.extract_candidates()
