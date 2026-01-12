"""MAL (Memory & Abstraction Library) module."""

from .retrieval import (
    MemoryStore,
    SolutionMemory,
    retrieve_similar,
    # New JSON-based macro storage
    MacroStore,
    StoredMacro,
    retrieve_macros,
    extract_task_tags,
)
from .macro_induction import (
    MacroLibrary,
    Macro,
    induce_macros,
    # Trace-based induction
    MacroInducer,
    CandidateMacro,
    extract_candidate_macros,
)
from .gating import (
    GatingMechanism,
    GatingDecision,
    # MDL-based macro acceptance
    MacroGate,
    MacroAcceptanceResult,
    accept_macro,
)

__all__ = [
    # Memory and retrieval
    "MemoryStore",
    "SolutionMemory",
    "retrieve_similar",
    "MacroStore",
    "StoredMacro",
    "retrieve_macros",
    "extract_task_tags",
    # Macro induction
    "MacroLibrary",
    "Macro",
    "induce_macros",
    "MacroInducer",
    "CandidateMacro",
    "extract_candidate_macros",
    # Gating
    "GatingMechanism",
    "GatingDecision",
    "MacroGate",
    "MacroAcceptanceResult",
    "accept_macro",
]
