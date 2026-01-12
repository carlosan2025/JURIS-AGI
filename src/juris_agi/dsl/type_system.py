"""
Type system for the DSL.

Provides typed primitives to ensure program correctness.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict


class DSLType(ABC):
    """Base class for DSL types."""

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    def is_subtype_of(self, other: "DSLType") -> bool:
        """Check if this type is a subtype of another."""
        return self == other


@dataclass(frozen=True)
class GridType(DSLType):
    """Type for grids."""

    def __str__(self) -> str:
        return "Grid"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GridType)

    def __hash__(self) -> int:
        return hash("Grid")


@dataclass(frozen=True)
class IntType(DSLType):
    """Type for integers."""

    def __str__(self) -> str:
        return "Int"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IntType)

    def __hash__(self) -> int:
        return hash("Int")


@dataclass(frozen=True)
class BoolType(DSLType):
    """Type for booleans."""

    def __str__(self) -> str:
        return "Bool"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BoolType)

    def __hash__(self) -> int:
        return hash("Bool")


@dataclass(frozen=True)
class ColorType(DSLType):
    """Type for colors (0-9)."""

    def __str__(self) -> str:
        return "Color"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ColorType)

    def __hash__(self) -> int:
        return hash("Color")

    def is_subtype_of(self, other: DSLType) -> bool:
        # Color is a subtype of Int
        return isinstance(other, (ColorType, IntType))


@dataclass(frozen=True)
class ListType(DSLType):
    """Type for lists of elements."""
    element_type: DSLType

    def __str__(self) -> str:
        return f"List[{self.element_type}]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ListType) and self.element_type == other.element_type

    def __hash__(self) -> int:
        return hash(("List", self.element_type))


@dataclass(frozen=True)
class ObjectType(DSLType):
    """Type for GridObjects."""

    def __str__(self) -> str:
        return "Object"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObjectType)

    def __hash__(self) -> int:
        return hash("Object")


@dataclass(frozen=True)
class PointType(DSLType):
    """Type for (row, col) points."""

    def __str__(self) -> str:
        return "Point"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PointType)

    def __hash__(self) -> int:
        return hash("Point")


@dataclass(frozen=True)
class BBoxType(DSLType):
    """Type for bounding boxes."""

    def __str__(self) -> str:
        return "BBox"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BBoxType)

    def __hash__(self) -> int:
        return hash("BBox")


@dataclass(frozen=True)
class FunctionType(DSLType):
    """Type for functions."""
    arg_types: Tuple[DSLType, ...]
    return_type: DSLType

    def __str__(self) -> str:
        args = ", ".join(str(t) for t in self.arg_types)
        return f"({args}) -> {self.return_type}"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, FunctionType) and
                self.arg_types == other.arg_types and
                self.return_type == other.return_type)

    def __hash__(self) -> int:
        return hash(("Function", self.arg_types, self.return_type))


@dataclass(frozen=True)
class ColorMapType(DSLType):
    """Type for color mappings."""

    def __str__(self) -> str:
        return "ColorMap"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ColorMapType)

    def __hash__(self) -> int:
        return hash("ColorMap")


class TypeCheckError(Exception):
    """Exception raised when type checking fails."""
    pass


def type_check(
    expected: DSLType,
    actual: DSLType,
    context: str = "",
) -> bool:
    """
    Check if actual type matches expected type.

    Raises TypeCheckError if types don't match.
    """
    if actual.is_subtype_of(expected):
        return True

    msg = f"Type mismatch: expected {expected}, got {actual}"
    if context:
        msg = f"{context}: {msg}"
    raise TypeCheckError(msg)


def infer_literal_type(value: Any) -> DSLType:
    """Infer the DSL type of a Python value."""
    if isinstance(value, bool):
        return BoolType()
    elif isinstance(value, int):
        if 0 <= value <= 9:
            return ColorType()
        return IntType()
    elif isinstance(value, list):
        if not value:
            return ListType(IntType())  # Default to List[Int]
        elem_type = infer_literal_type(value[0])
        return ListType(elem_type)
    elif isinstance(value, dict):
        return ColorMapType()
    elif isinstance(value, tuple) and len(value) == 2:
        return PointType()
    else:
        raise TypeCheckError(f"Cannot infer type for value: {value}")


# Convenience type constructors
GRID = GridType()
INT = IntType()
BOOL = BoolType()
COLOR = ColorType()
OBJECT = ObjectType()
POINT = PointType()
BBOX = BBoxType()
COLOR_MAP = ColorMapType()


def list_of(element_type: DSLType) -> ListType:
    """Create a List type."""
    return ListType(element_type)


def function_type(*args: DSLType, ret: DSLType) -> FunctionType:
    """Create a Function type."""
    return FunctionType(arg_types=tuple(args), return_type=ret)
