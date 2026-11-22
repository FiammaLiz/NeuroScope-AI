"""In-memory registry of typed scientific operation definitions."""

from __future__ import annotations

from .contracts import OperationDefinition


class OperationRegistry:
    """Stores operation contracts by stable ID; performs no discovery or execution."""

    def __init__(self) -> None:
        self._operations: dict[str, OperationDefinition] = {}

    def register(self, definition: OperationDefinition) -> None:
        if not isinstance(definition, OperationDefinition):
            raise TypeError("registry entries must be OperationDefinition values.")
        if definition.operation_id in self._operations:
            raise ValueError(f"Operation already registered: {definition.operation_id}")
        self._operations[definition.operation_id] = definition

    def get(
        self,
        operation_id: str,
        *,
        version: str | None = None,
    ) -> OperationDefinition:
        try:
            definition = self._operations[operation_id]
        except KeyError as exc:
            raise KeyError(f"Unknown operation ID: {operation_id}") from exc
        if version is not None and version != definition.version:
            raise ValueError(
                f"Operation {operation_id} version mismatch: requested {version}, "
                f"registered {definition.version}."
            )
        return definition

    def list_operations(self) -> tuple[OperationDefinition, ...]:
        return tuple(self._operations.values())

    def available(self) -> list[OperationDefinition]:
        """Compatibility spelling retained from the original registry scaffold."""
        return list(self._operations.values())


# Preserve the scaffold's registry import while directing it to typed definitions.
AnalysisRegistry = OperationRegistry
