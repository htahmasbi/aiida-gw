from __future__ import annotations


class GwError(Exception):
    """Base exception for all aiida-gw errors."""
    pass


class ConfigurationError(GwError):
    """Raised when configuration is invalid or missing."""
    pass


class WorkflowError(GwError):
    """Raised when a workflow execution fails."""

    def __init__(
        self,
        message: str,
        workflow_pk: int | None = None,
        exit_code: int | None = None,
    ) -> None:
        self.workflow_pk = workflow_pk
        self.exit_code = exit_code
        detail_parts = []
        if workflow_pk:
            detail_parts.append(f"workflow_pk={workflow_pk}")
        if exit_code:
            detail_parts.append(f"exit_code={exit_code}")
        detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
        super().__init__(f"{message}{detail}")


class StructureError(GwError):
    """Raised when a structure fails validation."""

    def __init__(self, message: str, structure_pk: int | None = None) -> None:
        self.structure_pk = structure_pk
        detail = f" (structure_pk={structure_pk})" if structure_pk else ""
        super().__init__(f"{message}{detail}")


class CalculationError(GwError):
    """Raised when a CP2K calculation fails."""
    pass
