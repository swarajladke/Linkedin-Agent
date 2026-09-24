"""Custom exception classes for goal compilation and validation."""


class InfeasibleGoalError(Exception):
    """Raised when a goal cannot be scheduled or achieved within given constraints."""

    pass
