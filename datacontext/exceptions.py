"""Public exceptions raised by DataContext."""


class DataContextError(Exception):
    """Base exception for DataContext errors."""


class InvalidInputError(DataContextError, ValueError):
    """Raised when an input cannot be profiled safely."""
