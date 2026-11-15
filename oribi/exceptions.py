"""Public exceptions raised by Oribi."""


class OribiError(Exception):
    """Base exception for Oribi errors."""


class InvalidInputError(OribiError, ValueError):
    """Raised when an input cannot be profiled safely."""
