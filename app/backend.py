"""Backward-compatible entrypoint.

The API application now lives in :mod:`app.api`.  This module is kept so that
existing commands (``uvicorn app.backend:app``) keep working.
"""

from app.api import app  # noqa: F401

__all__ = ["app"]
