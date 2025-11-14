"""Preprocessing utilities for the UDA Medical Imbalance Project."""

__all__ = ['ImbalanceHandlerFactory']


def __getattr__(name):
    if name == 'ImbalanceHandlerFactory':
        from .imbalance_handler import ImbalanceHandlerFactory  # Lazy import to avoid heavy deps
        return ImbalanceHandlerFactory
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
