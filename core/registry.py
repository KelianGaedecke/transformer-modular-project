"""
core/registry.py — Component Registry

A dead-simple registry: a dict of dicts mapping string names to classes.

Usage:
    # Register a class (usually done with @register decorator):
    @register('pe', 'rope')
    class RoPEEmbedding(PositionalEmbedding): ...

    # Look up and instantiate:
    cls = get('pe', 'rope')
    instance = cls(config)

    # List all registered names:
    list_registered('pe')  → ['sinusoidal', 'learned', 'rope', 'alibi']

This is the same pattern used by:
  - HuggingFace: AutoModel, AutoTokenizer
  - PyTorch Lightning: LightningModule registry
  - detectron2: component registry
"""

from typing import Type, Dict, Any

# The actual store: category -> {name -> class}
_REGISTRY: Dict[str, Dict[str, Type]] = {}


def register(category: str, name: str):
    """
    Decorator to register a class under a category and name.

    Example:
        @register('pe', 'rope')
        class RoPEEmbedding(PositionalEmbedding):
            ...
    """
    def decorator(cls: Type) -> Type:
        if category not in _REGISTRY:
            _REGISTRY[category] = {}
        if name in _REGISTRY[category]:
            raise ValueError(
                f"Name '{name}' already registered under '{category}'. "
                f"Existing: {_REGISTRY[category][name].__name__}, "
                f"New: {cls.__name__}"
            )
        _REGISTRY[category][name] = cls
        return cls  # return unchanged — decorator is transparent
    return decorator


def get(category: str, name: str) -> Type:
    """
    Retrieve a class by category and name.

    Raises a helpful error if not found (lists valid options).
    """
    if category not in _REGISTRY:
        raise KeyError(f"Unknown category '{category}'. Available: {list(_REGISTRY.keys())}")
    if name not in _REGISTRY[category]:
        valid = list(_REGISTRY[category].keys())
        raise KeyError(
            f"Unknown {category} strategy '{name}'.\n"
            f"  Valid options: {valid}\n"
            f"  To add a new one: @register('{category}', '{name}')"
        )
    return _REGISTRY[category][name]


def build(category: str, name: str, *args, **kwargs) -> Any:
    """Shortcut: look up class and instantiate it."""
    return get(category, name)(*args, **kwargs)


def list_registered(category: str = None) -> Dict | list:
    """List all registered names, optionally filtered by category."""
    if category:
        return list(_REGISTRY.get(category, {}).keys())
    return {cat: list(names.keys()) for cat, names in _REGISTRY.items()}


def is_registered(category: str, name: str) -> bool:
    return category in _REGISTRY and name in _REGISTRY[category]