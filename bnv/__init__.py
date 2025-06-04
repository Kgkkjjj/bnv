"""BNV: A simple text and code generation model."""

__all__ = [
    "train",
    "generate",
    "capabilities",
]

from .train import train
from .generate import generate
from . import capabilities
