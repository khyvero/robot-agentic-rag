"""
Zero-Shot Strategy Module

Exports:
- ZeroShotStrategy: Main strategy class
- All strategy-specific prompts
"""

from .strategy import ZeroShotStrategy
from . import prompts

__all__ = ['ZeroShotStrategy', 'prompts']