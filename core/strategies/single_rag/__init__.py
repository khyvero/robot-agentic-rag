"""
Single RAG Strategy Module

Exports:
- SingleRAGStrategy: Main strategy class
- All strategy-specific prompts
"""

from .strategy import SingleRAGStrategy
from . import prompts

__all__ = ['SingleRAGStrategy', 'prompts']