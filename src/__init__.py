"""
Tiny AI - 초경량 인공지능 라이브러리
"""

__version__ = "0.1.0"
__author__ = "Tiny AI Team"

from .perceptron import Perceptron
from .mlp import MLP
from . import activations

__all__ = ['Perceptron', 'MLP', 'activations']
