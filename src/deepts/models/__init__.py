"""
Module for deeps neural network models.
"""

from ._base import TimeseriesNeuralNet
from ._seq2seq import Seq2Seq

__all__ = ["TimeseriesNeuralNet", "Seq2Seq"]
