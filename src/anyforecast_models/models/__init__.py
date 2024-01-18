"""
Module for deepts neural network models.
"""

from anyforecast_models.models.base._nn import TimeseriesNeuralNet
from anyforecast_models.models.seq2seq import Seq2Seq

__all__ = ["TimeseriesNeuralNet", "Seq2Seq"]
