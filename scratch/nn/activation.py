from .module import Module
import numpy as np


"""
ReLU is a class that represents a rectified linear unit activation function.
"""
class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        # _input has shape NxF, self._mask has shape NxF
        self._mask = _input > 0
        return np.maximum(0, _input)

    def backward(self, _input, _gradPrev):
        # input and output vectors have same dimension
        return _gradPrev * self._mask

    def params(self):
        return None, None

    def name(self):
        return "ReLU Activation"


"""
Sigmoid is a class that represents a sigmoid activation function.
"""
class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        _output = 1. / (1 + np.exp(_input))
        self._mask = _output * (1 - _output)
        return _output

    def backward(self, _gradPrev):
        # calculate derivative on output vector space
        return _gradPrev * self._mask

    def name(self):
        return "Sigmoid Activation"