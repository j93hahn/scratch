from abc import abstractmethod


"""
Module is an abstract class that represents a neural network layer.
"""
class Module():
    def __init__(self):
        """
        Initialize the module.
        """
        self.train = True

    @abstractmethod
    def forward(self, _input):
        """
        _input is the input to the layer in the Sequential model when applying forward

        return self._output, which is the output of the layer and is passed to the
            next layer in the sequence
        """

    @abstractmethod
    def backward(self, _input, _gradPrev):
        """
        _gradPrev is the gradient/delta of the previous layer in the Sequential
            model when applying backwardagation

        return self._gradCurr multiplies the gradient of the current layer and
            passes it to the next layer in the sequence
        """

    @abstractmethod
    def params(self):
        """
        Return the parameters and their gradients.
        """

    def train(self):
        """
        Set the module in training mode.
        """
        self.train = True

    def eval(self):
        """
        Set the module in evaluation mode.
        """
        self.train = False

    @abstractmethod
    def name(self):
        """
        Return the name of the module.
        """


"""
Container class that holds a sequence of modules.
"""
class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()
        self.layers = [*args]

    def size(self):
        return len(self.layers)

    def components(self):
        for i in range(self.size()):
            print(self.layers[i].name())

    def forward(self, _input):
        self._inputs = [_input]
        for i in range(self.size()):
            self._inputs.append(self.layers[i].forward(self._inputs[i]))
        self._output = self._inputs[-1]
        return self._output

    def backward(self, _input, _gradPrev):
        self._gradPrevArray = [0] * (self.size() + 1)
        self._gradPrevArray[self.size()] = _gradPrev
        for i in reversed(range(self.size())):
            # think about the adjoint state
            self._gradPrevArray[i] = self.layers[i].backward(self._inputs[i], self._gradPrevArray[i + 1])
        self._adjoint = self._gradPrevArray[0]
        return self._adjoint

    def params(self):
        params = []
        gradParams = []
        for layer in self.layers:
            _p, _g = layer.params()
            if _p is not None:
                params.append(_p)
                gradParams.append(_g)
        return params, gradParams

    def train(self):
        self.train()
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.eval()
        for layer in self.layers:
            layer.eval()
