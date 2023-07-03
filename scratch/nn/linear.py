from .module import Module
import numpy as np


"""
Linear is a class that represents a linear layer in a neural network.
"""
class Linear(Module):
    def __init__(self, in_features, out_features, init_method="Gaussian") -> None:
        super(Linear, self).__init__()

        if init_method == "Zero":
            # Zeros initialization
            self.weights = np.zeros((out_features, in_features))
            self.biases = np.zeros(out_features)
        elif init_method == "Random":
            # Random distribution
            self.weights = np.random.randn(out_features, in_features)
            self.biases = np.random.randn(out_features)
        elif init_method == "Gaussian":
            # Gaussian normal distribution
            self.weights = np.random.normal(0, 1 / in_features, (out_features, in_features))
            self.biases = np.random.normal(0, 1, out_features)
        elif init_method == "He":
            # He initialization - https://arxiv.org/pdf/1502.01852.pdf
            self.weights = np.random.normal(0, np.sqrt(2 / in_features), (out_features, in_features))
            self.biases = np.zeros(out_features)
        elif init_method == "Xavier":
            # Xavier initialization - https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            # ideal for linear activation layers; non-ideal for non-linear activation layers (i.e., ReLU)
            self.weights = np.random.uniform(-1/np.sqrt(in_features), 1/np.sqrt(in_features), (out_features, in_features))
            self.biases = np.zeros(out_features)
        elif init_method == "XavierNorm":
            # Normalized Xavier initialization - https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            self.weights = np.random.uniform(-np.sqrt(6)/np.sqrt(in_features+out_features), np.sqrt(6)/np.sqrt(in_features+out_features), (out_features, in_features))
            self.biases = np.zeros(out_features)
        else:
            raise Exception("Initialization technique not recognized.")

        # Gradient descent initialization
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

    def forward(self, _input):
        # input is NxF where F is number of in features, and N is number of data samples
        return np.dot(_input, self.weights.T) + self.biases

    def backward(self, _input, _gradPrev):
        """
        _gradPrev has shape N x out_features

        Assume dL/dY has already been computed, a.k.a. _gradPrev
        dL/dX = dL/dY * W.T
        dL/dW = _input.T * dL/dY
        dL/dB = sum(dL/dY, axis=0)
        """
        self.gradWeights += _gradPrev.T.dot(_input) / _input.shape[0]
        self.gradBiases += np.sum(_gradPrev, axis=0) / _input.shape[0]

        return np.dot(_gradPrev, self.weights)

    def params(self):
        return [self.weights, self.biases], [self.gradWeights, self.gradBiases]

    def name(self):
        return "Linear Layer"


"""
Dropout is a class that represents a dropout layer in a neural network.
"""
class Dropout(Module):
    def __init__(self, p=0.5) -> None:
        """
        This class is implementing "inverse dropout" which can prevent the
        explosion or saturation of neurons. See https://bit.ly/3Ipmg12 for more info

        This is preferred to scaling during test-time
        """
        super().__init__()
        self.p = p # probability of keeping some unit active; higher p = less dropout

    def forward(self, _input):
        _output = _input
        if self.p > 0 and self.train:
            self.mask = np.random.binomial(n=1, p=self.p, size=_input.shape) / self.p
            _output *= self.mask
        return _output

    def backward(self, _input, _gradPrev):
        # scale the backwards pass by the same amount
        _output = _gradPrev
        if self.p > 0 and self.train:
            _output *= self.mask
        return _output

    def params(self):
        return None, None

    def name(self):
        return "Dropout Layer"


"""
BatchNorm1d is a class that represents a batch normalization layer in a neural network.
"""
class BatchNorm1d(Module):
    def __init__(self, channels, eps=1e-5, momentum=0.1) -> None:
        super().__init__()
        """
        Input has dimension (M x C), where M is the batchsize, and C = channels

        To calculate running statistics, use the following formulas:
        E[x] <-- np.mean(mean_beta)
        Var[x] <-- m/(m-1)*np.mean(var_beta)
        """
        self.train_first = True
        self.inf_first = True
        self.eps = eps
        self.momentum = momentum # another method for calculating running averages
        self.count = 0

        # initialize parameters
        self.gamma = np.ones(channels) # PyTorch implementation
        self.beta = np.zeros(channels)
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def forward(self, _input):
        if self.train: # training batch-normalized network
            if self.train_first: # retrieve shape of input from the first mini-batch
                self.m = _input.shape[0] # batch-size
                self.running_mean = np.zeros(_input.shape[1])
                self.running_var = np.zeros(_input.shape[1])
                self.train_first = False

            # calculate mini-batch statistics
            self.mean = np.mean(_input, axis=0)
            self.var = np.mean(np.square(_input - self.mean), axis=0)

            # update moving statistics
            self.running_mean += self.mean
            self.running_var += self.var

            # normalize data, then scale and shift via affine parameters
            self.x_hat = (_input - self.mean) / np.sqrt(self.var + self.eps) # m by C, broadcasted
            y = self.gamma * self.x_hat + self.beta # m by C, broadcasted

            # return output values
            self.count += 1
            return y

        else: # inference stage
            if self.inf_first:
                self.running_mean /= self.count
                self.running_var /= self.count
                self.running_var *= (self.m / (self.m - 1))
                self.count = 0 # training is over
                self.inf_first = False

            y = self.gamma * _input/(np.sqrt(self.running_var + self.eps))
            y += self.beta - ((self.gamma * self.running_mean) / (np.sqrt(self.running_var + self.eps)))
            return y

    def backward(self, _input, _gradPrev):
        """
        All gradient calculations taken directly from Ioffe and Szegedy 2015

        Requires mean, var, x_hat to be passed from the forward pass
        """
        _gradxhat = _gradPrev * self.gamma
        _gradVar = np.sum(_gradxhat * (_input - self.mean) * -1/2*np.power(self.var + self.eps, -3/2), axis=0)
        _gradMean = np.sum(_gradxhat * -1/np.sqrt(self.var + self.eps), axis=0)
        _gradCurr = _gradxhat * 1/np.sqrt(self.var + self.eps) + _gradVar*2*(_input - self.mean)/self.m + _gradMean*1/self.m
        self.gradGamma += np.sum(_gradPrev * self.x_hat, axis=0)
        self.gradBeta += np.sum(_gradPrev, axis=0)
        return _gradCurr

    def params(self):
        return [self.gamma, self.beta], [self.gradGamma, self.gradBeta]

    def name(self):
        return "Batch Normalization 1D Layer"
