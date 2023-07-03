from abc import abstractmethod
from typing import List


"""
Optimizer is an abstract class that represents an optimizer.
"""
class Optimizer():
    def __init__(self, params: List[list]):
        self.params = params[0]
        self.gradParams = params[1]
        """
        paramCount is not the actual number of parameters - it just describes
        how many groups of parameters are from each layer (i.e., the weights of a
        linear layer are considered as one parameter even though there might be
        many thousands of parameters for one individual linear layer)
        """
        self.paramCount = len(self.gradParams) * len(self.gradParams[0])

    def zero_grad(self):
        """
        Fill the gradient arrays with 0s - enables optimizer and model
            to refer to same pointers

        The first for-loop iterates through all of the layers with parameters,
            and the second for-loop iterates through all of the parameters
        """
        for i in range(len(self.gradParams)): # layer
            for j in range(len(self.gradParams[i])): # parameter type of each layer
                self.gradParams[i][j].fill(0)

    @abstractmethod
    def step(self):
        """
        Update the parameters based on the gradients.
        """

    def state_dict(self):
        """
        Return the state of the optimizer.
        """
        return self.params, self.gradParams

    def add_param_group(self, param_group):
        """
        Add a group of parameters to the optimizer's parameter list.
        """
        self.params.append(param_group[0])
        self.gradParams.append(param_group[1])



"""
Stochastic Gradient Descent with Momentum
"""
class SGDM(Optimizer):
    def __init__(self, params: List[list], alpha=0.1):
        super().__init__(params)
        self.alpha = alpha

    def step(self):
        for i in range(len(self.params)):
            for j in range(len(self.params[i])):
                self.params[i][j] -= self.alpha * self.gradParams[i][j]
