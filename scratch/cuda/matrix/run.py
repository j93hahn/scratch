"""
There are multiple ways of passing a PyTorch model to a custom CUDA C++ extension. Plenoxels instantiates
a grid class, where all the parameters are stored in PyTorch tensors, but are passed down to the C++ extension
where both the forward and backward passes are computed. Plenoxels hard-codes the gradients into the tensor
itself, instead of using the autograd engine via the loss.backward() call to handle the dynamic linking of
gradients automatically (as done in DvGO).

As such, implementing a Plenoxels-style CUDA C++ kernel is significantly more difficult as it requires implementing
the loss.backward() call in C++ as well; in addition, you must also directly implement the optimizer.step() call to
handle updates to the parameters with the computed gradients from the C++ extension.
"""


import torch
import torch.nn as nn
import utils
from einops import rearrange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_C = utils._get_cuda_extension()


def optimization(model, optimizer, loss_function, epochs=10):
    model.train()
    for epoch in range(1, epochs + 1):
        for data, target in utils.train_loader:
            data = rearrange(data, 'b c h w -> b (c h w)')
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            prediction = model(data)
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()
        inference(model, epoch)


def inference(model, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in utils.test_loader:
            data = rearrange(data, 'b c h w -> b (c h w)')
            data, target = data.to(device), target.to(device)
            prediction = model(data)
            correct += (torch.argmax(prediction, dim=1) == target).sum().item()
    print(f"Epoch {epoch} test accuracy: {correct / len(utils.test_loader.dataset)}")
    model.train()


# from . import load_cuda_kernels
# _C = load_cuda_kernels()    # must be instantiated by loading the CUDA C++ extensions
# class LinearKernel(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weights, bias):
#         output = _C.custom_forward_kernel(input, weights, bias)
#         ctx.save_for_backward(input, weights, bias)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weights, bias = ctx.saved_tensors    # retrieve saved tensors
#         grad_input, grad_weights, grad_bias = _C.custom_backward_kernel(grad_output, input, weights, bias)
#         return grad_input, grad_weights, grad_bias


# class Linear(torch.nn.Module):
#     def __init__(self):
#         super(Linear, self).__init__()
#         self.weights = nn.Parameter(torch.randn(10, 20, dtype=torch.float32))
#         self.bias = nn.Parameter(torch.randn(20, dtype=torch.float32))

#     def forward(self, input):
#         return LinearKernel.apply(input, self.weights, self.bias)


# class MLP(nn.Module):
#     """
#     Multi-layer perceptron class that will be passed to a custom CUDA C++ extension.
#     """
#     def __init__(
#         self,
#         in_dim,
#         out_dim,
#         hidden_dim,
#         num_layers,
#         activation: str = "relu",
#         output_activation: str = "linear",
#         device: Union[torch.device, str] = "cuda",
#     ):
#         super(MLP, self).__init__()
#         self.num_layers = num_layers
#         if self.num_layers == 1:
#             self.layers = nn.ParameterList(
#                 [torch.randn(in_dim, out_dim, dtype=torch.float32).to(device)]
#             )
#         else:
#             self.layers = nn.ParameterList(
#                 [torch.randn(in_dim, hidden_dim, dtype=torch.float32).to(device)] +
#                 [torch.randn(hidden_dim, hidden_dim, dtype=torch.float32).to(device) for _ in range(num_layers-2)] +
#                 [torch.randn(hidden_dim, out_dim, dtype=torch.float32).to(device)]
#             )
#         assert len(self.layers) == self.num_layers

#     def forward(self, x):

#         ...


#     # def _to_cpp(self):

#     #     return MLP_CPP(self.layers)


if __name__ == '__main__':

    model = nn.Sequential(
        nn.Linear(784, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    ).to(device=device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimization(model, optimizer, loss_function)

    # mlp = MLP(3, 3, 256, 8)

    # x = nn.Parameter(torch.randn(10, 20))
    # y = torch.randn(20, 30)
    # truth = torch.ones(10, 30)
    # loss = nn.CrossEntropyLoss()

    # loss = loss(x @ y, truth)
    # loss.backward()
    # breakpoint()
    # print(mlp)
