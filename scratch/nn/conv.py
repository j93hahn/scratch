from numpy.lib.stride_tricks import sliding_window_view
from einops import rearrange
from .module import Module
import numpy as np


"""
Conv2d is a class that represents a 2D convolutional layer in a neural network.
"""
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1,
                 stride=1, padding=0, pad_mode="zeros", init_method="Uniform") -> None:
        super(Conv2d, self).__init__()

        # initialize channel dimensions
        if in_channels % groups != 0:
            raise Exception("Input channels are not divisible by groups")
        if out_channels % groups != 0:
            raise Exception("Output channels are not divisible by groups")
        self.out_channels = out_channels
        self.feature_count = int(in_channels/groups)
        self.out_spatial_dim = -1

        # initialize parameters
        if init_method == "Zero":
            self.weights = np.zeros((out_channels, self.feature_count, kernel_size, kernel_size))
            self.biases = np.zeros(out_channels)
        elif init_method == "Random":
            self.weights = np.random.randn(out_channels, self.feature_count, kernel_size, kernel_size)
            self.biases = np.random.randn(out_channels)
        elif init_method == "Uniform":
            k = groups/(in_channels * (kernel_size ** 2))
            self.weights = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(out_channels, self.feature_count, kernel_size, kernel_size))
            self.biases = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=out_channels)
        else:
            raise Exception("Initialization technique not recognized")

        # initialize gradients
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

        # initialize hyperparameters - assume padding, kernel size, and stride are integers
        if pad_mode not in ["zeros", "reflect", "symmetric"]:
            raise Exception("Invalid padding mode specified")
        self.pad_mode = "constant" if pad_mode == "zeros" else pad_mode
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, _input):
        # calculate output spatial dimensions
        if self.out_spatial_dim == -1:
            self.out_spatial_dim = np.floor((_input.shape[-1] + 2*self.padding - self.kernel_size)/self.stride + 1).astype(int)
            assert self.out_spatial_dim > 0

        # pad the input if necessary
        _input = Conv2d.pad(_input, self.padding, self.pad_mode) if self.padding > 0 else _input

        # im2col and strides vectorization techniques
        self._inputWindows = sliding_window_view(_input, window_shape=(self.kernel_size, self.kernel_size), axis=(-2, -1))[:, :, ::self.stride, ::self.stride]
        _windows = rearrange(self._inputWindows, 'n c_in h w kh kw -> n h w (c_in kh kw)')
        _weights = rearrange(self.weights, 'c_out c_in kh kw -> c_out (c_in kh kw)')
        _output = np.einsum('n h w q, c q -> n c h w', _windows, _weights) # q is the collapsed dimension
        _biases = rearrange(self.biases, 'c_out -> 1 c_out 1 1')
        return _output + _biases

    def backward(self, _input, _gradPrev):
        # calculate parameter gradients
        self.gradWeights += np.einsum('n i h w k l, n o h w -> o i k l', self._inputWindows, _gradPrev) / _input.shape[0]
        self.gradBiases += np.mean(_gradPrev.sum(axis=(-2, -1)), axis=0)

        # convolve the adjoint with a rotated kernel to produce _gradCurr
        _pad = (self.kernel_size - 1) // 2
        _gradCurr = Conv2d.pad(np.zeros_like(_input), _pad, "constant")
        _rotKernel = np.rot90(self.weights, 2, axes=(-2, -1))

        # each element in _gradPrev corresponds to a square in _gradCurr with length self.kernel_size convolved with the filter
        inds0, inds1 = Conv2d.unroll_img_inds(range(0, _gradCurr.shape[-1] - self.kernel_size + 1, self.stride), self.kernel_size)
        inds2, inds3 = Conv2d.unroll_img_inds(range(0, _gradPrev.shape[-1], self.stride), 1)
        _gradCurr[:, :, inds0, inds1] += np.einsum('n o c d p q, o i k l -> n i c d p q', _gradPrev[:, :, inds2, inds3], _rotKernel)

        if self.padding > 0: # remove padding to match _input shape
            _gradCurr = _gradCurr[:, :, _pad:-_pad, _pad:-_pad]

        return _gradCurr

    def params(self):
        return [self.weights, self.biases], [self.gradWeights, self.gradBiases]

    @staticmethod
    def pad(_input, padding, mode):
        pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
        return np.pad(_input, pad_width=pad_width, mode=mode)

    @staticmethod # code taken from Haochen Wang - https://github.com/w-hc
    def unroll_img_inds(base_hinds, filter_h, base_winds=None, filter_w=None):
        # assume spatial dimensions are identical
        filter_w = filter_h if filter_w is None else filter_w
        base_winds = base_hinds if base_winds is None else base_winds

        outer_h, outer_w, inner_h, inner_w = np.ix_(
            base_hinds, base_winds, range(filter_h), range(filter_w)
        )

        return outer_h + inner_h, outer_w + inner_w

    def name(self):
        return "Conv2d Layer"


"""
Flatten2d is a module that flattens the input to a 2d tensor with shape (batch_size, num_features).
"""
class Flatten2d(Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = True

    def forward(self, _input): # flatten to batch dimension
        return rearrange(_input, 'n c h w -> n (c h w)')

    def backward(self, _input, _gradPrev):
        return _gradPrev.reshape(_input.shape)

    def params(self):
        return None, None

    def name(self):
        return "Flatten2d Layer"


"""
For all pooling layers, input should be (N x C x H x W)

Key Asumptions:
1) H == W, i.e. _input.shape[2] == _input.shape[3]
2) kernel_size and stride are both integers but do not need to be same value
3) kernel_size <= stride
4) No padding - the pooling can only happen over the given input image plane
5) For min or max pooling, if multiple elements have the same minimum or maximum value
within a kernel (very unlikely), pass the gradients to all elements with the min
or max value
"""
class Pooling2d(Module):
    def __init__(self, kernel_size, stride=None, mode="max", return_indices=True) -> None:
        super().__init__()
        self.kernel_size = kernel_size # input must be an integer
        self.stride = kernel_size if stride == None else stride

        # hard-code assumptions here
        if type(self.kernel_size) != type(self.stride):
            raise Exception("Stride type does not match kernel filter type")
        if not isinstance(self.stride, int):
            raise Exception("Invalid stride input type")
        if self.stride < self.kernel_size:
            raise Exception("Kernel size is larger than the stride")

        self.return_indices = return_indices # if true, return index of max value, necessary for MaxUnpool2d

        if mode not in ["max", "min", "avg"]:
            raise Exception("Invalid pooling mode specified")
        self.mode = mode

    def forward(self, _input):
        """
        Example Calculation:

        Note: Number of grids N = H / stride
        Each grid should go from 0+stride*n to stride*(n+1) from n is in range(0, N)

        Method 1: Use np.ix_ and np.stack to construct the output arrays
        Method 2: Use sliding_window_view to construct the output arrays

        Tried and tested using np.all() - both methods produce equivalent results
        and have similar runtime complexities
        """
        if _input.shape[2] % self.stride:
            raise Exception("Invalid stride shape for input size shape")
        if _input.shape[2] != _input.shape[3]:
            raise Exception("Input spatial dimension axes do not match")

        self.h = int(_input.shape[2] / self.stride) # determines output spatial dimension

        # Method 1
        # grids = [np.ix_(np.arange(self.stride*i, self.stride*i+self.kernel_size),
        #                 np.arange(self.stride*i, self.stride*i+self.kernel_size)) for i in range(self.h)]
        # _pooled = [_input[:, :, grids[i][0], [grids[j][1] for j in range(self.h)]] for i in range(self.h)]

        # Method 2
        _windows = sliding_window_view(_input, window_shape=(self.kernel_size, self.kernel_size), axis=(-2, -1))[:, :, ::self.stride, ::self.stride]

        if self.mode == "max":
            # _output = np.stack([hz.max(axis=(-1, -2)) for hz in _pooled], axis=2)
            _output = _windows.max(axis=(-1, -2))
        elif self.mode == "min":
            # _output = np.stack([hz.min(axis=(-1, -2)) for hz in _pooled], axis=2)
            _output = _windows.min(axis=(-1, -2))
        elif self.mode == "avg":
            # _output = np.stack([hz.mean(axis=(-1, -2)) for hz in _pooled], axis=2)
            _output = _windows.mean(axis=(-1, -2))

        if self.return_indices and self.mode in ["max", "min"]:
            self.indices = _output.repeat(self.stride, axis=-1).repeat(self.stride, axis=-2)

        return _output

    def backward(self, _input, _gradPrev):
        if not self.return_indices: # must be True to enable backpropagation
            raise Exception("Module not equipped to handle backwards propagation")
        if _gradPrev.shape[-1] != _gradPrev.shape[-2]:
            raise Exception("Adjoint state has incorrect spatial dimensions")
        if _gradPrev.shape[-1] != self.h:
            raise Exception("Adjoint state does not have matching dimensions with internals of the module")

        y = _gradPrev.repeat(self.stride, axis=-1).repeat(self.stride, axis=-2)

        # if stride > kernel_size, we have to zero out the elements from the gradient
        # which were not included in the kernel but are included in the stride
        if self.kernel_size < self.stride:
            mask = [] # assumes that _input spatial dimensions are squares
            diff = self.stride - self.kernel_size
            for i in range(diff):
                mask.append(np.arange(self.kernel_size+i, _input.shape[-1], self.stride))
            mask = np.concatenate(mask).astype(int).tolist()
            mask = np.ix_(mask, mask)

            y[:, :, mask[0], :] = 0 # zero out rows
            y[:, :, :, mask[1]] = 0 # zero out columns

        if self.mode == "max" or self.mode == "min":
            # apply a mask such that only the maximum or minimum value of each kernel
            # has the gradient passed backwards to it
            _gradCurr = np.equal(_input, self.indices).astype(int) * y
        elif self.mode == "avg":
            # scale gradient down by 1 / (H * W) - each element in the kernel gets
            # an equal proportion of the gradient
            _gradCurr = y / (self.kernel_size ** 2)

        if _gradCurr.shape != _input.shape:
            raise Exception("Current gradient does not match dimensions of input vector")

        return _gradCurr

    def params(self):
        return None, None

    def name(self):
        return self.mode.capitalize() + "Pool2d Layer"
