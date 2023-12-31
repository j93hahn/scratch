from torch.utils.cpp_extension import load
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os.path as osp
import torch
import unittest


def _get_cuda_extension():
    parent_dir = osp.dirname(osp.abspath(__file__))
    return load(
        name='utils_cuda',
        sources=[
            osp.join(parent_dir, "include", file)
            for file in ['math.cpp', 'math_kernel.cu', 'module.cpp', 'module_kernel.cu',
                         'loss.cpp', 'loss_kernel.cu', 'optim.cpp', 'optim_kernel.cu']
        ],
        verbose=True
    )


train_loader = DataLoader(
    datasets.MNIST(
        root='./data',
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    ),
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST(
        root='./data',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])),
    batch_size=64,
    shuffle=False
)


assert(torch.cuda.is_available(), 'CUDA not available')
device = torch.device('cuda')


class TestMatrix(unittest.TestCase):
    def test_sum(self):
        for _ in range(100):
            n = torch.randint(1, 100, size=()).item()
            a = torch.randn(n*n, dtype=torch.float32).reshape(n, n).to(device)
            b = torch.randn_like(a).to(device)

            correct = a + b
            test = _C.sum(a, b)
            assert torch.allclose(correct, test), 'CUDA kernel failed'
            torch.cuda.empty_cache()

        print('>>>> CUDA kernel passed all matrix summation tests <<<<')

    def test_matmul(self):
        for _ in range(100):
            n1 = torch.randint(1, 100, size=()).item()
            n2 = torch.randint(1, 100, size=()).item()
            n3 = torch.randint(1, 100, size=()).item()
            a = torch.randn(n1*n2, dtype=torch.float32).reshape(n1, n2).to(device)
            b = torch.randn(n2*n3, dtype=torch.float32).reshape(n2, n3).to(device)

            correct = torch.matmul(a, b)
            test = _C.matmul(a, b)
            assert torch.allclose(correct, test, atol=1e-5), 'CUDA kernel failed'   # atol=1e-5 for numerical stability on FP32
            torch.cuda.empty_cache()

        print('>>>> CUDA kernel passed all matrix multiplication tests <<<<')

    def test_linear(self):
        layer = torch.nn.Linear(100, 20).to(device)
        for _ in range(100):
            # generate random matrices
            n1 = torch.randint(1, 100, size=()).item()
            n2 = torch.randint(1, 100, size=()).item()
            n3 = torch.randint(1, 100, size=()).item()
            a = torch.randn(n1*n2, dtype=torch.float32).reshape(n1, n2).to(device)
            b = torch.randn(n2*n3, dtype=torch.float32).reshape(n2, n3).to(device)

            # compute correct answer and pass assertion
            correct = torch.matmul(a, b)
            test = _C.linear(a, b)
            assert torch.allclose(correct, test, atol=1e-5), 'CUDA kernel failed'


if __name__ == '__main__':
    unittest.main()
