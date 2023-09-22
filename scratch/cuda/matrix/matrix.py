from torch.utils.cpp_extension import load
import os.path as osp
import torch
import unittest


assert(torch.cuda.is_available())
device = torch.device('cuda:0')


# load the CUDA kernel - inspiration from DVGO codebase
def load_matrix():
    parent_dir = osp.dirname(osp.abspath(__file__))
    return load(
        name='matrix',
        sources=[
            osp.join(parent_dir, file)
            for file in ['matrix.cpp', 'matrix_kernel.cu']
        ],
        verbose=True
    )


class TestMatrix(unittest.TestCase):
    def test_sum(self):
        matrix = load_matrix()
        for _ in range(100):
            # generate random matrices
            n = torch.randint(1, 100, size=()).item()
            a = torch.randn(n*n, dtype=torch.float32).reshape(n, n).to(device)
            b = torch.randn_like(a).to(device)

            # compute correct answer and pass assertion
            correct = a + b
            test = matrix.sum(a, b)
            assert torch.allclose(correct, test), 'CUDA kernel failed'
            torch.cuda.empty_cache()

        print('>>>> CUDA kernel passed all matrix summation tests <<<<')

    def test_matmul(self):
        matrix = load_matrix()
        for _ in range(100):
            # generate random matrices
            n1 = torch.randint(1, 100, size=()).item()
            n2 = torch.randint(1, 100, size=()).item()
            n3 = torch.randint(1, 100, size=()).item()
            a = torch.randn(n1*n2, dtype=torch.float32).reshape(n1, n2).to(device)
            b = torch.randn(n2*n3, dtype=torch.float32).reshape(n2, n3).to(device)

            # compute correct answer and pass assertion
            correct = torch.matmul(a, b)
            test = matrix.matmul(a, b)
            breakpoint()
            assert torch.allclose(correct, test), 'CUDA kernel failed'
            torch.cuda.empty_cache()

        print('>>>> CUDA kernel passed all matrix multiplication tests <<<<')


if __name__ == '__main__':
    unittest.main()
