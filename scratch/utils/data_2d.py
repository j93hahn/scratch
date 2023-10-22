"""
A collection of generated 2D datasets, originally used for experimenting
with diffusion models. Repurposed for training a 2D SDF.
"""


import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


class BentRing():
    @staticmethod
    def deform(zs):
        x1 = np.sin(zs)
        x2 = np.cos(zs)
        x3 = x2 ** 2
        data = np.stack([x1, x2, x3], axis=-1)
        return data

    @classmethod
    def sample(cls, n):
        zs = 2 * math.pi * np.random.beta(3, 1, size=n)
        xs = cls.deform(zs)
        return xs


class SwissRoll():
    @staticmethod
    def sample(n):
        data = sklearn.datasets.make_swiss_roll(n_samples=n, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        assert data.shape == (n, 2)
        return data


class Circles():
    @staticmethod
    def sample(n):
        xs, ys = sklearn.datasets.make_circles(
            n_samples=n, factor=0.5, noise=0.08
        )
        xs = xs.astype("float32")
        xs *= 3
        assert xs.shape == (n, 2)
        return xs


class Rings():
    @staticmethod
    def sample(n):
        n_samples4 = n_samples3 = n_samples2 = n // 4
        n_samples1 = n - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=None)

        # Add noise
        X = X + np.random.normal(scale=0.08, size=X.shape)
        X = X.astype("float32")
        return X


class Moons():
    @staticmethod
    def sample(n):
        data = sklearn.datasets.make_moons(n_samples=n, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data


class EightGaussians():
    @staticmethod
    def sample(n):
        scale = 4.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(n):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset


class PinWheels():
    @staticmethod
    def sample(n):
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = n // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))


class TwoSpirals():
    """this is a symmetric two spiral"""
    @staticmethod
    def sample(n):
        n = n // 2  # only sample half; the other half is just mirrored by rotation
        num_revolutions = 1.5
        zs = np.random.rand(n)
        zs = np.sqrt(zs)  # denser towards 1
        zs = zs * (2 * np.pi) * num_revolutions
        d1x = zs * -np.cos(zs) + np.random.rand(n) * 0.5
        d1y = zs * np.sin(zs) + np.random.rand(n) * 0.5
        half = np.stack([d1x, d1y], -1)
        xs = np.concatenate([half, -half], axis=0)
        xs = xs / 3
        xs += np.random.randn(*xs.shape) * 0.1
        return xs


class CheckerBoard():
    @staticmethod
    def sample(n):
        x1 = np.random.rand(n) * 4 - 2
        x2_ = np.random.rand(n) - np.random.randint(0, 2, n) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2


class Line():
    @staticmethod
    def sample(n):
        x = np.random.rand(n) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)


class Cos():
    @staticmethod
    def sample(n):
        x = np.random.rand(n) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)


_ALL_SAMPLERS = [
    BentRing, SwissRoll, Circles,
    Rings, Moons, EightGaussians,
    PinWheels, TwoSpirals,
    CheckerBoard, Line, Cos,
]


def simple_2d_show(x, name):
    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1])   # only consider the first two spatial dimensions
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.set_yticks(np.arange(-4, 5, 1))
    ax.set_title(name)
    plt.savefig(f"plots/{name}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)


def display():
    for sample_cls in _ALL_SAMPLERS:
        sampler = sample_cls()
        xs = sampler.sample(1000)
        simple_2d_show(xs, sample_cls.__name__)


if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    display()
