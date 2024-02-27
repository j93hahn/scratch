import numpy as np
from internal import render


class Dataset():
    def __init__(self, path):
        self.path = path

    def load(self):
        return np.load(self.path)


class Blender(Dataset):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

    def load(self):
        return super().load()


class LLFF(Dataset):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

    def load(self):
        return super().load()


dataset_dict = {
    "blender": Blender,
    "llff": LLFF
}
