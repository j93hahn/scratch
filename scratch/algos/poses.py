import numpy as np
import torch
import json
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    print("open3d not found. Some functions may not work.")

class PoseCalculator():
    def __init__(self, K, R, t):
        self.K = K
        self.R = R
        self.t = t


if __name__ == '__main__':
    with open(Path(__file__).parent / 'transforms.json', 'r') as f:
        x = json.load(f)
    breakpoint()
