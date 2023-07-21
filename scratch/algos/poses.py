import torch
import numpy as np


class PoseCalculator():
    def __init__(self, K, R, t):
        self.K = K
        self.R = R
        self.t = t
