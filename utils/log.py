import numpy as np
from typing import List

class Log():
    def __init__(self, k_new: np.ndarray, weight: List[float], loss_train: float, duration: float):
        self.k_new = k_new
        self.weight = weight
        self.loss_train = loss_train
        self.duration = duration