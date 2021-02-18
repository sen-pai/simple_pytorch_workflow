import random
import glob
import os
import math
import itertools
import numpy as np

import torch
from torch.utils.data import Dataset


class SumOfDataset(Dataset):
    """
    The target is the sum of all elements in each data point
    """

    def __init__(self, is_train=True):

        self.train = is_train

        self.all_val = list(range(100))

        self.train_size = 1000
        self.test_size = 1000

    def __len__(self):
        if self.train:
            return self.train_size
        return self.test_size

    def __getitem__(self, i):
        max_val = random.randint(5,100)
        x = np.arange(max_val - 5, max_val)
        y = np.array([max_val + 1])

        return x, y



class DummyDataset(Dataset):
    """
    Only while testing, ignore otherwise
    """

    def __init__(self, is_train=True):

        self.train = is_train

        self.train_size = 1000
        self.test_size = 1000

    def __len__(self):
        if self.train:
            return self.train_size
        return self.test_size

    def __getitem__(self, i):
        max_val = random.randint(5,100)
        x = np.arange(max_val - 5, max_val)
        y = np.array([max_val + 1])

        return torch.tensor(x).float(), torch.tensor(y).float()
