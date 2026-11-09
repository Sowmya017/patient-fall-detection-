import pickle
import random

import numpy as np
from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        label_path,
        debug=False,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=-1,
        normalization=False,
        use_mmap=True,
    ):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap

        self.load_data()

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.sample_name = self.sample_name[:100]
            self.label = self.label[:100]
            self.data = self.data[:100]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])  # (C, T, V, M)
        label = self.label[index]

        # Your preprocessing already fixed T, centered, and normalized,
        # so keep augmentation minimal for the first run.
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)