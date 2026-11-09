import pickle

import numpy as np
from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        label_path,
        debug=False,
        normalization=False,
        use_mmap=True,
        p_interval=1,
        random_choose=False,
        random_shift=False,
        random_move=False,
        random_rot=False,
        window_size=-1,
        bone=False,
        vel=False,
    ):
        self.data_path = data_path
        self.label_path = label_path
        self.debug = debug
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_rot = random_rot
        self.window_size = window_size
        self.bone = bone
        self.vel = vel

        self.load_data()
        if normalization:
            self.get_mean_map()

        # BODY_25 edges (1-based in docs, converted to 0-based here)
        self.bone_pairs = (
            (1, 8), (1, 2), (1, 5), (2, 3), (3, 4),
            (5, 6), (6, 7), (8, 9), (9, 10), (10, 11),
            (8, 12), (12, 13), (13, 14), (1, 0), (0, 15),
            (15, 17), (0, 16), (16, 18), (14, 19), (19, 20),
            (14, 21), (11, 22), (22, 23), (11, 24),
        )

    def load_data(self):
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        if self.debug:
            self.sample_name = self.sample_name[:100]
            self.label = self.label[:100]
            self.data = self.data[:100]

    def get_mean_map(self):
        data = np.array(self.data)
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
        self.std_map[self.std_map == 0] = 1.0

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map

        if self.window_size > 0 and data_numpy.shape[1] != self.window_size:
            data_numpy = self._resize_temporal(data_numpy, self.window_size)

        if self.bone:
            bone_data = np.zeros_like(data_numpy)
            for v1, v2 in self.bone_pairs:
                bone_data[:, :, v1, :] = data_numpy[:, :, v1, :] - data_numpy[:, :, v2, :]
            data_numpy = bone_data

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    @staticmethod
    def _resize_temporal(data_numpy, target_len):
        C, T, V, M = data_numpy.shape
        if T == target_len:
            return data_numpy
        if T <= 1:
            return np.repeat(data_numpy, target_len, axis=1)[:, :target_len]

        src_idx = np.linspace(0, T - 1, T, dtype=np.float32)
        dst_idx = np.linspace(0, T - 1, target_len, dtype=np.float32)
        out = np.zeros((C, target_len, V, M), dtype=data_numpy.dtype)
        for c in range(C):
            for v in range(V):
                for m in range(M):
                    out[c, :, v, m] = np.interp(dst_idx, src_idx, data_numpy[c, :, v, m])
        return out
