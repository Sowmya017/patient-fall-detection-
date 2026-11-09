import numpy as np


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = 25
        self.self_link = [(i, i) for i in range(self.num_node)]

        # OpenPose BODY_25 edges, 0-based
        self.inward = [
            (0, 1), (1, 8),
            (2, 1), (3, 2), (4, 3),
            (5, 1), (6, 5), (7, 6),
            (9, 8), (10, 9), (11, 10),
            (12, 8), (13, 12), (14, 13),
            (15, 0), (16, 0), (17, 15), (18, 16),
            (19, 14), (20, 19), (21, 14),
            (22, 11), (23, 22), (24, 11),
        ]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            return get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        raise ValueError(f'Unsupported labeling mode: {labeling_mode}')


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A
