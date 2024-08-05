from torch import nn
import nd_utils

class Voxelizer(nn.Module):
    """
    Voxelize and estimate normal distributions of the input point cloud, one per voxel.
    """
    def __init__(self):
        pass

    def forward(self, x):
        return x


class Pruner(nn.Module):
    """
    Prune the normal distributions based on the Kulback-Leibler divergence between the two distributions.
    """
    def __init__(self):
        pass

    def forward(self, x):
        return x
