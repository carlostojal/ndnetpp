"""
MIT License

Copyright (c) 2024 Carlos Tojal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import torch
from torch import nn
import nd_utils

class VoxelizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, num_desired_dists: int, num_desired_dists_thres: float = 0.2):
        """
        Voxelize the input point cloud.

        Args:
            input (torch.Tensor): Input point cloud of shape (N, 3).
            num_desired_dists (int): Number of desired normal distributions.
            num_desired_dists_thres (float): Threshold for the number of desired normal distributions.

        Returns:
            torch.Tensor: Normal distribution means (N1, 3).
            torch.Tensor: Normal distribution covariances (N1, 3, 3).
        """

        # estimate the normal distributions
        voxels = nd_utils.normal_distributions.estimate_normal_distributions(input, num_desired_dists,
                num_desired_dists_thres)

        # TODO: prune the extra normal distributions based on the Kullback-Leibler divergences

        raise NotImplementedError("VoxelizerFunciton.forward is not implemented.")

    @staticmethod
    def backward(ctx, dists_grad: torch.Tensor):
        """
        Voxelization backward pass.

        Args:
            dists_grad (torch.Tensor): gradients of the distributions (voxels) losses (N1).

        Returns:
            torch.Tensor: gradients propagated to the points corresponding to each voxel (N).
        """

        # TODO: distribute the voxel gradients to the corresponding points
        raise NotImplementedError("VoxelizerFunction.backward is not implemented.")

class Voxelizer(nn.Module):
    """
    Voxelize and estimate normal distributions of the input point cloud, one per voxel.
    """
    def __init__(self, num_desired_dists: int, num_desired_dists_thres: float = 0.2):
        self.num_desired_dists = num_desired_dists
        self.num_desired_dists_thres = num_desired_dists_thres

    def forward(self, x):
        # apply the autograd function
        return VoxelizerFunction.apply(x, self.num_desired_dists, self.num_desired_dists_thres)

    def backward(self, x_grad):
        # TODO
        raise NotImplementedError("Voxelizer.backward is not implemented.")

