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
from typing import Tuple
import time
import nd_utils.voxelization
import nd_utils.normal_distributions

class VoxelizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, num_desired_dists: int, num_desired_dists_thres: float = 0.2) -> Tuple[torch.Tensor]:
        """
        Voxelize the input point cloud.

        Args:
            input (torch.Tensor): Input point cloud of shape (batch_size, n_points, 3).
            num_desired_dists (int): Number of desired normal distributions.
            num_desired_dists_thres (float): Threshold for the number of desired normal distributions.

        Returns:
            torch.Tensor: Normal distribution means and covariances (n_desired_dists, 12).
        """

        # estimate the normal distributions
        start = time.time()
        # normal distributions shaped (batch_size, voxels_x, voxels_y, voxels_z, 12)
        dists, sample_counts = nd_utils.normal_distributions.estimate_normal_distributions(input, num_desired_dists, num_desired_dists_thres)
        end = time.time()
        print(f"Normal distributions estimation time {dists.device}: {end - start}s - {(end-start)*1000}ms - {1.0 / (end-start)}Hz")

        # get the batch size
        batch_size = input.shape[0]

        # create a tensor for the clean normal distributions shaped (batch_size, n_desired_dists, 12)
        clean_dists = torch.empty((batch_size, num_desired_dists, 12), device=input.device)

        # remove random normal distributions until the desired number is reached in each batch
        for b in range(batch_size):
            # TODO: review this
            batch_dists = dists[b]
            batch_sample_counts = sample_counts[b]
            valid_dists = batch_dists[batch_sample_counts > 1]
            n_dists = valid_dists.shape[0]
            indices_to_keep = torch.randperm(n_dists, device=input.device)[:num_desired_dists]
            valid_dists = valid_dists[indices_to_keep]
            clean_dists[b] = valid_dists.view(-1, 12)

        # TODO: save the context needed for the backward pass

        # return the filtered normal distributions
        return clean_dists

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
        super().__init__()
        self.num_desired_dists = num_desired_dists
        self.num_desired_dists_thres = num_desired_dists_thres

    def forward(self, x):
        # apply the autograd function
        return VoxelizerFunction.apply(x, self.num_desired_dists, self.num_desired_dists_thres)
