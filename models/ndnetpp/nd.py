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
import nd_utils.point_clouds

class VoxelizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, num_desired_dists: int, voxel_size: float) -> torch.Tensor:
        """
        Voxelize the input point cloud.

        Args:
            input (torch.Tensor): Input point cloud of shape (batch_size, n_points, 3).
            num_desired_dists (int): Number of desired normal distributions.
            voxel_size (float): Size of the edge of each voxel.

        Returns:
            torch.Tensor: Normal distribution means and covariances (n_desired_dists, 12).
        """

        # estimate the normal distributions
        start = time.time()
        # normal distributions shaped (batch_size, voxels_x, voxels_y, voxels_z, 12)
        dists, _, min_coords, n_voxels = nd_utils.normal_distributions.estimate_normal_distributions_with_size(input, voxel_size)
        end = time.time()
        print(f"Normal distributions estimation time {dists.device}: {end - start}s - {(end-start)*1000}ms - {1.0 / (end-start)}Hz")

        # get the voxel indices of the input point cloud
        voxel_idxs_pcd = nd_utils.voxelization.metric_to_voxel_space(input, voxel_size, n_voxels, min_coords)

        # randomly sample the input point cloud
        sampled_pcd, sampled_idx = nd_utils.point_clouds.random_sample_point_cloud(input, num_desired_dists)

        # convert the sampled point cloud from metric to voxel space, to the the grid indices
        neighborhood_idxs = nd_utils.voxelization.metric_to_voxel_space(sampled_pcd, voxel_size, num_desired_dists, min_coords)

        # generate the batch indexes
        batch_idxs = torch.arange(input.shape[0]).view(-1, 1).expand(-1, num_desired_dists)

        # get the normal distributions at the indices
        filtered_dists = dists[batch_idxs, neighborhood_idxs[..., 0], neighborhood_idxs[..., 1], neighborhood_idxs[..., 2]]

        # save the context
        ctx.save_for_backward(voxel_idxs_pcd, filtered_dists, sampled_idx, neighborhood_idxs)

        # return the filtered normal distributions
        return filtered_dists

    @staticmethod
    def backward(ctx, dists_grad: torch.Tensor):
        """
        Voxelization backward pass.

        Args:
            dists_grad (torch.Tensor): gradients of the distributions (voxels) losses shaped (batch_size, n_dists, 12)

        Returns:
            torch.Tensor: gradients propagated to the points corresponding to each voxel shaped (batch_size, n_points, 3).
        """

        # retrieve the saved tensors
        voxel_idxs_pcd, out_dists, sampled_idx, neighborhood_idxs = ctx.saved_tensors

        # sum the last dimension (normal distribution) of the gradients
        dists_grad = dists_grad.sum(dim=-1)

        # create a mask where each point's voxel index matches the neighborhood index
        mask = (voxel_idxs_pcd.unsqueeze(2) == neighborhood_idxs.unsqueeze(1)).all(dim=-1)

        # broadcast the gradients to the points
        input_grad = torch.zeros_like(voxel_idxs_pcd, dtype=dists_grad.dtype)
        input_grad += (mask.float() * dists_grad.unsqueeze(1)).sum(dim=2).unsqueeze(-1)

        return input_grad, None, None

class Voxelizer(nn.Module):
    """
    Voxelize and estimate normal distributions of the input point cloud, one per voxel.
    """
    def __init__(self, num_desired_dists: int, voxel_size: float):
        super().__init__()
        self.num_desired_dists = num_desired_dists
        self.voxel_size = voxel_size 

    def forward(self, x):
        # apply the autograd function
        return VoxelizerFunction.apply(x, self.num_desired_dists, self.voxel_size)
