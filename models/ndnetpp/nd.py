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
    def forward(ctx, input: torch.Tensor, num_desired_dists: int, voxel_size: float) -> Tuple[torch.Tensor]:
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
        dists, _, min_coords, voxel_size = nd_utils.normal_distributions.estimate_normal_distributions_with_size(input, voxel_size)
        end = time.time()
        print(f"Normal distributions estimation time {dists.device}: {end - start}s - {(end-start)*1000}ms - {1.0 / (end-start)}Hz")

        # randomly sample the input point cloud
        sampled_pcd, sampled_idx = nd_utils.point_clouds.random_sample_point_cloud(input, num_desired_dists)

        # convert the sampled point cloud from metric to voxel space, to the the grid indices
        neighborhood_idxs = nd_utils.voxelization.metric_to_voxel_space(sampled_pcd, voxel_size, num_desired_dists, min_coords)

        # generate the batch indexes
        batch_idxs = torch.arange(input.shape[0]).view(-1, 1).expand(-1, num_desired_dists)

        # get the normal distributions at the indices
        filtered_dists = dists[batch_idxs, neighborhood_idxs[..., 0], neighborhood_idxs[..., 1], neighborhood_idxs[..., 2]]

        # return the filtered normal distributions
        return filtered_dists

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
