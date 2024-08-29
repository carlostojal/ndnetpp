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
from typing import Tuple
import nd_utils

def estimate_normal_distributions(points: torch.Tensor, n_desired_dists: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate normal distributions from the point coordinates.

    Args:
        points (torch.Tensor): Point coordinates (batch_size, n_points, 3)
        n_desired_dists (int): Desired number of normal distributions


    Returns:
        dists (torch.Tensor): Concatenated mean vectors and covariance matrices (batch_size, voxels_x, voxels_y, voxels_z, 12)
        sample_counts (torch.Tensor): Sample counts for each voxel (batch_size, voxels_x, voxels_y, voxels_z)
        min_coords (torch.Tensor): Point cloud minimum coordinates in each axis.
    """

    # find the point cloud limits and ranges
    min_coords, _, dimensions = nd_utils.voxelization.find_point_cloud_limits(points)

    # calculate the batch-wise voxel size and number of voxels
    voxel_size, n_voxels = nd_utils.voxelization.calculate_voxel_size(dimensions, int(n_desired_dists))

    grid_dim = torch.cat((torch.tensor([points.shape[0]]), n_voxels.cpu())).int()

    # create a tensor of means with shape (batch_size, voxels_x, voxels_y, voxels_z, 3)
    means = torch.zeros(torch.cat((grid_dim, torch.tensor([3]))).int().tolist(), device=points.device)

    # create a tensor of covariances with shape (batch_size, voxels_x, voxels_y, voxels_z, 3, 3)
    covs = torch.zeros(torch.cat((grid_dim, torch.tensor([3, 3]))).int().tolist(), device=points.device)

    # create a tensor of sample counts with shape (batch_size, voxels_x, voxels_y, voxels_z)
    sample_counts = torch.zeros(grid_dim.tolist(), device=points.device).int()

    # get the voxel indices for each point with shape (batch_size, n_points, 3)
    voxel_idxs = nd_utils.voxelization.metric_to_voxel_space(points, voxel_size, n_voxels, min_coords)

    # create indices for increments
    batch_size = points.shape[0]
    n_points = points.shape[1]
    batch_indices = torch.arange(batch_size, device=points.device).view(-1, 1).expand_as(voxel_idxs[..., 0]).to(points.device)
    indices = (batch_indices.flatten(),
               voxel_idxs[..., 0].flatten(),
               voxel_idxs[..., 1].flatten(),
               voxel_idxs[..., 2].flatten())

    # increment the sample_counts for each voxel at the indices enumerated by indices
    inc = torch.ones(n_points*batch_size, device=points.device).int()
    sample_counts.index_put_(indices, inc, accumulate=True)

    # calculate the means
    means.index_put_(indices, points.view(-1, 3), accumulate=True)
    means = means / sample_counts.unsqueeze(-1) # unsqueeze the point dimension

    # TODO: calculate the covariances
    deviations = points - means[indices].view(batch_size, n_points, 3)
    for i in range(3):
        for j in range(3):
            covs[..., i, j].index_put_(indices, (deviations[..., i] * deviations[..., j]).view(-1), accumulate=True)
    covs = covs / sample_counts.unsqueeze(-1).unsqueeze(-1)
    # flatten the covariance matrix
    covs = covs.reshape(*covs.shape[:-2], -1)

    # concatenate the means and covariances along the mean/flattened covariance dimension (last dimension)
    dists = torch.cat((means, covs), dim=-1)

    return dists, sample_counts, min_coords

def prune_normal_distributions(means: torch.Tensor, covs: torch.Tensor, valid_dists: torch.Tensor, n_desired_dists: int) -> torch.Tensor:
    """
    Prune the extra normal distributions based on the Kullback-Leibler divergences.

    Args:
        means (torch.Tensor): Tensor of normal distribution means (batch_size, voxels_x, voxels_y, voxels_z, 3)
        covs (torch.Tensor): Tensor of normal distribution covariances (batch_size, voxels_x, voxels_y, voxels_z, 3, 3)
        n_desired_dists (int): Number of desired normal distributions after the pruning proces.

    Returns:
        valid_dists (torch.Tensor): Tensor of booleans denoting either a normal distribution is valid or not.
    """

    # TODO
    raise NotImplementedError("Kullback-Leibler normal distribution pruning not implemented.")

