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
import nd_utils.voxelization


def estimate_normal_distributions_with_numel(points: torch.Tensor,
                                             n_elements: int,
                                             estimate_covariances: bool = True,
                                             mean_dims: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate normal distributions within a grid with a given number of elements from the point coordinates.

    Args:
        points (torch.Tensor): Point coordinates (optionally with covariances and feature vectors) (batch_size, n_points, d)
        n_elements (int): Desired number of grid elements. Will be rounded up.
        estimate_covariances (bool): Whether to estimate the covariances or not. Default: True.
        mean_dims (int): Number of dimensions to consider for mean calculation. Default: 3. -1 to consider all.


    Returns:
        dists (torch.Tensor): Concatenated mean vectors and covariance matrices (batch_size, voxels_x, voxels_y, voxels_z, 12)
        sample_counts (torch.Tensor): Sample counts for each voxel (batch_size, voxels_x, voxels_y, voxels_z)
        min_coords (torch.Tensor): Point cloud minimum coordinates in each axis.
        voxel_size (float): Generated voxel size.
    """

    # find the point cloud limits and ranges
    # ensure to only use the xyz coordinates (first 3 channels)
    min_coords, _, dimensions = nd_utils.voxelization.find_point_cloud_limits(points[:, :, :3])

    # calculate the batch-wise voxel size and number of voxels
    voxel_size, n_voxels = nd_utils.voxelization.calculate_voxel_size(dimensions, int(n_elements))

    # estimate the normal distribution grid
    dists, sample_counts = estimate_grid(points, min_coords, n_voxels, voxel_size,
                                         estimate_covariances, mean_dims)

    return dists, sample_counts, min_coords, voxel_size


def estimate_normal_distributions_with_size(points: torch.Tensor,
                                            voxel_size: float,
                                            estimate_covariances: bool = True,
                                            mean_dims: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate normal distributions with a given voxel size from the point coordinates.

    Args:
        points (torch.Tensor): Point coordinates (batch_size, n_points, 3)
        voxel_size (int): Size of the edge of each voxel.
        estimate_covariances (bool): Whether to estimate the covariances or not. Default: True
        mean_dims (int): Number of dimensions to consider for mean calculation. Default: 3. -1 to consider all.

    Returns:
        dists (torch.Tensor): Concatenated mean vectors and covariance matrices (batch_size, voxels_x, voxels_y, voxels_z, 12)
        sample_counts (torch.Tensor): Sample counts for each voxel (batch_size, voxels_x, voxels_y, voxels_z)
        min_coords (torch.Tensor): Point cloud minimum coordinates in each axis
        n_voxels (torch.Tensor): Number of voxels in each dimension
    """

    # find point cloud limits and dimensions
    min_coords, _, dimensions = nd_utils.voxelization.find_point_cloud_limits(points[:, :, 3])

    # calculate the number of voxels in each dimension
    n_voxels = nd_utils.voxelization.calculate_num_voxels(dimensions, voxel_size)

    # estimate the normal distribution grid
    dists, sample_counts = estimate_grid(points, min_coords, n_voxels, voxel_size, estimate_covariances, mean_dims)

    return dists, sample_counts, min_coords, n_voxels


def estimate_grid(points: torch.Tensor, min_coords: torch.Tensor,
                  n_voxels: torch.Tensor, voxel_size: float,
                  estimate_covariances: bool = True,
                  mean_dims: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate the normal distributions grid from the points and all grid information.

    Args:
        points (torch.Tensor): Point coordinates (batch_size, n_points, d)
        min_coords (torch.Tensor): Minimum point coordinates of the point cloud in each axis
        n_voxels (torch.Tensor): Number of voxels in each dimension
        voxel_size (float): Size of each voxel edge
        estimate_covariances (bool): Estimate the covariances or not. Default: True
        mean_dims (int): Number of dimensions to consider for mean calculation. (Eg: 3 for xyz). If -1, consider all point dimensions. Default: 3

    Returns:
        dists (torch.Tensor): Concatenated mean vectors and flattened covariance matrices (batch_size, voxels_x, voxels_y, voxels_x, 12)
        sample_counts (torch.Tensor): Sample counts for each voxel (batch_size, voxels_x, voxels_y, voxels_z)
    """

    if mean_dims == -1:
        mean_dims = points.size(2)
    elif mean_dims <= 0:
        raise ValueError("Invalid number of dimensions. Must be > 0 or ==-1.")

    # build the grid dimension with the batch dimension
    grid_dim = torch.cat((torch.tensor([points.shape[0]]), n_voxels.cpu())).int()

    # create a tensor of means with shape (batch_size, voxels_x, voxels_y, voxels_z, 3)
    means = torch.zeros(torch.cat((grid_dim, torch.tensor([mean_dims]))).int().tolist(), device=points.device)

    if estimate_covariances:
        # create a tensor of covariances with shape (batch_size, voxels_x, voxels_y, voxels_z, 3, 3)
        covs = torch.zeros(torch.cat((grid_dim, torch.tensor([mean_dims, mean_dims]))).int().tolist(), device=points.device)

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
    means.index_put_(indices, points.view(-1, mean_dims), accumulate=True)
    means = means / sample_counts.unsqueeze(-1)  # unsqueeze the point dimension

    dists = means

    if estimate_covariances:
        #  calculate the covariances
        deviations = points - means[indices].view(batch_size, n_points, mean_dims)
        for i in range(mean_dims):
            for j in range(mean_dims):
                covs[..., i, j].index_put_(indices, (deviations[..., i] * deviations[..., j]).view(-1), accumulate=True)
        covs = covs / sample_counts.unsqueeze(-1).unsqueeze(-1)
        # flatten the covariance matrix
        covs = covs.reshape(*covs.shape[:-2], -1)

    # concatenate the means and covariances along the mean/flattened covariance dimension (last dimension)
    dists = torch.cat((dists, covs), dim=-1)

    return dists, sample_counts

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

