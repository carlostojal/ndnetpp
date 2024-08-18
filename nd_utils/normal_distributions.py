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
import nd_utils.voxelization

def estimate_normal_distributions(points: torch.Tensor, n_desired_dists: int,
                                  n_desired_dists_thres: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate normal distributions from the point coordinates.

    Args:
        points (torch.Tensor): Point coordinates (batch_size, n_points, 3)
        n_desired_dists (int): Desired number of normal distributions

    Returns:
        means: Mean vectors (batch_size, voxels_x, voxels_y, voxels_z, 3)
        covs: Covariance matrices (batch_size, voxels_x, voxels_y, voxels_z, 3, 3)
    """

    # find the point cloud limits and ranges
    min_coords, max_coords, dimensions = nd_utils.voxelization.find_point_cloud_limits(points)

    # calculate the voxel size
    voxel_size, n_voxels = nd_utils.voxelization.calculate_voxel_size(dimensions, n_desired_dists*(1.0+n_desired_dists_thres))

    # create a tensor of means with shape (voxels_x, voxels_y, voxels_z, 3)
    means = torch.zeros(torch.cat(n_voxels, torch.Tensor([3])).int().tolist())

    # create a tensor of covariances with shape (voxels_x, voxels_y, voxels_z, 3, 3)
    covs = torch.zeros(torch.cat(n_voxels, torch.Tensor([3, 3])).int().tolist())

    # create a tensor of sample counts with shape (voxels_x, voxels_y, voxels_z)
    sample_counts = torch.zeros(n_voxels.int().tolist()).int()

    # get the voxel indices for each point with shape (n_points, 3)
    voxel_idxs = nd_utils.voxelization.metric_to_voxel_space(points, voxel_size, n_voxels, min_coords)

    # increment the sample counts for each voxel
    sample_counts = sample_counts.scatter_add(0, voxel_idxs, torch.ones(voxel_idxs.size(0)).int())

    # calculate the means
    means = means.scatter_add(0, voxel_idxs.unsqueeze(1).expand(-1, 3, -1), points.unsqueeze(2).expand(-1, -1, 3))

    # calculate the covariances
    covs = covs.scatter_add(0, voxel_idxs.unsqueeze(1).unsqueeze(1).expand(-1, 3, 3, -1),
                            (points.unsqueeze(2).expand(-1, -1, 3) - means[voxel_idxs].unsqueeze(3)).unsqueeze(3))
    
    # divide the means by the sample counts
    means = means / sample_counts.unsqueeze(3)

    # divide the covariances by the sample counts
    covs = covs / sample_counts.unsqueeze(3).unsqueeze(4)

    return means, covs
