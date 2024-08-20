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

def find_point_cloud_limits(point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the limits of the point cloud coordinates in each axis.

    Args:
        point_cloud (torch.Tensor): Point cloud data (batch_size, n_points, 3)

    Returns:
        torch.Tensor: minimum limits (batch_size, 3)
        torch.Tensor: maximum limits (batch_size, 3)
        torch.Tensor: dimension in each axis (batch_size, 3)
    """

    # find the limits
    max_coords = torch.max(point_cloud, dim=1).values
    min_coords = torch.min(point_cloud, dim=1).values

    # calculate the dimensions in each axis (max - min)
    dimensions = max_coords - min_coords

    return min_coords, max_coords, dimensions

def calculate_voxel_size(dimensions: torch.Tensor, n_desired_voxels: int) -> Tuple[float, torch.Tensor]:
    """
    Calculate the voxel size considering the point cloud characteristics and desired number of voxels batch-wise
    (don't confuse with normal distributions. voxels without samples don't count as normal distributions).
    The smallest voxel size of the batch is used for all batch samples to ensure compatability.

    Args:
        dimensions (torch.Tensor): Dimensions in each axis (batch_size, 3)
        n_desired_voxels (int): Desired number of voxels

    Returns:
        voxel_size (float): Calculated voxel size
        n_voxels (torch.Tensor): Number of voxels in each dimension (3)
    """

    # find a global bounding box (capable of containing all point clouds within)
    dimensions_global = dimensions.max(dim=0).values

    # calculate the voxel size (batch_size) (calculate the volume along the 3 dimensions and then calculate the cube root)
    voxel_size = torch.pow(torch.prod(dimensions_global) / n_desired_voxels, 1.0/3.0).item()

    # calculate the number of voxels in each dimension. the voxel_size is reshaped to (batch_size, 1) to allow broadcasting
    n_voxels = torch.ceil(dimensions_global / voxel_size).int()

    return voxel_size, n_voxels

def metric_to_voxel_space(points: torch.Tensor, voxel_size: torch.Tensor, n_voxels: torch.Tensor,
                          min_coords: torch.Tensor) -> torch.Tensor:
    """
    Map a point in metric space to voxel space.

    Args:
        points (torch.Tensor): Coordinates of the points (batch_size, n_points, 3)
        voxel_size (torch.Tensor): Voxel sizes (batch_size)
        n_voxels (torch.Tensor): Number of voxels in each dimension (batch_size, 3)
        min_coords (torch.Tensor): Minimum coordinate values in metric space (batch_size, 3)

    Returns:
        voxel_idx (torch.Tensor): Voxel indices in each dimension (batch_size, n_points, 3)
    """

    voxel_idx = torch.floor((points - min_coords) / voxel_size)

    # check out-of-bounds indices
    out_of_bounds = (voxel_idx < 0) | (voxel_idx >= n_voxels)
    if torch.any(out_of_bounds):
        raise ValueError("Point coordinates out of point cloud bounds.")
    
    return voxel_idx

def voxel_to_metric_space(voxels: torch.Tensor, voxel_size: torch.Tensor,
                          min_coords: torch.Tensor) -> torch.Tensor:
    """
    Map a voxel center to metric space.

    Args:
        voxels (torch.Tensor): Coordinates of the voxels in voxel space (batch_size, n_voxels, 3)
        voxel_size (torch.Tensor): Voxel size (batch_size)
        min_coords (torch.Tensor): Minimum coordinate values in metric space (batch_size, 3)

    Returns:
        coords (torch.Tensor): Coordinates of the voxel centers in metric space (n_voxels, 3)
    """
    
    # get the voxel edge and add half a voxel and the offset
    return (voxels * voxel_size) + (voxel_size / 2.0) + min_coords
