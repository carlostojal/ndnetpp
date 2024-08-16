import torch
from typing import Tuple

def find_point_cloud_limits(point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the limits of the point cloud coordinates in each axis.

    Args:
        point_cloud (torch.Tensor): Point cloud data (n_points, 3)

    Returns:
        torch.Tensor: minimum limits (3)
        torch.Tensor: maximum limits (3)
        torch.Tensor: dimension in each axis(3)
    """

    # find the limits
    max_coords = torch.max(point_cloud, dim=0).values
    min_coords = torch.min(point_cloud, dim=0).values

    # calculate the dimensions in each axis (max - min)
    dimensions = max_coords - min_coords

    return min_coords, max_coords, dimensions

def calculate_voxel_size(dimensions: torch.Tensor, n_desired_voxels: int) -> Tuple[float, torch.Tensor]:
    """
    Calculate the voxel size considering the point cloud characteristics and desired number of voxels
    (don't confuse with normal distributions. voxels without samples don't count as normal distributiondon't confuse with normal distributions. voxels without samples don't count as normal distributionss)

    Args:
        dimensions (torch.Tensor): Dimensions in each axis (3)
        n_desired_voxels (int): Desired number of voxels

    Returns:
        voxel_size (float): Calculated voxel size
        n_voxels (torch.Tensor): Number of voxels in each dimension (3)
    """

    # calculate the voxel size
    voxel_size = torch.pow(torch.prod(dimensions) / n_desired_voxels, 1.0/3.0)

    # calculate the number of voxels in each dimension
    n_voxels = torch.ceil(dimensions / voxel_size)

    return voxel_size, n_voxels

def metric_to_voxel_space(points: torch.Tensor, voxel_size: float, n_voxels: torch.Tensor,
                          min_coords: torch.Tensor) -> torch.Tensor:
    """
    Map a point in metric space to voxel space.

    Args:
        points (torch.Tensor): Coordinates of the points (n_points, 3)
        voxel_size (float): Voxel size
        n_voxels (torch.Tensor): Number of voxels in each dimension (3)
        min_coords (torch.Tensor): Minimum coordinate values in metric space (3)

    Returns:
        voxel_idx (torch.Tensor): Voxel indices in each dimension (3)
    """

    voxel_idx = torch.floor((points - min_coords) / voxel_size)

    # check out of bounds indices
    out_of_bounds = (voxel_idx < 0) | (voxel_idx >= n_voxels)
    if torch.any(out_of_bounds):
        raise ValueError("Point coordinates out of point cloud bounds.")
    
    return voxel_idx
