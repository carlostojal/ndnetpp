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
