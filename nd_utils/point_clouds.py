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

def random_sample_point_cloud(pcd: torch.Tensor, n_samples: int ,seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly sample a point cloud.

    Args:
        pcd (torch.Tensor): Point cloud (batch_size, n_points, 3)
        n_samples (int): Number of points to sample from each point cloud
        seed (int): Random seed
    
    Returns:
        torch.Tensor: Sampled point cloud (batch_size, n_samples, 3)
        torch.Tensor: Sampled point cloud indices (batch_size, n_samples)
    """

    # manually set the random seed
    torch.manual_seed(seed)

    # generate the random mask
    point_idx = torch.randperm(pcd.shape[1])[:n_samples]

    # sample the point cloud
    points = pcd[:, point_idx, :]

    return points, point_idx
