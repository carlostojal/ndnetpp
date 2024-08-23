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

class PointCloudNorm(torch.nn.Module):
    """
    Normalizes the input point cloud into a unit sphere.
    """
    def __init__(self):
        super().__init__()

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input point clouds into unit spheres.

        Args:
            points (torch.Tensor): Input point clouds shaped (batch_size, n_points, 3).

        Returns:
            norm_points (torch.Tensor): Output point clouds shaped (batch_size, n_points, 3).
        """

        # compute the centroid
        centroid = points.mean(dim=1)

        # translate to the centroid
        points = points - centroid.unsqueeze(1)

        # compute the maximum distance to the centroid
        max_dist = torch.sqrt((points**2).sum(dim=2)).max(dim=1)[0]

        # scale to the maximum distance
        return points / max_dist.unsqueeze(1).unsqueeze(2)
    