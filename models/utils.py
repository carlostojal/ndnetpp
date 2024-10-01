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
from typing import List
import torch
from torch import nn
from models.ndnetpp.nd import Voxelizer

def _generate_nd_layer(self, num_nds: int, voxel_size: float, feature_dims: List[int],
                           first: bool = True) -> nn.Module:
        """
        Generate a ND layer.

        Args:
            num_nds (int): Number of normal distributions estimated.
            voxel_size (float): Number of the edge of each voxel in the grid.
            feature_dims (List[int]): List of PointNet feature dimensions.
            first (bool): Is this the first ND layer of ND-Net? Default: True.

        Returns:
            nn.Module: The ND module.
        """

        # initialize the voxelizer layer
        nd = Voxelizer(num_nds, voxel_size, not first)

        # initialize the pointnet layer
        pointnet = _generate_pointnet_layer(feature_dims)

        # create the sequential module
        return nn.Sequential([nd, pointnet])


def _generate_pointnet_layer(self, feature_dims: List[int]) -> nn.Module:
    """
    Generate a PointNet layer.

    Args:
        feature_dims (List[int]): List of PointNet feature dimensions.

    Returns:
        nn.Module: The PointNet module.
    """

    in_channels = feature_dims[0]

    # generate each pointnet layer
    layers: List[nn.Module] = []
    for i in range(len(feature_dims)-1):
        conv = nn.Conv1d(in_channels, feature_dims[i+1], 1)
        bn = nn.BatchNorm1d(feature_dims[i+1])
        layer = nn.Sequential([conv, bn])
    layers.append(layer)

    # generate the sequential pointnet
    return nn.Sequential(*layers)

