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
from typing import Any, List
from ndnetpp.nd import Voxelizer
from ndnetpp.point_clouds import PointCloudNorm


class NDNetppBackbone(nn.Module):
    """
    NDNet++ backbone common to all variants.
    """

    def __init__(self, conf: Any) -> None:
        super().__init__()

        self.num_nds = conf.backbone.num_nds
        self.voxel_sizes = conf.backbone.voxel_sizes

        # create the point cloud normalization layer
        self.pcd_norm = PointCloudNorm()

        # generate the ND layers
        nd_layers_list: List[nn.Module] = []
        for i in range(conf.backbone.num_nd_layers):
            nd_layer = self._generate_nd_layer(conf.backbone.num_nds[i],
                                               conf.backbone.voxel_sizes[i],
                                               conf.backbone.pointnet_feature_dims[i])
            nd_layers_list.append(nd_layer)
        self.nd_layers = nn.Sequential(*nd_layers_list)

        raise NotImplementedError("ND-Net++ backbone not implemented.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NDNet++ backbone.

        Args:
            x (torch.Tensor): the input point cloud, shaped (batch_size, point_dim, num_points)

        Returns:
            torch.Tensor: point cloud feature map shaped (batch_size, feature_dim, distsN)
        """

        # normalize the point cloud coordinates
        x = self.pcd_norm(x)

        # pass through the ND layers
        # TODO: get the features at each level to construct a feature pyramid / U-Net for segmentation
        x = self.nd_layers(x)

        return x

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
        pointnet = self._generate_pointnet_layer(feature_dims)

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
