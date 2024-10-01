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
from models.ndnetpp.nd import Voxelizer
from models.ndnetpp.point_clouds import PointCloudNorm
from models.utils import _generate_nd_layer, _generate_pointnet_layer


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
        first: bool = True
        for i in range(conf.backbone.num_nd_layers):
            nd_layer = _generate_nd_layer(conf.backbone.num_nds[i],
                                               conf.backbone.voxel_sizes[i],
                                               conf.backbone.pointnet_feature_dims[i],
                                               first)
            nd_layers_list.append(nd_layer)
            first = False
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

