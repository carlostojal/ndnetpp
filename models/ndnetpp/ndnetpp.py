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

class NDNetppBackbone(nn.Module):
    """
    NDNet++ backbone common to all variants.
    """
    
    def __init__(self, conf: Any) -> None:
        super().__init__()
        self.conf = conf

        # TODO: generate ND layers
        # TODO: generate T-Nets

        raise NotImplementedError("ND-Net++ backbone not implemented.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NDNet++ backbone.
        
        Args:
        - x (torch.Tensor): the input point cloud, shaped (batch_size, point_dim, num_points)
        
        Returns:
        - torch.Tensor: point cloud feature map shaped (batch_size, feature_dim, distsN)
        """
        
        raise NotImplementedError("ND-Net++ backbone not implemented.")
    
    def _generate_nd_layer(self, n_point_samples: int, voxel_size: float, feature_dims: List[int]) -> nn.Module:
        """
        Generate the ND layers.
        """
        
        raise NotImplementedError("ND-Net++ backbone not implemented.")
    
    def _generate_tnet(self, feature_dims: List[int]) -> nn.Module:
        """
        Generate the T-Nets.
        """
        
        raise NotImplementedError("ND-Net++ backbone not implemented.")
