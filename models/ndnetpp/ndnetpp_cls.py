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
from models.ndnetpp.ndnetpp import NDNetppBackbone
from models.utils import GlobalMaxPool1d
from models.pointnet import PointNet
from typing import Any, List

class NDNetppClassifier(nn.Module):
    """
    ND-Net++ classifier module.
    """

    def __init__(self, conf: Any) -> None:
        """
        ND-Net++ classifier class constructor.

        Args: 
            conf (Any): Configuration object as per the conf. file
        """

        super().__init__()

        # build the backbone
        self.backbone = NDNetppBackbone(conf)

        # get the dimension of the feature map
        feature_dim = int(conf['backbone']['pointnet_feature_dims'][-1][-1])

        # build the classifier MLP
        self.classifier = self._build_classifier(feature_dim, conf['cls_head'])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ND-Net++ classifier forward pass.

        Args:
            x (torch.Tensor): point cloud shaped (batch_size, n_points, 3)

        Returns:
            torch.Tensor: probability distribution over the object classes
        """

        # extract features with the backbone
        x = self.backbone(x)

        # use the classifier
        x = self.classifier(x)

        return x

    def _build_classifier(self, in_features: int, conf: Any) -> nn.Sequential:
        """
        Build a classifier.

        Args:
            conf (Any): Classifier configuration object.
        """

        layers: List[nn.Module] = []

        # build the pointnet layer
        pointnet = PointNet(in_features, conf['pointnet_feature_dims'], conf['tnet_feature_dims'])
        maxpool = GlobalMaxPool1d(dim=2, keepdim=False)
        layers.extend([pointnet, maxpool])

        # generate the fully-connected layers
        last_in = conf['pointnet_feature_dims'][-1]
        for i in range(len(conf['fc_dims'])):
            l = nn.Linear(last_in, int(conf['fc_dims'][i]))  # create the FC layer
            d = nn.Dropout(conf['dropout_probs'][i])
            layers.extend([l, d])  # add the layer to the list
            last_in = int(conf['fc_dims'][i])  # update the input dimension of the next layer
        # add the last layer with the number of classes
        l = nn.Linear(last_in, int(conf['num_classes']))
        s = nn.Softmax(dim=1)
        layers.extend([l, s])

        # build a sequential module from the list
        return nn.Sequential(*layers)

