import torch
from torch import nn
from typing import List

class TNet(nn.Module):
    """
    Transformation Network
    """

    def __init__(self, in_dim: int = 64, 
                 feature_dims: List[int] = [64,128,1024,512,256]) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.feature_dims = feature_dims

        self.conv1 = nn.Conv1d(in_dim, self.feature_dims[0], 1)
        self.conv2 = nn.Conv1d(self.feature_dims[0], self.feature_dims[1], 1)
        self.conv3 = nn.Conv1d(self.feature_dims[1], self.feature_dims[2], 1)

        self.fc1 = nn.Linear(self.feature_dims[2], self.feature_dims[3])
        self.fc2 = nn.Linear(self.feature_dims[3], self.feature_dims[4])
        self.fc3 = nn.Linear(self.feature_dims[4], in_dim**2)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(self.feature_dims[0])
        self.bn2 = nn.BatchNorm1d(self.feature_dims[1])
        self.bn3 = nn.BatchNorm1d(self.feature_dims[2])
        self.bn4 = nn.BatchNorm1d(self.feature_dims[3])
        self.bn5 = nn.BatchNorm1d(self.feature_dims[4])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformation Network

        Args:
        - x (torch.Tensor): the input point cloud, shaped (batch_size, point_dim, num_points)

        Returns:
        - torch.Tensor: the output of the Transformation Network
        """

        # MLP
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x))) # (batch_size, n_points, 1024) shape

        # max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.feature_dims[2]) # (batch_size, 1024) shape

        # FC layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity matrix
        x += torch.eye(self.in_dim).view(1, self.in_dim**2).repeat(x.size(0), 1).to(x.device)
        x = x.view(-1, self.in_dim, self.in_dim)

        return x


class PointNet(nn.Module):
    """
    PointNet
    """

    def __init__(self, 
                 point_dim: int = 3, 
                 feature_dims: List[int] = [64,128,768],
                 tnet_feature_dims: List[int] = [64,128,1024,512,256]) -> None:
        """
        Constructor of the PointNet

        Args:
        - point_dim (int): the dimension of the input points. Default is 3.
        - feature_dim (int): the dimension of the output features. Default is 768.
        - extra_type (str): the type of the additional features. Can be "none", "covariances" or "feature_vector". Default is "covariances".
        """
        super().__init__()

        self.point_dim = point_dim
        self.feature_dims = feature_dims
        self.tnet_feature_dims = tnet_feature_dims

        self.conv1 = nn.Conv1d(self.point_dim, self.feature_dims[0], 1)
        self.conv2 = nn.Conv1d(self.feature_dims[0], self.feature_dims[1], 1)
        self.conv3 = nn.Conv1d(self.feature_dims[1], self.feature_dims[2], 1)

        self.bn1 = nn.BatchNorm1d(self.feature_dims[0])
        self.bn2 = nn.BatchNorm1d(self.feature_dims[1])
        self.bn3 = nn.BatchNorm1d(self.feature_dims[2])

        self.t1 = TNet(in_dim=point_dim, feature_dims=tnet_feature_dims)
        self.t2 = TNet(in_dim=self.feature_dims[0], feature_dims=tnet_feature_dims)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PointNet

        Args:
        - points (torch.Tensor): the points tensor, shaped (batch_size, num_points, point_dim)

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: tensor with shape (batch_size, feature_dim, num_nds) and the feature transform
        """

        B, N, D = x.size()

        x = x.transpose(2, 1) # [B, 12, N]

        # input transform
        t = self.t1(x) # [B, 3, 3]
        # apply the transformation matrix to the points
        x = torch.bmm(t, x) # [B, 3, N]
        x = torch.nan_to_num(x, nan=0.0)

        # MLP
        x = self.bn1(self.conv1(x))

        # feature transform
        t = self.t2(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, t)
        x = x.transpose(2, 1)

        x_t2 = x

        # MLP
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))

        # return a tensor with shape (batch_size, feature_dim, num_points) and the feature transform
        return x, x_t2
    
    def __repr__(self):

        return (f"{self.__class__.__name__}("
                f"point_dim={self.point_dim}"
                f"feature_dims={self.feature_dims}"
                f"tnet_feature_dims={self.tnet_feature_dims}"
        )
