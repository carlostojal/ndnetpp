import unittest
import torch
from nd_utils.voxelization import find_point_cloud_limits

class TestVoxelizer(unittest.TestCase):

    def test_find_limits(self):

        pcd = torch.tensor([[
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1]
        ]])

        min_coords, max_coors, dimensions = find_point_cloud_limits(pcd)

        self.assertTrue(torch.equal(min_coords, torch.tensor([-1, -1, -1])))
        self.assertTrue(torch.equal(max_coors, torch.tensor([1, 1, 1])))
        self.assertTrue(torch.equal(dimensions, torch.tensor([2, 2, 2])))
