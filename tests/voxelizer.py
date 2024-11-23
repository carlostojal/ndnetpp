import unittest
import torch
from nd_utils.voxelization import find_point_cloud_limits, calculate_voxel_size

class TestVoxelizer(unittest.TestCase):

    def test_find_limits(self) -> None:

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
    
        print("Finding point cloud limits...")
        min_coords, max_coors, dimensions = find_point_cloud_limits(pcd)

        print("Testing point cloud limits...")
        self.assertTrue(torch.equal(min_coords, torch.tensor([-1, -1, -1])))
        self.assertTrue(torch.equal(max_coors, torch.tensor([1, 1, 1])))
        self.assertTrue(torch.equal(dimensions, torch.tensor([2, 2, 2])))

    def test_calculate_voxel_size(self) -> None:

        print("Calculating voxel size and count...")
        voxel_size, n_voxels = calculate_voxel_size(torch.tensor([10, 10, 2]), 200)

        print("Testing voxel size and count...")
        self.assertAlmostEqual(voxel_size, 1.0)
        self.assertTrue(torch.equal(torch.tensor([10, 10, 2]), n_voxels))

