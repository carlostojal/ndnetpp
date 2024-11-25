import unittest
import torch
from nd_utils.voxelization import *

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

    def test_convert_metric_to_voxel_space(self) -> None:

        points: List[torch.Tensor] = []

        for x in range(2):
            for y in range(3):
                for z in range(3):
                    points.append(torch.tensor([x, y, z]))
    
        pcd = torch.stack(points, dim=0).unsqueeze(0)


        print("Converting points from metric to voxel space...")
        min_coords, _, _ = find_point_cloud_limits(pcd)
        voxel_idx = metric_to_voxel_space(pcd, 1, 9, min_coords)

        print("Testing voxel indices...")
        self.assertTrue(torch.equal(pcd, voxel_idx))



