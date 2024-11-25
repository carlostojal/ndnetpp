import unittest
import torch
from nd_utils.voxelization import *

class TestVoxelizer(unittest.TestCase):
    """
    Test suite for the voxelizer utility functions.
    """

    def test_find_limits(self) -> None:
        """
        Test the point cloud limit finding utility. Must return the minimum and maximum coordinates of a point cloud and dimensions in each axis.
        """

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
        """
        Test the voxel size calculation utility. Must return the voxel size and count across dimensions considering a given desired number of voxels.
        """

        print("Calculating voxel size and count...")
        voxel_size, n_voxels = calculate_voxel_size(torch.tensor([10, 10, 2]), 200)

        print("Testing voxel size and count...")
        self.assertAlmostEqual(voxel_size, 1.0)
        self.assertTrue(torch.equal(torch.tensor([10, 10, 2]), n_voxels))

    def test_convert_metric_to_voxel_space(self) -> None:
        """
        Test the conversion from metric to voxel space. Must return the voxel coordinates for each input point.
        """

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

    def test_convert_voxel_to_metric_space(self) -> None:
        """
        Test conversion from voxel to metric space. Must return the coordinate of the center of each voxel.
        """

        voxel_idx_l: List[torch.Tensor] = []

        for x in range(2):
            for y in range(3):
                for z in range(3):
                    voxel_idx_l.append(torch.tensor([x, y, z]))

        voxel_idx = torch.stack(voxel_idx_l, dim=0).unsqueeze(0)

        print("Converting voxel indices to metric space...")
        min_coords, _, _ = find_point_cloud_limits(voxel_idx)
        min_coords = min_coords.float()
        pcd = voxel_to_metric_space(voxel_idx, 1, min_coords)

        print("Testing voxel indices...")
        self.assertTrue(torch.equal(pcd, voxel_idx))

