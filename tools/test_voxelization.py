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
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import open3d as o3d
import numpy as np
import sys
sys.path.append(".")
from models.ndnetpp.nd import Voxelizer
from models.ndnetpp.point_clouds import PointCloudNorm
from datasets.ModelNet import ModelNet

if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="modelnet", required=False, type=str)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--n_points", type=int, required=False, default=10000)
    parser.add_argument("--n_dists", type=int, required=False, default=1000)
    parser.add_argument("--voxel_size", type=float, required=False, default=1)
    args = parser.parse_args()

    # check available devices
    device = device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # create the dataset and data loader instances
    valid_datasets = ("modelnet")
    dataset = None
    if args.dataset not in valid_datasets:
        raise RuntimeError(f"Invalid dataset {args.dataset}. Choose one from {valid_datasets}")
    if args.dataset == "modelnet":
        dataset = ModelNet(args.path, "test", num_sampled_points=int(args.n_points))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)

    # pick the first sample of the dataset
    sample, _ = next(iter(dataloader))
    sample = sample.to(device)
    # sample.requires_grad = True

    # create a normalizer and a voxelizer layer and pass the point cloud through it
    normalizer = PointCloudNorm()
    voxelizer = Voxelizer(int(args.n_dists), float(args.voxel_size))
    # voxelizer.train()
    norm_sample = normalizer(sample)
    voxels = voxelizer(norm_sample)

    # iterate the batch
    for i in range(voxels.shape[0]):
        # extract the batch
        batch_voxels = voxels[i]
        # slice the mean
        batch_voxels = batch_voxels[...,:3]

        # visualize using open3d
        # convert the tensor to a numpy array
        sample_np = batch_voxels.cpu().numpy()
        print(sample_np.shape)
        # convert the numpy array to an open3d point cloud
        sample_o3d = o3d.geometry.PointCloud()
        sample_o3d.points = o3d.utility.Vector3dVector(sample_np)
        # open a visualizer
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([sample_o3d, coordinate_frame])
    
    """
    # compute a fake loss
    loss = voxels.sum()
    loss.backward()
    print(loss.item())
    print(sample.grad)
    """

    print("Done")

    sys.exit(0)
