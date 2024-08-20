import torch
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np
import os
from typing import Tuple

class ModelNet(Dataset):
    """
    ModelNet dataset class.
    """

    def __init__(self, root_dir: str, stage: str = "train", num_sampled_points: int = 10000) -> None:
        """
        ModelNet class constructor.

        Args:
            root_dir (str): Root directory of the dataset.
            stage (str): Training stage (train/test).
            num_sampled_points (int): Number of points sampled from the mesh.
        """
        self.root_dir = root_dir
        self.stage = stage
        self.num_sampled_points = num_sampled_points

        if stage != "train" and stage != "test":
            raise NotImplementedError("Stage can only be \"train\" or \"test\".")

        # initialize dictionary by class and complete list
        self.files_per_class = {}
        self.files = []

        # check if the path exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError("The dataset path does not exist.")

        # get the class names
        self.classes = os.listdir(root_dir)
        self.classes.sort()

        # iterate the classes
        for c in self.classes:
            # get the files list
            l = os.listdir(os.path.join(root_dir, c, stage))
            l.sort()
            # index the files by class
            self.files_per_class[c] = l
            # add to the global list of files
            self.files.extend(l)

    def __len__(self) -> int:
        """
        ModelNet dataset length.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a ModelNet dataset sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            point_cloud (torch.Tensor): Tensor of points shaped (num_points, 3).
            class_idx (int): Index of the class.
        """
        # get the class name
        fname_short = self.files[idx]
        cname = fname_short.split("_")[0]

        # build the complete filename
        fname = os.path.join(self.root_dir, cname, self.stage, fname_short)

        # read the mesh
        mesh = o3d.io.read_triangle_mesh(fname)

        # sample a point cloud from the mesh (https://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Sampling)
        pcd = mesh.sample_points_uniformly(number_of_points=self.num_sampled_points)

        # convert the point cloud to a tensor
        pcd_np = np.asarray(pcd.points)
        point_cloud = torch.tensor(pcd_np, dtype=torch.float32)

        # get the class index
        class_idx = -1
        for idx, c in enumerate(self.classes):
            if c == cname:
                class_idx = idx
                break

        return point_cloud, class_idx

