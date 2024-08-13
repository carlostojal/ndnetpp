#ifndef NDNETPP_ND_UTILS_VOXELIZE_H_
#define NDNETPP_ND_UTILS_VOXELIZE_H_

#include <vector>
#include <torch/extension.h>

/*! \brief Voxelize the input point cloud, estimating a normal distribution in each voxel. Each normal distribution has a (0,0,0) mean.

    \param points (torch::Tensor): Input point cloud with shape (N, 3).
    \param desired_n_voxels (torch::Tensor): Desired number of voxels in each dimension.

    \return A vector of two tensors. The first tensor has shape (desired_n_voxels[0], desired_n_voxels[1], desired_n_voxels[2], 3) and contains the means of the normal distributions. The second tensor has shape (desired_n_voxels[0], desired_n_voxels[1], desired_n_voxels[2], 3, 3) and contains the covariances of the normal distributions.
 */
std::vector<torch::Tensor> voxelize_forward(
    torch::Tensor points,
    torch::Tensor desired_n_voxels);

#endif // NDNETPP_ND_UTILS_VOXELIZE_H_
