#include <torch/extension.h>

std::vector<torch::Tensor> voxelize_forward(
    torch::Tensor points,
    torch::Tensor desired_n_voxels);
