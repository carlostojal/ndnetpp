#include <cuda_runtime.h>
#include <cstdint>

/*! \brief Accumulate the normal distributions for each voxel.

    This function accumulates the normal distribution for each voxel. The normal distributions are
    represented by the point coordinate mean and covariances.

    \param points The point coordinates.
    \param n_points The number of points.
    \param n_desired_nds The number of desired normal distributions.
    \param voxels_x The number of voxels in x direction.
    \param voxels_y The number of voxels in y direction.
    \param voxels_z The number of voxels in z direction.
    \param normal_dists The normal distributions. Each normal distribution is 12-d (3-d mean and 9-d covariance matrix).
    \param normal_dist_samples The number of samples for each voxel.
    \param n_valid_dists The total number of valid normal distributions (voxels with more than one sample).
*/
__global__ void accumulate_normal_dists(float* points, long n_points, long n_desired_nds, 
                                                long voxels_x, long voxels_y, long voxels_z, 
                                                float* normal_dists, short* normal_dist_samples, long* n_valid_dists) {

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    // get the point coordinates from the index
    float x = points[3 * idx];
    float y = points[3 * idx + 1];
    float z = points[3 * idx + 2];

    // get the voxel coordinates from the point coordinates
    long voxel_x = (long) floor(x);
    long voxel_y = (long) floor(y);
    long voxel_z = (long) floor(z);

    // check if the voxel is within the bounds
    if (voxel_x < 0 || voxel_x >= voxels_x || voxel_y < 0 || voxel_y >= voxels_y || voxel_z < 0 || voxel_z >= voxels_z) return;

    // get the voxel index in contiguous memory
    long voxel_idx = voxel_x + voxels_x * (voxel_y + voxels_y * voxel_z);

    // add to the number of samples for the voxel
    atomicAdd(&normal_dist_samples[voxel_idx], 1);

    // get the current number of samples for the voxel
    short n_samples = normal_dist_samples[voxel_idx];

    // if the number is 1 (first voxel sample), add to the total number of valid normal distributions
    if (n_samples == 1) {
        atomicAdd(n_valid_dists, 1);
    }

    // get the current normal distribution for the voxel
    float* normal_dist = &normal_dists[12 * voxel_idx];

    // atomically add the point coordinates to the mean
    atomicAdd(&normal_dist[0], x);
    atomicAdd(&normal_dist[1], y);
    atomicAdd(&normal_dist[2], z);

    // add the outer product of the point coordinates to the covariance matrix
    atomicAdd(&normal_dist[3], x * x);
    atomicAdd(&normal_dist[4], x * y);
    atomicAdd(&normal_dist[5], x * z);
    atomicAdd(&normal_dist[6], y * x);
    atomicAdd(&normal_dist[7], y * y);
    atomicAdd(&normal_dist[8], y * z);
    atomicAdd(&normal_dist[9], z * x);
    atomicAdd(&normal_dist[10], z * y);
    atomicAdd(&normal_dist[11], z * z);

}

