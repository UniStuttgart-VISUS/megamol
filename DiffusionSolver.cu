//
// DiffusionSolver.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 16, 2013
// Author     : scharnkn
//

#include "stdafx.h"
#include "DiffusionSolver.h"

#ifdef WITH_CUDA

#include "cuda_error_check.h"
#include "cuda_helper.h"

using namespace megamol;
using namespace megamol::protein;


/*
 * DiffusionSolver::CalcGVF
 */
bool DiffusionSolver::CalcGVF(const float *startVol, size_t dim[3],
        float isovalue, float radius, float *gvf_D, unsigned int maxIt) {

    // Initialize the GVF field with the gradient in the starting regions
    if (!DiffusionSolver::initGVF(startVol, dim, isovalue, radius, gvf_D)) {
        return false;
    }

    return true;
}


/*
 * calcGradient_D
 * Computes the gradient of the given volume.
 */
__global__ void calcGradient_D(const float *vol_D, float *grad_D, uint3 voldim) {

    const uint idx = __umul24(__umul24(blockIdx.y, gridDim.x) +
            blockIdx.x, blockDim.x) + threadIdx.x;

    // Get grid coordinates
    uint3 gridCoord = make_uint3(
            idx % voldim.x,
            (idx / voldim.x) % voldim.y,
            (idx / voldim.x) / voldim.y);

    // Omit border cells (gradient remains zero)
    if (gridCoord.x == 0) return;
    if (gridCoord.y == 0) return;
    if (gridCoord.z == 0) return;
    if (gridCoord.x >= uint(voldim.x - 1)) return;
    if (gridCoord.y >= uint(voldim.y - 1)) return;
    if (gridCoord.z >= uint(voldim.z - 1)) return;

    float3 grad;
    uint3 x1, x2;

    x1 = make_uint3(gridCoord.x+1, gridCoord.y+0, gridCoord.z+0);
    x2 = make_uint3(gridCoord.x-1, gridCoord.y+0, gridCoord.z+0);
    grad.x = vol_D[voldim.x*(voldim.y*x1.z + x1.y) + x1.x]-
             vol_D[voldim.x*(voldim.y*x2.z + x2.y) + x2.x];

    x1 = make_uint3(gridCoord.x, gridCoord.y+1, gridCoord.z+0);
    x2 = make_uint3(gridCoord.x, gridCoord.y-1, gridCoord.z+0);
    grad.y = vol_D[voldim.x*(voldim.y*x1.z + x1.y) + x1.x]-
             vol_D[voldim.x*(voldim.y*x2.z + x2.y) + x2.x];

    x1 = make_uint3(gridCoord.x, gridCoord.y, gridCoord.z+1);
    x2 = make_uint3(gridCoord.x, gridCoord.y, gridCoord.z-1);
    grad.z = vol_D[voldim.x*(voldim.y*x1.z + x1.y) + x1.x]-
             vol_D[voldim.x*(voldim.y*x2.z + x2.y) + x2.x];

    float len = length(grad);
    if (len > 0) grad/= len;
    grad_D[4*idx+0] = grad.x;
    grad_D[4*idx+1] = grad.y;
    grad_D[4*idx+2] = grad.z;
}


/*
 * DiffusionSolver::initGVFCuda
 */
bool DiffusionSolver::initGVF(const float *startVol, size_t dim[3],
        float isovalue, float radius, float *gvf_D) {

    size_t volsize = dim[0]*dim[1]*dim[2];
    uint3 voldim = make_uint3(dim[0], dim[1], dim[2]);

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Calculate gradient using finite differences
    calcGradient_D <<< DiffusionSolver::Grid(volsize, 256), 256 >>> (startVol, gvf_D, voldim);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'calcGradient_D':                        %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return true;
}


/**
 * Returns a 1D grid definition based on the given threadsPerBlock value.
 *
 * @param size             The minimum number of threads
 * @param threadsPerBlock  The number of threads per block
 * @return The grid dimensions
 */
extern "C" dim3 DiffusionSolver::Grid(const unsigned int size, const int threadsPerBlock) {
    //TODO: remove hardcoded hardware capabilities :(
    // see: http://code.google.com/p/thrust/source/browse/thrust/detail/backend/cuda/arch.inl
    //   and http://code.google.com/p/thrust/source/browse/thrust/detail/backend/cuda/detail/safe_scan.inl
    //   for refactoring.
    // Get maximum grid size of CUDA device.
    //CUdevice device;
    //cuDeviceGet(&device, 0);
    //CUdevprop deviceProps;
    //cuDeviceGetProperties(&deviceProps, device);
    //this->gridSize = dim3(deviceProps.maxGridSize[0],
    //  deviceProps.maxGridSize[1],
    //  deviceProps.maxGridSize[2]);
    const dim3 maxGridSize(65535, 65535, 0);
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

    return grid;
}

#endif // WITH_CUDA




