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

#define USE_TIMER
//#define USE_CUDA_TIMER


/**
 * Samples the field at a given (integer) grid position.
 *
 * @param x,y,z Coordinates of the position
 * @return The sampled value of the field
 */
template <typename T>
inline __device__ T SampleFieldAt_D(uint x, uint y, uint z, T *field_D, uint3 dim) {
    return field_D[dim.x*(dim.y*z+y)+x];
}


/**
 * Answers the grid position index associated with the given coordinates.
 *
 * @param v0 The coordinates
 * @return The index
 */
inline __device__ uint GetPosIdxByGridCoords(uint3 v0, uint3 voldim) {
    return voldim.x*(voldim.y*v0.z + v0.y) + v0.x;
}

/**
 * Answers the cell index associated with the given coordinates.
 *
 * @param v0 The coordinates
 * @return The index
 */
inline __device__ uint GetCellIdxByGridCoords(int3 v0, uint3 voldim) {
    return (voldim.x-1)*((voldim.y-1)*v0.z + v0.y) + v0.x;
}

/**
 * Answers the grid position coordinates associated with a given cell index.
 * The returned position is the left/lower/back corner of the cell
 *
 * @param index The index
 * @return The coordinates
 */
inline __device__ uint3 GetGridCoordsByCellIdx(uint index, uint3 voldim) {
    return make_uint3(index % (voldim.x-1),
                      (index / (voldim.x-1)) % (voldim.y-1),
                      (index / (voldim.x-1)) / (voldim.y-1));
}

/**
 * Answers the cell coordinates associated with a given grid position index.
 *
 * @param index The index
 * @return The coordinates
 */
inline __device__ uint3 GetGridCoordsByPosIdx(uint index, uint3 voldim) {
    return make_uint3(index % voldim.x,
                      (index / voldim.x) % voldim.y,
                      (index / voldim.x) / voldim.y);
}


/*
 * updateGVF_D
 */
__global__ void updateGVF_D(
        float *gvfIn_D,
        float *gvfOut_D,
        float *gvfConstData_D, // b, c1, c2, c3
        float scl,
        uint3 voldim) {

    const uint idx = __umul24(__umul24(blockIdx.y, gridDim.x) +
            blockIdx.x, blockDim.x) + threadIdx.x;

    uint volsize = voldim.x*voldim.y*voldim.z;
    if (idx >= volsize) return;

    float3 gvf, gvfOld, gvfAdj[6];
    uint idxAdj[6];

    // Get grid coordinates
    uint3 gridC = make_uint3(
            idx % voldim.x,
            (idx / voldim.x) % voldim.y,
            (idx / voldim.x) / voldim.y);

    // Get const data
    float b = gvfConstData_D[4*idx+0];
    float c1 = gvfConstData_D[4*idx+1];
    float c2 = gvfConstData_D[4*idx+2];
    float c3 = gvfConstData_D[4*idx+3];

    /* Update isotropic diffusion for all vector components */

    // Get indices of adjacent values
    idxAdj[0] = ::GetPosIdxByGridCoords(make_uint3(uint(clamp(int(gridC.x)-1, 0, int(voldim.x-1))), gridC.y, gridC.z), voldim);
    idxAdj[1] = ::GetPosIdxByGridCoords(make_uint3(uint(clamp(int(gridC.x)+1, 0, int(voldim.x-1))), gridC.y, gridC.z), voldim);
    idxAdj[2] = ::GetPosIdxByGridCoords(make_uint3(gridC.x, uint(clamp(int(gridC.y)-1, 0, int(voldim.y-1))), gridC.z), voldim);
    idxAdj[3] = ::GetPosIdxByGridCoords(make_uint3(gridC.x, uint(clamp(int(gridC.y)+1, 0, int(voldim.y-1))), gridC.z), voldim);
    idxAdj[4] = ::GetPosIdxByGridCoords(make_uint3(gridC.x, gridC.y, uint(clamp(int(gridC.z)-1, 0, int(voldim.z-1)))), voldim);
    //idxAdj[4] = 0;
    idxAdj[5] = ::GetPosIdxByGridCoords(make_uint3(gridC.x, gridC.y, uint(clamp(int(gridC.z)+1, 0, int(voldim.z-1)))), voldim);

    // Get adjacent gvf values
    gvfOld = make_float3(gvfIn_D[4*idx+0], gvfIn_D[4*idx+1], gvfIn_D[4*idx+2]);
    gvfAdj[0] = make_float3(gvfIn_D[4*idxAdj[0]+0], gvfIn_D[4*idxAdj[0]+1], gvfIn_D[4*idxAdj[0]+2]);
    gvfAdj[1] = make_float3(gvfIn_D[4*idxAdj[1]+0], gvfIn_D[4*idxAdj[1]+1], gvfIn_D[4*idxAdj[1]+2]);
    gvfAdj[2] = make_float3(gvfIn_D[4*idxAdj[2]+0], gvfIn_D[4*idxAdj[2]+1], gvfIn_D[4*idxAdj[2]+2]);
    gvfAdj[3] = make_float3(gvfIn_D[4*idxAdj[3]+0], gvfIn_D[4*idxAdj[3]+1], gvfIn_D[4*idxAdj[3]+2]);
    gvfAdj[4] = make_float3(gvfIn_D[4*idxAdj[4]+0], gvfIn_D[4*idxAdj[4]+1], gvfIn_D[4*idxAdj[4]+2]);
    gvfAdj[5] = make_float3(gvfIn_D[4*idxAdj[5]+0], gvfIn_D[4*idxAdj[5]+1], gvfIn_D[4*idxAdj[5]+2]);

    // Compute diffusion
    gvf.x = (1.0-b)*gvfOld.x;
    gvf.x += (gvfAdj[0].x + gvfAdj[1].x + gvfAdj[2].x + gvfAdj[3].x +
              gvfAdj[4].x + gvfAdj[5].x -6*gvfOld.x)*scl;
    gvf.x += c1;

    gvf.y = (1.0-b)*gvfOld.y;
    gvf.y += (gvfAdj[0].y + gvfAdj[1].y + gvfAdj[2].y + gvfAdj[3].y +
            gvfAdj[4].y + gvfAdj[5].y -6*gvfOld.y)*scl;
    gvf.y += c2;

    gvf.z = (1.0-b)*gvfOld.z;
    gvf.z += (gvfAdj[0].z + gvfAdj[1].z + gvfAdj[2].z + gvfAdj[3].z +
            gvfAdj[4].z + gvfAdj[5].z -6*gvfOld.z)*scl;
    gvf.z += c3;

    float len = length(gvf);
    if (len > 0.0f) gvf /= len;

    //__syncthreads();
    gvfOut_D[4*idx+0] = gvf.x;
    gvfOut_D[4*idx+1] = gvf.y;
    gvfOut_D[4*idx+2] = gvf.z;
}


/*
 * DiffusionSolver::CalcGVF
 */
bool DiffusionSolver::CalcGVF(const float *startVol, float *gvfConstData_D,
        const unsigned int *cellStates_D,
        float *grad_D, size_t dim[3], float isovalue,
        float *gvfIn_D, float *gvfOut_D, unsigned int maxIt, float scl) {

    // Initialize the GVF field with the gradient in the starting regions
    if (!DiffusionSolver::initGVF(startVol, dim, cellStates_D, isovalue, grad_D,
            gvfConstData_D)) {
        return false;
    }

    uint volsize = dim[0]*dim[1]*dim[2];
    uint3 voldim = make_uint3(dim[0], dim[1], dim[2]);


    for (unsigned int it=(maxIt%2); it < maxIt+(maxIt%2); ++it) {

#ifdef USE_CUDA_TIMER
        float dt_ms;
        cudaEvent_t event1, event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);
        cudaEventRecord(event1, 0);
#endif
        if (it%2 == 0) {
            // Update diffusion
            updateGVF_D <<< DiffusionSolver::Grid(volsize, 256), 256 >>> (
                    gvfIn_D, gvfOut_D, gvfConstData_D, scl, voldim);

            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
        } else {
            // Update diffusion
            updateGVF_D <<< DiffusionSolver::Grid(volsize, 256), 256 >>> (
                    gvfOut_D, gvfIn_D, gvfConstData_D, scl, voldim);

            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
        }

#ifdef USE_CUDA_TIMER
        cudaEventRecord(event2, 0);
        cudaEventSynchronize(event1);
        cudaEventSynchronize(event2);
        cudaEventElapsedTime(&dt_ms, event1, event2);
        printf("CUDA time for 'updateGVF_D':                       %.10f sec\n",
                dt_ms/1000.0f);
        cudaEventRecord(event1, 0);
#endif
    }

    return true;
}


/*
 * DiffusionSolver::CalcTwoWayGVF
 */
bool DiffusionSolver::CalcTwoWayGVF(
        const float *volSource_D,
        const float *volTarget_D,
        float *gvfConstData_D,
        const unsigned int *cellStatesSource_D,
        const unsigned int *cellStatesTarget_D,
        float *grad_D,
        size_t dim[3],
        float isovalue,
        float *gvfIn_D,
        float *gvfOut_D,
        unsigned int maxIt,
        float scl) {

    // Initialize the GVF field with the gradient in the starting regions
    if (!DiffusionSolver::initTwoWayGVF(volSource_D, volTarget_D, dim,
            cellStatesSource_D, cellStatesTarget_D, isovalue, grad_D,
            gvfConstData_D)) {
        return false;
    }

    uint volsize = dim[0]*dim[1]*dim[2];
    uint3 voldim = make_uint3(dim[0], dim[1], dim[2]);


    for (unsigned int it=(maxIt%2); it < maxIt+(maxIt%2); ++it) {

#ifdef USE_CUDA_TIMER
        float dt_ms;
        cudaEvent_t event1, event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);
        cudaEventRecord(event1, 0);
#endif
        if (it%2 == 0) {
            // Update diffusion
            updateGVF_D <<< DiffusionSolver::Grid(volsize, 256), 256 >>> (
                    gvfIn_D, gvfOut_D, gvfConstData_D, scl, voldim);

            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
        } else {
            // Update diffusion
            updateGVF_D <<< DiffusionSolver::Grid(volsize, 256), 256 >>> (
                    gvfOut_D, gvfIn_D, gvfConstData_D, scl, voldim);

            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
        }

#ifdef USE_CUDA_TIMER
        cudaEventRecord(event2, 0);
        cudaEventSynchronize(event1);
        cudaEventSynchronize(event2);
        cudaEventElapsedTime(&dt_ms, event1, event2);
        printf("CUDA time for 'updateGVF_D':                       %.10f sec\n",
                dt_ms/1000.0f);
        cudaEventRecord(event1, 0);
#endif
    }

    return true;
}


/*
 * calcGradient_D
 * Computes the gradient of the given volume.
 */
__global__ void calcGradient_D(const float *vol_D, float *grad_D,
        const unsigned int *cellStates_D, uint3 voldim, float isovalue) {

    const uint idx = __umul24(__umul24(blockIdx.y, gridDim.x) +
            blockIdx.x, blockDim.x) + threadIdx.x;

    uint volsize = voldim.x*voldim.y*voldim.z;
    if (idx >= volsize) return;

    uint nCells = (voldim.x-1)*(voldim.y-1)*(voldim.z-1);

    int3 cellC;
    uint cellIdx;
    int active = 0;

    // Get grid coordinates
    int3 gridC = make_int3(
            idx % (voldim.x),
            (idx / voldim.x) % voldim.y,
            (idx / voldim.x) / voldim.y);

    int3 voldimI = make_int3(int(voldim.x), int(voldim.y), int(voldim.z));

    /* Check all eight adjacent cells */

    // (-1, -1, -1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, voldimI.x-2),
            clamp(gridC.y-1, 0, voldimI.y-2),
            clamp(gridC.z-1, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    active |= cellStates_D[cellIdx];

    // (-1, 1, -1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, voldimI.x-2),
            clamp(gridC.y, 0, voldimI.y-2),
            clamp(gridC.z-1, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    active |= cellStates_D[cellIdx];

    // (-1, 1, 1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, voldimI.x-2),
            clamp(gridC.y, 0, voldimI.y-2),
            clamp(gridC.z, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    active |= cellStates_D[cellIdx];

    // (-1, -1, 1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, voldimI.x-2),
            clamp(gridC.y-1, 0, voldimI.y-2),
            clamp(gridC.z, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    active |= cellStates_D[cellIdx];

    // (1, -1, -1)
    cellC = make_int3(
            clamp(gridC.x, 0, voldimI.x-2),
            clamp(gridC.y-1, 0, voldimI.y-2),
            clamp(gridC.z-1, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    active |= cellStates_D[cellIdx];

    // (1, 1, -1)
    cellC = make_int3(
            clamp(gridC.x, 0, voldimI.x-2),
            clamp(gridC.y, 0, voldimI.y-2),
            clamp(gridC.z-1, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    active |= cellStates_D[cellIdx];

    // (1, 1, 1)
    cellC = make_int3(
            clamp(gridC.x, 0, voldimI.x-2),
            clamp(gridC.y, 0, voldimI.y-2),
            clamp(gridC.z, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    active |= cellStates_D[cellIdx];

    // (1, -1, 1)
    cellC = make_int3(
            clamp(gridC.x, 0, voldimI.x-2),
            clamp(gridC.y-1, 0, voldimI.y-2),
            clamp(gridC.z, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    active |= cellStates_D[cellIdx];

    /* Sample gradient if necessary */

    if (active) {
        float3 grad;
        uint3 x1, x2;

        x1 = make_uint3(clamp(gridC.x+1, 0, voldimI.x-1), gridC.y, gridC.z);
        x2 = make_uint3(clamp(gridC.x-1, 0, voldimI.x-1), gridC.y, gridC.z);
        grad.x = vol_D[voldim.x*(voldim.y*x1.z + x1.y) + x1.x]-
                vol_D[voldim.x*(voldim.y*x2.z + x2.y) + x2.x];

        x1 = make_uint3(gridC.x, clamp(gridC.y+1, 0, voldimI.y-1), gridC.z);
        x2 = make_uint3(gridC.x, clamp(gridC.y-1, 0, voldimI.y-1), gridC.z);
        grad.y = vol_D[voldim.x*(voldim.y*x1.z + x1.y) + x1.x]-
                vol_D[voldim.x*(voldim.y*x2.z + x2.y) + x2.x];

        x1 = make_uint3(gridC.x, gridC.y, clamp(gridC.z+1, 0, voldimI.z-1));
        x2 = make_uint3(gridC.x, gridC.y, clamp(gridC.z-1, 0, voldimI.z-1));
        grad.z = vol_D[voldim.x*(voldim.y*x1.z + x1.y) + x1.x]-
                vol_D[voldim.x*(voldim.y*x2.z + x2.y) + x2.x];

        float len = length(grad);
        if (len > 0.0) grad/= len;

        grad_D[4*idx+0] = grad.x;
        grad_D[4*idx+1] = grad.y;
        grad_D[4*idx+2] = grad.z;
    } else {
        grad_D[4*idx+0] = 0.0;
        grad_D[4*idx+1] = 0.0;
        grad_D[4*idx+2] = 0.0;
    }
}


/*
 * prepareGVFDiffusion
 */
__global__ void prepareGVFDiffusion_D(
        float *grad_D,
        float *gvfConstData_D, // b, c1, c2, c3
        uint3 voldim) {

    const uint idx = __umul24(__umul24(blockIdx.y, gridDim.x) +
            blockIdx.x, blockDim.x) + threadIdx.x;

    uint volsize = voldim.x*voldim.y*voldim.z;
    if (idx >= volsize) return;

    // Get grid coordinates
    uint3 gridCoord = make_uint3(
            idx % voldim.x,
            (idx / voldim.x) % voldim.y,
            (idx / voldim.x) / voldim.y);

    // Get vector field
    float3 grad;
    grad.x = grad_D[4*idx+0];
    grad.y = grad_D[4*idx+1];
    grad.z = grad_D[4*idx+2];

    // Compute len^2
    float len = grad.x*grad.x + grad.y*grad.y + grad.z*grad.z;

    // Write b to device memory
    gvfConstData_D[4*idx+0] = len;
    // Write c1, c2, and c3 to device memory
    gvfConstData_D[4*idx+1] = len*grad.x;
    gvfConstData_D[4*idx+2] = len*grad.y;
    gvfConstData_D[4*idx+3] = len*grad.z;

}


/*
 * DiffusionSolver::initGVFCuda
 */
bool DiffusionSolver::initGVF(const float *startVol, size_t dim[3],
        const unsigned int *cellStates_D,
        float isovalue, float *grad_D, float *gvfConstData_D) {

    size_t volsize = dim[0]*dim[1]*dim[2];
    uint3 voldim = make_uint3(dim[0], dim[1], dim[2]);

#ifdef USE_CUDA_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Calculate gradient using finite differences
    calcGradient_D <<< DiffusionSolver::Grid(volsize, 256), 256 >>> (
            startVol, grad_D, cellStates_D, voldim, isovalue);

#ifdef USE_CUDA_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'calcGradient_D':                        %.10f sec\n",
            dt_ms/1000.0f);
    cudaEventRecord(event1, 0);
#endif

    // Precompute b,c1,c2, and c3
    prepareGVFDiffusion_D <<< DiffusionSolver::Grid(volsize, 256), 256 >>> (
            grad_D, gvfConstData_D, voldim);

#ifdef USE_CUDA_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'prepareGVFDiffusion_D':                 %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return true;
}


/*
 * calcGradient_D
 * Computes the gradient of the given volume.
 */
__global__ void calcTwoWayGradient_D(
        const float *sourceVol_D,
        const float *targetVol_D,
        float *grad_D,
        const unsigned int *cellStatesSource_D,
        const unsigned int *cellStatesTarget_D,
        uint3 voldim,
        float isovalue) {

    const uint idx = __umul24(__umul24(blockIdx.y, gridDim.x) +
            blockIdx.x, blockDim.x) + threadIdx.x;

    uint volsize = voldim.x*voldim.y*voldim.z;
    if (idx >= volsize) return;

    uint nCells = (voldim.x-1)*(voldim.y-1)*(voldim.z-1);

    int3 cellC;
    uint cellIdx;
    int activeSource = 0, activeTarget=0;

    // Get grid coordinates
    int3 gridC = make_int3(
            idx % (voldim.x),
            (idx / voldim.x) % voldim.y,
            (idx / voldim.x) / voldim.y);

    int3 voldimI = make_int3(int(voldim.x), int(voldim.y), int(voldim.z));

    /* Check all eight adjacent cells in source volume */

    // (-1, -1, -1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, voldimI.x-2),
            clamp(gridC.y-1, 0, voldimI.y-2),
            clamp(gridC.z-1, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeSource |= cellStatesSource_D[cellIdx];

    // (-1, 1, -1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, voldimI.x-2),
            clamp(gridC.y, 0, voldimI.y-2),
            clamp(gridC.z-1, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeSource |= cellStatesSource_D[cellIdx];

    // (-1, 1, 1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, voldimI.x-2),
            clamp(gridC.y, 0, voldimI.y-2),
            clamp(gridC.z, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeSource |= cellStatesSource_D[cellIdx];

    // (-1, -1, 1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, voldimI.x-2),
            clamp(gridC.y-1, 0, voldimI.y-2),
            clamp(gridC.z, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeSource |= cellStatesSource_D[cellIdx];

    // (1, -1, -1)
    cellC = make_int3(
            clamp(gridC.x, 0, voldimI.x-2),
            clamp(gridC.y-1, 0, voldimI.y-2),
            clamp(gridC.z-1, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeSource |= cellStatesSource_D[cellIdx];

    // (1, 1, -1)
    cellC = make_int3(
            clamp(gridC.x, 0, voldimI.x-2),
            clamp(gridC.y, 0, voldimI.y-2),
            clamp(gridC.z-1, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeSource |= cellStatesSource_D[cellIdx];

    // (1, 1, 1)
    cellC = make_int3(
            clamp(gridC.x, 0, voldimI.x-2),
            clamp(gridC.y, 0, voldimI.y-2),
            clamp(gridC.z, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeSource |= cellStatesSource_D[cellIdx];

    // (1, -1, 1)
    cellC = make_int3(
            clamp(gridC.x, 0, voldimI.x-2),
            clamp(gridC.y-1, 0, voldimI.y-2),
            clamp(gridC.z, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeSource |= cellStatesSource_D[cellIdx];


    /* Check all adjacent cells in the target volume */

    // (-1, -1, -1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, voldimI.x-2),
            clamp(gridC.y-1, 0, voldimI.y-2),
            clamp(gridC.z-1, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (-1, 1, -1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, voldimI.x-2),
            clamp(gridC.y, 0, voldimI.y-2),
            clamp(gridC.z-1, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (-1, 1, 1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, voldimI.x-2),
            clamp(gridC.y, 0, voldimI.y-2),
            clamp(gridC.z, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (-1, -1, 1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, voldimI.x-2),
            clamp(gridC.y-1, 0, voldimI.y-2),
            clamp(gridC.z, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (1, -1, -1)
    cellC = make_int3(
            clamp(gridC.x, 0, voldimI.x-2),
            clamp(gridC.y-1, 0, voldimI.y-2),
            clamp(gridC.z-1, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (1, 1, -1)
    cellC = make_int3(
            clamp(gridC.x, 0, voldimI.x-2),
            clamp(gridC.y, 0, voldimI.y-2),
            clamp(gridC.z-1, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (1, 1, 1)
    cellC = make_int3(
            clamp(gridC.x, 0, voldimI.x-2),
            clamp(gridC.y, 0, voldimI.y-2),
            clamp(gridC.z, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (1, -1, 1)
    cellC = make_int3(
            clamp(gridC.x, 0, voldimI.x-2),
            clamp(gridC.y-1, 0, voldimI.y-2),
            clamp(gridC.z, 0, voldimI.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC, voldim);
    activeTarget |= cellStatesTarget_D[cellIdx];


    /* Sample gradients  */

    float3 gradSource,gradTarget;
    uint3 x1, x2;

    x1 = make_uint3(clamp(gridC.x+1, 0, voldimI.x-1), gridC.y, gridC.z);
    x2 = make_uint3(clamp(gridC.x-1, 0, voldimI.x-1), gridC.y, gridC.z);
    gradSource.x = sourceVol_D[voldim.x*(voldim.y*x1.z + x1.y) + x1.x]-
            sourceVol_D[voldim.x*(voldim.y*x2.z + x2.y) + x2.x];

    x1 = make_uint3(gridC.x, clamp(gridC.y+1, 0, voldimI.y-1), gridC.z);
    x2 = make_uint3(gridC.x, clamp(gridC.y-1, 0, voldimI.y-1), gridC.z);
    gradSource.y = sourceVol_D[voldim.x*(voldim.y*x1.z + x1.y) + x1.x]-
            sourceVol_D[voldim.x*(voldim.y*x2.z + x2.y) + x2.x];

    x1 = make_uint3(gridC.x, gridC.y, clamp(gridC.z+1, 0, voldimI.z-1));
    x2 = make_uint3(gridC.x, gridC.y, clamp(gridC.z-1, 0, voldimI.z-1));
    gradSource.z = targetVol_D[voldim.x*(voldim.y*x1.z + x1.y) + x1.x]-
            targetVol_D[voldim.x*(voldim.y*x2.z + x2.y) + x2.x];

    float len = length(gradSource);
    if (len > 0.0) gradSource /= len;

    x1 = make_uint3(clamp(gridC.x+1, 0, voldimI.x-1), gridC.y, gridC.z);
    x2 = make_uint3(clamp(gridC.x-1, 0, voldimI.x-1), gridC.y, gridC.z);
    gradTarget.x = targetVol_D[voldim.x*(voldim.y*x1.z + x1.y) + x1.x]-
            targetVol_D[voldim.x*(voldim.y*x2.z + x2.y) + x2.x];

    x1 = make_uint3(gridC.x, clamp(gridC.y+1, 0, voldimI.y-1), gridC.z);
    x2 = make_uint3(gridC.x, clamp(gridC.y-1, 0, voldimI.y-1), gridC.z);
    gradTarget.y = targetVol_D[voldim.x*(voldim.y*x1.z + x1.y) + x1.x]-
            targetVol_D[voldim.x*(voldim.y*x2.z + x2.y) + x2.x];

    x1 = make_uint3(gridC.x, gridC.y, clamp(gridC.z+1, 0, voldimI.z-1));
    x2 = make_uint3(gridC.x, gridC.y, clamp(gridC.z-1, 0, voldimI.z-1));
    gradTarget.z = targetVol_D[voldim.x*(voldim.y*x1.z + x1.y) + x1.x]-
            targetVol_D[voldim.x*(voldim.y*x2.z + x2.y) + x2.x];

    len = length(gradTarget);
    if (len > 0.0) gradTarget /= len;


    /* Compute final gradient sample based on active cells */

    float3 gradFinal;
    gradFinal.x = activeSource*gradSource.x + activeTarget*gradTarget.x;
    gradFinal.y = activeSource*gradSource.y + activeTarget*gradTarget.y;
    gradFinal.z = activeSource*gradSource.z + activeTarget*gradTarget.z;

    len = length(gradFinal);
    if (len > 0.0) gradFinal /= len;

    grad_D[4*idx+0] = gradFinal.x;
    grad_D[4*idx+1] = gradFinal.y;
    grad_D[4*idx+2] = gradFinal.z;
}


/*
 * DiffusionSolver::initGVFCuda
 */
bool DiffusionSolver::initTwoWayGVF(
        const float *sourceVol_D,
        const float *targetVol_D,
        size_t dim[3],
        const unsigned int *cellStatesSource_D,
        const unsigned int *cellStatesTarget_D,
        float isovalue,
        float *grad_D,
        float *gvfConstData_D) {

    size_t volsize = dim[0]*dim[1]*dim[2];
    uint3 voldim = make_uint3(dim[0], dim[1], dim[2]);

#ifdef USE_CUDA_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Calculate gradient using finite differences
    calcTwoWayGradient_D <<< DiffusionSolver::Grid(volsize, 256), 256 >>> (
            sourceVol_D,
            targetVol_D,
            grad_D,
            cellStatesSource_D,
            cellStatesTarget_D,
            voldim,
            isovalue);

#ifdef USE_CUDA_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'calcGradient_D':                        %.10f sec\n",
            dt_ms/1000.0f);
    cudaEventRecord(event1, 0);
#endif

    // Precompute b,c1,c2, and c3
    prepareGVFDiffusion_D <<< DiffusionSolver::Grid(volsize, 256), 256 >>> (
            grad_D, gvfConstData_D, voldim);

#ifdef USE_CUDA_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'prepareGVFDiffusion_D':                 %.10f sec\n",
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
