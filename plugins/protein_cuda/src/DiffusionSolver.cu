//
// DiffusionSolver.cu
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 16, 2013
// Author     : scharnkn
//

#include "stdafx.h"
#include "DiffusionSolver.h"

#include "cuda_error_check.h"
#include "CUDAGrid.cuh"

#include "helper_cuda.h"
#include "helper_math.h"

#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::protein_cuda;

typedef unsigned int uint;

//#define USE_CUDA_TIMER

__constant__ __device__ float isoval_D;  // Isovalue defining the level sets

/*
 * DiffusionSolver_InitGVF_D
 */
__global__ void DiffusionSolver_InitGVF_D(
        const float *volTarget_D,
        const unsigned int *cellStatesTarget_D,
        float *gvfConstData_D) {

    const uint idx = __umul24(__umul24(blockIdx.y, gridDim.x) +
            blockIdx.x, blockDim.x) + threadIdx.x;

    int volsize = gridSize_D.x*gridSize_D.y*gridSize_D.z;
    if (idx >= volsize) return;

    int3 cellC;
    uint cellIdx;
    int activeTarget=0;

    int3 gridC = make_int3(
            idx % (gridSize_D.x),
            (idx / gridSize_D.x) % gridSize_D.y,
            (idx / gridSize_D.x) / gridSize_D.y);


    /* Check neighbor cells in target volume */

    // (-1, -1, -1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, gridSize_D.x-2),
            clamp(gridC.y-1, 0, gridSize_D.y-2),
            clamp(gridC.z-1, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (-1, 1, -1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, gridSize_D.x-2),
            clamp(gridC.y, 0, gridSize_D.y-2),
            clamp(gridC.z-1, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (-1, 1, 1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, gridSize_D.x-2),
            clamp(gridC.y, 0, gridSize_D.y-2),
            clamp(gridC.z, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (-1, -1, 1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, gridSize_D.x-2),
            clamp(gridC.y-1, 0, gridSize_D.y-2),
            clamp(gridC.z, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (1, -1, -1)
    cellC = make_int3(
            clamp(gridC.x, 0, gridSize_D.x-2),
            clamp(gridC.y-1, 0, gridSize_D.y-2),
            clamp(gridC.z-1, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (1, 1, -1)
    cellC = make_int3(
            clamp(gridC.x, 0, gridSize_D.x-2),
            clamp(gridC.y, 0, gridSize_D.y-2),
            clamp(gridC.z-1, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (1, 1, 1)
    cellC = make_int3(
            clamp(gridC.x, 0, gridSize_D.x-2),
            clamp(gridC.y, 0, gridSize_D.y-2),
            clamp(gridC.z, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (1, -1, 1)
    cellC = make_int3(
            clamp(gridC.x, 0, gridSize_D.x-2),
            clamp(gridC.y-1, 0, gridSize_D.y-2),
            clamp(gridC.z, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];


    /* Sample gradients */

    float3 gradTarget;
    int3 x1, x2;

    x1 = make_int3(clamp(gridC.x+1, 0, gridSize_D.x-1), gridC.y, gridC.z);
    x2 = make_int3(clamp(gridC.x-1, 0, gridSize_D.x-1), gridC.y, gridC.z);
    gradTarget.x =
            volTarget_D[gridSize_D.x*(gridSize_D.y*x1.z + x1.y) + x1.x]-
            volTarget_D[gridSize_D.x*(gridSize_D.y*x2.z + x2.y) + x2.x];

    x1 = make_int3(gridC.x, clamp(gridC.y+1, 0, gridSize_D.y-1), gridC.z);
    x2 = make_int3(gridC.x, clamp(gridC.y-1, 0, gridSize_D.y-1), gridC.z);
    gradTarget.y =
            volTarget_D[gridSize_D.x*(gridSize_D.y*x1.z + x1.y) + x1.x]-
            volTarget_D[gridSize_D.x*(gridSize_D.y*x2.z + x2.y) + x2.x];

    x1 = make_int3(gridC.x, gridC.y, clamp(gridC.z+1, 0, gridSize_D.z-1));
    x2 = make_int3(gridC.x, gridC.y, clamp(gridC.z-1, 0, gridSize_D.z-1));
    gradTarget.z =
            volTarget_D[gridSize_D.x*(gridSize_D.y*x1.z + x1.y) + x1.x]-
            volTarget_D[gridSize_D.x*(gridSize_D.y*x2.z + x2.y) + x2.x];

    float len = length(gradTarget);
    if (len > 0.0) gradTarget/= len;


    /* Extract cont data*/

    // Compute len^2
    len = gradTarget.x*gradTarget.x + gradTarget.y*gradTarget.y + gradTarget.z*gradTarget.z;

    // Write b to device memory
    gvfConstData_D[4*idx+0] = len;
    // Write c1, c2, and c3 to device memory
    gvfConstData_D[4*idx+1] = len*gradTarget.x;
    gvfConstData_D[4*idx+2] = len*gradTarget.y;
    gvfConstData_D[4*idx+3] = len*gradTarget.z;
}


/*
 * DiffusionSolver_InitTwoWayGVF_D
 */
__global__ void DiffusionSolver_InitTwoWayGVF_D(
        const float *volSource_D,
        const float *volTarget_D,
        const unsigned int *cellStatesSource_D,
        const unsigned int *cellStatesTarget_D,
        float *gvfConstData_D) {

    const uint idx = __umul24(__umul24(blockIdx.y, gridDim.x) +
            blockIdx.x, blockDim.x) + threadIdx.x;

    int volsize = gridSize_D.x*gridSize_D.y*gridSize_D.z;
    if (idx >= volsize) return;

    int3 cellC;
    uint cellIdx;
    int activeSource=0, activeTarget=0;

    int3 gridC = make_int3(
            idx % (gridSize_D.x),
            (idx / gridSize_D.x) % gridSize_D.y,
            (idx / gridSize_D.x) / gridSize_D.y);


    /* Check neighbor cells in source volume */

    // (-1, -1, -1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, gridSize_D.x-2),
            clamp(gridC.y-1, 0, gridSize_D.y-2),
            clamp(gridC.z-1, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeSource |= cellStatesSource_D[cellIdx];

    // (-1, 1, -1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, gridSize_D.x-2),
            clamp(gridC.y, 0, gridSize_D.y-2),
            clamp(gridC.z-1, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeSource |= cellStatesSource_D[cellIdx];

    // (-1, 1, 1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, gridSize_D.x-2),
            clamp(gridC.y, 0, gridSize_D.y-2),
            clamp(gridC.z, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeSource |= cellStatesSource_D[cellIdx];

    // (-1, -1, 1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, gridSize_D.x-2),
            clamp(gridC.y-1, 0, gridSize_D.y-2),
            clamp(gridC.z, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeSource |= cellStatesSource_D[cellIdx];

    // (1, -1, -1)
    cellC = make_int3(
            clamp(gridC.x, 0, gridSize_D.x-2),
            clamp(gridC.y-1, 0, gridSize_D.y-2),
            clamp(gridC.z-1, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeSource |= cellStatesSource_D[cellIdx];

    // (1, 1, -1)
    cellC = make_int3(
            clamp(gridC.x, 0, gridSize_D.x-2),
            clamp(gridC.y, 0, gridSize_D.y-2),
            clamp(gridC.z-1, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeSource |= cellStatesSource_D[cellIdx];

    // (1, 1, 1)
    cellC = make_int3(
            clamp(gridC.x, 0, gridSize_D.x-2),
            clamp(gridC.y, 0, gridSize_D.y-2),
            clamp(gridC.z, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeSource |= cellStatesSource_D[cellIdx];

    // (1, -1, 1)
    cellC = make_int3(
            clamp(gridC.x, 0, gridSize_D.x-2),
            clamp(gridC.y-1, 0, gridSize_D.y-2),
            clamp(gridC.z, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeSource |= cellStatesSource_D[cellIdx];


    /* Check neighbor cells in target volume */

    // (-1, -1, -1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, gridSize_D.x-2),
            clamp(gridC.y-1, 0, gridSize_D.y-2),
            clamp(gridC.z-1, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (-1, 1, -1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, gridSize_D.x-2),
            clamp(gridC.y, 0, gridSize_D.y-2),
            clamp(gridC.z-1, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (-1, 1, 1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, gridSize_D.x-2),
            clamp(gridC.y, 0, gridSize_D.y-2),
            clamp(gridC.z, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (-1, -1, 1)
    cellC = make_int3(
            clamp(gridC.x-1, 0, gridSize_D.x-2),
            clamp(gridC.y-1, 0, gridSize_D.y-2),
            clamp(gridC.z, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (1, -1, -1)
    cellC = make_int3(
            clamp(gridC.x, 0, gridSize_D.x-2),
            clamp(gridC.y-1, 0, gridSize_D.y-2),
            clamp(gridC.z-1, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (1, 1, -1)
    cellC = make_int3(
            clamp(gridC.x, 0, gridSize_D.x-2),
            clamp(gridC.y, 0, gridSize_D.y-2),
            clamp(gridC.z-1, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (1, 1, 1)
    cellC = make_int3(
            clamp(gridC.x, 0, gridSize_D.x-2),
            clamp(gridC.y, 0, gridSize_D.y-2),
            clamp(gridC.z, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];

    // (1, -1, 1)
    cellC = make_int3(
            clamp(gridC.x, 0, gridSize_D.x-2),
            clamp(gridC.y-1, 0, gridSize_D.y-2),
            clamp(gridC.z, 0, gridSize_D.z-2));
    cellIdx = ::GetCellIdxByGridCoords(cellC);
    activeTarget |= cellStatesTarget_D[cellIdx];


    /* Sample gradients */

    float3 gradSource, gradTarget, gradFinal;
    int3 x1, x2;

    x1 = make_int3(clamp(gridC.x+1, 0, gridSize_D.x-1), gridC.y, gridC.z);
    x2 = make_int3(clamp(gridC.x-1, 0, gridSize_D.x-1), gridC.y, gridC.z);
    gradSource.x =
            volSource_D[gridSize_D.x*(gridSize_D.y*x1.z + x1.y) + x1.x]-
            volSource_D[gridSize_D.x*(gridSize_D.y*x2.z + x2.y) + x2.x];

    x1 = make_int3(gridC.x, clamp(gridC.y+1, 0, gridSize_D.y-1), gridC.z);
    x2 = make_int3(gridC.x, clamp(gridC.y-1, 0, gridSize_D.y-1), gridC.z);
    gradSource.y =
            volSource_D[gridSize_D.x*(gridSize_D.y*x1.z + x1.y) + x1.x]-
            volSource_D[gridSize_D.x*(gridSize_D.y*x2.z + x2.y) + x2.x];

    x1 = make_int3(gridC.x, gridC.y, clamp(gridC.z+1, 0, gridSize_D.z-1));
    x2 = make_int3(gridC.x, gridC.y, clamp(gridC.z-1, 0, gridSize_D.z-1));
    gradSource.z =
            volSource_D[gridSize_D.x*(gridSize_D.y*x1.z + x1.y) + x1.x]-
            volSource_D[gridSize_D.x*(gridSize_D.y*x2.z + x2.y) + x2.x];

    float len = length(gradSource);
    if (len > 0.0) gradSource/= len;

    x1 = make_int3(clamp(gridC.x+1, 0, gridSize_D.x-1), gridC.y, gridC.z);
    x2 = make_int3(clamp(gridC.x-1, 0, gridSize_D.x-1), gridC.y, gridC.z);
    gradTarget.x =
            volTarget_D[gridSize_D.x*(gridSize_D.y*x1.z + x1.y) + x1.x]-
            volTarget_D[gridSize_D.x*(gridSize_D.y*x2.z + x2.y) + x2.x];

    x1 = make_int3(gridC.x, clamp(gridC.y+1, 0, gridSize_D.y-1), gridC.z);
    x2 = make_int3(gridC.x, clamp(gridC.y-1, 0, gridSize_D.y-1), gridC.z);
    gradTarget.y =
            volTarget_D[gridSize_D.x*(gridSize_D.y*x1.z + x1.y) + x1.x]-
            volTarget_D[gridSize_D.x*(gridSize_D.y*x2.z + x2.y) + x2.x];

    x1 = make_int3(gridC.x, gridC.y, clamp(gridC.z+1, 0, gridSize_D.z-1));
    x2 = make_int3(gridC.x, gridC.y, clamp(gridC.z-1, 0, gridSize_D.z-1));
    gradTarget.z =
            volTarget_D[gridSize_D.x*(gridSize_D.y*x1.z + x1.y) + x1.x]-
            volTarget_D[gridSize_D.x*(gridSize_D.y*x2.z + x2.y) + x2.x];

    len = length(gradTarget);
    if (len > 0.0) gradTarget/= len;


    /* Compute final gradient and extract const data*/

    gradFinal = activeSource*(activeSource-0.5*activeTarget)*gradSource +
                activeTarget*(activeTarget-0.5*activeSource)*gradTarget;

    // Compute len^2
    //len = gradFinal.x*gradFinal.x + gradFinal.y*gradFinal.y + gradFinal.z*gradFinal.z;
    len = float(activeSource||activeTarget);

    // Write b to device memory
    gvfConstData_D[4*idx+0] = len;
    // Write c1, c2, and c3 to device memory
    gvfConstData_D[4*idx+1] = len*gradFinal.x;
    gvfConstData_D[4*idx+2] = len*gradFinal.y;
    gvfConstData_D[4*idx+3] = len*gradFinal.z;
}


/*
 * DiffusionSolver_UpdateGVF_D
 */
__global__ void DiffusionSolver_UpdateGVF_D(
        float *gvfIn_D,
        float *gvfOut_D,
        float *gvfConstData_D, // b, c1, c2, c3
        float scl) {

    const uint idx = __umul24(__umul24(blockIdx.y, gridDim.x) +
            blockIdx.x, blockDim.x) + threadIdx.x;

    uint volsize = gridSize_D.x*gridSize_D.y*gridSize_D.z;
    if (idx >= volsize) return;

    /// Get const data from global device memory ///

    float b = gvfConstData_D[4*idx+0];
    float c1 = gvfConstData_D[4*idx+1];
    float c2 = gvfConstData_D[4*idx+2];
    float c3 = gvfConstData_D[4*idx+3];

    float3 gvf, gvfOld, gvfAdj[6];
    //uint idxAdj[6];

    // Get grid coordinates
    int3 gridC = make_int3(
            idx % gridSize_D.x,
           (idx / gridSize_D.x) % gridSize_D.y,
           (idx / gridSize_D.x) / gridSize_D.y);


    /// Update isotropic diffusion for all vector components ///

    // Get adjacent gvf values
    gvfOld = make_float3(gvfIn_D[4*idx+0], gvfIn_D[4*idx+1], gvfIn_D[4*idx+2]);
    uint idxAdj = ::GetPosIdxByGridCoords(make_int3(clamp(int(gridC.x)-1, 0, int(gridSize_D.x-1)), gridC.y, gridC.z));
    gvfAdj[0] = make_float3(gvfIn_D[4*idxAdj+0], gvfIn_D[4*idxAdj+1], gvfIn_D[4*idxAdj+2]);
    idxAdj = ::GetPosIdxByGridCoords(make_int3(clamp(int(gridC.x)+1, 0, int(gridSize_D.x-1)), gridC.y, gridC.z));
    gvfAdj[1] = make_float3(gvfIn_D[4*idxAdj+0], gvfIn_D[4*idxAdj+1], gvfIn_D[4*idxAdj+2]);
    idxAdj = ::GetPosIdxByGridCoords(make_int3(gridC.x, uint(clamp(int(gridC.y)-1, 0, int(gridSize_D.y-1))), gridC.z));
    gvfAdj[2] = make_float3(gvfIn_D[4*idxAdj+0], gvfIn_D[4*idxAdj+1], gvfIn_D[4*idxAdj+2]);
    idxAdj = ::GetPosIdxByGridCoords(make_int3(gridC.x, uint(clamp(int(gridC.y)+1, 0, int(gridSize_D.y-1))), gridC.z));
    gvfAdj[3] = make_float3(gvfIn_D[4*idxAdj+0], gvfIn_D[4*idxAdj+1], gvfIn_D[4*idxAdj+2]);
    idxAdj = ::GetPosIdxByGridCoords(make_int3(gridC.x, gridC.y, uint(clamp(int(gridC.z)-1, 0, int(gridSize_D.z-1)))));
    gvfAdj[4] = make_float3(gvfIn_D[4*idxAdj+0], gvfIn_D[4*idxAdj+1], gvfIn_D[4*idxAdj+2]);
    idxAdj = ::GetPosIdxByGridCoords(make_int3(gridC.x, gridC.y, uint(clamp(int(gridC.z)+1, 0, int(gridSize_D.z-1)))));
    gvfAdj[5] = make_float3(gvfIn_D[4*idxAdj+0], gvfIn_D[4*idxAdj+1], gvfIn_D[4*idxAdj+2]);

//    // Calculate maximum time step to ensure conversion
//    float dt = gridDelta_D.x*gridDelta_D.y*gridDelta_D.z/(scl*6);
//    dt /= 2.0;
//
//    // Compute diffusion
//    gvf.x = (1.0-b*dt)*gvfOld.x;
//    gvf.x += (gvfAdj[0].x + gvfAdj[1].x + gvfAdj[2].x + gvfAdj[3].x +
//              gvfAdj[4].x + gvfAdj[5].x -6*gvfOld.x)*scl;
//    gvf.x += c1*dt;
//
//    gvf.y = (1.0-b*dt)*gvfOld.y;
//    gvf.y += (gvfAdj[0].y + gvfAdj[1].y + gvfAdj[2].y + gvfAdj[3].y +
//            gvfAdj[4].y + gvfAdj[5].y -6*gvfOld.y)*scl;
//    gvf.y += c2*dt;
//
//    gvf.z = (1.0-b*dt)*gvfOld.z;
//    gvf.z += (gvfAdj[0].z + gvfAdj[1].z + gvfAdj[2].z + gvfAdj[3].z +
//            gvfAdj[4].z + gvfAdj[5].z -6*gvfOld.z)*scl;
//    gvf.z += c3*dt;


    // Calculate maximum time step to ensure conversion
    float minStep = min(gridDelta_D.z, min(gridDelta_D.x, gridDelta_D.y))/6.0;
    float dt = minStep*0.5;

    // Compute diffusion
    gvf.x = c1 + (1.0-b)*(gvfOld.x + dt*(gvfAdj[0].x + gvfAdj[1].x + gvfAdj[2].x + gvfAdj[3].x +
            gvfAdj[4].x + gvfAdj[5].x -6*gvfOld.x));

    gvf.y = c2 + (1.0-b)*(gvfOld.y + dt*(gvfAdj[0].y + gvfAdj[1].y + gvfAdj[2].y + gvfAdj[3].y +
            gvfAdj[4].y + gvfAdj[5].y -6*gvfOld.y));

    gvf.z = c3 + (1.0-b)*(gvfOld.z + dt*(gvfAdj[0].z + gvfAdj[1].z + gvfAdj[2].z + gvfAdj[3].z +
            gvfAdj[4].z + gvfAdj[5].z -6*gvfOld.z));

    gvfOut_D[4*idx+0] = gvf.x;
    gvfOut_D[4*idx+1] = gvf.y;
    gvfOut_D[4*idx+2] = gvf.z;
}


/*
 * DiffusionSolver::CalcGVF
 */
bool DiffusionSolver::CalcGVF(
        const float *volTarget_D,
        float *gvfConstData_D,
        const unsigned int *cellStatesTarget_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float *gvfIn_D,
        float *gvfOut_D,
        unsigned int maxIt,
        float scl) {

    using namespace vislib::sys;

    int volsize = volDim.x*volDim.y*volDim.z;
    uint3 voldim = make_uint3(volDim.x, volDim.y, volDim.z);

    // Init constant device parameters
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DiffusionSolver::ClassName());
        return false;
    }

#ifdef USE_CUDA_TIMER
        float dt_ms;
        cudaEvent_t event1, event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);
        cudaEventRecord(event1, 0);
#endif

    // Init diffusion by calculating cont data
        DiffusionSolver_InitGVF_D <<< Grid(volsize, 256), 256 >>> (
            volTarget_D,
            cellStatesTarget_D,
            gvfConstData_D);

#ifdef USE_CUDA_TIMER
        cudaEventRecord(event2, 0);
        cudaEventSynchronize(event1);
        cudaEventSynchronize(event2);
        cudaEventElapsedTime(&dt_ms, event1, event2);
        printf("CUDA time for 'initTwoWayGVF_D':                   %.10f sec\n",
                dt_ms/1000.0f);
#endif

    for (unsigned int it=(maxIt%2); it < maxIt+(maxIt%2); ++it) {

#ifdef USE_CUDA_TIMER
        cudaEventRecord(event1, 0);
#endif

        if (it%2 == 0) {
            // Update diffusion
            DiffusionSolver_UpdateGVF_D <<< Grid(volsize, 256), 256 >>> (
                    gvfIn_D, gvfOut_D, gvfConstData_D, scl);

            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
        } else {
            // Update diffusion
            DiffusionSolver_UpdateGVF_D <<< Grid(volsize, 256), 256 >>> (
                    gvfOut_D, gvfIn_D, gvfConstData_D, scl);

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
        const unsigned int *cellStatesSource_D,
        const unsigned int *cellStatesTarget_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float *gvfConstData_D,
        float *gvfIn_D,
        float *gvfOut_D,
        unsigned int maxIt,
        float scl) {

    using namespace vislib::sys;

    int volsize = volDim.x*volDim.y*volDim.z;
    uint3 voldim = make_uint3(volDim.x, volDim.y, volDim.z);

    // Init constant device parameters
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DiffusionSolver::ClassName());
        return false;
    }

#ifdef USE_CUDA_TIMER
        float dt_ms;
        cudaEvent_t event1, event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);
        cudaEventRecord(event1, 0);
#endif

    // Init diffusion by calculating cont data
        DiffusionSolver_InitTwoWayGVF_D <<< Grid(volsize, 256), 256 >>> (
            volSource_D, volTarget_D, cellStatesSource_D, cellStatesTarget_D,
            gvfConstData_D);

#ifdef USE_CUDA_TIMER
        cudaEventRecord(event2, 0);
        cudaEventSynchronize(event1);
        cudaEventSynchronize(event2);
        cudaEventElapsedTime(&dt_ms, event1, event2);
        printf("CUDA time for 'initTwoWayGVF_D':                   %.10f sec\n",
                dt_ms/1000.0f);
        cudaEventRecord(event1, 0);
#endif

    for (unsigned int it=(maxIt%2); it < maxIt+(maxIt%2); ++it) {

#ifdef USE_CUDA_TIMER
        cudaEventRecord(event1, 0);
#endif
        if (it%2 == 0) {
            // Update diffusion
            DiffusionSolver_UpdateGVF_D <<< Grid(volsize, 256), 256 >>> (
                    gvfIn_D, gvfOut_D, gvfConstData_D, scl);

            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
        } else {
            // Update diffusion
            DiffusionSolver_UpdateGVF_D <<< Grid(volsize, 256), 256 >>> (
                    gvfOut_D, gvfIn_D, gvfConstData_D, scl);

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
#endif
    }

    return true;
}
