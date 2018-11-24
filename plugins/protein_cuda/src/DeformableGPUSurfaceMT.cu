//
// DeformableGPUSurfaceMT.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 17, 2013
// Author     : scharnkn
//
#include "stdafx.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "DeformableGPUSurfaceMT.h"

//#ifndef CUDA_NO_SM_11_ATOMIC_INTRINSICS
//    printf("WARNING! Not using atomics!\n");
//#endif

#include "ogl_error_check.h"
#include "cuda_error_check.h"
#include "HostArr.h"
#include "DiffusionSolver.h"
#include "CUDAGrid.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#define WGL_NV_gpu_affinity
#include <cuda_gl_interop.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#include "vislib/Array.h"
#include "vislib/math/Vector.h"

//#define USE_TIMER

using namespace megamol;
using namespace megamol::protein_cuda;


/**
 * Samples the field at a given position using linear interpolation.
 *
 * @param pos The position
 * @return The sampled value of the field
 */
float4 SampleFieldAtPosTrilin(
        float pos[3],
        float4 *field,
        float gridOrg[3],
        float gridDelta[3],
        int gridSize[3]) {

    int cell[3];
    float x[3];

    // Get id of the cell containing the given position and interpolation
    // coefficients
    x[0] = (pos[0]-gridOrg[0])/gridDelta[0];
    x[1] = (pos[1]-gridOrg[1])/gridDelta[1];
    x[2] = (pos[2]-gridOrg[2])/gridDelta[2];
    cell[0] = (int)(x[0]);
    cell[1] = (int)(x[1]);
    cell[2] = (int)(x[2]);
    x[0] = x[0]-(float)cell[0]; // alpha
    x[1] = x[1]-(float)cell[1]; // beta
    x[2] = x[2]-(float)cell[2]; // gamma

    float alpha = x[0];
    float beta = x[1];
    float gamma = x[2];

    cell[0] = std::min(std::max(cell[0], int(0)), gridSize[0]-2);
    cell[1] = std::min(std::max(cell[1], int(0)), gridSize[1]-2);
    cell[2] = std::min(std::max(cell[2], int(0)), gridSize[2]-2);

    // Get values at corners of current cell
    float4 n0, n1, n2, n3, n4, n5, n6, n7;
//    printf("dim %i %i %i\n", gridSize[0], gridSize[1], gridSize[2]);
//    printf("cell %i %i %i\n", cell[0], cell[1], cell[2]);

    size_t fieldSize =gridSize[0]*gridSize[1]*gridSize[2];
    if (gridSize[0]*(gridSize[1]*(cell[2]+0) + (cell[1]+0))+cell[0]+0 > fieldSize) {
        printf("Overflow %i\n", gridSize[0]*(gridSize[1]*(cell[2]+0) + (cell[1]+0))+cell[0]+0);
    }
    n0 = field[gridSize[0]*(gridSize[1]*(cell[2]+0) + (cell[1]+0))+cell[0]+0];
    n1 = field[gridSize[0]*(gridSize[1]*(cell[2]+0) + (cell[1]+0))+cell[0]+1];
    n2 = field[gridSize[0]*(gridSize[1]*(cell[2]+0) + (cell[1]+1))+cell[0]+0];
    n3 = field[gridSize[0]*(gridSize[1]*(cell[2]+0) + (cell[1]+1))+cell[0]+1];
    n4 = field[gridSize[0]*(gridSize[1]*(cell[2]+1) + (cell[1]+0))+cell[0]+0];
    n5 = field[gridSize[0]*(gridSize[1]*(cell[2]+1) + (cell[1]+0))+cell[0]+1];
    n6 = field[gridSize[0]*(gridSize[1]*(cell[2]+1) + (cell[1]+1))+cell[0]+0];
    n7 = field[gridSize[0]*(gridSize[1]*(cell[2]+1) + (cell[1]+1))+cell[0]+1];

    float4 a, b, c, d, e, f, g, h;
    a = n0;
    b = n1 - n0;
    c = n2 - n0;
    d = n3 - n1 - n2 + n0;
    e = n4 - n0;
    f = n5 - n1 - n4 + n0;
    g = n6 - n2 - n4 + n0;
    h = n7 - n3 - n5 - n6 + n1 + n2 + n4 - n0;

    return a + b*alpha + c*beta + d*alpha*beta + e*gamma + f*alpha*gamma
            + g*beta*gamma + h*alpha*beta*gamma;

}


float SampleFieldAtPosTrilin(
        float pos[3],
        float *field,
        float gridOrg[3],
        float gridDelta[3],
        int gridSize[3]) {

    int cell[3];
    float x[3];

    // Get id of the cell containing the given position and interpolation
    // coefficients
    x[0] = (pos[0]-gridOrg[0])/gridDelta[0];
    x[1] = (pos[1]-gridOrg[1])/gridDelta[1];
    x[2] = (pos[2]-gridOrg[2])/gridDelta[2];
    cell[0] = (int)(x[0]);
    cell[1] = (int)(x[1]);
    cell[2] = (int)(x[2]);
    x[0] = x[0]-(float)cell[0]; // alpha
    x[1] = x[1]-(float)cell[1]; // beta
    x[2] = x[2]-(float)cell[2]; // gamma

    float alpha = x[0];
    float beta = x[1];
    float gamma = x[2];

    cell[0] = std::min(std::max(cell[0], int(0)), gridSize[0]-2);
    cell[1] = std::min(std::max(cell[1], int(0)), gridSize[1]-2);
    cell[2] = std::min(std::max(cell[2], int(0)), gridSize[2]-2);

    // Get values at corners of current cell
    float n0, n1, n2, n3, n4, n5, n6, n7;
    n0 = field[gridSize[0]*(gridSize[1]*(cell[2]+0) + (cell[1]+0))+cell[0]+0];
    n1 = field[gridSize[0]*(gridSize[1]*(cell[2]+0) + (cell[1]+0))+cell[0]+1];
    n2 = field[gridSize[0]*(gridSize[1]*(cell[2]+0) + (cell[1]+1))+cell[0]+0];
    n3 = field[gridSize[0]*(gridSize[1]*(cell[2]+0) + (cell[1]+1))+cell[0]+1];
    n4 = field[gridSize[0]*(gridSize[1]*(cell[2]+1) + (cell[1]+0))+cell[0]+0];
    n5 = field[gridSize[0]*(gridSize[1]*(cell[2]+1) + (cell[1]+0))+cell[0]+1];
    n6 = field[gridSize[0]*(gridSize[1]*(cell[2]+1) + (cell[1]+1))+cell[0]+0];
    n7 = field[gridSize[0]*(gridSize[1]*(cell[2]+1) + (cell[1]+1))+cell[0]+1];

    float a, b, c, d, e, f, g, h;
    a = n0;
    b = n1 - n0;
    c = n2 - n0;
    d = n3 - n1 - n2 + n0;
    e = n4 - n0;
    f = n5 - n1 - n4 + n0;
    g = n6 - n2 - n4 + n0;
    h = n7 - n3 - n5 - n6 + n1 + n2 + n4 - n0;

    return a + b*alpha + c*beta + d*alpha*beta + e*gamma + f*alpha*gamma
            + g*beta*gamma + h*alpha*beta*gamma;

}


/**
 * 'Safe' inverse sqrt, that prevents dividing by zero
 *
 * @param x The input value
 * @return The inverse sqrt if x>0, 0.0 otherwise
 */
inline __host__ __device__ float safeRsqrtf(float x) {
    if (x > 0.0) {
        return 1.0f/sqrtf(x);
    } else {
        return 0.0f;
    }
}

/**
 * 'Safe' normalize function for float3 that uses safe rsqrt
 *
 * @param v The input vector to be normalized
 * @return The normalized vector v
 */
inline __device__ float safeInvLength(float3 v) {
    return safeRsqrtf(dot(v, v));
}

/**
 * 'Safe' normalize function for float2 that uses safe rsqrt
 *
 * @param v The input vector to be normalized
 * @return The normalized vector v
 */
inline __device__ float2 safeNormalize(float2 v) {
    float invLen = safeRsqrtf(dot(v, v));
    return v * invLen;
}

/**
 * 'Safe' normalize function for float3 that uses safe rsqrt
 *
 * @param v The input vector to be normalized
 * @return The normalized vector v
 */
inline __host__ __device__ float3 safeNormalize(float3 v) {
    float invLen = safeRsqrtf(dot(v, v));
    return v * invLen;
}


////////////////////////////////////////////////////////////////////////////////
//  Inline device functions                                                   //
////////////////////////////////////////////////////////////////////////////////

/**
 * @return Returns the thread index based on the current CUDA grid dimensions
 */
inline __device__ uint getThreadIdx() {
    return __umul24(__umul24(blockIdx.y, gridDim.x) + blockIdx.x, blockDim.x) +
            threadIdx.x;
}


////////////////////////////////////////////////////////////////////////////////
//  Global device functions                                                   //
////////////////////////////////////////////////////////////////////////////////


/**
 * Computes the gradient of a given scalar field using central differences.
 * Border areas are omitted.
 *
 * @param[out] grad_D  The gradient field
 * @param[in]  field_D The scalar field
 */
__global__ void DeformableGPUSurfaceMT_CalcVolGradient_D(float4 *grad_D, float *field_D) {

    const uint idx = ::getThreadIdx();

    // Get grid coordinates
    uint3 gridCoord = make_uint3(
            idx % gridSize_D.x,
            (idx / gridSize_D.x) % gridSize_D.y,
            (idx / gridSize_D.x) / gridSize_D.y);

    // Omit border cells (gradient remains zero)
    if (gridCoord.x == 0) return;
    if (gridCoord.y == 0) return;
    if (gridCoord.z == 0) return;
    if (gridCoord.x >= gridSize_D.x - 1) return;
    if (gridCoord.y >= gridSize_D.y - 1) return;
    if (gridCoord.z >= gridSize_D.z - 1) return;

    float3 grad;

    grad.x =
            field_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x+1, gridCoord.y+0, gridCoord.z+0))]-
            field_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x-1, gridCoord.y+0, gridCoord.z+0))];

    grad.y =
            field_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x, gridCoord.y+1, gridCoord.z+0))]-
            field_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x, gridCoord.y-1, gridCoord.z+0))];

    grad.z =
            field_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x, gridCoord.y+0, gridCoord.z+1))]-
            field_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x, gridCoord.y+0, gridCoord.z-1))];

    grad = safeNormalize(grad);

    grad_D[idx].x = grad.x;
    grad_D[idx].y = grad.y;
    grad_D[idx].z = grad.z;
}


/**
 * Computes the gradient of a given scalar field using central differences.
 * Border areas are omitted.
 *
 * @param[out] grad_D  The gradient field
 * @param[in]  field_D The scalar field
 * @param[in]  field_D The distance field
 */
__global__ void DeformableGPUSurfaceMT_CalcVolGradientWithDistField_D(float4 *grad_D, float *field_D,
        float *distField_D, float minDist, float isovalue) {

    const uint idx = ::getThreadIdx();

    // Get grid coordinates
    uint3 gridCoord = ::GetGridCoordsByPosIdx(idx);

    // Omit border cells (gradient remains zero)
    if (gridCoord.x == 0) return;
    if (gridCoord.y == 0) return;
    if (gridCoord.z == 0) return;
    if (gridCoord.x >= gridSize_D.x - 1) return;
    if (gridCoord.y >= gridSize_D.y - 1) return;
    if (gridCoord.z >= gridSize_D.z - 1) return;

    float distSample = ::SampleFieldAt_D<float, false>(gridCoord, distField_D);
    float volSample = ::SampleFieldAt_D<float, false>(gridCoord, field_D);

    float3 grad = make_float3(0.0, 0.0, 0.0);

    if (distSample > minDist) {
        grad.x =
                distField_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x+1, gridCoord.y+0, gridCoord.z+0))]-
                distField_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x+0, gridCoord.y+0, gridCoord.z+0))];

        grad.y =
                distField_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x, gridCoord.y+1, gridCoord.z+0))]-
                distField_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x, gridCoord.y+0, gridCoord.z+0))];

        grad.z =
                distField_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x, gridCoord.y+0, gridCoord.z+1))]-
                distField_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x, gridCoord.y+0, gridCoord.z+0))];

        if (volSample < isovalue) {
            grad.x *= -1.0;
            grad.y *= -1.0;
            grad.z *= -1.0;
        }

    } else {

        grad.x =
                field_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x+1, gridCoord.y+0, gridCoord.z+0))]-
                field_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x+0, gridCoord.y+0, gridCoord.z+0))];

        grad.y =
                field_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x, gridCoord.y+1, gridCoord.z+0))]-
                field_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x, gridCoord.y+0, gridCoord.z+0))];

        grad.z =
                field_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x, gridCoord.y+0, gridCoord.z+1))]-
                field_D[GetPosIdxByGridCoords(make_uint3(gridCoord.x, gridCoord.y+0, gridCoord.z+0))];
    }


    grad = safeNormalize(grad);

    grad_D[idx].x = grad.x;
    grad_D[idx].y = grad.y;
    grad_D[idx].z = grad.z;
}


/**
 * Computes a distance field based on the vertex positions.
 *
 * @param[in]  vertexPos_D The vertex data buffer (device memory)
 * @param[out] distField_D The distance field buffer (device memory)
 * @param[in]  vertexCnt   The number of vertices
 * @param[in]  dataArrOffs The vertex position offset for the vertex data buffer
 * @param[in]  dataArrSize The stride of the vertex data buffer
 */
__global__ void DeformableGPUSurfaceMT_ComputeDistField_D(
        float *vertexPos_D,
        float *distField_D,
        uint vertexCnt,
        uint dataArrOffs,
        uint dataArrSize) {

    // TODO This is very slow since it basically bruteforces all vertex
    //      positions and stores the distance to the nearest one.

    const uint idx = getThreadIdx();

    if (idx >= gridSize_D.x*gridSize_D.y*gridSize_D.z) {
        return;
    }

    // Get world space position of gridPoint
    uint3 gridCoords = GetGridCoordsByPosIdx(idx);
    float3 latticePos = TransformToWorldSpace(make_float3(
            gridCoords.x,
            gridCoords.y,
            gridCoords.z));

    // Loop through all vertices to find minimal distance
    float3 pos = make_float3(vertexPos_D[0], vertexPos_D[1], vertexPos_D[2]);
    float len;
    len = (latticePos.x-pos.x)*(latticePos.x-pos.x)+
          (latticePos.y-pos.y)*(latticePos.y-pos.y)+
          (latticePos.z-pos.z)*(latticePos.z-pos.z);
    float dist2 = len;


    for (uint i = 0; i < vertexCnt; ++i) {
        pos = make_float3(
                vertexPos_D[dataArrSize*i+dataArrOffs+0],
                vertexPos_D[dataArrSize*i+dataArrOffs+1],
                vertexPos_D[dataArrSize*i+dataArrOffs+2]);
        len = (latticePos.x-pos.x)*(latticePos.x-pos.x)+
              (latticePos.y-pos.y)*(latticePos.y-pos.y)+
              (latticePos.z-pos.z)*(latticePos.z-pos.z);
        dist2 = min(dist2, len);
    }

    distField_D[idx] = sqrt(dist2);
}


/**
 * Writes a flag for every vertex that is adjacent to a corrupt triangles.
 *
 * @param[in,out] vertexData_D              The buffer with the vertex data
 * @param[in]     vertexDataStride          The stride for the vertex data
 *                                          buffer
 * @param[in]     vertexDataOffsPos         The position offset in the vertex
 *                                          data buffer
 * @param[in]     vertexDataOffsCorruptFlag The corruption flag offset in the
 *                                          vertex data buffer
 * @param[in]     triangleVtxIdx_D          Array with triangle vertex indices
 * @param[in]     volume_D                  The target volume defining the
 *                                          iso-surface
 * @param[in]     externalForcesScl_D       Array with the scale factor for the
 *                                          external force
 * @param[in]     triangleCnt               The number of triangles
 * @param[in]     minDispl                  Minimum force scale to keep going
 * @param[in]     isoval                    The iso-value defining the iso-surface
 *
 * TODO
 */
__global__ void DeformableGPUSurfaceMT_FlagCorruptTriangles_D(
        float *vertexFlag_D,
        float *corruptTriangles_D,
        float *vertexData_D,
        uint vertexDataStride,
        uint vertexDataOffsPos,
        uint vertexDataOffsNormal,
        uint *triangleVtxIdx_D,
        float *targetVol_D,
        const unsigned int *targetActiveCells_D,
        float4 *externalForces_D,
        uint triangleCnt,
        float isoval) {

    const uint idx = ::getThreadIdx();
    if (idx >= triangleCnt) {
        return;
    }

    /* Alternative 1: Sample volume at triangle midpoint */

//    const uint baseIdx0 = vertexDataStride*triangleVtxIdx_D[3*idx+0];
//    const uint baseIdx1 = vertexDataStride*triangleVtxIdx_D[3*idx+1];
//    const uint baseIdx2 = vertexDataStride*triangleVtxIdx_D[3*idx+2];
//    const float3 p0 = make_float3(vertexData_D[baseIdx0+vertexDataOffsPos+0],
//                                  vertexData_D[baseIdx0+vertexDataOffsPos+1],
//                                  vertexData_D[baseIdx0+vertexDataOffsPos+2]);
//    const float3 p1 = make_float3(vertexData_D[baseIdx1+vertexDataOffsPos+0],
//                                  vertexData_D[baseIdx1+vertexDataOffsPos+1],
//                                  vertexData_D[baseIdx1+vertexDataOffsPos+2]);
//    const float3 p2 = make_float3(vertexData_D[baseIdx2+vertexDataOffsPos+0],
//                                  vertexData_D[baseIdx2+vertexDataOffsPos+1],
//                                  vertexData_D[baseIdx2+vertexDataOffsPos+2]);
//    // Sample volume at midpoint
//    const float3 midPoint = (p0+p1+p2)/3.0;
//    const float volSampleMidPoint = ::SampleFieldAtPosTricub_D<float>(midPoint, targetVol_D);
//    float flag = float(::fabs(volSampleMidPoint-isoval) > 0.3);
//    vertexFlag_D[triangleVtxIdx_D[3*idx+0]] = flag;
//    vertexFlag_D[triangleVtxIdx_D[3*idx+1]] = flag;
//    vertexFlag_D[triangleVtxIdx_D[3*idx+2]] = flag;

    /* Alternative 2: use area and angles */

//    const uint baseIdx0 = vertexDataStride*triangleVtxIdx_D[3*idx+0];
//    const uint baseIdx1 = vertexDataStride*triangleVtxIdx_D[3*idx+1];
//    const uint baseIdx2 = vertexDataStride*triangleVtxIdx_D[3*idx+2];
//    const float3 p0 = make_float3(
//            vertexData_D[baseIdx0+vertexDataOffsPos+0],
//            vertexData_D[baseIdx0+vertexDataOffsPos+1],
//            vertexData_D[baseIdx0+vertexDataOffsPos+2]);
//    const float3 p1 = make_float3(
//            vertexData_D[baseIdx1+vertexDataOffsPos+0],
//            vertexData_D[baseIdx1+vertexDataOffsPos+1],
//            vertexData_D[baseIdx1+vertexDataOffsPos+2]);
//    const float3 p2 = make_float3(
//            vertexData_D[baseIdx2+vertexDataOffsPos+0],
//            vertexData_D[baseIdx2+vertexDataOffsPos+1],
//            vertexData_D[baseIdx2+vertexDataOffsPos+2]);
//
//    float3 v01 = (p0-p1);
//    float3 v02 = (p0-p2);
//    float3 v10 = (p1-p0);
//    float3 v12 = (p1-p2);
//    float3 v21 = (p2-p1);
//    float3 v20 = (p2-p0);
//
//    // Compute minimum angle
//    float dot0 = acos(dot(normalize(v01), normalize(v02)));
//    float dot1 = acos(dot(normalize(v10), normalize(v12)));
//    float dot2 = acos(dot(normalize(v21), normalize(v20)));
//    float minDot = min(dot0, min(dot1, dot2));
//
//    // Compute area of the triangle
//    float3 midPnt = (p0+p1)*0.5;
//    float3 hVec = p2 - midPnt;
//    float area = length(p0-p1)*length(hVec)*0.5;
//    area = gridDelta_D.x*gridDelta_D.y-1;
//
//    float maxCellFaceArea = gridDelta_D.x*gridDelta_D.y; // Find max grid delta
//
//    //float flag = float((minDot < 0.1)||(area > maxCellFaceArea));
//    float flag = float(minDot < 0.2);
//
//    // TODO Is there no atomic write?
////    vertexFlag_D[triangleVtxIdx_D[3*idx+0]] = float(bool(currFlag0) || bool(flag));
////    vertexFlag_D[triangleVtxIdx_D[3*idx+1]] = float(bool(currFlag1) || bool(flag));
////    vertexFlag_D[triangleVtxIdx_D[3*idx+2]] = float(bool(currFlag2) || bool(flag));
//
//    // DEBUG
//    if (flag == 1.0) {
//        vertexFlag_D[triangleVtxIdx_D[3*idx+0]] = 1.0;
//        vertexFlag_D[triangleVtxIdx_D[3*idx+1]] = 1.0;
//        vertexFlag_D[triangleVtxIdx_D[3*idx+2]] = 1.0;
//    }
//    // END DEBUG
//    corruptTriangles_D[idx] = flag;

//    /* Alternative 3 Check whether the vertex lies in an active cell of the
//       target volume */
//
//    const uint baseIdx0 = vertexDataStride*triangleVtxIdx_D[3*idx+0];
//    const uint baseIdx1 = vertexDataStride*triangleVtxIdx_D[3*idx+1];
//    const uint baseIdx2 = vertexDataStride*triangleVtxIdx_D[3*idx+2];
//    const float3 p0 = make_float3(
//            vertexData_D[baseIdx0+vertexDataOffsPos+0],
//            vertexData_D[baseIdx0+vertexDataOffsPos+1],
//            vertexData_D[baseIdx0+vertexDataOffsPos+2]);
//    const float3 p1 = make_float3(
//            vertexData_D[baseIdx1+vertexDataOffsPos+0],
//            vertexData_D[baseIdx1+vertexDataOffsPos+1],
//            vertexData_D[baseIdx1+vertexDataOffsPos+2]);
//    const float3 p2 = make_float3(
//            vertexData_D[baseIdx2+vertexDataOffsPos+0],
//            vertexData_D[baseIdx2+vertexDataOffsPos+1],
//            vertexData_D[baseIdx2+vertexDataOffsPos+2]);
//
//    // Sample volume at midpoint
//    const float3 midpoint = (p0+p1+p2)/3.0;
//
//    // Get integer cell index
//    int3 coords;
//    coords.x = int((midpoint.x-gridOrg_D.x)/gridDelta_D.x);
//    coords.y = int((midpoint.y-gridOrg_D.y)/gridDelta_D.y);
//    coords.z = int((midpoint.z-gridOrg_D.z)/gridDelta_D.z);
//
//    int cellIDx = ::GetCellIdxByGridCoords(coords);
//    uint cellState = targetActiveCells_D[cellIDx];
//
//    float currFlag0 = vertexFlag_D[triangleVtxIdx_D[3*idx+0]];
//    float currFlag1 = vertexFlag_D[triangleVtxIdx_D[3*idx+1]];
//    float currFlag2 = vertexFlag_D[triangleVtxIdx_D[3*idx+2]];
////    __syncthreads();
////    vertexFlag_D[triangleVtxIdx_D[3*idx+0]] = float(bool(currFlag0) || bool(1-cellState));
////    vertexFlag_D[triangleVtxIdx_D[3*idx+1]] = float(bool(currFlag1) || bool(1-cellState));
////    vertexFlag_D[triangleVtxIdx_D[3*idx+2]] = float(bool(currFlag2) || bool(1-cellState));
////    vertexFlag_D[triangleVtxIdx_D[3*idx+0]] = 1.0;
////    vertexFlag_D[triangleVtxIdx_D[3*idx+1]] = 1.0;
////    vertexFlag_D[triangleVtxIdx_D[3*idx+2]] = 1.0;
//
//
//    corruptTriangles_D[idx] = float(1-cellState);


    /* Alternative 4 Check whether all the vertices lies in an active cell of the
       target volume */

    const uint baseIdx0 = vertexDataStride*triangleVtxIdx_D[3*idx+0];
    const uint baseIdx1 = vertexDataStride*triangleVtxIdx_D[3*idx+1];
    const uint baseIdx2 = vertexDataStride*triangleVtxIdx_D[3*idx+2];
    const float3 p0 = make_float3(
            vertexData_D[baseIdx0+vertexDataOffsPos+0],
            vertexData_D[baseIdx0+vertexDataOffsPos+1],
            vertexData_D[baseIdx0+vertexDataOffsPos+2]);
    const float3 p1 = make_float3(
            vertexData_D[baseIdx1+vertexDataOffsPos+0],
            vertexData_D[baseIdx1+vertexDataOffsPos+1],
            vertexData_D[baseIdx1+vertexDataOffsPos+2]);
    const float3 p2 = make_float3(
            vertexData_D[baseIdx2+vertexDataOffsPos+0],
            vertexData_D[baseIdx2+vertexDataOffsPos+1],
            vertexData_D[baseIdx2+vertexDataOffsPos+2]);

    float3 vec0 = (p1 - p0);
    float3 vec1 = (p2 - p0);
    float3 norm = normalize(cross(vec0, vec1));

    // Sample volume at midpoint
    const float3 midpoint = (p0+p1+p2)/3.0;

    // Sample gradient from external forces
    float4 externalForces = SampleFieldAtPosTrilin_D<float4, false>(midpoint, externalForces_D);
    float3 normField = make_float3(externalForces.x, externalForces.y, externalForces.z);
    float dotNormsAbs = dot(norm, normField);

    // Get integer cell index
    int3 coords;
    coords.x = int((midpoint.x-gridOrg_D.x)/gridDelta_D.x);
    coords.y = int((midpoint.y-gridOrg_D.y)/gridDelta_D.y);
    coords.z = int((midpoint.z-gridOrg_D.z)/gridDelta_D.z);
    int3 coords0;
    coords0.x = int((p0.x-gridOrg_D.x)/gridDelta_D.x);
    coords0.y = int((p0.y-gridOrg_D.y)/gridDelta_D.y);
    coords0.z = int((p0.z-gridOrg_D.z)/gridDelta_D.z);
    int3 coords1;
    coords1.x = int((p1.x-gridOrg_D.x)/gridDelta_D.x);
    coords1.y = int((p1.y-gridOrg_D.y)/gridDelta_D.y);
    coords1.z = int((p1.z-gridOrg_D.z)/gridDelta_D.z);
    int3 coords2;
    coords2.x = int((p2.x-gridOrg_D.x)/gridDelta_D.x);
    coords2.y = int((p2.y-gridOrg_D.y)/gridDelta_D.y);
    coords2.z = int((p2.z-gridOrg_D.z)/gridDelta_D.z);

    int cellIDx = ::GetCellIdxByGridCoords(coords);
    int cellIDx0 = ::GetCellIdxByGridCoords(coords0);
    int cellIDx1 = ::GetCellIdxByGridCoords(coords1);
    int cellIDx2 = ::GetCellIdxByGridCoords(coords2);
    uint cellState = targetActiveCells_D[cellIDx];
    uint cellState0 = targetActiveCells_D[cellIDx0];
    uint cellState1 = targetActiveCells_D[cellIDx1];
    uint cellState2 = targetActiveCells_D[cellIDx2];

//    float currFlag0 = vertexFlag_D[triangleVtxIdx_D[3*idx+0]];
//    float currFlag1 = vertexFlag_D[triangleVtxIdx_D[3*idx+1]];
//    float currFlag2 = vertexFlag_D[triangleVtxIdx_D[3*idx+2]];
//    __syncthreads();
//    vertexFlag_D[triangleVtxIdx_D[3*idx+0]] = float(bool(currFlag0) || bool(1-cellState));
//    vertexFlag_D[triangleVtxIdx_D[3*idx+1]] = float(bool(currFlag1) || bool(1-cellState));
//    vertexFlag_D[triangleVtxIdx_D[3*idx+2]] = float(bool(currFlag2) || bool(1-cellState));
//    vertexFlag_D[triangleVtxIdx_D[3*idx+0]] = 1.0;
//    vertexFlag_D[triangleVtxIdx_D[3*idx+1]] = 1.0;
//    vertexFlag_D[triangleVtxIdx_D[3*idx+2]] = 1.0;

    // Criteria for good triangles
    bool flag = bool(cellState) &&
                bool(cellState0) &&
                bool(cellState1) &&
                bool(cellState2);
               // (dotNormsAbs >= 0);
               //&& (dotNormsAbs <= 0.5);

    corruptTriangles_D[idx] = float(!flag);
}

/**
 * TODO
 * @return Position and path length addition
 */
__device__ float4 UpdateVtxPosSingle_D (
        float3 posStart,                // Starting position
        float4 *gradient_D,             // External forces
        float *targetVol_D,             // The target volume
        float minDisplScl,              // Minimum displacement for convergence
        float forcesScl,                // General scaling factor for forces
        float isovalue) {               // Isovalue

    float3 pos = posStart;

    float sample = SampleFieldAtPosTrilin_D<float, false>(pos, targetVol_D);
    bool outside = sample <= isovalue;
    float extForcesScl;

    if (outside) extForcesScl = 1.0;
    else extForcesScl = -1.0;
    float len = 0.0f;
    bool converged = false;
    int steps = 0;
    const int maxSteps = 3;
    do {
        // Get volume sample
        float sample = SampleFieldAtPosTrilin_D<float, false>(pos, targetVol_D);

        // Switch sign and scale down if necessary
        bool negative = extForcesScl < 0;
        bool outside = sample <= isovalue;
        int switchSign = int((negative && outside)||(!negative && !outside));
        extForcesScl = extForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
        extForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));

        // Get external forces sample and scale
        float4 extForceTmp = SampleFieldAtPosTrilin_D<float4, false>(pos, gradient_D);
        float3 extForce = make_float3(extForceTmp.x, extForceTmp.y, extForceTmp.z);
        extForce = safeNormalize(extForce);
        // Accumulate path
        len += extForcesScl*forcesScl;
        extForce *= extForcesScl*forcesScl;

        // Propagate vertex and increase path length
        pos += extForce;


        if (length(extForce) <= minDisplScl) {
            converged = true;
        }
        steps++;

    } while (!converged || steps < maxSteps);

    return make_float4(pos.x, pos.y, pos.z, len);
}


/**
 * TODO
 */
__device__ float DeformableGPUSurfaceMT_IntUncertaintyOverCorruptAreaRec_D(
        float3 pos1, float3 pos2, float3 pos3, // Vertex positions of the triangle
        float len1, float len2, float len3,    // Vertex path lengths of the triangle
        float4 *gradient_D,                    // External forces
        float *targetVol_D,                    // The target volume
        unsigned int *targetActiveCells_D,     // Active cells of the target volume
        float minDisplScl,                     // Minimum displacement for convergence
        float forcesScl,                       // General scaling factor for forces
        float isovalue,                        // Isovalue
        float &triArea,
        uint depth
        ) {

    const uint maxDepth = 2;

    // 1. Propagate vertices until they converge to a fixed position

    float4 newPosLen1, newPosLen2, newPosLen3;
    newPosLen1 = UpdateVtxPosSingle_D (pos1, gradient_D, targetVol_D,
            minDisplScl, forcesScl, isovalue);
    newPosLen2 = UpdateVtxPosSingle_D (pos2, gradient_D, targetVol_D,
            minDisplScl, forcesScl, isovalue);
    newPosLen3 = UpdateVtxPosSingle_D (pos3, gradient_D, targetVol_D,
            minDisplScl, forcesScl, isovalue);
    float3 newPos1, newPos2, newPos3;
    newPos1 = make_float3(newPosLen1.x, newPosLen1.y, newPosLen1.z);
    newPos2 = make_float3(newPosLen2.x, newPosLen2.y, newPosLen2.z);
    newPos3 = make_float3(newPosLen3.x, newPosLen3.y, newPosLen3.z);

    // 2. Check whether the resulting triangle is valid

    float3 midpoint = (newPos1+newPos2+newPos3)/3.0;
    int3 coords;
    coords.x = int((midpoint.x-gridOrg_D.x)/gridDelta_D.x);
    coords.y = int((midpoint.y-gridOrg_D.y)/gridDelta_D.y);
    coords.z = int((midpoint.z-gridOrg_D.z)/gridDelta_D.z);
    int cellIDx = ::GetCellIdxByGridCoords(coords);
    uint cellState = targetActiveCells_D[cellIDx];

    if ((cellState == 1)||(depth >= maxDepth)) {

//        printf("%.16f;%.16f;%.16f;%.16f;%.16f;%.16f;%.16f;%.16f;%.16f\n",
//                newPos1.x, newPos1.y, newPos1.z,
//                newPos2.x, newPos2.y, newPos2.z,
//                newPos3.x, newPos3.y, newPos3.z);

//        if (depth >= 2) printf("Thread %u, depth %u\n",::getThreadIdx(), depth);

        // 3a. Cell is active, therefore triangle is valid
        // --> Compute integrated uncertainty value
        // Get triangle area
        float a = length(newPos1 - newPos2);
        float b = length(newPos1 - newPos3);
        float c = length(newPos2 - newPos3);

        // Compute area (Heron's formula)
        float rad = (a + b - c)*(c + a - b)*(a + b + c)*(b + c - a);
        // Make sure radicand is not negative
        rad = rad > 0.0f ? rad : 0.0f;
        float area = 0.25f*sqrt(rad);
        triArea = area;

        // Get average value
        float avgValue = (len1+newPosLen1.w+len2+newPosLen2.w+len3+newPosLen3.w)/3.0f;

        // Approximate integration
        return triArea*avgValue;
    } else {
        float triArea1, triArea2, triArea3, triArea4;
        // 3b. Cell is not active, therefore, triangle is not valid
        // --> Subdivide and call recursively

        float3 p12 = (newPos1+newPos2)/2.0;
        float3 p13 = (newPos1+newPos3)/2.0;
        float3 p32 = (newPos3+newPos2)/2.0;
        float l12 = (len1+newPosLen1.w+len2+newPosLen2.w)/2.0;
        float l13 = (len1+newPosLen1.w+len3+newPosLen3.w)/2.0;
        float l32 = (len3+newPosLen3.w+len2+newPosLen2.w)/2.0;

        float intUncertainty1 =
                DeformableGPUSurfaceMT_IntUncertaintyOverCorruptAreaRec_D(
                        newPos1, p12, p13,
                        len1+newPosLen1.w, l12, l13,
                        gradient_D, targetVol_D, targetActiveCells_D,
                        minDisplScl, forcesScl, isovalue, triArea1,
                        depth+1);

        float intUncertainty2 =
                DeformableGPUSurfaceMT_IntUncertaintyOverCorruptAreaRec_D(
                        p13, p32, newPos3,
                        l13, l32, len3+newPosLen3.w,
                        gradient_D, targetVol_D, targetActiveCells_D,
                        minDisplScl, forcesScl, isovalue, triArea2,
                        depth+1);

        float intUncertainty3 =
                DeformableGPUSurfaceMT_IntUncertaintyOverCorruptAreaRec_D(
                        p12, p13, p32,
                        l12, l13, l32,
                        gradient_D, targetVol_D, targetActiveCells_D,
                        minDisplScl, forcesScl, isovalue, triArea3,
                        depth+1);

        float intUncertainty4 =
                DeformableGPUSurfaceMT_IntUncertaintyOverCorruptAreaRec_D(
                        p12, p32, newPos2,
                        l12, l32, len2+newPosLen2.w,
                        gradient_D, targetVol_D, targetActiveCells_D,
                        minDisplScl, forcesScl, isovalue, triArea4,
                        depth+1);


        triArea = triArea1 + triArea2 + triArea3 + triArea4;

        return intUncertainty1 + intUncertainty2 + intUncertainty3 + intUncertainty4;
    }
}


/**
 * TODO
 */
__global__ void DeformableGPUSurfaceMT_IntUncertaintyOverCorruptArea_D(
        float *corruptTriangles_D,
        float *vertexData_D,
        float *vertexPathLen_D,
        uint vertexDataStride,
        uint vertexDataOffsPos,
        uint vertexDataOffsNormal,
        uint *triangleVtxIdx_D,
        float *targetVol_D,
        float4 *gradient_D,
        unsigned int *targetActiveCells_D,
        uint triangleCnt,
        float isovalue,
        float minDisplScl,
        float forcesScl,
        float *corruptTrianglesIntUncertainty_D,
        float *trianglesArea_D) {

    const uint idx = ::getThreadIdx();
    if (idx >= triangleCnt) {
        return;
    }

    // Triangle is not corrupt
    if (corruptTriangles_D[idx] == 0) {
        return;
    }

    // Get initial positions from main memory
    uint baseIdx0 = vertexDataStride*triangleVtxIdx_D[3*idx+0];
    uint baseIdx1 = vertexDataStride*triangleVtxIdx_D[3*idx+1];
    uint baseIdx2 = vertexDataStride*triangleVtxIdx_D[3*idx+2];
    float3 pos1 = make_float3(
            vertexData_D[baseIdx0+vertexDataOffsPos+0],
            vertexData_D[baseIdx0+vertexDataOffsPos+1],
            vertexData_D[baseIdx0+vertexDataOffsPos+2]);
    float3 pos2 = make_float3(
            vertexData_D[baseIdx1+vertexDataOffsPos+0],
            vertexData_D[baseIdx1+vertexDataOffsPos+1],
            vertexData_D[baseIdx1+vertexDataOffsPos+2]);
    float3 pos3 = make_float3(
            vertexData_D[baseIdx2+vertexDataOffsPos+0],
            vertexData_D[baseIdx2+vertexDataOffsPos+1],
            vertexData_D[baseIdx2+vertexDataOffsPos+2]);

    // Get initial path lengths from previous morphing
    float len1 = vertexPathLen_D[triangleVtxIdx_D[3*idx+0]];
    float len2 = vertexPathLen_D[triangleVtxIdx_D[3*idx+1]];
    float len3 = vertexPathLen_D[triangleVtxIdx_D[3*idx+2]];

    float triArea = 0.0;

    // Integrate path lengths
    float intUncertainty = DeformableGPUSurfaceMT_IntUncertaintyOverCorruptAreaRec_D(
            pos1, pos2, pos3,        // Vertex positions of the triangle
            len1, len2, len3,        // Vertex path lengths of the triangle
            gradient_D,              // External forces
            targetVol_D,             // The target volume
            targetActiveCells_D,     // Active cells of the target volume
            minDisplScl,             // Minimum displacement for convergence
            forcesScl,               // General scaling factor for forces
            isovalue,                // Isovalue
            triArea,                 // Area associated with this triangle
            0                        // Initial recursion depth
    );

    corruptTrianglesIntUncertainty_D[idx] = intUncertainty;
    trianglesArea_D[idx] = triArea;


}


/**
 * Initializes the scale factor for the external forces with either -1.0 (if the
 * starting position of the vector is inside the isosurface, or 1.0 (vice
 * versa).
 *
 * @param[in] arr_D       The external forces data buffer
 * @param[in] volume_D    The volume the isosurface is extracted from
 * @param[in] vertexPos_D The vertex data buffer
 * @param[in] nElements   The number of vertices
 * @param[in] isoval      The isovalue that defines the isosurface
 * @param[in] dataArrOffs The offset for vertex positions in the vertex
 *                        data buffer
 * @param[in] dataArrSize The stride of the vertex data buffer TODO
 */
__global__ void DeformableGPUSurfaceMT_InitExternalForceScl_D (
        float *arr_D,
        float *displLen_D,
        float *volume_D,
        float *vertexPos_D,
        float minDispl,
        uint nElements,
        float isoval,
        uint dataArrOffs,
        uint dataArrSize) {

    const uint idx = getThreadIdx();

    if (idx >= nElements) {
        return;
    }

    float3 pos = make_float3(
            vertexPos_D[dataArrSize*idx+dataArrOffs+0],
            vertexPos_D[dataArrSize*idx+dataArrOffs+1],
            vertexPos_D[dataArrSize*idx+dataArrOffs+2]);

    // If the sampled value is smaller than isoval, we are outside the
    // isosurface TODO Make this smarter
    if (SampleFieldAtPosTrilin_D<float, false>(pos, volume_D) <= isoval) {
        arr_D[idx] = 1.0;
    } else {
        arr_D[idx] = -1.0;
    }

    // Init last displ scl with something bigger then minDispl;
    displLen_D[idx] = minDispl + 0.1;
}


__global__ void DeformableGPUSurfaceMT_MeshLaplacian_D(
        float *in_D,
        uint inOffs,
        uint inStride,
        int *vertexNeighbours_D,
        uint maxNeighbours,
        uint vertexCnt,
        float *out_D,
        uint outOffs,
        uint outStride) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    // Get initial position from global device memory
     float3 inOwn = make_float3(
             in_D[inStride*idx+inOffs+0],
             in_D[inStride*idx+inOffs+1],
             in_D[inStride*idx+inOffs+2]);

    uint activeNeighbourCnt = 0;
    float3 out = make_float3(0.0, 0.0, 0.0);
    for(int i = 0; i < maxNeighbours; ++i) {
        int isIdxValid = int(vertexNeighbours_D[maxNeighbours*idx+i] >= 0); // Check if idx != -1
        float3 in;
        int tmpIdx = isIdxValid*vertexNeighbours_D[maxNeighbours*idx+i]; // Map negative indices to 0
        in.x = in_D[inStride*tmpIdx+inOffs+0];
        in.y = in_D[inStride*tmpIdx+inOffs+1];
        in.z = in_D[inStride*tmpIdx+inOffs+2];
        out += (in - inOwn)*isIdxValid;
        activeNeighbourCnt += 1.0f*isIdxValid;
    }
    out /= activeNeighbourCnt;

    out_D[outStride*idx+outOffs+0] = out.x;
    out_D[outStride*idx+outOffs+1] = out.y;
    out_D[outStride*idx+outOffs+2] = out.z;
}


/**
 * Updates the positions of all vertices based on external and internal forces.
 * The external force is computed on the fly based on a the given volume.
 * Samples are aquired using tricubic interpolation.
 *
 * @param[in]      targetVolume_D         The volume the isosurface is extracted
 *                                        from
 * @param[in,out]  vertexPosMapped_D      The vertex data buffer
 * @param[in]      vertexExternalForces_D The external force and scale factor
 *                                        (in 'w') for all vertices
 * @param[in]      vertexNeighbours_D     The neighbour indices of all vertices
 * @param[in]      gradient_D             Array with the gradient
 * @param[in]      vtxNormals_D           The current normals of all vertices
 * @param[in]      vertexCount            The number of vertices
 * @param[in]      externalWeight         Weighting factor for the external
 *                                        forces. The factor for internal forces
 *                                        is implicitely defined by
 *                                        1.0-'externalWeight'
 * @param[in]      forcesScl              General scale factor for the final
 *                                        combined force
 * @param[in]      stiffness              The stiffness of the springs defining
 *                                        the internal forces
 * @param[in]      isoval                 The isovalue defining the isosurface
 * @param[in]      minDispl               The minimum displacement for the
 *                                        vertices to be updated
 * @param[in]      dataArrOffs            The vertex position offset in the
 *                                        vertex data buffer
 * @param[in]      dataArrSize            The stride of the vertex data buffer TODO
 */
__global__ void DeformableGPUSurfaceMT_UpdateVtxPos_D(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        float *displLen_D,
        float *vtxUncertainty_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        float3 *laplacian2_D,
        uint vertexCnt,
        float externalWeight,
        float forcesScl,
        float stiffness,
        float isoval,
        float minDispl,
        bool useCubicInterpolation,
        bool trackPath,
        uint dataArrOffsPos,
        uint dataArrOffsNormal,
        uint dataArrSize) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    // Check convergence criterion
    float lastDisplLen = displLen_D[idx];
    if (lastDisplLen <= minDispl) return; // Vertex is converged

    const uint posBaseIdx = dataArrSize*idx+dataArrOffsPos;


    /* Retrieve stuff from global device memory */

    // Get initial position from global device memory
    float3 posOld = make_float3(
            vertexPosMapped_D[posBaseIdx+0],
            vertexPosMapped_D[posBaseIdx+1],
            vertexPosMapped_D[posBaseIdx+2]);

    // Get initial scale factor for external forces
    float externalForcesScl = vertexExternalForcesScl_D[idx];

    // Get partial derivatives
    float3 laplacian = laplacian_D[idx];
    float3 laplacian2 = laplacian2_D[idx];


    /* Update position */

    // No warp divergence here, since useCubicInterpolation is the same for all
    // threads
    const float sampleDens = useCubicInterpolation
                    ? SampleFieldAtPosTricub_D<float, false>(posOld, targetVolume_D)
                    : SampleFieldAtPosTrilin_D<float, false>(posOld, targetVolume_D);

    // Switch sign and scale down if necessary
    bool negative = externalForcesScl < 0;
    bool outside = sampleDens <= isoval;
    int switchSign = int((negative && outside)||(!negative && !outside));
    externalForcesScl = externalForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
    externalForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));
    //externalForcesScl *= (1.0*(1-switchSign) + (switchSign));

    // Sample gradient by cubic interpolation
    float4 externalForceTmp = useCubicInterpolation
            ? SampleFieldAtPosTricub_D<float4, false>(posOld, gradient_D)
            : SampleFieldAtPosTrilin_D<float4, false>(posOld, gradient_D);

    float3 externalForce;
    externalForce.x = externalForceTmp.x;
    externalForce.y = externalForceTmp.y;
    externalForce.z = externalForceTmp.z;

   // externalForce = safeNormalize(externalForce);
    externalForce *= forcesScl*externalForcesScl*externalWeight;

    float3 internalForce = (1.0-externalWeight)*forcesScl*((1.0 - stiffness)*laplacian - stiffness*laplacian2);

    // Umbrella internal force
    float3 force = externalForce + internalForce;
    float3 posNew = posOld + force;

    /* Write back to global device memory */

    // New pos
    vertexPosMapped_D[posBaseIdx+0] = posNew.x;
    vertexPosMapped_D[posBaseIdx+1] = posNew.y;
    vertexPosMapped_D[posBaseIdx+2] = posNew.z;

    // Write external forces scale factor back to global device memory
    vertexExternalForcesScl_D[idx] = externalForcesScl;

    // No branching occurs here, since the parameter is set globally
    float3 diff = posNew-posOld;
    float diffLen = length(diff);
    //float diffLenInternal = length(forcesScl*((1.0 - stiffness)*laplacian - stiffness*laplacian2));
    if ((trackPath)&&(abs(externalForcesScl) == 1.0f)) {
        //vtxUncertainty_D[idx] += length(externalForce);
        vtxUncertainty_D[idx] += diffLen;
    }
    // Displ scl for convergence
    displLen_D[idx] = diffLen;
}


/**
 * Updates the positions of all vertices based on external and internal forces.
 * The external force is computed on the fly based on a the given volume.
 * Samples are aquired using tricubic interpolation.
 *
 * @param[in]      targetVolume_D         The volume the isosurface is extracted
 *                                        from
 * @param[in,out]  vertexPosMapped_D      The vertex data buffer
 * @param[in]      vertexExternalForces_D The external force and scale factor
 *                                        (in 'w') for all vertices
 * @param[in]      vertexNeighbours_D     The neighbour indices of all vertices
 * @param[in]      gradient_D             Array with the gradient
 * @param[in]      vtxNormals_D           The current normals of all vertices
 * @param[in]      vertexCount            The number of vertices
 * @param[in]      externalWeight         Weighting factor for the external
 *                                        forces. The factor for internal forces
 *                                        is implicitely defined by
 *                                        1.0-'externalWeight'
 * @param[in]      forcesScl              General scale factor for the final
 *                                        combined force
 * @param[in]      stiffness              The stiffness of the springs defining
 *                                        the internal forces
 * @param[in]      isoval                 The isovalue defining the isosurface
 * @param[in]      minDispl               The minimum displacement for the
 *                                        vertices to be updated
 * @param[in]      dataArrOffs            The vertex position offset in the
 *                                        vertex data buffer
 * @param[in]      dataArrSize            The stride of the vertex data buffer TODO
 */
__global__ void DeformableGPUSurfaceMT_UpdateVtxPosNoThinPlate_D(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        float *displLen_D,
        float *vtxUncertainty_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        uint vertexCnt,
        float externalWeight,
        float forcesScl,
        float isoval,
        float minDispl,
        bool useCubicInterpolation,
        bool trackPath,
        uint dataArrOffsPos,
        uint dataArrOffsNormal,
        uint dataArrSize) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    // Check convergence criterion
    float lastDisplLen = displLen_D[idx];
    if (lastDisplLen <= minDispl) {
        displLen_D[idx] = 0.0;
        return; // Vertex is converged
    }

    const uint posBaseIdx = dataArrSize*idx+dataArrOffsPos;


    /* Retrieve stuff from global device memory */

    // Get initial position from global device memory
    float3 posOld = make_float3(
            vertexPosMapped_D[posBaseIdx+0],
            vertexPosMapped_D[posBaseIdx+1],
            vertexPosMapped_D[posBaseIdx+2]);

    // Get initial scale factor for external forces
    float externalForcesScl = vertexExternalForcesScl_D[idx];

    // Get partial derivatives
    float3 laplacian = laplacian_D[idx];


    /* Update position */

    // No warp divergence here, since useCubicInterpolation is the same for all
    // threads
    const float sampleDens = useCubicInterpolation
                    ? SampleFieldAtPosTricub_D<float, false>(posOld, targetVolume_D)
                    : SampleFieldAtPosTrilin_D<float, false>(posOld, targetVolume_D);

    // Switch sign and scale down if necessary
    bool negative = externalForcesScl < 0;
    bool outside = sampleDens <= isoval;
    int switchSign = int((negative && outside)||(!negative && !outside));
    externalForcesScl = externalForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
    externalForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));
    //externalForcesScl *= (1.0*(1-switchSign) + (switchSign));

    // Sample gradient by cubic interpolation
    float4 externalForceTmp = useCubicInterpolation
            ? SampleFieldAtPosTricub_D<float4, false>(posOld, gradient_D)
            : SampleFieldAtPosTrilin_D<float4, false>(posOld, gradient_D);

    float3 externalForce;
    externalForce.x = externalForceTmp.x;
    externalForce.y = externalForceTmp.y;
    externalForce.z = externalForceTmp.z;

   // externalForce = safeNormalize(externalForce);
    externalForce *= forcesScl*externalForcesScl*externalWeight;

    float3 internalForce = (1.0-externalWeight)*forcesScl*laplacian;

    // Umbrella internal force
    float3 force = externalForce + internalForce;
    float3 posNew = posOld + force;

    /* Write back to global device memory */

    // New pos
    vertexPosMapped_D[posBaseIdx+0] = posNew.x;
    vertexPosMapped_D[posBaseIdx+1] = posNew.y;
    vertexPosMapped_D[posBaseIdx+2] = posNew.z;

    // Write external forces scale factor back to global device memory
    vertexExternalForcesScl_D[idx] = externalForcesScl;

    // No branching occurs here, since the parameter is set globally
    float3 diff = posNew-posOld;
    float diffLen = length(diff);
    //float diffLenInternal = length(forcesScl*((1.0 - stiffness)*laplacian - stiffness*laplacian2));
    if ((trackPath)&&(abs(externalForcesScl) == 1.0f)) {
        //vtxUncertainty_D[idx] += length(externalForce);
        vtxUncertainty_D[idx] += diffLen;
    }
    // Displ scl for convergence
    displLen_D[idx] = diffLen;
}



/**
 * Updates the positions of all vertices based on external and internal forces.
 * The external force is computed on the fly based on a the given volume.
 * Samples are aquired using tricubic interpolation.
 *
 * @param[in]      targetVolume_D         The volume the isosurface is extracted
 *                                        from
 * @param[in,out]  vertexPosMapped_D      The vertex data buffer
 * @param[in]      vertexExternalForces_D The external force and scale factor
 *                                        (in 'w') for all vertices
 * @param[in]      vertexCount            The number of vertices
 * @param[in]      forcesScl              General scale factor for the final
 *                                        combined force
 * @param[in]      isoval                 The isovalue defining the isosurface
 * @param[in]      minDispl               The minimum displacement for the
 *                                        vertices to be updated
 * @param[in]      dataArrOffs            The vertex position offset in the
 *                                        vertex data buffer
 * @param[in]      dataArrSize            The stride of the vertex data buffer TODO
 */
__global__ void DeformableGPUSurfaceMT_UpdateVtxPosExternalOnly_D(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        float *displLen_D,
        float *vtxUncertainty_D,
        float4 *gradient_D,
        int *accumPath_D,
        uint vertexCnt,
        float forcesScl,
        float isoval,
        float minDispl,
        bool useCubicInterpolation,
        bool trackPath,
        uint dataArrOffsPos,
        uint dataArrOffsNormal,
        uint dataArrSize) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    // Check convergence criterion
    float lastDisplLen = displLen_D[idx];
    if (lastDisplLen <= minDispl) {
        displLen_D[idx] = 0.0;

        return; // Vertex is converged
    }

    const uint posBaseIdx = dataArrSize*idx+dataArrOffsPos;


    /* Retrieve stuff from global device memory */

    // Get initial position from global device memory
    float3 posOld = make_float3(
            vertexPosMapped_D[posBaseIdx+0],
            vertexPosMapped_D[posBaseIdx+1],
            vertexPosMapped_D[posBaseIdx+2]);

    // Get initial scale factor for external forces
    float externalForcesScl = vertexExternalForcesScl_D[idx];

    // Check whether the difflen is to be subtracted or added
    int accumFactorOld = accumPath_D[idx];


    /* Update position */

    // No warp divergence here, since useCubicInterpolation is the same for all
    // threads
    const float sampleDens = useCubicInterpolation
                    ? SampleFieldAtPosTricub_D<float, false>(posOld, targetVolume_D)
                    : SampleFieldAtPosTrilin_D<float, false>(posOld, targetVolume_D);

    // Switch sign and scale down if necessary
    bool negative = externalForcesScl < 0;
    bool outside = sampleDens <= isoval;
    int switchSign = int((negative && outside)||(!negative && !outside));
    externalForcesScl = externalForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
    externalForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));

//    if (bool(switchSign) && (accumPath_D[idx] != 0)) {
//        accumPath_D[idx] = 0;
//    } else if (bool(switchSign) && (accumPath_D[idx] == 0)) {
//        accumPath_D[idx] = 1;
//    }
    // Change to zero if one and to one if zero
    int accumFactorNew = (1-accumFactorOld);
    int accumFactor = switchSign*accumFactorNew + (1-switchSign)*accumFactorOld;


    // Sample gradient by cubic interpolation
    float4 externalForceTmp = useCubicInterpolation
            ? SampleFieldAtPosTricub_D<float4, false>(posOld, gradient_D)
            : SampleFieldAtPosTrilin_D<float4, false>(posOld, gradient_D);
    float3 externalForce;
    externalForce.x = externalForceTmp.x;
    externalForce.y = externalForceTmp.y;
    externalForce.z = externalForceTmp.z;

    //externalForce = safeNormalize(externalForce);
    externalForce = normalize(externalForce);
    externalForce *= forcesScl*externalForcesScl;

    // Umbrella internal force
    float3 posNew = posOld + externalForce;

    /* Write back to global device memory */

    // New pos
    vertexPosMapped_D[posBaseIdx+0] = posNew.x;
    vertexPosMapped_D[posBaseIdx+1] = posNew.y;
    vertexPosMapped_D[posBaseIdx+2] = posNew.z;

    // Write external forces scale factor back to global device memory
    vertexExternalForcesScl_D[idx] = externalForcesScl;

    //float3 diff = posNew-posOld;
    //float diffLen = length(diff);
    float diffLen = abs(forcesScl*externalForcesScl);

    accumPath_D[idx] = accumFactor;

    // No branching since trackpath is equal for all threads
    if (trackPath) {
//        if (accumPath_D[idx] == 0) {
//            vtxUncertainty_D[idx] += diffLen;
//        } else if(accumPath_D[idx] != 0) {
//            vtxUncertainty_D[idx] -= diffLen;
//        }
        vtxUncertainty_D[idx] += (1-accumFactor)*diffLen - accumFactor*diffLen;
    }

    // Displ scl for convergence
    displLen_D[idx] = diffLen;
}



/**
 * Updates the positions of all vertices based on external and internal forces.
 * The external force is computed on the fly based on a the given volume.
 * Samples are aquired using tricubic interpolation.
 *
 * @param[in]      targetVolume_D         The volume the isosurface is extracted
 *                                        from
 * @param[in,out]  vertexPosMapped_D      The vertex data buffer
 * @param[in]      vertexExternalForces_D The external force and scale factor
 *                                        (in 'w') for all vertices
 * @param[in]      vertexCount            The number of vertices
 * @param[in]      forcesScl              General scale factor for the final
 *                                        combined force
 * @param[in]      isoval                 The isovalue defining the isosurface
 * @param[in]      minDispl               The minimum displacement for the
 *                                        vertices to be updated
 * @param[in]      dataArrOffs            The vertex position offset in the
 *                                        vertex data buffer
 * @param[in]      dataArrSize            The stride of the vertex data buffer TODO
 */
__global__ void DeformableGPUSurfaceMT_UpdateVtxPosExternalOnlySubdiv_D(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        float *displLen_D,
        float *vtxUncertainty_D,
        float4 *gradient_D,
        int *accumPath_D,
        float *vertexFlag_D,
        uint vertexCnt,
        float forcesScl,
        float isoval,
        float minDispl,
        bool useCubicInterpolation,
        bool trackPath,
        uint dataArrOffsPos,
        uint dataArrOffsNormal,
        uint dataArrSize) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    // Check convergence criterion
    float lastDisplLen = displLen_D[idx];
//    if ((lastDisplLen <= minDispl)||(vertexFlag_D[idx] == 0.0)) {
//        displLen_D[idx] = 0.0;
//        return; // Vertex is converged
//    }
    if (lastDisplLen <= minDispl) {
        displLen_D[idx] = 0.0;
        return; // Vertex is converged
    }


    const uint posBaseIdx = dataArrSize*idx+dataArrOffsPos;


    /* Retrieve stuff from global device memory */

    // Get initial position from global device memory
    float3 posOld = make_float3(
            vertexPosMapped_D[posBaseIdx+0],
            vertexPosMapped_D[posBaseIdx+1],
            vertexPosMapped_D[posBaseIdx+2]);

    // Get initial scale factor for external forces
    float externalForcesScl = vertexExternalForcesScl_D[idx];

    // Check whether the difflen is to be subtracted or added
    int accumFactorOld = accumPath_D[idx];


    /* Update position */

    // No warp divergence here, since useCubicInterpolation is the same for all
    // threads
    const float sampleDens = useCubicInterpolation
                    ? SampleFieldAtPosTricub_D<float, false>(posOld, targetVolume_D)
                    : SampleFieldAtPosTrilin_D<float, false>(posOld, targetVolume_D);

    // Switch sign and scale down if necessary
    bool negative = externalForcesScl < 0;
    bool outside = sampleDens <= isoval;
    int switchSign = int((negative && outside)||(!negative && !outside));
    externalForcesScl = externalForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
    externalForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));
    //externalForcesScl *= (1.0*(1-switchSign) + (switchSign));

    //    if (bool(switchSign) && (accumPath_D[idx] != 0)) {
    //        accumPath_D[idx] = 0;
    //    } else if (bool(switchSign) && (accumPath_D[idx] == 0)) {
    //        accumPath_D[idx] = 1;
    //    }
    // Change to zero if one and to one if zero
    int accumFactorNew = (1-accumFactorOld);
    int accumFactor = switchSign*accumFactorNew + (1-switchSign)*accumFactorOld;

    // Sample gradient by cubic interpolation
    float4 externalForceTmp = useCubicInterpolation
            ? SampleFieldAtPosTricub_D<float4, false>(posOld, gradient_D)
            : SampleFieldAtPosTrilin_D<float4, false>(posOld, gradient_D);

    float3 externalForce;
    externalForce.x = externalForceTmp.x;
    externalForce.y = externalForceTmp.y;
    externalForce.z = externalForceTmp.z;

    externalForce = safeNormalize(externalForce);
    externalForce *= forcesScl*externalForcesScl;

    // Umbrella internal force
    float3 posNew = posOld + externalForce;

    /* Write back to global device memory */

    // New pos
    vertexPosMapped_D[posBaseIdx+0] = posNew.x;
    vertexPosMapped_D[posBaseIdx+1] = posNew.y;
    vertexPosMapped_D[posBaseIdx+2] = posNew.z;

    // Write external forces scale factor back to global device memory
    vertexExternalForcesScl_D[idx] = externalForcesScl;

    //float3 diff = posNew-posOld;
    //float diffLen = length(diff);
    float diffLen = abs(forcesScl*externalForcesScl);

    accumPath_D[idx] = accumFactor;

    // No branching since trackpath is equal for all threads
    if (trackPath) {
//        if (accumPath_D[idx] == 0) {
//            vtxUncertainty_D[idx] += diffLen;
//        } else if(accumPath_D[idx] != 0) {
//            vtxUncertainty_D[idx] -= diffLen;
//        }
        vtxUncertainty_D[idx] += (1-accumFactor)*diffLen - accumFactor*diffLen;
    }
    // Displ scl for convergence
    displLen_D[idx] = diffLen;
}


/*
 * DeformableGPUSurfaceMT::DeformableGPUSurfaceMT
 */
DeformableGPUSurfaceMT::DeformableGPUSurfaceMT() : GPUSurfaceMT(),
        vboCorruptTriangleVertexFlag(0), vboVtxPath(0), vboVtxAttr(0),
        nFlaggedVertices(0) {

}


/*
 * DeformableGPUSurfaceMT::DeformableGPUSurfaceMT
 */
DeformableGPUSurfaceMT::DeformableGPUSurfaceMT(const DeformableGPUSurfaceMT& other) :
    GPUSurfaceMT(other) {

    CudaSafeCall(this->vertexExternalForcesScl_D.Validate(other.vertexExternalForcesScl_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexExternalForcesScl_D.Peek(),
            other.vertexExternalForcesScl_D.PeekConst(),
            this->vertexExternalForcesScl_D.GetCount()*sizeof(float2),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->externalForces_D.Validate(other.externalForces_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->externalForces_D.Peek(),
            other.externalForces_D.PeekConst(),
            this->externalForces_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->laplacian_D.Validate(other.laplacian_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->laplacian_D.Peek(),
            other.laplacian_D.PeekConst(),
            this->laplacian_D.GetCount()*sizeof(float3),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->laplacian2_D.Validate(other.laplacian2_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->laplacian2_D.Peek(),
            other.laplacian2_D.PeekConst(),
            this->laplacian2_D.GetCount()*sizeof(float3),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->displLen_D.Validate(other.displLen_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->displLen_D.Peek(),
            other.displLen_D.PeekConst(),
            this->displLen_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->accTriangleData_D.Validate(other.accTriangleData_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->accTriangleData_D.Peek(),
            other.accTriangleData_D.PeekConst(),
            this->accTriangleData_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));


    /* Make deep copy of corrupt triangle flag buffer */

    if (other.vboCorruptTriangleVertexFlag) {
        // Destroy if necessary
        if (this->vboCorruptTriangleVertexFlag) {
            glBindBufferARB(GL_ARRAY_BUFFER, this->vboCorruptTriangleVertexFlag);
            glDeleteBuffersARB(1, &this->vboCorruptTriangleVertexFlag);
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
            this->vboCorruptTriangleVertexFlag = 0;
        }

        // Create vertex buffer object for triangle indices
        glGenBuffersARB(1, &this->vboCorruptTriangleVertexFlag);

        CheckForGLError();

        // Map as copy buffer
        glBindBufferARB(GL_COPY_READ_BUFFER, other.vboCorruptTriangleVertexFlag);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, this->vboCorruptTriangleVertexFlag);
        glBufferDataARB(GL_COPY_WRITE_BUFFER,
                sizeof(float)*this->vertexCnt, 0, GL_DYNAMIC_DRAW);
        // Copy data
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                sizeof(float)*this->vertexCnt);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, 0);
        glBindBufferARB(GL_COPY_READ_BUFFER, 0);
        CheckForGLError();
    }

    /* Make deep copy of uncertainty vbo */

    if (other.vboVtxPath) {
        // Destroy if necessary
        if (this->vboVtxPath) {
            glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxPath);
            glDeleteBuffersARB(1, &this->vboVtxPath);
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
            this->vboVtxPath = 0;
        }

        // Create vertex buffer object for triangle indices
        glGenBuffersARB(1, &this->vboVtxPath);

        CheckForGLError();

        // Map as copy buffer
        glBindBufferARB(GL_COPY_READ_BUFFER, other.vboVtxPath);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, this->vboVtxPath);
        glBufferDataARB(GL_COPY_WRITE_BUFFER,
                sizeof(float)*this->vertexCnt, 0, GL_DYNAMIC_DRAW);
        // Copy data
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                sizeof(float)*this->vertexCnt);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, 0);
        glBindBufferARB(GL_COPY_READ_BUFFER, 0);
        CheckForGLError();
    }
}


/*
 * DeformableGPUSurfaceMT::~DeformableGPUSurfaceMT
 */
DeformableGPUSurfaceMT::~DeformableGPUSurfaceMT() {
}


/*
 * ComputeTriangleArea_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeTriangleAreas_D(
        float *trianglesArea_D,
        float *vertexData_D,
        uint *triangleIdx_D,
        uint triangleCnt) {

    const int vertexDataOffsPos = 0;
    //const int vertexDataOffsNormal = 3;
    //const int vertexDataOffsTexCoord = 6;
    const int vertexDataStride = 9;

    const uint idx = ::getThreadIdx();

    if (idx >= triangleCnt) {
        return;
    }

    float3 pos0, pos1, pos2;
    pos0.x = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+0]+vertexDataOffsPos+0];
    pos0.y = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+0]+vertexDataOffsPos+1];
    pos0.z = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+0]+vertexDataOffsPos+2];

    pos1.x = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+1]+vertexDataOffsPos+0];
    pos1.y = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+1]+vertexDataOffsPos+1];
    pos1.z = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+1]+vertexDataOffsPos+2];

    pos2.x = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+2]+vertexDataOffsPos+0];
    pos2.y = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+2]+vertexDataOffsPos+1];
    pos2.z = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+2]+vertexDataOffsPos+2];

    // compute edge lengths
    float a = length(pos0 - pos1);
    float b = length(pos0 - pos2);
    float c = length(pos1 - pos2);

    // Compute area (Heron's formula)
    float rad = (a + b - c)*(c + a - b)*(a + b + c)*(b + c - a);
    // Make sure radicand is not negative
    rad = rad > 0.0f ? rad : 0.0f;
    float area = 0.25f*sqrt(rad);
    trianglesArea_D[idx] = area;
}


/*
 * DeformableGPUSurfaceMT::GetTotalSurfArea
 */
float DeformableGPUSurfaceMT::GetTotalSurfArea() {
    // Compute triangle areas of all (non-corrupt) triangles
    if (!CudaSafeCall(this->accTriangleArea_D.Validate(this->triangleCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->accTriangleArea_D.Set(0x00))) {
        return false;
    }
    cudaGraphicsResource* cudaTokens[2];

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1],
            this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    uint *triangleIdxPt;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&triangleIdxPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    DeformableGPUSurfaceMT_ComputeTriangleAreas_D <<< Grid(this->triangleCnt, 256), 256 >>> (
            this->accTriangleArea_D.Peek(),
            vboPt,
            triangleIdxPt,
            this->triangleCnt);

    ::CheckForCudaErrorSync();

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeTriangleArea_D                   %.10f sec\n",
            dt_ms/1000.0);
#endif

    // Compute sum of all (non-corrupt) triangle areas
    float totalArea = thrust::reduce(
            thrust::device_ptr<float>(this->accTriangleArea_D.Peek()),
            thrust::device_ptr<float>(this->accTriangleArea_D.Peek() + this->triangleCnt));

    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

//    // DEBUG Copy back and accumuluate
//    HostArr<float> accTriangleArea;
//    accTriangleArea.Validate(this->accTriangleArea_D.GetCount());
//    this->accTriangleArea_D.CopyToHost(accTriangleArea.Peek());
//    float sum = 0.0f;
//    for (int i = 0; i < this->accTriangleArea_D.GetCount(); ++i) {
//        sum = sum + accTriangleArea.Peek()[i];
//    }
//    printf("sum: %f, triangles %i\n", sum, this->triangleCnt);
//    return sum;
//    // END DEBUG

    return totalArea;
}


/*
 * ComputeTriangleArea_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeValidTriangleAreas_D(
        float *trianglesArea_D,
        float *vertexData_D,
        uint *triangleIdx_D,
        float *corruptTriFlag_D,
        uint triangleCnt) {

    const int vertexDataOffsPos = 0;
    //const int vertexDataOffsNormal = 3;
    //const int vertexDataOffsTexCoord = 6;
    const int vertexDataStride = 9;

    const uint idx = ::getThreadIdx();

    if (idx >= triangleCnt) {
        return;
    }

    float3 pos0, pos1, pos2;
    pos0.x = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+0]+vertexDataOffsPos+0];
    pos0.y = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+0]+vertexDataOffsPos+1];
    pos0.z = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+0]+vertexDataOffsPos+2];

    pos1.x = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+1]+vertexDataOffsPos+0];
    pos1.y = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+1]+vertexDataOffsPos+1];
    pos1.z = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+1]+vertexDataOffsPos+2];

    pos2.x = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+2]+vertexDataOffsPos+0];
    pos2.y = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+2]+vertexDataOffsPos+1];
    pos2.z = vertexData_D[vertexDataStride*triangleIdx_D[3*idx+2]+vertexDataOffsPos+2];

    // compute edge lengths
    float a = length(pos0 - pos1);
    float b = length(pos0 - pos2);
    float c = length(pos1 - pos2);

    // Compute area (Heron's formula)
    float rad = (a + b - c)*(c + a - b)*(a + b + c)*(b + c - a);
    // Make sure radicand is not negative
    rad = rad > 0.0f ? rad : 0.0f;
    float area = 0.25f*sqrt(rad);
    trianglesArea_D[idx] = area*(1.0-corruptTriFlag_D[idx]);
}


/*
 * DeformableGPUSurfaceMT::GetTotalValidSurfArea
 */
float DeformableGPUSurfaceMT::GetTotalValidSurfArea() {
    // Compute triangle areas of all (non-corrupt) triangles
    if (!CudaSafeCall(this->accTriangleArea_D.Validate(this->triangleCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->accTriangleArea_D.Set(0x00))) {
        return false;
    }
    cudaGraphicsResource* cudaTokens[2];

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1],
            this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    uint *triangleIdxPt;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&triangleIdxPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    DeformableGPUSurfaceMT_ComputeValidTriangleAreas_D <<< Grid(this->triangleCnt, 256), 256 >>> (
            this->accTriangleArea_D.Peek(),
            vboPt,
            triangleIdxPt,
            this->corruptTriangles_D.Peek(),
            this->triangleCnt);

    ::CheckForCudaErrorSync();

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeTriangleArea_D                   %.10f sec\n",
            dt_ms/1000.0);
#endif

    // Compute sum of all (non-corrupt) triangle areas
    float totalArea = thrust::reduce(
            thrust::device_ptr<float>(this->accTriangleArea_D.Peek()),
            thrust::device_ptr<float>(this->accTriangleArea_D.Peek() + this->triangleCnt));

    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

//    // DEBUG Copy back and accumuluate
//    HostArr<float> accTriangleArea;
//    accTriangleArea.Validate(this->accTriangleArea_D.GetCount());
//    this->accTriangleArea_D.CopyToHost(accTriangleArea.Peek());
//    float sum = 0.0f;
//    for (int i = 0; i < this->accTriangleArea_D.GetCount(); ++i) {
//        sum = sum + accTriangleArea.Peek()[i];
//    }
//    printf("sum: %f, triangles %i\n", sum, this->triangleCnt);
//    return sum;
//    // END DEBUG

    return totalArea;
}


/*
 * DeformableGPUSurfaceMT::FlagCorruptTriangles
 */
bool DeformableGPUSurfaceMT::FlagCorruptTriangles(
        float *volume_D,
        const uint *targetActiveCells,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue) {

    using namespace vislib::sys;

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    if (!this->InitCorruptFlagVBO(this->vertexCnt)) {
        return false;
    }

    // Allocate memory for corrupt triangles
    if (!CudaSafeCall(this->corruptTriangles_D.Validate(this->triangleCnt))) {
        return false;
    }
    // Init with zero
    if (!CudaSafeCall(this->corruptTriangles_D.Set(0x00))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    cudaGraphicsResource* cudaTokens[3];

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1],
            this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[2],
            this->vboCorruptTriangleVertexFlag,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    // Map cuda ressource handles
    if (!CudaSafeCall(cudaGraphicsMapResources(3, cudaTokens, 0))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    /* Get mapped pointers to the vertex data buffer */

    float *vboPt;
    size_t vboSize;
    float* vboFlagPt;
    unsigned int *vboTriangleIdxPt;

    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt),
            &vboSize,
            cudaTokens[0]))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboTriangleIdxPt),
            &vboSize,
            cudaTokens[1]))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboFlagPt),
            &vboSize,
            cudaTokens[2]))) {
        return false;
    }

    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaMemset(vboFlagPt, 0x00, this->vertexCnt*sizeof(float)))) {
        return false;
    }


    // Call kernel
    DeformableGPUSurfaceMT_FlagCorruptTriangles_D <<< Grid(this->triangleCnt, 256), 256 >>> (
            vboFlagPt,
            this->corruptTriangles_D.Peek(),
            vboPt,
            AbstractGPUSurface::vertexDataStride,
            AbstractGPUSurface::vertexDataOffsPos,
            AbstractGPUSurface::vertexDataOffsNormal,
            vboTriangleIdxPt,
            volume_D,
            targetActiveCells,
            (float4*)(this->externalForces_D.Peek()),
            this->triangleCnt,
            isovalue);

    ::CheckForCudaErrorSync();

    // Set vertex flags according to triangle flags
    HostArr<float> triFlags, vtxFlags;
    HostArr<uint> triIdx;
    triFlags.Validate(this->triangleCnt);
    triIdx.Validate(this->triangleCnt*3);
    vtxFlags.Validate(this->vertexCnt);
    cudaMemcpy(vtxFlags.Peek(), vboFlagPt,
            sizeof(float)*this->vertexCnt, cudaMemcpyDeviceToHost);
    cudaMemcpy(triIdx.Peek(), vboTriangleIdxPt,
            sizeof(uint)*this->triangleCnt*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(triFlags.Peek(), this->corruptTriangles_D.Peek(),
            sizeof(float)*this->triangleCnt, cudaMemcpyDeviceToHost);
    vtxFlags.Set(0x00);
    for (int i = 0; i < this->triangleCnt; ++i) {
        float triFlag = triFlags.Peek()[i];
        if (triFlag == 1.0) {
            vtxFlags.Peek()[triIdx.Peek()[3*i+0]] = 1.0;
            vtxFlags.Peek()[triIdx.Peek()[3*i+1]] = 1.0;
            vtxFlags.Peek()[triIdx.Peek()[3*i+2]] = 1.0;
        }
    }

    // DEBUG Check validity of vertex flags
    HostArr<bool> vtxFlagValid;
    vtxFlagValid.Validate(this->vertexCnt);
    vtxFlagValid.Set(0x00);
    for (int i = 0; i < this->triangleCnt; ++i) {
        float triFlag = triFlags.Peek()[i];
        float vtxFlag0 = vtxFlags.Peek()[triIdx.Peek()[3*i+0]];
        float vtxFlag1 = vtxFlags.Peek()[triIdx.Peek()[3*i+1]];
        float vtxFlag2 = vtxFlags.Peek()[triIdx.Peek()[3*i+2]];
        if (triFlag == 1.0) {
            if (vtxFlag0 == 1.0) {
                vtxFlagValid.Peek()[triIdx.Peek()[3*i+0]] = true;
            } else {
                printf("INVALIV zero VERTEX FLAG %i (0)\n", triIdx.Peek()[3*i+0]);
            }
            if (vtxFlag1 == 1.0) {
                vtxFlagValid.Peek()[triIdx.Peek()[3*i+1]] = true;
            } else {
                printf("INVALIV zero VERTEX FLAG %i (1)\n", triIdx.Peek()[3*i+1]);
            }
            if (vtxFlag2 == 1.0) {
                vtxFlagValid.Peek()[triIdx.Peek()[3*i+2]] = true;
            } else {
                printf("INVALIV zero VERTEX FLAG %i (2)\n", triIdx.Peek()[3*i+2]);
            }
        }
    }
    for (int i = 0; i < this->vertexCnt; ++i) {
        if (vtxFlags.Peek()[i] == 1.0) {
            if (vtxFlagValid.Peek()[i] == false) {
                printf("INVALIV one VERTEX FLAG %i\n", i);
            }
        }
    }
    vtxFlagValid.Release();
    // END DEBUG

    cudaMemcpy(vboFlagPt, vtxFlags.Peek(),
                sizeof(float)*this->vertexCnt, cudaMemcpyHostToDevice);
    if (!CudaSafeCall(cudaGetLastError())) {
        return false;
    }

    triIdx.Release();
    vtxFlags.Release();
    triFlags.Release();

    if (!CudaSafeCall(cudaGetLastError())) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnmapResources(3, cudaTokens, 0))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[2]))) {
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::InitCorruptFlagVBO
 */
bool DeformableGPUSurfaceMT::InitCorruptFlagVBO(size_t vertexCnt) {

    // Destroy if necessary
    if (this->vboCorruptTriangleVertexFlag) {
        glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, this->vboCorruptTriangleVertexFlag);
        glDeleteBuffersARB(1, &this->vboCorruptTriangleVertexFlag);
        this->vboCorruptTriangleVertexFlag = 0;
    }

    // Create vertex buffer object for corrupt vertex flag
    glGenBuffersARB(1, &this->vboCorruptTriangleVertexFlag);
    glBindBufferARB(GL_ARRAY_BUFFER, this->vboCorruptTriangleVertexFlag);
    glBufferDataARB(GL_ARRAY_BUFFER, sizeof(float)*vertexCnt, 0, GL_DYNAMIC_DRAW);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

    return CheckForGLError();
}


/*
 * DeformableGPUSurfaceMT::InitVtxPathVBO
 */
bool DeformableGPUSurfaceMT::InitVtxPathVBO(size_t vertexCnt) {

    // Destroy if necessary
    if (this->vboVtxPath) {
        glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxPath);
        glDeleteBuffersARB(1, &this->vboVtxPath);
        this->vboVtxPath = 0;
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
    }

    // Create vertex buffer object for corrupt vertex flag
    glGenBuffersARB(1, &this->vboVtxPath);
    glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxPath);
    glBufferDataARB(GL_ARRAY_BUFFER, sizeof(float)*vertexCnt, 0, GL_DYNAMIC_DRAW);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

//    printf("InitVtxPathVBO: %u bytes\n", sizeof(float)*vertexCnt);

    return CheckForGLError();
}


/*
 * DeformableGPUSurfaceMT::InitVtxAttribVBO
 */
bool DeformableGPUSurfaceMT::InitVtxAttribVBO(size_t vertexCnt) {

    // Destroy if necessary
    if (this->vboVtxAttr) {
        glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxAttr);
        glDeleteBuffersARB(1, &this->vboVtxAttr);
        this->vboVtxAttr = 0;
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
    }

    // Create vertex buffer object for corrupt vertex flag
    glGenBuffersARB(1, &this->vboVtxAttr);
    glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxAttr);
    glBufferDataARB(GL_ARRAY_BUFFER, sizeof(float)*vertexCnt, 0, GL_DYNAMIC_DRAW);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

//    printf("InitVtxPathVBO: %u bytes\n", sizeof(float)*vertexCnt);

    return CheckForGLError();
}


/*
 * DeformableGPUSurfaceMT::initExtForcesGradient
 */
bool DeformableGPUSurfaceMT::initExtForcesGradient(float *volTarget_D,
        int3 volDim, float3 volOrg, float3 volDelta) {
    using namespace vislib::sys;

    int volSize = volDim.x*volDim.y*volDim.z;

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    // Allocate memory
    if (!CudaSafeCall(this->externalForces_D.Validate(volSize*4))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not allocate memory",
                this->ClassName());
        return false;
    }

    // Init with zero
    if (!CudaSafeCall(this->externalForces_D.Set(0))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init memory",
                this->ClassName());
        return false;
    }

#ifdef USE_CUDA_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Calculate gradient using finite differences
    DeformableGPUSurfaceMT_CalcVolGradient_D <<< Grid(volSize, 256), 256 >>> (
            (float4*)this->externalForces_D.Peek(), volTarget_D);

#ifdef USE_CUDA_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'CalcVolGradient_D':                     %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return true;

}


/*
 * DeformableGPUSurfaceMT::initExtForcesDistfield
 */
bool DeformableGPUSurfaceMT::initExtForcesDistfield(
        float *volume_D,
        float *vertexBuffer_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float distfieldDist,
        float isovalue) {

    using namespace vislib::sys;

    int volSize = volDim.x*volDim.y*volDim.z;

    /* Init grid parameters */

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }


    // Compute distance field
    if (!CudaSafeCall(this->distField_D.Validate(volSize))) {
        return false;
    }

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_ComputeDistField_D <<< Grid(volSize, 256), 256 >>> (
            vertexBuffer_D,
            this->distField_D.Peek(),
            this->vertexCnt,
            this->vertexDataOffsPos,
            this->vertexDataStride);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeDistField_D':                    %.10f sec\n",
            dt_ms/1000.0f);
#endif

    // Compute gradient
    if (!CudaSafeCall(this->externalForces_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->externalForces_D.Set(0))) {
        return false;
    }

#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    // Calculate gradient using finite differences
    DeformableGPUSurfaceMT_CalcVolGradientWithDistField_D <<< Grid(volSize, 256), 256 >>> (
            (float4*)this->externalForces_D.Peek(),
            volume_D,
            this->distField_D.Peek(), distfieldDist, isovalue);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'CalcVolGradientWithDistField_D':        %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return CudaSafeCall(cudaGetLastError());
}


bool DeformableGPUSurfaceMT::initExtForcesGVF(
        float *volumeTarget_D,
        const unsigned int *cellStatesTarget_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue,
        float gvfScl,
        unsigned int gvfIt) {

    using namespace vislib::sys;

    /* Init grid parameters */

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }


    int volSize = volDim.x*volDim.y*volDim.z;

    // Compute external forces
    if (!CudaSafeCall(this->externalForces_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->externalForces_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfTmp_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfTmp_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfConstData_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfConstData_D.Set(0))) {
        return false;
    }

    // Use GVF
    if (!DiffusionSolver::CalcGVF(
            volumeTarget_D,
            this->gvfConstData_D.Peek(),
            cellStatesTarget_D,
            volDim,
            volDelta,
            volOrg,
            this->externalForces_D.Peek(),
            this->gvfTmp_D.Peek(),
            gvfIt,
            gvfScl)) {
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::initExtForcesTwoWayGVF
 */
bool DeformableGPUSurfaceMT::initExtForcesTwoWayGVF(
        float *volumeSource_D,
        float *volumeTarget_D,
        const unsigned int *cellStatesSource_D,
        const unsigned int *cellStatesTarget_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue,
        float gvfScl,
        unsigned int gvfIt) {

    using namespace vislib::sys;

    /* Init grid parameters */

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }


    using namespace vislib::sys;

    //#ifdef USE_TIMER
        float dt_ms;
        cudaEvent_t event1, event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);
        cudaEventRecord(event1, 0);
    //#endif

    int volSize = volDim.x*volDim.y*volDim.z;

    // Compute external forces
    if (!CudaSafeCall(this->externalForces_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->externalForces_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfTmp_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfTmp_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfConstData_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfConstData_D.Set(0))) {
        return false;
    }

    // Calculate two way gvf by using isotropic diffusion
    if (!DiffusionSolver::CalcTwoWayGVF(
           volumeSource_D,
           volumeTarget_D,
           cellStatesSource_D,
           cellStatesTarget_D,
           volDim,
           volOrg,
           volDelta,
           this->gvfConstData_D.Peek(),
           this->externalForces_D.Peek(),
           this->gvfTmp_D.Peek(),
           gvfIt,
           gvfScl)) {
        return false;
    }

    //#ifdef USE_TIMER
        cudaEventRecord(event2, 0);
        cudaEventSynchronize(event1);
        cudaEventSynchronize(event2);
        cudaEventElapsedTime(&dt_ms, event1, event2);
//        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
//                "%s: Time for bi-directional diffusion %f\n",
//                "DeformableGPUSurfaceMT", dt_ms/1000.0f);
    //#endif
//        printf("GVF : %.10f\n",
//                dt_ms/1000.0f);

    return true;
}


/*
 * DeformableGPUSurfaceMT::InitGridParams
 */
bool DeformableGPUSurfaceMT::InitGridParams(uint3 gridSize, float3 org, float3 delta) {
    cudaMemcpyToSymbol(gridSize_D, &gridSize, sizeof(uint3));
    cudaMemcpyToSymbol(gridOrg_D, &org, sizeof(float3));
    cudaMemcpyToSymbol(gridDelta_D, &delta, sizeof(float3));
//    printf("Init grid with org %f %f %f, delta %f %f %f, dim %u %u %u\n", org.x,
//            org.y, org.z, delta.x, delta.y, delta.z, gridSize.x, gridSize.y,
//            gridSize.z);
    return CudaSafeCall(cudaGetLastError());
}



/*
 * DeformableGPUSurfaceMT_IntOverTriangles_D
 */
__global__ void DeformableGPUSurfaceMT_IntOverTriangles_D(
        float *trianglesAreaWeightedVertexVals_D,
        float *trianglesArea_D,
        uint *triangleIdx_D,
        float *scalarValue_D,
        uint triangleCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= triangleCnt) {
        return;
    }

    // Compute average
    float avgVal = (scalarValue_D[triangleIdx_D[idx*3+0]] +
                    scalarValue_D[triangleIdx_D[idx*3+1]] +
                    scalarValue_D[triangleIdx_D[idx*3+2]])/3.0;

    trianglesAreaWeightedVertexVals_D[idx] = avgVal*trianglesArea_D[idx];
}


/*
 * DeformableGPUSurfaceMT_IntOverValidTriangles_D
 */
__global__ void DeformableGPUSurfaceMT_IntOverValidTriangles_D(
        float *trianglesAreaWeightedVertexVals_D,
        float *trianglesArea_D,
        uint *triangleIdx_D,
        float *scalarValue_D,
        float *corruptTriFlag_D,
        uint triangleCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= triangleCnt) {
        return;
    }

    // Compute average
    float avgVal = (scalarValue_D[triangleIdx_D[idx*3+0]] +
                    scalarValue_D[triangleIdx_D[idx*3+1]] +
                    scalarValue_D[triangleIdx_D[idx*3+2]])/3.0;

    trianglesAreaWeightedVertexVals_D[idx] = avgVal*trianglesArea_D[idx]*(1.0-corruptTriFlag_D[idx]);
}


/*
 * DeformableGPUSurfaceMT::IntOverSurfArea
 */
float DeformableGPUSurfaceMT::IntOverSurfArea(float *value_D) {

    // Compute triangle areas of all (non-corrupt) triangles
    if (!CudaSafeCall(this->accTriangleData_D.Validate(this->triangleCnt))) {
        return false;
    }
    cudaGraphicsResource* cudaTokens[1];

    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(1, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    uint *triangleIdxPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&triangleIdxPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    DeformableGPUSurfaceMT_IntOverTriangles_D <<< Grid(this->triangleCnt, 256), 256 >>> (
            this->accTriangleData_D.Peek(),
            this->accTriangleArea_D.Peek(),
            triangleIdxPt,
            value_D,
            this->triangleCnt);

    ::CheckForCudaErrorSync();

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'intOverTriangles_D                      %.10f sec\n",
            dt_ms/1000.0);
#endif

    // Compute sum of all (non-corrupt) triangle areas
    float integralVal = thrust::reduce(
            thrust::device_ptr<float>(this->accTriangleData_D.Peek()),
            thrust::device_ptr<float>(this->accTriangleData_D.Peek() + this->triangleCnt));

    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, cudaTokens, 0))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }

    return integralVal;

}


/**
 * Integrate scalar value (given per vertex in value_D) over surface area.
 *
 * @return The integral value
 */
float DeformableGPUSurfaceMT::IntVtxPathOverSurfArea() {

    // TODO Assumes triangle area to be computed

    // Device array for accumulated data
     if (!CudaSafeCall(this->accTriangleData_D.Validate(this->triangleCnt))) {
         return false;
     }
     if (!CudaSafeCall(this->accTriangleData_D.Set(0x00))) {
         return false;
     }
     cudaGraphicsResource* cudaTokens[2];

     if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
             &cudaTokens[0],
             this->vboTriangleIdx,
             cudaGraphicsMapFlagsNone))) {
         return false;
     }

     if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
             &cudaTokens[1],
             this->vboVtxPath,
             cudaGraphicsMapFlagsNone))) {
         return false;
     }

     if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
         return false;
     }

     // Get mapped pointers to the vertex data buffers
     uint *triangleIdxPt;
     size_t vboSizeTri;
     if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
             reinterpret_cast<void**>(&triangleIdxPt), // The mapped pointer
             &vboSizeTri,              // The size of the accessible data
             cudaTokens[0]))) {                 // The mapped resource
         return false;
     }

     // Get mapped pointers to the vertex data buffers
     float *uncertaintyPt;
     size_t vboSizeUncertainty;
     if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
             reinterpret_cast<void**>(&uncertaintyPt), // The mapped pointer
             &vboSizeUncertainty,              // The size of the accessible data
             cudaTokens[1]))) {                 // The mapped resource
         return false;
     }

 #ifdef USE_TIMER
     float dt_ms;
     cudaEvent_t event1, event2;
     cudaEventCreate(&event1);
     cudaEventCreate(&event2);
     cudaEventRecord(event1, 0);
 #endif

     // Call kernel
     DeformableGPUSurfaceMT_IntOverTriangles_D <<< Grid(this->triangleCnt, 256), 256 >>> (
             this->accTriangleData_D.Peek(),
             this->accTriangleArea_D.Peek(),
             triangleIdxPt,
             uncertaintyPt,
             this->triangleCnt);

     ::CheckForCudaErrorSync();

 #ifdef USE_TIMER
     cudaEventRecord(event2, 0);
     cudaEventSynchronize(event1);
     cudaEventSynchronize(event2);
     cudaEventElapsedTime(&dt_ms, event1, event2);
     printf("CUDA time for 'intOverTriangles_D                     %.10f sec\n",
             dt_ms/1000.0);
 #endif

     // Compute sum of all (non-corrupt) triangle areas
     float integralVal = thrust::reduce(
             thrust::device_ptr<float>(this->accTriangleData_D.Peek()),
             thrust::device_ptr<float>(this->accTriangleData_D.Peek() + this->triangleCnt));

     if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
         return false;
     }

 //    ::CheckForCudaErrorSync();

     if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
         return false;
     }
     if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
         return false;
     }

     return integralVal;

}


/**
 * Integrate scalar value (given per vertex in value_D) over surface area.
 *
 * @return The integral value
 */
float DeformableGPUSurfaceMT::IntVtxPathOverValidSurfArea() {

    // TODO Assumes triangle area to be computed

    // Device array for accumulated data
     if (!CudaSafeCall(this->accTriangleData_D.Validate(this->triangleCnt))) {
         return false;
     }
     if (!CudaSafeCall(this->accTriangleData_D.Set(0x00))) {
         return false;
     }
     cudaGraphicsResource* cudaTokens[2];

     if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
             &cudaTokens[0],
             this->vboTriangleIdx,
             cudaGraphicsMapFlagsNone))) {
         return false;
     }

     if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
             &cudaTokens[1],
             this->vboVtxPath,
             cudaGraphicsMapFlagsNone))) {
         return false;
     }

     if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
         return false;
     }

     // Get mapped pointers to the vertex data buffers
     uint *triangleIdxPt;
     size_t vboSizeTri;
     if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
             reinterpret_cast<void**>(&triangleIdxPt), // The mapped pointer
             &vboSizeTri,              // The size of the accessible data
             cudaTokens[0]))) {                 // The mapped resource
         return false;
     }

     // Get mapped pointers to the vertex data buffers
     float *uncertaintyPt;
     size_t vboSizeUncertainty;
     if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
             reinterpret_cast<void**>(&uncertaintyPt), // The mapped pointer
             &vboSizeUncertainty,              // The size of the accessible data
             cudaTokens[1]))) {                 // The mapped resource
         return false;
     }

 #ifdef USE_TIMER
     float dt_ms;
     cudaEvent_t event1, event2;
     cudaEventCreate(&event1);
     cudaEventCreate(&event2);
     cudaEventRecord(event1, 0);
 #endif

     // Call kernel
     DeformableGPUSurfaceMT_IntOverValidTriangles_D <<< Grid(this->triangleCnt, 256), 256 >>> (
             this->accTriangleData_D.Peek(),
             this->accTriangleArea_D.Peek(),
             triangleIdxPt,
             uncertaintyPt,
             this->corruptTriangles_D.Peek(),
             this->triangleCnt);

     ::CheckForCudaErrorSync();

 #ifdef USE_TIMER
     cudaEventRecord(event2, 0);
     cudaEventSynchronize(event1);
     cudaEventSynchronize(event2);
     cudaEventElapsedTime(&dt_ms, event1, event2);
     printf("CUDA time for 'intOverTriangles_D                     %.10f sec\n",
             dt_ms/1000.0);
 #endif

     // Compute sum of all (non-corrupt) triangle areas
     float integralVal = thrust::reduce(
             thrust::device_ptr<float>(this->accTriangleData_D.Peek()),
             thrust::device_ptr<float>(this->accTriangleData_D.Peek() + this->triangleCnt));

     if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
         return false;
     }

 //    ::CheckForCudaErrorSync();

     if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
         return false;
     }
     if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
         return false;
     }

     return integralVal;

}


/**
 * Integrate scalar value (given per vertex in value_D) over surface area.
 *
 * @return The integral value
 */
float DeformableGPUSurfaceMT::IntVtxAttribOverSurfArea() {

    // TODO Assumes triangle area to be computed


    // Device array for accumulated data
     if (!CudaSafeCall(this->accTriangleData_D.Validate(this->triangleCnt))) {
         return false;
     }
     if (!CudaSafeCall(this->accTriangleData_D.Set(0x00))) {
         return false;
     }
     cudaGraphicsResource* cudaTokens[2];

     if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
             &cudaTokens[0],
             this->vboTriangleIdx,
             cudaGraphicsMapFlagsNone))) {
         return false;
     }

     if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
             &cudaTokens[1],
             this->vboVtxAttr,
             cudaGraphicsMapFlagsNone))) {
         return false;
     }

     if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
         return false;
     }

     // Get mapped pointers to the vertex data buffers
     uint *triangleIdxPt;
     size_t vboSizeTri;
     if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
             reinterpret_cast<void**>(&triangleIdxPt), // The mapped pointer
             &vboSizeTri,              // The size of the accessible data
             cudaTokens[0]))) {                 // The mapped resource
         return false;
     }

     // Get mapped pointers to the vertex data buffers
     float *vertexAttrPt;
     size_t vboVertexAttrSize;
     if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
             reinterpret_cast<void**>(&vertexAttrPt), // The mapped pointer
             &vboVertexAttrSize,              // The size of the accessible data
             cudaTokens[1]))) {                 // The mapped resource
         return false;
     }

 #ifdef USE_TIMER
     float dt_ms;
     cudaEvent_t event1, event2;
     cudaEventCreate(&event1);
     cudaEventCreate(&event2);
     cudaEventRecord(event1, 0);
 #endif

     // Call kernel
     DeformableGPUSurfaceMT_IntOverTriangles_D <<< Grid(this->triangleCnt, 256), 256 >>> (
             this->accTriangleData_D.Peek(),
             this->accTriangleArea_D.Peek(),
             triangleIdxPt,
             vertexAttrPt,
             this->triangleCnt);

     ::CheckForCudaErrorSync();

 #ifdef USE_TIMER
     cudaEventRecord(event2, 0);
     cudaEventSynchronize(event1);
     cudaEventSynchronize(event2);
     cudaEventElapsedTime(&dt_ms, event1, event2);
     printf("CUDA time for 'intOverTriangles_D                     %.10f sec\n",
             dt_ms/1000.0);
 #endif

     // Compute sum of all (non-corrupt) triangle areas
     float integralVal = thrust::reduce(
             thrust::device_ptr<float>(this->accTriangleData_D.Peek()),
             thrust::device_ptr<float>(this->accTriangleData_D.Peek() + this->triangleCnt));

     if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
         return false;
     }

 //    ::CheckForCudaErrorSync();

     if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
         return false;
     }
     if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
         return false;
     }

     return integralVal;

}


/**
 * Integrate scalar value (given per vertex in value_D) over surface area.
 *
 * @return The integral value
 */
float DeformableGPUSurfaceMT::IntVtxAttribOverValidSurfArea() {

    // TODO Assumes triangle area to be computed


    // Device array for accumulated data
     if (!CudaSafeCall(this->accTriangleData_D.Validate(this->triangleCnt))) {
         return false;
     }
     if (!CudaSafeCall(this->accTriangleData_D.Set(0x00))) {
         return false;
     }
     cudaGraphicsResource* cudaTokens[2];

     if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
             &cudaTokens[0],
             this->vboTriangleIdx,
             cudaGraphicsMapFlagsNone))) {
         return false;
     }

     if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
             &cudaTokens[1],
             this->vboVtxAttr,
             cudaGraphicsMapFlagsNone))) {
         return false;
     }

     if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
         return false;
     }

     // Get mapped pointers to the vertex data buffers
     uint *triangleIdxPt;
     size_t vboSizeTri;
     if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
             reinterpret_cast<void**>(&triangleIdxPt), // The mapped pointer
             &vboSizeTri,              // The size of the accessible data
             cudaTokens[0]))) {                 // The mapped resource
         return false;
     }

     // Get mapped pointers to the vertex data buffers
     float *vertexAttrPt;
     size_t vboVertexAttrSize;
     if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
             reinterpret_cast<void**>(&vertexAttrPt), // The mapped pointer
             &vboVertexAttrSize,              // The size of the accessible data
             cudaTokens[1]))) {                 // The mapped resource
         return false;
     }

 #ifdef USE_TIMER
     float dt_ms;
     cudaEvent_t event1, event2;
     cudaEventCreate(&event1);
     cudaEventCreate(&event2);
     cudaEventRecord(event1, 0);
 #endif

     // Call kernel
     DeformableGPUSurfaceMT_IntOverValidTriangles_D <<< Grid(this->triangleCnt, 256), 256 >>> (
             this->accTriangleData_D.Peek(),
             this->accTriangleArea_D.Peek(),
             triangleIdxPt,
             vertexAttrPt,
             this->corruptTriangles_D.Peek(),
             this->triangleCnt);

     ::CheckForCudaErrorSync();

 #ifdef USE_TIMER
     cudaEventRecord(event2, 0);
     cudaEventSynchronize(event1);
     cudaEventSynchronize(event2);
     cudaEventElapsedTime(&dt_ms, event1, event2);
     printf("CUDA time for 'intOverTriangles_D                     %.10f sec\n",
             dt_ms/1000.0);
 #endif

     // Compute sum of all (non-corrupt) triangle areas
     float integralVal = thrust::reduce(
             thrust::device_ptr<float>(this->accTriangleData_D.Peek()),
             thrust::device_ptr<float>(this->accTriangleData_D.Peek() + this->triangleCnt));

     if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
         return false;
     }

 //    ::CheckForCudaErrorSync();

     if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
         return false;
     }
     if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
         return false;
     }

     return integralVal;

}


/*
 * DeformableGPUSurfaceMT::IntOverCorruptSurfArea
 */
float DeformableGPUSurfaceMT::IntOverCorruptSurfArea() {
    return 0.0f;
}


/**
 * TODO
 * @return Position and path length addition
 */
float4 UpdateVtxPosSingle (
        float3 posStart,              // Starting position
        float4 *gradient,             // External forces
        float *targetVol,             // The target volume
        float minDisplScl,            // Minimum displacement for convergence
        float forcesScl,              // General scaling factor for forces
        float isovalue,
        float org[3], float delta[3], int dim[3],
        int maxSteps,
        int maxLevel,
        float initStepSize) {             // Isovalue

    float3 pos = posStart;

    float sample = SampleFieldAtPosTrilin((float*)(&pos), targetVol, org, delta, dim);
    bool outside = sample <= isovalue;
    float extForcesScl;

    if (outside) extForcesScl = 1.0;
    else extForcesScl = -1.0;
    float len = 0.0f;
    bool converged = false;
    int steps = 0;
    do {
//        printf("current pos: %f %f %f\n", pos.x, pos.y, pos.z);
        // Get volume sample
        float sample = SampleFieldAtPosTrilin((float*)(&pos), targetVol, org, delta, dim);

        // Switch sign and scale down if necessary
        bool negative = extForcesScl < 0;
        bool outside = sample <= isovalue;
        int switchSign = int((negative && outside)||(!negative && !outside));
        extForcesScl = extForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
        extForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));

        // Get external forces sample and scale
        float4 extForceTmp = SampleFieldAtPosTrilin((float*)(&pos), gradient, org, delta, dim);
        float3 extForce = make_float3(extForceTmp.x, extForceTmp.y, extForceTmp.z);
        extForce = safeNormalize(extForce);
        // Accumulate path
        len += extForcesScl*forcesScl;
        extForce *= extForcesScl*forcesScl;

        // Propagate vertex and increase path length
        pos += extForce;


        if (length(extForce) <= minDisplScl) {
            converged = true;
        }
        steps++;
    } while (!converged && steps < maxSteps);

    return make_float4(pos.x, pos.y, pos.z, len);
}



/**
 * TODO
 */
float DeformableGPUSurfaceMT::IntUncertaintyOverCorruptAreaRec(
        float3 pos1, float3 pos2, float3 pos3, // Vertex positions of the triangle
        float len1, float len2, float len3,    // Vertex path lengths of the triangle
        float4 *gradient,                      // External forces
        float *targetVol,                      // The target volume
        unsigned int *targetActiveCells,       // Active cells of the target volume
        float minDisplScl,                     // Minimum displacement for convergence
        float forcesScl,                       // General scaling factor for forces
        float isovalue,                        // Isovalue
        float &triArea,
        uint depth,
        float org[3], float delta[3], int dim[3],
        vislib::Array<float> &triArr,
        int maxSteps,
        int maxLevel,
        float initStepSize) {

//    printf("depth: %i\n", depth);

    // 1. Propagate vertices until they converge to a fixed position

    float4 newPosLen1, newPosLen2, newPosLen3;
    newPosLen1 = UpdateVtxPosSingle(pos1, gradient, targetVol,
            minDisplScl, forcesScl, isovalue, org, delta, dim,
            maxSteps,
            maxLevel,
            initStepSize);
    newPosLen2 = UpdateVtxPosSingle(pos2, gradient, targetVol,
            minDisplScl, forcesScl, isovalue, org, delta, dim,
            maxSteps,
            maxLevel,
            initStepSize);
    newPosLen3 = UpdateVtxPosSingle(pos3, gradient, targetVol,
            minDisplScl, forcesScl, isovalue, org, delta, dim,
            maxSteps,
            maxLevel,
            initStepSize);
    float3 newPos1, newPos2, newPos3;
    newPos1 = make_float3(newPosLen1.x, newPosLen1.y, newPosLen1.z);
    newPos2 = make_float3(newPosLen2.x, newPosLen2.y, newPosLen2.z);
    newPos3 = make_float3(newPosLen3.x, newPosLen3.y, newPosLen3.z);

    // 2. Check whether the resulting triangle is valid

    float3 midpoint = (newPos1+newPos2+newPos3)/3.0;
    int3 coords;
    coords.x = int((midpoint.x-org[0])/delta[0]);
    coords.y = int((midpoint.y-org[1])/delta[1]);
    coords.z = int((midpoint.z-org[2])/delta[2]);
    //int cellIDx = ::GetCellIdxByGridCoords(coords);
    int cellIdx = (dim[0]-1)*((dim[1]-1)*coords.z + coords.y) + coords.x;
    uint cellState = targetActiveCells[cellIdx];

	if ((cellState == 1) || (depth >= (int)maxLevel)) {

        triArr.Add(newPos1.x);
        triArr.Add(newPos1.y);
        triArr.Add(newPos1.z);

        triArr.Add(newPos2.x);
        triArr.Add(newPos2.y);
        triArr.Add(newPos2.z);

        triArr.Add(newPos3.x);
        triArr.Add(newPos3.y);
        triArr.Add(newPos3.z);

//        printf("%.16f;%.16f;%.16f;%.16f;%.16f;%.16f;%.16f;%.16f;%.16f\n",
//                newPos1.x, newPos1.y, newPos1.z,
//                newPos2.x, newPos2.y, newPos2.z,
//                newPos3.x, newPos3.y, newPos3.z);

        // 3a. Cell is active, therefore triangle is valid
        // --> Compute integrated uncertainty value
        // Get triangle area
        float a = length(newPos1 - newPos2);
        float b = length(newPos1 - newPos3);
        float c = length(newPos2 - newPos3);

        // Compute area (Heron's formula)
        float rad = (a + b - c)*(c + a - b)*(a + b + c)*(b + c - a);
        // Make sure radicand is not negative
        rad = rad > 0.0f ? rad : 0.0f;
        float area = 0.25f*sqrt(rad);
        triArea = area;

        // Get average value
        float avgValue = (len1+newPosLen1.w+len2+newPosLen2.w+len3+newPosLen3.w)/3.0f;

        // Approximate integration
        return triArea*avgValue;
    } else {
        float triArea1, triArea2, triArea3, triArea4;
        // 3b. Cell is not active, therefore, triangle is not valid
        // --> Subdivide and call recursively

        float3 p12 = (newPos1+newPos2)/2.0;
        float3 p13 = (newPos1+newPos3)/2.0;
        float3 p32 = (newPos3+newPos2)/2.0;
        float l12 = (len1+newPosLen1.w+len2+newPosLen2.w)/2.0;
        float l13 = (len1+newPosLen1.w+len3+newPosLen3.w)/2.0;
        float l32 = (len3+newPosLen3.w+len2+newPosLen2.w)/2.0;

        float intUncertainty1 =
                DeformableGPUSurfaceMT::IntUncertaintyOverCorruptAreaRec(
                        newPos1, p12, p13,
                        len1+newPosLen1.w, l12, l13,
                        gradient, targetVol, targetActiveCells,
                        minDisplScl, forcesScl, isovalue, triArea1,
                        depth+1, org, delta, dim, triArr,
                        maxSteps,
                        maxLevel,
                        initStepSize);

        float intUncertainty2 =
                DeformableGPUSurfaceMT::IntUncertaintyOverCorruptAreaRec(
                        p13, p32, newPos3,
                        l13, l32, len3+newPosLen3.w,
                        gradient, targetVol, targetActiveCells,
                        minDisplScl, forcesScl, isovalue, triArea2,
                        depth+1, org, delta, dim, triArr,
                        maxSteps,
                        maxLevel,
                        initStepSize);

        float intUncertainty3 =
                DeformableGPUSurfaceMT::IntUncertaintyOverCorruptAreaRec(
                        p12, p13, p32,
                        l12, l13, l32,
                        gradient, targetVol, targetActiveCells,
                        minDisplScl, forcesScl, isovalue, triArea3,
                        depth+1, org, delta, dim, triArr,
                        maxSteps,
                        maxLevel,
                        initStepSize);

        float intUncertainty4 =
                DeformableGPUSurfaceMT::IntUncertaintyOverCorruptAreaRec(
                        p12, p32, newPos2,
                        l12, l32, len2+newPosLen2.w,
                        gradient, targetVol, targetActiveCells,
                        minDisplScl, forcesScl, isovalue, triArea4,
                        depth+1, org, delta, dim, triArr,
                        maxSteps,
                        maxLevel,
                        initStepSize);


        triArea = triArea1 + triArea2 + triArea3 + triArea4;

        return intUncertainty1 + intUncertainty2 + intUncertainty3 + intUncertainty4;
    }
}



/*
 * DeformableGPUSurfaceMT::IntUncertaintyOverCorruptSurfArea
 */
float DeformableGPUSurfaceMT::IntUncertaintyOverCorruptSurfArea(
        float &corruptArea,
        float minDisplScl,
        float isovalue,
        float forcesScl,
        unsigned int *targetActiveCells_D,
        float *targetVol_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        vislib::Array<float> &triArr,
        int maxDepth,
        int maxLevel,
        float initStepSize) {
    using namespace vislib::sys;

    size_t fieldSize = volDim.x*volDim.y*volDim.z;
    size_t cellCnt = (volDim.x-1)*(volDim.y-1)*(volDim.z-1);

//    // Allocate memory for corrupt triangles
//    if (!CudaSafeCall(this->intUncertaintyCorrupt_D.Validate(this->triangleCnt))) {
//        return false;
//    }
//    // Init with zero
//    if (!CudaSafeCall(this->intUncertaintyCorrupt_D.Set(0x00))) {
//        return false;
//    }
//
//    if (!CudaSafeCall(this->accTriangleArea_D.Validate(this->triangleCnt))) {
//        return false;
//    }
//    if (!CudaSafeCall(this->accTriangleArea_D.Set(0x00))){
//        return false;
//    }
//
//    // Init constant device params
//    if (!initGridParams(volDim, volOrg, volDelta)) {
//        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
//                "%s: could not init constant device params",
//                this->ClassName());
//        return false;
//    }
//
    cudaGraphicsResource* cudaTokens[3];

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1],
            this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[2],
            this->vboVtxPath,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    // Map cuda ressource handles
    if (!CudaSafeCall(cudaGraphicsMapResources(3, cudaTokens, 0))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    /* Get mapped pointers to the vertex data buffer */

    float *vboPt;
    size_t vboSize;
    float* vboVtxPathPt;
    unsigned int *vboTriangleIdxPt;

    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt),
            &vboSize,
            cudaTokens[0]))) {
        return false;
    }

    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboTriangleIdxPt),
            &vboSize,
            cudaTokens[1]))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboVtxPathPt),
            &vboSize,
            cudaTokens[2]))) {
        return false;
    }
//
//#ifdef USE_TIMER
//    float dt_ms;
//    cudaEvent_t event1, event2;
//    cudaEventCreate(&event1);
//    cudaEventCreate(&event2);
//    cudaEventRecord(event1, 0);
//#endif
//
//    ::CheckForCudaErrorSync();
//
//    // Call kernel
//    DeformableGPUSurfaceMT_IntUncertaintyOverCorruptArea_D <<< Grid(this->triangleCnt, 256), 256 >>> (
//            this->corruptTriangles_D.Peek(),
//            vboPt,
//            vboVtxPathPt,
//            this->vertexDataStride,
//            this->vertexDataOffsPos,
//            this->vertexDataOffsNormal,
//            vboTriangleIdxPt,
//            targetVol_D,
//            (float4*)this->externalForces_D.Peek(),
//            targetActiveCells_D,
//            this->triangleCnt,
//            isovalue,
//            minDisplScl,
//            forcesScl,
//            this->intUncertaintyCorrupt_D.Peek(),
//            this->accTriangleArea_D.Peek());
//
//    ::CheckForCudaErrorSync();
//
//#ifdef USE_TIMER
//    cudaEventRecord(event2, 0);
//    cudaEventSynchronize(event1);
//    cudaEventSynchronize(event2);
//    cudaEventElapsedTime(&dt_ms, event1, event2);
//    printf("CUDA time for 'intOverTriangles_D                     %.10f sec\n",
//            dt_ms/1000.0);
//#endif
//
//    // Compute sum of all (non-corrupt) triangle areas
//    float integralVal = thrust::reduce(
//            thrust::device_ptr<float>(this->intUncertaintyCorrupt_D.Peek()),
//            thrust::device_ptr<float>(this->intUncertaintyCorrupt_D.Peek() + this->triangleCnt));
//
//    corruptArea = thrust::reduce(
//            thrust::device_ptr<float>(this->accTriangleArea_D.Peek()),
//            thrust::device_ptr<float>(this->accTriangleArea_D.Peek() + this->triangleCnt));
//
//    ::CheckForCudaErrorSync();
//
//    if (!CudaSafeCall(cudaGetLastError())) {
//        return false;
//    }


    float integralVal = 0.0f;
    corruptArea = 0.0f;

    // Get necessary data from GPU
    HostArr<float> corruptTriangles;
    HostArr<float> vertexBuffer;
    HostArr<unsigned int> triangleIdx;
    HostArr<float> uncertainty;
    HostArr<float> gradient;
    HostArr<float> targetVol;
    HostArr<unsigned int> targetActiveCells;

    corruptTriangles.Validate(this->corruptTriangles_D.GetCount());
    vertexBuffer.Validate(this->vertexDataStride*this->vertexCnt);
    triangleIdx.Validate(this->triangleCnt*3);
    uncertainty.Validate(this->vertexCnt);
    gradient.Validate(fieldSize*4);
    targetVol.Validate(fieldSize);
    targetActiveCells.Validate(cellCnt);

    if (!CudaSafeCall(cudaMemcpy(corruptTriangles.Peek(), this->corruptTriangles_D.Peek(),
            corruptTriangles.GetCount()*sizeof(float), cudaMemcpyDeviceToHost))) {
        return false;
    }

    if (!CudaSafeCall(cudaMemcpy(vertexBuffer.Peek(), vboPt,
            vertexBuffer.GetCount()*sizeof(float), cudaMemcpyDeviceToHost))) {
        return false;
    }

    if (!CudaSafeCall(cudaMemcpy(triangleIdx.Peek(), vboTriangleIdxPt,
            triangleIdx.GetCount()*sizeof(unsigned int), cudaMemcpyDeviceToHost))) {
        return false;
    }

    if (!CudaSafeCall(cudaMemcpy(uncertainty.Peek(), vboVtxPathPt,
            uncertainty.GetCount()*sizeof(float), cudaMemcpyDeviceToHost))) {
        return false;
    }

    if (!CudaSafeCall(cudaMemcpy(gradient.Peek(), this->externalForces_D.Peek(),
            gradient.GetCount()*sizeof(float), cudaMemcpyDeviceToHost))) {
        return false;
    }

    if (!CudaSafeCall(cudaMemcpy(targetVol.Peek(), targetVol_D,
            targetVol.GetCount()*sizeof(float), cudaMemcpyDeviceToHost))) {
        return false;
    }

    if (!CudaSafeCall(cudaMemcpy(targetActiveCells.Peek(), targetActiveCells_D,
            targetActiveCells.GetCount()*sizeof(float), cudaMemcpyDeviceToHost))) {
        return false;
    }


    // Loop over all corrupt triangles
    for (int idx = 0; idx < this->triangleCnt; ++idx) {
        // Check whether the triangle is corrupt
        if (corruptTriangles.Peek()[idx] == 1.0f) {

            // Get initial positions from main memory
            uint baseIdx0 = vertexDataStride*triangleIdx.Peek()[3*idx+0];
            uint baseIdx1 = vertexDataStride*triangleIdx.Peek()[3*idx+1];
            uint baseIdx2 = vertexDataStride*triangleIdx.Peek()[3*idx+2];
            float3 pos1 = make_float3(
                    vertexBuffer.Peek()[baseIdx0+vertexDataOffsPos+0],
                    vertexBuffer.Peek()[baseIdx0+vertexDataOffsPos+1],
                    vertexBuffer.Peek()[baseIdx0+vertexDataOffsPos+2]);
            float3 pos2 = make_float3(
                    vertexBuffer.Peek()[baseIdx1+vertexDataOffsPos+0],
                    vertexBuffer.Peek()[baseIdx1+vertexDataOffsPos+1],
                    vertexBuffer.Peek()[baseIdx1+vertexDataOffsPos+2]);
            float3 pos3 = make_float3(
                    vertexBuffer.Peek()[baseIdx2+vertexDataOffsPos+0],
                    vertexBuffer.Peek()[baseIdx2+vertexDataOffsPos+1],
                    vertexBuffer.Peek()[baseIdx2+vertexDataOffsPos+2]);

            // Get initial path lengths from previous morphing
            float len1 = uncertainty.Peek()[triangleIdx.Peek()[3*idx+0]];
            float len2 = uncertainty.Peek()[triangleIdx.Peek()[3*idx+1]];
            float len3 = uncertainty.Peek()[triangleIdx.Peek()[3*idx+2]];

            integralVal += this->IntUncertaintyOverCorruptAreaRec(
                    pos1, pos2, pos3, // Vertex positions of the triangle
                    len1, len2, len3,    // Vertex path lengths of the triangle
                    (float4*)(gradient.Peek()),                      // External forces
                    targetVol.Peek(),                      // The target volume
                    targetActiveCells.Peek(),       // Active cells of the target volume
                    minDisplScl,                           // Minimum displacement for convergence
                    forcesScl,                             // General scaling factor for forces
                    isovalue,                              // Isovalue
                    corruptArea,
                    0,
                    (float*)&volOrg,
                    (float*)&volDelta,
                    (int*)&volDim,
                    triArr,
                    maxDepth,
                    maxLevel,
                    initStepSize);
        }
    }

    // Cleanup
    vertexBuffer.Release();
    corruptTriangles.Release();
    triangleIdx.Release();
    uncertainty.Release();
    gradient.Release();
    targetVol.Release();
    targetActiveCells.Release();

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnmapResources(3, cudaTokens, 0))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[2]))) {
        return false;
    }


    return integralVal;
}


/*
 * DeformableGPUSurfaceMT::MorphToVolumeGradient
 */
bool DeformableGPUSurfaceMT::MorphToVolumeGradient(
        float *volume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue,
        InterpolationMode interpMode,
        size_t maxIt,
        float surfMappedMinDisplScl,
        float springStiffness,
        float forceScl,
        float externalForcesWeight) {

    using vislib::sys::Log;

    /* Init grid parameters */

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }


    cudaGraphicsResource* cudaTokens[2];

    if ((!this->triangleIdxReady)||(!this->neighboursReady)) {
        return false;
    }

    if (volume_D == NULL) {
        return false;
    }

    if (!initExtForcesGradient(volume_D,
            volDim, volOrg, volDelta)) {
        return false;
    }

    // Init vbo with uncertainty information
    if (!this->InitVtxPathVBO(this->vertexCnt)) {
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], this->vboVtxPath,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboVtxPathPt;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboVtxPathPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }

    // Init forces scale factor with -1 or 1, depending on whether they start
    // outside or inside the isosurface
    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt))) {
        return false;
    }
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_InitExternalForceScl_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            (float*)this->vertexExternalForcesScl_D.Peek(),
            this->displLen_D.Peek(),
            volume_D,
            vboPt,
            surfMappedMinDisplScl,
            this->vertexCnt,
            isovalue,
            this->vertexDataOffsPos,
            this->vertexDataStride);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'InitExternalForceScl_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif

    // Iterations for new position
    if (!this->updateVtxPos(
            volume_D,
            vboPt,
            vboVtxPathPt,
            volDim,
            volOrg,
            volDelta,
            isovalue,
            (interpMode == INTERP_CUBIC),
            maxIt,
            surfMappedMinDisplScl,
            springStiffness,
            forceScl,
            externalForcesWeight,
            false,
            false)) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::MorphToVolumeDistfield
 */
bool DeformableGPUSurfaceMT::MorphToVolumeDistfield(
        float *volume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue,
        InterpolationMode interpMode,
        size_t maxIt,
        float surfMappedMinDisplScl,
        float springStiffness,
        float forceScl,
        float externalForcesWeight,
        float distfieldDist) {

    using vislib::sys::Log;

    /* Init grid parameters */

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }


    cudaGraphicsResource* cudaTokens[2];

    if ((!this->triangleIdxReady)||(!this->neighboursReady)) {
        return false;
    }

    if (volume_D == NULL) {
        return false;
    }

    // Init vbo with uncertainty information
    if (!this->InitVtxPathVBO(this->vertexCnt)) {
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], this->vboVtxPath,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboVtxPathPt;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboVtxPathPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }


    if (!this->initExtForcesDistfield(
            volume_D,
            vboPt,
            volDim,
            volOrg,
            volDelta,
            distfieldDist,
            isovalue)) {
        return false;
    }


    // Init forces scale factor with -1 or 1, depending on whether they start
    // outside or inside the isosurface
    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt))) {
        return false;
    }
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_InitExternalForceScl_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            (float*)this->vertexExternalForcesScl_D.Peek(),
            this->displLen_D.Peek(),
            volume_D,
            vboPt,
            surfMappedMinDisplScl,
            this->vertexCnt,
            isovalue,
            this->vertexDataOffsPos,
            this->vertexDataStride);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'InitExternalForceScl_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif

    // Iterations for new position
    if (!this->updateVtxPos(
            volume_D,
            vboPt,
            vboVtxPathPt,
            volDim,
            volOrg,
            volDelta,
            isovalue,
            (interpMode == INTERP_CUBIC),
            maxIt,
            surfMappedMinDisplScl,
            springStiffness,
            forceScl,
            externalForcesWeight,
            true)) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::MorphToVolumeGVF
 */
bool DeformableGPUSurfaceMT::MorphToVolumeGVF(float *volumeSource_D,
        float *volumeTarget_D,
        const unsigned int *targetCubeStates_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue,
        InterpolationMode interpMode,
        size_t maxIt,
        float surfMappedMinDisplScl,
        float springStiffness,
        float forceScl,
        float externalForcesWeight,
        float gvfScl,
        unsigned int gvfIt) {

    using namespace vislib::sys;

    /* Init grid parameters */

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }


    using vislib::sys::Log;

    cudaGraphicsResource* cudaTokens[2];

    if ((!this->triangleIdxReady)||(!this->neighboursReady)) {
        return false;
    }

    if (volumeTarget_D == NULL) {
        return false;
    }

    if (!this->initExtForcesGVF(
            volumeTarget_D,
            targetCubeStates_D,
            volDim,
            volOrg,
            volDelta,
            isovalue,
            gvfScl,
            gvfIt)) {
        return false;
    }

    // Init vbo with uncertainty information
    if (!this->InitVtxPathVBO(this->vertexCnt)) {
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], this->vboVtxPath,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboVtxPathPt;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboVtxPathPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }

    // Init forces scale factor with -1 or 1, depending on whether they start
    // outside or inside the isosurface
    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt))) {
        return false;
    }
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_InitExternalForceScl_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            (float*)this->vertexExternalForcesScl_D.Peek(),
            this->displLen_D.Peek(),
            volumeTarget_D,
            vboPt,
            surfMappedMinDisplScl,
            this->vertexCnt,
            isovalue,
            this->vertexDataOffsPos,
            this->vertexDataStride);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'InitExternalForceScl_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif

    // Iterations for new position
    if (!this->updateVtxPos(
            volumeTarget_D,
            vboPt,
            vboVtxPathPt,
            volDim,
            volOrg,
            volDelta,
            isovalue,
            (interpMode == INTERP_CUBIC),
            maxIt,
            surfMappedMinDisplScl,
            springStiffness,
            forceScl,
            externalForcesWeight,
            true)) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::MorphToVolumeTwoWayGVFBM
 */
bool DeformableGPUSurfaceMT::MorphToVolumeTwoWayGVFBM(
        float *volumeSource_D,
        float *volumeTarget_D,
        const unsigned int *cellStatesSource_D,
        const unsigned int *cellStatesTarget_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue,
        InterpolationMode interpMode,
        size_t maxIt,
        float surfMappedMinDisplScl,
        float springStiffness,
        float forceScl,
        float externalForcesWeight,
        float gvfScl,
        unsigned int gvfIt,
        bool trackPath,
        bool recomputeGVF,
        float &t_gvf,
        float &t_map) {

    using vislib::sys::Log;

//    printf("MORPH\n");

    /* Init grid parameters */

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    cudaGraphicsResource* cudaTokens[2];

    if ((!this->triangleIdxReady)||(!this->neighboursReady)) {
        return false;
    }

    if ((volumeTarget_D == NULL)||(volumeSource_D == NULL)) {
        return false;
    }

    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);

    if (recomputeGVF) {
        if (!this->initExtForcesTwoWayGVF(
                volumeSource_D,
                volumeTarget_D,
                cellStatesSource_D,
                cellStatesTarget_D,
                volDim, volOrg, volDelta,
                isovalue, gvfScl, gvfIt)) {
            return false;
        }
    }

    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    t_gvf = dt_ms;

//    printf("GVF %f ms\n", t_gvf);

    if (trackPath) {
        // Init vbo with uncertainty information
        if (!this->InitVtxPathVBO(this->vertexCnt)) {
            return false;
        }
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], this->vboVtxPath,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboVtxPathPt;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboVtxPathPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }

    // Init forces scale factor with -1 or 1, depending on whether they start
    // outside or inside the isosurface
    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt))) {
        return false;
    }

    cudaEventRecord(event1, 0);

    DeformableGPUSurfaceMT_InitExternalForceScl_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            (float*)this->vertexExternalForcesScl_D.Peek(),
            this->displLen_D.Peek(),
            volumeTarget_D,
            vboPt,
            surfMappedMinDisplScl,
            this->vertexCnt,
            isovalue,
            this->vertexDataOffsPos,
            this->vertexDataStride);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'InitExternalForceScl_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif

    // Iterations for new position
    if (!this->updateVtxPos(
            volumeTarget_D,
            vboPt,
            vboVtxPathPt,
            volDim,
            volOrg,
            volDelta,
            isovalue,
            (interpMode == INTERP_CUBIC),
            maxIt,
            surfMappedMinDisplScl,
            springStiffness,
            forceScl,
            externalForcesWeight,
            trackPath, // Track path
            true)) { // Use external forces only
        return false;
    }

    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    t_map = dt_ms;

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::MorphToVolumeTwoWayGVF
 */
bool DeformableGPUSurfaceMT::MorphToVolumeTwoWayGVF(
        float *volumeSource_D,
        float *volumeTarget_D,
        const unsigned int *cellStatesSource_D,
        const unsigned int *cellStatesTarget_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue,
        InterpolationMode interpMode,
        size_t maxIt,
        float surfMappedMinDisplScl,
        float springStiffness,
        float forceScl,
        float externalForcesWeight,
        float gvfScl,
        unsigned int gvfIt,
        bool trackPath,
        bool recomputeGVF) {

    using vislib::sys::Log;

//    printf("MORPH\n");

    /* Init grid parameters */

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    cudaGraphicsResource* cudaTokens[2];

    if ((!this->triangleIdxReady)||(!this->neighboursReady)) {
        return false;
    }

    if ((volumeTarget_D == NULL)||(volumeSource_D == NULL)) {
        return false;
    }
//#define USE_TIMER
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    if (recomputeGVF) {
        if (!this->initExtForcesTwoWayGVF(
                volumeSource_D,
                volumeTarget_D,
                cellStatesSource_D,
                cellStatesTarget_D,
                volDim, volOrg, volDelta,
                isovalue, gvfScl, gvfIt)) {
            return false;
        }
    }

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for GVF:                %.10f sec\n",
            dt_ms/1000.0f);
#endif

    if (trackPath) {
        // Init vbo with uncertainty information
        if (!this->InitVtxPathVBO(this->vertexCnt)) {
            return false;
        }
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], this->vboVtxPath,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboVtxPathPt;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboVtxPathPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }

    // Init forces scale factor with -1 or 1, depending on whether they start
    // outside or inside the isosurface
    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt))) {
        return false;
    }

    DeformableGPUSurfaceMT_InitExternalForceScl_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            (float*)this->vertexExternalForcesScl_D.Peek(),
            this->displLen_D.Peek(),
            volumeTarget_D,
            vboPt,
            surfMappedMinDisplScl,
            this->vertexCnt,
            isovalue,
            this->vertexDataOffsPos,
            this->vertexDataStride);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'InitExternalForceScl_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif

    // Iterations for new position
    if (!this->updateVtxPos(
            volumeTarget_D,
            vboPt,
            vboVtxPathPt,
            volDim,
            volOrg,
            volDelta,
            isovalue,
            (interpMode == INTERP_CUBIC),
            maxIt,
            surfMappedMinDisplScl,
            springStiffness,
            forceScl,
            externalForcesWeight,
            trackPath, // Track path
            true)) { // Use external forces only
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

#undef USE_TIMER

    return true;
}



/*
 * DeformableGPUSurfaceMT::MorphToVolumeTwoWayGVF
 */
bool DeformableGPUSurfaceMT::MorphToVolumeTwoWayGVFSubdiv(
        float *volumeSource_D,
        float *volumeTarget_D,
        const unsigned int *cellStatesSource_D,
        const unsigned int *cellStatesTarget_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue,
        InterpolationMode interpMode,
        size_t maxIt,
        float surfMappedMinDisplScl,
        float springStiffness,
        float forceScl,
        float externalForcesWeight,
        float gvfScl,
        unsigned int gvfIt,
        bool trackPath,
        bool recomputeGVF) {

    using vislib::sys::Log;

//    printf("MORPH\n");

    /* Init grid parameters */

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    cudaGraphicsResource* cudaTokens[2];

    if ((!this->triangleIdxReady)||(!this->neighboursReady)) {
        return false;
    }

    if ((volumeTarget_D == NULL)||(volumeSource_D == NULL)) {
        return false;
    }

    if (recomputeGVF) {
        if (!this->initExtForcesTwoWayGVF(
                volumeSource_D,
                volumeTarget_D,
                cellStatesSource_D,
                cellStatesTarget_D,
                volDim, volOrg, volDelta,
                isovalue, gvfScl, gvfIt)) {
            return false;
        }
    }

    if (trackPath) {
        // Init vbo with uncertainty information
        if (!this->InitVtxPathVBO(this->vertexCnt)) {
            return false;
        }
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], this->vboVtxPath,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboVtxPathPt;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboVtxPathPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }

    // Init forces scale factor with -1 or 1, depending on whether they start
    // outside or inside the isosurface
    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt))) {
        return false;
    }
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_InitExternalForceScl_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            (float*)this->vertexExternalForcesScl_D.Peek(),
            this->displLen_D.Peek(),
            volumeTarget_D,
            vboPt,
            surfMappedMinDisplScl,
            this->vertexCnt,
            isovalue,
            this->vertexDataOffsPos,
            this->vertexDataStride);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'InitExternalForceScl_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif

    // Iterations for new position
    if (!this->updateVtxPosSubdiv(
            volumeTarget_D,
            vboPt,
            vboVtxPathPt,
            volDim,
            volOrg,
            volDelta,
            isovalue,
            (interpMode == INTERP_CUBIC),
            maxIt,
            surfMappedMinDisplScl,
            springStiffness,
            forceScl,
            externalForcesWeight,
            trackPath, // Track path
            true)) { // Use external forces only
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::operator=
 */
DeformableGPUSurfaceMT& DeformableGPUSurfaceMT::operator=(const DeformableGPUSurfaceMT &rhs) {
    GPUSurfaceMT::operator =(rhs);


    CudaSafeCall(this->vertexExternalForcesScl_D.Validate(rhs.vertexExternalForcesScl_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexExternalForcesScl_D.Peek(),
            rhs.vertexExternalForcesScl_D.PeekConst(),
            this->vertexExternalForcesScl_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->displLen_D.Validate(rhs.displLen_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->displLen_D.Peek(),
            rhs.displLen_D.PeekConst(),
            this->displLen_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->externalForces_D.Validate(rhs.externalForces_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->externalForces_D.Peek(),
            rhs.externalForces_D.PeekConst(),
            this->externalForces_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->laplacian_D.Validate(rhs.laplacian_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->laplacian_D.Peek(),
            rhs.laplacian_D.PeekConst(),
            this->laplacian_D.GetCount()*sizeof(float3),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->laplacian2_D.Validate(rhs.laplacian2_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->laplacian2_D.Peek(),
            rhs.laplacian2_D.PeekConst(),
            this->laplacian2_D.GetCount()*sizeof(float3),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->gvfTmp_D.Validate(rhs.gvfTmp_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->gvfTmp_D.Peek(),
            rhs.gvfTmp_D.PeekConst(),
            this->gvfTmp_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->gvfConstData_D.Validate(rhs.gvfConstData_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->gvfConstData_D.Peek(),
            rhs.gvfConstData_D.PeekConst(),
            this->gvfConstData_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->distField_D.Validate(rhs.distField_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->distField_D.Peek(),
            rhs.distField_D.PeekConst(),
            this->distField_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->accTriangleData_D.Validate(rhs.accTriangleData_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->accTriangleData_D.Peek(),
            rhs.accTriangleData_D.PeekConst(),
            this->accTriangleData_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->corruptTriangles_D.Validate(rhs.corruptTriangles_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->corruptTriangles_D.Peek(),
            rhs.corruptTriangles_D.PeekConst(),
            this->corruptTriangles_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->accTriangleArea_D.Validate(rhs.accTriangleArea_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->accTriangleArea_D.Peek(),
            rhs.accTriangleArea_D.PeekConst(),
            this->accTriangleArea_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));


    /* Make deep copy of corrupt triangle flag buffer */

    if (rhs.vboCorruptTriangleVertexFlag) {
        // Destroy if necessary
        if (this->vboCorruptTriangleVertexFlag) {
            glBindBufferARB(GL_ARRAY_BUFFER, this->vboCorruptTriangleVertexFlag);
            glDeleteBuffersARB(1, &this->vboCorruptTriangleVertexFlag);
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
            this->vboCorruptTriangleVertexFlag = 0;
        }

        // Create vertex buffer object for triangle indices
        glGenBuffersARB(1, &this->vboCorruptTriangleVertexFlag);

        CheckForGLError();

        // Map as copy buffer
        glBindBufferARB(GL_COPY_READ_BUFFER, rhs.vboCorruptTriangleVertexFlag);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, this->vboCorruptTriangleVertexFlag);
        glBufferDataARB(GL_COPY_WRITE_BUFFER,
                sizeof(float)*this->vertexCnt, 0, GL_DYNAMIC_DRAW);
        // Copy data
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                sizeof(float)*this->vertexCnt);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, 0);
        glBindBufferARB(GL_COPY_READ_BUFFER, 0);
        CheckForGLError();
    }


    /* Make deep copy of uncertainty vbo */

    if (rhs.vboVtxPath) {
        // Destroy if necessary
        if (this->vboVtxPath) {
            glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxPath);
            glDeleteBuffersARB(1, &this->vboVtxPath);
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
            this->vboVtxPath = 0;
        }

        // Create vertex buffer object for triangle indices
        glGenBuffersARB(1, &this->vboVtxPath);

        CheckForGLError();

        // Map as copy buffer
        glBindBufferARB(GL_COPY_READ_BUFFER, rhs.vboVtxPath);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, this->vboVtxPath);
        glBufferDataARB(GL_COPY_WRITE_BUFFER,
                sizeof(float)*this->vertexCnt, 0, GL_DYNAMIC_DRAW);
        // Copy data
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                sizeof(float)*this->vertexCnt);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, 0);
        glBindBufferARB(GL_COPY_READ_BUFFER, 0);
        CheckForGLError();
    }

    return *this;
}


/*
 * DeformableGPUSurfaceMT_GetTriangleEdgeCnt_D
 */
__global__ void DeformableGPUSurfaceMT_GetTriangleEdgeCnt_D (
        int *triangleEdgeOffs_D,
        uint *triangleNeighbors_D,
        uint triangleCnt) {

    const uint triIdx = ::getThreadIdx();
    if (triIdx >= triangleCnt) return;

    uint cnt = 0;
    uint n0 = triangleNeighbors_D[3*triIdx+0];
    cnt = cnt + int(n0 > triIdx);
    uint n1 = triangleNeighbors_D[3*triIdx+1];
    cnt = cnt + int(n1 > triIdx);
    uint n2 = triangleNeighbors_D[3*triIdx+2];
    cnt = cnt + int(n2 > triIdx);

    triangleEdgeOffs_D[triIdx] = cnt;
}


__device__ uint2 getAdjEdge_D (uint v0, uint v1, uint v2,
                               uint w0, uint w1, uint w2) {

    int idx0=-1, idx1=-1;
    int v[3], w[3];
    v[0] = v0; v[1] = v1; v[2] = v2;
    w[0] = w0; w[1] = w1; w[2] = w2;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (v[i] == w[j]) {
                if (idx0 < 0) {
                    idx0 = v[i];
                } else {
                    if (v[i] != idx0) {
                        idx1 = v[i];
                    }
                }
            }
        }
    }

   return make_uint2(idx0, idx1);
}


__device__ bool hasAdjEdge_D (uint v0, uint v1, uint v2,
                               uint w0, uint w1, uint w2) {

    int cnt = 0;
    int idx0 = -1;
    int v[3], w[3];
    v[0] = v0; v[1] = v1; v[2] = v2;
    w[0] = w0; w[1] = w1; w[2] = w2;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (v[i] == w[j]) {
                if (idx0 < 0) {
                    idx0 = v[i];
                    cnt++;
                } else {
                    if (v[i] != idx0) {
                        cnt++;
                    }
                }
            }
        }
    }

    if (cnt >=2) return true;
    else return false;
}


/*
 * DeformableGPUSurfaceMT_BuildEdgeList_D
 */
__global__ void DeformableGPUSurfaceMT_BuildEdgeList_D (
        uint *edgeList_D,
        int *triangleEdgeOffs_D,
        uint *triangleNeighbors_D,
        uint *triangleIdx_D,
        uint triangleCnt) {

    const uint triIdx = ::getThreadIdx();
    if (triIdx >= triangleCnt) return;

    uint3 idx = make_uint3(triangleIdx_D[3*triIdx+0],
                           triangleIdx_D[3*triIdx+1],
                           triangleIdx_D[3*triIdx+2]);



    uint cnt = 0;
    uint n0 = triangleNeighbors_D[3*triIdx+0];
    uint offs = triangleEdgeOffs_D[triIdx];
    // TODO Comparing all three vertex indices necessary? Use only two?
    if (n0 > triIdx) {
        uint3 nIdx = make_uint3(triangleIdx_D[3*n0+0],
                               triangleIdx_D[3*n0+1],
                               triangleIdx_D[3*n0+2]);
        uint2 e = getAdjEdge_D(idx.x, idx.y, idx.z, nIdx.x, nIdx.y, nIdx.z);
//        printf("%u %u: %u %u %u, %u %u %u\n", e.x, e.y, idx.x, idx.y, idx.z, nIdx.x, nIdx.y, nIdx.z);
        edgeList_D[2*offs+0] = e.x;
        edgeList_D[2*offs+1] = e.y;
//        printf("edge %u %u\n", e.x, e.y);
        cnt++;
    }
    uint n1 = triangleNeighbors_D[3*triIdx+1];
    if (n1 > triIdx) {
        uint3 nIdx = make_uint3(triangleIdx_D[3*n1+0],
                                triangleIdx_D[3*n1+1],
                                triangleIdx_D[3*n1+2]);
        uint2 e = getAdjEdge_D(idx.x, idx.y, idx.z, nIdx.x, nIdx.y, nIdx.z);
        edgeList_D[2*(offs+cnt)+0] = e.x;
        edgeList_D[2*(offs+cnt)+1] = e.y;
        cnt++;
    }
    uint n2 = triangleNeighbors_D[3*triIdx+2];
    if (n2 > triIdx) {
        uint3 nIdx = make_uint3(triangleIdx_D[3*n2+0],
                                triangleIdx_D[3*n2+1],
                                triangleIdx_D[3*n2+2]);
        uint2 e = getAdjEdge_D(idx.x, idx.y, idx.z, nIdx.x, nIdx.y, nIdx.z);
        edgeList_D[2*(offs+cnt)+0] = e.x;
        edgeList_D[2*(offs+cnt)+1] = e.y;
    }
}


__device__ uint getLocalEdgeOffsInTriangle_D(
        uint i0,
        uint i1,
        uint *triangleNeighbors_D,
        uint *triangleIdx_D,
        uint triIdx) {

    uint cnt = 0;

    uint v[3];
    v[0] = triangleIdx_D[3*triIdx+0];
    v[1] = triangleIdx_D[3*triIdx+1];
    v[2] = triangleIdx_D[3*triIdx+2];

    uint n[3];
    n[0] = triangleNeighbors_D[3*triIdx+0];
    n[1] = triangleNeighbors_D[3*triIdx+1];
    n[2] = triangleNeighbors_D[3*triIdx+2];

    for (int i = 0; i < 3; ++i) {
        if (n[i] < triIdx) continue; // This edge is not associated with this triangle
        if ((v[i] == i0)&&(v[(i+1)%3] == i1)||
            (v[i] == i1)&&(v[(i+1)%3] == i0)) {
            cnt++;
            break;
        } else {
            cnt++;
        }
    }

    return cnt-1;
}


/*
 * DeformableGPUSurfaceMT_ComputeTriEdgeList_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeTriEdgeList_D (
        uint *triEdgeList_D,
        int *triangleEdgeOffs_D,
        uint *triangleNeighbors_D,
        uint *triangleIdx_D,
        uint triangleCnt) {

    const uint triIdx = ::getThreadIdx();
    if (triIdx >= triangleCnt) return;

    uint3 idx = make_uint3(triangleIdx_D[3*triIdx+0],
                           triangleIdx_D[3*triIdx+1],
                           triangleIdx_D[3*triIdx+2]);

    //uint offs = triangleEdgeOffs_D[triIdx];

    // Get first edge
    uint n0 = triangleNeighbors_D[3*triIdx+0];
    uint nGlobalOffs;
    uint nLocalOffs;
    if (n0 < triIdx) { // Edge is associated with neighbor
        nGlobalOffs = triangleEdgeOffs_D[n0];
        nLocalOffs = getLocalEdgeOffsInTriangle_D(idx.x, idx.y,
                triangleNeighbors_D, triangleIdx_D, n0);
    } else { // Egde is associated with self
        nGlobalOffs = triangleEdgeOffs_D[triIdx];
        nLocalOffs = getLocalEdgeOffsInTriangle_D(idx.x, idx.y,
                triangleNeighbors_D, triangleIdx_D, triIdx);
    }
    triEdgeList_D[3*triIdx+0] = nGlobalOffs + nLocalOffs;

    // Get second edge
    uint n1 = triangleNeighbors_D[3*triIdx+1];
    if (n1 < triIdx) { // Edge is associated with neighbor
        nGlobalOffs = triangleEdgeOffs_D[n1];
        nLocalOffs = getLocalEdgeOffsInTriangle_D(idx.y, idx.z,
                triangleNeighbors_D, triangleIdx_D, n1);
    } else { // Egde is associated with self
        nGlobalOffs = triangleEdgeOffs_D[triIdx];
        nLocalOffs = getLocalEdgeOffsInTriangle_D(idx.y, idx.z,
                triangleNeighbors_D, triangleIdx_D, triIdx);
    }
    triEdgeList_D[3*triIdx+1] = nGlobalOffs + nLocalOffs;

    // Get third edge
    uint n2 = triangleNeighbors_D[3*triIdx+2];
    if (n2 < triIdx) { // Edge is associated with neighbor
        nGlobalOffs = triangleEdgeOffs_D[n2];
        nLocalOffs = getLocalEdgeOffsInTriangle_D(idx.z, idx.x,
                triangleNeighbors_D, triangleIdx_D, n2);
    } else { // Egde is associated with self
        nGlobalOffs = triangleEdgeOffs_D[triIdx];
        nLocalOffs = getLocalEdgeOffsInTriangle_D(idx.z, idx.x,
                triangleNeighbors_D, triangleIdx_D, triIdx);
    }
    triEdgeList_D[3*triIdx+2] = nGlobalOffs + nLocalOffs;
}


__global__ void FlagLongEdges_D(
        uint *edgeFlag_D,
        uint *edges_D,
        float *vertexData_D,
        float maxLenSqrt,
        uint edgeCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= edgeCnt) return;

    float3 pos0 = make_float3(vertexData_D[9*edges_D[2*idx+0]+0],
                              vertexData_D[9*edges_D[2*idx+0]+1],
                              vertexData_D[9*edges_D[2*idx+0]+2]);

    float3 pos1 = make_float3(vertexData_D[9*edges_D[2*idx+1]+0],
                              vertexData_D[9*edges_D[2*idx+1]+1],
                              vertexData_D[9*edges_D[2*idx+1]+2]);

    float lenSqrt = (pos0.x - pos1.x)*(pos0.x - pos1.x) +
                (pos0.y - pos1.y)*(pos0.y - pos1.y) +
                (pos0.z - pos1.z)*(pos0.z - pos1.z);

    edgeFlag_D[idx] = uint(lenSqrt > maxLenSqrt);

}


__global__ void ComputeNewVertices(
        float *newVertices_D,
        float *vertexFlag_D,
        uint *subDivEdgeIdxOffs_D,
        uint *edgeFlag_D,
        uint *edges_D,
        float *vertexData_D,
        uint oldVertexCnt,
        uint edgeCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= edgeCnt) return;
    if (edgeFlag_D[idx] == 0) return;

    float3 pos0 = make_float3(vertexData_D[9*edges_D[2*idx+0]+0],
                              vertexData_D[9*edges_D[2*idx+0]+1],
                              vertexData_D[9*edges_D[2*idx+0]+2]);

    float3 pos1 = make_float3(vertexData_D[9*edges_D[2*idx+1]+0],
                              vertexData_D[9*edges_D[2*idx+1]+1],
                              vertexData_D[9*edges_D[2*idx+1]+2]);

    float3 posNew = (pos1+pos0)*0.5;

    uint edgeIdxOffs = subDivEdgeIdxOffs_D[idx];

    newVertices_D[3*edgeIdxOffs+0] = posNew.x;
    newVertices_D[3*edgeIdxOffs+1] = posNew.y;
    newVertices_D[3*edgeIdxOffs+2] = posNew.z;
    vertexFlag_D[oldVertexCnt+edgeIdxOffs] = 1.0; // mark this vertex as new
//    printf("Vertex %f %f %f\n", posNew.x, posNew.y, posNew.z);
}


__global__ void ComputeSubdivCnt_D(
        uint *subdivCnt_D,
        uint *triangleEdgeList_D,
        uint *edgeFlag_D,
        uint *edges_D,
        uint *oldTrianglesIdxOffset,
        uint triangleCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= triangleCnt) return;

    uint edgeIdx0 = triangleEdgeList_D[3*idx+0];
    uint edgeIdx1 = triangleEdgeList_D[3*idx+1];
    uint edgeIdx2 = triangleEdgeList_D[3*idx+2];

    bool flag0 = bool(edgeFlag_D[edgeIdx0]);
    bool flag1 = bool(edgeFlag_D[edgeIdx1]);
    bool flag2 = bool(edgeFlag_D[edgeIdx2]);

    if (flag0 && flag1 && flag2) {
        subdivCnt_D[idx] = 4;
        oldTrianglesIdxOffset[idx] = 0;
    } else if ((flag0 && flag1)||(flag1 && flag2)||(flag2 && flag0)) {
        subdivCnt_D[idx] = 3;
        oldTrianglesIdxOffset[idx] = 0;
    } else if (flag0 || flag1 || flag2) {
        subdivCnt_D[idx] = 2;
        oldTrianglesIdxOffset[idx] = 0;
    } else {
        subdivCnt_D[idx] = 0;
        oldTrianglesIdxOffset[idx] = 1;
    }
}


// TODO Orientation of new triangles should match neighbor triangles
__global__ void ComputeSubdiv_D(
        uint *newTriangles,
        uint *newTriangleIdxOffsets,
        uint *triangleEdgeList_D,
        uint *triangleIdx_D,
        uint *edgeFlag_D,
        uint *edges_D,
        uint *subDivEdgeIdxOffs_D,
        uint *oldSubDivLevels_D,
        uint *subDivLevels_D,
        uint *oldTrianglesIdxOffsets_D,
        uint vertexCntOld,
        uint keptTrianglesCnt,
        uint triangleCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= triangleCnt) return;

    uint edgeIdx0 = triangleEdgeList_D[3*idx+0];
    uint edgeIdx1 = triangleEdgeList_D[3*idx+1];
    uint edgeIdx2 = triangleEdgeList_D[3*idx+2];

    bool flag0 = bool(edgeFlag_D[edgeIdx0]);
    bool flag1 = bool(edgeFlag_D[edgeIdx1]);
    bool flag2 = bool(edgeFlag_D[edgeIdx2]);

    uint v0 = triangleIdx_D[3*idx+0];
    uint v1 = triangleIdx_D[3*idx+1];
    uint v2 = triangleIdx_D[3*idx+2];

    uint e0 = triangleEdgeList_D[3*idx+0];
    uint e1 = triangleEdgeList_D[3*idx+1];
    uint e2 = triangleEdgeList_D[3*idx+2];

    uint triIdxOffs = newTriangleIdxOffsets[idx];

    if (flag0 && flag1 && flag2) { // Spawn 4 new triangles

        uint vNew0 = vertexCntOld + subDivEdgeIdxOffs_D[e0];
        uint vNew1 = vertexCntOld + subDivEdgeIdxOffs_D[e1];
        uint vNew2 = vertexCntOld + subDivEdgeIdxOffs_D[e2];

        // #0
        newTriangles[3*triIdxOffs+0] = v0;
        newTriangles[3*triIdxOffs+1] = vNew0;
        newTriangles[3*triIdxOffs+2] = vNew2;
        // #1
        newTriangles[3*triIdxOffs+3] = v1;
        newTriangles[3*triIdxOffs+4] = vNew1;
        newTriangles[3*triIdxOffs+5] = vNew0;
        // #2
        newTriangles[3*triIdxOffs+6] = v2;
        newTriangles[3*triIdxOffs+7] = vNew2;
        newTriangles[3*triIdxOffs+8] = vNew1;
        // #3
        newTriangles[3*triIdxOffs+9] = vNew0;
        newTriangles[3*triIdxOffs+10] = vNew1;
        newTriangles[3*triIdxOffs+11] = vNew2;

        // Write subdiv levels
        uint parentSubdiv = oldSubDivLevels_D[idx];
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+0] = parentSubdiv + 1;
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+1] = parentSubdiv + 1;
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+2] = parentSubdiv + 1;
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+3] = parentSubdiv + 1;


    } else if (flag0 && flag1) { // Spawn 3 new triangles

        uint vNew0 = vertexCntOld + subDivEdgeIdxOffs_D[e0];
        uint vNew1 = vertexCntOld + subDivEdgeIdxOffs_D[e1];

        // #0
        newTriangles[3*triIdxOffs+0] = v1;
        newTriangles[3*triIdxOffs+1] = vNew1;
        newTriangles[3*triIdxOffs+2] = vNew0;
        // #1
        newTriangles[3*triIdxOffs+3] = v0;
        newTriangles[3*triIdxOffs+4] = vNew0;
        newTriangles[3*triIdxOffs+5] = vNew1;
        // #2
        newTriangles[3*triIdxOffs+6] = v2;
        newTriangles[3*triIdxOffs+7] = v0;
        newTriangles[3*triIdxOffs+8] = vNew1;

        // Write subdiv levels
        uint parentSubdiv = oldSubDivLevels_D[idx];
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+0] = parentSubdiv + 1;
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+1] = parentSubdiv + 1;
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+2] = parentSubdiv + 1;

    } else if (flag1 && flag2) { // Spawn 3 new triangles

        uint vNew1 = vertexCntOld + subDivEdgeIdxOffs_D[e1];
        uint vNew2 = vertexCntOld + subDivEdgeIdxOffs_D[e2];

        // #0
        newTriangles[3*triIdxOffs+0] = v2;
        newTriangles[3*triIdxOffs+1] = vNew2;
        newTriangles[3*triIdxOffs+2] = vNew1;
        // #1
        newTriangles[3*triIdxOffs+3] = v0;
        newTriangles[3*triIdxOffs+4] = vNew1;
        newTriangles[3*triIdxOffs+5] = vNew2;
        // #2
        newTriangles[3*triIdxOffs+6] = v0;
        newTriangles[3*triIdxOffs+7] = v1;
        newTriangles[3*triIdxOffs+8] = vNew1;

        // Write subdiv levels
        uint parentSubdiv = oldSubDivLevels_D[idx];
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+0] = parentSubdiv + 1;
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+1] = parentSubdiv + 1;
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+2] = parentSubdiv + 1;

    } else if (flag2 && flag0) { // Spawn 3 new triangles

        uint vNew2 = vertexCntOld + subDivEdgeIdxOffs_D[e2];
        uint vNew0 = vertexCntOld + subDivEdgeIdxOffs_D[e0];

        // #0
        newTriangles[3*triIdxOffs+0] = v0;
        newTriangles[3*triIdxOffs+1] = vNew0;
        newTriangles[3*triIdxOffs+2] = vNew2;
        // #1
        newTriangles[3*triIdxOffs+3] = v2;
        newTriangles[3*triIdxOffs+4] = vNew2;
        newTriangles[3*triIdxOffs+5] = vNew0;
        // #2
        newTriangles[3*triIdxOffs+6] = v1;
        newTriangles[3*triIdxOffs+7] = v2;
        newTriangles[3*triIdxOffs+8] = vNew0;

        // Write subdiv levels
        uint parentSubdiv = oldSubDivLevels_D[idx];
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+0] = parentSubdiv + 1;
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+1] = parentSubdiv + 1;
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+2] = parentSubdiv + 1;

    } else if (flag0) { // Spawn 2 new triangles

        uint vNew0 = vertexCntOld + subDivEdgeIdxOffs_D[e0];

        // #0
        newTriangles[3*triIdxOffs+0] = v0;
        newTriangles[3*triIdxOffs+1] = vNew0;
        newTriangles[3*triIdxOffs+2] = v2;
        // #1
        newTriangles[3*triIdxOffs+3] = v1;
        newTriangles[3*triIdxOffs+4] = v2;
        newTriangles[3*triIdxOffs+5] = vNew0;

        // Write subdiv levels
        uint parentSubdiv = oldSubDivLevels_D[idx];
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+0] = parentSubdiv + 1;
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+1] = parentSubdiv + 1;

    } else if (flag1) { // Spawn 2 new triangles

        uint vNew1 = vertexCntOld + subDivEdgeIdxOffs_D[e1];

        // #0
        newTriangles[3*triIdxOffs+0] = v0;
        newTriangles[3*triIdxOffs+1] = v1;
        newTriangles[3*triIdxOffs+2] = vNew1;
        // #1
        newTriangles[3*triIdxOffs+3] = v0;
        newTriangles[3*triIdxOffs+4] = vNew1;
        newTriangles[3*triIdxOffs+5] = v2;

        // Write subdiv levels
        uint parentSubdiv = oldSubDivLevels_D[idx];
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+0] = parentSubdiv + 1;
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+1] = parentSubdiv + 1;

    } else if (flag2) { // Spawn 2 new triangles

        uint vNew2 = vertexCntOld + subDivEdgeIdxOffs_D[e2];

        // #0
        newTriangles[3*triIdxOffs+0] = v0;
        newTriangles[3*triIdxOffs+1] = v1;
        newTriangles[3*triIdxOffs+2] = vNew2;
        // #1
        newTriangles[3*triIdxOffs+3] = v1;
        newTriangles[3*triIdxOffs+4] = v2;
        newTriangles[3*triIdxOffs+5] = vNew2;

        // Write subdiv levels
        uint parentSubdiv = oldSubDivLevels_D[idx];
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+0] = parentSubdiv + 1;
        subDivLevels_D[keptTrianglesCnt+triIdxOffs+1] = parentSubdiv + 1;
    } else {
        // Write back subdiv level
        subDivLevels_D[oldTrianglesIdxOffsets_D[idx]] = oldSubDivLevels_D[idx];
    }
}


// TODO: !!! This method assumed a certain ordering in the three neighbors of
//       !!! a triangle. Is this actually true?
__global__ void ComputeSubdivTriNeighbors_D (
        uint *newTriangleNeighbors_D,
        uint *oldTriangleNeighbors_D,
        uint *newTriangleIdxOffsets,
        uint *triangleEdgeList_D,
        uint *triangleIdx_D,
        uint *edgeFlag_D,
        uint *edges_D,
        uint *subDivEdgeIdxOffs_D,
        uint *subdivCnt_D,
        uint *oldTriangleIdxOffset,
        uint *newTriangles_D,
        uint vertexCntOld,
        uint numberOfKeptTriangles,
        uint oldTriangleCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= oldTriangleCnt) return;

    uint edgeIdx0 = triangleEdgeList_D[3*idx+0];
    uint edgeIdx1 = triangleEdgeList_D[3*idx+1];
    uint edgeIdx2 = triangleEdgeList_D[3*idx+2];

    bool flag0 = bool(edgeFlag_D[edgeIdx0]);
    bool flag1 = bool(edgeFlag_D[edgeIdx1]);
    bool flag2 = bool(edgeFlag_D[edgeIdx2]);

    uint v0 = triangleIdx_D[3*idx+0];
    uint v1 = triangleIdx_D[3*idx+1];
    uint v2 = triangleIdx_D[3*idx+2];

    uint e0 = triangleEdgeList_D[3*idx+0];
    uint e1 = triangleEdgeList_D[3*idx+1];
    uint e2 = triangleEdgeList_D[3*idx+2];

    uint triIdxOffs = newTriangleIdxOffsets[idx];

    if (!(flag0 || flag1 || flag2)) { // No subdivision

        uint newIdx = oldTriangleIdxOffset[idx];

        uint oldN0 = oldTriangleNeighbors_D[3*idx+0];
        uint oldN1 = oldTriangleNeighbors_D[3*idx+1];
        uint oldN2 = oldTriangleNeighbors_D[3*idx+2];

        uint subDivCntN0 = subdivCnt_D[oldN0];
        if (subDivCntN0 > 0) {
            for (int i = 0; i < subDivCntN0; ++i) {
                uint u0, u1, u2;
                u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+0];
                u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+1];
                u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+2];
                if (hasAdjEdge_D (v0, v1, v2, u0, u1, u2)) {
                    newTriangleNeighbors_D[3*newIdx+0] =
                            numberOfKeptTriangles + newTriangleIdxOffsets[oldN0]+i;
                }
            }
        } else {
            newTriangleNeighbors_D[3*newIdx+0] = oldTriangleIdxOffset[oldN0];
        }

        uint subDivCntN1 = subdivCnt_D[oldN1];
        if (subDivCntN1 > 0) {
            for (int i = 0; i < subDivCntN1; ++i) {
                uint u0, u1, u2;
                u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+0];
                u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+1];
                u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+2];
                if (hasAdjEdge_D (v0, v1, v2, u0, u1, u2)) {
                    newTriangleNeighbors_D[3*newIdx+1]=
                            numberOfKeptTriangles + newTriangleIdxOffsets[oldN1]+i;
                }
            }
        } else {
            newTriangleNeighbors_D[3*newIdx+1] = oldTriangleIdxOffset[oldN1];
        }

        uint subDivCntN2 = subdivCnt_D[oldN2];
        if (subDivCntN2 > 0) {
            for (int i = 0; i < subDivCntN2; ++i) {
                uint u0, u1, u2;
                u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+0];
                u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+1];
                u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+2];
                if (hasAdjEdge_D (v0, v1, v2, u0, u1, u2)) {
                    newTriangleNeighbors_D[3*newIdx+2]=
                            numberOfKeptTriangles + newTriangleIdxOffsets[oldN2]+i;
                }
            }
        } else {
            newTriangleNeighbors_D[3*newIdx+2] = oldTriangleIdxOffset[oldN2];
        }

    } else if (flag0 && !flag1 && !flag2) { // 2 new triangles have been spawned

        uint vNew0 = vertexCntOld + subDivEdgeIdxOffs_D[e0];

        // Get index of neighbors of old triangle
        uint oldN0 = oldTriangleNeighbors_D[3*idx+0];
        uint oldN1 = oldTriangleNeighbors_D[3*idx+1];
        uint oldN2 = oldTriangleNeighbors_D[3*idx+2];


        /* Get neighbors of triangle #0 */

        // Get respective vertex indices of this triangle
        uint w0 = v0;
        uint w1 = vNew0;
        uint w2 = v2;

        // This neighbor has to be determined by comparing vertex indices
        uint subDivCntN0 = subdivCnt_D[oldN0];
        if (subDivCntN0 > 0) {
            for (int i = 0; i < subDivCntN0; ++i) {
                uint u0, u1, u2;
                u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+0];
                u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+1];
                u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+2];
                if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                    newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0]=
                            numberOfKeptTriangles + newTriangleIdxOffsets[oldN0]+i;
                }
            }
        } else {
            newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0]=
                    oldTriangleIdxOffset[oldN0];
        }

        // This neighbor is the other subdivision
        newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+1] =
                numberOfKeptTriangles+triIdxOffs+1;

        // This neighbor has to be determined by comparing vertex indices
        uint subDivCntN2 = subdivCnt_D[oldN2];
        if (subDivCntN2 > 0) {
            for (int i = 0; i < subDivCntN2; ++i) {
                uint u0, u1, u2;
                u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+0];
                u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+1];
                u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+2];
                if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                    newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2]=
                            numberOfKeptTriangles + newTriangleIdxOffsets[oldN2]+i;
                }
            }
        } else {
            newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2]=
                    oldTriangleIdxOffset[oldN2];
        }

        /* Get neighbors of triangle #1 */

        // Get respective vertex indices of this triangle
        w0 = v1;
        w1 = v2;
        w2 = vNew0;

        // This neighbor has to be determined by comparing vertex indices
        uint subDivCntN1 = subdivCnt_D[oldN1];
        if (subDivCntN1 > 0) {
            for (int i = 0; i < subDivCntN1; ++i) {
                uint u0, u1, u2;
                u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+0];
                u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+1];
                u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+2];
                if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                    newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+3]=
                            numberOfKeptTriangles + newTriangleIdxOffsets[oldN1]+i;
                }
            }
        } else {
            newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+3]=
                    oldTriangleIdxOffset[oldN1];
        }

        // This neighbor is the other subdivision
        newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+4] =
                numberOfKeptTriangles+triIdxOffs;

        // This neighbor has to be determined by comparing vertex indices
        subDivCntN0 = subdivCnt_D[oldN0];
        if (subDivCntN0 > 0) {
            for (int i = 0; i < subDivCntN0; ++i) {
                uint u0, u1, u2;
                u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+0];
                u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+1];
                u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+2];
                if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                    newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+5]=
                            numberOfKeptTriangles + newTriangleIdxOffsets[oldN0]+i;
                }
            }
        } else {
            newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+5]=
                    oldTriangleIdxOffset[oldN0];
        }

    } else if (!flag0 && flag1 && !flag2) { // 2 new triangles have been spawned

        uint vNew1 = vertexCntOld + subDivEdgeIdxOffs_D[e1];

        // Get index of neighbors of old triangle
        uint oldN0 = oldTriangleNeighbors_D[3*idx+0];
        uint oldN1 = oldTriangleNeighbors_D[3*idx+1];
        uint oldN2 = oldTriangleNeighbors_D[3*idx+2];

        // #0
        uint w0 = v0;
        uint w1 = v1;
        uint w2 = vNew1;


        // This neighbor has to be determined by comparing vertex indices
        uint subDivCntN0 = subdivCnt_D[oldN0];
        if (subDivCntN0 > 0) {
            for (int i = 0; i < subDivCntN0; ++i) {
                uint u0, u1, u2;
                u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+0];
                u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+1];
                u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+2];
                if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                    newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0]=
                            numberOfKeptTriangles + newTriangleIdxOffsets[oldN0]+i;
                }
            }
        } else {
            newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0]=
                    oldTriangleIdxOffset[oldN0];
        }

        // This neighbor has to be determined by comparing vertex indices
        uint subDivCntN1 = subdivCnt_D[oldN1];
        if (subDivCntN1 > 0) {
            for (int i = 0; i < subDivCntN1; ++i) {
                uint u0, u1, u2;
                u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+0];
                u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+1];
                u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+2];
                if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                    newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+1]=
                            numberOfKeptTriangles + newTriangleIdxOffsets[oldN1]+i;
                }
            }
        } else {
            newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+1]=
                    oldTriangleIdxOffset[oldN1];
        }

        // This neighbor is the other subdivision
        newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2] =
                numberOfKeptTriangles+triIdxOffs+1;

        // #1
        w0 = v0;
        w1 = vNew1;
        w2 = v2;

        // This neighbor is the other subdivision
        newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+3] =
                numberOfKeptTriangles+triIdxOffs;

         // This neighbor has to be determined by comparing vertex indices
         subDivCntN1 = subdivCnt_D[oldN1];
         if (subDivCntN1 > 0) {
             for (int i = 0; i < subDivCntN1; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+4]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN1]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+4]=
                     oldTriangleIdxOffset[oldN1];
         }

         // This neighbor has to be determined by comparing vertex indices
          uint subDivCntN2 = subdivCnt_D[oldN2];
          if (subDivCntN2 > 0) {
              for (int i = 0; i < subDivCntN2; ++i) {
                  uint u0, u1, u2;
                  u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+0];
                  u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+1];
                  u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+2];
                  if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                      newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+5]=
                              numberOfKeptTriangles + newTriangleIdxOffsets[oldN2]+i;
                  }
              }
          } else {
              newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+5]=
                      oldTriangleIdxOffset[oldN2];
          }


    } else if (!flag0 && !flag1 && flag2) { // 2 new triangles have been spawned

        uint vNew2 = vertexCntOld + subDivEdgeIdxOffs_D[e2];

        // Get index of neighbors of old triangle
        uint oldN0 = oldTriangleNeighbors_D[3*idx+0];
        uint oldN1 = oldTriangleNeighbors_D[3*idx+1];
        uint oldN2 = oldTriangleNeighbors_D[3*idx+2];

        // #0
        uint w0 = v0;
        uint w1 = v1;
        uint w2 = vNew2;

        // This neighbor has to be determined by comparing vertex indices
         uint subDivCntN0 = subdivCnt_D[oldN0];
         if (subDivCntN0 > 0) {
             for (int i = 0; i < subDivCntN0; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN0]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0]=
                     oldTriangleIdxOffset[oldN0];
         }

         // This neighbor is the other subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+1] =
                 numberOfKeptTriangles+triIdxOffs + 1;

         // This neighbor has to be determined by comparing vertex indices
         uint subDivCntN2 = subdivCnt_D[oldN2];
         if (subDivCntN2 > 0) {
             for (int i = 0; i < subDivCntN2; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN2]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2]=
                     oldTriangleIdxOffset[oldN2];
         }

        // #1
        w0 = v1;
        w1 = v2;
        w2 = vNew2;


        // This neighbor has to be determined by comparing vertex indices
         uint subDivCntN1 = subdivCnt_D[oldN1];
         if (subDivCntN1 > 0) {
             for (int i = 0; i < subDivCntN1; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+3]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN1]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+3]=
                     oldTriangleIdxOffset[oldN1];
         }

         // This neighbor has to be determined by comparing vertex indices
         subDivCntN2 = subdivCnt_D[oldN2];
         if (subDivCntN2 > 0) {
             for (int i = 0; i < subDivCntN2; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+4]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN2]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+4]=
                     oldTriangleIdxOffset[oldN2];
         }

         // This neighbor is the other subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+5] =
                 numberOfKeptTriangles+triIdxOffs;

    } else if (flag0 && flag1 && !flag2) { // 3 new triangles have been spawned

        uint vNew0 = vertexCntOld + subDivEdgeIdxOffs_D[e0];
        uint vNew1 = vertexCntOld + subDivEdgeIdxOffs_D[e1];

        // Get index of neighbors of old triangle
        uint oldN0 = oldTriangleNeighbors_D[3*idx+0];
        uint oldN1 = oldTriangleNeighbors_D[3*idx+1];
        uint oldN2 = oldTriangleNeighbors_D[3*idx+2];

        // #0
        uint w0 = v1;
        uint w1 = vNew1;
        uint w2 = vNew0;

        // This neighbor has to be determined by comparing vertex indices
         uint subDivCntN1 = subdivCnt_D[oldN1];
         if (subDivCntN1 > 0) {
             for (int i = 0; i < subDivCntN1; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN1]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0] =
                     oldTriangleIdxOffset[oldN1];
         }

         // This neighbor is the other subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+1] =
                 numberOfKeptTriangles+triIdxOffs + 1;

         // This neighbor has to be determined by comparing vertex indices
         uint subDivCntN0 = subdivCnt_D[oldN0];
         if (subDivCntN0 > 0) {
             for (int i = 0; i < subDivCntN0; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN0]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2]=
                     oldTriangleIdxOffset[oldN0];
         }




        // #1
        w0 = v0;
        w1 = vNew0;
        w2 = vNew1;


        // This neighbor has to be determined by comparing vertex indices
         subDivCntN0 = subdivCnt_D[oldN0];
         if (subDivCntN0 > 0) {
             for (int i = 0; i < subDivCntN0; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+3]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN0]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+3] =
                     oldTriangleIdxOffset[oldN0];
         }

         // This neighbor is the other subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+4] =
                 numberOfKeptTriangles+triIdxOffs;

         // This neighbor is the other subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+5] =
                 numberOfKeptTriangles+triIdxOffs + 2;



        // #2
        w0 = v2;
        w1 = v0;
        w2 = vNew1;


        // This neighbor has to be determined by comparing vertex indices
         uint subDivCntN2 = subdivCnt_D[oldN2];
         if (subDivCntN2 > 0) {
             for (int i = 0; i < subDivCntN2; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+6]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN2]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+6] =
                     oldTriangleIdxOffset[oldN2];
         }

         // This neighbor is the other subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs) + 7] =
                 numberOfKeptTriangles+triIdxOffs + 1;

         // This neighbor has to be determined by comparing vertex indices
          subDivCntN1 = subdivCnt_D[oldN1];
          if (subDivCntN1 > 0) {
              for (int i = 0; i < subDivCntN1; ++i) {
                  uint u0, u1, u2;
                  u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+0];
                  u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+1];
                  u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+2];
                  if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                      newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+8]=
                              numberOfKeptTriangles + newTriangleIdxOffsets[oldN1]+i;
                  }
              }
          } else {
              newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+8] =
                      oldTriangleIdxOffset[oldN1];
          }


    } else if (!flag0 && flag1 && flag2) { // 3 new triangles have been spawned

        uint vNew1 = vertexCntOld + subDivEdgeIdxOffs_D[e1];
        uint vNew2 = vertexCntOld + subDivEdgeIdxOffs_D[e2];

        // Get index of neighbors of old triangle
        uint oldN0 = oldTriangleNeighbors_D[3*idx+0];
        uint oldN1 = oldTriangleNeighbors_D[3*idx+1];
        uint oldN2 = oldTriangleNeighbors_D[3*idx+2];

        // #0
        uint w0 = v2;
        uint w1 = vNew2;
        uint w2 = vNew1;


        // This neighbor has to be determined by comparing vertex indices
         uint subDivCntN2 = subdivCnt_D[oldN2];
         if (subDivCntN2 > 0) {
             for (int i = 0; i < subDivCntN2; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN2]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0] =
                     oldTriangleIdxOffset[oldN2];
         }

         // This neighbor is the other subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs) + 1] =
                 numberOfKeptTriangles+triIdxOffs + 1;

         // This neighbor has to be determined by comparing vertex indices
         uint subDivCntN1 = subdivCnt_D[oldN1];
          if (subDivCntN1 > 0) {
              for (int i = 0; i < subDivCntN1; ++i) {
                  uint u0, u1, u2;
                  u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+0];
                  u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+1];
                  u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+2];
                  if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                      newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2]=
                              numberOfKeptTriangles + newTriangleIdxOffsets[oldN1]+i;
                  }
              }
          } else {
              newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2] =
                      oldTriangleIdxOffset[oldN1];
          }




        // #1
        w0 = v0;
        w1 = vNew1;
        w2 = vNew2;

        // This neighbor is the other subdivision
        newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs) + 3] =
                numberOfKeptTriangles+triIdxOffs + 2;

        // This neighbor is the other subdivision
        newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs) + 4] =
                numberOfKeptTriangles+triIdxOffs;


        // This neighbor has to be determined by comparing vertex indices
         subDivCntN2 = subdivCnt_D[oldN2];
         if (subDivCntN2 > 0) {
             for (int i = 0; i < subDivCntN2; ++i) {

                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+5]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN2]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+5] =
                     oldTriangleIdxOffset[oldN2];
         }



        // #2
        w0 = v0;
        w1 = v1;
        w2 = vNew1;


        // This neighbor has to be determined by comparing vertex indices
         uint subDivCntN0 = subdivCnt_D[oldN0];
         if (subDivCntN0 > 0) {
             for (int i = 0; i < subDivCntN0; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+6]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN0]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+6] =
                     oldTriangleIdxOffset[oldN0];
         }

         // This neighbor has to be determined by comparing vertex indices
          subDivCntN1 = subdivCnt_D[oldN1];
          if (subDivCntN1 > 0) {
              for (int i = 0; i < subDivCntN1; ++i) {
                  uint u0, u1, u2;
                  u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+0];
                  u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+1];
                  u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+2];
                  if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                      newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+7]=
                              numberOfKeptTriangles + newTriangleIdxOffsets[oldN1]+i;
                  }
              }
          } else {
              newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+7] =
                      oldTriangleIdxOffset[oldN1];
          }

         // This neighbor is the other subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs) + 8] =
                 numberOfKeptTriangles+triIdxOffs + 1;



    } else if (flag0 && !flag1 && flag2) { // 3 new triangles have been spawned

        // Get index of neighbors of old triangle
        uint oldN0 = oldTriangleNeighbors_D[3*idx+0];
        uint oldN1 = oldTriangleNeighbors_D[3*idx+1];
        uint oldN2 = oldTriangleNeighbors_D[3*idx+2];

        uint vNew2 = vertexCntOld + subDivEdgeIdxOffs_D[e2];
        uint vNew0 = vertexCntOld + subDivEdgeIdxOffs_D[e0];

        // #0
        uint w0 = v0;
        uint w1 = vNew0;
        uint w2 = vNew2;


        // This neighbor has to be determined by comparing vertex indices
        // TODO DEBUG!!
        uint subDivCntN0 = subdivCnt_D[oldN0];
        if (subDivCntN0 > 0) {
            for (int i = 0; i < subDivCntN0; ++i) {
                uint u0, u1, u2;
                u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+0];
                u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+1];
                u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+2];
                if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                    newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0]=
                            numberOfKeptTriangles + newTriangleIdxOffsets[oldN0]+i;
                }
            }
        } else {
            newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0] =
                    oldTriangleIdxOffset[oldN0];
        }

         // This neighbor is the other subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs) + 1] =
                 numberOfKeptTriangles+triIdxOffs + 1;

         // This neighbor has to be determined by comparing vertex indices
          uint subDivCntN2 = subdivCnt_D[oldN2];
          if (subDivCntN2 > 0) {
              for (int i = 0; i < subDivCntN2; ++i) {
                  uint u0, u1, u2;
                  u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+0];
                  u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+1];
                  u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+2];
                  if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                      newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2]=
                              numberOfKeptTriangles + newTriangleIdxOffsets[oldN2]+i;
                  }
              }
          } else {
              newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2] =
                      oldTriangleIdxOffset[oldN2];
          }


        // #1
        w0 = v2;
        w1 = vNew2;
        w2 = vNew0;

        // This neighbor has to be determined by comparing vertex indices
         subDivCntN2 = subdivCnt_D[oldN2];
         if (subDivCntN2 > 0) {
             for (int i = 0; i < subDivCntN2; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+3]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN2]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+3] =
                     oldTriangleIdxOffset[oldN2];
         }

        // This neighbor is the other subdivision
        newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs) + 4] =
                numberOfKeptTriangles+triIdxOffs;

        newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs) + 5] =
                 numberOfKeptTriangles+triIdxOffs+2;



        // #2
        w0 = v1;
        w1 = v2;
        w2 = vNew0;


        // This neighbor has to be determined by comparing vertex indices
         uint subDivCntN1 = subdivCnt_D[oldN1];
         if (subDivCntN1 > 0) {
             for (int i = 0; i < subDivCntN1; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+6]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN1]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+6] =
                     oldTriangleIdxOffset[oldN1];
         }

         // This neighbor is the other subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs) + 7] =
                 numberOfKeptTriangles+triIdxOffs + 1;

         subDivCntN0 = subdivCnt_D[oldN0];
         if (subDivCntN0 > 0) {
             for (int i = 0; i < subDivCntN0; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+8]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN0]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+8] =
                     oldTriangleIdxOffset[oldN0];
         }

    } else if (flag0 && flag1 && flag2) { // 4 new triangles have been spawned

        uint vNew0 = vertexCntOld + subDivEdgeIdxOffs_D[e0];
        uint vNew1 = vertexCntOld + subDivEdgeIdxOffs_D[e1];
        uint vNew2 = vertexCntOld + subDivEdgeIdxOffs_D[e2];

        // Get index of neighbors of old triangle
        uint oldN0 = oldTriangleNeighbors_D[3*idx+0];
        uint oldN1 = oldTriangleNeighbors_D[3*idx+1];
        uint oldN2 = oldTriangleNeighbors_D[3*idx+2];

        // #0
        uint w0 = v0;
        uint w1 = vNew0;
        uint w2 = vNew2;

        // This neighbor has to be determined by comparing vertex indices
         uint subDivCntN0 = subdivCnt_D[oldN0];
         if (subDivCntN0 > 0) {
             for (int i = 0; i < subDivCntN0; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN0]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+0]=
                     oldTriangleIdxOffset[oldN0];
         }

         // This neighbor is the middle subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+1] =
                 numberOfKeptTriangles+triIdxOffs + 3;

         // This neighbor has to be determined by comparing vertex indices
         uint subDivCntN2 = subdivCnt_D[oldN2];
         if (subDivCntN2 > 0) {
             for (int i = 0; i < subDivCntN2; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN2]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+2]=
                     oldTriangleIdxOffset[oldN2];
         }

        // #1
        w0 = v1;
        w1 = vNew1;
        w2 = vNew0;


        // This neighbor has to be determined by comparing vertex indices
        uint subDivCntN1 = subdivCnt_D[oldN1];
         if (subDivCntN1 > 0) {
             for (int i = 0; i < subDivCntN1; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+3]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN1]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+3]=
                     oldTriangleIdxOffset[oldN1];
         }

         // This neighbor is the middle subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+4] =
                 numberOfKeptTriangles+triIdxOffs + 3;

         // This neighbor has to be determined by comparing vertex indices
         subDivCntN0 = subdivCnt_D[oldN0];
         if (subDivCntN0 > 0) {
             for (int i = 0; i < subDivCntN0; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN0]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+5]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN0]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+5]=
                     oldTriangleIdxOffset[oldN0];
         }



        // #2
        w0 = v2;
        w1 = vNew2;
        w2 = vNew1;


        // This neighbor has to be determined by comparing vertex indices
        subDivCntN2 = subdivCnt_D[oldN2];
         if (subDivCntN2 > 0) {
             for (int i = 0; i < subDivCntN2; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN2]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+6]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN2]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+6]=
                     oldTriangleIdxOffset[oldN2];
         }

         // This neighbor is the middle subdivision
         newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+7] =
                 numberOfKeptTriangles+triIdxOffs + 3;

         // This neighbor has to be determined by comparing vertex indices
         subDivCntN1 = subdivCnt_D[oldN1];
         if (subDivCntN1 > 0) {
             for (int i = 0; i < subDivCntN1; ++i) {
                 uint u0, u1, u2;
                 u0 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+0];
                 u1 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+1];
                 u2 = newTriangles_D[3*(newTriangleIdxOffsets[oldN1]+i)+2];
                 if (hasAdjEdge_D (w0, w1, w2, u0, u1, u2)) {
                     newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+8]=
                             numberOfKeptTriangles + newTriangleIdxOffsets[oldN1]+i;
                 }
             }
         } else {
             newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+8]=
                     oldTriangleIdxOffset[oldN1];
         }

        // #3 This is the middle triangle
        w0 = vNew0;
        w1 = vNew1;
        w2 = vNew2;

        // This neighbor is the middle subdivision
        newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+9] =
                numberOfKeptTriangles+triIdxOffs + 1;

        // This neighbor is the middle subdivision
        newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+10] =
                numberOfKeptTriangles+triIdxOffs + 2;

        // This neighbor is the middle subdivision
        newTriangleNeighbors_D[3*(numberOfKeptTriangles+triIdxOffs)+11] =
                numberOfKeptTriangles+triIdxOffs + 0;
    }
}


__global__ void CopyNewDataToVertexBuffer_D(
        float *newVertices_D,
        float *newBuffer_D,
        uint oldVertexCnt,
        uint newVertexCnt) {

    const uint vertexDataStride = 9;

    const uint idx = ::getThreadIdx();
    if (idx >= newVertexCnt) return;

    newBuffer_D[vertexDataStride*(oldVertexCnt+idx)+0] = newVertices_D[3*idx+0];
    newBuffer_D[vertexDataStride*(oldVertexCnt+idx)+1] = newVertices_D[3*idx+1];
    newBuffer_D[vertexDataStride*(oldVertexCnt+idx)+2] = newVertices_D[3*idx+2];

    newBuffer_D[vertexDataStride*(oldVertexCnt+idx)+3] = 1.0; // Normal
    newBuffer_D[vertexDataStride*(oldVertexCnt+idx)+4] = 0.0;
    newBuffer_D[vertexDataStride*(oldVertexCnt+idx)+5] = 0.0;

    newBuffer_D[vertexDataStride*(oldVertexCnt+idx)+6] = 0.0; // TC
    newBuffer_D[vertexDataStride*(oldVertexCnt+idx)+7] = 0.0;
    newBuffer_D[vertexDataStride*(oldVertexCnt+idx)+8] = 0.0;

}

__global__ void CopyOldDataToTriangleBuffer_D(
        uint *oldTriangleIdx_D,
        uint *oldTriangleIdxOffs_D,
        uint *newTriangleIdx_D,
        uint *subdivCnt_D,
        uint oldTriangleCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= oldTriangleCnt) return;
    if (subdivCnt_D[idx] > 0) return; // Subdivided triangles are dismissed
    uint newIdx = oldTriangleIdxOffs_D[idx];

    newTriangleIdx_D[3*newIdx+0] = oldTriangleIdx_D[3*idx+0];
    newTriangleIdx_D[3*newIdx+1] = oldTriangleIdx_D[3*idx+1];
    newTriangleIdx_D[3*newIdx+2] = oldTriangleIdx_D[3*idx+2];
}


/*
 * DeformableGPUSurfaceMT::RefineMesh
 */
int DeformableGPUSurfaceMT::RefineMesh(
        uint maxSubdivLevel,
        float *volume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue,
        float maxEdgeLen) {

    using vislib::sys::Log;

    // Init grid parameters

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return -1;
    }

    cudaGraphicsResource* cudaTokens[2];

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could register buffer",
                this->ClassName());
        return -1;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not register buffer",
                this->ClassName());
        return -1;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not map recources",
                this->ClassName());
        return -1;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not obtain device pointer",
                this->ClassName());
        return -1;
    }

    // Get mapped pointers to the vertex data buffers
    unsigned int *vboTriIdxPt;
    size_t vboTriSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboTriIdxPt), // The mapped pointer
            &vboTriSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not obtain device pointer",
                this->ClassName());
        return -1;
    }

    /* 1. Compute edge list */
//#define USE_TIMER
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2, eventStart, eventEnd;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventCreate(&eventStart);
    cudaEventCreate(&eventEnd);
    cudaEventRecord(event1, 0);
    cudaEventRecord(eventStart, 0);
#endif

    const uint edgeCnt = (this->triangleCnt*3)/2;
//    printf("EDGE COUNT %u\n", edgeCnt);


    // Get the number of edges associated with each triangle
    if (!CudaSafeCall(this->triangleEdgeOffs_D.Validate(this->triangleCnt))) {
        return -1;
    }
    if (!CudaSafeCall(this->triangleEdgeOffs_D.Set(0x00))) {
        return -1;
    }
    // Check whether triangle neighbors have been computed
    if (this->triangleNeighbors_D.GetCount() != this->triangleCnt*3) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: need triangle neighbors",
                this->ClassName());
        return -1;
    }
    DeformableGPUSurfaceMT_GetTriangleEdgeCnt_D <<< Grid(this->triangleCnt, 256), 256 >>>(
           this->triangleEdgeOffs_D.Peek(),
           this->triangleNeighbors_D.Peek(),
           this->triangleCnt);
    if (!CheckForCudaError()) {
        return -1;
    }

    // Compute prefix sum
    thrust::exclusive_scan(
            thrust::device_ptr<int>(this->triangleEdgeOffs_D.Peek()),
            thrust::device_ptr<int>(this->triangleEdgeOffs_D.Peek() + this->triangleCnt),
            thrust::device_ptr<int>(this->triangleEdgeOffs_D.Peek()));

    if (!CheckForCudaError()) {
        return -1;
    }

    // Build up edge list based on the offsets
    if (!CudaSafeCall(this->edges_D.Validate(edgeCnt*2))) {
        return -1;
    }
    if (!CudaSafeCall(this->edges_D.Set(0x00))) {
        return -1;
    }
    DeformableGPUSurfaceMT_BuildEdgeList_D <<< Grid(this->triangleCnt, 256), 256 >>>(
            this->edges_D.Peek(),
            this->triangleEdgeOffs_D.Peek(),
            this->triangleNeighbors_D.Peek(),
            vboTriIdxPt,
            this->triangleCnt);
    if (!CheckForCudaError()) {
        return -1;
    }

//    // DEBUG Print edges
//    this->edges.Validate(this->edges_D.GetCount());
//    if (!CudaSafeCall(this->edges_D.CopyToHost(this->edges.Peek()))){
//        return false;
//    }
//    for (int e = 0; e < edgeCnt; ++e) {
//        printf("EDGE %i: %u %u\n", e,
//                this->edges.Peek()[2*e+0],
//                this->edges.Peek()[2*e+1]);
//    }
//    // END DEBUG

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for computing edge list:                     %.10f sec\n",
            dt_ms/1000.0f);
    cudaEventRecord(event1, 0);
#endif


    /* 2. Flag long edges and determine number of newly created vertices */

    // Build up edge list based on the offsets
    if (!CudaSafeCall(this->subDivEdgeFlag_D.Validate(edgeCnt))) {
        return -1;
    }
    if (!CudaSafeCall(this->subDivEdgeFlag_D.Set(0x00))) { // Set to 'false'
        return -1;
    }
    FlagLongEdges_D <<< Grid(edgeCnt, 256), 256 >>> (
            this->subDivEdgeFlag_D.Peek(),
            this->edges_D.Peek(),
            vboPt,
            maxEdgeLen*maxEdgeLen,
            this->edges_D.GetCount()/2);
    if (!CheckForCudaError()) {
        return -1;
    }

    // Compute prefix sum
    if (!CudaSafeCall(this->subDivEdgeIdxOffs_D.Validate(edgeCnt))) {
        return -1;
    }
    if (!CudaSafeCall(this->subDivEdgeIdxOffs_D.Set(0x00))) { // Set to 'false'
        return -1;
    }
    thrust::exclusive_scan(
            thrust::device_ptr<uint>(this->subDivEdgeFlag_D.Peek()),
            thrust::device_ptr<uint>(this->subDivEdgeFlag_D.Peek() + edgeCnt),
            thrust::device_ptr<uint>(this->subDivEdgeIdxOffs_D.Peek()));

    uint accTmp;
    if (!CudaSafeCall(cudaMemcpy(&accTmp, this->subDivEdgeFlag_D.Peek()+(edgeCnt-1), sizeof(uint),
            cudaMemcpyDeviceToHost))) {
        return -1;
    }
    this->newVertexCnt = accTmp;
    if (!CudaSafeCall(cudaMemcpy(&accTmp, this->subDivEdgeIdxOffs_D.Peek()+(edgeCnt-1), sizeof(uint),
            cudaMemcpyDeviceToHost))) {
        return -1;
    }

    this->newVertexCnt += accTmp;
    this->nFlaggedVertices += this->newVertexCnt;
    if (this->newVertexCnt == 0) {
        // !! Unmap/registers vbos because they will be reinitialized
        CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0));
        CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]));
        CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]));
        return 0;
    }
//    printf("Need %i new vertices (old triangle count %u)\n", newVertexCnt, this->triangleCnt);

//    // DEBUG print edge flag
//    HostArr<uint> edgeFlag;
//    edgeFlag.Validate(this->subDivEdgeFlag_D.GetCount());
//    this->subDivEdgeFlag_D.CopyToHost(edgeFlag.Peek());
//    for (int i = 0; i < edgeCnt; ++i) {
//        printf("EDGEFLAG %i %u\n", i, edgeFlag.Peek()[i]);
//    }
//    edgeFlag.Release();
//    // END DEBUG

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for flagging edges and thrust reduce:        %.10f sec\n",
            dt_ms/1000.0f);
    cudaEventRecord(event1, 0);
#endif


    /* 3. Interpolate new vertex positions associated with the flagged edges */

    if (!CudaSafeCall(this->newVertices_D.Validate(this->newVertexCnt*3))) {
        return -1;
    }

    if (this->vertexFlag_D.GetCount() != this->vertexCnt) { // First subdivision round

        if (!CudaSafeCall(this->vertexFlag_D.Validate(this->newVertexCnt + this->vertexCnt))) {
            return -1;
        }
        if (!CudaSafeCall(this->vertexFlag_D.Set(0x00))) {
            return -1;
        }
    } else { // Need to save old flags

        if (!CudaSafeCall(this->vertexFlagTmp_D.Validate(this->vertexFlag_D.GetCount()))) {
            return -1;
        }
        if (!CudaSafeCall(cudaMemcpy(
                this->vertexFlagTmp_D.Peek(),
                this->vertexFlag_D.Peek(),
                sizeof(float)*this->vertexFlag_D.GetCount(),
                cudaMemcpyDeviceToDevice))) {
            return -1;
        }
        if (!CudaSafeCall(this->vertexFlag_D.Validate(this->newVertexCnt + this->vertexCnt))) {
            return -1;
        }
        if (!CudaSafeCall(this->vertexFlag_D.Set(0x00))) {
            return -1;
        }
        if (!CudaSafeCall(cudaMemcpy(
                this->vertexFlag_D.Peek(),
                this->vertexFlagTmp_D.Peek(),
                sizeof(float)*this->vertexFlagTmp_D.GetCount(),
                cudaMemcpyDeviceToDevice))) {
            return -1;
        }
    }

    ComputeNewVertices <<< Grid(edgeCnt, 256), 256 >>> (
            this->newVertices_D.Peek(),
            this->vertexFlag_D.Peek(),
            this->subDivEdgeIdxOffs_D.Peek(),
            this->subDivEdgeFlag_D.Peek(),
            this->edges_D.Peek(),
            vboPt,
            this->vertexCnt,
            edgeCnt);
    if (!CheckForCudaError()) {
        return -1;
    }

    // Compute number of flagged vertices
    this->nFlaggedVertices = thrust::reduce(
            thrust::device_ptr<float>(this->vertexFlag_D.Peek()),
            thrust::device_ptr<float>(this->vertexFlag_D.Peek() + this->vertexCnt));

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for interpolating new vertices:              %.10f sec\n",
            dt_ms/1000.0f);
    cudaEventRecord(event1, 0);
#endif


    /* 4. Build triangle-edge-list */

    if (this->triangleNeighbors_D.GetCount() != this->triangleCnt*3) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: need triangle neighbors",
                this->ClassName());
        // !! Unmap/registers vbos because they will be reinitialized
        CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0));
        CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]));
        CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]));
        return -1;
    }

    if (!CudaSafeCall(this->triangleEdgeList_D.Validate(this->triangleCnt*3))) {
        return -1;
    }
    DeformableGPUSurfaceMT_ComputeTriEdgeList_D  <<< Grid(this->triangleCnt, 256), 256 >>> (
            this->triangleEdgeList_D.Peek(),
            this->triangleEdgeOffs_D.Peek(),
            this->triangleNeighbors_D.Peek(),
            vboTriIdxPt,
            this->triangleCnt);
    if (!CheckForCudaErrorSync()) {
        return -1;
    }

//    // DEBUG Triangle edge list
//    HostArr<unsigned int> triangleEdgeList;
//    triangleEdgeList.Validate(this->triangleEdgeList_D.GetCount());
//    if (!CudaSafeCall(this->triangleEdgeList_D.CopyToHost(triangleEdgeList.Peek()))){
//        return false;
//    }
//    for (int e = 0; e < this->triangleCnt; ++e) {
//        printf("Tri %i, edges: %u %u %u\n", e,
//                triangleEdgeList.Peek()[3*e+0],
//                triangleEdgeList.Peek()[3*e+1],
//                triangleEdgeList.Peek()[3*e+2]);
//    }
//    triangleEdgeList.Release();
//    // END DEBUG

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for triangle edge list:                      %.10f sec\n",
            dt_ms/1000.0f);
    cudaEventRecord(event1, 0);
#endif

    /* 5. Determine number of newly created triangles */

    if (!CudaSafeCall(this->subDivCnt_D.Validate(this->triangleCnt))) {
        return -1;
    }
    if (!CudaSafeCall(this->subDivCnt_D.Set(0x00))) {
        return -1;
    }
    if (!CudaSafeCall(this->oldTrianglesIdxOffs_D.Validate(this->triangleCnt))) {
        return -1;
    }
    ComputeSubdivCnt_D <<< Grid(this->triangleCnt, 256), 256 >>> (
            this->subDivCnt_D.Peek(),
            this->triangleEdgeList_D.Peek(),
            this->subDivEdgeFlag_D.Peek(),
            this->edges_D.Peek(),
            this->oldTrianglesIdxOffs_D.Peek(),
            this->triangleCnt);
    if (!CheckForCudaErrorSync()) {
        return -1;
    }

    if (!CudaSafeCall(this->newTrianglesIdxOffs_D.Validate(this->triangleCnt))) {
        return -1;
    }

    // Compute prefix sum

    thrust::exclusive_scan(
            thrust::device_ptr<uint>(this->subDivCnt_D.Peek()),
            thrust::device_ptr<uint>(this->subDivCnt_D.Peek() + this->triangleCnt),
            thrust::device_ptr<uint>(this->newTrianglesIdxOffs_D.Peek()));

    uint newTrianglesCnt;
    if (!CudaSafeCall(cudaMemcpy(&accTmp, this->subDivCnt_D.Peek()+(this->triangleCnt-1), sizeof(uint),
            cudaMemcpyDeviceToHost))) {
        return -1;
    }
    newTrianglesCnt = accTmp;
    if (!CudaSafeCall(cudaMemcpy(&accTmp, this->newTrianglesIdxOffs_D.Peek()+(this->triangleCnt-1), sizeof(uint),
            cudaMemcpyDeviceToHost))) {
        return -1;
    }
    newTrianglesCnt += accTmp;
//    printf("Need %i new triangles\n", newTrianglesCnt);

    uint nOldTriangles = thrust::reduce(
            thrust::device_ptr<uint>(this->oldTrianglesIdxOffs_D.Peek()),
            thrust::device_ptr<uint>(this->oldTrianglesIdxOffs_D.Peek() + this->triangleCnt));

    thrust::exclusive_scan(
             thrust::device_ptr<uint>(this->oldTrianglesIdxOffs_D.Peek()),
             thrust::device_ptr<uint>(this->oldTrianglesIdxOffs_D.Peek() + this->triangleCnt),
             thrust::device_ptr<uint>(this->oldTrianglesIdxOffs_D.Peek()));

//     printf("Keep %i old triangles\n", nOldTriangles);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for computing number of new  triangles:      %.10f sec\n",
            dt_ms/1000.0f);
    cudaEventRecord(event1, 0);
#endif



    /* 6. Create new triangles with respective vertex indices */

    if (this->subDivLevels_D.GetCount() != this->triangleCnt) {
        // This is the first subdivision
        if (!CudaSafeCall(this->oldSubDivLevels_D.Validate(this->triangleCnt))) {
            return -1;
        }
        if (!CudaSafeCall(this->oldSubDivLevels_D.Set(0x00))) {
            return -1;
        }
    } else { // Store old subdivision levels
        if (!CudaSafeCall(this->oldSubDivLevels_D.Validate(this->triangleCnt))) {
            return -1;
        }
        if (!CudaSafeCall(cudaMemcpy(this->oldSubDivLevels_D.Peek(),
                this->subDivLevels_D.Peek(), sizeof(unsigned int)*this->triangleCnt,
                cudaMemcpyDeviceToDevice))){
            return -1;
        }
    }
    // Allocate memory for new subdivision levels (old and new triangles)
    if (!CudaSafeCall(this->subDivLevels_D.Validate(nOldTriangles+newTrianglesCnt))) {
        return -1;
    }

    if (!CudaSafeCall(this->newTriangles_D.Validate(newTrianglesCnt*3))) {
        return -1;
    }
    ComputeSubdiv_D <<< Grid(this->triangleCnt, 256), 256 >>> (
            this->newTriangles_D.Peek(),
            this->newTrianglesIdxOffs_D.Peek(),
            this->triangleEdgeList_D.Peek(),
            vboTriIdxPt,
            this->subDivEdgeFlag_D.Peek(),
            this->edges_D.Peek(),
            this->subDivEdgeIdxOffs_D.Peek(),
            this->oldSubDivLevels_D.Peek(),
            this->subDivLevels_D.Peek(),
            this->oldTrianglesIdxOffs_D.Peek(),
            this->vertexCnt,
            nOldTriangles,
            this->triangleCnt);

    if (!CheckForCudaErrorSync()) {
        return -1;
    }

//    // DEBUG Print new triangles
//    HostArr<uint> newTriangles;
//    newTriangles.Validate(this->newTriangles_D.GetCount());
//    this->newTriangles_D.CopyToHost(newTriangles.Peek());
//    for (int i = 0; i < this->newTriangles_D.GetCount()/3; ++i) {
//        printf("NEW TRI %i: %u %u %u\n", i,
//                newTriangles.Peek()[3*i+0],
//                newTriangles.Peek()[3*i+1],
//                newTriangles.Peek()[3*i+2]);
//    }
//    newTriangles.Release();
//    // END DEBUG

//    // DEBUG Print subdivision levels
//    HostArr<uint> subDivisionLevels;
//    subDivisionLevels.Validate(this->subDivLevels_D.GetCount());
//    this->subDivLevels_D.CopyToHost(subDivisionLevels.Peek());
//    for (int i = 0; i < this->subDivLevels_D.GetCount(); ++i) {
//        printf("SUBDIV LVL %i: %u \n", i,
//                subDivisionLevels.Peek()[i]);
//    }
//    subDivisionLevels.Release();
//    // END DEBUG

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for computing new  triangles:                %.10f sec\n",
            dt_ms/1000.0f);
    cudaEventRecord(event1, 0);
#endif



    /* 7. (Re-)compute triangle neighbors */

    if (!CudaSafeCall(this->newTriangleNeighbors_D.Validate((nOldTriangles+newTrianglesCnt)*3))) {
        return -1;
    }
    if (!CudaSafeCall(this->newTriangleNeighbors_D.Set(0x00))) {
        return -1;
    }
    ComputeSubdivTriNeighbors_D <<< Grid(this->triangleCnt, 256), 256 >>> (
            this->newTriangleNeighbors_D.Peek(),
            this->triangleNeighbors_D.Peek(),
            this->newTrianglesIdxOffs_D.Peek(),
            this->triangleEdgeList_D.Peek(),
            vboTriIdxPt,
            this->subDivEdgeFlag_D.Peek(),
            this->edges_D.Peek(),
            this->subDivEdgeIdxOffs_D.Peek(),
            this->subDivCnt_D.Peek(),
            this->oldTrianglesIdxOffs_D.Peek(),
            this->newTriangles_D.Peek(),
            this->vertexCnt,
            nOldTriangles,
            this->triangleCnt);

    // Reallocate old array TODO Simply swap pointers?
    if (!CudaSafeCall(this->triangleNeighbors_D.Validate(this->newTriangleNeighbors_D.GetCount()))) {
        return -1;
    }
    if (!CudaSafeCall(cudaMemcpy(
            this->triangleNeighbors_D.Peek(),
            this->newTriangleNeighbors_D.Peek(),
            this->newTriangleNeighbors_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice))) {
        return -1;
    }

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for updating triangle neighbors:             %.10f sec\n",
            dt_ms/1000.0f);
    cudaEventRecord(event1, 0);
#endif


    /* 8. Update VBOs for vertex data and triangle indices */

//    // DEBUG Print oldTriangles index offset and subdivision count
//    HostArr<unsigned int> oldTrianglesIdxOffs;
//    oldTrianglesIdxOffs.Validate(this->oldTrianglesIdxOffs_D.GetCount());
//    if (!CudaSafeCall(this->oldTrianglesIdxOffs_D.CopyToHost(oldTrianglesIdxOffs.Peek()))) {
//        return -1;
//    }
//    HostArr<unsigned int> subDivCnt;
//    subDivCnt.Validate(this->subDivCnt_D.GetCount());
//    if (!CudaSafeCall(this->subDivCnt_D.CopyToHost(subDivCnt.Peek()))) {
//        return -1;
//    }
//    for (int i = 0; i < this->triangleCnt; ++i) {
//        printf("%i: offs: %u, subdiv %u\n", i, oldTrianglesIdxOffs.Peek()[i],
//                subDivCnt.Peek()[i]);
//    }
//    subDivCnt.Release();
//    oldTrianglesIdxOffs.Release();
//    // END DEBUG

//    // DEBUG print old vertex buffer
//    HostArr<float> vertexBuffer;
//    vertexBuffer.Validate(this->vertexDataStride*this->vertexCnt);
//    cudaMemcpy(vertexBuffer.Peek(), vboPt, vertexBuffer.GetCount()*sizeof(float), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < this->vertexCnt; ++i) {
//        printf("Old Vertex Buffer %i: %f %f %f, %f %f %f, %f %f %f\n", i,
//                vertexBuffer.Peek()[9*i+0],
//                vertexBuffer.Peek()[9*i+1],
//                vertexBuffer.Peek()[9*i+2],
//                vertexBuffer.Peek()[9*i+3],
//                vertexBuffer.Peek()[9*i+4],
//                vertexBuffer.Peek()[9*i+5],
//                vertexBuffer.Peek()[9*i+6],
//                vertexBuffer.Peek()[9*i+7],
//                vertexBuffer.Peek()[9*i+8]);
//    }
//    vertexBuffer.Release();
//    // END DEBUG

//    // DEBUG print old triangle index buffer
//    HostArr<uint> triangleBuffer;
//    triangleBuffer.Validate(3*this->triangleCnt);
//    cudaMemcpy(triangleBuffer.Peek(), vboTriIdxPt,
//            triangleBuffer.GetCount()*sizeof(uint), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < this->triangleCnt; ++i) {
//        printf("Old Triangle Buffer %i: %u %u %u\n",i,
//                triangleBuffer.Peek()[3*i+0],
//                triangleBuffer.Peek()[3*i+1],
//                triangleBuffer.Peek()[3*i+2]);
//    }
//    triangleBuffer.Release();
//    // END DEBUG

    // Make copy of old data
    if (!CudaSafeCall(this->oldTriangles_D.Validate(this->triangleCnt*3))) {
        return -1;
    }
    if (!CudaSafeCall(this->trackedSubdivVertexData_D.Validate(this->vertexCnt*this->vertexDataStride))) {
        return -1;
    }
    if (!CudaSafeCall(cudaMemcpy(this->oldTriangles_D.Peek(), vboTriIdxPt,
            sizeof(unsigned int)*this->oldTriangles_D.GetCount(),
            cudaMemcpyDeviceToDevice))) {
        return -1;
    }
    if (!CudaSafeCall(cudaMemcpy(this->trackedSubdivVertexData_D.Peek(), vboPt,
            sizeof(float)*this->trackedSubdivVertexData_D.GetCount(),
            cudaMemcpyDeviceToDevice))) {
        return -1;
    }

    // !! Unmap/registers vbos because they will be reinitialized
    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return -1;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return -1;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return -1;
    }

    // Re-initialize VBOS
    uint oldVertexCnt = this->vertexCnt;
    this->vertexCnt += newVertexCnt;
    uint oldTriangleCount = this->triangleCnt;
    this->triangleCnt = nOldTriangles + newTrianglesCnt;
    this->InitTriangleIdxVBO(this->triangleCnt);
    this->InitVertexDataVBO(this->vertexCnt);

    // Register and get pointers

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {

        return -1;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {

        return -1;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return -1;
    }

    // Get mapped pointers to the vertex data buffers
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return -1;
    }

    // Get mapped pointers to the vertex data buffers
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboTriIdxPt), // The mapped pointer
            &vboTriSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return -1;
    }

    // Copy old vertex data to new buffer
    if (!CudaSafeCall(cudaMemcpy(vboPt, this->trackedSubdivVertexData_D.Peek(),
            sizeof(float)*this->vertexDataStride*oldVertexCnt,
            cudaMemcpyDeviceToDevice))) {
        return -1;
    }
    // Copy new vertex data to new buffer
    CopyNewDataToVertexBuffer_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            this->newVertices_D.Peek(),
            vboPt,
            oldVertexCnt,
            newVertexCnt);
    if (!CheckForCudaError()) {
        return -1;
    }

//    // DEBUG print old vertex buffer
//    vertexBuffer.Validate(this->vertexDataStride*this->vertexCnt);
//    cudaMemcpy(vertexBuffer.Peek(), vboPt, vertexBuffer.GetCount()*sizeof(float), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < this->vertexCnt; ++i) {
//        printf("New Vertex Buffer %i: %f %f %f, %f %f %f, %f %f %f\n", i,
//                vertexBuffer.Peek()[9*i+0],
//                vertexBuffer.Peek()[9*i+1],
//                vertexBuffer.Peek()[9*i+2],
//                vertexBuffer.Peek()[9*i+3],
//                vertexBuffer.Peek()[9*i+4],
//                vertexBuffer.Peek()[9*i+5],
//                vertexBuffer.Peek()[9*i+6],
//                vertexBuffer.Peek()[9*i+7],
//                vertexBuffer.Peek()[9*i+8]);
//    }
//    vertexBuffer.Release();
//    // END DEBUG


    // Copy old triangle indices to VBO
    CopyOldDataToTriangleBuffer_D <<< Grid(oldTriangleCount, 256), 256 >>> (
            this->oldTriangles_D.Peek(),
            this->oldTrianglesIdxOffs_D.Peek(),
            vboTriIdxPt,
            this->subDivCnt_D.Peek(),
            oldTriangleCount);
    // Copy new data to triangle VBO
    if (!CudaSafeCall(cudaMemcpy(
            vboTriIdxPt + 3*nOldTriangles, // New data starts after old data
            this->newTriangles_D.Peek(),
            sizeof(uint)*this->newTriangles_D.GetCount(),
            cudaMemcpyDeviceToDevice))) {
        return -1;
    }


//    // DEBUG Print new triangle neighbors
//    HostArr<uint> triNeighbors;
//    triNeighbors.Validate(this->triangleNeighbors_D.GetCount());
//    HostArr<uint> triangleBuffer;
//    triangleBuffer.Validate(3*this->triangleCnt);
//    cudaMemcpy(triangleBuffer.Peek(), vboTriIdxPt,
//            triangleBuffer.GetCount()*sizeof(uint), cudaMemcpyDeviceToHost);
//    if (!CudaSafeCall(this->triangleNeighbors_D.CopyToHost(triNeighbors.Peek()))) {
//        return -1;
//    }
//    for (int i = 0; i < this->triangleNeighbors_D.GetCount()/3; ++i) {
//
////        printf("TRI NEIGHBORS %i: %u %u %u\n", i,
////                triNeighbors.Peek()[3*i+0],
////                triNeighbors.Peek()[3*i+1],
////                triNeighbors.Peek()[3*i+2]);
//
//        // Check neighbor consistency
//        uint v0 = triangleBuffer.Peek()[3*i+0];
//        uint v1 = triangleBuffer.Peek()[3*i+1];
//        uint v2 = triangleBuffer.Peek()[3*i+2];
//
//        uint n00 = triangleBuffer.Peek()[3*triNeighbors.Peek()[3*i+0]+0];
//        uint n01 = triangleBuffer.Peek()[3*triNeighbors.Peek()[3*i+0]+1];
//        uint n02 = triangleBuffer.Peek()[3*triNeighbors.Peek()[3*i+0]+2];
//
//        uint n10 = triangleBuffer.Peek()[3*triNeighbors.Peek()[3*i+1]+0];
//        uint n11 = triangleBuffer.Peek()[3*triNeighbors.Peek()[3*i+1]+1];
//        uint n12 = triangleBuffer.Peek()[3*triNeighbors.Peek()[3*i+1]+2];
//
//        uint n20 = triangleBuffer.Peek()[3*triNeighbors.Peek()[3*i+2]+0];
//        uint n21 = triangleBuffer.Peek()[3*triNeighbors.Peek()[3*i+2]+1];
//        uint n22 = triangleBuffer.Peek()[3*triNeighbors.Peek()[3*i+2]+2];
//
////        printf("n0 %u %u %u, n1 %u %u %u, n2 %u %u %u\n",
////                n00, n01, n02, n10, n11, n12, n20, n21, n22);
//
//        uint cnt = 0;
//        bool flag0=false, flag1=false, flag2=false;
//        if (v0 == n00) cnt++; if (v0 == n01) cnt++; if (v0 == n02) cnt++;
//        if (v1 == n00) cnt++; if (v1 == n01) cnt++; if (v1 == n02) cnt++;
//        if (v2 == n00) cnt++; if (v2 == n01) cnt++; if (v2 == n02) cnt++;
//        if (cnt < 2) {
//            flag0 = true;
//
//        }
//
//        cnt = 0;
//        if (v0 == n10) cnt++; if (v0 == n11) cnt++; if (v0 == n12) cnt++;
//        if (v1 == n10) cnt++; if (v1 == n11) cnt++; if (v1 == n12) cnt++;
//        if (v2 == n10) cnt++; if (v2 == n11) cnt++; if (v2 == n12) cnt++;
//        if (cnt < 2) {
//            flag1 = true;
//        }
//
//        cnt = 0;
//        if (v0 == n20) cnt++; if (v0 == n21) cnt++; if (v0 == n22) cnt++;
//        if (v1 == n20) cnt++; if (v1 == n21) cnt++; if (v1 == n22) cnt++;
//        if (v2 == n20) cnt++; if (v2 == n21) cnt++; if (v2 == n22) cnt++;
//        if (cnt < 2) {
//            flag2 = true;
//        }
//
//        if (flag0||flag1||flag2) {
//            printf("TRI NEIGHBORS %i: %u %u %u\n", i,
//                    triNeighbors.Peek()[3*i+0],
//                    triNeighbors.Peek()[3*i+1],
//                    triNeighbors.Peek()[3*i+2]);
//        }
//        if (flag0) printf("----> %u inconsistent\n", triNeighbors.Peek()[3*i+0]);
//        if (flag1) printf("----> %u inconsistent\n", triNeighbors.Peek()[3*i+1]);
//        if (flag2) printf("----> %u inconsistent\n", triNeighbors.Peek()[3*i+2]);
//
//    }
//    triangleBuffer.Release();
//    triNeighbors.Release();
//    // END DEBUG
//
////    // DEBUG print new triangle index buffer
//////    HostArr<uint> triangleBuffer;
////    triangleBuffer.Validate(3*this->triangleCnt);
////    cudaMemcpy(triangleBuffer.Peek(), vboTriIdxPt,
////            triangleBuffer.GetCount()*sizeof(uint), cudaMemcpyDeviceToHost);
////    for (int i = 0; i < this->triangleCnt; ++i) {
////    if ((i > 8200)&&(i < 8300)) {
////        printf("New Triangle Buffer %i: %u %u %u (vertex count %u)\n", i,
////                triangleBuffer.Peek()[3*i+0],
////                triangleBuffer.Peek()[3*i+1],
////                triangleBuffer.Peek()[3*i+2],
////                this->vertexCnt);
////       }
////    }
////    triangleBuffer.Release();
////    // END DEBUG


#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for updating VBOs:                           %.10f sec\n",
            dt_ms/1000.0f);
    cudaEventRecord(event1, 0);

    cudaEventRecord(eventEnd, 0);
    cudaEventSynchronize(eventStart);
    cudaEventSynchronize(eventEnd);
    cudaEventElapsedTime(&dt_ms, eventStart, eventEnd);
    printf("==> Total CUDA time for mesh refinement:               %.10f sec\n",
            dt_ms/1000.0f);
#endif

    // Cleanup

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return -1;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return -1;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return -1;
    }

    return newTrianglesCnt;

#undef USE_TIMER
}


/*
 * DeformableGPUSurfaceMT::Release
 */
void DeformableGPUSurfaceMT::Release() {
    GPUSurfaceMT::Release();
    CudaSafeCall(this->vertexExternalForcesScl_D.Release());
    CudaSafeCall(this->gvfTmp_D.Release());
    CudaSafeCall(this->gvfConstData_D.Release());
    CudaSafeCall(this->laplacian_D.Release());
    CudaSafeCall(this->laplacian2_D.Release());
    CudaSafeCall(this->displLen_D.Release());
    CudaSafeCall(this->distField_D.Release());
    CudaSafeCall(this->externalForces_D.Release());
    CudaSafeCall(this->accTriangleData_D.Release());
    CudaSafeCall(this->accTriangleArea_D.Release());
    CudaSafeCall(this->corruptTriangles_D.Release());

    CudaSafeCall(this->intUncertaintyCorrupt_D.Release());
    CudaSafeCall(this->accumPath_D.Release());
    CudaSafeCall(triangleEdgeOffs_D.Release());
    CudaSafeCall(triangleEdgeList_D.Release());
    CudaSafeCall(subDivEdgeFlag_D.Release());
    CudaSafeCall(subDivEdgeIdxOffs_D.Release());
    CudaSafeCall(newVertices_D.Release());
    CudaSafeCall(newTriangles_D.Release());
    CudaSafeCall(oldTriangles_D.Release());
    CudaSafeCall(trackedSubdivVertexData_D.Release());
    CudaSafeCall(subDivCnt_D.Release());
    CudaSafeCall(newTrianglesIdxOffs_D.Release());
    CudaSafeCall(oldTrianglesIdxOffs_D.Release());
    CudaSafeCall(newTriangleNeighbors_D.Release());
    CudaSafeCall(subDivLevels_D.Release());
    CudaSafeCall(oldSubDivLevels_D.Release());
    CudaSafeCall(vertexFlag_D.Release());
    CudaSafeCall(vertexFlagTmp_D.Release());
    CudaSafeCall(vertexUncertaintyTmp_D.Release());

    CudaSafeCall(triangleFaceNormals_D.Release());
    CudaSafeCall(triangleIdxTmp_D.Release());
    CudaSafeCall(outputArrayTmp_D.Release());
    CudaSafeCall(reducedVertexKeysTmp_D.Release());
    CudaSafeCall(reducedNormalsTmp_D.Release());
    CudaSafeCall(vertexNormalsIndxOffs_D.Release());
    CudaSafeCall(this->geometricLaplacian_D.Release());



    if (this->vboCorruptTriangleVertexFlag) {
        glBindBufferARB(GL_ARRAY_BUFFER, this->vboCorruptTriangleVertexFlag);
        glDeleteBuffersARB(1, &this->vboCorruptTriangleVertexFlag);
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
        this->vboCorruptTriangleVertexFlag = 0;
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
        CheckForGLError();
    }

    if (this->vboVtxPath) {
        glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxPath);
        glDeleteBuffersARB(1, &this->vboVtxPath);
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
        this->vboVtxPath = 0;
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
        CheckForGLError();
    }

    if (this->vboVtxAttr) {
        glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxAttr);
        glDeleteBuffersARB(1, &this->vboVtxAttr);
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
        this->vboVtxAttr = 0;
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
        CheckForGLError();
    }

    ::CheckForGLError();
}


/*
 * DeformableGPUSurfaceMT::updateVtxPos
 */
bool DeformableGPUSurfaceMT::updateVtxPos(
        float* volTarget_D,
        float* vertexBuffer_D,
        float* vtxUncertainty_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue,
        bool useCubicInterpolation,
        size_t maxIt,
        float surfMappedMinDisplScl,
        float springStiffness,
        float forceScl,
        float externalForcesWeight,
        bool trackPath,
        bool externalForcesOnly,
        bool useThinPlate) {

    using namespace vislib::sys;

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    if (!CudaSafeCall(this->laplacian_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->laplacian_D.Set(0))) {
        return false;
    }

    if (!CudaSafeCall(this->laplacian2_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->laplacian2_D.Set(0))) {
        return false;
    }

    if (!CudaSafeCall(this->accumPath_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->accumPath_D.Set(0x00))) {
        return false;
    }

    // Init uncertainty buffer with zero
    if (trackPath) {
        if (!CudaSafeCall(cudaMemset(vtxUncertainty_D, 0x00, this->vertexCnt*sizeof(float)))) {
            return false;
        }
    }

//#define USE_TIMER
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

#ifdef USE_TIMER
    cudaEvent_t eventStart, eventEnd;
    cudaEventCreate(&eventStart);
    cudaEventCreate(&eventEnd);
#endif

    int iterationsNeeded = maxIt;

    if (!externalForcesOnly) {

        // TODO Timer
        for (uint i = 0; i < maxIt; ++i) {

            // Calc laplacian
            DeformableGPUSurfaceMT_MeshLaplacian_D <<< Grid(this->vertexCnt, 256), 256 >>> (
                    vertexBuffer_D,
                    this->vertexDataOffsPos,
                    this->vertexDataStride,
                    this->vertexNeighbours_D.Peek(),
                    18,
                    this->vertexCnt,
                    (float*)this->laplacian_D.Peek(),
                    0,
                    3);

            ::CheckForCudaErrorSync();

            if (useThinPlate) {

                // Calc laplacian^2
                DeformableGPUSurfaceMT_MeshLaplacian_D <<< Grid(this->vertexCnt, 256), 256 >>> (
                        (float*)this->laplacian_D.Peek(),
                        0,
                        3,
                        this->vertexNeighbours_D.Peek(),
                        18,
                        this->vertexCnt,
                        (float*)this->laplacian2_D.Peek(),
                        0,
                        3);

                ::CheckForCudaErrorSync();

                // Update vertex position
                DeformableGPUSurfaceMT_UpdateVtxPos_D <<< Grid(this->vertexCnt, 256), 256 >>> (
                        volTarget_D,
                        vertexBuffer_D,
                        (float*)this->vertexExternalForcesScl_D.Peek(),
                        this->displLen_D.Peek(),
                        vtxUncertainty_D,
                        (float4*)this->externalForces_D.Peek(),
                        this->laplacian_D.Peek(),
                        this->laplacian2_D.Peek(),
                        this->vertexCnt,
                        externalForcesWeight,
                        forceScl,
                        springStiffness,
                        isovalue,
                        surfMappedMinDisplScl,
                        useCubicInterpolation,
                        trackPath, // Track path of vertices
                        this->vertexDataOffsPos,
                        this->vertexDataOffsNormal,
                        this->vertexDataStride);
            } else { // No thin plate aspect
                // Update vertex position
                DeformableGPUSurfaceMT_UpdateVtxPosNoThinPlate_D <<< Grid(this->vertexCnt, 256), 256 >>> (
                        volTarget_D,
                        vertexBuffer_D,
                        (float*)this->vertexExternalForcesScl_D.Peek(),
                        this->displLen_D.Peek(),
                        vtxUncertainty_D,
                        (float4*)this->externalForces_D.Peek(),
                        this->laplacian_D.Peek(),
                        this->vertexCnt,
                        externalForcesWeight,
                        forceScl,
                        isovalue,
                        surfMappedMinDisplScl,
                        useCubicInterpolation,
                        trackPath, // Track path of vertices
                        this->vertexDataOffsPos,
                        this->vertexDataOffsNormal,
                        this->vertexDataStride);
            }

            // Accumulate displacement length of this iteration step
            float avgDisplLen = 0.0f;
            avgDisplLen = thrust::reduce(
                    thrust::device_ptr<float>(this->displLen_D.Peek()),
                    thrust::device_ptr<float>(this->displLen_D.Peek() + this->vertexCnt));
            if (!CudaSafeCall(cudaGetLastError())) {
                return false;
            }
            avgDisplLen /= static_cast<float>(this->vertexCnt);
//            if (i%5 == 0) printf("It: %i, avgDispl: %.16f, min %.16f\n", i, avgDisplLen, surfMappedMinDisplScl);
//            printf("It Reg: %i, avgDispl: %.16f, min %.16f\n", i, avgDisplLen, surfMappedMinDisplScl);
            if (avgDisplLen < surfMappedMinDisplScl) {
                iterationsNeeded =i+1;
                break;
            }

            ::CheckForCudaErrorSync();
        }
    } else {

        for (uint i = 0; i < maxIt; ++i) {
//            this->PrintVertexBuffer(1);

//            // DEBUG print parameters
//            printf("PARAMS:\n");
//            printf("vertex count %u\n", this->vertexCnt);
//            printf("forcesScl %f\n", forceScl);
//            printf("isovalue %f\n", isovalue);
//            printf("surfMappedMinDisplScl %f\n", surfMappedMinDisplScl);
//            if (useCubicInterpolation) printf("useCubicInterpolation TRUE\n");
//            else printf("useCubicInterpolation FALSE\n");
//            if (trackPath) printf("trackPath TRUE\n");
//                        else printf("trackPath FALSE\n");
//            // END DEBUG

//            // DEBUG Print voltarget_D
//            if (i == 0) {
//                HostArr<float> volTarget;
//                size_t gridSize = volDim.x*volDim.y*volDim.z;
//                volTarget.Validate(gridSize);
//                CudaSafeCall(cudaMemcpy(volTarget.Peek(), volTarget_D,
//                        sizeof(float)*gridSize,
//                        cudaMemcpyDeviceToHost));
//
//                for (int i = 0; i < gridSize; ++i) {
//                    printf("VOL %.16f\n", volTarget.Peek()[i]);
//                }
//
//                volTarget.Release();
//            }
//            // END DEBUG

//            cudaEventRecord(eventStart, 0);

            // Update vertex position
            DeformableGPUSurfaceMT_UpdateVtxPosExternalOnly_D <<< Grid(this->vertexCnt, 256), 256 >>> (
                    volTarget_D,
                    vertexBuffer_D,
                    (float*)this->vertexExternalForcesScl_D.Peek(),
                    this->displLen_D.Peek(),
                    vtxUncertainty_D,
                    (float4*)this->externalForces_D.Peek(),
                    this->accumPath_D.Peek(),
                    this->vertexCnt,
                    forceScl,
                    isovalue,
                    surfMappedMinDisplScl,
                    useCubicInterpolation,
                    trackPath, // Track path of vertices
                    this->vertexDataOffsPos,
                    this->vertexDataOffsNormal,
                    this->vertexDataStride);

//            cudaEventRecord(eventEnd, 0);
//            cudaEventSynchronize(eventEnd);
//            cudaEventSynchronize(eventStart);
//            cudaEventElapsedTime(&dt_ms, eventStart, eventEnd);
////            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
////                    "%s: Time for iteration (%u vertices): %f sec\n",
////                    "DeformableGPUSurfaceMT",
////                    this->vertexCnt,
////                    dt_ms/1000.0f);
//            cudaEventRecord(eventStart, 0);

            // Accumulate displacement length of this iteration step
            float avgDisplLen = 0.0f;
            avgDisplLen = thrust::reduce(
                    thrust::device_ptr<float>(this->displLen_D.Peek()),
                    thrust::device_ptr<float>(this->displLen_D.Peek() + this->vertexCnt));
            if (!CudaSafeCall(cudaGetLastError())) {
                return false;
            }
//            cudaEventRecord(eventEnd, 0);
//            cudaEventSynchronize(eventEnd);
//            cudaEventSynchronize(eventStart);
//            cudaEventElapsedTime(&dt_ms, eventStart, eventEnd);
//            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
//                    "%s: Time for thrust::reduce (%u vertices): %f sec\n",
//                    "DeformableGPUSurfaceMT",
//                    this->vertexCnt,
//                    dt_ms/1000.0f);
            avgDisplLen /= static_cast<float>(this->vertexCnt);
//            if (i%5 == 0) printf("It %i, avgDispl: %.16f, min %.16f\n", i, avgDisplLen, surfMappedMinDisplScl);
//            printf("It: %i, avgDispl: %.16f, min %.16f\n", i, avgDisplLen, surfMappedMinDisplScl);
            if (avgDisplLen < surfMappedMinDisplScl) {
                iterationsNeeded =i+1;
                break;
            }

            ::CheckForCudaErrorSync();
        }
    }

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "%s: Time for mapping (%u iterations, %u vertices): %f sec\n",
            "DeformableGPUSurfaceMT",
            iterationsNeeded, this->vertexCnt, dt_ms/1000.0f);
    //printf("Mapping : %.10f\n",
    //        dt_ms/1000.0f);
#endif
#undef USE_TIMER
    return CudaSafeCall(cudaGetLastError());
}


/*
 * DeformableGPUSurfaceMT::updateVtxPos
 */
bool DeformableGPUSurfaceMT::updateVtxPosSubdiv(
        float* volTarget_D,
        float* vertexBuffer_D,
        float* vtxUncertainty_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue,
        bool useCubicInterpolation,
        size_t maxIt,
        float surfMappedMinDisplScl,
        float springStiffness,
        float forceScl,
        float externalForcesWeight,
        bool trackPath,
        bool externalForcesOnly,
        bool useThinPlate) {

    using namespace vislib::sys;

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    if (!CudaSafeCall(this->laplacian_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->laplacian_D.Set(0))) {
        return false;
    }

    if (!CudaSafeCall(this->laplacian2_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->laplacian2_D.Set(0))) {
        return false;
    }

    if (!CudaSafeCall(this->accumPath_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->accumPath_D.Set(0x00))) {
        return false;
    }

    // Init uncertainty buffer with zero
    if (trackPath) {
        if (!CudaSafeCall(cudaMemset(vtxUncertainty_D, 0x00, this->vertexCnt*sizeof(float)))) {
            return false;
        }
    }

//#ifdef USE_TIMER
    //float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
//#endif


    int iterationsNeeded = maxIt;

    if (!externalForcesOnly) {

        // TODO Timer
        for (uint i = 0; i < maxIt; ++i) {

            // Calc laplacian
            DeformableGPUSurfaceMT_MeshLaplacian_D <<< Grid(this->vertexCnt, 256), 256 >>> (
                    vertexBuffer_D,
                    this->vertexDataOffsPos,
                    this->vertexDataStride,
                    this->vertexNeighbours_D.Peek(),
                    18,
                    this->vertexCnt,
                    (float*)this->laplacian_D.Peek(),
                    0,
                    3);

            ::CheckForCudaErrorSync();

            if (useThinPlate) {

                // Calc laplacian^2
                DeformableGPUSurfaceMT_MeshLaplacian_D <<< Grid(this->vertexCnt, 256), 256 >>> (
                        (float*)this->laplacian_D.Peek(),
                        0,
                        3,
                        this->vertexNeighbours_D.Peek(),
                        18,
                        this->vertexCnt,
                        (float*)this->laplacian2_D.Peek(),
                        0,
                        3);

                ::CheckForCudaErrorSync();

                // Update vertex position
                DeformableGPUSurfaceMT_UpdateVtxPos_D <<< Grid(this->vertexCnt, 256), 256 >>> (
                        volTarget_D,
                        vertexBuffer_D,
                        (float*)this->vertexExternalForcesScl_D.Peek(),
                        this->displLen_D.Peek(),
                        vtxUncertainty_D,
                        (float4*)this->externalForces_D.Peek(),
                        this->laplacian_D.Peek(),
                        this->laplacian2_D.Peek(),
                        this->vertexCnt,
                        externalForcesWeight,
                        forceScl,
                        springStiffness,
                        isovalue,
                        surfMappedMinDisplScl,
                        useCubicInterpolation,
                        trackPath, // Track path of vertices
                        this->vertexDataOffsPos,
                        this->vertexDataOffsNormal,
                        this->vertexDataStride);
            } else { // No thin plate aspect
                // Update vertex position
                DeformableGPUSurfaceMT_UpdateVtxPosNoThinPlate_D <<< Grid(this->vertexCnt, 256), 256 >>> (
                        volTarget_D,
                        vertexBuffer_D,
                        (float*)this->vertexExternalForcesScl_D.Peek(),
                        this->displLen_D.Peek(),
                        vtxUncertainty_D,
                        (float4*)this->externalForces_D.Peek(),
                        this->laplacian_D.Peek(),
                        this->vertexCnt,
                        externalForcesWeight,
                        forceScl,
                        isovalue,
                        surfMappedMinDisplScl,
                        useCubicInterpolation,
                        trackPath, // Track path of vertices
                        this->vertexDataOffsPos,
                        this->vertexDataOffsNormal,
                        this->vertexDataStride);
            }

            // Accumulate displacement length of this iteration step
            float avgDisplLen = 0.0f;
            avgDisplLen = thrust::reduce(
                    thrust::device_ptr<float>(this->displLen_D.Peek()),
                    thrust::device_ptr<float>(this->displLen_D.Peek() + this->vertexCnt));
            if (!CudaSafeCall(cudaGetLastError())) {
                return false;
            }
            avgDisplLen /= static_cast<float>(this->vertexCnt);
//            if (i%5 == 0) printf("It: %i, avgDispl: %.16f, min %.16f\n", i, avgDisplLen, surfMappedMinDisplScl);
//            printf("It: %i, avgDispl: %.16f, min %.16f\n", i, avgDisplLen, surfMappedMinDisplScl);
            if (avgDisplLen < surfMappedMinDisplScl) {
                iterationsNeeded =i+1;
                break;
            }

            ::CheckForCudaErrorSync();
        }
    } else {
        // TODO Timer
        for (uint i = 0; i < maxIt; ++i) {

            // Update vertex position
            DeformableGPUSurfaceMT_UpdateVtxPosExternalOnlySubdiv_D <<< Grid(this->vertexCnt, 256), 256 >>> (
                    volTarget_D,
                    vertexBuffer_D,
                    (float*)this->vertexExternalForcesScl_D.Peek(),
                    this->displLen_D.Peek(),
                    vtxUncertainty_D,
                    (float4*)this->externalForces_D.Peek(),
                    this->accumPath_D.Peek(),
                    this->vertexFlag_D.Peek(),
                    this->vertexCnt,
                    forceScl,
                    isovalue,
                    surfMappedMinDisplScl,
                    useCubicInterpolation,
                    trackPath, // Track path of vertices
                    this->vertexDataOffsPos,
                    this->vertexDataOffsNormal,
                    this->vertexDataStride);

            // Accumulate displacement length of this iteration step
            float avgDisplLen = 0.0f;
            avgDisplLen = thrust::reduce(
                    thrust::device_ptr<float>(this->displLen_D.Peek()),
                    thrust::device_ptr<float>(this->displLen_D.Peek() + this->vertexCnt));
            if (!CudaSafeCall(cudaGetLastError())) {
                return false;
            }
            avgDisplLen /= static_cast<float>(this->nFlaggedVertices);
//            printf("New vertex count %u\n", this->nFlaggedVertices);
//            if (i%5 == 0) printf("It %i, avgDispl: %.16f, min %.16f\n", i, avgDisplLen, surfMappedMinDisplScl);
//            printf("It: %i, avgDispl: %.16f, min %.1f\n", i, avgDisplLen, surfMappedMinDisplScl);
            if (avgDisplLen < surfMappedMinDisplScl) {
                iterationsNeeded =i+1;
                break;
            }

            ::CheckForCudaErrorSync();
        }
    }

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "%s: Time for mapping (%u iterations, %u vertices): %f sec\n",
            "DeformableGPUSurfaceMT",
            iterationsNeeded, this->vertexCnt, dt_ms/1000.0f);
    //printf("Mapping : %.10f\n",
    //        dt_ms/1000.0f);
#endif

    return CudaSafeCall(cudaGetLastError());
}


/*
 * ComputeVtxDiffValue0_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeVtxDiffValue0_D(
        float *diff_D,
        float *tex0_D,
        float *vtxData0_D,
        size_t vertexCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    const int vertexDataStride = 9; // TODO
    const int vertexDataOffsPos = 0;

    float3 pos;
    pos.x = vtxData0_D[vertexDataStride*idx + vertexDataOffsPos +0];
    pos.y = vtxData0_D[vertexDataStride*idx + vertexDataOffsPos +1];
    pos.z = vtxData0_D[vertexDataStride*idx + vertexDataOffsPos +2];

    diff_D[idx] = ::SampleFieldAtPosTrilin_D<float, true>(pos, tex0_D);
}


/*
 * ComputeVtxDiffValue1_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeVtxDiffValue1_D(
        float *diff_D,
        float *tex1_D,
        float *vtxData1_D,
        size_t vertexCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    const int vertexDataStride = 9; // TODO
    const int vertexDataOffsPos = 0;

    float valFirst = diff_D[idx];
    float3 pos;
    pos.x = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +0];
    pos.y = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +1];
    pos.z = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +2];

    float valSec = ::SampleFieldAtPosTrilin_D<float, true>(pos, tex1_D);
    valSec = abs(valSec-valFirst);
    diff_D[idx] = valSec;
}


/*
 * DeformableGPUSurfaceMT_ComputeVtxDiffValue1Fitted_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeVtxDiffValue1Fitted_D(
        float *diff_D,
        float *tex1_D,
        float *vtxData1_D,
        float *rotation_D,
        float3 translation,
        float3 centroid,
        size_t vertexCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    const int vertexDataStride = 9; // TODO
    const int vertexDataOffsPos = 0;

    //float valFirst = diff_D[idx];
    float3 pos;
    pos.x = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +0];
    pos.y = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +1];
    pos.z = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +2];

    // Revert translation to move to origin
    pos.x -= translation.x;
    pos.y -= translation.y;
    pos.z -= translation.z;

    // Revert rotation
    float3 posRot;
    posRot.x = rotation_D[0] * pos.x +
               rotation_D[3] * pos.y +
               rotation_D[6] * pos.z;
    posRot.y = rotation_D[1] * pos.x +
               rotation_D[4] * pos.y +
               rotation_D[7] * pos.z;
    posRot.z = rotation_D[2] * pos.x +
               rotation_D[5] * pos.y +
               rotation_D[8] * pos.z;

    // Move to old centroid
    posRot.x += centroid.x;
    posRot.y += centroid.y;
    posRot.z += centroid.z;

    float valSec = ::SampleFieldAtPosTrilin_D<float, true>(posRot, tex1_D);
    //valSec = abs(valSec-valFirst);
    diff_D[idx] = valSec;
    printf("%f\n", valSec);
}


/*
 * ComputeVtxSignDiffValue1_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeVtxSignDiffValue1_D(
        float *signdiff_D,
        float *tex1_D,
        float *vtxData1_D,
        size_t vertexCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    const int vertexDataStride = 9; // TODO
    const int vertexDataOffsPos = 0;

    float valFirst = signdiff_D[idx];
    float3 pos;
    pos.x = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +0];
    pos.y = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +1];
    pos.z = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +2];


    float valSec = ::SampleFieldAtPosTrilin_D<float, true>(pos, tex1_D);
    valSec = float(valSec*valFirst < 0); // TODO Use binary operator
    signdiff_D[idx] = valSec;
}


/*
 * ComputeVtxSignDiffValue1_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeVtxSignDiffValue1Fitted_D(
        float *signdiff_D,
        float *tex1_D,
        float *vtxData1_D,
        float *rotation_D,
        float3 translation,
        float3 centroid,
        size_t vertexCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    const int vertexDataStride = 9; // TODO
    const int vertexDataOffsPos = 0;

    float valFirst = signdiff_D[idx];
//    float3 pos;
//    pos.x = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +0];
//    pos.y = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +1];
//    pos.z = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +2];

    float3 pos;
    pos.x = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +0];
    pos.y = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +1];
    pos.z = vtxData1_D[vertexDataStride*idx + vertexDataOffsPos +2];

       // Revert translation to move to origin
       pos.x -= translation.x;
       pos.y -= translation.y;
       pos.z -= translation.z;

       // Revert rotation
       float3 posRot;
       posRot.x = rotation_D[0] * pos.x +
               rotation_D[3] * pos.y +
               rotation_D[6] * pos.z;
       posRot.y = rotation_D[1] * pos.x +
               rotation_D[4] * pos.y +
               rotation_D[7] * pos.z;
       posRot.z = rotation_D[2] * pos.x +
               rotation_D[5] * pos.y +
               rotation_D[8] * pos.z;

       // Move to old centroid
       posRot.x += centroid.x;
       posRot.y += centroid.y;
       posRot.z += centroid.z;

    float valSec = ::SampleFieldAtPosTrilin_D<float, true>(posRot, tex1_D);
    valSec = float(valSec*valFirst < 0); // TODO Use binary operator
    signdiff_D[idx] = valSec;
}


/*
 * DeformableGPUSurfaceMT::ComputeVtxDiffValue
 */
bool DeformableGPUSurfaceMT::ComputeVtxDiffValue(
        float *diff_D,
        float *tex0_D,
        int3 texDim0,
        float3 texOrg0,
        float3 texDelta0,
        float *tex1_D,
        int3 texDim1,
        float3 texOrg1,
        float3 texDelta1,
        GLuint vtxDataVBO0,
        GLuint vtxDataVBO1,
        size_t vertexCnt) {

    using namespace vislib::sys;

    /* Get pointers to vertex data */

    cudaGraphicsResource* cudaTokens[2];

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], vtxDataVBO0,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], vtxDataVBO1,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt0, *vboPt1;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt0), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt1), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }


    // Init CUDA grid for texture #0
    if (!initGridParams(texDim0, texOrg0, texDelta0)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DeformableGPUSurfaceMT::ClassName());
        return false;
    }

    // Call first kernel
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_ComputeVtxDiffValue0_D <<< Grid(vertexCnt, 256), 256 >>> (
            diff_D,
            tex0_D,
            vboPt0,
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVtxDiffValue0_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif

    // Init CUDA grid for texture #1
    if (!initGridParams(texDim1, texOrg1, texDelta1)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DeformableGPUSurfaceMT::ClassName());
        return false;
    }

    // Call second kernel
#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_ComputeVtxDiffValue1_D <<< Grid(vertexCnt, 256), 256 >>> (
            diff_D,
            tex1_D,
            vboPt1,
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVtxDiffValue1_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif


    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

    return true;

}


/*
 * DeformableGPUSurfaceMT::ComputeVtxDiffValueFitted
 */
bool DeformableGPUSurfaceMT::ComputeVtxDiffValueFitted(
        float *diff_D,
        float centroid[3],
        float rotMat[9],
        float transVec[3],
        float *tex0_D,
        int3 texDim0,
        float3 texOrg0,
        float3 texDelta0,
        float *tex1_D,
        int3 texDim1,
        float3 texOrg1,
        float3 texDelta1,
        GLuint vtxDataVBO0,
        GLuint vtxDataVBO1,
        size_t vertexCnt) {

    CudaDevArr<float> rotate_D;
    // Rotate for best fit
    rotate_D.Validate(9);
    if (!CudaSafeCall(cudaMemcpy((void *)rotate_D.Peek(), &rotMat[0],
            9*sizeof(float), cudaMemcpyHostToDevice))) {
        return false;
    }

    using namespace vislib::sys;

    /* Get pointers to vertex data */

    cudaGraphicsResource* cudaTokens[2];

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], vtxDataVBO0,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], vtxDataVBO1,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt0, *vboPt1;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt0), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt1), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }


    // Init CUDA grid for texture #0
    if (!initGridParams(texDim0, texOrg0, texDelta0)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DeformableGPUSurfaceMT::ClassName());
        return false;
    }

    // Call first kernel
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_ComputeVtxDiffValue0_D <<< Grid(vertexCnt, 256), 256 >>> (
            diff_D,
            tex0_D,
            vboPt0,
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVtxDiffValue0_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif

    // Init CUDA grid for texture #1
    if (!initGridParams(texDim1, texOrg1, texDelta1)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DeformableGPUSurfaceMT::ClassName());
        return false;
    }

    // Call second kernel
#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_ComputeVtxDiffValue1Fitted_D <<< Grid(vertexCnt, 256), 256 >>> (
            diff_D,
            tex1_D,
            vboPt1,
            rotate_D.Peek(),
            make_float3(transVec[0],transVec[1],transVec[2]),
            make_float3(centroid[0],centroid[1],centroid[2]),
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVtxDiffValue1_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif


    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

    if (!CudaSafeCall(rotate_D.Release())) {
        return false;
    }

    return true;

}



/*
 * DeformableGPUSurfaceMT::ComputeVtxSignDiffValue
 */
bool DeformableGPUSurfaceMT::ComputeVtxSignDiffValue(
        float *signdiff_D,
        float *tex0_D,
        int3 texDim0,
        float3 texOrg0,
        float3 texDelta0,
        float *tex1_D,
        int3 texDim1,
        float3 texOrg1,
        float3 texDelta1,
        GLuint vtxDataVBO0,
        GLuint vtxDataVBO1,
        size_t vertexCnt) {

    using namespace vislib::sys;

    /* Get pointers to vertex data */

    cudaGraphicsResource* cudaTokens[2];

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], vtxDataVBO0,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], vtxDataVBO1,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt0;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt0), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt1;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt1), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }


    // Init CUDA grid for texture #0
    if (!initGridParams(texDim0, texOrg0, texDelta0)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DeformableGPUSurfaceMT::ClassName());
        return false;
    }

    // Call first kernel
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_ComputeVtxDiffValue0_D <<< Grid(vertexCnt, 256), 256 >>> (
            signdiff_D,
            tex0_D,
            vboPt0,
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVtxSignDiffValue0_D':            %.10f sec\n",
            dt_ms/1000.0f);
#endif

    // Init CUDA grid for texture #1
    if (!initGridParams(texDim1, texOrg1, texDelta1)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DeformableGPUSurfaceMT::ClassName());
        return false;
    }

    // Call second kernel
#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_ComputeVtxSignDiffValue1_D <<< Grid(vertexCnt, 256), 256 >>> (
            signdiff_D,
            tex1_D,
            vboPt1,
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVtxDiffValue1_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif


    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

    return true;

}


/*
 * DeformableGPUSurfaceMT::ComputeVtxSignDiffValueFitted
 */
bool DeformableGPUSurfaceMT::ComputeVtxSignDiffValueFitted(
        float *signdiff_D,
        float centroid[3],
        float rotMat[9],
        float transVec[3],
        float *tex0_D,
        int3 texDim0,
        float3 texOrg0,
        float3 texDelta0,
        float *tex1_D,
        int3 texDim1,
        float3 texOrg1,
        float3 texDelta1,
        GLuint vtxDataVBO0,
        GLuint vtxDataVBO1,
        size_t vertexCnt) {

    CudaDevArr<float> rotate_D;
    // Rotate for best fit
    rotate_D.Validate(9);
    if (!CudaSafeCall(cudaMemcpy((void *)rotate_D.Peek(), &rotMat[0],
            9*sizeof(float), cudaMemcpyHostToDevice))) {
        return false;
    }

    using namespace vislib::sys;

    /* Get pointers to vertex data */

    cudaGraphicsResource* cudaTokens[2];

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], vtxDataVBO0,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], vtxDataVBO1,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt0;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt0), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt1;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt1), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }


    // Init CUDA grid for texture #0
    if (!initGridParams(texDim0, texOrg0, texDelta0)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DeformableGPUSurfaceMT::ClassName());
        return false;
    }

    // Call first kernel
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_ComputeVtxDiffValue0_D <<< Grid(vertexCnt, 256), 256 >>> (
            signdiff_D,
            tex0_D,
            vboPt0,
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVtxSignDiffValue0_D':            %.10f sec\n",
            dt_ms/1000.0f);
#endif

    // Init CUDA grid for texture #1
    if (!initGridParams(texDim1, texOrg1, texDelta1)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DeformableGPUSurfaceMT::ClassName());
        return false;
    }

    // Call second kernel
#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_ComputeVtxSignDiffValue1Fitted_D <<< Grid(vertexCnt, 256), 256 >>> (
            signdiff_D,
            tex1_D,
            vboPt1,
            rotate_D.Peek(),
            make_float3(transVec[0],transVec[1],transVec[2]),
            make_float3(centroid[0],centroid[1],centroid[2]),
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVtxDiffValue1_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif


    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

    if (!CudaSafeCall(rotate_D.Release())) {
        return false;
    }

    return true;

}


/*
 * DeformableGPUSurfaceMT_CalcHausdorffDistance_D
 */
__global__ void DeformableGPUSurfaceMT_CalcHausdorffDistance_D(
        float *vtxData1,
        float *vtxData2,
        float *hausdorffdistVtx_D,
        uint vertexCnt1,
        uint vertexCnt2) {

    const uint posOffs = 0; // TODO Define const device vars
    const uint stride = 9;

    const uint idx = ::getThreadIdx();

    if (idx >= vertexCnt1) {
        return;
    }

    float3 pos1 = make_float3(
            vtxData1[stride*idx+posOffs+0],
            vtxData1[stride*idx+posOffs+1],
            vtxData1[stride*idx+posOffs+2]);

    float3 pos2;
    float distSqr;
    float minDistSqr = 10000000.0;
    for (int i = 0; i < vertexCnt2; ++i) {
        pos2 = make_float3(
                vtxData2[stride*i+posOffs+0],
                vtxData2[stride*i+posOffs+1],
                vtxData2[stride*i+posOffs+2]);
        distSqr = (pos2.x-pos1.x)*(pos2.x-pos1.x) +
                (pos2.y-pos1.y)*(pos2.y-pos1.y) +
                (pos2.z-pos1.z)*(pos2.z-pos1.z);

        minDistSqr = min(minDistSqr,distSqr);
    }

    hausdorffdistVtx_D[idx] = minDistSqr;
}


/*
 * DeformableGPUSurfaceMT::CalcHausdorffDistance
 */
float DeformableGPUSurfaceMT::CalcHausdorffDistance(
        DeformableGPUSurfaceMT *surf1,
        DeformableGPUSurfaceMT *surf2,
        float *hausdorffdistVtx_D,
        bool symmetric) {

    // TODO Implement symmetric version


    /* Get pointers to vertex data */

    cudaGraphicsResource* cudaTokens[2];

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0], surf1->GetVtxDataVBO(),
            cudaGraphicsMapFlagsNone))) {
        return 0.0f;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1], surf2->GetVtxDataVBO(),
            cudaGraphicsMapFlagsNone))) {
        return 0.0f;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return 0.0f;
    }

    // Get mapped pointers to the vertex data buffers
    float *vboPt0, *vboPt1;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt0), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return 0.0f;
    }

    // Get mapped pointers to the vertex data buffers
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt1), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return 0.0f;
    }


    // Calc kernel
    // TODO Implement less lazy and faster version of Hausdorff distance
    DeformableGPUSurfaceMT_CalcHausdorffDistance_D <<< Grid(surf1->GetVertexCnt(), 256), 256 >>> (
            vboPt0,
            vboPt1,
            hausdorffdistVtx_D,
            surf1->GetVertexCnt(),
            surf2->GetVertexCnt());


    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return 0.0f;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return 0.0f;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return 0.0f;
    }

    float res = 0.0;

    res = thrust::reduce(
            thrust::device_ptr<float>(hausdorffdistVtx_D),
            thrust::device_ptr<float>(hausdorffdistVtx_D + surf1->GetVertexCnt()),
            -1.0,
            thrust::maximum<float>());

    return sqrt(res);
}


__global__ void TrackPathSubdivVertices_D(
        float *sourceVolume_D,
        float *vertexData_D,
        float *vertexFlag_D,
        float *vertexExternalForcesScl_D,
        float *displLen_D,
        float *vtxUncertainty_D,
        float4 *gradient_D,
        int *accumPath_D,
        uint vertexCnt,
        float forcesScl,
        float isoval,
        float minDispl) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }
    if (vertexFlag_D[idx] == 0.0) {
        vertexData_D[9*idx+3] = 0.0;
        vertexData_D[9*idx+4] = 1.0;
        vertexData_D[9*idx+5] = 0.0;
        displLen_D[idx] = 0.0; // Old vertices are per definition converged
        return; // This is an old vertex
    }

    // Check convergence criterion
    float lastDisplLen = displLen_D[idx];
    if (lastDisplLen <= minDispl) {
        displLen_D[idx] = 0.0;
        return; // Vertex is converged
    }


    /* Retrieve stuff from global device memory */

    // Get initial position from global device memory
    float3 posOld = make_float3(
            vertexData_D[9*idx+0],
            vertexData_D[9*idx+1],
            vertexData_D[9*idx+2]);

    // Get initial scale factor for external forces
    float externalForcesScl = vertexExternalForcesScl_D[idx];
    //float externalForcesSclOld = externalForcesScl;


    /* Update position */

    // No warp divergence here, since useCubicInterpolation is the same for all
    // threads
    //const float sampleDens = SampleFieldAtPosTrilin_D<float>(posOld, sourceVolume_D);
    const float sampleDens = SampleFieldAtPosTricub_D<float, false>(posOld, sourceVolume_D);

    // Switch sign and scale down if necessary
    bool negative = externalForcesScl < 0;
    bool outside = sampleDens <= isoval;
    int switchSign = int((negative && outside)||(!negative && !outside));
    externalForcesScl = externalForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
    externalForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));

    if (bool(switchSign) && (accumPath_D[idx] != 0)) {
        accumPath_D[idx] = 0;
    } else if (bool(switchSign) && (accumPath_D[idx] == 0)) {
        accumPath_D[idx] = 1;
    }

    // Sample gradient
    //float4 externalForceTmp = SampleFieldAtPosTrilin_D<float4>(posOld, gradient_D);
    float4 externalForceTmp = SampleFieldAtPosTricub_D<float4, false>(posOld, gradient_D);

    float3 externalForce;
    externalForce.x = externalForceTmp.x;
    externalForce.y = externalForceTmp.y;
    externalForce.z = externalForceTmp.z;

    externalForce = safeNormalize(externalForce);
    externalForce *= forcesScl*externalForcesScl;

    float3 posNew = posOld + externalForce; // Integrate backwards

    /* Write back to global device memory */

    // New pos
    vertexData_D[9*idx+0] = posNew.x;
    vertexData_D[9*idx+1] = posNew.y;
    vertexData_D[9*idx+2] = posNew.z;

    // Write external forces scale factor back to global device memory
    vertexExternalForcesScl_D[idx] = externalForcesScl;

//    float3 diff = posNew-posOld;
//    float diffLen = length(diff);
    float diffLen = abs(forcesScl*externalForcesScl);
//    if ((abs(externalForcesScl) == 1.0f)) {
//        vtxUncertainty_D[idx] += diffLen;
//    }

    if (accumPath_D[idx] == 0) {
        vtxUncertainty_D[idx] += diffLen;
    } else if(accumPath_D[idx] != 0) {
        vtxUncertainty_D[idx] -= diffLen;
    }

    // Displ scl for convergence
    displLen_D[idx] = diffLen;
    //displLen_D[idx] = 0.1;
    vertexData_D[9*idx+3] = 1.0;
    vertexData_D[9*idx+4] = 0.0;
    vertexData_D[9*idx+5] = 1.0;
}


/*
 * DeformableGPUSurfaceMT::ComputeUncertaintyForSubdivVertices
 */
bool DeformableGPUSurfaceMT::TrackPathSubdivVertices(
        float *sourceVolume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float forcesScl,
        float minDispl,
        float isoval,
        uint maxIt) {

    using namespace vislib::sys;

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    /* 1. Reinitialize VBO and copy back uncertainty values */

    cudaGraphicsResource* cudaTokens[1];
    cudaGraphicsResource* cudaTokens2[2];

    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboVtxPath,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsMapResources(1, cudaTokens, 0))) {
        return false;
    }
    // Get mapped pointers to the vertex data buffers
    float *uncertaintyPt;
    size_t vboVtxPathSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&uncertaintyPt), // The mapped pointer
            &vboVtxPathSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Copy old values to temporary array
    if (!CudaSafeCall(this->vertexUncertaintyTmp_D.Validate(vboVtxPathSize/sizeof(float)))) {
        return false;
    }
    if (!CudaSafeCall(cudaMemcpy(
            this->vertexUncertaintyTmp_D.Peek(),
            uncertaintyPt,
            vboVtxPathSize,
            cudaMemcpyDeviceToDevice))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }

    // Re-initiaize VBO
    if (!this->InitVtxPathVBO(this->vertexCnt)) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens2[0],
            this->vboVtxPath,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens2[1],
            this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens2, 0))) {
        return false;
    }

    float *vboVertexPt;
    size_t vboVertexSize;
    // Get mapped pointers to the vertex data buffers
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&uncertaintyPt), // The mapped pointer
            &vboVtxPathSize,                      // The size of the accessible data
            cudaTokens2[0]))) {                        // The mapped resource
        return false;
    }
    // Get mapped pointers to the vertex data buffers
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboVertexPt), // The mapped pointer
            &vboVertexSize,                      // The size of the accessible data
            cudaTokens2[1]))) {                        // The mapped resource
        return false;
    }

    if (!CudaSafeCall(cudaMemset(uncertaintyPt, 0x00, vboVtxPathSize))) {
        return false;
    }
    if (!CudaSafeCall(cudaMemcpy(
            uncertaintyPt,
            this->vertexUncertaintyTmp_D.Peek(),
            sizeof(float)*this->vertexUncertaintyTmp_D.GetCount(),
            cudaMemcpyDeviceToDevice))) {
        return false;
    }

    /* 2. Write uncertainty values of new vertices */

    // Get copy of vertex buffer
    if (!CudaSafeCall(this->trackedSubdivVertexData_D.Validate(this->vertexCnt*this->vertexDataStride))) {
        return false;
    }
    if (!CudaSafeCall(cudaMemcpy(this->trackedSubdivVertexData_D.Peek(),
            vboVertexPt,
            vboVertexSize,
            cudaMemcpyDeviceToDevice))) {
        return false;
    }
    // Check/prepare necessary arrays
    if (sourceVolume_D == NULL) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Set(0x00))) {
        return false;
    }

    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt))) {
        return false;
    }
    DeformableGPUSurfaceMT_InitExternalForceScl_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            (float*)this->vertexExternalForcesScl_D.Peek(),
            this->displLen_D.Peek(),
            sourceVolume_D,
            this->trackedSubdivVertexData_D.Peek(),
            minDispl,
            this->vertexCnt,
            isoval,
            this->vertexDataOffsPos,
            this->vertexDataStride);
    if (!CheckForCudaError()) {
        return false;
    }

    if (this->vertexFlag_D.GetCount() != this->vertexCnt) {
        if (!CudaSafeCall(this->vertexFlag_D.Validate(this->vertexCnt))) {
            return -1;
        }
        if (!CudaSafeCall(this->vertexFlag_D.Set(0x00))) {
            return -1;
        }
    }

    if (!CudaSafeCall(this->accumPath_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->accumPath_D.Set(0x00))) {
        return false;
    }
    uint iterationsNeeded = 0;
    for (uint i = 0; i < maxIt; ++i) {

        // Update vertex position
        TrackPathSubdivVertices_D <<< Grid(this->vertexCnt, 256), 256 >>> (
                sourceVolume_D,
                this->trackedSubdivVertexData_D.Peek(),
                this->vertexFlag_D.Peek(),
                this->vertexExternalForcesScl_D.Peek(),
                this->displLen_D.Peek(),
                uncertaintyPt,
                (float4*)(this->externalForces_D.Peek()),
                this->accumPath_D.Peek(),
                this->vertexCnt,
                forcesScl,
                isoval,
                minDispl);
        if (!CheckForCudaError()) {
            return false;
        }

        // Accumulate displacement length of this iteration step
        float avgDisplLen = 0.0f;
        avgDisplLen = thrust::reduce(
                thrust::device_ptr<float>(this->displLen_D.Peek()),
                thrust::device_ptr<float>(this->displLen_D.Peek() + this->vertexCnt));
        if (!CudaSafeCall(cudaGetLastError())) {
            return false;
        }
//        printf("Number of flagged vertices %u, %f\n", this->nFlaggedVertices, avgDisplLen);
        avgDisplLen /= static_cast<float>(this->nFlaggedVertices);
//        if (i%10 == 0) printf("It %i, avgDispl: %.16f, min %.16f\n", i, avgDisplLen, minDispl);
//        printf("It: %i, avgDispl: %.16f, min %.16f\n", i, avgDisplLen, minDispl);
        if (avgDisplLen < minDispl) {
            iterationsNeeded = i+1;
            break;
        }

        ::CheckForCudaErrorSync();
    }

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens2, 0))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens2[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens2[1]))) {
        return false;
    }

    return CheckForCudaError();
}


/*
 * DeformableGPUSurfaceMT_ComputeSurfAttribDiff0_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeSurfAttribSignDiff0_D (
        float *vertexAttrib_D,
        float *vertexDataEnd_D,
        float *tex0_D,
        uint vertexCnt) {
    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    const int vertexDataStride = 9; // TODO
    const int vertexDataOffsPos = 0;

    float3 pos;
    pos.x = vertexDataEnd_D[vertexDataStride*idx + vertexDataOffsPos +0];
    pos.y = vertexDataEnd_D[vertexDataStride*idx + vertexDataOffsPos +1];
    pos.z = vertexDataEnd_D[vertexDataStride*idx + vertexDataOffsPos +2];

    vertexAttrib_D[idx] = ::SampleFieldAtPosTrilin_D<float, true>(pos, tex0_D);
}


/*
 * DeformableGPUSurfaceMT_ComputeSurfAttribDiff1_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeSurfAttribSignDiff1_D (
        float *vertexAttrib_D,
        float *vertexDataStart_D,
        float *vertexDataTrackedBack_D,
        float *vertexFlag_D,
        float *tex1_D,
        float *rotation_D,
        float3 translation,
        float3 centroid,
        uint vertexCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    const int vertexDataStride = 9; // TODO
    const int vertexDataOffsPos = 0;

    float3 pos;
    if (vertexFlag_D[idx] == 1.0) {
        pos.x = vertexDataTrackedBack_D[vertexDataStride*idx + vertexDataOffsPos +0];
        pos.y = vertexDataTrackedBack_D[vertexDataStride*idx + vertexDataOffsPos +1];
        pos.z = vertexDataTrackedBack_D[vertexDataStride*idx + vertexDataOffsPos +2];
    } else {
        pos.x = vertexDataStart_D[vertexDataStride*idx + vertexDataOffsPos +0];
        pos.y = vertexDataStart_D[vertexDataStride*idx + vertexDataOffsPos +1];
        pos.z = vertexDataStart_D[vertexDataStride*idx + vertexDataOffsPos +2];
    }

    // Revert translation to move to origin
    pos.x -= translation.x;
    pos.y -= translation.y;
    pos.z -= translation.z;

    // Revert rotation
    float3 posRot;
    posRot.x = rotation_D[0] * pos.x +
            rotation_D[3] * pos.y +
            rotation_D[6] * pos.z;
    posRot.y = rotation_D[1] * pos.x +
            rotation_D[4] * pos.y +
            rotation_D[7] * pos.z;
    posRot.z = rotation_D[2] * pos.x +
            rotation_D[5] * pos.y +
            rotation_D[8] * pos.z;

    // Move to old centroid
    posRot.x += centroid.x;
    posRot.y += centroid.y;
    posRot.z += centroid.z;

    float attribOld = vertexAttrib_D[idx];
    float attribNew = ::SampleFieldAtPosTrilin_D<float, true>(posRot, tex1_D);
    vertexAttrib_D[idx] = int(attribOld*attribNew < 0); // 1.0 or 0.0
}


/*
 * DeformableGPUSurfaceMT::ComputeSurfAttribSignDiff
 */
bool DeformableGPUSurfaceMT::ComputeSurfAttribSignDiff(
        DeformableGPUSurfaceMT &surfStart,
        float centroid[3], // In case the start surface has been fitted using RMSD
        float rotMat[9],
        float transVec[3],
        float *tex0_D,
        int3 texDim0,
        float3 texOrg0,
        float3 texDelta0,
        float *tex1_D,
        int3 texDim1,
        float3 texOrg1,
        float3 texDelta1) {

    using namespace vislib::sys;


    if (!this->InitVtxAttribVBO(this->vertexCnt)) {
        return false;
    }

    // Get pointer to vertex attribute array
    cudaGraphicsResource* cudaTokens[3];
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboVtxAttr,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1],
            this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[2],
            surfStart.GetVtxDataVBO(),
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsMapResources(3, cudaTokens, 0))) {
        return false;
    }
    // Get mapped pointers to the vertex data buffers
    float *vertexAttrib_D;
    size_t vboVtxAttribSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vertexAttrib_D), // The mapped pointer
            &vboVtxAttribSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }
    // Get mapped pointers to the vertex data buffers
    float *vertexDataEnd_D;
    size_t vboEndSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vertexDataEnd_D), // The mapped pointer
            &vboEndSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }
    // Get mapped pointers to the vertex data buffers
    float *vertexDataStart_D;
    size_t vboStartSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vertexDataStart_D), // The mapped pointer
            &vboStartSize,              // The size of the accessible data
            cudaTokens[2]))) {                 // The mapped resource
        return false;
    }

    // Init grid params
    // Init CUDA grid for texture #0
    if (!initGridParams(texDim0, texOrg0, texDelta0)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DeformableGPUSurfaceMT::ClassName());
        return false;
    }

    // Compute difference for new and old vertices (after subdivision)
    // Part one: sample value for new vertices
    DeformableGPUSurfaceMT_ComputeSurfAttribSignDiff0_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            vertexAttrib_D,
            vertexDataEnd_D,
            tex0_D,
            this->vertexCnt);

    CudaDevArr<float> rotate_D;
    // Rotate for best fit
    rotate_D.Validate(9);
    if (!CudaSafeCall(cudaMemcpy((void *)rotate_D.Peek(), &rotMat[0],
            9*sizeof(float), cudaMemcpyHostToDevice))) {
        return false;
    }

    // Init grid params
    // Init CUDA grid for texture #0
    if (!initGridParams(texDim1, texOrg1, texDelta1)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DeformableGPUSurfaceMT::ClassName());
        return false;
    }

    if (this->vertexFlag_D.GetCount() == 0) {
        if (!CudaSafeCall(this->vertexFlag_D.Validate(this->vertexCnt))) {
            return false;
        }
        if (!CudaSafeCall(this->vertexFlag_D.Set(0x00))) {
            return false;
        }
    }

    // Compute difference for new and old vertices (after subdivision)
    // Part two: sample value for old/tracked back vertices
    DeformableGPUSurfaceMT_ComputeSurfAttribSignDiff1_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            vertexAttrib_D,
            vertexDataStart_D,
            this->trackedSubdivVertexData_D.Peek(), // Tracked back vertices, needed for sampling
            this->vertexFlag_D.Peek(),
            tex1_D,
            rotate_D.Peek(),
            make_float3(transVec[0],transVec[1],transVec[2]),
            make_float3(centroid[0],centroid[1],centroid[2]),
            this->vertexCnt);

    if (!CheckForCudaError()) {
        return false;
    }

    rotate_D.Release();

    if (!CudaSafeCall(cudaGraphicsUnmapResources(3, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[2]))) {
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT_ComputeSurfAttribDiff0_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeSurfAttribDiff0_D (
        float *vertexAttrib_D,
        float *vertexDataEnd_D,
        float *tex0_D,
        uint vertexCnt) {
    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    const int vertexDataStride = 9; // TODO
    const int vertexDataOffsPos = 0;

    float3 pos;
    pos.x = vertexDataEnd_D[vertexDataStride*idx + vertexDataOffsPos +0];
    pos.y = vertexDataEnd_D[vertexDataStride*idx + vertexDataOffsPos +1];
    pos.z = vertexDataEnd_D[vertexDataStride*idx + vertexDataOffsPos +2];

    vertexAttrib_D[idx] = ::SampleFieldAtPosTrilin_D<float, true>(pos, tex0_D);
}


/*
 * DeformableGPUSurfaceMT_ComputeSurfAttribDiff1_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeSurfAttribDiff1_D (
        float *vertexAttrib_D,
        float *vertexDataStart_D,
        float *vertexDataTrackedBack_D,
        float *vertexFlag_D,
        float *tex1_D,
        float *rotation_D,
        float3 translation,
        float3 centroid,
        uint vertexCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    const int vertexDataStride = 9; // TODO
    const int vertexDataOffsPos = 0;

    float3 pos;
    if (vertexFlag_D[idx] == 1.0) {
        pos.x = vertexDataTrackedBack_D[vertexDataStride*idx + vertexDataOffsPos +0];
        pos.y = vertexDataTrackedBack_D[vertexDataStride*idx + vertexDataOffsPos +1];
        pos.z = vertexDataTrackedBack_D[vertexDataStride*idx + vertexDataOffsPos +2];
    } else {
        pos.x = vertexDataStart_D[vertexDataStride*idx + vertexDataOffsPos +0];
        pos.y = vertexDataStart_D[vertexDataStride*idx + vertexDataOffsPos +1];
        pos.z = vertexDataStart_D[vertexDataStride*idx + vertexDataOffsPos +2];
    }

    // Revert translation to move to origin
    pos.x -= translation.x;
    pos.y -= translation.y;
    pos.z -= translation.z;

    // Revert rotation
    float3 posRot;
    posRot.x = rotation_D[0] * pos.x +
            rotation_D[3] * pos.y +
            rotation_D[6] * pos.z;
    posRot.y = rotation_D[1] * pos.x +
            rotation_D[4] * pos.y +
            rotation_D[7] * pos.z;
    posRot.z = rotation_D[2] * pos.x +
            rotation_D[5] * pos.y +
            rotation_D[8] * pos.z;

    // Move to old centroid
    posRot.x += centroid.x;
    posRot.y += centroid.y;
    posRot.z += centroid.z;

    vertexAttrib_D[idx] = abs(vertexAttrib_D[idx] - ::SampleFieldAtPosTrilin_D<float, true>(posRot, tex1_D));
}


/*
 * DeformableGPUSurfaceMT::ComputeSurfAttribDiff
 */
bool DeformableGPUSurfaceMT::ComputeSurfAttribDiff(
        DeformableGPUSurfaceMT &surfStart,
        float centroid[3], // In case the start surface has been fitted using RMSD
        float rotMat[9],
        float transVec[3],
        float *tex0_D,
        int3 texDim0,
        float3 texOrg0,
        float3 texDelta0,
        float *tex1_D,
        int3 texDim1,
        float3 texOrg1,
        float3 texDelta1) {

    using namespace vislib::sys;


    if (!this->InitVtxAttribVBO(this->vertexCnt)) {
        return false;
    }

    // Get pointer to vertex attribute array
    cudaGraphicsResource* cudaTokens[3];
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboVtxAttr,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1],
            this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[2],
            surfStart.GetVtxDataVBO(),
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsMapResources(3, cudaTokens, 0))) {
        return false;
    }
    // Get mapped pointers to the vertex data buffers
    float *vertexAttrib_D;
    size_t vboVtxAttribSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vertexAttrib_D), // The mapped pointer
            &vboVtxAttribSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }
    // Get mapped pointers to the vertex data buffers
    float *vertexDataEnd_D;
    size_t vboEndSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vertexDataEnd_D), // The mapped pointer
            &vboEndSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }
    // Get mapped pointers to the vertex data buffers
    float *vertexDataStart_D;
    size_t vboStartSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vertexDataStart_D), // The mapped pointer
            &vboStartSize,              // The size of the accessible data
            cudaTokens[2]))) {                 // The mapped resource
        return false;
    }

    // Init grid params
    // Init CUDA grid for texture #0
    if (!initGridParams(texDim0, texOrg0, texDelta0)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DeformableGPUSurfaceMT::ClassName());
        return false;
    }

    // Compute difference for new and old vertices (after subdivision)
    // Part one: sample value for new vertices
    DeformableGPUSurfaceMT_ComputeSurfAttribDiff0_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            vertexAttrib_D,
            vertexDataEnd_D,
            tex0_D,
            this->vertexCnt);

    CudaDevArr<float> rotate_D;
    // Rotate for best fit
    rotate_D.Validate(9);
    if (!CudaSafeCall(cudaMemcpy((void *)rotate_D.Peek(), &rotMat[0],
            9*sizeof(float), cudaMemcpyHostToDevice))) {
        return false;
    }

    // Init grid params
    // Init CUDA grid for texture #1
    if (!initGridParams(texDim1, texOrg1, texDelta1)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                DeformableGPUSurfaceMT::ClassName());
        return false;
    }

    if (this->vertexFlag_D.GetCount() == 0) {
        if (!CudaSafeCall(this->vertexFlag_D.Validate(this->vertexCnt))) {
            return false;
        }
        if (!CudaSafeCall(this->vertexFlag_D.Set(0x00))) {
            return false;
        }
    }

    // Compute difference for new and old vertices (after subdivision)
    // Part two: sample value for old/tracked back vertices
    DeformableGPUSurfaceMT_ComputeSurfAttribDiff1_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            vertexAttrib_D,
            vertexDataStart_D,
            this->trackedSubdivVertexData_D.Peek(), // Tracked back vertices, needed for sampling
            this->vertexFlag_D.Peek(),
            tex1_D,
            rotate_D.Peek(),
            make_float3(transVec[0],transVec[1],transVec[2]),
            make_float3(centroid[0],centroid[1],centroid[2]),
            this->vertexCnt);

    if (!CheckForCudaError()) {
        return false;
    }

    rotate_D.Release();

    if (!CudaSafeCall(cudaGraphicsUnmapResources(3, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[2]))) {
        return false;
    }

    return true;
}


__global__ void DeformableGPUSurfaceMT_ComputeTriangleFaceNormal_D(
        float3 *triFaceNormals_D,
        float *vertexData_D,
        uint *triangleidx_D,
        uint triangleCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= triangleCnt) {
        return;
    }

    float3 pos0 = make_float3(
            vertexData_D[9*triangleidx_D[3*idx+0]+0],
            vertexData_D[9*triangleidx_D[3*idx+0]+1],
            vertexData_D[9*triangleidx_D[3*idx+0]+2]);

    float3 pos1 = make_float3(
            vertexData_D[9*triangleidx_D[3*idx+1]+0],
            vertexData_D[9*triangleidx_D[3*idx+1]+1],
            vertexData_D[9*triangleidx_D[3*idx+1]+2]);

    float3 pos2 = make_float3(
            vertexData_D[9*triangleidx_D[3*idx+2]+0],
            vertexData_D[9*triangleidx_D[3*idx+2]+1],
            vertexData_D[9*triangleidx_D[3*idx+2]+2]);

    float3 vec0 = (pos1 - pos0);
    float3 vec1 = (pos2 - pos0);

    float3 norm = normalize(cross(vec0, vec1));

    // Write normal
    triFaceNormals_D[idx*3+0] = norm;
    triFaceNormals_D[idx*3+1] = norm;
    triFaceNormals_D[idx*3+2] = norm;
}


__global__ void DeformableGPUSurfaceMT_CheckTriNormals_D(
        float3 *triFaceNormals_D,
        uint *triangleNeighbors_D,
        uint triangleCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= triangleCnt) {
        return;
    }

    uint n0 = triangleNeighbors_D[3*idx+0];
    uint n1 = triangleNeighbors_D[3*idx+1];
    uint n2 = triangleNeighbors_D[3*idx+2];

    float3 norm = normalize(triFaceNormals_D[idx]);
    float3 norm0 = normalize(triFaceNormals_D[n0]);
    float3 norm1 = normalize(triFaceNormals_D[n1]);
    float3 norm2 = normalize(triFaceNormals_D[n2]);
    float3 avgNorm = (norm0+norm1+norm2)*0.3;

    __syncthreads();

    if ((dot(norm, avgNorm) < 0)) {
        triFaceNormals_D[idx] = make_float3(0.0, 0.0, 0.0);
    }

}



__global__ void DeformableGPUSurfaceMT_ComputeNormalsSubdiv_D(
        float *vertexData_D,
        float3 *normals_D,
        uint vertexCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    float3 norm = normalize(normals_D[idx]);

    // Write normal
    vertexData_D[idx*9+3] = norm.x;
    vertexData_D[idx*9+4] = norm.y;
    vertexData_D[idx*9+5] = norm.z;
}


/*
 * see http://blog.csdn.net/newtonbear/article/details/12768377
 */
template <typename Key, typename Value>
int reduce_by_key_with_raw_pointers(Key* d_key, Key* d_key_last, Value* d_value,
        Key* d_okey, Value* d_ovalue) {
    thrust::device_ptr<Key> d_keyp = thrust::device_pointer_cast(d_key);
    thrust::device_ptr<Key> d_key_lastp = thrust::device_pointer_cast(d_key_last);
    thrust::device_ptr<Value> d_valuep = thrust::device_pointer_cast(d_value);
    thrust::device_ptr<Key> d_okeyp = thrust::device_pointer_cast(d_okey);
    thrust::device_ptr<Value> d_ovaluep = thrust::device_pointer_cast(d_ovalue);
    thrust::pair<thrust::device_ptr<Key>, thrust::device_ptr<Value> > new_end;
    new_end = thrust::reduce_by_key(d_keyp, d_key_lastp, d_valuep, d_okeyp, d_ovaluep);
    return new_end.first - d_okeyp;
}


void OutputDevArrayUint(uint* d_array, int count, const char* name) {
    // DEBUG Print
    HostArr<uint> h_array;
    h_array.Validate(count);
    if (!CudaSafeCall(cudaMemcpy(h_array.Peek(), d_array, sizeof(uint)*count, cudaMemcpyDeviceToHost))) {
        return;
    }
    for (int i = 0; i < count; ++i) {
        printf("%s %i: %u\n", name, i, h_array.Peek()[i]);
    }
    h_array.Release();
    // END DEBUG
}


void OutputDevArrayFloat3(float3* d_array, int count, const char* name) {
    // DEBUG Print
    HostArr<float3> h_array;
    h_array.Validate(count);
    if (!CudaSafeCall(cudaMemcpy(h_array.Peek(), d_array, sizeof(float3)*count,
            cudaMemcpyDeviceToHost))) {
        return;
    }
    for (int i = 0; i < count; ++i) {
        printf("%s %i: %f %f %f\n", name, i,
                h_array.Peek()[i].x,
                h_array.Peek()[i].y,
                h_array.Peek()[i].z);
    }
    h_array.Release();
    // END DEBUG
}


/*
 * DeformableGPUSurfaceMT::ComputeNormalsSubdiv
 */
bool DeformableGPUSurfaceMT::ComputeNormalsSubdiv() {

    // Get pointer to vertex attribute array
    cudaGraphicsResource* cudaTokens[2];
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1],
            this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }
    // Get mapped pointers to the vertex data buffers
    float *vertexBuffer_D;
    size_t vboVertexBufferSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vertexBuffer_D), // The mapped pointer
            &vboVertexBufferSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }
    // Get mapped pointers to the vertex data buffers
    unsigned int *triIdx_D;
    size_t vboTriIdxSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&triIdx_D), // The mapped pointer
            &vboTriIdxSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }

    // 1. Compute triangle face normals
    if (!CudaSafeCall(this->triangleFaceNormals_D.Validate(this->triangleCnt*3))) {
        return false;
    }
    DeformableGPUSurfaceMT_ComputeTriangleFaceNormal_D <<< Grid(this->triangleCnt, 256), 256 >>> (
            this->triangleFaceNormals_D.Peek(),
            vertexBuffer_D,
            triIdx_D,
            this->triangleCnt);
    if (!CheckForCudaError()) {
        return false;
    }

//    // DEBUG CHECK FACE NORMALS
//    DeformableGPUSurfaceMT_CheckTriNormals_D <<< Grid(this->triangleCnt, 256), 256 >>> (
//            this->triangleFaceNormals_D.Peek(),
//            this->triangleNeighbors_D.Peek(),
//            this->triangleCnt);

    // 2. Sort triangle normals by key
    // Copy triangle indices
    if (!CudaSafeCall(this->triangleIdxTmp_D.Validate(this->triangleCnt*3))) {
        return false;
    }
    if (!CudaSafeCall(cudaMemcpy(this->triangleIdxTmp_D.Peek(), triIdx_D,
            sizeof(uint)*this->triangleCnt*3, cudaMemcpyDeviceToDevice))) {
        return false;
    }

    thrust::sort_by_key(
            thrust::device_ptr<uint>(this->triangleIdxTmp_D.Peek()),
            thrust::device_ptr<uint>(this->triangleIdxTmp_D.Peek() + this->triangleCnt*3),
            thrust::device_ptr<float3>(this->triangleFaceNormals_D.Peek()));
    if (!CheckForCudaError()) {
        return false;
    }
//    OutputDevArrayUint(this->triangleIdxTmp_D.Peek(), this->triangleCnt*3, "TRI IDX");

    // 3. Reduce vertex normals by key
//    if (!CudaSafeCall(this->vertexNormalsIndxOffs_D.Validate(this->triangleCnt*3))) {
//        return false;
//    }
//    if (!CudaSafeCall(this->reducedVertexKeysTmp_D.Validate(this->vertexCnt))) {
//        return false;
//    }
//    thrust::device_ptr<uint> D = thrust::device_ptr<uint>(this->vertexNormalsIndxOffs_D.Peek());
//    thrust::fill(D, D + this->vertexCnt, 1);
//    thrust::device_ptr<uint> dev_ptr(this->vertexNormalsIndxOffs_D.Peek());
//    thrust::fill(dev_ptr, dev_ptr + this->triangleCnt*3, 1);

//    int n = reduce_by_key_with_raw_pointers<uint, uint>(
//            this->triangleIdxTmp_D.Peek(),
//            this->triangleIdxTmp_D.Peek() + this->triangleCnt*3,
//            this->vertexNormalsIndxOffs_D.Peek(),
//            this->triangleIdxTmp_D.Peek(),
//            this->reducedVertexKeysTmp_D.Peek());

    //OutputDevArrayUint(this->reducedVertexKeysTmp_D.Peek(), this->vertexCnt, "NORMAL CNT");

    if (!CudaSafeCall(this->reducedNormalsTmp_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->outputArrayTmp_D.Validate(this->vertexCnt))) {
        return false;
    }

    int n = reduce_by_key_with_raw_pointers<uint, float3>(
            this->triangleIdxTmp_D.Peek(),
            this->triangleIdxTmp_D.Peek() + this->triangleCnt*3,
            this->triangleFaceNormals_D.Peek(),
            this->outputArrayTmp_D.Peek(),
            this->reducedNormalsTmp_D.Peek());

//    OutputDevArrayFloat3(this->reducedNormalsTmp_D.Peek(), this->vertexCnt, "NORMAL ");
//    printf("N %u, vertexCnt %u\n", n, this->vertexCnt);

    // Compute actual normals
    DeformableGPUSurfaceMT_ComputeNormalsSubdiv_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            vertexBuffer_D,
            this->reducedNormalsTmp_D.Peek(),
            this->vertexCnt);
    if (!CheckForCudaError()) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }
    return ::CheckForCudaError();
}


/*
 * DeformableGPUSurfaceMT::PrintVertexBuffer
 */
void DeformableGPUSurfaceMT::PrintVertexBuffer(size_t cnt) {
    // Get pointer to vertex attribute array
    cudaGraphicsResource* cudaTokens[1];
    CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboVtxData,
            cudaGraphicsMapFlagsNone));
    CudaSafeCall(cudaGraphicsMapResources(1, cudaTokens, 0));

    // Get mapped pointers to the vertex data buffers
    float *vertexBuffer_D;
    size_t vboVertexBufferSize;
    CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vertexBuffer_D), // The mapped pointer
            &vboVertexBufferSize,              // The size of the accessible data
            cudaTokens[0]));

    HostArr<float> vertexBuffer;
    vertexBuffer.Validate(cnt*this->vertexDataStride);
    CudaSafeCall(cudaMemcpy(vertexBuffer.Peek(), vertexBuffer_D,
            sizeof(float)*cnt*this->vertexDataStride,
            cudaMemcpyDeviceToHost));

    for (int i = 0; i < cnt; ++i) {
        printf("VERTEX BUFFER %f %f %f, %f %f %f, %f %f %f\n",
                vertexBuffer.Peek()[i*this->vertexDataStride+0],
                vertexBuffer.Peek()[i*this->vertexDataStride+1],
                vertexBuffer.Peek()[i*this->vertexDataStride+2],
                vertexBuffer.Peek()[i*this->vertexDataStride+3],
                vertexBuffer.Peek()[i*this->vertexDataStride+4],
                vertexBuffer.Peek()[i*this->vertexDataStride+5],
                vertexBuffer.Peek()[i*this->vertexDataStride+6],
                vertexBuffer.Peek()[i*this->vertexDataStride+7],
                vertexBuffer.Peek()[i*this->vertexDataStride+8]);
    }

    vertexBuffer.Release();

    CudaSafeCall(cudaGraphicsUnmapResources(1, cudaTokens, 0));
    CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]));
}

void DeformableGPUSurfaceMT::PrintExternalForces(size_t cnt) {

    HostArr<float> externalForces;
    externalForces.Validate(cnt*4);
    CudaSafeCall(cudaMemcpy(externalForces.Peek(), this->externalForces_D.Peek(),
            sizeof(float)*cnt*4,
            cudaMemcpyDeviceToHost));

    for (int i = 0; i < cnt; ++i) {
        printf("EXT FORCES %f %f %f\n",
                externalForces.Peek()[4*i+0],
                externalForces.Peek()[4*i+1],
                externalForces.Peek()[4*i+2]);
    }
    externalForces.Release();
}


void DeformableGPUSurfaceMT::PrintCubeStates(size_t cnt) {
    HostArr<unsigned int> cubeStates;
    cubeStates.Validate(cnt);
    CudaSafeCall(cudaMemcpy(cubeStates.Peek(), this->cubeStates_D.Peek(),
            sizeof(unsigned int)*cnt,
            cudaMemcpyDeviceToHost));

    for (int i = 0; i < cnt; ++i) {
        printf("CUBESTATES %u\n", cubeStates.Peek()[i]);
    }
    cubeStates.Release();
}

/*
 * DeformableGPUSurfaceMT::ComputeMeshLaplacian
 */
bool DeformableGPUSurfaceMT::ComputeMeshLaplacian() {
    typedef vislib::math::Vector<float, 3> Vec3f;

    // Get pointer to vertex attribute array
    cudaGraphicsResource* cudaTokens[2];
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1],
            this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
        return false;
    }
    // Get mapped pointers to the vertex data buffers
    float *vertexBuffer_D;
    size_t vboVertexBufferSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vertexBuffer_D), // The mapped pointer
            &vboVertexBufferSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }
    // Get mapped pointers to the vertex data buffers
    unsigned int *triIdx_D;
    size_t vboTriIdxSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&triIdx_D), // The mapped pointer
            &vboTriIdxSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }

    // Copy vertex data and triangle indices to CPU
    HostArr<float> vertexData;
    HostArr<unsigned int> triIdx;
    vertexData.Validate(this->vertexCnt*9);
    if (!CudaSafeCall(cudaMemcpy(vertexData.Peek(), vertexBuffer_D,
            sizeof(float)*this->vertexCnt*9, cudaMemcpyDeviceToHost))) {
        return false;
    }
    triIdx.Validate(this->triangleCnt*3);
    if (!CudaSafeCall(cudaMemcpy(triIdx.Peek(), triIdx_D,
            sizeof(uint)*this->triangleCnt*3, cudaMemcpyDeviceToHost))) {
        return false;
    }

    // Build vertex neighbor list
    vislib::Array<vislib::Array<uint> > vtxNeighbors;
    vtxNeighbors.SetCount(this->vertexCnt);
    // Loop through all triangles
    for (size_t tri = 0; tri < this->triangleCnt; ++tri) {
        uint idx0 = triIdx.Peek()[3*tri+0];
        uint idx1 = triIdx.Peek()[3*tri+1];
        uint idx2 = triIdx.Peek()[3*tri+2];
        if (vtxNeighbors[idx0].Find(idx1) == NULL) {
            vtxNeighbors[idx0].Append(idx1);
        }
        if (vtxNeighbors[idx0].Find(idx2) == NULL) {
            vtxNeighbors[idx0].Append(idx2);
        }
        if (vtxNeighbors[idx1].Find(idx0) == NULL) {
            vtxNeighbors[idx1].Append(idx0);
        }
        if (vtxNeighbors[idx1].Find(idx2) == NULL) {
            vtxNeighbors[idx1].Append(idx2);
        }
        if (vtxNeighbors[idx2].Find(idx0) == NULL) {
            vtxNeighbors[idx2].Append(idx0);
        }
        if (vtxNeighbors[idx2].Find(idx1) == NULL) {
            vtxNeighbors[idx2].Append(idx1);
        }
    }

//    // DEBUG printf vertex neighbor list
//    printf("Computing vertex neighbor list...\n");
//    for (size_t v = 0; v < this->vertexCnt; ++v) {
//        printf("%u: ", v);
//        for (size_t n = 0; n < vtxNeighbors[v].Count(); ++n) {
//            printf("%u ", vtxNeighbors[v][n]);
//        }
//        printf("\n");
//    }
//    // End DEBUG

    printf("Computing mesh Laplacian ...\n");
    HostArr<float> vtxLaplacian;
    vtxLaplacian.Validate(this->vertexCnt*3);
    // Loop through all vertices
    for (size_t v = 0; v < this->vertexCnt; ++v) {
        float normSum = 0.0f;
        vtxLaplacian.Peek()[3*v+0] = 0.0f;
        vtxLaplacian.Peek()[3*v+1] = 0.0f;
        vtxLaplacian.Peek()[3*v+2] = 0.0f;
        Vec3f pos(vertexData.Peek()[9*v+0],
                  vertexData.Peek()[9*v+1],
                  vertexData.Peek()[9*v+2]);
        //float minAngle = 1000.0f;
        //float maxAngle = 0.0f;
        Vec3f currNPos;
        Vec3f nextNPos;
        for (size_t n = 0; n < vtxNeighbors[v].Count(); ++n) {
            // Get position of neighbor
            uint nIdxCurr = vtxNeighbors[v][n];
            if (n == vtxNeighbors[v].Count()-1)
                uint nIdxNext = vtxNeighbors[v][0];
            else
                uint nIdxNext = vtxNeighbors[v][n+1];
            currNPos.Set(vertexData.Peek()[9*nIdxCurr+0],
                       vertexData.Peek()[9*nIdxCurr+1],
                       vertexData.Peek()[9*nIdxCurr+2]);
            nextNPos.Set(vertexData.Peek()[9*nIdxCurr+0],
                       vertexData.Peek()[9*nIdxCurr+1],
                       vertexData.Peek()[9*nIdxCurr+2]);
//            normSum += (pos-posN).Length();
//            Vec3f dist = pos-posN;
//            dist.Normalise();
//            vtxLaplacian.Peek()[3*v+0] += dist.X();
//            vtxLaplacian.Peek()[3*v+1] += dist.Y();
//            vtxLaplacian.Peek()[3*v+2] += dist.Z();
        }

        // Normalize
        vtxLaplacian.Peek()[3*v+0] /= normSum;
        vtxLaplacian.Peek()[3*v+1] /= normSum;
        vtxLaplacian.Peek()[3*v+2] /= normSum;
    }

//    // DEBUG Print mesh Laplacian norm
//    for (size_t v = 0; v < this->vertexCnt; ++v) {
//        printf("Laplacian %u: %f\n", v, vtxLaplacian.Peek()[v]);
//    }
//    // End DEBUG

    // Write to vertex attribute array
    if (!CudaSafeCall(this->geometricLaplacian_D.Validate(this->vertexCnt*3))) {
        return false;
    }
    if (!CudaSafeCall(cudaMemcpy(this->geometricLaplacian_D.Peek(), vtxLaplacian.Peek(),
            sizeof(float)*this->vertexCnt*3, cudaMemcpyHostToDevice))) {
        return false;
    }

    // Cleanup
    vertexData.Release();
    triIdx.Release();
    vtxLaplacian.Release();
    vtxNeighbors.Clear();
    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }
    return ::CheckForCudaError();
}


/*
 * DeformableGPUSurfaceMT_ComputeSurfAttribDiff1_D
 */
__global__ void DeformableGPUSurfaceMT_ComputeAttribDiff_D (
        float *vertexAttrib_D,
        float *meshLaplacian_D,
        float *meshLaplacianOther_D,
        uint vertexCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }
    float3 otherAttrib = make_float3(
            meshLaplacianOther_D[3*idx+0],
            meshLaplacianOther_D[3*idx+1],
            meshLaplacianOther_D[3*idx+2]);
    float3 thisAttrib = make_float3(
            meshLaplacian_D[3*idx+0],
            meshLaplacian_D[3*idx+1],
            meshLaplacian_D[3*idx+2]);

    //vertexAttrib_D[idx] = abs(thisAttrib-otherAttrib);
    vertexAttrib_D[idx] = length(thisAttrib-otherAttrib);
}



/*
 * DeformableGPUSurfaceMT::ComputeMeshLaplacianDiff
 */
bool DeformableGPUSurfaceMT::ComputeMeshLaplacianDiff(
        DeformableGPUSurfaceMT &surfStart) {

    if (this->nFlaggedVertices != 0) {
        printf("No subdivision allowed in this case!\n");
        return false;
    }

    typedef vislib::math::Vector<float, 3> Vec3f;

    if (!this->InitVtxAttribVBO(this->vertexCnt)) {
        return false;
    }

    if (!surfStart.ComputeMeshLaplacian()) {
        return false;
    }
    if (!this->ComputeMeshLaplacian()) {
        return false;
    }

    // Get pointer to vertex attribute array
    cudaGraphicsResource* cudaTokens[1];
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboVtxAttr,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsMapResources(1, cudaTokens, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *vertexAttrib_D;
    size_t vertexAttribSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vertexAttrib_D), // The mapped pointer
            &vertexAttribSize,              // The size of the accessible data
            cudaTokens[0]))) {                 // The mapped resource
        return false;
    }

    // Compute difference
    DeformableGPUSurfaceMT_ComputeAttribDiff_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            vertexAttrib_D,
            this->PeekGeomLaplacian(),
            surfStart.PeekGeomLaplacian(),
            this->vertexCnt);
    if (!CheckForCudaError()) {
        return false;
    }


    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, cudaTokens, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }

    return ::CheckForCudaError();
}
