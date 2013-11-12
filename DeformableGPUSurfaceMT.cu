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

#include <glh/glh_extensions.h>
#include "DeformableGPUSurfaceMT.h"
#ifdef WITH_CUDA

#include "ogl_error_check.h"
#include "cuda_error_check.h"
#include "HostArr.h"
#include "DiffusionSolver.h"
#include "CUDAGrid.cuh"
#include "cuda_helper.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

//#define USE_TIMER

using namespace megamol;
using namespace megamol::protein;


/**
 * 'Safe' inverse sqrt, that prevents dividing by zero
 *
 * @param x The input value
 * @return The inverse sqrt if x>0, 0.0 otherwise
 */
inline __device__ float safeRsqrtf(float x) {
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
inline __device__ float3 safeNormalize(float3 v) {
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

    float distSample = ::SampleFieldAt_D<float>(gridCoord, distField_D);
    float volSample = ::SampleFieldAt_D<float>(gridCoord, field_D);

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

    /* Alternative 2: calc variance of angle between normals */

//    const uint baseIdx0 = vertexDataStride*triangleVtxIdx_D[3*idx+0];
//    const uint baseIdx1 = vertexDataStride*triangleVtxIdx_D[3*idx+1];
//    const uint baseIdx2 = vertexDataStride*triangleVtxIdx_D[3*idx+2];
//    const float3 n0 = make_float3(vertexData_D[baseIdx0+vertexDataOffsNormal+0],
//                                  vertexData_D[baseIdx0+vertexDataOffsNormal+1],
//                                  vertexData_D[baseIdx0+vertexDataOffsNormal+2]);
//    const float3 n1 = make_float3(vertexData_D[baseIdx1+vertexDataOffsNormal+0],
//                                  vertexData_D[baseIdx1+vertexDataOffsNormal+1],
//                                  vertexData_D[baseIdx1+vertexDataOffsNormal+2]);
//    const float3 n2 = make_float3(vertexData_D[baseIdx2+vertexDataOffsNormal+0],
//                                  vertexData_D[baseIdx2+vertexDataOffsNormal+1],
//                                  vertexData_D[baseIdx2+vertexDataOffsNormal+2]);
//    // Sample volume at midpoint
//    const float3 avgNormal = (n0+n1+n2)/3.0;
//    float dot0 = clamp(dot(n0, avgNormal), 0.0, 1.0);
//    float dot1 = clamp(dot(n1, avgNormal), 0.0, 1.0);
//    float dot2 = clamp(dot(n2, avgNormal), 0.0, 1.0);
//    float maxDot = max(dot0, max(dot1, dot2));
//    float flag = float(maxDot > 0.9);
//    vertexFlag_D[triangleVtxIdx_D[3*idx+0]] = flag;
//    vertexFlag_D[triangleVtxIdx_D[3*idx+1]] = flag;
//    vertexFlag_D[triangleVtxIdx_D[3*idx+2]] = flag;

    /* Alternative 3 Check whether the vertex lies in a active cell of the
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

    // Sample volume at midpoint
    const float3 midpoint = (p0+p1+p2)/3.0;

    // Get integer cell index
    int3 coords;
    coords.x = int((midpoint.x-gridOrg_D.x)/gridDelta_D.x);
    coords.y = int((midpoint.y-gridOrg_D.y)/gridDelta_D.y);
    coords.z = int((midpoint.z-gridOrg_D.z)/gridDelta_D.z);

    int cellIDx = ::GetCellIdxByGridCoords(coords);
    uint cellState = targetActiveCells_D[cellIDx];

    if (cellState == 0) {
        vertexFlag_D[triangleVtxIdx_D[3*idx+0]] = 1.0;
        vertexFlag_D[triangleVtxIdx_D[3*idx+1]] = 1.0;
        vertexFlag_D[triangleVtxIdx_D[3*idx+2]] = 1.0;
    }

    corruptTriangles_D[idx] = float(1-cellState);
}



///**
// * Writes a flag for every vertex that is adjacent to a corrupt triangles.
// *
// * @param[in,out] vertexData_D              The buffer with the vertex data
// * @param[in]     vertexDataStride          The stride for the vertex data
// *                                          buffer
// * @param[in]     vertexDataOffsPos         The position offset in the vertex
// *                                          data buffer
// * @param[in]     vertexDataOffsCorruptFlag The corruption flag offset in the
// *                                          vertex data buffer
// * @param[in]     triangleVtxIdx_D          Array with triangle vertex indices
// * @param[in]     volume_D                  The target volume defining the
// *                               iso-surface
// * @param[in]     externalForcesScl_D       Array with the scale factor for the external force
// * @param[in]     triangleCnt               The number of triangles
// * @param[in]     minDispl                  Minimum force scale to keep going
// * @param[in]     isoval                    The iso-value defining the iso-surface
// *
// * TODO
// */
//__global__ void DeformableGPUSurfaceMT_FlagCorruptTriangles_D(
//        float *corruptTriangles_D,
//        float *vertexData_D,
//        uint vertexDataStride,
//        uint vertexDataOffsPos,
//        uint vertexDataOffsNormal,
//        uint *triangleVtxIdx_D,
//        float *targetVol_D,
//        uint triangleCnt,
//        float isoval) {
//
//    const uint idx = ::getThreadIdx();
//    if (idx >= triangleCnt) {
//        return;
//    }
//
//    /* Alternative 1: Sample volume at triangle midpoint */
//
//    const uint baseIdx0 = vertexDataStride*triangleVtxIdx_D[3*idx+0];
//    const uint baseIdx1 = vertexDataStride*triangleVtxIdx_D[3*idx+1];
//    const uint baseIdx2 = vertexDataStride*triangleVtxIdx_D[3*idx+2];
//
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
//
//
//    float flag = float(::fabs(volSampleMidPoint-isoval) > 0.3);
//    corruptTriangles_D[idx] = flag;
//
//}



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
 * @param[in] dataArrSize The stride of the vertex data buffer
 */
__global__ void DeformableGPUSurfaceMT_InitExternalForceScl_D (
        float *arr_D,
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
    if (SampleFieldAtPosTrilin_D<float>(pos, volume_D) <= isoval) {
        arr_D[2*idx+0] = 1.0;
    } else {
        arr_D[2*idx+0] = -1.0;
    }

    // Init last displ scl with something bigger then minDispl;
    arr_D[2*idx+1] = minDispl + 0.1; // TODO Mult times two by shift
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
    float lastDisplLen = vertexExternalForcesScl_D[2*idx+1];
    if (lastDisplLen <= minDispl) return; // Vertex is converged

    const uint posBaseIdx = dataArrSize*idx+dataArrOffsPos;


    /* Retrieve stuff from global device memory */

    // Get initial position from global device memory
    float3 posOld = make_float3(
            vertexPosMapped_D[posBaseIdx+0],
            vertexPosMapped_D[posBaseIdx+1],
            vertexPosMapped_D[posBaseIdx+2]);

    // Get initial scale factor for external forces
    float externalForcesScl = vertexExternalForcesScl_D[2*idx];
    float externalForcesSclOld = externalForcesScl;

    // Get partial derivatives
    float3 laplacian = laplacian_D[idx];
    float3 laplacian2 = laplacian2_D[idx];


    /* Update position */

    // No warp divergence here, since useCubicInterpolation is the same for all
    // threads
    const float sampleDens = useCubicInterpolation
                    ? SampleFieldAtPosTricub_D<float>(posOld, targetVolume_D)
                    : SampleFieldAtPosTrilin_D<float>(posOld, targetVolume_D);

    // Switch sign and scale down if necessary
    bool negative = externalForcesScl < 0;
    bool outside = sampleDens <= isoval;
    int switchSign = int((negative && outside)||(!negative && !outside));
    externalForcesScl = externalForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
    externalForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));

    // Sample gradient by cubic interpolation
    float4 externalForceTmp = useCubicInterpolation
            ? SampleFieldAtPosTricub_D<float4>(posOld, gradient_D)
            : SampleFieldAtPosTrilin_D<float4>(posOld, gradient_D);

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
    vertexExternalForcesScl_D[2*idx] = externalForcesScl;

    // No branching occurs here, since the parameter is set globally
    float3 diff = posNew-posOld;
    float diffLen = length(diff);
    if ((trackPath)&&(abs(externalForcesScl) == 1.0f)) {
        //float3 diff = externalForce;
        vtxUncertainty_D[idx] += length(externalForce);
        //vtxUncertainty_D[idx] += 1.0f;
    }
    // Displ scl for convergence
    vertexExternalForcesScl_D[2*idx+1] = diffLen;
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
        float *vtxUncertainty_D,
        float4 *gradient_D,
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
    float lastDisplLen = vertexExternalForcesScl_D[2*idx+1];
    if (lastDisplLen <= minDispl) return; // Vertex is converged

    const uint posBaseIdx = dataArrSize*idx+dataArrOffsPos;


    /* Retrieve stuff from global device memory */

    // Get initial position from global device memory
    float3 posOld = make_float3(
            vertexPosMapped_D[posBaseIdx+0],
            vertexPosMapped_D[posBaseIdx+1],
            vertexPosMapped_D[posBaseIdx+2]);

    // Get initial scale factor for external forces
    float externalForcesScl = vertexExternalForcesScl_D[2*idx];

    /* Update position */

    // No warp divergence here, since useCubicInterpolation is the same for all
    // threads
    const float sampleDens = useCubicInterpolation
                    ? SampleFieldAtPosTricub_D<float>(posOld, targetVolume_D)
                    : SampleFieldAtPosTrilin_D<float>(posOld, targetVolume_D);

    // Switch sign and scale down if necessary
    bool negative = externalForcesScl < 0;
    bool outside = sampleDens <= isoval;
    int switchSign = int((negative && outside)||(!negative && !outside));
    externalForcesScl = externalForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
    externalForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));

    // Sample gradient by cubic interpolation
    float4 externalForceTmp = useCubicInterpolation
            ? SampleFieldAtPosTricub_D<float4>(posOld, gradient_D)
            : SampleFieldAtPosTrilin_D<float4>(posOld, gradient_D);

    float3 externalForce;
    externalForce.x = externalForceTmp.x;
    externalForce.y = externalForceTmp.y;
    externalForce.z = externalForceTmp.z;

    externalForce = safeNormalize(externalForce);
    externalForce *= forcesScl*externalForcesScl;

    float3 posNew = posOld + externalForce;

    /* Write back to global device memory */

    // New pos
    vertexPosMapped_D[posBaseIdx+0] = posNew.x;
    vertexPosMapped_D[posBaseIdx+1] = posNew.y;
    vertexPosMapped_D[posBaseIdx+2] = posNew.z;

    // Write external forces scale factor back to global device memory
    vertexExternalForcesScl_D[2*idx] = externalForcesScl;

    // No branching occurs here, since the parameter is set globally
    float3 diff = posNew-posOld;
    float diffLen = length(diff);
    if ((trackPath)&&(abs(externalForcesScl) == 1.0f)) {
        //float3 diff = externalForce;
        vtxUncertainty_D[idx] += length(externalForce);
        //vtxUncertainty_D[idx] += 1.0f;
    }
    // Displ scl for convergence
    vertexExternalForcesScl_D[2*idx+1] = diffLen;
}


/*
 * DeformableGPUSurfaceMT::DeformableGPUSurfaceMT
 */
DeformableGPUSurfaceMT::DeformableGPUSurfaceMT() : GPUSurfaceMT(),
        vboCorruptTriangleVertexFlag(0), vboUncertainty(0) {

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

    if (other.vboUncertainty) {
        // Destroy if necessary
        if (this->vboUncertainty) {
            glBindBufferARB(GL_ARRAY_BUFFER, this->vboUncertainty);
            glDeleteBuffersARB(1, &this->vboUncertainty);
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
            this->vboUncertainty = 0;
        }

        // Create vertex buffer object for triangle indices
        glGenBuffersARB(1, &this->vboUncertainty);

        CheckForGLError();

        // Map as copy buffer
        glBindBufferARB(GL_COPY_READ_BUFFER, other.vboUncertainty);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, this->vboUncertainty);
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
__global__ void DeformableGPUSurfaceMT_ComputeValidTriangleAreas_D(
        float *trianglesArea_D,
        float *corruptTriangleFlag_D,
        float *vertexPos_D,
        uint *triangleIdx_D,
        uint triangleCnt) {

    const uint idx = ::getThreadIdx();
    if (idx >= triangleCnt) {
        return;
    }

    float flag = corruptTriangleFlag_D[idx];
    //float flag = 0.0f;

    float3 pos0, pos1, pos2;
    pos0.x = vertexPos_D[3*triangleIdx_D[3+idx+0]+0];
    pos0.y = vertexPos_D[3*triangleIdx_D[3+idx+0]+1];
    pos0.z = vertexPos_D[3*triangleIdx_D[3+idx+0]+2];
    pos1.x = vertexPos_D[3*triangleIdx_D[3+idx+1]+0];
    pos1.y = vertexPos_D[3*triangleIdx_D[3+idx+1]+1];
    pos1.z = vertexPos_D[3*triangleIdx_D[3+idx+1]+2];
    pos2.x = vertexPos_D[3*triangleIdx_D[3+idx+2]+0];
    pos2.y = vertexPos_D[3*triangleIdx_D[3+idx+2]+1];
    pos2.z = vertexPos_D[3*triangleIdx_D[3+idx+2]+2];

    float3 midPnt = (pos0+pos1)*0.5;
    float3 hVec = pos2 - midPnt;
    trianglesArea_D[idx] = length(pos0-pos1)*length(hVec)*0.5*(1.0-flag);
}


/*
 * DeformableGPUSurfaceMT::GetTotalValidSurfArea
 */
float DeformableGPUSurfaceMT::GetTotalValidSurfArea() {
    // Compute triangle areas of all (non-corrupt) triangles
    if (!CudaSafeCall(this->accTriangleArea_D.Validate(this->triangleCnt))) {
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
            this->corruptTriangles_D.Peek(),
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
    float areaValidTriangles = thrust::reduce(
            thrust::device_ptr<float>(this->accTriangleArea_D.Peek()),
            thrust::device_ptr<float>(this->accTriangleArea_D.Peek() + this->triangleCnt));

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

    return areaValidTriangles;
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

//    // DEBUG Printf triangle indices
//    HostArr<unsigned int> triangleIdx;
//    triangleIdx.Validate(this->triangleCnt*3);
//    cudaMemcpy(triangleIdx.Peek(), vboTriangleIdxPt, sizeof(unsigned int)*this->triangleCnt*3, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < this->triangleCnt; ++i) {
//        if ((triangleIdx.Peek()[i*3+0] > this->vertexCnt) ||
//                (triangleIdx.Peek()[i*3+0] > this->vertexCnt)||
//                (triangleIdx.Peek()[i*3+0] > this->vertexCnt)) {
//
//            printf("wrong vertex index idx %i: %u %u %u (vtxCnt %u\n", i,
//                    triangleIdx.Peek()[i*3+0],
//                    triangleIdx.Peek()[i*3+1],
//                    triangleIdx.Peek()[i*3+2],
//                    this->vertexCnt);
//        }
//    }
//    // END DEBUG

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
            this->triangleCnt,
            isovalue);

    ::CheckForCudaErrorSync();

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
 * DeformableGPUSurfaceMT::InitUncertaintyVBO
 */
bool DeformableGPUSurfaceMT::InitUncertaintyVBO(size_t vertexCnt) {

    // Destroy if necessary
    if (this->vboUncertainty) {
        glBindBufferARB(GL_ARRAY_BUFFER, this->vboUncertainty);
        glDeleteBuffersARB(1, &this->vboUncertainty);
        this->vboUncertainty = 0;
    }

    // Create vertex buffer object for corrupt vertex flag
    glGenBuffersARB(1, &this->vboUncertainty);
    glBindBufferARB(GL_ARRAY_BUFFER, this->vboUncertainty);
    glBufferDataARB(GL_ARRAY_BUFFER, sizeof(float)*vertexCnt, 0, GL_DYNAMIC_DRAW);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

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
float DeformableGPUSurfaceMT::IntUncertaintyOverSurfArea() {


    // Compute triangle areas of all (non-corrupt) triangles
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
             this->vboUncertainty,
             cudaGraphicsMapFlagsNone))) {
         return false;
     }

     if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokens, 0))) {
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

     // Get mapped pointers to the vertex data buffers
     float *uncertaintyPt;
     if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
             reinterpret_cast<void**>(&uncertaintyPt), // The mapped pointer
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
    if (!this->InitUncertaintyVBO(this->vertexCnt)) {
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
            &cudaTokens[1], this->vboUncertainty,
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
    float *vboUncertaintyPt;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboUncertaintyPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }

    // Init forces scale factor with -1 or 1, depending on whether they start
    // outside or inside the isosurface
    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt))) {
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
            vboUncertaintyPt,
            volDim,
            volOrg,
            volDelta,
            isovalue,
            (interpMode == INTERP_CUBIC),
            maxIt,
            surfMappedMinDisplScl,
            springStiffness,
            forceScl,
            externalForcesWeight)) {
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
    if (!this->InitUncertaintyVBO(this->vertexCnt)) {
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
            &cudaTokens[1], this->vboUncertainty,
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
    float *vboUncertaintyPt;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboUncertaintyPt), // The mapped pointer
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
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    DeformableGPUSurfaceMT_InitExternalForceScl_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            (float*)this->vertexExternalForcesScl_D.Peek(),
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
            vboUncertaintyPt,
            volDim,
            volOrg,
            volDelta,
            isovalue,
            (interpMode == INTERP_CUBIC),
            maxIt,
            surfMappedMinDisplScl,
            springStiffness,
            forceScl,
            externalForcesWeight)) {
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
    if (!this->InitUncertaintyVBO(this->vertexCnt)) {
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
            &cudaTokens[1], this->vboUncertainty,
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
    float *vboUncertaintyPt;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboUncertaintyPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }

    // Init forces scale factor with -1 or 1, depending on whether they start
    // outside or inside the isosurface
    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt))) {
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
            vboUncertaintyPt,
            volDim,
            volOrg,
            volDelta,
            isovalue,
            (interpMode == INTERP_CUBIC),
            maxIt,
            surfMappedMinDisplScl,
            springStiffness,
            forceScl,
            externalForcesWeight)) {
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
        unsigned int gvfIt) {

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

    if ((volumeTarget_D == NULL)||(volumeSource_D == NULL)) {
        return false;
    }

    if (!this->initExtForcesTwoWayGVF(
            volumeSource_D,
            volumeTarget_D,
            cellStatesSource_D,
            cellStatesTarget_D,
            volDim, volOrg, volDelta,
            isovalue, gvfScl, gvfIt)) {
        return false;
    }

    // Init vbo with uncertainty information
    if (!this->InitUncertaintyVBO(this->vertexCnt)) {
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
            &cudaTokens[1], this->vboUncertainty,
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
    float *vboUncertaintyPt;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboUncertaintyPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            cudaTokens[1]))) {                 // The mapped resource
        return false;
    }

    // Init forces scale factor with -1 or 1, depending on whether they start
    // outside or inside the isosurface
    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt))) {
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
            vboUncertaintyPt,
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
            false)) { // use external and internal forces
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
            this->vertexExternalForcesScl_D.GetCount()*sizeof(float2),
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

    CudaSafeCall(this->displLen_D.Validate(rhs.displLen_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->displLen_D.Peek(),
            rhs.displLen_D.PeekConst(),
            this->displLen_D.GetCount()*sizeof(float),
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

    if (rhs.vboUncertainty) {
        // Destroy if necessary
        if (this->vboUncertainty) {
            glBindBufferARB(GL_ARRAY_BUFFER, this->vboUncertainty);
            glDeleteBuffersARB(1, &this->vboUncertainty);
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
            this->vboUncertainty = 0;
        }

        // Create vertex buffer object for triangle indices
        glGenBuffersARB(1, &this->vboUncertainty);

        CheckForGLError();

        // Map as copy buffer
        glBindBufferARB(GL_COPY_READ_BUFFER, rhs.vboUncertainty);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, this->vboUncertainty);
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

    if (this->vboCorruptTriangleVertexFlag) {
        glBindBufferARB(GL_ARRAY_BUFFER, this->vboCorruptTriangleVertexFlag);
        glDeleteBuffersARB(1, &this->vboCorruptTriangleVertexFlag);
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
        this->vboCorruptTriangleVertexFlag = 0;
    }

    if (this->vboUncertainty) {
        glBindBufferARB(GL_ARRAY_BUFFER, this->vboUncertainty);
        glDeleteBuffersARB(1, &this->vboUncertainty);
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
        this->vboUncertainty = 0;
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
        bool externalForcesOnly) {

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

    if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Set(0xff))) {
        return false;
    }

    // Init uncertainty buffer with zero
    if (!CudaSafeCall(cudaMemset(vtxUncertainty_D, 0x00, this->vertexCnt*sizeof(float)))) {
        return false;
    }

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

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
                    true, // Track path of vertices
                    this->vertexDataOffsPos,
                    this->vertexDataOffsNormal,
                    this->vertexDataStride);

            //        // DEBUG Print uncertainty
            //        HostArr<float> uncertainty;
            //        uncertainty.Validate(this->vertexCnt);
            //        cudaMemcpy(uncertainty.Peek(), vtxUncertainty_D, sizeof(float)*this->vertexCnt, cudaMemcpyDeviceToHost);
            ////        for (int i = 0; i < this->vertexCnt; ++i) {
            //        printf("---------------------------------------------\n");
            //        for (int i = 300; i < 310; ++i) {
            //            printf("uncertainty %i: %f\n", i, uncertainty.Peek()[i]);
            //        }
            //        uncertainty.Release();
            //        // END DEBUG

            ::CheckForCudaErrorSync();
        }
    } else {
        // TODO Timer
        for (uint i = 0; i < maxIt; ++i) {

            // Update vertex position
            DeformableGPUSurfaceMT_UpdateVtxPosExternalOnly_D <<< Grid(this->vertexCnt, 256), 256 >>> (
                    volTarget_D,
                    vertexBuffer_D,
                    (float*)this->vertexExternalForcesScl_D.Peek(),
                    vtxUncertainty_D,
                    (float4*)this->externalForces_D.Peek(),
                    this->vertexCnt,
                    forceScl,
                    isovalue,
                    surfMappedMinDisplScl,
                    useCubicInterpolation,
                    true, // Track path of vertices
                    this->vertexDataOffsPos,
                    this->vertexDataOffsNormal,
                    this->vertexDataStride);

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
            maxIt, this->vertexCnt, dt_ms/1000.0f);
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

    diff_D[idx] = ::SampleFieldAtPosTrilin_D<float>(pos, tex0_D);
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

    float valSec = ::SampleFieldAtPosTrilin_D<float>(pos, tex1_D);
    valSec = abs(valSec-valFirst);
    diff_D[idx] = valSec;
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

    float valSec = ::SampleFieldAtPosTrilin_D<float>(pos, tex1_D);
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

#endif // WITH_CUDA
