//
// ComparativeSurfacePotentialRenderer_surface_mapping.cu
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 10, 2013
//     Author: scharnkn
//

#include "ComparativeSurfacePotentialRenderer.cuh"
#include "ComparativeSurfacePotentialRenderer_inline_device_functions.cuh"
#include "constantGridParams.cuh"
#include "cuda_helper.h"
#include <cstdio>


// Toggle performance measurement and respective messages
//#define USE_TIMER
#define UPDATE_VERTEX_POSITION_BLOCKDIM 256

// Shut up eclipse syntax error highlighting
#ifdef __CDT_PARSER__
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __host__
#endif


/**
 * Computes the gradient of a given scalar field using central differences.
 * Border areas are omitted.
 *
 * @param[out] grad_D  The gradient field
 * @param[in]  field_D The scalar field
 */
__global__ void CalcVolGradient_D(float4 *grad_D, float *field_D) {

    const uint idx = ::GetThreadIndex();

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


/*
 * CalcVolGradient
 */
extern "C"
cudaError_t CalcVolGradient(float4 *grad_D, float *field_D, uint gridSize) {


#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Calculate gradient using finite differences
    CalcVolGradient_D <<< Grid(gridSize, 256), 256 >>> (grad_D, field_D);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'CalcVolGradient_D':                     %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
}


/**
 * Computes the gradient of a given scalar field using central differences.
 * Border areas are omitted.
 *
 * @param[out] grad_D  The gradient field
 * @param[in]  field_D The scalar field
 * @param[in]  field_D The distance field
 */
__global__ void CalcVolGradientWithDistField_D(float4 *grad_D, float *field_D,
        float *distField_D, float minDist, float isovalue) {

    const uint idx = ::GetThreadIndex();

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


/*
 * CalcVolGradientWithDistField
 */
cudaError_t CalcVolGradientWithDistField(float4 *grad_D, float *field_D,
        float *distField_D, float minDist, float isovalue, uint gridSize) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Calculate gradient using finite differences
    CalcVolGradientWithDistField_D <<< Grid(gridSize, 256), 256 >>> (
            grad_D, field_D, distField_D, minDist, isovalue);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'CalcVolGradientWithDistField_D':        %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
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
__global__ void ComputeDistField_D(
        float *vertexPos_D,
        float *distField_D,
        uint vertexCnt,
        uint dataArrOffs,
        uint dataArrSize) {

    // TODO This is very slow since it basically bruteforces all vertex
    //      positions and stores the distance to the nearest one.

    const uint idx = GetThreadIndex();

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


/*
 * ComputeDistField
 */
extern "C"
cudaError_t ComputeDistField(
        float *vertexPos_D,
        float *distField_D,
        uint volSize,
        uint vertexCnt,
        uint dataArrOffs,
        uint dataArrSize) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    ComputeDistField_D <<< Grid(volSize, 256), 256 >>> (
            vertexPos_D,
            distField_D,
            vertexCnt,
            dataArrOffs,
            dataArrSize);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeDistField_D':                    %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
}


/**
 * Taken from: 'Realtime collision detection'
 */
__device__ float3 ClosestPtPoint2Triangle(float3 p, float3 a, float3 b, float3 c) {
    float3 ab = b - a;
    float3 ac = c - a;
    float3 bc = c - b;

    // Compute parametric position s for projection P’ of P on AB,
    // P’ = A + s*AB, s = snom/(snom+sdenom)
    float snom = dot(p - a, ab), sdenom = dot(p - b, a - b);

    // Compute parametric position t for projection P’ of P on AC,
    // P’ = A + t*AC, s = tnom/(tnom+tdenom)
    float tnom = dot(p - a, ac), tdenom = dot(p - c, a - c);
    if (snom <= 0.0f && tnom <= 0.0f) {
        return a;
    }
    // Vertex region early out
    // Compute parametric position u for projection P’ of P on BC,
    // P’ = B + u*BC, u = unom/(unom+udenom)
    float unom = dot(p - b, bc), udenom = dot(p - c, b - c);
    if (sdenom <= 0.0f && unom <= 0.0f) return b; // Vertex region early out
    if (tdenom <= 0.0f && udenom <= 0.0f) return c; // Vertex region early out

    // P is outside (or on) AB if the triple scalar product [N PA PB] <= 0
    float3 n = cross(b - a, c - a);
    float vc = dot(n, cross(a - p, b - p));

    // If P outside AB and within feature region of AB,
    // return projection of P onto AB
    if (vc <= 0.0f && snom >= 0.0f && sdenom >= 0.0f) {
        return a + snom / (snom + sdenom) * ab;
    }

    // P is outside (or on) BC if the triple scalar product [N PB PC] <= 0
    float va = dot(n, cross(b - p, c - p));

    // If P outside BC and within feature region of BC,
    // return projection of P onto BC
    if (va <= 0.0f && unom >= 0.0f && udenom >= 0.0f)
        return b + unom / (unom + udenom) * bc;

    // P is outside (or on) CA if the triple scalar product [N PC PA] <= 0
    float vb = dot(n, cross(c - p, a - p));

    // If P outside CA and within feature region of CA,
    // return projection of P onto CA
    if (vb <= 0.0f && tnom >= 0.0f && tdenom >= 0.0f)
        return a + tnom / (tnom + tdenom) * ac;

    // P must project inside face region. Compute Q using barycentric coordinates
    float u = va / (va + vb + vc);
    float v = vb / (va + vb + vc);
    float w = 1.0f - u - v; // = vc / (va + vb + vc)
    return u * a + v * b + w * c;
}


/**
 * TODO
 */
__global__ void ComputeHausdorffDistance_D(
        float *vertexPos_D,
        float *vertexPosOld_D,
        float *hausdorffDist_D,
        uint vertexCnt,
        uint vertexCntOld,
        uint dataArrOffsPos,
        uint dataArrSize) {

    const uint idx = GetThreadIndex();

    if (idx >= vertexCnt) {
        return;
    }

    // Loop through all vertices to find minimal distance
    float3 posOld, pos = make_float3(
            vertexPos_D[dataArrSize*idx+dataArrOffsPos+0],
            vertexPos_D[dataArrSize*idx+dataArrOffsPos+1],
            vertexPos_D[dataArrSize*idx+dataArrOffsPos+2]);
    float dist2 = 10000.0;
    float len;

    for (uint i = 0; i < vertexCntOld; ++i) {
        posOld = make_float3(
                vertexPosOld_D[dataArrSize*i+dataArrOffsPos+0],
                vertexPosOld_D[dataArrSize*i+dataArrOffsPos+1],
                vertexPosOld_D[dataArrSize*i+dataArrOffsPos+2]);
        // Compute squared distance
        len = (posOld.x-pos.x)*(posOld.x-pos.x)+
              (posOld.y-pos.y)*(posOld.y-pos.y)+
              (posOld.z-pos.z)*(posOld.z-pos.z);
        dist2 = min(dist2, len);
    }

    hausdorffDist_D[idx] = sqrt(dist2);
}


/*
 * ComputeHausdorffDistance
 */
extern "C"
cudaError_t ComputeHausdorffDistance(
        float *vertexPos_D,
        float *vertexPosOld_D,
        float *hausdorffDist_D,
        uint vertexCnt,
        uint vertexCntOld,
        uint dataArrOffsPos,
        uint dataArrSiz) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    ComputeHausdorffDistance_D <<< Grid(vertexCnt, 256), 256 >>> (
            vertexPos_D,
            vertexPosOld_D,
            hausdorffDist_D,
            vertexCnt,
            vertexCntOld,
            dataArrOffsPos,
            dataArrSiz);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeHausdorffDistance_D':            %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
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
 *                               iso-surface
 * @param[in]     externalForcesScl_D       Array with the scale factor for the external force
 * @param[in]     triangleCnt               The number of triangles
 * @param[in]     minDispl                  Minimum force scale to keep going
 * @param[in]     isoval                    The iso-value defining the iso-surface
 */
__global__ void FlagVerticesInCorruptTriangles_D(
        float *vertexData_D,
        uint vertexDataStride,
        uint vertexDataOffsPos,
        uint vertexDataOffsCorruptFlag,
        uint *triangleVtxIdx_D,
        float *volume_D,
        float *externalForcesScl_D,
        uint triangleCnt,
        float minDispl,
        float isoval) {

    const uint idx = GetThreadIndex();
    if (idx >= triangleCnt) {
        return;
    }

    const uint baseIdx0 = vertexDataStride*triangleVtxIdx_D[3*idx+0];
    const uint baseIdx1 = vertexDataStride*triangleVtxIdx_D[3*idx+1];
    const uint baseIdx2 = vertexDataStride*triangleVtxIdx_D[3*idx+2];
    const float3 p0 = make_float3(vertexData_D[baseIdx0+vertexDataOffsPos+0],
                                  vertexData_D[baseIdx0+vertexDataOffsPos+1],
                                  vertexData_D[baseIdx0+vertexDataOffsPos+2]);
    const float3 p1 = make_float3(vertexData_D[baseIdx1+vertexDataOffsPos+0],
                                  vertexData_D[baseIdx1+vertexDataOffsPos+1],
                                  vertexData_D[baseIdx1+vertexDataOffsPos+2]);
    const float3 p2 = make_float3(vertexData_D[baseIdx2+vertexDataOffsPos+0],
                                  vertexData_D[baseIdx2+vertexDataOffsPos+1],
                                  vertexData_D[baseIdx2+vertexDataOffsPos+2]);

    // Sample volume at midpoint
    const float3 midPoint = (p0+p1+p2)/3.0;
    const float volSampleMidPoint = ::SampleFieldAtPosTricub_D<float>(midPoint, volume_D);
    float flag = float(::fabs(volSampleMidPoint-isoval) > 0.1);
    vertexData_D[baseIdx0+vertexDataOffsCorruptFlag] = flag;
    vertexData_D[baseIdx1+vertexDataOffsCorruptFlag] = flag;
    vertexData_D[baseIdx2+vertexDataOffsCorruptFlag] = flag;
}

extern "C"
cudaError_t FlagVerticesInCorruptTriangles(
        float *vertexData_D,
        uint vertexDataStride,
        uint vertexDataOffsPos,
        uint vertexDataOffsCorruptFlag,
        uint *triangleVtxIdx_D,
        float *volume_D,
        float *externalForcesScl_D,
        uint triangleCnt,
        float minDispl,
        float isoval) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    FlagVerticesInCorruptTriangles_D <<< Grid(triangleCnt, 256), 256 >>> (
            vertexData_D, vertexDataStride, vertexDataOffsPos,
            vertexDataOffsCorruptFlag, triangleVtxIdx_D, volume_D,
            externalForcesScl_D, triangleCnt, minDispl, isoval);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'FindCorruptTriangles_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
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
 * @param[in] dataArrSize The stride of the vertex data buffer
 */
__global__ void InitExternalForceScl_D (
        float *arr_D,
        float *volume_D,
        float *vertexPos_D,
        uint nElements,
        float isoval,
        uint dataArrOffs,
        uint dataArrSize) {

    const uint idx = GetThreadIndex();

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
        arr_D[idx] = 1.0;
    } else {
        arr_D[idx] = -1.0;
    }
}


/*
 * InitExternalForceScl
 */
extern "C"
cudaError_t InitExternalForceScl(
        float *arr_D,
        float *volume_D,
        float *vertexPos_D,
        uint nElements,
        float isoval,
        uint dataArrOffs,
        uint dataArrSize) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    InitExternalForceScl_D <<< Grid(nElements, 256), 256 >>> (
            arr_D,
            volume_D,
            vertexPos_D,
            nElements,
            isoval,
            dataArrOffs,
            dataArrSize);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'InitExternalForceScl_D':                %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
}


/**
 * Exchanges float vectors with three components from one data buffer to
 * another.
 *
 * @param[out] dataOut_D     The output buffer (device memory)
 * @param[in]  dataOutStride The stride of the output buffer
 * @param[in]  dataOutOffs   The data offset for the output buffer
 * @param[in]  dataIn_D      The input buffer (device memory)
 * @param[in]  dataInStride  The stride of the input buffer
 * @param[in]  dataInOffs    The data offset for the input buffer
 */
__global__ void InitVertexData3_D (
        float *dataOut_D,
        uint dataOutStride,
        uint dataOutOffs,
        float *dataIn_D,
        uint dataInStride,
        uint dataInOffs,
        uint vertexCnt) {

    const uint idx = GetThreadIndex();
    if (idx >= vertexCnt) {
        return;
    }
    const uint idxOut = dataOutStride*idx+dataOutOffs;
    const uint idxIn = dataInStride*idx+dataInOffs;
    dataOut_D[idxOut+0] = dataIn_D[idxIn+0];
    dataOut_D[idxOut+1] = dataIn_D[idxIn+1];
    dataOut_D[idxOut+2] = dataIn_D[idxIn+2];
}


/*
 * InitVertexData3
 */
extern "C"
cudaError_t InitVertexData3(
        float *dataOut_D,
        uint dataOutStride,
        uint dataOutOffs,
        float *dataIn_D,
        uint dataInStride,
        uint dataInOffs,
        uint vertexCnt) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    InitVertexData3_D <<< Grid(vertexCnt, 256), 256 >>> (
            dataOut_D,
            dataOutStride,
            dataOutOffs,
            dataIn_D,
            dataInStride,
            dataInOffs,
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'InitVertexData3_D':                     %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
}


// Initialize constant values. This needs to be done separately for every file,
// since constant variables are only visible at file scope
extern "C"
cudaError InitVolume_surface_mapping(uint3 gridSize, float3 org, float3 delta) {
    cudaMemcpyToSymbol(gridSize_D, &gridSize, sizeof(uint3));
    cudaMemcpyToSymbol(gridOrg_D, &org, sizeof(float3));
    cudaMemcpyToSymbol(gridDelta_D, &delta, sizeof(float3));
    return cudaGetLastError();
}


////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// The following set of functions calculates both external force and the      //
// updated vertex position in one kernel. This allows to perform              //
// several iterations in one kernel call, which reduces access to global      //
// memory. However when using tricubic interpolation register spilling        //
// might occur.                                                               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


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
__global__ void UpdateVertexPositionTricubic_D(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        uint vertexCount,
        float externalWeight,
        float forcesScl,
        float stiffness,
        float isoval,
        float minDispl,
        uint dataArrOffsPos,
        uint dataArrOffsNormal,
        uint dataArrSize) {

    const uint idx = GetThreadIndex();
    if (idx >= vertexCount) {
        return;
    }

    const uint posBaseIdx = dataArrSize*idx+dataArrOffsPos;

    /* Retrieve stuff from global device memory */

    // Get initial position from global device memory
    float3 pos = make_float3(
            vertexPosMapped_D[posBaseIdx+0],
            vertexPosMapped_D[posBaseIdx+1],
            vertexPosMapped_D[posBaseIdx+2]);

    // Get initial scale factor for external forces
    float externalForcesScl = vertexExternalForcesScl_D[idx];

    // Get neighbor indices
    int nIdx[MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS];
    for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
        nIdx[i] = vertexNeighbours_D[MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS*idx+i];
    }

    /* Update position */

    for (int i = 0; i < UPDATE_VTX_POS_ITERATIONS_PER_KERNEL; ++i) {

        if (fabs(externalForcesScl) < minDispl) {
            vertexExternalForcesScl_D[idx] = externalForcesScl;
            return;
        }

        const float sampleDens = ::SampleFieldAtPosTricub_D<float>(pos, targetVolume_D);

        // Switch sign and scale down if necessary
        bool negative = externalForcesScl < 0;
        bool outside = sampleDens <= isoval;
        int switchSign = int((negative && outside)||(!negative && !outside));
        externalForcesScl = externalForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
        externalForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));

        // Sample gradient by trilinear interpolation
        float4 externalForceTmp = ::SampleFieldAtPosTricub_D(pos, gradient_D);
        float3 externalForce;
        externalForce.x = externalForceTmp.x;
        externalForce.y = externalForceTmp.y;
        externalForce.z = externalForceTmp.z;

        externalForce = safeNormalize(externalForce);
        externalForce *= forcesScl*externalForcesScl*externalWeight;

        float3 normal = make_float3(
                vertexPosMapped_D[dataArrSize*idx+dataArrOffsNormal+0],
                vertexPosMapped_D[dataArrSize*idx+dataArrOffsNormal+1],
                vertexPosMapped_D[dataArrSize*idx+dataArrOffsNormal+2]);

        if (outside) {
            if (dot(normalize(normal), normalize(externalForce)) > 0.0) {
                externalForce *= 0.0;
                externalWeight = 0.0;
            }
        } else {
            if (dot(normalize(normal), normalize(externalForce)) < 0.0) {
                externalForce *= 0.0;
                externalWeight = 0.0;
            }
        }

        // Calculate new position when using external forces only
        float3 vertexPosExternalOnly;
        vertexPosExternalOnly.x = pos.x + externalForce.x;
        vertexPosExternalOnly.y = pos.y + externalForce.y;
        vertexPosExternalOnly.z = pos.z + externalForce.z;

        // Calculate correction vector based on internal spring forces
        float3 correctionVec = make_float3(0.0, 0.0, 0.0);
        float3 displ;
        float activeNeighbourCnt = 0.00001f; // Prevent division by zero (this should never happen)
        for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
            int isIdxValid = int(nIdx[i] >= 0); // Check if idx != -1
            float3 posNeighbour;
            int tmpIdx = isIdxValid*nIdx[i]; // Map negative indices to 0
            posNeighbour.x = vertexPosMapped_D[dataArrSize*tmpIdx+dataArrOffsPos+0];
            posNeighbour.y = vertexPosMapped_D[dataArrSize*tmpIdx+dataArrOffsPos+1];
            posNeighbour.z = vertexPosMapped_D[dataArrSize*tmpIdx+dataArrOffsPos+2];
            displ = (posNeighbour - pos);
//            correctionVec += displ*stiffness*isIdxValid;
            correctionVec += displ*isIdxValid;
            activeNeighbourCnt += 1.0f*isIdxValid;
        }
        //correctionVec *= stiffness;
        correctionVec /= activeNeighbourCnt; // Represents internal force

//        normal = safeNormalize(normal);
//        float3 internalForce = correctionVec - dot(correctionVec, normal)*normal; // With projection
        float3 internalForce = correctionVec;                                       // Without projection

        laplacian_D[idx] = internalForce;
        __syncthreads();

        // Calculate correction vector based on internal spring forces
        float3 laplacian2 = make_float3(0.0, 0.0, 0.0);
        //activeNeighbourCnt = 0.00001f; // Prevent division by zero (this should never happen)
        for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
            int isIdxValid = int(nIdx[i] >= 0); // Check if idx != -1
            int tmpIdx = isIdxValid*nIdx[i]; // Map negative indices to 0
            laplacian2 += (laplacian_D[tmpIdx ]- correctionVec)*isIdxValid;
            //activeNeighbourCnt += 1.0f*isIdxValid;
        }
        laplacian2 /= activeNeighbourCnt; // Represents internal force


        // Project internal force onto surface
//        float3 crossTmp0 = cross(externalForce, correctionVec);
//        float3 crossTmp1 = cross(crossTmp0, externalForce);
//        float dotTmp = dot(crossTmp1, correctionVec);
//        dotTmp = clamp(dotTmp, 0.0, 1.0);

//        // Umbrella internal force
//        pos = vertexPosExternalOnly
//                + (1.0-externalWeight)*forcesScl*normalize(crossTmp1)*dotTmp;



        // Umbrella internal force
        pos = vertexPosExternalOnly + (1.0-externalWeight)*forcesScl*
                ((1.0 - stiffness)*internalForce - stiffness*laplacian2);

        vertexPosMapped_D[posBaseIdx+0] = pos.x;
        vertexPosMapped_D[posBaseIdx+1] = pos.y;
        vertexPosMapped_D[posBaseIdx+2] = pos.z;

        __syncthreads();
    }

    // Write external forces scale factor back to global device memory
    vertexExternalForcesScl_D[idx] = externalForcesScl;
}


/*
 * UpdateVertexPositionTricubic
 */
extern "C"
cudaError_t UpdateVertexPositionTricubic(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        uint vertexCount,
        float externalWeight,
        float forcesScl,
        float stiffness,
        float isoval,
        float minDispl,
        uint dataArrOffsPos,
        uint dataArrOffsNormal,
        uint dataArrSize) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif




    // Update position for all vertices
    UpdateVertexPositionTricubic_D <<< Grid(vertexCount, 256), 256 >>> (
            targetVolume_D,
            vertexPosMapped_D,
            vertexExternalForcesScl_D,
            vertexNeighbours_D,
            gradient_D,
            laplacian_D,
            vertexCount,
            externalWeight,
            forcesScl,
            stiffness,
            isoval,
            minDispl,
            dataArrOffsPos,
            dataArrOffsNormal,
            dataArrSize);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'UpdateVertexPositionTricubic_D':        %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
}


/**
 * Updates the positions of all vertices based on external and internal forces.
 * The external force is computed on the fly based on a the given volume.
 * Samples are aquired using trilinear interpolation.
 *
 * @param[in]      targetVolume_D         The volume the isosurface is extracted
 *                                        from
 * @param[in,out]  vertexPosMapped_D      The vertex data buffer
 * @param[in]      vertexExternalForcesScl_D The external force and scale factor
 *                                        for all vertices
 * @param[in]      vertexNeighbours_D     The neighbour indices of all vertices
 * @param[in]      gradient_D             Array with the gradient
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
 * @param[in]      dataArrSize            The stride of the vertex data buffer
 */
__global__ void UpdateVertexPositionTrilinear_D(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        uint vertexCount,
        float externalWeight,
        float forcesScl,
        float stiffness,
        float isoval,
        float minDispl,
        uint dataArrOffs,
        uint dataArrSize) {

    const uint idx = GetThreadIndex();
    if (idx >= vertexCount) {
        return;
    }

    const uint posBaseIdx = dataArrSize*idx+dataArrOffs;

    /* Retrieve stuff from global device memory */

    // Get initial position from global device memory
    float3 pos = make_float3(
            vertexPosMapped_D[posBaseIdx+0],
            vertexPosMapped_D[posBaseIdx+1],
            vertexPosMapped_D[posBaseIdx+2]);

    // Get initial scale factor for external forces
    float externalForcesScl = vertexExternalForcesScl_D[idx];

    // Get neighbor indices
    int nIdx[MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS];
    for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
        nIdx[i] = vertexNeighbours_D[MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS*idx+i];
    }

    /* Update position */

    for (int i = 0; i < UPDATE_VTX_POS_ITERATIONS_PER_KERNEL; ++i) {

        if (fabs(externalForcesScl) < minDispl) {
            vertexExternalForcesScl_D[idx] = externalForcesScl;
            return;
        }

        const float sampleDens = ::SampleFieldAtPosTrilin_D<float>(pos, targetVolume_D);

        // Switch sign and scale down if necessary
        bool negative = externalForcesScl < 0;
        bool outside = sampleDens <= isoval;
        int switchSign = int((negative && outside)||(!negative && !outside));
        externalForcesScl = externalForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
        externalForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));

        // Sample gradient by trilinear interpolation
        float4 externalForceTmp = ::SampleFieldAtPosTrilin_D(pos, gradient_D);
        float3 externalForce;
        externalForce.x = externalForceTmp.x;
        externalForce.y = externalForceTmp.y;
        externalForce.z = externalForceTmp.z;

        externalForce = safeNormalize(externalForce);

        // Calculate new position when using external forces only
        float3 vertexPosExternalOnly;
        vertexPosExternalOnly.x = pos.x +
                forcesScl*externalForcesScl*externalWeight*externalForce.x;
        vertexPosExternalOnly.y = pos.y +
                forcesScl*externalForcesScl*externalWeight*externalForce.y;
        vertexPosExternalOnly.z = pos.z +
                forcesScl*externalForcesScl*externalWeight*externalForce.z;

        // Calculate correction vector based on internal spring forces
        float3 correctionVec = make_float3(0.0, 0.0, 0.0);
        float3 displ;
        float activeNeighbourCnt = 0.00001f; // Prevent division by zero (this should never happen)
        for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
            int isIdxValid = int(nIdx[i] >= 0); // Check if idx != -1
            float3 posNeighbour;
            int tmpIdx = isIdxValid*nIdx[i]; // Map negative indices to 0
            posNeighbour.x = vertexPosMapped_D[dataArrSize*tmpIdx+dataArrOffs+0];
            posNeighbour.y = vertexPosMapped_D[dataArrSize*tmpIdx+dataArrOffs+1];
            posNeighbour.z = vertexPosMapped_D[dataArrSize*tmpIdx+dataArrOffs+2];
            displ = (posNeighbour - pos);
            //correctionVec += displ*stiffness*isIdxValid;
            correctionVec += displ*isIdxValid;
            activeNeighbourCnt += 1.0f*isIdxValid;
        }
        correctionVec /= activeNeighbourCnt; // Represents internal force

//        float3 normal = externalForce;
//        normal = safeNormalize(normal);
//        float3 internalForce = correctionVec - dot(correctionVec, normal)*normal;

        //laplacian_D[idx] = internalForce;
        laplacian_D[idx] = correctionVec;
        __syncthreads();

        // Calculate correction vector based on internal spring forces
        float3 laplacian2 = make_float3(0.0, 0.0, 0.0);
        //activeNeighbourCnt = 0.00001f; // Prevent division by zero (this should never happen)
        for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
            int isIdxValid = int(nIdx[i] >= 0); // Check if idx != -1
            int tmpIdx = isIdxValid*nIdx[i]; // Map negative indices to 0
            laplacian2 += (laplacian_D[tmpIdx ]- correctionVec)*isIdxValid;
            //activeNeighbourCnt += 1.0f*isIdxValid;
        }
        laplacian2 /= activeNeighbourCnt; // Represents internal force

        float3 normal = externalForce;
        normal = safeNormalize(normal);
        float3 internalForce = (1.0 - stiffness)*correctionVec - stiffness*laplacian2;
        internalForce = internalForce - dot(internalForce, normal)*normal;


        // Project internal force onto surface
//        float3 crossTmp0 = cross(externalForce, correctionVec);
//        float3 crossTmp1 = cross(crossTmp0, externalForce);
//        float dotTmp = dot(crossTmp1, correctionVec);
//        dotTmp = clamp(dotTmp, 0.0, 1.0);

//        // Umbrella internal force
//        pos = vertexPosExternalOnly
//                + (1.0-externalWeight)*forcesScl*normalize(crossTmp1)*dotTmp;



        // Umbrella internal force
        pos = vertexPosExternalOnly + (1.0-externalWeight)*forcesScl*
                internalForce;

        vertexPosMapped_D[posBaseIdx+0] = pos.x;
        vertexPosMapped_D[posBaseIdx+1] = pos.y;
        vertexPosMapped_D[posBaseIdx+2] = pos.z;

        __syncthreads();
    }

    // Write external forces scale factor back to global device memory
    vertexExternalForcesScl_D[idx] = externalForcesScl;
}


/*
 * UpdateVertexPositionTrilinear
 */
extern "C"
cudaError_t UpdateVertexPositionTrilinear(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        uint vertexCount,
        float externalWeight,
        float forcesScl,
        float stiffness,
        float isoval,
        float minDispl,
        uint dataArrOffs,
        uint dataArrSize) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Update position for all vertices
    UpdateVertexPositionTrilinear_D <<< Grid(vertexCount, 256), 256 >>> (
            targetVolume_D,
            vertexPosMapped_D,
            vertexExternalForcesScl_D,
            vertexNeighbours_D,
            gradient_D,
            laplacian_D,
            vertexCount,
            externalWeight,
            forcesScl,
            stiffness,
            isoval,
            minDispl,
            dataArrOffs,
            dataArrSize);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'UpdateVertexPositionTrilinear_D':       %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
}


// DEBUG versions that keep track of displacement length

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
 * @param[in]      dataArrSize            The stride of the vertex data buffer
 */
__global__ void UpdateVertexPositionTricubicWithDispl_D(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        float *displLen_D,
        uint vertexCount,
        float externalWeight,
        float forcesScl,
        float stiffness,
        float isoval,
        float minDispl,
        uint dataArrOffs,
        uint dataArrSize) {

    const uint idx = GetThreadIndex();
    if (idx >= vertexCount) {
        return;
    }

    //if (displLen_D[idx] <= minDispl) return;

    const uint posBaseIdx = dataArrSize*idx+dataArrOffs;

    /* Retrieve stuff from global device memory */

    // Get initial position from global device memory
    float3 pos = make_float3(
            vertexPosMapped_D[posBaseIdx+0],
            vertexPosMapped_D[posBaseIdx+1],
            vertexPosMapped_D[posBaseIdx+2]);
    float3 posOld = pos;

    // Get initial scale factor for external forces
    float externalForcesScl = vertexExternalForcesScl_D[idx];

    // Get neighbor indices
    int nIdx[MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS];
    for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
        nIdx[i] = vertexNeighbours_D[MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS*idx+i];
    }

    /* Update position */

    for (int i = 0; i < UPDATE_VTX_POS_ITERATIONS_PER_KERNEL; ++i) {

//        if (fabs(externalForcesScl) < minDispl) {
//            vertexExternalForcesScl_D[idx] = externalForcesScl;
//            displLen_D[idx] = 0.0f;
//            return;
//        }

        const float sampleDens = ::SampleFieldAtPosTricub_D<float>(pos, targetVolume_D);

        // Switch sign and scale down if necessary
        bool negative = externalForcesScl < 0;
        bool outside = sampleDens <= isoval;
        int switchSign = int((negative && outside)||(!negative && !outside));
        externalForcesScl = externalForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
        externalForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));

        // Sample gradient by trilinear interpolation
        float4 externalForceTmp = ::SampleFieldAtPosTricub_D(pos, gradient_D);
        float3 externalForce;
        externalForce.x = externalForceTmp.x;
        externalForce.y = externalForceTmp.y;
        externalForce.z = externalForceTmp.z;

        externalForce = safeNormalize(externalForce);

        // Calculate new position when using external forces only
        float3 vertexPosExternalOnly;
        vertexPosExternalOnly.x = pos.x +
                forcesScl*externalForcesScl*externalWeight*externalForce.x;
        vertexPosExternalOnly.y = pos.y +
                forcesScl*externalForcesScl*externalWeight*externalForce.y;
        vertexPosExternalOnly.z = pos.z +
                forcesScl*externalForcesScl*externalWeight*externalForce.z;

        // Calculate correction vector based on internal spring forces
        float3 correctionVec = make_float3(0.0, 0.0, 0.0);
        float3 displ;
        float activeNeighbourCnt = 0.00001f; // Prevent division by zero (this should never happen)
        for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
            int isIdxValid = int(nIdx[i] >= 0); // Check if idx != -1
            float3 posNeighbour;
            int tmpIdx = isIdxValid*nIdx[i]; // Map negative indices to 0
            posNeighbour.x = vertexPosMapped_D[dataArrSize*tmpIdx+dataArrOffs+0];
            posNeighbour.y = vertexPosMapped_D[dataArrSize*tmpIdx+dataArrOffs+1];
            posNeighbour.z = vertexPosMapped_D[dataArrSize*tmpIdx+dataArrOffs+2];
            displ = (posNeighbour - pos);
            //correctionVec += displ*stiffness*isIdxValid;
            correctionVec += displ*isIdxValid;
            activeNeighbourCnt += 1.0f*isIdxValid;
        }
        correctionVec /= activeNeighbourCnt; // Represents internal force

        float3 normal = externalForce;
        normal = safeNormalize(normal);
        float3 internalForce = correctionVec - dot(correctionVec, normal)*normal;

        laplacian_D[idx] = internalForce;
        __syncthreads();

        // Calculate correction vector based on internal spring forces
        float3 laplacian2 = make_float3(0.0, 0.0, 0.0);
        //activeNeighbourCnt = 0.00001f; // Prevent division by zero (this should never happen)
        for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
            int isIdxValid = int(nIdx[i] >= 0); // Check if idx != -1
            int tmpIdx = isIdxValid*nIdx[i]; // Map negative indices to 0
            laplacian2 += (laplacian_D[tmpIdx ]- correctionVec)*isIdxValid;
            //activeNeighbourCnt += 1.0f*isIdxValid;
        }
        laplacian2 /= activeNeighbourCnt; // Represents internal force

        // Umbrella internal force
        pos = vertexPosExternalOnly + (1.0-externalWeight)*forcesScl*
                ((1.0 - stiffness)*internalForce - stiffness*laplacian2);

        vertexPosMapped_D[posBaseIdx+0] = pos.x;
        vertexPosMapped_D[posBaseIdx+1] = pos.y;
        vertexPosMapped_D[posBaseIdx+2] = pos.z;

        __syncthreads();
    }

    // Write external forces scale factor back to global device memory
    vertexExternalForcesScl_D[idx] = externalForcesScl;
//    if (length(pos-posOld) < minDispl) displLen_D[idx] = 0.0f;
//    else displLen_D[idx] = length(pos-posOld);
    displLen_D[idx] = length(pos-posOld);
}


/*
 * UpdateVertexPositionTricubic
 */
extern "C"
cudaError_t UpdateVertexPositionTricubicWithDispl(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        float *displLen_D,
        uint vertexCount,
        float externalWeight,
        float forcesScl,
        float stiffness,
        float isoval,
        float minDispl,
        uint dataArrOffs,
        uint dataArrSize) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Update position for all vertices
    UpdateVertexPositionTricubicWithDispl_D <<< Grid(vertexCount, 256), 256 >>> (
            targetVolume_D,
            vertexPosMapped_D,
            vertexExternalForcesScl_D,
            vertexNeighbours_D,
            gradient_D,
            laplacian_D,
            displLen_D,
            vertexCount,
            externalWeight,
            forcesScl,
            stiffness,
            isoval,
            minDispl,
            dataArrOffs,
            dataArrSize);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'UpdateVertexPositionTricubic_D':        %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
}


/**
 * Updates the positions of all vertices based on external and internal forces.
 * The external force is computed on the fly based on a the given volume.
 * Samples are aquired using trilinear interpolation.
 *
 * @param[in]      targetVolume_D         The volume the isosurface is extracted
 *                                        from
 * @param[in,out]  vertexPosMapped_D      The vertex data buffer
 * @param[in]      vertexExternalForcesScl_D The external force and scale factor
 *                                        for all vertices
 * @param[in]      vertexNeighbours_D     The neighbour indices of all vertices
 * @param[in]      gradient_D             Array with the gradient
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
 * @param[in]      dataArrSize            The stride of the vertex data buffer
 */
__global__ void UpdateVertexPositionTrilinearWithDispl_D(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        float *displLen_D,
        uint vertexCount,
        float externalWeight,
        float forcesScl,
        float stiffness,
        float isoval,
        float minDispl,
        uint dataArrOffs,
        uint dataArrSize) {

    const uint idx = GetThreadIndex();
    if (idx >= vertexCount) {
        return;
    }

    //if (displLen_D[idx] <= minDispl) return;

    const uint posBaseIdx = dataArrSize*idx+dataArrOffs;

    /* Retrieve stuff from global device memory */

    // Get initial position from global device memory
    float3 pos = make_float3(
            vertexPosMapped_D[posBaseIdx+0],
            vertexPosMapped_D[posBaseIdx+1],
            vertexPosMapped_D[posBaseIdx+2]);
    float3 posOld = pos;

    // Get initial scale factor for external forces
    float externalForcesScl = vertexExternalForcesScl_D[idx];

    // Get neighbor indices
    int nIdx[MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS];
    for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
        nIdx[i] = vertexNeighbours_D[MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS*idx+i];
    }

    /* Update position */

    for (int i = 0; i < UPDATE_VTX_POS_ITERATIONS_PER_KERNEL; ++i) {

//        if (fabs(externalForcesScl) < minDispl) {
//            vertexExternalForcesScl_D[idx] = externalForcesScl;
//            displLen_D[idx] = 0.0f;
//            return;
//        }

        const float sampleDens = ::SampleFieldAtPosTrilin_D<float>(pos, targetVolume_D);

        // Switch sign and scale down if necessary
        bool negative = externalForcesScl < 0;
        bool outside = sampleDens <= isoval;
        int switchSign = int((negative && outside)||(!negative && !outside));
        externalForcesScl = externalForcesScl*(1.0*(1-switchSign) - 1.0*switchSign);
        externalForcesScl *= (1.0*(1-switchSign) + 0.5*(switchSign));

        // Sample gradient by trilinear interpolation
        float4 externalForceTmp = ::SampleFieldAtPosTrilin_D(pos, gradient_D);
        float3 externalForce;
        externalForce.x = externalForceTmp.x;
        externalForce.y = externalForceTmp.y;
        externalForce.z = externalForceTmp.z;

        externalForce = safeNormalize(externalForce);

        // Calculate new position when using external forces only
        float3 vertexPosExternalOnly;
        vertexPosExternalOnly.x = pos.x +
                forcesScl*externalForcesScl*externalWeight*externalForce.x;
        vertexPosExternalOnly.y = pos.y +
                forcesScl*externalForcesScl*externalWeight*externalForce.y;
        vertexPosExternalOnly.z = pos.z +
                forcesScl*externalForcesScl*externalWeight*externalForce.z;

        // Calculate correction vector based on internal spring forces
        float3 correctionVec = make_float3(0.0, 0.0, 0.0);
        float3 displ;
        float activeNeighbourCnt = 0.00001f; // Prevent division by zero (this should never happen)
        for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
            int isIdxValid = int(nIdx[i] >= 0); // Check if idx != -1
            float3 posNeighbour;
            int tmpIdx = isIdxValid*nIdx[i]; // Map negative indices to 0
            posNeighbour.x = vertexPosMapped_D[dataArrSize*tmpIdx+dataArrOffs+0];
            posNeighbour.y = vertexPosMapped_D[dataArrSize*tmpIdx+dataArrOffs+1];
            posNeighbour.z = vertexPosMapped_D[dataArrSize*tmpIdx+dataArrOffs+2];
            displ = (posNeighbour - pos);
            //correctionVec += displ*stiffness*isIdxValid;
            correctionVec += displ*isIdxValid;
            activeNeighbourCnt += 1.0f*isIdxValid;
        }
        correctionVec /= activeNeighbourCnt; // Represents internal force

        float3 normal = externalForce;
        normal = safeNormalize(normal);
        float3 internalForce = correctionVec - dot(correctionVec, normal)*normal;

        laplacian_D[idx] = internalForce;
        __syncthreads();

        // Calculate correction vector based on internal spring forces
        float3 laplacian2 = make_float3(0.0, 0.0, 0.0);
        //activeNeighbourCnt = 0.00001f; // Prevent division by zero (this should never happen)
        for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
            int isIdxValid = int(nIdx[i] >= 0); // Check if idx != -1
            int tmpIdx = isIdxValid*nIdx[i]; // Map negative indices to 0
            laplacian2 += (laplacian_D[tmpIdx ]- correctionVec)*isIdxValid;
            //activeNeighbourCnt += 1.0f*isIdxValid;
        }
        laplacian2 /= activeNeighbourCnt; // Represents internal force

        // Umbrella internal force
        pos = vertexPosExternalOnly + (1.0-externalWeight)*forcesScl*
                ((1.0 - stiffness)*internalForce - stiffness*laplacian2);

        vertexPosMapped_D[posBaseIdx+0] = pos.x;
        vertexPosMapped_D[posBaseIdx+1] = pos.y;
        vertexPosMapped_D[posBaseIdx+2] = pos.z;

        __syncthreads();
    }

    // Write external forces scale factor back to global device memory
    vertexExternalForcesScl_D[idx] = externalForcesScl;
//    if (length(pos-posOld) < minDispl) displLen_D[idx] = 0.0f;
//    else displLen_D[idx] = length(pos-posOld);
    displLen_D[idx] = length(pos-posOld);
}


/*
 * UpdateVertexPositionTrilinear
 */
extern "C"
cudaError_t UpdateVertexPositionTrilinearWithDispl(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        float *displLen_D,
        uint vertexCount,
        float externalWeight,
        float forcesScl,
        float stiffness,
        float isoval,
        float minDispl,
        uint dataArrOffs,
        uint dataArrSize) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Update position for all vertices
    UpdateVertexPositionTrilinearWithDispl_D <<< Grid(vertexCount, 256), 256 >>> (
            targetVolume_D,
            vertexPosMapped_D,
            vertexExternalForcesScl_D,
            vertexNeighbours_D,
            gradient_D,
            laplacian_D,
            displLen_D,
            vertexCount,
            externalWeight,
            forcesScl,
            stiffness,
            isoval,
            minDispl,
            dataArrOffs,
            dataArrSize);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'UpdateVertexPositionTrilinear_D':       %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
}
