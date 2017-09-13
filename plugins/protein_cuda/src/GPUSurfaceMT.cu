//
// GPUSurfaceMT.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 17, 2013
// Author     : scharnkn
//

#include "stdafx.h"
#include "GPUSurfaceMT.h"

#include "cuda_error_check.h"
#include "HostArr.h"
#include "sort_triangles.cuh"
#include "CUDAGrid.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include "vislib/graphics/gl/IncludeAllGL.h"
#define WGL_NV_gpu_affinity
#include <cuda_gl_interop.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

using namespace megamol;
using namespace megamol::protein_cuda;



void GPUSurfaceMT::ComputeMinMaxCoords(float3 &minC, float3 &maxC) {

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return;
    }

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0))) {
        return;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,             // The size of the accessible data
            this->vertexDataResource))) {                   // The mapped resource
        return;
    }

    HostArr<float> vertexBuffer;
    vertexBuffer.Validate(this->vertexCnt*this->vertexDataStride);
    CudaSafeCall(cudaMemcpy(vertexBuffer.Peek(), vboPt, vboSize, cudaMemcpyDeviceToHost));

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
        return;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return;
    }

    minC.x = vertexBuffer.Peek()[0];
    minC.y = vertexBuffer.Peek()[1];
    minC.z = vertexBuffer.Peek()[2];
    maxC.x = vertexBuffer.Peek()[0];
    maxC.y = vertexBuffer.Peek()[1];
    maxC.z = vertexBuffer.Peek()[2];
    for (int i = 0; i < this->vertexCnt; ++i) {
        minC.x = std::min(vertexBuffer.Peek()[9*i + 0], minC.x);
        minC.y = std::min(vertexBuffer.Peek()[9*i + 1], minC.y);
        minC.z = std::min(vertexBuffer.Peek()[9*i + 2], minC.z);
        maxC.x = std::max(vertexBuffer.Peek()[9*i + 0], maxC.x);
        maxC.y = std::max(vertexBuffer.Peek()[9*i + 1], maxC.y);
        maxC.z = std::max(vertexBuffer.Peek()[9*i + 2], maxC.z);
//        printf("min %f %f %f, max %f %f %f\n", minC.x, minC.y, minC.z,
//                maxC.x, maxC.y, maxC.z);
    }

    vertexBuffer.Release();
}

//#define USE_TIMER // Toggle performance measurements

/**
 * @return Returns the thread index based on the current CUDA grid dimensions
 */
inline __device__ uint getThreadIdx() {
    return __umul24(__umul24(blockIdx.y, gridDim.x) + blockIdx.x, blockDim.x) +
            threadIdx.x;
}

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

__device__ __shared__ char tetrahedronTriangles_S[16][6];
__device__ __constant__ char tetrahedronTriangles[16][6] = {
    {-1, -1, -1, -1, -1, -1}, // #0
    { 0,  3,  2, -1, -1, -1}, // #1
    { 0,  1,  4, -1, -1, -1}, // #2
    { 1,  4,  2,  2,  4,  3}, // #3
    { 1,  2,  5, -1, -1, -1}, // #4
    { 0,  3,  5,  0,  5,  1}, // #5
    { 0,  2,  5,  0,  5,  4}, // #6
    { 5,  4,  3, -1, -1, -1}, // #7
    { 3,  4,  5, -1, -1, -1}, // #8
    { 4,  5,  0,  5,  2,  0}, // #9
    { 1,  5,  0,  5,  3,  0}, // #10
    { 5,  2,  1, -1, -1, -1}, // #11
    { 3,  4,  2,  2,  4,  1}, // #12
    { 4,  1,  0, -1, -1, -1}, // #13
    { 2,  3,  0, -1, -1, -1}, // #14
    {-1, -1, -1, -1, -1, -1}  // #15
};
inline __device__ void LoadTetrahedronTrianglesToSharedMemory() {
    // Load tetrahedron triangle table into shared memory.
    if (threadIdx.x < 16) {
        for (int i = 0; i < 6; ++i) {
            tetrahedronTriangles_S[threadIdx.x][i] = tetrahedronTriangles[threadIdx.x][i];
        }
    }
}

__device__ __shared__ unsigned char tetrahedronEdgeFlags_S[16];
__device__ __constant__ unsigned char tetrahedronEdgeFlags[16] = {
    0x00, 0x0d, 0x13, 0x1e, 0x26, 0x2b, 0x35, 0x38,
    0x38, 0x35, 0x2b, 0x26, 0x1e, 0x13, 0x0d, 0x00
};
__device__ __shared__ char tetrahedronEdgeConnections_S[6][2];
__device__ __constant__ char tetrahedronEdgeConnections[6][2] = {
    {0, 1},  {1, 2},  {2, 0},  {0, 3},  {1, 3},  {2, 3}
};
inline __device__
void LoadTetrahedronEdgeFlagsAndConnectionsToSharedMemory() {
    // Load tetrahedron edge flags into shared memory.
    if (threadIdx.x < 16) {
        tetrahedronEdgeFlags_S[threadIdx.x] = tetrahedronEdgeFlags[threadIdx.x];
    }
    // Load tetrahedron edge connection table into shared memory.
    if (threadIdx.x < 6) {
        tetrahedronEdgeConnections_S[threadIdx.x][0] = tetrahedronEdgeConnections[threadIdx.x][0];
        tetrahedronEdgeConnections_S[threadIdx.x][1] = tetrahedronEdgeConnections[threadIdx.x][1];
    }
}

// [tetrahedronIdx][vtx][tetrahedronEdge]
// -1 indicates undefined values
__device__ __shared__ int VertexIdxPerTetrahedronIdx_S[6][2][2];
__device__ __constant__ int VertexIdxPerTetrahedronIdx[6][2][2] = {
        {{ 0,  2}, { 6,  3}}, // Tetrahedron #0
        {{ 4,  2}, {-1, -1}}, // Tetrahedron #1
        {{ 1,  2}, {-1, -1}}, // Tetrahedron #2
        {{ 5,  2}, {-1, -1}}, // Tetrahedron #3
        {{ 2,  2}, {-1, -1}}, // Tetrahedron #4
        {{ 3,  2}, {-1, -1}}, // Tetrahedron #5
};
inline __device__ void LoadVertexIdxPerTetrahedronIdxToSharedMemory() {
    // Load cube vertex offsets into shared memory
    if (threadIdx.x < 6) {
        VertexIdxPerTetrahedronIdx_S[threadIdx.x][0][0] = VertexIdxPerTetrahedronIdx[threadIdx.x][0][0];
        VertexIdxPerTetrahedronIdx_S[threadIdx.x][0][1] = VertexIdxPerTetrahedronIdx[threadIdx.x][0][1];
        VertexIdxPerTetrahedronIdx_S[threadIdx.x][1][0] = VertexIdxPerTetrahedronIdx[threadIdx.x][1][0];
        VertexIdxPerTetrahedronIdx_S[threadIdx.x][1][1] = VertexIdxPerTetrahedronIdx[threadIdx.x][1][1];
    }
}


__device__ __shared__ uint tetrahedronsInACube_S[6][4];
__device__ __constant__ uint tetrahedronsInACube[6][4] = {
    {0, 5, 1, 6},
    {0, 1, 2, 6},
    {0, 2, 3, 6},
    {0, 3, 7, 6},
    {0, 7, 4, 6},
    {0, 4, 5, 6}
};
inline __device__ void LoadTetrahedronsInACubeToSharedMemory() {
    // Load cube vertex offsets into shared memory
    if (threadIdx.x < 6) {
        tetrahedronsInACube_S[threadIdx.x][0] = tetrahedronsInACube[threadIdx.x][0];
        tetrahedronsInACube_S[threadIdx.x][1] = tetrahedronsInACube[threadIdx.x][1];
        tetrahedronsInACube_S[threadIdx.x][2] = tetrahedronsInACube[threadIdx.x][2];
        tetrahedronsInACube_S[threadIdx.x][3] = tetrahedronsInACube[threadIdx.x][3];
    }
}


__device__ __shared__ uint cubeVertexOffsets_S[8][3];
__device__ __constant__ uint cubeVertexOffsets[8][3] = {
    {0, 0, 0},
    {1, 0, 0},
    {1, 1, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 0, 1},
    {1, 1, 1},
    {0, 1, 1}
};
inline __device__ void LoadCubeOffsetsToSharedMemory() {
    // Load cube vertex offsets into shared memory
//    if (threadIdx.x < 32) {
//        const uint idx0 = clamp(int(threadIdx.x/8), 0, 7);
//        const uint idx1 = threadIdx.x%3;
//        cubeVertexOffsets_S[idx0][idx1] = cubeVertexOffsets[idx0][idx1];
//    }
    if (threadIdx.x < 8) {
        cubeVertexOffsets_S[threadIdx.x][0] = cubeVertexOffsets[threadIdx.x][0];
        cubeVertexOffsets_S[threadIdx.x][1] = cubeVertexOffsets[threadIdx.x][1];
        cubeVertexOffsets_S[threadIdx.x][2] = cubeVertexOffsets[threadIdx.x][2];
    }
}

__device__ __shared__ uint tetrahedronVertexCount_S[16];
__device__ __constant__ uint tetrahedronVertexCount[16] = {
    0, 3, 3, 6, 3, 6, 6, 3,
    3, 6, 6, 3, 6, 3, 3, 0
};
inline __device__ void LoadTetrahedronVertexCountToSharedMemory() {
    // Load tetrahedron vertex count into shared memory.
    if (threadIdx.x < 16) {
        tetrahedronVertexCount_S[threadIdx.x] = tetrahedronVertexCount[threadIdx.x];
    }
}


// Maps the index based on the tetrahedron index and the edge
// index to a global vertex index based on cubeIdx*7 + localIdx. The layout is
// {cubeIdxOffs.x, cubeIdxOffs.y, cubeIdxOffs.z, localIdx}
__shared__ __device__ int TetrahedronEdgeVertexIdxOffset_S[6][6][4];
__constant__ __device__ int TetrahedronEdgeVertexIdxOffset[6][6][4] = {
    {{0, 0, 0, 3}, {1, 0, 0, 2}, {0, 0, 0, 0}, {0, 0, 0, 6}, {1, 0, 1, 1}, {1, 0, 0, 5}}, // Tetrahedron #0
    {{0, 0, 0, 0}, {1, 0, 0, 1}, {0, 0, 0, 4}, {0, 0, 0, 6}, {1, 0, 0, 5}, {1, 1, 0, 2}}, // Tetrahedron #1
    {{0, 0, 0, 4}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 6}, {1, 1, 0, 2}, {0, 1, 0, 3}}, // Tetrahedron #2
    {{0, 0, 0, 1}, {0, 1, 0, 2}, {0, 0, 0, 5}, {0, 0, 0, 6}, {0, 1, 0, 3}, {0, 1, 1, 0}}, // Tetrahedron #3
    {{0, 0, 0, 5}, {0, 0, 1, 1}, {0, 0, 0, 2}, {0, 0, 0, 6}, {0, 1, 1, 0}, {0, 0, 1, 4}}, // Tetrahedron #4
    {{0, 0, 0, 2}, {0, 0, 1, 0}, {0, 0, 0, 3}, {0, 0, 0, 6}, {0, 0, 1, 4}, {1, 0, 1, 1}}  // Tetrahedron #5
};
__device__ void LoadTetrahedronEdgeVertexIdxOffsetToSharedMemory() {
//    if (threadIdx.x < 6) {
//        for (int i = 0; i < 6; ++i) {
//            TetrahedronEdgeVertexIdxOffset_S[threadIdx.x][i][0] = TetrahedronEdgeVertexIdxOffset[threadIdx.x][i][0];
//            TetrahedronEdgeVertexIdxOffset_S[threadIdx.x][i][1] = TetrahedronEdgeVertexIdxOffset[threadIdx.x][i][1];
//            TetrahedronEdgeVertexIdxOffset_S[threadIdx.x][i][2] = TetrahedronEdgeVertexIdxOffset[threadIdx.x][i][2];
//            TetrahedronEdgeVertexIdxOffset_S[threadIdx.x][i][3] = TetrahedronEdgeVertexIdxOffset[threadIdx.x][i][3];
//        }
//    }
    if (threadIdx.x < 64) {
        const uint idx0 = clamp(int(threadIdx.x/6), 0, 5);
        const uint idx1 = threadIdx.x%6;
        TetrahedronEdgeVertexIdxOffset_S[idx0][idx1][0] = TetrahedronEdgeVertexIdxOffset[idx0][idx1][0];
        TetrahedronEdgeVertexIdxOffset_S[idx0][idx1][1] = TetrahedronEdgeVertexIdxOffset[idx0][idx1][1];
        TetrahedronEdgeVertexIdxOffset_S[idx0][idx1][2] = TetrahedronEdgeVertexIdxOffset[idx0][idx1][2];
        TetrahedronEdgeVertexIdxOffset_S[idx0][idx1][3] = TetrahedronEdgeVertexIdxOffset[idx0][idx1][3];
    }
}

// Describes connections inside the tetrahedron for every tetrahedron edge
// based on the tetrahedron edge flags (0-16)
__constant__ __device__ unsigned char TetrahedronEdgeConnections[16][6] = {
// edges   #0      #1      #2      #3      #4      #5
        {0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // vertices active 0000 #00 (= #15)
        {0x0C, 0x00, 0x09, 0x05, 0x00, 0x00}, // vertices active 0001 #01 (= #14)
        {0x12, 0x11, 0x00, 0x00, 0x03, 0x00}, // vertices active 0010 #02 (= #13)
        {0x00, 0x14, 0x1A, 0x14, 0x0E, 0x00}, // vertices active 0011 #03 (= #12)
        {0x00, 0x24, 0x22, 0x00, 0x00, 0x06}, // vertices active 0100 #04 (= #11)
        {0x2A, 0x21, 0x00, 0x21, 0x00, 0x0B}, // vertices active 0101 #05 (= #10)
        {0x34, 0x00, 0x21, 0x00, 0x21, 0x15}, // vertices active 0110 #06 (= #09)
        {0x00, 0x00, 0x00, 0x30, 0x28, 0x18}, // vertices active 0111 #07 (= #08)
        {0x00, 0x00, 0x00, 0x30, 0x28, 0x18}, // vertices active 1000 #08 (= #07)
        {0x34, 0x00, 0x21, 0x00, 0x21, 0x15}, // vertices active 1001 #09 (= #06)
        {0x2A, 0x21, 0x00, 0x21, 0x00, 0x0B}, // vertices active 1010 #10 (= #05)
        {0x00, 0x24, 0x22, 0x00, 0x00, 0x06}, // vertices active 1011 #11 (= #04)
        {0x00, 0x14, 0x1A, 0x14, 0x0E, 0x00}, // vertices active 1100 #12 (= #03)
        {0x12, 0x11, 0x00, 0x00, 0x03, 0x00}, // vertices active 1101 #13 (= #02)
        {0x0C, 0x00, 0x09, 0x05, 0x00, 0x00}, // vertices active 1110 #14 (= #01)
        {0x00, 0x00, 0x00, 0x00, 0x00, 0x00}  // vertices active 1111 #15 (= #00)
};

// Contains all neighbouring tetrahedrons of a vertex (v0-v6), defined by
// a global cube offset and a local tetrahedron index. Values beyond [-1, 1]
// indicate undefined values
__shared__ __device__ int VertexNeighbouringTetrahedrons_S[7][6][4];
__constant__ __device__ int VertexNeighbouringTetrahedrons[7][6][4] = {
        {{ 0,  0,  0,  0}, { 0,  0,  0,  1}, { 0, -1,  0,  2}, { 0, -1, -1,  3}, { 0, -1, -1,  4}, { 0,  0, -1,  5}}, // v0
        {{-1,  0, -1,  0}, {-1,  0,  0,  1}, { 0,  0,  0,  2}, { 0,  0,  0,  3}, { 0,  0, -1,  4}, {-1,  0, -1,  5}}, // v1
        {{-1,  0,  0,  0}, {-1, -1,  0,  1}, {-1, -1,  0,  2}, { 0, -1,  0,  3}, { 0,  0,  0,  4}, { 0,  0,  0,  5}}, // v2
        {{ 0,  0,  0,  0}, {99, 99, 99,  1}, { 0, -1,  0,  2}, { 0, -1,  0,  3}, {99, 99, 99,  4}, { 0,  0,  0,  5}}, // v3
        {{99, 99, 99,  0}, { 0,  0,  0,  1}, { 0,  0,  0,  2}, {99, 99, 99,  3}, { 0,  0, -1,  4}, { 0,  0, -1,  5}}, // v4
        {{-1,  0,  0,  0}, {-1,  0,  0,  1}, {99, 99, 99,  2}, { 0,  0,  0,  3}, { 0,  0,  0,  4}, {99, 99, 99,  5}}, // v5
        {{ 0,  0,  0,  0}, { 0,  0,  0,  1}, { 0,  0,  0,  2}, { 0,  0,  0,  3}, { 0,  0,  0,  4}, { 0,  0,  0,  5}}  // v6
};
__device__ void LoadVertexNeighbouringTetrahedronsToSharedMemory() {
//    if (threadIdx.x < 7) {
//        for (int i = 0; i < 6; ++i) {
//            VertexNeighbouringTetrahedrons_S[threadIdx.x][i][0] = VertexNeighbouringTetrahedrons[threadIdx.x][i][0];
//            VertexNeighbouringTetrahedrons_S[threadIdx.x][i][1] = VertexNeighbouringTetrahedrons[threadIdx.x][i][1];
//            VertexNeighbouringTetrahedrons_S[threadIdx.x][i][2] = VertexNeighbouringTetrahedrons[threadIdx.x][i][2];
//            VertexNeighbouringTetrahedrons_S[threadIdx.x][i][3] = VertexNeighbouringTetrahedrons[threadIdx.x][i][3];
//        }
//    }
    if (threadIdx.x < 64) {
        const uint idx0 = clamp(int(threadIdx.x/7), 0, 6);
        const uint idx1 = threadIdx.x%6;
        VertexNeighbouringTetrahedrons_S[idx0][idx1][0] = VertexNeighbouringTetrahedrons[idx0][idx1][0];
        VertexNeighbouringTetrahedrons_S[idx0][idx1][1] = VertexNeighbouringTetrahedrons[idx0][idx1][1];
        VertexNeighbouringTetrahedrons_S[idx0][idx1][2] = VertexNeighbouringTetrahedrons[idx0][idx1][2];
        VertexNeighbouringTetrahedrons_S[idx0][idx1][3] = VertexNeighbouringTetrahedrons[idx0][idx1][3];
    }
}

// Contains the edge index every vertex (v0-v6) has inside its adjacent
// tetrahedrons, -1 indicates undefined values
__constant__ __device__ int VertexNeighbouringTetrahedronsOwnEdgeIdx[7][6] = {
        { 2,  0,  1,  5,  4,  1}, // v0
        { 4,  1,  2,  0,  1,  5}, // v1
        { 1,  5,  4,  1,  2,  0}, // v2
        { 0, -1,  5,  4, -1,  2}, // v3
        {-1,  2,  0, -1,  5,  4}, // v4
        { 5,  4, -1,  2,  0, -1}, // v5
        { 3,  3,  3,  3,  3,  3}  // v6
};

// Defines the neighbour index for all possible connected edges for all vertices
// -1 indicates, that there is no connection possible (self)
__constant__ __device__ int TetrahedronToNeighbourIdx[7][6][6] = {

        {{ 0,  1, -1,  2,  3,  4}, {-1,  5,  6,  2,  4,  7}, { 8, -1,  9, 10,  1,  0},
         {11, 12, 13, 14, 15, -1}, {13,  9, 16, 14, -1,  8}, {12, -1, 15, 17,  6,  5}}, // #v0

        {{ 0,  1,  2,  3, -1,  4}, { 5, -1,  6,  7,  8,  9}, {10, 11, -1, 12, 13, 14},
         {-1,  9,  8, 12, 14, 15}, { 4, -1,  1, 16, 11, 10}, {17,  5,  0,  3,  6, -1}}, // #v1

        {{ 0, -1,  1,  2,  3,  4}, { 5,  6,  7,  8,  9, -1}, { 7,  1, 10,  8, -1,  0},
         { 6, -1,  9, 11, 12, 13}, { 4,  3, -1, 14, 15, 16}, {-1, 13, 12, 14, 16, 17}}, // #v2

        {{-1,  0,  1,  2,  3,  4}, {-1, -1, -1, -1, -1, -1}, { 5,  1,  6,  7,  0, -1},
         { 6,  8,  9,  7, -1, 10}, {-1, -1, -1, -1, -1, -1}, { 8, 10, -1,  2, 11,  3}}, // #v3

        {{-1, -1, -1, -1, -1, -1}, { 0,  1, -1,  2,  3,  4}, {-1,  5,  6,  2,  4,  7},
         {-1, -1, -1, -1, -1, -1}, { 8,  6,  9,  10, 5, -1}, { 9,  0, 11, 10, -1,  1}}, // #v4

        {{ 0,  1,  2,  3,  4, -1}, { 2,  5,  6,  3, -1,  7}, {-1, -1, -1, -1, -1, -1},
         { 5,  7, -1,  8,  9, 10}, {-1,  4,  1,  8, 10, 11}, {-1, -1, -1, -1, -1, -1}}, // #v5

        {{ 0,  1,  2, -1,  3,  4}, { 2,  5,  6, -1,  4,  7}, { 6,  8,  9, -1,  7, 10},
         { 9, 11, 12, -1, 10, 13}, {12, 14, 15, -1, 13, 16}, {15, 17,  0, -1, 16,  3}}, // #v6
};


__constant__ __device__ uint TriangleCrossPoductVtxIdx[3][2] = {
        {1, 2}, {1, 0}, {0, 2}
};


inline __device__ void LoadTetrahedronsInACube() {
    // Load tetrahedron vertex index to cube index map into shared memory.
    if (threadIdx.x < 6) {
        for (int i = 0; i < 4; ++i) {
            tetrahedronsInACube_S[threadIdx.x][i] = tetrahedronsInACube[threadIdx.x][i];
        }
    }
}


/*
 * getTetrahedronEdgeVertexIdx_D
 */
inline __device__ uint getTetrahedronEdgeVertexIdx_D(
        uint activeCubeIndex,
        uint tetrahedronIdx,
        uint edgeIdx,
        uint *cubeMap_D,
        uint *cubeOffs_D) {

    uint cubeIdx = cubeMap_D[activeCubeIndex];
    uint offset = (gridSize_D.x-1)*(
            (gridSize_D.y-1)*TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx][2] // Global cube index
                    + TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx][1])           // Global cube index
            + TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx][0];
    uint cubeIdxNew = cubeOffs_D[cubeIdx + offset];
    return 7*cubeIdxNew + TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx][3];
}


/**
 * Computes flags for one tetrahedron. The flags define what vertices of the
 * tetrahedron are active.
 *
 * @param[in] cubeVertex0       The origin of the current cell
 * @param[in] tetrahedronIndex  Local index of the tetrahedron inside the cell
 * @param[in] thresholdValue    The isovalue that defines the isosurface
 * @param[in] volume_D          The volume the isosurface is extracted from
 *
 * @return The tetrahedron flags
 */
inline __device__ unsigned char tetrahedronFlags_D(
        uint3 cubeVertex0,
        int tetrahedronIndex,
        float thresholdValue,
        float *volume_D) {

    unsigned char flags = 0;
    // Loop through all four vertices of the tetrahedron
    for (int idx = 0; idx < 4; ++idx) {
        const uint3 cubeVertexOffset = make_uint3(
                cubeVertexOffsets_S[tetrahedronsInACube_S[tetrahedronIndex][idx]][0],
                cubeVertexOffsets_S[tetrahedronsInACube_S[tetrahedronIndex][idx]][1],
                cubeVertexOffsets_S[tetrahedronsInACube_S[tetrahedronIndex][idx]][2]);
        if(::SampleFieldAt_D<float, false>(cubeVertex0 + cubeVertexOffset, volume_D) <= thresholdValue) {
            flags |= 1 << static_cast<unsigned char>(idx);
        }
    }
    return flags;
}


/**
 * Setup mapping from the list containing all cells to the list containing only
 * active cells.
 *
 * @param[out] cubeMap_D     The mapping from the cell list to the active cells'
 *                           list
 * @param[in]  cubeOffs_D    Index of the cells in the active cell's list
 * @param[in]  cubeStates_D  The flags of the cells
 * @param[in]  cubeCount     The number of cells to be processed
 */
__global__ void GPUSurfaceMT_CalcCubeMap_D(
        uint* cubeMap_D,     // output
        uint* cubeOffs_D,    // input
        uint* cubeStates_D,  // input
        uint cubeCount) {    // input

    const uint cubeIndex = ::getThreadIdx();
    if (cubeIndex >= cubeCount) {
        return;
    }

    if(cubeStates_D[cubeIndex] != 0) {
        // Map from active cubes list to cube index
        cubeMap_D[cubeOffs_D[cubeIndex]] = cubeIndex;
    }
}


/**
 * Setup mapping function from active vertex list to vertex list (based on
 * active cells).
 *
 * @param[out] vertexMap_D       Mapping from active vertex' list to global
 *                               vertex list
 * @param[in]  vertexIdxOffs_D   Offsets for vertex indices
 * @param[in]  activeVertexIdx_D Active vertex flags, '1' if vertex is active
 * @param[in]  vtxCount          The number of vertices
 */
__global__ void GPUSurfaceMT_CalcVertexMap_D(
        uint* vertexMap_D,
        uint* vertexIdxOffs_D,
        uint* activeVertexIdx_D,
        uint vtxCount) {

    const uint vtxIndex = ::getThreadIdx();
    if (vtxIndex >= vtxCount) {
        return;
    }

    if(activeVertexIdx_D[vtxIndex] != 0) {
        vertexMap_D[vertexIdxOffs_D[vtxIndex]] = vtxIndex;
    }
}


/**
 * Compute vertex positions in active cells. Every active cell is associated
 * with 7 potentially active vertices. For all 7 vertices their active flag is
 * set and, if possible, the position is computed. One kernel processes one
 * tetrahedron
 *
 * @param[out] activeVertexIdx_D The flag that shows whether a vertex is active
 *                               ('1') or not ('0')
 * @param[out] activeVertexPos_D The position of active vertices
 * @param[in]  cubeMapInv_D      Mapping from the global cell list to the active
 *                               cells' list
 * @param[in]  isoval            The isovalue that defines the isosurface
 * @param[in]  activeCubeCnt     The number of active cells
 * @param[in]  volume_D          The volume the isosurface is extracted from
 */
__global__ void GPUSurfaceMT_CalcVertexPositions_D(
        uint*activeVertexIdx_D,
        float3 *activeVertexPos_D,
        uint* cubeMap_D,
        float isoval,
        uint activeCubeCount,
        float *volume_D) {

    // Load LUTs to shared memory
    LoadCubeOffsetsToSharedMemory();
    LoadTetrahedronsInACubeToSharedMemory();
    LoadVertexIdxPerTetrahedronIdxToSharedMemory();
    LoadTetrahedronEdgeFlagsAndConnectionsToSharedMemory();
    __syncthreads();

    // Thread index (= active cube index)
    uint globalTetraIdx = ::getThreadIdx();
    if (globalTetraIdx >= activeCubeCount*6) {
        return;
    }

    uint activeCubeIdx = globalTetraIdx/6;
    uint localTetraIdx = globalTetraIdx%6; // 0 ... 5

    // Compute cell origin
    const uint3 cellOrg = GetGridCoordsByCellIdx(cubeMap_D[activeCubeIdx]);

    // Get bitmap to classify the tetrahedron
    unsigned char tetrahedronFlags = tetrahedronFlags_D(cellOrg, localTetraIdx, isoval, volume_D);

    for (int i = 0; i < 2; ++i) {
        if (VertexIdxPerTetrahedronIdx_S[localTetraIdx][i][0] < 0) {
            continue;
        }
        uint localVtxIdx = VertexIdxPerTetrahedronIdx_S[localTetraIdx][i][0];
        uint edgeIdx = VertexIdxPerTetrahedronIdx_S[localTetraIdx][i][1];
        if (tetrahedronEdgeFlags_S[tetrahedronFlags] & (1 << static_cast<unsigned char>(edgeIdx))) {

            // Interpolate vertex position
            const uint3 v0 = cellOrg + make_uint3(
                    cubeVertexOffsets_S[tetrahedronsInACube_S[localTetraIdx][tetrahedronEdgeConnections_S[edgeIdx][0]]][0],
                    cubeVertexOffsets_S[tetrahedronsInACube_S[localTetraIdx][tetrahedronEdgeConnections_S[edgeIdx][0]]][1],
                    cubeVertexOffsets_S[tetrahedronsInACube_S[localTetraIdx][tetrahedronEdgeConnections_S[edgeIdx][0]]][2]);
            const uint3 v1 = cellOrg + make_uint3(
                    cubeVertexOffsets_S[tetrahedronsInACube_S[localTetraIdx][tetrahedronEdgeConnections_S[edgeIdx][1]]][0],
                    cubeVertexOffsets_S[tetrahedronsInACube_S[localTetraIdx][tetrahedronEdgeConnections_S[edgeIdx][1]]][1],
                    cubeVertexOffsets_S[tetrahedronsInACube_S[localTetraIdx][tetrahedronEdgeConnections_S[edgeIdx][1]]][2]);

            // Linear interpolation
            const float f0 = ::SampleFieldAt_D<float, false>(v0, volume_D);
            const float f1 = ::SampleFieldAt_D<float, false>(v1, volume_D);
            const float interpolator = (isoval - f0) / (f1 - f0);
            float3 vertex = lerp(make_float3(v0.x, v0.y, v0.z),
                    make_float3(v1.x, v1.y, v1.z), interpolator);

            // Save position and mark vertex index as 'active'
            activeVertexIdx_D[activeCubeIdx*7+localVtxIdx] = 1;
            activeVertexPos_D[activeCubeIdx*7+localVtxIdx] = TransformToWorldSpace(vertex);
        }
    }
}

//   9 == undefined
//   4 == diagonal
__constant__ __device__ int faceIdxbyEdgeIndices[6][6] = {
      //  0, 1, 2, 3, 4, 5
        { 9, 3, 3, 0, 0, 4 }, // 0
        { 3, 9, 3, 4, 1, 1 }, // 1
        { 3, 3, 9, 2, 4, 2 }, // 2
        { 0, 4, 2, 9, 0, 2 }, // 3
        { 0, 1, 4, 0, 9, 1 }, // 4
        { 4, 1, 2, 2, 1, 9 }  // 5
};

// Faces defined by edge indices
__constant__ __device__ uint TetrahedronFaces_C [4][3] = {
        {0, 3, 4}, // 0
        {1, 4, 5}, // 1
        {2, 3, 5}, // 2
        {0, 1, 2}  // 3
};

// [localTetraIdx][tetraFaceIdx][offs.x, offs.y, offs.z, localTetraIdx, faceIdx]
__constant__ __device__ int GetAdjTetraFace_C[6][4][5] = {
        { // local tetrahedron #0
                { 0, 0, 0, 5, 2}, // local face #0
                { 1, 0, 0, 4, 3}, // local face #1
                { 0, 0, 0, 1, 0}, // local face #2
                { 0,-1, 0, 2, 1}, // local face #3
        },
        { // local tetrahedron #1
                { 0, 0, 0, 0, 2}, // local face #0
                { 1, 0, 0, 3, 3}, // local face #1
                { 0, 0, 0, 2, 0}, // local face #2
                { 0, 0,-1, 5, 1}, // local face #3
        },
        { // local tetrahedron #2
                { 0, 0, 0, 1, 2}, // local face #0
                { 0, 1, 0, 0, 3}, // local face #1
                { 0, 0, 0, 3, 0}, // local face #2
                { 0, 0,-1, 4, 1}, // local face #3
        },
        { // local tetrahedron #3
                { 0, 0, 0, 2, 2}, // local face #0
                { 0, 1, 0, 5, 3}, // local face #1
                { 0, 0, 0, 4, 0}, // local face #2
                {-1, 0, 0, 1, 1}, // local face #3
        },
        { // local tetrahedron #4
                { 0, 0, 0, 3, 2}, // local face #0
                { 0, 0, 1, 2, 3}, // local face #1
                { 0, 0, 0, 5, 0}, // local face #2
                {-1, 0, 0, 0, 1}, // local face #3
        },
        { // local tetrahedron #5
                { 0, 0, 0, 4, 2}, // local face #0
                { 0, 0, 1, 1, 3}, // local face #1
                { 0, 0, 0, 0, 0}, // local face #2
                { 0,-1, 0, 3, 1}, // local face #3
        }
};


// Each tetrahedron contains either one or two triangles. This LUT answers
// whether the triangle a tetrahedron face is adjacent to none, the first or the
// second defined triangle (-1, 0, 1)
__constant__ __device__ int IsFaceAdjacentToTri_C[16][4] = {
            {-1, -1, -1, -1}, // #0
            { 0, -1,  0,  0}, // #1
            { 0,  0, -1,  0}, // #2
            { 1,  0,  1,  0}, // #3
            {-1,  0,  0,  0}, // #4
            { 0,  1,  0,  1}, // #5
            { 1,  1,  0,  0}, // #6
            { 0,  0,  0, -1}, // #7
            { 0,  0,  0, -1}, // #8
            { 0,  0,  1,  1}, // #9
            { 1,  0,  1,  0}, // #10
            {-1,  0,  0,  0}, // #11
            { 0,  1,  0,  1}, // #12
            { 0,  0, -1,  0}, // #13
            { 0, -1,  0,  0}, // #14
            {-1, -1, -1, -1}  // #15
};


/*
 * GPUSurfaceMT_ComputeTriangleNeighbors_D
 */
__global__ void GPUSurfaceMT_ComputeTriangleNeighbors_D (
        uint *triangleNeighbors_D,
        uint* vertexOffsets_D,
        uint* cubeMap_D,
        uint* cubeMapInv_D,
        float isoval,
        uint *vertexMapInv_D,
        float *volume_D,
        uint activeCellCnt) {

    const uint id = getThreadIdx();

    // Load cube vertex offsets into shared memory
    if (threadIdx.x < 32) {
        const int clampedThreadIdx = min(threadIdx.x, 7);
        cubeVertexOffsets_S[clampedThreadIdx][0] = cubeVertexOffsets[clampedThreadIdx][0];
        cubeVertexOffsets_S[clampedThreadIdx][1] = cubeVertexOffsets[clampedThreadIdx][1];
        cubeVertexOffsets_S[clampedThreadIdx][2] = cubeVertexOffsets[clampedThreadIdx][2];
    }

    // Load tetrahedrons in a cube to shared memory
    if (threadIdx.x >= 32 && threadIdx.x < 64) {
        const int clampedThreadIdx = min(threadIdx.x, 37) - 32;
        tetrahedronsInACube_S[clampedThreadIdx][0] = tetrahedronsInACube[clampedThreadIdx][0];
        tetrahedronsInACube_S[clampedThreadIdx][1] = tetrahedronsInACube[clampedThreadIdx][1];
        tetrahedronsInACube_S[clampedThreadIdx][2] = tetrahedronsInACube[clampedThreadIdx][2];
        tetrahedronsInACube_S[clampedThreadIdx][3] = tetrahedronsInACube[clampedThreadIdx][3];
    }
    __syncthreads();

    if (id >= activeCellCnt*6) {
        return;
    }

    const uint activeCellIndex = id / 6;
    const int tetrahedronIndex = id % 6;
    const uint3 cellOrg = ::GetGridCoordsByCellIdx(cubeMap_D[activeCellIndex]);

    // Get bitmap to classify the tetrahedron
    unsigned char tetrahedronFlags = tetrahedronFlags_D(cellOrg,
            tetrahedronIndex, isoval, volume_D);

    uint vertexBaseOffset = vertexOffsets_D[id];

//    if (vertexBaseOffset == 7*3) {
//        printf("localTetraIdx %i, flags %i\n", tetrahedronIndex, tetrahedronFlags);
//    }

    const uint edges[6][2] = {
            {0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}
    };

    // Loop through all 6 possible edges
    for (int i = 0; i < 6; ++i) {

        uint idx0 = edges[i][0];
        uint idx1 = edges[i][1];

        if (tetrahedronTriangles[tetrahedronFlags][idx0] < 0) continue; // No second triangle

        // Get local tetrahedron edge indices describing that edge
        uint localTetraEdgeIdx0 = tetrahedronTriangles[tetrahedronFlags][idx0];
        uint localTetraEdgeIdx1 = tetrahedronTriangles[tetrahedronFlags][idx1];

//        if (vertexBaseOffset == 7*3) {
//            printf("edge %i: %u %u\n", i, localTetraEdgeIdx0, localTetraEdgeIdx1);
//        }

        // Obtain face on which the edge lays
        uint localFaceIdx = faceIdxbyEdgeIndices[localTetraEdgeIdx0][localTetraEdgeIdx1];

//        if (vertexBaseOffset == 7*3) {
//            printf("face Idx %u\n", localFaceIdx);
//        }

        if (localFaceIdx == 4) { // 4 means diagonal

            // In this case, the adjacent triangle is simply the other triangle
            // in this tetrahedron
            // TODO Special treatment necessary?
            triangleNeighbors_D[vertexBaseOffset + i] =
                    vertexBaseOffset/3 + int(i <= 2);
        } else {
            // Get adjacent face
            int3 cellOffs;
            cellOffs.x =       GetAdjTetraFace_C[tetrahedronIndex][localFaceIdx][0];
            cellOffs.y =       GetAdjTetraFace_C[tetrahedronIndex][localFaceIdx][1];
            cellOffs.z =       GetAdjTetraFace_C[tetrahedronIndex][localFaceIdx][2];
            uint adjTetraIdx = GetAdjTetraFace_C[tetrahedronIndex][localFaceIdx][3];
            uint adjFaceIdx =  GetAdjTetraFace_C[tetrahedronIndex][localFaceIdx][4];

//            if (vertexBaseOffset == 7*3) {
//                printf("offs %i %i %i, adjTetra %u, adjFace %u\n",
//                        cellOffs.x, cellOffs.y, cellOffs.z,
//                        adjTetraIdx, adjFaceIdx);
//            }

            int3 adjCell = make_int3(cellOrg.x + cellOffs.x,
                                     cellOrg.y + cellOffs.y,
                                     cellOrg.z + cellOffs.z);

//            if (vertexBaseOffset == 7*3) {
//                printf("adjCell %i %i %i\n",
//                        adjCell.x, adjCell.y, adjCell.z);
//            }

            uint adjCellIdx = ::GetCellIdxByGridCoords(adjCell);
            uint adjActiveCellIdx = cubeMapInv_D[adjCellIdx];

//            if (vertexBaseOffset == 7*3) {
//                printf("adjCellIdx %u\n", adjCellIdx);
//            }
//
//            if (vertexBaseOffset == 7*3) {
//                printf("adjActiveCellIdx %u\n", adjActiveCellIdx);
//            }


            // Compute global tetrahedron index of the adjacent tetrahedron
            uint globalAdjTetraIdx = adjActiveCellIdx*6 + adjTetraIdx;

//            if (vertexBaseOffset == 7*3) {
//                printf("globalAdjTetraIdx %u\n", globalAdjTetraIdx);
//            }

            unsigned char adjTetrahedronFlags =  tetrahedronFlags_D(
                    make_uint3(adjCell.x, adjCell.y, adjCell.z),
                    adjTetraIdx, isoval, volume_D);

            // Get triangle index offset
            // Note: this should never be negative
            int triOffs = IsFaceAdjacentToTri_C[adjTetrahedronFlags][adjFaceIdx];

//            if (vertexBaseOffset == 7*3) {
//                printf("triOffs %i\n", triOffs);
//            }

            // Write to global dev memory
            triangleNeighbors_D[vertexBaseOffset + i] = vertexOffsets_D[globalAdjTetraIdx] / 3 + triOffs;
//            if (vertexBaseOffset == 7*3) {
//                printf("ADD NEIGHBOR %i\n",
//                        vertexOffsets_D[globalAdjTetraIdx] / 3 + triOffs);
//            }
        }
    }
}


/*
 * GPUSurfaceMT::ComputeTriangleNeighbors
 */
bool GPUSurfaceMT::ComputeTriangleNeighbors(
        float *volume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue) {

    using namespace vislib::sys;

    /// Init grid parameters for all files ///

    const uint blocksize = 256;

    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    if (!CudaSafeCall(this->triangleNeighbors_D.Validate(this->triangleCnt*3))) {
        return false;
    }
    // Initialize triangle index array
    GPUSurfaceMT_ComputeTriangleNeighbors_D <<< Grid(this->activeCellCnt*6, blocksize), blocksize >>> (
            this->triangleNeighbors_D.Peek(),
            this->tetrahedronVertexOffsets_D.Peek(),
            this->cubeMap_D.Peek(),
            this->cubeOffsets_D.Peek(),
            isovalue,
            this->vertexIdxOffs_D.Peek(),
            volume_D,
            this->activeCellCnt);

    if (!CheckForCudaError()) {
        return false;
    }

//    // DEBUG print triangle neighbors
//    HostArr<unsigned int> triangleNeighbors;
//    triangleNeighbors.Validate(this->triangleNeighbors_D.GetCount());
//    if (!CudaSafeCall(this->triangleNeighbors_D.CopyToHost(triangleNeighbors.Peek()))){
//        return false;
//    }
//    for (int e = 0; e < this->triangleCnt; ++e) {
//        printf("TRIANGLE NEIGHBORS %i: %u %u %u\n", e,
//                triangleNeighbors.Peek()[3*e+0],
//                triangleNeighbors.Peek()[3*e+1],
//                triangleNeighbors.Peek()[3*e+2]);
//    }
//    triangleNeighbors.Release();
//    // END DEBUG

    return true;
}


/**
 * Writes all active vertex positions to a compacted array.
 * @param[out] vertexPos_D       The array with the compacted positions
 * @param[in]  vertexStates_D    Contains flags that show the activity of the
 *                               vertices
 * @param[in]  vertexIdxOffs_D   The index of the vertex in the compacted vertex
 *                               list
 * @param[in]  activeVertexPos_D The array with non-compacted vertex positions
 * @param[in]  vertexCount       The number of vertices (active and non-active)
 * @param[in]  outputArrOffs     The output buffer offset to store vertex
 *                               positions
 * @param[in]  outputArrDataSize The output buffer stride
 */
__global__ void GPUSurfaceMT_CompactActiveVertexPositions_D(
        float *vertexPos_D,
        uint *vertexStates_D,
        uint *vertexIdxOffs_D,
        float3 *activeVertexPos_D,
        uint vertexCount,
        uint outputArrOffs,
        uint outputArrDataSize) {

    // Thread index (= vertex index)
    uint idx = getThreadIdx();
    if (idx >= vertexCount) {
        return;
    }

    if (vertexStates_D[idx] == 1) {
        vertexPos_D[outputArrDataSize*vertexIdxOffs_D[idx]+outputArrOffs+0] = activeVertexPos_D[idx].x;
        vertexPos_D[outputArrDataSize*vertexIdxOffs_D[idx]+outputArrOffs+1] = activeVertexPos_D[idx].y;
        vertexPos_D[outputArrDataSize*vertexIdxOffs_D[idx]+outputArrOffs+2] = activeVertexPos_D[idx].z;
    }
}


/// Answers the two vertices that build an edge associated with a tetrahedron
/// based on its local tetrahedron index (0...5) and its tetrahedron flags (0...15)
/// The vertices are defined by their tetrahedron edge indices (0...5), -1
/// indicates undefined values
// TODO Use shared memory?
__constant__ __device__ int EdgesByTetraFlags_C[16][6][4][2] = {

{{{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}}}, // #0  // 0000

{{{ 0, 2}, { 2, 3}, { 3, 0}, {-1,-1}},
 {{ 0, 2}, { 2, 3}, {-1,-1}, {-1,-1}},
 {{ 0, 2}, { 2, 3}, {-1,-1}, {-1,-1}},
 {{ 0, 2}, { 2, 3}, {-1,-1}, {-1,-1}},
 {{ 0, 2}, { 2, 3}, {-1,-1}, {-1,-1}},
 {{ 0, 2}, {-1,-1}, {-1,-1}, {-1,-1}}}, // #1  // 0001

{{{ 0, 1}, { 4, 0}, {-1,-1}, {-1,-1}},
 {{ 0, 1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 0, 1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 0, 1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 0, 1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 0, 1}, {-1,-1}, {-1,-1}, {-1,-1}}}, // #2  // 0010

{{{ 2, 3}, { 4, 2}, { 1, 2}, { 3, 4}},
 {{ 2, 3}, { 4, 2}, { 1, 2}, {-1,-1}},
 {{ 2, 3}, { 4, 2}, { 1, 2}, {-1,-1}},
 {{ 2, 3}, { 4, 2}, { 1, 2}, {-1,-1}},
 {{ 2, 3}, { 4, 2}, { 1, 2}, {-1,-1}},
 {{ 1, 2}, { 4, 2}, {-1,-1}, {-1,-1}}}, // #3  // 0011

{{{ 5, 2}, { 2, 1}, {-1,-1}, {-1,-1}},
 {{ 5, 2}, { 2, 1}, {-1,-1}, {-1,-1}},
 {{ 5, 2}, { 2, 1}, {-1,-1}, {-1,-1}},
 {{ 5, 2}, { 2, 1}, {-1,-1}, {-1,-1}},
 {{ 5, 2}, { 2, 1}, {-1,-1}, {-1,-1}},
 {{ 1, 2}, {-1,-1}, {-1,-1}, {-1,-1}}}, // #4  // 0100

{{{ 0, 1}, { 5, 3}, { 0, 5}, { 3, 0}},
 {{ 0, 1}, { 5, 3}, { 0, 5}, {-1,-1}},
 {{ 0, 1}, { 5, 3}, { 0, 5}, {-1,-1}},
 {{ 0, 1}, { 5, 3}, { 0, 5}, {-1,-1}},
 {{ 0, 1}, { 5, 3}, { 0, 5}, {-1,-1}},
 {{ 0, 1}, { 0, 5}, {-1,-1}, {-1,-1}}}, // #5  // 0101

{{{ 0, 2}, { 2, 5}, { 0, 5}, { 4, 0}},
 {{ 0, 2}, { 2, 5}, { 0, 5}, {-1,-1}},
 {{ 0, 2}, { 2, 5}, { 0, 5}, {-1,-1}},
 {{ 0, 2}, { 2, 5}, { 0, 5}, {-1,-1}},
 {{ 0, 2}, { 2, 5}, { 0, 5}, {-1,-1}},
 {{ 0, 2}, { 0, 5}, {-1,-1}, {-1,-1}}}, // #6  // 0110

{{{ 5, 3}, { 3, 4}, {-1,-1}, {-1,-1}},
 {{ 5, 3}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 5, 3}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 5, 3}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 5, 3}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}}}, // #7  // 0111

{{{ 5, 3}, { 3, 4}, {-1,-1}, {-1,-1}},
 {{ 5, 3}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 5, 3}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 5, 3}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 5, 3}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}}}, // #8  // 1000

{{{ 0, 2}, { 2, 5}, { 0, 5}, { 4, 0}},
 {{ 0, 2}, { 2, 5}, { 0, 5}, {-1,-1}},
 {{ 0, 2}, { 2, 5}, { 0, 5}, {-1,-1}},
 {{ 0, 2}, { 2, 5}, { 0, 5}, {-1,-1}},
 {{ 0, 2}, { 2, 5}, { 0, 5}, {-1,-1}},
 {{ 0, 2}, { 0, 5}, {-1,-1}, {-1,-1}}}, // #9  // 1001

{{{ 0, 1}, { 5, 3}, { 0, 5}, { 3, 0}},
 {{ 0, 1}, { 5, 3}, { 0, 5}, {-1,-1}},
 {{ 0, 1}, { 5, 3}, { 0, 5}, {-1,-1}},
 {{ 0, 1}, { 5, 3}, { 0, 5}, {-1,-1}},
 {{ 0, 1}, { 5, 3}, { 0, 5}, {-1,-1}},
 {{ 0, 1}, { 0, 5}, {-1,-1}, {-1,-1}}}, // #10 // 1010

{{{ 5, 2}, { 2, 1}, {-1,-1}, {-1,-1}},
 {{ 5, 2}, { 2, 1}, {-1,-1}, {-1,-1}},
 {{ 5, 2}, { 2, 1}, {-1,-1}, {-1,-1}},
 {{ 5, 2}, { 2, 1}, {-1,-1}, {-1,-1}},
 {{ 5, 2}, { 2, 1}, {-1,-1}, {-1,-1}},
 {{ 1, 2}, {-1,-1}, {-1,-1}, {-1,-1}}}, // #11 // 1011

{{{ 2, 3}, { 4, 2}, { 1, 2}, { 3, 4}},
 {{ 2, 3}, { 4, 2}, { 1, 2}, {-1,-1}},
 {{ 2, 3}, { 4, 2}, { 1, 2}, {-1,-1}},
 {{ 2, 3}, { 4, 2}, { 1, 2}, {-1,-1}},
 {{ 2, 3}, { 4, 2}, { 1, 2}, {-1,-1}},
 {{ 1, 2}, { 4, 2}, {-1,-1}, {-1,-1}}}, // #12 // 1100

{{{ 0, 1}, { 4, 0}, {-1,-1}, {-1,-1}},
 {{ 0, 1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 0, 1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 0, 1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 0, 1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{ 0, 1}, {-1,-1}, {-1,-1}, {-1,-1}}}, // #13 // 1101

{{{ 0, 2}, { 2, 3}, { 3, 0}, {-1,-1}},
 {{ 0, 2}, { 2, 3}, {-1,-1}, {-1,-1}},
 {{ 0, 2}, { 2, 3}, {-1,-1}, {-1,-1}},
 {{ 0, 2}, { 2, 3}, {-1,-1}, {-1,-1}},
 {{ 0, 2}, { 2, 3}, {-1,-1}, {-1,-1}},
 {{ 0, 2}, {-1,-1}, {-1,-1}, {-1,-1}}}, // #14 // 1110

{{{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}},
 {{-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}}}  // #15 // 1111

};


/*
 * TODO
 */
__global__ void GPUSurfaceMT_ComputeEdgeList_D(
        uint *edges_D,
        uint *tetraEdgeIdxOffsets_D,
        uint *edgesPerTetrahedron_D,
        uint *vertexMapInv_D,
        uint* cubeMap_D,
        uint* cubeMapInv_D,
        float thresholdValue,
        float *volume_D,
        uint activeCubeCount) {

    // Thread index (= tetrahedron index)
    const uint id = getThreadIdx();

    // Load cube vertex offsets into shared memory
    if (threadIdx.x < 32) {
        const int clampedThreadIdx = min(threadIdx.x, 7);
        cubeVertexOffsets_S[clampedThreadIdx][0] = cubeVertexOffsets[clampedThreadIdx][0];
        cubeVertexOffsets_S[clampedThreadIdx][1] = cubeVertexOffsets[clampedThreadIdx][1];
        cubeVertexOffsets_S[clampedThreadIdx][2] = cubeVertexOffsets[clampedThreadIdx][2];
    }

    // Load tetrahedrons in a cube to shared memory
    if (threadIdx.x >= 32 && threadIdx.x < 64) {
        const int clampedThreadIdx = min(threadIdx.x, 37) - 32;
        tetrahedronsInACube_S[clampedThreadIdx][0] = tetrahedronsInACube[clampedThreadIdx][0];
        tetrahedronsInACube_S[clampedThreadIdx][1] = tetrahedronsInACube[clampedThreadIdx][1];
        tetrahedronsInACube_S[clampedThreadIdx][2] = tetrahedronsInACube[clampedThreadIdx][2];
        tetrahedronsInACube_S[clampedThreadIdx][3] = tetrahedronsInACube[clampedThreadIdx][3];
    }
    __syncthreads();

    if (id >= activeCubeCount*6) {
        return;
    }

    const uint activeCubeIndex = id / 6;
    const int tetrahedronIndex = id % 6;
    const uint3 cellOrg = ::GetGridCoordsByCellIdx(cubeMap_D[activeCubeIndex]);

    // Get bitmap to classify the tetrahedron
    unsigned char tetrahedronFlags = tetrahedronFlags_D(cellOrg,
            tetrahedronIndex, thresholdValue, volume_D);

    const uint nEdges = edgesPerTetrahedron_D[id];
    const uint idxOffs = tetraEdgeIdxOffsets_D[id];

    // Loop through all edges associated with this tetratedron
    for (int e = 0; e < nEdges; ++e) {

        // First vertex TODO
        int edgeIndex0 = EdgesByTetraFlags_C[tetrahedronFlags][tetrahedronIndex][e][0];
        if (edgeIndex0 < 0) {
            edges_D[2*(idxOffs+e)+0] = 0;
            edges_D[2*(idxOffs+e)+1] = 0;
            continue;
        }
        int vertexIdx0 = getTetrahedronEdgeVertexIdx_D(
                activeCubeIndex, tetrahedronIndex, edgeIndex0, cubeMap_D,
                cubeMapInv_D);

        // Second vertex TODO
        int edgeIndex1 = EdgesByTetraFlags_C[tetrahedronFlags][tetrahedronIndex][e][1];
        if (edgeIndex1 < 0) {
            edges_D[2*(idxOffs+e)+0] = 0;
            edges_D[2*(idxOffs+e)+1] = 0;
            continue;
        }
        int vertexIdx1 = getTetrahedronEdgeVertexIdx_D(
                activeCubeIndex, tetrahedronIndex, edgeIndex1, cubeMap_D,
                cubeMapInv_D);

        edges_D[2*(idxOffs+e)+0] = vertexMapInv_D[vertexIdx0];
        edges_D[2*(idxOffs+e)+1] = vertexMapInv_D[vertexIdx1];
    }
}


/// Answers the number of edges associated with a tetrahedron based on its local
/// tetrahedron index (0...5) and its tetrahedron flags (0...15)
// TODO Use shared memory?
__constant__ __device__ unsigned int EdgeCntByTetraFlags_C[16][6] = {
        {0, 0, 0, 0, 0, 0}, // #0  // 0000
        {3, 2, 2, 2, 2, 1}, // #1  // 0001
        {2, 1, 1, 1, 1, 1}, // #2  // 0010
        {4, 3, 3, 3, 3, 2}, // #3  // 0011
        {2, 2, 2, 2, 2, 1}, // #4  // 0100
        {4, 3, 3, 3, 3, 2}, // #5  // 0101
        {4, 3, 3, 3, 3, 2}, // #6  // 0110
        {2, 1, 1, 1, 1, 0}, // #7  // 0111
        {2, 1, 1, 1, 1, 0}, // #8  // 1000
        {4, 3, 3, 3, 3, 2}, // #9  // 1001
        {4, 3, 3, 3, 3, 2}, // #10 // 1010
        {2, 2, 2, 2, 2, 1}, // #11 // 1011
        {4, 3, 3, 3, 3, 2}, // #12 // 1100
        {2, 1, 1, 1, 1, 1}, // #13 // 1101
        {3, 2, 2, 2, 2, 1}, // #14 // 1110
        {0, 0, 0, 0, 0, 0}  // #15 // 1111
};


/*
 * TODO
 */
__global__ void GPUSurfaceMT_ComputeEdgesPerTetrahedron_D(
        uint *edgesPerTetrahedron_D,
        float *volume_D,
        uint *cubeMap_D,
        float isoval,
        uint activeCubeCount) {

    // Load LUTs to shared memory
    // TODO Which of those are necessary
    LoadCubeOffsetsToSharedMemory();
    LoadTetrahedronsInACubeToSharedMemory();
    LoadVertexIdxPerTetrahedronIdxToSharedMemory();
    LoadTetrahedronEdgeFlagsAndConnectionsToSharedMemory();
    __syncthreads();

    // Thread index
    uint globalTetraIdx = ::getThreadIdx();
    if (globalTetraIdx >= activeCubeCount*6) {
        return;
    }

    uint activeCubeIdx = globalTetraIdx/6;
    uint localTetraIdx = globalTetraIdx%6; // 0 ... 5

    // Compute cell origin
    const uint3 cellOrg = GetGridCoordsByCellIdx(cubeMap_D[activeCubeIdx]);

    // Get bitmap to classify the tetrahedron
    unsigned char tetrahedronFlags = tetrahedronFlags_D(cellOrg, localTetraIdx, isoval, volume_D);

    edgesPerTetrahedron_D[globalTetraIdx] = EdgeCntByTetraFlags_C[tetrahedronFlags][localTetraIdx];
}


/**
 * Identifies neighbouring vertices for all vertices and stores them as vertex
 * indices. -1 indicates invalid neighbours
 *
 * @param[out] vertexNeighbours_D The vertex connectivity information
 * @param[in]  activeVertexIdx_D  Array with vertex activity flags
 * @param[in]  activeVertexCnt    The number of vertices
 * @param[in]  vertexMap_D        Vertex mapping from active to global index
 * @param[in]  vertexMapInv_D     Inverse mapping to vertexMap_D
 * @param[in]  cubeMap_D          Mapping from active cells's list to global
 *                                cell list
 * @param[in]  cubeMapInv_D       Inverse mapping to cubeMap_D
 * @param[in]  cubeStates_D       Flags active cells
 * @param[in]  volume_D           The volume the isosurface is extracted from
 * @param[in]  isoval             The isovalue that defines the isosurface
 */
__global__ void GPUSurfaceMT_ComputeVertexConnectivity_D(
        int *vertexNeighbours_D,
        uint *activeVertexIdx_D,
        uint activeVertexCnt,
        uint *vertexMap_D,
        uint *vertexMapInv_D,
        uint *cubeMap_D,
        uint *cubeMapInv_D,
        uint *cubeStates_D,
        float *volume_D,
        float isoval) {

    // Get different indices
    uint idx = ::getThreadIdx();
    uint activeVertexIdx = idx/6;
    uint i = idx - __umul24(activeVertexIdx, 6); // == idx%6;

    /* 1. Load LUTs to shared memory */

    // Note: We have 6 warps per block (each warp with 32 threads)

    // Load cube vertex offsets into shared memory
    // Use warp #0
//    LoadCubeOffsetsToSharedMemory();
    if (threadIdx.x < 32) {
        const uint idx0 = clamp(int(threadIdx.x), 0, 7);
        cubeVertexOffsets_S[idx0][0] = cubeVertexOffsets[idx0][0];
        cubeVertexOffsets_S[idx0][1] = cubeVertexOffsets[idx0][1];
        cubeVertexOffsets_S[idx0][2] = cubeVertexOffsets[idx0][2];
    }

    // Load cube vertex offsets into shared memory
    // Use warp #1
    //LoadTetrahedronsInACubeToSharedMemory();
    if (threadIdx.x >= 32 && threadIdx.x < 64) {
        const uint idx0 = clamp(int(threadIdx.x-32), 0, 7);
        tetrahedronsInACube_S[idx0][0] = tetrahedronsInACube[idx0][0];
        tetrahedronsInACube_S[idx0][1] = tetrahedronsInACube[idx0][1];
        tetrahedronsInACube_S[idx0][2] = tetrahedronsInACube[idx0][2];
        tetrahedronsInACube_S[idx0][3] = tetrahedronsInACube[idx0][3];
    }

    // Load vertex indices of neighbouring tetrahedrons to shared memory
    // Use warps #2 and #3
    //LoadVertexNeighbouringTetrahedronsToSharedMemory();
    if (threadIdx.x >= 64 && threadIdx.x < 128) { // We need 2 warps here
        const uint idx0 = clamp(int((threadIdx.x-64)/7), 0, 6);
        const uint idx1 = threadIdx.x%6;
        VertexNeighbouringTetrahedrons_S[idx0][idx1][0] = VertexNeighbouringTetrahedrons[idx0][idx1][0];
        VertexNeighbouringTetrahedrons_S[idx0][idx1][1] = VertexNeighbouringTetrahedrons[idx0][idx1][1];
        VertexNeighbouringTetrahedrons_S[idx0][idx1][2] = VertexNeighbouringTetrahedrons[idx0][idx1][2];
        VertexNeighbouringTetrahedrons_S[idx0][idx1][3] = VertexNeighbouringTetrahedrons[idx0][idx1][3];
    }

    // Use vertex index offset to shared memory
    // Use warps #4 and #5
    //LoadTetrahedronEdgeVertexIdxOffsetToSharedMemory();
    if (threadIdx.x >= 128 && threadIdx.x < 192) {
        const uint idx0 = clamp(int((threadIdx.x-128)/6), 0, 5);
        const uint idx1 = threadIdx.x%6;
        TetrahedronEdgeVertexIdxOffset_S[idx0][idx1][0] = TetrahedronEdgeVertexIdxOffset[idx0][idx1][0];
        TetrahedronEdgeVertexIdxOffset_S[idx0][idx1][1] = TetrahedronEdgeVertexIdxOffset[idx0][idx1][1];
        TetrahedronEdgeVertexIdxOffset_S[idx0][idx1][2] = TetrahedronEdgeVertexIdxOffset[idx0][idx1][2];
        TetrahedronEdgeVertexIdxOffset_S[idx0][idx1][3] = TetrahedronEdgeVertexIdxOffset[idx0][idx1][3];
    }

    // Use first warp of every block to load vertex data for 6 threads
    // from global device memory to shared memory

    __shared__ int VertIdxGlobal_S[32];
    __shared__ int VertIdxLocal_S[32];
    __shared__ uint CellIdx_S[32];
    __shared__ uint3 CellOrg_S[32];

    if (threadIdx.x < 32) {
        //uint vtxIdx = clamp(32*blockIdx.x+threadIdx.x, uint(0), activeVertexCnt-1);
        uint vtxIdx = 32*blockIdx.x+threadIdx.x;

        if (vtxIdx < activeVertexCnt) {
            //VertIdxGlobal_S[threadIdx.x] = static_cast<int>(vertexMap_D[vtxIdx]); <-- this is actually slower
            VertIdxGlobal_S[threadIdx.x] = static_cast<int>(vertexMap_D[32*blockIdx.x+threadIdx.x]);
            CellIdx_S[threadIdx.x]       = VertIdxGlobal_S[threadIdx.x]/7;
            VertIdxLocal_S[threadIdx.x]  = VertIdxGlobal_S[threadIdx.x] - CellIdx_S[threadIdx.x]*7;
            CellOrg_S[threadIdx.x]       = ::GetGridCoordsByCellIdx(cubeMap_D[CellIdx_S[threadIdx.x]]);
        }
    }

    __syncthreads(); // Sync to make sure all writing operations are done!

    if (activeVertexIdx >= activeVertexCnt) {
        return;
    }

    // Load vertex data for current adjacent tetrahedron to registers
    const uint sharedMemoryIdx = threadIdx.x/6;
    //int vertIdx = VertIdxGlobal_S[sharedMemoryIdx];
    int v       = VertIdxLocal_S[sharedMemoryIdx];
    //uint cubeId = CellIdx_S[sharedMemoryIdx];
    uint3 cellOrg = CellOrg_S[sharedMemoryIdx];

    if (cellOrg.x >= gridSize_D.x-2) return;
    if (cellOrg.y >= gridSize_D.y-2) return;
    if (cellOrg.z >= gridSize_D.z-2) return;
    if (cellOrg.x <= 0) return;
    if (cellOrg.y <= 0) return;
    if (cellOrg.z <= 0) return;

    //--- From here on everything depends on 'i' -----------------------------//

    unsigned char terahedronFlagsTmp;
    unsigned char connectionFlags;
    uint ownEdgeIdx;

    // From here on stuff that depends on 'i'
    if (VertexNeighbouringTetrahedrons_S[v][i][0] == 99) return;

    // Get origin of the cell containing the adjacent tetrahedron
    int3 cellOrgTemp = make_int3(
            cellOrg.x + VertexNeighbouringTetrahedrons_S[v][i][0],
            cellOrg.y + VertexNeighbouringTetrahedrons_S[v][i][1],
            cellOrg.z + VertexNeighbouringTetrahedrons_S[v][i][2]);

    // Get tetrahedron flags of the adjacent tetrahedron
    terahedronFlagsTmp = tetrahedronFlags_D(
            make_uint3(cellOrgTemp.x, cellOrgTemp.y, cellOrgTemp.z),
            VertexNeighbouringTetrahedrons_S[v][i][3], isoval, volume_D);

    // Edge index of this vertex in the adjacent tetrahedron
    ownEdgeIdx = VertexNeighbouringTetrahedronsOwnEdgeIdx[v][i];

    // Look up connections
    connectionFlags = TetrahedronEdgeConnections[terahedronFlagsTmp][ownEdgeIdx];

    // Loop through possible connections
    for(int j = 0; j < 6; ++j) {
        if (connectionFlags & (1 << static_cast<unsigned char>(j))) {
            int3 tempOffs = make_int3(
                    TetrahedronEdgeVertexIdxOffset_S[i][j][0],
                    TetrahedronEdgeVertexIdxOffset_S[i][j][1],
                    TetrahedronEdgeVertexIdxOffset_S[i][j][2]);
            int3 neighbourVertexIdxOffs = cellOrgTemp + tempOffs;
            int vertexIdx =
                    static_cast<int>(cubeMapInv_D[GetCellIdxByGridCoords(neighbourVertexIdxOffs)]*7) +
                    TetrahedronEdgeVertexIdxOffset_S[i][j][3];
            vertexNeighbours_D[18*activeVertexIdx+TetrahedronToNeighbourIdx[v][i][j]] = vertexMapInv_D[vertexIdx];
            //vertexNeighbours_D[18*activeVertexIdx+TetrahedronToNeighbourIdx[v][i][j]] = vertexIdx;
        }
    }
}


/*
 * GPUSurfaceMT_ComputeVertexNormals_D
 *
 * TODO This method is highly inefficient
 */
__global__ void GPUSurfaceMT_ComputeVertexNormals_D(
        float *dataBuffer_D,
        uint *vertexMap_D,
        uint *vertexMapInv_D,
        uint *cubeMap_D,
        uint *cubeMapInv_D,
        float *volume_D,
        float isoval,
        uint activeVertexCnt,
        uint arrDataOffsPos,
        uint arrDataOffsNormals,
        uint arrDataSize){

    // Get thread index
    uint activeVertexIdx = ::getThreadIdx();

    // Load cube offsets to shared memory
    if (threadIdx.x < 8) {
        cubeVertexOffsets_S[threadIdx.x][0] = cubeVertexOffsets[threadIdx.x][0];
        cubeVertexOffsets_S[threadIdx.x][1] = cubeVertexOffsets[threadIdx.x][1];
        cubeVertexOffsets_S[threadIdx.x][2] = cubeVertexOffsets[threadIdx.x][2];
    }
    // Load tetrahedrons in a cube to shared memory
    if (threadIdx.x < 6) {
        tetrahedronsInACube_S[threadIdx.x][0] = tetrahedronsInACube[threadIdx.x][0];
        tetrahedronsInACube_S[threadIdx.x][1] = tetrahedronsInACube[threadIdx.x][1];
        tetrahedronsInACube_S[threadIdx.x][2] = tetrahedronsInACube[threadIdx.x][2];
        tetrahedronsInACube_S[threadIdx.x][3] = tetrahedronsInACube[threadIdx.x][3];
    }

    __syncthreads();

    if (activeVertexIdx >= activeVertexCnt) {
        return;
    }

    int vertIdx = static_cast<int>(vertexMap_D[activeVertexIdx]);

    int v = vertIdx%7; // We have 7 vertices per cube
    int cubeId = vertIdx/7;
    unsigned char terahedronFlagsTmp;
    uint ownEdgeIdx;

    // Omit border cells
    uint3 cellOrgUint = GetGridCoordsByCellIdx(cubeMap_D[cubeId]);
    int3 cellOrg = make_int3(cellOrgUint.x, cellOrgUint.y, cellOrgUint.z);
    if (cellOrg.x >= gridSize_D.x-2) return;
    if (cellOrg.y >= gridSize_D.y-2) return;
    if (cellOrg.z >= gridSize_D.z-2) return;
    if (cellOrg.x <= 0) return;
    if (cellOrg.y <= 0) return;
    if (cellOrg.z <= 0) return;


    float3 normal = make_float3(0.0, 0.0, 0.0);
    float3 pos = make_float3(
            dataBuffer_D[arrDataSize*activeVertexIdx+arrDataOffsPos+0],
            dataBuffer_D[arrDataSize*activeVertexIdx+arrDataOffsPos+1],
            dataBuffer_D[arrDataSize*activeVertexIdx+arrDataOffsPos+2]);

    int maxIdx = 0;

    // Loop through all adjacent tetrahedrons
    for(int tetrahedronIdx = 0; tetrahedronIdx < 6; ++tetrahedronIdx) {

        // Check whether =? 99 (is tetrahedron neighbour)
        if(VertexNeighbouringTetrahedrons[v][tetrahedronIdx][0] == 99) continue;

        // Get origin of the cell containing the adjacent tetrahedron
        int3 cellOrgTemp = make_int3(
                cellOrg.x + VertexNeighbouringTetrahedrons[v][tetrahedronIdx][0],
                cellOrg.y + VertexNeighbouringTetrahedrons[v][tetrahedronIdx][1],
                cellOrg.z + VertexNeighbouringTetrahedrons[v][tetrahedronIdx][2]);


        // Get tetrahedron flags of the adjacent tetrahedron
        terahedronFlagsTmp = tetrahedronFlags_D(
                make_uint3(cellOrgTemp.x, cellOrgTemp.y, cellOrgTemp.z),
                VertexNeighbouringTetrahedrons[v][tetrahedronIdx][3], isoval, volume_D);


        // Edge index of this vertex in the adjacent tetrahedron
        ownEdgeIdx = VertexNeighbouringTetrahedronsOwnEdgeIdx[v][tetrahedronIdx];

        // Loop both possible triangles
        for(int triangleIdx = 0; triangleIdx < 2; ++triangleIdx) {
            if(tetrahedronTriangles[terahedronFlagsTmp][3*triangleIdx+0] < 0) {
                continue;
            }

            for(int vtx = 0; vtx < 3; vtx++) {

                if(tetrahedronTriangles[terahedronFlagsTmp][triangleIdx*3 + vtx] == ownEdgeIdx) {

                    uint edgeIdx0 = tetrahedronTriangles[terahedronFlagsTmp][triangleIdx*3 + (vtx+1)%3];
                    uint edgeIdx1 = tetrahedronTriangles[terahedronFlagsTmp][triangleIdx*3 + (vtx+2)%3];

                    int3 cubeIdx0 = make_int3(
                            TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx0][0],
                            TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx0][1],
                            TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx0][2]) + cellOrgTemp;

                    int3 cubeIdx1 = make_int3(
                            TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx1][0],
                            TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx1][1],
                            TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx1][2]) + cellOrgTemp;

                    int vertexIdx0 = vertexMapInv_D[cubeMapInv_D[GetCellIdxByGridCoords(cubeIdx0)]*7 +
                                                    TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx0][3]];

                    int vertexIdx1 = vertexMapInv_D[cubeMapInv_D[GetCellIdxByGridCoords(cubeIdx1)]*7 +
                                                    TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx1][3]];

                    maxIdx = max(maxIdx, max(vertexIdx0, vertexIdx1));

                    float3 pos0, pos1;
                    pos0 = make_float3(
                            dataBuffer_D[arrDataSize*vertexIdx0+arrDataOffsPos+0],
                            dataBuffer_D[arrDataSize*vertexIdx0+arrDataOffsPos+1],
                            dataBuffer_D[arrDataSize*vertexIdx0+arrDataOffsPos+2]);
                    pos1 = make_float3(
                            dataBuffer_D[arrDataSize*vertexIdx1+arrDataOffsPos+0],
                            dataBuffer_D[arrDataSize*vertexIdx1+arrDataOffsPos+1],
                            dataBuffer_D[arrDataSize*vertexIdx1+arrDataOffsPos+2]);
                    float3 vec0 = safeNormalize(pos0 - pos);
                    float3 vec1 = safeNormalize(pos1 - pos);

                    normal += cross(vec0, vec1);
                }
            }
        }
    }

    normal = safeNormalize(normal);
    dataBuffer_D[arrDataSize*activeVertexIdx+arrDataOffsNormals+0] = normal.x;
    dataBuffer_D[arrDataSize*activeVertexIdx+arrDataOffsNormals+1] = normal.y;
    dataBuffer_D[arrDataSize*activeVertexIdx+arrDataOffsNormals+2] = normal.z;
}


/*
 * GPUSurfaceMT_ComputeVertexTexCoords_D
 */
__global__ void GPUSurfaceMT_ComputeVertexTexCoords_D(float *dataBuff_D,
        float volMinX, float volMinY, float volMinZ,
        float volMaxX, float volMaxY, float volMaxZ,
        uint activeVertexCnt,
        uint arrDataOffsPos,
        uint arrDataOffsTexCoords,
        uint arrDataSize) {

    // Get thread index
    uint activeVertexIdx = getThreadIdx();
    if (activeVertexIdx >= activeVertexCnt) {
        return;
    }

    dataBuff_D[arrDataSize*activeVertexIdx+arrDataOffsTexCoords+0] =
            (dataBuff_D[arrDataSize*activeVertexIdx+arrDataOffsPos+0] - volMinX) / (volMaxX-volMinX);
    dataBuff_D[arrDataSize*activeVertexIdx+arrDataOffsTexCoords+1] =
            (dataBuff_D[arrDataSize*activeVertexIdx+arrDataOffsPos+1] - volMinY) / (volMaxY-volMinY);
    dataBuff_D[arrDataSize*activeVertexIdx+arrDataOffsTexCoords+2] =
            (dataBuff_D[arrDataSize*activeVertexIdx+arrDataOffsPos+2] - volMinZ) / (volMaxZ-volMinZ);
}


/*
 * GPUSurfaceMT_ComputeVertexTexCoordsOfFittedPos_D
 */
__global__ void GPUSurfaceMT_ComputeVertexTexCoordsOfFittedPos_D(
        float *dataBuff_D,
        float *rotation_D,
        float3 translation,
        float3 centroid,
        float volMinX, float volMinY, float volMinZ,
        float volMaxX, float volMaxY, float volMaxZ,
        uint activeVertexCnt,
        uint arrDataOffsPos,
        uint arrDataOffsTexCoords,
        uint arrDataSize) {

    // Get thread index
    uint activeVertexIdx = getThreadIdx();
    if (activeVertexIdx >= activeVertexCnt) {
        return;
    }

    float3 pos = make_float3(
            dataBuff_D[arrDataSize*activeVertexIdx+arrDataOffsPos+0],
            dataBuff_D[arrDataSize*activeVertexIdx+arrDataOffsPos+1],
            dataBuff_D[arrDataSize*activeVertexIdx+arrDataOffsPos+2]);

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

    dataBuff_D[arrDataSize*activeVertexIdx+arrDataOffsTexCoords+0] =
            (posRot.x - volMinX) / (volMaxX-volMinX);
    dataBuff_D[arrDataSize*activeVertexIdx+arrDataOffsTexCoords+1] =
            (posRot.y - volMinY) / (volMaxY-volMinY);
    dataBuff_D[arrDataSize*activeVertexIdx+arrDataOffsTexCoords+2] =
            (posRot.z - volMinZ) / (volMaxZ-volMinZ);
}


/*
 * GPUSurfaceMT_FlagGridCells_D
 */
__global__ void GPUSurfaceMT_FlagGridCells_D(
        uint* activeCellFlag_D,  // Output
        float *volume_D,         // Input
        float isoval,            // Input
        uint cubeCount) {        // Input

    const uint cellIdx = ::getThreadIdx();

    if (cellIdx >= cubeCount) {
        return;
    }

    const uint3 cellOrg = ::GetGridCoordsByCellIdx(cellIdx);

    // Put into registers by the compiler since the array size is constant
    const float cellVertexOffsets[8][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {1, 1, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 1},
        {1, 1, 1},
        {0, 1, 1}
    };

    // Add vertex states of a cube (0: inactive, 1: active)
    float volSample = ::SampleFieldAt_D<float, false>(cellOrg, volume_D);
    unsigned char cubeFlags = static_cast<uint>(volSample <= isoval);

#pragma unroll
    for (int v = 1; v < 8; ++v) {
        const uint3 pos = make_uint3(
                cellOrg.x + cellVertexOffsets[v][0],
                cellOrg.y + cellVertexOffsets[v][1],
                cellOrg.z + cellVertexOffsets[v][2]);
        volSample = ::SampleFieldAt_D<float, false>(pos, volume_D);
        cubeFlags += (unsigned char)(volSample <= isoval);
    }

    // Reduce vertex states to one cube state
    cubeFlags = (unsigned char)(!((cubeFlags == 8)||(cubeFlags == 0)));

    activeCellFlag_D[cellIdx] = cubeFlags;
}


/**
 * Determines for every tetrahedron the number of active vertices. This is
 * necessary to get the actual triangle count.
 *
 * @param[out] verticesPerTetrahedron_D The number of vertices in every
 *                                      tetrahedron (can be either 0, 3 or 6)
 * @param[in]  cubeMap_D                mapping from active cells' list to
 *                                      global cell list
 * @param[in]  thresholdValue           The isovalue
 * @param[in]  activeCubeCount          The number of active cells
 * @param[in]  volume_D                 The volume the isosurface us extracted
 *                                      from
 */
__global__ void GPUSurfaceMT_FlagTetrahedrons_D(
        uint* verticesPerTetrahedron_D,
        uint* cubeMap_D,
        float thresholdValue,
        uint activeCubeCount,
        float *volume_D) {

    const uint activeCubeIndex = ::getThreadIdx();

    // Load cube vertex offsets into shared memory
    if (threadIdx.x < 32) {
        const int clampedThreadIdx = min(threadIdx.x, 7);
        cubeVertexOffsets_S[clampedThreadIdx][0] = cubeVertexOffsets[clampedThreadIdx][0];
        cubeVertexOffsets_S[clampedThreadIdx][1] = cubeVertexOffsets[clampedThreadIdx][1];
        cubeVertexOffsets_S[clampedThreadIdx][2] = cubeVertexOffsets[clampedThreadIdx][2];
    }

    // Load tetrahedrons in a cube to shared memory
    if (threadIdx.x >= 32 && threadIdx.x < 64) {
        const int clampedThreadIdx = min(threadIdx.x, 37) - 32;
        tetrahedronsInACube_S[clampedThreadIdx][0] = tetrahedronsInACube[clampedThreadIdx][0];
        tetrahedronsInACube_S[clampedThreadIdx][1] = tetrahedronsInACube[clampedThreadIdx][1];
        tetrahedronsInACube_S[clampedThreadIdx][2] = tetrahedronsInACube[clampedThreadIdx][2];
        tetrahedronsInACube_S[clampedThreadIdx][3] = tetrahedronsInACube[clampedThreadIdx][3];
    }
    __syncthreads();

    // Prevent non-power of two writes.
    if (activeCubeIndex >= activeCubeCount) {
        return;
    }
    const uint3 cubeVertex0 = GetGridCoordsByCellIdx(cubeMap_D[activeCubeIndex]);
    // Classify all tetrahedrons in a cube
#pragma unroll
    for (int tetrahedronIndex = 0; tetrahedronIndex < 6; ++tetrahedronIndex) {
        // Compute tetrahedron flags
        unsigned char tetrahedronFlags = tetrahedronFlags_D(cubeVertex0, tetrahedronIndex, thresholdValue, volume_D);
        // Store number of vertices
        verticesPerTetrahedron_D[activeCubeIndex * 6 + tetrahedronIndex] = tetrahedronVertexCount[tetrahedronFlags];
    }
}


/**
 * Obtains the vertex indices for all triangles.
 *
 * @param[in]  vertexOffsets_D Offsets for vertex indices
 * @param[in]  cubeMap_D       Mapping from active cells's list to global cell list
 * @param[in]  cubeMapInv_D    Inverse mapping to cubeMap_D
 * @param[in]  thresholdValue  The isovalue
 * @param[in]  tetrahedronCount The number of tetrahedrons
 * @param[in]  activeCubeCount The number of active cells
 * @param[out] triangleVertexIdx_D The triangles' vertex indices
 * @param[in]  vertexMapInv_D Inverse mapping to vertexOffsets_D
 * @param[in]  volume_D The volume the isosurface is extracted from
 */
__global__ void GPUSurfaceMT_GetTrianglesIdx_D(
        uint* vertexOffsets_D,
        uint* cubeMap_D,
        uint* cubeMapInv_D,
        float thresholdValue,
        uint tetrahedronCount,
        uint activeCubeCount,
        uint *triangleVertexIdx_D,
        uint *vertexMapInv_D,
        float *volume_D) {

    const uint id = getThreadIdx();

    // Load cube vertex offsets into shared memory
    if (threadIdx.x < 32) {
        const int clampedThreadIdx = min(threadIdx.x, 7);
        cubeVertexOffsets_S[clampedThreadIdx][0] = cubeVertexOffsets[clampedThreadIdx][0];
        cubeVertexOffsets_S[clampedThreadIdx][1] = cubeVertexOffsets[clampedThreadIdx][1];
        cubeVertexOffsets_S[clampedThreadIdx][2] = cubeVertexOffsets[clampedThreadIdx][2];
    }

    // Load tetrahedrons in a cube to shared memory
    if (threadIdx.x >= 32 && threadIdx.x < 64) {
        const int clampedThreadIdx = min(threadIdx.x, 37) - 32;
        tetrahedronsInACube_S[clampedThreadIdx][0] = tetrahedronsInACube[clampedThreadIdx][0];
        tetrahedronsInACube_S[clampedThreadIdx][1] = tetrahedronsInACube[clampedThreadIdx][1];
        tetrahedronsInACube_S[clampedThreadIdx][2] = tetrahedronsInACube[clampedThreadIdx][2];
        tetrahedronsInACube_S[clampedThreadIdx][3] = tetrahedronsInACube[clampedThreadIdx][3];
    }
    __syncthreads();

    if (id >= tetrahedronCount) {
        return;
    }

    const uint activeCubeIndex = id / 6;
    const int tetrahedronIndex = id % 6;
    const uint3 cellOrg = ::GetGridCoordsByCellIdx(cubeMap_D[activeCubeIndex]);

    // Get bitmap to classify the tetrahedron
    unsigned char tetrahedronFlags = tetrahedronFlags_D(cellOrg,
            tetrahedronIndex, thresholdValue, volume_D);

    uint vertexBaseOffset;

    /// Compute vertex indices ///

    // First triangle
    if (tetrahedronTriangles[tetrahedronFlags][0] >= 0) {

        vertexBaseOffset = vertexOffsets_D[id];

        // First corner
        int edgeIndex0 = tetrahedronTriangles[tetrahedronFlags][0];
        int vertexIdx0 = getTetrahedronEdgeVertexIdx_D(
                activeCubeIndex, tetrahedronIndex, edgeIndex0, cubeMap_D,
                cubeMapInv_D);

        // Second corner
        int edgeIndex1 = tetrahedronTriangles[tetrahedronFlags][1];
        int vertexIdx1 = getTetrahedronEdgeVertexIdx_D(
                activeCubeIndex, tetrahedronIndex, edgeIndex1, cubeMap_D,
                cubeMapInv_D);

        // Third corner
        int edgeIndex2 = tetrahedronTriangles[tetrahedronFlags][2];
        int vertexIdx2 = getTetrahedronEdgeVertexIdx_D(
                activeCubeIndex, tetrahedronIndex, edgeIndex2, cubeMap_D,
                cubeMapInv_D);

        triangleVertexIdx_D[vertexBaseOffset] = vertexMapInv_D[vertexIdx0];
        triangleVertexIdx_D[vertexBaseOffset + 1] = vertexMapInv_D[vertexIdx1];
        triangleVertexIdx_D[vertexBaseOffset + 2] = vertexMapInv_D[vertexIdx2];
    }

    // Second (possible) triangle
    if (tetrahedronTriangles[tetrahedronFlags][3] >= 0) {

        // First corner
        int edgeIndex0 = tetrahedronTriangles[tetrahedronFlags][3];
        int vertexIdx0 = getTetrahedronEdgeVertexIdx_D(
                activeCubeIndex, tetrahedronIndex, edgeIndex0, cubeMap_D,
                cubeMapInv_D);

        // Second corner
        int edgeIndex1 = tetrahedronTriangles[tetrahedronFlags][4];
        int vertexIdx1 = getTetrahedronEdgeVertexIdx_D(
                activeCubeIndex, tetrahedronIndex, edgeIndex1, cubeMap_D,
                cubeMapInv_D);

        // Third corner
        int edgeIndex2 = tetrahedronTriangles[tetrahedronFlags][5];
        int vertexIdx2 = getTetrahedronEdgeVertexIdx_D(
                activeCubeIndex, tetrahedronIndex, edgeIndex2, cubeMap_D,
                cubeMapInv_D);

        triangleVertexIdx_D[vertexBaseOffset + 5] = vertexMapInv_D[vertexIdx2];
        triangleVertexIdx_D[vertexBaseOffset + 4] = vertexMapInv_D[vertexIdx1];
        triangleVertexIdx_D[vertexBaseOffset + 3] = vertexMapInv_D[vertexIdx0];
    }
}


/*
 * GPUSurfaceMT_RotatePos_D
 */
__global__ void GPUSurfaceMT_RotatePos_D(
        float *vertexData_D,
        uint vertexDataStride,
        uint vertexDataOffsPos,
        float *rotation_D,
        uint vertexCnt) {

    const uint idx = getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }

    const uint vertexDataIdx = vertexDataStride*idx+vertexDataOffsPos;

    float xtemp, ytemp, ztemp;
    xtemp = rotation_D[0] * vertexData_D[vertexDataIdx+0] +
            rotation_D[3] * vertexData_D[vertexDataIdx+1] +
            rotation_D[6] * vertexData_D[vertexDataIdx+2];
    ytemp = rotation_D[1] * vertexData_D[vertexDataIdx+0] +
            rotation_D[4] * vertexData_D[vertexDataIdx+1] +
            rotation_D[7] * vertexData_D[vertexDataIdx+2];
    ztemp = rotation_D[2] * vertexData_D[vertexDataIdx+0] +
            rotation_D[5] * vertexData_D[vertexDataIdx+1] +
            rotation_D[8] * vertexData_D[vertexDataIdx+2];
    vertexData_D[vertexDataIdx+0] = xtemp;
    vertexData_D[vertexDataIdx+1] = ytemp;
    vertexData_D[vertexDataIdx+2] = ztemp;
}


/*
 * GPUSurfaceMT_TranslatePos_D
 */
__global__ void GPUSurfaceMT_TranslatePos_D(
        float *vertexData_D,
        uint vertexDataStride,
        uint vertexDataOffsPos,
        float3 translation,
        uint vertexCnt) {

    const uint idx = getThreadIdx();
    if (idx >= vertexCnt) {
        return;
    }
    const uint vertexDataIdx = vertexDataStride*idx+vertexDataOffsPos;

    vertexData_D[vertexDataIdx+0] += translation.x;
    vertexData_D[vertexDataIdx+1] += translation.y;
    vertexData_D[vertexDataIdx+2] += translation.z;
}


/*
 * GPUSurfaceMT::GPUSurfaceMT
 */
GPUSurfaceMT::GPUSurfaceMT() : AbstractGPUSurface() , neighboursReady(false) {
}


/*
 * GPUSurfaceMT::GPUSurfaceMT
 */
GPUSurfaceMT::GPUSurfaceMT(const GPUSurfaceMT& other) : AbstractGPUSurface(other) {

    // Copy GPU memory

    CudaSafeCall(this->cubeStates_D.Validate(other.cubeStates_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeStates_D.Peek(),
            other.cubeStates_D.PeekConst(),
            this->cubeStates_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->cubeOffsets_D.Validate(other.cubeOffsets_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeOffsets_D.Peek(),
            other.cubeOffsets_D.PeekConst(),
            this->cubeOffsets_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->cubeMap_D.Validate(other.cubeMap_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeMap_D.Peek(),
            other.cubeMap_D.PeekConst(),
            this->cubeMap_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexStates_D.Validate(other.vertexStates_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexStates_D.Peek(),
            other.vertexStates_D.PeekConst(),
            this->vertexStates_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->activeVertexPos_D.Validate(other.activeVertexPos_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->activeVertexPos_D.Peek(),
            other.activeVertexPos_D.PeekConst(),
            this->activeVertexPos_D.GetCount()*sizeof(float3),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexIdxOffs_D.Validate(other.vertexIdxOffs_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexIdxOffs_D.Peek(),
            other.vertexIdxOffs_D.PeekConst(),
            this->vertexIdxOffs_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexMap_D.Validate(other.vertexMap_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexMap_D.Peek(),
            other.vertexMap_D.PeekConst(),
            this->vertexMap_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexNeighbours_D.Validate(other.vertexNeighbours_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexNeighbours_D.Peek(),
            other.vertexNeighbours_D.PeekConst(),
            this->vertexNeighbours_D.GetCount()*sizeof(int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->verticesPerTetrahedron_D.Validate(other.verticesPerTetrahedron_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->verticesPerTetrahedron_D.Peek(),
            other.verticesPerTetrahedron_D.PeekConst(),
            this->verticesPerTetrahedron_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->tetrahedronVertexOffsets_D.Validate(other.tetrahedronVertexOffsets_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->tetrahedronVertexOffsets_D.Peek(),
            other.tetrahedronVertexOffsets_D.PeekConst(),
            this->tetrahedronVertexOffsets_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->triangleCamDistance_D.Validate(other.triangleCamDistance_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->triangleCamDistance_D.Peek(),
            other.triangleCamDistance_D.PeekConst(),
            this->triangleCamDistance_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    // The number of active cells
    this->activeCellCnt = other.activeCellCnt;

    // Check whether neighbors have been computed
    this->neighboursReady = other.neighboursReady;
}


/*
 * GPUSurfaceMT::~GPUSurfaceMT
 */
GPUSurfaceMT::~GPUSurfaceMT() {
}


/*
 * GPUSurfaceMT::ComputeConnectivity
 */
bool GPUSurfaceMT::ComputeConnectivity(
        float *volume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue) {

    CheckForCudaErrorSync();

    using namespace vislib::sys;

    /// Init grid parameters for all files ///

    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    /* Compute neighbours */

    CheckForCudaErrorSync();

    if (!CudaSafeCall(vertexNeighbours_D.Validate(this->vertexCnt*18))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: could not allocate device memory",
                this->ClassName());
        return false;
    }
    CheckForCudaErrorSync();
    //if (!CudaSafeCall(vertexNeighbours_D.Set(-1))) {
    if (!CudaSafeCall(vertexNeighbours_D.Set(0xff))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: could not init device memory",
                this->ClassName());
        return false;
    }
    CheckForCudaErrorSync();

    const uint blockSize = 192; // == 6 * 32, 32 = warpsize

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    GPUSurfaceMT_ComputeVertexConnectivity_D <<< Grid(this->vertexCnt*6, blockSize), blockSize >>> (
            this->vertexNeighbours_D.Peek(),
            this->vertexStates_D.Peek(),
            this->vertexCnt,
            this->vertexMap_D.Peek(),
            this->vertexIdxOffs_D.Peek(),
            this->cubeMap_D.Peek(),
            this->cubeOffsets_D.Peek(),
            this->cubeStates_D.Peek(),
            volume_D,
            isovalue);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVertexConnectivity_D' :          %.10f sec\n",
            dt_ms/1000.0);
#endif

    CheckForCudaErrorSync();

    this->neighboursReady = true;
    return true;
}


/*
 * GPUSurfaceMT::ComputeEdgeList
 */
bool GPUSurfaceMT::ComputeEdgeList(
        float *volume_D,
        float isoval,
        int3 volDim,
        float3 volOrg,
        float3 volDelta) {

    using namespace vislib::sys;

    const uint blockSize = 256;
    //const uint cellCnt = (volDim.x-1)*(volDim.y-1)*(volDim.z-1);

    /* Init grid parameters for all files */

    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName(),
                __FILE__,
                __LINE__);
        return false;
    }


    /* Obtain number of edges associated with each cell */

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    if (this->cubeMap_D.GetCount() != this->activeCellCnt) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: need cube map to compute edges (%s:%i)",
                this->ClassName(),
                __FILE__,
                __LINE__);
        return false;
    }

    //(Re-)allocate memory to count number of edges per cell
    if (!CudaSafeCall(this->edgesPerTetrahedron_D.Validate(this->activeCellCnt*6))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not allocate device memory (%s:%i)",
                this->ClassName(),
                __FILE__,
                __LINE__);
        return false;
    }
    // Init with zero
    if (!CudaSafeCall(this->edgesPerTetrahedron_D.Set(0x00))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init device memory (%s:%i)",
                this->ClassName(),
                __FILE__,
                __LINE__);
        return false;
    }

    GPUSurfaceMT_ComputeEdgesPerTetrahedron_D <<< Grid(this->activeCellCnt*6, blockSize), blockSize >>> (
           this->edgesPerTetrahedron_D.Peek(),
           volume_D,
           this->cubeMap_D.Peek(),
           isoval,
           this->activeCellCnt);

    if (!CheckForCudaError()) {
        return false;
    }

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'GPUSurfaceMT_ComputeEdgesPerCell_D' :   %.10f sec\n",
            dt_ms/1000.0);
    cudaEventRecord(event1, 0);
#endif

    /* Compute prefix sum to obtain index offsets for edges */

    //(Re-)allocate memory to count number of edges per cell
    if (!CudaSafeCall(this->tetraEdgeIdxOffsets_D.Validate(this->activeCellCnt*6))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not allocate device memory (%s:%i)",
                this->ClassName(),
                __FILE__,
                __LINE__);
        return false;
    }
    thrust::exclusive_scan(
            thrust::device_ptr<unsigned int>(this->edgesPerTetrahedron_D.Peek()),
            thrust::device_ptr<unsigned int>(this->edgesPerTetrahedron_D.Peek() + 6*this->activeCellCnt),
            thrust::device_ptr<unsigned int>(this->tetraEdgeIdxOffsets_D.Peek()));

    if (!CheckForCudaError()) {
        return false;
    }

//    // DEBUG print prefix list
//    HostArr<unsigned int> tetraEdgeIdxOffsets;
//    tetraEdgeIdxOffsets.Validate(this->tetraEdgeIdxOffsets_D.GetCount());
//    if (!CudaSafeCall(this->tetraEdgeIdxOffsets_D.CopyToHost(tetraEdgeIdxOffsets.Peek()))){
//        return false;
//    }
//    HostArr<unsigned int> edgesPerTetrahedron;
//    edgesPerTetrahedron.Validate(this->edgesPerTetrahedron_D.GetCount());
//    if (!CudaSafeCall(this->edgesPerTetrahedron_D.CopyToHost(edgesPerTetrahedron.Peek()))){
//        return false;
//    }
//    for (int e = 0; e < 6*this->activeCellCnt; ++e) {
//        printf("%i: edgeCnt %u, index %u\n", e,
//                edgesPerTetrahedron.Peek()[e],
//                tetraEdgeIdxOffsets.Peek()[e]);
//    }
//    tetraEdgeIdxOffsets.Release();
//    edgesPerTetrahedron.Release();
//    // END DEBUG

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'thrust::exclusive_scan' :               %.10f sec\n",
            dt_ms/1000.0);
    cudaEventRecord(event1, 0);
#endif

    // Compute actual edges
    unsigned int edgeCnt = this->triangleCnt*3/2;

    //(Re-)allocate memory to count number of edges per cell
    if (!CudaSafeCall(this->edges_D.Validate(edgeCnt*2))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not allocate device memory (%s:%i)",
                this->ClassName(),
                __FILE__,
                __LINE__);
        return false;
    }
    // Init with zero
    if (!CudaSafeCall(this->edges_D.Set(0x00))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init device memory (%s:%i)",
                this->ClassName(),
                __FILE__,
                __LINE__);
        return false;
    }

    GPUSurfaceMT_ComputeEdgeList_D <<< Grid(this->activeCellCnt*6, blockSize), blockSize >>> (
            this->edges_D.Peek(),
            this->tetraEdgeIdxOffsets_D.Peek(),
            this->edgesPerTetrahedron_D.Peek(),
            this->vertexIdxOffs_D.Peek(),
            this->cubeMap_D.Peek(),
            this->cubeOffsets_D.Peek(),
            isoval,
            volume_D,
            this->activeCellCnt);

    if (!CheckForCudaErrorSync()) {
        return false;
    }

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'GPUSurfaceMT_ComputeEdgeList_D' :       %.10f sec\n",
            dt_ms/1000.0);
#endif

    this->edges.Validate(this->edges_D.GetCount());
    if (!CudaSafeCall(this->edges_D.CopyToHost(edges.Peek()))){
        return false;
    }

//    // DEBUG print edge list
//    for (int e = 0; e < edgeCnt; ++e) {
//        printf("EDGE %i: %u %u\n", e, this->edges.Peek()[2*e+0], this->edges.Peek()[2*e+1]);
//    }
//    // END DEBUG

    return true;
}




/*
 * GPUSurfaceMT::computeVertexNormals
 */
bool GPUSurfaceMT::ComputeNormals(
        float *volume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue) {

    using vislib::sys::Log;

    if (!this->triangleIdxReady) { // We need the triangles mesh info
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: triangles not computed",
                this->ClassName());
        return false;
    }

    /// Init grid parameters ///

    // Init constant device params
    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not register vertex buffer",
                this->ClassName());

        return false;
    }

    // Get mapped pointer to the vbo
    float *vboPt_D;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not map resources",
                this->ClassName());
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt_D), // The mapped pointer
            &vboSize,             // The size of the accessible data
            this->vertexDataResource))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not acquire mapped pointer",
                this->ClassName());
        return false;
    }

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    GPUSurfaceMT_ComputeVertexNormals_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            vboPt_D,
            this->vertexMap_D.Peek(),
            this->vertexIdxOffs_D.Peek(),
            this->cubeMap_D.Peek(),
            this->cubeOffsets_D.Peek(),
            volume_D,
            isovalue,
            this->vertexCnt,
            this->vertexDataOffsPos,
            this->vertexDataOffsNormal,
            this->vertexDataStride);


#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVertexNormals_D' :               %.10f sec\n",
            dt_ms/1000.0);
#endif

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not unmap resources",
                this->ClassName());
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not unregister buffers",
                this->ClassName());
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::ComputeTexCoords
 */
bool GPUSurfaceMT::ComputeTexCoords(float minCoords[3], float maxCoords[3]) {
    if (!this->triangleIdxReady) { // We need the triangles mesh info
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0));
    CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,             // The size of the accessible data
            this->vertexDataResource));                   // The mapped resource


    GPUSurfaceMT_ComputeVertexTexCoords_D <<< Grid(this->vertexCnt, 256), 256 >>>(
            vboPt,
            minCoords[0],
            minCoords[1],
            minCoords[2],
            maxCoords[0],
            maxCoords[1],
            maxCoords[2],
            this->vertexCnt,
            this->vertexDataOffsPos,
            this->vertexDataOffsTexCoord,
            this->vertexDataStride);

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }

    return true;
}


/*
 * GPUSurfaceMT::ComputeTexCoordsOfRMSDFittedPositions
 */
bool GPUSurfaceMT::ComputeTexCoordsOfRMSDFittedPositions(
        float minCoords[3],
        float maxCoords[3],
        float centroid[3],
        float rotMat[9],
        float transVec[3]) {

    CudaDevArr<float> rotate_D;

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0));
    CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,             // The size of the accessible data
            this->vertexDataResource));                   // The mapped resource

    // Rotate for best fit
    rotate_D.Validate(9);
    if (!CudaSafeCall(cudaMemcpy((void *)rotate_D.Peek(), &rotMat[0],
            9*sizeof(float), cudaMemcpyHostToDevice))) {
        return false;
    }

    GPUSurfaceMT_ComputeVertexTexCoordsOfFittedPos_D <<< Grid(this->vertexCnt, 256), 256 >>>(
            vboPt,
            rotate_D.Peek(),
            make_float3(transVec[0],transVec[1],transVec[2]),
            make_float3(centroid[0],centroid[1],centroid[2]),
            minCoords[0],
            minCoords[1],
            minCoords[2],
            maxCoords[0],
            maxCoords[1],
            maxCoords[2],
            this->vertexCnt,
            this->vertexDataOffsPos,
            this->vertexDataOffsTexCoord,
            this->vertexDataStride);

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }

    if (!CudaSafeCall(rotate_D.Release())) {
        return false;
    }

    return true;

}


/*
 * GPUSurfaceMT::computeTriangles
 */
bool GPUSurfaceMT::ComputeTriangles(
        float *volume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue) {

    using namespace vislib::sys;

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
#endif

    /// Init grid parameters ///

    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

    if (!this->vertexDataReady) { // We need vertex data to generate triangles
        return false;
    }

    size_t triangleVtxCnt;


    /// Calc vertex index map ///

    if (!CudaSafeCall(this->vertexMap_D.Validate(this->vertexCnt))) {
        return false;
    }

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    GPUSurfaceMT_CalcVertexMap_D <<< Grid(7*this->activeCellCnt, 256), 256 >>> (
            this->vertexMap_D.Peek(),
            this->vertexIdxOffs_D.Peek(),
            this->vertexStates_D.Peek(),
             7*this->activeCellCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'CalcVertexMap_D' :                      %.10f sec\n",
            dt_ms/1000.0);
#endif

    if (!::CheckForCudaErrorSync()) {
        return false;
    }


    /// Flag tetrahedrons ///

    if (!CudaSafeCall(this->verticesPerTetrahedron_D.Validate(6*this->activeCellCnt))) {
        return false;
    }

#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    GPUSurfaceMT_FlagTetrahedrons_D <<< Grid(this->activeCellCnt, 256), 256 >>> (
            this->verticesPerTetrahedron_D.Peek(),
            this->cubeMap_D.Peek(),
            isovalue,
            this->activeCellCnt,
            volume_D);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'FlagTetrahedrons_D' :                   %.10f sec\n",
            dt_ms/1000.0);
#endif

    if (!::CheckForCudaErrorSync()) {
        return false;
    }


    /// Scan tetrahedrons ///

    if (!CudaSafeCall(this->tetrahedronVertexOffsets_D.Validate(6*this->activeCellCnt))) {
        return false;
    }

#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    thrust::exclusive_scan(
            thrust::device_ptr<uint>(this->verticesPerTetrahedron_D.Peek()),
            thrust::device_ptr<uint>(this->verticesPerTetrahedron_D.Peek() + 6*this->activeCellCnt),
            thrust::device_ptr<uint>(this->tetrahedronVertexOffsets_D.Peek()));

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'thrust::exclusive_scan' :               %.10f sec\n",
            dt_ms/1000.0);
#endif

    if (!::CheckForCudaErrorSync()) {
        return false;
    }


    /// Get triangle vertex count ///

    triangleVtxCnt =
            this->tetrahedronVertexOffsets_D.GetAt(this->activeCellCnt*6-1) +
            this->verticesPerTetrahedron_D.GetAt(this->activeCellCnt*6-1);

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

    this->triangleCnt = triangleVtxCnt/3;


    /// Create vertex buffer object and register with CUDA ///

    // Create empty vbo to hold the triangle indices
    if (!this->InitTriangleIdxVBO(this->triangleCnt)) {
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->triangleIdxResource,
            this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    unsigned int *vboTriangleIdxPt;
    size_t vboTriangleIdxSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->triangleIdxResource, 0))) {

        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
            return false;
        }
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboTriangleIdxPt), // The mapped pointer
            &vboTriangleIdxSize,             // The size of the accessible data
            this->triangleIdxResource))) {                   // The mapped resource

        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->triangleIdxResource))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
            return false;
        }
        return false;
    }


    /// Generate triangles ///

    if (!CudaSafeCall(cudaMemset(vboTriangleIdxPt, 0x00, vboTriangleIdxSize))) {

        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->triangleIdxResource))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
            return false;
        }
        return false;
    }

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    GPUSurfaceMT_GetTrianglesIdx_D <<< Grid(this->activeCellCnt*6, 256), 256 >>> (
            this->tetrahedronVertexOffsets_D.Peek(),
            this->cubeMap_D.Peek(),
            this->cubeOffsets_D.Peek(),
            isovalue,
            this->activeCellCnt*6,
            this->activeCellCnt,
            vboTriangleIdxPt,
            this->vertexIdxOffs_D.Peek(),
            volume_D);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'GetTrianglesIdx_D' :                    %.10f sec\n",
            dt_ms/1000.0);
#endif

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

//    // DEBUG print triangle indices
//    HostArr<unsigned int> vboTriangleIdx;
//    vboTriangleIdx.Validate(this->triangleCnt*3);
//    cudaMemcpy(vboTriangleIdx.Peek(), vboTriangleIdxPt, vboTriangleIdxSize, cudaMemcpyDeviceToHost);
//    for (int t = 0; t < this->triangleCnt; ++t) {
//        printf("TRIANGLE %i: %u %u %u\n", t, vboTriangleIdx.Peek()[3*t+0],
//                vboTriangleIdx.Peek()[3*t+1], vboTriangleIdx.Peek()[3*t+2]);
//    }
//    // END DEBUG

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->triangleIdxResource))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::ComputeVertexPositions
 */
bool GPUSurfaceMT::ComputeVertexPositions(
        float *volume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue) {

    using vislib::sys::Log;

    size_t gridCellCnt = (volDim.x-1)*(volDim.y-1)*(volDim.z-1);

#ifdef USE_TIMER
    cudaEvent_t event1, event2;
    float dt_ms;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
#endif


    /// Init grid parameters ///

    if (!initGridParams(volDim, volOrg, volDelta)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init constant device params",
                this->ClassName());
        return false;
    }

    if (!::CheckForCudaErrorSync()) {
        return false;
    }


    /// Find active grid cells ///

    if (!CudaSafeCall(this->cubeStates_D.Validate(gridCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->cubeOffsets_D.Validate(gridCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->cubeStates_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(this->cubeOffsets_D.Set(0x00))) {
        return false;
    }

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    GPUSurfaceMT_FlagGridCells_D <<< Grid(gridCellCnt, 256), 256 >>> (
            this->cubeStates_D.Peek(),
            volume_D,
            isovalue,
            gridCellCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'FlagGridCells_D' :                      %.10f sec\n",
            dt_ms/1000.0);
#endif

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    thrust::exclusive_scan(
            thrust::device_ptr<uint>(this->cubeStates_D.Peek()),
            thrust::device_ptr<uint>(this->cubeStates_D.Peek() + gridCellCnt),
            thrust::device_ptr<uint>(this->cubeOffsets_D.Peek()));

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'thrust::exclusive_scan' :               %.10f sec\n",
            dt_ms/1000.0);
#endif


    /// Get number of active grid cells ///

    this->activeCellCnt =
            this->cubeStates_D.GetAt(gridCellCnt-1) +
            this->cubeOffsets_D.GetAt(gridCellCnt-1);

    if (!::CheckForCudaErrorSync()) {
        return false;
    }


    /// Prepare cube map ///

    if (!CudaSafeCall(this->cubeMap_D.Validate(this->activeCellCnt))) {
        return false;
    }

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    GPUSurfaceMT_CalcCubeMap_D <<< Grid(gridCellCnt, 256), 256 >>> (
            this->cubeMap_D.Peek(),
            this->cubeOffsets_D.Peek(),
            this->cubeStates_D.Peek(),
            gridCellCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'CalcCubeMap_D' :                        %.10f sec\n",
            dt_ms/1000.0);
#endif

    if (!::CheckForCudaErrorSync()) {
        return false;
    }


    /// Get vertex positions ///

    if (!CudaSafeCall(this->vertexStates_D.Validate(7*this->activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->activeVertexPos_D.Validate(7*this->activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->vertexIdxOffs_D.Validate(7*this->activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->vertexStates_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(this->activeVertexPos_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(this->vertexIdxOffs_D.Set(0x00))) {
        return false;
    }

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

    const uint threadPerBlock = 128;

#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    GPUSurfaceMT_CalcVertexPositions_D <<< Grid(this->activeCellCnt*6, threadPerBlock ), threadPerBlock  >>> (
            this->vertexStates_D.Peek(),
            this->activeVertexPos_D.Peek(),
            this->cubeMap_D.Peek(),
            isovalue,
            this->activeCellCnt,
            volume_D
    );

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'CalcVertexPositions_D' :                %.10f sec\n",
            dt_ms/1000.0);
#endif

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

    thrust::exclusive_scan(
            thrust::device_ptr<uint>(this->vertexStates_D.Peek()),
            thrust::device_ptr<uint>(this->vertexStates_D.Peek() + 7*this->activeCellCnt),
            thrust::device_ptr<uint>(this->vertexIdxOffs_D.Peek()));

    if (!::CheckForCudaErrorSync()) {
        return false;
    }


    //// Get number of active vertices ///

    this->vertexCnt =
            this->vertexStates_D.GetAt(7*this->activeCellCnt-1) +
            this->vertexIdxOffs_D.GetAt(7*this->activeCellCnt-1);

    if (!::CheckForCudaErrorSync()) {
        return false;
    }


    //// Create vertex buffer object and register with CUDA ///

    // Create empty vbo to hold vertex data for the surface
    if (!this->InitVertexDataVBO(this->vertexCnt)) {
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,             // The size of the accessible data
            this->vertexDataResource))) {                   // The mapped resource
        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
            return false;
        }
        return false;
    }

    // Init with zeros
    if (!CudaSafeCall(cudaMemset(vboPt, 0, vboSize))) {
        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
            return false;
        }
        return false;
    }


#ifdef USE_TIMER
    cudaEventRecord(event1, 0);
#endif

    GPUSurfaceMT_CompactActiveVertexPositions_D <<< Grid(this->activeCellCnt*7, 256), 256 >>> (
            vboPt,
            this->vertexStates_D.Peek(),
            this->vertexIdxOffs_D.Peek(),
            this->activeVertexPos_D.Peek(),
            this->activeCellCnt*7,
            this->vertexDataOffsPos,
            this->vertexDataStride);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'CompactActiveVertexPositions_D' :       %.10f sec\n",
            dt_ms/1000.0);
#endif

    if (!::CheckForCudaErrorSync()) {
        return false;
    }

    /// Unmap CUDA graphics resource ///

    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }

    return ::CheckForCudaErrorSync();
}


/*
 * GPUSurfaceMT::Rotate
 */
bool GPUSurfaceMT::Rotate(float rotMat[9]) {
    CudaDevArr<float> rotate_D;

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,                         // The size of the accessible data
            this->vertexDataResource))) {     // The mapped resource
        return false;
    }

    // Rotate for best fit
    rotate_D.Validate(9);
    if (!CudaSafeCall(cudaMemcpy((void *)rotate_D.Peek(), &rotMat[0],
            9*sizeof(float), cudaMemcpyHostToDevice))) {
        return false;
    }

#ifdef USE_TIMER
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Initialize triangle index array
    GPUSurfaceMT_RotatePos_D <<< Grid(this->vertexCnt, 256), 256 >>> (
            vboPt,
            this->vertexDataStride,
            this->vertexDataOffsPos,
            rotate_D.Peek(),
            this->vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Kernel execution time 'RotatePos_D':                      %f sec\n", dt_ms/1000.0f);
#endif

    // Clean up
    rotate_D.Release();

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }

    return true;
}


/*
 * GPUSurfaceMT::SortTrianglesByCamDist
 */
bool GPUSurfaceMT::SortTrianglesByCamDist(float camPos[3]) {

    if (!CudaSafeCall(this->triangleCamDistance_D.Validate(triangleCnt))) {
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->triangleIdxResource,
            this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // We need both cuda graphics resources to be mapped at the same time
    cudaGraphicsResource *cudaToken[2];
    cudaToken[0] = this->vertexDataResource;
    cudaToken[1] = this->triangleIdxResource;
    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaToken, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data and the triangle indices
    float *vboPt;
    uint *vboTriangleIdxPt;
    size_t vboSize, vboTriangleIdxSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,                         // The size of the accessible data
            cudaToken[0]))) {                 // The mapped resource
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboTriangleIdxPt), // The mapped pointer
            &vboTriangleIdxSize,              // The size of the accessible data
            cudaToken[1]))) {                 // The mapped resource
        return false;
    }

    if (!CudaSafeCall(SortTrianglesByCamDistance(
            vboPt,
            this->vertexDataStride,
            this->vertexDataOffsPos,
            make_float3(camPos[0], camPos[1], camPos[2]),
            vboTriangleIdxPt,
            this->triangleCnt,
            this->triangleCamDistance_D.Peek()))) {

        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaToken, 0))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
            return false;
        }

        return false;
    }

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaToken, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
        return false;
    }


    return true;
}


/*
 * GPUSurfaceMT::Translate
 */
bool GPUSurfaceMT::Translate(float transVec[3]) {

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,                         // The size of the accessible data
            this->vertexDataResource))) {     // The mapped resource
        return false;
    }

#ifdef USE_TIMER
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Initialize triangle index array
    GPUSurfaceMT_TranslatePos_D <<< Grid(vertexCnt, 256), 256 >>> (
            vboPt,
            this->vertexDataStride,
            this->vertexDataOffsPos,
            make_float3(transVec[0], transVec[1], transVec[2]),
            this->vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Kernel execution time 'Translateos_D': %f sec\n", dt_ms/1000.0f);
#endif

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }

    return true;
}


/*
 * GPUSurfaceMT::Release
 */
void GPUSurfaceMT::Release() {
    AbstractGPUSurface::Release();
    CudaSafeCall(this->cubeStates_D.Release());
    CudaSafeCall(this->cubeOffsets_D.Release());
    CudaSafeCall(this->cubeMap_D.Release());
    CudaSafeCall(this->vertexStates_D.Release());
    CudaSafeCall(this->activeVertexPos_D.Release());
    CudaSafeCall(this->vertexIdxOffs_D.Release());
    CudaSafeCall(this->vertexMap_D.Release());
    CudaSafeCall(this->vertexNeighbours_D.Release());
    CudaSafeCall(this->verticesPerTetrahedron_D.Release());
    CudaSafeCall(this->tetrahedronVertexOffsets_D.Release());
    CudaSafeCall(this->triangleCamDistance_D.Release());
    CudaSafeCall(this->edges_D.Release());
    CudaSafeCall(this->edgesPerTetrahedron_D.Release());
    CudaSafeCall(this->tetraEdgeIdxOffsets_D.Release());
    CudaSafeCall(this->triangleNeighbors_D.Release());
    this->edges.Release();
    this->neighboursReady = false;
    this->activeCellCnt = 0;
}

/*
 * GPUSurfaceMT_SetSubdivVertexFlag
 */
__global__ void GPUSurfaceMT_SetSubdivVertexFlag(
        bool *subdivFlag_D,    // Output
        float *vertexBuffer_D, // Input
        uint *edges_D,         // Input
        float maxEdgeLenSqrt,  // Input
        uint edgeCnt)          // Input
{
    const uint vertexBufferStride = 9;
    const uint vertexBufferPosOffs = 0;

    const uint edgeIdx = ::getThreadIdx();
    if (edgeIdx >= edgeCnt) return;

    uint idx0 = edges_D[2*edgeIdx+0];
    uint idx1 = edges_D[2*edgeIdx+1];

    float3 pos0;
    pos0.x = vertexBuffer_D[vertexBufferStride*idx0 + vertexBufferPosOffs + 0];
    pos0.y = vertexBuffer_D[vertexBufferStride*idx0 + vertexBufferPosOffs + 1];
    pos0.z = vertexBuffer_D[vertexBufferStride*idx0 + vertexBufferPosOffs + 2];

    float3 pos1;
    pos1.x = vertexBuffer_D[vertexBufferStride*idx1 + vertexBufferPosOffs + 0];
    pos1.y = vertexBuffer_D[vertexBufferStride*idx1 + vertexBufferPosOffs + 1];
    pos1.z = vertexBuffer_D[vertexBufferStride*idx1 + vertexBufferPosOffs + 2];

    float lenSqrt = (pos0.x-pos1.x)*(pos0.x-pos1.x) +
                    (pos0.y-pos1.y)*(pos0.y-pos1.y) +
                    (pos0.z-pos1.z)*(pos0.z-pos1.z);

    if (lenSqrt > maxEdgeLenSqrt) {
        subdivFlag_D[idx0] = true;
        subdivFlag_D[idx1] = true;
    }

    if (edgeIdx < 1) {
        subdivFlag_D[idx0] = true;
        subdivFlag_D[idx1] = true;
    }

}


/*
 * GPUSurfaceMT::operator=
 */
GPUSurfaceMT& GPUSurfaceMT::operator=(const GPUSurfaceMT &rhs) {
    AbstractGPUSurface::operator=(rhs);

    // Copy GPU memory

    CudaSafeCall(this->cubeStates_D.Validate(rhs.cubeStates_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeStates_D.Peek(),
            rhs.cubeStates_D.PeekConst(),
            this->cubeStates_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->cubeOffsets_D.Validate(rhs.cubeOffsets_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeOffsets_D.Peek(),
            rhs.cubeOffsets_D.PeekConst(),
            this->cubeOffsets_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->cubeMap_D.Validate(rhs.cubeMap_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeMap_D.Peek(),
            rhs.cubeMap_D.PeekConst(),
            this->cubeMap_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexStates_D.Validate(rhs.vertexStates_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexStates_D.Peek(),
            rhs.vertexStates_D.PeekConst(),
            this->vertexStates_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->activeVertexPos_D.Validate(rhs.activeVertexPos_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->activeVertexPos_D.Peek(),
            rhs.activeVertexPos_D.PeekConst(),
            this->activeVertexPos_D.GetCount()*sizeof(float3),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexIdxOffs_D.Validate(rhs.vertexIdxOffs_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexIdxOffs_D.Peek(),
            rhs.vertexIdxOffs_D.PeekConst(),
            this->vertexIdxOffs_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexMap_D.Validate(rhs.vertexMap_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexMap_D.Peek(),
            rhs.vertexMap_D.PeekConst(),
            this->vertexMap_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexNeighbours_D.Validate(rhs.vertexNeighbours_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexNeighbours_D.Peek(),
            rhs.vertexNeighbours_D.PeekConst(),
            this->vertexNeighbours_D.GetCount()*sizeof(int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->verticesPerTetrahedron_D.Validate(rhs.verticesPerTetrahedron_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->verticesPerTetrahedron_D.Peek(),
            rhs.verticesPerTetrahedron_D.PeekConst(),
            this->verticesPerTetrahedron_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->tetrahedronVertexOffsets_D.Validate(rhs.tetrahedronVertexOffsets_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->tetrahedronVertexOffsets_D.Peek(),
            rhs.tetrahedronVertexOffsets_D.PeekConst(),
            this->tetrahedronVertexOffsets_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->edges_D.Validate(rhs.edges_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->edges_D.Peek(),
            rhs.edges_D.PeekConst(),
            this->edges_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    this->edges.Validate(rhs.edges.GetCount());
    memcpy(this->edges.Peek(),
            rhs.edges.PeekConst(),
            this->edges.GetCount()*sizeof(unsigned int));

    CudaSafeCall(this->edgesPerTetrahedron_D.Validate(rhs.edgesPerTetrahedron_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->edgesPerTetrahedron_D.Peek(),
            rhs.edgesPerTetrahedron_D.PeekConst(),
            this->edgesPerTetrahedron_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->tetraEdgeIdxOffsets_D.Validate(rhs.tetraEdgeIdxOffsets_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->tetraEdgeIdxOffsets_D.Peek(),
            rhs.tetraEdgeIdxOffsets_D.PeekConst(),
            this->tetraEdgeIdxOffsets_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->triangleNeighbors_D.Validate(rhs.triangleNeighbors_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->triangleNeighbors_D.Peek(),
            rhs.triangleNeighbors_D.PeekConst(),
            this->triangleNeighbors_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    // The number of active cells
    this->activeCellCnt = rhs.activeCellCnt;

    /// Flag whether the neighbors have been computed
    this->neighboursReady = rhs.neighboursReady;

    return *this;

}
