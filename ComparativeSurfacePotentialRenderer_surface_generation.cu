//
// ComparativeSurfacePotentialRenderer_surface_generation.cu
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
#include "cuda_error_check.h"
#include <cstdio>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

// Toggle performance measurement and respective messages
//#define USE_TIMER

// The number of threads per block used in GenerateTriangles_D
#define GET_TRIANGLE_IDX_BLOCKSIZE 128

// Shut up eclipse syntax error highlighting
#ifdef __CDT_PARSER__
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __host__
#endif

// Initialize constant values. This needs to be done separately for every file,
// since constant variables are only visible at file scope
extern "C"
cudaError InitVolume_surface_generation(uint3 gridSize, float3 org, float3 delta) {
    cudaMemcpyToSymbol(gridSize_D, &gridSize, sizeof(uint3));
    cudaMemcpyToSymbol(gridOrg_D, &org, sizeof(float3));
    cudaMemcpyToSymbol(gridDelta_D, &delta, sizeof(float3));
    return cudaGetLastError();
}


/**
 * Setup mapping from the list containing all cells to the list containing only
 * active cells.
 *
 * @param[out] cubeMap_D     The mapping from the cell list to the active cells'
 *                           list
 * @param[out] cubeMapInv_D  The mapping from the active cells' list to the
 *                           global cell list
 * @param[in]  cubeOffs_D    Index of the cells in the active cell's list
 * @param[in]  cubeStates_D  The flags of the cells
 * @param[in]  cubeCount     The number of cells to be processed
 */
// TODO cubemapInv_D is pointless, since it contains the same information as
//      cubeOffs_D
__global__ void CalcCubeMap_D(
        uint* cubeMap_D,     // output
        uint* cubeMapInv_D,  // output
        uint* cubeOffs_D,    // input
        uint* cubeStates_D,  // input
        uint cubeCount) {    // input

    const uint cubeIndex = ::GetThreadIndex();
    if (cubeIndex >= cubeCount) {
        return;
    }

    if(cubeStates_D[cubeIndex] != 0) {
        // Map from active cubes list to cube index
        cubeMap_D[cubeOffs_D[cubeIndex]] = cubeIndex;
        cubeMapInv_D[cubeIndex] = cubeOffs_D[cubeIndex];
    }
}


/*
 * CalcCubeMap
 */
extern "C"
cudaError CalcCubeMap(
        uint *cubeMap_D,
        uint *cubeMapInv_D,
        uint *cubeOffs_D,
        uint *cubeStates_D,
        uint cubeCount) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    CalcCubeMap_D <<< Grid(cubeCount, 256), 256 >>> (
            cubeMap_D,
            cubeMapInv_D,
            cubeOffs_D,
            cubeStates_D,
            cubeCount);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'CalcCubeMap_D' :                        %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();
}



__global__ void FindActiveGridCells_D(
        uint* activeCellFlag_D,  // Output
        float *volume_D,         // Input
        float isoval,            // Input
        uint cubeCount) {        // Input

    const uint cellIdx = ::GetThreadIndex();

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
    float volSample = ::SampleFieldAt_D<float>(cellOrg, volume_D);
    unsigned char cubeFlags = static_cast<uint>(volSample <= isoval);

#pragma unroll
    for (int v = 1; v < 8; ++v) {
        const uint3 pos = make_uint3(
                cellOrg.x + cellVertexOffsets[v][0],
                cellOrg.y + cellVertexOffsets[v][1],
                cellOrg.z + cellVertexOffsets[v][2]);
        volSample = ::SampleFieldAt_D<float>(pos, volume_D);
        cubeFlags |= static_cast<uint>(volSample <= isoval) * (1 << v);
    }

    // Reduce vertex states to one cube state
    activeCellFlag_D[cellIdx] = min(cubeFlags % 255, 1);
}


extern "C"
cudaError FindActiveGridCells(uint *cubeStates_D, uint *cubeOffs_D,
        uint cubeCount, float isoval, float *volume_D) {

#ifdef USE_TIMER
    cudaEvent_t event1, event2;
    float dt_ms;
    //Create events
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    //Record events around kernel launch
    cudaEventRecord(event1, 0); //where 0 is the default stream
#endif

    // Classify cells
    FindActiveGridCells_D <<< Grid(cubeCount, 256), 256 >>> (
            cubeStates_D,
            volume_D,
            isoval,
            cubeCount);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'FindActiveGridCells_D' :                %.10f sec\n",
            dt_ms/1000.0);
#endif

    ::ComputePrefixSumExclusiveScan(cubeStates_D, cubeOffs_D, cubeCount-1);

    return cudaGetLastError();
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
inline __device__ unsigned char TetrahedronFlags_D(
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
        if(::SampleFieldAt_D<float>(cubeVertex0 + cubeVertexOffset, volume_D) <= thresholdValue) {
            flags |= 1 << static_cast<unsigned char>(idx);
        }
    }
    return flags;
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
__global__ void CalcVertexPositions_D(
        uint*activeVertexIdx_D,
        float3 *activeVertexPos_D,
        uint* cubeMap_D,
        float isoval,
        uint activeCubeCount,
        float *volume_D) {

    // Thread index (= active cube index)
    uint globalTetraIdx = ::GetThreadIndex();
    if (globalTetraIdx >= activeCubeCount*6) {
        return;
    }

    // Load LUTs to shared memory
    LoadCubeOffsetsToSharedMemory();
    LoadTetrahedronsInACubeToSharedMemory();
    LoadVertexIdxPerTetrahedronIdxToSharedMemory();
    LoadTetrahedronEdgeFlagsAndConnectionsToSharedMemory();
    __syncthreads();

    uint activeCubeIdx = globalTetraIdx/6;
    uint localTetraIdx = globalTetraIdx%6; // 0 ... 5

    // Compute cell origin
    const uint3 cellOrg = GetGridCoordsByCellIdx(cubeMap_D[activeCubeIdx]);

    // Get bitmap to classify the tetrahedron
    unsigned char tetrahedronFlags = TetrahedronFlags_D(cellOrg, localTetraIdx, isoval, volume_D);

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
            const float f0 = ::SampleFieldAt_D<float>(v0, volume_D);
            const float f1 = ::SampleFieldAt_D<float>(v1, volume_D);
            const float interpolator = (isoval - f0) / (f1 - f0);
            float3 vertex = lerp(make_float3(v0.x, v0.y, v0.z),
                    make_float3(v1.x, v1.y, v1.z), interpolator);

            // Save position and mark vertex index as 'active'
            activeVertexIdx_D[activeCubeIdx*7+localVtxIdx] = 1;
            activeVertexPos_D[activeCubeIdx*7+localVtxIdx] = TransformToWorldSpace(vertex);
        }
    }
}


extern "C"
cudaError_t CalcVertexPositions(uint *vertexStates_D, float3 *activeVertexPos_D,
        uint *vertexIdxOffs_D, uint *cubeMap_D, uint activeCubeCount, float isoval,
        float *volume_D) {

    const uint threadPerBlock = 128;

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    CalcVertexPositions_D <<< Grid(activeCubeCount*6, threadPerBlock ), threadPerBlock  >>> (
            vertexStates_D,
            activeVertexPos_D,
            cubeMap_D,
            isoval,
            activeCubeCount,
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

    ::ComputePrefixSumExclusiveScan(
            vertexStates_D,
            vertexIdxOffs_D,
            7*activeCubeCount-1);

    return cudaGetLastError();
}


/**
 * Setup mapping function from active vertex list to vertex list (based on
 * active cells).
 *
 * @param[out] vertexMap_D       Mapping from active vertex' list to global
 *                               vertex list
 * @param[out] vertexMapInv_D    Mapping from global vertex list to active
 *                               vertex' list
 * @param[in]  vertexIdxOffs_D   Offsets for vertex indices
 * @param[in]  activeVertexIdx_D Active vertex flags, '1' if vertex is active
 * @param[in]  vtxCount          The number of vertices
 */
// TODO vertexIdxOffs_D is pointless
__global__ void CalcVertexMap_D(
        uint* vertexMap_D,
        uint* vertexMapInv_D,
        uint* vertexIdxOffs_D,
        uint* activeVertexIdx_D,
        uint vtxCount) {

    const uint vtxIndex = ::GetThreadIndex();
    if (vtxIndex >= vtxCount) {
        return;
    }

    if(activeVertexIdx_D[vtxIndex] != 0) {
        // Map from active vertices list to vtx idx
        vertexMap_D[vertexIdxOffs_D[vtxIndex]] = vtxIndex;
        vertexMapInv_D[vtxIndex] = vertexIdxOffs_D[vtxIndex];
    }
}

extern "C"
cudaError_t CalcVertexMap(uint *vertexMap_D, uint *vertexMapInv_D,
        uint *vertexIdxOffs_D, uint *vertexStates_D, uint activeCellsCount) {

    CalcVertexMap_D <<< Grid(7*activeCellsCount, 256), 256 >>> (
            vertexMap_D, vertexMapInv_D, vertexIdxOffs_D, vertexStates_D,
            7*activeCellsCount);

    return cudaGetLastError();
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


// Returns unsigned char from 00000 to 111111 describing the activity of the
// vertices inside the cube of index cubeId.
__device__
unsigned char CubeFlags(uint cubeId, uint *activeVertexIdx_D) {
    unsigned char flags = 0x00;
    for(int idx = 0; idx < 6; ++idx) {
        if(activeVertexIdx_D[6*cubeId] == 1) {
            flags |= 1 << static_cast<unsigned char>(idx);
        }
    }
    return flags;
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


inline __device__
void LoadTetrahedronsInACube() {
    // Load tetrahedron vertex index to cube index map into shared memory.
    if (threadIdx.x < 6) {
        for (int i = 0; i < 4; ++i) {
            tetrahedronsInACube_S[threadIdx.x][i] = tetrahedronsInACube[threadIdx.x][i];
        }
    }
}

/*
 * freudenthal_subdiv::GetTetrahedronEdgeVertexIdxOffset
 */
inline __device__
uint GetTetrahedronEdgeVertexIdx(uint activeCubeIndex, uint tetrahedronIdx, uint edgeIdx, uint *cubeMap, uint *cubeMapInv) {
    uint cubeIdx = cubeMap[activeCubeIndex];
    uint offset = (gridSize_D.x-1)*(
            (gridSize_D.y-1)*TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx][2] // Global cube index
            + TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx][1])           // Global cube index
            + TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx][0];
    uint cubeIdxNew = cubeMapInv[cubeIdx + offset];
    return 7*cubeIdxNew + TetrahedronEdgeVertexIdxOffset[tetrahedronIdx][edgeIdx][3];
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
__global__
void CompactActiveVertexPositions_D(
        float *vertexPos_D,
        uint *vertexStates_D,
        uint *vertexIdxOffs_D,
        float3 *activeVertexPos_D,
        uint vertexCount,
        uint outputArrOffs,
        uint outputArrDataSize) {

    // Thread index (= vertex index)
    uint idx = GetThreadIndex();
    if (idx >= vertexCount) {
        return;
    }

    if (vertexStates_D[idx] == 1) {
        vertexPos_D[outputArrDataSize*vertexIdxOffs_D[idx]+outputArrOffs+0] = activeVertexPos_D[idx].x;
        vertexPos_D[outputArrDataSize*vertexIdxOffs_D[idx]+outputArrOffs+1] = activeVertexPos_D[idx].y;
        vertexPos_D[outputArrDataSize*vertexIdxOffs_D[idx]+outputArrOffs+2] = activeVertexPos_D[idx].z;
    }

}

extern "C"
cudaError_t CompactActiveVertexPositions(float *vertexPos_D, uint *vertexStates_D,
        uint *vertexIdxOffs_D, float3 *activeVertexPos_D, uint activeCellCount,
        uint outputArrOffs, uint outputArrDataSize) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    CompactActiveVertexPositions_D <<< Grid(activeCellCount*7, 256), 256 >>> (
            vertexPos_D, vertexStates_D, vertexIdxOffs_D, activeVertexPos_D,
            activeCellCount*7, outputArrOffs, outputArrDataSize);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'CompactActiveVertexPositions_D' :       %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();
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
__global__ void FlagTetrahedrons_D(
        uint* verticesPerTetrahedron_D,
        uint* cubeMap_D,
        float thresholdValue,
        uint activeCubeCount,
        float *volume_D) {

    const uint activeCubeIndex = GetThreadIndex();

    LoadCubeOffsetsToSharedMemory();
    LoadTetrahedronsInACubeToSharedMemory();
    __syncthreads();

    // Prevent non-power of two writes.
    if (activeCubeIndex >= activeCubeCount) {
        return;
    }
    const uint3 cubeVertex0 = GetGridCoordsByCellIdx(cubeMap_D[activeCubeIndex]);
    // Classify all tetrahedrons in a cube.
    for (int tetrahedronIndex = 0; tetrahedronIndex < 6; ++tetrahedronIndex) {
        // Compute tetrahedron flags.
        unsigned char tetrahedronFlags = TetrahedronFlags_D(cubeVertex0, tetrahedronIndex, thresholdValue, volume_D);
        // Store number of vertices.
        verticesPerTetrahedron_D[activeCubeIndex * 6 + tetrahedronIndex] = tetrahedronVertexCount[tetrahedronFlags];
    }
}

extern "C"
cudaError_t FlagTetrahedrons(
        uint *verticesPerTetrahedron_D,
        uint *cubeMap_D,
        float isoval,
        uint activeCellCount,
        float *volume_D) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    FlagTetrahedrons_D <<< Grid(activeCellCount, 256), 256 >>> (
            verticesPerTetrahedron_D,
            cubeMap_D,
            isoval,
            activeCellCount,
            volume_D);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'FlagTetrahedrons_D' :       %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();
}


extern "C"
cudaError_t GetTetrahedronVertexOffsets(
        uint *tetrahedronVertexOffsets_D,
        uint *verticesPerTetrahedron_D,
        uint tetrahedronCount) {


    thrust::exclusive_scan(
            thrust::device_ptr<uint>(verticesPerTetrahedron_D),
            thrust::device_ptr<uint>(verticesPerTetrahedron_D + tetrahedronCount),
            thrust::device_ptr<uint>(tetrahedronVertexOffsets_D));

//    ::ComputePrefixSumExclusiveScan(
//            verticesPerTetrahedron_D,
//            tetrahedronVertexOffsets_D,
//            tetrahedronCount); // TODO This is unintuitive

    return cudaGetLastError();
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
__global__
void GetTrianglesIdx_D(
        uint* vertexOffsets_D,
        uint* cubeMap_D,
        uint* cubeMapInv_D,
        float thresholdValue,
        uint tetrahedronCount,
        uint activeCubeCount,
        uint *triangleVertexIdx_D,
        uint *vertexMapInv_D,
        float *volume_D) {

    const uint id = GetThreadIndex();
    const uint activeCubeIndex = id / 6;
    const int tetrahedronIndex = id % 6;


    // Load tables from constant to shared memory
    LoadCubeOffsetsToSharedMemory();
    LoadTetrahedronsInACubeToSharedMemory();
    __syncthreads();

    // Prevent non-power of two writes.
    if (id >= tetrahedronCount) {
        return;
    }

    const uint3 cubeVertex0 = GetGridCoordsByCellIdx(cubeMap_D[activeCubeIndex]);

    // Get bitmap to classify the tetrahedron
    unsigned char tetrahedronFlags = TetrahedronFlags_D(cubeVertex0,
            tetrahedronIndex, thresholdValue, volume_D);

    // Skip inactive tetrahedrons
    if (tetrahedronFlags == 0x00 || tetrahedronFlags == 0x0F) {
        return;
    }
    __shared__ uint edgeVertexIdx[6 * GET_TRIANGLE_IDX_BLOCKSIZE];

    // Find intersection of the surface with each edge.
    for (int edgeIndex = 0; edgeIndex < 6; edgeIndex++) {
        // Test if edge intersects with surface.
        if (tetrahedronEdgeFlags[tetrahedronFlags] & (1 << static_cast<unsigned char>(edgeIndex)))  {
            edgeVertexIdx[threadIdx.x * 6 + edgeIndex] =
                    GetTetrahedronEdgeVertexIdx(activeCubeIndex,
                            tetrahedronIndex, edgeIndex, cubeMap_D, cubeMapInv_D);
        }
    }

    __syncthreads();

    // Write vertices.
    for (int triangleIndex = 0; triangleIndex < 2; triangleIndex++) {
        if (tetrahedronTriangles[tetrahedronFlags][3 * triangleIndex] >= 0) {
            for (int cornerIndex = 0; cornerIndex < 3; cornerIndex++) {
                int edgeIndex = threadIdx.x * 6 + tetrahedronTriangles[tetrahedronFlags][3 * triangleIndex + cornerIndex];
                uint vertexOffset = vertexOffsets_D[id] + 3 * triangleIndex + cornerIndex;
                triangleVertexIdx_D[vertexOffset] = vertexMapInv_D[edgeVertexIdx[edgeIndex]];
            }
        }
    }
}


extern "C"
cudaError_t GetTrianglesIdx(uint *tetrahedronVertexOffsets_D, uint *cubeMap_D,
        uint *cubeMapInv_D,
        float isoval,
        uint tetrahedronCount,
        uint activeCellCount,
        uint *triangleVertexIdx_D,
        uint *vertexMapInv_D,
        float *volume_D) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // TODO Use dnymic shared memory here
    GetTrianglesIdx_D <<< Grid(tetrahedronCount, GET_TRIANGLE_IDX_BLOCKSIZE), GET_TRIANGLE_IDX_BLOCKSIZE >>> (
            tetrahedronVertexOffsets_D,
            cubeMap_D,
            cubeMapInv_D,
            isoval,
            tetrahedronCount,
            activeCellCount,
            triangleVertexIdx_D,
            vertexMapInv_D,
            volume_D);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'GetTrianglesIdx_D' :                    %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();
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
__global__
void ComputeVertexConnectivity_D(
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
    uint idx = ::GetThreadIndex();
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
    int vertIdx = VertIdxGlobal_S[sharedMemoryIdx];
    int v       = VertIdxLocal_S[sharedMemoryIdx];
    uint cubeId = CellIdx_S[sharedMemoryIdx];
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
    terahedronFlagsTmp = TetrahedronFlags_D(
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

extern "C"
cudaError_t ComputeVertexConnectivity(int *vertexNeighbours_D, uint *vertexStates_D,
        uint *vertexMap_D, uint *vertexMapInv_D, uint *cubeMap_D,
        uint *cubeMapInv_D, uint *cubeStates_D, uint activeVertexCnt, float *volume_D,
        float isoval) {

    const uint blockSize = 192; // == 6 * 32, 32 = warpsize

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

//    cudaDeviceProp devProp;
//    cudaGetDeviceProperties(&devProp, 0);
//    printf("Shared memory per block %u bytes\n", devProp.sharedMemPerBlock);
//    printf("Number of blocks %u\n", Grid(activeVertexCnt*6, blockSize).x);

//    CheckForCudaErrorSync();

    ComputeVertexConnectivity_D <<< Grid(activeVertexCnt*6, blockSize), blockSize >>> (
            vertexNeighbours_D,
            vertexStates_D,
            activeVertexCnt,
            vertexMap_D,
            vertexMapInv_D,
            cubeMap_D,
            cubeMapInv_D,
            cubeStates_D,
            volume_D,
            isoval);

//    CheckForCudaErrorSync();

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVertexConnectivity_D' :          %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();
}

__constant__ __device__ uint TriangleCrossPoductVtxIdx[3][2] = {
        {1, 2}, {1, 0}, {0, 2}
};

__global__
void ComputeVertexNormals_D(
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
    uint activeVertexIdx = GetThreadIndex();
    LoadCubeOffsetsToSharedMemory();
    LoadTetrahedronsInACubeToSharedMemory();
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
        terahedronFlagsTmp = TetrahedronFlags_D(
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
//
//                    if (vertexIdx0 >= activeVertexCnt) continue;
//                    if (vertexIdx1 >= activeVertexCnt) continue;
//
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
//    dataBuffer_D[arrDataSize*activeVertexIdx+arrDataOffsNormals+0] = maxIdx;
//    dataBuffer_D[arrDataSize*activeVertexIdx+arrDataOffsNormals+1] = maxIdx;
//    dataBuffer_D[arrDataSize*activeVertexIdx+arrDataOffsNormals+2] = maxIdx;
}

extern "C"
cudaError_t ComputeVertexNormals(
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
        uint arrDataSize) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    ComputeVertexNormals_D <<< Grid(activeVertexCnt, 256), 256 >>> (
            dataBuffer_D,
            vertexMap_D,
            vertexMapInv_D,
            cubeMap_D,
            cubeMapInv_D,
            volume_D,
            isoval,
            activeVertexCnt,
            arrDataOffsPos,
            arrDataOffsNormals,
            arrDataSize);


#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVertexNormals_D' :               %.10f sec\n",
            dt_ms/1000.0);
#endif

    //return cudaGetLastError();
    return cudaDeviceSynchronize(); // Appearently this is necessary
}

__global__
void ComputeVertexTexCoords_D(float *dataBuff_D,
        float volMinX, float volMinY, float volMinZ,
        float volMaxX, float volMaxY, float volMaxZ,
        uint activeVertexCnt,
        uint arrDataOffsPos,
        uint arrDataOffsTexCoords,
        uint arrDataSize) {

    // Get thread index
    uint activeVertexIdx = GetThreadIndex();
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

extern "C"
cudaError_t ComputeVertexTexCoords(float *dataBuff_D,
        float volMinX, float volMinY, float volMinZ,
        float volMaxX, float volMaxY, float volMaxZ,
        uint activeVertexCnt,
        uint arrDataOffsPos,
        uint arrDataOffsTexCoords,
        uint arrDataSize) {


#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Calc difference field using the iso value
    ComputeVertexTexCoords_D <<< Grid(activeVertexCnt, 256), 256 >>> (
            dataBuff_D,
            volMinX, volMinY, volMinZ,
            volMaxX, volMaxY, volMaxZ,
                    activeVertexCnt,
                    arrDataOffsPos,
                    arrDataOffsTexCoords,
                    arrDataSize);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVertexTexCoords_D' :             %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();
}
