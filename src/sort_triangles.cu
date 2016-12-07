//
// sort_triangles.cu
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 29, 2013
//     Author: scharnkn
//

#include "sort_triangles.cuh"

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_ptr.h>

#include "helper_math.h"

//#define SORT_TRIANGLES_CUDA_USE_TIMER

// Shut up eclipse syntax error highlighting
#ifdef __CDT_PARSER__
#define __device__
#define __global__
#define __shared__
#define __constant__
#endif

typedef unsigned int uint;


__global__
void TrianglesCalcDistToCam_D(
        float *dataBuff_D,
        uint dataBuffSize,
        uint dataBuffOffsPos,
        uint *triangleVtxIdx_D,
        float *triangleCamDistance_D,
        float3 camPos,
        uint triangleCnt) {

    const uint idx = __umul24(__umul24(blockIdx.y, gridDim.x) +
            blockIdx.x, blockDim.x) + threadIdx.x;
    if (idx >= triangleCnt) {
        return;
    }

    // Alternative 1: Use midpoint
    float3 pos0 = make_float3(dataBuff_D[dataBuffSize*triangleVtxIdx_D[3*idx+0]+dataBuffOffsPos+0],
            dataBuff_D[dataBuffSize*triangleVtxIdx_D[3*idx+0]+dataBuffOffsPos+1],
            dataBuff_D[dataBuffSize*triangleVtxIdx_D[3*idx+0]+dataBuffOffsPos+2]);

    float3 pos1 = make_float3(dataBuff_D[dataBuffSize*triangleVtxIdx_D[3*idx+1]+dataBuffOffsPos+0],
            dataBuff_D[dataBuffSize*triangleVtxIdx_D[3*idx+1]+dataBuffOffsPos+1],
            dataBuff_D[dataBuffSize*triangleVtxIdx_D[3*idx+1]+dataBuffOffsPos+2]);

    float3 pos2 = make_float3(dataBuff_D[dataBuffSize*triangleVtxIdx_D[3*idx+2]+dataBuffOffsPos+0],
            dataBuff_D[dataBuffSize*triangleVtxIdx_D[3*idx+2]+dataBuffOffsPos+1],
            dataBuff_D[dataBuffSize*triangleVtxIdx_D[3*idx+2]+dataBuffOffsPos+2]);

    float3 trianglePos = (pos0 + pos1 + pos2)/3.0;

    // Alternative 2: Use first vertex
    //float3 trianglePos = vertexPos_D[triangleVtxIdx_D[3*idx]+0];
    triangleCamDistance_D[idx] = length(trianglePos - camPos);
}

extern "C"
cudaError_t SortTrianglesByCamDistance(
        float *dataBuff_D,
        uint dataBuffSize,
        uint dataBuffOffsPos,
        float3 camPos,
        uint *triangleVtxIdx_D,
        uint triangleCnt,
        float *triangleCamDistance_D) {

#ifdef SORT_TRIANGLES_CUDA_USE_TIMER
    //Create events
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    //Record events around kernel launch
    cudaEventRecord(event1, 0); //where 0 is the default stream
#endif

    const uint threadsPerBlock = 256;

    // Define grid
    const dim3 maxGridSize(65535, 65535, 0);
    const int blocksPerGrid = (triangleCnt + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

    // Compute distance to camera position
    TrianglesCalcDistToCam_D <<< grid, threadsPerBlock >>> (
            dataBuff_D, dataBuffSize, dataBuffOffsPos, triangleVtxIdx_D,
            triangleCamDistance_D, camPos, triangleCnt);

    // Sort triangles using thrust library
    thrust::stable_sort_by_key(
            thrust::device_ptr<float>(triangleCamDistance_D), // Keys
            thrust::device_ptr<float>(triangleCamDistance_D + triangleCnt),
            thrust::device_ptr<uint3>(reinterpret_cast<uint3*>(triangleVtxIdx_D)),          // Values
            thrust::greater<float>());                        // Sort in descending order

#ifdef SORT_TRIANGLES_CUDA_USE_TIMER
    cudaEventRecord(event2, 0);
    // Synchronize
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!
    // Calculate time
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'triangle sorting' :                     %.10f sec\n",
               dt_ms/1000.0);
#endif

    return cudaGetLastError();
}





