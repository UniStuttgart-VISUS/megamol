//
// ComparativeSurfacePotentialRenderer.cu
//
// Contains CUDA functionality used by the PotentialVolumeRendererCuda class.
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 13, 2013
//     Author: scharnkn
//

#include "ComparativeSurfacePotentialRenderer.cuh"
#include "ComparativeSurfacePotentialRenderer_inline_device_functions.cuh"
#include "interpol.cuh"

#include "cuda_helper.h"
#include <device_functions.h>

#include <cassert>


// Shut up eclipse syntax error highlighting
#ifdef __CDT_PARSER__
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __host__
#endif

// Toggle performance measurement and respective messages
//#define USE_TIMER

extern "C"
cudaError InitVolume(uint3 gridSize, float3 org, float3 delta) {
    cudaMemcpyToSymbol(gridSize_D, &gridSize, sizeof(uint3));
    cudaMemcpyToSymbol(gridOrg_D, &org, sizeof(float3));
    cudaMemcpyToSymbol(gridDelta_D, &delta, sizeof(float3));
//    printf("Init grid with org %f %f %f, delta %f %f %f, dim %u %u %u\n", org.x,
//            org.y, org.z, delta.x, delta.y, delta.z, gridSize.x, gridSize.y,
//            gridSize.z);
    return cudaGetLastError();
}

///**
// * Returns a 1D grid definition based on the given threadsPerBlock value.
// *
// * @param size             The minimum number of threads
// * @param threadsPerBlock  The number of threads per block
// * @return The grid dimensions
// */
//extern "C" dim3 Grid(const uint size, const int threadsPerBlock) {
//    //TODO: remove hardcoded hardware capabilities :(
//    // see: http://code.google.com/p/thrust/source/browse/thrust/detail/backend/cuda/arch.inl
//    //   and http://code.google.com/p/thrust/source/browse/thrust/detail/backend/cuda/detail/safe_scan.inl
//    //   for refactoring.
//    // Get maximum grid size of CUDA device.
//    //CUdevice device;
//    //cuDeviceGet(&device, 0);
//    //CUdevprop deviceProps;
//    //cuDeviceGetProperties(&deviceProps, device);
//    //this->gridSize = dim3(deviceProps.maxGridSize[0],
//    //  deviceProps.maxGridSize[1],
//    //  deviceProps.maxGridSize[2]);
//    const dim3 maxGridSize(65535, 65535, 0);
//    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
//    dim3 grid(blocksPerGrid, 1, 1);
//
//    return grid;
//}





//-------------//
// RMS fitting //
//-------------//

__global__
void TranslatePos_D(float *vertexData_D, uint vertexDataStride,
        uint vertexDataOffsPos, float3 translation, uint vertexCnt) {

    const uint idx = GetThreadIndex();
    if (idx >= vertexCnt) {
        return;
    }
    const uint vertexDataIdx = vertexDataStride*idx+vertexDataOffsPos;

    vertexData_D[vertexDataIdx+0] += translation.x;
    vertexData_D[vertexDataIdx+1] += translation.y;
    vertexData_D[vertexDataIdx+2] += translation.z;
}

extern "C"
cudaError_t TranslatePos(float *vertexData_D, uint vertexDataStride,
        uint vertexDataOffsPos, float3 translation, uint vertexCnt) {

#ifdef USE_TIMER
    //Create events
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    //Record events around kernel launch
    cudaEventRecord(event1, 0); //where 0 is the default stream
#endif

    // Initialize triangle index array
    TranslatePos_D <<< Grid(vertexCnt, 256), 256 >>> (vertexData_D,
            vertexDataStride, vertexDataOffsPos, translation, vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    // Synchronize
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!
    // Calculate time
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Kernel execution time 'Translateos_D': %f sec\n", dt_ms/1000.0f);
#endif

    return cudaGetLastError();
}

__global__
void RotatePos_D(float *vertexData_D, uint vertexDataStride,
        uint vertexDataOffsPos, float *rotation_D, uint vertexCnt) {

    const uint idx = GetThreadIndex();
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

extern "C"
cudaError_t RotatePos(float *vertexData_D, uint vertexDataStride,
        uint vertexDataOffsPos, float *rotation_D, uint vertexCnt) {

#ifdef USE_TIMER
    //Create events
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    //Record events around kernel launch
    cudaEventRecord(event1, 0); //where 0 is the default stream
#endif

    // Initialize triangle index array
    RotatePos_D <<< Grid(vertexCnt, 256), 256 >>> (vertexData_D,
            vertexDataStride, vertexDataOffsPos, rotation_D, vertexCnt);
#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    // Synchronize
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!
    // Calculate time
    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Kernel execution time 'RotatePos_D': %f sec\n", dt_ms/1000.0f);
#endif

    return cudaGetLastError();
}
