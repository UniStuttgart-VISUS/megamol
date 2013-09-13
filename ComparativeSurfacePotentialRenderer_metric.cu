//
// ComparativeSurfacePotentialRenderer_metric.cu
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 22, 2013
//     Author: scharnkn
//

//#include <stdafx.h>
#include "ComparativeSurfacePotentialRenderer.cuh"
#include "ComparativeSurfacePotentialRenderer_inline_device_functions.cuh"

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

__constant__ __device__ int3 potentialGridDim0;
__constant__ __device__ float3 potentialMinC0;
__constant__ __device__ float3 potentialDelta0;

__constant__ __device__ int3 potentialGridDim1;
__constant__ __device__ float3 potentialMinC1;
__constant__ __device__ float3 potentialDelta1;

cudaError_t InitPotentialTexParams(int idx, int3 dim, float3 minC, float3 delta) {
    if (idx == 0) {
        cudaMemcpyToSymbol(potentialGridDim0, &dim, sizeof(int3));
        cudaMemcpyToSymbol(potentialMinC0, &minC, sizeof(float3));
        cudaMemcpyToSymbol(potentialDelta0, &delta, sizeof(float3));
    } else {
        cudaMemcpyToSymbol(potentialGridDim1, &dim, sizeof(int3));
        cudaMemcpyToSymbol(potentialMinC1, &minC, sizeof(float3));
        cudaMemcpyToSymbol(potentialDelta1, &delta, sizeof(float3));
    }

    return cudaGetLastError();
}

texture<float, 3, cudaReadModeElementType> potentialTexRef0;
texture<float, 3, cudaReadModeElementType> potentialTexRef1;

cudaError_t BindTexRef0ToArray(cudaArray *texArray) {
    // specify mutable texture reference parameters
    potentialTexRef0.normalized = true;
    potentialTexRef0.filterMode = cudaFilterModeLinear;
    potentialTexRef0.addressMode[0] = cudaAddressModeWrap;
    potentialTexRef0.addressMode[1] = cudaAddressModeWrap;
    potentialTexRef0.addressMode[2] = cudaAddressModeWrap;

    // bind texture reference to array
    cudaBindTextureToArray(potentialTexRef0, texArray);

    return cudaGetLastError();
}

cudaError_t BindTexRef1ToArray(cudaArray *texArray) {
    // specify mutable texture reference parameters
    potentialTexRef1.normalized = true;
    potentialTexRef1.filterMode = cudaFilterModeLinear;
    potentialTexRef1.addressMode[0] = cudaAddressModeWrap;
    potentialTexRef1.addressMode[1] = cudaAddressModeWrap;
    potentialTexRef1.addressMode[2] = cudaAddressModeWrap;

    // bind texture reference to array
    cudaBindTextureToArray(potentialTexRef1, texArray);

    return cudaGetLastError();
}

extern "C"
cudaError_t InitVolume_metric(uint3 gridSize, float3 org, float3 delta) {
    cudaMemcpyToSymbol(gridSize_D, &gridSize, sizeof(uint3));
    cudaMemcpyToSymbol(gridOrg_D, &org, sizeof(float3));
    cudaMemcpyToSymbol(gridDelta_D, &delta, sizeof(float3));
//    printf("Init grid with org %f %f %f, delta %f %f %f, dim %u %u %u\n", org.x,
//            org.y, org.z, delta.x, delta.y, delta.z, gridSize.x, gridSize.y,
//            gridSize.z);
    return cudaGetLastError();
}


inline __device__ float SamplePotentialTexAtPosTrilin_D(float3 pos, float *tex_D,
        float3 org, float3 delta, int3 gridSize) {

    uint3 c;
    float3 f;

    // Get id of the cell containing the given position and interpolation
    // coefficients
    f.x = (pos.x-org.x)/delta.x;
    f.y = (pos.y-org.y)/delta.y;
    f.z = (pos.z-org.z)/delta.z;
    c.x = (uint)(f.x);
    c.y = (uint)(f.y);
    c.z = (uint)(f.z);
    f.x = f.x-(float)c.x; // alpha
    f.y = f.y-(float)c.y; // beta
    f.z = f.z-(float)c.z; // gamma

    // Get values at corners of current cell
    float s[8];
    s[0] = tex_D[gridSize.x*(gridSize.y*(c.z+0) + (c.y+0))+c.x+0];
    s[1] = tex_D[gridSize.x*(gridSize.y*(c.z+0) + (c.y+0))+c.x+1];
    s[2] = tex_D[gridSize.x*(gridSize.y*(c.z+0) + (c.y+1))+c.x+0];
    s[3] = tex_D[gridSize.x*(gridSize.y*(c.z+0) + (c.y+1))+c.x+1];
    s[4] = tex_D[gridSize.x*(gridSize.y*(c.z+1) + (c.y+0))+c.x+0];
    s[5] = tex_D[gridSize.x*(gridSize.y*(c.z+1) + (c.y+0))+c.x+1];
    s[6] = tex_D[gridSize.x*(gridSize.y*(c.z+1) + (c.y+1))+c.x+0];
    s[7] = tex_D[gridSize.x*(gridSize.y*(c.z+1) + (c.y+1))+c.x+1];

    // Use trilinear interpolation to sample the potential texture
   return InterpFieldTrilin_D<float>(s, f.x, f.y, f.z);
}


/*
 * ComputeVertexPotentialDiff_D
 */
__global__ void ComputeVertexPotentialDiffSqr_D(
        float *vertexPotentialDiff_D,
        float *vertexPosNew_D,
        float *vertexPosOld_D,
        float *potentialTex0_D,
        float *potentialTex1_D,
        float volMinXOld, float volMinYOld, float volMinZOld,
        float volMaxXOld, float volMaxYOld, float volMaxZOld,
        float volMinXNew, float volMinYNew, float volMinZNew,
        float volMaxXNew, float volMaxYNew, float volMaxZNew,
        uint vertexCnt) {

    const uint idx = ::GetThreadIndex();
    if (idx >= vertexCnt) {
        return;
    }

    float3 posOld = make_float3(
            vertexPosOld_D[3*idx+0],
            vertexPosOld_D[3*idx+1],
            vertexPosOld_D[3*idx+2]);

    float3 posNew = make_float3(
            vertexPosNew_D[3*idx+0],
            vertexPosNew_D[3*idx+1],
            vertexPosNew_D[3*idx+2]);

    vertexPotentialDiff_D[idx] =
            ::fabs(SamplePotentialTexAtPosTrilin_D(posNew, potentialTex0_D, potentialMinC0, potentialDelta0, potentialGridDim0)
                    - SamplePotentialTexAtPosTrilin_D(posOld, potentialTex1_D, potentialMinC1, potentialDelta1, potentialGridDim1));

    vertexPotentialDiff_D[idx] *= vertexPotentialDiff_D[idx];
}

/*
 * ComputeVertexPosDist
 */
cudaError_t ComputeVertexPotentialDiffSqr(
        float *vertexPotentialDiff_D,
        float *vertexPosNew_D,
        float *vertexPosOld_D,
        float *potentialTex0_D,
        float *potentialTex1_D,
        float volMinXOld, float volMinYOld, float volMinZOld,
        float volMaxXOld, float volMaxYOld, float volMaxZOld,

        float volMinXNew, float volMinYNew, float volMinZNew,
        float volMaxXNew, float volMaxYNew, float volMaxZNew,

        uint vertexCnt,
        float3 *texCoords_D // DEBUG
        ) {

    // Create 1D grid layout
    const uint threadsPerBlock = 256;
    const uint blocksPerGrid = (vertexCnt+threadsPerBlock-1)/threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    ComputeVertexPotentialDiffSqr_D <<< grid, threadsPerBlock >>> (
            vertexPotentialDiff_D,
            vertexPosNew_D,
            vertexPosOld_D,
            potentialTex0_D,
            potentialTex1_D,
            volMinXOld, volMinYOld, volMinZOld,
            volMaxXOld, volMaxYOld, volMaxZOld,
            volMinXNew, volMinYNew, volMinZNew,
            volMaxXNew, volMaxYNew, volMaxZNew,
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVertexPosDist_D                  %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();

}


/*
 * ComputeVertexPotentialDiff_D
 */
__global__ void ComputeVertexPotentialDiff_D(
        float *vertexPotentialDiff_D,
        float *vertexPosNew_D,
        float *vertexPosOld_D,
        float *potentialTex0_D,
        float *potentialTex1_D,
        float volMinXOld, float volMinYOld, float volMinZOld,
        float volMaxXOld, float volMaxYOld, float volMaxZOld,
        float volMinXNew, float volMinYNew, float volMinZNew,
        float volMaxXNew, float volMaxYNew, float volMaxZNew,
        uint vertexCnt) {

    const uint idx = ::GetThreadIndex();
    if (idx >= vertexCnt) {
        return;
    }

    float3 posOld = make_float3(
            vertexPosOld_D[3*idx+0],
            vertexPosOld_D[3*idx+1],
            vertexPosOld_D[3*idx+2]);

    float3 posNew = make_float3(
            vertexPosNew_D[3*idx+0],
            vertexPosNew_D[3*idx+1],
            vertexPosNew_D[3*idx+2]);

    vertexPotentialDiff_D[idx] =
            ::fabs(SamplePotentialTexAtPosTrilin_D(posNew, potentialTex0_D, potentialMinC0, potentialDelta0, potentialGridDim0)
                    - SamplePotentialTexAtPosTrilin_D(posOld, potentialTex1_D, potentialMinC1, potentialDelta1, potentialGridDim1));
}


/*
 * ComputeVertexPosDist
 */
cudaError_t ComputeVertexPotentialDiff(
        float *vertexPotentialDiff_D,
        float *vertexPosNew_D,
        float *vertexPosOld_D,
        float *potentialTex0_D,
        float *potentialTex1_D,
        float volMinXOld, float volMinYOld, float volMinZOld,
        float volMaxXOld, float volMaxYOld, float volMaxZOld,
        float volMinXNew, float volMinYNew, float volMinZNew,
        float volMaxXNew, float volMaxYNew, float volMaxZNew,
        uint vertexCnt) {

    // Create 1D grid layout
    const uint threadsPerBlock = 256;
    const uint blocksPerGrid = (vertexCnt+threadsPerBlock-1)/threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    ComputeVertexPotentialDiff_D <<< grid, threadsPerBlock >>> (
            vertexPotentialDiff_D,
            vertexPosNew_D,
            vertexPosOld_D,
            potentialTex0_D,
            potentialTex1_D,
            volMinXOld, volMinYOld, volMinZOld,
            volMaxXOld, volMaxYOld, volMaxZOld,
            volMinXNew, volMinYNew, volMinZNew,
            volMaxXNew, volMaxYNew, volMaxZNew,
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVertexPosDist_D                  %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();

}


/*
 * ComputeVertexPotentialDiff_D
 */
__global__ void ComputeVertexPotentialSignDiff_D(
        float *vertexPotentialSignDiff_D,
        float *vertexPosNew_D,
        float *vertexPosOld_D,
        float *potentialTex0_D,
        float *potentialTex1_D,
        float volMinXOld, float volMinYOld, float volMinZOld,
        float volMaxXOld, float volMaxYOld, float volMaxZOld,
        float volMinXNew, float volMinYNew, float volMinZNew,
        float volMaxXNew, float volMaxYNew, float volMaxZNew,
        uint vertexCnt) {

    const uint idx = ::GetThreadIndex();
    if (idx >= vertexCnt) {
        return;
    }

    float3 posOld = make_float3(
            vertexPosOld_D[3*idx+0],
            vertexPosOld_D[3*idx+1],
            vertexPosOld_D[3*idx+2]);

    float3 posNew = make_float3(
            vertexPosNew_D[3*idx+0],
            vertexPosNew_D[3*idx+1],
            vertexPosNew_D[3*idx+2]);

    float sample0 = SamplePotentialTexAtPosTrilin_D(posNew, potentialTex0_D,
            potentialMinC0, potentialDelta0, potentialGridDim0);
    float sample1 = SamplePotentialTexAtPosTrilin_D(posOld, potentialTex1_D,
            potentialMinC1, potentialDelta1, potentialGridDim1);

    vertexPotentialSignDiff_D[idx] = float(sample0*sample1 < 0);
}

cudaError_t ComputeVertexPotentialSignDiff(
        float *vertexPotentialSignDiff_D,
        float *vertexPosNew_D,
        float *vertexPosOld_D,
        float *potentialTex0_D,
        float *potentialTex1_D,
        float volMinXOld, float volMinYOld, float volMinZOld,
        float volMaxXOld, float volMaxYOld, float volMaxZOld,
        float volMinXNew, float volMinYNew, float volMinZNew,
        float volMaxXNew, float volMaxYNew, float volMaxZNew,
        uint vertexCnt) {

    // Create 1D grid layout
    const uint threadsPerBlock = 256;
    const uint blocksPerGrid = (vertexCnt+threadsPerBlock-1)/threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    ComputeVertexPotentialSignDiff_D <<< grid, threadsPerBlock >>> (
            vertexPotentialSignDiff_D,
            vertexPosNew_D,
            vertexPosOld_D,
            potentialTex0_D,
            potentialTex1_D,
            volMinXOld, volMinYOld, volMinZOld,
            volMaxXOld, volMaxYOld, volMaxZOld,
            volMinXNew, volMinYNew, volMinZNew,
            volMaxXNew, volMaxYNew, volMaxZNew,
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVertexPosDist_D                  %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();

}


/*
 * ComputeVertexPosDist_D
 */
__global__ void ComputeVertexPosDist_D(
        float *vertexPosDist_D,
        float *vertexPosNew_D,
        float *vertexPosOld_D,
        uint vertexCnt) {

    const uint idx = ::GetThreadIndex();
    if (idx >= vertexCnt) {
        return;
    }

    float3 posOld = make_float3(
            vertexPosOld_D[3*idx+0],
            vertexPosOld_D[3*idx+1],
            vertexPosOld_D[3*idx+2]);

    float3 posNew = make_float3(
            vertexPosNew_D[3*idx+0],
            vertexPosNew_D[3*idx+1],
            vertexPosNew_D[3*idx+2]);

    vertexPosDist_D[idx] = length(posNew-posOld);
}


/*
 * ComputeVertexPosDist
 */
cudaError_t ComputeVertexPosDist(
        float *vertexPosDist_D,
        float *vertexPosNew_D,
        float *vertexPosOld_D,
        uint vertexCnt) {

    // Create 1D grid layout
    const uint threadsPerBlock = 256;
    const uint blocksPerGrid = (vertexCnt+threadsPerBlock-1)/threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    ComputeVertexPosDist_D <<< grid, threadsPerBlock >>> (
            vertexPosDist_D,
            vertexPosNew_D,
            vertexPosOld_D,
            vertexCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeVertexPosDist_D                  %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();

}


/*
 * ComputeTriangleArea_D
 */
__global__ void ComputeTriangleArea_D(
        float *trianglesArea_D,
        float *corruptTriangleFlag_D,
        float *vertexPos_D,
        uint *triangleIdx_D,
        uint triangleCnt) {

    const uint idx = ::GetThreadIndex();
    if (idx >= triangleCnt) {
        return;
    }

    float flag = corruptTriangleFlag_D[idx];

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
 * ComputeTriangleArea
 */
cudaError_t ComputeTriangleArea(
        float *trianglesArea_D,
        float *corruptTriangleFlag_D,
        float *vertexPos_D,
        uint *triangleIdx_D,
        uint triangleCnt) {

    // Create 1D grid layout
    const uint threadsPerBlock = 256;
    const uint blocksPerGrid = (triangleCnt+threadsPerBlock-1)/threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    ComputeTriangleArea_D <<< grid, threadsPerBlock >>> (
            trianglesArea_D,
            corruptTriangleFlag_D,
            vertexPos_D,
            triangleIdx_D,
            triangleCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeTriangleArea_D                   %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();
}


/*
 * ComputeTriangleAreaAll_D
 */
__global__ void ComputeTriangleAreaAll_D(
        float *trianglesArea_D,
        float *vertexPos_D,
        uint *triangleIdx_D,
        uint triangleCnt) {

    const uint idx = ::GetThreadIndex();
    if (idx >= triangleCnt) {
        return;
    }

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
    trianglesArea_D[idx] = length(pos0-pos1)*length(hVec)*0.5;
}


/*
 * ComputeTriangleArea
 */
cudaError_t ComputeTriangleAreaAll(
        float *trianglesArea_D,
        float *vertexPos_D,
        uint *triangleIdx_D,
        uint triangleCnt) {

    // Create 1D grid layout
    const uint threadsPerBlock = 256;
    const uint blocksPerGrid = (triangleCnt+threadsPerBlock-1)/threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    ComputeTriangleAreaAll_D <<< grid, threadsPerBlock >>> (
            trianglesArea_D,
            vertexPos_D,
            triangleIdx_D,
            triangleCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeTriangleArea_D                   %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();
}


/*
 * ComputeTriangleAreaCorrupt_D
 */
__global__ void ComputeTriangleAreaCorrupt_D(
        float *trianglesArea_D,
        float *corruptTriangleFlag_D,
        float *vertexPos_D,
        uint *triangleIdx_D,
        uint triangleCnt) {

    const uint idx = ::GetThreadIndex();
    if (idx >= triangleCnt) {
        return;
    }

    float flag = corruptTriangleFlag_D[idx];

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
    trianglesArea_D[idx] = length(pos0-pos1)*length(hVec)*0.5*flag;
}


/*
 * ComputeTriangleArea
 */
cudaError_t ComputeTriangleAreaCorrupt(
        float *trianglesArea_D,
        float *corruptTriangleFlag_D,
        float *vertexPos_D,
        uint *triangleIdx_D,
        uint triangleCnt) {

    // Create 1D grid layout
    const uint threadsPerBlock = 256;
    const uint blocksPerGrid = (triangleCnt+threadsPerBlock-1)/threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    ComputeTriangleAreaCorrupt_D <<< grid, threadsPerBlock >>> (
            trianglesArea_D,
            corruptTriangleFlag_D,
            vertexPos_D,
            triangleIdx_D,
            triangleCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'ComputeTriangleArea_D                   %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();
}


/*
 * IntegrateScalarValueOverTriangles_D
 */
__global__ void IntegrateScalarValueOverTriangles_D(
        float *trianglesAreaWeightedVertexVals_D,
        float *corruptTriangleFlag_D,
        float *trianglesArea_D,
        uint *triangleIdx_D,
        float *scalarValue_D,
        uint triangleCnt) {

    const uint idx = ::GetThreadIndex();
    if (idx >= triangleCnt) {
        return;
    }

    // Compute average
    float avgVal = (scalarValue_D[triangleIdx_D[idx*3+0]] +
            scalarValue_D[triangleIdx_D[idx*3+1]] +
            scalarValue_D[triangleIdx_D[idx*3+2]])/3.0;

    trianglesAreaWeightedVertexVals_D[idx] = avgVal*trianglesArea_D[idx]*(1.0 - corruptTriangleFlag_D[idx]);
}


/*
 * IntegrateScalarValueOverTriangles
 */
cudaError_t IntegrateScalarValueOverTriangles(
        float *trianglesAreaWeightedVertexVals_D,
        float *corruptTriangleFlag_D,
        float *trianglesArea_D,
        uint *triangleIdx_D,
        float *scalarValue_D,
        uint triangleCnt) {

    // Create 1D grid layout
    const uint threadsPerBlock = 256;
    const uint blocksPerGrid = (triangleCnt+threadsPerBlock-1)/threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    IntegrateScalarValueOverTriangles_D <<< grid, threadsPerBlock >>> (
            trianglesAreaWeightedVertexVals_D,
            corruptTriangleFlag_D,
            trianglesArea_D,
            triangleIdx_D,
            scalarValue_D,
            triangleCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'IntegrateScalarValueOverTriangles_D     %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();
}



/*
 * IntegrateScalarValueOverTriangles_D
 */
__global__ void FlagCorruptTriangles_D(
        float *corruptTriangleFlag_D,
        float *vertexPos_D,
        uint *triangleVtxIdx_D,
        float *volume_D,
        uint triangleCnt,
        uint vertexCnt, // TODO unnecessary
        float isoval) {

    const uint idx = GetThreadIndex();
    if (idx >= triangleCnt) {
        return;
    }

    // Get vertex positions of this triangle
    const float3 p0 = make_float3(
            vertexPos_D[3*triangleVtxIdx_D[3*idx+0]+0],
            vertexPos_D[3*triangleVtxIdx_D[3*idx+0]+1],
            vertexPos_D[3*triangleVtxIdx_D[3*idx+0]+2]);
    const float3 p1 = make_float3(
            vertexPos_D[3*triangleVtxIdx_D[3*idx+1]+0],
            vertexPos_D[3*triangleVtxIdx_D[3*idx+1]+1],
            vertexPos_D[3*triangleVtxIdx_D[3*idx+1]+2]);
    const float3 p2 = make_float3(
            vertexPos_D[3*triangleVtxIdx_D[3*idx+2]+0],
            vertexPos_D[3*triangleVtxIdx_D[3*idx+2]+1],
            vertexPos_D[3*triangleVtxIdx_D[3*idx+2]+2]);

    // Sample volume at midpoint
    const float3 midPoint = (p0+p1+p2)/3.0;
    const float volSampleMidPoint = ::SampleFieldAtPosTricub_D<float>(midPoint, volume_D);
    corruptTriangleFlag_D[idx] = float((::fabs(volSampleMidPoint-isoval) > 0.1));
}


// This one saves the flag per triangle
cudaError_t FlagCorruptTriangles(float *corruptTriangleFlag_D,
        float *vertexPos_D, uint *triangleVtxIdx_D,
        float *volume_D, uint triangleCnt, uint vertexCnt, float isoval) {

    // Create 1D grid layout
    const uint threadsPerBlock = 256;
    const uint blocksPerGrid = (triangleCnt+threadsPerBlock-1)/threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    FlagCorruptTriangles_D <<< grid, threadsPerBlock >>> (
            corruptTriangleFlag_D,
            vertexPos_D,
            triangleVtxIdx_D,
            volume_D,
            triangleCnt,
            vertexCnt,
            isoval);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'FindCorruptTrianglesFlag_D              %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();

}


/*
 * IntegrateScalarValueOverTriangles_D
 */
__global__ void MultTriangleAreaWithWeight_D(
        float *corruptTriangleFlag_D,
                float *trianglesArea_D, uint triangleCnt) {

    const uint idx = GetThreadIndex();
    if (idx >= triangleCnt) {
        return;
    }

    trianglesArea_D[idx] *= corruptTriangleFlag_D[idx];
}


cudaError_t MultTriangleAreaWithWeight(float *corruptTriangleFlag_D,
        float *trianglesArea_D, uint triangleCnt) {

    // Create 1D grid layout
    const uint threadsPerBlock = 256;
    const uint blocksPerGrid = (triangleCnt+threadsPerBlock-1)/threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    MultTriangleAreaWithWeight_D <<< grid, threadsPerBlock >>> (
            corruptTriangleFlag_D,
            trianglesArea_D,
            triangleCnt);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'FindCorruptTrianglesFlag_D              %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();

}



__global__ void  CalcInternalForceLen_D(
        float *internalForceLen_D,
        float *vertexPosMapped_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        uint vertexCount,
        float stiffness,
        float forcesScl,
        uint dataArrOffs,
        uint dataArrSize) {


    const uint idx = GetThreadIndex();
    if (idx >= vertexCount) {
        return;
    }

    const uint posBaseIdx = dataArrSize*idx+dataArrOffs;

    float3 pos = make_float3(
            vertexPosMapped_D[posBaseIdx+0],
            vertexPosMapped_D[posBaseIdx+1],
            vertexPosMapped_D[posBaseIdx+2]);

    // Sample gradient by trilinear interpolation
    float4 normalTmp = ::SampleFieldAtPosTricub_D(pos, gradient_D);
    float3 normal;
    normal.x = normalTmp.x;
    normal.y = normalTmp.y;
    normal.z = normalTmp.z;
    normal = normalize(normal);

    // Calculate correction vector based on internal spring forces
    float3 correctionVec = make_float3(0.0, 0.0, 0.0);
    float3 displ;
    float activeNeighbourCnt = 0.00001f; // Prevent division by zero (this should never happen)
    for(int i = 0; i < MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS; ++i) {
        int isIdxValid = int(vertexNeighbours_D[MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS*idx+i] >= 0); // Check if idx != -1
        float3 posNeighbour;
        int tmpIdx = isIdxValid*vertexNeighbours_D[MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS*idx+i]; // Map negative indices to 0
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
        int isIdxValid = int(vertexNeighbours_D[MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS*idx+i] >= 0); // Check if idx != -1
        int tmpIdx = isIdxValid*vertexNeighbours_D[MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS*idx+i]; // Map negative indices to 0
        laplacian2 += (laplacian_D[tmpIdx ]- correctionVec)*isIdxValid;
        //activeNeighbourCnt += 1.0f*isIdxValid;
    }
    laplacian2 /= activeNeighbourCnt; // Represents internal force

    float3 internalForce = (1.0 - stiffness)*correctionVec - stiffness*laplacian2;
    internalForce = internalForce - dot(internalForce, normal)*normal;
    internalForceLen_D[idx] = length(internalForce);
}


cudaError_t CalcInternalForceLen(
        float *internalForceLen_D,
        float *vertexPosMapped_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        uint vertexCount,
        float stiffness,
        float forcesScl,
        uint dataArrOffs,
        uint dataArrSize) {

    // Create 1D grid layout
    const uint threadsPerBlock = 256;
    const uint blocksPerGrid = (vertexCount+threadsPerBlock-1)/threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    // Call kernel
    CalcInternalForceLen_D <<< grid, threadsPerBlock >>> (
            internalForceLen_D,
            vertexPosMapped_D,
            vertexNeighbours_D,
            gradient_D,
            laplacian_D,
            vertexCount,
            stiffness,
            forcesScl,
            dataArrOffs,
            dataArrSize);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'FindCorruptTrianglesFlag_D              %.10f sec\n",
            dt_ms/1000.0);
#endif

    return cudaGetLastError();

}

