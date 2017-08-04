/*
 * CUDACurl.cu
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#include "CUDACurl.cuh"
#include "helper_cuda.h"
#include "helper_math.h"

using namespace megamol;

// TODO
// + Avoid calculation of grid position


// Grid parameters in constant memory
__constant__ CurlGridParams curlParams;


/*
 *  protein_cuda::CudaGetGridPos
 */
__device__
uint3 protein_cuda::CudaGetGridPos(unsigned int idx) {
    uint3 pos;
    // Get position in grid
    // idx = x * w_x * w_z + y * w_z + z
    //     = w_z * ( x * w_y + y ) + z
    pos.z = idx%curlParams.gridDim.z;
    pos.y = (idx/curlParams.gridDim.z)%curlParams.gridDim.y;
    pos.x = ((idx/curlParams.gridDim.z)/curlParams.gridDim.y);
    return pos;
}




/*
 *  protein_cuda::CudaCurlX_D
 */
__global__
void protein_cuda::CudaCurlX_D(float *gridVecFieldD,
                          float *gridCurlD,
                          float hInvHalf) {

    // Get thread index
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= curlParams.gridDim.x*curlParams.gridDim.y*curlParams.gridDim.z) return;

    float curlX = 0.0f;

    /// Calc curl x value ///
    uint3 gridPos = CudaGetGridPos(idx);
    if((gridPos.x == 0)||(gridPos.x == curlParams.gridDim.x-1)) return; // Discard border areas
    if((gridPos.y == 0)||(gridPos.y == curlParams.gridDim.y-1)) return;
    if((gridPos.z == 0)||(gridPos.z == curlParams.gridDim.z-1)) return;

    //uint idx_dyUp = idx + curlParams.gridDim.z;
    uint idx_dyUp = idx + curlParams.gridDim.x;
    float zVal_dyUp = gridVecFieldD[idx_dyUp*3+2];

    //uint idx_dzDown = idx - 1;
    uint idx_dzDown = idx - curlParams.gridDim.x*curlParams.gridDim.y;
    float yVal_dzDown = gridVecFieldD[idx_dzDown*3+1];

    curlX += zVal_dyUp;
    curlX += yVal_dzDown;

    __syncthreads(); // Because otherwise different threads could read the same memory location TODO ?

    //uint idx_dyDown = idx - curlParams.gridDim.z;
    uint idx_dyDown = idx - curlParams.gridDim.x;
    float zVal_dyDown = gridVecFieldD[idx_dyDown*3+2];

    //uint idx_dzUp = idx + 1;
    uint idx_dzUp = idx + curlParams.gridDim.x*curlParams.gridDim.y;
    float yVal_dzUp = gridVecFieldD[idx_dzUp*3+1];

    curlX -= zVal_dyDown;
    curlX -= yVal_dzUp;

    /// Write result ///

    gridCurlD[idx*3+0] = curlX*hInvHalf;
}


/*
 *  protein_cuda::CudaCurlY_D
 */
__global__
void protein_cuda::CudaCurlY_D(float *gridVecFieldD,
                          float *gridCurlD,
                          float hInvHalf) {

    // Get thread index
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= curlParams.gridDim.x*curlParams.gridDim.y*curlParams.gridDim.z) return;

    float curlY = 0.0f;

    /// Calc curl y value ///
    uint3 gridPos = CudaGetGridPos(idx);
    if((gridPos.x == 0)||(gridPos.x == curlParams.gridDim.x-1)) return; // Discard border areas
    if((gridPos.y == 0)||(gridPos.y == curlParams.gridDim.y-1)) return;
    if((gridPos.z == 0)||(gridPos.z == curlParams.gridDim.z-1)) return;

    //uint idx_dzUp = idx + 1;
    uint idx_dzUp = idx + curlParams.gridDim.x*curlParams.gridDim.y;
    float xVal_dzUp = gridVecFieldD[idx_dzUp*3+0];

    //uint idx_dxDown = idx - curlParams.gridDim.z*curlParams.gridDim.y;
    uint idx_dxDown = idx - 1;
    float zVal_dxDown = gridVecFieldD[idx_dxDown*3+2];
    //float zVal_dxDown = gridVecFieldD[0];


    curlY += xVal_dzUp;
    curlY += zVal_dxDown;

    __syncthreads(); // Because otherwise different threads could read the same memory location TODO ?

    //uint idx_dzDown = idx - 1;
    uint idx_dzDown = idx - curlParams.gridDim.x*curlParams.gridDim.y;
    float xVal_dzDown = gridVecFieldD[idx_dzDown*3+0];

    //uint idx_dxUp = idx + curlParams.gridDim.z*curlParams.gridDim.y;
    uint idx_dxUp = idx + 1;
    float zVal_dxUp = gridVecFieldD[idx_dxUp*3+2];

    curlY -= xVal_dzDown;
    curlY -= zVal_dxUp;

    /// Write result ///

    gridCurlD[idx*3+1] = curlY*hInvHalf;
}


/*
 *  protein_cuda::CudaCurlZ_D
 */
__global__
void protein_cuda::CudaCurlZ_D(float *gridVecFieldD,
                          float *gridCurlD,
                          float hInvHalf) {

    // Get thread index
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= curlParams.gridDim.x*curlParams.gridDim.y*curlParams.gridDim.z) return;

    float curlZ = 0.0f;

    /// Calc curl z value ///
    uint3 gridPos = CudaGetGridPos(idx);
    if((gridPos.x == 0)||(gridPos.x == curlParams.gridDim.x-1)) return; // Discard border areas
    if((gridPos.y == 0)||(gridPos.y == curlParams.gridDim.y-1)) return;
    if((gridPos.z == 0)||(gridPos.z == curlParams.gridDim.z-1)) return;

    //uint idx_dxUp = idx + curlParams.gridDim.z*curlParams.gridDim.y;
    uint idx_dxUp = idx + 1;
    float yVal_dxUp = gridVecFieldD[idx_dxUp*3+1];

    //uint idx_dyDown = idx - curlParams.gridDim.z;
    uint idx_dyDown = idx - curlParams.gridDim.x;
    float xVal_dyDown = gridVecFieldD[idx_dyDown*3+0];

    curlZ += yVal_dxUp;
    curlZ += xVal_dyDown;

    __syncthreads(); // Because otherwise different threads could read the same memory location TODO ?

    //uint idx_dxDown = idx - curlParams.gridDim.z*curlParams.gridDim.y;
    uint idx_dxDown = idx - 1;
    float yVal_dxDown = gridVecFieldD[idx_dxDown*3+1];

    //uint idx_dyUp = idx + curlParams.gridDim.z;
    uint idx_dyUp = idx + curlParams.gridDim.x;
    float xVal_dyUp = gridVecFieldD[idx_dyUp*3+0];

    curlZ -= yVal_dxDown;
    curlZ -= xVal_dyUp;


    /// Write result ///

    gridCurlD[idx*3+2] = curlZ*hInvHalf;
}


/*
 * Device function computing the magnitude of the curl value.
 */
__global__
void protein_cuda::CudaCurlMag_D(float *gridCurlD,
                            float *gridCurlMagD) {

    // Get thread index
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= curlParams.gridDim.x*curlParams.gridDim.y*curlParams.gridDim.z) return;

    // Compute magnitude
    float3 vec = make_float3(gridCurlD[3*idx+0],
                             gridCurlD[3*idx+1],
                             gridCurlD[3*idx+2]);

    float res = sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    gridCurlMagD[idx] = res;
}


/*
 * protein_cuda::CudaNormalizeGrid
 */
__global__
void protein_cuda::CudaNormalizeGrid_D(float *gridVecFieldD) {
    // Get thread index
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= curlParams.gridDim.x*curlParams.gridDim.y*curlParams.gridDim.z) return;

    // Compute magnitude
    float3 vec = make_float3(gridVecFieldD[3*idx+0],
                             gridVecFieldD[3*idx+1],
                             gridVecFieldD[3*idx+2]);

    float magVec = sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    //float magVec = 1.0f;

    if(magVec != 0.0) {
        gridVecFieldD[3*idx+0] /= magVec;
        gridVecFieldD[3*idx+1] /= magVec;
        gridVecFieldD[3*idx+2] /= magVec;
    }
}


/**
 * Device function computing the gradient in x direction
 */
__global__
void protein_cuda::CudaGradX_D(float *gridVecFieldD,
                 float *gridGradD,
                 float hInvHalf) {
    // Get thread index
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= curlParams.gridDim.x*curlParams.gridDim.y*curlParams.gridDim.z) return;

    // Discard border areas
    uint3 gridPos = CudaGetGridPos(idx);
    if((gridPos.x == 0)||(gridPos.x == curlParams.gridDim.x-1)) return;
    if((gridPos.y == 0)||(gridPos.y == curlParams.gridDim.y-1)) return;
    if((gridPos.z == 0)||(gridPos.z == curlParams.gridDim.z-1)) return;

    float3 grad = make_float3(0.0, 0.0, 0.0);

    uint idx_dxUp = idx + 1;
    grad.x += gridVecFieldD[idx_dxUp*3+0];
    grad.y += gridVecFieldD[idx_dxUp*3+1];
    grad.z += gridVecFieldD[idx_dxUp*3+2];

    __syncthreads(); // Because otherwise different threads could read the same memory location TODO ?

    uint idx_dxDown = idx - 1;

    grad.x -= gridVecFieldD[idx_dxDown*3+0];
    grad.y -= gridVecFieldD[idx_dxDown*3+1];
    grad.z -= gridVecFieldD[idx_dxDown*3+2];

    grad *= hInvHalf;

    // Write result

    gridGradD[idx*3+0] = grad.x;
    gridGradD[idx*3+1] = grad.y;
    gridGradD[idx*3+2] = grad.z;
}

/**
 * Device function computing the gradient in y direction
 */
__global__
void protein_cuda::CudaGradY_D(float *gridVecFieldD,
                 float *gridGradD,
                 float hInvHalf) {

    // Get thread index
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= curlParams.gridDim.x*curlParams.gridDim.y*curlParams.gridDim.z) return;

    // Discard border areas
    uint3 gridPos = CudaGetGridPos(idx);
    if((gridPos.x == 0)||(gridPos.x == curlParams.gridDim.x-1)) return;
    if((gridPos.y == 0)||(gridPos.y == curlParams.gridDim.y-1)) return;
    if((gridPos.z == 0)||(gridPos.z == curlParams.gridDim.z-1)) return;

    float3 grad = make_float3(0.0, 0.0, 0.0);

    uint idx_dyUp = idx + curlParams.gridDim.x;
    grad.x += gridVecFieldD[idx_dyUp*3+0];
    grad.y += gridVecFieldD[idx_dyUp*3+1];
    grad.z += gridVecFieldD[idx_dyUp*3+2];

    __syncthreads(); // Because otherwise different threads could read the same memory location TODO ?

    uint idx_dyDown = idx - curlParams.gridDim.x;
    grad.x -= gridVecFieldD[idx_dyDown*3+0];
    grad.y -= gridVecFieldD[idx_dyDown*3+1];
    grad.z -= gridVecFieldD[idx_dyDown*3+2];

    grad*=hInvHalf;

    // Write result
    gridGradD[idx*3+0] = grad.x;
    gridGradD[idx*3+1] = grad.y;
    gridGradD[idx*3+2] = grad.z;
}

/**
 * Device function computing the gradient in z direction
 */
__global__
void protein_cuda::CudaGradZ_D(float *gridVecFieldD,
                 float *gridGradD,
                 float hInvHalf) {

    // Get thread index
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= curlParams.gridDim.x*curlParams.gridDim.y*curlParams.gridDim.z) return;

    // Discard border areas
    uint3 gridPos = CudaGetGridPos(idx);
    if((gridPos.x == 0)||(gridPos.x == curlParams.gridDim.x-1)) return;
    if((gridPos.y == 0)||(gridPos.y == curlParams.gridDim.y-1)) return;
    if((gridPos.z == 0)||(gridPos.z == curlParams.gridDim.z-1)) return;

    float3 grad = make_float3(0.0, 0.0, 0.0);

    uint idx_dzUp = idx + curlParams.gridDim.x*curlParams.gridDim.y;
    grad.x += gridVecFieldD[idx_dzUp*3+0];
    grad.y += gridVecFieldD[idx_dzUp*3+1];
    grad.z += gridVecFieldD[idx_dzUp*3+2];

    __syncthreads(); // Because otherwise different threads could read the same memory location TODO ?

    uint idx_dzDown = idx - curlParams.gridDim.x*curlParams.gridDim.y;

    grad.x -= gridVecFieldD[idx_dzDown*3+0];
    grad.y -= gridVecFieldD[idx_dzDown*3+1];
    grad.z -= gridVecFieldD[idx_dzDown*3+2];

    grad*=hInvHalf;

    // Write result
    gridGradD[idx*3+0] = grad.x;
    gridGradD[idx*3+1] = grad.y;
    gridGradD[idx*3+2] = grad.z;
}



extern "C" {


/*
 * protein_cuda::CUDASetCurlParams
 */
cudaError_t protein_cuda::CUDASetCurlParams(CurlGridParams *hostParams) {
    // Copy parameters to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(curlParams, hostParams, sizeof(CurlGridParams)));
    return cudaGetLastError();
}


/*
 * protein_cuda::CudaGetCurlMagnitude
 */
cudaError_t protein_cuda::CudaGetCurlMagnitude(float *gridVecFieldD,
                                          float *gridCurlD,
                                          float *gridCurlMagD,
                                          unsigned int nVoxels,
                                          float gridSpacing) {

    uint nThreadsPerBlock = min(512, nVoxels);
    uint nBlocks  = (uint)ceil((float)nVoxels/(float)nThreadsPerBlock);

    // Set magnitude and curl vector to zero
    checkCudaErrors(cudaMemset(gridCurlMagD, 0, nVoxels*sizeof(float)));
    checkCudaErrors(cudaMemset(gridCurlD, 0, nVoxels*sizeof(float)*3));

    float hInvHalf = 1.0f/(gridSpacing*2.0f);

    protein_cuda::CudaNormalizeGrid_D <<< nBlocks, nThreadsPerBlock >>>
             (gridVecFieldD);
    //getLastCudaError("protein_cuda::CudaNormalizeGrid_D"); // DEBUG

    protein_cuda::CudaCurlX_D <<< nBlocks, nThreadsPerBlock >>>
             (gridVecFieldD, gridCurlD, hInvHalf);
    //getLastCudaError("protein_cuda::CudaCurlX_D"); // DEBUG

    protein_cuda::CudaCurlY_D <<< nBlocks, nThreadsPerBlock >>>
             (gridVecFieldD, gridCurlD, hInvHalf);
    //getLastCudaError("protein_cuda::CudaCurlY_D"); // DEBUG

    protein_cuda::CudaCurlZ_D <<< nBlocks, nThreadsPerBlock >>>
             (gridVecFieldD, gridCurlD, hInvHalf);
    //getLastCudaError("protein_cuda::CudaCurlZ_D"); // DEBUG

    protein_cuda::CudaCurlMag_D <<< nBlocks, nThreadsPerBlock >>>
             (gridCurlD, gridCurlMagD);
    //getLastCudaError("protein_cuda::CudaCurlMagD"); // DEBUG

    return cudaGetLastError();
}


/*
 * protein_cuda::CudaGetGradient
 */
cudaError_t CudaGetGradX(float *gridVecFieldD,
                            float *gridGrad_D,
                            float *gridGrad,
                            unsigned int nVoxels,
                            float gridSpacing) {

    uint nThreadsPerBlock = min(512, nVoxels);
    uint nBlocks  = (uint)ceil((float)nVoxels/(float)nThreadsPerBlock);

    // Set gradient to zero
    checkCudaErrors(cudaMemset(gridGrad_D, 0, nVoxels*sizeof(float)*3));

    float hInvHalf = 1.0f/(gridSpacing*2.0f);

    protein_cuda::CudaGradX_D <<< nBlocks, nThreadsPerBlock >>>
             (gridVecFieldD, gridGrad_D, hInvHalf);
    //getLastCudaError("protein_cuda::CudaGradX_D"); // DEBUG

    // Copy result to host
    checkCudaErrors(cudaMemcpy(gridGrad, gridGrad_D, sizeof(float)*nVoxels*3,
            cudaMemcpyDeviceToHost));

    return cudaGetLastError();
}


/*
 * protein_cuda::CudaGetGradient
 */
cudaError_t CudaGetGradY(float *gridVecFieldD,
                            float *gridGrad_D,
                            float *gridGrad,
                            unsigned int nVoxels,
                            float gridSpacing) {

    uint nThreadsPerBlock = min(512, nVoxels);
    uint nBlocks  = (uint)ceil((float)nVoxels/(float)nThreadsPerBlock);

    // Set gradient to zero
    checkCudaErrors(cudaMemset(gridGrad_D, 0, nVoxels*sizeof(float)*3));

    float hInvHalf = 1.0f/(gridSpacing*2.0f);

    protein_cuda::CudaGradY_D <<< nBlocks, nThreadsPerBlock >>>
             (gridVecFieldD, gridGrad_D, hInvHalf);
    //getLastCudaError("protein_cuda::CudaGradY_D"); // DEBUG

    // Copy result to host
    checkCudaErrors(cudaMemcpy(gridGrad, gridGrad_D, sizeof(float)*nVoxels*3,
            cudaMemcpyDeviceToHost));

    return cudaGetLastError();
}


/*
 * protein_cuda::CudaGetGradient
 */
cudaError_t CudaGetGradZ(float *gridVecFieldD,
                            float *gridGrad_D,
                            float *gridGrad,
                            unsigned int nVoxels,
                            float gridSpacing) {

    uint nThreadsPerBlock = min(512, nVoxels);
    uint nBlocks  = (uint)ceil((float)nVoxels/(float)nThreadsPerBlock);

    // Set gradient to zero
    checkCudaErrors(cudaMemset(gridGrad_D, 0, nVoxels*sizeof(float)*3));

    float hInvHalf = 1.0f/(gridSpacing*2.0f);

    protein_cuda::CudaGradZ_D <<< nBlocks, nThreadsPerBlock >>>
             (gridVecFieldD, gridGrad_D, hInvHalf);
    //getLastCudaError("protein_cuda::CudaGradZ_D"); // DEBUG

    // Copy result to host
    checkCudaErrors(cudaMemcpy(gridGrad, gridGrad_D, sizeof(float)*nVoxels*3,
            cudaMemcpyDeviceToHost));

    return cudaGetLastError();
}


}
