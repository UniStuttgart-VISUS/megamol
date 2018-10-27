/*
 * CUDACurl.cuh
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#ifndef MMPROTEINCUDAPLUGIN_CUDACURL_CUH
#define MMPROTEINCUDAPLUGIN_CUDACURL_CUH

#include <vector_types.h>
#include <driver_types.h>


/** Struct holding several grid params */
struct CurlGridParams {
    uint3   gridDim;   // Grid dimension defining the number of voxels
    float3  gridOrg;   // Origin of the grid in world coords
    float3  gridStep;  // Size of grid cells
    float3  gridXAxis; // X axis of the grid, not normalized!
    float3  gridYAxis; // Y axis of the grid, not normalized!
    float3  gridZAxis; // Z axis of the grid, not normalized!
};

namespace megamol {
namespace protein_cuda {

/**
 * Device function to map thread index to 3D grid position
 */
__device__
uint3 CudaGetGridPos(unsigned int idx);

/**
 * Device function computing the curl value in x direction
 */
__global__
void CudaCurlX_D(float *gridVecFieldD,
		         float *gridCurlD,
		         float hInvHalf);

/**
 * Device function computing the curl value in y direction
 */
__global__
void CudaCurlY_D(float *gridVecFieldD,
		         float *gridCurlD,
		         float hInvHalf);

/**
 * Device function computing the curl value in z direction
 */
__global__
void CudaCurlZ_D(float *gridVecFieldD,
		         float *gridCurlD,
		         float hInvHalf);

/**
 * Device function computing the gradient in x direction
 */
__global__
void CudaGradX_D(float *gridVecFieldD,
		         float *gridCurlD,
		         float hInvHalf);

/**
 * Device function computing the gradient in y direction
 */
__global__
void CudaGradY_D(float *gridVecFieldD,
		         float *gridCurlD,
		         float hInvHalf);

/**
 * Device function computing the gradient in z direction
 */
__global__
void CudaGradZ_D(float *gridVecFieldD,
		         float *gridCurlD,
		         float hInvHalf);

/**
 * Device function computing the magnitude of the curl value.
 */
__global__
void CudaCurlMag_D(float *gridCurlD,
		           float *gridCurlMagD);

/**
 * Device function which normalizes all vectors in the grid and stores filter
 * information.
 */
__global__
void CudaNormalizeGrid_D(float *gridVecFieldD);


extern "C" {


/**
 * Copy grid parameters to constant device memory.
 *
 * @return 'cudaSuccess' on success, the according cuda error otherwise
 */
cudaError_t CUDASetCurlParams(CurlGridParams *hostParams);


/**
 * Apply curl operator to a given vector field.
 *
 * @param gridVecFieldD 3D uniform grid containing the vector field (device memory)
 * @param gridCurlD     3D uniform grid containing the curl vector(device memory)
 * @param gridCurlMagD  3D uniform grid containing the magnitude of the curl vector (device memory)
 * @param nVoxels       The number of voxels in the grid
 * @param gridSpacing   The cell size of the grid
 * @return 'cudaSuccess' on success, the according cuda error otherwise
 */
cudaError_t CudaGetCurlMagnitude(float *gridVecFieldD,
                                 float *gridCurlD,
                                 float *gridCurlMagD,
                                 unsigned int nVoxels,
                                 float gridSpacing);

/**
 * Compute gradient of a given uniform vector field in x direction.
 *
 * @param gridVecFieldD 3D uniform grid containing the vector field (device memory)
 * @param gridGradD     3D uniform grid containing the gradient (device memory)
 * @param gridGrad      3D uniform grid containing the gradient (host memory)
 * @param nVoxels       The number of voxels in the grid
 * @param gridSpacing   The cell size of the grid
 * @return 'cudaSuccess' on success, the according cuda error otherwise
 */
cudaError_t CudaGetGradX(float *gridVecFieldD,
                            float *gridGradD,
                            float *gridGrad,
                            unsigned int nVoxels,
                            float gridSpacing);

/**
 * Compute gradient of a given uniform vector field in y direction.
 *
 * @param gridVecFieldD 3D uniform grid containing the vector field (device memory)
 * @param gridGradD     3D uniform grid containing the gradient (device memory)
 * @param gridGrad      3D uniform grid containing the gradient (host memory)
 * @param nVoxels       The number of voxels in the grid
 * @param gridSpacing   The cell size of the grid
 * @return 'cudaSuccess' on success, the according cuda error otherwise
 */
cudaError_t CudaGetGradY(float *gridVecFieldD,
                            float *gridGradD,
                            float *gridGrad,
                            unsigned int nVoxels,
                            float gridSpacing);

/**
 * Compute gradient of a given uniform vector field in z direction.
 *
 * @param gridVecFieldD 3D uniform grid containing the vector field (device memory)
 * @param gridGradD     3D uniform grid containing the gradient (device memory)
 * @param gridGrad      3D uniform grid containing the gradient (host memory)
 * @param nVoxels       The number of voxels in the grid
 * @param gridSpacing   The cell size of the grid
 * @return 'cudaSuccess' on success, the according cuda error otherwise
 */
cudaError_t CudaGetGradZ(float *gridVecFieldD,
                            float *gridGradD,
                            float *gridGrad,
                            unsigned int nVoxels,
                            float gridSpacing);

}

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MMPROTEINCUDAPLUGIN_CUDACURL_CUH */
