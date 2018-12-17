//
// CUDAFieldTopology.cuh
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MMPROTEINCUDAPLUGIN_CUDAFIELDTOPOLOGY_CUH
#define MMPROTEINCUDAPLUGIN_CUDAFIELDTOPOLOGY_CUH

#include <vector_types.h>
#include <driver_types.h>

typedef unsigned int uint;

extern "C" {

/// Streamline integration

/** TODO */
cudaError_t CalcGradient(float *scalarDield_D, float *gradientField_D, uint volsize);

cudaError_t InitStartPos(float *vertexDataBuffer_D, float *streamlinePos_D,
        uint vertexDataBufferStride, uint vertexDataBufferOffsPos, uint vertexCnt);

cudaError_t UpdateStreamlinePos(float *streamlinePos_D,float *gradientField_D, uint vertexCnt, uint step);







    
/**
 * Copy grid dimensions to constant device memory.
 *
 * @param dim      The dimensions of the grid
 * @param org      The origin of the grid
 * @param maxCoord The maximum coordinates ofthe grid (origin = minimum)
 * @param dim      The spacing of the grid
 * @return 'cudaSuccess' on success, the according cuda error otherwise
 */
cudaError_t SetGridParams(uint3 dim_h, float3 org_h, float3 maxCoord_h, 
                          float3 spacing_h);

/**
 * Copy streamline stepsize to constant device memory.
 *
 * @param[in] stepsize The streamline stepsize
 * qparam[in] maxSteps The maximum number of steps
 * @return 'cudaSuccess' on success, the according cuda error otherwise
 */
cudaError_t SetStreamlineParams(float stepsize_h, uint maxSteps);

/**
 * Copy the number of positions to constant device memory.
 *
 * @param[in] nPos The number of positions
 * @return 'cudaSuccess' on success, the according cuda error otherwise
 */
cudaError_t SetNumberOfPos(uint nPos_h);

/**
 * Integrate vector field at current position to determine next position. This
 * function is used to find ending points of streamlines with respect to a set of 
 * given starting points. Integration is done using a fourth order Runge Kutta
 * method.
 *
 * @param vecField The vector field
 * @param dim      The dimensions of the grid defining the vector field
 * @param pos      The array containing the current positions
 * @param nPos     The number of positions to keep track of
 * @param maxIt    The maximum number of iteration steps before the positions
 *                 are copied back to host memory
 * @param backward If 'true', the integration is done backward
 * @return 'cudaSuccess' on success, the according cuda error otherwise
 */
cudaError_t UpdatePositionRK4(
        const float *vecField,
        uint3 dim,
        const float *pos,
        uint nPos,
        uint maxIt, 
        bool backward = false);

/**
 * Search for cells in which the vector field is vanishing. Afterwards the
 * output array contains alpha, beta, gamma values for all cells which mark
 * the center of the subcell containing the critical point. If a cell does not
 * contain a critical point, alpha, beta and gamma are set to -1.
 *
 * @param vecField     The vector field
 * @param dim          The dimensions of the grid containing the vector fiel
 * @param cellCoords   Output array for the coordinates inside the cells
 * @param maxStackSize Maximum stack size
 * @return 'cudaSuccess' on success, the according cuda error otherwise
 */
cudaError_t SearchNullPoints(
        const float *vecField, // Input
        uint3 dim,             // Input
        float3 org,            // Input
        float3 spacing,        // Input
        float *cellCoords,     // Output
        unsigned int maxStackSize);

}

#endif // MMPROTEINCUDAPLUGIN_CUDAFIELDTOPOLOGY_CUH
