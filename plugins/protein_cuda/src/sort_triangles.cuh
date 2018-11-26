//
// sort_triangles.cuh
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 29, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_SORT_TRIANGLES_CUH_INCLUDED
#define MMPROTEINCUDAPLUGIN_SORT_TRIANGLES_CUH_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

typedef unsigned int uint;

/**
 * Sorts an array of triangles (defined by index array and vertex positions)
 * by their distance to the camera position 'camPos'. The distance is calculated
 * qith respect to the triangles midpoint.
 *
 * @param[in] dataBuff_D           Array containing the vertex data (device
 *                                 memory)
 * @param[in] dataBuffSize         Stride for the vertex data
 * @param[in] dataBuffOffsPos      Offset for vertex position in the vertex data
 *                                 buffer
 * @param[in] camPos               The camera position
 * @param[in,out] triangleVtxIdx_D The array containing the triangle vertex
 *                                 indices (device memory). This is also the
 *                                 output array.
 * @param[in] triangleCnt          The number of triangles to be sorted
 * @param[in, out] triangleCamDistance_D Array containing the distance to the
 *                                       camera position of all triangles
 * @return 'cudaSuccess' on success, the respective error enum otherwise
 */
extern "C" cudaError_t SortTrianglesByCamDistance(
        float *dataBuff_D,
        uint dataBuffSize,
        uint dataBuffOffsPos,
        float3 camPos,
        uint *triangleVtxIdx_D,
        uint triangleCnt,
        float *triangleCamDistance_D);

#endif /* SORT_TRIANGLES_CUH_ */
