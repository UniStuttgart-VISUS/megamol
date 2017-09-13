//
// ComparativeSurfacePotentialRenderer_inline_device_functions.cuh
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 10, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_COMPARATIVESURFACEPOTENTIALRENDERER_INLINE_DEVICE_FUNCTIONS_CUH_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_COMPARATIVESURFACEPOTENTIALRENDERER_INLINE_DEVICE_FUNCTIONS_CUH_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ComparativeSurfacePotentialRenderer.cuh"
#include "constantGridParams.cuh"

#include "helper_math.h"
#include "interpol.cuh"
#include <cstdio>
#include <cassert>


////////////////////////////////////////////////////////////////////////////////
//  Redifinitions of some mathfunctions that prevent dividing by zero         //
////////////////////////////////////////////////////////////////////////////////

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

/**
 * 'Safe' normalize function for float4 that uses safe rsqrt
 *
 * @param v The input vector to be normalized
 * @return The normalized vector v
 */
inline __device__ float4 safeNormalize(float4 v) {
    float invLen = safeRsqrtf(dot(v, v));
    return v * invLen;
}


////////////////////////////////////////////////////////////////////////////////
//  Grid utility functions offering coversion of different indices. The term  //
//  'cell' refers to grid centers rather than grid corners. Thus, there are   //
//  gridSize - 1 cells in every dimension.                                    //
////////////////////////////////////////////////////////////////////////////////

/**
 * Answers the grid position index associated with the given coordinates.
 *
 * @param v0 The coordinates
 * @return The index
 */
inline __device__ uint GetPosIdxByGridCoords(uint3 v0) {
    return gridSize_D.x*(gridSize_D.y*v0.z + v0.y) + v0.x;
}

/**
 * Answers the cell index associated with the given coordinates.
 *
 * @param v0 The coordinates
 * @return The index
 */
inline __device__ uint GetCellIdxByGridCoords(int3 v0) {
    return (gridSize_D.x-1)*((gridSize_D.y-1)*v0.z + v0.y) + v0.x;
}

/**
 * Answers the grid position coordinates associated with a given cell index.
 * The returned position is the left/lower/back corner of the cell
 *
 * @param index The index
 * @return The coordinates
 */
inline __device__ uint3 GetGridCoordsByCellIdx(uint index) {
    return make_uint3(index % (gridSize_D.x-1),
                      (index / (gridSize_D.x-1)) % (gridSize_D.y-1),
                      (index / (gridSize_D.x-1)) / (gridSize_D.y-1));
}

/**
 * Answers the cell coordinates associated with a given grid position index.
 *
 * @param index The index
 * @return The coordinates
 */
inline __device__ uint3 GetGridCoordsByPosIdx(uint index) {
    return make_uint3(index % gridSize_D.x,
                      (index / gridSize_D.x) % gridSize_D.y,
                      (index / gridSize_D.x) / gridSize_D.y);
}

/**
 * Transforms grid coordinates to world space coordinates based on the grid
 * origin and spacing
 *
 * @param[in] pos The grid coordinates
 *
 * @return The coordinates in world space
 */
inline __device__ float3 TransformToWorldSpace(float3 pos) { // TODO Renameand take uint3 as input
    return make_float3(gridOrg_D.x + pos.x*gridDelta_D.x,
                       gridOrg_D.y + pos.y*gridDelta_D.y,
                       gridOrg_D.z + pos.z*gridDelta_D.z);
}

/**
 * Transforms world space  coordinates to grid coordinates based on the grid
 * origin and spacing
 *
 * @param[in] pos The coordinates in world space
 *
 * @return The grid coordinates
 */
//inline __device__ uint3 WSToGridCoords(float3 pos) {
//    return make_uint3(pos.x/gridDelta_D.x - gridOrg_D.x,
//                      pos.y/gridDelta_D.y - gridOrg_D.y,
//                      pos.z/gridDelta_D.z - gridOrg_D.z);
//}


////////////////////////////////////////////////////////////////////////////////
//  Sampling routines using linear and cubic interpolation schemes            //
////////////////////////////////////////////////////////////////////////////////

/**
 * Samples the field at a given (integer) grid position.
 *
 * @param x,y,z Coordinates of the position
 * @return The sampled value of the field
 */
template <typename T>
inline __device__ T SampleFieldAt_D(uint x, uint y, uint z, T *field_D) {
    return field_D[gridSize_D.x*(gridSize_D.y*z+y)+x];
}

/**
 * Samples the field at a given (integer) grid position.
 *
 * @param index The position
 * @return The sampled value of the field
 */
template <typename T>
inline __device__ T SampleFieldAt_D(uint3 index, T *field_D) {
    return SampleFieldAt_D<T>(index.x, index.y, index.z, field_D);
}

/**
 * Samples the field at a given position using linear interpolation.
 *
 * @param pos The position
 * @return The sampled value of the field
 */
template <typename T>
inline __device__ T SampleFieldAtPosTrilin_D(float3 pos, T *field_D) {



    uint3 c;
    float3 f;

    // Get id of the cell containing the given position and interpolation
    // coefficients
    f.x = (pos.x-gridOrg_D.x)/gridDelta_D.x;
    f.y = (pos.y-gridOrg_D.y)/gridDelta_D.y;
    f.z = (pos.z-gridOrg_D.z)/gridDelta_D.z;
    c.x = (uint)(f.x);
    c.y = (uint)(f.y);
    c.z = (uint)(f.z);
    f.x = f.x-(float)c.x; // alpha
    f.y = f.y-(float)c.y; // beta
    f.z = f.z-(float)c.z; // gamma

    c.x = clamp(c.x, uint(0), gridSize_D.x-2);
    c.y = clamp(c.y, uint(0), gridSize_D.y-2);
    c.z = clamp(c.z, uint(0), gridSize_D.z-2);

    // Get values at corners of current cell
    T s[8];
    s[0] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+0) + (c.y+0))+c.x+0];
    s[1] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+0) + (c.y+0))+c.x+1];
    s[2] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+0) + (c.y+1))+c.x+0];
    s[3] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+0) + (c.y+1))+c.x+1];
    s[4] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+1) + (c.y+0))+c.x+0];
    s[5] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+1) + (c.y+0))+c.x+1];
    s[6] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+1) + (c.y+1))+c.x+0];
    s[7] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+1) + (c.y+1))+c.x+1];

    // Use trilinear interpolation to sample the volume
   return InterpFieldTrilin_D(s, f.x, f.y, f.z);
}

/**
 * Samples the field at a given position using cubic interpolation.
 *
 * @param pos The position
 * @return The sampled value of the field
 */
template <typename T>
inline __device__ T SampleFieldAtPosTricub_D(float x, float y, float z, T *field_D) {
    ::SampleFieldAtPosTricub_D<T>(make_float3(x, y, z));
}

/**
 * Samples the field at a given position using cubic interpolation.
 *
 * @param pos The position
 * @return The sampled value of the field
 */
template <typename T>
inline __device__ T SampleFieldAtPosTricub_D(float3 pos, T *field_D) {

    uint3 c;
    float3 f;

    // Get id of the cell containing the given position and interpolation
    // coefficients
    f.x = (pos.x-gridOrg_D.x)/gridDelta_D.x;
    f.y = (pos.y-gridOrg_D.y)/gridDelta_D.y;
    f.z = (pos.z-gridOrg_D.z)/gridDelta_D.z;
    c.x = (uint)(f.x);
    c.y = (uint)(f.y);
    c.z = (uint)(f.z);

    c.x = clamp(c.x, uint(1), gridSize_D.x-3);
    c.y = clamp(c.y, uint(1), gridSize_D.y-3);
    c.z = clamp(c.z, uint(1), gridSize_D.z-3);

    f.x = f.x-(float)c.x; // alpha
    f.y = f.y-(float)c.y; // beta
    f.z = f.z-(float)c.z; // gamma



    // Get values at cell corners; coords are defined relative to cell origin

    volatile const uint one = 1;
    //volatile const uint two = 2;


    T sX0 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y - one, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y - one, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y - one, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y - one, c.z + 2), field_D),
            f.z);

    T sX1 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y, c.z + 2), field_D),
            f.z);

    T sX2  = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y + one, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y + one, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y + one, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y + one, c.z + 2), field_D),
            f.z);

    T sX3 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y + 2, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y + 2, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y + 2, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x - one, c.y + 2, c.z + 2), field_D),
            f.z);

    T s0 = ::InterpFieldCubicSepArgs_D<T>(sX0, sX1, sX2, sX3, f.y);

    sX0 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x, c.y - one, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x, c.y - one, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x, c.y - one, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x, c.y - one, c.z + 2), field_D),
            f.z);

    sX1 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x, c.y, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x, c.y, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x, c.y, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x, c.y, c.z + 2), field_D),
            f.z);

    sX2 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x, c.y + one, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x, c.y + one, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x, c.y + one, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x, c.y + one, c.z + 2), field_D),
            f.z);

    sX3 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x, c.y + 2, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x, c.y + 2, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x, c.y + 2, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x, c.y + 2, c.z + 2), field_D),
            f.z);

    T s1 = ::InterpFieldCubicSepArgs_D<T>(sX0, sX1, sX2, sX3, f.y);

    sX0 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y - one, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y - one, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y - one, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y - one, c.z + 2), field_D),
            f.z);

    sX1 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y, c.z + 2), field_D),
            f.z);

    sX2 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y + one, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y + one, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y + one, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y + one, c.z + 2), field_D),
            f.z);

    sX3 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y + 2, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y + 2, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y + 2, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + one, c.y + 2, c.z + 2), field_D),
            f.z);

    T s2 = ::InterpFieldCubicSepArgs_D<T>(sX0, sX1, sX2, sX3, f.y);

    sX0 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y - one, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y - one, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y - one, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y - one, c.z + 2), field_D),
            f.z);

    sX1 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y, c.z + 2), field_D),
            f.z);

    sX2 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y + one, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y + one, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y + one, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y + one, c.z + 2), field_D),
            f.z);

    sX3 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y + 2, c.z - one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y + 2, c.z), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y + 2, c.z + one), field_D),
            SampleFieldAt_D<T>(make_uint3(c.x + 2, c.y + 2, c.z + 2), field_D),
            f.z);

    T s3 = ::InterpFieldCubicSepArgs_D<T>(sX0, sX1, sX2, sX3, f.y);

    return ::InterpFieldCubicSepArgs_D<T>(s0, s1, s2, s3, f.x);
}


////////////////////////////////////////////////////////////////////////////////
//  Utility functions concerning the CUDA grid definition.                    //
////////////////////////////////////////////////////////////////////////////////

/**
 * @return Returns the thread index based on the current CUDA grid dimensions
 */
inline __device__ uint GetThreadIndex() {
    return __umul24(__umul24(blockIdx.y, gridDim.x) + blockIdx.x, blockDim.x) +
            threadIdx.x;
}

#endif // MMPROTEINCUDAPLUGIN_COMPARATIVESURFACEPOTENTIALRENDERER_INLINE_DEVICE_FUNCTIONS_CUH_H_INCLUDED
