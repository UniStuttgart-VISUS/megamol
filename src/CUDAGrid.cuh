//
// CUDAGrid.cuh
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Oct 16, 2013
// Author     : scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_CUDAGRID_CUH_INCLUDED
#define MMPROTEINCUDAPLUGIN_CUDAGRID_CUH_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "interpol.cuh"
#include "helper_math.h"

__constant__ __device__ int3 gridSize_D;     // The size of the volume texture
__constant__ __device__ float3 gridOrg_D;    // The origin of the volume texture
__constant__ __device__ float3 gridDelta_D;  // The spacing of the volume texture

/*
 * initGridParams
 */
inline bool initGridParams(int3 gridSize, float3 org, float3 delta) {
    cudaMemcpyToSymbol(gridSize_D, &gridSize, sizeof(uint3));
    cudaMemcpyToSymbol(gridOrg_D, &org, sizeof(float3));
    cudaMemcpyToSymbol(gridDelta_D, &delta, sizeof(float3));
//    printf("Init grid with org %f %f %f, delta %f %f %f, dim %u %u %u\n", org.x,
//            org.y, org.z, delta.x, delta.y, delta.z, gridSize.x, gridSize.y,
//            gridSize.z);
    return CudaSafeCall(cudaGetLastError());
}


/**
 * Samples the field at a given (integer) grid position.
 *
 * @param x,y,z Coordinates of the position
 * @return The sampled value of the field
 */
template <typename T, bool W>
inline __device__ T SampleFieldAt_D(uint x, uint y, uint z, T *field_D) {
    // Texture wrapping
    if (W) { // W should be a global parameter to avoid branching
        x = x%gridSize_D.x;
        y = y%gridSize_D.y;
        z = z%gridSize_D.z;
    }
    return field_D[gridSize_D.x*(gridSize_D.y*z+y)+x];
}

/**
 * Samples the field at a given (integer) grid position.
 *
 * @param index The position
 * @return The sampled value of the field
 */
template <typename T, bool W>
inline __device__ T SampleFieldAt_D(uint3 index, T *field_D) {
    return SampleFieldAt_D<T, W>(index.x, index.y, index.z, field_D);
}


/**
 * Samples the field at a given position using linear interpolation.
 *
 * @param pos The position
 * @return The sampled value of the field
 */
template <typename T, bool W>
inline __device__ T SampleFieldAtPosTrilin_D(float x, float y, float z, T *field_D) {
    ::SampleFieldAtPosTrilin_D<T, W>(make_float3(x, y, z));
}


/**
 * Samples the field at a given position using linear interpolation.
 *
 * @param pos The position
 * @return The sampled value of the field
 */
template <typename T, bool W>
inline __device__ T SampleFieldAtPosTrilin_D(float3 pos, T *field_D) {

    int3 c, c1;
    float3 f;

    // Get id of the cell containing the given position and interpolation
    // coefficients
    f.x = (pos.x-gridOrg_D.x)/gridDelta_D.x;
    f.y = (pos.y-gridOrg_D.y)/gridDelta_D.y;
    f.z = (pos.z-gridOrg_D.z)/gridDelta_D.z;
    c.x = (int)(f.x);
    c.y = (int)(f.y);
    c.z = (int)(f.z);
    f.x = f.x-(float)c.x; // alpha
    f.y = f.y-(float)c.y; // beta
    f.z = f.z-(float)c.z; // gamma

    if (W) { // Wrap coordinates
        c.x = c.x%gridSize_D.x;
        c.y = c.y%gridSize_D.y;
        c.z = c.z%gridSize_D.z;
        c1.x = (c.x+1)%gridSize_D.x;
        c1.y = (c.y+1)%gridSize_D.y;
        c1.z = (c.z+1)%gridSize_D.z;
    } else {
        c.x = clamp(c.x, int(0), gridSize_D.x-2);
        c.y = clamp(c.y, int(0), gridSize_D.y-2);
        c.z = clamp(c.z, int(0), gridSize_D.z-2);
        c1.x = clamp(c.x+1, int(0), gridSize_D.x-1);
        c1.y = clamp(c.y+1, int(0), gridSize_D.y-1);
        c1.z = clamp(c.z+1, int(0), gridSize_D.z-1);
    }

    // Get values at corners of current cell
    T s[8];
    s[0] = field_D[gridSize_D.x*(gridSize_D.y*(c.z) + (c.y))+c.x];
    s[1] = field_D[gridSize_D.x*(gridSize_D.y*(c.z) + (c.y))+c1.x];
    s[2] = field_D[gridSize_D.x*(gridSize_D.y*(c.z) + (c1.y))+c.x];
    s[3] = field_D[gridSize_D.x*(gridSize_D.y*(c.z) + (c1.y))+c1.x];
    s[4] = field_D[gridSize_D.x*(gridSize_D.y*(c1.z) + (c.y))+c.x];
    s[5] = field_D[gridSize_D.x*(gridSize_D.y*(c1.z) + (c.y))+c1.x];
    s[6] = field_D[gridSize_D.x*(gridSize_D.y*(c1.z) + (c1.y))+c.x];
    s[7] = field_D[gridSize_D.x*(gridSize_D.y*(c1.z) + (c1.y))+c1.x];

//    s[0] = SampleFieldAt_D<T, true>(make_uint3(c.x+0, c.y+0, c.z+0), field_D);
//    s[1] = SampleFieldAt_D<T, true>(make_uint3(c.x+1, c.y+0, c.z+0), field_D);
//    s[2] = SampleFieldAt_D<T, true>(make_uint3(c.x+0, c.y+1, c.z+0), field_D);
//    s[3] = SampleFieldAt_D<T, true>(make_uint3(c.x+1, c.y+1, c.z+0), field_D);
//    s[4] = SampleFieldAt_D<T, true>(make_uint3(c.x+0, c.y+0, c.z+1), field_D);
//    s[5] = SampleFieldAt_D<T, true>(make_uint3(c.x+1, c.y+0, c.z+1), field_D);
//    s[6] = SampleFieldAt_D<T, true>(make_uint3(c.x+0, c.y+1, c.z+1), field_D);
//    s[7] = SampleFieldAt_D<T, true>(make_uint3(c.x+1, c.y+1, c.z+1), field_D);

    // Use trilinear interpolation to sample the volume
   return InterpFieldTrilin_D<T>(s, f.x, f.y, f.z);
}

/**
 * Samples the field at a given position using cubic interpolation.
 *
 * @param pos The position
 * @return The sampled value of the field
 */
template <typename T, bool W>
inline __device__ T SampleFieldAtPosTricub_D(float x, float y, float z, T *field_D) {
    ::SampleFieldAtPosTricub_D<T, W>(make_float3(x, y, z));
}

/**
 * Samples the field at a given position using cubic interpolation.
 *
 * @param pos The position
 * @return The sampled value of the field
 */
template <typename T, bool W>
inline __device__ T SampleFieldAtPosTricub_D(float3 pos, T *field_D) {

    int3 c;
    float3 f;

    // Get id of the cell containing the given position and interpolation
    // coefficients
    f.x = (pos.x-gridOrg_D.x)/gridDelta_D.x;
    f.y = (pos.y-gridOrg_D.y)/gridDelta_D.y;
    f.z = (pos.z-gridOrg_D.z)/gridDelta_D.z;
    c.x = (int)(f.x);
    c.y = (int)(f.y);
    c.z = (int)(f.z);

    c.x = clamp(c.x, int(1), gridSize_D.x-3);
    c.y = clamp(c.y, int(1), gridSize_D.y-3);
    c.z = clamp(c.z, int(1), gridSize_D.z-3);

    f.x = f.x-(float)c.x; // alpha
    f.y = f.y-(float)c.y; // beta
    f.z = f.z-(float)c.z; // gamma



    // Get values at cell corners; coords are defined relative to cell origin

    volatile const uint one = 1;
    //volatile const uint two = 2;


    T sX0 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y - one, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y - one, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y - one, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y - one, c.z + 2), field_D),
            f.z);

    T sX1 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y, c.z + 2), field_D),
            f.z);

    T sX2  = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y + one, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y + one, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y + one, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y + one, c.z + 2), field_D),
            f.z);

    T sX3 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y + 2, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y + 2, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y + 2, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x - one, c.y + 2, c.z + 2), field_D),
            f.z);

    T s0 = ::InterpFieldCubicSepArgs_D<T>(sX0, sX1, sX2, sX3, f.y);

    sX0 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y - one, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y - one, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y - one, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y - one, c.z + 2), field_D),
            f.z);

    sX1 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y, c.z + 2), field_D),
            f.z);

    sX2 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y + one, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y + one, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y + one, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y + one, c.z + 2), field_D),
            f.z);

    sX3 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y + 2, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y + 2, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y + 2, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x, c.y + 2, c.z + 2), field_D),
            f.z);

    T s1 = ::InterpFieldCubicSepArgs_D<T>(sX0, sX1, sX2, sX3, f.y);

    sX0 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y - one, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y - one, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y - one, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y - one, c.z + 2), field_D),
            f.z);

    sX1 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y, c.z + 2), field_D),
            f.z);

    sX2 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y + one, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y + one, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y + one, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y + one, c.z + 2), field_D),
            f.z);

    sX3 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y + 2, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y + 2, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y + 2, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + one, c.y + 2, c.z + 2), field_D),
            f.z);

    T s2 = ::InterpFieldCubicSepArgs_D<T>(sX0, sX1, sX2, sX3, f.y);

    sX0 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y - one, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y - one, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y - one, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y - one, c.z + 2), field_D),
            f.z);

    sX1 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y, c.z + 2), field_D),
            f.z);

    sX2 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y + one, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y + one, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y + one, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y + one, c.z + 2), field_D),
            f.z);

    sX3 = ::InterpFieldCubicSepArgs_D<T>(
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y + 2, c.z - one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y + 2, c.z), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y + 2, c.z + one), field_D),
            SampleFieldAt_D<T, W>(make_uint3(c.x + 2, c.y + 2, c.z + 2), field_D),
            f.z);

    T s3 = ::InterpFieldCubicSepArgs_D<T>(sX0, sX1, sX2, sX3, f.y);

    return ::InterpFieldCubicSepArgs_D<T>(s0, s1, s2, s3, f.x);
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
 * Answers the grid position index associated with the given coordinates.
 *
 * @param v0 The coordinates
 * @return The index
 */
inline __device__ uint GetPosIdxByGridCoords(int3 v0) {
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
 * Answers the cell index associated with the given coordinates.
 *
 * @param v0 The coordinates
 * @return The index
 */
inline __device__ uint GetCellIdxByGridCoords(uint3 v0) {
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
 * Checks whether a given WS position is inside the grid.
 *
 * @param pos The world space position
 * @return 'True' if the position is inside the grid
 */
inline __device__ bool IsValidGridpos(float3 pos) {
    /*printf("IsValidGridPos: %f %f %f\n", pos.X(), pos.Y(), pos.Z());
    printf("Min %f %f %f\n", this->orgX, this->orgY, this->orgZ);
    printf("Max %f %f %f\n", (this->orgX + (this->dimX-1)*this->spacingX),
            (this->orgY + (this->dimY-1)*this->spacingY),
            (this->orgZ + (this->dimZ-1)*this->spacingZ));*/
    if (pos.x < gridOrg_D.x) return false;
    if (pos.y < gridOrg_D.y) return false;
    if (pos.z < gridOrg_D.z) return false;
    if (pos.x >= (gridOrg_D.x + (gridSize_D.x-1)*gridDelta_D.x)) return false;
    if (pos.y >= (gridOrg_D.y + (gridSize_D.y-1)*gridDelta_D.y)) return false;
    if (pos.z >= (gridOrg_D.z + (gridSize_D.z-1)*gridDelta_D.z)) return false;
    return true;
}

#endif // MMPROTEINCUDAPLUGIN_CUDAGRID_CUH_INCLUDED
