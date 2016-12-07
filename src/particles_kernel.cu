/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

/* 
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include "math_constants.h"
#include "particles_kernel.cuh"
#include "helper_math.h"

// texture for particle position
texture<float4, 1, cudaReadModeElementType> oldPosTex;
// texture for atom position
texture<float4, 1, cudaReadModeElementType> atomPosTex;
// texture for number of neighbor atoms
texture<uint, 1, cudaReadModeElementType> neighborCountTex;
// texture for neighbor atoms (indices)
texture<uint, 1, cudaReadModeElementType> neighborsTex;
// texture for small circles (vector to center 
texture<float4, 1, cudaReadModeElementType> smallCirclesTex;
// texture for arcs
texture<float4, 1, cudaReadModeElementType> arcsTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;

texture<float4, 1, cudaReadModeElementType> visibleAtomsTex;
texture<uint, 1, cudaReadModeElementType> visibleAtomsIdTex;
//texture<uint4, 1, cudaReadModeElementType> point1Tex;

texture<float4, 2, cudaReadModeElementType> inVisibilityTex;

// simulation parameters in constant memory
__constant__ SimParams params;

// Reduced Surfaec parameters in constant memory
__constant__ RSParams rsParams;

///////////////////////////////////////////////////////////////////////////////
// calculate position in uniform grid
///////////////////////////////////////////////////////////////////////////////
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

///////////////////////////////////////////////////////////////////////////////
// calculate address in grid from position (clamping to edges)
///////////////////////////////////////////////////////////////////////////////
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

///////////////////////////////////////////////////////////////////////////////
// calculate grid hash value for each particle
///////////////////////////////////////////////////////////////////////////////
__global__
void calcHashD(uint*   gridParticleHash,  // output
               uint*   gridParticleIndex, // output
               float4* pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;
    
    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

///////////////////////////////////////////////////////////////////////////////
// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
///////////////////////////////////////////////////////////////////////////////
__global__
void reorderDataAndFindCellStartD(uint*   cellStart,        // output: cell start index
                                  uint*   cellEnd,          // output: cell end index
                                  float4* sortedPos,        // output: sorted positions
                                  uint *  gridParticleHash, // input: sorted grid hashes
                                  uint *  gridParticleIndex,// input: sorted particle indices
                                  float4* oldPos,           // input: sorted position array
                                  uint    numParticles)
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    
    uint hash;
    // handle case when no. of particles not multiple of block size
    if (index < numParticles) {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look 
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    __syncthreads();
    
    if( index < numParticles ) {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x]) {
            cellStart[hash] = index;
            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1) {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos data
        uint sortedIndex = gridParticleIndex[index];
        //float4 pos = FETCH( oldPos, sortedIndex);       // macro does either global read or texture fetch
        float4 pos = oldPos[sortedIndex];       // Do not use texture for large arrays!

        sortedPos[index] = pos;
    }
}

///////////////////////////////////////////////////////////////////////////////
// count all neighbor atoms in a given cell
///////////////////////////////////////////////////////////////////////////////
__device__
uint countNeighborsInCell( uint*   neighbors,     // output: neighbor indices
                           float4* smallCircles,  // output: small circles
                           uint    neighborIndex, // input: first index for writing in neighbor list
                           uint    atomIndex,     // input: atom index for writing in neighbor list
                           int3    gridPos,
                           uint    index,
                           float4  pos,
                           float4* atomPos,
                           uint*   gridParticleIndex,    // input: sorted atom indices
                           uint*   cellStart,
                           uint*   cellEnd) {
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH( cellStart, gridHash);

    uint count = 0;
    float4 pos2;
    float3 relPos;
    float dist;
    float neighborDist;
    float r;
    float3 vec;
    float4 smallCircle;
    if( startIndex != 0xffffffff ) {	// cell is not empty
        // iterate over atoms in this cell
        uint endIndex = FETCH( cellEnd, gridHash);
        for( uint j = startIndex; j < endIndex; j++) {
            // do not count self
            if( j != index) {
                // get position of potential neighbor
                pos2 = FETCH( atomPos, j);
                // check distance
                relPos = make_float3( pos2.x, pos2.y, pos2.z) - make_float3( pos.x, pos.y, pos.z);
                dist = length( relPos);
                neighborDist = pos.w + pos2.w + 2.0f * params.probeRadius;
                if( dist < neighborDist ) {
                    // check number of neighbors
                    if( ( neighborIndex + count) >= params.maxNumNeighbors ) return count;
                    //neighbors[atomIndex*params.maxNumNeighbors+neighborIndex+count] = gridParticleIndex[j];
                    neighbors[atomIndex*params.maxNumNeighbors+neighborIndex+count] = j;
                    // compute small circle / intersection plane
                    r = ( (pos.w + params.probeRadius)*(pos.w + params.probeRadius))
                        + ( dist * dist) //dot( relPos, relPos) 
                        - ( (pos2.w + params.probeRadius)*(pos2.w + params.probeRadius));
                    r = r / (2.0f * dist * dist); // dot( relPos, relPos));
                    /*
                    r = (pos.w + params.probeRadius)*(pos.w + params.probeRadius)
                        - (pos2.w + params.probeRadius)*(pos2.w + params.probeRadius);
                    r = r / (2.0f * dot( relPos, relPos));
                    r = r + 0.5f;
                    */
                    vec = relPos * r;
                    smallCircle.x = vec.x;
                    smallCircle.y = vec.y;
                    smallCircle.z = vec.z;
                    smallCircle.w = 1.0f;
                    smallCircles[atomIndex*params.maxNumNeighbors+neighborIndex+count] = smallCircle;
                    // increment the neighbor counter
                    count++;
                }
            }
        }
    }
    return count;
}

///////////////////////////////////////////////////////////////////////////////
// count all neighbor atoms in a given cell
///////////////////////////////////////////////////////////////////////////////
__device__
uint countNeighborsInCell( uint*   neighbors,     // output: neighbor indices
                           uint    neighborIndex, // input: first index for writing in neighbor list
                           uint    atomIndex,     // input: atom index for writing in neighbor list
                           int3    gridPos,
                           uint    index,
                           float4  pos,
                           float4* atomPos,
                           uint*   gridParticleIndex,    // input: sorted atom indices
                           uint*   cellStart,
                           uint*   cellEnd) {
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH( cellStart, gridHash);

    uint count = 0;
    float4 pos2;
    float3 relPos;
    float dist;
    float neighborDist;
    if( startIndex != 0xffffffff ) {	// cell is not empty
        // iterate over atoms in this cell
        uint endIndex = FETCH( cellEnd, gridHash);
        for( uint j = startIndex; j < endIndex; j++) {
            // do not count self
            if( j != index) {
                // get position of potential neighbor
                pos2 = FETCH( atomPos, j);
                // check distance
                relPos = make_float3( pos2.x,  pos2.y,  pos2.z) - make_float3( pos.x, pos.y, pos.z);
                dist = length( relPos);
                neighborDist = pos.w + pos2.w + 2.0f * params.probeRadius;
                if( dist < neighborDist ) {
                    // check number of neighbors
                    if( ( neighborIndex + count) >= params.maxNumNeighbors ) return count;
                    //neighbors[atomIndex*params.maxNumNeighbors+neighborIndex+count] = gridParticleIndex[j];
                    neighbors[atomIndex*params.maxNumNeighbors+neighborIndex+count] = j;
                    // increment the neighbor counter
                    count++;
                }
            }
        }
    }
    return count;
}

///////////////////////////////////////////////////////////////////////////////
// count all neighbor atoms in a given cell
///////////////////////////////////////////////////////////////////////////////
__device__
uint countProbeNeighborsInCell( //uint*   neighbors,     // output: neighbor indices
                           float3* neighbors,     // output: neighbor positions
                           uint    neighborIndex, // input: first index for writing in neighbor list
                           uint    atomIndex,     // input: atom index for writing in neighbor list
                           int3    gridPos,
                           uint    index,
                           float4  pos,
                           float4* atomPos,
                           uint*   gridParticleIndex,    // input: sorted atom indices
                           uint*   cellStart,
                           uint*   cellEnd) {
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH( cellStart, gridHash);

    uint count = 0;
    float4 pos2;
    float3 relPos;
    float dist;
    float neighborDist;
    if( startIndex != 0xffffffff ) {	// cell is not empty
        // iterate over atoms in this cell
        uint endIndex = FETCH( cellEnd, gridHash);
        for( uint j = startIndex; j < endIndex; j++) {
            // do not count self
            if( j != index) {
                // get position of potential neighbor
                pos2 = atomPos[j];
                // check distance
                relPos = make_float3( pos2.x, pos2.y, pos2.z) - make_float3( pos.x, pos.y, pos.z);
                dist = length( relPos);
                neighborDist = 2.0f * params.probeRadius;
                if( dist < neighborDist ) {
                    // check number of neighbors
                    if( ( neighborIndex + count) >= rsParams.maxNumProbeNeighbors ) return count;
                    neighbors[atomIndex*rsParams.maxNumProbeNeighbors+neighborIndex+count] = make_float3( pos2.x, pos2.y, pos2.z);
                    // increment the neighbor counter
                    count++;
                }
            }
        }
    }
    return count;
}

///////////////////////////////////////////////////////////////////////////////
// counting function
///////////////////////////////////////////////////////////////////////////////
__global__
void countNeighbors( uint*   neighborCount,        // output: number of neighbors
                     uint*   neighbors,            // output: neighbor indices
                     float4* smallCircles,         // output: small circles
                     float4* atomPos,              // input: sorted atom positions
                     uint*   gridParticleIndex,    // input: sorted atom indices
                     uint*   cellStart,
                     uint*   cellEnd,
                     uint    numAtoms) {
    uint index = __mul24( blockIdx.x, blockDim.x) + threadIdx.x;
    if( index >= numAtoms ) return;
    
    // read original unsorted atom location
    uint originalIndex = gridParticleIndex[index];

    // read atom data from sorted arrays
    float4 pos = FETCH( atomPos, index);

    // get address in grid
    int3 gridPos = calcGridPos( make_float3( pos.x, pos.y, pos.z));

    int3 gridSize;
    gridSize.x = int( params.gridSize.x);
    gridSize.y = int( params.gridSize.y);
    gridSize.z = int( params.gridSize.z);
    // search range for neighbor atoms: max atom diameter + probe diameter
    float range = ( pos.w + 3.0f + 2.0f * params.probeRadius);
    // compute number of grid cells
    int3 cellsInRange;
    cellsInRange.x = ceil( range / params.cellSize.x);
    cellsInRange.y = ceil( range / params.cellSize.y);
    cellsInRange.z = ceil( range / params.cellSize.z);
    int3 start = gridPos - cellsInRange;
    int3 end = gridPos + cellsInRange;

    // examine neighbouring cells
    uint count = 0;
    int3 neighborPos;
    for( int z = ( start.z > 0 ? start.z : 0); z < ( end.z > gridSize.z ? gridSize.z : end.z) ; z++ ) {
        for( int y = ( start.y > 0 ? start.y : 0); y < ( end.y > gridSize.y ? gridSize.y : end.y) ; y++ ) {
            for( int x = ( start.x > 0 ? start.x : 0); x < ( end.x > gridSize.x ? gridSize.x : end.x) ; x++ ) {
                neighborPos = make_int3( x, y, z);
                count += countNeighborsInCell( neighbors, smallCircles, count, originalIndex, neighborPos, index, pos, atomPos, gridParticleIndex, cellStart, cellEnd);
            }
        }
    }

    // write new neighbor atom count back to original unsorted location
    neighborCount[originalIndex] = count;
}

///////////////////////////////////////////////////////////////////////////////
// counting function
///////////////////////////////////////////////////////////////////////////////
__global__
void countNeighbors( uint*   neighborCount,        // output: number of neighbors
                     uint*   neighbors,            // output: neighbor indices
                     float4* atomPos,              // input: sorted atom positions
                     uint*   gridParticleIndex,    // input: sorted atom indices
                     uint*   cellStart,
                     uint*   cellEnd,
                     uint    numAtoms) {
    uint index = __mul24( blockIdx.x, blockDim.x) + threadIdx.x;
    if( index >= numAtoms ) return;
    
    // read original unsorted atom location
    uint originalIndex = gridParticleIndex[index];

    // read atom data from sorted arrays
    float4 pos = FETCH( atomPos, index);

    // get address in grid
    int3 gridPos = calcGridPos( make_float3( pos));

    int3 gridSize;
    gridSize.x = int( params.gridSize.x);
    gridSize.y = int( params.gridSize.y);
    gridSize.z = int( params.gridSize.z);
    // search range for neighbor atoms: max atom diameter + probe diameter
    float range = ( pos.w + 3.0f + 2.0f * params.probeRadius);
    // compute number of grid cells
    int3 cellsInRange;
    cellsInRange.x = ceil( range / params.cellSize.x);
    cellsInRange.y = ceil( range / params.cellSize.y);
    cellsInRange.z = ceil( range / params.cellSize.z);
    int3 start = gridPos - cellsInRange;
    int3 end = gridPos + cellsInRange;

    // examine neighbouring cells
    uint count = 0;
    int3 neighborPos;
    for( int z = ( start.z > 0 ? start.z : 0); z < ( end.z > gridSize.z ? gridSize.z : end.z) ; z++ ) {
        for( int y = ( start.y > 0 ? start.y : 0); y < ( end.y > gridSize.y ? gridSize.y : end.y) ; y++ ) {
            for( int x = ( start.x > 0 ? start.x : 0); x < ( end.x > gridSize.x ? gridSize.x : end.x) ; x++ ) {
                neighborPos = make_int3( x, y, z);
                count += countNeighborsInCell( neighbors, count, originalIndex, neighborPos, index, pos, atomPos, gridParticleIndex, cellStart, cellEnd);
            }
        }
    }

    // write new neighbor atom count back to original unsorted location
    neighborCount[originalIndex] = count;
}

///////////////////////////////////////////////////////////////////////////////
// counting function
///////////////////////////////////////////////////////////////////////////////
__global__
void countProbeNeighbors( //uint*   probeNeighborCount, // output: number of neighbors
                     float3* probeNeighborCount,        // output: number of neighbors
                     //uint*   probeNeighbors,          // output: neighbor indices
                     float3* probeNeighbors,            // output: neighbor indices
                     float4* probePos,                  // input: sorted atom positions
                     uint*   gridParticleIndex,         // input: sorted atom indices
                     uint*   cellStart,
                     uint*   cellEnd,
                     uint    numProbes) {
    uint index = __mul24( blockIdx.x, blockDim.x) + threadIdx.x;
    if( index >= numProbes ) return;
    
    // read original unsorted atom location
    uint originalIndex = gridParticleIndex[index];

    // read atom data from sorted arrays
    float4 pos = probePos[index];

    // get address in grid
    int3 gridPos = calcGridPos( make_float3( pos.x, pos.y, pos.z));

    int3 gridSize;
    gridSize.x = int( params.gridSize.x);
    gridSize.y = int( params.gridSize.y);
    gridSize.z = int( params.gridSize.z);
    // search range for neighbor probes: 2x probe diameter
    float range = 2.0f * params.probeRadius;
    // compute number of grid cells
    int3 cellsInRange;
    cellsInRange.x = ceil( range / params.cellSize.x);
    cellsInRange.y = ceil( range / params.cellSize.y);
    cellsInRange.z = ceil( range / params.cellSize.z);
    int3 start = gridPos - cellsInRange;
    int3 end = gridPos + cellsInRange;

    // examine neighbouring cells
    uint count = 0;
    int3 neighborPos;
    for( int z = ( start.z > 0 ? start.z : 0); z < ( end.z > gridSize.z ? gridSize.z : end.z) ; z++ ) {
        for( int y = ( start.y > 0 ? start.y : 0); y < ( end.y > gridSize.y ? gridSize.y : end.y) ; y++ ) {
            for( int x = ( start.x > 0 ? start.x : 0); x < ( end.x > gridSize.x ? gridSize.x : end.x) ; x++ ) {
                neighborPos = make_int3( x, y, z);
                count += countProbeNeighborsInCell( probeNeighbors, count, originalIndex, neighborPos, index, pos, probePos, gridParticleIndex, cellStart, cellEnd);
            }
        }
    }

    // write new neighbor atom count back to original unsorted location
    probeNeighborCount[originalIndex] = make_float3( float( count), 0.0f, float( originalIndex));
}

///////////////////////////////////////////////////////////////////////////////
// compute the arcs
///////////////////////////////////////////////////////////////////////////////
__global__
void computeArcs( float4* arcs,                 // output: arcs
                  uint*   neighborCount,        // input: number of neighbors
                  uint*   neighbors,            // input: neighbor indices
                  float4* smallCircles,         // input: small circles
                  float4* atomPos,              // input: sorted atom positions
                  uint*   gridParticleIndex,    // input: sorted atom indices
                  uint    numAtoms) {
    // get atom index
    uint atomIdx = blockIdx.x;
    // get neighbor atom index
    uint neighborIdx = threadIdx.x;
    // check, if atom index is within bounds
    if( atomIdx >= numAtoms ) return;
    // check, if neighbor index is within bounds
    if( neighborIdx >= params.maxNumNeighbors ) return;
    // read original unsorted atom location
    uint origAtomIdx = gridParticleIndex[atomIdx];
    // check, if neighbor index is within bounds
    uint numNeighbors = neighborCount[origAtomIdx];
    if( neighborIdx >= numNeighbors ) return;

    // read atom position from sorted arrays
    float4 atom = FETCH( atomPos, atomIdx);
    // read neighbor position from sorted arrays
    //float4 ak = FETCH( atomPos, FETCH( neighbors, origAtomIdx * params.maxNumNeighbors + neighborIdx));
    float3 ai = make_float3( atom);
    float3 aj;

    float3 rm;
    float4 rj4;
    float3 rj;
    float4 rk4 = FETCH( smallCircles, origAtomIdx * params.maxNumNeighbors + neighborIdx);
    float3 rk = make_float3( rk4.x, rk4.y, rk4.z);
    float Ri = atom.w + params.probeRadius;
    float Ri2 = Ri * Ri;
    float numer1, numer2, denom, rj_dot_rk, rj2, rk2;
    float3 p1, p2, tmpFloat3, cross_rj_rk;
    uint numArcs = 0;

    rk2 = dot( rk, rk);

    float3 e1, e2;
    
    for( uint cnt = 0; cnt < numNeighbors; ++cnt ) {
        if( cnt == neighborIdx ) continue;
        
        aj = make_float3( FETCH( atomPos, FETCH( neighbors, origAtomIdx * params.maxNumNeighbors + cnt)));
        // compute the auxiliary vector rm (plane intersection)
        rj4 = FETCH( smallCircles, origAtomIdx * params.maxNumNeighbors + cnt);
        rj = make_float3( rj4);
        rj2 = dot( rj, rj);
        rj_dot_rk = dot( rj, rk);
        numer1 = ( rj2 - rj_dot_rk) * rk2;
        numer2 = ( rk2 - rj_dot_rk) * rj2;
        denom = rj2 * rk2 - rj_dot_rk * rj_dot_rk;
        rm = rj * ( numer1 / denom) + rk * ( numer2 / denom);

        // continue to next small circle, if this one does not intersect
        if( dot( rm, rm) > Ri2 ) continue;

        // compute the start- and endpoint of the newly found arc
        cross_rj_rk = cross( rj, rk);
        tmpFloat3 = cross_rj_rk * sqrt( ( Ri2 - dot( rm, rm)) / dot( cross_rj_rk, cross_rj_rk));
        
        if( dot( rj, aj - ai) < 0.0f ) {
            // x1
            p1 = rm + tmpFloat3;
            // x2
            p2 = rm - tmpFloat3;
        } else {
            // x2
            p2 = rm + tmpFloat3;
            // x1
            p1 = rm - tmpFloat3;
        }

        if( dot( cross( p1 - rk, p2 - rk), rk) < 0.0f ) {
            tmpFloat3 = p1;
            p1 = p2;
            p2 = tmpFloat3;
        }
        
        // if the current arc ist the first:
        if( numArcs == 0 ) {
            // write the first arc
            //arcs[origAtomIdx * params.maxNumNeighbors * 4 + 0] = make_float4( p1, 1.0f);
            //arcs[origAtomIdx * params.maxNumNeighbors * 4 + 1] = make_float4( p2, 1.0f);
            numArcs++;

            e1 = p1;
            e2 = p2;
            continue;
        }

        // compute angles
        /*
        tmpAngle1 = acos( dot( normalize( p1 - rk), e1));
        tmpAngle = dot( normalize( p1 - rk), e2);
        if( tmpAngle < 0.0f )
            tmpAngle1 = 6.2831853f - tmpAngle1;
        tmpAngle2 = acos( dot( normalize( p2 - rk), e1));
        tmpAngle = dot( normalize( p2 - rk), e2);
        if( tmpAngle < 0.0f )
            tmpAngle2 = 6.2831853f - tmpAngle2;

        // check cases
        if( tmpAngle1 > angle1 ) {
            angle1 = tmpAngle1;
            arcs[origAtomIdx * params.maxNumNeighbors * 4 + 0] = make_float4( p1, 1.0f);
        }
        if( tmpAngle2 < angle2 ) {
            angle2 = tmpAngle2;
            arcs[origAtomIdx * params.maxNumNeighbors * 4 + 1] = make_float4( p2, 1.0f);
        }
        if( tmpAngle1 < angle2 && tmpAngle2 > angle1 ) {
            angle2 = tmpAngle2;
            arcs[origAtomIdx * params.maxNumNeighbors * 4 + 1] = make_float4( p2, 1.0f);
            // TODO: second arc segment
        }
        */
        float d1 = dot( aj - ai, e1 - rj);
        float d2 = dot( aj - ai, e2 - rj);
        float d3 = dot( rk, aj - ai) * dot( cross( e1, p1), e2);

        if( d1 > 0.0f ) {
            if( d2 > 0.0f ) {
                if( d3 > 0.0f ) {
                    e1 = make_float3( 0.0f, 0.0f, 0.0f);
                    e2 = make_float3( 0.0f, 0.0f, 0.0f);
                } else { // d3 < 0
                    e1 = p1;
                    e2 = p2;
                }
            } else { // d2 < 0
                if( d3 > 0.0f ) {
                    e1 = p1;
                } else { // d3 < 0
                    e1 = p2;
                }
            }
        } else { // d1 < 0
            if( d2 > 0.0f ) {
                if( d3 > 0.0f ) {
                    e2 = p1;
                } else { // d3 < 0
                    e2 = p2;
                }
            } else { // d2 < 0
                if( d3 > 0.0f ) {
                    // Teilung in zwei B�gen
                } else { // d3 < 0
                    // keine Auswirkung
                }
            }
        }
    }
    arcs[origAtomIdx * params.maxNumNeighbors * 4 + 0] = make_float4( e1, 1.0f);
    arcs[origAtomIdx * params.maxNumNeighbors * 4 + 1] = make_float4( e2, 1.0f);

    /*
    for( uint cnt = 0; cnt < numNeighbors; ++cnt ) {
        if( cnt == neighborIdx ) continue;
        
        //aj = FETCH( atomPos, FETCH( neighbors, origAtomIdx * params.maxNumNeighbors + cnt));
        // do nothing if the neighboring atoms do not intersect
        //if( length( make_float3( aj) - make_float3( ak)) > ( ak.w + aj.w + 2.0f * params.probeRadius) ) continue;

        // compute the auxiliary vector rm (plane intersection)
        rj4 = FETCH( smallCircles, origAtomIdx * params.maxNumNeighbors + cnt);
        rj = make_float3( rj4);
        rj2 = dot( rj, rj);
        rj_dot_rk = dot( rj, rk);
        numer1 = ( rj2 - rj_dot_rk) * rk2;
        numer2 = ( rk2 - rj_dot_rk) * rj2;
        denom = rj2 * rk2 - rj_dot_rk * rj_dot_rk;
        rm = rj * ( numer1 / denom) + rk * ( numer2 / denom);

        // continue to next small circle, if this one does not intersect
        if( dot( rm, rm) > Ri2 ) continue;

        // compute the start- and endpoint of the newly found arc
        cross_rj_rk = cross( rj, rk);
        tmpFloat3 = cross_rj_rk * sqrt( ( Ri2 - dot( rm, rm)) / dot( cross_rj_rk, cross_rj_rk));
        p1 = rm + tmpFloat3;
        p2 = rm - tmpFloat3;

        // COMPUTATION WITH GLOBAL MEMORY ...
        // if the current arc ist the first:
        if( numArcs == 0 ) {
            // write the first arc
            arcs[origAtomIdx * params.maxNumNeighbors * 4 + 0] = make_float4( p1, 1.0f);
            numArcs++;
            arcs[origAtomIdx * params.maxNumNeighbors * 4 + 1] = make_float4( p2, 1.0f);
            numArcs++;
            sk = 1.0f;
            continue;
        }

        // compute c1, c2, d to determine the case
        c1 = dot( rj, make_float3( arcs[origAtomIdx * params.maxNumNeighbors * 4 + 0]) - rj);
        c2 = dot( rj, make_float3( arcs[origAtomIdx * params.maxNumNeighbors * 4 + 1]) - rj);
        d = sk * dot( cross( make_float3( arcs[origAtomIdx * params.maxNumNeighbors * 4 + 0]), p1), make_float3( arcs[origAtomIdx * params.maxNumNeighbors * 4 + 1]));

        if( c1 > 0.0f && c2 < 0.0f ) {
            if( d > 0.0f ) {
                arcs[origAtomIdx * params.maxNumNeighbors * 4 + 0] = make_float4( p1, 1.0f);
            } else {
                arcs[origAtomIdx * params.maxNumNeighbors * 4 + 0] = make_float4( p2, 1.0f);
            }
        } else if( c1 < 0.0f && c2 > 0.0f ) {
            if( d > 0.0f ) {
                arcs[origAtomIdx * params.maxNumNeighbors * 4 + 1] = make_float4( p1, 1.0f);
            } else {
                arcs[origAtomIdx * params.maxNumNeighbors * 4 + 1] = make_float4( p2, 1.0f);
            }
        } else if( c1 > 0.0f && c2 > 0.0f ) {
            if( d > 0.0f ) {
                return;
            } else {
                arcs[origAtomIdx * params.maxNumNeighbors * 4 + 0] = make_float4( p1, 1.0f);
                arcs[origAtomIdx * params.maxNumNeighbors * 4 + 1] = make_float4( p2, 1.0f);
            }
        }
        else { // c1 < 0.0f && c2 < 0.0f
            if( d > 0.0f ) {
                arcs[origAtomIdx * params.maxNumNeighbors * 4 + 2] = arcs[origAtomIdx * params.maxNumNeighbors * 4 + 1];
                arcs[origAtomIdx * params.maxNumNeighbors * 4 + 1] = make_float4( p1, 1.0f);
                arcs[origAtomIdx * params.maxNumNeighbors * 4 + 3] = make_float4( p2, 1.0f);
            }
        }
        // ... COMPUTATION WITH GLOBAL MEMORY
    }
    */

}


///////////////////////////////////////////////////////////////////////////////
// compute reduced surfaces
///////////////////////////////////////////////////////////////////////////////
__global__
void computeReducedSurface( uint4* point1,      // output: the atom indices
                  float4* probePos,             // output: the probe position and orientation
                  uint*   neighborCount,        // input: number of neighbors
                  uint*   neighbors,            // input: neighbor indices
                  float4* atomPos,              // input: sorted atom positions
                  uint*   gridParticleIndex,    // input: sorted atom indices
                  float4* visibleAtoms,         // input: visible atoms position and radius
                  uint*   visibleAtomsId ) {    // input: visible atoms original index list

    // get atom index
    uint visibleAtomIdx = blockIdx.y * blockDim.y + threadIdx.y;
    // check bounds of visible atoms
    if( visibleAtomIdx >= rsParams.visibleAtomCount ) {
        return;
    }
    // get combined neighbor index
    uint idxX = blockIdx.x * blockDim.x + threadIdx.x;
    // check bounds of neighbor index
    if( idxX >= ( params.maxNumNeighbors * params.maxNumNeighbors) ) {
        return;
    }
    // get neighbor atom indices
    uint id1 = idxX % params.maxNumNeighbors;
    uint id2 = ( idxX - id1) / params.maxNumNeighbors;
    // check if the id's are the same
    if( id1 == id2 ) {
        return;
    }
    
    // ---------- FRAGMENT SHADER CODE ... ---------
    
    // get the Id of the current atom
    float atomId = FETCH( visibleAtomsId, visibleAtomIdx);

    // read number of vicinity atoms from vicinity texture
    uint vicinityCnt = FETCH( neighborCount, atomId);

    // temp variables
    int cutId;

    float4 akTmp = FETCH( visibleAtoms, visibleAtomIdx);
    uint aiIdx = FETCH( neighbors, params.maxNumNeighbors * atomId + id1);
    float4 aiTmp = FETCH( atomPos, aiIdx);
    uint ajIdx = FETCH( neighbors, params.maxNumNeighbors * atomId + id2);
    float4 ajTmp = FETCH( atomPos, ajIdx);
    float3 ak = { akTmp.x, akTmp.y, akTmp.z}; 
    float rk = akTmp.w;
    float3 ai = { aiTmp.x, aiTmp.y, aiTmp.z};
    float ri = aiTmp.w;
    float3 aj = { ajTmp.x, ajTmp.y, ajTmp.z};
    float rj = ajTmp.w;
    
    // names of the variables according to: Connolly "Analytical Molecular Surface Calculation", 1983
    float3 uij, uik, tij, tik, uijk, utb, bijk, pijk0, pijk1;
    float dij, dik, djk, hijk, wijk, tmpFloat;
    
    dij = length( aj - ai);
    dik = length( ak - ai);
    djk = length( ak - aj);
    
    uij = ( aj - ai)/dij;
    uik = ( ak - ai)/dik;

    if( ( ( ri - rj)*( ri - rj) > dij*dij ) || 
        ( ( ri - rk)*( ri - rk) > dik*dik ) || 
        ( ( rj - rk)*( rj - rk) > djk*djk ) ) {
        return;
    }
    tij = 0.5f*( ai + aj) + 0.5f*( aj - ai) * ( ( ri + params.probeRadius)*( ri + params.probeRadius) - ( rj + params.probeRadius)*( rj + params.probeRadius))/( dij*dij);
    tik = 0.5f*( ai + ak) + 0.5f*( ak - ai) * ( ( ri + params.probeRadius)*( ri + params.probeRadius) - ( rk + params.probeRadius)*( rk + params.probeRadius))/( dik*dik);
    wijk = acos( dot( uij, uik) );
    uijk = cross( uij, uik) / sin( wijk);
    utb = cross( uijk, uij);
    bijk = tij + utb * ( dot( uik, tik - tij) / sin( wijk));
    tmpFloat = ( ri + params.probeRadius)*( ri + params.probeRadius) - length( bijk - ai)*length( bijk - ai);
    if( tmpFloat < 0.0f ) {
        return;
    }
    hijk = sqrt( tmpFloat);
    pijk0 = bijk + uijk * hijk;
    pijk1 = bijk - uijk * hijk;

    bool draw0, draw1;
    draw0 = true;
    draw1 = true;
    
    int stop1 = min( id1, id2);
    int stop2 = max( id1, id2);
    float4 voxel;
    float3 voxelPos;
    for( cutId = 0; cutId < stop1; ++cutId ) {
        voxel = FETCH( atomPos, FETCH( neighbors, params.maxNumNeighbors * atomId + cutId));
        voxelPos = make_float3( voxel.x, voxel.y, voxel.z);
        if( length( pijk0 - voxelPos ) < ( params.probeRadius + voxel.w - 0.001f ) )
            draw0 = false;
        if( length( pijk1 - voxelPos ) < ( params.probeRadius + voxel.w - 0.001f ) )
            draw1 = false;
    }
    for( cutId = stop1+1; cutId < stop2; ++cutId ) {
        voxel = FETCH( atomPos, FETCH( neighbors, params.maxNumNeighbors * atomId + cutId));
        voxelPos = make_float3( voxel.x, voxel.y, voxel.z);
        if( length( pijk0 - voxelPos ) < ( params.probeRadius + voxel.w - 0.001f ) )
            draw0 = false;
        if( length( pijk1 - voxelPos ) < ( params.probeRadius + voxel.w - 0.001f ) )
            draw1 = false;
    }
    for( cutId = stop2+1; cutId < vicinityCnt; ++cutId ) {
        voxel = FETCH( atomPos, FETCH( neighbors, params.maxNumNeighbors * atomId + cutId));
        voxelPos = make_float3( voxel.x, voxel.y, voxel.z);
        if( length( pijk0 - voxelPos ) < ( params.probeRadius + voxel.w - 0.001f ) )
            draw0 = false;
        if( length( pijk1 - voxelPos ) < ( params.probeRadius + voxel.w - 0.001f ) )
            draw1 = false;
    }
    
    if( draw0 && draw1 ) {
        point1[visibleAtomIdx * params.maxNumNeighbors * params.maxNumNeighbors + idxX] = make_uint4( aiIdx, ajIdx, visibleAtomIdx, 1);
        probePos[visibleAtomIdx * params.maxNumNeighbors * params.maxNumNeighbors + idxX] = make_float4( pijk0.x, pijk0.y, pijk0.z,-1.0f);
    } else if( draw0 && !draw1 ) {
        point1[visibleAtomIdx * params.maxNumNeighbors * params.maxNumNeighbors + idxX] = make_uint4( aiIdx, ajIdx, visibleAtomIdx, 1);
        probePos[visibleAtomIdx * params.maxNumNeighbors * params.maxNumNeighbors + idxX] = make_float4( pijk0.x, pijk0.y, pijk0.z, 1.0f);
    } else if( !draw0 && draw1 ) {
        point1[visibleAtomIdx * params.maxNumNeighbors * params.maxNumNeighbors + idxX] = make_uint4( aiIdx, ajIdx, visibleAtomIdx, 1);
        probePos[visibleAtomIdx * params.maxNumNeighbors * params.maxNumNeighbors + idxX] = make_float4( pijk1.x, pijk1.y, pijk1.z, 1.0f);
    } else {
        return;
    }
    // ---------- ... FRAGMENT SHADER CODE ---------

}

///////////////////////////////////////////////////////////////////////////////
// compute the triangle vbo
///////////////////////////////////////////////////////////////////////////////
__global__
void computeTriangleVBO( float3* vbo,           // output: triangle vertices and colors
                  //float4* point1,               // input: point 1 of the RS face
                  //float4* point2,               // input: point 2 of the RS face
                  //float4* point3 ) {            // input: point 3 of the RS face
                  uint4* point1,                // input: point 1 of the RS face
                  float4* atomPos,              // input: sorted atom positions
                  float4* visibleAtoms,         // input: visible atoms position and radius
                  uint offset ) {

    // get atom index
    uint visibleAtomIdx = offset + blockIdx.y * blockDim.y + threadIdx.y;
    // check bounds of visible atoms
    if( visibleAtomIdx >= rsParams.visibleAtomCount ) {
        return;
    }
    // get combined neighbor index
    uint idxX = blockIdx.x * blockDim.x + threadIdx.x;
    // check bounds of neighbor index
    if( idxX >= ( params.maxNumNeighbors * params.maxNumNeighbors) ) {
        return;
    }

    // the color value for visibility checking
    //float3 color = { float( visibleAtomIdx), float( idxX), 0.0f};
    float3 color = { float( idxX), float( visibleAtomIdx), 0.0f};
    float4 pos; 
    // compute the index of the array
    uint pointIdx = visibleAtomIdx * params.maxNumNeighbors * params.maxNumNeighbors + idxX;
    // offset: 3 vertices + 3 colors
    //uint vboIdx = pointIdx * 6;
    uint vboIdx = ( ( visibleAtomIdx - offset) * params.maxNumNeighbors * params.maxNumNeighbors + idxX) * 6;
    // write the positions
    //uint4 point = FETCH( point1, pointIdx);
    uint4 point = point1[pointIdx];
    //pos = FETCH( atomPos, point1[pointIdx].x);
    pos = FETCH( atomPos, point.x);
    pos.w = 1.0f;
    vbo[vboIdx+0] = make_float3( pos);

    color.z = 0.0f;
    vbo[vboIdx+1] = color;

    //pos = FETCH( atomPos, point1[pointIdx].y);
    pos = FETCH( atomPos, point.y);
    pos.w = 1.0f;
    vbo[vboIdx+2] = make_float3( pos);

    color.z = 1.0f;
    vbo[vboIdx+3] = color;

    //pos = FETCH( visibleAtoms, point1[pointIdx].z);
    pos = FETCH( visibleAtoms, point.z);
    pos.w = 1.0f;
    vbo[vboIdx+4] = make_float3( pos);

    color.z = 2.0f;
    vbo[vboIdx+5] = color;
}

///////////////////////////////////////////////////////////////////////////////
// compute the triangle vbo
///////////////////////////////////////////////////////////////////////////////
__global__
void computeVisibleTriangleVBO( float3* vbo,           // output: triangle vertices and colors
                                uint4* point1,         // input: point 1 of the RS face
                                float4* atomPos,       // input: sorted atom positions
                                float4* visibleAtoms,  // input: visible atoms position and radius
                                uint offset ) {

    // get atom index
    uint visibleAtomIdx = offset + blockIdx.y * blockDim.y + threadIdx.y;
    // check bounds of visible atoms
    if( visibleAtomIdx >= rsParams.visibleAtomCount ) {
        return;
    }
    // get combined neighbor index
    uint idxX = blockIdx.x * blockDim.x + threadIdx.x;
    // check bounds of neighbor index
    if( idxX >= ( params.maxNumNeighbors * params.maxNumNeighbors) ) {
        return;
    }

    // the color value for visibility checking
    //float3 color = { float( visibleAtomIdx), float( idxX), 0.0f};
    float3 color = { float( idxX), float( visibleAtomIdx), 0.0f};
    float4 pos; 
    // compute the index of the array
    uint pointIdx = visibleAtomIdx * params.maxNumNeighbors * params.maxNumNeighbors + idxX;
    // offset: 3 vertices + 3 colors
    //uint vboIdx = pointIdx * 6;
    uint vboIdx = ( ( visibleAtomIdx - offset) * params.maxNumNeighbors * params.maxNumNeighbors + idxX) * 6;
    // get the visiblity
    float visible = tex2D( inVisibilityTex, idxX, visibleAtomIdx).x;
    // write the positions
    //uint4 point = FETCH( point1, pointIdx);
    uint4 point = point1[pointIdx];
    //pos = FETCH( atomPos, point1[pointIdx].x);
    pos = FETCH( atomPos, point.x);
    pos.w = 1.0f;
    vbo[vboIdx+0] = make_float3( pos) * visible;

    color.z = 0.0f;
    vbo[vboIdx+1] = color;

    //pos = FETCH( atomPos, point1[pointIdx].y);
    pos = FETCH( atomPos, point.y);
    pos.w = 1.0f;
    vbo[vboIdx+2] = make_float3( pos) * visible;

    color.z = 1.0f;
    vbo[vboIdx+3] = color;

    //pos = FETCH( visibleAtoms, point1[pointIdx].z);
    pos = FETCH( visibleAtoms, point.z);
    pos.w = 1.0f;
    vbo[vboIdx+4] = make_float3( pos) * visible;

    color.z = 2.0f;
    vbo[vboIdx+5] = color;
}


///////////////////////////////////////////////////////////////////////////////
// compute the torus vbo
///////////////////////////////////////////////////////////////////////////////
__global__
void computeTorusVBO(
        float4* outTorusVBO,    // the output VBO (positions + attributes for torus drawing)
        float4* outSTriaVBO,    // the output VBO (positions + attributes for spherical triangle drawing)
        float4* inVBO,          // the input VBO (indices of visible triangles)
        float4* atomPos,        // the sorted atom positions (for neighboring atoms)
        float4* visibleAtoms,   // the visible atoms' positions
        uint4* point1,          // the atom index array
        float4* probePos ) {    // the probe position array
    // get the index
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    // get the coordinates of the atom index array
    float4 indicesFloat = inVBO[idx];
    uint xIdx = uint( floor( indicesFloat.x + 0.5f));
    uint yIdx = uint( floor( indicesFloat.y + 0.5f));
    uint arrayIdx = params.maxNumNeighbors * params.maxNumNeighbors * yIdx + xIdx;
    uint4 pointIdx = point1[arrayIdx];
    // get the points
    float4 aiTmp = FETCH( atomPos, pointIdx.x);
    float3 ai = make_float3( aiTmp.x, aiTmp.y, aiTmp.z);
    float ri = aiTmp.w;
    float4 ajTmp = FETCH( atomPos, pointIdx.y);
    float3 aj = make_float3( ajTmp.x, ajTmp.y, ajTmp.z);
    float rj = ajTmp.w;
    float4 akTmp = FETCH( visibleAtoms, pointIdx.z);
    float3 ak = make_float3( akTmp.x, akTmp.y, akTmp.z);
    float rk = akTmp.w;
    // get the probe position
    float4 probe = probePos[arrayIdx];

    // names of the variables according to: Connolly "Analytical Molecular Surface Calculation", 1983
    float3 tij, tik, tjk, pijk0, pijk1;
    float dij, dik, djk, rij, rik, rjk;

    dij = length( aj - ai);
    dik = length( ak - ai);
    djk = length( ak - aj);
    
    tij = 0.5f*( ai + aj) + 0.5f*( aj - ai) * ( ( ri + params.probeRadius)*( ri + params.probeRadius) - ( rj + params.probeRadius)*( rj + params.probeRadius))/( dij*dij);
    tik = 0.5f*( ai + ak) + 0.5f*( ak - ai) * ( ( ri + params.probeRadius)*( ri + params.probeRadius) - ( rk + params.probeRadius)*( rk + params.probeRadius))/( dik*dik);
    tjk = 0.5f*( aj + ak) + 0.5f*( ak - aj) * ( ( rj + params.probeRadius)*( rj + params.probeRadius) - ( rk + params.probeRadius)*( rk + params.probeRadius))/( djk*djk);
    rij = 0.5f*sqrt( float( (ri + rj + 2.0f*params.probeRadius)*(ri + rj + 2.0f*params.probeRadius) - dij*dij)) * ( sqrt( float( dij*dij - ( ri - rj)*( ri - rj))) / dij);
    rik = 0.5f*sqrt( float( (ri + rk + 2.0f*params.probeRadius)*(ri + rk + 2.0f*params.probeRadius) - dik*dik)) * ( sqrt( float( dik*dik - ( ri - rk)*( ri - rk))) / dik);
    rjk = 0.5f*sqrt( float( (rj + rk + 2.0f*params.probeRadius)*(rj + rk + 2.0f*params.probeRadius) - djk*djk)) * ( sqrt( float( djk*djk - ( rj - rk)*( rj - rk))) / djk);
    pijk0 = make_float3( probe.x, probe.y, probe.z);
    pijk1 = pijk0*probe.w;

    //////////////////////////////////////////////
    // emit varyings and position for torus i-j //
    //////////////////////////////////////////////
    // get the rotation axis of the torus
    float3 torusAxis = normalize( ai - tij);
    // get the axis for rotating the torus rotations axis on the z-axis
    float3 rotAxis = normalize( cross( torusAxis, make_float3( 0.0f, 0.0f, 1.0f)));
    // compute quaternion
    float4 quatC;
    float angle = acos( dot( torusAxis, make_float3( 0.0f, 0.0f, 1.0f)));
    float len = length( rotAxis);
    float halfAngle = 0.5f * angle;
    if( len > 0.0f ) {
        len = sin( halfAngle);
        quatC.x = rotAxis.x * len;
        quatC.y = rotAxis.y * len;
        quatC.z = rotAxis.z * len;
        quatC.w = cos( halfAngle);
    } else {
        quatC = make_float4( 0.0f, 0.0f, 0.0f, 1.0f);
    }
    // compute the tangential point X2 of the spheres
    float3 P = tij + rotAxis * rij;
    float3 X1 = normalize( P - ai) * ri;
    float3 X2 = normalize( P - aj) * rj;
    float3 C = ai - aj;
    C = ( length( P - aj) / ( length( P - ai) + length( P - aj) ) ) * C;
    float distance = length( X2 - C);
    C = ( C + aj) - tij;
    // write the parameters
    outTorusVBO[idx*3*4] = make_float4( tij.x, tij.y, tij.z, 1.0f);
    outTorusVBO[idx*3*4+1] = make_float4( params.probeRadius, rij, 1.0f, 1.0f);
    outTorusVBO[idx*3*4+2] = quatC;
    outTorusVBO[idx*3*4+3] = make_float4( C.x, C.y, C.z, distance);

    //////////////////////////////////////////////
    // emit varyings and position for torus i-k //
    //////////////////////////////////////////////
    // get the rotation axis of the torus
    torusAxis = normalize( ai - tik);
    // get the axis for rotating the torus rotations axis on the z-axis
    rotAxis = normalize( cross( torusAxis, make_float3( 0.0f, 0.0f, 1.0f)));
    // compute quaternion
    angle = acos( dot( torusAxis, make_float3( 0.0f, 0.0f, 1.0f)));
    len = length( rotAxis);
    halfAngle = 0.5f * angle;
    if( len > 0.0f ) {
        len = sin( halfAngle);
        quatC.x = rotAxis.x * len;
        quatC.y = rotAxis.y * len;
        quatC.z = rotAxis.z * len;
        quatC.w = cos( halfAngle);
    } else {
        quatC = make_float4( 0.0f, 0.0f, 0.0f, 1.0f);
    }
    // compute the tangential point X2 of the spheres
    P = tik + rotAxis * rik;
    X1 = normalize( P - ai) * ri;
    X2 = normalize( P - ak) * rk;
    C = ai - ak;
    C = ( length( P - ak) / ( length( P - ai) + length( P - ak) ) ) * C;
    distance = length( X2 - C);
    C = ( C + ak) - tik;
    // write the parameters
    outTorusVBO[idx*3*4+4] = make_float4( tik.x, tik.y, tik.z, 1.0f);
    outTorusVBO[idx*3*4+5] = make_float4( params.probeRadius, rik, 1.0f, 1.0f);
    outTorusVBO[idx*3*4+6] = quatC;
    outTorusVBO[idx*3*4+7] = make_float4( C.x, C.y, C.z, distance);

    //////////////////////////////////////////////
    // emit varyings and position for torus i-k //
    //////////////////////////////////////////////
    // get the rotation axis of the torus
    torusAxis = normalize( aj - tjk);
    // get the axis for rotating the torus rotations axis on the z-axis
    rotAxis = normalize( cross( torusAxis, make_float3( 0.0f, 0.0f, 1.0f)));
    // compute quaternion
    angle = acos( dot( torusAxis, make_float3( 0.0f, 0.0f, 1.0f)));
    len = length( rotAxis);
    halfAngle = 0.5f * angle;
    if( len > 0.0f ) {
        len = sin(halfAngle);
        quatC.x = rotAxis.x * len;
        quatC.y = rotAxis.y * len;
        quatC.z = rotAxis.z * len;
        quatC.w = cos( halfAngle);
    } else {
        quatC = make_float4( 0.0f, 0.0f, 0.0f, 1.0f);
    }
    // compute the tangential point X2 of the spheres
    P = tjk + rotAxis * rjk;
    X1 = normalize( P - aj) * rj;
    X2 = normalize( P - ak) * rk;
    C = aj - ak;
    C = ( length( P - ak) / ( length( P - aj) + length( P - ak) ) ) * C;
    distance = length( X2 - C);
    C = ( C + ak) - tjk;
    // write the parameters
    outTorusVBO[idx*3*4+8] = make_float4( tjk.x, tjk.y, tjk.z, 1.0f);
    outTorusVBO[idx*3*4+9] = make_float4( params.probeRadius, rjk, 1.0f, 1.0f);
    outTorusVBO[idx*3*4+10] = quatC;
    outTorusVBO[idx*3*4+11] = make_float4( C.x, C.y, C.z, distance);

    /////////////////////////////////////////////////////////////
    // emit varyings and position for first spherical triangle //
    /////////////////////////////////////////////////////////////
    outSTriaVBO[idx*2*4] = make_float4( pijk0, params.probeRadius);
    outSTriaVBO[idx*2*4+1] = make_float4( ai - pijk0, 1.0f);
    outSTriaVBO[idx*2*4+2] = make_float4( aj - pijk0, 1.0f);
    outSTriaVBO[idx*2*4+3] = make_float4( ak - pijk0, params.probeRadius*params.probeRadius);
    
    //////////////////////////////////////////////////////////////
    // emit varyings and position for second spherical triangle //
    //////////////////////////////////////////////////////////////
    outSTriaVBO[idx*2*4+4] = make_float4( pijk1, params.probeRadius);
    outSTriaVBO[idx*2*4+5] = make_float4( ai - pijk1, 1.0f);
    outSTriaVBO[idx*2*4+6] = make_float4( aj - pijk1, 1.0f);
    outSTriaVBO[idx*2*4+7] = make_float4( ak - pijk1, params.probeRadius*params.probeRadius);
    
    ///////////////////////////////////////////////////////////////////////
    // ==> The two spherical triangles are potentially the same!
    //     This does not matter, since rendering if fast enough and
    //     singularity handling will test probe distances.
    ///////////////////////////////////////////////////////////////////////
    
}


///////////////////////////////////////////////////////////////////////////////
// write the probe positions to a new array
///////////////////////////////////////////////////////////////////////////////
__global__
void writeProbePositions(
        float4* probePos,   // output (probe positions)
        float4* sTriaVbo,   // input (probe positions)
        uint numProbes ) {  // the number of probes
    // get the index
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    // check bounds
    if( idx >= numProbes ) return;
    // copy data
    probePos[idx] = sTriaVbo[idx*4];
    probePos[idx].w = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////
// write the singularities to the PBO
///////////////////////////////////////////////////////////////////////////////
__global__
void writeSingularities(
        float3* outArray,
        uint*  probeNeighbors,
        float4* probePos ) {
    // get the indices
    uint probeIdx = blockIdx.y * blockDim.y + threadIdx.y;
    // check bounds of visible atoms
    if( probeIdx >= rsParams.probeCount ) {
        return;
    }
    // get combined neighbor index
    uint neighborIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // check bounds of neighbor index
    if( neighborIdx >= rsParams.maxNumProbeNeighbors ) {
        return;
    }
    // read the neighbor probe position
    uint nIdx = probeNeighbors[probeIdx*rsParams.maxNumProbeNeighbors+neighborIdx];
    float4 pos = probePos[nIdx];
    // write the neighbor probe position
    outArray[probeIdx*rsParams.maxNumProbeNeighbors+neighborIdx] = make_float3( pos.x, pos.y, pos.z);
}

///////////////////////////////////////////////////////////////////////////////
// find all adjacent, occluded RS-faces
///////////////////////////////////////////////////////////////////////////////
__global__
void findAdjacentTriangles( 
        float* outPbo,
        uint4* point1, 
        float4* probePos,
        uint* neighborCount,
        uint* neighbors,
        float4* atomPos, 
        float4* visibleAtoms, 
        uint* visibleAtomsId ) {
    // get atom index
    uint visibleAtomIdx = blockIdx.y * blockDim.y + threadIdx.y;
    // check bounds of visible atoms
    if( visibleAtomIdx >= rsParams.visibleAtomCount ) {
        return;
    }
    // get neighbor index
    uint nIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // check bounds of neighbor index
    if( nIdx >= params.maxNumNeighbors ) {
        return;
    }
    
    uint cnt;
    bool visible = false;
    bool secondVisible = false;
    float4 visibleTria[3];
    float4 visibleTriaNormal;
    float4 invisibleTriaNormal[64];
    uint invisibleTriaId[64];
    float4 tmp;
    float3 tmpDir, tmpDirTS;
    int smallest = -1;

    uint arrayIdx;
    uint4 pointIdx;

    uint counter = 0;
    
    int xcoord = int( nIdx)*params.maxNumNeighbors;

    float visibility;

    // check number of visible triangles for this edge
    for( cnt = 0; cnt < params.maxNumNeighbors; ++cnt ) {
        // compute the array index
        arrayIdx = params.maxNumNeighbors * params.maxNumNeighbors * visibleAtomIdx + xcoord + cnt;
        // get the indices of the atoms
        pointIdx = point1[arrayIdx];
        // get visibility information
        visibility = tex2D( inVisibilityTex, xcoord+cnt, visibleAtomIdx).x;
        // copy visibility information to PBO
        outPbo[arrayIdx] = visibility;
        // check visibility
        if( tex2D( inVisibilityTex, xcoord+cnt, visibleAtomIdx).x > 0.5f ) {
            // if a second visible triangle was found: do nothing!
            if( visible ) secondVisible = true;
            // get the points
            visibleTria[0] = FETCH( atomPos, pointIdx.x);
            visibleTria[1] = FETCH( atomPos, pointIdx.y);
            visibleTria[2] = FETCH( visibleAtoms, pointIdx.z);
            // get the probe position
            visibleTriaNormal = probePos[arrayIdx];
            visible = true;
        } else {
            tmp = FETCH( atomPos, pointIdx.y);
            if( tmp.w > 0.5f ) {
                invisibleTriaNormal[counter] = probePos[arrayIdx];
                invisibleTriaId[counter] = cnt;
                counter++;
            }
        }
    }

    // if no or two visible triangles were found: do nothing!
    if( !visible || secondVisible ) return;
    
    float angle, tmpAngle;
    // The transformation matrix is:
    //     (Tx Ty Tz 0)
    // M = (Bx By Bz 0)
    //     (Nx Ny Nz 0)
    //     ( 0  0  0 1)
    // where T is the tangent, B is binormal and N is the normal (all in object space).
    // T = shared edge ai-ak; B = pijk - tik; N = T x B
    float3 ai = make_float3( visibleTria[0].x, visibleTria[0].y, visibleTria[0].z);
    float ri = visibleTria[0].w;
    float3 aj = make_float3( visibleTria[1].x, visibleTria[1].y, visibleTria[1].z);
    float3 ak = make_float3( visibleTria[2].x, visibleTria[2].y, visibleTria[2].z);
    float rk = visibleTria[2].w;

    float3 T = normalize( ai - ak);

    float dik = length( ak - ai);
    float3 tik = 0.5f*( ai + ak) + 0.5f*( ak - ai) * ( ( ri + params.probeRadius)*( ri + params.probeRadius) - ( rk + params.probeRadius)*( rk + params.probeRadius))/( dik*dik);
    float3 B = normalize( make_float3( visibleTriaNormal.x, visibleTriaNormal.y, visibleTriaNormal.z) - tik);
    
    float3 N = normalize( cross( T, B));
    
    // set angle to more than 2*PI
    angle = 7.0f;
    for( cnt = 0; cnt < counter; ++cnt ) {
        // get direction to pijk'
        tmpDir = normalize( make_float3( invisibleTriaNormal[cnt].x, invisibleTriaNormal[cnt].y, invisibleTriaNormal[cnt].z) - tik);
        // project direction to tangent space
        tmpDirTS = make_float3( dot(T, tmpDir), dot(B, tmpDir), dot(N, tmpDir));
        tmpAngle = atan2f( tmpDirTS.z, tmpDirTS.y) + 3.14159265f;
        if( tmpAngle < angle ) {
            angle = tmpAngle;
            smallest = invisibleTriaId[cnt];
        }
        // get direction to pijk'2
        tmpDir = normalize( make_float3( invisibleTriaNormal[cnt].x, invisibleTriaNormal[cnt].y, invisibleTriaNormal[cnt].z)*invisibleTriaNormal[cnt].w - tik);
        // project direction to tangent space
        tmpDirTS = make_float3( dot(T, tmpDir), dot(B, tmpDir), dot(N, tmpDir));
        tmpAngle = atan2f( tmpDirTS.z, tmpDirTS.y) + 3.14159265f;
        if( tmpAngle < angle ) {
            angle = tmpAngle;
            smallest = invisibleTriaId[cnt];
        }
    }
    
    if( smallest >= 0 )
        outPbo[params.maxNumNeighbors * params.maxNumNeighbors * visibleAtomIdx + xcoord + uint(smallest)] = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////
//
// CONTOUR BUILDUP KERNELS
//
///////////////////////////////////////////////////////////////////////////////

// find all neighbor atoms in a given cell
__device__ uint findNeighborsInCellCBCuda(
        uint*   neighbors,      // out: neighbor indices
        float4* smallCircles,   // out: small circles
        uint    neighborIndex,  // in: first index for writing in neighbor list
        int3    gridPos,        // in: the current grid cell
        uint    index,          // in: the index of the atom
        float4  pos,            // in: the position of the atom
        float4* atomPos,        // in: the (sorted) atom position array
        uint*   cellStart,
        uint*   cellEnd) {
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH( cellStart, gridHash);

    uint count = 0;
    float4 pos2;
    float3 relPos;
    float dist;
    float neighborDist;
    float r;
    float3 vec;
    float4 smallCircle;
    if( startIndex != 0xffffffff ) {	// cell is not empty
        // iterate over atoms in this cell
        uint endIndex = FETCH( cellEnd, gridHash);
        for( uint j = startIndex; j < endIndex; j++) {
            // do not count self
            if( j != index) {
                // get position of potential neighbor
                pos2 = FETCH( atomPos, j);
                // check distance
                relPos = make_float3( pos2.x, pos2.y, pos2.z) - make_float3( pos.x, pos.y, pos.z);
                dist = length( relPos);
                neighborDist = pos.w + pos2.w + 2.0f * params.probeRadius;
                if( dist < neighborDist ) {
                    // check number of neighbors
                    if( ( neighborIndex + count) >= params.maxNumNeighbors ) return count;
                    // write the (sorted) neighbor index
                    neighbors[index*params.maxNumNeighbors+neighborIndex+count] = j;
                    // compute small circle / intersection plane
                    r = ( (pos.w + params.probeRadius)*(pos.w + params.probeRadius))
                        + ( dist * dist)
                        - ( (pos2.w + params.probeRadius)*(pos2.w + params.probeRadius));
                    r = r / (2.0f * dist * dist);
                    vec = relPos * r;
                    // set small circle
                    smallCircle.x = vec.x;
                    smallCircle.y = vec.y;
                    smallCircle.z = vec.z;
                    //smallCircle.w = 1.0f;
                    smallCircle.w = sqrt(((pos.w + params.probeRadius) * (pos.w + params.probeRadius)) - dot( vec, vec));
                    smallCircles[index*params.maxNumNeighbors+neighborIndex+count] = smallCircle;
                    // increment the neighbor counter
                    count++;
                }
            }
        }
    }
    return count;
}

// Find and count neighbors and write them to the neighbor array.
// Also computes small circles for each neighbor and stores them, too.
__global__ void findNeighborsCBCuda(
        uint*   neighborCount,     // out: number of neighbors
        uint*   neighbors,         // out: neighbor indices
        float4* smallCircles,      // out: small circles
        float4* atomPos,           // in: sorted atom positions
        uint*   cellStart,
        uint*   cellEnd,
        uint    numAtoms) {
    uint index = __mul24( blockIdx.x, blockDim.x) + threadIdx.x;
    if( index >= numAtoms ) return;

    // read atom data from sorted arrays
    float4 pos = FETCH( atomPos, index);

    // get address in grid
    int3 gridPos = calcGridPos( make_float3( pos));

    int3 gridSize;
    gridSize.x = int( params.gridSize.x);
    gridSize.y = int( params.gridSize.y);
    gridSize.z = int( params.gridSize.z);
    // search range for neighbor atoms: max atom diameter + probe diameter
    float range = ( pos.w + 3.0f + 2.0f * params.probeRadius);
    // compute number of grid cells
    int3 cellsInRange;
    cellsInRange.x = ceil( range / params.cellSize.x);
    cellsInRange.y = ceil( range / params.cellSize.y);
    cellsInRange.z = ceil( range / params.cellSize.z);
    int3 start = gridPos - cellsInRange;
    int3 end = gridPos + cellsInRange;

    // examine neighbouring cells
    uint count = 0;
    int3 neighborPos;
    for( int z = ( start.z > 0 ? start.z : 0); z < ( end.z > gridSize.z ? gridSize.z : end.z) ; z++ ) {
        for( int y = ( start.y > 0 ? start.y : 0); y < ( end.y > gridSize.y ? gridSize.y : end.y) ; y++ ) {
            for( int x = ( start.x > 0 ? start.x : 0); x < ( end.x > gridSize.x ? gridSize.x : end.x) ; x++ ) {
                neighborPos = make_int3( x, y, z);
                count += findNeighborsInCellCBCuda( neighbors, smallCircles, count, neighborPos, index, pos, atomPos, cellStart, cellEnd);
            }
        }
    }

    // write new neighbor atom count back to (sorted) index location
    neighborCount[index] = count;
}

// find and remove unnecessary small circles
__global__ void removeCoveredSmallCirclesCBCuda(
        float4* smallCircles,         // in/out: small circles
        uint*   smallCircleVisible,   // out: small circle visibility
        uint*   neighborCount,        // in/out: number of neighbors
        uint*   neighbors,            // input: neighbor indices
        float4* atomPos,              // input: sorted atom positions
        uint    numAtoms) {
    // get atom index
    uint atomIdx = blockIdx.y * blockDim.y + threadIdx.y;
    // get neighbor atom index
    uint jIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // check, if atom index is within bounds
    if( atomIdx >= numAtoms ) return;
    // check, if neighbor index is within bounds
    if( jIdx >= params.maxNumNeighbors ) return;
    // set small circle visibility to false
    smallCircleVisible[atomIdx * params.maxNumNeighbors + jIdx] = 0;
    // check, if neighbor index is within bounds
    uint numNeighbors = neighborCount[atomIdx];
    if( jIdx >= numNeighbors ) return;

    // read position and radius of atom i from sorted array
    float4 atomi = FETCH( atomPos, atomIdx);
    float3 pi = make_float3( atomi.x, atomi.y, atomi.z);
    float R = atomi.w + params.probeRadius;

    // flag wether j sould be added (true) is cut off (false)
    bool addJ = true;

    // the atom index of j
    uint j = FETCH( neighbors, atomIdx * params.maxNumNeighbors + jIdx);
    // get small circle j
    float4 scj = smallCircles[atomIdx * params.maxNumNeighbors + jIdx];
    // vj = the small circle center
    float3 vj = make_float3( scj.x, scj.y, scj.z);
    // pj = center of atom j
    float4 aj = FETCH( atomPos, j);
    float3 pj = make_float3( aj.x, aj.y, aj.z);
    // variables for k
    uint k;
    float4 sck;
    float3 vk;
    float4 ak;
    float3 pk;
    float vjvk;
    float denom;
    float3 h;
    float3 nj;
    float3 nk;
    float3 q;
    float3 mj;
    float3 mk;

    // check j with all other neighbors k
    for( uint kCnt = 0; kCnt < numNeighbors; kCnt++ ) {
        // don't compare the circle with itself
        if( jIdx != kCnt ) {
            // the atom index of k
            k = FETCH( neighbors, atomIdx * params.maxNumNeighbors + kCnt);
            // pk = center of atom k
            ak = FETCH( atomPos, k);
            pk = make_float3( ak.x, ak.y, ak.z);
            // get small circle k
            sck = smallCircles[atomIdx * params.maxNumNeighbors + kCnt];
            // vk = the small circle center
            vk = make_float3( sck.x, sck.y, sck.z);
            // vj * vk
            vjvk = dot( vj, vk);
            // denominator
            denom = dot( vj, vj) * dot( vk, vk) - vjvk * vjvk;
            // point on straight line (intersection of small circle planes)
            h = vj * ( dot( vj, vj - vk) * dot( vk, vk) ) / denom + vk * ( dot( vk - vj, vk) * dot( vj, vj) ) / denom;
            // compute cases
            nj = normalize( pi - pj);
            nk = normalize( pi - pk);
            q = vk - vj;
            // if normals are the same (unrealistic, yet theoretically possible)
            if( dot( nj, nk) == 1.0f ) {
                if( dot( nj, nk) > 0.0f ) {
                    if( dot( nj, q) > 0.0f ) {
                        // k cuts off j --> remove j
                        addJ = false;
                    }
                }
            } else if( length( h) > R ) {
                mj = ( vj - h);
                mk = ( vk - h);
                if( dot( nj, nk) > 0.0f ) {
                    if( dot( mj, mk) > 0.0f && dot( nj, q) > 0.0f ) {
                        // k cuts off j --> remove j
                        addJ = false;
                    }
                } else {
                    if( dot( mj, mk) > 0.0f && dot( nj, q) < 0.0f ) {
                        // atom i has no contour
                        neighborCount[atomIdx] = 0;
                    }
                }
            }
        }
    }
    // all k were tested, see if j is cut off
    if( !addJ ) {
        smallCircles[atomIdx * params.maxNumNeighbors + jIdx].w = -1.0f;
    }
}

// compute all arcs of atom j on the surface of atom i
__global__ void computeArcsCBCuda(
        uint*   smallCircleVisible,		// out: small circle visibility
        float4* arcs,					// out: the arcs
        uint*   arcCount,				// out: the number of arcs
        uint    numAtoms) {
    // get atom index
    uint atomIdx = blockIdx.y * blockDim.y + threadIdx.y;
    // get neighbor atom index
    uint jIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // check, if atom index is within bounds
    if( atomIdx >= numAtoms ) return;
    // check, if neighbor index is within bounds
    if( jIdx >= params.maxNumNeighbors ) return;
    // check, if neighbor index is within bounds
    uint numNeighbors = FETCH( neighborCount, atomIdx);
    if( jIdx >= numNeighbors ) return;

    // read position and radius of atom i from sorted array
    float4 atomi = FETCH( atomPos, atomIdx);
    float3 pi = make_float3( atomi.x, atomi.y, atomi.z);
    float R = atomi.w + params.probeRadius;

    // the atom index of j
    //uint j = neighbors[atomIdx * params.maxNumNeighbors + jIdx];
    uint j = FETCH( neighbors, atomIdx * params.maxNumNeighbors + jIdx);
    // get small circle j
    float4 scj = FETCH( smallCircles, atomIdx * params.maxNumNeighbors + jIdx);
    // do nothing if small circle j has radius -1 (removed)
    if( scj.w < 0.0f )
        return;
    // vj = the small circle center
    float3 vj = make_float3( scj.x, scj.y, scj.z);
    // pj = center of atom j
    float4 aj = FETCH( atomPos, j);
    float3 pj = make_float3( aj.x, aj.y, aj.z);
    // store all arcs
    float start[64];
    float end[64];
    uint startkIndex[64];
    uint endkIndex[64];
    bool arcValid[64];
    start[0] = 0.0f;
    end[0] = 6.28318530718f;
    startkIndex[0] = 0;
    endkIndex[0] = 0;
    arcValid[0] = true;
    uint arcCnt = 1;
    // temporary arc arrays for new arcs
    float tmpStart[16];
    float tmpEnd[16];
    uint tmpStartkIndex[16];
    uint tmpEndkIndex[16];
    uint tmpArcCnt = 0;
    // compute axes of local coordinate system
    float3 ex = make_float3( 1.0f, 0.0f, 0.0f);
    float3 ey = make_float3( 0.0f, 1.0f, 0.0f);
    float3 xAxis = cross( vj, ey);
    if( dot( xAxis, xAxis) == 0.0f ) {
        xAxis = cross( vj, ex);
    }
    xAxis = normalize( xAxis);
    float3 yAxis = cross( xAxis, vj);
    yAxis = normalize( yAxis);

    // variables for k
    uint k;
    float4 sck;
    float3 vk;
    float4 ak;
    float3 pk;
    float vjvk;
    float denom;
    float3 h;
    float root;
    float3 x1;
    float3 x2;
    float3 tmpVec;
    float xX1;
    float yX1;
    float xX2;
    float yX2;
    float angleX1;
    float angleX2;
    uint aCnt;
    float s;
    float e;
    float skIndex;
    float ekIndex;

    // check j with all other neighbors k
    for( uint kCnt = 0; kCnt < numNeighbors; kCnt++ ) {
        // don't compare the circle with itself
        if( jIdx == kCnt ) 
            continue;
        // the atom index of k
        //k = neighbors[atomIdx * params.maxNumNeighbors + kCnt];
        k = FETCH( neighbors, atomIdx * params.maxNumNeighbors + kCnt);
        // get small circle k
        sck = FETCH( smallCircles, atomIdx * params.maxNumNeighbors + kCnt);
        // do nothing if small circle k has radius -1 (removed)
        if( sck.w < 0.0f )
            continue;
        // vk = the small circle center
        vk = make_float3( sck.x, sck.y, sck.z);
        // pk = center of atom k
        ak = FETCH( atomPos, k);
        pk = make_float3( ak.x, ak.y, ak.z);
        // vj * vk
        vjvk = dot( vj, vk);
        // denominator
        denom = dot( vj, vj) * dot( vk, vk) - vjvk * vjvk;
        // point on straight line (intersection of small circle planes)
        h = vj * ( dot( vj, vj - vk) * dot( vk, vk) ) / denom + vk * ( dot( vk - vj, vk) * dot( vj, vj) ) / denom;

        // do nothing if h is outside of the extended sphere of atom i
        if( length( h) > R ) 
            continue;
        // compute the root
        root = sqrt( ( R*R - dot( h, h)) / dot( cross( vk, vj), cross( vk, vj)));
        // compute the two intersection points
        x1 = h + cross( vk, vj) * root;
        x2 = h - cross( vk, vj) * root;
        // swap x1 & x2 if vj points in the opposit direction of pj-pi
        if( dot( vk, pk - pi) < 0.0f ) {
            tmpVec = x1;
            x1 = x2;
            x2 = tmpVec;
        }
        // transform x1 and x2 to small circle coordinate system
        xX1 = dot( x1 - vj, xAxis);
        yX1 = dot( x1 - vj, yAxis);
        xX2 = dot( x2 - vj, xAxis);
        yX2 = dot( x2 - vj, yAxis);
        angleX1 = atan2( yX1, xX1);
        angleX2 = atan2( yX2, xX2);
        // limit angles to 0..2*PI
        if( angleX1 > 6.28318530718f ) {
            angleX1 = fmod( angleX1, 6.28318530718f);
            angleX2 = fmod( angleX2, 6.28318530718f);
        }
        // angle of x2 has to be larger than angle of x1 (add 2 PI)
        if( angleX2 < angleX1 ) {
            angleX2 += 6.28318530718f;
        }
        // make all angles positive (add 2 PI)
        if( angleX1 < 0.0f ) {
            angleX1 += 6.28318530718f;
            angleX2 += 6.28318530718f;
        }

        // check all existing arcs with new arc k
        for( aCnt = 0; aCnt < arcCnt; aCnt++ ) {
            s = start[aCnt];
            e = end[aCnt];
            skIndex = startkIndex[aCnt];
            ekIndex = endkIndex[aCnt];
            if( arcValid[aCnt] ) {
                if( angleX1 < s ) {
                    // case (1) & (10)
                    if( ( s - angleX1) > ( angleX2 - angleX1)) {
                        if( ( ( s - angleX1) + ( e - s)) > 6.28318530718f ) {
                            if( ( ( s - angleX1) + ( e - s)) < ( 6.28318530718f + angleX2 - angleX1) ) {
                                // case (10)
                                start[aCnt] = angleX1;
                                startkIndex[aCnt] = k;
                                end[aCnt] = fmod( e, 6.28318530718f);
                                // second angle check
                                if( end[aCnt] < start[aCnt] )
                                    end[aCnt] += 6.28318530718f;
                            } else {
                                start[aCnt] = angleX1;
                                startkIndex[aCnt] = k;
                                end[aCnt] = fmod( angleX2, 6.28318530718f);
                                endkIndex[aCnt] = k;
                                // second angle check
                                if( end[aCnt] < start[aCnt] )
                                    end[aCnt] += 6.28318530718f;
                            }
                        } else {
                            // case (1)
                            //arcAngles.RemoveAt( aCnt);
                            //aCnt--;
                            arcValid[aCnt] = false;
                        }
                    } else {
                        if( ( ( s - angleX1) + ( e - s)) > ( angleX2 - angleX1) ) {
                            // case (5)
                            end[aCnt] = fmod( angleX2, 6.28318530718f);
                            endkIndex[aCnt] = k;
                            // second angle check
                            if( end[aCnt] < start[aCnt] )
                                end[aCnt] += 6.28318530718f;
                            if( ( ( s - angleX1) + ( e - s)) > 6.28318530718f ) {
                                // case (6)
                                tmpStart[tmpArcCnt] = angleX1;
                                tmpStartkIndex[tmpArcCnt] = k;
                                tmpEnd[tmpArcCnt] = fmod( e, 6.28318530718f);
                                tmpEndkIndex[tmpArcCnt] = ekIndex;
                                // second angle check
                                if( tmpEnd[tmpArcCnt] < tmpStart[tmpArcCnt] )
                                    tmpEnd[tmpArcCnt] += 6.28318530718f;
                                tmpArcCnt++;
                            }
                        }
                    } // case (4): Do nothing!
                } else { // angleX1 > s
                    // case (2) & (9)
                    if( ( angleX1 - s) > ( e - s)) {
                        if( ( ( angleX1 - s) + ( angleX2 - angleX1)) > 6.28318530718f ) {
                            if( ( ( angleX1 - s) + ( angleX2 - angleX1)) < ( 6.28318530718f + e - s)) {
                                // case (9)
                                end[aCnt] = fmod( angleX2, 6.28318530718f);
                                endkIndex[aCnt] = k;
                                // second angle check
                                if( end[aCnt] < start[aCnt] )
                                    end[aCnt] += 6.28318530718f;
                            }
                        } else {
                            // case (2)
                            //arcAngles.RemoveAt( aCnt);
                            //aCnt--;
                            arcValid[aCnt] = false;
                        }
                    } else {
                        if( ( ( angleX1 - s) + ( angleX2 - angleX1)) > ( e - s) ) {
                            // case (7)
                            start[aCnt] = angleX1;
                            startkIndex[aCnt] = k;
                            // second angle check
                            end[aCnt] = fmod( end[aCnt], 6.28318530718f);
                            if( end[aCnt] < start[aCnt] )
                                end[aCnt] += 6.28318530718f;
                            if( ( ( angleX1 - s) + ( angleX2 - angleX1)) > 6.28318530718f ) {
                                // case (8)
                                tmpStart[tmpArcCnt] = s;
                                tmpStartkIndex[tmpArcCnt] = skIndex;
                                tmpEnd[tmpArcCnt] = fmod( angleX2, 6.28318530718f);
                                tmpEndkIndex[tmpArcCnt] = k;
                                // second angle check
                                if( tmpEnd[tmpArcCnt] < tmpStart[tmpArcCnt] )
                                    tmpEnd[tmpArcCnt] += 6.28318530718f;
                                tmpArcCnt++;
                            }
                        } else {
                            // case (3)
                            start[aCnt] = angleX1;
                            startkIndex[aCnt] = k;
                            end[aCnt] = fmod( angleX2, 6.28318530718f);
                            endkIndex[aCnt] = k;
                            // second angle check
                            if( end[aCnt] < start[aCnt] )
                                end[aCnt] += 6.28318530718f;
                        }
                    }
                }
            } // if( arcValid[aCnt] )
        } // for( uint aCnt = 0; aCnt < arcCnt; aCnt++ )

        // copy new arcs to arc array
        for( aCnt = 0; aCnt < tmpArcCnt; aCnt++ ) {
            start[aCnt + arcCnt] = tmpStart[aCnt];
            end[aCnt + arcCnt] = tmpEnd[aCnt];
            startkIndex[aCnt + arcCnt] = tmpStartkIndex[aCnt];
            endkIndex[aCnt + arcCnt] = tmpEndkIndex[aCnt];
            arcValid[aCnt + arcCnt] = true;
        }
        // add new arcs to arc count
        arcCnt += tmpArcCnt;
        // "reset" temporary arc array
        tmpArcCnt = 0;

        // fill gaps (overwrite invalid arcs)
        uint counter = 0;
        for( aCnt = 0; aCnt < arcCnt; aCnt++ ) {
            if( arcValid[aCnt] ) {
                start[aCnt - counter] = start[aCnt];
                end[aCnt - counter] = end[aCnt];
                startkIndex[aCnt - counter] = startkIndex[aCnt];
                endkIndex[aCnt - counter] = endkIndex[aCnt];
                arcValid[aCnt - counter] = arcValid[aCnt];
            } else {
                counter++;
            }
        }
        // subtract number of invalid arcs from total number of arcs
        arcCnt -= counter;
    } // for( uint kCnt = 0; kCnt < numNeighbors; kCnt++ )

    // TODO: remove/merge split arcs ( x..2*PI / 0..y --> x..y+2*PI )

    // merge arcs if arc with angle 0 and arc with angle 2*PI exist
    int idx0 = -1;
    int idx2pi = -1;
    for( aCnt = 0; aCnt < arcCnt; aCnt++ ) {
        if( start[aCnt] < 0.00001f ) {
            idx0 = int( aCnt);
        } else if( abs( end[aCnt] - 6.28318530718f) < 0.0001f ) {
            idx2pi = int( aCnt);
        }
    }
    if( idx0 >= 0 && idx2pi >= 0 ) {
        start[uint(idx0)] = start[uint(idx2pi)];
        startkIndex[uint(idx0)] = startkIndex[uint(idx2pi)];
        // second angle check
        end[uint(idx0)] = fmod( end[uint(idx0)], 6.28318530718f);
        if( end[uint(idx0)] < start[uint(idx0)] )
            end[uint(idx0)] += 6.28318530718f;
        // fill gaps (overwrite removed arc idx2pi)
        for( aCnt = uint(idx2pi); aCnt < arcCnt - 1; aCnt++ ) {
            start[aCnt] = start[aCnt + 1];
            end[aCnt] = end[aCnt + 1];
            startkIndex[aCnt] = startkIndex[aCnt + 1];
            endkIndex[aCnt] = endkIndex[aCnt + 1];
            arcValid[aCnt] = true;
        }
        // subtract the removed arc from total number of arcs
        arcCnt--;
    }

    uint arcWritten = 0;
    // copy arcs to global arc array
    for( aCnt = 0; aCnt < arcCnt; aCnt++ ) {
        if( atomIdx < j ) {
            if( j < startkIndex[aCnt] ) {
                arcs[atomIdx * params.maxNumNeighbors * params.maxNumNeighbors + jIdx * params.maxNumNeighbors + arcWritten] = 
                    make_float4( pi + vj + ( cos( start[aCnt]) * xAxis + sin( start[aCnt]) * yAxis) * scj.w, float( startkIndex[aCnt]) + 0.2f); //start[aCnt]);
                arcWritten++;
            }
            if( j < endkIndex[aCnt] ) {
                arcs[atomIdx * params.maxNumNeighbors * params.maxNumNeighbors + jIdx * params.maxNumNeighbors + arcWritten] = 
                    make_float4( pi + vj + ( cos( end[aCnt]) * xAxis + sin( end[aCnt]) * yAxis) * scj.w, float( endkIndex[aCnt]) + 0.2f); //end[aCnt]);
                arcWritten++;
            }
        }
    }

    // write number of arcs
    arcCount[atomIdx * params.maxNumNeighbors + jIdx] = arcWritten;

    // set small circle j visible if at least one arc was created and i < j
    if( atomIdx < j && arcCnt > 0 ) {
    //if( arcWritten > 0 ) {
        smallCircleVisible[atomIdx * params.maxNumNeighbors + jIdx] = 1;
    }
    // DO NOT USE THIS!! It will create false, internal arcs!
    //if( arcCnt == 0 ) {
    //	smallCircles[atomIdx * params.maxNumNeighbors + jIdx].w = -1.0f;
    //}

}

// Write all arc start and end points to a vertex array.
__global__ void writeProbePositionsCBCuda(
        float4*	probePos,		// out: the probe positions
        float4*	sphereTriaVec1,	// out: the spherical triangle vector 1
        float4*	sphereTriaVec2,	// out: the spherical triangle vector 2
        float4*	sphereTriaVec3,	// out: the spherical triangle vector 3
        float4*	torusPos,		// out: torus center
        float4*	torusVS,		// out: torus visibility sphere
        float4*	torusAxis,		// out: torus axis
        float4* arcs,			// in: the arc start and end points
        uint*	arcCount,		// in: the number of probes
        uint*	arcCountScan,	// in: the prefix sum of "arcCount"
        uint*	scCount,		// in: the small circle count
        uint*	scCountScan,	// in: the prefix sum of "scCount"
        uint    numAtoms) {
    // get atom index
    uint atomIdx = blockIdx.y * blockDim.y + threadIdx.y;
    // get neighbor atom index
    uint jIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // check, if atom index is within bounds
    if( atomIdx >= numAtoms ) return;
    // check, if neighbor index is within bounds
    uint numNeighbors = FETCH( neighborCount, atomIdx);
    if( jIdx >= numNeighbors ) return;

    float4 ai = FETCH( atomPos, atomIdx);
    float4 aj = FETCH( atomPos, FETCH( neighbors, atomIdx * params.maxNumNeighbors + jIdx));
    float4 ak;

    // ---------- write spherical triangle ----------

    // get number of probes and the sum of previous probes for this neighbor
    uint numProbes = arcCount[atomIdx * params.maxNumNeighbors + jIdx];
    uint numPreviousProbes = arcCountScan[atomIdx * params.maxNumNeighbors + jIdx];

    float4 tmpProbePos;
    uint kIdx;

    for( uint cnt = 0; cnt < numProbes; cnt++ ) {
        tmpProbePos = arcs[atomIdx * params.maxNumNeighbors * params.maxNumNeighbors + jIdx * params.maxNumNeighbors + cnt];
        probePos[numPreviousProbes + cnt] = make_float4( tmpProbePos.x, tmpProbePos.y, tmpProbePos.z, params.probeRadius);
        kIdx = uint( floor( tmpProbePos.w));
        //ak = FETCH( atomPos, FETCH( neighbors, kIdx));
        ak = FETCH( atomPos, kIdx);
        sphereTriaVec1[numPreviousProbes + cnt] = make_float4( make_float3( ai) - make_float3( tmpProbePos), 1.0f);
        sphereTriaVec2[numPreviousProbes + cnt] = make_float4( make_float3( aj) - make_float3( tmpProbePos), 1.0f);
        sphereTriaVec3[numPreviousProbes + cnt] = make_float4( make_float3( ak) - make_float3( tmpProbePos), params.probeRadius * params.probeRadius);
    }

    // ---------- write torus ----------

    if( scCount[atomIdx * params.maxNumNeighbors + jIdx] > 0 ) {
        float4 sc;
        uint torusIdx = scCountScan[atomIdx * params.maxNumNeighbors + jIdx];
        sc = FETCH( smallCircles, atomIdx * params.maxNumNeighbors + jIdx);
        // torus axis
        float3 ta = normalize( make_float3( sc));
        // torus center
        float3 tc = make_float3( sc) + make_float3( ai);
        float3 ortho = normalize( cross( ta, make_float3( 0.0f, 0.0f, 1.0f)));

        // compute the tangential point X2 of the spheres
        float3 P = tc + ( ortho * sc.w);
        float3 X = normalize( P - make_float3( ai)) * ai.w;
        float3 C = ( length( P - make_float3( ai)) /
            ( length( P - make_float3( aj)) + length( P - make_float3( ai)))) * 
            ( make_float3( aj) - make_float3( ai));
        float distance = length( X - C);
        C = ( C + make_float3( ai)) - tc;

        // write torus center & torus radius R
        torusPos[torusIdx] = make_float4( tc, sc.w);
        // write torus axis & probe radius (= torus radius r)
        torusAxis[torusIdx] = make_float4( ta, params.probeRadius);
        // write visibility sphere
        torusVS[torusIdx] = make_float4( C, distance);
    }

}

// find all probe neighbors in a given cell
__device__ uint findProbeNeighborsInCellCBCuda(
        float3* singTex,        // out: singularity texture
        uint    neighborIndex,  // in: first index for writing in neighbor list
        int3    gridPos,        // in: the current grid cell
        uint    index,          // in: the index of the probe
        float4  pos,            // in: the position of the probe
        float4* probePos,       // in: the (sorted) probe position array
        uint*   gridProbeIndex, // input: sorted probe indices
        uint*   cellStart,
        uint*   cellEnd,
        uint    numNeighbors ) {
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH( cellStart, gridHash);

    uint count = 0;
    float4 pos2;
    float3 relPos;
    float dist;
    float neighborDist;
    if( startIndex != 0xffffffff ) {	// cell is not empty
        // iterate over atoms in this cell
        uint endIndex = FETCH( cellEnd, gridHash);
        for( uint j = startIndex; j < endIndex; j++) {
            // do not count self
            if( j != index) {
                // get position of potential neighbor
                pos2 = probePos[j];
                // check distance
                relPos = make_float3( pos2.x, pos2.y, pos2.z) - make_float3( pos.x, pos.y, pos.z);
                dist = length( relPos);
                neighborDist = params.probeRadius + params.probeRadius;
                if( dist < neighborDist ) {
                    // check number of neighbors
                    if( ( neighborIndex + count) >= numNeighbors ) return count;
                    // write singTex
                    singTex[index * numNeighbors + neighborIndex + count] = make_float3( pos2);
                    // increment the neighbor counter
                    count++;
                }
            }
        }
    }
    return count;
}

// Find and count probe neighbors and write them to the singularity texture
__global__ void findProbeNeighborsCBCuda(
        float3* texCoord,       // out: #neighbors + texture coords
        float3* singTex,        // out: singularity texture
        float4* probePos,       // in: probe positions
        uint*   gridProbeIndex, // input: sorted probe indices
        uint*   cellStart,
        uint*   cellEnd,
        uint    numProbes,
        uint    numNeighbors ) {
    uint index = __mul24( blockIdx.x, blockDim.x) + threadIdx.x;
    if( index >= numProbes ) return;

    // read atom data from sorted arrays
    float4 pos = probePos[index];

    // get address in grid
    int3 gridPos = calcGridPos( make_float3( pos));

    int3 gridSize;
    gridSize.x = int( params.gridSize.x);
    gridSize.y = int( params.gridSize.y);
    gridSize.z = int( params.gridSize.z);
    // search range for neighbor atoms: max atom diameter + probe diameter
    float range = params.probeRadius + params.probeRadius;
    // compute number of grid cells
    int3 cellsInRange;
    cellsInRange.x = ceil( range / params.cellSize.x);
    cellsInRange.y = ceil( range / params.cellSize.y);
    cellsInRange.z = ceil( range / params.cellSize.z);
    int3 start = gridPos - cellsInRange;
    int3 end = gridPos + cellsInRange;

    // examine neighbouring cells
    uint count = 0;
    int3 neighborPos;
    for( int z = ( start.z > 0 ? start.z : 0); z < ( end.z > gridSize.z ? gridSize.z : end.z) ; z++ ) {
        for( int y = ( start.y > 0 ? start.y : 0); y < ( end.y > gridSize.y ? gridSize.y : end.y) ; y++ ) {
            for( int x = ( start.x > 0 ? start.x : 0); x < ( end.x > gridSize.x ? gridSize.x : end.x) ; x++ ) {
                neighborPos = make_int3( x, y, z);
                count += findProbeNeighborsInCellCBCuda( singTex, count, neighborPos, index, pos, probePos, gridProbeIndex, cellStart, cellEnd, numNeighbors);
            }
        }
    }

    uint xCoord = index / params.texSize;
    // write probe neighbor count and texture coordinates
    texCoord[gridProbeIndex[index]] = make_float3( float( count), float( xCoord), float( index % params.texSize));
}

#endif
