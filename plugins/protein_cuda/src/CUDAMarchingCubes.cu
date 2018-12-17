/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: CUDAMarchingCubes.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.8 $       $Date: 2011/11/22 21:17:57 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA Marching Cubes Implementation
 *
 * LICENSE:
 *   UIUC Open Source License
 *   http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html
 *
 ***************************************************************************/

//
// Description: This class computes an isosurface for a given density grid
//              using a CUDA Marching Cubes (MC) alorithm. 
//              The implementation is based on the MC demo from the 
//              Nvidia GPU Computing SDK, but has been improved 
//              and extended.  This implementation achieves higher 
//              performance by reducing the number of temporary memory
//              buffers, reduces the number of scan calls by using vector
//              integer types, and allows extraction of per-vertex normals 
//              optionally computes per-vertex colors if provided with a 
//              volumetric texture map.
//
// Author: Michael Krone <michael.krone@visus.uni-stuttgart.de>
//         John Stone <johns@ks.uiuc.edu>
//
// Copyright 2011
//

#include "CUDAKernels.h"
#define CUDAMARCHINGCUBES_INTERNAL 1
#include "CUDAMarchingCubes.h"
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

// The number of threads to use for triangle generation 
// (limited by shared memory size)
#define NTHREADS 48

#define USE_CUDA_3D_TEXTURE

//
// Various math operators for vector types not already 
// provided by the regular CUDA headers
//
// "+" operator
inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ uint3 operator+(uint3 a, uint3 b) {
  return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ uint2 operator+(uint2 a, uint2 b) {
  return make_uint2(a.x + b.x, a.y + b.y);
}

// "-" operator
inline __host__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ uint3 operator-(uint3 a, uint3 b) {
  return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// "*" operator
inline __host__ __device__ float3 operator*(float b, float3 a) {
  return make_float3(b * a.x, b * a.y, b * a.z);
}

// dot()
inline __host__ __device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// lerp()
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t) {
  return a + t*(b-a);
}

// length()
inline __host__ __device__ float length(float3 v) {
  return sqrtf(dot(v, v));
}

//
// CUDA textures containing marching cubes look-up tables
// Note: SIMD marching cubes implementations have no need for the edge table
//
texture<int, 1, cudaReadModeElementType> triTex;
texture<unsigned int, 1, cudaReadModeElementType> numVertsTex;
// 3D 24-bit RGB texture
texture<float, 3, cudaReadModeElementType> volumeTex;

// sample volume data set at a point
__device__ float sampleVolume(float *data, uint3 p, uint3 gridSize) {
    // gridPos CAN NEVER BE OUT OF BOUNDS
    //p.x = (p.x >= gridSize.x) ? gridSize.x - 1 : p.x;
    //p.y = (p.y >= gridSize.y) ? gridSize.y - 1 : p.y;
    //p.z = (p.z >= gridSize.z) ? gridSize.z - 1 : p.z;
    unsigned int i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
    return data[i];
}


// sample volume data set at a point
__device__ float3 sampleColors(float3 *data, uint3 p, uint3 gridSize) {
    // gridPos CAN NEVER BE OUT OF BOUNDS
    //p.x = (p.x >= gridSize.x) ? gridSize.x - 1 : p.x;
    //p.y = (p.y >= gridSize.y) ? gridSize.y - 1 : p.y;
    //p.z = (p.z >= gridSize.z) ? gridSize.z - 1 : p.z;
    unsigned int i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
    return data[i];
}


// compute position in 3d grid from 1d index
__device__ uint3 calcGridPos(unsigned int i, uint3 gridSize) {
    uint3 gridPos;
    unsigned int gridsizexy = gridSize.x * gridSize.y;
    gridPos.z = i / gridsizexy;
    unsigned int tmp1 = i - (gridsizexy * gridPos.z);
    gridPos.y = tmp1 / gridSize.x;
    gridPos.x = tmp1 - (gridSize.x * gridPos.y);
    return gridPos;
}


// compute interpolated vertex along an edge
__device__ float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1) {
    float t = (isolevel - f0) / (f1 - f0);
    return lerp(p0, p1, t);
}


// classify voxel based on number of vertices it will generate one thread per two voxels
// due to type system changes in CUDA, uint2 had to be replaced by myuint2
template <int gridis3d, int subgrid>
__global__ void classifyVoxel(myuint2* voxelVerts, float *volume,
                              uint3 gridSize, unsigned int numVoxels, float3 voxelSize,
                              uint3 subGridStart, uint3 subGridEnd,
                              float isoValue) {
    uint3 gridPos;
    unsigned int i;

    // Compute voxel indices and address using either 2-D or 3-D 
    // thread indexing depending on the caller-provided gridis3d parameter
    if (gridis3d) {
      // Compute voxel index using 3-D thread indexing
      // compute 3D grid position
      gridPos.x = blockIdx.x * blockDim.x + threadIdx.x;
      gridPos.y = blockIdx.y * blockDim.y + threadIdx.y;
      gridPos.z = blockIdx.z * blockDim.z + threadIdx.z;

      // safety check
      if (gridPos.x >= gridSize.x || 
          gridPos.y >= gridSize.y || 
          gridPos.z >= gridSize.z)
        return;

      // compute 1D grid index
      i = gridPos.z*gridSize.x*gridSize.y + gridPos.y*gridSize.x + gridPos.x;
    } else {
      // Compute voxel index using 2-D thread indexing
      unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
      i = (blockId * blockDim.x) + threadIdx.x;

      // safety check
      if (i >= numVoxels)
        return;

      // compute current grid position
      gridPos = calcGridPos(i, gridSize);
    }

    // If we are told to compute the isosurface for only a sub-region
    // of the volume, we use a more complex boundary test, otherwise we
    // use just the maximum voxel dimension
    uint2 numVerts = make_uint2(0, 0); // initialize vertex output to zero
    if (subgrid) {
      if (gridPos.x < subGridStart.x || 
          gridPos.y < subGridStart.y || 
          gridPos.z < subGridStart.z ||
          gridPos.x >= subGridEnd.x || 
          gridPos.y >= subGridEnd.y || 
          gridPos.z >= subGridEnd.z) {
        voxelVerts[i] = numVerts; // no vertices returned
        return;
      }
    } else {
      if (gridPos.x > (gridSize.x - 2) || gridPos.y > (gridSize.y - 2) || gridPos.z > (gridSize.z - 2)) {
        voxelVerts[i] = numVerts; // no vertices returned
        return;
      }
    }

    // read field values at neighbouring grid vertices
    float field[8];
    field[0] = sampleVolume(volume, gridPos, gridSize);
    // TODO early exit test for
    //if( field[0] < 0.000001f )  {
    //    voxelVerts[i] = numVerts;
    //    return;
    //}
    field[1] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
    field[2] = sampleVolume(volume, gridPos + make_uint3(1, 1, 0), gridSize);
    field[3] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
    field[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
    field[5] = sampleVolume(volume, gridPos + make_uint3(1, 0, 1), gridSize);
    field[6] = sampleVolume(volume, gridPos + make_uint3(1, 1, 1), gridSize);
    field[7] = sampleVolume(volume, gridPos + make_uint3(0, 1, 1), gridSize);

    // calculate flag indicating if each vertex is inside or outside isosurface
    unsigned int cubeindex;
    cubeindex =  ((unsigned int) (field[0] < isoValue));
    cubeindex += ((unsigned int) (field[1] < isoValue))*2;
    cubeindex += ((unsigned int) (field[2] < isoValue))*4;
    cubeindex += ((unsigned int) (field[3] < isoValue))*8;
    cubeindex += ((unsigned int) (field[4] < isoValue))*16;
    cubeindex += ((unsigned int) (field[5] < isoValue))*32;
    cubeindex += ((unsigned int) (field[6] < isoValue))*64;
    cubeindex += ((unsigned int) (field[7] < isoValue))*128;

    // read number of vertices from texture
    numVerts.x = tex1Dfetch(numVertsTex, cubeindex);
    numVerts.y = (numVerts.x > 0);

    voxelVerts[i] = numVerts;
}


// compact voxel array
// due to type system changes in CUDA, uint2 had to be replaced by myuint2
__global__ void compactVoxels(unsigned int *compactedVoxelArray, myuint2 *voxelOccupied, unsigned int lastVoxel, unsigned int numVoxels, unsigned int numVoxelsp1) {
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    unsigned int i = (blockId * blockDim.x) + threadIdx.x;

    if ((i < numVoxels) && ( (i < numVoxelsp1) ? voxelOccupied[i].y < voxelOccupied[i+1].y : lastVoxel) ) {
      compactedVoxelArray[ voxelOccupied[i].y ] = i;
    }
}


// version that calculates no surface normal or color,  only triangle vertices
// due to type system changes in CUDA, uint2 had to be replaced by myuint2
__global__ void generateTriangleVerticesSMEM(float3 *pos, unsigned int *compactedVoxelArray, myuint2 *numVertsScanned, float *volume,
                   uint3 gridSize, float3 voxelSize, float isoValue, unsigned int activeVoxels, unsigned int maxVertsM3) {
    unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    unsigned int i = zOffset * ( blockDim.x * blockDim.y) + (blockId * blockDim.x) + threadIdx.x;

    if (i >= activeVoxels )
        return;

    unsigned int voxel = compactedVoxelArray[i];

    // compute position in 3d grid
    uint3 gridPos = calcGridPos(voxel, gridSize);

    float3 p;
    p.x = gridPos.x * voxelSize.x;
    p.y = gridPos.y * voxelSize.y;
    p.z = gridPos.z * voxelSize.z;

    // calculate cell vertex positions
    float3 v[8];
    v[0] = p;
    v[1] = p + make_float3(voxelSize.x, 0, 0);
    v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
    v[3] = p + make_float3(0, voxelSize.y, 0);
    v[4] = p + make_float3(0, 0, voxelSize.z);
    v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
    v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
    v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

    float field[8];
    field[0] = sampleVolume(volume, gridPos, gridSize);
    field[1] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
    field[2] = sampleVolume(volume, gridPos + make_uint3(1, 1, 0), gridSize);
    field[3] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
    field[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
    field[5] = sampleVolume(volume, gridPos + make_uint3(1, 0, 1), gridSize);
    field[6] = sampleVolume(volume, gridPos + make_uint3(1, 1, 1), gridSize);
    field[7] = sampleVolume(volume, gridPos + make_uint3(0, 1, 1), gridSize);

    // recalculate flag
    unsigned int cubeindex;
    cubeindex =  ((unsigned int)(field[0] < isoValue)); 
    cubeindex += ((unsigned int)(field[1] < isoValue))*2; 
    cubeindex += ((unsigned int)(field[2] < isoValue))*4; 
    cubeindex += ((unsigned int)(field[3] < isoValue))*8; 
    cubeindex += ((unsigned int)(field[4] < isoValue))*16; 
    cubeindex += ((unsigned int)(field[5] < isoValue))*32; 
    cubeindex += ((unsigned int)(field[6] < isoValue))*64; 
    cubeindex += ((unsigned int)(field[7] < isoValue))*128;

    // find the vertices where the surface intersects the cube 
    // Note: SIMD marching cubes implementations have no need
    //       for an edge table, because branch divergence eliminates any
    //       potential performance gain from only computing the per-edge
    //       vertices when indicated by the edgeTable.

    // Use shared memory to keep register pressure under control.
    // No need to call __syncthreads() since each thread uses its own
    // private shared memory buffer.
    __shared__ float3 vertlist[12*NTHREADS];

    vertlist[threadIdx.x] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
    vertlist[NTHREADS+threadIdx.x] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
    vertlist[(NTHREADS*2)+threadIdx.x] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
    vertlist[(NTHREADS*3)+threadIdx.x] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
    vertlist[(NTHREADS*4)+threadIdx.x] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
    vertlist[(NTHREADS*5)+threadIdx.x] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
    vertlist[(NTHREADS*6)+threadIdx.x] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
    vertlist[(NTHREADS*7)+threadIdx.x] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
    vertlist[(NTHREADS*8)+threadIdx.x] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
    vertlist[(NTHREADS*9)+threadIdx.x] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
    vertlist[(NTHREADS*10)+threadIdx.x] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
    vertlist[(NTHREADS*11)+threadIdx.x] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);

    // output triangle vertices
    unsigned int numVerts = tex1Dfetch(numVertsTex, cubeindex);
    for(int i=0; i<numVerts; i+=3) {
        unsigned int index = numVertsScanned[voxel].x + i;

        float3 *vert[3];
        int edge;
        edge = tex1Dfetch(triTex, (cubeindex*16) + i);
        vert[0] = &vertlist[(edge*NTHREADS)+threadIdx.x];

        edge = tex1Dfetch(triTex, (cubeindex*16) + i + 1);
        vert[1] = &vertlist[(edge*NTHREADS)+threadIdx.x];

        edge = tex1Dfetch(triTex, (cubeindex*16) + i + 2);
        vert[2] = &vertlist[(edge*NTHREADS)+threadIdx.x];

        if (index < maxVertsM3) {
            pos[index  ] = *vert[0];
            pos[index+1] = *vert[1];
            pos[index+2] = *vert[2];
        }
    }
}


__global__ void offsetTriangleVertices(float3 *pos, float3 origin, unsigned int numVertsM1) {
    unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    unsigned int i = zOffset + (blockId * blockDim.x) + threadIdx.x;

    if (i > numVertsM1)
      return;

    float3 p = pos[i];
    p.x += origin.x;
    p.y += origin.y;
    p.z += origin.z;
    pos[i] = p;
}


// version that calculates the surface normal for each triangle vertex
__global__ void generateTriangleNormals(float3 *pos, float3 *norm, float3 gridSizeInv, float3 bBoxInv, unsigned int numVerts) {
    unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    unsigned int i = zOffset + (blockId * blockDim.x) + threadIdx.x;

    if (i > numVerts - 1)
      return;

    float3 n;
    float3 p, p1, p2;
    // normal calculation using central differences
    // TODO
    //p = ( pos[i] + make_float3( 1.0f)) * 0.5f;
    p = pos[i];
    p.x *= bBoxInv.x;
    p.y *= bBoxInv.y;
    p.z *= bBoxInv.z;
    p1 = p + make_float3( gridSizeInv.x * 1.0f, 0.0f, 0.0f);
    p2 = p - make_float3( gridSizeInv.x * 1.0f, 0.0f, 0.0f);
    n.x = tex3D( volumeTex, p2.x, p2.y, p2.z) - tex3D( volumeTex, p1.x, p1.y, p1.z);
    p1 = p + make_float3( 0.0f, gridSizeInv.y * 1.0f, 0.0f);
    p2 = p - make_float3( 0.0f, gridSizeInv.y * 1.0f, 0.0f);
    n.y = tex3D( volumeTex, p2.x, p2.y, p2.z) - tex3D( volumeTex, p1.x, p1.y, p1.z);
    p1 = p + make_float3( 0.0f, 0.0f, gridSizeInv.z * 1.0f);
    p2 = p - make_float3( 0.0f, 0.0f, gridSizeInv.z * 1.0f);
    n.z = tex3D( volumeTex, p2.x, p2.y, p2.z) - tex3D( volumeTex, p1.x, p1.y, p1.z);

    norm[i] = n;
}


inline __host__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

// version that calculates the surface normal for each triangle vertex
__global__ void generateTriangleNormalsNo3DTex(float3 *pos, float3 *norm, float *volume, uint3 gridSize, float3 voxelSize, unsigned int numVerts) {
    unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    unsigned int i = zOffset + (blockId * blockDim.x) + threadIdx.x;

    if (i > numVerts - 1)
      return;
    
    float3 n;
    float3 p1, p2, p3;
    p1 = pos[3*i+0];
    p2 = pos[3*i+1] - p1;
    p3 = pos[3*i+2] - p1;
    n = cross(p2, p3);
    /*
    float3 p;
    p = pos[i];
    // compute position in 3d grid
    uint3 gridPos = make_uint3(
        (unsigned int)(p.x / voxelSize.x),
        (unsigned int)(p.y / voxelSize.y),
        (unsigned int)(p.z / voxelSize.z));
    
    float field[6];
    field[0] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
    field[1] = sampleVolume(volume, gridPos - make_uint3(1, 0, 0), gridSize);
    field[2] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
    field[3] = sampleVolume(volume, gridPos - make_uint3(0, 1, 0), gridSize);
    field[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
    field[5] = sampleVolume(volume, gridPos - make_uint3(0, 0, 1), gridSize);
    // normal calculation using central differences
    n.x = field[1] - field[0];
    n.y = field[3] - field[2];
    n.z = field[5] - field[4];
    */
    float lenN = length(n);
    n.x = n.x / lenN;
    n.y = n.y / lenN;
    n.z = n.z / lenN;
    
    norm[3*i+0] = n;
    norm[3*i+1] = n;
    norm[3*i+2] = n;
}


// version that calculates the surface normal and color for each triangle vertex
__global__ void generateTriangleColorNormal(float3 *pos, float3 *col, float3 *norm, float3 *colors, uint3 gridSize, float3 gridSizeInv, float3 bBoxInv, unsigned int numVerts) {
    unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    unsigned int i = zOffset + (blockId * blockDim.x) + threadIdx.x;

    if (i > numVerts - 1)
      return;

    float3 p = pos[i];
    p.x *= bBoxInv.x;
    p.y *= bBoxInv.y;
    p.z *= bBoxInv.z;
    // color computation
    float3 gridPosF = p;
    gridPosF.x *= float( gridSize.x);
    gridPosF.y *= float( gridSize.y);
    gridPosF.z *= float( gridSize.z);
    float3 gridPosFloor;
    // Without the offset, rounding errors can occur
    // TODO why do we need the offset??
    gridPosFloor.x = floorf(gridPosF.x + 0.0001f);
    gridPosFloor.y = floorf(gridPosF.y + 0.0001f);
    gridPosFloor.z = floorf(gridPosF.z + 0.0001f);
    float3 gridPosCeil;
    // Without the offset, rounding errors can occur
    // TODO why do we need the offset??
    gridPosCeil.x = ceilf(gridPosF.x - 0.0001f);
    gridPosCeil.y = ceilf(gridPosF.y - 0.0001f);
    gridPosCeil.z = ceilf(gridPosF.z - 0.0001f);
    uint3 gridPos0;
    gridPos0.x = gridPosFloor.x;
    gridPos0.y = gridPosFloor.y;
    gridPos0.z = gridPosFloor.z;
    uint3 gridPos1;
    gridPos1.x = gridPosCeil.x;
    gridPos1.y = gridPosCeil.y;
    gridPos1.z = gridPosCeil.z;

    float3 field[2];
    field[0] = sampleColors(colors, gridPos0, gridSize);
    field[1] = sampleColors(colors, gridPos1, gridSize);

    float3 tmp = gridPosF - gridPosFloor;
    float a = max( max( tmp.x, tmp.y), tmp.z);
    float3 c = lerp( field[0], field[1], a);

    float3 p1, p2;
    // normal calculation using central differences
    float3 n;
    p1 = p + make_float3( gridSizeInv.x * 1.0f, 0.0f, 0.0f);
    p2 = p - make_float3( gridSizeInv.x * 1.0f, 0.0f, 0.0f);
    n.x = tex3D( volumeTex, p2.x, p2.y, p2.z) - tex3D( volumeTex, p1.x, p1.y, p1.z);
    p1 = p + make_float3( 0.0f, gridSizeInv.y * 1.0f, 0.0f);
    p2 = p - make_float3( 0.0f, gridSizeInv.y * 1.0f, 0.0f);
    n.y = tex3D( volumeTex, p2.x, p2.y, p2.z) - tex3D( volumeTex, p1.x, p1.y, p1.z);
    p1 = p + make_float3( 0.0f, 0.0f, gridSizeInv.z * 1.0f);
    p2 = p - make_float3( 0.0f, 0.0f, gridSizeInv.z * 1.0f);
    n.z = tex3D( volumeTex, p2.x, p2.y, p2.z) - tex3D( volumeTex, p1.x, p1.y, p1.z);

    norm[i] = n;
    col[i] = c;
}

// version that calculates the surface normal and color for each triangle vertex
__global__ void generateTriangleColorNormalNo3DTex(float3 *pos, float3 *col, float3 *norm, float *volume, float3 *colors, uint3 gridSize, float3 voxelSize, float3 bBoxInv, unsigned int numVerts) {
    unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    unsigned int i = zOffset + (blockId * blockDim.x) + threadIdx.x;

    if (i > numVerts - 1)
      return;

    float3 p = pos[3*i+0];
    p.x *= bBoxInv.x;
    p.y *= bBoxInv.y;
    p.z *= bBoxInv.z;
    // color computation
    float3 gridPosF = p;
    gridPosF.x *= float( gridSize.x);
    gridPosF.y *= float( gridSize.y);
    gridPosF.z *= float( gridSize.z);
    float3 gridPosFloor;
    // Without the offset, rounding errors can occur
    // TODO why do we need the offset??
    gridPosFloor.x = floorf(gridPosF.x + 0.0001f);
    gridPosFloor.y = floorf(gridPosF.y + 0.0001f);
    gridPosFloor.z = floorf(gridPosF.z + 0.0001f);
    float3 gridPosCeil;
    // Without the offset, rounding errors can occur
    // TODO why do we need the offset??
    gridPosCeil.x = ceilf(gridPosF.x - 0.0001f);
    gridPosCeil.y = ceilf(gridPosF.y - 0.0001f);
    gridPosCeil.z = ceilf(gridPosF.z - 0.0001f);
    uint3 gridPos0;
    gridPos0.x = gridPosFloor.x;
    gridPos0.y = gridPosFloor.y;
    gridPos0.z = gridPosFloor.z;
    uint3 gridPos1;
    gridPos1.x = gridPosCeil.x;
    gridPos1.y = gridPosCeil.y;
    gridPos1.z = gridPosCeil.z;

    float3 field[2];
    field[0] = sampleColors(colors, gridPos0, gridSize);
    field[1] = sampleColors(colors, gridPos1, gridSize);

    float3 tmp = gridPosF - gridPosFloor;
    float a = max( max( tmp.x, tmp.y), tmp.z);
    float3 c = lerp( field[0], field[1], a);
    
    float3 n;
    float3 p1, p2, p3;
    p1 = pos[3*i+0];
    p2 = pos[3*i+1] - p1;
    p3 = pos[3*i+2] - p1;
    n = cross(p2, p3);
    float lenN = length(n);
    n.x = n.x / lenN;
    n.y = n.y / lenN;
    n.z = n.z / lenN;
    
    norm[3*i+0] = n;
    norm[3*i+1] = n;
    norm[3*i+2] = n;
    col[3*i+0] = c;
    col[3*i+1] = c;
    col[3*i+2] = c;
}


void allocateTextures(int **d_triTable, unsigned int **d_numVertsTable) {
    cudaChannelFormatDesc channelDescSigned = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
    cudaMalloc((void**) d_triTable, 256*16*sizeof(int));
    cudaMemcpy((void *)*d_triTable, (void *)triTable, 256*16*sizeof(int), cudaMemcpyHostToDevice);
    cudaBindTexture(0, triTex, *d_triTable, channelDescSigned);

    cudaChannelFormatDesc channelDescUnsigned = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaMalloc((void**) d_numVertsTable, 256*sizeof(unsigned int));
    cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable, 256*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaBindTexture(0, numVertsTex, *d_numVertsTable, channelDescUnsigned);
}


void bindVolumeTexture( cudaArray *d_vol, cudaChannelFormatDesc desc) {
    // set texture parameters
    volumeTex.normalized = 1;
    volumeTex.filterMode = cudaFilterModeLinear;
    //volumeTex.filterMode = cudaFilterModePoint;
    volumeTex.addressMode[0] = cudaAddressModeClamp;
    volumeTex.addressMode[1] = cudaAddressModeClamp;
    volumeTex.addressMode[2] = cudaAddressModeClamp;
    // bind array to 3D texture
    cudaBindTextureToArray( volumeTex, d_vol, desc);
}


#if 0
void ThrustScanWrapper(unsigned int* output, unsigned int* input, unsigned int numElements) {
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input), 
                           thrust::device_ptr<unsigned int>(input + numElements),
                           thrust::device_ptr<unsigned int>(output));
}
#endif

/**
 * This whole method used to have all uint2 instead of myuint2. Changes in the type system of CUDA made this stunt necessary,
 * since the assignment operators of uint2 are not overloaded properly.
 */
void ThrustScanWrapperUint2(myuint2* output, myuint2* input, unsigned int numElements) {
    const myuint2 zero{0, 0};
    thrust::exclusive_scan(thrust::device_ptr<myuint2>(input),
                           thrust::device_ptr<myuint2>(input + numElements),
                           thrust::device_ptr<myuint2>(output),
                           zero);
}


void ThrustScanWrapperArea(float* output, float* input, unsigned int numElements) {
    thrust::inclusive_scan(thrust::device_ptr<float>(input), 
                           thrust::device_ptr<float>(input + numElements),
                           thrust::device_ptr<float>(output));
}


__global__ void computeTriangleAreas( float3 *pos, float *area, unsigned int maxTria) {
    unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    unsigned int i = zOffset + (blockId * blockDim.x) + threadIdx.x;

    // prevent overrunning of array boundary
    if (i >= maxTria)
      return;

    // get all three triangle vertices
    float3 v0 = pos[3*i];
    float3 v1 = pos[3*i+1];
    float3 v2 = pos[3*i+2];

    // compute edge lengths
    float a = length( v0 - v1);
    float b = length( v0 - v2);
    float c = length( v1 - v2);

    // compute area (Heron's formula)
    float rad = ( a + b + c) * ( a + b - c) * ( b + c - a) * ( c + a - b);
    // make sure radicand is not negative
    rad = rad > 0.0f ? rad : 0.0f;
    area[i] = 0.25f * sqrt( rad);
}


///////////////////////////////////////////////////////////////////////////////
//
// class CUDAMarchingCubes
//
///////////////////////////////////////////////////////////////////////////////

CUDAMarchingCubes::CUDAMarchingCubes() {
    // initialize values
    isoValue = 0.5f;
    maxNumVoxels = 0;
    numVoxels    = 0;
    activeVoxels = 0;
    totalVerts   = 0;
    d_volume = 0;
    d_colors = 0;
    d_volumeArray = 0;
    d_voxelVerts = 0;
    d_numVertsTable = 0;
    d_triTable = 0;
    useColor = false;
    useSubGrid = false;
    initialized = false;
    setdata = false;
    cudadevice = 0;
    cudacomputemajor = 0;
    d_areas = 0;
    areaMemSize = 0;
    totalArea = 0.0f;

    // Query GPU device attributes so we can launch the best kernel type
    cudaDeviceProp deviceProp;
    memset(&deviceProp, 0, sizeof(cudaDeviceProp));

    if (cudaGetDevice(&cudadevice) != cudaSuccess) {
      // XXX do something more useful here...
    }

    if (cudaGetDeviceProperties(&deviceProp, cudadevice) != cudaSuccess) {
      // XXX do something more useful here...
    }

    cudacomputemajor = deviceProp.major;
}


CUDAMarchingCubes::~CUDAMarchingCubes() {
    Cleanup();
}


void CUDAMarchingCubes::Cleanup() {
    if( d_triTable ) cudaFree(d_triTable);
    if( d_numVertsTable ) cudaFree(d_numVertsTable);
    if( d_voxelVerts ) cudaFree(d_voxelVerts);
    if( d_compVoxelArray ) cudaFree(d_compVoxelArray);
    if( ownCudaDataArrays ) {
        if( d_volume ) cudaFree(d_volume);
        if( d_colors ) cudaFree(d_colors);
    }
    if( d_volumeArray) cudaFreeArray(d_volumeArray);

    maxNumVoxels = 0;
    numVoxels    = 0;
    d_triTable = 0;
    d_numVertsTable = 0;
    d_voxelVerts = 0;
    d_compVoxelArray = 0;
    d_volume = 0;
    d_colors = 0;
    ownCudaDataArrays = false;
    d_volumeArray = 0;
    initialized = false;
    setdata = false;
}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void CUDAMarchingCubes::computeIsosurface( float3* vertOut, float3* normOut, float3* colOut, unsigned int maxverts) {
    // check if data is available
    if( !this->setdata ) return;

    //cudaEvent_t start, stop;
    //float time;
    //float totalTime = 0.0f;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    int threads = 256;
    //dim3 grid(int(ceil(float(numVoxels) / float( 2 * threads))), 1, 1); // for Mult2
    dim3 grid((unsigned int) (ceil(float(numVoxels) / float(threads))), 1, 1);
    // get around maximum grid size of 65535 in each dimension
    if (grid.x > 65535) {
        grid.y = (unsigned int) (ceil(float( grid.x) / 32768.0f));
        grid.x = 32768;
    }

    dim3 threads3D( 256, 1, 1);
    dim3 grid3D((unsigned int) (ceil(float(gridSize.x) / float(threads3D.x))), 
                (unsigned int) (ceil(float(gridSize.y) / float(threads3D.y))), 
                (unsigned int) (ceil(float(gridSize.z) / float(threads3D.z))));

    //cudaEventRecord( start, 0);

    cudaThreadSynchronize();

    // calculate number of vertices need per voxel
    if (cudacomputemajor >= 2) {
      // launch a 3-D grid if we have a new enough device (Fermi or later...)
      if (useSubGrid) {
        classifyVoxel<1,1><<<grid3D, threads3D>>>(d_voxelVerts, d_volume, 
                             gridSize, numVoxels, voxelSize, 
                             subGridStart, subGridEnd, isoValue);
      } else {
        classifyVoxel<1,0><<<grid3D, threads3D>>>(d_voxelVerts, d_volume, 
                             gridSize, numVoxels, voxelSize, 
                             subGridStart, subGridEnd, isoValue);
      }
    } else {
      // launch a 2-D grid for older devices
      if (useSubGrid) {
        classifyVoxel<0,1><<<grid, threads>>>(d_voxelVerts, d_volume, 
                             gridSize, numVoxels, voxelSize,
                             subGridStart, subGridEnd, isoValue);
      } else {
        classifyVoxel<0,0><<<grid, threads>>>(d_voxelVerts, d_volume, 
                             gridSize, numVoxels, voxelSize,
                             subGridStart, subGridEnd, isoValue);
      }
    }

    // check for errors
    cudaThreadSynchronize();
    //printf( "CUDA-MC: launch_classifyVoxel: %s\n", cudaGetErrorString( cudaGetLastError()));
    //cudaEventRecord( stop, 0);
    //cudaEventSynchronize( stop);
    //cudaEventElapsedTime(&time, start, stop);
    //printf ("-----------------------------------------------------\n");
    //printf ("Time for the launch_classifyVoxel kernel: %f ms\n", time);
    //totalTime += time;

    //cudaEventRecord( start, 0);
    // scan voxel vertex/occupation array (use in-place prefix sum for lower memory consumption)
    uint2 lastElement, lastScanElement;
    cudaMemcpy((void *) &lastElement, (void *)(d_voxelVerts + numVoxels-1), sizeof(uint2), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    ThrustScanWrapperUint2(d_voxelVerts, d_voxelVerts, numVoxels);
    // check for errors
    cudaThreadSynchronize();
    //printf( "CUDA-MC: ThrustScanWrapper I: %s\n", cudaGetErrorString( cudaGetLastError()));
    //cudaEventRecord( stop, 0);
    //cudaEventSynchronize( stop);
    //cudaEventElapsedTime(&time, start, stop);
    //printf ("Time for the ThrustScanWrapper kernel: %f ms\n", time);
    //totalTime += time;

    // read back values to calculate total number of non-empty voxels
    // since we are using an exclusive scan, the total is the last value of
    // the scan result plus the last value in the input array
    cudaMemcpy((void *) &lastScanElement, (void *) (d_voxelVerts + numVoxels-1), sizeof(uint2), cudaMemcpyDeviceToHost);
    activeVoxels = lastElement.y + lastScanElement.y;
    // add up total number of vertices
    totalVerts = lastElement.x + lastScanElement.x;
    totalVerts = totalVerts < maxverts ? totalVerts : maxverts; // min

    if (activeVoxels==0) {
        // return if there are no full voxels
        totalVerts = 0;
        return;
    }

    grid.x = (unsigned int) (ceil(float(numVoxels) / float(threads)));
    grid.y = grid.z = 1;
    // get around maximum grid size of 65535 in each dimension
    if (grid.x > 65535) {
        grid.y = (unsigned int) (ceil(float(grid.x) / 32768.0f));
        grid.x = 32768;
    }

    //cudaEventRecord( start, 0);
    // compact voxel index array
    compactVoxels<<<grid, threads>>>(d_compVoxelArray, d_voxelVerts, lastElement.y, numVoxels, numVoxels + 1);
    // check for errors
    //cudaThreadSynchronize();
    //printf( "CUDA-MC: launch_compactVoxels: %s\n", cudaGetErrorString( cudaGetLastError()));
    //cudaEventRecord( stop, 0);
    //cudaEventSynchronize( stop);
    //cudaEventElapsedTime(&time, start, stop);
    //printf ("Time for the launch_compactVoxels kernel: %f ms\n", time);
    //totalTime += time;


    dim3 grid2((unsigned int) (ceil(float(activeVoxels) / (float) NTHREADS)), 1, 1);
    while(grid2.x > 65535) {
        grid2.x = (unsigned int) (ceil(float(grid2.x) / 2.0f));
        grid2.y *= 2;
    }

    dim3 grid3((unsigned int) (ceil(float(totalVerts) / (float) threads)), 1, 1);
    while(grid3.x > 65535) {
        grid3.x = (unsigned int) (ceil(float(grid3.x) / 2.0f));
        grid3.y *= 2;
    }
    while(grid3.y > 65535) {
        grid3.y = (unsigned int) (ceil(float(grid3.y) / 2.0f));
        grid3.z *= 2;
    }

    //cudaEventRecord( start, 0);

    // separate computation of vertices and vertex color/normal for higher occupancy and speed
    generateTriangleVerticesSMEM<<<grid2, NTHREADS>>>( vertOut, 
        d_compVoxelArray, d_voxelVerts, d_volume, gridSize, voxelSize, 
        isoValue, activeVoxels, maxverts - 3);
    // check for errors
    cudaThreadSynchronize();
    //printf( "CUDA-MC: launch_generateTriangleVertices: %s\n", cudaGetErrorString( cudaGetLastError()));

    float3 gridSizeInv = make_float3( 1.0f / float( gridSize.x), 1.0f / float( gridSize.y), 1.0f / float( gridSize.z));
    float3 bBoxInv = make_float3( 1.0f / bBox.x, 1.0f / bBox.y, 1.0f / bBox.z);
#ifdef USE_CUDA_3D_TEXTURE
    if( this->useColor ) {
        generateTriangleColorNormal<<<grid3, threads>>>(vertOut, colOut, normOut, (float3*)d_colors, this->gridSize, gridSizeInv, bBoxInv, totalVerts);
    } else {
        generateTriangleNormals<<<grid3, threads>>>(vertOut, normOut, gridSizeInv, bBoxInv, totalVerts);
    }
#else
    if( this->useColor ) {
        //generateTriangleColorNormal<<<grid3, threads>>>(vertOut, colOut, normOut, (float3*)d_colors, this->gridSize, gridSizeInv, bBoxInv, totalVerts);
        generateTriangleColorNormalNo3DTex<<<grid3, threads>>>(vertOut, colOut, normOut, d_volume, (float3*)d_colors, gridSize, voxelSize, bBoxInv, totalVerts/3);
    } else {
        generateTriangleNormalsNo3DTex<<<grid3, threads>>>(vertOut, normOut, d_volume, gridSize, voxelSize, totalVerts/3);
    }
#endif // USE_CUDA_3D_TEXTURE

    // check for errors
    cudaThreadSynchronize();
    //printf( "CUDA-MC: launch_generateTriangleColorNormal: %s\n", cudaGetErrorString( cudaGetLastError()));

    offsetTriangleVertices<<<grid3, threads>>>(vertOut, this->origin, totalVerts - 1);

    // check for errors
    cudaThreadSynchronize();
    //printf( "CUDA-MC: launch_offsetTriangleVertices: %s\n", cudaGetErrorString( cudaGetLastError()));

    //cudaEventRecord( stop, 0);
    //cudaEventSynchronize( stop);
    //cudaEventElapsedTime(&time, start, stop);
    //printf ("Time for the generateTriangles kernel: %f ms\n", time);
    //totalTime += time;
    
    //printf (" CUDA MC: Total kernel runtime time: %f ms = %.3f fps\n", totalTime, 1000.0f/totalTime);
    // check for errors
    //cudaThreadSynchronize();
    //printf( "compute Marching Cubes: %s\n", cudaGetErrorString( cudaGetLastError()));
}


bool CUDAMarchingCubes::Initialize(uint3 maxgridsize) {
    // check if already initialized
    if (initialized) return false;

    // use max grid size initially
    maxGridSize = maxgridsize;
    gridSize = maxGridSize;
    maxNumVoxels = gridSize.x*gridSize.y*gridSize.z;
    numVoxels = maxNumVoxels;
    
    // initialize subgrid dimensions to the entire volume by default
    subGridStart = make_uint3(0, 0, 0);
    subGridEnd = gridSize - make_uint3(1, 1, 1);

    // allocate textures
    allocateTextures(&d_triTable, &d_numVertsTable);

    // allocate device memory
    if (cudaMalloc((void**) &d_voxelVerts, sizeof(uint2) * numVoxels) != cudaSuccess) {
        Cleanup();
        return false;
    }
    if (cudaMalloc((void**) &d_compVoxelArray, sizeof(unsigned int) * numVoxels) != cudaSuccess) {
        Cleanup();
        return false;
    }

    // success
    initialized = true;
    return true;
}


bool CUDAMarchingCubes::SetVolumeData(float *volume, float *colors, uint3 gridsize, 
                                      float3 gridOrigin, float3 boundingBox, bool cudaArray) {
  bool newgridsize = false;

  // check if initialize was successful
  if (!initialized) return false;

#if 0
  // return if volume data was already set
  if (setdata) return false;
  if (d_volume != NULL) return false;
#endif

  // check if the grid size matches
  if (gridsize.x != gridSize.x ||
      gridsize.y != gridSize.y ||
      gridsize.z != gridSize.z) {
    newgridsize = true;
    int nv = gridsize.x*gridsize.y*gridsize.z;
    if (nv > maxNumVoxels)
      return false;

    gridSize = gridsize;
    numVoxels = nv;

    // initialize subgrid dimensions to the entire volume by default
    subGridStart = make_uint3(0, 0, 0);
    subGridEnd = gridSize - make_uint3(1, 1, 1);
  }

  // copy the grid origin, bounding box dimensions, 
  // and update dependent variables
  origin = gridOrigin;
  bBox = boundingBox;
  voxelSize = make_float3(bBox.x / gridSize.x,
                          bBox.y / gridSize.y,
                          bBox.z / gridSize.z);

  // check colors
  useColor = colors ? true : false;

  // copy cuda array pointers or create cuda arrays and copy data
  if (cudaArray) {
    // check ownership flag and free if necessary
    if (ownCudaDataArrays) {
      if (d_volume) cudaFree(d_volume);
      d_volume = NULL;

      if (d_colors) cudaFree(d_colors);
      d_colors = NULL;
    }

    // copy data pointers
    d_volume = volume;
    d_colors = colors;

    // set ownership flag
    ownCudaDataArrays = false;
  } else {
    // create the volume array (using max size) and copy data
    unsigned int size = numVoxels * sizeof(float);
    unsigned int maxsize = maxNumVoxels * sizeof(float);

    // check data array allocation
    if (!d_volume) {
      if (cudaMalloc((void**) &d_volume, maxsize) != cudaSuccess) {
        Cleanup();
        return false;
      }
      if (cudaMemcpy(d_volume, volume, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        Cleanup();
        return false;
      }
    }

    if (colors) {
      if (!d_colors) {
        // create the color array and copy colors
        if (cudaMalloc((void**) &d_colors, maxsize*3) != cudaSuccess) {
          Cleanup();
          return false;
        }
        if (cudaMemcpy(d_colors, colors, size*3, cudaMemcpyHostToDevice) != cudaSuccess ) {
          Cleanup();
          return false;
        }
      }
    }

    // set ownership flag
    ownCudaDataArrays = true;
  }

  // Compute grid extents and channel description for the 3-D array
  cudaExtent gridExtent = make_cudaExtent(gridSize.x, gridSize.y, gridSize.z);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  // Check to see if existing 3D array allocation matches current grid size,
  // deallocate it if not so that we regenerate it with the correct size.
  if (d_volumeArray && newgridsize) { 
    cudaFreeArray(d_volumeArray);
    d_volumeArray = NULL;
  }

#ifdef USE_CUDA_3D_TEXTURE
  // allocate the 3D array if needed
  if (!d_volumeArray) { 
    // create 3D array
    if (cudaMalloc3DArray(&d_volumeArray, &channelDesc, gridExtent) != cudaSuccess) {
      Cleanup();
      return false;
    }
  }

  // copy data to 3D array
  cudaMemcpy3DParms copyParams = {0};
  //copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_volume, 0, 0, 0);
  copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_volume, sizeof(float)*gridSize.x, gridSize.x, gridSize.y);
  copyParams.dstArray = d_volumeArray;
  copyParams.extent   = gridExtent;
  copyParams.kind     = cudaMemcpyDeviceToDevice;
  if (cudaMemcpy3D(&copyParams) != cudaSuccess) {
    Cleanup();
    return false;
  }

  // bind the array to a volume texture
  bindVolumeTexture(d_volumeArray, channelDesc);
#endif

  // success
  setdata = true;

  return true;
}


void CUDAMarchingCubes::SetSubVolume(uint3 start, uint3 end) {
  subGridStart = start;
  subGridEnd = end;
  useSubGrid = true;

  if (subGridEnd.x >= gridSize.x)
    subGridEnd.x = gridSize.x - 1;
  if (subGridEnd.y >= gridSize.y)
    subGridEnd.y = gridSize.y - 1;
  if (subGridEnd.z >= gridSize.z)
    subGridEnd.z = gridSize.z - 1;
}


bool CUDAMarchingCubes::computeIsosurface(float *volume, float *colors, 
                                          uint3 gridsize, float3 gridOrigin, 
                                          float3 boundingBox, bool cudaArray, 
                                          float3* vertOut, float3* normOut, 
                                          float3* colOut, unsigned int maxverts) {
    // Setup
    if (!Initialize(gridsize)) {
        return false;
    }

    if (!SetVolumeData(volume, colors, gridsize, gridOrigin, boundingBox, cudaArray)) {
        return false;
    }

    // Compute and Render Isosurface
    computeIsosurface(vertOut, normOut, colOut, maxverts);

    // Tear down and free resources
    Cleanup();

    return true;
}


float CUDAMarchingCubes::computeSurfaceArea( float3 *verts, unsigned int triaCount) {
    // do nothing for zero triangles
    if(triaCount <= 0) return 0.0f;

    // initialize and allocate device arrays
    size_t memSize = sizeof(float) * triaCount;
    if (memSize > this->areaMemSize ) {
        if (this->areaMemSize > 0) {
            // clean up
            cudaFree(this->d_areas);
            // check for errors
            cudaThreadSynchronize();
        }
        //printf( "cudaFree: %s\n", cudaGetErrorString( cudaGetLastError()));
        if (cudaMalloc((void**) &this->d_areas, memSize) != cudaSuccess) {
            return -1.0f;
        }
        this->areaMemSize = memSize;
        // check for errors
        cudaThreadSynchronize();
        //printf( "cudaMalloc: %s\n", cudaGetErrorString( cudaGetLastError()));
    }

    // compute area for each triangle
    int threads = 256;
    dim3 grid(int(ceil(float(triaCount) / float(threads))), 1, 1);

    // get around maximum grid size of 65535 in each dimension
    while(grid.x > 65535) {
        grid.x = (unsigned int) (ceil(float(grid.x) / 2.0f));
        grid.y *= 2;
    }
    while(grid.y > 65535) {
        grid.y = (unsigned int) (ceil(float(grid.y) / 2.0f));
        grid.z *= 2;
    }

    computeTriangleAreas<<<grid, threads>>>(verts, this->d_areas, triaCount);
    // check for errors
    cudaThreadSynchronize();
    //printf( "launch_computeTriangleAreas: %s\n", cudaGetErrorString( cudaGetLastError()));

    // use prefix sum to compute total surface area
    ThrustScanWrapperArea(this->d_areas, this->d_areas, triaCount);
    // check for errors
    cudaThreadSynchronize();
    //printf( "ThrustScanWrapperArea: %s\n", cudaGetErrorString( cudaGetLastError()));

    // readback total surface area
    cudaMemcpy((void *)&this->totalArea, (void *)(this->d_areas + triaCount - 1), sizeof(float), cudaMemcpyDeviceToHost);
    // check for errors
    cudaThreadSynchronize();
    //printf( "cudaMemcpy: %s\n", cudaGetErrorString( cudaGetLastError()));

    // return result
    return totalArea;
}



