// includes for windows!
#ifdef _WIN32
#include <windows.h>
#endif
#include "vislib/graphics/gl/IncludeAllGL.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <algorithm>
#define WGL_NV_gpu_affinity
#include <cuda_gl_interop.h>

#if CUDART_VERSION < 4000
#error The VMD QuickSurf feature requires CUDA 4.0 or later
#endif

#include "utilities.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#include "CUDAKernels.h" 
#include "CUDAMarchingCubes.h"
#include "CUDAQuickSurfArray.h"
#include <thrust/device_ptr.h>
//
// multi-threaded direct summation implementation
//

surface<void, 3> outputSurface;

inline __host__ __device__ float dot(float3 a, float3 b) { 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float length(float3 v){
    return sqrtf(dot(v, v));
}

#if 1
#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  printf("Thread aborting...\n"); \
  return NULL; }}
#else
#define CUERR
#endif

__global__ static void setArrayToIntQSA( int count, int* array, int value) {
    // get global index
    const unsigned int idx = __umul24(__umul24(blockIdx.y, gridDim.x) + blockIdx.x, blockDim.x) + threadIdx.x;
    // early exit if this thread is outside of the grid bounds
    if (idx >= count)
        return;
    array[idx] = value;
}

//
// Linear-time density kernels that use spatial hashing of atoms 
// into a uniform grid of atom bins to reduce the number of 
// density computations by truncating the gaussian to a given radius
// and only considering bins of atoms that fall within that radius.
//
#include <thrust/sort.h> // need thrust sorting primitives

#define GRID_CELL_EMPTY 0xffffffff

// calculate cell address as the hash value for each atom
__global__ void hashAtomsQSA(unsigned int natoms,
                          const float4 *xyzr,
                          int3 numvoxels,
                          float invgridspacing,
                          unsigned int *atomIndex,
                          unsigned int *atomHash) {
  unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= natoms)
    return;

  float4 atom = xyzr[index];

  int3 cell;
  cell.x = atom.x * invgridspacing;
  cell.y = atom.y * invgridspacing;
  cell.z = atom.z * invgridspacing;

  cell.x = min(cell.x, numvoxels.x-1);
  cell.y = min(cell.y, numvoxels.y-1);
  cell.z = min(cell.z, numvoxels.z-1);

  unsigned int hash = (cell.z * numvoxels.y * numvoxels.x) + 
                      (cell.y * numvoxels.x) + cell.x;

  atomIndex[index] = index; // original atom index
  atomHash[index] = hash;   // atoms hashed to voxel address
}


// build cell lists and reorder atoms and colors using sorted atom index list
__global__ void sortAtomsGenCellListsQSA(unsigned int natoms,
                           const float4 *xyzr_d,
                           const float4 *color_d,
                           const unsigned int *atomIndex_d,
                           unsigned int *sorted_atomIndex_d,
                           const unsigned int *atomHash_d,
                           float4 *sorted_xyzr_d,
                           float4 *sorted_color_d,
                           uint2 *cellStartEnd_d) {
  extern __shared__ unsigned int hash_s[]; // blockSize + 1 elements
  unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int hash;

  // do nothing if current index exceeds the number of atoms
  if (index < natoms) {
    hash = atomHash_d[index];
    hash_s[threadIdx.x+1] = hash; // use smem to avoid redundant loads
    if (index > 0 && threadIdx.x == 0) {
      // first thread in block must load neighbor particle hash
      hash_s[0] = atomHash_d[index-1];
    }
  }
  __syncthreads();

  if (index < natoms) {
    // Since atoms are sorted, if this atom has a different cell
    // than its predecessor, it is the first atom in its cell, and 
    // it's index marks the end of the previous cell.
    if (index == 0 || hash != hash_s[threadIdx.x]) {
      cellStartEnd_d[hash].x = index; // set start
      if (index > 0)
        cellStartEnd_d[hash_s[threadIdx.x]].y = index; // set end
    }
	
    if (index == natoms - 1) {
      cellStartEnd_d[hash].y = index + 1; // set end
    }
    // Reorder atoms according to sorted indices
    unsigned int sortedIndex = atomIndex_d[index];
    sorted_atomIndex_d[sortedIndex] = index;
    float4 pos = xyzr_d[sortedIndex];
    sorted_xyzr_d[index] = pos;
 
    // Reorder colors according to sorted indices, if provided
    if (color_d != NULL) {
      float4 col = color_d[sortedIndex];
      sorted_color_d[index] = col;
    }
  }
}


int vmd_cuda_build_density_atom_gridQSA(int natoms,
                                     const float4 * xyzr_d,
                                     const float4 * color_d,
                                     float4 * sorted_xyzr_d,
                                     float4 * sorted_color_d,
                                     unsigned int *atomIndex_d,
                                     unsigned int *sorted_atomIndex_d,
                                     unsigned int *atomHash_d,
                                     uint2 * cellStartEnd_d,
                                     int3 volsz,
                                     float invgridspacing) {

  // Compute block and grid sizes to use for various kernels
  dim3 hBsz(256, 1, 1);
  cudaError_t err;

  err = cudaGetLastError();
  if (err != cudaSuccess) {
	  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
	  return -1;
  }
  // If we have a very large atom count, we must either use 
  // larger thread blocks, or use multi-dimensional grids of thread blocks. 
  // We can use up to 65535 blocks in a 1-D grid, so we can use
  // 256-thread blocks for less than 16776960 atoms, and use 512-thread
  // blocks for up to 33553920 atoms.  Beyond that, we have to use 2-D grids
  // and modified kernels.
  if (natoms > 16000000)
    hBsz.x = 512; // this will get us 

  dim3 hGsz(((natoms+hBsz.x-1) / hBsz.x), 1, 1);

  // Compute grid cell address as atom hash
  // XXX need to use 2-D indexing for large atom counts or we exceed the
  //     per-dimension 65535 block grid size limitation
  hashAtomsQSA<<<hGsz, hBsz>>>(natoms, xyzr_d, volsz, invgridspacing,
                            atomIndex_d, atomHash_d);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
	  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
	  return -1;
  }
  // Sort atom indices by their grid cell address
  // (wrapping the device pointers with vector iterators)
  thrust::sort_by_key(thrust::device_ptr<unsigned int>(atomHash_d),
                      thrust::device_ptr<unsigned int>(atomHash_d + natoms),
                      thrust::device_ptr<unsigned int>(atomIndex_d));

  err = cudaGetLastError();
  if (err != cudaSuccess) {
	  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
	  return -1;
  }
  // Initialize all cells to empty
  int ncells = volsz.x * volsz.y * volsz.z;
  cudaMemset(cellStartEnd_d, GRID_CELL_EMPTY, ncells*sizeof(uint2));

  err = cudaGetLastError();
  if (err != cudaSuccess) {
	  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
	  return -1;
  }
  // Reorder atoms into sorted order and find start and end of each cell
  // XXX need to use 2-D indexing for large atom counts or we exceed the
  //     per-dimension 65535 block grid size limitation
  unsigned int smemSize = sizeof(unsigned int)*(hBsz.x+1);
  sortAtomsGenCellListsQSA<<<hGsz, hBsz, smemSize>>>(
                       natoms, xyzr_d, color_d, atomIndex_d, sorted_atomIndex_d, 
                       atomHash_d, sorted_xyzr_d, sorted_color_d, cellStartEnd_d);

#if 1
  // XXX when the code is ready for production use, we can disable
  //     detailed error checking and use a more all-or-nothing approach
  //     where errors fall through all of the CUDA API calls until the
  //     end and we do the cleanup only at the end.
  cudaThreadSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    return -1;
  }
#endif

  return 0;
}


// 
// Parameters for linear-time range-limited gaussian density kernels
//
#define GGRIDSZ   8.0f
#define GBLOCKSZX 8
#define GBLOCKSZY 8

#define GTEXBLOCKSZZ 2
#define GTEXUNROLL   4
#define GBLOCKSZZ    2
#define GUNROLL      4

__global__ static void gaussdensity_fast_texQSA(int natoms,
                                         const float4 *sorted_xyzr, 
                                         const float4 *sorted_color, 
                                         int3 numvoxels,
                                         int3 acncells,
                                         float acgridspacing,
                                         float invacgridspacing,
                                         const uint2 * cellStartEnd,
                                         float gridspacing, unsigned int z, 
                                         cudaSurfaceObject_t densitygrid,
                                         float3 *voltexmap,
                                         float invisovalue,
                                         bool storeNearestNeighbor,
                                         int* nearestNeighbor,
                                         unsigned int* atomIndex) {
  unsigned int xindex  = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int zindex  = ((blockIdx.z * blockDim.z) + threadIdx.z) * GTEXUNROLL;

  // shave register use slightly
  unsigned int outaddr = zindex * numvoxels.x * numvoxels.y + 
                         yindex * numvoxels.x + xindex;

  // early exit if this thread is outside of the grid bounds
  if (xindex >= numvoxels.x || yindex >= numvoxels.y || zindex >= numvoxels.z)
    return;

  zindex += z;

  // compute ac grid index of lower corner minus gaussian radius
  int xabmin = ((blockIdx.x * blockDim.x) * gridspacing - acgridspacing) * invacgridspacing;
  int yabmin = ((blockIdx.y * blockDim.y) * gridspacing - acgridspacing) * invacgridspacing;
  int zabmin = ((z + blockIdx.z * blockDim.z * GTEXUNROLL) * gridspacing - acgridspacing) * invacgridspacing;

  // compute ac grid index of upper corner plus gaussian radius
  int xabmax = (((blockIdx.x+1) * blockDim.x) * gridspacing + acgridspacing) * invacgridspacing;
  int yabmax = (((blockIdx.y+1) * blockDim.y) * gridspacing + acgridspacing) * invacgridspacing;
  int zabmax = ((z + (blockIdx.z+1) * blockDim.z * GTEXUNROLL) * gridspacing + acgridspacing) * invacgridspacing;

  xabmin = (xabmin < 0) ? 0 : xabmin;
  yabmin = (yabmin < 0) ? 0 : yabmin;
  zabmin = (zabmin < 0) ? 0 : zabmin;
  xabmax = (xabmax >= acncells.x-1) ? acncells.x-1 : xabmax;
  yabmax = (yabmax >= acncells.y-1) ? acncells.y-1 : yabmax;
  zabmax = (zabmax >= acncells.z-1) ? acncells.z-1 : zabmax;

  float coorx = gridspacing * xindex;
  float coory = gridspacing * yindex;
  float coorz = gridspacing * zindex;

  float densityval1=0.0f;
  float3 densitycol1=make_float3(0.0f, 0.0f, 0.0f);
#if GTEXUNROLL >= 2
  float densityval2=0.0f;
  float3 densitycol2=densitycol1;
#endif
#if GTEXUNROLL >= 4
  float densityval3=0.0f;
  float3 densitycol3=densitycol1;
  float densityval4=0.0f;
  float3 densitycol4=densitycol1;
#endif

  // the minimum distance to the next atom
  float minDist1 = 1000000.0f;
#if GTEXUNROLL >= 2
  float minDist2 = minDist1;
#endif
#if GTEXUNROLL >= 4
  float minDist3 = minDist1;
  float minDist4 = minDist1;
#endif
  // the index of the next atom
  int neighbor1 = -1;
#if GTEXUNROLL >= 2
  int neighbor2 = -1;
#endif
#if GTEXUNROLL >= 4
  int neighbor3 = -1;
  int neighbor4 = -1;
#endif

  int acplanesz = acncells.x * acncells.y;
  int xab, yab, zab;
  for (zab=zabmin; zab<=zabmax; zab++) {
    for (yab=yabmin; yab<=yabmax; yab++) {
      for (xab=xabmin; xab<=xabmax; xab++) {
        int abcellidx = zab * acplanesz + yab * acncells.x + xab;
        uint2 atomstartend = cellStartEnd[abcellidx];
        if (atomstartend.x != GRID_CELL_EMPTY) {
          unsigned int atomid;
          for (atomid=atomstartend.x; atomid<atomstartend.y; atomid++) {
            float4 atom  = sorted_xyzr[atomid];
            float4 color = sorted_color[atomid];
            float dx = coorx - atom.x;
            float dy = coory - atom.y;
            float dxy2 = dx*dx + dy*dy;
            float dz = coorz - atom.z;
            float r21 = (dxy2 + dz*dz) * atom.w;
            float tmp1 = exp2f(r21);
            densityval1 += tmp1;
            tmp1 *= invisovalue;
            densitycol1.x += tmp1 * color.x;
            densitycol1.y += tmp1 * color.y;
            densitycol1.z += tmp1 * color.z;
            // store nearest neighbor
            if( (dxy2 + dz*dz) < minDist1 ) {
                minDist1 = (dxy2 + dz*dz);
                neighbor1 = atomid;
            }

#if GTEXUNROLL >= 2
            float dz2 = dz + gridspacing;
            float r22 = (dxy2 + dz2*dz2) * atom.w;
            float tmp2 = exp2f(r22);
            densityval2 += tmp2;
            tmp2 *= invisovalue;
            densitycol2.x += tmp2 * color.x;
            densitycol2.y += tmp2 * color.y;
            densitycol2.z += tmp2 * color.z;
            // store nearest neighbor
            if( (dxy2 + dz2*dz2) < minDist2 ) {
                minDist2 = (dxy2 + dz2*dz2);
                neighbor2 = atomid;
            }
#endif
#if GTEXUNROLL >= 4
            float dz3 = dz2 + gridspacing;
            float r23 = (dxy2 + dz3*dz3) * atom.w;
            float tmp3 = exp2f(r23);
            densityval3 += tmp3;
            tmp3 *= invisovalue;
            densitycol3.x += tmp3 * color.x;
            densitycol3.y += tmp3 * color.y;
            densitycol3.z += tmp3 * color.z;
            // store nearest neighbor
            if( (dxy2 + dz3*dz3) < minDist3 ) {
                minDist3 = (dxy2 + dz3*dz3);
                neighbor3 = atomid;
            }

            float dz4 = dz3 + gridspacing;
            float r24 = (dxy2 + dz4*dz4) * atom.w;
            float tmp4 = exp2f(r24);
            densityval4 += tmp4;
            tmp4 *= invisovalue;
            densitycol4.x += tmp4 * color.x;
            densitycol4.y += tmp4 * color.y;
            densitycol4.z += tmp4 * color.z;
            // store nearest neighbor
            if( (dxy2 + dz4*dz4) < minDist4 ) {
                minDist4 = (dxy2 + dz4*dz4);
                neighbor4 = atomid;
            }
#endif
          }
        }
      }
    }
  }

  //densitygrid[outaddr          ] = densityval1;
  surf3Dwrite(densityval1, densitygrid, xindex * sizeof(float), yindex, zindex);
  voltexmap[outaddr          ].x = densitycol1.x;
  voltexmap[outaddr          ].y = densitycol1.y;
  voltexmap[outaddr          ].z = densitycol1.z;

#if GTEXUNROLL >= 2
  int planesz = numvoxels.x * numvoxels.y;
  //densitygrid[outaddr + planesz] = densityval2;
  surf3Dwrite(densityval2, densitygrid, xindex * sizeof(float), yindex, zindex + 1);
  voltexmap[outaddr + planesz].x = densitycol2.x;
  voltexmap[outaddr + planesz].y = densitycol2.y;
  voltexmap[outaddr + planesz].z = densitycol2.z;
#endif
#if GTEXUNROLL >= 4
  //densitygrid[outaddr + 2*planesz] = densityval3;
  surf3Dwrite(densityval3, densitygrid, xindex * sizeof(float), yindex, zindex + 2);
  voltexmap[outaddr + 2*planesz].x = densitycol3.x;
  voltexmap[outaddr + 2*planesz].y = densitycol3.y;
  voltexmap[outaddr + 2*planesz].z = densitycol3.z;

  //densitygrid[outaddr + 3*planesz] = densityval4;
  surf3Dwrite(densityval4, densitygrid, xindex * sizeof(float), yindex, zindex + 3);
  voltexmap[outaddr + 3*planesz].x = densitycol4.x;
  voltexmap[outaddr + 3*planesz].y = densitycol4.y;
  voltexmap[outaddr + 3*planesz].z = densitycol4.z;
#endif
  
  if( storeNearestNeighbor ) {
    nearestNeighbor[outaddr          ] = atomIndex[neighbor1];
#if GUNROLL >= 2
    int planesz = numvoxels.x * numvoxels.y;
    nearestNeighbor[outaddr + planesz] = atomIndex[neighbor2];
#endif
#if GUNROLL >= 4
    nearestNeighbor[outaddr + 2*planesz] = atomIndex[neighbor3];
    nearestNeighbor[outaddr + 3*planesz] = atomIndex[neighbor4];
#endif
  }
}

__global__ static void gaussdensity_fastQSA(int natoms,
                                         const float4 *sorted_xyzr, 
                                         int3 numvoxels,
                                         int3 acncells,
                                         float acgridspacing,
                                         float invacgridspacing,
                                         const uint2 * cellStartEnd,
                                         float gridspacing, unsigned int z, 
                                         cudaSurfaceObject_t densitygrid,
                                         bool storeNearestNeighbor,
                                         int* nearestNeighbor,
                                         unsigned int* atomIndex) {
  unsigned int xindex  = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int zindex  = ((blockIdx.z * blockDim.z) + threadIdx.z) * GUNROLL;
  unsigned int outaddr = zindex * numvoxels.x * numvoxels.y + 
                         yindex * numvoxels.x + 
                         xindex;

  // early exit if this thread is outside of the grid bounds
  if (xindex >= numvoxels.x || yindex >= numvoxels.y || zindex >= numvoxels.z)
    return;

  zindex += z;

  // compute ac grid index of lower corner minus gaussian radius
  int xabmin = ((blockIdx.x * blockDim.x) * gridspacing - acgridspacing) * invacgridspacing;
  int yabmin = ((blockIdx.y * blockDim.y) * gridspacing - acgridspacing) * invacgridspacing;
  int zabmin = ((z + blockIdx.z * blockDim.z * GUNROLL) * gridspacing - acgridspacing) * invacgridspacing;

  // compute ac grid index of upper corner plus gaussian radius
  int xabmax = (((blockIdx.x+1) * blockDim.x) * gridspacing + acgridspacing) * invacgridspacing;
  int yabmax = (((blockIdx.y+1) * blockDim.y) * gridspacing + acgridspacing) * invacgridspacing;
  int zabmax = ((z + (blockIdx.z+1) * blockDim.z * GUNROLL) * gridspacing + acgridspacing) * invacgridspacing;

  xabmin = (xabmin < 0) ? 0 : xabmin;
  yabmin = (yabmin < 0) ? 0 : yabmin;
  zabmin = (zabmin < 0) ? 0 : zabmin;
  xabmax = (xabmax >= acncells.x-1) ? acncells.x-1 : xabmax;
  yabmax = (yabmax >= acncells.y-1) ? acncells.y-1 : yabmax;
  zabmax = (zabmax >= acncells.z-1) ? acncells.z-1 : zabmax;

  float coorx = gridspacing * xindex;
  float coory = gridspacing * yindex;
  float coorz = gridspacing * zindex;

  float densityval1=0.0f;
#if GUNROLL >= 2
  float densityval2=0.0f;
#endif
#if GUNROLL >= 4
  float densityval3=0.0f;
  float densityval4=0.0f;
#endif
  
  // the minimum distance to the next atom
  float minDist1 = 1000000.0f;
#if GTEXUNROLL >= 2
  float minDist2 = minDist1;
#endif
#if GTEXUNROLL >= 4
  float minDist3 = minDist1;
  float minDist4 = minDist1;
#endif
  // the index of the next atom
  int neighbor1 = -1;
#if GTEXUNROLL >= 2
  int neighbor2 = -1;
#endif
#if GTEXUNROLL >= 4
  int neighbor3 = -1;
  int neighbor4 = -1;
#endif

  int acplanesz = acncells.x * acncells.y;
  int xab, yab, zab;
  for (zab=zabmin; zab<=zabmax; zab++) {
    for (yab=yabmin; yab<=yabmax; yab++) {
      for (xab=xabmin; xab<=xabmax; xab++) {
        int abcellidx = zab * acplanesz + yab * acncells.x + xab;
        uint2 atomstartend = cellStartEnd[abcellidx];
        if (atomstartend.x != GRID_CELL_EMPTY) {
          unsigned int atomid;
          for (atomid=atomstartend.x; atomid<atomstartend.y; atomid++) {
            float4 atom = sorted_xyzr[atomid];
            float dx = coorx - atom.x;
            float dy = coory - atom.y;
            float dxy2 = dx*dx + dy*dy;
  
            float dz = coorz - atom.z;
            float r21 = (dxy2 + dz*dz) * atom.w;
            densityval1 += exp2f(r21);
            // store nearest neighbor
            if( (dxy2 + dz*dz) < minDist1 ) {
                minDist1 = (dxy2 + dz*dz);
                neighbor1 = atomid;
            }

#if GUNROLL >= 2
            float dz2 = dz + gridspacing;
            float r22 = (dxy2 + dz2*dz2) * atom.w;
            densityval2 += exp2f(r22);
            // store nearest neighbor
            if( (dxy2 + dz2*dz2) < minDist2 ) {
                minDist2 = (dxy2 + dz2*dz2);
                neighbor2 = atomid;
            }
#endif
#if GUNROLL >= 4
            float dz3 = dz2 + gridspacing;
            float r23 = (dxy2 + dz3*dz3) * atom.w;
            densityval3 += exp2f(r23);
            // store nearest neighbor
            if( (dxy2 + dz3*dz3) < minDist3 ) {
                minDist3 = (dxy2 + dz3*dz3);
                neighbor3 = atomid;
            }

            float dz4 = dz3 + gridspacing;
            float r24 = (dxy2 + dz4*dz4) * atom.w;
            densityval4 += exp2f(r24);
            // store nearest neighbor
            if( (dxy2 + dz4*dz4) < minDist4 ) {
                minDist4 = (dxy2 + dz4*dz4);
                neighbor4 = atomid;
            }
#endif
          }
        }
      }
    }
  }

  densityval1 = 1.0f;
  densityval2 = 1.0f;
  densityval3 = 1.0f;
  densityval4 = 1.0f;

  //densitygrid[outaddr            ] = densityval1;
  //surf3Dwrite(densityval1, densitygrid, xindex * sizeof(float), yindex, zindex);
  surf3Dwrite(densityval1, outputSurface, xindex * sizeof(float), yindex, zindex);
#if GUNROLL >= 2
  int planesz = numvoxels.x * numvoxels.y;
  //densitygrid[outaddr +   planesz] = densityval2;
  //surf3Dwrite(densityval2, densitygrid, xindex * sizeof(float), yindex, zindex + 1);
  surf3Dwrite(densityval2, outputSurface, xindex * sizeof(float), yindex, zindex + 1);
#endif
#if GUNROLL >= 4
  //densitygrid[outaddr + 2*planesz] = densityval3;
  //surf3Dwrite(densityval3, densitygrid, xindex * sizeof(float), yindex, zindex + 2);
  surf3Dwrite(densityval3, outputSurface, xindex * sizeof(float), yindex, zindex + 2);
  //densitygrid[outaddr + 3*planesz] = densityval4;
  //surf3Dwrite(densityval4, densitygrid, xindex * sizeof(float), yindex, zindex + 3);
  surf3Dwrite(densityval4, outputSurface, xindex * sizeof(float), yindex, zindex + 3);
#endif
  
  if( storeNearestNeighbor ) {
    nearestNeighbor[outaddr          ] = atomIndex[neighbor1];
#if GUNROLL >= 2
    int planesz = numvoxels.x * numvoxels.y;
    nearestNeighbor[outaddr + planesz] = atomIndex[neighbor2];
#endif
#if GUNROLL >= 4
    nearestNeighbor[outaddr + 2*planesz] = atomIndex[neighbor3];
    nearestNeighbor[outaddr + 3*planesz] = atomIndex[neighbor4];
#endif
  }
}

// per-GPU handle with various memory buffer pointers, etc.
typedef struct {
  /// max grid sizes and attributes the current allocations will support
  long int natoms;
  int colorperatom;
  int gx;
  int gy;
  int gz;

  CUDAMarchingCubes *mc;     ///< Marching cubes class used to extract surface

  cudaArray *devdensity;         ///< density map stored in GPU memory
  float *devvoltexmap;       ///< volumetric texture map
  float4 *xyzr_d;            ///< atom coords and radii
  float4 *sorted_xyzr_d;     ///< cell-sorted coords and radii
  float4 *color_d;           ///< colors
  float4 *sorted_color_d;    ///< cell-sorted colors
  int *nearest_atom_d;           ///< colors
  
  unsigned int *atomIndex_d; ///< cell index for each atom
  unsigned int *sorted_atomIndex_d; ///< original index for each sorted atom
  unsigned int *atomHash_d;  ///<  
  uint2 *cellStartEnd_d;     ///< cell start/end indices 

  void *safety;

  cudaSurfaceObject_t devdensitySurface;

} qsurf_gpuhandle;


CUDAQuickSurfArray::CUDAQuickSurfArray() {
  voidgpu = calloc(1, sizeof(qsurf_gpuhandle));
  surfaceArea = 0.0f;
//  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
}


CUDAQuickSurfArray::~CUDAQuickSurfArray() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

  // free all working buffers if not done already
  free_bufs_map();

  // delete marching cubes object
  delete gpuh->mc;

  free(voidgpu);
}


int CUDAQuickSurfArray::free_bufs_map() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

  // zero out max buffer capacities
  gpuh->natoms = 0;
  gpuh->colorperatom = 0;
  gpuh->gx = 0;
  gpuh->gy = 0;
  gpuh->gz = 0;

  if (gpuh->safety != NULL)
    cudaFree(gpuh->safety);
  gpuh->safety=NULL;

  //if (gpuh->devdensity != NULL)
  //  cudaFree(gpuh->devdensity);
  //gpuh->devdensity=NULL;

  if (gpuh->devdensity != NULL) {
      cudaDestroySurfaceObject(gpuh->devdensitySurface);
      cudaFreeArray(gpuh->devdensity);
  }
  gpuh->devdensity = NULL;

  if (gpuh->devvoltexmap != NULL)
    cudaFree(gpuh->devvoltexmap);
  gpuh->devvoltexmap=NULL;

  if (gpuh->xyzr_d != NULL)
    cudaFree(gpuh->xyzr_d);
  gpuh->xyzr_d=NULL;

  if (gpuh->sorted_xyzr_d != NULL)
    cudaFree(gpuh->sorted_xyzr_d);  
  gpuh->sorted_xyzr_d=NULL;

  if (gpuh->color_d != NULL)
    cudaFree(gpuh->color_d);
  gpuh->color_d=NULL;

  if (gpuh->sorted_color_d != NULL)
    cudaFree(gpuh->sorted_color_d);
  gpuh->sorted_color_d=NULL;
  
  if (gpuh->nearest_atom_d != NULL)
      cudaFree(gpuh->nearest_atom_d);
  gpuh->nearest_atom_d=NULL;
  
  if (gpuh->atomIndex_d != NULL)
    cudaFree(gpuh->atomIndex_d);
  gpuh->atomIndex_d=NULL;

  if (gpuh->sorted_atomIndex_d != NULL)
    cudaFree(gpuh->sorted_atomIndex_d);
  gpuh->sorted_atomIndex_d=NULL;

  if (gpuh->atomHash_d != NULL)
    cudaFree(gpuh->atomHash_d);
  gpuh->atomHash_d=NULL;

  if (gpuh->cellStartEnd_d != NULL)
    cudaFree(gpuh->cellStartEnd_d);
  gpuh->cellStartEnd_d=NULL;
  
  return 0;
}


int CUDAQuickSurfArray::check_bufs(long int natoms, int colorperatom, 
                              int gx, int gy, int gz) {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

  if (natoms <= gpuh->natoms &&
      colorperatom <= gpuh->colorperatom &&
      gx <= gpuh->gx &&
      gy <= gpuh->gy &&
      gz <= gpuh->gz)
    return 0;
 
  return -1; // no existing bufs, or too small to be used 
}

int CUDAQuickSurfArray::alloc_bufs_map(long int natoms, int colorperatom, 
                                  int gx, int gy, int gz,
                                  bool storeNearestAtom) {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
  cudaError_t err;

  // early exit from allocation call if we've already got existing
  // buffers that are large enough to support the request
  if (check_bufs(natoms, colorperatom, gx, gy, gz) == 0)
    return 0;

  // If we have any existing allocations, trash them as they weren't
  // usable for this new request and we need to reallocate them from scratch
  free_bufs_map();

  long int ncells = gx * gy * gz;
  long int volmemsz = ncells * sizeof(float);

  // Allocate all of the memory buffers our algorithms will need up-front,
  // so we can retry and gracefully reduce the sizes of various buffers
  // to attempt to fit within available GPU memory 


  //cudaMalloc((void**)&gpuh->devdensity, volmemsz);
  //cudaMemset(gpuh->devdensity, 0, volmemsz);
  cudaExtent volumeSize;
  volumeSize.width = gx;
  volumeSize.height = gy;
  volumeSize.depth = gz;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaMalloc3DArray(&gpuh->devdensity, &channelDesc, volumeSize, cudaArraySurfaceLoadStore);
  // Specify surface
  /*
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  // Create the surface objects
  resDesc.res.array.array = gpuh->devdensity;
  gpuh->devdensitySurface = 0;
  err = cudaCreateSurfaceObject(&gpuh->devdensitySurface, &resDesc);
  if (err != cudaSuccess)
	  return -1;
  */

  if (colorperatom) {
    cudaMalloc((void**)&gpuh->devvoltexmap, 3*volmemsz);
    cudaMalloc((void**)&gpuh->color_d, natoms * sizeof(float4));
    cudaMalloc((void**)&gpuh->sorted_color_d, natoms * sizeof(float4));
  }
  if (storeNearestAtom) {
      cudaMalloc((void**)&gpuh->nearest_atom_d, ncells * sizeof(int));
  }
  cudaMalloc((void**)&gpuh->xyzr_d, natoms * sizeof(float4));
  cudaMalloc((void**)&gpuh->sorted_xyzr_d, natoms * sizeof(float4));
  cudaMalloc((void**)&gpuh->atomIndex_d, natoms * sizeof(unsigned int));
  cudaMalloc((void**)&gpuh->sorted_atomIndex_d, natoms * sizeof(unsigned int));
  cudaMalloc((void**)&gpuh->atomHash_d, natoms * sizeof(unsigned int));
  cudaMalloc((void**)&gpuh->cellStartEnd_d, ncells * sizeof(uint2));
  
  // Allocate an extra phantom array to act as a safety net to
  // ensure that subsequent allocations performed internally by 
  // the NVIDIA thrust template library or by our 
  // marching cubes implementation don't fail, since we can't 
  // currently pre-allocate all of them.
  cudaMalloc(&gpuh->safety, natoms*sizeof(float4) +           // thrust
             8 * gx * gy * sizeof(float));
  
  err = cudaGetLastError();
  if (err != cudaSuccess)
    return -1;

  // kroneml
  gpuh->natoms = natoms;
  gpuh->colorperatom = colorperatom;
  gpuh->gx = gx;
  gpuh->gy = gy;
  gpuh->gz = gz;

  return 0;
}

int CUDAQuickSurfArray::get_chunk_bufs_map(int testexisting,
                                  long int natoms, int colorperatom, 
                                  int gx, int gy, int gz,
                                  int &cx, int &cy, int &cz,
                                  int &sx, int &sy, int &sz,
                                  bool storeNearestAtom) {
  dim3 Bsz(GBLOCKSZX, GBLOCKSZY, GBLOCKSZZ);
  if (colorperatom)
    Bsz.z = GTEXBLOCKSZZ;

  cudaError_t err = cudaGetLastError(); // eat error so next CUDA op succeeds

  // enter loop to attempt a single-pass computation, but if the
  // allocation fails, cut the chunk size Z dimension by half
  // repeatedly until we either run chunks of 8 planes at a time,
  // otherwise we assume it is hopeless.
  cz <<= 1; // premultiply by two to simplify loop body
  int chunkiters = 0;
  int chunkallocated = 0;
  while (!chunkallocated) {
    // Cut the Z chunk size in half
    chunkiters++;
    cz >>= 1;

    // if we've already dropped to a subvolume size, subtract off the
    // four extra Z planes from last time before we do the modulo padding
    // calculation so we don't hit an infinite loop trying to go below 
    // 16 planes due the padding math below.
    if (cz != gz)
      cz-=4;

    // Pad the chunk to a multiple of the computational tile size since
    // each thread computes multiple elements (unrolled in the Z direction)
    cz += (8 - (cz % 8));

    // The density map "slab" size is the chunk size but without the extra
    // plane used to copy the last plane of the previous chunk's density
    // into the start, for use by the marching cubes.
    sx = cx;
    sy = cy;
    sz = cz;

    // Add four extra Z-planes for copying the previous end planes into 
    // the start of the next chunk.
    cz+=4;

#if 0
    printf("  Trying slab size: %d (test: %d)\n", sz, testexisting);
#endif

#if 1
    // test to see if total number of thread blocks exceeds maximum
    // number we can reasonably run prior to a kernel timeout error
    dim3 tGsz((sx+Bsz.x-1) / Bsz.x, 
              (sy+Bsz.y-1) / Bsz.y,
              (sz+(Bsz.z*GUNROLL)-1) / (Bsz.z * GUNROLL));
    if (colorperatom) {
      tGsz.z = (sz+(Bsz.z*GTEXUNROLL)-1) / (Bsz.z * GTEXUNROLL);
    }
    if (tGsz.x * tGsz.y * tGsz.z > 65535)
      continue; 
#endif

    // Bail out if we can't get enough memory to run at least
    // 8 slices in a single pass (making sure we've freed any allocations
    // beforehand, so they aren't leaked).
    if (sz <= 8) {
      return -1;
    }
 
    if (testexisting) {
      if (check_bufs(natoms, colorperatom, cx, cy, cz) != 0)
        continue;
    } else {
      if (alloc_bufs_map(natoms, colorperatom, cx, cy, cz, storeNearestAtom) != 0)
        continue;
    }

    chunkallocated=1;
  }

  return 0;
}

/*
 * CUDAQuickSurf::calc_map
 */
int CUDAQuickSurfArray::calc_map(long int natoms, const float *xyzr_f, 
                             const float *colors_f,
                             int colorperatom,
                             float *origin, int *numvoxels, float maxrad,
                             float radscale, float gridspacing, 
                             float isovalue, float gausslim, 
                             bool storeNearestAtom) {
    qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

    float4 *colors = (float4 *) colors_f;
    int3 volsz;
    volsz.x = numvoxels[0];
    volsz.y = numvoxels[1];
    volsz.z = numvoxels[2];

    float log2e = log2(2.718281828);

    // short-term fix until a new CUDA kernel takes care of this
    int i, i4;
    float4 *xyzr = (float4 *) malloc(natoms * sizeof(float4));
    for (i=0,i4=0; i<natoms; i++,i4+=4) {
        xyzr[i].x = xyzr_f[i4    ];
        xyzr[i].y = xyzr_f[i4 + 1];
        xyzr[i].z = xyzr_f[i4 + 2];

        float scaledrad = xyzr_f[i4 + 3] * radscale;
        float arinv = -1.0f * log2e / (2.0f*scaledrad*scaledrad);

        xyzr[i].w = arinv;
    }
	
    wkf_timerhandle globaltimer = wkf_timer_create();
    wkf_timer_start(globaltimer);

    cudaError_t err;
    cudaDeviceProp deviceProp;
    int dev;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        wkf_timer_destroy(globaltimer);
        free(xyzr);
        return -1;
    }
 
    memset(&deviceProp, 0, sizeof(cudaDeviceProp));
  
    if (cudaGetDeviceProperties(&deviceProp, dev) != cudaSuccess) {
        wkf_timer_destroy(globaltimer);
        err = cudaGetLastError(); // eat error so next CUDA op succeeds
        free(xyzr);
        return -1;
    }

    // This code currently requires compute capability 1.3 or 2.x
    // because we absolutely depend on hardware broadcasts for 
    // global memory reads by multiple threads reading the same element,
    // and the code more generally assumes the Fermi L1 cache and prefers
    // to launch 3-D grids where possible.  The current code will run on 
    // GT200 with reasonable performance so we allow it currently.  More
    // testing will be needed to determine if laptop integrated 
    // GT200 devices are truly fast enough to outrun quad core CPUs or
    // if we should trigger CPU fallback on devices with smaller SM counts.
    if ((deviceProp.major < 2) &&
        ((deviceProp.major == 1) && (deviceProp.minor < 3))) {
        wkf_timer_destroy(globaltimer);
        free(xyzr);
        return -1;
    }

    // compute grid spacing for the acceleration grid
    float acgridspacing = gausslim * radscale * maxrad;
    
    // ensure acceleration grid spacing >= density grid spacing
    if (acgridspacing < gridspacing)
        acgridspacing = gridspacing;


    // Allocate output arrays for the gaussian density map and 3-D texture map
    // We test for errors carefully here since this is the most likely place
    // for a memory allocation failure due to the size of the grid.
    int3 chunksz = volsz;
    int3 slabsz = volsz;

    int3 accelcells;
    accelcells.x = max(int((volsz.x*gridspacing) / acgridspacing), 1);
    accelcells.y = max(int((volsz.y*gridspacing) / acgridspacing), 1);
    accelcells.z = max(int((volsz.z*gridspacing) / acgridspacing), 1);

    dim3 Bsz(GBLOCKSZX, GBLOCKSZY, GBLOCKSZZ);
    if (colorperatom)
        Bsz.z = GTEXBLOCKSZZ;

    // check to see if it's possible to use an existing allocation,
    // if so, just leave things as they are, and do the computation 
    // using the existing buffers
    if (gpuh->natoms == 0 ||
            // TODO Make sure the computation stops if not enough memory available!
            get_chunk_bufs_map(0, natoms, colorperatom,
                volsz.x, volsz.y, volsz.z,
                chunksz.x, chunksz.y, chunksz.z,
                slabsz.x, slabsz.y, slabsz.z, storeNearestAtom) == -1) {
        // reset the chunksz and slabsz after failing to try and
        // fit them into the existing allocations...
        chunksz = volsz;
        slabsz = volsz;

        // reallocate the chunk buffers from scratch since we weren't
        // able to reuse them
        if (get_chunk_bufs_map(0, natoms, colorperatom,
                volsz.x, volsz.y, volsz.z,
                chunksz.x, chunksz.y, chunksz.z,
                slabsz.x, slabsz.y, slabsz.z, storeNearestAtom) == -1) {
            wkf_timer_destroy(globaltimer);
            free(xyzr);
            return -1;
        }
    }

    // Free the "safety padding" memory we allocate to ensure we dont
    // have trouble with thrust calls that allocate their own memory later
    if (gpuh->safety != NULL)
        cudaFree(gpuh->safety);
    gpuh->safety = NULL;

    cudaMemcpy(gpuh->xyzr_d, xyzr, natoms * sizeof(float4), cudaMemcpyHostToDevice);
    if (colorperatom)
        cudaMemcpy(gpuh->color_d, colors, natoms * sizeof(float4), cudaMemcpyHostToDevice);
    // set all nearest neighbor values to -1 (no neighbor)
    if (storeNearestAtom) {
        int gridDim = gpuh->gx * gpuh->gy * gpuh->gz;
        const dim3 maxGridSize(65535, 65535, 0);
        const int blocksPerGrid = (gridDim + 256 - 1) / 256;
        dim3 grid(blocksPerGrid, 1, 1);
        // Test if grid needs to be extended to 2D.
        while (grid.x > maxGridSize.x) {
            grid.x /= 2;
            grid.y *= 2;
        }
        setArrayToIntQSA<<<grid, 256>>>( gridDim, gpuh->nearest_atom_d, -1);
    }
  
    free(xyzr);
 
    // build uniform grid acceleration structure
    if (vmd_cuda_build_density_atom_gridQSA(natoms, gpuh->xyzr_d, gpuh->color_d,
                                        gpuh->sorted_xyzr_d,
                                        gpuh->sorted_color_d,
                                        gpuh->atomIndex_d, gpuh->sorted_atomIndex_d,
                                        gpuh->atomHash_d, gpuh->cellStartEnd_d, 
                                        accelcells, 1.0f / acgridspacing) != 0) {
        wkf_timer_destroy(globaltimer);
        free_bufs_map();
        return -1;
    }

    double sorttime = wkf_timer_timenow(globaltimer);
    double lastlooptime = sorttime;

    double densitykerneltime = 0.0f;

    int lzplane = GBLOCKSZZ * GUNROLL;
    if (colorperatom)
        lzplane = GTEXBLOCKSZZ * GTEXUNROLL;

    // guarantee that the method finishes with one run
    if (volsz.z > slabsz.z) {
        wkf_timer_destroy(globaltimer);
        free_bufs_map();
        return -1;
    }

    int3 curslab = slabsz;

    int slabplanesz = curslab.x * curslab.y;

    dim3 Gsz((curslab.x + Bsz.x - 1) / Bsz.x,
        (curslab.y + Bsz.y - 1) / Bsz.y,
        (curslab.z + (Bsz.z*GUNROLL) - 1) / (Bsz.z * GUNROLL));
    if (colorperatom)
        Gsz.z = (curslab.z + (Bsz.z*GTEXUNROLL) - 1) / (Bsz.z * GTEXUNROLL);

    // For SM 2.x, we can run the entire slab in one pass by launching
    // a 3-D grid of thread blocks.
    // If we are running on SM 1.x, we can only launch 1-D grids so we
    // must loop over planar grids until we have processed the whole slab.
    dim3 Gszslice = Gsz;
    if (deviceProp.major < 2)
        Gszslice.z = 1;

	err = cudaBindSurfaceToArray(outputSurface, gpuh->devdensity);
	if (err != cudaSuccess) {
		printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		return -1;
	}

    if (colorperatom) {
        gaussdensity_fast_texQSA<<<Gszslice, Bsz, 0>>>(natoms, 
            gpuh->sorted_xyzr_d, gpuh->sorted_color_d, 
            make_int3(0, 0, 0), accelcells, acgridspacing,
            1.0f / acgridspacing, gpuh->cellStartEnd_d, gridspacing, 0,
            gpuh->devdensitySurface, (float3*)gpuh->devvoltexmap, 1.0f / isovalue, storeNearestAtom, gpuh->nearest_atom_d, gpuh->atomIndex_d);
    } else {
        gaussdensity_fastQSA<<<Gszslice, Bsz, 0>>>(natoms, 
            gpuh->sorted_xyzr_d, 
            make_int3(0, 0, 0), accelcells, acgridspacing, 1.0f / acgridspacing,
            gpuh->cellStartEnd_d, gridspacing, 0, gpuh->devdensitySurface, storeNearestAtom, gpuh->nearest_atom_d, gpuh->atomIndex_d);
    }
    cudaThreadSynchronize(); 
    densitykerneltime = wkf_timer_timenow(globaltimer);

    // catch any errors that may have occured so that at the very least,
    // all of the subsequent resource deallocation calls will succeed
    err = cudaGetLastError();

    wkf_timer_stop(globaltimer);
    double totalruntime = wkf_timer_time(globaltimer);
    wkf_timer_destroy(globaltimer);

    // If an error occured, we print it and return an error code, once
    // all of the memory deallocations have completed.
    if (err != cudaSuccess) { 
        printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        return -1;
    }

#if VERBOSE
  printf("  GPU generated density map (%i x %i x %i) in %d passes\n", volsz.x, volsz.y, volsz.z, chunkcount);

  printf("  GPU time (%s): %.3f [sort: %.3f density %.3f copy: %.3f]\n", 
         (deviceProp.major == 1 && deviceProp.minor == 3) ? "SM 1.3" : "SM 2.x",
         totalruntime, sorttime, densitytime, copytime);
#endif

  return 0;
}

int CUDAQuickSurfArray::getMapSizeX() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
  return gpuh->gx;
}

int CUDAQuickSurfArray::getMapSizeY() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
  return gpuh->gy;
}

int CUDAQuickSurfArray::getMapSizeZ() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
  return gpuh->gz;
}

cudaArray* CUDAQuickSurfArray::getMap() {
    qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *)voidgpu;
    return gpuh->devdensity;
}

float* CUDAQuickSurfArray::getColorMap() {
    qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
    return gpuh->devvoltexmap;
}

int* CUDAQuickSurfArray::getNeighborMap() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
  return gpuh->nearest_atom_d;
}
