/***************************************************************************
 *cr
 *cr            (C) Copyright 2007-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: CUDAQuickSurf.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.38 $      $Date: 2011/12/03 20:24:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA accelerated gaussian density calculation
 *
 ***************************************************************************/

//#define VERBOSE 1
//#define CUDA_TIMER

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

//#define WRITE_DATRAW_FILE
//#define WRITE_DATRAW_FILE_MAP

#include "utilities.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#include "CUDAKernels.h" 
#include "CUDAMarchingCubes.h"
#include "CUDAQuickSurfAlternative.h"
#include <thrust/device_ptr.h>
//
// multi-threaded direct summation implementation
//

inline __host__ __device__ float dot(float3 a, float3 b) { 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float length(float3 v){
    return sqrtf(dot(v, v));
}

typedef struct {
  float isovalue;
  float radscale;
  float4 *xyzr;
  float4 *colors;
  float *volmap;
  float *voltexmap;
  int numplane;
  int numcol;
  int numpt;
  long int natoms;
  float gridspacing;
} enthrparms;

/* thread prototype */
static void * cudadensitythread(void *);

#if 1
#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  printf("Thread aborting...\n"); \
  return NULL; }}
#else
#define CUERR
#endif

// 
// Parameters for direct summation gaussian density kernels
//
#if 1

#define DUNROLLX   8 
#define DBLOCKSZX  8  // make large enough to allow coalesced global mem ops
#define DBLOCKSZY  8  // make as small as possible for finer granularity

#else

#define DUNROLLX   4
#define DBLOCKSZX 16  // make large enough to allow coalesced global mem ops
#define DBLOCKSZY 16  // make as small as possible for finer granularity

#endif

#define DBLOCKSZ  (DBLOCKSZX*DBLOCKSZY)

// FLOP counting
#if DUNROLLX == 8
#define FLOPSPERATOMEVALTEX (109.0/8.0)
#define FLOPSPERATOMEVAL    (53.0/8.0)
#elif DUNROLLX == 4 
#define FLOPSPERATOMEVALTEX (57.0/4.0)
#define FLOPSPERATOMEVAL    (29.0/4.0)
#elif DUNROLLX == 2 
#define FLOPSPERATOMEVALTEX (30.0/2.0)
#define FLOPSPERATOMEVAL    (17.0/2.0)
#endif


__global__ static void setArrayToInt( int count, int* array, int value) {
    // get global index
    const unsigned int idx = __umul24(__umul24(blockIdx.y, gridDim.x) + blockIdx.x, blockDim.x) + threadIdx.x;
    // early exit if this thread is outside of the grid bounds
    if (idx >= count)
        return;
    array[idx] = value;
}

__global__ static void gaussdensity_direct_tex(int natoms, 
                                        const float4 *xyzr, 
                                        const float4 *colors,
                                        float gridspacing, unsigned int z,
                                        float *densitygrid, 
                                        float3 *voltexmap, 
                                        float invisovalue) {
  unsigned int xindex  = (blockIdx.x * blockDim.x) * DUNROLLX + threadIdx.x;
  unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int zindex  = (blockIdx.z * blockDim.z) + threadIdx.z;
  unsigned int outaddr = 
    ((gridDim.x * blockDim.x) * DUNROLLX) * (gridDim.y * blockDim.y) * zindex + 
    ((gridDim.x * blockDim.x) * DUNROLLX) * yindex + xindex;
  zindex += z;

  float coorx = gridspacing * xindex;
  float coory = gridspacing * yindex;
  float coorz = gridspacing * zindex;

  float densityvalx1=0.0f;
  float densityvalx2=0.0f;
  float3 densitycolx1;
  densitycolx1=make_float3(0.0f, 0.0f, 0.0f);
  float3 densitycolx2=densitycolx1;

#if DUNROLLX >= 4
  float densityvalx3=0.0f;
  float densityvalx4=0.0f;
  float3 densitycolx3=densitycolx1;
  float3 densitycolx4=densitycolx1;
#endif
#if DUNROLLX >= 8
  float densityvalx5=0.0f;
  float densityvalx6=0.0f;
  float densityvalx7=0.0f;
  float densityvalx8=0.0f;

  float3 densitycolx5=densitycolx1;
  float3 densitycolx6=densitycolx1;
  float3 densitycolx7=densitycolx1;
  float3 densitycolx8=densitycolx1;
#endif

  float gridspacing_coalesce = gridspacing * DBLOCKSZX;

  int atomid;
  for (atomid=0; atomid<natoms; atomid++) {
    float4 atom = xyzr[atomid];
    float4 color = colors[atomid];

    float dy = coory - atom.y;
    float dz = coorz - atom.z;
    float dyz2 = dy*dy + dz*dz;

    float dx1 = coorx - atom.x;
    float r21 = (dx1*dx1 + dyz2) * atom.w;
    float tmp1 = exp2f(-r21);
    densityvalx1 += tmp1;
    tmp1 *= invisovalue;
    densitycolx1.x += tmp1 * color.x;
    densitycolx1.y += tmp1 * color.y;
    densitycolx1.z += tmp1 * color.z;

    float dx2 = dx1 + gridspacing_coalesce;
    float r22 = (dx2*dx2 + dyz2) * atom.w;
    float tmp2 = exp2f(-r22);
    densityvalx2 += tmp2;
    tmp2 *= invisovalue;
    densitycolx2.x += tmp2 * color.x;
    densitycolx2.y += tmp2 * color.y;
    densitycolx2.z += tmp2 * color.z;

#if DUNROLLX >= 4
    float dx3 = dx2 + gridspacing_coalesce;
    float r23 = (dx3*dx3 + dyz2) * atom.w;
    float tmp3 = exp2f(-r23);
    densityvalx3 += tmp3;
    tmp3 *= invisovalue;
    densitycolx3.x += tmp3 * color.x;
    densitycolx3.y += tmp3 * color.y;
    densitycolx3.z += tmp3 * color.z;

    float dx4 = dx3 + gridspacing_coalesce;
    float r24 = (dx4*dx4 + dyz2) * atom.w;
    float tmp4 = exp2f(-r24);
    densityvalx4 += tmp4;
    tmp4 *= invisovalue;
    densitycolx4.x += tmp4 * color.x;
    densitycolx4.y += tmp4 * color.y;
    densitycolx4.z += tmp4 * color.z;
#endif
#if DUNROLLX >= 8
    float dx5 = dx4 + gridspacing_coalesce;
    float r25 = (dx5*dx5 + dyz2) * atom.w;
    float tmp5 = exp2f(-r25);
    densityvalx5 += tmp5;
    tmp5 *= invisovalue;
    densitycolx5.x += tmp5 * color.x;
    densitycolx5.y += tmp5 * color.y;
    densitycolx5.z += tmp5 * color.z;

    float dx6 = dx5 + gridspacing_coalesce;
    float r26 = (dx6*dx6 + dyz2) * atom.w;
    float tmp6 = exp2f(-r26);
    densityvalx6 += tmp6;
    tmp6 *= invisovalue;
    densitycolx6.x += tmp6 * color.x;
    densitycolx6.y += tmp6 * color.y;
    densitycolx6.z += tmp6 * color.z;

    float dx7 = dx6 + gridspacing_coalesce;
    float r27 = (dx7*dx7 + dyz2) * atom.w;
    float tmp7 = exp2f(-r27);
    densityvalx7 += tmp7;
    tmp7 *= invisovalue;
    densitycolx7.x += tmp7 * color.x;
    densitycolx7.y += tmp7 * color.y;
    densitycolx7.z += tmp7 * color.z;

    float dx8 = dx7 + gridspacing_coalesce;
    float r28 = (dx8*dx8 + dyz2) * atom.w;
    float tmp8 = exp2f(-r28);
    densityvalx8 += tmp8;
    tmp8 *= invisovalue;
    densitycolx8.x += tmp8 * color.x;
    densitycolx8.y += tmp8 * color.y;
    densitycolx8.z += tmp8 * color.z;
#endif
  }

  densitygrid[outaddr             ] += densityvalx1;
  voltexmap[outaddr             ].x += densitycolx1.x;
  voltexmap[outaddr             ].y += densitycolx1.y;
  voltexmap[outaddr             ].z += densitycolx1.z;

  densitygrid[outaddr+1*DBLOCKSZX] += densityvalx2;
  voltexmap[outaddr+1*DBLOCKSZX].x += densitycolx2.x;
  voltexmap[outaddr+1*DBLOCKSZX].y += densitycolx2.y;
  voltexmap[outaddr+1*DBLOCKSZX].z += densitycolx2.z;

#if DUNROLLX >= 4
  densitygrid[outaddr+2*DBLOCKSZX] += densityvalx3;
  voltexmap[outaddr+2*DBLOCKSZX].x += densitycolx3.x;
  voltexmap[outaddr+2*DBLOCKSZX].y += densitycolx3.y;
  voltexmap[outaddr+2*DBLOCKSZX].z += densitycolx3.z;

  densitygrid[outaddr+3*DBLOCKSZX] += densityvalx4;
  voltexmap[outaddr+3*DBLOCKSZX].x += densitycolx4.x;
  voltexmap[outaddr+3*DBLOCKSZX].y += densitycolx4.y;
  voltexmap[outaddr+3*DBLOCKSZX].z += densitycolx4.z;
#endif
#if DUNROLLX >= 8
  densitygrid[outaddr+4*DBLOCKSZX] += densityvalx5;
  voltexmap[outaddr+4*DBLOCKSZX].x += densitycolx5.x;
  voltexmap[outaddr+4*DBLOCKSZX].y += densitycolx5.y;
  voltexmap[outaddr+4*DBLOCKSZX].z += densitycolx5.z;

  densitygrid[outaddr+5*DBLOCKSZX] += densityvalx6;
  voltexmap[outaddr+5*DBLOCKSZX].x += densitycolx6.x;
  voltexmap[outaddr+5*DBLOCKSZX].y += densitycolx6.y;
  voltexmap[outaddr+5*DBLOCKSZX].z += densitycolx6.z;

  densitygrid[outaddr+6*DBLOCKSZX] += densityvalx7;
  voltexmap[outaddr+6*DBLOCKSZX].x += densitycolx7.x;
  voltexmap[outaddr+6*DBLOCKSZX].y += densitycolx7.y;
  voltexmap[outaddr+6*DBLOCKSZX].z += densitycolx7.z;

  densitygrid[outaddr+7*DBLOCKSZX] += densityvalx8;
  voltexmap[outaddr+7*DBLOCKSZX].x += densitycolx8.x;
  voltexmap[outaddr+7*DBLOCKSZX].y += densitycolx8.y;
  voltexmap[outaddr+7*DBLOCKSZX].z += densitycolx8.z;
#endif
}


__global__ static void gaussdensity_direct_alt(int natoms,
                                    const float4 *xyzr, 
                                    float gridspacing, unsigned int z, 
                                    float *densitygrid) {
  unsigned int xindex  = (blockIdx.x * blockDim.x) * DUNROLLX + threadIdx.x;
  unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int zindex  = (blockIdx.z * blockDim.z) + threadIdx.z;
  unsigned int outaddr = 
    ((gridDim.x * blockDim.x) * DUNROLLX) * (gridDim.y * blockDim.y) * zindex + 
    ((gridDim.x * blockDim.x) * DUNROLLX) * yindex + xindex;
  zindex += z;

  float coorx = gridspacing * xindex;
  float coory = gridspacing * yindex;
  float coorz = gridspacing * zindex;

  float densityvalx1=0.0f;
  float densityvalx2=0.0f;
#if DUNROLLX >= 4
  float densityvalx3=0.0f;
  float densityvalx4=0.0f;
#endif
#if DUNROLLX >= 8
  float densityvalx5=0.0f;
  float densityvalx6=0.0f;
  float densityvalx7=0.0f;
  float densityvalx8=0.0f;
#endif

  float gridspacing_coalesce = gridspacing * DBLOCKSZX;

  int atomid;
  for (atomid=0; atomid<natoms; atomid++) {
    float4 atom = xyzr[atomid];
    float dy = coory - atom.y;
    float dz = coorz - atom.z;
    float dyz2 = dy*dy + dz*dz;

    float dx1 = coorx - atom.x;
    float r21 = (dx1*dx1 + dyz2) * atom.w;
    densityvalx1 += exp2f(-r21);

    float dx2 = dx1 + gridspacing_coalesce;
    float r22 = (dx2*dx2 + dyz2) * atom.w;
    densityvalx2 += exp2f(-r22);

#if DUNROLLX >= 4
    float dx3 = dx2 + gridspacing_coalesce;
    float r23 = (dx3*dx3 + dyz2) * atom.w;
    densityvalx3 += exp2f(-r23);

    float dx4 = dx3 + gridspacing_coalesce;
    float r24 = (dx4*dx4 + dyz2) * atom.w;
    densityvalx4 += exp2f(-r24);
#endif
#if DUNROLLX >= 8
    float dx5 = dx4 + gridspacing_coalesce;
    float r25 = (dx5*dx5 + dyz2) * atom.w;
    densityvalx5 += exp2f(-r25);

    float dx6 = dx5 + gridspacing_coalesce;
    float r26 = (dx6*dx6 + dyz2) * atom.w;
    densityvalx6 += exp2f(-r26);

    float dx7 = dx6 + gridspacing_coalesce;
    float r27 = (dx7*dx7 + dyz2) * atom.w;
    densityvalx7 += exp2f(-r27);

    float dx8 = dx7 + gridspacing_coalesce;
    float r28 = (dx8*dx8 + dyz2) * atom.w;
    densityvalx8 += exp2f(-r28);
#endif
  }

  densitygrid[outaddr             ] += densityvalx1;
  densitygrid[outaddr+1*DBLOCKSZX] += densityvalx2;
#if DUNROLLX >= 4
  densitygrid[outaddr+2*DBLOCKSZX] += densityvalx3;
  densitygrid[outaddr+3*DBLOCKSZX] += densityvalx4;
#endif
#if DUNROLLX >= 8
  densitygrid[outaddr+4*DBLOCKSZX] += densityvalx5;
  densitygrid[outaddr+5*DBLOCKSZX] += densityvalx6;
  densitygrid[outaddr+6*DBLOCKSZX] += densityvalx7;
  densitygrid[outaddr+7*DBLOCKSZX] += densityvalx8;
#endif
}


// required GPU array padding to match thread block size
// XXX note: this code requires block size dimensions to be a power of two
#define TILESIZEX DBLOCKSZX*DUNROLLX
#define TILESIZEY DBLOCKSZY
#define GPU_X_ALIGNMASK (TILESIZEX - 1)
#define GPU_Y_ALIGNMASK (TILESIZEY - 1)

int vmd_cuda_gaussdensity_direct_alt(long int natoms, float4 *xyzr, float4 *colors,
                                 float* volmap, float *voltexmap, int3 volsz,
                                 float radscale, float gridspacing, 
                                 float isovalue, float gausslim) {
  enthrparms parms;
  wkf_timerhandle globaltimer;
  double totalruntime;
  int rc=0;

  int numprocs = wkf_thread_numprocessors();
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount))
    return -1;
  if (deviceCount < 1)
    return -1;

  /* take the lesser of the number of CPUs and GPUs */
  /* and execute that many threads                  */
  if (deviceCount < numprocs) {
    numprocs = deviceCount;
  }

  printf("Using %d CUDA GPUs\n", numprocs);
  printf("GPU padded grid size: %d x %d x %d\n", 
         (volsz.x + GPU_X_ALIGNMASK) & ~(GPU_X_ALIGNMASK),
         (volsz.y + GPU_Y_ALIGNMASK) & ~(GPU_Y_ALIGNMASK),
          volsz.z);

  parms.isovalue = isovalue;
  parms.radscale = radscale;
  parms.xyzr = xyzr;
  parms.colors = colors;
  parms.volmap = volmap;
  parms.voltexmap = voltexmap;
  parms.numplane = volsz.z;
  parms.numcol = volsz.y;
  parms.numpt = volsz.x;
  parms.natoms = natoms;
  parms.gridspacing = gridspacing;

  globaltimer = wkf_timer_create();
  wkf_timer_start(globaltimer);

  /* spawn child threads to do the work */
  wkf_tasktile_t tile;
  tile.start=0;
  tile.end=volsz.z;
  rc = wkf_threadlaunch(numprocs, &parms, cudadensitythread, &tile);

  // Measure GFLOPS
  wkf_timer_stop(globaltimer);
  totalruntime = wkf_timer_time(globaltimer);
  wkf_timer_destroy(globaltimer);

  if (!rc) {
    double atomevalssec = ((double) volsz.x * volsz.y * volsz.z * natoms) / (totalruntime * 1000000000.0);
    printf("  %g billion atom evals/second, %g GFLOPS\n",
           atomevalssec, 
           atomevalssec * ((voltexmap==NULL) ? FLOPSPERATOMEVAL : FLOPSPERATOMEVALTEX));
  } else {
    printf( "A GPU encountered an unrecoverable error.\n");
    printf( "Calculation will continue using the main CPU.\n");
  }
  return rc;
}





static void * cudadensitythread(void *voidparms) {
  dim3 volsize, Gsz, Bsz;
  float *devdensity = NULL;
  float *devtexmap = NULL;
  float *hostdensity = NULL;
  float *hosttexmap = NULL;
  enthrparms *parms = NULL;
  int threadid=0;

  wkf_threadlaunch_getid(voidparms, &threadid, NULL);
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);

  /* 
   * copy in per-thread parameters 
   */
  float isovalue = parms->isovalue;
//  float radscale = parms->radscale;
  const float4 *xyzr = parms->xyzr;
  const float4 *colors = parms->colors;
  float *volmap = parms->volmap;
  float *voltexmap = parms->voltexmap;
  const long int numplane = parms->numplane;
  const int numcol = parms->numcol;
  const int numpt = parms->numpt;
  const int natoms = parms->natoms;
  const float gridspacing = parms->gridspacing;
  double lasttime, totaltime;

  cudaError_t rc;
  rc = cudaSetDevice(threadid);
  if (rc != cudaSuccess) {
#if CUDART_VERSION >= 2010
    rc = cudaGetLastError(); // query last error and reset error state
    if (rc != cudaErrorSetOnActiveProcess)
      return NULL; // abort and return an error
#else
    cudaGetLastError(); // just ignore and reset error state, since older CUDA
                        // revs don't have a cudaErrorSetOnActiveProcess enum
#endif
  }

  // setup density grid size, padding out arrays for peak GPU memory performance
  volsize.x = (numpt  + GPU_X_ALIGNMASK) & ~(GPU_X_ALIGNMASK);
  volsize.y = (numcol + GPU_Y_ALIGNMASK) & ~(GPU_Y_ALIGNMASK);
  volsize.z = 1;      // we only do one plane at a time

  // setup CUDA grid and block sizes
  Bsz.x = DBLOCKSZX;
  Bsz.y = DBLOCKSZY;
  Bsz.z = 1;
  Gsz.x = volsize.x / (Bsz.x * DUNROLLX);
  Gsz.y = volsize.y / Bsz.y; 
  Gsz.z = 1;
  int maxplanes = 4;
  int volplanesz = sizeof(float) * volsize.x * volsize.y * volsize.z;
  int volmemsz = maxplanes * volplanesz;

  printf("Thread %d started for CUDA device %d, grid size %dx%dx%d\n", 
         threadid, threadid, Gsz.x, Gsz.y, Gsz.z);
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);
  wkfmsgtimer * msgt = wkf_msg_timer_create(5);

  // allocate and initialize the GPU output array
  cudaMalloc((void**)&devdensity, volmemsz);
  if (voltexmap != NULL) cudaMalloc((void**)&devtexmap, volmemsz*3);
  CUERR // check and clear any existing errors

  hostdensity = (float *) malloc(volmemsz); // allocate working buffer
  if (voltexmap != NULL) hosttexmap = (float *) malloc(volmemsz*3); // allocate working buffer

  float *devatoms=NULL, *devradii=NULL, *devcolors=NULL;
  cudaMalloc((void**)&devatoms, natoms*4*sizeof(float));
  cudaMemcpy(devatoms, xyzr, natoms*4*sizeof(float), cudaMemcpyHostToDevice);
  if (colors) {
    cudaMalloc((void**)&devcolors, natoms*4*sizeof(float));  
    cudaMemcpy(devcolors, colors, natoms*4*sizeof(float), cudaMemcpyHostToDevice);
  }
  CUERR // check and clear any existing errors

  // For each point in the cube...
  int computedplanes=0;
  wkf_tasktile_t tile;
  while (wkf_threadlaunch_next_tile(voidparms, maxplanes, &tile) != WKF_SCHED_DONE) {
    int z, k;
    int numk=tile.end - tile.start;
    int runplanes = min(numk, maxplanes);
    // Set CUDA grid Z dimension
    Gsz.z = runplanes;
    for (k=tile.start; k<tile.end; k+=runplanes) {
// printf("k[%d/%d] run: %d Gsz.z: %d\n", k, tile.end, runplanes, Gsz.z);
      int y;
      computedplanes++; // track work done by this GPU for progress reporting

#if 1
      cudaMemset(devdensity, 0, volmemsz);
      if (hosttexmap != NULL)
        cudaMemset(devtexmap, 0, 3*volmemsz);
      CUERR // check and clear any existing errors
#else
      // XXX no need for this currently... 
      // Copy density grid into GPU padded input
      for (z=k; z<k+runplanes; z++) {
        for (y=0; y<numcol; y++) {
          long densaddr = z*numcol*numpt + y*numpt;
          memcpy(&hostdensity[(z-k)*volsize.x*volsize.y + y*volsize.x], 
                 &volmap[densaddr], numpt * sizeof(float));
          if (hosttexmap != NULL)
            memcpy(&hosttexmap[((z-k)*volsize.x*volsize.y + y*volsize.x)*3], 
                   &voltexmap[densaddr*3], 3 * numpt * sizeof(float));
        }
      }

      // Copy the Host input data to the GPU..
      cudaMemcpy(devdensity, hostdensity, volmemsz, cudaMemcpyHostToDevice);
      if (devtexmap != NULL)
        cudaMemcpy(devtexmap, hosttexmap, volmemsz*3, cudaMemcpyHostToDevice);
      CUERR // check and clear any existing errors
#endif

      lasttime = wkf_timer_timenow(timer);
      // RUN the kernel...
      if (devtexmap != NULL) {
        gaussdensity_direct_tex<<<Gsz, Bsz, 0>>>(natoms, 
                                          (float4*) devatoms,
                                          (float4*) devcolors,
                                          gridspacing, k,
                                          devdensity, 
                                          (float3*) devtexmap,
                                          1.0f / isovalue);
      } else {
        gaussdensity_direct_alt<<<Gsz, Bsz, 0>>>(natoms, 
                                      (float4*) devatoms,
                                      gridspacing, k, devdensity);
      }
      cudaThreadSynchronize();
	  getLastCudaError("kernel failed");
      CUERR // check and clear any existing errors

      // Copy the GPU output data back to the host and use/store it..
      cudaMemcpy(hostdensity, devdensity, volmemsz, cudaMemcpyDeviceToHost);
      if (devtexmap != NULL)
        cudaMemcpy(hosttexmap, devtexmap, volmemsz*3, cudaMemcpyDeviceToHost);
      CUERR // check and clear any existing errors

      // Copy GPU blocksize padded array back down to the original size
      for (z=k; z<k+runplanes; z++) {
        for (y=0; y<numcol; y++) {
          long densaddr = z*numcol*numpt + y*numpt;
          memcpy(&volmap[densaddr], 
                 &hostdensity[(z-k)*volsize.x*volsize.y + y*volsize.x], 
                 numpt * sizeof(float));
          if (hosttexmap != NULL) 
            memcpy(&voltexmap[densaddr*3], 
                   &hosttexmap[((z-k)*volsize.x*volsize.y + y*volsize.x)*3], 
                   3 * numpt * sizeof(float));
        }
      }
 
      totaltime = wkf_timer_timenow(timer);
      if (wkf_msg_timer_timeout(msgt)) {
        // XXX: we have to use printf here as msgInfo is not thread-safe yet.
        printf("thread[%d] plane %d/%ld (%d computed) time %.2f, elapsed %.1f, est. total: %.1f\n",
               threadid, k, numplane, computedplanes,
               totaltime - lasttime, totaltime,
               totaltime * numplane / (k+1));
      }
    }
  }

  wkf_timer_destroy(timer); // free timer
  wkf_msg_timer_destroy(msgt); // free timer
  free(hostdensity);    // free working buffer
  if (hosttexmap != NULL)
    free(hosttexmap);    // free working buffer
  cudaFree(devdensity); // free CUDA memory buffer
  if (devtexmap != NULL)
    cudaFree(devtexmap); // free CUDA memory buffer
  if (devatoms != NULL)
    cudaFree(devatoms);
  if (devradii != NULL)
    cudaFree(devradii);
  if (devcolors != NULL)
    cudaFree(devcolors);

  CUERR // check and clear any existing errors

  return NULL;
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
__global__ void hashAtomsAlt(unsigned int natoms,
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
__global__ void sortAtomsGenCellListsAlt(unsigned int natoms,
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


int vmd_cuda_build_density_atom_grid_alt(int natoms,
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
  hashAtomsAlt<<<hGsz, hBsz>>>(natoms, xyzr_d, volsz, invgridspacing,
                            atomIndex_d, atomHash_d);
  getLastCudaError("kernel failed");

  // Sort atom indices by their grid cell address
  // (wrapping the device pointers with vector iterators)
  thrust::sort_by_key(thrust::device_ptr<unsigned int>(atomHash_d),
                      thrust::device_ptr<unsigned int>(atomHash_d + natoms),
                      thrust::device_ptr<unsigned int>(atomIndex_d));

  // Initialize all cells to empty
  int ncells = volsz.x * volsz.y * volsz.z;
  cudaMemset(cellStartEnd_d, GRID_CELL_EMPTY, ncells*sizeof(uint2));

  // Reorder atoms into sorted order and find start and end of each cell
  // XXX need to use 2-D indexing for large atom counts or we exceed the
  //     per-dimension 65535 block grid size limitation
  unsigned int smemSize = sizeof(unsigned int)*(hBsz.x+1);
  sortAtomsGenCellListsAlt<<<hGsz, hBsz, smemSize>>>(
                       natoms, xyzr_d, color_d, atomIndex_d, sorted_atomIndex_d, 
                       atomHash_d, sorted_xyzr_d, sorted_color_d, cellStartEnd_d);
#if 1
  // XXX when the code is ready for production use, we can disable
  //     detailed error checking and use a more all-or-nothing approach
  //     where errors fall through all of the CUDA API calls until the
  //     end and we do the cleanup only at the end.
  cudaThreadSynchronize();
  getLastCudaError("kernel failed");
  cudaError_t err = cudaGetLastError();
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

#if 1
#define GTEXBLOCKSZZ 2
#define GTEXUNROLL   4
#define GBLOCKSZZ    2
#define GUNROLL      4
#else
#define GTEXBLOCKSZZ 8
#define GTEXUNROLL   1
#define GBLOCKSZZ    8
#define GUNROLL      1
#endif

__global__ static void gaussdensity_fast_tex(int natoms,
                                         const float4 *sorted_xyzr, 
                                         const float4 *sorted_color, 
                                         int3 numvoxels,
                                         int3 acncells,
                                         float acgridspacing,
                                         float invacgridspacing,
                                         const uint2 * cellStartEnd,
                                         float gridspacing, unsigned int z, 
                                         float *densitygrid,
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
            float tmp1 = exp2f(r21) * color.w; // schatzkn: scale the gaussian by the concentration that is written in the w-component of the color
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
			float tmp2 = exp2f(r22) * color.w; // schatzkn: scale the gaussian by the concentration that is written in the w-component of the color
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
			float tmp3 = exp2f(r23) * color.w; // schatzkn: scale the gaussian by the concentration that is written in the w-component of the color
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
			float tmp4 = exp2f(r24) * color.w; // schatzkn: scale the gaussian by the concentration that is written in the w-component of the color
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

  densitygrid[outaddr          ] = densityval1;
  voltexmap[outaddr          ].x = densitycol1.x;
  voltexmap[outaddr          ].y = densitycol1.y;
  voltexmap[outaddr          ].z = densitycol1.z;

#if GTEXUNROLL >= 2
  int planesz = numvoxels.x * numvoxels.y;
  densitygrid[outaddr + planesz] = densityval2;
  voltexmap[outaddr + planesz].x = densitycol2.x;
  voltexmap[outaddr + planesz].y = densitycol2.y;
  voltexmap[outaddr + planesz].z = densitycol2.z;
#endif
#if GTEXUNROLL >= 4
  densitygrid[outaddr + 2*planesz] = densityval3;
  voltexmap[outaddr + 2*planesz].x = densitycol3.x;
  voltexmap[outaddr + 2*planesz].y = densitycol3.y;
  voltexmap[outaddr + 2*planesz].z = densitycol3.z;

  densitygrid[outaddr + 3*planesz] = densityval4;
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

__global__ static void gaussdensity_fast_tex(int natoms,
                                         const float4 *sorted_xyzr, 
                                         const float4 *sorted_color, 
                                         int3 numvoxels,
                                         int3 acncells,
                                         float acgridspacing,
                                         float invacgridspacing,
                                         const uint2 * cellStartEnd,
                                         float3 gridspacing, unsigned int z, 
                                         float *densitygrid,
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
  int xabmin = ((blockIdx.x * blockDim.x) * gridspacing.x - acgridspacing) * invacgridspacing;
  int yabmin = ((blockIdx.y * blockDim.y) * gridspacing.y - acgridspacing) * invacgridspacing;
  int zabmin = ((z + blockIdx.z * blockDim.z * GTEXUNROLL) * gridspacing.z - acgridspacing) * invacgridspacing;

  // compute ac grid index of upper corner plus gaussian radius
  int xabmax = (((blockIdx.x+1) * blockDim.x) * gridspacing.x + acgridspacing) * invacgridspacing;
  int yabmax = (((blockIdx.y+1) * blockDim.y) * gridspacing.y + acgridspacing) * invacgridspacing;
  int zabmax = ((z + (blockIdx.z+1) * blockDim.z * GTEXUNROLL) * gridspacing.z + acgridspacing) * invacgridspacing;

  xabmin = (xabmin < 0) ? 0 : xabmin;
  yabmin = (yabmin < 0) ? 0 : yabmin;
  zabmin = (zabmin < 0) ? 0 : zabmin;
  xabmax = (xabmax >= acncells.x-1) ? acncells.x-1 : xabmax;
  yabmax = (yabmax >= acncells.y-1) ? acncells.y-1 : yabmax;
  zabmax = (zabmax >= acncells.z-1) ? acncells.z-1 : zabmax;

  float coorx = gridspacing.x * xindex;
  float coory = gridspacing.y * yindex;
  float coorz = gridspacing.z * zindex;

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
            float dz2 = dz + gridspacing.z;
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
            float dz3 = dz2 + gridspacing.z;
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

            float dz4 = dz3 + gridspacing.z;
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

  densitygrid[outaddr          ] = densityval1;
  voltexmap[outaddr          ].x = densitycol1.x;
  voltexmap[outaddr          ].y = densitycol1.y;
  voltexmap[outaddr          ].z = densitycol1.z;

#if GTEXUNROLL >= 2
  int planesz = numvoxels.x * numvoxels.y;
  densitygrid[outaddr + planesz] = densityval2;
  voltexmap[outaddr + planesz].x = densitycol2.x;
  voltexmap[outaddr + planesz].y = densitycol2.y;
  voltexmap[outaddr + planesz].z = densitycol2.z;
#endif
#if GTEXUNROLL >= 4
  densitygrid[outaddr + 2*planesz] = densityval3;
  voltexmap[outaddr + 2*planesz].x = densitycol3.x;
  voltexmap[outaddr + 2*planesz].y = densitycol3.y;
  voltexmap[outaddr + 2*planesz].z = densitycol3.z;

  densitygrid[outaddr + 3*planesz] = densityval4;
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


__global__ static void gaussdensity_fast(int natoms,
                                         const float4 *sorted_xyzr, 
                                         int3 numvoxels,
                                         int3 acncells,
                                         float acgridspacing,
                                         float invacgridspacing,
                                         const uint2 * cellStartEnd,
                                         float gridspacing, unsigned int z, 
                                         float *densitygrid,
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
  
  densitygrid[outaddr            ] = densityval1;
#if GUNROLL >= 2
  int planesz = numvoxels.x * numvoxels.y;
  densitygrid[outaddr +   planesz] = densityval2;
#endif
#if GUNROLL >= 4
  densitygrid[outaddr + 2*planesz] = densityval3;
  densitygrid[outaddr + 3*planesz] = densityval4;
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

__global__ static void gaussdensity_fast(int natoms,
                                         const float4 *sorted_xyzr, 
                                         int3 numvoxels,
                                         int3 acncells,
                                         float acgridspacing,
                                         float invacgridspacing,
                                         const uint2 * cellStartEnd,
                                         float3 gridspacing, unsigned int z, 
                                         float *densitygrid,
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
  int xabmin = ((blockIdx.x * blockDim.x) * gridspacing.x - acgridspacing) * invacgridspacing;
  int yabmin = ((blockIdx.y * blockDim.y) * gridspacing .y- acgridspacing) * invacgridspacing;
  int zabmin = ((z + blockIdx.z * blockDim.z * GUNROLL) * gridspacing.z - acgridspacing) * invacgridspacing;

  // compute ac grid index of upper corner plus gaussian radius
  int xabmax = (((blockIdx.x+1) * blockDim.x) * gridspacing.x + acgridspacing) * invacgridspacing;
  int yabmax = (((blockIdx.y+1) * blockDim.y) * gridspacing.y + acgridspacing) * invacgridspacing;
  int zabmax = ((z + (blockIdx.z+1) * blockDim.z * GUNROLL) * gridspacing.z + acgridspacing) * invacgridspacing;

  xabmin = (xabmin < 0) ? 0 : xabmin;
  yabmin = (yabmin < 0) ? 0 : yabmin;
  zabmin = (zabmin < 0) ? 0 : zabmin;
  xabmax = (xabmax >= acncells.x-1) ? acncells.x-1 : xabmax;
  yabmax = (yabmax >= acncells.y-1) ? acncells.y-1 : yabmax;
  zabmax = (zabmax >= acncells.z-1) ? acncells.z-1 : zabmax;

  float coorx = gridspacing.x * xindex;
  float coory = gridspacing.y * yindex;
  float coorz = gridspacing.z * zindex;

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
            float dz2 = dz + gridspacing.z;
            float r22 = (dxy2 + dz2*dz2) * atom.w;
            densityval2 += exp2f(r22);
            // store nearest neighbor
            if( (dxy2 + dz2*dz2) < minDist2 ) {
                minDist2 = (dxy2 + dz2*dz2);
                neighbor2 = atomid;
            }
#endif
#if GUNROLL >= 4
            float dz3 = dz2 + gridspacing.z;
            float r23 = (dxy2 + dz3*dz3) * atom.w;
            densityval3 += exp2f(r23);
            // store nearest neighbor
            if( (dxy2 + dz3*dz3) < minDist3 ) {
                minDist3 = (dxy2 + dz3*dz3);
                neighbor3 = atomid;
            }

            float dz4 = dz3 + gridspacing.z;
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
  
  densitygrid[outaddr            ] = densityval1;
#if GUNROLL >= 2
  int planesz = numvoxels.x * numvoxels.y;
  densitygrid[outaddr +   planesz] = densityval2;
#endif
#if GUNROLL >= 4
  densitygrid[outaddr + 2*planesz] = densityval3;
  densitygrid[outaddr + 3*planesz] = densityval4;
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


__global__ static void wendlanddensity_fast(int natoms,
                                         const float4 *sorted_xyzr, 
                                         int3 numvoxels,
                                         int3 acncells,
                                         float acgridspacing,
                                         float invacgridspacing,
                                         const uint2 * cellStartEnd,
                                         float3 gridspacing, unsigned int z, 
                                         float *densitygrid,
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
  int xabmin = ((blockIdx.x * blockDim.x) * gridspacing.x - acgridspacing) * invacgridspacing;
  int yabmin = ((blockIdx.y * blockDim.y) * gridspacing .y- acgridspacing) * invacgridspacing;
  int zabmin = ((z + blockIdx.z * blockDim.z * GUNROLL) * gridspacing.z - acgridspacing) * invacgridspacing;

  // compute ac grid index of upper corner plus gaussian radius
  int xabmax = (((blockIdx.x+1) * blockDim.x) * gridspacing.x + acgridspacing) * invacgridspacing;
  int yabmax = (((blockIdx.y+1) * blockDim.y) * gridspacing.y + acgridspacing) * invacgridspacing;
  int zabmax = ((z + (blockIdx.z+1) * blockDim.z * GUNROLL) * gridspacing.z + acgridspacing) * invacgridspacing;

  xabmin = (xabmin < 0) ? 0 : xabmin;
  yabmin = (yabmin < 0) ? 0 : yabmin;
  zabmin = (zabmin < 0) ? 0 : zabmin;
  xabmax = (xabmax >= acncells.x-1) ? acncells.x-1 : xabmax;
  yabmax = (yabmax >= acncells.y-1) ? acncells.y-1 : yabmax;
  zabmax = (zabmax >= acncells.z-1) ? acncells.z-1 : zabmax;

  float coorx = gridspacing.x * xindex;
  float coory = gridspacing.y * yindex;
  float coorz = gridspacing.z * zindex;

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
            float r = length( make_float3( dx, dy, dz));

            // TODO fix this!
            float cutoff = 3.1; // Einheit: Partikelabstaende!
            float h=cutoff/2.0;
            float q=r/h;
            float pi = 3.141592653589793f;

            // TODO mass!! m_j

            if(r<=cutoff)
                densityval1 += 21.0f/16.0f/pi/h/h/h * (1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(2.0f*q+1.0f);

            // store nearest neighbor
            if( (dxy2 + dz*dz) < minDist1 ) {
                minDist1 = (dxy2 + dz*dz);
                neighbor1 = atomid;
            }

#if GUNROLL >= 2
            float dz2 = dz + gridspacing.z;
            r = length( make_float3( dx, dy, dz2));
            q = r/h;
            if(r<=cutoff)
                densityval2 += 21.0f/16.0f/pi/h/h/h * (1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(2.0f*q+1.0f);
            // store nearest neighbor
            if( (dxy2 + dz2*dz2) < minDist2 ) {
                minDist2 = (dxy2 + dz2*dz2);
                neighbor2 = atomid;
            }
#endif
#if GUNROLL >= 4
            float dz3 = dz2 + gridspacing.z;
            r = length( make_float3( dx, dy, dz3));
            q = r/h;
            if(r<=cutoff)
                densityval3 += 21.0f/16.0f/pi/h/h/h * (1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(2.0f*q+1.0f);
            // store nearest neighbor
            if( (dxy2 + dz3*dz3) < minDist3 ) {
                minDist3 = (dxy2 + dz3*dz3);
                neighbor3 = atomid;
            }

            float dz4 = dz3 + gridspacing.z;
            r = length( make_float3( dx, dy, dz4));
            q = r/h;
            if(r<=cutoff)
                densityval4 += 21.0f/16.0f/pi/h/h/h * (1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(2.0f*q+1.0f);
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
  
  densitygrid[outaddr            ] = densityval1;
#if GUNROLL >= 2
  int planesz = numvoxels.x * numvoxels.y;
  densitygrid[outaddr +   planesz] = densityval2;
#endif
#if GUNROLL >= 4
  densitygrid[outaddr + 2*planesz] = densityval3;
  densitygrid[outaddr + 3*planesz] = densityval4;
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

__global__ static void wendlanddensity_fast_tex(int natoms,
                                         const float4 *sorted_xyzr, 
                                         const float4 *sorted_color, 
                                         int3 numvoxels,
                                         int3 acncells,
                                         float acgridspacing,
                                         float invacgridspacing,
                                         const uint2 * cellStartEnd,
                                         float3 gridspacing, unsigned int z, 
                                         float *densitygrid,
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
  int xabmin = ((blockIdx.x * blockDim.x) * gridspacing.x - acgridspacing) * invacgridspacing;
  int yabmin = ((blockIdx.y * blockDim.y) * gridspacing.y - acgridspacing) * invacgridspacing;
  int zabmin = ((z + blockIdx.z * blockDim.z * GTEXUNROLL) * gridspacing.z - acgridspacing) * invacgridspacing;

  // compute ac grid index of upper corner plus gaussian radius
  int xabmax = (((blockIdx.x+1) * blockDim.x) * gridspacing.x + acgridspacing) * invacgridspacing;
  int yabmax = (((blockIdx.y+1) * blockDim.y) * gridspacing.y + acgridspacing) * invacgridspacing;
  int zabmax = ((z + (blockIdx.z+1) * blockDim.z * GTEXUNROLL) * gridspacing.z + acgridspacing) * invacgridspacing;

  xabmin = (xabmin < 0) ? 0 : xabmin;
  yabmin = (yabmin < 0) ? 0 : yabmin;
  zabmin = (zabmin < 0) ? 0 : zabmin;
  xabmax = (xabmax >= acncells.x-1) ? acncells.x-1 : xabmax;
  yabmax = (yabmax >= acncells.y-1) ? acncells.y-1 : yabmax;
  zabmax = (zabmax >= acncells.z-1) ? acncells.z-1 : zabmax;

  float coorx = gridspacing.x * xindex;
  float coory = gridspacing.y * yindex;
  float coorz = gridspacing.z * zindex;

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
            
            float r = length( make_float3( dx, dy, dz));

            // TODO fix this!
            float cutoff = 3.1; // Einheit: Partikelabstaende!
            float h=cutoff/2.0;
            float q=r/h;
            float pi = 3.141592653589793f;
            float tmp1;

            if(r<=cutoff)
                tmp1 = 21.0f/16.0f/pi/h/h/h * (1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(2.0f*q+1.0f);
            else
                tmp1 = 0.0f;

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
            float dz2 = dz + gridspacing.z;
            float tmp2;
            r = length( make_float3( dx, dy, dz2));
            q = r/h;
            if(r<=cutoff)
                tmp2 = 21.0f/16.0f/pi/h/h/h * (1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(2.0f*q+1.0f);
            else
                tmp2 = 0.0f;
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
            float dz3 = dz2 + gridspacing.z;
            float tmp3;
            r = length( make_float3( dx, dy, dz3));
            q = r/h;
            if(r<=cutoff)
                tmp3 = 21.0f/16.0f/pi/h/h/h * (1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(2.0f*q+1.0f);
            else
                tmp3 = 0.0f;
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

            float dz4 = dz3 + gridspacing.z;
            float tmp4;
            r = length( make_float3( dx, dy, dz4));
            q = r/h;
            if(r<=cutoff)
                tmp4 = 21.0f/16.0f/pi/h/h/h * (1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(1.0f - q/2.0f)*(2.0f*q+1.0f);
            else
                tmp4 = 0.0f;
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

  densitygrid[outaddr          ] = densityval1;
  voltexmap[outaddr          ].x = densitycol1.x;
  voltexmap[outaddr          ].y = densitycol1.y;
  voltexmap[outaddr          ].z = densitycol1.z;

#if GTEXUNROLL >= 2
  int planesz = numvoxels.x * numvoxels.y;
  densitygrid[outaddr + planesz] = densityval2;
  voltexmap[outaddr + planesz].x = densitycol2.x;
  voltexmap[outaddr + planesz].y = densitycol2.y;
  voltexmap[outaddr + planesz].z = densitycol2.z;
#endif
#if GTEXUNROLL >= 4
  densitygrid[outaddr + 2*planesz] = densityval3;
  voltexmap[outaddr + 2*planesz].x = densitycol3.x;
  voltexmap[outaddr + 2*planesz].y = densitycol3.y;
  voltexmap[outaddr + 2*planesz].z = densitycol3.z;

  densitygrid[outaddr + 3*planesz] = densityval4;
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


// per-GPU handle with various memory buffer pointers, etc.
typedef struct {
  /// max grid sizes and attributes the current allocations will support
  long int natoms;
  int colorperatom;
  int gx;
  int gy;
  int gz;

  CUDAMarchingCubes *mc;     ///< Marching cubes class used to extract surface

  float *devdensity;         ///< density map stored in GPU memory
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
  float *v3f_d;
  float *n3f_d;
  float *c3f_d;
  
  GLuint v3f_vbo;
  GLuint n3f_vbo;
  GLuint c3f_vbo;

  //cudaGraphicsResource *v3f_res;
  //cudaGraphicsResource *n3f_res;
  //cudaGraphicsResource *c3f_res;

} qsurf_gpuhandle;


CUDAQuickSurfAlternative::CUDAQuickSurfAlternative() {
  voidgpu = calloc(1, sizeof(qsurf_gpuhandle));
  useGaussKernel = true;
  surfaceArea = 0.0f;
//  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
}


CUDAQuickSurfAlternative::~CUDAQuickSurfAlternative() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

  // free all working buffers if not done already
  free_bufs();

  // delete marching cubes object
  delete gpuh->mc;

  free(voidgpu);
}


int CUDAQuickSurfAlternative::free_bufs() {
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

  if (gpuh->devdensity != NULL)
    cudaFree(gpuh->devdensity);
  gpuh->devdensity=NULL;

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

  if (gpuh->v3f_d != NULL)
    cudaFree(gpuh->v3f_d);
  gpuh->v3f_d=NULL;

  if (gpuh->n3f_d != NULL)
    cudaFree(gpuh->n3f_d);
  gpuh->n3f_d=NULL;

  if (gpuh->c3f_d != NULL)
    cudaFree(gpuh->c3f_d);
  gpuh->c3f_d=NULL;

  //GL
  if (gpuh->v3f_vbo) {
      cudaGLUnregisterBufferObject( gpuh->v3f_vbo);
      glBindBuffer(1, gpuh->v3f_vbo);
      glDeleteBuffers(1, &gpuh->v3f_vbo);
      //cudaGraphicsUnregisterResource( gpuh->v3f_res);
  }
  if (gpuh->n3f_vbo) {
      cudaGLUnregisterBufferObject( gpuh->n3f_vbo);
      glBindBuffer(1, gpuh->n3f_vbo);
      glDeleteBuffers(1, &gpuh->n3f_vbo);
      //cudaGraphicsUnregisterResource( gpuh->n3f_res);
  }
  if (gpuh->c3f_vbo) {
      cudaGLUnregisterBufferObject( gpuh->c3f_vbo);
      glBindBuffer(1, gpuh->c3f_vbo);
      glDeleteBuffers(1, &gpuh->c3f_vbo);
      //cudaGraphicsUnregisterResource( gpuh->c3f_res);
  }
  
  return 0;
}


int CUDAQuickSurfAlternative::free_bufs_map() {
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

  if (gpuh->devdensity != NULL)
    cudaFree(gpuh->devdensity);
  gpuh->devdensity=NULL;

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


int CUDAQuickSurfAlternative::check_bufs(long int natoms, int colorperatom,
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

int CUDAQuickSurfAlternative::alloc_bufs(long int natoms, int colorperatom,
                              int gx, int gy, int gz) {

  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

  // early exit from allocation call if we've already got existing
  // buffers that are large enough to support the request
  if (check_bufs(natoms, colorperatom, gx, gy, gz) == 0)
    return 0;

  // If we have any existing allocations, trash them as they weren't
  // usable for this new request and we need to reallocate them from scratch
  free_bufs();

  long int ncells = gx * gy * gz;
  long int volmemsz = ncells * sizeof(float);

  // Allocate all of the memory buffers our algorithms will need up-front,
  // so we can retry and gracefully reduce the sizes of various buffers
  // to attempt to fit within available GPU memory 
  cudaMalloc((void**)&gpuh->devdensity, volmemsz);
  if (colorperatom) {
    cudaMalloc((void**)&gpuh->devvoltexmap, 3*volmemsz);
    cudaMalloc((void**)&gpuh->color_d, natoms * sizeof(float4));
    cudaMalloc((void**)&gpuh->sorted_color_d, natoms * sizeof(float4));
  }
  cudaMalloc((void**)&gpuh->xyzr_d, natoms * sizeof(float4));
  cudaMalloc((void**)&gpuh->sorted_xyzr_d, natoms * sizeof(float4));
  cudaMalloc((void**)&gpuh->atomIndex_d, natoms * sizeof(unsigned int));
  cudaMalloc((void**)&gpuh->sorted_atomIndex_d, natoms * sizeof(unsigned int));
  cudaMalloc((void**)&gpuh->atomHash_d, natoms * sizeof(unsigned int));
  cudaMalloc((void**)&gpuh->cellStartEnd_d, ncells * sizeof(uint2));

  // allocate marching cubes output buffers
  int chunkmaxverts = 3 * ncells;
  cudaMalloc((void**)&gpuh->v3f_d, 3 * chunkmaxverts * sizeof(float4));
  cudaMalloc((void**)&gpuh->n3f_d, 3 * chunkmaxverts * sizeof(float4));
  cudaMalloc((void**)&gpuh->c3f_d, 3 * chunkmaxverts * sizeof(float4));

    // GL
  if( ogl_IsVersionGEQ(2,0) != GL_TRUE ) {
        return -1;
    }
    glGenBuffers( 1, &gpuh->v3f_vbo);
    glGenBuffers( 1, &gpuh->n3f_vbo);
    glGenBuffers( 1, &gpuh->c3f_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, gpuh->v3f_vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * chunkmaxverts * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, gpuh->n3f_vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * chunkmaxverts * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, gpuh->c3f_vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * chunkmaxverts * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    //cudaError_t reg1 = cudaGraphicsGLRegisterBuffer(&gpuh->v3f_res, gpuh->v3f_vbo, cudaGraphicsMapFlagsWriteDiscard);
    //cudaError_t reg2 = cudaGraphicsGLRegisterBuffer(&gpuh->n3f_res, gpuh->n3f_vbo, cudaGraphicsMapFlagsWriteDiscard);
    //cudaError_t reg3 = cudaGraphicsGLRegisterBuffer(&gpuh->c3f_res, gpuh->c3f_vbo, cudaGraphicsMapFlagsWriteDiscard);
    cudaGLRegisterBufferObject( gpuh->v3f_vbo);
    cudaGLRegisterBufferObject( gpuh->n3f_vbo);
    cudaGLRegisterBufferObject( gpuh->c3f_vbo);

  // Allocate an extra phantom array to act as a safety net to
  // ensure that subsequent allocations performed internally by 
  // the NVIDIA thrust template library or by our 
  // marching cubes implementation don't fail, since we can't 
  // currently pre-allocate all of them.
  cudaMalloc(&gpuh->safety, natoms*sizeof(float4) +           // thrust
             8 * gx * gy * sizeof(float) +                    // thrust
             CUDAMarchingCubes::MemUsageMC(gx, gy, gz));      // mcubes
  
  cudaError_t err = cudaGetLastError();
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


int CUDAQuickSurfAlternative::alloc_bufs_map(long int natoms, int colorperatom,
                                  int gx, int gy, int gz,
                                  bool storeNearestAtom) {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

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
  cudaMalloc((void**)&gpuh->devdensity, volmemsz);
  cudaMemset(gpuh->devdensity, 0, volmemsz);
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
  
  cudaError_t err = cudaGetLastError();
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


int CUDAQuickSurfAlternative::get_chunk_bufs(int testexisting,
                                  long int natoms, int colorperatom, 
                                  int gx, int gy, int gz,
                                  int &cx, int &cy, int &cz,
                                  int &sx, int &sy, int &sz) {
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
      if (alloc_bufs(natoms, colorperatom, cx, cy, cz) != 0)
        continue;
    }

    chunkallocated=1;
  }

  return 0;
}

int CUDAQuickSurfAlternative::get_chunk_bufs_map(int testexisting,
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

int CUDAQuickSurfAlternative::calc_surf(long int natoms, const float *xyzr_f,
                             const float *colors_f,
                             int colorperatom,
                             float *origin, int *numvoxels, float maxrad,
                             float radscale, float gridspacing, 
                             float isovalue, float gausslim,
                             int &numverts, float *&v, float *&n, float *&c,
                             int &numfacets, int *&f) {
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

  int chunkmaxverts=0;
  int chunknumverts=0; 
  numverts=0;
  numfacets=0;

  wkf_timerhandle globaltimer = wkf_timer_create();
  wkf_timer_start(globaltimer);

#ifdef CUDA_TIMER
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0);
#endif

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
      get_chunk_bufs(1, natoms, colorperatom,
                     volsz.x, volsz.y, volsz.z,
                     chunksz.x, chunksz.y, chunksz.z,
                     slabsz.x, slabsz.y, slabsz.z) == -1) {
    // reset the chunksz and slabsz after failing to try and
    // fit them into the existing allocations...
    chunksz = volsz;
    slabsz = volsz;

    // reallocate the chunk buffers from scratch since we weren't
    // able to reuse them
    if (get_chunk_bufs(0, natoms, colorperatom,
                       volsz.x, volsz.y, volsz.z,
                       chunksz.x, chunksz.y, chunksz.z,
                       slabsz.x, slabsz.y, slabsz.z) == -1) {
      wkf_timer_destroy(globaltimer);
      free(xyzr);
      return -1;
    }
  }
  chunkmaxverts = 3 * chunksz.x * chunksz.y * chunksz.z;

  // Free the "safety padding" memory we allocate to ensure we dont
  // have trouble with thrust calls that allocate their own memory later
  if (gpuh->safety != NULL)
    cudaFree(gpuh->safety);
  gpuh->safety = NULL;

#if 0
  if (chunkiters > 1)
    printf("  Using GPU chunk size: %d\n", chunksz.z);

  printf("  Accel grid(%d, %d, %d) spacing %f\n",
         accelcells.x, accelcells.y, accelcells.z, acgridspacing);
#endif

  cudaMemcpy(gpuh->xyzr_d, xyzr, natoms * sizeof(float4), cudaMemcpyHostToDevice);
  if (colorperatom)
    cudaMemcpy(gpuh->color_d, colors, natoms * sizeof(float4), cudaMemcpyHostToDevice);
  free(xyzr);
 
  // build uniform grid acceleration structure
  if (vmd_cuda_build_density_atom_grid_alt(natoms, gpuh->xyzr_d, gpuh->color_d,
                                       gpuh->sorted_xyzr_d,
                                       gpuh->sorted_color_d,
                                       gpuh->atomIndex_d, gpuh->sorted_atomIndex_d,
                                       gpuh->atomHash_d, gpuh->cellStartEnd_d, 
                                       accelcells, 1.0f / acgridspacing) != 0) {
    wkf_timer_destroy(globaltimer);
    free_bufs();
    return -1;
  }

  double sorttime = wkf_timer_timenow(globaltimer);
  double lastlooptime = sorttime;

  double densitykerneltime = 0.0f;
  double densitytime = 0.0f;
  double mckerneltime = 0.0f;
  double mctime = 0.0f; 
  double copycalltime = 0.0f;
  double copytime = 0.0f;

  float *volslab_d = NULL;
  float *texslab_d = NULL;

  int lzplane = GBLOCKSZZ * GUNROLL;
  if (colorperatom)
    lzplane = GTEXBLOCKSZZ * GTEXUNROLL;

  // initialize CUDA marching cubes class instance or rebuild it if needed
  uint3 mgsz = make_uint3(chunksz.x, chunksz.y, chunksz.z);
  if (gpuh->mc == NULL) {
    gpuh->mc = new CUDAMarchingCubes(); 
    if (!gpuh->mc->Initialize(mgsz)) {
      printf("MC Initialize() failed\n");
    }
  } else {
    uint3 mcmaxgridsize = gpuh->mc->GetMaxGridSize();
	if (slabsz.x <= (int)mcmaxgridsize.x &&
		slabsz.y <= (int)mcmaxgridsize.y &&
		slabsz.z <= (int)mcmaxgridsize.z) {
#if VERBOSE
      printf("Reusing MC object...\n");
#endif
    } else {
      printf("*** Allocating new MC object...\n");
      // delete marching cubes object
      delete gpuh->mc;

      // create and initialize CUDA marching cubes class instance
      gpuh->mc = new CUDAMarchingCubes(); 

      if (!gpuh->mc->Initialize(mgsz)) {
        printf("MC Initialize() failed while recreating MC object\n");
      }
    } 
  }
  
#ifdef WRITE_FILE
  //FILE *vertexFile = fopen( "vertices.dat", "wb");
  //FILE *indexFile = fopen( "indices.dat", "wb");
  FILE *objFile = fopen( "frame.obj", "w");
  unsigned int globalIndexCounter = 0;
#endif // WRITE_FILE

#ifdef WRITE_DATRAW_FILE
  FILE *qsDatFile = fopen( "qsvolume.dat", "w");
  FILE *qsRawFile = fopen( "qsvolume.raw", "wb");
#endif // WRITE_DATRAW_FILE

  int z;
  int chunkcount=0;
  this->surfaceArea = 0.0f;
  for (z=0; z<volsz.z; z+=slabsz.z) {
    int3 curslab = slabsz;
    if (z+curslab.z > volsz.z)
      curslab.z = volsz.z - z; 
  
    int slabplanesz = curslab.x * curslab.y;

    dim3 Gsz((curslab.x+Bsz.x-1) / Bsz.x, 
             (curslab.y+Bsz.y-1) / Bsz.y,
             (curslab.z+(Bsz.z*GUNROLL)-1) / (Bsz.z * GUNROLL));
    if (colorperatom)
      Gsz.z = (curslab.z+(Bsz.z*GTEXUNROLL)-1) / (Bsz.z * GTEXUNROLL);

    // For SM 2.x, we can run the entire slab in one pass by launching
    // a 3-D grid of thread blocks.
    // If we are running on SM 1.x, we can only launch 1-D grids so we
    // must loop over planar grids until we have processed the whole slab.
    dim3 Gszslice = Gsz;
    if (deviceProp.major < 2)
      Gszslice.z = 1;

#if VERBOSE
    printf("CUDA device %d, grid size %dx%dx%d\n", 
           0, Gsz.x, Gsz.y, Gsz.z);
    printf("CUDA: vol(%d,%d,%d) accel(%d,%d,%d)\n",
           curslab.x, curslab.y, curslab.z,
           accelcells.x, accelcells.y, accelcells.z);
    printf("Z=%d, curslab.z=%d\n", z, curslab.z);
#endif

    // For all but the first density slab, we copy the last four 
    // planes of the previous run into the start of the next run so
    // that we can extract the isosurface with no discontinuities
    if (z == 0) {
      volslab_d = gpuh->devdensity;
      if (colorperatom)
        texslab_d = gpuh->devvoltexmap;
    } else {
      cudaMemcpy(gpuh->devdensity,
                 volslab_d + (slabsz.z-4) * slabplanesz, 
                 4 * slabplanesz * sizeof(float), cudaMemcpyDeviceToDevice);
      if (colorperatom)
        cudaMemcpy(gpuh->devvoltexmap,
                   texslab_d + (slabsz.z-4) * 3 * slabplanesz, 
                   4*3 * slabplanesz * sizeof(float), cudaMemcpyDeviceToDevice);

      volslab_d = gpuh->devdensity + (4 * slabplanesz);
      if (colorperatom)
        texslab_d = gpuh->devvoltexmap + (4 * 3 * slabplanesz);
    }

    for (int lz=0; (int)lz<Gsz.z; lz+=Gszslice.z) {
      int lzinc = lz * lzplane;
      float *volslice_d = volslab_d + lzinc * slabplanesz;

      if (colorperatom) {
        float *texslice_d = texslab_d + lzinc * slabplanesz * 3;
        gaussdensity_fast_tex<<<Gszslice, Bsz, 0>>>(natoms, 
            gpuh->sorted_xyzr_d, gpuh->sorted_color_d, 
            curslab, accelcells, acgridspacing,
            1.0f / acgridspacing, gpuh->cellStartEnd_d, gridspacing, z+lzinc,
            volslice_d, (float3 *) texslice_d, 1.0f / isovalue, false, 0, 0);
      } else {
        gaussdensity_fast<<<Gszslice, Bsz, 0>>>(natoms, 
            gpuh->sorted_xyzr_d, 
            curslab, accelcells, acgridspacing, 1.0f / acgridspacing, 
            gpuh->cellStartEnd_d, gridspacing, z+lzinc, volslice_d, false, 0, 0);
      }
    }
    cudaThreadSynchronize(); 
	getLastCudaError("kernel failed");
    densitykerneltime = wkf_timer_timenow(globaltimer);
    
#ifdef CUDA_TIMER		
    cudaDeviceSynchronize();
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for sort + density: %f ms\n", time);
#endif

#if VERBOSE
    printf("  CUDA mcubes..."); fflush(stdout);
#endif

#ifdef CUDA_TIMER
    cudaEventRecord( start, 0);
#endif

    int vsz[3];
    vsz[0]=curslab.x;
    vsz[1]=curslab.y;
    vsz[2]=curslab.z;

    // For all but the first density slab, we copy the last four
    // planes of the previous run into the start of the next run so
    // that we can extract the isosurface with no discontinuities
    if (z != 0)
      vsz[2]=curslab.z + 4;

    float bbox[3];
    bbox[0] = vsz[0] * gridspacing;
    bbox[1] = vsz[1] * gridspacing;
    bbox[2] = vsz[2] * gridspacing;


    float gorigin[3];
    gorigin[0] = origin[0];
    gorigin[1] = origin[1];
    gorigin[2] = origin[2] + (z * gridspacing);

    if (z != 0)
      gorigin[2] = origin[2] + ((z-4) * gridspacing);

#if VERBOSE
printf("\n  ... vsz: %d %d %d\n", vsz[0], vsz[1], vsz[2]);
printf("  ... org: %.2f %.2f %.2f\n", gorigin[0], gorigin[1], gorigin[2]);
printf("  ... bxs: %.2f %.2f %.2f\n", bbox[0], bbox[1], bbox[2]);
printf("  ... bbe: %.2f %.2f %.2f\n", 
  gorigin[0]+bbox[0], gorigin[1]+bbox[1], gorigin[2]+bbox[2]);
#endif

    // If we are computing the volume using multiple passes, we have to 
    // overlap the marching cubes grids and compute a sub-volume to exclude
    // the end planes, except for the first and last sub-volume, in order to
    // get correct per-vertex normals at the edges of each sub-volume 
    int skipstartplane=0;
    int skipendplane=0;
    if (chunksz.z < volsz.z) {
      // on any but the first pass, we skip the first Z plane
      if (z != 0)
        skipstartplane=1;

      // on any but the last pass, we skip the last Z plane
      if (z+curslab.z < volsz.z)
        skipendplane=1;
    }

    //
    // Extract density map isosurface using marching cubes
    //
    uint3 gvsz = make_uint3(vsz[0], vsz[1], vsz[2]);
    float3 gorg = make_float3(gorigin[0], gorigin[1], gorigin[2]);
    float3 gbnds = make_float3(bbox[0], bbox[1], bbox[2]);
    
    gpuh->mc->SetIsovalue(isovalue);
    if (!gpuh->mc->SetVolumeData(gpuh->devdensity, gpuh->devvoltexmap, 
                                 gvsz, gorg, gbnds, true)) {
      printf("MC SetVolumeData() failed\n");
      err = cudaGetLastError();
      // If an error occured, we print it
      if (err != cudaSuccess)
          printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    // set the sub-volume starting/ending indices if needed
    if (skipstartplane || skipendplane) {
      uint3 volstart = make_uint3(0, 0, 0);
      uint3 volend = make_uint3(gvsz.x, gvsz.y, gvsz.z);

      if (skipstartplane)
        volstart.z = 2;

      if (skipendplane)
        volend.z = gvsz.z - 2;

      gpuh->mc->SetSubVolume(volstart, volend);
    }
    
#ifdef WRITE_DATRAW_FILE
    float *vol;
    vol = new float[gvsz.x * gvsz.y * gvsz.z];
    // copy
    cudaMemcpy( vol, gpuh->devdensity, gvsz.x * gvsz.y * gvsz.z * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    fprintf( qsDatFile, "ObjectFileName: qsvolume.raw\n");
    fprintf( qsDatFile, "TaggedFileName: ---\n");
    fprintf( qsDatFile, "Resolution:     %i %i %i\n", gvsz.x, gvsz.y, gvsz.z);
    fprintf( qsDatFile, "SliceThickness: 1 1 1\n");
    fprintf( qsDatFile, "Format:         FLOAT\n");
    fprintf( qsDatFile, "NbrTags:        0\n");
    fprintf( qsDatFile, "ObjectType:     TEXTURE_VOLUME_OBJECT\n");
    fprintf( qsDatFile, "ObjectModel:    DENSITY\n");
    fprintf( qsDatFile, "GridType:       EQUIDISTANT\n");
    fflush( qsDatFile);
    fwrite( vol, sizeof(float), gvsz.x * gvsz.y * gvsz.z, qsRawFile);
    fflush( qsRawFile);
    delete[] vol;
#endif // WRITE_DATRAW_FILE

//#ifndef CUDA_ARRAY
//    // map VBOs for writing
//    size_t num_bytes;
//    cudaGraphicsMapResources(1, &gpuh->v3f_res, 0);
//    cudaGraphicsResourceGetMappedPointer((void**)&gpuh->v3f_d, &num_bytes, gpuh->v3f_res);
//    cudaGraphicsMapResources(1, &gpuh->n3f_res, 0);
//    cudaGraphicsResourceGetMappedPointer((void**)&gpuh->n3f_d, &num_bytes, gpuh->n3f_res);
//    cudaGraphicsMapResources(1, &gpuh->c3f_res, 0);
//    cudaGraphicsResourceGetMappedPointer((void**)&gpuh->c3f_d, &num_bytes, gpuh->c3f_res);
//#endif

    gpuh->mc->computeIsosurface((float3 *) gpuh->v3f_d, (float3 *) gpuh->n3f_d, 
                                (float3 *) gpuh->c3f_d, chunkmaxverts);

    chunknumverts = gpuh->mc->GetVertexCount();
    
    // TEST compute surface area
    this->surfaceArea += gpuh->mc->computeSurfaceArea((float3*)gpuh->v3f_d, chunknumverts/3);

//#ifndef CUDA_ARRAY
//    // unmap VBOs
//    cudaGraphicsUnmapResources(1, &gpuh->v3f_res, 0);
//    cudaGraphicsUnmapResources(1, &gpuh->n3f_res, 0);
//    cudaGraphicsUnmapResources(1, &gpuh->c3f_res, 0);
//#endif

    if (chunknumverts == chunkmaxverts)
      printf("  *** Exceeded marching cubes vertex limit (%d verts)\n", chunknumverts);

    cudaThreadSynchronize(); 
    mckerneltime = wkf_timer_timenow(globaltimer);

    //int l;
    //int vertstart = 3 * numverts;
    //int vertbufsz = 3 * (numverts + chunknumverts) * sizeof(float);
    //int facebufsz = (numverts + chunknumverts) * sizeof(int);
    int chunkvertsz = 3 * chunknumverts * sizeof(float);

//#ifdef CUDA_ARRAY
//    v = (float*) realloc(v, vertbufsz);
//    n = (float*) realloc(n, vertbufsz);
//    c = (float*) realloc(c, vertbufsz);
//    f = (int*)   realloc(f, facebufsz);
//    cudaMemcpy(v+vertstart, gpuh->v3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
//    cudaMemcpy(n+vertstart, gpuh->n3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
//    if (colorperatom) {
//      cudaMemcpy(c+vertstart, gpuh->c3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
//    } else {
//      float *color = c+vertstart;
//      for (l=0; l<chunknumverts*3; l+=3) {
//        color[l + 0] = colors[0].x;
//        color[l + 1] = colors[0].y;
//        color[l + 2] = colors[0].z;
//      }
//    }
//    for (l=numverts; l<numverts+chunknumverts; l++) {
//      f[l]=l;
//    }
//#else
#if 1
    // map VBOs for writing
    //size_t num_bytes;
    float *v3f, *n3f, *c3f;
    //cudaGraphicsMapResources(1, &gpuh->v3f_res, 0);
    //cudaGraphicsResourceGetMappedPointer((void**)&v3f, &num_bytes, gpuh->v3f_res);
    //cudaGraphicsMapResources(1, &gpuh->n3f_res, 0);
    //cudaGraphicsResourceGetMappedPointer((void**)&n3f, &num_bytes, gpuh->n3f_res);
    //cudaGraphicsMapResources(1, &gpuh->c3f_res, 0);
    //cudaGraphicsResourceGetMappedPointer((void**)&c3f, &num_bytes, gpuh->c3f_res);
    cudaGLMapBufferObject( (void **)&v3f, gpuh->v3f_vbo);
    cudaGLMapBufferObject( (void **)&n3f, gpuh->n3f_vbo);
    cudaGLMapBufferObject( (void **)&c3f, gpuh->c3f_vbo);

    // copy
    cudaMemcpy( v3f, gpuh->v3f_d, chunkvertsz, cudaMemcpyDeviceToDevice);
    cudaMemcpy( n3f, gpuh->n3f_d, chunkvertsz, cudaMemcpyDeviceToDevice);
    cudaMemcpy( c3f, gpuh->c3f_d, chunkvertsz, cudaMemcpyDeviceToDevice);
    
#ifdef WRITE_FILE
    float *v, *n, *c;
    v = new float[3 * chunknumverts];
    //n = new float[3 * chunknumverts];
    //c = new float[3 * chunknumverts];
    //Vertex *vb = new Vertex[chunknumverts];
    //unsigned int *ib = new unsigned int[chunknumverts];
    // copy
    cudaMemcpy( v, gpuh->v3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
    //cudaMemcpy( n, gpuh->n3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
    //cudaMemcpy( c, gpuh->c3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    for( unsigned int i = 0; i < chunknumverts; i++ ) {
        //vb[i].pos[0] = v[i*3+0];
        //vb[i].pos[1] = v[i*3+1];
        //vb[i].pos[2] = v[i*3+2];
        //vb[i].normal[0] = n[i*3+0];
        //vb[i].normal[1] = n[i*3+1];
        //vb[i].normal[2] = n[i*3+2];
        //vb[i].color[0] = c[i*3+0];
        //vb[i].color[1] = c[i*3+1];
        //vb[i].color[2] = c[i*3+2];
        //vb[i].color[3] = 0.1f;
        //ib[i] = globalIndexCounter;
        //globalIndexCounter++;
        fprintf( objFile, "v %f %f %f\n", v[i*3+0], v[i*3+1], v[i*3+2]);
    }
    for( unsigned int i = 0; i < chunknumverts / 3; i++ ) {
        //vb[i].pos[0] = v[i*3+0];
        //vb[i].pos[1] = v[i*3+1];
        //vb[i].pos[2] = v[i*3+2];
        //vb[i].normal[0] = n[i*3+0];
        //vb[i].normal[1] = n[i*3+1];
        //vb[i].normal[2] = n[i*3+2];
        //vb[i].color[0] = c[i*3+0];
        //vb[i].color[1] = c[i*3+1];
        //vb[i].color[2] = c[i*3+2];
        //vb[i].color[3] = 0.1f;
        //ib[i] = globalIndexCounter;
        fprintf( objFile, "f %i %i %i\n", globalIndexCounter+1, globalIndexCounter+2, globalIndexCounter+3);
        globalIndexCounter+=3;
    }
    unsigned int sizeofvertex = sizeof(Vertex);
    //fwrite( vb, sizeof(Vertex), chunknumverts, vertexFile);
    //fflush( vertexFile);
    //fwrite( vb, sizeof(Vertex), chunknumverts, objFile);
    //fflush( objFile);
    //unsigned int writtenBytes = fwrite( ib, sizeof(unsigned int), chunknumverts, indexFile); 
    //fflush( indexFile);
    //fwrite( ib, sizeof(unsigned int), chunknumverts, objFile); 
    fflush( objFile);
    delete[] v;
    //delete[] n;
    //delete[] c;
    //delete[] vb;
    //delete[] ib;
#endif // WRITE_FILE

    // unmap VBOs
    //cudaGraphicsUnmapResources(1, &gpuh->v3f_res, 0);
    //cudaGraphicsUnmapResources(1, &gpuh->n3f_res, 0);
    //cudaGraphicsUnmapResources(1, &gpuh->c3f_res, 0);
    cudaGLUnmapBufferObject( gpuh->v3f_vbo);
    cudaGLUnmapBufferObject( gpuh->n3f_vbo);
    cudaGLUnmapBufferObject( gpuh->c3f_vbo);

    // GL
    glBindBuffer(GL_ARRAY_BUFFER, gpuh->v3f_vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, gpuh->n3f_vbo);
    glNormalPointer(GL_FLOAT, 0, 0);
    glEnableClientState(GL_NORMAL_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, gpuh->c3f_vbo);
    glColorPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_COLOR_ARRAY);

    glColor3f(0.9f, 0.6f, 0.2f);
    glDrawArrays(GL_TRIANGLES, 0, chunknumverts);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif

    numverts+=chunknumverts;
    numfacets+=chunknumverts/3;
    copycalltime = wkf_timer_timenow(globaltimer);

    densitytime += densitykerneltime - lastlooptime;
    mctime += mckerneltime - densitykerneltime;
    copytime += copycalltime - mckerneltime;

    lastlooptime = wkf_timer_timenow(globaltimer);

    chunkcount++; // increment number of chunks processed
  }
  
#ifdef WRITE_FILE
  //fclose( vertexFile);
  //fclose( indexFile);
  fclose( objFile);
#endif
  
#ifdef WRITE_DATRAW_FILE
  fclose( qsDatFile);
  fclose( qsRawFile);
#endif // WRITE_DATRAW_FILE

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
  printf("  GPU generated %d vertices, %d facets, in %d passes\n", numverts, numfacets, chunkcount);

  printf("  GPU time (%s): %.3f [sort: %.3f density %.3f mcubes: %.3f copy: %.3f]\n", 
         (deviceProp.major == 1 && deviceProp.minor == 3) ? "SM 1.3" : "SM 2.x",
         totalruntime, sorttime, densitytime, mctime, copytime);

  printf("  Total surface area: %.0f\n", this->surfaceArea);
#endif
  
#ifdef CUDA_TIMER
    cudaDeviceSynchronize();
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for MC: %f ms\n", time);
#endif


  return 0;
}

int CUDAQuickSurfAlternative::calc_surf(long int natoms, const float *xyzr_f,
                             const float *colors_f,
                             int colorperatom,
                             float *origin, int *numvoxels, float maxrad,
                             float radscale, float3 gridspacing, 
                             float isovalue, float gausslim,
                             int &numverts, float *&v, float *&n, float *&c,
                             int &numfacets, int *&f) {
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

  int chunkmaxverts=0;
  int chunknumverts=0; 
  numverts=0;
  numfacets=0;

  wkf_timerhandle globaltimer = wkf_timer_create();
  wkf_timer_start(globaltimer);
  
#ifdef CUDA_TIMER
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0);
#endif

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
  float maxGridSpacing = std::max(gridspacing.x, std::max(gridspacing.y, gridspacing.z));
  if (acgridspacing < maxGridSpacing)
      acgridspacing = maxGridSpacing;


  // Allocate output arrays for the gaussian density map and 3-D texture map
  // We test for errors carefully here since this is the most likely place
  // for a memory allocation failure due to the size of the grid.
  int3 chunksz = volsz;
  int3 slabsz = volsz;

  int3 accelcells;
  accelcells.x = max(int((volsz.x*maxGridSpacing) / acgridspacing), 1);
  accelcells.y = max(int((volsz.y*maxGridSpacing) / acgridspacing), 1);
  accelcells.z = max(int((volsz.z*maxGridSpacing) / acgridspacing), 1);

  dim3 Bsz(GBLOCKSZX, GBLOCKSZY, GBLOCKSZZ);
  if (colorperatom)
    Bsz.z = GTEXBLOCKSZZ;

  // check to see if it's possible to use an existing allocation,
  // if so, just leave things as they are, and do the computation 
  // using the existing buffers
  if (gpuh->natoms == 0 ||
      get_chunk_bufs(1, natoms, colorperatom,
                     volsz.x, volsz.y, volsz.z,
                     chunksz.x, chunksz.y, chunksz.z,
                     slabsz.x, slabsz.y, slabsz.z) == -1) {
    // reset the chunksz and slabsz after failing to try and
    // fit them into the existing allocations...
    chunksz = volsz;
    slabsz = volsz;

    // reallocate the chunk buffers from scratch since we weren't
    // able to reuse them
    if (get_chunk_bufs(0, natoms, colorperatom,
                       volsz.x, volsz.y, volsz.z,
                       chunksz.x, chunksz.y, chunksz.z,
                       slabsz.x, slabsz.y, slabsz.z) == -1) {
      wkf_timer_destroy(globaltimer);
      free(xyzr);
      return -1;
    }
  }
  chunkmaxverts = 3 * chunksz.x * chunksz.y * chunksz.z;

  // Free the "safety padding" memory we allocate to ensure we dont
  // have trouble with thrust calls that allocate their own memory later
  if (gpuh->safety != NULL)
    cudaFree(gpuh->safety);
  gpuh->safety = NULL;

#if 0
  if (chunkiters > 1)
    printf("  Using GPU chunk size: %d\n", chunksz.z);

  printf("  Accel grid(%d, %d, %d) spacing %f\n",
         accelcells.x, accelcells.y, accelcells.z, acgridspacing);
#endif

  cudaMemcpy(gpuh->xyzr_d, xyzr, natoms * sizeof(float4), cudaMemcpyHostToDevice);
  if (colorperatom)
    cudaMemcpy(gpuh->color_d, colors, natoms * sizeof(float4), cudaMemcpyHostToDevice);
  free(xyzr);
 
  // build uniform grid acceleration structure
  if (vmd_cuda_build_density_atom_grid_alt(natoms, gpuh->xyzr_d, gpuh->color_d,
                                       gpuh->sorted_xyzr_d,
                                       gpuh->sorted_color_d,
                                       gpuh->atomIndex_d, gpuh->sorted_atomIndex_d,
                                       gpuh->atomHash_d, gpuh->cellStartEnd_d, 
                                       accelcells, 1.0f / acgridspacing) != 0) {
    wkf_timer_destroy(globaltimer);
    free_bufs();
    return -1;
  }

  double sorttime = wkf_timer_timenow(globaltimer);
  double lastlooptime = sorttime;

  double densitykerneltime = 0.0f;
  double densitytime = 0.0f;
  double mckerneltime = 0.0f;
  double mctime = 0.0f; 
  double copycalltime = 0.0f;
  double copytime = 0.0f;

  float *volslab_d = NULL;
  float *texslab_d = NULL;

  int lzplane = GBLOCKSZZ * GUNROLL;
  if (colorperatom)
    lzplane = GTEXBLOCKSZZ * GTEXUNROLL;

  // initialize CUDA marching cubes class instance or rebuild it if needed
  uint3 mgsz = make_uint3(chunksz.x, chunksz.y, chunksz.z);
  if (gpuh->mc == NULL) {
    gpuh->mc = new CUDAMarchingCubes(); 
    if (!gpuh->mc->Initialize(mgsz)) {
      printf("MC Initialize() failed\n");
    }
  } else {
    uint3 mcmaxgridsize = gpuh->mc->GetMaxGridSize();
    if (slabsz.x <= mcmaxgridsize.x &&
        slabsz.y <= mcmaxgridsize.y &&
        slabsz.z <= mcmaxgridsize.z) {
#if VERBOSE
      printf("Reusing MC object...\n");
#endif
    } else {
      printf("*** Allocating new MC object...\n");
      // delete marching cubes object
      delete gpuh->mc;

      // create and initialize CUDA marching cubes class instance
      gpuh->mc = new CUDAMarchingCubes(); 

      if (!gpuh->mc->Initialize(mgsz)) {
        printf("MC Initialize() failed while recreating MC object\n");
      }
    } 
  }

#ifdef WRITE_FILE
  //FILE *vertexFile = fopen( "vertices.dat", "wb");
  //FILE *indexFile = fopen( "indices.dat", "wb");
  FILE *objFile = fopen( "frame.obj", "w");
  unsigned int globalIndexCounter = 0;
#endif // WRITE_FILE

  int z;
  int chunkcount=0;
  this->surfaceArea = 0.0f;
  for (z=0; z<volsz.z; z+=slabsz.z) {
    int3 curslab = slabsz;
    if (z+curslab.z > volsz.z)
      curslab.z = volsz.z - z; 
  
    int slabplanesz = curslab.x * curslab.y;

    dim3 Gsz((curslab.x+Bsz.x-1) / Bsz.x, 
             (curslab.y+Bsz.y-1) / Bsz.y,
             (curslab.z+(Bsz.z*GUNROLL)-1) / (Bsz.z * GUNROLL));
    if (colorperatom)
      Gsz.z = (curslab.z+(Bsz.z*GTEXUNROLL)-1) / (Bsz.z * GTEXUNROLL);

    // For SM 2.x, we can run the entire slab in one pass by launching
    // a 3-D grid of thread blocks.
    // If we are running on SM 1.x, we can only launch 1-D grids so we
    // must loop over planar grids until we have processed the whole slab.
    dim3 Gszslice = Gsz;
    if (deviceProp.major < 2)
      Gszslice.z = 1;

#if VERBOSE
    printf("CUDA device %d, grid size %dx%dx%d\n", 
           0, Gsz.x, Gsz.y, Gsz.z);
    printf("CUDA: vol(%d,%d,%d) accel(%d,%d,%d)\n",
           curslab.x, curslab.y, curslab.z,
           accelcells.x, accelcells.y, accelcells.z);
    printf("Z=%d, curslab.z=%d\n", z, curslab.z);
#endif

    // For all but the first density slab, we copy the last four 
    // planes of the previous run into the start of the next run so
    // that we can extract the isosurface with no discontinuities
    if (z == 0) {
      volslab_d = gpuh->devdensity;
      if (colorperatom)
        texslab_d = gpuh->devvoltexmap;
    } else {
      cudaMemcpy(gpuh->devdensity,
                 volslab_d + (slabsz.z-4) * slabplanesz, 
                 4 * slabplanesz * sizeof(float), cudaMemcpyDeviceToDevice);
      if (colorperatom)
        cudaMemcpy(gpuh->devvoltexmap,
                   texslab_d + (slabsz.z-4) * 3 * slabplanesz, 
                   4*3 * slabplanesz * sizeof(float), cudaMemcpyDeviceToDevice);

      volslab_d = gpuh->devdensity + (4 * slabplanesz);
      if (colorperatom)
        texslab_d = gpuh->devvoltexmap + (4 * 3 * slabplanesz);
    }

	for (int lz = 0; lz<(int)Gsz.z; lz += Gszslice.z) {
      int lzinc = lz * lzplane;
      float *volslice_d = volslab_d + lzinc * slabplanesz;

      if (colorperatom) {
        float *texslice_d = texslab_d + lzinc * slabplanesz * 3;
        if (useGaussKernel) {
          gaussdensity_fast_tex<<<Gszslice, Bsz, 0>>>(natoms, 
              gpuh->sorted_xyzr_d, gpuh->sorted_color_d, 
              curslab, accelcells, acgridspacing,
              1.0f / acgridspacing, gpuh->cellStartEnd_d, gridspacing, z+lzinc,
              volslice_d, (float3 *) texslice_d, 1.0f / isovalue, false, 0, 0);
        } else {
          wendlanddensity_fast_tex<<<Gszslice, Bsz, 0>>>(natoms, 
              gpuh->sorted_xyzr_d, gpuh->sorted_color_d, 
              curslab, accelcells, acgridspacing,
              1.0f / acgridspacing, gpuh->cellStartEnd_d, gridspacing, z+lzinc,
              volslice_d, (float3 *) texslice_d, 1.0f / isovalue, false, 0, 0);
        }
      } else {
        if (useGaussKernel) {
          gaussdensity_fast<<<Gszslice, Bsz, 0>>>(natoms, 
              gpuh->sorted_xyzr_d, 
              curslab, accelcells, acgridspacing, 1.0f / acgridspacing, 
              gpuh->cellStartEnd_d, gridspacing, z+lzinc, volslice_d, false, 0, 0);
        } else {
          wendlanddensity_fast<<<Gszslice, Bsz, 0>>>(natoms, 
              gpuh->sorted_xyzr_d, 
              curslab, accelcells, acgridspacing, 1.0f / acgridspacing, 
              gpuh->cellStartEnd_d, gridspacing, z+lzinc, volslice_d, false, 0, 0);
        }
      }
    }
    cudaThreadSynchronize(); 
	getLastCudaError("kernel failed");
    densitykerneltime = wkf_timer_timenow(globaltimer);
    
#ifdef CUDA_TIMER		
    cudaDeviceSynchronize();
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for sort + density: %f ms\n", time);
#endif

#if VERBOSE
    printf("  CUDA mcubes..."); fflush(stdout);
#endif

#ifdef CUDA_TIMER
    cudaEventRecord( start, 0);
#endif

    int vsz[3];
    vsz[0]=curslab.x;
    vsz[1]=curslab.y;
    vsz[2]=curslab.z;

    // For all but the first density slab, we copy the last four
    // planes of the previous run into the start of the next run so
    // that we can extract the isosurface with no discontinuities
    if (z != 0)
      vsz[2]=curslab.z + 4;

    float bbox[3];
    bbox[0] = vsz[0] * gridspacing.x;
    bbox[1] = vsz[1] * gridspacing.y;
    bbox[2] = vsz[2] * gridspacing.z;


    float gorigin[3];
    gorigin[0] = origin[0];
    gorigin[1] = origin[1];
    gorigin[2] = origin[2] + (z * gridspacing.z);

    if (z != 0)
      gorigin[2] = origin[2] + ((z-4) * gridspacing.z);

#if VERBOSE
printf("\n  ... vsz: %d %d %d\n", vsz[0], vsz[1], vsz[2]);
printf("  ... org: %.2f %.2f %.2f\n", gorigin[0], gorigin[1], gorigin[2]);
printf("  ... bxs: %.2f %.2f %.2f\n", bbox[0], bbox[1], bbox[2]);
printf("  ... bbe: %.2f %.2f %.2f\n", 
  gorigin[0]+bbox[0], gorigin[1]+bbox[1], gorigin[2]+bbox[2]);
#endif

    // If we are computing the volume using multiple passes, we have to 
    // overlap the marching cubes grids and compute a sub-volume to exclude
    // the end planes, except for the first and last sub-volume, in order to
    // get correct per-vertex normals at the edges of each sub-volume 
    int skipstartplane=0;
    int skipendplane=0;
    if (chunksz.z < volsz.z) {
      // on any but the first pass, we skip the first Z plane
      if (z != 0)
        skipstartplane=1;

      // on any but the last pass, we skip the last Z plane
      if (z+curslab.z < volsz.z)
        skipendplane=1;
    }

    //
    // Extract density map isosurface using marching cubes
    //
    uint3 gvsz = make_uint3(vsz[0], vsz[1], vsz[2]);
    float3 gorg = make_float3(gorigin[0], gorigin[1], gorigin[2]);
    float3 gbnds = make_float3(bbox[0], bbox[1], bbox[2]);

    gpuh->mc->SetIsovalue(isovalue);
    if (!gpuh->mc->SetVolumeData(gpuh->devdensity, gpuh->devvoltexmap, 
                                 gvsz, gorg, gbnds, true)) {
      printf("MC SetVolumeData() failed\n");
      err = cudaGetLastError();
      // If an error occured, we print it
      if (err != cudaSuccess)
          printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    // set the sub-volume starting/ending indices if needed
    if (skipstartplane || skipendplane) {
      uint3 volstart = make_uint3(0, 0, 0);
      uint3 volend = make_uint3(gvsz.x, gvsz.y, gvsz.z);

      if (skipstartplane)
        volstart.z = 2;

      if (skipendplane)
        volend.z = gvsz.z - 2;

      gpuh->mc->SetSubVolume(volstart, volend);
    }
    
//#ifndef CUDA_ARRAY
//    // map VBOs for writing
//    size_t num_bytes;
//    cudaGraphicsMapResources(1, &gpuh->v3f_res, 0);
//    cudaGraphicsResourceGetMappedPointer((void**)&gpuh->v3f_d, &num_bytes, gpuh->v3f_res);
//    cudaGraphicsMapResources(1, &gpuh->n3f_res, 0);
//    cudaGraphicsResourceGetMappedPointer((void**)&gpuh->n3f_d, &num_bytes, gpuh->n3f_res);
//    cudaGraphicsMapResources(1, &gpuh->c3f_res, 0);
//    cudaGraphicsResourceGetMappedPointer((void**)&gpuh->c3f_d, &num_bytes, gpuh->c3f_res);
//#endif

    gpuh->mc->computeIsosurface((float3 *) gpuh->v3f_d, (float3 *) gpuh->n3f_d, 
                                (float3 *) gpuh->c3f_d, chunkmaxverts);

    chunknumverts = gpuh->mc->GetVertexCount();
    
    // TEST compute surface area
    this->surfaceArea += gpuh->mc->computeSurfaceArea((float3*)gpuh->v3f_d, chunknumverts/3);

//#ifndef CUDA_ARRAY
//    // unmap VBOs
//    cudaGraphicsUnmapResources(1, &gpuh->v3f_res, 0);
//    cudaGraphicsUnmapResources(1, &gpuh->n3f_res, 0);
//    cudaGraphicsUnmapResources(1, &gpuh->c3f_res, 0);
//#endif

    if (chunknumverts == chunkmaxverts)
      printf("  *** Exceeded marching cubes vertex limit (%d verts)\n", chunknumverts);

    cudaThreadSynchronize(); 
    mckerneltime = wkf_timer_timenow(globaltimer);

    //int l;
    //int vertstart = 3 * numverts;
    //int vertbufsz = 3 * (numverts + chunknumverts) * sizeof(float);
    int facebufsz = (numverts + chunknumverts) * sizeof(int);
    int chunkvertsz = 3 * chunknumverts * sizeof(float);

//#ifdef CUDA_ARRAY
//    v = (float*) realloc(v, vertbufsz);
//    n = (float*) realloc(n, vertbufsz);
//    c = (float*) realloc(c, vertbufsz);
//    f = (int*)   realloc(f, facebufsz);
//    cudaMemcpy(v+vertstart, gpuh->v3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
//    cudaMemcpy(n+vertstart, gpuh->n3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
//    if (colorperatom) {
//      cudaMemcpy(c+vertstart, gpuh->c3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
//    } else {
//      float *color = c+vertstart;
//      for (l=0; l<chunknumverts*3; l+=3) {
//        color[l + 0] = colors[0].x;
//        color[l + 1] = colors[0].y;
//        color[l + 2] = colors[0].z;
//      }
//    }
//    for (l=numverts; l<numverts+chunknumverts; l++) {
//      f[l]=l;
//    }
//#else
#if 1
    // map VBOs for writing
    //size_t num_bytes;
    float *v3f, *n3f, *c3f;
    //cudaGraphicsMapResources(1, &gpuh->v3f_res, 0);
    //cudaGraphicsResourceGetMappedPointer((void**)&v3f, &num_bytes, gpuh->v3f_res);
    //cudaGraphicsMapResources(1, &gpuh->n3f_res, 0);
    //cudaGraphicsResourceGetMappedPointer((void**)&n3f, &num_bytes, gpuh->n3f_res);
    //cudaGraphicsMapResources(1, &gpuh->c3f_res, 0);
    //cudaGraphicsResourceGetMappedPointer((void**)&c3f, &num_bytes, gpuh->c3f_res);
    cudaGLMapBufferObject( (void **)&v3f, gpuh->v3f_vbo);
    cudaGLMapBufferObject( (void **)&n3f, gpuh->n3f_vbo);
    cudaGLMapBufferObject( (void **)&c3f, gpuh->c3f_vbo);

    // copy
    cudaMemcpy( v3f, gpuh->v3f_d, chunkvertsz, cudaMemcpyDeviceToDevice);
    cudaMemcpy( n3f, gpuh->n3f_d, chunkvertsz, cudaMemcpyDeviceToDevice);
    cudaMemcpy( c3f, gpuh->c3f_d, chunkvertsz, cudaMemcpyDeviceToDevice);
    
#ifdef WRITE_FILE
    float *v, *n, *c;
    v = new float[3 * chunknumverts];
    //n = new float[3 * chunknumverts];
    //c = new float[3 * chunknumverts];
    Vertex *vb = new Vertex[chunknumverts];
    unsigned int *ib = new unsigned int[chunknumverts];
    // copy
    cudaMemcpy( v, gpuh->v3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
    //cudaMemcpy( n, gpuh->n3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
    //cudaMemcpy( c, gpuh->c3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for( unsigned int i = 0; i < chunknumverts; i++ ) {
        vb[i].pos[0] = v[i*3+0];
        vb[i].pos[1] = v[i*3+1];
        vb[i].pos[2] = v[i*3+2];
        //vb[i].normal[0] = n[i*3+0];
        //vb[i].normal[1] = n[i*3+1];
        //vb[i].normal[2] = n[i*3+2];
        //vb[i].color[0] = c[i*3+0];
        //vb[i].color[1] = c[i*3+1];
        //vb[i].color[2] = c[i*3+2];
        //vb[i].color[3] = 0.1f;
        ib[i] = globalIndexCounter;
        globalIndexCounter++;
    }
    unsigned int sizeofvertex = sizeof(Vertex);
    //fwrite( vb, sizeof(Vertex), chunknumverts, vertexFile);
    //fflush( vertexFile);
    //unsigned int writtenBytes = fwrite( ib, sizeof(unsigned int), chunknumverts, indexFile); 
    //fflush( indexFile);
    delete[] v;
    //delete[] n;
    //delete[] c;
    delete[] vb;
    delete[] ib;
#endif // WRITE_FILE

    // unmap VBOs
    //cudaGraphicsUnmapResources(1, &gpuh->v3f_res, 0);
    //cudaGraphicsUnmapResources(1, &gpuh->n3f_res, 0);
    //cudaGraphicsUnmapResources(1, &gpuh->c3f_res, 0);
    cudaGLUnmapBufferObject( gpuh->v3f_vbo);
    cudaGLUnmapBufferObject( gpuh->n3f_vbo);
    cudaGLUnmapBufferObject( gpuh->c3f_vbo);

    // GL
    glBindBuffer(GL_ARRAY_BUFFER, gpuh->v3f_vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, gpuh->n3f_vbo);
    glNormalPointer(GL_FLOAT, 0, 0);
    glEnableClientState(GL_NORMAL_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, gpuh->c3f_vbo);
    glColorPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_COLOR_ARRAY);

    glColor3f(0.9f, 0.6f, 0.2f);
    glDrawArrays(GL_TRIANGLES, 0, chunknumverts);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif

    numverts+=chunknumverts;
    numfacets+=chunknumverts/3;
    copycalltime = wkf_timer_timenow(globaltimer);

    densitytime += densitykerneltime - lastlooptime;
    mctime += mckerneltime - densitykerneltime;
    copytime += copycalltime - mckerneltime;

    lastlooptime = wkf_timer_timenow(globaltimer);

    chunkcount++; // increment number of chunks processed
  }
  
#ifdef WRITE_FILE
  //fclose( vertexFile);
  //fclose( indexFile);
  fclose(objFile);
#endif

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
  printf("  GPU generated %d vertices, %d facets, in %d passes\n", numverts, numfacets, chunkcount);

  printf("  GPU time (%s): %.3f [sort: %.3f density %.3f mcubes: %.3f copy: %.3f]\n", 
         (deviceProp.major == 1 && deviceProp.minor == 3) ? "SM 1.3" : "SM 2.x",
         totalruntime, sorttime, densitytime, mctime, copytime);

  printf("  Total surface area (%4i x %4i x %4i): %.0f\n", numvoxels[0], numvoxels[1], numvoxels[2], surfaceArea);
#endif
  
  
#ifdef CUDA_TIMER
    cudaDeviceSynchronize();
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for MC: %f ms\n", time);
#endif

  return 0;
}


/*
 * CUDAQuickSurf::calc_map
 */
int CUDAQuickSurfAlternative::calc_map(long int natoms, const float *xyzr_f,
                             const float *colors_f,
                             int colorperatom,
                             float *origin, int *numvoxels, float maxrad,
                             float radscale, float gridspacing, 
                             float isovalue, float gausslim, 
                             bool storeNearestAtom,
							 int fileIndex,
							 int resolution) {
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

//#if 1
//  if (chunkiters > 1)
//    printf("  Using GPU chunk size: %d\n", chunksz.z);
//
//  printf("  Accel grid(%d, %d, %d) spacing %f\n",
//         accelcells.x, accelcells.y, accelcells.z, acgridspacing);
//#endif

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
      setArrayToInt<<<grid, 256>>>( gridDim, gpuh->nearest_atom_d, -1);
	  getLastCudaError("kernel failed");
  }
  
  free(xyzr);
 
  // build uniform grid acceleration structure
  if (vmd_cuda_build_density_atom_grid_alt(natoms, gpuh->xyzr_d, gpuh->color_d,
                                       gpuh->sorted_xyzr_d,
                                       gpuh->sorted_color_d,
                                       gpuh->atomIndex_d, gpuh->sorted_atomIndex_d,
                                       gpuh->atomHash_d, gpuh->cellStartEnd_d, 
                                       accelcells, 1.0f / acgridspacing) != 0) {
    wkf_timer_destroy(globaltimer);
    free_bufs();
    return -1;
  }

  double sorttime = wkf_timer_timenow(globaltimer);
  double lastlooptime = sorttime;

  double densitykerneltime = 0.0f;
  //double densitytime = 0.0f;
  //double copycalltime = 0.0f;
  //double copytime = 0.0f;

  float *volslab_d = NULL;
  float *texslab_d = NULL;

  int lzplane = GBLOCKSZZ * GUNROLL;
  if (colorperatom)
    lzplane = GTEXBLOCKSZZ * GTEXUNROLL;

  int z;
  int chunkcount=0;
  for (z=0; z<volsz.z; z+=slabsz.z) {
    int3 curslab = slabsz;
    if (z+curslab.z > volsz.z)
      curslab.z = volsz.z - z; 
  
    int slabplanesz = curslab.x * curslab.y;

    dim3 Gsz((curslab.x+Bsz.x-1) / Bsz.x, 
             (curslab.y+Bsz.y-1) / Bsz.y,
             (curslab.z+(Bsz.z*GUNROLL)-1) / (Bsz.z * GUNROLL));
    if (colorperatom)
      Gsz.z = (curslab.z+(Bsz.z*GTEXUNROLL)-1) / (Bsz.z * GTEXUNROLL);

    // For SM 2.x, we can run the entire slab in one pass by launching
    // a 3-D grid of thread blocks.
    // If we are running on SM 1.x, we can only launch 1-D grids so we
    // must loop over planar grids until we have processed the whole slab.
    dim3 Gszslice = Gsz;
    if (deviceProp.major < 2)
      Gszslice.z = 1;

#if VERBOSE
    printf("CUDA device %d, grid size %dx%dx%d\n", 
           0, Gsz.x, Gsz.y, Gsz.z);
    printf("CUDA: vol(%d,%d,%d) accel(%d,%d,%d)\n",
           curslab.x, curslab.y, curslab.z,
           accelcells.x, accelcells.y, accelcells.z);
    printf("Z=%d, curslab.z=%d\n", z, curslab.z);
#endif

    // For all but the first density slab, we copy the last four 
    // planes of the previous run into the start of the next run so
    // that we can extract the isosurface with no discontinuities
    if (z == 0) {
      volslab_d = gpuh->devdensity;
      if (colorperatom)
        texslab_d = gpuh->devvoltexmap;
    } else {
      cudaMemcpy(gpuh->devdensity,
                 volslab_d + (slabsz.z-4) * slabplanesz, 
                 4 * slabplanesz * sizeof(float), cudaMemcpyDeviceToDevice);
      if (colorperatom)
        cudaMemcpy(gpuh->devvoltexmap,
                   texslab_d + (slabsz.z-4) * 3 * slabplanesz, 
                   4*3 * slabplanesz * sizeof(float), cudaMemcpyDeviceToDevice);

      volslab_d = gpuh->devdensity + (4 * slabplanesz);
      if (colorperatom)
        texslab_d = gpuh->devvoltexmap + (4 * 3 * slabplanesz);
    }

	for (int lz = 0; lz<(int)Gsz.z; lz += Gszslice.z) {
      int lzinc = lz * lzplane;
      float *volslice_d = volslab_d + lzinc * slabplanesz;

      if (colorperatom) {
        float *texslice_d = texslab_d + lzinc * slabplanesz * 3;
        gaussdensity_fast_tex<<<Gszslice, Bsz, 0>>>(natoms, 
            gpuh->sorted_xyzr_d, gpuh->sorted_color_d, 
            curslab, accelcells, acgridspacing,
            1.0f / acgridspacing, gpuh->cellStartEnd_d, gridspacing, z+lzinc,
            volslice_d, (float3 *) texslice_d, 1.0f / isovalue, storeNearestAtom, gpuh->nearest_atom_d, gpuh->atomIndex_d);
      } else {
        gaussdensity_fast<<<Gszslice, Bsz, 0>>>(natoms, 
            gpuh->sorted_xyzr_d, 
            curslab, accelcells, acgridspacing, 1.0f / acgridspacing, 
            gpuh->cellStartEnd_d, gridspacing, z+lzinc, volslice_d, storeNearestAtom, gpuh->nearest_atom_d, gpuh->atomIndex_d);
      }
    }
    cudaThreadSynchronize(); 
    densitykerneltime = wkf_timer_timenow(globaltimer);

    lastlooptime = wkf_timer_timenow(globaltimer);

    chunkcount++; // increment number of chunks processed
  }
  getLastCudaError("kernel failed");

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

#ifdef WRITE_DATRAW_FILE_MAP
  std::string name1 = "Volume_data/qsvolume_" + std::to_string(resolution) + "_" + std::to_string(fileIndex) + ".dat";
  std::string name2 = "Volume_data/qsvolume_" + std::to_string(resolution) + "_" + std::to_string(fileIndex) + ".raw";
  std::string name3 = "qsvolume_" + std::to_string(resolution) + "_" + std::to_string(fileIndex) + ".raw";

  FILE *qsDatFile = fopen(name1.c_str(), "w");
  FILE *qsRawFile = fopen(name2.c_str(), "wb");

  uint3 gvsz = make_uint3(numvoxels[0], numvoxels[1], numvoxels[2]);

  printf("size: %i %i %i\n", this->getMapSizeX(), this->getMapSizeY(), this->getMapSizeZ());

  float *vol;
  vol = new float[gvsz.x * gvsz.y * gvsz.z * 3];
  // copy
  checkCudaErrors(cudaMemcpy(vol, gpuh->devvoltexmap, gvsz.x * gvsz.y * gvsz.z * sizeof(float) * 3, cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  float thicknessX = 10.0f / (float)gvsz.x;
  float thicknessY = 10.0f / (float)gvsz.y;
  float thicknessZ = 10.0f / (float)gvsz.z;

  std::string filenameText = "ObjectFileName: " + name3 + "\n";
  fprintf(qsDatFile, filenameText.c_str());
  fprintf(qsDatFile, "Format:         FLOAT\n");
  fprintf(qsDatFile, "GridType:       EQUIDISTANT\n");
  fprintf(qsDatFile, "Components:     3\n");
  fprintf(qsDatFile, "Dimensions:     3\n");
  fprintf(qsDatFile, "TimeSteps:      1\n");
  fprintf(qsDatFile, "ByteOrder:      LITTLE_ENDIAN\n");
  fprintf(qsDatFile, "Resolution:     %i %i %i\n", gvsz.x, gvsz.y, gvsz.z);
  fprintf(qsDatFile, "SliceThickness: %1.5f %1.5f %1.5f\n", thicknessX, thicknessY, thicknessZ);
  fprintf(qsDatFile, "Origin:         -5.0 -5.0 0.0\n");
  fprintf(qsDatFile, "Time:           %i.0\n", fileIndex);
  fflush(qsDatFile);
  fwrite(vol, sizeof(float), gvsz.x * gvsz.y * gvsz.z * 3, qsRawFile);
  fflush(qsRawFile);
  delete[] vol;

  fclose(qsDatFile);
  fclose(qsRawFile);
#endif // WRITE_DATRAW_FILE_MAP

  getLastCudaError("kernel failed");

  return 0;
}

int CUDAQuickSurfAlternative::getMapSizeX() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
  return gpuh->gx;
}

int CUDAQuickSurfAlternative::getMapSizeY() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
  return gpuh->gy;
}

int CUDAQuickSurfAlternative::getMapSizeZ() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
  return gpuh->gz;
}

float* CUDAQuickSurfAlternative::getMap() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
  return gpuh->devdensity;
}

float* CUDAQuickSurfAlternative::getColorMap() {
    qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
    return gpuh->devvoltexmap;
}

int* CUDAQuickSurfAlternative::getNeighborMap() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
  return gpuh->nearest_atom_d;
}

/*void CUDAQuickSurf::setDensFilterVals(float rad, int minN) {
    qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;
    gpuh->nFilterMinNeighbours = minN;
    gpuh->nFilterSqrtRad = rad;
}*/


