/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/*
 * CUDA accelerated coulombic potential grid test code
 *   John E. Stone and Chris Rodrigues
 *   http://www.ks.uiuc.edu/~johns/
 */

#include <stdio.h>
#include <stdlib.h>

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return -1; }}

// max constant buffer size is 64KB, minus whatever
// the CUDA runtime and compiler are using that we don't know about
// At 16 bytes for atom, for this program 4070 atoms is about the max
// we can store in the constant buffer.
#define MAXATOMS 4000
__constant__ float4 atominfo[MAXATOMS];

#define UNROLLX       8
#define UNROLLY       1
#define BLOCKSIZEX   16
#define BLOCKSIZEY   16
#define BLOCKSIZE    BLOCKSIZEX * BLOCKSIZEY

// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.
//
// This kernel was written by Chris Rodrigues of Wen-mei's group
//
__global__ void cenergy(int numatoms, float gridspacing, float * energygrid) {
  unsigned int xindex  = __umul24(blockIdx.x, blockDim.x) * UNROLLX
                         + threadIdx.x;
  unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int outaddr = (__umul24(gridDim.x, blockDim.x) * UNROLLX) * yindex
                         + xindex;

  float coory = gridspacing * yindex;
  float coorx = gridspacing * xindex;

  float energyvalx1=0.0f;
  float energyvalx2=0.0f;
  float energyvalx3=0.0f;
  float energyvalx4=0.0f;
#if UNROLLX == 8
  float energyvalx5=0.0f;
  float energyvalx6=0.0f;
  float energyvalx7=0.0f;
  float energyvalx8=0.0f;
#endif

  float gridspacing_u = gridspacing * BLOCKSIZEX;

  //
  // XXX 59/8 FLOPS per atom
  //
  int atomid;
  for (atomid=0; atomid<numatoms; atomid++) {
    float dy = coory - atominfo[atomid].y;
    float dyz2 = (dy * dy) + atominfo[atomid].z;
    float atomq=atominfo[atomid].w;

    float dx1 = coorx - atominfo[atomid].x;
    float dx2 = dx1 + gridspacing_u;
    float dx3 = dx2 + gridspacing_u;
    float dx4 = dx3 + gridspacing_u;
#if UNROLLX == 8
    float dx5 = dx4 + gridspacing_u;
    float dx6 = dx5 + gridspacing_u;
    float dx7 = dx6 + gridspacing_u;
    float dx8 = dx7 + gridspacing_u;
#endif
    

    energyvalx1 += atomq * rsqrtf(dx1*dx1 + dyz2);
    energyvalx2 += atomq * rsqrtf(dx2*dx2 + dyz2);
    energyvalx3 += atomq * rsqrtf(dx3*dx3 + dyz2);
    energyvalx4 += atomq * rsqrtf(dx4*dx4 + dyz2);
#if UNROLLX == 8
    energyvalx5 += atomq * rsqrtf(dx5*dx5 + dyz2);
    energyvalx6 += atomq * rsqrtf(dx6*dx6 + dyz2);
    energyvalx7 += atomq * rsqrtf(dx7*dx7 + dyz2);
    energyvalx8 += atomq * rsqrtf(dx8*dx8 + dyz2);
#endif
  }

  energygrid[outaddr             ] += energyvalx1;
  energygrid[outaddr+1*BLOCKSIZEX] += energyvalx2;
  energygrid[outaddr+2*BLOCKSIZEX] += energyvalx3;
  energygrid[outaddr+3*BLOCKSIZEX] += energyvalx4;
#if UNROLLX == 8
  energygrid[outaddr+4*BLOCKSIZEX] += energyvalx5;
  energygrid[outaddr+5*BLOCKSIZEX] += energyvalx6;
  energygrid[outaddr+6*BLOCKSIZEX] += energyvalx7;
  energygrid[outaddr+7*BLOCKSIZEX] += energyvalx8;
#endif
}



int copyatomstoconstbuf(float *atoms, int count, float zplane) {
  CUERR // check and clear any existing errors

  if (count > MAXATOMS) {
    printf("Atom count exceeds constant buffer storage capacity\n");
    return -1;
  }

  float atompre[4*MAXATOMS];
  int i;
  for (i=0; i<count*4; i+=4) {
    atompre[i    ] = atoms[i    ];
    atompre[i + 1] = atoms[i + 1];
    float dz = zplane - atoms[i + 2];
    atompre[i + 2]  = dz*dz;
    atompre[i + 3] = atoms[i + 3];
  }

  cudaMemcpyToSymbol(atominfo, atompre, count * 4 * sizeof(float), 0);
  CUERR // check and clear any existing errors

  return 0;
}


int initatoms(float **atombuf, int count, dim3 volsize, float gridspacing) {
  dim3 size;
  int i;
  float *atoms;

  atoms = (float *) malloc(count * 4 * sizeof(float));
  *atombuf = atoms;

  // compute grid dimensions in angstroms
  size.x = (unsigned int) gridspacing * volsize.x;
  size.y = (unsigned int) gridspacing * volsize.y;
  size.z = (unsigned int) gridspacing * volsize.z;

  for (i=0; i<count; i++) {
    int addr = i * 4;
    atoms[addr    ] = (rand() / (float) RAND_MAX) * size.x; 
    atoms[addr + 1] = (rand() / (float) RAND_MAX) * size.y; 
    atoms[addr + 2] = (rand() / (float) RAND_MAX) * size.z; 
    atoms[addr + 3] = ((rand() / (float) RAND_MAX) * 2.0) - 1.0;  // charge
  }  

  return 0;
}
