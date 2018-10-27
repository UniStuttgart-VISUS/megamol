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
 *	$RCSfile: CUDAQuickSurf.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.2 $	$Date: 2011/12/03 20:24:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Fast gaussian surface representation
 ***************************************************************************/
#ifndef CUDAQUICKSURF_H
#define CUDAQUICKSURF_H

#include <vector_types.h>
#include <driver_types.h>


// Write file for Daniel Kauker (puxel)
//#define WRITE_FILE

typedef unsigned int uint;

struct triangleCustom {
	float3 v1;
	float3 v2;
	float3 v3;
};

/*
* greater than comparison for int2 (compares x values)
*/
struct greater_triangleCustom {
	__device__
		bool operator()(const triangleCustom& lhs, const triangleCustom& rhs) const;
};

class CUDAQuickSurf {
  void *voidgpu; ///< pointer to structs containing private per-GPU pointers 

public:

struct Vertex
{
     float pos[3];
     //float normal[3];
     //float color[4];
};

  CUDAQuickSurf(void);
  
  int free_bufs(void);
  
  int free_bufs_map(void);
  
  int check_bufs(long int natoms, int colorperatom, 
                 int gx, int gy, int gz);
  
  int alloc_bufs(long int natoms, int colorperatom, 
                 int gx, int gy, int gz);
  
  int alloc_bufs_map(long int natoms, int colorperatom, 
                     int gx, int gy, int gz,
                     bool storeNearestAtom = false);
  
  int get_chunk_bufs(int testexisting,
                     long int natoms, int colorperatom, 
                     int gx, int gy, int gz,
                     int &cx, int &cy, int &cz,
                     int &sx, int &sy, int &sz);
  
  int get_chunk_bufs_map(int testexisting,
                     long int natoms, int colorperatom, 
                     int gx, int gy, int gz,
                     int &cx, int &cy, int &cz,
                     int &sx, int &sy, int &sz,
                     bool storeNearestAtom = false);
  
  int calc_surf(long int natoms, const float *xyzr, const float *colors,
                int colorperatom, float *origin, int* numvoxels, float maxrad,
                float radscale, float gridspacing,
                float isovalue, float gausslim,
                int &numverts, float *&v, float *&n, float *&c,
                int &numfacets, int *&f, bool sortTriangles = false);

  int calc_surf(long int natoms, const float *xyzr, const float *colors,
                int colorperatom, float *origin, int* numvoxels, float maxrad,
                float radscale, float3 gridspacing,
                float isovalue, float gausslim,
                int &numverts, float *&v, float *&n, float *&c,
                int &numfacets, int *&f, bool sortTriangles = false);

  int calc_map(long int natoms, const float *xyzr, const float *colors,
               int colorperatom, float *origin, int* numvoxels, float maxrad,
               float radscale, float gridspacing,
               float isovalue, float gausslim, bool storeNearestAtom = false);
  
  //int calc_map_alt(long int natoms, const float *xyzr, const float *colors,
  //             int colorperatom, float *origin, int* numvoxels, float maxrad,
  //             float radscale, float gridspacing,
  //             float isovalue, float gausslim);

  int getMapSizeX();
  int getMapSizeY();
  int getMapSizeZ();
  float* getMap();
  float* getColorMap();
  int* getNeighborMap();
  bool useGaussKernel;
  float surfaceArea;

  cudaError_t copyCamPosToDevice(float3 camPos);

  //void setDensFilterVals(float rad, int minN);

  ~CUDAQuickSurf(void);
private:
	cudaError SortTrianglesDevice(uint triaCnt, triangleCustom * vertices, 
		triangleCustom* verticesCopy, triangleCustom* colors, triangleCustom* normals);
};

#endif

