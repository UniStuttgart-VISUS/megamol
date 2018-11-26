#ifndef CUDAQUICKSURFARRAY_H
#define CUDAQUICKSURFARRAY_H

#include <vector_types.h>

// Write file for Daniel Kauker (puxel)
//#define WRITE_FILE

class CUDAQuickSurfArray {
  void *voidgpu; ///< pointer to structs containing private per-GPU pointers 

public:

struct Vertex
{
     float pos[3];
     //float normal[3];
     //float color[4];
};

  CUDAQuickSurfArray(void);
  
  int free_bufs_map(void);
  
  int check_bufs(long int natoms, int colorperatom, 
                 int gx, int gy, int gz);
  
  int alloc_bufs_map(long int natoms, int colorperatom, 
                     int gx, int gy, int gz,
                     bool storeNearestAtom = false);
  
  int get_chunk_bufs_map(int testexisting,
                     long int natoms, int colorperatom, 
                     int gx, int gy, int gz,
                     int &cx, int &cy, int &cz,
                     int &sx, int &sy, int &sz,
                     bool storeNearestAtom = false);
  
  int calc_map(long int natoms, const float *xyzr, const float *colors,
               int colorperatom, float *origin, int* numvoxels, float maxrad,
               float radscale, float gridspacing,
               float isovalue, float gausslim, bool storeNearestAtom = false);
  
  int getMapSizeX();
  int getMapSizeY();
  int getMapSizeZ();
  cudaArray* getMap();
  float* getColorMap();
  int* getNeighborMap();
  float surfaceArea;

  ~CUDAQuickSurfArray(void);
};

#endif

