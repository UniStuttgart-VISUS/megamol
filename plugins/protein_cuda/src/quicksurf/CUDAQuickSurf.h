/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *  $RCSfile: CUDAQuickSurf.h,v $
 *  $Author: johns $    $Locker:  $     $State: Exp $
 *  $Revision: 1.11 $   $Date: 2021/10/05 06:03:00 $
 *
 ***************************************************************************/
/**
 * \file CUDAQuickSurf.h
 * \brief CUDA accelerated QuickSurf gaussian density and surface calculation.
 *
 *  This work is described in the following papers:
 *
 *  "GPU-Accelerated Molecular Visualization on Petascale Supercomputing
 *   Platforms"
 *  John E. Stone, Kirby L. Vandivort, and Klaus Schulten.
 *  UltraVis'13: Proceedings of the 8th International Workshop on
 *  Ultrascale Visualization, pp. 6:1-6:8, 2013.
 *  http://dx.doi.org/10.1145/2535571.2535595
 *
 *  "Early Experiences Scaling VMD Molecular Visualization and
 *   Analysis Jobs on Blue Waters"
 *  John E. Stone, Barry Isralewitz, and Klaus Schulten.
 *  Extreme Scaling Workshop (XSW), pp. 43-50, 2013.
 *  http://dx.doi.org/10.1109/XSW.2013.10
 *
 *  "Fast Visualization of Gaussian Density Surfaces for
 *   Molecular Dynamics and Particle System Trajectories"
 *  Michael Krone, John E. Stone, Thomas Ertl, and Klaus Schulten.
 *  EuroVis - Short Papers, pp. 67-71, 2012.
 *  http://dx.doi.org/10.2312/PE/EuroVisShort/EuroVisShort2012/067-071
 */

#ifndef CUDAQUICKSURF_H
#define CUDAQUICKSURF_H

#include "glm/glm.hpp"
#include "glowl/BufferObject.hpp"
#include <memory>
#include <vector>

namespace megamol::protein_cuda {

struct QuickSurfGraphicBuffer {
    std::shared_ptr<glowl::BufferObject> positionBuffer_ = nullptr;
    std::shared_ptr<glowl::BufferObject> normalBuffer_ = nullptr;
    std::shared_ptr<glowl::BufferObject> colorBuffer_ = nullptr;
    int vertCount_ = 0;
};

class CUDAQuickSurf {
    void* voidgpu; ///< pointer to structs containing private per-GPU pointers

public:
    enum class VolTexFormat { RGB3F, RGB4U }; ///< which texture map format to use

    /** Ctor. */
    CUDAQuickSurf(void);

    /** Dtor. */
    ~CUDAQuickSurf(void);

    /**
     *
     */
    int calc_surf(long int natoms, const float* xyzr, const float* colors, int colorperatom, VolTexFormat vtexformat,
        float* origin, int* numvoxels, float maxrad, float radscale, float gridspacing, float isovalue, float gausslim,
        std::vector<QuickSurfGraphicBuffer>& meshResult);

    /**
     *
     */
    int calc_map(long int natoms, const float* xyzr_f, const float* colors_f, int colorperatom, VolTexFormat vtexformat,
        float* origin, int* numvoxels, float maxrad, float radscale, float gridspacing, float isovalue, float gausslim,
        bool storeNearestAtom);

    /**
     *
     */
    glm::ivec3 getMapSize();

    /**
     *
     */
    float* getMap();

    /**
     *
     */
    float* getColorMap();

    /**
     *
     */
    int* getNeighborMap();

private:
    /**
     *
     */
    int free_bufs(void);

    /**
     *
     */
    int check_bufs(
        long int natoms, int colorperatom, VolTexFormat vtexformat, int acx, int acy, int acz, int gx, int gy, int gz);

    /**
     *
     */
    int alloc_bufs(
        long int natoms, int colorperatom, VolTexFormat vtexformat, int acx, int acy, int acz, int gx, int gy, int gz);

    /**
     *
     */
    int alloc_bufs_map(long int natoms, int colorperatom, VolTexFormat vtexformat, int acx, int acy, int acz, int gx,
        int gy, int gz, bool storeNearestAtom = false);

    /**
     *
     */
    int get_chunk_bufs(int testexisting, long int natoms, int colorperatom, VolTexFormat vtexformat, int acx, int acy,
        int acz, int gx, int gy, int gz, int& cx, int& cy, int& cz, int& sx, int& sy, int& sz);

    /**
     *
     */
    int get_chunk_bufs_map(int testexisting, long int natoms, int colorperatom, VolTexFormat vtexformat, int acx,
        int acy, int acz, int gx, int gy, int gz, int& cx, int& cy, int& cz, int& sx, int& sy, int& sz,
        bool storeNearestAtom = false);
};

} // namespace megamol::protein_cuda

#endif
