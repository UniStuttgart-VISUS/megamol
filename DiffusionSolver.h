//
// DiffusionSolver.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 16, 2013
// Author     : scharnkn
//

#ifndef MMPROTEINPLUGIN_DIFFUSIONSOLVER_H_INCLUDED
#define MMPROTEINPLUGIN_DIFFUSIONSOLVER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#ifdef WITH_CUDA

namespace megamol {
namespace protein {

/*
 * This class provides static methods for numerically solving diffusion
 * equations, such as the Gradient Vector Flow (GVF).
 */
class DiffusionSolver {

public:


    /**
     * CUDA implementation of an iterative method to compute the Gradient
     * Vector Field (GVF) based on a given vector field.
     * Note: The input vector field 'v' is also used as output array. TODO
     *
     * @param v      The vector field (also used as output)
     * @param dim    The dimensions of the vector field
     * @param maxIt  The maximum number of iterations
     * @return 'True' on success, 'false' otherwise
     */
    static bool CalcGVF(
            const float *startVol, float *gvfConstData_D, const unsigned int *cellStates_D,
                    float *grad_D, size_t dim[3], float isovalue,
                    float *gvf_D, unsigned int maxIt, float scl);

protected:

    /** TODO */
    static dim3 Grid(const unsigned int size, const int threadsPerBlock);

    /**
     * Initializes the texture for the isotropic diffusion used in the
     * GVF computation.
     *
     * @param startVol_D  The initial volume texture provided by the used
     *                    (device memory). This can be any arbitrary volume
     *                    texture containing a level set.
     * @param isovalue    The isovalue defining the level set in 'startVol'.
     * @param radius      The sampling radius for the volume.
     * @param gvfVol_D    The resoluting volume that can then be used in the
     *                    GVF computation.
     */
    static bool initGVF(
            const float *startVol, size_t dim[3], const unsigned int *cellStates_D,
                    float isovalue, float *grad_D, float *gvfConstData_D);

private:

};

} // namespace protein
} // namespace megamol

#endif // WITH_CUDA
#endif // MMPROTEINPLUGIN_DIFFUSIONSOLVER_H_INCLUDED
