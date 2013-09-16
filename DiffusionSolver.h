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

namespace megamol {
namespace protein {

/*
 * This class provides static methods for numerically solving diffusion
 * equations, such as the Gradient Vector Flow (GVF).
 */
class DiffusionSolver {

public:

#ifdef WITH_CUDA
    /**
     * CUDA implementation of an iterative method to compute the Gradient
     * Vector Field (GVF) based on a given vector field.
     * Note: The input vector field 'v' is also used as output array.
     *
     * @param v      The vector field (also used as output)
     * @param dim    The dimensions of the vector field
     * @param maxIt  The maximum number of iterations
     * @return 'True' on success, 'false' otherwise
     */
    static bool CalcGVFCuda(
            double *v,
            size_t dim[3],
            unsigned int maxIt);
#endif // WITH_CUDA

protected:

private:

};

} // namespace protein
} // namespace megamol

#endif // MMPROTEINPLUGIN_DIFFUSIONSOLVER_H_INCLUDED
