//
// DiffusionSolver.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 16, 2013
// Author     : scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_DIFFUSIONSOLVER_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_DIFFUSIONSOLVER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

namespace megamol {
namespace protein_cuda {

/*
 * This class provides static methods for numerically solving diffusion
 * equations, such as the Gradient Vector Flow (GVF).
 */
class DiffusionSolver {

public:

    /**
     * CUDA implementation of an iterative method to compute the Gradient
     * Vector Field (GVF) based on a given volume gradient and a level set.
     *
     * @param volTarget_D         The target volume texture (device memory)
     * @param gvfConstData_D      Temporary array for gvf const data (device
     *                            memory)
     * @param cellStatesTarget_D  Flags about cell activity for the level set
     *                            (device memory)
     * @param volDim              Dimensions of the volume texture
     * @param volOrg              WS origin of the volume texture
     * @param volDelta            WS spacing of the volume texture
     * @param gvfIn_D             Temporary array for gvf (device memory)
     * @param gvfOut_D            Output array for gvf (device memory)
     * @param maxIt               The number of iterations for the gvf
     *                            computation
     * @param scl                 Scale factor for the gvf
     * @return 'True' on success, 'false' otherwise
     */
    static bool CalcGVF(
            const float *volTarget_D,
            float *gvfConstData_D,
            const unsigned int *cellStatesTarget_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float *gvfIn_D,
            float *gvfOut_D,
            unsigned int maxIt,
            float scl);

    /**
     * CUDA implementation of an iterative method to compute the Gradient
     * Vector Field (GVF) based on a given volume gradient and a level set.
     * This method applies the diffusion in two directions.
     *
     * @param volSource_D         The source volume texture (device memory)
     * @param volTarget_D         The target volume texture (device memory)
     * @param cellStatesSource_D  Flags about cell activity for the source level
     *                            set (device memory)
     * @param cellStatesTarget_D  Flags about cell activity for the target level
     *                            set (device memory)
     * @param volDim              Dimensions of the volume texture
     * @param volOrg              WS origin of the volume texture
     * @param volDelta            WS spacing of the volume texture
     * @param gvfConstData_D      Temporary array for gvf const data (device
     *                            memory)
     * @param gvfIn_D             Temporary array for gvf (device memory)
     * @param gvfOut_D            Output array for gvf (device memory)
     * @param maxIt               The number of iterations for the gvf
     *                            computation
     * @param scl                 Scale factor for the gvf
     * @return 'True' on success, 'false' otherwise
     */
    static bool CalcTwoWayGVF(
            const float *volSource_D,
            const float *volTarget_D,
            const unsigned int *cellStatesSource_D,
            const unsigned int *cellStatesTarget_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float *gvfConstData_D,
            float *gvfIn_D,
            float *gvfOut_D,
            unsigned int maxIt,
            float scl);

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "DiffusionSolver";
    }

protected:

private:

};

} // namespace protein_cuda
} // namespace megamol

#endif // MMPROTEINCUDAPLUGIN_DIFFUSIONSOLVER_H_INCLUDED
