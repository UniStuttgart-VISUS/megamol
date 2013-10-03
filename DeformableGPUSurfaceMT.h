//
// DeformableGPUSurfaceMT.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 17, 2013
// Author     : scharnkn
//

#ifdef WITH_CUDA

#ifndef MMPROTEINPLUGIN_DEFORMABLEGPUSURFACEMT_H_INCLUDED
#define MMPROTEINPLUGIN_DEFORMABLEGPUSURFACEMT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "GPUSurfaceMT.h"

namespace megamol {
namespace protein {

/*
 * TODO
 */
class DeformableGPUSurfaceMT : public GPUSurfaceMT {

public:

    // Interpolation mode used when computing external forces based on gradient
    enum InterpolationMode {INTERP_LINEAR=0, INTERP_CUBIC};

    /** DTor */
    DeformableGPUSurfaceMT();

    /**
     * Copy constructor that does a deep copy of another surface object.
     *
     * @param other The other surface object
     */
    DeformableGPUSurfaceMT(const DeformableGPUSurfaceMT& other);

    /** CTor */
    virtual ~DeformableGPUSurfaceMT();

    /**
     * TODO
     */
    bool MorphToVolume(float *volume_D, size_t volDim[3],
            float volWSOrg[3], float volWSDelta[3], float isovalue,
            InterpolationMode interpMode, size_t maxIt,
            float surfMappedMinDisplScl,
            float springStiffness, float forceScl,
            float externalForcesWeight); // TODO

    /**
     * TODO
     */
    bool MorphToVolumeGVF(float *volume_D, const unsigned int *targetCubeStates_D,
            size_t volDim[3],
            float volWSOrg[3], float volWSDelta[3], float isovalue,
            InterpolationMode interpMode, size_t maxIt,
            float surfMappedMinDisplScl,
            float springStiffness, float forceScl,
            float externalForcesWeight, float gvfScl, unsigned int gvfIt); // TODO

    /**
     * Assignment operator (makes deep copy).
     *
     * @param rhs The assigned surface object
     * @return The returned surface object
     */
    DeformableGPUSurfaceMT& operator=(const DeformableGPUSurfaceMT &rhs);

    // DEBUG
    const float *PeekGVF() {
        return this->gvf_D.Peek();
    }

    const unsigned int *PeekCubeStates() {
        return this->cubeStates_D.Peek();
    }

protected:

private:

    /* Device arrays for external forces */

    /// Device pointer to external forces for every vertex
    CudaDevArr<float> vertexExternalForcesScl_D;

    /// TODO
    CudaDevArr<float> gvf_D;

    /// TODO
    CudaDevArr<float> gvfConstData_D;

    /// TODO
    CudaDevArr<float> grad_D;

    /// Device pointer to gradient field
    CudaDevArr<float4> volGradient_D;

    /// Array for laplacian
    CudaDevArr<float3> laplacian_D;

    /// Array to safe displacement length
    CudaDevArr<float> displLen_D;

    /// Flag whether the neighbors have been computed
    bool neighboursReady;

};

} // namespace protein
} // namespace megamol

#endif // MMPROTEINPLUGIN_DEFORMABLEGPUSURFACEMT_H_INCLUDED
#endif // WITH_CUDA
