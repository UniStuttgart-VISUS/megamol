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

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "DeformableGPUSurfaceMT";
    }

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
     * Flag vertices adjacent to corrupt triangles in the current mesh. This
     * is mainly for rendering purposes where vertex attributes are needed.
     *
     * TODO
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool FlagCorruptTriangleVertices(float *volume_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float isovalue);

    /**
     * Answers the GPU handle for the VBO with the vertex data. Needs
     * the 'ready flag to be true.
     *
     * @return The GPU handle for the vertex buffer object or NULL if !ready
     */
    GLuint GetCorruptTriangleVtxFlagVBO() const {
        return this->vboCorruptTriangleVertexFlag;
    }

    /**
     * TODO
     */
    bool InitCorruptFlagVBO(size_t vertexCnt);

    /**
     * TODO
     */
    bool MorphToVolumeGradient(
            float *volume_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float isovalue,
            InterpolationMode interpMode,
            size_t maxIt,
            float surfMappedMinDisplScl,
            float springStiffness,
            float forceScl,
            float externalForcesWeight); // TODO

    /**
     * TODO
     */
    bool MorphToVolumeDistfield(
            float *volume_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float isovalue,
            InterpolationMode interpMode,
            size_t maxIt,
            float surfMappedMinDisplScl,
            float springStiffness,
            float forceScl,
            float externalForcesWeight,
            float distfieldDist); // TODO

    /**
     * TODO
     */
    bool MorphToVolumeGVF(
            float *volumeSource_D,
            float *volumeTarget_D,
            const unsigned int *targetCubeStates_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float isovalue,
            InterpolationMode interpMode,
            size_t maxIt,
            float surfMappedMinDisplScl,
            float springStiffness,
            float forceScl,
            float externalForcesWeight,
            float gvfScl,
            unsigned int gvfIt); // TODO

    /**
     * TODO
     */
    bool MorphToVolumeTwoWayGVF(
            float *volumeSource_D,
            float *volumeTarget_D,
            const unsigned int *cellStatesSource_D,
            const unsigned int *cellStatesTarget_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float isovalue,
            InterpolationMode interpMode,
            size_t maxIt,
            float surfMappedMinDisplScl,
            float springStiffness,
            float forceScl,
            float externalForcesWeight,
            float gvfScl,
            unsigned int gvfIt); // TODO

    /**
     * TODO
     */
    bool InitGridParams(uint3 gridSize, float3 org, float3 delta);




    /**
     * Assignment operator (makes deep copy).
     *
     * @param rhs The assigned surface object
     * @return The returned surface object
     */
    DeformableGPUSurfaceMT& operator=(const DeformableGPUSurfaceMT &rhs);

    /** TODO */
    const unsigned int *PeekCubeStates() {
        return this->cubeStates_D.Peek();
    }

    /** TODO */
    const float *PeekExternalForces() {
        return this->externalForces_D.Peek();
    }

    /**
     * Free all the device memory allocated in this class.
     */
    void Release();

protected:

    /**
     * TODO
     */
    bool initExtForcesGradient(
            float *volTarget_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta);

    /**
     * TODO
     */
    bool initExtForcesDistfield(
            float *volume_D,
            float *vertexBuffer_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float distfieldDist,
            float isovalue);

    /**
     * TODO
     */
    bool initExtForcesGVF(
            float *volumeTarget_D,
            const unsigned int *cellStatesTarget_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float isovalue,
            float gvfScl,
            unsigned int gvfIt);

    /**
     * TODO
     */
    bool initExtForcesTwoWayGVF(
            float *volumeSource_D,
            float *volumeTarget_D,
            const unsigned int *cellStatesSource_D,
            const unsigned int *cellStatesTarget_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float isovalue,
            float gvfScl,
            unsigned int gvfIt);

    /**
     * TODO
     */
    bool updateVtxPos(
            float* volTarget_D,
            float* vertexBuffer_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float isovalue,
            bool useCubicInterpolation,
            size_t maxIt,
            float surfMappedMinDisplScl,
            float springStiffness,
            float forceScl,
            float externalForcesWeight);

private:

    /* Device arrays for external forces */

    /// Device pointer to external forces for every vertex
    CudaDevArr<float> vertexExternalForcesScl_D;

    /// TODO
    CudaDevArr<float> gvfTmp_D;

    /// TODO
    CudaDevArr<float> gvfConstData_D;

    /// Array for laplacian
    CudaDevArr<float3> laplacian_D;

    /// Array for laplacian
    CudaDevArr<float3> laplacian2_D;

    /// Array to safe displacement length
    CudaDevArr<float> displLen_D;

    /// Array for distance field
    CudaDevArr<float> distField_D;

    /// Device array for external forces
    CudaDevArr<float> externalForces_D;

    /// Vertex Buffer Object handle for vertex data
    GLuint vboCorruptTriangleVertexFlag;

};

} // namespace protein
} // namespace megamol

#endif // MMPROTEINPLUGIN_DEFORMABLEGPUSURFACEMT_H_INCLUDED
#endif // WITH_CUDA
