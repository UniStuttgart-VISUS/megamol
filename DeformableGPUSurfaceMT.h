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

    /**
     * Computes the Hausdorff distance from surf1 to surf 2.
     *
     * @param surf1 Pointer to the first surface.
     * @param surf2 Pointer to the second surface.
     * @return The Hausdorff distance
     */
    static float CalcHausdorffDistance(
            DeformableGPUSurfaceMT *surf1,
            DeformableGPUSurfaceMT *surf2,
            float *hausdorffdistVtx_D,
            bool symmetric=false);

    /**
     * In this class, because it needs the offsets TODO
     */
    static bool ComputeVtxDiffValue(
            float *diff_D,
            float *tex0_D,
            int3 texDim0,
            float3 texOrg0,
            float3 texDelta0,
            float *tex1_D,
            int3 texDim1,
            float3 texOrg1,
            float3 texDelta1,
            GLuint vtxDataVBO0,
            GLuint vtxDataVBO1,
            size_t vertexCnt);

    /**
     * In this class, because it needs the offsets TODO
     */
    static bool ComputeVtxSignDiffValue(
            float *signdiff_D,
            float *tex0_D,
            int3 texDim0,
            float3 texOrg0,
            float3 texDelta0,
            float *tex1_D,
            int3 texDim1,
            float3 texOrg1,
            float3 texDelta1,
            GLuint vtxDataVBO0,
            GLuint vtxDataVBO1,
            size_t vertexCnt);


    /**
     * In this class, because it needs the offsets TODO
     */
    static bool ComputeVtxDiffValueFitted(
            float *diff_D,
            float centroid[3],
            float rotMat[9],
            float transVec[3],
            float *tex0_D,
            int3 texDim0,
            float3 texOrg0,
            float3 texDelta0,
            float *tex1_D,
            int3 texDim1,
            float3 texOrg1,
            float3 texDelta1,
            GLuint vtxDataVBO0,
            GLuint vtxDataVBO1,
            size_t vertexCnt);

    /**
     * In this class, because it needs the offsets TODO
     */
    static bool ComputeVtxSignDiffValueFitted(
            float *signdiff_D,
            float centroid[3],
            float rotMat[9],
            float transVec[3],
            float *tex0_D,
            int3 texDim0,
            float3 texOrg0,
            float3 texDelta0,
            float *tex1_D,
            int3 texDim1,
            float3 texOrg1,
            float3 texDelta1,
            GLuint vtxDataVBO0,
            GLuint vtxDataVBO1,
            size_t vertexCnt);

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
     * Compute the total surface area of all valid triangles.
     *
     * @return The total area of all valid triangles
     */
    float GetTotalValidSurfArea();

    /**
     * Flag corrupt triangles in the current mesh.
     * TODO
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool FlagCorruptTriangles(
            float *volume_D,
            const unsigned int *targetActiveCells,
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
     * Answers the OpenGL handle for the vbo holding uncertainty information.
     *
     * @return The GPU handle for the vertex buffer object or NULL if !ready
     */
    GLuint GetUncertaintyVBO() const {
        return this->vboUncertainty;
    }

    /**
     * TODO
     */
    bool InitCorruptFlagVBO(size_t vertexCnt);

    /**
     * TODO
     */
    bool InitUncertaintyVBO(size_t vertexCnt);

    /**
     * Integrate scalar value (given per vertex in value_D) over surface area.
     *
     * @param value_D The value to be integrated
     * @return The integral value
     */
    float IntOverSurfArea(float *value_D);

    /**
     * Integrate scalar value (given per vertex in value_D) over surface area.
     *
     * @return The integral value
     */
    float IntUncertaintyOverSurfArea();

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
            float* vtxUncertainty_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float isovalue,
            bool useCubicInterpolation,
            size_t maxIt,
            float surfMappedMinDisplScl,
            float springStiffness,
            float forceScl,
            float externalForcesWeight,
            bool externalForcesOnly=false,
            bool useThinPlate=true);

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

    /// Array to store displacement length
    CudaDevArr<float> displLen_D;

    /// Array for distance field
    CudaDevArr<float> distField_D;

    /// Device array for external forces
    CudaDevArr<float> externalForces_D;

    /// Vertex Buffer Object handle for vertex data
    GLuint vboCorruptTriangleVertexFlag;

    /// Vertex buffer object for the vertex uncertainty information
    GLuint vboUncertainty;

    // Device array that flags corrupt triangles
    CudaDevArr<float> corruptTriangles_D;

    /// Device area needed to accumulate triangle data
    CudaDevArr<float> accTriangleArea_D;

    /// Device area needed to accumulate triangle data
    CudaDevArr<float> accTriangleData_D;

};

} // namespace protein
} // namespace megamol

#endif // MMPROTEINPLUGIN_DEFORMABLEGPUSURFACEMT_H_INCLUDED
#endif // WITH_CUDA
