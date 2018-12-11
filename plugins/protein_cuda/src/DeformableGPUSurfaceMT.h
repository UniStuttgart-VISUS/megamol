//
// DeformableGPUSurfaceMT.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 17, 2013
// Author     : scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_DEFORMABLEGPUSURFACEMT_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_DEFORMABLEGPUSURFACEMT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "GPUSurfaceMT.h"
#include "vislib/Array.h"
#include "helper_math.h"

namespace megamol {
namespace protein_cuda {

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
     * Compute normals if subdivision has been performed.
     *
     * TODO
     *
     * @return 'true' on success, 'false' otherwise
     */
    bool ComputeNormalsSubdiv();

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
     * Compute the total surface area of all triangles.
     *
     * @return The total area of all valid triangles
     */
    float GetTotalSurfArea();

    /**
     * Compute the total surface area of all non-corrupt triangles.
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
    GLuint GetVtxPathVBO() const {
        return this->vboVtxPath;
    }

    /**
     * Answers the OpenGL handle for the vbo holding uncertainty information.
     *
     * @return The GPU handle for the vertex buffer object or NULL if !ready
     */
    GLuint GetVtxAttribVBO() const {
        return this->vboVtxAttr;
    }

    /**
     * TODO
     */
    bool InitVtxPathVBO(size_t vertexCnt);

    /**
     * TODO
     */
    bool InitVtxAttribVBO(size_t vertexCnt);

    bool InitCorruptFlagVBO(size_t vertexCnt);


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
    float IntVtxPathOverSurfArea();

    /**
     * Integrate scalar value (given per vertex in value_D) over surface area.
     *
     * @return The integral value
     */
    float IntVtxPathOverValidSurfArea();

    /**
     * Integrate scalar value (given per vertex in value_D) over surface area.
     *
     * @return The integral value
     */
    float IntVtxAttribOverSurfArea();

    /**
     * Integrate scalar value (given per vertex in value_D) over surface area.
     *
     * @return The integral value
     */
    float IntVtxAttribOverValidSurfArea();

    /**
     * Integrate scalar value over corrupt surface area by using virtual
     * subdivision.
     *
     * TODO
     * @return The integral value
     */
    float IntOverCorruptSurfArea();

    /**
     * Integrate uncertainty value over corrupt surface area.
     *
     * TODO
     * @return The integral value
     */
    float IntUncertaintyOverCorruptSurfArea(
            float &corruptArea,
            float minDisplScl,
            float isovalue,
            float forcesScl,
            unsigned int *targetActiveCells_D,
            float *targetVol_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            vislib::Array<float> &triArr,
            int maxSteps,
            int maxLevel,
            float initStepSize);

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
            unsigned int gvfIt,
            bool trackPath,
            bool recomputeGVF); // TODO


    /**
     * TODO Only for benchmarking
     */
    bool MorphToVolumeTwoWayGVFBM(
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
            unsigned int gvfIt,
            bool trackPath,
            bool recomputeGVF,
            float &t_gvf,
            float &t_map); // TODO

    /**
     * TODO
     */
    bool MorphToVolumeTwoWayGVFSubdiv(
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
            unsigned int gvfIt,
            bool trackPath,
            bool recomputeGVF); // TODO

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
     * Attention: this is potentially slow, since it performs the subdivison
     * on the CPU and copies all the data back to the GPU!
     *
     * @param maxSubdivLevel The maximum number of subdivisions to be performed
     * TODO
     * @return The number of newly created triangles or '-1' if something went
     *         wrong.
     */
    int RefineMesh(uint maxSubdivLevel,
            float *volume_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float isovalue,
            float maxEdgeLen);

    /**
     * Free all the device memory allocated in this class.
     */
    void Release();

    /**
     * TODO
     */
    float *PeekVertexFlagDevPt() {
        return this->vertexFlag_D.Peek();
    }

    /**
     * TODO
     */
    float *PeekOldVertexDataDevPt() {
        return this->trackedSubdivVertexData_D.Peek();
    }

    /**
     * TODO
     */
    bool TrackPathSubdivVertices(
            float *sourceVolume_D,
            int3 volDim,
            float3 volOrg,
            float3 volDelta,
            float forcesScl,
            float minDispl,
            float isoval,
            uint maxIt);

    bool ComputeSurfAttribDiff( DeformableGPUSurfaceMT &surfStart,
            float centroid[3], // In case the start surface has been fitted using RMSD
            float rotMat[9],
            float transVec[3],
            float *tex0_D,
            int3 texDim0,
            float3 texOrg0,
            float3 texDelta0,
            float *tex1_D,
            int3 texDim1,
            float3 texOrg1,
            float3 texDelta1);

    bool ComputeSurfAttribSignDiff( DeformableGPUSurfaceMT &surfStart,
            float centroid[3], // In case the start surface has been fitted using RMSD
            float rotMat[9],
            float transVec[3],
            float *tex0_D,
            int3 texDim0,
            float3 texOrg0,
            float3 texDelta0,
            float *tex1_D,
            int3 texDim1,
            float3 texOrg1,
            float3 texDelta1);

    void PrintVertexBuffer(size_t cnt);
    void PrintExternalForces(size_t cnt);
    void PrintCubeStates(size_t cnt);

    bool ComputeMeshLaplacian();
    bool ComputeMeshLaplacianDiff(DeformableGPUSurfaceMT &surfStart);

    float *PeekGeomLaplacian() {
        return this->geometricLaplacian_D.Peek();
    }

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
            bool trackPath,
            bool externalForcesOnly=false,
            bool useThinPlate=true);

    /**
     * TODO
     */
    bool updateVtxPosSubdiv(
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
            bool trackPath,
            bool externalForcesOnly=false,
            bool useThinPlate=true);

    float IntUncertaintyOverCorruptAreaRec(
            float3 pos1, float3 pos2, float3 pos3, // Vertex positions of the triangle
            float len1, float len2, float len3,    // Vertex path lengths of the triangle
            float4 *gradient,                      // External forces
            float *targetVol,                      // The target volume
            unsigned int *targetActiveCells,       // Active cells of the target volume
            float minDisplScl,                     // Minimum displacement for convergence
            float forcesScl,                       // General scaling factor for forces
            float isovalue,                        // Isovalue
            float &triArea,
            uint depth,
            float org[3], float delta[3], int dim[3],
            vislib::Array<float> &triArr,
            int maxSteps,
            int maxLevel,
            float initStepSize);

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
    GLuint vboVtxPath;

    /// Vertex buffer object for an arbitrary vertex attribute (scalar)
    GLuint vboVtxAttr;

    // Device array that flags corrupt triangles
    CudaDevArr<float> corruptTriangles_D;

    /// Device area needed to accumulate triangle data
    CudaDevArr<float> accTriangleArea_D;

    /// Device area needed to accumulate triangle data
    CudaDevArr<float> accTriangleData_D;

    // Device array that flags corrupt triangles
    CudaDevArr<float> intUncertaintyCorrupt_D;

    /// Device array to remember whether to accumulate or to decrease the path
    CudaDevArr<int> accumPath_D;


    /* Subdivision */

    CudaDevArr<int> triangleEdgeOffs_D;
    CudaDevArr<uint> triangleEdgeList_D;
    CudaDevArr<uint> subDivEdgeFlag_D;
    CudaDevArr<uint> subDivEdgeIdxOffs_D;
    CudaDevArr<float> newVertices_D;
    CudaDevArr<uint> newTriangles_D;
    CudaDevArr<uint> oldTriangles_D;
    CudaDevArr<float> trackedSubdivVertexData_D;
    CudaDevArr<uint> subDivCnt_D;
    CudaDevArr<uint> newTrianglesIdxOffs_D;
    CudaDevArr<uint> oldTrianglesIdxOffs_D;
    CudaDevArr<uint> newTriangleNeighbors_D;
    CudaDevArr<uint> subDivLevels_D;
    CudaDevArr<uint> oldSubDivLevels_D;
    CudaDevArr<float> vertexFlag_D;
    CudaDevArr<float> vertexFlagTmp_D;
    CudaDevArr<float> vertexUncertaintyTmp_D;
    uint newVertexCnt;
    uint oldVertexCnt;
    uint nFlaggedVertices;

    CudaDevArr<float3> triangleFaceNormals_D;
    CudaDevArr<uint> triangleIdxTmp_D;
    CudaDevArr<uint> outputArrayTmp_D;
    CudaDevArr<uint> reducedVertexKeysTmp_D;
    CudaDevArr<float3> reducedNormalsTmp_D;
    CudaDevArr<uint> vertexNormalsIndxOffs_D;

    CudaDevArr<float> geometricLaplacian_D;
};

} // namespace protein_cuda
} // namespace megamol

#endif // MMPROTEINCUDAPLUGIN_DEFORMABLEGPUSURFACEMT_H_INCLUDED
