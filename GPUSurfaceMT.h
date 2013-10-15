//
// GPUSurfaceMT.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 17, 2013
// Author     : scharnkn
//

#ifdef WITH_CUDA

#ifndef MMPROTEINPLUGIN_GPUSURFACEMT_H_INCLUDED
#define MMPROTEINPLUGIN_GPUSURFACEMT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractGPUSurface.h"
#include "CudaDevArr.h"

namespace megamol {
namespace protein {

/*
 * TODO
 */
class GPUSurfaceMT : public AbstractGPUSurface {

public:

    /** TODO */
    static dim3 Grid(const unsigned int size, const int threadsPerBlock);

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "GPUSurfaceMT";
    }

    /**
     * TODO
     */
    bool ComputeConnectivity(float *volume_D,
            size_t volDim[3], float volWSOrg[3], float volWSDelta[3],
            float isovalue);

    /**
     * Computes the vertex positions based on a given level set using the
     * Marching Tetrahedra method. Needs vertex data to be ready.
     *
     * @param volume     THe 3D volume texture (device memory)
     * @param volDim     The dimensions of the volume
     * @param volWSOrg   The world space origin of the volume
     * @param volWSDelta The world space spacing of the volume lattice
     * @param isovalue   The isovalue defining the level set
     * @return 'True' on success, 'false' otherwise
     */
    bool ComputeTriangles(
            float *volume_D,
            size_t volDim[3],
            float volWSOrg[3],
            float volWSDelta[3],
            float isovalue);

    /**
     * Computes vertex normals. Needs vertex data to be ready.
     *
     * @param volume_D   The 3D volume texture (device memory)
     * @param volDim     The dimensions of the volume
     * @param volWSOrg   The world space origin of the volume
     * @param volWSDelta The world space spacing of the volume lattice
     * @param isovalue   The isovalue defining the level set
     * @return 'True' on success, 'false' otherwise
     */
    bool ComputeNormals(
            float *volume_D,
            size_t volDim[3],
            float volWSOrg[3],
            float volWSDelta[3],
            float isovalue);

    /**
     * Initializes the VBO and computes vertex positions based on the given
     * level set. Sets vertex data ready flag to true.
     *
     * @param volume     THe 3D volume texture (device memory)
     * @param volDim     The dimensions of the volume
     * @param volWSOrg   The world space origin of the volume
     * @param volWSDelta The world space spacing of the volume lattice
     * @param isovalue   The isovalue defining the level set
     * @return 'True' on success, 'false' otherwise
     */
    bool ComputeVertexPositions(
            float *volume_D,
            size_t volDim[3],
            float volWSOrg[3],
            float volWSDelta[3],
            float isovalue);

    /**
     * Computes texture coordinates based on the given texture dimensions and
     * world space information. Needs 'ready' to be true.
     *
     * @param minCoords The minimum world space coordinates of the texture
     * @param minCoords The maximum world space coordinates of the texture
     * @return 'True' on success, 'false' otherwise
     */
    bool ComputeTexCoords(float minCoords[3], float maxCoords[3]);

    /** DTor */
    GPUSurfaceMT();

    /**
     * Copy constructor that makes a deep copy of another surface object.
     */
    GPUSurfaceMT(const GPUSurfaceMT& other);

    /** CTor */
    virtual ~GPUSurfaceMT();

    /**
     * Assignment operator (makes deep copy).
     *
     * @param rhs The assigned surface object
     * @return The returned surface object
     */
    GPUSurfaceMT& operator=(const GPUSurfaceMT &rhs);

    /**
     * Apply rotation to all vertex positions.
     *
     * @param rotMat The rotation matrix
     * @return 'True' on success, 'false' otherwise
     */
    bool Rotate(float rotMat[9]);

    /**
     * Sorts all triangles defined by the triangle index array based on their
     * distance to the camera.
     *
     * @param camPos The camera position
     * @return 'True' on success, 'false' otherwise
     */
    bool SortTrianglesByCamDist(float camPos[3]);

    /**
     * Apply translation to all vertex positions.
     *
     * @param rotMat The rotation matrix
     * @return 'True' on success, 'false' otherwise
     */
    bool Translate(float transVec[3]);

    /**
     * Free all CUDA arrays allocated by this class.
     */
    void Release();

protected:


    /* Surface triangulation */

    /// Array containing activity information for all grid cells (device memory)
    CudaDevArr<unsigned int> cubeStates_D;

    /// Array containing activity information for all grid cells (device memory)
    CudaDevArr<unsigned int> cubeOffsets_D;

    /// Mapping from list of active cells to general cell list (and vice versa)
    CudaDevArr<unsigned int> cubeMap_D, cubeMapInv_D;

    /// Activity of vertices
    CudaDevArr<unsigned int> vertexStates_D;

    /// Positions of active vertices
    CudaDevArr<float3> activeVertexPos_D;

    /// Index offsets for active vertices
    CudaDevArr<unsigned int> vertexIdxOffs_D;

    /// Mapping from active vertex indices to general index list (and vice versa)
    CudaDevArr<unsigned int> vertexMap_D, vertexMapInv_D;

    /// Connectivity information for all vertices (at most 18 neighbours per vertex)
    CudaDevArr<int> vertexNeighbours_D;

    /// Array containing number of vertices for each tetrahedron
    CudaDevArr<unsigned int> verticesPerTetrahedron_D;

    /// Vertex index offsets for all tetrahedrons
    CudaDevArr<unsigned int> tetrahedronVertexOffsets_D;

    /// Cuda graphics resource associated with the vertex data VBO
    struct cudaGraphicsResource *vertexDataResource;

    /// Cuda graphics resource associated with the triangle index VBO
    struct cudaGraphicsResource *triangleIdxResource;

    /// The number of active cells
    size_t activeCellCnt;

    /* Triangle sorting */

    /// Array used for triangles sorting
    CudaDevArr<float> triangleCamDistance_D;

    /// Flag to tell whether the connectivity information has been computed yet
    bool neighboursReady;

private:




};

} // namespace protein
} // namespace megamol

#endif // MMPROTEINPLUGIN_GPUSURFACEMT_H_INCLUDED
#endif // WITH_CUDA
