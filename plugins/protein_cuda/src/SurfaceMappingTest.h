//
// SurfaceMappingTest.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jul 15, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_SURFACEMAPPINGTEST_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_SURFACEMAPPINGTEST_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif // (defined(_MSC_VER) && (_MSC_VER > 1000))

//#include <glh/glh_genext.h>
#include "mmcore/job/AbstractJob.h"
#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "CudaDevArr.h"
#include "HostArr.h"
#include "protein_calls/MolecularDataCall.h"
#include "CUDAQuickSurf.h"
#include "gridParams.h"
#include <cmath>

#define USE_DISTANCE_FIELD

typedef unsigned int uint;

namespace megamol {
namespace protein_cuda {

class SurfaceMappingTest : public core::job::AbstractJob, public core::Module {

public:

    /// Enum defining the differend heuristics
    enum Heuristic {RMS_VALUE=0, SURFACE_POTENTIAL, SURFACE_POTENTIAL_SIGN, MEAN_HAUSDORFF_DIST, HAUSDORFF_DIST};

    /// Enum describing different ways of using RMS fitting
    enum RMSFittingMode {RMS_ALL=0, RMS_BACKBONE, RMS_C_ALPHA};

    /// Interpolation mode used when computing external forces based on gradient
    enum InterpolationMode {INTERP_LINEAR=0, INTERP_CUBIC};

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "SurfaceMappingTest";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Job module to test the surface mapping routines.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /**
     * Disallow usage in quickstarts
     *
     * @return false
     */
    static bool SupportQuickstart(void) {
        return false;
    }

    /**
     * Ctor
     */
    SurfaceMappingTest();

    /**
     * Dtor
     */
    virtual ~SurfaceMappingTest();

    /**
     * Answers whether or not this job is still running.
     *
     * @return 'true' if this job is still running, 'false' if it has
     *         finished.
     */
    virtual bool IsRunning(void) const;

    /**
     * Starts the job thread.
     *
     * @return true if the job has been successfully started.
     */
    virtual bool Start(void);

    /**
     * Terminates the job thread.
     *
     * @return true to acknowledge that the job will finish as soon
     *         as possible, false if termination is not possible.
     */
    virtual bool Terminate(void);

protected:

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

private:

    /**
     * Translate and rotate an array of positions according to the current
     * transformation obtained by RMS fitting (a translation vector and
     * rotation matrix).
     *
     * @param mol          A Molecular data call containing the particle
     *                     positions of the corresponding data set. This
     *                     is necessary to compute the centroid of the
     *                     particles.
     * @param vertexPos_D  The device array containing the vertex
     *                     positions to be transformed (device memory)
     * @param vertexCnt    The number of vertices to be transformed
     * @param rotation     The rotation matrix
     * @param translation  The translation matrix
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool applyRMSFittingToPosArray(
            megamol::protein_calls::MolecularDataCall *mol,
            CudaDevArr<float> &vertexPos_D,
            uint vertexCnt,
            float rotation[3][3],
            float translation[3]);

    /**
     * (Re-)computes a smooth density map based on an array of givwen particle
     * positions using a CUDA implementation.
     *
     * @param mol           The data call containing the particle positions
     * @param cqs           The CUDAQuickSurf object used to compute the density
     *                      map
     * @param gridDensMap   Grid parameters for the resulting density map
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool computeDensityMap(
            megamol::protein_calls::MolecularDataCall *mol,
            CUDAQuickSurf *cqs,
            gridParams &gridDensMap);

#if defined(USE_DISTANCE_FIELD)
    /**
     * Computes a distance field based on a given set of vertices. For every
     * lattice point, the distance to the nearest vertex is stored.
     *
     * @param vertexPos_D   Array with the vertex positions (device memory)
     * @param vertexCnt     The number of vertices
     * @param distField_D   Array containing the distance field (device memory)
     * @param gridDistField Grid parameters of the distance field lattice
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool computeDistField(
            CudaDevArr<float> &vertexPos_D,
            uint vertexCnt,
            CudaDevArr<float> &distField_D,
            gridParams &gridDistField);
#endif // defined(USE_DISTANCE_FIELD)

    /**
     * Computes the RMS (Root Mean Square) value of two variants.
     *
     * @param atomPos0 Array with atom positions for variant 0
     * @param atomPos1 Array with atom positions for variant 1
     * @param cnt      The number of elements in both arrays (must be equal)
     * @param fit      If 'true', the actual fitting of performed, otherwise
     *                 only the RMS value is computed
     * @param flag     If == 0, rms deviation is calculated but no structure
     *                 will move,
     *                 if == 1, rms deviation is calculated, Vec moves back, but
     *                 toFitVec's centroid moved to (0,0,0). Alignment will be
     *                 done in the calling function.
     *                 if == 2, rms deviation is calculated and toFitVec will
     *                 align to Vec.
     *                 (see also RMS.h)
     * @param rotation    Saves the rotation matrix if wanted (may be null)
     * @param translation Saves the translation vector if wanted (may be null)
     *
     * @return The RMS value of the two variants
     */
    float getRMS(float *atomPos0, float *atomPos1, unsigned int cnt, bool fit,
            int flag, float rotation[3][3], float translation[3]);

    /**
     * Extracts all atom positions from a moleculr data call, that are used to
     * compute the RMS value (either all protein atoms, or only backbone atoms,
     * or only C alpha atoms).
     *
     * @param mol    The data call comtaining the particles
     * @param posArr The array for the extracted positions
     * @param cnt    The number of extracted elements
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool getRMSPosArray(megamol::protein_calls::MolecularDataCall *mol, HostArr<float> &posArr,
            unsigned int &cnt);

    /**
     * Extracts an isosurface from a given volume texture using Marching
     * Tetrahedra with the Freudenthal subdivision scheme.
     *
     * @param volume_D            The volume data (device memory)
     * @param cubeMap_D           Array for the cube map (device memory)
     * @param cubeMapInv_D        Array for the inverse cube map (device memory)
     * @param vertexMap_D         Array for the vertex map (device memory)
     * @param vertexMapInv_D      Array for the inverse vertex map (device
     *                            memory)
     * @param vertexNeighbours_D  Array containing neighbours of all vertices
     * @param gridDensMap         Grid parameters for the volume
     * @param vertexCount         The number of vertices
     * @param vertexPos_D         Array with vertex positions
     * @param triangleCount       The number of triangles
     * @param triangleIdx_D       Array with triangle indices
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool isosurfComputeVertices(
            float *volume_D,
            CudaDevArr<uint> &cubeMap_D,
            CudaDevArr<uint> &cubeMapInv_D,
            CudaDevArr<uint> &vertexMap_D,
            CudaDevArr<uint> &vertexMapInv_D,
            CudaDevArr<int> &vertexNeighbours_D,
            gridParams gridDensMap,
            uint &activeVertexCount,
            CudaDevArr<float> &vertexPos_D,
            uint &triangleCount,
            CudaDevArr<uint> &triangleIdx_D);

    /**
     * Maps an isosurface defined by an array of vertex positions with
     * connectivity information to a given volumetric isosurface (defined by
     * a texture and an isovalue). To achieve this a deformable model approach
     * is used which combines internal spring forces with an external force
     * obtained from the volume gradient.
     * The potential used for the external forces is a combination of a distance
     * field and a density map.
     *
     * @param volume_D                The volume the vertices are to be mapped
     *                                to (device memory)
     * @param gridDensMap             Grid parameters for the volume
     * @param triangleIdx_D           Array with triangle indices
     * @param vertexCnt               The number of vertices
     * @param triangleCnt             The number of triangles
     * @param vertexNeighbours_D      Connectivity information of the vertices
     *                                (device memory)
     * @param maxIt                   The number of iterations for the mapping
     * @param springStiffness         The stiffness of the springs defining the
     *                                internal spring forces
     * @param forceScl                An overall scaling for the combined force
     * @param externalForcesWeight    The weighting of the external force. The
     *                                weight for the internal forces is
     *                                1.0 - externalForcesWeight
     * @param interpMode              Detemines whether linear or cubic
     *                                interpolation is to be used when computing
     *                                the external forces
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool mapIsosurfaceToVolume(
            float *volume_D,
            gridParams gridDensMap,
            CudaDevArr<float> &vertexPos_D,
            CudaDevArr<uint> &triangleIdx_D,
            uint vertexCnt,
            uint triangleCnt,
            CudaDevArr<int> &vertexNeighbours_D,
            uint maxIt,
            float springStiffness,
            float forceScl,
            float externalForcesWeight,
            InterpolationMode interpMode);

    /**
     * TODO
     */
    bool plotMappingIterationsByExternalForceWeighting();

    /**
     * TODO
     */
    bool plotMappingIterationsByForcesScl();

    /**
     * TODO
     */
    bool plotMappingIterationsByRigidity();

    /**
     * TODO
     */
    bool plotCorruptTriangleAreaByRigidity();

    /**
     * TODO
     */
    bool plotRegIterationsByExternalForceWeighting();

    /**
     * TODO
     */
    bool plotRegIterationsByRigidity();

    /**
     * TODO
     */
    bool plotRegIterationsByForcesScl();

    /**
     * Maps an isosurface defined by an array of vertex positions with
     * connectivity information to a given volumetric isosurface (defined by
     * a texture and an isovalue). To achieve this a deformable model approach
     * is used which combines internal spring forces with an external force
     * obtained from the volume gradient.
     *
     * @param volume_D                The volume the vertices are to be mapped
     *                                to (device memory)
     * @param gridDensMap             Grid parameters for the volume
     * @param vertexPos_D             Array with vertex positions
     * @param vertexCnt               The number of vertices
     * @param vertexNeighbours_D      Connectivity information of the vertices
     *                                (device memory)
     * @param maxIt                   The number of iterations for the mapping
     * @param springStiffness         The stiffness of the springs defining the
     *                                internal spring forces
     * @param forceScl                An overall scaling for the combined force
     * @param externalForcesWeight    The weighting of the external force. The
     *                                weight for the internal forces is
     *                                1.0 - externalForcesWeight
     * @param interpMode              Detemines whether linear or cubic
     *                                interpolation is to be used when computing
     *                                the external forces
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool regularizeSurface(
            float *volume_D,
            gridParams gridDensMap,
            CudaDevArr<float> &vertexPos_D,
            uint vertexCnt,
            CudaDevArr<int> &vertexNeighbours_D,
            uint maxIt,
            float springStiffness,
            float forceScl,
            float externalForcesWeight,
            InterpolationMode interpMode);

    bool initDensMapParams(megamol::protein_calls::MolecularDataCall *mol);

    /// Flag whether the job is done
    bool jobDone;

    /* Data caller/callee slots */

    /// Caller slot for vertex data
    megamol::core::CallerSlot particleDataCallerSlot;

    /// Caller slot volume data
    megamol::core::CallerSlot volumeDataCallerSlot;

    /* Hardcoded parameters for the 'quicksurf' class */

    /// Parameter for assumed radius of density grid data
    static const float qsParticleRad;

    /// Parameter for the cutoff radius for the gaussian kernel
    static const float qsGaussLim;

    /// Parameter for assumed radius of density grid data
    static const float qsGridSpacing;

    /// Parameter to toggle scaling by van der Waals radius
    static const bool qsSclVanDerWaals;

    /// Parameter for iso value for volume rendering
    static const float qsIsoVal;

    /* RMS fitting */

    HostArr<float> rmsPosVec0;  ///> Position vector #0 for rms fitting
    HostArr<float> rmsPosVec1;  ///> Position vector #1 for rms fitting
    HostArr<float> rmsWeights;  ///> Particle weights
    HostArr<int> rmsMask;       ///> Mask for particles


    /* Volume generation */

    void *cudaqsurf0, *cudaqsurf1;   ///> Pointer to CUDAQuickSurf objects
    HostArr<float> gridDataPos;      ///> Data array for intermediate calculations

    /* Surface mapping */

    /// Device pointer to external forces for every vertex
    CudaDevArr<float> vertexExternalForcesScl_D;

    /// Array containing activity information for all grid cells (device memory)
    CudaDevArr<uint> cubeStates_D;

    /// Array containing activity information for all grid cells (device memory)
    CudaDevArr<uint> cubeOffsets_D;

    /// Mapping from list of active cells to generall cell list (and vice versa)
    CudaDevArr<uint> cubeMap0_D, cubeMapInv0_D;
    CudaDevArr<uint> cubeMap1_D, cubeMapInv1_D;

    /// Activity of vertices
    CudaDevArr<uint> vertexStates_D;

    /// Positions of active vertices
    CudaDevArr<float> vertexPos0_D, vertexPos1_D, vertexPosMapped_D, vertexPosRMSTransformed_D;

    /// Positions of active vertices (holds intermediate results)
    CudaDevArr<float3> activeVertexPos_D;

    /// Index offsets for active vertices
    CudaDevArr<uint> vertexIdxOffs_D;

    /// Mapping from active vertex indices to general index list (and vice versa)
    CudaDevArr<uint> vertexMap0_D, vertexMapInv0_D;
    CudaDevArr<uint> vertexMap1_D, vertexMapInv1_D;

    /// Connectivity information for all vertices (at most 18 neighbours per vertex)
    CudaDevArr<int> vertexNeighbours0_D, vertexNeighbours1_D;

    /// Array containing number of vertices for each tetrahedron
    CudaDevArr<uint> verticesPerTetrahedron_D;

    /// Vertex index offsets for all tetrahedrons
    CudaDevArr<uint> tetrahedronVertexOffsets_D;

#if defined(USE_DISTANCE_FIELD)
    /// Array holding the distance field (device memory)
    CudaDevArr<float> distField_D;

    /// Max dist to use Gaussian volume instead
    static const float maxGaussianVolumeDist;
#endif // defined(USE_DISTANCE_FIELD)

    /// Array for volume gradient
    CudaDevArr<float4> volGradient_D;

    /// The bounding boxes of the density maps
    gridParams gridDensMap;

    /// The bounding boxes for potential maps
    gridParams gridPotential0, gridPotential1;

    /// Triangle indices
    CudaDevArr<uint> triangleIdx0_D, triangleIdx1_D;

    /// The number of active vertices
    uint vertexCnt0, vertexCnt1;

    /// The number of triangle indices
    uint triangleCnt0, triangleCnt1;

    /// Minimum force to keep going
    static const float minDispl;

    /* Surface area summation */

    /// The area of all triangles
    CudaDevArr<float> trianglesArea_D;

    /// The averaged vertex values for all triangles (potential difference)
    CudaDevArr<float> trianglesAreaWeightedPotentialDiff_D;

    /// The averaged vertex values for all triangles (hausdorff distance)
    CudaDevArr<float> trianglesAreaWeightedHausdorffDist_D;

    /// The averaged vertex values for all triangles (potential sign)
    CudaDevArr<float> trianglesAreaWeightedPotentialSign_D;

    /// Distance between new and old potential values
    CudaDevArr<float> vertexPotentialDiff_D;

    /// Flags whether the sign has changed
    CudaDevArr<float> vertexPotentialSignDiff_D;

    /// Flag for corrupt triangles
    CudaDevArr<float> corruptTriangleFlag_D;

    /// Per-vertex hausdorff distance
    CudaDevArr<float> vtxHausdorffDist_D;

    /// Per-vertex length of the internal force
    CudaDevArr<float> internalForceLen_D;

    /// Array for laplacian
    CudaDevArr<float3> laplacian_D;



    /* Boolean flags */

    CudaDevArr<float> potentialTex0_D;
    CudaDevArr<float> potentialTex1_D;

    /// Array to safe displacement length
    CudaDevArr<float> displLen_D;

    RMSFittingMode fittingMode;

};

} // end namespace protein_cuda
} // end namespace megamol

#endif // MMPROTEINCUDAPLUGIN_SURFACEMAPPINGTEST_H_INCLUDED
