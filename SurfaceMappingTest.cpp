//
// SurfaceMappingTest.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jul 15, 2013
//     Author: scharnkn
//

#include <stdafx.h>

#if (defined(WITH_CUDA) && (WITH_CUDA))

#include "SurfaceMappingTest.h"
#include "param/BoolParam.h"
#include "param/IntParam.h"
#include "param/StringParam.h"
#include "param/EnumParam.h"

#include "VTIDataCall.h"
#include "MolecularDataCall.h"
#include "DiagramCall.h"
#include "VariantMatchDataCall.h"
#include "MolecularSurfaceFeature.h"
#include "vislib_vector_typedefs.h"
#include "param/EnumParam.h"
#include "param/FloatParam.h"
#include "param/BoolParam.h"
#include "param/IntParam.h"
#include "RMS.h"
#include "vislib/Log.h"
#include <ctime>
#include <algorithm>
//#include <cmath>

#include "CudaDevArr.h"
#include "ComparativeSurfacePotentialRenderer.cuh"
#include "cuda_error_check.h"
#include "ogl_error_check.h"
#include "CUDAQuickSurf.h"
#include "gridParams.h"
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_ptr.h>
#include "vislib/mathtypes.h"

using namespace megamol;
using namespace megamol::protein;
using namespace megamol::core;

#if defined(USE_DISTANCE_FIELD)
const float SurfaceMappingTest::maxGaussianVolumeDist = 0.75;
#endif //  defined(USE_DISTANCE_FIELD)

// Hardcoded parameters for 'quicksurf' class
const float SurfaceMappingTest::qsParticleRad = 1.0f;
const float SurfaceMappingTest::qsGaussLim = 10.0f;
const float SurfaceMappingTest::qsGridSpacing = 1.0f;
const bool SurfaceMappingTest::qsSclVanDerWaals = true;
const float SurfaceMappingTest::qsIsoVal = 0.5f;

/// Minimum force to keep going
const float SurfaceMappingTest::minDispl = SurfaceMappingTest::qsGridSpacing/1000.0;

using namespace megamol;
using namespace megamol::protein;

// Toggle performance measurement and the respective messages
#define USE_TIMER

// Toggle output messages about progress of computations
#define OUTPUT_PROGRESS

// Toggle more detailed output messages
//#define VERBOSE


/*
 * SurfaceMappingTest::IsRunning
 */
bool SurfaceMappingTest::IsRunning(void) const {
    return (!(this->jobDone));
}


/*
 * SurfaceMappingTest::Start
 */
bool SurfaceMappingTest::Start(void) {
    using namespace vislib::sys;

//    if (!this->plotMappingIterationsByExternalForceWeighting()) {
//        this->jobDone = true;
//        return false;
//    }

//    if (!this->plotMappingIterationsByRigidity()) {
//        this->jobDone = true;
//        return false;
//    }

//    if (!this->plotMappingIterationsByForcesScl()) {
//        this->jobDone = true;
//        return false;
//    }

//    if (!this->plotCorruptTriangleAreaByRigidity()) {
//        this->jobDone = true;
//        return false;
//    }

//    if (!this->plotRegIterationsByExternalForceWeighting()) {
//        this->jobDone = true;
//        return false;
//    }

//    if (!this->plotRegIterationsByRigidity()) {
//        this->jobDone = true;
//        return false;
//    }

    if (!this->plotRegIterationsByForcesScl()) {
        this->jobDone = true;
        return false;
    }

    this->jobDone = true;
    return true;
}


/*
 * SurfaceMappingTest::Terminate
 */
bool SurfaceMappingTest::Terminate(void) {
    return true;
}


/*
 * SurfaceMappingTest::SurfaceMappingTest
 */
SurfaceMappingTest::SurfaceMappingTest(void) : Module() , AbstractJob(),
        particleDataCallerSlot("getParticleData", "Connects the module with the partidle data source"),
        volumeDataCallerSlot("getVolumeData", "Connects the module with the volume data source"),
        /* Parameters for surface mapping */
        cudaqsurf0(NULL), cudaqsurf1(NULL),
        vertexCnt0(0), vertexCnt1(0), triangleCnt0(0), triangleCnt1(0),
        jobDone(false) {

    // Data caller for particle data
    this->particleDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->particleDataCallerSlot);

    // Data caller for volume data
    this->volumeDataCallerSlot.SetCompatibleCall<VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->volumeDataCallerSlot);

    this->fittingMode = RMS_ALL;
}


/*
 * SurfaceMappingTest::~SurfaceMappingTest
 */
SurfaceMappingTest::~SurfaceMappingTest(void) {
    this->Release();
}


/*
 * SurfaceMappingTest::create
 */
bool SurfaceMappingTest::create(void) {

    // Create quicksurf objects
    if(!this->cudaqsurf0) {
        this->cudaqsurf0 = new CUDAQuickSurf();
    }
    if(!this->cudaqsurf1) {
        this->cudaqsurf1 = new CUDAQuickSurf();
    }

    return true;
}


/*
 * SurfaceMappingTest::release
 */
void SurfaceMappingTest::release(void) {

    if (this->cudaqsurf0 != NULL) {
        CUDAQuickSurf *cqs = (CUDAQuickSurf *)this->cudaqsurf0;
        delete cqs;
    }

    CheckForGLError();

    if (this->cudaqsurf1 != NULL) {
        CUDAQuickSurf *cqs = (CUDAQuickSurf *)this->cudaqsurf1;
        delete cqs;
    }

    CheckForGLError();
}


/*
 * SurfaceMappingTest::applyRMSFittingToPosArray
 */
bool SurfaceMappingTest::applyRMSFittingToPosArray(
        MolecularDataCall *mol,
        CudaDevArr<float> &vertexPos_D,
        uint vertexCnt,
        float rotation[3][3],
        float translation[3]) {

    // Note: all particles have the same weight
    Vec3f centroid(0.0f, 0.0f, 0.0f);
    CudaDevArr<float> rotate_D;


//    // DEBUG Print mapped positions
//    printf("Apply RMS,  positions before:\n");
//    HostArr<float> vertexPos;
//    vertexPos.Validate(vertexCnt*3);
//    cudaMemcpy(vertexPos.Peek(), vertexPos_D.Peek(),
//            sizeof(float)*vertexCnt*3,
//            cudaMemcpyDeviceToHost);
//    for (int k = 0; k < 10; ++k) {
//        printf("%i: Vertex position (%f %f %f)\n", k, vertexPos.Peek()[3*k+0],
//                vertexPos.Peek()[3*k+1], vertexPos.Peek()[3*k+2]);
//
//    }
//    // End DEBUG


    // Compute centroid
    for (int cnt = 0; cnt < static_cast<int>(mol->AtomCount()); ++cnt) {

        centroid += Vec3f(mol->AtomPositions()[cnt*3],
                mol->AtomPositions()[cnt*3+1],
                mol->AtomPositions()[cnt*3+2]);
    }
    centroid /= static_cast<float>(mol->AtomCount());

    // Move vertex positions to origin (with respect to centroid)
    if (!CudaSafeCall(TranslatePos(
            vertexPos_D.Peek(),
            3,
            0,
            make_float3(-centroid.X(), -centroid.Y(), -centroid.Z()),
            vertexCnt))) {
        return false;
    }

    // Rotate for best fit
    rotate_D.Validate(9);
    if (!CudaSafeCall(cudaMemcpy((void *)rotate_D.Peek(), rotation,
            9*sizeof(float), cudaMemcpyHostToDevice))) {
        return false;
    }
    if (!CudaSafeCall(RotatePos(
            vertexPos_D.Peek(),
            3,
            0,
            rotate_D.Peek(),
            vertexCnt))) {
        return false;
    }

    // Move vertex positions to centroid of second data set
    if (!CudaSafeCall(TranslatePos(
            vertexPos_D.Peek(),
            3,
            0,
            make_float3(translation[0],
                    translation[1],
                    translation[2]),
                    vertexCnt))) {
        return false;
    }

    // Clean up
    rotate_D.Release();

//    // DEBUG
//    printf("RMS centroid %f %f %f\n", centroid.X(), centroid.Y(), centroid.Z());
//    printf("RMS translation %f %f %f\n", translation[0],
//            translation[1], translation[2]);
//    printf("RMS rotation \n %f %f %f\n%f %f %f\n%f %f %f\n",
//            rotation[0][0], rotation[0][1], rotation[0][2],
//            rotation[1][0], rotation[1][1], rotation[1][2],
//            rotation[2][0], rotation[2][1], rotation[2][2]);
//
//
//    // DEBUG Print mapped positions
//    printf("Apply RMS,  positions after:\n");
////    HostArr<float> vertexPos;
//    vertexPos.Validate(vertexCnt*3);
//    cudaMemcpy(vertexPos.Peek(), vertexPos_D.Peek(),
//            sizeof(float)*vertexCnt*3,
//            cudaMemcpyDeviceToHost);
//    for (int k = 0; k < 10; ++k) {
//        printf("%i: Vertex position (%f %f %f)\n", k, vertexPos.Peek()[3*k+0],
//                vertexPos.Peek()[3*k+1], vertexPos.Peek()[3*k+2]);
//
//    }
//    // End DEBUG


    return true;
}


bool initDensMapParams(MolecularDataCall *mol) {
    return true;

}

/*
 * SurfaceMappingTest::computeDensityMap
 */
bool SurfaceMappingTest::computeDensityMap(
        MolecularDataCall *mol,
        CUDAQuickSurf *cqs,
        gridParams &gridDensMap) {

    using namespace vislib::sys;
    using namespace vislib::math;

    float gridXAxisLen, gridYAxisLen, gridZAxisLen;
    float maxAtomRad, padding;

    // (Re-)allocate memory for intermediate particle data
    this->gridDataPos.Validate(mol->AtomCount()*4);

    // Set particle radii and compute maximum particle radius
    maxAtomRad = 0.0f;

    // Gather atom positions for the density map
    uint particleCnt = 0;
    for (uint c = 0; c < mol->ChainCount(); ++c) { // Loop through chains
        // Only use non-solvent atoms for density map
        if (mol->Chains()[c].Type() != MolecularDataCall::Chain::SOLVENT) {

            MolecularDataCall::Chain chainTmp = mol->Chains()[c];
            for (uint m = 0; m < chainTmp.MoleculeCount(); ++m) { // Loop through molecules

                MolecularDataCall::Molecule molTmp = mol->Molecules()[chainTmp.FirstMoleculeIndex()+m];
                for (uint res = 0; res < molTmp.ResidueCount(); ++res) { // Loop through residues

                    const MolecularDataCall::Residue *resTmp = mol->Residues()[molTmp.FirstResidueIndex()+res];

                    for (uint at = 0; at < resTmp->AtomCount(); ++at) { // Loop through atoms
                        uint atomIdx = resTmp->FirstAtomIndex() + at;

                        this->gridDataPos.Peek()[4*particleCnt+0] = mol->AtomPositions()[3*atomIdx+0];
                        this->gridDataPos.Peek()[4*particleCnt+1] = mol->AtomPositions()[3*atomIdx+1];
                        this->gridDataPos.Peek()[4*particleCnt+2] = mol->AtomPositions()[3*atomIdx+2];
                        if(this->qsSclVanDerWaals) {
                            gridDataPos.Peek()[4*particleCnt+3] = mol->AtomTypes()[mol->AtomTypeIndices()[atomIdx]].Radius();
                        }
                        else {
                            gridDataPos.Peek()[4*particleCnt+3] = 1.0f;
                        }
                        if(gridDataPos.Peek()[4*particleCnt+3] > maxAtomRad) {
                            maxAtomRad = gridDataPos.Peek()[4*particleCnt+3];
                        }
                        particleCnt++;
                    }
                }
            }
        }
    }

    // Compute padding for the density map
    padding = maxAtomRad*this->qsParticleRad + this->qsGridSpacing*10;

    Cuboid<float> bboxParticles = mol->AccessBoundingBoxes().ObjectSpaceBBox();

//    // DEBUG
//    printf("bbox , org %f %f %f, maxCoord %f %f %f\n",
//            bboxParticles.GetOrigin().X(),
//            bboxParticles.GetOrigin().Y(),
//            bboxParticles.GetOrigin().Z(),
//            bboxParticles.GetRightTopFront().X(),
//            bboxParticles.GetRightTopFront().Y(),
//            bboxParticles.GetRightTopFront().Z());

    // Init grid parameters
    gridDensMap.minC[0] = bboxParticles.GetLeft()   - padding;
    gridDensMap.minC[1] = bboxParticles.GetBottom() - padding;
    gridDensMap.minC[2] = bboxParticles.GetBack()   - padding;
    gridDensMap.maxC[0] = bboxParticles.GetRight() + padding;
    gridDensMap.maxC[1] = bboxParticles.GetTop()   + padding;
    gridDensMap.maxC[2] = bboxParticles.GetFront() + padding;
    gridXAxisLen = gridDensMap.maxC[0] - gridDensMap.minC[0];
    gridYAxisLen = gridDensMap.maxC[1] - gridDensMap.minC[1];
    gridZAxisLen = gridDensMap.maxC[2] - gridDensMap.minC[2];
    gridDensMap.size[0] = (int) ceil(gridXAxisLen / this->qsGridSpacing);
    gridDensMap.size[1] = (int) ceil(gridYAxisLen / this->qsGridSpacing);
    gridDensMap.size[2] = (int) ceil(gridZAxisLen / this->qsGridSpacing);
    gridXAxisLen = (gridDensMap.size[0]-1) * this->qsGridSpacing;
    gridYAxisLen = (gridDensMap.size[1]-1) * this->qsGridSpacing;
    gridZAxisLen = (gridDensMap.size[2]-1) * this->qsGridSpacing;
    gridDensMap.maxC[0] = gridDensMap.minC[0] + gridXAxisLen;
    gridDensMap.maxC[1] = gridDensMap.minC[1] + gridYAxisLen;
    gridDensMap.maxC[2] = gridDensMap.minC[2] + gridZAxisLen;
    gridDensMap.delta[0] = this->qsGridSpacing;
    gridDensMap.delta[1] = this->qsGridSpacing;
    gridDensMap.delta[2] = this->qsGridSpacing;

//    // DEBUG Density grid params
//    printf("Density grid size %u %u %u\n", gridDensMap.size[0], gridDensMap.size[1],
//            gridDensMap.size[2]);
//    printf("Density grid org %f %f %f\n", gridDensMap.minC[0], gridDensMap.minC[1],
//            gridDensMap.minC[2]);
//    // DEBUG

    // Set particle positions
#pragma omp parallel for
    for (int cnt = 0; cnt < static_cast<int>(mol->AtomCount()); ++cnt) {
            this->gridDataPos.Peek()[4*cnt+0] -= gridDensMap.minC[0];
            this->gridDataPos.Peek()[4*cnt+1] -= gridDensMap.minC[1];
            this->gridDataPos.Peek()[4*cnt+2] -= gridDensMap.minC[2];
    }


    // Compute uniform grid
    int rc = cqs->calc_map(
            particleCnt,
            &this->gridDataPos.Peek()[0],
            NULL,                 // Pointer to 'color' array
            false,                // Do not use 'color' array
            (float*)&gridDensMap.minC,
            (int*)&gridDensMap.size,
            maxAtomRad,
            this->qsParticleRad, // Radius scaling
            this->qsGridSpacing,
            this->qsIsoVal,
            this->qsGaussLim);

    if (rc != 0) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Quicksurf class returned val != 0\n", this->ClassName());
        return false;
    }

    return true;
}

#if defined(USE_DISTANCE_FIELD)
/*
 * SurfaceMappingTest::computeDistField
 */
bool SurfaceMappingTest::computeDistField(
        CudaDevArr<float> &vertexPos_D,
        uint vertexCnt,
        CudaDevArr<float> &distField_D,
        gridParams &gridDistField) {

    size_t volSize = gridDistField.size[0]*gridDistField.size[1]*
            gridDistField.size[2];

    // (Re)allocate memory if necessary
    if (!CudaSafeCall(distField_D.Validate(volSize))) {
        return false;
    }

    // Init grid parameters
    if (!CudaSafeCall(InitVolume(
            make_uint3(gridDistField.size[0], gridDistField.size[1], gridDistField.size[2]),
            make_float3(gridDistField.minC[0], gridDistField.minC[1], gridDistField.minC[2]),
            make_float3(gridDistField.delta[0], gridDistField.delta[1], gridDistField.delta[2])))) {
        return false;
    }

    // Compute distance field
    if (!CudaSafeCall(ComputeDistField(
            vertexPos_D.Peek(),
            distField_D.Peek(),
            volSize,
            vertexCnt,
            0,
            3))) {
        return false;
    }

    return true;
}
#endif // defined(USE_DISTANCE_FIELD)


/*
 * SurfaceMappingTest::getRMS
 */
float SurfaceMappingTest::getRMS(float *atomPos0, float *atomPos1,
        unsigned int cnt, bool fit, int flag, float rotation[3][3],
        float translation[3]) {

    // All particles are weighted with one
    if (this->rmsMask.GetCount() < cnt) {
        this->rmsMask.Validate(cnt);
        this->rmsWeights.Validate(cnt);
#pragma omp parallel for
        for (int a = 0; a < static_cast<int>(cnt) ; ++a) {
            this->rmsMask.Peek()[a] = 1;
            this->rmsWeights.Peek()[a] = 1.0f;
        }
    }

    float rmsValue = CalculateRMS(
            cnt,                     // Number of positions in each vector
            fit,
            flag,
            this->rmsWeights.Peek(), // Weights for the particles
            this->rmsMask.Peek(),    // Which particles should be considered
            this->rmsPosVec1.Peek(), // Vector to be fit
            this->rmsPosVec0.Peek(), // Vector
            rotation,                    // Saves the rotation matrix
            translation                  // Saves the translation vector
    );

    if (rotation != NULL) {

        float tempRot[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                tempRot[i][j] = rotation[i][j];
            }

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                rotation[i][j] = tempRot[j][i];
            }

//        printf("RMS translation %f %f %f\n", translation[0],
//                translation[1], translation[2]);
//        printf("RMS rotation\n %f %f %f\n%f %f %f\n%f %f %f\n",
//                rotation[0][0], rotation[0][1], rotation[0][2],
//                rotation[1][0], rotation[1][1], rotation[1][2],
//                rotation[2][0], rotation[2][1], rotation[2][2]);
    }

//    printf("RMS value %f\n", rmsValue);

    return rmsValue;
}


/*
 * SurfaceMappingTest::getRMSPosArray
 */
bool SurfaceMappingTest::getRMSPosArray(MolecularDataCall *mol,
        HostArr<float> &posArr, unsigned int &cnt) {
    using namespace vislib::sys;

    cnt = 0;

    // Use all particles for RMS fitting
    if (this->fittingMode == RMS_ALL) {

        // Extracting protein atoms from mol 0
        for (uint sec = 0; sec < mol->SecondaryStructureCount(); ++sec) {
            for (uint acid = 0; acid < mol->SecondaryStructures()[sec].AminoAcidCount(); ++acid) {
                const MolecularDataCall::AminoAcid *aminoAcid =
                        dynamic_cast<const MolecularDataCall::AminoAcid*>(
                                (mol->Residues()[mol->SecondaryStructures()[sec].
                                                  FirstAminoAcidIndex()+acid]));
                if (aminoAcid == NULL) {
                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                            "%s: Unable to perform RMS fitting using all protein\
atoms (residue mislabeled as 'amino acid')", this->ClassName());
                    return false;
                }
                for (uint at = 0; at < aminoAcid->AtomCount(); ++at) {
                    posArr.Peek()[3*cnt+0] =
                            mol->AtomPositions()[3*(aminoAcid->FirstAtomIndex()+at)+0];
                    posArr.Peek()[3*cnt+1] =
                            mol->AtomPositions()[3*(aminoAcid->FirstAtomIndex()+at)+1];
                    posArr.Peek()[3*cnt+2] =
                            mol->AtomPositions()[3*(aminoAcid->FirstAtomIndex()+at)+2];
                    cnt++;
                }
            }
        }
    } else if (this->fittingMode == RMS_BACKBONE) { // Use backbone atoms for RMS fitting

        // Extracting backbone atoms from mol 0
        for (uint sec = 0; sec < mol->SecondaryStructureCount(); ++sec) {

            for (uint acid = 0; acid < mol->SecondaryStructures()[sec].AminoAcidCount(); ++acid) {

                uint cAlphaIdx =
                        ((const MolecularDataCall::AminoAcid*)
                                (mol->Residues()[mol->SecondaryStructures()[sec].
                                                  FirstAminoAcidIndex()+acid]))->CAlphaIndex();
                uint cCarbIdx =
                        ((const MolecularDataCall::AminoAcid*)
                                (mol->Residues()[mol->SecondaryStructures()[sec].
                                                  FirstAminoAcidIndex()+acid]))->CCarbIndex();
                uint nIdx =
                        ((const MolecularDataCall::AminoAcid*)
                                (mol->Residues()[mol->SecondaryStructures()[sec].
                                                  FirstAminoAcidIndex()+acid]))->NIndex();
                uint oIdx =
                        ((const MolecularDataCall::AminoAcid*)
                                (mol->Residues()[mol->SecondaryStructures()[sec].
                                                  FirstAminoAcidIndex()+acid]))->OIndex();

                posArr.Peek()[3*cnt+0] = mol->AtomPositions()[3*cAlphaIdx+0];
                posArr.Peek()[3*cnt+1] = mol->AtomPositions()[3*cAlphaIdx+1];
                posArr.Peek()[3*cnt+2] = mol->AtomPositions()[3*cAlphaIdx+2];
                cnt++;
                posArr.Peek()[3*cnt+0] = mol->AtomPositions()[3*cCarbIdx+0];
                posArr.Peek()[3*cnt+1] = mol->AtomPositions()[3*cCarbIdx+1];
                posArr.Peek()[3*cnt+2] = mol->AtomPositions()[3*cCarbIdx+2];
                cnt++;
                posArr.Peek()[3*cnt+0] = mol->AtomPositions()[3*oIdx+0];
                posArr.Peek()[3*cnt+1] = mol->AtomPositions()[3*oIdx+1];
                posArr.Peek()[3*cnt+2] = mol->AtomPositions()[3*oIdx+2];
                cnt++;
                posArr.Peek()[3*cnt+0] = mol->AtomPositions()[3*nIdx+0];
                posArr.Peek()[3*cnt+1] = mol->AtomPositions()[3*nIdx+1];
                posArr.Peek()[3*cnt+2] = mol->AtomPositions()[3*nIdx+2];
                cnt++;
            }
        }
    } else if (this->fittingMode == RMS_C_ALPHA) { // Use C alpha atoms for RMS fitting
        // Extracting C alpha atoms from mol 0
        for (uint sec = 0; sec < mol->SecondaryStructureCount(); ++sec) {
            MolecularDataCall::SecStructure secStructure = mol->SecondaryStructures()[sec];
            for (uint acid = 0; acid < secStructure.AminoAcidCount(); ++acid) {
                uint cAlphaIdx =
                        ((const MolecularDataCall::AminoAcid*)
                                (mol->Residues()[secStructure.
                                                  FirstAminoAcidIndex()+acid]))->CAlphaIndex();
                posArr.Peek()[3*cnt+0] = mol->AtomPositions()[3*cAlphaIdx+0];
                posArr.Peek()[3*cnt+1] = mol->AtomPositions()[3*cAlphaIdx+1];
                posArr.Peek()[3*cnt+2] = mol->AtomPositions()[3*cAlphaIdx+2];
                cnt++;
            }
        }
    }

    return true;
}


/*
 * SurfaceMappingTest::isosurfComputeVertices
 */
bool SurfaceMappingTest::isosurfComputeVertices(
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
        CudaDevArr<uint> &triangleIdx_D) {

    using vislib::sys::Log;

    uint gridCellCnt = (gridDensMap.size[0]-1)*(gridDensMap.size[1]-1)*(gridDensMap.size[2]-1);
    uint activeCellCnt, triangleVtxCnt;


    /* Init grid parameters */

    if (!CudaSafeCall(InitVolume(
            make_uint3(gridDensMap.size[0], gridDensMap.size[1], gridDensMap.size[2]),
            make_float3(gridDensMap.minC[0], gridDensMap.minC[1], gridDensMap.minC[2]),
            make_float3(gridDensMap.delta[0], gridDensMap.delta[1], gridDensMap.delta[2])))) {
        return false;
    }

    if (!CudaSafeCall(InitVolume_surface_generation(
            make_uint3(gridDensMap.size[0], gridDensMap.size[1], gridDensMap.size[2]),
            make_float3(gridDensMap.minC[0], gridDensMap.minC[1], gridDensMap.minC[2]),
            make_float3(gridDensMap.delta[0], gridDensMap.delta[1], gridDensMap.delta[2])))) {
        return false;
    }


    /* Find active grid cells */

    if (!CudaSafeCall(this->cubeStates_D.Validate(gridCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->cubeOffsets_D.Validate(gridCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->cubeStates_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(this->cubeOffsets_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(FindActiveGridCells(
            this->cubeStates_D.Peek(),
            this->cubeOffsets_D.Peek(),
            gridCellCnt,
            this->qsIsoVal,
            volume_D))) {
        return false;
    }

//    // DEBUG Print Cube states and offsets
//    HostArr<uint> cubeStates;
//    HostArr<uint> cubeOffsets;
//    cubeStates.Validate(gridCellCnt);
//    cubeOffsets.Validate(gridCellCnt);
//    this->cubeStates_D.CopyToHost(cubeStates.Peek());
//    this->cubeOffsets_D.CopyToHost(cubeOffsets.Peek());
//    for (int i = 0; i < gridCellCnt; ++i) {
//        printf ("Cell %i: state %u, offs %u\n", i, cubeStates.Peek()[i],
//                cubeOffsets.Peek()[i]);
//    }
//    // END DEBUG


    /* Get number of active grid cells */

    activeCellCnt =
            this->cubeStates_D.GetAt(gridCellCnt-1) +
            this->cubeOffsets_D.GetAt(gridCellCnt-1);
    if (!CheckForCudaError()) {
        return false;
    }


    //printf("Active cell count %u\n", activeCellCnt); // DEBUG


    /* Prepare cube map */

    if (!CudaSafeCall(cubeMapInv_D.Validate(gridCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(cubeMapInv_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(cubeMap_D.Validate(activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(CalcCubeMap(
            cubeMap_D.Peek(),
            cubeMapInv_D.Peek(),
            this->cubeOffsets_D.Peek(),
            this->cubeStates_D.Peek(),
            gridCellCnt))) {
        return false;
    }

//    printf("Cube map size %u\n", activeCellCnt);

    // DEBUG Cube map
//    HostArr<uint> cubeMap;
//    HostArr<uint> cubeMapInv;
//    cubeMap.Validate(activeCellCnt);
//    cubeMapInv.Validate(gridCellCnt);
//    cubeMapInv_D.CopyToHost(cubeMapInv.Peek());
//    cubeMap_D.CopyToHost(cubeMap.Peek());
//    for (int i = 0; i < gridCellCnt; ++i) {
//        printf ("Cell %i: cubeMapInv %u\n", i, cubeMapInv.Peek()[i]);
//    }
//    for (int i = 0; i < activeCellCnt; ++i) {
//        printf ("Cell %i: cubeMap %u\n", i, cubeMap.Peek()[i]);
//    }
    // END DEBUG


    /* Get vertex positions */

    if (!CudaSafeCall(this->vertexStates_D.Validate(7*activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->activeVertexPos_D.Validate(7*activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->vertexIdxOffs_D.Validate(7*activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->vertexStates_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(this->activeVertexPos_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(this->vertexIdxOffs_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(CalcVertexPositions(
            this->vertexStates_D.Peek(),
            this->activeVertexPos_D.Peek(),
            this->vertexIdxOffs_D.Peek(),
            cubeMap_D.Peek(),
            activeCellCnt,
            this->qsIsoVal,
            volume_D))) {
        return false;
    }

//    // DEBUG Print active vertex positions
//    HostArr<float3> activeVertexPos;
//    HostArr<uint> vertexStates;
//    HostArr<uint> vertexIdxOffsets;
//    activeVertexPos.Validate(7*activeCellCnt);
//    vertexIdxOffsets.Validate(7*activeCellCnt);
//    vertexStates.Validate(7*activeCellCnt);
//    cudaMemcpy(vertexStates.Peek(), this->vertexStates_D.Peek(), 7*activeCellCnt,
//            cudaMemcpyDeviceToHost);
//    cudaMemcpy(activeVertexPos.Peek(), this->activeVertexPos_D.Peek(), 7*activeCellCnt,
//            cudaMemcpyDeviceToHost);
//    cudaMemcpy(vertexIdxOffsets.Peek(), this->vertexIdxOffs_D.Peek(), 7*activeCellCnt,
//            cudaMemcpyDeviceToHost);
//    for (int i = 0; i < 7*activeCellCnt; ++i) {
//        printf("#%i: active vertexPos %f %f %f (state = %u)\n", i,
//                activeVertexPos.Peek()[i].x,
//                activeVertexPos.Peek()[i].y,
//                activeVertexPos.Peek()[i].z,
//                vertexStates.Peek()[i]);
//    }
//
//    for (int i = 0; i < 7*activeCellCnt; ++i) {
//        printf("#%i: vertex index offset %u (state %u)\n",i,
//                vertexIdxOffsets.Peek()[i],
//                vertexStates.Peek()[i]);
//    }
//    // END DEBUG


    /* Get number of active vertices */

    activeVertexCount =
            this->vertexStates_D.GetAt(7*activeCellCnt-1) +
            this->vertexIdxOffs_D.GetAt(7*activeCellCnt-1);
    if (!CheckForCudaError()) {
        return false;
    }
    if (!CudaSafeCall(vertexPos_D.Validate(activeVertexCount*3))) {
        return false;
    }

    /* Compact list of vertex positions (keep only active vertices) */

    if (!CudaSafeCall(CompactActiveVertexPositions(
            vertexPos_D.Peek(),
            this->vertexStates_D.Peek(),
            this->vertexIdxOffs_D.Peek(),
            this->activeVertexPos_D.Peek(),
            activeCellCnt,
            0,  // Array data  offset
            3    // Array data element size
            ))) {
        return false;
    }

//    // DEBUG Print vertex positions
//    HostArr<float> vertexPos;
//    vertexPos.Validate(activeVertexCount*this->vboBuffDataSize);
//    cudaMemcpy(vertexPos.Peek(), vboPt, activeVertexCount*this->vboBuffDataSize,
//            cudaMemcpyDeviceToHost);
//    for (int i = 0; i < activeVertexCount; ++i) {
//        printf("#%i: vertexPos %f %f %f\n", i, vertexPos.Peek()[9*i+0],
//                vertexPos.Peek()[9*i+1], vertexPos.Peek()[9*i+2]);
//    }
//    // END DEBUG


    /* Calc vertex index map */

    if (!CudaSafeCall(vertexMap_D.Validate(activeVertexCount))) return false;
    if (!CudaSafeCall(vertexMapInv_D.Validate(7*activeCellCnt))) return false;
    if (!CudaSafeCall(CalcVertexMap(
            vertexMap_D.Peek(),
            vertexMapInv_D.Peek(),
            this->vertexIdxOffs_D.Peek(),
            this->vertexStates_D.Peek(),
            activeCellCnt))) {
        return false;
    }

//    // DEBUG Print vertex map
//    HostArr<uint> vertexMap;
//    vertexMap.Validate(activeVertexCount);
//    vertexMap_D.CopyToHost(vertexMap.Peek());
//    for (int i = 0; i < vertexMap_D.GetSize(); ++i) {
//        printf("Vertex mapping %i: %u\n", i, vertexMap.Peek()[i]);
//    }
//    // END DEBUG


    /* Flag tetrahedrons */

    if (!CudaSafeCall(this->verticesPerTetrahedron_D.Validate(6*activeCellCnt))) return false;
    if (!CudaSafeCall(FlagTetrahedrons(
            this->verticesPerTetrahedron_D.Peek(),
            cubeMap_D.Peek(),
            this->qsIsoVal,
            activeCellCnt,
            volume_D))) {
        return false;
    }


    /* Scan tetrahedrons */

    if (!CudaSafeCall(this->tetrahedronVertexOffsets_D.Validate(6*activeCellCnt))) return false;
    if (!CudaSafeCall(GetTetrahedronVertexOffsets(
            this->tetrahedronVertexOffsets_D.Peek(),
            this->verticesPerTetrahedron_D.Peek(),
            activeCellCnt*6))) {
        return false;
    }


    /* Get triangle vertex count */

    triangleVtxCnt =
            this->tetrahedronVertexOffsets_D.GetAt(activeCellCnt*6-1) +
            this->verticesPerTetrahedron_D.GetAt(activeCellCnt*6-1);
    if (!CheckForCudaError()) {
        return false;
    }

    triangleCount = triangleVtxCnt/3;


    /* Generate triangles */

    if (!CudaSafeCall(triangleIdx_D.Validate(triangleVtxCnt))) {
        return false;
    }
    if (!CudaSafeCall(triangleIdx_D.Set(0x00))) {
        return false;
    }

    if (!CudaSafeCall(GetTrianglesIdx(
            this->tetrahedronVertexOffsets_D.Peek(),
            cubeMap_D.Peek(),
            cubeMapInv_D.Peek(),
            this->qsIsoVal,
            activeCellCnt*6,
            activeCellCnt,
            triangleIdx_D.Peek(),
            vertexMapInv_D.Peek(),
            volume_D))) {
        return false;
    }

    /* Compute neighbours */

    if (!CudaSafeCall(vertexNeighbours_D.Validate(activeVertexCount*18))) {
        return false;
    }
    if (!CudaSafeCall(vertexNeighbours_D.Set(-1))) {
        return false;
    }
    if (!CudaSafeCall(ComputeVertexConnectivity(
            vertexNeighbours_D.Peek(),
            this->vertexStates_D.Peek(),
            vertexMap_D.Peek(),
            vertexMapInv_D.Peek(),
            cubeMap_D.Peek(),
            cubeMapInv_D.Peek(),
            this->cubeStates_D.Peek(),
            activeVertexCount,
            volume_D,
            this->qsIsoVal))) {

//        // DEBUG Print neighbour indices
//        HostArr<int> vertexNeighbours;
//        vertexNeighbours.Validate(vertexNeighbours_D.GetSize());
//        vertexNeighbours_D.CopyToHost(vertexNeighbours.Peek());
//        for (int i = 0; i < vertexNeighbours_D.GetSize()/18; ++i) {
//            printf("Neighbours vtx #%i: ", i);
//            for (int j = 0; j < 18; ++j) {
//                printf("%i ", vertexNeighbours.Peek()[i*18+j]);
//            }
//            printf("\n");
//        }
//        // END DEBUG

        return false;
    }


    // This is actually slower ...
//    if (!CudaSafeCall(MapNeighbourIdxToActiveList(
//            vertexNeighbours_D.Peek(),
//            vertexMapInv_D.Peek(),
//            vertexNeighbours_D.GetCount()))) {
//        return false;
//    }


//    // DEBUG Print neighbour indices
//    HostArr<int> vertexNeighbours;
//    vertexNeighbours.Validate(vertexNeighbours_D.GetSize());
//    vertexNeighbours_D.CopyToHost(vertexNeighbours.Peek());
//    for (int i = 0; i < vertexNeighbours_D.GetSize()/18; ++i) {
//        printf("Neighbours vtx #%i: ", i);
//        for (int j = 0; j < 18; ++j) {
//            printf("%i ", vertexNeighbours.Peek()[i*18+j]);
//        }
//        printf("\n");
//    }
//    // END DEBUG

//    // DEBUG Print initial positions
//    HostArr<float> vertexPos;
//    vertexPos.Validate(activeVertexCount*3);
//    cudaMemcpy(vertexPos.Peek(), vertexPos_D.Peek(),
//            sizeof(float)*activeVertexCount*3,
//            cudaMemcpyDeviceToHost);
//    printf("Initial positions:\n");
//    for (int i = 0; i < 10; ++i) {
//        printf("%i: Vertex position (%f %f %f)\n", i,
//                vertexPos.Peek()[3*i+0],
//                vertexPos.Peek()[3*i+1],
//                vertexPos.Peek()[3*i+2]);
//    }
//    // End DEBUG

    return CheckForCudaErrorSync(); // Error check with sync
}


/*
 * SurfaceMappingTest::mapIsosurfaceToVolume
 */
bool SurfaceMappingTest::mapIsosurfaceToVolume(
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
        InterpolationMode interpMode) {

    using vislib::sys::Log;

    if (volume_D == NULL) {
        return false;
    }

#if (defined(USE_TIMER) && defined(VERBOSE))
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif // defined(USE_TIMER)


    /* Init grid parameters for all files */

    if (!CudaSafeCall(InitVolume(
            make_uint3(gridDensMap.size[0], gridDensMap.size[1], gridDensMap.size[2]),
            make_float3(gridDensMap.minC[0], gridDensMap.minC[1], gridDensMap.minC[2]),
            make_float3(gridDensMap.delta[0], gridDensMap.delta[1], gridDensMap.delta[2])))) {
        return false;
    }

    if (!CudaSafeCall(InitVolume_surface_mapping(
            make_uint3(gridDensMap.size[0], gridDensMap.size[1], gridDensMap.size[2]),
            make_float3(gridDensMap.minC[0], gridDensMap.minC[1], gridDensMap.minC[2]),
            make_float3(gridDensMap.delta[0], gridDensMap.delta[1], gridDensMap.delta[2])))) {
        return false;
    }

    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(vertexCnt))) {
        return false;
    }

    // Init forces scale factor with -1 or 1, depending on whether they start
    // outside or inside the isosurface
    if (!CudaSafeCall (InitExternalForceScl(
            this->vertexExternalForcesScl_D.Peek(),
            volume_D,
            vertexPos_D.Peek(),
            this->vertexExternalForcesScl_D.GetCount(),
            this->qsIsoVal,
            0,
            3))) {
        return false;
    }


    // Compute gradient
    if (!CudaSafeCall(this->volGradient_D.Validate(
            gridDensMap.size[0]*gridDensMap.size[1]*gridDensMap.size[2]))) {
        return false;
    }
    if (!CudaSafeCall(this->volGradient_D.Set(0))) {
        return false;
    }
#if defined(USE_DISTANCE_FIELD)

//    // DEBUG Print distance field
//    HostArr<float> distFieldTest;
//    distFieldTest.Validate(this->distField_D.GetSize());
//    if (!CudaSafeCall(this->distField_D.CopyToHost(distFieldTest.Peek()))) {
//        return false;
//    }
//    for (int i = 0; i < this->gradientDensMap0_D.GetSize(); ++i) {
//         printf("Distfield: %f \n", distFieldTest.Peek()[i]);
//    }
//    // END DEBUG

    if (!CudaSafeCall(CalcVolGradientWithDistField(
            this->volGradient_D.Peek(),
            volume_D,
            this->distField_D.Peek(),
            this->maxGaussianVolumeDist,
            this->qsIsoVal,
            this->volGradient_D.GetSize()))) {
        return false;
    }

#else
    if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(), volume_D,
            this->volGradient_D.GetSize()))) {
        return false;
    }
#endif // defined(USE_DISTANCE_FIELD)

//    // DEBUG Print mapped positions
//    printf("Mapped positions before:\n");
//    HostArr<float> vertexPos;
//    vertexPos.Validate(vertexCnt*3);
//    cudaMemcpy(vertexPos.Peek(), vertexPos_D.Peek(),
//            sizeof(float)*vertexCnt*3,
//            cudaMemcpyDeviceToHost);
//    for (int k = 0; k < 10; ++k) {
//        printf("%i: Vertex position (%f %f %f)\n", k, vertexPos.Peek()[3*k+0],
//                vertexPos.Peek()[3*k+1], vertexPos.Peek()[3*k+2]);
//
//    }
//    // End DEBUG

    if (!CudaSafeCall(this->displLen_D.Validate(vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Set(0))) {
        return false;
    }

    // Laplacian
    if (!CudaSafeCall(this->laplacian_D.Validate(vertexCnt))) {
        return false;
    }

    if (interpMode == INTERP_LINEAR) {
        for (uint i = 0; i < maxIt; i += UPDATE_VTX_POS_ITERATIONS_PER_KERNEL) {

            // Update position for all vertices
            if (!CudaSafeCall(UpdateVertexPositionTrilinear(
                    volume_D,
                    vertexPos_D.Peek(),
                    this->vertexExternalForcesScl_D.Peek(),
                    vertexNeighbours_D.Peek(),
                    this->volGradient_D.Peek(),
                    this->laplacian_D.Peek(),
                    vertexCnt,
                    externalForcesWeight,
                    forceScl,
                    springStiffness,
                    this->qsIsoVal,
                    this->minDispl,
                    0,     // Offset for positions in array
                    3))) { // Stride in array
                return false;
            }
        }
    } else {
        for (uint i = 0; i < maxIt; i += UPDATE_VTX_POS_ITERATIONS_PER_KERNEL) {

//            // Update position for all vertices
//            if (!CudaSafeCall(UpdateVertexPositionTricubic(
//                    volume_D,
//                    vertexPos_D.Peek(),
//                    this->vertexExternalForcesScl_D.Peek(),
//                    vertexNeighbours_D.Peek(),
//                    this->volGradient_D.Peek(),
//                    this->laplacian_D.Peek(),
//                    vertexCnt,
//                    externalForcesWeight,
//                    forceScl,
//                    springStiffness,
//                    this->qsIsoVal,
//                    this->minDispl,
//                    0,     // Offset for positions in array
//                    3))) { // Stride in array
//                return false;
//            }
        }
    }

//    // DEBUG Print mapped positions
//    printf("Mapped positions after:\n");
////    HostArr<float> vertexPos;
//    vertexPos.Validate(vertexCnt*3);
//    cudaMemcpy(vertexPos.Peek(), vertexPos_D.Peek(),
//            sizeof(float)*vertexCnt*3,
//            cudaMemcpyDeviceToHost);
//    for (int k = 0; k < 10; ++k) {
//        printf("%i: Vertex position (%f %f %f)\n", k, vertexPos.Peek()[3*k+0],
//                vertexPos.Peek()[3*k+1], vertexPos.Peek()[3*k+2]);
//
//    }
//
//    // End DEBUG

#if (defined(USE_TIMER) && defined(VERBOSE))
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "%s: Time for surface mapping (%u iterations, %u vertices): %f sec\n",
            this->ClassName(),
            maxIt, vertexCnt, dt_ms/1000.0f);
#endif // defined(USE_TIMER)

    return CheckForCudaErrorSync(); // Error check with sync
}

/*
 * SurfaceMappingTest::regularizeSurface
 */
bool SurfaceMappingTest::regularizeSurface(
        float *volume_D,
        gridParams gridDensMap,
        CudaDevArr<float> &vertexPos_D,
        uint vertexCnt,
        CudaDevArr<int> &vertexNeighbours_D,
        uint maxIt,
        float springStiffness,
        float forceScl,
        float externalForcesWeight,
        InterpolationMode interpMode) {

    using vislib::sys::Log;

    if (volume_D == NULL) {
        return false;
    }

    /* Init grid parameters */

    if (!CudaSafeCall(InitVolume(
            make_uint3(gridDensMap.size[0], gridDensMap.size[1], gridDensMap.size[2]),
            make_float3(gridDensMap.minC[0], gridDensMap.minC[1], gridDensMap.minC[2]),
            make_float3(gridDensMap.delta[0], gridDensMap.delta[1], gridDensMap.delta[2])))) {
        return false;
    }

    if (!CudaSafeCall(InitVolume_surface_mapping(
            make_uint3(gridDensMap.size[0], gridDensMap.size[1], gridDensMap.size[2]),
            make_float3(gridDensMap.minC[0], gridDensMap.minC[1], gridDensMap.minC[2]),
            make_float3(gridDensMap.delta[0], gridDensMap.delta[1], gridDensMap.delta[2])))) {
        return false;
    }

    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Set(0x00))) {
        return false;
    }

    // Init forces scale factor with -1 or 1, depending on whether they start
    // outside or inside the isosurface
    if (!CudaSafeCall (InitExternalForceScl(
            this->vertexExternalForcesScl_D.Peek(),
            volume_D,
            vertexPos_D.Peek(),
            static_cast<uint>(this->vertexExternalForcesScl_D.GetCount()),
            this->qsIsoVal,
            0,
            3))) {
        return false;
    }

    // Compute gradient
    if (!CudaSafeCall(this->volGradient_D.Validate(
            gridDensMap.size[0]*gridDensMap.size[1]*gridDensMap.size[2]))) {
        return false;
    }
    if (!CudaSafeCall(this->volGradient_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(), volume_D,
            this->volGradient_D.GetSize()))) {
        return false;
    }

//    CheckForCudaErrorSync();

//    // DEBUG Print regularized positions
//    printf("Positions before\n");
//    HostArr<float> vertexPos;
//    vertexPos.Validate(vertexCnt*3);
//    cudaMemcpy(vertexPos.Peek(), vertexPos_D.Peek(),
//            sizeof(float)*vertexCnt*3,
//            cudaMemcpyDeviceToHost);
//    for (int k = 0; k < 10; ++k) {
//        printf("%i: Vertex position (%f %f %f)\n", k, vertexPos.Peek()[3*k+0],
//                vertexPos.Peek()[3*k+1], vertexPos.Peek()[3*k+2]);
//
//    }
//    // End DEBUG

    if (!CudaSafeCall(this->displLen_D.Validate(vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Set(0))) {
        return false;
    }

    // Laplacian
    if (!CudaSafeCall(this->laplacian_D.Validate(vertexCnt))) {
        return false;
    }

    if (!CudaSafeCall(this->laplacian_D.Set(0))) {
        return false;
    }

    if (interpMode == INTERP_LINEAR) {
        for (uint i = 0; i < maxIt; ++i) {

            if (!CudaSafeCall(UpdateVertexPositionTrilinear(
                    volume_D,
                    vertexPos_D.Peek(),
                    this->vertexExternalForcesScl_D.Peek(),
                    vertexNeighbours_D.Peek(),
                    this->volGradient_D.Peek(),
                    this->laplacian_D.Peek(),
                    vertexCnt,
                    externalForcesWeight,
                    forceScl,
                    springStiffness,
                    this->qsIsoVal,
                    this->minDispl,
                    0,     // Offset for positions in array
                    3))) { // Stride in array
                return false;
            }
        }
    } else {
        for (uint i = 0; i < maxIt; ++i) {

//            if (!CudaSafeCall(UpdateVertexPositionTricubic(
//                    volume_D,
//                    vertexPos_D.Peek(),
//                    this->vertexExternalForcesScl_D.Peek(),
//                    vertexNeighbours_D.Peek(),
//                    this->volGradient_D.Peek(),
//                    this->laplacian_D.Peek(),
//                    vertexCnt,
//                    externalForcesWeight,
//                    forceScl,
//                    springStiffness,
//                    this->qsIsoVal,
//                    this->minDispl,
//                    0,     // Offset for positions in array
//                    3))) { // Stride in array
//                return false;
//            }
        }
    }

//    printf("Parameters: vertex count %u, externalForcesWeight %f, forceScl %f, springStiffness %f, springEquilibriumLength %f, this->qsIsoVal %f minDispl %f\n",
//            vertexCnt, externalForcesWeight, forceScl, springStiffness, springEquilibriumLength,
//            this->qsIsoVal, 0.0);
//
//
//    // DEBUG Print regularized positions
//    printf("Positions after\n");
////    HostArr<float> vertexPos;
//    vertexPos.Validate(vertexCnt*3);
//    cudaMemcpy(vertexPos.Peek(), vertexPos_D.Peek(),
//            sizeof(float)*vertexCnt*3,
//            cudaMemcpyDeviceToHost);
//    for (int k = 0; k < 10; ++k) {
//        printf("%i: Vertex position (%f %f %f)\n", k, vertexPos.Peek()[3*k+0],
//                vertexPos.Peek()[3*k+1], vertexPos.Peek()[3*k+2]);
//
//    }
//    // End DEBUG

    return CheckForCudaErrorSync(); // Error check with sync
}


bool SurfaceMappingTest::plotMappingIterationsByExternalForceWeighting() {

    printf("Plotting iterations by external force weighting...\n");

    unsigned int posCnt0, posCnt1;
    float rotation[3][3];
    float translation[3];
    float rmsVal;

    /* Set parameters */

    const uint frame0 = 0;

    const uint regMaxIt = 500;
    const float regSpringStiffness = 0.3f;
    const float regForcesScl = 0.1f;
    const float regExternalForcesWeight = 0.5f;
    const InterpolationMode regInterpolMode = INTERP_LINEAR;

    const float mapSpringStiffness = 0.3f;
    const float mapForcesScl = 0.1f;
    const InterpolationMode mapInterpolMode =  INTERP_LINEAR;
    const float step = 0.05f;


    int itCntAvg = 0.0f;
    int itCntMax = -1;
    int itCntMin= 10000000;

    float corruptTrianglesAvg = 0.0f;
    float corruptTrianglesMin= 10000000;
    float corruptTrianglesMax = -1;

    float hausdorffAvg = 0.0f;
    float hausdorffMin= 10000000;
    float hausdorffMax = -1;


    // Get particles
    MolecularDataCall *molCall = this->particleDataCallerSlot.CallAs<MolecularDataCall>();
    if (molCall == NULL) {
        return false;
    }

    // Get potential texture
    VTIDataCall *volCall = this->volumeDataCallerSlot.CallAs<VTIDataCall>();
    if (volCall == NULL) {
        return false;
    }

    /* Generate target surface */

    molCall->SetFrameID(frame0, true); // Set frame id and force flag
    if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }
    if (!(*molCall)(MolecularDataCall::CallForGetData)) {
        return false;
    }
    volCall->SetFrameID(frame0, true); // Set frame id and force flag
    if (!(*volCall)(VTIDataCall::CallForGetExtent)) {
        return false;
    }
    if (!(*volCall)(VTIDataCall::CallForGetData)) {
        return false;
    }

    this->rmsPosVec0.Validate(molCall->AtomCount()*3);

    // Get rms atom positions
    if (!this->getRMSPosArray(molCall, this->rmsPosVec0, posCnt0)) {
        return false;
    }

    // 1. Compute density map of variant #0
    if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf0,
            this->gridDensMap)) {
        return false;
    }

    // 2. Compute initial triangulation for variant #0
    if (!this->isosurfComputeVertices(
            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
            this->cubeMap0_D,
            this->cubeMapInv0_D,
            this->vertexMap0_D,
            this->vertexMapInv0_D,
            this->vertexNeighbours0_D,
            this->gridDensMap,
            this->vertexCnt0,
            this->vertexPos0_D,
            this->triangleCnt0,
            this->triangleIdx0_D)) {
        return false;
    }

    // 3. Make mesh more regular by using deformable model approach
    if (!this->regularizeSurface(
            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
            this->gridDensMap,
            this->vertexPos0_D,
            this->vertexCnt0,
            this->vertexNeighbours0_D,
            regMaxIt,
            regSpringStiffness,
            regForcesScl,
            regExternalForcesWeight,
            regInterpolMode)) {
        return false;
    }


#if defined(USE_DISTANCE_FIELD)
    // 4. Compute distance field based on regularized vertices of surface 0
    if (!this->computeDistField(
            this->vertexPos0_D,
            this->vertexCnt0,
            this->distField_D,
            this->gridDensMap)) {
        return false;
    }
#endif // defined(USE_DISTANCE_FIELD)

    // Init volume grid constants for metric functions (because they are
    // only visible at file scope
    if (!CudaSafeCall(InitVolume_metric(
            make_uint3(this->gridDensMap.size[0],
                    this->gridDensMap.size[1],
                    this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0],
                            this->gridDensMap.minC[1],
                            this->gridDensMap.minC[2]),
                            make_float3(this->gridDensMap.delta[0],
                                    this->gridDensMap.delta[1],
                                    this->gridDensMap.delta[2])))) {
        return false;
    }

    molCall->Unlock();


    // Loop through all values of the external force
    for (float w = 0.05f; w <= 1.00f; w += step) {


        itCntAvg = 0.0f;
        itCntMax = -1;
        itCntMin= 10000000;

        corruptTrianglesAvg = 0.0f;
        corruptTrianglesMin= 10000000;
        corruptTrianglesMax = -1;

        hausdorffAvg = 0.0f;
        hausdorffMin= 10000000;
        hausdorffMax = -1;


        if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
            return false;
        }

        if (!(*volCall)(VTIDataCall::CallForGetExtent)) {
            return false;
        }

        // Loop through all frames
        for (unsigned int fr = 0; fr < molCall->FrameCount(); ++fr) {

            /* Generate vertices for source shape */

            molCall->SetFrameID(fr, true);
            if (!(*molCall)(MolecularDataCall::CallForGetData)) {
                return false;
            }


//            volCall->SetFrameID(fr, true); // Set frame id and force flag
//            if (!(*volCall)(MolecularDataCall::CallForGetData)) {
//                return false;
//            }

            this->rmsPosVec1.Validate(molCall->AtomCount()*3);
            // Get atom positions
            if (!this->getRMSPosArray(molCall, this->rmsPosVec1, posCnt1)) {
                return false;
            }

            // Compute RMS value and transformation
            if (posCnt0 != posCnt1) {
                return false;
            }
            rmsVal = this->getRMS(this->rmsPosVec0.Peek(),
                    this->rmsPosVec1.Peek(), posCnt0, true, 2, rotation,
                    translation);
            if (rmsVal > 10.0f) {
                return false;
            }

            // 1. Compute density map of variant #1
            if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf1,
                    this->gridDensMap)) {
                return false;
            }


            // 2. Compute initial triangulation for variant #1
            if (!this->isosurfComputeVertices(
                    ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
                    this->cubeMap1_D,
                    this->cubeMapInv1_D,
                    this->vertexMap1_D,
                    this->vertexMapInv1_D,
                    this->vertexNeighbours1_D,
                    this->gridDensMap,
                    this->vertexCnt1,
                    this->vertexPos1_D,
                    this->triangleCnt1,
                    this->triangleIdx1_D)) {
                return false;
            }

//            // DEBUG Print mapped positions
//            HostArr<float> vertexPos;
//            vertexPos.Validate(vertexCnt1*3);
//            cudaMemcpy(vertexPos.Peek(), this->vertexPos1_D.Peek(),
//                    sizeof(float)*vertexCnt1*3, cudaMemcpyDeviceToHost);
//            for (int i = 0; i < 10; ++i) {
//
//                printf("%i: Vertex position (%f %f %f)\n", i,
//                        vertexPos.Peek()[3*i+0],
//                        vertexPos.Peek()[3*i+1],
//                        vertexPos.Peek()[3*i+2]);
//
//            }
//            // End DEBUG


            // 3. Make mesh more regular by using deformable model approach
            if (!this->regularizeSurface(
                    ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
                    this->gridDensMap,
                    this->vertexPos1_D,
                    this->vertexCnt1,
                    this->vertexNeighbours1_D,
                    regMaxIt,
                    regSpringStiffness,
                    regForcesScl,
                    regExternalForcesWeight,
                    regInterpolMode)) {
                return false;
            }


            // Init mapped pos array with old position
            if (!CudaSafeCall(this->vertexPosMapped_D.Validate(this->vertexCnt1*3))) {
                return false;
            }
            if (!CudaSafeCall(InitVertexData3(
                    this->vertexPosMapped_D.Peek(), 3, 0,
                    this->vertexPos1_D.Peek(), 3, 0,
                    this->vertexCnt1))) {
                return false;
            }

            // Transform new positions based on RMS fitting
            if (!this->applyRMSFittingToPosArray(
                    molCall,
                    this->vertexPosMapped_D,
                    this->vertexCnt1,
                    rotation,
                    translation)) {
                return false;
            }

            // Store rms transformed, but unmapped positions, because we need
            // later on
            if (!CudaSafeCall(this->vertexPosRMSTransformed_D.Validate(this->vertexCnt1*3))) {
                return false;
            }
            if (!CudaSafeCall(cudaMemcpy(this->vertexPosRMSTransformed_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    sizeof(float)*this->vertexCnt1*3,
                    cudaMemcpyDeviceToDevice))) {
                return false;
            }

            /* Surface mapping */

            /* Init grid parameters for all files */

            if (!CudaSafeCall(InitVolume(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_surface_mapping(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_metric(
                     make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                     make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                     make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                 return false;
            }

            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Set(0x00))) {
                return false;
            }


            // Init forces scale factor with -1 or 1, depending on whether they start
            // outside or inside the isosurface
            if (!CudaSafeCall (InitExternalForceScl(
                    this->vertexExternalForcesScl_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->vertexPosMapped_D.Peek(),
                    this->vertexExternalForcesScl_D.GetCount(),
                    this->qsIsoVal,
                    0,
                    3))) {
                return false;
            }

            // Compute gradient
            if (!CudaSafeCall(this->volGradient_D.Validate(
                    this->gridDensMap.size[0]*this->gridDensMap.size[1]*this->gridDensMap.size[2]))) {
                return false;
            }
            if (!CudaSafeCall(this->volGradient_D.Set(0))) {
                return false;
            }
#if defined(USE_DISTANCE_FIELD)

            if (!CudaSafeCall(CalcVolGradientWithDistField(
                    this->volGradient_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->distField_D.Peek(),
                    this->maxGaussianVolumeDist,
                    this->qsIsoVal,
                    this->volGradient_D.GetSize()))) {
                return false;
            }

#else
            if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(), volume_D,
                    this->volGradient_D.GetSize()))) {
                return false;
            }
#endif // defined(USE_DISTANCE_FIELD)


            if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(this->displLen_D.Set(0xff))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Validate(this->vertexCnt1))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Set(0))) {
                return false;
            }



//            // DEBUG Print mapped positions
//            HostArr<float> vertexPos;
//            vertexPos.Validate(vertexCnt1*3);
//            cudaMemcpy(vertexPos.Peek(), this->vertexPosMapped_D.Peek(),
//                    sizeof(float)*vertexCnt1*3, cudaMemcpyDeviceToHost);
//            for (int i = 0; i < 10; ++i) {
//
//                printf("%i: Vertex position (%f %f %f)\n", i,
//                        vertexPos.Peek()[3*i+0],
//                        vertexPos.Peek()[3*i+1],
//                        vertexPos.Peek()[3*i+2]);
//
//            }
//            // End DEBUG


            uint cnt = 0;
            while(true) {
//            for (int i = 0; i < 1000; ++i) {
                if (mapInterpolMode == INTERP_LINEAR) {

                    // Update position for all vertices
                    if (!CudaSafeCall(UpdateVertexPositionTrilinearWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPosMapped_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours1_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt1,
                            w,
                            mapForcesScl,
                            mapSpringStiffness,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }

                } else {

                    // Update position for all vertices
                    if (!CudaSafeCall(UpdateVertexPositionTricubicWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPosMapped_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours1_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt1,
                            w,
                            mapForcesScl,
                            mapSpringStiffness,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }
                }
//
//                // DEBUG Print mapped positions
//                HostArr<float> vertexPos;
//                vertexPos.Validate(vertexCnt1*3);
//                cudaMemcpy(vertexPos.Peek(), this->vertexPosMapped_D.Peek(),
//                        sizeof(float)*vertexCnt1*3, cudaMemcpyDeviceToHost);
//                for (int i = 0; i < 10; ++i) {
//
//                    printf("%i: Vertex position (%f %f %f)\n", i,
//                            vertexPos.Peek()[3*i+0],
//                            vertexPos.Peek()[3*i+1],
//                            vertexPos.Peek()[3*i+2]);
//
//                }
//                // End DEBUG

//                // DEBUG Print all displ lengths
//                HostArr<float> displLen;
//                displLen.Validate(this->vertexCnt1);
//                this->displLen_D.CopyToHost(displLen.Peek());
//                for (int v = 0; v < this->vertexCnt1; ++v) {
//                    printf("%i: Displlen %f\n", v, displLen.Peek()[v]);
//                }
//                //return true;
//                // END DEBUG

                float avgDisplLen = 0.0f;
                if (!CudaSafeCall(AccumulateFloat(avgDisplLen, this->displLen_D.Peek(), this->vertexCnt1))) {
                    return false;
                }
                avgDisplLen /= static_cast<float>(this->vertexCnt1);
                //if (cnt%1000==0)
                    //printf("%.16f\n", avgDisplLen);
                //if (avgDisplLen < ::vislib::math::FLOAT_EPSILON) {
                if (avgDisplLen < this->minDispl) {
                    //printf("fr %u cnt %u\n", fr, cnt);
                    break;
                }
                cnt++;

            }


            /* Compute needed number of iterations */

            itCntAvg += cnt;
            itCntMax = std::max(itCntMax, static_cast<int>(cnt));
            itCntMin = std::min(itCntMin, static_cast<int>(cnt));


            /* Compute area with corrupt triangles */

            // Flag corrupt triangles
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Validate(this->triangleCnt1))) {
                return false;
            }
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(FlagCorruptTriangles(
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->triangleCnt1,
                    this->vertexCnt1,
                    this->qsIsoVal))) {
                return false;
            }

            // Compute triangle areas of all (non-corrupt) triangles
            if (!CudaSafeCall(this->trianglesArea_D.Validate(this->triangleCnt1))) {
                return false;
            }
            if (!CudaSafeCall(ComputeTriangleAreaCorrupt(
                    this->trianglesArea_D.Peek(),
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute sum of all (non-corrupt) triangle areas
            float areaCorrupt;
            if (!CudaSafeCall(AccumulateFloat(areaCorrupt,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            if (!CudaSafeCall(ComputeTriangleAreaAll(
                    this->trianglesArea_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute sum of all (non-corrupt) triangle areas
            float areaAll;
            if (!CudaSafeCall(AccumulateFloat(areaAll,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }


            corruptTrianglesAvg += areaCorrupt/areaAll;
            corruptTrianglesMin = ::std::min(areaCorrupt/areaAll, corruptTrianglesMin);
            corruptTrianglesMax = ::std::max(areaCorrupt/areaAll, corruptTrianglesMax);


            /* Compute mean hausdorff difference per vertex */

            // Compute area of non corrupt triangles
            if (!CudaSafeCall(ComputeTriangleArea(
                     this->trianglesArea_D.Peek(),
                     this->corruptTriangleFlag_D.Peek(),
                     this->vertexPosMapped_D.Peek(),
                     this->triangleIdx1_D.Peek(),
                     this->triangleCnt1))) {
                 return false;
             }

            // Compute sum of all (non-corrupt) triangle areas
            float areaNonCorrupt;
            if (!CudaSafeCall(AccumulateFloat(areaNonCorrupt,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            if (!CudaSafeCall(this->vtxHausdorffDist_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(ComputeHausdorffDistance(
                    this->vertexPosMapped_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->vtxHausdorffDist_D.Peek(),
                    this->vertexCnt1,
                    this->vertexCnt0,
                    0,
                    3))) {
                return false;
            }

//            // Integrate hausdorff difference values over (non-corrupt) triangle areas
//            if (!CudaSafeCall(this->trianglesAreaWeightedHausdorffDist_D.Validate(this->triangleCnt1))) {
//                return false;
//            }
//            if (!CudaSafeCall(IntegrateScalarValueOverTriangles(
//                    this->trianglesAreaWeightedHausdorffDist_D.Peek(),
//                    this->corruptTriangleFlag_D.Peek(),
//                    this->trianglesArea_D.Peek(),
//                    this->triangleIdx1_D.Peek(),
//                    this->vtxHausdorffDist_D.Peek(),
//                    this->triangleCnt1))) {
//                return false;
//            }

            // Compute sum of all triangle integrated values
            float hausdorffAll;
            if (!CudaSafeCall(AccumulateFloat(hausdorffAll,
                    this->vtxHausdorffDist_D.Peek(),
                    this->vertexCnt1))) {
                return false;
            }
            hausdorffAll /= static_cast<float>(this->vertexCnt1);


            hausdorffAvg += hausdorffAll/areaAll;
            hausdorffMin= ::std::min(hausdorffAll/areaAll, hausdorffMin);
            hausdorffMax= ::std::max(hausdorffAll/areaAll, hausdorffMax);


            molCall->Unlock(); // Unlock the frame

        }

        itCntAvg /= molCall->FrameCount();
        corruptTrianglesAvg /= molCall->FrameCount();
        hausdorffAvg /= molCall->FrameCount();

        printf("(%.2f, %f) +- (%f, %f)\n", w, corruptTrianglesAvg,
                corruptTrianglesAvg - corruptTrianglesMin,
                corruptTrianglesMax - corruptTrianglesAvg);

//        printf("(%.2f, %.10f) +- (%.10f, %.10f)\n", w, hausdorffAvg,
//                hausdorffAvg - hausdorffMin,
//                hausdorffMax - hausdorffAvg);

//        // Print result for this value of external force weight
//        printf("weight: %f\n", w);
//        printf("(%f, %i) +- (%i, %i)\n",w , itCntAvg, itCntAvg-itCntMin, itCntMax-itCntAvg);
//        printf("    -> corrupt triangles avg %f, min %f, max %f\n",
//                corruptTrianglesAvg, corruptTrianglesAvg-corruptTrianglesMin,
//                corruptTrianglesMax-corruptTrianglesAvg);
//        printf("    -> hausdorff %f, min %f, max %f\n",
//                hausdorffAvg, hausdorffAvg-hausdorffMin,
//                hausdorffMax-hausdorffAvg);

    }

    return true;
}




bool SurfaceMappingTest::plotMappingIterationsByRigidity() {

    printf("Plotting iterations by rigidity...\n");

    unsigned int posCnt0, posCnt1;
    float rotation[3][3];
    float translation[3];
    float rmsVal;

    /* Set parameters */

    const uint frame0 = 0;

    const uint regMaxIt = 500;
    const float regSpringStiffness = 0.3f;
    const float regForcesScl = 0.1f;
    const float regExternalForcesWeight = 0.5f;
    const InterpolationMode regInterpolMode = INTERP_CUBIC;

    //const float mapSpringStiffness = 0.3f;
    const float mapWeight = 0.75f;
    const float mapForcesScl = 0.1f;
    const InterpolationMode mapInterpolMode =  INTERP_CUBIC;
    const float step = 0.05f;


    int itCntAvg = 0.0f;
    int itCntMax = -1;
    int itCntMin= 10000000;

    float corruptTrianglesAvg = 0.0f;
    float corruptTrianglesMin= 10000000;
    float corruptTrianglesMax = -1;

    float hausdorffAvg = 0.0f;
    float hausdorffMin= 10000000;
    float hausdorffMax = -1;


    // Get particles
    MolecularDataCall *molCall = this->particleDataCallerSlot.CallAs<MolecularDataCall>();
    if (molCall == NULL) {
        return false;
    }

    // Get potential texture
    VTIDataCall *volCall = this->volumeDataCallerSlot.CallAs<VTIDataCall>();
    if (volCall == NULL) {
        return false;
    }

    /* Generate target surface */

    molCall->SetFrameID(frame0, true); // Set frame id and force flag
    if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }
    if (!(*molCall)(MolecularDataCall::CallForGetData)) {
        return false;
    }
    volCall->SetFrameID(frame0, true); // Set frame id and force flag
    if (!(*volCall)(VTIDataCall::CallForGetExtent)) {
        return false;
    }
    if (!(*volCall)(VTIDataCall::CallForGetData)) {
        return false;
    }

    this->rmsPosVec0.Validate(molCall->AtomCount()*3);

    // Get rms atom positions
    if (!this->getRMSPosArray(molCall, this->rmsPosVec0, posCnt0)) {
        return false;
    }

    // 1. Compute density map of variant #0
    if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf0,
            this->gridDensMap)) {
        return false;
    }

    // 2. Compute initial triangulation for variant #0
    if (!this->isosurfComputeVertices(
            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
            this->cubeMap0_D,
            this->cubeMapInv0_D,
            this->vertexMap0_D,
            this->vertexMapInv0_D,
            this->vertexNeighbours0_D,
            this->gridDensMap,
            this->vertexCnt0,
            this->vertexPos0_D,
            this->triangleCnt0,
            this->triangleIdx0_D)) {
        return false;
    }

    // 3. Make mesh more regular by using deformable model approach
    if (!this->regularizeSurface(
            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
            this->gridDensMap,
            this->vertexPos0_D,
            this->vertexCnt0,
            this->vertexNeighbours0_D,
            regMaxIt,
            regSpringStiffness,
            regForcesScl,
            regExternalForcesWeight,
            regInterpolMode)) {
        return false;
    }


#if defined(USE_DISTANCE_FIELD)
    // 4. Compute distance field based on regularized vertices of surface 0
    if (!this->computeDistField(
            this->vertexPos0_D,
            this->vertexCnt0,
            this->distField_D,
            this->gridDensMap)) {
        return false;
    }
#endif // defined(USE_DISTANCE_FIELD)

    // Init volume grid constants for metric functions (because they are
    // only visible at file scope
    if (!CudaSafeCall(InitVolume_metric(
            make_uint3(this->gridDensMap.size[0],
                    this->gridDensMap.size[1],
                    this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0],
                            this->gridDensMap.minC[1],
                            this->gridDensMap.minC[2]),
                            make_float3(this->gridDensMap.delta[0],
                                    this->gridDensMap.delta[1],
                                    this->gridDensMap.delta[2])))) {
        return false;
    }

    molCall->Unlock();


    // Loop through all values of the external force
    for (float r = 0.05f; r <= 1.00f; r += step) {


        itCntAvg = 0.0f;
        itCntMax = -1;
        itCntMin= 10000000;

        corruptTrianglesAvg = 0.0f;
        corruptTrianglesMin= 10000000;
        corruptTrianglesMax = -1;

        hausdorffAvg = 0.0f;
        hausdorffMin= 10000000;
        hausdorffMax = -1;


        if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
            return false;
        }

        if (!(*volCall)(VTIDataCall::CallForGetExtent)) {
            return false;
        }

        // Loop through all frames
        for (unsigned int fr = 0; fr < molCall->FrameCount(); ++fr) {

            /* Generate vertices for source shape */

            molCall->SetFrameID(fr, true);
            if (!(*molCall)(MolecularDataCall::CallForGetData)) {
                return false;
            }


//            volCall->SetFrameID(fr, true); // Set frame id and force flag
//            if (!(*volCall)(MolecularDataCall::CallForGetData)) {
//                return false;
//            }

            this->rmsPosVec1.Validate(molCall->AtomCount()*3);
            // Get atom positions
            if (!this->getRMSPosArray(molCall, this->rmsPosVec1, posCnt1)) {
                return false;
            }

            // Compute RMS value and transformation
            if (posCnt0 != posCnt1) {
                return false;
            }
            rmsVal = this->getRMS(this->rmsPosVec0.Peek(),
                    this->rmsPosVec1.Peek(), posCnt0, true, 2, rotation,
                    translation);
            if (rmsVal > 10.0f) {
                return false;
            }

            // 1. Compute density map of variant #1
            if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf1,
                    this->gridDensMap)) {
                return false;
            }


            // 2. Compute initial triangulation for variant #1
            if (!this->isosurfComputeVertices(
                    ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
                    this->cubeMap1_D,
                    this->cubeMapInv1_D,
                    this->vertexMap1_D,
                    this->vertexMapInv1_D,
                    this->vertexNeighbours1_D,
                    this->gridDensMap,
                    this->vertexCnt1,
                    this->vertexPos1_D,
                    this->triangleCnt1,
                    this->triangleIdx1_D)) {
                return false;
            }

//            // DEBUG Print mapped positions
//            HostArr<float> vertexPos;
//            vertexPos.Validate(vertexCnt1*3);
//            cudaMemcpy(vertexPos.Peek(), this->vertexPos1_D.Peek(),
//                    sizeof(float)*vertexCnt1*3, cudaMemcpyDeviceToHost);
//            for (int i = 0; i < 10; ++i) {
//
//                printf("%i: Vertex position (%f %f %f)\n", i,
//                        vertexPos.Peek()[3*i+0],
//                        vertexPos.Peek()[3*i+1],
//                        vertexPos.Peek()[3*i+2]);
//
//            }
//            // End DEBUG


            // 3. Make mesh more regular by using deformable model approach
            if (!this->regularizeSurface(
                    ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
                    this->gridDensMap,
                    this->vertexPos1_D,
                    this->vertexCnt1,
                    this->vertexNeighbours1_D,
                    regMaxIt,
                    regSpringStiffness,
                    regForcesScl,
                    regExternalForcesWeight,
                    regInterpolMode)) {
                return false;
            }


            // Init mapped pos array with old position
            if (!CudaSafeCall(this->vertexPosMapped_D.Validate(this->vertexCnt1*3))) {
                return false;
            }
            if (!CudaSafeCall(InitVertexData3(
                    this->vertexPosMapped_D.Peek(), 3, 0,
                    this->vertexPos1_D.Peek(), 3, 0,
                    this->vertexCnt1))) {
                return false;
            }

            // Transform new positions based on RMS fitting
            if (!this->applyRMSFittingToPosArray(
                    molCall,
                    this->vertexPosMapped_D,
                    this->vertexCnt1,
                    rotation,
                    translation)) {
                return false;
            }

            // Store rms transformed, but unmapped positions, because we need
            // later on
            if (!CudaSafeCall(this->vertexPosRMSTransformed_D.Validate(this->vertexCnt1*3))) {
                return false;
            }
            if (!CudaSafeCall(cudaMemcpy(this->vertexPosRMSTransformed_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    sizeof(float)*this->vertexCnt1*3,
                    cudaMemcpyDeviceToDevice))) {
                return false;
            }

            /* Surface mapping */

            /* Init grid parameters for all files */

            if (!CudaSafeCall(InitVolume(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_surface_mapping(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_metric(
                     make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                     make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                     make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                 return false;
            }

            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Set(0x00))) {
                return false;
            }


            // Init forces scale factor with -1 or 1, depending on whether they start
            // outside or inside the isosurface
            if (!CudaSafeCall (InitExternalForceScl(
                    this->vertexExternalForcesScl_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->vertexPosMapped_D.Peek(),
                    this->vertexExternalForcesScl_D.GetCount(),
                    this->qsIsoVal,
                    0,
                    3))) {
                return false;
            }

            // Compute gradient
            if (!CudaSafeCall(this->volGradient_D.Validate(
                    this->gridDensMap.size[0]*this->gridDensMap.size[1]*this->gridDensMap.size[2]))) {
                return false;
            }
            if (!CudaSafeCall(this->volGradient_D.Set(0))) {
                return false;
            }
#if defined(USE_DISTANCE_FIELD)

            if (!CudaSafeCall(CalcVolGradientWithDistField(
                    this->volGradient_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->distField_D.Peek(),
                    this->maxGaussianVolumeDist,
                    this->qsIsoVal,
                    this->volGradient_D.GetSize()))) {
                return false;
            }

#else
            if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(), volume_D,
                    this->volGradient_D.GetSize()))) {
                return false;
            }
#endif // defined(USE_DISTANCE_FIELD)


            if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(this->displLen_D.Set(0xff))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Validate(this->vertexCnt1))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Set(0))) {
                return false;
            }



//            // DEBUG Print mapped positions
//            HostArr<float> vertexPos;
//            vertexPos.Validate(vertexCnt1*3);
//            cudaMemcpy(vertexPos.Peek(), this->vertexPosMapped_D.Peek(),
//                    sizeof(float)*vertexCnt1*3, cudaMemcpyDeviceToHost);
//            for (int i = 0; i < 10; ++i) {
//
//                printf("%i: Vertex position (%f %f %f)\n", i,
//                        vertexPos.Peek()[3*i+0],
//                        vertexPos.Peek()[3*i+1],
//                        vertexPos.Peek()[3*i+2]);
//
//            }
//            // End DEBUG


            uint cnt = 0;
            while(true) {
//            for (int i = 0; i < 1000; ++i) {
                if (mapInterpolMode == INTERP_LINEAR) {

                    // Update position for all vertices
                    if (!CudaSafeCall(UpdateVertexPositionTrilinearWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPosMapped_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours1_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt1,
                            mapWeight,
                            mapForcesScl,
                            r,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }

                } else {

                    // Update position for all vertices
                    if (!CudaSafeCall(UpdateVertexPositionTricubicWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPosMapped_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours1_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt1,
                            mapWeight,
                            mapForcesScl,
                            r,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }
                }
//
//                // DEBUG Print mapped positions
//                HostArr<float> vertexPos;
//                vertexPos.Validate(vertexCnt1*3);
//                cudaMemcpy(vertexPos.Peek(), this->vertexPosMapped_D.Peek(),
//                        sizeof(float)*vertexCnt1*3, cudaMemcpyDeviceToHost);
//                for (int i = 0; i < 10; ++i) {
//
//                    printf("%i: Vertex position (%f %f %f)\n", i,
//                            vertexPos.Peek()[3*i+0],
//                            vertexPos.Peek()[3*i+1],
//                            vertexPos.Peek()[3*i+2]);
//
//                }
//                // End DEBUG

//                // DEBUG Print all displ lengths
//                HostArr<float> displLen;
//                displLen.Validate(this->vertexCnt1);
//                this->displLen_D.CopyToHost(displLen.Peek());
//                for (int v = 0; v < this->vertexCnt1; ++v) {
//                    printf("%i: Displlen %f\n", v, displLen.Peek()[v]);
//                }
//                //return true;
//                // END DEBUG

                float avgDisplLen = 0.0f;
                if (!CudaSafeCall(AccumulateFloat(avgDisplLen, this->displLen_D.Peek(), this->vertexCnt1))) {
                    return false;
                }
                avgDisplLen /= static_cast<float>(this->vertexCnt1);
                //if (cnt%1000==0)
                    //printf("%.16f\n", avgDisplLen);
                //if (avgDisplLen < ::vislib::math::FLOAT_EPSILON) {
                if (avgDisplLen < this->minDispl) {
                    //printf("fr %u cnt %u\n", fr, cnt);
                    break;
                }
                cnt++;

            }


            /* Compute needed number of iterations */

            itCntAvg += cnt;
            itCntMax = std::max(itCntMax, static_cast<int>(cnt));
            itCntMin = std::min(itCntMin, static_cast<int>(cnt));


            /* Compute area with corrupt triangles */

            // Flag corrupt triangles
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Validate(this->triangleCnt1))) {
                return false;
            }
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(FlagCorruptTriangles(
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->triangleCnt1,
                    this->vertexCnt1,
                    this->qsIsoVal))) {
                return false;
            }

            // Compute triangle areas of all (non-corrupt) triangles
            if (!CudaSafeCall(this->trianglesArea_D.Validate(this->triangleCnt1))) {
                return false;
            }
            if (!CudaSafeCall(ComputeTriangleAreaCorrupt(
                    this->trianglesArea_D.Peek(),
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute sum of all (non-corrupt) triangle areas
            float areaCorrupt;
            if (!CudaSafeCall(AccumulateFloat(areaCorrupt,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            if (!CudaSafeCall(ComputeTriangleAreaAll(
                    this->trianglesArea_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute sum of all (non-corrupt) triangle areas
            float areaAll;
            if (!CudaSafeCall(AccumulateFloat(areaAll,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }


            corruptTrianglesAvg += areaCorrupt/areaAll;
            corruptTrianglesMin = ::std::min(areaCorrupt/areaAll, corruptTrianglesMin);
            corruptTrianglesMax = ::std::max(areaCorrupt/areaAll, corruptTrianglesMax);


            /* Compute mean hausdorff difference per vertex */

            // Compute area of non corrupt triangles
            if (!CudaSafeCall(ComputeTriangleArea(
                     this->trianglesArea_D.Peek(),
                     this->corruptTriangleFlag_D.Peek(),
                     this->vertexPosMapped_D.Peek(),
                     this->triangleIdx1_D.Peek(),
                     this->triangleCnt1))) {
                 return false;
             }

            // Compute sum of all (non-corrupt) triangle areas
            float areaNonCorrupt;
            if (!CudaSafeCall(AccumulateFloat(areaNonCorrupt,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            if (!CudaSafeCall(this->vtxHausdorffDist_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(ComputeHausdorffDistance(
                    this->vertexPosMapped_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->vtxHausdorffDist_D.Peek(),
                    this->vertexCnt1,
                    this->vertexCnt0,
                    0,
                    3))) {
                return false;
            }

//            // Integrate hausdorff difference values over (non-corrupt) triangle areas
//            if (!CudaSafeCall(this->trianglesAreaWeightedHausdorffDist_D.Validate(this->triangleCnt1))) {
//                return false;
//            }
//            if (!CudaSafeCall(IntegrateScalarValueOverTriangles(
//                    this->trianglesAreaWeightedHausdorffDist_D.Peek(),
//                    this->corruptTriangleFlag_D.Peek(),
//                    this->trianglesArea_D.Peek(),
//                    this->triangleIdx1_D.Peek(),
//                    this->vtxHausdorffDist_D.Peek(),
//                    this->triangleCnt1))) {
//                return false;
//            }

            // Compute sum of all triangle integrated values
            float hausdorffAll;
            if (!CudaSafeCall(AccumulateFloat(hausdorffAll,
                    this->vtxHausdorffDist_D.Peek(),
                    this->vertexCnt1))) {
                return false;
            }
            hausdorffAll /= static_cast<float>(this->vertexCnt1);


            hausdorffAvg += hausdorffAll/areaAll;
            hausdorffMin= ::std::min(hausdorffAll/areaAll, hausdorffMin);
            hausdorffMax= ::std::max(hausdorffAll/areaAll, hausdorffMax);


            molCall->Unlock(); // Unlock the frame

        }

        itCntAvg /= molCall->FrameCount();
        corruptTrianglesAvg /= molCall->FrameCount();
        hausdorffAvg /= molCall->FrameCount();

        printf("(%.2f, %f) +- (%f, %f)\n", r, corruptTrianglesAvg,
                corruptTrianglesAvg - corruptTrianglesMin,
                corruptTrianglesMax - corruptTrianglesAvg);

//        printf("(%.2f, %.10f) +- (%.10f, %.10f)\n", r, hausdorffAvg,
//                hausdorffAvg - hausdorffMin,
//                hausdorffMax - hausdorffAvg);

//        // Print result for this value of external force weight
//        printf("weight: %f\n", w);
//        printf("(%f, %i) +- (%i, %i)\n",r , itCntAvg, itCntAvg-itCntMin, itCntMax-itCntAvg);
//        printf("    -> corrupt triangles avg %f, min %f, max %f\n",
//                corruptTrianglesAvg, corruptTrianglesAvg-corruptTrianglesMin,
//                corruptTrianglesMax-corruptTrianglesAvg);
//        printf("    -> hausdorff %f, min %f, max %f\n",
//                hausdorffAvg, hausdorffAvg-hausdorffMin,
//                hausdorffMax-hausdorffAvg);

    }

    return true;
}



bool SurfaceMappingTest::plotMappingIterationsByForcesScl() {

    printf("Plotting iterations by force scl...\n");

    unsigned int posCnt0, posCnt1;
    float rotation[3][3];
    float translation[3];
    float rmsVal;

    /* Set parameters */

    const uint frame0 = 0;

    const uint regMaxIt = 500;
    const float regSpringStiffness = 0.3f;
    const float regForcesScl = 0.1f;
    const float regExternalForcesWeight = 0.5f;
    const InterpolationMode regInterpolMode = INTERP_LINEAR;

    const float mapSpringStiffness = 0.3f;
    const float mapWeight = 0.75f;
    //const float mapForcesScl = 0.1f;
    const InterpolationMode mapInterpolMode =  INTERP_LINEAR;
    const float step = 0.01f;


    int itCntAvg = 0.0f;
    int itCntMax = -1;
    int itCntMin= 10000000;

    float corruptTrianglesAvg = 0.0f;
    float corruptTrianglesMin= 10000000;
    float corruptTrianglesMax = -1;

    float hausdorffAvg = 0.0f;
    float hausdorffMin= 10000000;
    float hausdorffMax = -1;


    // Get particles
    MolecularDataCall *molCall = this->particleDataCallerSlot.CallAs<MolecularDataCall>();
    if (molCall == NULL) {
        return false;
    }

    // Get potential texture
    VTIDataCall *volCall = this->volumeDataCallerSlot.CallAs<VTIDataCall>();
    if (volCall == NULL) {
        return false;
    }

    /* Generate target surface */

    molCall->SetFrameID(frame0, true); // Set frame id and force flag
    if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }
    if (!(*molCall)(MolecularDataCall::CallForGetData)) {
        return false;
    }
    volCall->SetFrameID(frame0, true); // Set frame id and force flag
    if (!(*volCall)(VTIDataCall::CallForGetExtent)) {
        return false;
    }
    if (!(*volCall)(VTIDataCall::CallForGetData)) {
        return false;
    }

    this->rmsPosVec0.Validate(molCall->AtomCount()*3);

    // Get rms atom positions
    if (!this->getRMSPosArray(molCall, this->rmsPosVec0, posCnt0)) {
        return false;
    }

    // 1. Compute density map of variant #0
    if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf0,
            this->gridDensMap)) {
        return false;
    }

    // 2. Compute initial triangulation for variant #0
    if (!this->isosurfComputeVertices(
            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
            this->cubeMap0_D,
            this->cubeMapInv0_D,
            this->vertexMap0_D,
            this->vertexMapInv0_D,
            this->vertexNeighbours0_D,
            this->gridDensMap,
            this->vertexCnt0,
            this->vertexPos0_D,
            this->triangleCnt0,
            this->triangleIdx0_D)) {
        return false;
    }

    // 3. Make mesh more regular by using deformable model approach
    if (!this->regularizeSurface(
            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
            this->gridDensMap,
            this->vertexPos0_D,
            this->vertexCnt0,
            this->vertexNeighbours0_D,
            regMaxIt,
            regSpringStiffness,
            regForcesScl,
            regExternalForcesWeight,
            regInterpolMode)) {
        return false;
    }


#if defined(USE_DISTANCE_FIELD)
    // 4. Compute distance field based on regularized vertices of surface 0
    if (!this->computeDistField(
            this->vertexPos0_D,
            this->vertexCnt0,
            this->distField_D,
            this->gridDensMap)) {
        return false;
    }
#endif // defined(USE_DISTANCE_FIELD)

    // Init volume grid constants for metric functions (because they are
    // only visible at file scope
    if (!CudaSafeCall(InitVolume_metric(
            make_uint3(this->gridDensMap.size[0],
                    this->gridDensMap.size[1],
                    this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0],
                            this->gridDensMap.minC[1],
                            this->gridDensMap.minC[2]),
                            make_float3(this->gridDensMap.delta[0],
                                    this->gridDensMap.delta[1],
                                    this->gridDensMap.delta[2])))) {
        return false;
    }

    molCall->Unlock();


    // Loop through all values of the external force
    for (float s = 0.01f; s <= 0.2f; s += step) {


        itCntAvg = 0.0f;
        itCntMax = -1;
        itCntMin= 10000000;

        corruptTrianglesAvg = 0.0f;
        corruptTrianglesMin= 10000000;
        corruptTrianglesMax = -1;

        hausdorffAvg = 0.0f;
        hausdorffMin= 10000000;
        hausdorffMax = -1;


        if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
            return false;
        }

        if (!(*volCall)(VTIDataCall::CallForGetExtent)) {
            return false;
        }

        // Loop through all frames
        for (unsigned int fr = 0; fr < molCall->FrameCount(); ++fr) {

            /* Generate vertices for source shape */

            molCall->SetFrameID(fr, true);
            if (!(*molCall)(MolecularDataCall::CallForGetData)) {
                return false;
            }


//            volCall->SetFrameID(fr, true); // Set frame id and force flag
//            if (!(*volCall)(MolecularDataCall::CallForGetData)) {
//                return false;
//            }

            this->rmsPosVec1.Validate(molCall->AtomCount()*3);
            // Get atom positions
            if (!this->getRMSPosArray(molCall, this->rmsPosVec1, posCnt1)) {
                return false;
            }

            // Compute RMS value and transformation
            if (posCnt0 != posCnt1) {
                return false;
            }
            rmsVal = this->getRMS(this->rmsPosVec0.Peek(),
                    this->rmsPosVec1.Peek(), posCnt0, true, 2, rotation,
                    translation);
            if (rmsVal > 10.0f) {
                return false;
            }

            // 1. Compute density map of variant #1
            if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf1,
                    this->gridDensMap)) {
                return false;
            }


            // 2. Compute initial triangulation for variant #1
            if (!this->isosurfComputeVertices(
                    ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
                    this->cubeMap1_D,
                    this->cubeMapInv1_D,
                    this->vertexMap1_D,
                    this->vertexMapInv1_D,
                    this->vertexNeighbours1_D,
                    this->gridDensMap,
                    this->vertexCnt1,
                    this->vertexPos1_D,
                    this->triangleCnt1,
                    this->triangleIdx1_D)) {
                return false;
            }

//            // DEBUG Print mapped positions
//            HostArr<float> vertexPos;
//            vertexPos.Validate(vertexCnt1*3);
//            cudaMemcpy(vertexPos.Peek(), this->vertexPos1_D.Peek(),
//                    sizeof(float)*vertexCnt1*3, cudaMemcpyDeviceToHost);
//            for (int i = 0; i < 10; ++i) {
//
//                printf("%i: Vertex position (%f %f %f)\n", i,
//                        vertexPos.Peek()[3*i+0],
//                        vertexPos.Peek()[3*i+1],
//                        vertexPos.Peek()[3*i+2]);
//
//            }
//            // End DEBUG


            // 3. Make mesh more regular by using deformable model approach
            if (!this->regularizeSurface(
                    ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
                    this->gridDensMap,
                    this->vertexPos1_D,
                    this->vertexCnt1,
                    this->vertexNeighbours1_D,
                    regMaxIt,
                    regSpringStiffness,
                    regForcesScl,
                    regExternalForcesWeight,
                    regInterpolMode)) {
                return false;
            }


            // Init mapped pos array with old position
            if (!CudaSafeCall(this->vertexPosMapped_D.Validate(this->vertexCnt1*3))) {
                return false;
            }
            if (!CudaSafeCall(InitVertexData3(
                    this->vertexPosMapped_D.Peek(), 3, 0,
                    this->vertexPos1_D.Peek(), 3, 0,
                    this->vertexCnt1))) {
                return false;
            }

            // Transform new positions based on RMS fitting
            if (!this->applyRMSFittingToPosArray(
                    molCall,
                    this->vertexPosMapped_D,
                    this->vertexCnt1,
                    rotation,
                    translation)) {
                return false;
            }

            // Store rms transformed, but unmapped positions, because we need
            // later on
            if (!CudaSafeCall(this->vertexPosRMSTransformed_D.Validate(this->vertexCnt1*3))) {
                return false;
            }
            if (!CudaSafeCall(cudaMemcpy(this->vertexPosRMSTransformed_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    sizeof(float)*this->vertexCnt1*3,
                    cudaMemcpyDeviceToDevice))) {
                return false;
            }

            /* Surface mapping */

            /* Init grid parameters for all files */

            if (!CudaSafeCall(InitVolume(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_surface_mapping(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_metric(
                     make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                     make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                     make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                 return false;
            }

            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Set(0x00))) {
                return false;
            }


            // Init forces scale factor with -1 or 1, depending on whether they start
            // outside or inside the isosurface
            if (!CudaSafeCall (InitExternalForceScl(
                    this->vertexExternalForcesScl_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->vertexPosMapped_D.Peek(),
                    this->vertexExternalForcesScl_D.GetCount(),
                    this->qsIsoVal,
                    0,
                    3))) {
                return false;
            }

            // Compute gradient
            if (!CudaSafeCall(this->volGradient_D.Validate(
                    this->gridDensMap.size[0]*this->gridDensMap.size[1]*this->gridDensMap.size[2]))) {
                return false;
            }
            if (!CudaSafeCall(this->volGradient_D.Set(0))) {
                return false;
            }
#if defined(USE_DISTANCE_FIELD)

            if (!CudaSafeCall(CalcVolGradientWithDistField(
                    this->volGradient_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->distField_D.Peek(),
                    this->maxGaussianVolumeDist,
                    this->qsIsoVal,
                    this->volGradient_D.GetSize()))) {
                return false;
            }

#else
            if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(), volume_D,
                    this->volGradient_D.GetSize()))) {
                return false;
            }
#endif // defined(USE_DISTANCE_FIELD)


            if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(this->displLen_D.Set(0xff))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Validate(this->vertexCnt1))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Set(0))) {
                return false;
            }



//            // DEBUG Print mapped positions
//            HostArr<float> vertexPos;
//            vertexPos.Validate(vertexCnt1*3);
//            cudaMemcpy(vertexPos.Peek(), this->vertexPosMapped_D.Peek(),
//                    sizeof(float)*vertexCnt1*3, cudaMemcpyDeviceToHost);
//            for (int i = 0; i < 10; ++i) {
//
//                printf("%i: Vertex position (%f %f %f)\n", i,
//                        vertexPos.Peek()[3*i+0],
//                        vertexPos.Peek()[3*i+1],
//                        vertexPos.Peek()[3*i+2]);
//
//            }
//            // End DEBUG


            uint cnt = 0;
            while(true) {
//            for (int i = 0; i < 1000; ++i) {
                if (mapInterpolMode == INTERP_LINEAR) {

                    // Update position for all vertices
                    if (!CudaSafeCall(UpdateVertexPositionTrilinearWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPosMapped_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours1_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt1,
                            mapWeight,
                            s,
                            mapSpringStiffness,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }

                } else {

                    // Update position for all vertices
                    if (!CudaSafeCall(UpdateVertexPositionTricubicWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPosMapped_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours1_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt1,
                            mapWeight,
                            s,
                            mapSpringStiffness,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }
                }
//
//                // DEBUG Print mapped positions
//                HostArr<float> vertexPos;
//                vertexPos.Validate(vertexCnt1*3);
//                cudaMemcpy(vertexPos.Peek(), this->vertexPosMapped_D.Peek(),
//                        sizeof(float)*vertexCnt1*3, cudaMemcpyDeviceToHost);
//                for (int i = 0; i < 10; ++i) {
//
//                    printf("%i: Vertex position (%f %f %f)\n", i,
//                            vertexPos.Peek()[3*i+0],
//                            vertexPos.Peek()[3*i+1],
//                            vertexPos.Peek()[3*i+2]);
//
//                }
//                // End DEBUG

//                // DEBUG Print all displ lengths
//                HostArr<float> displLen;
//                displLen.Validate(this->vertexCnt1);
//                this->displLen_D.CopyToHost(displLen.Peek());
//                for (int v = 0; v < this->vertexCnt1; ++v) {
//                    printf("%i: Displlen %f\n", v, displLen.Peek()[v]);
//                }
//                //return true;
//                // END DEBUG

                float avgDisplLen = 0.0f;
                if (!CudaSafeCall(AccumulateFloat(avgDisplLen, this->displLen_D.Peek(), this->vertexCnt1))) {
                    return false;
                }
                avgDisplLen /= static_cast<float>(this->vertexCnt1);
                //if (cnt%1000==0)
                    //printf("%.16f\n", avgDisplLen);
                //if (avgDisplLen < ::vislib::math::FLOAT_EPSILON) {
                if (avgDisplLen < this->minDispl) {
                    //printf("fr %u cnt %u\n", fr, cnt);
                    break;
                }
                cnt++;

            }


            /* Compute needed number of iterations */

            itCntAvg += cnt;
            itCntMax = std::max(itCntMax, static_cast<int>(cnt));
            itCntMin = std::min(itCntMin, static_cast<int>(cnt));


            /* Compute area with corrupt triangles */

            // Flag corrupt triangles
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Validate(this->triangleCnt1))) {
                return false;
            }
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(FlagCorruptTriangles(
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->triangleCnt1,
                    this->vertexCnt1,
                    this->qsIsoVal))) {
                return false;
            }

            // Compute triangle areas of all (non-corrupt) triangles
            if (!CudaSafeCall(this->trianglesArea_D.Validate(this->triangleCnt1))) {
                return false;
            }
            if (!CudaSafeCall(ComputeTriangleAreaCorrupt(
                    this->trianglesArea_D.Peek(),
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute sum of all (non-corrupt) triangle areas
            float areaCorrupt;
            if (!CudaSafeCall(AccumulateFloat(areaCorrupt,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            if (!CudaSafeCall(ComputeTriangleAreaAll(
                    this->trianglesArea_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute sum of all (non-corrupt) triangle areas
            float areaAll;
            if (!CudaSafeCall(AccumulateFloat(areaAll,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }


            corruptTrianglesAvg += areaCorrupt/areaAll;
            corruptTrianglesMin = ::std::min(areaCorrupt/areaAll, corruptTrianglesMin);
            corruptTrianglesMax = ::std::max(areaCorrupt/areaAll, corruptTrianglesMax);


            /* Compute mean hausdorff difference per vertex */

            // Compute area of non corrupt triangles
            if (!CudaSafeCall(ComputeTriangleArea(
                     this->trianglesArea_D.Peek(),
                     this->corruptTriangleFlag_D.Peek(),
                     this->vertexPosMapped_D.Peek(),
                     this->triangleIdx1_D.Peek(),
                     this->triangleCnt1))) {
                 return false;
             }

            // Compute sum of all (non-corrupt) triangle areas
            float areaNonCorrupt;
            if (!CudaSafeCall(AccumulateFloat(areaNonCorrupt,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            if (!CudaSafeCall(this->vtxHausdorffDist_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(ComputeHausdorffDistance(
                    this->vertexPosMapped_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->vtxHausdorffDist_D.Peek(),
                    this->vertexCnt1,
                    this->vertexCnt0,
                    0,
                    3))) {
                return false;
            }

//            // Integrate hausdorff difference values over (non-corrupt) triangle areas
//            if (!CudaSafeCall(this->trianglesAreaWeightedHausdorffDist_D.Validate(this->triangleCnt1))) {
//                return false;
//            }
//            if (!CudaSafeCall(IntegrateScalarValueOverTriangles(
//                    this->trianglesAreaWeightedHausdorffDist_D.Peek(),
//                    this->corruptTriangleFlag_D.Peek(),
//                    this->trianglesArea_D.Peek(),
//                    this->triangleIdx1_D.Peek(),
//                    this->vtxHausdorffDist_D.Peek(),
//                    this->triangleCnt1))) {
//                return false;
//            }

            // Compute sum of all triangle integrated values
            float hausdorffAll;
            if (!CudaSafeCall(AccumulateFloat(hausdorffAll,
                    this->vtxHausdorffDist_D.Peek(),
                    this->vertexCnt1))) {
                return false;
            }
            hausdorffAll /= static_cast<float>(this->vertexCnt1);


            hausdorffAvg += hausdorffAll/areaAll;
            hausdorffMin= ::std::min(hausdorffAll/areaAll, hausdorffMin);
            hausdorffMax= ::std::max(hausdorffAll/areaAll, hausdorffMax);


            molCall->Unlock(); // Unlock the frame

        }

        itCntAvg /= molCall->FrameCount();
        corruptTrianglesAvg /= molCall->FrameCount();
        hausdorffAvg /= molCall->FrameCount();

        printf("(%.2f, %f) +- (%f, %f)\n", s, corruptTrianglesAvg,
                corruptTrianglesAvg - corruptTrianglesMin,
                corruptTrianglesMax - corruptTrianglesAvg);

//        printf("(%.2f, %.10f) +- (%.10f, %.10f)\n", w, hausdorffAvg,
//                hausdorffAvg - hausdorffMin,
//                hausdorffMax - hausdorffAvg);

//        // Print result for this value of external force weight
//        printf("weight: %f\n", w);
//        printf("(%f, %i) +- (%i, %i)\n",s , itCntAvg, itCntAvg-itCntMin, itCntMax-itCntAvg);
//        printf("    -> corrupt triangles avg %f, min %f, max %f\n",
//                corruptTrianglesAvg, corruptTrianglesAvg-corruptTrianglesMin,
//                corruptTrianglesMax-corruptTrianglesAvg);
//        printf("    -> hausdorff %f, min %f, max %f\n",
//                hausdorffAvg, hausdorffAvg-hausdorffMin,
//                hausdorffMax-hausdorffAvg);

    }

    return true;
}




bool SurfaceMappingTest::plotCorruptTriangleAreaByRigidity() {
    printf("Plotting iterations by external force weighting...\n");

    unsigned int posCnt0, posCnt1;
    float rotation[3][3];
    float translation[3];
    float rmsVal;

    /* Set parameters */

    const uint frame0 = 0;

    const uint regMaxIt = 500;
    const float regSpringStiffness = 0.3f;
    const float regForcesScl = 0.1f;
    const float regExternalForcesWeight = 0.5f;
    const InterpolationMode regInterpolMode = INTERP_CUBIC;


    const float mapForcesScl = 0.1f;
    const float mapExternalForcesWeight = 0.95f;
    const InterpolationMode mapInterpolMode =  INTERP_CUBIC;
    const float step = 0.05f;


//    int itCntAvg = 0.0f;
//    int itCntMax = -1;
//    int itCntMin= 10000000;

    float corruptTrianglesAvg = 0.0f;
    float corruptTrianglesMin= 10000000;
    float corruptTrianglesMax = -1;

    float hausdorffAvg = 0.0f;
    float hausdorffMin= 10000000;
    float hausdorffMax = -1;


    // Get particles
    MolecularDataCall *molCall = this->particleDataCallerSlot.CallAs<MolecularDataCall>();
    if (molCall == NULL) {
        return false;
    }

    // Get potential texture
    VTIDataCall *volCall = this->volumeDataCallerSlot.CallAs<VTIDataCall>();
    if (volCall == NULL) {
        return false;
    }

    /* Generate target surface */

    molCall->SetFrameID(frame0, true); // Set frame id and force flag
    if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }
    if (!(*molCall)(MolecularDataCall::CallForGetData)) {
        return false;
    }
    volCall->SetFrameID(frame0, true); // Set frame id and force flag
    if (!(*volCall)(VTIDataCall::CallForGetExtent)) {
        return false;
    }
    if (!(*volCall)(VTIDataCall::CallForGetData)) {
        return false;
    }

    this->rmsPosVec0.Validate(molCall->AtomCount()*3);

    // Get rms atom positions
    if (!this->getRMSPosArray(molCall, this->rmsPosVec0, posCnt0)) {
        return false;
    }

    // 1. Compute density map of variant #0
    if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf0,
            this->gridDensMap)) {
        return false;
    }

    // 2. Compute initial triangulation for variant #0
    if (!this->isosurfComputeVertices(
            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
            this->cubeMap0_D,
            this->cubeMapInv0_D,
            this->vertexMap0_D,
            this->vertexMapInv0_D,
            this->vertexNeighbours0_D,
            this->gridDensMap,
            this->vertexCnt0,
            this->vertexPos0_D,
            this->triangleCnt0,
            this->triangleIdx0_D)) {
        return false;
    }

    // 3. Make mesh more regular by using deformable model approach
    if (!this->regularizeSurface(
            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
            this->gridDensMap,
            this->vertexPos0_D,
            this->vertexCnt0,
            this->vertexNeighbours0_D,
            regMaxIt,
            regSpringStiffness,
            regForcesScl,
            regExternalForcesWeight,
            regInterpolMode)) {
        return false;
    }


#if defined(USE_DISTANCE_FIELD)
    // 4. Compute distance field based on regularized vertices of surface 0
    if (!this->computeDistField(
            this->vertexPos0_D,
            this->vertexCnt0,
            this->distField_D,
            this->gridDensMap)) {
        return false;
    }
#endif // defined(USE_DISTANCE_FIELD)

    // Init volume grid constants for metric functions (because they are
    // only visible at file scope
    if (!CudaSafeCall(InitVolume_metric(
            make_uint3(this->gridDensMap.size[0],
                    this->gridDensMap.size[1],
                    this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0],
                            this->gridDensMap.minC[1],
                            this->gridDensMap.minC[2]),
                            make_float3(this->gridDensMap.delta[0],
                                    this->gridDensMap.delta[1],
                                    this->gridDensMap.delta[2])))) {
        return false;
    }

    molCall->Unlock();


    // Loop through all values of the external force
    for (float r = 0.05f; r <= 1.00f; r += step) {


//        itCntAvg = 0.0f;
//        itCntMax = -1;
//        itCntMin= 10000000;

        corruptTrianglesAvg = 0.0f;
        corruptTrianglesMin= 10000000;
        corruptTrianglesMax = -1;

        hausdorffAvg = 0.0f;
        hausdorffMin= 10000000;
        hausdorffMax = -1;


        if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
            return false;
        }

        if (!(*volCall)(VTIDataCall::CallForGetExtent)) {
            return false;
        }

        // Loop through all frames
        for (unsigned int fr = 0; fr < molCall->FrameCount(); ++fr) {

            /* Generate vertices for source shape */

            molCall->SetFrameID(fr, true);
            if (!(*molCall)(MolecularDataCall::CallForGetData)) {
                return false;
            }


//            volCall->SetFrameID(fr, true); // Set frame id and force flag
//            if (!(*volCall)(MolecularDataCall::CallForGetData)) {
//                return false;
//            }

            this->rmsPosVec1.Validate(molCall->AtomCount()*3);
            // Get atom positions
            if (!this->getRMSPosArray(molCall, this->rmsPosVec1, posCnt1)) {
                return false;
            }

            // Compute RMS value and transformation
            if (posCnt0 != posCnt1) {
                return false;
            }
            rmsVal = this->getRMS(this->rmsPosVec0.Peek(),
                    this->rmsPosVec1.Peek(), posCnt0, true, 2, rotation,
                    translation);
            if (rmsVal > 10.0f) {
                return false;
            }

            // 1. Compute density map of variant #1
            if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf1,
                    this->gridDensMap)) {
                return false;
            }


            // 2. Compute initial triangulation for variant #1
            if (!this->isosurfComputeVertices(
                    ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
                    this->cubeMap1_D,
                    this->cubeMapInv1_D,
                    this->vertexMap1_D,
                    this->vertexMapInv1_D,
                    this->vertexNeighbours1_D,
                    this->gridDensMap,
                    this->vertexCnt1,
                    this->vertexPos1_D,
                    this->triangleCnt1,
                    this->triangleIdx1_D)) {
                return false;
            }

//            // DEBUG Print mapped positions
//            HostArr<float> vertexPos;
//            vertexPos.Validate(vertexCnt1*3);
//            cudaMemcpy(vertexPos.Peek(), this->vertexPos1_D.Peek(),
//                    sizeof(float)*vertexCnt1*3, cudaMemcpyDeviceToHost);
//            for (int i = 0; i < 10; ++i) {
//
//                printf("%i: Vertex position (%f %f %f)\n", i,
//                        vertexPos.Peek()[3*i+0],
//                        vertexPos.Peek()[3*i+1],
//                        vertexPos.Peek()[3*i+2]);
//
//            }
//            // End DEBUG


            // 3. Make mesh more regular by using deformable model approach
            if (!this->regularizeSurface(
                    ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
                    this->gridDensMap,
                    this->vertexPos1_D,
                    this->vertexCnt1,
                    this->vertexNeighbours1_D,
                    regMaxIt,
                    regSpringStiffness,
                    regForcesScl,
                    regExternalForcesWeight,
                    regInterpolMode)) {
                return false;
            }


            // Init mapped pos array with old position
            if (!CudaSafeCall(this->vertexPosMapped_D.Validate(this->vertexCnt1*3))) {
                return false;
            }
            if (!CudaSafeCall(InitVertexData3(
                    this->vertexPosMapped_D.Peek(), 3, 0,
                    this->vertexPos1_D.Peek(), 3, 0,
                    this->vertexCnt1))) {
                return false;
            }

            // Transform new positions based on RMS fitting
            if (!this->applyRMSFittingToPosArray(
                    molCall,
                    this->vertexPosMapped_D,
                    this->vertexCnt1,
                    rotation,
                    translation)) {
                return false;
            }

            // Store rms transformed, but unmapped positions, because we need
            // later on
            if (!CudaSafeCall(this->vertexPosRMSTransformed_D.Validate(this->vertexCnt1*3))) {
                return false;
            }
            if (!CudaSafeCall(cudaMemcpy(this->vertexPosRMSTransformed_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    sizeof(float)*this->vertexCnt1*3,
                    cudaMemcpyDeviceToDevice))) {
                return false;
            }

            /* Surface mapping */

            /* Init grid parameters for all files */

            if (!CudaSafeCall(InitVolume(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_surface_mapping(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_metric(
                     make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                     make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                     make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                 return false;
            }

            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Set(0x00))) {
                return false;
            }


            // Init forces scale factor with -1 or 1, depending on whether they start
            // outside or inside the isosurface
            if (!CudaSafeCall (InitExternalForceScl(
                    this->vertexExternalForcesScl_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->vertexPosMapped_D.Peek(),
                    this->vertexExternalForcesScl_D.GetCount(),
                    this->qsIsoVal,
                    0,
                    3))) {
                return false;
            }

            // Compute gradient
            if (!CudaSafeCall(this->volGradient_D.Validate(
                    this->gridDensMap.size[0]*this->gridDensMap.size[1]*this->gridDensMap.size[2]))) {
                return false;
            }
            if (!CudaSafeCall(this->volGradient_D.Set(0))) {
                return false;
            }
#if defined(USE_DISTANCE_FIELD)

            if (!CudaSafeCall(CalcVolGradientWithDistField(
                    this->volGradient_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->distField_D.Peek(),
                    this->maxGaussianVolumeDist,
                    this->qsIsoVal,
                    this->volGradient_D.GetSize()))) {
                return false;
            }

#else
            if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(), volume_D,
                    this->volGradient_D.GetSize()))) {
                return false;
            }
#endif // defined(USE_DISTANCE_FIELD)


            if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(this->displLen_D.Set(0xff))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Validate(this->vertexCnt1))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Set(0))) {
                return false;
            }



//            // DEBUG Print mapped positions
//            HostArr<float> vertexPos;
//            vertexPos.Validate(vertexCnt1*3);
//            cudaMemcpy(vertexPos.Peek(), this->vertexPosMapped_D.Peek(),
//                    sizeof(float)*vertexCnt1*3, cudaMemcpyDeviceToHost);
//            for (int i = 0; i < 10; ++i) {
//
//                printf("%i: Vertex position (%f %f %f)\n", i,
//                        vertexPos.Peek()[3*i+0],
//                        vertexPos.Peek()[3*i+1],
//                        vertexPos.Peek()[3*i+2]);
//
//            }
//            // End DEBUG


//            uint cnt = 0;
            //while(true) {
            for (int i = 0; i < 1000; ++i) {
                if (mapInterpolMode == INTERP_LINEAR) {

                    // Update position for all vertices
                    if (!CudaSafeCall(UpdateVertexPositionTrilinearWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPosMapped_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours1_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt1,
                            mapExternalForcesWeight,
                            mapForcesScl,
                            r,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }

                } else {

                    // Update position for all vertices
                    if (!CudaSafeCall(UpdateVertexPositionTricubicWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPosMapped_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours1_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt1,
                            mapExternalForcesWeight,
                            mapForcesScl,
                            r,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }
                }
//
//                // DEBUG Print mapped positions
//                HostArr<float> vertexPos;
//                vertexPos.Validate(vertexCnt1*3);
//                cudaMemcpy(vertexPos.Peek(), this->vertexPosMapped_D.Peek(),
//                        sizeof(float)*vertexCnt1*3, cudaMemcpyDeviceToHost);
//                for (int i = 0; i < 10; ++i) {
//
//                    printf("%i: Vertex position (%f %f %f)\n", i,
//                            vertexPos.Peek()[3*i+0],
//                            vertexPos.Peek()[3*i+1],
//                            vertexPos.Peek()[3*i+2]);
//
//                }
//                // End DEBUG

//                // DEBUG Print all displ lengths
//                HostArr<float> displLen;
//                displLen.Validate(this->vertexCnt1);
//                this->displLen_D.CopyToHost(displLen.Peek());
//                for (int v = 0; v < this->vertexCnt1; ++v) {
//                    printf("%i: Displlen %f\n", v, displLen.Peek()[v]);
//                }
//                //return true;
//                // END DEBUG

//                float avgDisplLen = 0.0f;
//                if (!CudaSafeCall(AccumulateFloat(avgDisplLen, this->displLen_D.Peek(), this->vertexCnt1))) {
//                    return false;
//                }
//                avgDisplLen /= static_cast<float>(this->vertexCnt1);
//                if (cnt%1000==0)
//                    printf("%.16f\n", avgDisplLen);
//                //if (avgDisplLen < ::vislib::math::FLOAT_EPSILON) {
//                if (avgDisplLen < 0.001) {
//                    printf("fr %u cnt %u\n", fr, cnt);
//                    break;
//                }
//                cnt++;

            }


            /* Compute needed number of iterations */

//            itCntAvg += cnt;
            //itCntMax = std::max(itCntMax, static_cast<int>(cnt));
            //itCntMin = std::min(itCntMin, static_cast<int>(cnt));


            /* Compute area with corrupt triangles */

            // Flag corrupt triangles
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Validate(this->triangleCnt1))) {
                return false;
            }
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(FlagCorruptTriangles(
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->triangleCnt1,
                    this->vertexCnt1,
                    this->qsIsoVal))) {
                return false;
            }

            // Compute triangle areas of all (non-corrupt) triangles
            if (!CudaSafeCall(this->trianglesArea_D.Validate(this->triangleCnt1))) {
                return false;
            }
            if (!CudaSafeCall(ComputeTriangleAreaCorrupt(
                    this->trianglesArea_D.Peek(),
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute sum of all (non-corrupt) triangle areas
            float areaCorrupt;
            if (!CudaSafeCall(AccumulateFloat(areaCorrupt,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            if (!CudaSafeCall(ComputeTriangleAreaAll(
                    this->trianglesArea_D.Peek(),
                    this->vertexPosMapped_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute sum of all (non-corrupt) triangle areas
            float areaAll;
            if (!CudaSafeCall(AccumulateFloat(areaAll,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }


            corruptTrianglesAvg += areaCorrupt/areaAll;
            corruptTrianglesMin = ::std::min(areaCorrupt/areaAll, corruptTrianglesMin);
            corruptTrianglesMax = ::std::max(areaCorrupt/areaAll, corruptTrianglesMax);


            /* Compute mean hausdorff difference per vertex */

            // Compute area of non corrupt triangles
            if (!CudaSafeCall(ComputeTriangleArea(
                     this->trianglesArea_D.Peek(),
                     this->corruptTriangleFlag_D.Peek(),
                     this->vertexPosMapped_D.Peek(),
                     this->triangleIdx1_D.Peek(),
                     this->triangleCnt1))) {
                 return false;
             }

            // Compute sum of all (non-corrupt) triangle areas
            float areaNonCorrupt;
            if (!CudaSafeCall(AccumulateFloat(areaNonCorrupt,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            if (!CudaSafeCall(this->vtxHausdorffDist_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(ComputeHausdorffDistance(
                    this->vertexPosMapped_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->vtxHausdorffDist_D.Peek(),
                    this->vertexCnt1,
                    this->vertexCnt0,
                    0,
                    3))) {
                return false;
            }

            // Integrate hausdorff difference values over (non-corrupt) triangle areas
            if (!CudaSafeCall(this->trianglesAreaWeightedHausdorffDist_D.Validate(this->triangleCnt1))) {
                return false;
            }
            if (!CudaSafeCall(IntegrateScalarValueOverTriangles(
                    this->trianglesAreaWeightedHausdorffDist_D.Peek(),
                    this->corruptTriangleFlag_D.Peek(),
                    this->trianglesArea_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    this->vtxHausdorffDist_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute sum of all triangle integrated values
            float hausdorffAll;
            if (!CudaSafeCall(AccumulateFloat(hausdorffAll,
                    this->vtxHausdorffDist_D.Peek(),
                    this->vertexCnt1))) {
                return false;
            }
            hausdorffAll /= static_cast<float>(this->vertexCnt1);


            hausdorffAvg += hausdorffAll/areaAll;
            hausdorffMin= ::std::min(hausdorffAll/areaAll, hausdorffMin);
            hausdorffMax= ::std::max(hausdorffAll/areaAll, hausdorffMax);


            molCall->Unlock(); // Unlock the frame

        }

//        itCntAvg /= molCall->FrameCount();
        corruptTrianglesAvg /= molCall->FrameCount();
        hausdorffAvg /= molCall->FrameCount();

//        printf("(%.2f, %f) +- (%f, %f)\n", r, corruptTrianglesAvg,
//                corruptTrianglesAvg - corruptTrianglesMin,
//                corruptTrianglesMax - corruptTrianglesAvg);

        printf("(%.2f, %.16f) +- (%f, %f)\n", r, hausdorffAvg,
                hausdorffAvg - hausdorffMin,
                hausdorffMax - hausdorffAvg);

//        // Print result for this value of external force weight
//        printf("weight: %f\n", w);
//        printf("    -> iterations avg %i, min %i, max %i\n", itCntAvg, itCntAvg-itCntMin, itCntMax-itCntAvg);
//        printf("    -> corrupt triangles avg %f, min %f, max %f\n",
//                corruptTrianglesAvg, corruptTrianglesAvg-corruptTrianglesMin,
//                corruptTrianglesMax-corruptTrianglesAvg);
//        printf("    -> hausdorff %f, min %f, max %f\n",
//                hausdorffAvg, hausdorffAvg-hausdorffMin,
//                hausdorffMax-hausdorffAvg);

    }

    return true;
}



bool SurfaceMappingTest::plotRegIterationsByExternalForceWeighting() {

    printf("Plotting iterations by external force weighting...\n");

    /* Set parameters */

    const float regSpringStiffness = 0.3f;
    const float regForcesScl = 0.1f;
    const InterpolationMode regInterpolMode = INTERP_CUBIC;
    const float step = 0.05f;


    // Get particles
    MolecularDataCall *molCall = this->particleDataCallerSlot.CallAs<MolecularDataCall>();
    if (molCall == NULL) {
        return false;
    }

    int itCntAvg = 0.0f;
    int itCntMax = -1;
    int itCntMin= 10000000;

    float internalForceAvg = 0.0f;
    float internalForceMin= 10000000;
    float internalForceMax = -1;
    bool fail = false;
    int failCnt = 0;

    // Loop through all values of the external force
    for (float w = 0.05f; w <= 1.01f; w += step) {

        itCntAvg = 0.0f;
        itCntMax = -1;
        itCntMin= 10000000;

        internalForceAvg = 0.0f;
        internalForceMin= 10000000;
        internalForceMax = -1;

        failCnt = 0;

        if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
            return false;
        }

        // Loop through all frames
        for (unsigned int fr = 0; fr < molCall->FrameCount(); ++fr) {

            /* Generate source surface */

            molCall->SetFrameID(fr, true); // Set frame id and force flag
            if (!(*molCall)(MolecularDataCall::CallForGetData)) {
                return false;
            }

            // 1. Compute density map of variant #0
            if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf0,
                    this->gridDensMap)) {
                printf("ERROR: Could not compute density map.");
                return false;
            }

            // 2. Compute initial triangulation for variant #0
            if (!this->isosurfComputeVertices(
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->cubeMap0_D,
                    this->cubeMapInv0_D,
                    this->vertexMap0_D,
                    this->vertexMapInv0_D,
                    this->vertexNeighbours0_D,
                    this->gridDensMap,
                    this->vertexCnt0,
                    this->vertexPos0_D,
                    this->triangleCnt0,
                    this->triangleIdx0_D)) {
                return false;
            }


            /* Init grid parameters */

            if (!CudaSafeCall(InitVolume(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_surface_mapping(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_metric(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Set(0x00))) {
                return false;
            }

            // Init forces scale factor with -1 or 1, depending on whether they start
            // outside or inside the isosurface
            if (!CudaSafeCall (InitExternalForceScl(
                    this->vertexExternalForcesScl_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->vertexPos0_D.Peek(),
                    static_cast<uint>(this->vertexExternalForcesScl_D.GetCount()),
                    this->qsIsoVal,
                    0,
                    3))) {
                return false;
            }

            // Compute gradient
            if (!CudaSafeCall(this->volGradient_D.Validate(
                    this->gridDensMap.size[0]*this->gridDensMap.size[1]*
                    this->gridDensMap.size[2]))) {
                return false;
            }
            if (!CudaSafeCall(this->volGradient_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->volGradient_D.GetSize()))) {
                return false;
            }

            if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->displLen_D.Set(0))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Validate(this->vertexCnt0))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Set(0))) {
                return false;
            }

            uint cnt = 0;
            while(true) {
                if (regInterpolMode == INTERP_LINEAR) {

                    if (!CudaSafeCall(UpdateVertexPositionTrilinearWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPos0_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours0_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt0,
                            w,
                            regForcesScl,
                            regSpringStiffness,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }

                } else {

                    if (!CudaSafeCall(UpdateVertexPositionTricubicWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPos0_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours0_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt0,
                            w,
                            regForcesScl,
                            regSpringStiffness,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }

                }



//                // DEBUG Print all displ lengths
//                HostArr<float> displLen;
//                displLen.Validate(this->vertexCnt0);
//                this->displLen_D.CopyToHost(displLen.Peek());
//                for (int v = 0; v < this->vertexCnt0; ++v) {
//                    printf("%i: Displlen %f\n", v, displLen.Peek()[v]);
//                }
//                //return true;
//                // END DEBUG

                // Accumulate displlen
                float avgDisplLen = 0.0f;
                if (!CudaSafeCall(AccumulateFloat(avgDisplLen, this->displLen_D.Peek(), this->vertexCnt0))) {
                    return false;
                }
                avgDisplLen /= static_cast<float>(this->vertexCnt0);
//                printf("%.16f\n", avgDisplLen);
                //if (avgDisplLen < vislib::math::FLOAT_EPSILON) {
                if (avgDisplLen < this->minDispl) {
                //if (avgDisplLen < 0.00001) {
                    break;
                }
//                if (cnt > 10000) {
//                    fail = true;
//                    failCnt += 1;
//                    printf("FAIL");
//                    break;
//                }
                cnt++;
            }

//            if (!fail) {
            itCntAvg += cnt;
            itCntMax = std::max(int(cnt), itCntMax);
            itCntMin = std::min(int(cnt), itCntMin);
//            }
//            printf("Iterations (%.2f, %u)\n", w, cnt); // Print necessary number of iterations


            /* Find out percentage of surface area that is corrupt */

            // Flag corrupt triangles
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Validate(this->triangleCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(FlagCorruptTriangles(
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->triangleIdx0_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->triangleCnt0,
                    this->vertexCnt0,
                    this->qsIsoVal))) {
                return false;
            }

            // Compute triangle areas of all triangles
            if (!CudaSafeCall(this->trianglesArea_D.Validate(this->triangleCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->trianglesArea_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(ComputeTriangleAreaAll(
                    this->trianglesArea_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->triangleIdx0_D.Peek(),
                    this->triangleCnt0))) {
                return false;
            }

            // Compute sum of all triangle areas
            float areaAll;
            if (!CudaSafeCall(AccumulateFloat(areaAll,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt0))) {
                return false;
            }

            // Compute triangle areas of all triangles
            if (!CudaSafeCall(this->trianglesArea_D.Validate(this->triangleCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->trianglesArea_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(ComputeTriangleAreaCorrupt(
                    this->trianglesArea_D.Peek(),
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->triangleIdx0_D.Peek(),
                    this->triangleCnt0))) {
                return false;
            }

            // Compute sum of all triangle areas
            float areaCorrupt;
            if (!CudaSafeCall(AccumulateFloat(areaCorrupt,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt0))) {
                return false;
            }


            /* Compute average internal force length */

            if (!CudaSafeCall(this->internalForceLen_D.Validate(this->vertexCnt0))) {
                return false;
            }

            if (!CudaSafeCall(CalcInternalForceLen(
                    this->internalForceLen_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->vertexNeighbours0_D.Peek(),
                    this->volGradient_D.Peek(),
                    this->laplacian_D.Peek(),
                    this->vertexCnt0,
                    regSpringStiffness,
                    regForcesScl,
                    0,
                    3))) {
                return false;
            }

            // Compute sum of all triangle areas
            float lenAll;
            if (!CudaSafeCall(AccumulateFloat(lenAll,
                    this->internalForceLen_D.Peek(),
                    this->vertexCnt0))) {
                return false;
            }
            lenAll /= static_cast<float>(this->vertexCnt0);
//            if (!fail) {
            internalForceAvg +=  lenAll;
            internalForceMin = std::min(lenAll, internalForceMin);
            internalForceMax = std::max(lenAll, internalForceMax);
//            }


            fail = false;

//            printf("(%.2f, %f)\n", w, lenAll/static_cast<float>(this->vertexCnt0));
        }


        itCntAvg /= (molCall->FrameCount()-failCnt);
        internalForceAvg /= static_cast<float>(molCall->FrameCount());
        // Print result for this value of external force weight
//        printf("weight: %f\n", w);
//        printf("    -> iterations avg %i, min %i, max %i\n", itCntAvg, itCntAvg-itCntMin, itCntMax-itCntAvg);
        printf("(%.16f, %.16f) +- (%.16f, %.16f)\n", w,
                internalForceAvg, internalForceAvg-internalForceMin,
                internalForceMax-internalForceAvg);
    }
    return true;
}



bool SurfaceMappingTest::plotRegIterationsByRigidity() {

    printf("Plotting iterations by external force weighting...\n");

    /* Set parameters */

    //const float regSpringStiffness = 0.3f;
    const float regForcesScl = 0.1f;
    const float forceWeighting = 0.5f;
    const InterpolationMode regInterpolMode = INTERP_CUBIC;
    const float step = 0.05f;


    // Get particles
    MolecularDataCall *molCall = this->particleDataCallerSlot.CallAs<MolecularDataCall>();
    if (molCall == NULL) {
        return false;
    }

    int itCntAvg = 0.0f;
    int itCntMax = -1;
    int itCntMin= 10000000;

    float internalForceAvg = 0.0f;
    float internalForceMin= 10000000;
    float internalForceMax = -1;
    bool fail = false;
    int failCnt = 0;

    // Loop through all values of the external force
    for (float r = 0.05f; r <= 1.00f; r += step) {

        itCntAvg = 0.0f;
        itCntMax = -1;
        itCntMin= 10000000;

        internalForceAvg = 0.0f;
        internalForceMin= 10000000;
        internalForceMax = -1;

        failCnt = 0;

        if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
            return false;
        }

        // Loop through all frames
        for (unsigned int fr = 0; fr < molCall->FrameCount(); ++fr) {

            /* Generate source surface */

            molCall->SetFrameID(fr, true); // Set frame id and force flag
            if (!(*molCall)(MolecularDataCall::CallForGetData)) {
                return false;
            }

            // 1. Compute density map of variant #0
            if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf0,
                    this->gridDensMap)) {
                printf("ERROR: Could not compute density map.");
                return false;
            }

            // 2. Compute initial triangulation for variant #0
            if (!this->isosurfComputeVertices(
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->cubeMap0_D,
                    this->cubeMapInv0_D,
                    this->vertexMap0_D,
                    this->vertexMapInv0_D,
                    this->vertexNeighbours0_D,
                    this->gridDensMap,
                    this->vertexCnt0,
                    this->vertexPos0_D,
                    this->triangleCnt0,
                    this->triangleIdx0_D)) {
                return false;
            }


            /* Init grid parameters */

            if (!CudaSafeCall(InitVolume(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_surface_mapping(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_metric(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Set(0x00))) {
                return false;
            }

            // Init forces scale factor with -1 or 1, depending on whether they start
            // outside or inside the isosurface
            if (!CudaSafeCall (InitExternalForceScl(
                    this->vertexExternalForcesScl_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->vertexPos0_D.Peek(),
                    static_cast<uint>(this->vertexExternalForcesScl_D.GetCount()),
                    this->qsIsoVal,
                    0,
                    3))) {
                return false;
            }

            // Compute gradient
            if (!CudaSafeCall(this->volGradient_D.Validate(
                    this->gridDensMap.size[0]*this->gridDensMap.size[1]*
                    this->gridDensMap.size[2]))) {
                return false;
            }
            if (!CudaSafeCall(this->volGradient_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->volGradient_D.GetSize()))) {
                return false;
            }

            if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->displLen_D.Set(0))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Validate(this->vertexCnt0))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Set(0))) {
                return false;
            }

            uint cnt = 0;
            while(true) {
                if (regInterpolMode == INTERP_LINEAR) {

                    if (!CudaSafeCall(UpdateVertexPositionTrilinearWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPos0_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours0_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt0,
                            forceWeighting,
                            regForcesScl,
                            r,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }

                } else {

                    if (!CudaSafeCall(UpdateVertexPositionTricubicWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPos0_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours0_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt0,
                            forceWeighting,
                            regForcesScl,
                            r,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }

                }



//                // DEBUG Print all displ lengths
//                HostArr<float> displLen;
//                displLen.Validate(this->vertexCnt0);
//                this->displLen_D.CopyToHost(displLen.Peek());
//                for (int v = 0; v < this->vertexCnt0; ++v) {
//                    printf("%i: Displlen %f\n", v, displLen.Peek()[v]);
//                }
//                //return true;
//                // END DEBUG

                // Accumulate displlen
                float avgDisplLen = 0.0f;
                if (!CudaSafeCall(AccumulateFloat(avgDisplLen, this->displLen_D.Peek(), this->vertexCnt0))) {
                    return false;
                }
                avgDisplLen /= static_cast<float>(this->vertexCnt0);
//                printf("%.16f\n", avgDisplLen);
                //if (avgDisplLen < vislib::math::FLOAT_EPSILON) {
                if (avgDisplLen < this->minDispl) {
                //if (avgDisplLen < 0.00001) {
                    break;
                }
//                if (cnt > 10000) {
//                    fail = true;
//                    failCnt += 1;
//                    printf("FAIL");
//                    break;
//                }
                cnt++;
            }

//            if (!fail) {
            itCntAvg += cnt;
            itCntMax = std::max(int(cnt), itCntMax);
            itCntMin = std::min(int(cnt), itCntMin);
//            }
//            printf("Iterations (%.2f, %u)\n", w, cnt); // Print necessary number of iterations


            /* Find out percentage of surface area that is corrupt */

            // Flag corrupt triangles
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Validate(this->triangleCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(FlagCorruptTriangles(
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->triangleIdx0_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->triangleCnt0,
                    this->vertexCnt0,
                    this->qsIsoVal))) {
                return false;
            }

            // Compute triangle areas of all triangles
            if (!CudaSafeCall(this->trianglesArea_D.Validate(this->triangleCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->trianglesArea_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(ComputeTriangleAreaAll(
                    this->trianglesArea_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->triangleIdx0_D.Peek(),
                    this->triangleCnt0))) {
                return false;
            }

            // Compute sum of all triangle areas
            float areaAll;
            if (!CudaSafeCall(AccumulateFloat(areaAll,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt0))) {
                return false;
            }

            // Compute triangle areas of all triangles
            if (!CudaSafeCall(this->trianglesArea_D.Validate(this->triangleCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->trianglesArea_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(ComputeTriangleAreaCorrupt(
                    this->trianglesArea_D.Peek(),
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->triangleIdx0_D.Peek(),
                    this->triangleCnt0))) {
                return false;
            }

            // Compute sum of all triangle areas
            float areaCorrupt;
            if (!CudaSafeCall(AccumulateFloat(areaCorrupt,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt0))) {
                return false;
            }


            /* Compute average internal force length */

            if (!CudaSafeCall(this->internalForceLen_D.Validate(this->vertexCnt0))) {
                return false;
            }

            if (!CudaSafeCall(CalcInternalForceLen(
                    this->internalForceLen_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->vertexNeighbours0_D.Peek(),
                    this->volGradient_D.Peek(),
                    this->laplacian_D.Peek(),
                    this->vertexCnt0,
                    r,
                    regForcesScl,
                    0,
                    3))) {
                return false;
            }

            // Compute sum of all triangle areas
            float lenAll;
            if (!CudaSafeCall(AccumulateFloat(lenAll,
                    this->internalForceLen_D.Peek(),
                    this->vertexCnt0))) {
                return false;
            }
            lenAll /= static_cast<float>(this->vertexCnt0);
//            if (!fail) {
            internalForceAvg +=  lenAll;
            internalForceMin = std::min(lenAll, internalForceMin);
            internalForceMax = std::max(lenAll, internalForceMax);
//            }


            fail = false;

//            printf("(%.2f, %f)\n", w, lenAll/static_cast<float>(this->vertexCnt0));
        }


        itCntAvg /= (molCall->FrameCount()-failCnt);
        internalForceAvg /= static_cast<float>(molCall->FrameCount());
        // Print result for this value of external force weight
//        printf("weight: %f\n", w);
//        printf("    -> iterations avg %i, min %i, max %i\n", itCntAvg, itCntAvg-itCntMin, itCntMax-itCntAvg);
        printf("(%.16f, %.16f) +- (%.16f, %.16f)\n", r,
                internalForceAvg, internalForceAvg-internalForceMin,
                internalForceMax-internalForceAvg);
    }
    return true;
}


bool SurfaceMappingTest::plotRegIterationsByForcesScl() {

    printf("Plotting iterations by external force weighting...\n");

    /* Set parameters */

    const float regSpringStiffness = 0.3f;
    //const float regForcesScl = 0.1f;
    const float forceWeighting = 0.5f;
    const InterpolationMode regInterpolMode = INTERP_CUBIC;
    const float step = 0.01f;


    // Get particles
    MolecularDataCall *molCall = this->particleDataCallerSlot.CallAs<MolecularDataCall>();
    if (molCall == NULL) {
        return false;
    }

    int itCntAvg = 0.0f;
    int itCntMax = -1;
    int itCntMin= 10000000;

    float internalForceAvg = 0.0f;
    float internalForceMin= 10000000;
    float internalForceMax = -1;
    bool fail = false;
    int failCnt = 0;

    // Loop through all values of the external force
    for (float s = 0.01f; s <= 0.2f; s += step) {

        itCntAvg = 0.0f;
        itCntMax = -1;
        itCntMin= 10000000;

        internalForceAvg = 0.0f;
        internalForceMin= 10000000;
        internalForceMax = -1;

        failCnt = 0;

        if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
            return false;
        }

        // Loop through all frames
        for (unsigned int fr = 0; fr < molCall->FrameCount(); ++fr) {

            /* Generate source surface */

            molCall->SetFrameID(fr, true); // Set frame id and force flag
            if (!(*molCall)(MolecularDataCall::CallForGetData)) {
                return false;
            }

            // 1. Compute density map of variant #0
            if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf0,
                    this->gridDensMap)) {
                printf("ERROR: Could not compute density map.");
                return false;
            }

            // 2. Compute initial triangulation for variant #0
            if (!this->isosurfComputeVertices(
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->cubeMap0_D,
                    this->cubeMapInv0_D,
                    this->vertexMap0_D,
                    this->vertexMapInv0_D,
                    this->vertexNeighbours0_D,
                    this->gridDensMap,
                    this->vertexCnt0,
                    this->vertexPos0_D,
                    this->triangleCnt0,
                    this->triangleIdx0_D)) {
                return false;
            }


            /* Init grid parameters */

            if (!CudaSafeCall(InitVolume(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_surface_mapping(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(InitVolume_metric(
                    make_uint3(this->gridDensMap.size[0], this->gridDensMap.size[1], this->gridDensMap.size[2]),
                    make_float3(this->gridDensMap.minC[0], this->gridDensMap.minC[1], this->gridDensMap.minC[2]),
                    make_float3(this->gridDensMap.delta[0], this->gridDensMap.delta[1], this->gridDensMap.delta[2])))) {
                return false;
            }

            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->vertexExternalForcesScl_D.Set(0x00))) {
                return false;
            }

            // Init forces scale factor with -1 or 1, depending on whether they start
            // outside or inside the isosurface
            if (!CudaSafeCall (InitExternalForceScl(
                    this->vertexExternalForcesScl_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->vertexPos0_D.Peek(),
                    static_cast<uint>(this->vertexExternalForcesScl_D.GetCount()),
                    this->qsIsoVal,
                    0,
                    3))) {
                return false;
            }

            // Compute gradient
            if (!CudaSafeCall(this->volGradient_D.Validate(
                    this->gridDensMap.size[0]*this->gridDensMap.size[1]*
                    this->gridDensMap.size[2]))) {
                return false;
            }
            if (!CudaSafeCall(this->volGradient_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->volGradient_D.GetSize()))) {
                return false;
            }

            if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->displLen_D.Set(0))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Validate(this->vertexCnt0))) {
                return false;
            }

            if (!CudaSafeCall(this->laplacian_D.Set(0))) {
                return false;
            }

            uint cnt = 0;
            while(true) {
                if (regInterpolMode == INTERP_LINEAR) {

                    if (!CudaSafeCall(UpdateVertexPositionTrilinearWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPos0_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours0_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt0,
                            forceWeighting,
                            s,
                            regSpringStiffness,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }

                } else {

                    if (!CudaSafeCall(UpdateVertexPositionTricubicWithDispl(
                            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                            this->vertexPos0_D.Peek(),
                            this->vertexExternalForcesScl_D.Peek(),
                            this->vertexNeighbours0_D.Peek(),
                            this->volGradient_D.Peek(),
                            this->laplacian_D.Peek(),
                            this->displLen_D.Peek(),
                            this->vertexCnt0,
                            forceWeighting,
                            s,
                            regSpringStiffness,
                            this->qsIsoVal,
                            this->minDispl,
                            0,     // Offset for positions in array
                            3))) { // Stride in array
                        return false;
                    }

                }



//                // DEBUG Print all displ lengths
//                HostArr<float> displLen;
//                displLen.Validate(this->vertexCnt0);
//                this->displLen_D.CopyToHost(displLen.Peek());
//                for (int v = 0; v < this->vertexCnt0; ++v) {
//                    printf("%i: Displlen %f\n", v, displLen.Peek()[v]);
//                }
//                //return true;
//                // END DEBUG

                // Accumulate displlen
                float avgDisplLen = 0.0f;
                if (!CudaSafeCall(AccumulateFloat(avgDisplLen, this->displLen_D.Peek(), this->vertexCnt0))) {
                    return false;
                }
                avgDisplLen /= static_cast<float>(this->vertexCnt0);
//                printf("%.16f\n", avgDisplLen);
                //if (avgDisplLen < vislib::math::FLOAT_EPSILON) {
                if (avgDisplLen < this->minDispl) {
                //if (avgDisplLen < 0.00001) {
                    break;
                }
//                if (cnt > 10000) {
//                    fail = true;
//                    failCnt += 1;
//                    printf("FAIL");
//                    break;
//                }
                cnt++;
            }

//            if (!fail) {
            itCntAvg += cnt;
            itCntMax = std::max(int(cnt), itCntMax);
            itCntMin = std::min(int(cnt), itCntMin);
//            }
//            printf("Iterations (%.2f, %u)\n", w, cnt); // Print necessary number of iterations


            /* Find out percentage of surface area that is corrupt */

            // Flag corrupt triangles
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Validate(this->triangleCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->corruptTriangleFlag_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(FlagCorruptTriangles(
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->triangleIdx0_D.Peek(),
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->triangleCnt0,
                    this->vertexCnt0,
                    this->qsIsoVal))) {
                return false;
            }

            // Compute triangle areas of all triangles
            if (!CudaSafeCall(this->trianglesArea_D.Validate(this->triangleCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->trianglesArea_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(ComputeTriangleAreaAll(
                    this->trianglesArea_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->triangleIdx0_D.Peek(),
                    this->triangleCnt0))) {
                return false;
            }

            // Compute sum of all triangle areas
            float areaAll;
            if (!CudaSafeCall(AccumulateFloat(areaAll,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt0))) {
                return false;
            }

            // Compute triangle areas of all triangles
            if (!CudaSafeCall(this->trianglesArea_D.Validate(this->triangleCnt0))) {
                return false;
            }
            if (!CudaSafeCall(this->trianglesArea_D.Set(0))) {
                return false;
            }
            if (!CudaSafeCall(ComputeTriangleAreaCorrupt(
                    this->trianglesArea_D.Peek(),
                    this->corruptTriangleFlag_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->triangleIdx0_D.Peek(),
                    this->triangleCnt0))) {
                return false;
            }

            // Compute sum of all triangle areas
            float areaCorrupt;
            if (!CudaSafeCall(AccumulateFloat(areaCorrupt,
                    this->trianglesArea_D.Peek(),
                    this->triangleCnt0))) {
                return false;
            }


            /* Compute average internal force length */

            if (!CudaSafeCall(this->internalForceLen_D.Validate(this->vertexCnt0))) {
                return false;
            }

            if (!CudaSafeCall(CalcInternalForceLen(
                    this->internalForceLen_D.Peek(),
                    this->vertexPos0_D.Peek(),
                    this->vertexNeighbours0_D.Peek(),
                    this->volGradient_D.Peek(),
                    this->laplacian_D.Peek(),
                    this->vertexCnt0,
                    regSpringStiffness,
                    s,
                    0,
                    3))) {
                return false;
            }

            // Compute sum of all triangle areas
            float lenAll;
            if (!CudaSafeCall(AccumulateFloat(lenAll,
                    this->internalForceLen_D.Peek(),
                    this->vertexCnt0))) {
                return false;
            }
            lenAll /= static_cast<float>(this->vertexCnt0);
//            if (!fail) {
            internalForceAvg +=  lenAll;
            internalForceMin = std::min(lenAll, internalForceMin);
            internalForceMax = std::max(lenAll, internalForceMax);
//            }


            fail = false;

//            printf("(%.2f, %f)\n", w, lenAll/static_cast<float>(this->vertexCnt0));
        }


        itCntAvg /= (molCall->FrameCount()-failCnt);
        internalForceAvg /= static_cast<float>(molCall->FrameCount());
        // Print result for this value of external force weight
//        printf("weight: %f\n", w);
//        printf("    -> iterations avg %i, min %i, max %i\n", itCntAvg, itCntAvg-itCntMin, itCntMax-itCntAvg);
        printf("(%.16f, %.16f) +- (%.16f, %.16f)\n", s,
                internalForceAvg, internalForceAvg-internalForceMin,
                internalForceMax-internalForceAvg);
    }
    return true;
}

#endif // (defined(WITH_CUDA) && (WITH_CUDA))

