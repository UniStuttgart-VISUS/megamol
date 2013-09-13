//
// ProteinVariantMatch.cpp
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 11, 2013
//     Author: scharnkn
//

// Toggle performance measurement and the respective messages
#define USE_TIMER

// Toggle output messages about progress of computations
#define OUTPUT_PROGRESS

// Toggle more detailed output messages
#define VERBOSE

#include "stdafx.h"
#include "ProteinVariantMatch.h"

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

#if (defined(WITH_CUDA) && (WITH_CUDA))
#include "CudaDevArr.h"
#include "ComparativeSurfacePotentialRenderer.cuh"
#include "cuda_error_check.h"
#include "ogl_error_check.h"
#include "CUDAQuickSurf.h"
#include "gridParams.h"
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_ptr.h>
#endif // (defined(WITH_CUDA) && (WITH_CUDA))

using namespace megamol;
using namespace megamol::protein;
using namespace megamol::core;

// TODO Make diagram renderer react to singleFrame-parameter

#if (defined(WITH_CUDA) && (WITH_CUDA))

#if defined(USE_DISTANCE_FIELD)
const float ProteinVariantMatch::maxGaussianVolumeDist = 1.0;
#endif //  defined(USE_DISTANCE_FIELD)

/// Minimum force to keep going
const float ProteinVariantMatch::minDispl = 0.0f;

// Hardcoded parameters for 'quicksurf' class
const float ProteinVariantMatch::qsParticleRad = 1.0f;
const float ProteinVariantMatch::qsGaussLim = 4.0f;
const float ProteinVariantMatch::qsGridSpacing = 1.0f;
const bool ProteinVariantMatch::qsSclVanDerWaals = true;
const float ProteinVariantMatch::qsIsoVal = 0.5f;

#endif // (defined(WITH_CUDA) && (WITH_CUDA))


/*
 * ProteinVariantMatch::ProteinVariantMatch
 */
ProteinVariantMatch::ProteinVariantMatch(void) : Module() ,
        particleDataCallerSlot("getParticleData", "Connects the module with the partidle data source"),
#if (defined(WITH_CUDA) && (WITH_CUDA))
        volumeDataCallerSlot("getVolumeData", "Connects the module with the volume data source"),
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
        diagDataOutCalleeSlot("diagDataOut", "Slot to output results of the computations"),
        matrixDataOutCalleeSlot("matrixDataOut", "Slot to output results of the computations"),
        /* Global mapping options */
        theheuristicSlot("heuristic", "Chose the heuristic the matching is based on"),
        fittingModeSlot("rmsMode", "RMS fitting mode"),
        singleFrameIdxSlot("singleFrame", "Idx of the single frame"),
#if (defined(WITH_CUDA) && (WITH_CUDA))
        /* Parameters for surface mapping */
        surfMapMaxItSlot("surfmap::maxIt", "Number of iterations when mapping the mesh"),
        surfMapRegMaxItSlot("surfmap::regMaxIt", "Number of iterations when regularizing the mesh"),
        surfMapInterpolModeSlot("surfmap::Interpolation", "Interpolation method used for external forces calculation"),
        surfMapSpringStiffnessSlot("surfmap::stiffness", "Stiffness of the internal springs"),
        surfMapExternalForcesWeightSlot("surfmap::externalForcesWeight", "Weight of the external forces"),
        surfMapForcesSclSlot("surfmap::forcesScl", "Scaling of overall force"),
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
        nVariants(0),
#if (defined(WITH_CUDA) && (WITH_CUDA))
        cudaqsurf0(NULL), cudaqsurf1(NULL),
        vertexCnt0(0), vertexCnt1(0), triangleCnt0(0), triangleCnt1(0),
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
        triggerComputeMatch(true),
        triggerComputeMatchRMSD(true)
#if (defined(WITH_CUDA) && (WITH_CUDA))
,       triggerComputeMatchSurfMapping(true)
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
{

    // Data caller for particle data
    this->particleDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->particleDataCallerSlot);

#if (defined(WITH_CUDA) && (WITH_CUDA))
    // Data caller for volume data
    this->volumeDataCallerSlot.SetCompatibleCall<VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->volumeDataCallerSlot);
#endif // (defined(WITH_CUDA) && (WITH_CUDA))

    // Make output slot for diagram data available
    this->diagDataOutCalleeSlot.SetCallback(DiagramCall::ClassName(),
            DiagramCall::FunctionName(DiagramCall::CallForGetData),
            &ProteinVariantMatch::getDiagData);
    this->MakeSlotAvailable(&this->diagDataOutCalleeSlot);

    // Make output slot for matrix data available
    this->matrixDataOutCalleeSlot.SetCallback(VariantMatchDataCall::ClassName(),
            VariantMatchDataCall::FunctionName(VariantMatchDataCall::CallForGetData),
            &ProteinVariantMatch::getMatrixData);
    this->MakeSlotAvailable(&this->matrixDataOutCalleeSlot);


    /* Global mapping parameters */

    // Param slot for heuristic to perfom match
    this->theheuristic = RMS_VALUE;
    core::param::EnumParam* fm = new core::param::EnumParam(this->theheuristic);
#if (defined(WITH_CUDA) && (WITH_CUDA))
    fm->SetTypePair(SURFACE_POTENTIAL, "Mean potential diff");
    fm->SetTypePair(SURFACE_POTENTIAL_SIGN, "Mean potential sign diff");
    fm->SetTypePair(MEAN_HAUSDORFF_DIST, "Mean Hausdorff");
    fm->SetTypePair(HAUSDORFF_DIST, "Hausdorff");
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
    fm->SetTypePair(RMS_VALUE, "RMSD");
    this->theheuristicSlot.SetParameter(fm);
    this->MakeSlotAvailable(&this->theheuristicSlot);

    // Parameter for the RMS fitting mode
    this->fittingMode = RMS_ALL;
    param::EnumParam *rms = new core::param::EnumParam(int(this->fittingMode));
    rms->SetTypePair(RMS_ALL, "All");
    rms->SetTypePair(RMS_BACKBONE, "Backbone");
    rms->SetTypePair(RMS_C_ALPHA, "C alpha");
    this->fittingModeSlot << rms;
    this->MakeSlotAvailable(&this->fittingModeSlot);

    // Parameter for single frame
    this->singleFrameIdx = 0;
    this->singleFrameIdxSlot.SetParameter(new core::param::IntParam(this->singleFrameIdx, 0));
    this->MakeSlotAvailable(&this->singleFrameIdxSlot);

#if (defined(WITH_CUDA) && (WITH_CUDA))
    /* Parameters for surface mapping */

    // Maximum number of iterations when mapping the mesh
    this->surfMapMaxIt = 0;
    this->surfMapMaxItSlot.SetParameter(new core::param::IntParam(this->surfMapMaxIt, 0));
    this->MakeSlotAvailable(&this->surfMapMaxItSlot);

    // Maximum number of iterations when regularizing the mesh
    this->surfMapRegMaxIt = 10;
    this->surfMapRegMaxItSlot.SetParameter(new core::param::IntParam(this->surfMapRegMaxIt, 0));
    this->MakeSlotAvailable(&this->surfMapRegMaxItSlot);

    // Interpolation method used when computing external forces
    this->surfMapInterpolMode = INTERP_LINEAR;
    param::EnumParam *s0i = new core::param::EnumParam(int(this->surfMapInterpolMode));
    s0i->SetTypePair(INTERP_LINEAR, "Linear");
    s0i->SetTypePair(INTERP_CUBIC, "Cubic");
    this->surfMapInterpolModeSlot << s0i;
    this->MakeSlotAvailable(&this->surfMapInterpolModeSlot);

    // Stiffness of the springs defining the spring forces in surface #0
    this->surfMapSpringStiffness = 1.0f;
    this->surfMapSpringStiffnessSlot.SetParameter(new core::param::FloatParam(this->surfMapSpringStiffness, 0.1f));
    this->MakeSlotAvailable(&this->surfMapSpringStiffnessSlot);

    // Weighting of the external forces in surface #0, note that the weight
    // of the internal forces is implicitely defined by
    // 1.0 - surf0ExternalForcesWeight
    this->surfMapExternalForcesWeight = 0.0f;
    this->surfMapExternalForcesWeightSlot.SetParameter(new core::param::FloatParam(this->surfMapExternalForcesWeight, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surfMapExternalForcesWeightSlot);

    // Overall scaling for the forces acting upon surface #0
    this->surfMapForcesScl = 1.0f;
    this->surfMapForcesSclSlot.SetParameter(new core::param::FloatParam(this->surfMapForcesScl, 0.0f));
    this->MakeSlotAvailable(&this->surfMapForcesSclSlot);

#endif //(defined(WITH_CUDA) && (WITH_CUDA))

    // Inititalize min/max values
    this->minMatchRMSDVal = 0.0f;
    this->maxMatchRMSDVal = 0.0f;

#if (defined(WITH_CUDA) && (WITH_CUDA))
    this->minMatchSurfacePotentialVal = 0.0f;
    this->maxMatchSurfacePotentialVal = 0.0f;

    this->minMatchSurfacePotentialSignVal = 0.0f;
    this->maxMatchSurfacePotentialSignVal = 0.0f;

    this->minMatchMeanHausdorffDistanceVal = 0.0f;
    this->maxMatchMeanHausdorffDistanceVal = 0.0f;

    this->minMatchHausdorffDistanceVal = 0.0f;
    this->maxMatchHausdorffDistanceVal = 0.0f;
#endif //(defined(WITH_CUDA) && (WITH_CUDA))
}


/*
 * ProteinVariantMatch::~ProteinVariantMatch
 */
ProteinVariantMatch::~ProteinVariantMatch(void) {
    this->Release();
}


/*
 * ProteinVariantMatch::create
 */
bool ProteinVariantMatch::create(void) {

#if (defined(WITH_CUDA) && (WITH_CUDA))

    // Create quicksurf objects
    if(!this->cudaqsurf0) {
        this->cudaqsurf0 = new CUDAQuickSurf();
    }
    if(!this->cudaqsurf1) {
        this->cudaqsurf1 = new CUDAQuickSurf();
    }

#endif

    return true;
}


/*
 * ProteinVariantMatch::release
 */
void ProteinVariantMatch::release(void) {

#if (defined(WITH_CUDA) && (WITH_CUDA))

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

#endif
}


/*
 * ProteinVariantMatch::getData
 */
bool ProteinVariantMatch::getDiagData(core::Call& call) {

    DiagramCall *diaCall;
    MolecularSurfaceFeature *ms = NULL;

    diaCall = dynamic_cast<DiagramCall*>(&call);
    if (diaCall == NULL) {
        return false;
    }


    // Update parameter slots
    this->updatParams();

    // (Re)-compute match if necessary
    if (this->triggerComputeMatch) {
        if (!this->computeMatch()) {
            return false;
        }
        this->triggerComputeMatch = false;
    }



    // Add series to diagram call if necessary (this only needs to be done once,
    // subsequently data can be added to the series)
    if (this->featureList.Count() < 1) {
#if (defined(WITH_CUDA) && (WITH_CUDA))
        this->featureList.SetCount(5);
#else
        this->featureList.SetCount(1);
#endif // (defined(WITH_CUDA) && (WITH_CUDA))

        // Define feature list for rmsd
        this->featureList[0] = new DiagramCall::DiagramSeries("RMSD",
                new MolecularSurfaceFeature(this->nVariants,
                        Vec3f(0.0f, 0.0f, 0.0f)));
        this->featureList[0]->SetColor(1.0f, 1.0f, 0.0f, 1.0f);

#if (defined(WITH_CUDA) && (WITH_CUDA))
        // Define feature list for absolute hausdorff distance
        this->featureList[1] = new DiagramCall::DiagramSeries("Hausdorff distance",
                new MolecularSurfaceFeature(this->nVariants,
                        Vec3f(0.0f, 0.0f, 0.0f)));
        this->featureList[1]->SetColor(1.0f, 0.0f, 1.0f, 1.0f);

        // Define feature list for absolute hausdorff distance
        this->featureList[2] = new DiagramCall::DiagramSeries("Mean hausdorff distance",
                new MolecularSurfaceFeature(this->nVariants,
                        Vec3f(0.0f, 0.0f, 0.0f)));
        this->featureList[2]->SetColor(0.0f, 1.0f, 0.0f, 1.0f);

        // Define feature list for mean potential difference
        this->featureList[3] = new DiagramCall::DiagramSeries("Mean potential difference",
                new MolecularSurfaceFeature(this->nVariants,
                        Vec3f(0.0f, 0.0f, 0.0f)));
        this->featureList[3]->SetColor(0.0f, 1.0f, 1.0f, 1.0f);

        // Define feature list for mean sign switch
        this->featureList[4] = new DiagramCall::DiagramSeries("Potential sign switch",
                new MolecularSurfaceFeature(this->nVariants,
                        Vec3f(0.0f, 0.0f, 0.0f)));
        this->featureList[4]->SetColor(0.0f, 0.0f, 1.0f, 1.0f);

#endif // (defined(WITH_CUDA) && (WITH_CUDA))
    }
    if (diaCall->GetSeriesCount() < 1) {
        diaCall->AddSeries(this->featureList[0]);
#if (defined(WITH_CUDA) && (WITH_CUDA))
        diaCall->AddSeries(this->featureList[1]);
        diaCall->AddSeries(this->featureList[2]);
        diaCall->AddSeries(this->featureList[3]);
        diaCall->AddSeries(this->featureList[4]);
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
    }

    // To avoid segfaults
    this->singleFrameIdx = std::min(this->singleFrameIdx, this->nVariants-1);

    /* Add data from matching matrices */
    // Note: all values are normalized to 0.0 - 1.0

    // RMSD
    ms = static_cast<MolecularSurfaceFeature*>(this->featureList[0]->GetMappable());
    for (unsigned int i = 0; i < this->nVariants; ++i) {
        ms->AppendValue(i, this->matchRMSD[this->singleFrameIdx*this->nVariants+i]/this->maxMatchRMSDVal);
    }

#if (defined(WITH_CUDA) && (WITH_CUDA))
    // Surface potential
    ms = static_cast<MolecularSurfaceFeature*>(this->featureList[3]->GetMappable());
//    printf("Surface potential: ");
    for (unsigned int i = 0; i < this->nVariants; ++i) {
        ms->AppendValue(i,
                this->matchSurfacePotential[this->singleFrameIdx*this->nVariants+i]/
                this->maxMatchSurfacePotentialVal);
//        printf("%f ", this->matchSurfacePotential[this->singleFrameIdx*this->nVariants+i]/
//                this->maxMatchSurfacePotentialVal);
    }
//    printf("\n");

    // Surface potential sign change
    printf("Mean hausdorff: ");
    ms = static_cast<MolecularSurfaceFeature*>(this->featureList[4]->GetMappable());
    for (unsigned int i = 0; i < this->nVariants; ++i) {
        ms->AppendValue(i,
                this->matchSurfacePotentialSign[this->singleFrameIdx*this->nVariants+i]/
                this->maxMatchSurfacePotentialSignVal);
        printf("%f ", this->matchSurfacePotentialSign[this->singleFrameIdx*this->nVariants+i]/
                this->maxMatchSurfacePotentialSignVal);
    }
    printf("\n");

    // Mean hausdorff distance
//    printf("Mean hausdorff: ");
    ms = static_cast<MolecularSurfaceFeature*>(this->featureList[2]->GetMappable());
    for (unsigned int i = 0; i < this->nVariants; ++i) {
        ms->AppendValue(i,
                this->matchMeanHausdorffDistance[this->singleFrameIdx*this->nVariants+i]/
                this->maxMatchMeanHausdorffDistanceVal);
//        printf("%f ", this->matchMeanHausdorffDistance[this->singleFrameIdx*this->nVariants+i]/
//                this->maxMatchMeanHausdorffDistanceVal);
    }
//    printf("\n");

    // Hausdorff distance
    ms = static_cast<MolecularSurfaceFeature*>(this->featureList[1]->GetMappable());
    for (unsigned int i = 0; i < this->nVariants; ++i) {
        ms->AppendValue(i,
                this->matchHausdorffDistance[this->singleFrameIdx*this->nVariants+i]/
                this->maxMatchHausdorffDistanceVal);
    }
#endif // (defined(WITH_CUDA) && (WITH_CUDA))

    return true;
}


/*
 * ProteinVariantMatch::getMatrixData
 */
bool ProteinVariantMatch::getMatrixData(core::Call& call) {

    VariantMatchDataCall *dc;

    // Update parameter slots
    this->updatParams();

    // (Re)-compute match if necessary
    if (this->triggerComputeMatch) {
        if (!this->computeMatch()) {
            return false;
        }
        this->triggerComputeMatch = false;
    }



    dc = dynamic_cast<VariantMatchDataCall*>(&call);
    if (dc == NULL) {
        return false;
    }

    dc->SetVariantCnt(this->nVariants);
    switch(this->theheuristic) {
    case SURFACE_POTENTIAL :
#if (defined(WITH_CUDA) && (WITH_CUDA))
        dc->SetMatch(this->matchSurfacePotential.PeekElements());
        dc->SetMatchRange(this->minMatchSurfacePotentialVal, this->maxMatchSurfacePotentialVal);
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
        break;
    case SURFACE_POTENTIAL_SIGN :
#if (defined(WITH_CUDA) && (WITH_CUDA))
        dc->SetMatch(this->matchSurfacePotentialSign.PeekElements());
        dc->SetMatchRange(this->minMatchSurfacePotentialSignVal, this->maxMatchSurfacePotentialSignVal);
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
        break;
    case MEAN_HAUSDORFF_DIST :
#if (defined(WITH_CUDA) && (WITH_CUDA))
        dc->SetMatch(this->matchMeanHausdorffDistance.PeekElements());
        dc->SetMatchRange(this->minMatchMeanHausdorffDistanceVal, this->maxMatchMeanHausdorffDistanceVal);
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
        break;
    case HAUSDORFF_DIST :
#if (defined(WITH_CUDA) && (WITH_CUDA))
        dc->SetMatch(this->matchHausdorffDistance.PeekElements());
        dc->SetMatchRange(this->minMatchHausdorffDistanceVal, this->maxMatchHausdorffDistanceVal);
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
        break;
    case RMS_VALUE :
        dc->SetMatch(this->matchRMSD.PeekElements());
        dc->SetMatchRange(this->minMatchRMSDVal, this->maxMatchRMSDVal);
        break;
    }

    return true;
}


#if (defined(WITH_CUDA) && (WITH_CUDA))
/*
 * ProteinVariantMatch::applyRMSFittingToPosArray
 */
bool ProteinVariantMatch::applyRMSFittingToPosArray(
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
#endif // (defined(WITH_CUDA) && (WITH_CUDA))

#if (defined(WITH_CUDA) && (WITH_CUDA))

/*
 * ProteinVariantMatch::computeDensityMap
 */
bool ProteinVariantMatch::computeDensityMap(
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
            mol->AtomCount(),
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
 * ProteinVariantMatch::computeDistField
 */
bool ProteinVariantMatch::computeDistField(
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
#endif // (defined(WITH_CUDA) && (WITH_CUDA))


/*
 * ProteinVariantMatch::computeMatch
 */
bool ProteinVariantMatch::computeMatch() {

#if (defined(WITH_CUDA) && (WITH_CUDA))
    VTIDataCall *vtiCall = this->volumeDataCallerSlot.CallAs<VTIDataCall>();
    if (vtiCall == NULL) {
        return false;
    }
#endif // (defined(WITH_CUDA) && (WITH_CUDA))

    MolecularDataCall *molCall = this->particleDataCallerSlot.CallAs<MolecularDataCall>();
    if (molCall == NULL) {
        return false;
    }

    if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }

    if (!(*molCall)(MolecularDataCall::CallForGetData)) {
        return false;
    }

#if (defined(WITH_CUDA) && (WITH_CUDA))
    if (!(*vtiCall)(VTIDataCall::CallForGetExtent)) {
        return false;
    }
    if (!(*vtiCall)(VTIDataCall::CallForGetData)) {
        return false;
    }
#endif // (defined(WITH_CUDA) && (WITH_CUDA))

#if (defined(WITH_CUDA) && (WITH_CUDA))
    this->nVariants = std::min(molCall->FrameCount(), vtiCall->FrameCount());
#else
    this->nVariants = molCall->FrameCount();
#endif// (defined(WITH_CUDA) && (WITH_CUDA))

//    printf("Matching %u variants ...\n", this->nVariants); // DEBUG

    // Compute match based on chosen heuristic
    switch(this->theheuristic) {
    case SURFACE_POTENTIAL :
#if (defined(WITH_CUDA) && (WITH_CUDA))
        this->matchSurfacePotential.SetCount(this->nVariants*this->nVariants);
        this->matchSurfacePotentialSign.SetCount(this->nVariants*this->nVariants);
        this->matchMeanHausdorffDistance.SetCount(this->nVariants*this->nVariants);
        this->matchHausdorffDistance.SetCount(this->nVariants*this->nVariants);
        if (this->triggerComputeMatchSurfMapping) {
            if (!this->computeMatchSurfMapping()) {
                return false;
            }
            this->triggerComputeMatchSurfMapping = false;
        }
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
        break;
    case SURFACE_POTENTIAL_SIGN :
#if (defined(WITH_CUDA) && (WITH_CUDA))
        this->matchSurfacePotential.SetCount(this->nVariants*this->nVariants);
        this->matchSurfacePotentialSign.SetCount(this->nVariants*this->nVariants);
        this->matchMeanHausdorffDistance.SetCount(this->nVariants*this->nVariants);
        this->matchHausdorffDistance.SetCount(this->nVariants*this->nVariants);
        if (this->triggerComputeMatchSurfMapping) {
            if (!this->computeMatchSurfMapping()) {
                return false;
            }
            this->triggerComputeMatchSurfMapping = false;
        }
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
        break;
    case MEAN_HAUSDORFF_DIST :
#if (defined(WITH_CUDA) && (WITH_CUDA))
        this->matchSurfacePotential.SetCount(this->nVariants*this->nVariants);
        this->matchSurfacePotentialSign.SetCount(this->nVariants*this->nVariants);
        this->matchMeanHausdorffDistance.SetCount(this->nVariants*this->nVariants);
        this->matchHausdorffDistance.SetCount(this->nVariants*this->nVariants);
        if (this->triggerComputeMatchSurfMapping) {
            if (!this->computeMatchSurfMapping()) {
                return false;
            }
            this->triggerComputeMatchSurfMapping = false;
        }
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
        break;
    case HAUSDORFF_DIST :
#if (defined(WITH_CUDA) && (WITH_CUDA))
        this->matchSurfacePotential.SetCount(this->nVariants*this->nVariants);
        this->matchSurfacePotentialSign.SetCount(this->nVariants*this->nVariants);
        this->matchMeanHausdorffDistance.SetCount(this->nVariants*this->nVariants);
        this->matchHausdorffDistance.SetCount(this->nVariants*this->nVariants);
        if (this->triggerComputeMatchSurfMapping) {
            if (!this->computeMatchSurfMapping()) {
                return false;
            }
            this->triggerComputeMatchSurfMapping = false;
        }
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
        break;
    case RMS_VALUE :
        this->matchRMSD.SetCount(this->nVariants*this->nVariants);
        if (this->triggerComputeMatchRMSD) {
            if (!this->computeMatchRMS()) {
                return false;
            }
            this->triggerComputeMatchRMSD = false;
        }
        break;
    }
    return true;
}


/*
 * ProteinVariantMatch::computeMatchRMS
 */
bool ProteinVariantMatch::computeMatchRMS() {
    using namespace vislib::sys;

    unsigned int posCnt0, posCnt1;

#if defined(OUTPUT_PROGRESS)
    //float steps = (this->nVariants*(this->nVariants+1))*0.5f;
    float steps = this->nVariants*this->nVariants;
    float currStep = 0.0f;
#endif // defined(OUTPUT_PROGRESS)

#if defined(USE_TIMER)
    time_t t = clock();
#endif // defined(USE_TIMER)

    MolecularDataCall *molCall = this->particleDataCallerSlot.CallAs<MolecularDataCall>();
    if (molCall == NULL) {
        return false;
    }

    this->minMatchRMSDVal = 1000000.0f;
    this->maxMatchRMSDVal = 0.0f;
    float translation[3];
    float rotation[3][3];

    // Loop through all variants
    for (unsigned int i = 0; i < this->nVariants; ++i) {

        molCall->SetFrameID(i, true); // Set frame id and force flag
        if (!(*molCall)(MolecularDataCall::CallForGetData)) {
            return false;
        }
        this->rmsPosVec0.Validate(molCall->AtomCount()*3);

        // Get atom positions
        if (!this->getRMSPosArray(molCall, this->rmsPosVec0, posCnt0)) {
            return false;
        }

        molCall->Unlock();

        // Loop through all variants
        for (unsigned int j = 0; j < this->nVariants; ++j) {

#if defined(VERBOSE)
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                    "%s: matching variants %i and %i...",
                    this->ClassName(), i, j);
#endif // defined(VERBOSE)

            molCall->SetFrameID(j, true); // Set frame id and force flag
            if (!(*molCall)(MolecularDataCall::CallForGetData)) {
                return false;
            }

            this->rmsPosVec1.Validate(molCall->AtomCount()*3);
            // Get atom positions
            if (!this->getRMSPosArray(molCall, this->rmsPosVec1, posCnt1)) {
                return false;
            }

            molCall->Unlock();

//            printf("rmsPosVec0: %f %f %f %f %f %f\n",
//                    this->rmsPosVec0.Peek()[0],
//                    this->rmsPosVec0.Peek()[1],
//                    this->rmsPosVec0.Peek()[2],
//                    this->rmsPosVec0.Peek()[3],
//                    this->rmsPosVec0.Peek()[4],
//                    this->rmsPosVec0.Peek()[5]);
//
//            printf("rmsPosVec1: %f %f %f %f %f %f\n",
//                    this->rmsPosVec1.Peek()[0],
//                    this->rmsPosVec1.Peek()[1],
//                    this->rmsPosVec1.Peek()[2],
//                    this->rmsPosVec1.Peek()[3],
//                    this->rmsPosVec1.Peek()[4],
//                    this->rmsPosVec1.Peek()[5]);

            if (posCnt0 != posCnt1) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: Unable to perform RMS fitting (non-equal atom \
count (%u vs. %u))", this->ClassName(), posCnt0, posCnt1);
                return false;
            }

            // Compute RMS value and store it in output matrix
            // Note: If the actual rms deviation is not computed, the method
            //       returns the RMS value of the unfitted positions.
            //       Therefore, the 'fit' flag has to be true here, although
            //       we do not actually need the translation/rotation
            this->matchRMSD[this->nVariants*i+j] = this->getRMS(this->rmsPosVec0.Peek(),
                    this->rmsPosVec1.Peek(), posCnt0, true, 2, rotation, translation);

            if (i != j) {
                this->minMatchRMSDVal = std::min(this->minMatchRMSDVal, this->matchRMSD[this->nVariants*i+j]);
                this->maxMatchRMSDVal = std::max(this->maxMatchRMSDVal, this->matchRMSD[this->nVariants*i+j]);
            }

#if defined(VERBOSE)
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                    "%s: RMS value %f",
                    this->ClassName(), this->matchRMSD[this->nVariants*i+j]);
#endif // defined(VERBOSE)

#if defined(OUTPUT_PROGRESS)
            // Ouput progress
            currStep += 1.0f;
            if (static_cast<unsigned int>(currStep)%100 == 0) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                        "%s: matching RMS values %3u%%", this->ClassName(),
                        static_cast<unsigned int>(currStep/steps*100));
            }
#endif // defined(OUTPUT_PROGRESS)
        }
    }

#if defined(USE_TIMER)
    vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_INFO,
    "%s: time for matching RMS values of %u variants %f sec", this->ClassName(),
    this->nVariants,(double(clock()-t)/double(CLOCKS_PER_SEC)));
#endif // defined(USE_TIMER)

    return true;
}

#if (defined(WITH_CUDA) && (WITH_CUDA))
/*
 * ProteinVariantMatch::computeMatchSurfMapping
 */
bool ProteinVariantMatch::computeMatchSurfMapping() {
    using namespace vislib::sys;

    unsigned int posCnt0, posCnt1;
    this->minMatchSurfacePotentialVal = 1000000.0f;
    this->maxMatchSurfacePotentialVal = 0.0f;
    this->minMatchMeanHausdorffDistanceVal = 1000000.0f;
    this->maxMatchMeanHausdorffDistanceVal = 0.0f;
    this->minMatchHausdorffDistanceVal = 1000000.0f;
    this->maxMatchHausdorffDistanceVal = 0.0f;
    this->minMatchSurfacePotentialSignVal = 1000000.0f;
    this->maxMatchSurfacePotentialSignVal = 0.0f;
    float rotation[3][3];
    float translation[3];
    float rmsVal;

    float3 minCOld, minCNew, maxCOld, maxCNew;

#if defined(OUTPUT_PROGRESS)
    //float steps = (this->nVariants*(this->nVariants+1))*0.5f;
    float steps = this->nVariants*this->nVariants;
    float currStep = 0.0f;
#endif // defined(OUTPUT_PROGRESS)

#if defined(USE_TIMER)
    time_t t = clock();
#endif // defined(USE_TIMER)

    // Init matching matrix with zero
    for (uint i = 0; i < this->matchSurfacePotential.Count(); ++i) {
        this->matchSurfacePotential[i] = 0.0f;
    }

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

    // Loop through all variantsPotentialVal
    for (unsigned int i = 0; i < this->nVariants; ++i) {

        molCall->SetFrameID(i, true); // Set frame id and force flag
        if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
            return false;
        }
        if (!(*molCall)(MolecularDataCall::CallForGetData)) {
            return false;
        }
        volCall->SetFrameID(i, true); // Set frame id and force flag
        if (!(*volCall)(MolecularDataCall::CallForGetExtent)) {
            return false;
        }
        if (!(*volCall)(MolecularDataCall::CallForGetData)) {
            return false;
        }

        if (!CudaSafeCall(this->potentialTex0_D.Validate(volCall->GetGridsize().X()*
                volCall->GetGridsize().Y()*volCall->GetGridsize().Z()))) {
            return false;
        }
        if (!CudaSafeCall(cudaMemcpy(this->potentialTex0_D.Peek(),
                (float*)(volCall->GetPointDataByIdx(0,0)),
                sizeof(float)*volCall->GetGridsize().X()*
                volCall->GetGridsize().Y()*volCall->GetGridsize().Z(),
                cudaMemcpyHostToDevice))) {
            return false;
        }

        if (!CudaSafeCall(InitPotentialTexParams(0,
                make_int3(volCall->GetGridsize().X(), volCall->GetGridsize().Y(), volCall->GetGridsize().Z()),
                make_float3(volCall->GetOrigin().X(), volCall->GetOrigin().Y(), volCall->GetOrigin().Z()),
                make_float3(volCall->GetSpacing().X(), volCall->GetSpacing().Y(), volCall->GetSpacing().Z())))) {
            return false;
        }

        // Setup bounding box
        minCNew.x = volCall->GetBoundingBoxes().ObjectSpaceBBox().Left();
        minCNew.y = volCall->GetBoundingBoxes().ObjectSpaceBBox().Bottom();
        minCNew.z = volCall->GetBoundingBoxes().ObjectSpaceBBox().Back();
        maxCNew.x = volCall->GetBoundingBoxes().ObjectSpaceBBox().Right();
        maxCNew.y = volCall->GetBoundingBoxes().ObjectSpaceBBox().Top();
        maxCNew.z = volCall->GetBoundingBoxes().ObjectSpaceBBox().Front();
//
//        printf("Init potential map (new), frame %u\n", volCall->FrameID());
//        printf("Min coord %f %f %f\n",
//                minCNew.x, minCNew.y, minCNew.z);
//        printf("Max coord %f %f %f\n",
//                maxCNew.x, maxCNew.y, maxCNew.z);

        this->rmsPosVec0.Validate(molCall->AtomCount()*3);

        // Get rms atom positions
        if (!this->getRMSPosArray(molCall, this->rmsPosVec0, posCnt0)) {
            return false;
        }

        // 1. Compute density map of variant #0
        if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf0,
                this->gridDensMap0)) {
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
                this->gridDensMap0,
                this->vertexCnt0,
                this->vertexPos0_D,
                this->triangleCnt0,
                this->triangleIdx0_D)) {
            return false;
        }

        // 3. Make mesh more regular by using deformable model approach
        if (!this->regularizeSurface(
                ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                this->gridDensMap0,
                this->vertexPos0_D,
                this->vertexCnt0,
                this->vertexNeighbours0_D,
                this->surfMapRegMaxIt,
                this->surfMapSpringStiffness,
                this->surfMapForcesScl,
                this->surfMapExternalForcesWeight,
                this->surfMapInterpolMode)) {
            return false;
        }

#if defined(USE_DISTANCE_FIELD)
        // 4. Compute distance field based on regularized vertices of surface 0
        if (!this->computeDistField(
                this->vertexPos0_D,
                this->vertexCnt0,
                this->distField_D,
                this->gridDensMap0)) {
            return false;
        }
#endif // defined(USE_DISTANCE_FIELD)

        // Init volume grid constants for metric functions (because they are
        // only visible at file scope
        if (!CudaSafeCall(InitVolume_metric(
                make_uint3(this->gridDensMap0.size[0],
                        this->gridDensMap0.size[1],
                        this->gridDensMap0.size[2]),
                make_float3(this->gridDensMap0.minC[0],
                        this->gridDensMap0.minC[1],
                        this->gridDensMap0.minC[2]),
                make_float3(this->gridDensMap0.delta[0],
                        this->gridDensMap0.delta[1],
                        this->gridDensMap0.delta[2])))) {
            return false;
        }

        molCall->Unlock();

        // Loop through all variants
        for (unsigned int j = 0; j < this->nVariants; ++j) {

#if defined(VERBOSE)
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                    "%s: matching variants %i and %i...",
                    this->ClassName(), i, j);
#endif // defined(VERBOSE)

            molCall->SetFrameID(j, true); // Set frame id and force flag
            if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
                return false;
            }

            if (!(*molCall)(MolecularDataCall::CallForGetData)) {
                return false;
            }


            volCall->SetFrameID(j, true); // Set frame id and force flag
            if (!(*volCall)(MolecularDataCall::CallForGetExtent)) {
                return false;
            }

            if (!(*volCall)(MolecularDataCall::CallForGetData)) {
                return false;
            }


            if (!CudaSafeCall(this->potentialTex1_D.Validate(volCall->GetGridsize().X()*
                    volCall->GetGridsize().Y()*volCall->GetGridsize().Z()))) {
                return false;
            }

            if (!CudaSafeCall(cudaMemcpy(this->potentialTex1_D.Peek(),
                    (float*)(volCall->GetPointDataByIdx(0,0)),
                    sizeof(float)*volCall->GetGridsize().X()*
                    volCall->GetGridsize().Y()*volCall->GetGridsize().Z(),
                    cudaMemcpyHostToDevice))) {
                return false;
            }

            if (!CudaSafeCall(InitPotentialTexParams(1,
                    make_int3(volCall->GetGridsize().X(), volCall->GetGridsize().Y(), volCall->GetGridsize().Z()),
                    make_float3(volCall->GetOrigin().X(), volCall->GetOrigin().Y(), volCall->GetOrigin().Z()),
                    make_float3(volCall->GetSpacing().X(), volCall->GetSpacing().Y(), volCall->GetSpacing().Z())))) {
                return false;
            }

            // Setup bounding box
            minCOld.x = volCall->GetBoundingBoxes().ObjectSpaceBBox().Left();
            minCOld.y = volCall->GetBoundingBoxes().ObjectSpaceBBox().Bottom();
            minCOld.z = volCall->GetBoundingBoxes().ObjectSpaceBBox().Back();
            maxCOld.x = volCall->GetBoundingBoxes().ObjectSpaceBBox().Right();
            maxCOld.y = volCall->GetBoundingBoxes().ObjectSpaceBBox().Top();
            maxCOld.z = volCall->GetBoundingBoxes().ObjectSpaceBBox().Front();

//            printf("Init potential map (old), frame %u\n", volCall->FrameID());
//            printf("grid dim %u %u %u\n", volCall->GetGridsize().X(),
//                    volCall->GetGridsize().Y(),
//                    volCall->GetGridsize().Z());
//            printf("Min coord %f %f %f\n",
//                    minCOld.x, minCOld.y, minCOld.z);
//            printf("Max coord %f %f %f\n",
//                    maxCOld.x, maxCOld.y, maxCOld.z);

            this->rmsPosVec1.Validate(molCall->AtomCount()*3);
            // Get atom positions
            if (!this->getRMSPosArray(molCall, this->rmsPosVec1, posCnt1)) {
                return false;
            }

            // Compute RMS value and transformation
            if (posCnt0 != posCnt1) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: Unable to perform RMS fitting (non-equal atom \
count (%u vs. %u))", this->ClassName(), posCnt0, posCnt1);
                return false;
            }
            rmsVal = this->getRMS(this->rmsPosVec0.Peek(),
                    this->rmsPosVec1.Peek(), posCnt0, true, 2, rotation,
                    translation);
            if (rmsVal > 10.0f) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: Unable to perform surface matching (rms val = %f)",
                        this->ClassName(), rmsVal);
                return false;
            }

//            // DEBUG
//            printf("RMS value %f\n", rmsVal);
//            printf("RMS translation %f %f %f\n",
//                    translation[0],
//                    translation[1],
//                    translation[2]);
//            printf("RMS rotation \n %f %f %f\n%f %f %f\n%f %f %f\n",
//                    rotation[0][0], rotation[0][1], rotation[0][2],
//                    rotation[1][0], rotation[1][1], rotation[1][2],
//                    rotation[2][0], rotation[2][1], rotation[2][2]);

            // 1. Compute density map of variant #1
            if (!this->computeDensityMap(molCall, (CUDAQuickSurf *)this->cudaqsurf1,
                    this->gridDensMap1)) {
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
                    this->gridDensMap1,
                    this->vertexCnt1,
                    this->vertexPos1_D,
                    this->triangleCnt1,
                    this->triangleIdx1_D)) {
                return false;
            }

            // 3. Make mesh more regular by using deformable model approach
            if (!this->regularizeSurface(
                    ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
                    this->gridDensMap1,
                    this->vertexPos1_D,
                    this->vertexCnt1,
                    this->vertexNeighbours1_D,
                    this->surfMapRegMaxIt,
                    this->surfMapSpringStiffness,
                    this->surfMapForcesScl,
                    this->surfMapExternalForcesWeight,
                    this->surfMapInterpolMode)) {
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

//            // DEBUG print start positions
//            HostArr<float> vertexPosMapped;
//            vertexPosMapped.Validate(this->vertexPosMapped_D.GetSize());
//            this->vertexPosMapped_D.CopyToHost(vertexPosMapped.Peek());
//            for (int k = 0; k < 10; ++k) {
//                printf("%i Pos start %f\n", k, vertexPosMapped.Peek()[k]);
//            }
//            // END DEBUG

            // Map surface of variant #1 to volume #0
            if (!this->mapIsosurfaceToVolume(
                    ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                    this->gridDensMap0,
                    this->vertexPosMapped_D,
                    this->triangleIdx1_D,
                    this->vertexCnt1,
                    this->triangleCnt1,
                    this->vertexNeighbours1_D,
                    this->surfMapMaxIt,
                    this->surfMapSpringStiffness,
                    this->surfMapForcesScl,
                    this->surfMapExternalForcesWeight,
                    this->surfMapInterpolMode)) {
                return false;
            }

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

//            // DEBUG print mapped vertex positions
//            //HostArr<float> vertexPosMapped;
//            vertexPosMapped.Validate(this->vertexPosMapped_D.GetSize());
//            this->vertexPosMapped_D.CopyToHost(vertexPosMapped.Peek());
//            for (int k = 0; k < 10; ++k) {
//                printf("%i Pos mapped %f\n", k, vertexPosMapped.Peek()[k]);
//            }
//            // END DEBUG

            // Compute potential difference per vertex
            if (!CudaSafeCall(this->vertexPotentialDiff_D.Validate(this->vertexCnt1))) {
                return false;
            }

//            float *vertexPotentialDiff_D,
//            float *vertexPosNew_D,
//            float *vertexPosOld_D,
//            float volMinXOld, float volMinYOld, float volMinZOld,
//            float volMaxXOld, float volMaxYOld, float volMaxZOld,
//            float volMinXNew, float volMinYNew, float volMinZNew,
//            float volMaxXNew, float volMaxYNew, float volMaxZNew,
//            uint vertexCnt

            // DEBUG Save texture coordinates
//            CudaDevArr<float3> texCoords;
//            HostArr<float3> texCoords_H;
//            texCoords.Validate(this->vertexCnt1*2);
//            texCoords_H.Validate(this->vertexCnt1*2);

            if (!CudaSafeCall(ComputeVertexPotentialDiff(
                    this->vertexPotentialDiff_D.Peek(),
                    this->vertexPosMapped_D.Peek(),  // New position of vertices
                    this->vertexPos1_D.Peek(),       // Old position without RMS transformation
                    this->potentialTex0_D.Peek(),
                    this->potentialTex1_D.Peek(),
                    minCOld.x, minCOld.y, minCOld.z,
                    maxCOld.x, maxCOld.y, maxCOld.z,
                    minCNew.x, minCNew.y, minCNew.z,
                    maxCNew.x, maxCNew.y, maxCNew.z,
                    this->vertexCnt1))) {
                return false;
            }

//            // DEBUG Print resulting vertex positions, tex coords, etc.
//            HostArr<float> vertexPosMapped;
//            vertexPosMapped.Validate(this->vertexPosMapped_D.GetSize());
//            this->vertexPosMapped_D.CopyToHost(vertexPosMapped.Peek());
//            texCoords.CopyToHost(texCoords_H.Peek());
//            for (int i = 0; i < 10; ++i) {
//                printf("%i: pos new (%f %f %f), tex coords new (%f %f %f), tex coords old (%f %f %f)\n", i,
//                        vertexPosMapped.Peek()[i*3 + 0],
//                        vertexPosMapped.Peek()[i*3 + 1],
//                        vertexPosMapped.Peek()[i*3 + 2],
//                        texCoords_H.Peek()[i*2 + 0].x,
//                        texCoords_H.Peek()[i*2 + 0].y,
//                        texCoords_H.Peek()[i*2 + 0].z,
//                        texCoords_H.Peek()[i*2 + 1].x,
//                        texCoords_H.Peek()[i*2 + 1].y,
//                        texCoords_H.Peek()[i*2 + 1].z);
//            }
//            // END DEBUG

//            // DEBUG Print resulting potential values and difference
//            HostArr<float> vertexPosMapped;
//            vertexPosMapped.Validate(this->vertexPosMapped_D.GetSize());
//            this->vertexPosMapped_D.CopyToHost(vertexPosMapped.Peek());
//            texCoords.CopyToHost(texCoords_H.Peek());
//            HostArr<float> vertexPosDiff;
//            vertexPosDiff.Validate(this->vertexPotentialDiff_D.GetSize());
//            this->vertexPotentialDiff_D.CopyToHost(vertexPosDiff.Peek());
//            for (int j = 0; j < 10; ++j) {
//                printf("%i: pos new (%f %f %f), pot new %f, pot old %f, diff %f\n", j,
//                        vertexPosMapped.Peek()[j*3 + 0],
//                        vertexPosMapped.Peek()[j*3 + 1],
//                        vertexPosMapped.Peek()[j*3 + 2],
//                        texCoords_H.Peek()[2*j].x,
//                        texCoords_H.Peek()[2*j+1].x,
//                        vertexPosDiff.Peek()[j]);
//            }
//            // END DEBUG

            // Compute triangle areas of all (non-corrupt) triangles
            if (!CudaSafeCall(this->trianglesArea_D.Validate(this->triangleCnt1))) {
                return false;
            }
            if (!CudaSafeCall(ComputeTriangleArea(
                    this->trianglesArea_D.Peek(),
                    this->corruptTriangleFlag_D.Peek(),
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

//            // Weight non-corrupt triangles with 0.0, corrupt with 1.0
//            if (!CudaSafeCall(MultTriangleAreaWithWeight(
//                    this->corruptTriangleFlag_D.Peek(),
//                    this->trianglesArea_D.Peek(),
//                    this->triangleCnt1))) {
//                return false;
//            }

//            // Compute sum of all triangle areas
//            float areaCorrupt;
//            if (!CudaSafeCall(AccumulateFloat(areaCorrupt,
//                    this->trianglesArea_D.Peek(),
//                    this->triangleCnt1))) {
//                return false;
//            }

            /* Compute potential difference metric */

            // Integrate potential difference values over (non-corrupt) triangle areas
            if (!CudaSafeCall(this->trianglesAreaWeightedPotentialDiff_D.Validate(this->triangleCnt1))) {
                return false;
            }
            if (!CudaSafeCall(IntegrateScalarValueOverTriangles(
                    this->trianglesAreaWeightedPotentialDiff_D.Peek(),
                    this->corruptTriangleFlag_D.Peek(),
                    this->trianglesArea_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    this->vertexPotentialDiff_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute sum of all triangle integrated values
            float potDiffAll;
            if (!CudaSafeCall(AccumulateFloat(potDiffAll,
                    this->trianglesAreaWeightedPotentialDiff_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute absolute mean value
            this->matchSurfacePotential[i*this->nVariants+j] = potDiffAll/areaAll;

            if (i != j) {
                this->minMatchSurfacePotentialVal = std::min(this->minMatchSurfacePotentialVal, this->matchSurfacePotential[this->nVariants*i+j]);
                this->maxMatchSurfacePotentialVal = std::max(this->maxMatchSurfacePotentialVal, this->matchSurfacePotential[this->nVariants*i+j]);
            }


            /* Compute potential sign difference metric */

            // Compute potential difference per vertex
            if (!CudaSafeCall(this->vertexPotentialSignDiff_D.Validate(this->vertexCnt1))) {
                return false;
            }

            if (!CudaSafeCall(ComputeVertexPotentialSignDiff(
                    this->vertexPotentialSignDiff_D.Peek(),
                    this->vertexPosMapped_D.Peek(),  // New position of vertices
                    this->vertexPos1_D.Peek(),       // Old position without RMS transformation
                    this->potentialTex0_D.Peek(),
                    this->potentialTex1_D.Peek(),
                    minCOld.x, minCOld.y, minCOld.z,
                    maxCOld.x, maxCOld.y, maxCOld.z,
                    minCNew.x, minCNew.y, minCNew.z,
                    maxCNew.x, maxCNew.y, maxCNew.z,
                    this->vertexCnt1))) {
                return false;
            }

            // Integrate potential difference values over (non-corrupt) triangle areas
            if (!CudaSafeCall(this->trianglesAreaWeightedPotentialSign_D.Validate(this->triangleCnt1))) {
                return false;
            }
            if (!CudaSafeCall(IntegrateScalarValueOverTriangles(
                    this->trianglesAreaWeightedPotentialSign_D.Peek(),
                    this->corruptTriangleFlag_D.Peek(),
                    this->trianglesArea_D.Peek(),
                    this->triangleIdx1_D.Peek(),
                    this->vertexPotentialSignDiff_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute sum of all triangle integrated values
            float potSignDiffAll;
            if (!CudaSafeCall(AccumulateFloat(potSignDiffAll,
                    this->trianglesAreaWeightedPotentialSign_D.Peek(),
                    this->triangleCnt1))) {
                return false;
            }

            // Compute absolute mean value
            this->matchSurfacePotentialSign[i*this->nVariants+j] = potSignDiffAll/areaAll;

            if (i != j) {
                this->minMatchSurfacePotentialSignVal =
                        std::min(this->minMatchSurfacePotentialSignVal, this->matchSurfacePotentialSign[this->nVariants*i+j]);
                this->maxMatchSurfacePotentialSignVal =
                        std::max(this->maxMatchSurfacePotentialSignVal, this->matchSurfacePotentialSign[this->nVariants*i+j]);
            }


            /* Compute mean hausdorff difference per vertex */

            if (!CudaSafeCall(this->vtxHausdorffDist_D.Validate(this->vertexCnt1))) {
                return false;
            }
            if (!CudaSafeCall(ComputeHausdorffDistance(
                    this->vertexPosMapped_D.Peek(),
                    this->vertexPosRMSTransformed_D.Peek(),
                    this->vtxHausdorffDist_D.Peek(),
                    this->vertexCnt1,
                    this->vertexCnt1,
                    0,
                    3))) {
                return false;
            }

            float hausdorff;
            if (!CudaSafeCall(AccumulateFloat(hausdorff,
                    this->vtxHausdorffDist_D.Peek(),
                    this->vertexCnt1))) {
                return false;
            }
            hausdorff /= static_cast<float>(this->vertexCnt1);

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

//            // Compute sum of all triangle integrated values
//            float hausdorffAll;
//            if (!CudaSafeCall(AccumulateFloat(hausdorffAll,
//                    this->trianglesAreaWeightedHausdorffDist_D.Peek(),
//                    this->triangleCnt1))) {
//                return false;
//            }
//
//            // Compute absolute mean value
//            this->matchMeanHausdorffDistance[i*this->nVariants+j] = hausdorffAll/areaAll;

            if (i != j) {
                this->minMatchMeanHausdorffDistanceVal =
                        std::min(this->minMatchMeanHausdorffDistanceVal, this->matchMeanHausdorffDistance[this->nVariants*i+j]);
                this->maxMatchMeanHausdorffDistanceVal =
                        std::max(this->maxMatchMeanHausdorffDistanceVal, this->matchMeanHausdorffDistance[this->nVariants*i+j]);
            }

            /* Compute hausdorff difference */

            float hausdorffDist;
            if (!CudaSafeCall(::ReduceToMax(hausdorffDist,
                    this->vtxHausdorffDist_D.Peek(), this->vertexCnt1, -100.0))) {
                return false;
            }

            // Compute absolute mean value
            this->matchHausdorffDistance[i*this->nVariants+j] = hausdorffDist;

            if (i != j) {
                this->minMatchHausdorffDistanceVal =
                        std::min(this->minMatchHausdorffDistanceVal, this->matchHausdorffDistance[this->nVariants*i+j]);
                this->maxMatchHausdorffDistanceVal =
                        std::max(this->maxMatchHausdorffDistanceVal, this->matchHausdorffDistance[this->nVariants*i+j]);
            }

            molCall->Unlock(); // Unlock the frame

#if defined(VERBOSE)
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                    "%s: triangle area sum (all)     %f",
                    this->ClassName(), areaAll);
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                    "%s: mean hausdorff distance     %f",
                    this->ClassName(), hausdorff);
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                    "%s: mean potential difference   %f",
                    this->ClassName(), this->matchSurfacePotential[i*this->nVariants+j]);
#endif // defined(VERBOSE)

#if defined(OUTPUT_PROGRESS)
            // Ouput progress
            currStep += 1.0f;
            if (static_cast<unsigned int>(currStep)%10 == 0) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                        "%s: matching surfaces %3u%%", this->ClassName(),
                        static_cast<unsigned int>(currStep/steps*100));
            }
#endif // defined(OUTPUT_PROGRESS)
        }
    }

#if defined(USE_TIMER)
    vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_INFO,
    "%s: time for matching surfaces of %u variants %f sec", this->ClassName(),
    this->nVariants,(double(clock()-t)/double(CLOCKS_PER_SEC)));
#endif // defined(USE_TIMER)

    return true;
}

#endif // (defined(WITH_CUDA) && (WITH_CUDA))


/*
 * ProteinVariantMatch::getRMS
 */
float ProteinVariantMatch::getRMS(float *atomPos0, float *atomPos1,
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
 * ProteinVariantMatch::getRMSPosArray
 */
bool ProteinVariantMatch::getRMSPosArray(MolecularDataCall *mol,
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


#if (defined(WITH_CUDA) && (WITH_CUDA))
/*
 * ProteinVariantMatch::isosurfComputeVertices
 */
bool ProteinVariantMatch::isosurfComputeVertices(
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

#endif // (defined(WITH_CUDA) && (WITH_CUDA))


#if (defined(WITH_CUDA) && (WITH_CUDA))
/*
 * ProteinVariantMatch::mapIsosurfaceToVolume
 */
bool ProteinVariantMatch::mapIsosurfaceToVolume(
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

    // Laplacian
    if (!CudaSafeCall(this->laplacian_D.Validate(vertexCnt))) {
        return false;
    }

    if (!CudaSafeCall(this->laplacian_D.Set(0))) {
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

            // Update position for all vertices
            if (!CudaSafeCall(UpdateVertexPositionTricubic(
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

#endif // (defined(WITH_CUDA) && (WITH_CUDA))


#if (defined(WITH_CUDA) && (WITH_CUDA))
/*
 * ProteinVariantMatch::regularizeSurface
 */
bool ProteinVariantMatch::regularizeSurface(
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

    // Laplacian
    if (!CudaSafeCall(this->laplacian_D.Validate(vertexCnt))) {
        return false;
    }

    if (!CudaSafeCall(this->laplacian_D.Set(0))) {
        return false;
    }

    if (interpMode == INTERP_LINEAR) {
        for (uint i = 0; i < maxIt; i += UPDATE_VTX_POS_ITERATIONS_PER_KERNEL) {

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

            if (!CudaSafeCall(UpdateVertexPositionTricubic(
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
#endif // (defined(WITH_CUDA) && (WITH_CUDA))


/*
 * ProteinVariantMatch::updatParams
 */
void ProteinVariantMatch::updatParams() {

    /* Global mapping parameters */

    // Param slot for heuristic to perfom match
    if (this->theheuristicSlot.IsDirty()) {
         this->theheuristic = static_cast<Heuristic>(this->theheuristicSlot.Param<core::param::EnumParam>()->Value());
         this->theheuristicSlot.ResetDirty();
         this->triggerComputeMatch = true;
    }

    // Parameter for the RMS fitting mode
    if (this->fittingModeSlot.IsDirty()) {
         this->fittingMode = static_cast<RMSFittingMode>(this->fittingModeSlot.Param<core::param::EnumParam>()->Value());
         this->fittingModeSlot.ResetDirty();
         this->triggerComputeMatch = true;
    }

    // Parameter for single frame
    if (this->singleFrameIdxSlot.IsDirty()) {
         this->singleFrameIdx = this->singleFrameIdxSlot.Param<core::param::IntParam>()->Value();
         this->singleFrameIdxSlot.ResetDirty();
         // Single frame has changed, so the feature lists have to be cleared
         this->featureList.Clear();
    }

#if (defined(WITH_CUDA) && (WITH_CUDA))
    /* Parameters for surface mapping */

    // Maximum number of iterations when mapping the mesh
    if (this->surfMapMaxItSlot.IsDirty()) {
        this->surfMapMaxIt = this->surfMapMaxItSlot.Param<core::param::IntParam>()->Value();
        this->surfMapMaxItSlot.ResetDirty();
        this->triggerComputeMatch = true;
    }

    // Maximum number of iterations when regularizing the mesh
    if (this->surfMapRegMaxItSlot.IsDirty()) {
        this->surfMapRegMaxIt = this->surfMapRegMaxItSlot.Param<core::param::IntParam>()->Value();
        this->surfMapRegMaxItSlot.ResetDirty();
        this->triggerComputeMatch = true;
    }

    // Interpolation method used when computing external forces
    if (this->surfMapInterpolModeSlot.IsDirty()) {
        this->surfMapInterpolMode = static_cast<InterpolationMode>(
                this->surfMapInterpolModeSlot.Param<core::param::EnumParam>()->Value());
        this->surfMapInterpolModeSlot.ResetDirty();
        this->triggerComputeMatch = true;
    }

    // Stiffness of the springs defining the spring forces in the surface
    if (this->surfMapSpringStiffnessSlot.IsDirty()) {
        this->surfMapSpringStiffness = this->surfMapSpringStiffnessSlot.Param<core::param::FloatParam>()->Value();
        this->surfMapSpringStiffnessSlot.ResetDirty();
        this->triggerComputeMatch = true;
    }

    // Weighting of the external forces in surface #1, note that the weight
    // of the internal forces is implicitely defined by
    // 1.0 - surf0ExternalForcesWeight
    if (this->surfMapExternalForcesWeightSlot.IsDirty()) {
        this->surfMapExternalForcesWeight = this->surfMapExternalForcesWeightSlot.Param<core::param::FloatParam>()->Value();
        this->surfMapExternalForcesWeightSlot.ResetDirty();
        this->triggerComputeMatch = true;
    }

    // Overall scaling for the forces acting upon surface #1
    if (this->surfMapForcesSclSlot.IsDirty()) {
        this->surfMapForcesScl = this->surfMapForcesSclSlot.Param<core::param::FloatParam>()->Value();
        this->surfMapForcesSclSlot.ResetDirty();
        this->triggerComputeMatch = true;
    }
#endif // (defined(WITH_CUDA) && (WITH_CUDA))

}
