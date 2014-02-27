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
#include "Module.h"
#include "DiagramCall.h"
#include "VariantMatchDataCall.h"
#include "MolecularSurfaceFeature.h"
#include "param/EnumParam.h"
#include "param/FloatParam.h"
#include "param/BoolParam.h"
#include "param/IntParam.h"
#include "param/ButtonParam.h"
#include "RMS.h"
#include "vislib/Log.h"
#include <ctime>


#include "CudaDevArr.h"
#include "ComparativeSurfacePotentialRenderer.cuh"
#include "cuda_error_check.h"
#include "ogl_error_check.h"
#include "CUDAQuickSurf.h"
#include "gridParams.h"
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_ptr.h>


using namespace megamol;
using namespace megamol::protein;
using namespace megamol::core;

// TODO Make diagram renderer react to singleFrame-parameter

/// Minimum force to keep going
const float ProteinVariantMatch::minDispl = 0.000001f;

// Hardcoded parameters for 'quicksurf' class
//const float ProteinVariantMatch::qsParticleRad = 1.0f;
const float ProteinVariantMatch::qsGaussLim = 8.0f;
//const float ProteinVariantMatch::qsGridSpacing = 1.0f;
const bool ProteinVariantMatch::qsSclVanDerWaals = true;
//const float ProteinVariantMatch::qsIsoVal = 0.5f;


/*
 * ProteinVariantMatch::ProteinVariantMatch
 */
ProteinVariantMatch::ProteinVariantMatch(void) : Module() ,
        particleDataCallerSlot("getParticleData", "Connects the module with the partidle data source"),
        volumeDataCallerSlot("getVolumeData", "Connects the module with the volume data source"),
        diagDataOutCalleeSlot("diagDataOut", "Slot to output results of the computations"),
        matrixDataOutCalleeSlot("matrixDataOut", "Slot to output results of the computations"),
        /* Global mapping options */
        triggerMatchingSlot("triggerMatching", "Triggers the computation"),
        theheuristicSlot("heuristic", "Chose the heuristic the matching is based on"),
        fittingModeSlot("rmsMode", "RMS fitting mode"),
        singleFrameIdxSlot("singleFrame", "Idx of the single frame"),
        maxVariantsSlot("maxVariants", "..." ),
        /* Parameters for surface mapping */
        surfMapMaxItSlot("surfmap::maxIt", "Number of iterations when mapping the mesh"),
        surfMapInterpolModeSlot("surfmap::Interpolation", "Interpolation method used for external forces calculation"),
        surfMapSpringStiffnessSlot("surfmap::stiffness", "Stiffness of the internal springs"),
        surfMapExternalForcesWeightSlot("surfmap::externalForcesWeight", "Weight of the external forces"),
        surfMapForcesSclSlot("surfmap::forcesScl", "Scaling of overall force"),
        /* Parameters for surface regularization */
        surfRegMaxItSlot("surfreg::regMaxIt", "Number of iterations when regularizing the mesh"),
        surfRegInterpolModeSlot("surfreg::Interpolation", "Interpolation method used for external forces calculation"),
        surfRegSpringStiffnessSlot("surfreg::stiffness", "Stiffness of the internal springs"),
        surfRegExternalForcesWeightSlot("surfreg::externalForcesWeight", "Weight of the external forces"),
        surfRegForcesSclSlot("surfreg::forcesScl", "Scaling of overall force"),
        /* Quicksurf parameters */
        /// Parameter slot for atom radius scaling
        qsRadSclSlot("quicksurf::qsRadScl","..."),
        qsGridDeltaSlot("quicksurf::qsGridDelta","..."),
        qsIsoValSlot("quicksurf::qsIsoVal","..."),
        nVariants(0),
        cudaqsurf0(NULL), cudaqsurf1(NULL),
        triggerComputeMatch(true) {

    // Data caller for particle data
    this->particleDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->particleDataCallerSlot);

    // Data caller for volume data
    this->volumeDataCallerSlot.SetCompatibleCall<VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->volumeDataCallerSlot);

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

    // Parameter for label size
    this->maxVariants = 10;
    this->maxVariantsSlot.SetParameter(new core::param::IntParam(1));
    this->MakeSlotAvailable(&this->maxVariantsSlot);


    /* Global mapping parameters */

    this->triggerMatchingSlot << new core::param::ButtonParam();
    this->triggerMatchingSlot.SetUpdateCallback(&ProteinVariantMatch::computeMatch);
    this->MakeSlotAvailable(&this->triggerMatchingSlot);

    // Param slot for heuristic to perfom match
    this->theheuristic = RMS_VALUE;
    core::param::EnumParam* fm = new core::param::EnumParam(this->theheuristic);
    fm->SetTypePair(SURFACE_POTENTIAL, "Mean potential diff");
    fm->SetTypePair(SURFACE_POTENTIAL_SIGN, "Mean potential sign diff");
    fm->SetTypePair(MEAN_VERTEX_PATH, "Mean vertex path");
//    fm->SetTypePair(HAUSDORFF_DIST, "Hausdorff distance");
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


    /* Parameters for surface mapping */

    // Maximum number of iterations when mapping the mesh
    this->surfMapMaxIt = 0;
    this->surfMapMaxItSlot.SetParameter(new core::param::IntParam(this->surfMapMaxIt, 0));
    this->MakeSlotAvailable(&this->surfMapMaxItSlot);

    // Interpolation method used when computing external forces
    this->surfMapInterpolMode = DeformableGPUSurfaceMT::INTERP_LINEAR;
    param::EnumParam *s0i = new core::param::EnumParam(int(this->surfMapInterpolMode));
    s0i->SetTypePair(DeformableGPUSurfaceMT::INTERP_LINEAR, "Linear");
    s0i->SetTypePair(DeformableGPUSurfaceMT::INTERP_CUBIC, "Cubic");
    this->surfMapInterpolModeSlot << s0i;
    this->MakeSlotAvailable(&this->surfMapInterpolModeSlot);

    // Stiffness of the springs defining the spring forces in surface #0
    this->surfMapSpringStiffness = 0.5f;
    this->surfMapSpringStiffnessSlot.SetParameter(new core::param::FloatParam(this->surfMapSpringStiffness, 0.0f, 1.0f));
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


    /* Parameters for surface regularization */

    // Maximum number of iterations when regularizing the mesh
    this->surfRegMaxIt = 10;
    this->surfRegMaxItSlot.SetParameter(new core::param::IntParam(this->surfRegMaxIt, 0));
    this->MakeSlotAvailable(&this->surfRegMaxItSlot);

    // Interpolation method used when computing external forces
    this->surfRegInterpolMode = DeformableGPUSurfaceMT::INTERP_LINEAR;
    param::EnumParam *s1i = new core::param::EnumParam(int(this->surfRegInterpolMode));
    s1i->SetTypePair(DeformableGPUSurfaceMT::INTERP_LINEAR, "Linear");
    s1i->SetTypePair(DeformableGPUSurfaceMT::INTERP_CUBIC, "Cubic");
    this->surfRegInterpolModeSlot << s1i;
    this->MakeSlotAvailable(&this->surfRegInterpolModeSlot);

    // Stiffness of the springs defining the spring forces in surface #0
    this->surfRegSpringStiffness = 0.5f;
    this->surfRegSpringStiffnessSlot.SetParameter(new core::param::FloatParam(this->surfRegSpringStiffness, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surfRegSpringStiffnessSlot);

    // Weighting of the external forces in surface #0, note that the weight
    // of the internal forces is implicitely defined by
    // 1.0 - surf0ExternalForcesWeight
    this->surfRegExternalForcesWeight = 0.0f;
    this->surfRegExternalForcesWeightSlot.SetParameter(new core::param::FloatParam(this->surfRegExternalForcesWeight, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surfRegExternalForcesWeightSlot);

    // Overall scaling for the forces acting upon surface #0
    this->surfRegForcesScl = 1.0f;
    this->surfRegForcesSclSlot.SetParameter(new core::param::FloatParam(this->surfRegForcesScl, 0.0f));
    this->MakeSlotAvailable(&this->surfRegForcesSclSlot);


    /* Quicksurf parameter */

    // Quicksurf radius scale
    this->qsRadScl = 1.0f;
    this->qsRadSclSlot.SetParameter(
            new core::param::FloatParam(this->qsRadScl));
    this->MakeSlotAvailable(&this->qsRadSclSlot);

    // Quicksurf grid scaling
    this->qsGridDelta = 2.0f;
    this->qsGridDeltaSlot.SetParameter(
            new core::param::FloatParam(this->qsGridDelta));
    this->MakeSlotAvailable(&this->qsGridDeltaSlot);

    // Quicksurf isovalue
    this->qsIsoVal = 0.5f;
    this->qsIsoValSlot.SetParameter(
            new core::param::FloatParam(this->qsIsoVal));
    this->MakeSlotAvailable(&this->qsIsoValSlot);


    // Inititalize min/max values
    this->minMatchRMSDVal = 0.0f;
    this->maxMatchRMSDVal = 0.0f;


    this->minMatchSurfacePotentialVal = 0.0f;
    this->maxMatchSurfacePotentialVal = 0.0f;

    this->minMatchSurfacePotentialSignVal = 0.0f;
    this->maxMatchSurfacePotentialSignVal = 0.0f;

    this->minMatchMeanVertexPathVal = 0.0f;
    this->maxMatchMeanVertexPathVal = 0.0f;

    this->minMatchHausdorffDistanceVal = 0.0f;
    this->maxMatchHausdorffDistanceVal = 0.0f;

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

    // Create quicksurf objects
    if(!this->cudaqsurf0) {
        this->cudaqsurf0 = new CUDAQuickSurf();
    }
    if(!this->cudaqsurf1) {
        this->cudaqsurf1 = new CUDAQuickSurf();
    }

    if (!DeformableGPUSurfaceMT::InitExtensions()) {
        return false;
    }

    return true;
}


/*
 * ProteinVariantMatch::release
 */
void ProteinVariantMatch::release(void) {

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

    this->atomPosFitted.Release();
    this->atomPos0.Release();
    this->gridDataPos.Release();
    this->rmsPosVec0.Release();
    this->rmsPosVec1.Release();
    this->rmsWeights.Release();
    this->rmsMask.Release();

    CudaSafeCall(this->hausdorffdistVtx_D.Release());
    CudaSafeCall(this->vertexPotentialDiff_D.Release());
    CudaSafeCall(this->vertexPotentialSignDiff_D.Release());

    this->surfEnd.Release();
    this->surfStart.Release();
}


/*
 * ProteinVariantMatch::getData
 */
bool ProteinVariantMatch::getDiagData(core::Call& call) {

    VTIDataCall *vtiCall = this->volumeDataCallerSlot.CallAs<VTIDataCall>();
    if (vtiCall == NULL) {
        return false;
    }

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

    if (!(*vtiCall)(VTIDataCall::CallForGetExtent)) {
        return false;
    }
    if (!(*vtiCall)(VTIDataCall::CallForGetData)) {
        return false;
    }
    this->nVariants = std::min(molCall->FrameCount(), vtiCall->FrameCount());

    DiagramCall *diaCall;
    MolecularSurfaceFeature *ms = NULL;

    diaCall = dynamic_cast<DiagramCall*>(&call);
    if (diaCall == NULL) {
        return false;
    }

    // Update parameter slots
    this->updatParams();

    // Add series to diagram call if necessary (this only needs to be done once,
    // subsequently data can be added to the series)
    if (this->featureList.Count() < 1) {
        this->featureList.SetCount(5);

        // Define feature list for rmsd
        this->featureList[0] = new DiagramCall::DiagramSeries("RMSD",
                new MolecularSurfaceFeature(this->nVariants,
                        Vec3f(0.0f, 0.0f, 0.0f)));
        this->featureList[0]->SetColor(1.0f, 1.0f, 0.0f, 1.0f);

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

    }
    if (diaCall->GetSeriesCount() < 1) {
        diaCall->AddSeries(this->featureList[0]);
        diaCall->AddSeries(this->featureList[1]);
        diaCall->AddSeries(this->featureList[2]);
        diaCall->AddSeries(this->featureList[3]);
        diaCall->AddSeries(this->featureList[4]);
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
//    printf("Mean hausdorff: ");
    ms = static_cast<MolecularSurfaceFeature*>(this->featureList[4]->GetMappable());
    for (unsigned int i = 0; i < this->nVariants; ++i) {
        ms->AppendValue(i,
                this->matchSurfacePotentialSign[this->singleFrameIdx*this->nVariants+i]/
                this->maxMatchSurfacePotentialSignVal);
        printf("%f ", this->matchSurfacePotentialSign[this->singleFrameIdx*this->nVariants+i]/
                this->maxMatchSurfacePotentialSignVal);
    }
//    printf("\n");

    // Mean hausdorff distance
//    printf("Mean hausdorff: ");
    ms = static_cast<MolecularSurfaceFeature*>(this->featureList[2]->GetMappable());
    for (unsigned int i = 0; i < this->nVariants; ++i) {
        ms->AppendValue(i,
                this->matchMeanVertexPath[this->singleFrameIdx*this->nVariants+i]/
                this->maxMatchMeanVertexPathVal);
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

    return true;
}


/*
 * ProteinVariantMatch::getMatrixData
 */
bool ProteinVariantMatch::getMatrixData(core::Call& call) {

    VTIDataCall *vtiCall = this->volumeDataCallerSlot.CallAs<VTIDataCall>();
    if (vtiCall == NULL) {
        return false;
    }

    MolecularDataCall *molCall = this->particleDataCallerSlot.CallAs<MolecularDataCall>();
    if (molCall == NULL) {
        return false;
    }

    if (!(*molCall)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }

//    if (!(*molCall)(MolecularDataCall::CallForGetData)) {
//        return false;
//    }

    if (!(*vtiCall)(VTIDataCall::CallForGetExtent)) {
        return false;
    }
//    if (!(*vtiCall)(VTIDataCall::CallForGetData)) {
//        return false;
//    }
    this->nVariants = std::min(molCall->FrameCount(), vtiCall->FrameCount());
    this->nVariants = std::min(static_cast<unsigned int>(this->maxVariants), this->nVariants);

    if (this->matchRMSD.Count() != this->nVariants*this->nVariants) {
        this->matchSurfacePotential.SetCount(this->nVariants*this->nVariants);
        this->matchSurfacePotentialSign.SetCount(this->nVariants*this->nVariants);
        this->matchMeanVertexPath.SetCount(this->nVariants*this->nVariants);
        this->matchHausdorffDistance.SetCount(this->nVariants*this->nVariants);
        this->matchRMSD.SetCount(this->nVariants*this->nVariants);
#pragma omp for
        for (int i = 0; i < static_cast<int>(this->nVariants*this->nVariants); ++i) {
            this->matchSurfacePotential[i] = 0.0f;
            this->matchSurfacePotentialSign[i] = 0.0f;
            this->matchMeanVertexPath[i] = 0.0f;
            this->matchHausdorffDistance[i] = 0.0f;
            this->matchRMSD[i] = 0.0f;
        }
        // Loop through files to obtain labels
        this->labels.SetCount(this->nVariants);
        for (unsigned int v = 0; v < this->nVariants; ++v) {
            vtiCall->SetFrameID(v);
            if (!(*vtiCall)(VTIDataCall::CallForGetExtent)) {
                printf("call for get extent fail\n");
                return false;
            }
            if (!(*vtiCall)(VTIDataCall::CallForGetData)) {
                printf("call for get data fail\n");
                return false;
            }
            this->labels[v] = vtiCall->GetPointDataArrayId(0, 0);
        }
    }

    VariantMatchDataCall *dc;

    // Update parameter slots
    this->updatParams();

    dc = dynamic_cast<VariantMatchDataCall*>(&call);
    if (dc == NULL) {
        return false;
    }

    dc->SetVariantCnt(this->nVariants);
    switch(this->theheuristic) {
    case SURFACE_POTENTIAL :
        dc->SetMatch(this->matchSurfacePotential.PeekElements());
        dc->SetMatchRange(this->minMatchSurfacePotentialVal, this->maxMatchSurfacePotentialVal);
        break;
    case SURFACE_POTENTIAL_SIGN :
        dc->SetMatch(this->matchSurfacePotentialSign.PeekElements());
        dc->SetMatchRange(this->minMatchSurfacePotentialSignVal, this->maxMatchSurfacePotentialSignVal);
        break;
    case MEAN_VERTEX_PATH :
        dc->SetMatch(this->matchMeanVertexPath.PeekElements());
        dc->SetMatchRange(this->minMatchMeanVertexPathVal, this->maxMatchMeanVertexPathVal);
        break;
    case HAUSDORFF_DIST :
        dc->SetMatch(this->matchHausdorffDistance.PeekElements());
        dc->SetMatchRange(this->minMatchHausdorffDistanceVal, this->maxMatchHausdorffDistanceVal);
        break;
    case RMS_VALUE :
        dc->SetMatch(this->matchRMSD.PeekElements());
        dc->SetMatchRange(this->minMatchRMSDVal, this->maxMatchRMSDVal);
        break;
    }

    // Set labels
    dc->SetLabels(this->labels.PeekElements());

    return true;
}


/*
 * ProteinVariantMatch::getAtomPosArray
 */
void ProteinVariantMatch::getAtomPosArray(MolecularDataCall *mol, HostArr<float> &posArr, size_t &particleCnt) {

    using namespace vislib::sys;
    using namespace vislib::math;

    // (Re-)allocate memory for intermediate particle data
    posArr.Validate(mol->AtomCount()*4);

    // Gather atom positions for the density map
    particleCnt = 0;
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

                        posArr.Peek()[4*particleCnt+0] = mol->AtomPositions()[3*atomIdx+0];
                        posArr.Peek()[4*particleCnt+1] = mol->AtomPositions()[3*atomIdx+1];
                        posArr.Peek()[4*particleCnt+2] = mol->AtomPositions()[3*atomIdx+2];

                        if(this->qsSclVanDerWaals) {
                            posArr.Peek()[4*particleCnt+3] = mol->AtomTypes()[mol->AtomTypeIndices()[atomIdx]].Radius();
                        }
                        else {
                            posArr.Peek()[4*particleCnt+3] = 1.0f;
                        }

                        this->maxAtomRad = std::max(this->maxAtomRad, posArr.Peek()[4*particleCnt+3]);
                        this->minAtomRad = std::min(this->minAtomRad, posArr.Peek()[4*particleCnt+3]);

                        particleCnt++;
                    }
                }
            }
        }
    }
}


/*
 * ProteinVariantMatch::computeDensityMap
 */
bool ProteinVariantMatch::computeDensityMap(
        float* atomPos,
        size_t atomCnt,
        CUDAQuickSurf *cqs) {

//    printf("Compute density map particleCount %u\n", atomCnt);

    this->atomPosTmp.Validate(atomCnt*4);

    using namespace vislib::sys;
    using namespace vislib::math;

    float gridXAxisLen, gridYAxisLen, gridZAxisLen;
    float padding;

    // Compute padding for the density map
    padding = this->maxAtomRad*this->qsRadScl + this->qsGridDelta*10;

    // Init grid parameters
    this->volOrg.x = this->bboxParticles.GetLeft()   - padding;
    this->volOrg.y = this->bboxParticles.GetBottom() - padding;
    this->volOrg.z = this->bboxParticles.GetBack()   - padding;
    this->volMaxC.x = this->bboxParticles.GetRight() + padding;
    this->volMaxC.y = this->bboxParticles.GetTop()   + padding;
    this->volMaxC.z = this->bboxParticles.GetFront() + padding;
    gridXAxisLen = this->volMaxC.x - this->volOrg.x;
    gridYAxisLen = this->volMaxC.y - this->volOrg.y;
    gridZAxisLen = this->volMaxC.z - this->volOrg.z;
    this->volDim.x = (int) ceil(gridXAxisLen / this->qsGridDelta);
    this->volDim.y = (int) ceil(gridYAxisLen / this->qsGridDelta);
    this->volDim.z = (int) ceil(gridZAxisLen / this->qsGridDelta);
    gridXAxisLen = (this->volDim.x-1) * this->qsGridDelta;
    gridYAxisLen = (this->volDim.y-1) * this->qsGridDelta;
    gridZAxisLen = (this->volDim.z-1) * this->qsGridDelta;
    this->volMaxC.x = this->volOrg.x + gridXAxisLen;
    this->volMaxC.y = this->volOrg.y + gridYAxisLen;
    this->volMaxC.z = this->volOrg.z + gridZAxisLen;
    this->volDelta.x = this->qsGridDelta;
    this->volDelta.y = this->qsGridDelta;
    this->volDelta.z = this->qsGridDelta;

    // Set particle positions
#pragma omp parallel for
    for (int cnt = 0; cnt < static_cast<int>(atomCnt); ++cnt) {
        this->atomPosTmp.Peek()[4*cnt+0] = atomPos[4*cnt+0] - this->volOrg.x;
        this->atomPosTmp.Peek()[4*cnt+1] = atomPos[4*cnt+1] - this->volOrg.y;
        this->atomPosTmp.Peek()[4*cnt+2] = atomPos[4*cnt+2] - this->volOrg.z;
        this->atomPosTmp.Peek()[4*cnt+3] = atomPos[4*cnt+3];
    }

//    for (int cnt = 0; cnt < static_cast<int>(atomCnt); ++cnt) {
//        printf("data pos %i %f %f %f\n",cnt,
//                atomPos[4*cnt+0],
//                atomPos[4*cnt+1],
//                atomPos[4*cnt+2]);
//    }

//    printf("volOrg %f %f %f\n", this->volOrg.x,this->volOrg.y,this->volOrg.z);
    //printf("volDim %i %i %i\n", this->volDim.x,this->volDim.y,this->volDim.z);

    //printf("%i\n", this->volDim.x*this->volDim.y*this->volDim.z);

//#ifdef USE_TIMER
//    float dt_ms;
//    cudaEvent_t event1, event2;
//    cudaEventCreate(&event1);
//    cudaEventCreate(&event2);
//    cudaEventRecord(event1, 0);
//#endif // USE_TIMER

    // Compute uniform grid
    int rc = cqs->calc_map(
            atomCnt,
            this->atomPosTmp.Peek(),
            NULL,   // Pointer to 'color' array
            false,  // Do not use 'color' array
            (float*)&this->volOrg,
            (int*)&this->volDim,
            this->maxAtomRad,
            this->qsRadScl, // Radius scaling
            this->qsGridDelta,
            this->qsIsoVal,
            this->qsGaussLim);

//    printf("max atom rad %f\n", this->maxAtomRad);

//#ifdef VERBOSE
//    cudaEventRecord(event2, 0);
//    cudaEventSynchronize(event1);
//    cudaEventSynchronize(event2);
//    cudaEventElapsedTime(&dt_ms, event1, event2);
//    printf("CUDA time for 'quicksurf':                             %.10f sec\n",
//            dt_ms/1000.0f);
//#endif // USE_TIMER

    if (rc != 0) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Quicksurf class returned val != 0\n", this->ClassName());
        return false;
    }

    return CheckForCudaErrorSync();
}


/*
 *  ProteinVariantMatch::computeDensityBBox
 */
void ProteinVariantMatch::computeDensityBBox(
        const float *atomPos1,
        const float *atomPos2,
        size_t atomCnt1,
        size_t atomCnt2) {

    float3 minC, maxC;
    minC.x = maxC.x = atomPos1[0];
    minC.y = maxC.y = atomPos1[1];
    minC.z = maxC.z = atomPos1[2];
    for (size_t i = 0; i < atomCnt1; ++i) {
        minC.x = std::min(minC.x, atomPos1[4*i+0]);
        minC.y = std::min(minC.y, atomPos1[4*i+1]);
        minC.z = std::min(minC.z, atomPos1[4*i+2]);
        maxC.x = std::max(maxC.x, atomPos1[4*i+0]);
        maxC.y = std::max(maxC.y, atomPos1[4*i+1]);
        maxC.z = std::max(maxC.z, atomPos1[4*i+2]);
    }
    for (size_t i = 0; i < atomCnt2; ++i) {
        minC.x = std::min(minC.x, atomPos2[4*i+0]);
        minC.y = std::min(minC.y, atomPos2[4*i+1]);
        minC.z = std::min(minC.z, atomPos2[4*i+2]);
        maxC.x = std::max(maxC.x, atomPos2[4*i+0]);
        maxC.y = std::max(maxC.y, atomPos2[4*i+1]);
        maxC.z = std::max(maxC.z, atomPos2[4*i+2]);
    }

    this->bboxParticles.Set(
            minC.x, minC.y, minC.z,
            maxC.x, maxC.y, maxC.z);
//
//    // DEBUG Print new bounding box
//    printf("bbboxParticles: %f %f %f %f %f %f\n",
//            minC.x, minC.y, minC.z,
//            maxC.x, maxC.y, maxC.z);
//    printf("atomCnt0: %u\n",atomCnt1);
//    printf("atomCnt1: %u\n",atomCnt2);
//    // END DEBUG

}

/*
 * ProteinVariantMatch::computeMatch
 */
bool ProteinVariantMatch::computeMatch(param::ParamSlot& p) {

    if (!this->computeMatchSurfMapping()) {
        return false;
    }

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
    this->minMatchMeanVertexPathVal = 1000000.0f;
    this->maxMatchMeanVertexPathVal = 0.0f;
    this->minMatchHausdorffDistanceVal = 1000000.0f;
    this->maxMatchHausdorffDistanceVal = 0.0f;
    this->minMatchSurfacePotentialSignVal = 1000000.0f;
    this->maxMatchSurfacePotentialSignVal = 0.0f;
    float rotation[3][3];
    float translation[3];
    float rmsVal;

    int3 texDim0, texDim1;
    float3 texOrg0, texOrg1, texDelta0, texDelta1;
    size_t particleCnt0, particleCnt1;

#if defined(OUTPUT_PROGRESS)
    //float steps = (this->nVariants*(this->nVariants+1))*0.5f;
    float steps = this->nVariants*this->nVariants;
    float currStep = 0.0f;
#endif // defined(OUTPUT_PROGRESS)

#if defined(USE_TIMER)
    time_t t = clock();
#endif // defined(USE_TIMER)

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

        // Copy potential texture #0 to device memory
        if (!CudaSafeCall(this->potentialTex0_D.Validate(
                volCall->GetGridsize().X()*
                volCall->GetGridsize().Y()*
                volCall->GetGridsize().Z()))) {
            return false;
        }
        if (!CudaSafeCall(cudaMemcpy(this->potentialTex0_D.Peek(),
                (float*)(volCall->GetPointDataByIdx(0,0)),
                sizeof(float)*volCall->GetGridsize().X()*
                volCall->GetGridsize().Y()*volCall->GetGridsize().Z(),
                cudaMemcpyHostToDevice))) {
            return false;
        }

        // Keep track of potential texture dimensions
        texDim0.x = volCall->GetGridsize().X();
        texDim0.y = volCall->GetGridsize().Y();
        texDim0.z = volCall->GetGridsize().Z();
        texOrg0.x = volCall->GetOrigin().X();
        texOrg0.y = volCall->GetOrigin().Y();
        texOrg0.z = volCall->GetOrigin().Z();
        texDelta0.x = volCall->GetSpacing().X();
        texDelta0.y = volCall->GetSpacing().Y();
        texDelta0.z = volCall->GetSpacing().Z();
//
//        printf("Init potential map (new), frame %u\n", volCall->FrameID());
//        printf("Min coord %f %f %f\n",
//                minCNew.x, minCNew.y, minCNew.z);
//        printf("Max coord %f %f %f\n",
//                maxCNew.x, maxCNew.y, maxCNew.z);

        // Get rms atom positions
        this->rmsPosVec0.Validate(molCall->AtomCount()*3);
        if (!this->getRMSPosArray(molCall, this->rmsPosVec0, posCnt0)) {
            return false;
        }

        this->maxAtomRad = 0.0f;
        this->minAtomRad = 1000000.0f;
        // Get atom pos vector (dismisses solvent molecules)
        this->getAtomPosArray(
                molCall,
                this->atomPos0,
                particleCnt0);

        molCall->Unlock(); // Unlock the frame
        volCall->Unlock(); // Unlock the frame

        // Loop through all variants
        for (unsigned int j = 0; j < this->nVariants; ++j) {

#if defined(VERBOSE)
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                    "%s: matching variants %i and %i (%u variants)",
                    this->ClassName(), i, j, this->nVariants);
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

            // Copy potential texture #1 tp device memory
            if (!CudaSafeCall(this->potentialTex1_D.Validate(
                    volCall->GetGridsize().X()*
                    volCall->GetGridsize().Y()*
                    volCall->GetGridsize().Z()))) {
                return false;
            }
            if (!CudaSafeCall(cudaMemcpy(this->potentialTex1_D.Peek(),
                    (float*)(volCall->GetPointDataByIdx(0,0)),
                    sizeof(float)*volCall->GetGridsize().X()*
                    volCall->GetGridsize().Y()*volCall->GetGridsize().Z(),
                    cudaMemcpyHostToDevice))) {
                return false;
            }
            // Keep track of potential tex dimensions
            texDim1.x = volCall->GetGridsize().X();
            texDim1.y = volCall->GetGridsize().Y();
            texDim1.z = volCall->GetGridsize().Z();
            texOrg1.x = volCall->GetOrigin().X();
            texOrg1.y = volCall->GetOrigin().Y();
            texOrg1.z = volCall->GetOrigin().Z();
            texDelta1.x = volCall->GetSpacing().X();
            texDelta1.y = volCall->GetSpacing().Y();
            texDelta1.z = volCall->GetSpacing().Z();

//            printf("Init potential map (old), frame %u\n", volCall->FrameID());
//            printf("grid dim %u %u %u\n", volCall->GetGridsize().X(),
//                    volCall->GetGridsize().Y(),
//                    volCall->GetGridsize().Z());
//            printf("Min coord %f %f %f\n",
//                    minCOld.x, minCOld.y, minCOld.z);
//            printf("Max coord %f %f %f\n",
//                    maxCOld.x, maxCOld.y, maxCOld.z);


#define COMPUTE_RUNTIME
#ifdef COMPUTE_RUNTIME
            float dt_ms;
            cudaEvent_t event1, event2;
            cudaEventCreate(&event1);
            cudaEventCreate(&event2);
            cudaEventRecord(event1, 0);
#endif // COMPUTE_RUNTIME

            // Get atom positions
            this->rmsPosVec1.Validate(molCall->AtomCount()*3);
            if (!this->getRMSPosArray(molCall, this->rmsPosVec1, posCnt1)) {
                return false;
            }

            // Print atom count
            //printf("%u\n", molCall->AtomCount());

            // Get atom pos vector (dismisses solvent molecules)
            this->getAtomPosArray(
                    molCall,
                    this->atomPosFitted,
                    particleCnt1);

            // Compute RMS value and transformation
            int posCnt = std::min(posCnt0, posCnt1);
//            if (posCnt0 != posCnt1) {
//                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
//                        "%s: Unable to perform RMS fitting (non-equal atom \
//count (%u vs. %u))", this->ClassName(), posCnt0, posCnt1);
//                return false;
//            }
            rmsVal = this->getRMS(
                    this->rmsPosVec0.Peek(),
                    this->rmsPosVec1.Peek(),
                    posCnt, true, 2, rotation, translation);
            if (rmsVal > 100.0f) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: Unable to perform surface matching (rms val = %f)",
                        this->ClassName(), rmsVal);
                return false;
            }

            // Safe RMSD in matrix
            this->matchRMSD[i*this->nVariants+j] = rmsVal;
            this->minMatchRMSDVal =
                    std::min(this->minMatchRMSDVal, this->matchRMSD[this->nVariants*i+j]);
            this->maxMatchRMSDVal =
                    std::max(this->maxMatchRMSDVal, this->matchRMSD[this->nVariants*i+j]);

            // Fit atom positions of source structure
            Vec3f centroid(0.0f, 0.0f, 0.0f);
            for (int cnt = 0; cnt < static_cast<int>(particleCnt1); ++cnt) {
                centroid += Vec3f(
                        this->atomPosFitted.Peek()[cnt*4+0],
                        this->atomPosFitted.Peek()[cnt*4+1],
                        this->atomPosFitted.Peek()[cnt*4+2]);
            }
            centroid /= static_cast<float>(particleCnt1);

            // Rotate/translate positions
#pragma omp parallel for
            for (int a = 0; a < static_cast<int>(particleCnt1); ++a) {

                Vec3f pos(this->atomPosFitted.Peek()[4*a+0],
                          this->atomPosFitted.Peek()[4*a+1],
                          this->atomPosFitted.Peek()[4*a+2]);
                pos -= centroid;
                pos = this->rmsRotation*pos;
                pos += this->rmsTranslation;
                this->atomPosFitted.Peek()[4*a+0] = pos.X();
                this->atomPosFitted.Peek()[4*a+1] = pos.Y();
                this->atomPosFitted.Peek()[4*a+2] = pos.Z();
            }

//            printf("CALL 1 frame %u\n", molCall->FrameID());

//#ifdef COMPUTE_RUNTIME
//            cudaEventRecord(event2, 0);
//            cudaEventSynchronize(event1);
//            cudaEventSynchronize(event2);
//            cudaEventElapsedTime(&dt_ms, event1, event2);
//            printf("RMSD : %.10f\n",
//                    dt_ms/1000.0f);
//            cudaEventRecord(event1, 0);
//#endif // COMPUTE_RUNTIME

            this->computeDensityBBox(
                    this->atomPos0.Peek(),
                    this->atomPosFitted.Peek(),
                    particleCnt0,
                    particleCnt1);

            // Compute density map of variant #0
            if (!this->computeDensityMap(
                    this->atomPos0.Peek(),
                    particleCnt0,
                    (CUDAQuickSurf *)this->cudaqsurf0)) {
                return false;
            }

            // Compute density map of variant #1
            if (!this->computeDensityMap(
                    this->atomPosFitted.Peek(),
                    particleCnt1,
                    (CUDAQuickSurf *)this->cudaqsurf1)) {
                return false;
            }

#ifdef COMPUTE_RUNTIME
            cudaEventRecord(event2, 0);
            cudaEventSynchronize(event1);
            cudaEventSynchronize(event2);
            cudaEventElapsedTime(&dt_ms, event1, event2);
            printf("Vol : %.10f sec\n",
                    dt_ms/1000.0f);
            cudaEventRecord(event1, 0);
#endif // COMPUTE_RUNTIME


            // Get vertex positions based on the level set
            if (!surfTarget.ComputeVertexPositions(
                    ((CUDAQuickSurf*)this->cudaqsurf0)->getMap(),
                    this->volDim,
                    this->volOrg,
                    this->volDelta,
                    this->qsIsoVal)) {

                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: could not compute vertex positions #1",
                        this->ClassName());

                return false;
            }

            // Get vertex positions based on the level set
            if (!surfStart.ComputeVertexPositions(
                    ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                    this->volDim,
                    this->volOrg,
                    this->volDelta,
                    this->qsIsoVal)) {

                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: could not compute vertex positions #1",
                        this->ClassName());

                return false;
            }
            ::CheckForCudaErrorSync();

            // Build triangle mesh from vertices
            if (!surfStart.ComputeTriangles(
                    ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                    this->volDim,
                    this->volOrg,
                    this->volDelta,
                    this->qsIsoVal)) {

                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: could not compute vertex triangles #1",
                        this->ClassName());

                return false;
            }
            ::CheckForCudaErrorSync();

            // Compute vertex connectivity
            if (!surfStart.ComputeConnectivity(
                    ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                    this->volDim,
                    this->volOrg,
                    this->volDelta,
                    this->qsIsoVal)) {

                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: could not compute vertex connectivity #1",
                        this->ClassName());

                return false;
            }
            ::CheckForCudaErrorSync();


            if (!this->surfStart.ComputeTriangleNeighbors(
                    ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                    this->volDim,
                    this->volOrg,
                    this->volDelta,
                    this->qsIsoVal)) {

                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: could not compute triangle neighbors #2",
                        this->ClassName());
                return false;
            }

            // print vertex count
            //printf("%u\n", surfStart.GetVertexCnt());

            // Regularize the mesh of surface #1
            if (!this->surfStart.MorphToVolumeGradient(
                    ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                    this->volDim,
                    this->volOrg,
                    this->volDelta,
                    this->qsIsoVal,
                    this->surfRegInterpolMode,
                    this->surfRegMaxIt,
                    qsGridDelta/100.0f*this->surfRegForcesScl,
                    this->surfRegSpringStiffness,
                    this->surfRegForcesScl,
                    this->surfRegExternalForcesWeight)) {

                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: could not regularize surface #1",
                        this->ClassName());

                return false;
            }

#ifdef COMPUTE_RUNTIME
            cudaEventRecord(event2, 0);
            cudaEventSynchronize(event1);
            cudaEventSynchronize(event2);
            cudaEventElapsedTime(&dt_ms, event1, event2);
            printf("Mesh : %.10f\n",
                    dt_ms/1000.0f);
            cudaEventRecord(event1, 0);
#endif // COMPUTE_RUNTIME

//#ifdef COMPUTE_RUNTIME
//            cudaEventRecord(event2, 0);
//            cudaEventSynchronize(event1);
//            cudaEventSynchronize(event2);
//            cudaEventElapsedTime(&dt_ms, event1, event2);
//            printf("Regularization %.10f sec\n",
//                    dt_ms/1000.0f);
//            cudaEventRecord(event1, 0);
//#endif // COMPUTE_RUNTIME

            // Make deep copy of start surface
            this->surfEnd = this->surfStart;

            // Morph end surface to its final position using two-way gradient
            // vector flow
            if (!this->surfEnd.MorphToVolumeTwoWayGVF(
                    ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                    ((CUDAQuickSurf*)this->cudaqsurf0)->getMap(),
                    this->surfEnd.PeekCubeStates(),
                    this->surfStart.PeekCubeStates(),
                    this->volDim,
                    this->volOrg,
                    this->volDelta,
                    this->qsIsoVal,
                    this->surfMapInterpolMode,
                    this->surfMapMaxIt,
                    qsGridDelta/100.0f*this->surfMapForcesScl,
                    this->surfMapSpringStiffness,
                    this->surfMapForcesScl,
                    this->surfMapExternalForcesWeight,
                    0.1f,
                    100,
                    true,
                    true)) { // TODO Change back
                return false;
            }

            // Refine mesh
            // Perform subdivision with subsequent deformation to create a fine
            // target mesh enough
            float initialStep = 1.0;
            const uint maxSubdivLevel = 10;
            int newTris;
            for (uint i = 0; i < maxSubdivLevel; ++i) {
                printf("SUBDIV #%i\n", i);

                newTris = this->surfEnd.RefineMesh(
                        1,
                        ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                        this->volDim,
                        this->volOrg,
                        this->volDelta,
                        this->qsIsoVal,
                        this->volDelta.x*2); // TODO Whats a good maximum edge length?

                if (newTris < 0) {

                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                            "%s: could not refine mesh",
                            this->ClassName());

                    return false;
                }

                if (newTris == 0) break; // No more subdivisions needed

                // Perform morphing
                // Morph surface #2 to shape #1 using Two-Way-GVF

                initialStep*=0.1f;

                if (!this->surfEnd.MorphToVolumeTwoWayGVF( // TODo DO not compute GVF every time
                        ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                        ((CUDAQuickSurf*)this->cudaqsurf0)->getMap(),
                        this->surfStart.PeekCubeStates(),
                        this->surfEnd.PeekCubeStates(),
                        this->volDim,
                        this->volOrg,
                        this->volDelta,
                        this->qsIsoVal,
                        this->surfMapInterpolMode,
                        this->surfMapMaxIt,
                        this->qsGridDelta/100.0f*this->surfMapForcesScl*initialStep,
                        this->surfMapSpringStiffness,
                        this->surfMapForcesScl*initialStep,
                        this->surfMapExternalForcesWeight,
                        0.1f,
                        100,  // TODO Change back to 30 or so?
                        false,
                        false)) {

                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                            "%s: could not compute Two-Way-GVF deformation",
                            this->ClassName());

                    return false;
                }
            }


#ifdef COMPUTE_RUNTIME
            cudaEventRecord(event2, 0);
            cudaEventSynchronize(event1);
            cudaEventSynchronize(event2);
            cudaEventElapsedTime(&dt_ms, event1, event2);
            printf("Mapping %.10f sec\n",
                    dt_ms/1000.0f);
            cudaEventRecord(event1, 0);
#endif // COMPUTE_RUNTIME

            // Compute surface area
            float surfArea = this->surfEnd.GetTotalSurfArea();
            printf("Surface area %f, vertexCnt %u\n", surfArea, this->surfEnd.GetVertexCnt());

            /* Compute different metrics on a per-vertex basis */

            // Compute texture coordinates
            Mat3f rmsRotInv(this->rmsRotation);
            if (!rmsRotInv.Invert()) {
                printf("Could not invert rotation matrix\n");
                return false;
            }

            // 1. Compute potential difference per vertex

//            if (!CudaSafeCall(this->vertexPotentialDiff_D.Validate(surfEnd.GetVertexCnt()))) {
//                return false;
//            }
//
//            if (!DeformableGPUSurfaceMT::ComputeVtxDiffValueFitted(
//                    this->vertexPotentialDiff_D.Peek(),
//                    centroid.PeekComponents(),
//                    rmsRotInv.PeekComponents(),
//                    this->rmsTranslation.PeekComponents(),
//                    this->potentialTex0_D.Peek(),
//                    texDim0, texOrg0, texDelta0,
//                    this->potentialTex1_D.Peek(),
//                    texDim1, texOrg1, texDelta1,
//                    surfStart.GetVtxDataVBO(),
//                    surfEnd.GetVtxDataVBO(),
//                    surfStart.GetVertexCnt()
//                    )) {
//                return false;
//            }
//
//            // Integrate over surface area and write to matrix
//            float meanPotentialDiff = surfEnd.IntOverSurfArea(this->vertexPotentialDiff_D.Peek());
//
////            printf("Surface area %f\, potentialDiff %f, meanPotentialDiff %f\n", surfArea, meanPotentialDiff, meanPotentialDiff/surfArea);
//            this->matchSurfacePotential[i*this->nVariants+j] = meanPotentialDiff/surfArea;
//
//
////            if (i !=j) {
//            this->minMatchSurfacePotentialVal =
//                    std::min(this->minMatchSurfacePotentialVal, this->matchSurfacePotential[this->nVariants*i+j]);
//            this->maxMatchSurfacePotentialVal =
//                    std::max(this->maxMatchSurfacePotentialVal, this->matchSurfacePotential[this->nVariants*i+j]);
////            }

            // 2. Compute potential sign difference per vertex

//            if (!CudaSafeCall(this->vertexPotentialSignDiff_D.Validate(surfEnd.GetVertexCnt()))) {
//                return false;
//            }
//            if (!DeformableGPUSurfaceMT::ComputeVtxSignDiffValueFitted(
//                    this->vertexPotentialSignDiff_D.Peek(),
//                    centroid.PeekComponents(),
//                    rmsRotInv.PeekComponents(),
//                    this->rmsTranslation.PeekComponents(),
//                    this->potentialTex0_D.Peek(),
//                    texDim0, texOrg0, texDelta0,
//                    this->potentialTex1_D.Peek(),
//                    texDim1, texOrg1, texDelta1,
//                    surfStart.GetVtxDataVBO(),
//                    surfEnd.GetVtxDataVBO(),
//                    surfStart.GetVertexCnt()
//                    )) {
//                return false;
//            }
//            // Integrate over surface area and write to matrix
//            float meanPotentialSignDiff = surfEnd.IntOverSurfArea(this->vertexPotentialSignDiff_D.Peek());
//            this->matchSurfacePotentialSign[i*this->nVariants+j] = meanPotentialSignDiff/surfArea;
//
//            this->minMatchSurfacePotentialSignVal =
//                    std::min(this->minMatchSurfacePotentialSignVal, this->matchSurfacePotentialSign[this->nVariants*i+j]);
//            this->maxMatchSurfacePotentialSignVal =
//                    std::max(this->maxMatchSurfacePotentialSignVal, this->matchSurfacePotentialSign[this->nVariants*i+j]);


            // 3. Compute mean vertex path

            // Compute uncertainty for new vertices
            // Update uncertainty VBO for new vertices
            if (!this->surfEnd.TrackPathSubdivVertices(
                    ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                    this->volDim,
                    this->volOrg,
                    this->volDelta,
                    this->surfMapForcesScl,
                    this->qsGridDelta/100.0f*this->surfMapForcesScl,
                    this->qsIsoVal,
                    this->surfMapMaxIt)) {
                return false;
            }

            float meanVertexPath = this->surfEnd.IntVtxPathOverSurfArea();

            this->matchMeanVertexPath[i*this->nVariants+j] = meanVertexPath/surfArea;

//            if (i != j) {
                this->minMatchMeanVertexPathVal =
                        std::min(this->minMatchMeanVertexPathVal, this->matchMeanVertexPath[this->nVariants*i+j]);
                this->maxMatchMeanVertexPathVal =
                        std::max(this->maxMatchMeanVertexPathVal, this->matchMeanVertexPath[this->nVariants*i+j]);
//            }
            printf("Mean vertex path: min %f, max %f\n",
                    this->minMatchMeanVertexPathVal,
                    this->maxMatchMeanVertexPathVal);

#ifdef COMPUTE_RUNTIME
            cudaEventRecord(event2, 0);
            cudaEventSynchronize(event1);
            cudaEventSynchronize(event2);
            cudaEventElapsedTime(&dt_ms, event1, event2);
            printf("metrics : %.10f\n", dt_ms/1000.0f);
#endif // COMPUTE_RUNTIME

            molCall->Unlock(); // Unlock the frame
            volCall->Unlock(); // Unlock the frame

#if defined(OUTPUT_PROGRESS)
            // Ouput progress
            currStep += 1.0f;
           // if (static_cast<unsigned int>(currStep)%10 == 0) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                        "%s: matching surfaces %3u%%", this->ClassName(),
                        static_cast<unsigned int>(currStep/steps*100));
           // }
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
float ProteinVariantMatch::getRMS(
        float *atomPos0,
        float *atomPos1,
        unsigned int cnt,
        bool fit,
        int flag,
        float rotation[3][3],
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

        this->rmsTranslation.Set(translation[0], translation[1], translation[2]);
        this->rmsRotation.SetAt(0, 0, rotation[0][0]);
        this->rmsRotation.SetAt(0, 1, rotation[0][1]);
        this->rmsRotation.SetAt(0, 2, rotation[0][2]);
        this->rmsRotation.SetAt(1, 0, rotation[1][0]);
        this->rmsRotation.SetAt(1, 1, rotation[1][1]);
        this->rmsRotation.SetAt(1, 2, rotation[1][2]);
        this->rmsRotation.SetAt(2, 0, rotation[2][0]);
        this->rmsRotation.SetAt(2, 1, rotation[2][1]);
        this->rmsRotation.SetAt(2, 2, rotation[2][2]);
    }

    return rmsValue;
}


/*
 * ProteinVariantMatch::getRMSPosArray
 */
bool ProteinVariantMatch::getRMSPosArray(
        MolecularDataCall *mol,
        HostArr<float> &posArr,
        unsigned int &cnt) {
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
 * ProteinVariantMatch::updatParams
 */
void ProteinVariantMatch::updatParams() {

    /* Global mapping parameters */

    // Param slot for heuristic to perfom match
    if (this->theheuristicSlot.IsDirty()) {
         this->theheuristic = static_cast<Heuristic>(this->theheuristicSlot.Param<core::param::EnumParam>()->Value());
         this->theheuristicSlot.ResetDirty();
    }

    // Parameter for the RMS fitting mode
    if (this->fittingModeSlot.IsDirty()) {
         this->fittingMode = static_cast<RMSFittingMode>(this->fittingModeSlot.Param<core::param::EnumParam>()->Value());
         this->fittingModeSlot.ResetDirty();
    }

    // Parameter for single frame
    if (this->singleFrameIdxSlot.IsDirty()) {
         this->singleFrameIdx = this->singleFrameIdxSlot.Param<core::param::IntParam>()->Value();
         this->singleFrameIdxSlot.ResetDirty();
         // Single frame has changed, so the feature lists have to be cleared
         this->featureList.Clear();
    }

    // Max variants
    if (this->maxVariantsSlot.IsDirty()) {
        this->maxVariants = this->maxVariantsSlot.Param<core::param::IntParam>()->Value();
        this->maxVariantsSlot.ResetDirty();
    }


    /* Parameters for surface mapping */

    // Maximum number of iterations when mapping the mesh
    if (this->surfMapMaxItSlot.IsDirty()) {
        this->surfMapMaxIt = this->surfMapMaxItSlot.Param<core::param::IntParam>()->Value();
        this->surfMapMaxItSlot.ResetDirty();
    }

    // Interpolation method used when computing external forces
    if (this->surfMapInterpolModeSlot.IsDirty()) {
        this->surfMapInterpolMode = static_cast<DeformableGPUSurfaceMT::InterpolationMode>(
                this->surfMapInterpolModeSlot.Param<core::param::EnumParam>()->Value());
        this->surfMapInterpolModeSlot.ResetDirty();
    }

    // Stiffness of the springs defining the spring forces in the surface
    if (this->surfMapSpringStiffnessSlot.IsDirty()) {
        this->surfMapSpringStiffness = this->surfMapSpringStiffnessSlot.Param<core::param::FloatParam>()->Value();
        this->surfMapSpringStiffnessSlot.ResetDirty();
    }

    // Weighting of the external forces in surface #1, note that the weight
    // of the internal forces is implicitely defined by
    // 1.0 - surf0ExternalForcesWeight
    if (this->surfMapExternalForcesWeightSlot.IsDirty()) {
        this->surfMapExternalForcesWeight = this->surfMapExternalForcesWeightSlot.Param<core::param::FloatParam>()->Value();
        this->surfMapExternalForcesWeightSlot.ResetDirty();
    }

    // Weighting of the external forces in surface #1, note that the weight
    // of the internal forces is implicitely defined by
    // 1.0 - surf0ExternalForcesWeight
    if (this->surfRegExternalForcesWeightSlot.IsDirty()) {
        this->surfRegExternalForcesWeight = this->surfRegExternalForcesWeightSlot.Param<core::param::FloatParam>()->Value();
        this->surfRegExternalForcesWeightSlot.ResetDirty();
    }

    // Overall scaling for the forces acting upon surface #1
    if (this->surfMapForcesSclSlot.IsDirty()) {
        this->surfMapForcesScl = this->surfMapForcesSclSlot.Param<core::param::FloatParam>()->Value();
        this->surfMapForcesSclSlot.ResetDirty();
    }


    /* Surface regularization parameters */

    // Maximum number of iterations when regularizing the mesh
    if (this->surfRegMaxItSlot.IsDirty()) {
        this->surfRegMaxIt = this->surfRegMaxItSlot.Param<core::param::IntParam>()->Value();
        this->surfRegMaxItSlot.ResetDirty();
    }

    // Interpolation method used when computing external forces
    if (this->surfRegInterpolModeSlot.IsDirty()) {
        this->surfRegInterpolMode = static_cast<DeformableGPUSurfaceMT::InterpolationMode>(
                this->surfRegInterpolModeSlot.Param<core::param::EnumParam>()->Value());
        this->surfRegInterpolModeSlot.ResetDirty();
    }

    // Stiffness of the springs defining the spring forces in the surface
    if (this->surfRegSpringStiffnessSlot.IsDirty()) {
        this->surfRegSpringStiffness = this->surfRegSpringStiffnessSlot.Param<core::param::FloatParam>()->Value();
        this->surfRegSpringStiffnessSlot.ResetDirty();
    }

    // Weighting of the external forces in surface #1, note that the weight
    // of the internal forces is implicitely defined by
    // 1.0 - surf0ExternalForcesWeight
    if (this->surfRegExternalForcesWeightSlot.IsDirty()) {
        this->surfRegExternalForcesWeight = this->surfRegExternalForcesWeightSlot.Param<core::param::FloatParam>()->Value();
        this->surfRegExternalForcesWeightSlot.ResetDirty();
    }

    // Overall scaling for the forces acting upon surface #1
    if (this->surfRegForcesSclSlot.IsDirty()) {
        this->surfRegForcesScl = this->surfRegForcesSclSlot.Param<core::param::FloatParam>()->Value();
        this->surfRegForcesSclSlot.ResetDirty();
    }


    /* Quicksurf parameter */

    // Quicksurf radius scale
    if (this->qsRadSclSlot.IsDirty()) {
        this->qsRadScl = this->qsRadSclSlot.Param<core::param::FloatParam>()->Value();
        this->qsRadSclSlot.ResetDirty();
    }

    // Quicksurf grid scaling
    if (this->qsGridDeltaSlot.IsDirty()) {
        this->qsGridDelta = this->qsGridDeltaSlot.Param<core::param::FloatParam>()->Value();
        this->qsGridDeltaSlot.ResetDirty();
    }

    // Quicksurf isovalue
    if (this->qsIsoValSlot.IsDirty()) {
        this->qsIsoVal = this->qsIsoValSlot.Param<core::param::FloatParam>()->Value();
        this->qsIsoValSlot.ResetDirty();
    }
}
