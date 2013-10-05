//
// ComparativeMolSurfaceRenderer.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 16, 2013
// Author     : scharnkn
//

#include "stdafx.h"
#include "ComparativeMolSurfaceRenderer.h"

#ifdef WITH_CUDA

#define USE_TIMER

#include "VBODataCall.h"
#include "VTIDataCall.h"
#include "MolecularDataCall.h"
#include "RMS.h"
#include "ogl_error_check.h"
#include "cuda_error_check.h"
#include "DiffusionSolver.h"

#include "CoreInstance.h"
#include "view/AbstractCallRender3D.h"
#include "view/CallRender3D.h"
#include "param/EnumParam.h"
#include "param/FloatParam.h"
#include "param/IntParam.h"

#include <cmath>

using namespace megamol;
using namespace megamol::protein;

// Hardcoded parameters for 'quicksurf' class
const float ComparativeMolSurfaceRenderer::qsParticleRad = 0.8f;
const float ComparativeMolSurfaceRenderer::qsGaussLim = 10.0f;
const float ComparativeMolSurfaceRenderer::qsGridSpacing = 1.0f;
const bool ComparativeMolSurfaceRenderer::qsSclVanDerWaals = true;
const float ComparativeMolSurfaceRenderer::qsIsoVal = 0.5f;

// Hardcoded colors for surface rendering
const Vec3f ComparativeMolSurfaceRenderer::uniformColorSurf1 = Vec3f(0.36f, 0.36f, 1.0f);
const Vec3f ComparativeMolSurfaceRenderer::uniformColorSurf2 = Vec3f(0.8f, 0.8f, 0.0f);
const Vec3f ComparativeMolSurfaceRenderer::uniformColorSurfMapped = Vec3f(0.5f, 0.5f, 0.5f);
const Vec3f ComparativeMolSurfaceRenderer::colorMaxPotential = Vec3f(0.0f, 0.0f, 1.0f);
const Vec3f ComparativeMolSurfaceRenderer::colorMinPotential = Vec3f(1.0f, 0.0f, 0.0f);

// Maximum RMS value to enable valid mapping
const float ComparativeMolSurfaceRenderer::maxRMSVal = 10.0f;


/*
 * ComparativeMolSurfaceRenderer::applyRMSFitting
 */
bool ComparativeMolSurfaceRenderer::applyRMSFitting(
        MolecularDataCall *mol,
        DeformableGPUSurfaceMT *surf) {

    // Note: all particles have the same weight
    Vec3f centroid(0.0f, 0.0f, 0.0f);
    CudaDevArr<float> rotate_D;

    if (this->fittingMode != RMS_NONE) {

        // Compute centroid
        for (int cnt = 0; cnt < static_cast<int>(mol->AtomCount()); ++cnt) {

            centroid += Vec3f(mol->AtomPositions()[cnt*3],
                    mol->AtomPositions()[cnt*3+1],
                    mol->AtomPositions()[cnt*3+2]);
        }
        centroid /= static_cast<float>(mol->AtomCount());

        // Move center to origin
        float transVec[] = {-centroid.X(), -centroid.Y(), -centroid.Z()};
        if (!surf->Translate(transVec)) {
            return false;
        }

        // Rotate
        if (!surf->Rotate(this->rmsRotation.PeekComponents())) {
            return false;
        }

        // Translate to target center
        if (!surf->Translate(this->rmsTranslation.PeekComponents())) {
            return false;
        }
    }

    return true;
}


/*
 * ComparativeMolSurfaceRenderer::ComparativeMolSurfaceRenderer
 */
ComparativeMolSurfaceRenderer::ComparativeMolSurfaceRenderer(void) :
        Renderer3DModuleDS(),
        volOutputSlot("volOut", "Initial external forces representing the target shape"),
        molDataSlot1("molIn1", "Input molecule #1"),
        molDataSlot2("molIn2", "Input molecule #2"),
        volDataSlot1("volIn1", "Connects the rendering with data storage"),
        volDataSlot2("volIn2", "Connects the rendering with data storage"),
        rendererCallerSlot("protren", "Connects the renderer with another render module"),
        /* Parameters for frame by frame comparison */
        cmpModeSlot("cmpMode", "Param slot for compare mode"),
        singleFrame1Slot("singleFrame1", "Param for single frame #1"),
        singleFrame2Slot("singleFrame2", "Param for single frame #2"),
        /* Global mapping options */
        interpolModeSlot("interpolation", "Change interpolation method"),
        /* Parameters for mapped surface */
        fittingModeSlot("surfMap::RMSDfitting", "RMSD fitting for the mapped surface"),
        surfMappedExtForceSlot("surfMap::externalForces", "External forces for surface mapping"),
        surfaceMappingExternalForcesWeightSclSlot("surfMap::externalForcesScl","Scale factor for the external forces weighting"),
        surfaceMappingForcesSclSlot("surfMap::sclAll","Overall scale factor for all forces"),
        surfaceMappingMaxItSlot("surfMap::maxIt","Maximum number of iterations for the surface mapping"),
        surfMappedMinDisplSclSlot("surfMap::minDisplScl", "Minimum displacement for vertices"),
        surfMappedSpringStiffnessSlot("surfMap::stiffness", "Spring stiffness"),
        surfMappedGVFSclSlot("surfMap::GVFScl", "GVF scale factor"),
        surfMappedGVFItSlot("surfMap::GVFIt", "GVF iterations"),
        /* Surface regularization */
        regMaxItSlot("surfReg::regMaxIt", "Maximum number of iterations for regularization"),
        regSpringStiffnessSlot("surfReg::stiffness", "Spring stiffness"),
        regExternalForcesWeightSlot("surfReg::externalForcesWeight", "Weight of the external forces"),
        regForcesSclSlot("surfReg::forcesScl", "Scaling of overall force"),
        /* Surface rendering */
        surface1RMSlot("surf1::render", "Render mode for the first surface"),
        surface1ColorModeSlot("surf1::color", "Color mode for the first surface"),
        surf1AlphaSclSlot("surf1::alphaScl", "Transparency scale factor"),
        surface2RMSlot("surf2::render", "Render mode for the second surface"),
        surface2ColorModeSlot("surf2::color", "Render mode for the second surface"),
        surf2AlphaSclSlot("surf2::alphaScl", "Transparency scale factor"),
        surfaceMappedRMSlot("surfMap::render", "Render mode for the mapped surface"),
        surfaceMappedColorModeSlot("surfMap::color", "Color mode for the mapped surface"),
        surfMaxPosDiffSlot("surfMap::maxPosDiff", "Maximum value for euclidian distance"),
        surfMappedAlphaSclSlot("surfMap::alphaScl", "Transparency scale factor"),
        /* DEBUG external forces */
        filterXMaxParam("posFilter::XMax", "The maximum position in x-direction"),
        filterYMaxParam("posFilter::YMax", "The maximum position in y-direction"),
        filterZMaxParam("posFilter::ZMax", "The maximum position in z-direction"),
        filterXMinParam("posFilter::XMin", "The minimum position in x-direction"),
        filterYMinParam("posFilter::YMin", "The minimum position in y-direction"),
        filterZMinParam("posFilter::ZMin", "The minimum position in z-direction"),
        cudaqsurf1(NULL), cudaqsurf2(NULL),
        triggerComputeVolume(true), triggerInitPotentialTex(true),
        triggerComputeSurfacePoints1(true), triggerComputeSurfacePoints2(true),
        triggerSurfaceMapping(true), triggerRMSFit(true), triggerComputeLines(true) {

    /* Make data caller/callee slots available */

    // Molecular data input #1
    this->molDataSlot1.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataSlot1);

    // Molecular data input #2
    this->molDataSlot2.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataSlot2);

    // Data caller slot for the surface attribute texture #1
    this->volDataSlot1.SetCompatibleCall<VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->volDataSlot1);

    // Data caller slot for the surface attribute texture #2
    this->volDataSlot2.SetCompatibleCall<VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->volDataSlot2);

    // Renderer caller slot
    this->rendererCallerSlot.SetCompatibleCall<core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rendererCallerSlot);


    /* Parameters for frame-by-frame comparison */

    // Param slot for compare mode
    this->cmpMode = COMPARE_1_1;
    core::param::EnumParam *cmpm = new core::param::EnumParam(int(this->cmpMode));
    cmpm->SetTypePair(COMPARE_1_1, "1-1");
    cmpm->SetTypePair(COMPARE_1_N, "1-n");
    cmpm->SetTypePair(COMPARE_N_1, "n-1");
    cmpm->SetTypePair(COMPARE_N_N, "n-n");
    this->cmpModeSlot << cmpm;
    this->MakeSlotAvailable(&this->cmpModeSlot);

    // Param for single frame #1
    this->singleFrame1 = 0;
    this->singleFrame1Slot.SetParameter(new core::param::IntParam(this->singleFrame1, 0));
    this->MakeSlotAvailable(&this->singleFrame1Slot);

    // Param for single frame #2
    this->singleFrame2 = 0;
    this->singleFrame2Slot.SetParameter(new core::param::IntParam(this->singleFrame2, 0));
    this->MakeSlotAvailable(&this->singleFrame2Slot);


    /* Global mapping options */

    // Interpolation method used when computing external forces
    this->interpolMode = DeformableGPUSurfaceMT::INTERP_LINEAR;
    core::param::EnumParam *smi = new core::param::EnumParam(int(this->interpolMode));
    smi->SetTypePair(DeformableGPUSurfaceMT::INTERP_LINEAR, "Linear");
    smi->SetTypePair(DeformableGPUSurfaceMT::INTERP_CUBIC, "Cubic");
    this->interpolModeSlot << smi;
    this->MakeSlotAvailable(&this->interpolModeSlot);


    /* Parameters for mapped surface */

    // Parameter for the RMS fitting mode
    this->fittingMode = RMS_NONE;
    core::param::EnumParam *rms = new core::param::EnumParam(int(this->fittingMode));
    rms->SetTypePair(RMS_NONE, "None");
    rms->SetTypePair(RMS_ALL, "All");
    rms->SetTypePair(RMS_BACKBONE, "Backbone");
    rms->SetTypePair(RMS_C_ALPHA, "C alpha");
    this->fittingModeSlot << rms;
    this->MakeSlotAvailable(&this->fittingModeSlot);

    // Param slot for compare mode
    this->surfMappedExtForce = METABALLS;
    core::param::EnumParam *ext = new core::param::EnumParam(int(this->surfMappedExtForce));
    ext->SetTypePair(METABALLS, "Meta balls");
    ext->SetTypePair(METABALLS_DISTFIELD, "Meta balls + distance field");
    ext->SetTypePair(GVF, "GVF");
    this->surfMappedExtForceSlot << ext;
    this->MakeSlotAvailable(&this->surfMappedExtForceSlot);

    // Weighting for external forces when mapping the surface
    this->surfaceMappingExternalForcesWeightScl = 0.5f;
    this->surfaceMappingExternalForcesWeightSclSlot.SetParameter(
            new core::param::FloatParam(this->surfaceMappingExternalForcesWeightScl, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surfaceMappingExternalForcesWeightSclSlot);

    // Overall scaling of the resulting forces
    this->surfaceMappingForcesScl = 1.0f;
    this->surfaceMappingForcesSclSlot.SetParameter(new core::param::FloatParam(this->surfaceMappingForcesScl, 0.0f));
    this->MakeSlotAvailable(&this->surfaceMappingForcesSclSlot);

    // Maximum number of iterations when mapping the surface
    this->surfaceMappingMaxIt = 0;
    this->surfaceMappingMaxItSlot.SetParameter(new core::param::IntParam(this->surfaceMappingMaxIt, 0));
    this->MakeSlotAvailable(&this->surfaceMappingMaxItSlot);

    // Spring stiffness
    this->surfMappedSpringStiffness = 0.1;
    this->surfMappedSpringStiffnessSlot.SetParameter(new core::param::FloatParam(this->surfMappedSpringStiffness, 0.01f));
    this->MakeSlotAvailable(&this->surfMappedSpringStiffnessSlot);

    // Minimum displacement
    this->surfMappedMinDisplScl = 0.1;
    this->surfMappedMinDisplSclSlot.SetParameter(new core::param::FloatParam(this->surfMappedMinDisplScl, 0.0f));
    this->MakeSlotAvailable(&this->surfMappedMinDisplSclSlot);

    /// GVF scale factor
    this->surfMappedGVFScl = 1.0f;
    this->surfMappedGVFSclSlot.SetParameter(new core::param::FloatParam(this->surfMappedGVFScl, 0.0f));
    this->MakeSlotAvailable(&this->surfMappedGVFSclSlot);

    /// GVF iterations
    this->surfMappedGVFIt = 0;
    this->surfMappedGVFItSlot.SetParameter(new core::param::IntParam(this->surfMappedGVFIt, 0));
    this->MakeSlotAvailable(&this->surfMappedGVFItSlot);


    /* Parameters for surface regularization */

    // Maximum number of iterations when regularizing the mesh #0
    this->regMaxIt = 0;
    this->regMaxItSlot.SetParameter(new core::param::IntParam(this->regMaxIt, 0));
    this->MakeSlotAvailable(&this->regMaxItSlot);

    // Stiffness of the springs defining the spring forces in surface #0
    this->regSpringStiffness = 1.0f;
    this->regSpringStiffnessSlot.SetParameter(new core::param::FloatParam(this->regSpringStiffness, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->regSpringStiffnessSlot);

    // Weighting of the external forces in surface #0, note that the weight
    // of the internal forces is implicitely defined by
    // 1.0 - surf0ExternalForcesWeight
    this->regExternalForcesWeight = 0.0f;
    this->regExternalForcesWeightSlot.SetParameter(new core::param::FloatParam(this->regExternalForcesWeight, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->regExternalForcesWeightSlot);

    // Overall scaling for the forces acting upon surface #0
    this->regForcesScl = 1.0f;
    this->regForcesSclSlot.SetParameter(new core::param::FloatParam(this->regForcesScl, 0.0f));
    this->MakeSlotAvailable(&this->regForcesSclSlot);


    /* Rendering options of surface #1 and #2 */

    // Parameter for surface 1 render mode
    this->surface1RM = SURFACE_NONE;
    core::param::EnumParam *msrm1 = new core::param::EnumParam(int(this->surface1RM));
    msrm1->SetTypePair(SURFACE_NONE, "None");
    msrm1->SetTypePair(SURFACE_POINTS, "Points");
    msrm1->SetTypePair(SURFACE_WIREFRAME, "Wireframe");
    msrm1->SetTypePair(SURFACE_FILL, "Fill");
    this->surface1RMSlot << msrm1;
    this->MakeSlotAvailable(&this->surface1RMSlot);

    // Parameter for surface color 1 mode
    this->surface1ColorMode = SURFACE_UNI;
    core::param::EnumParam *mscm1 = new core::param::EnumParam(int(this->surface1ColorMode));
    mscm1->SetTypePair(SURFACE_UNI, "Uniform");
    mscm1->SetTypePair(SURFACE_NORMAL, "Normal");
    mscm1->SetTypePair(SURFACE_TEXCOORDS, "TexCoords");
    mscm1->SetTypePair(SURFACE_POTENTIAL, "Potential");
    this->surface1ColorModeSlot << mscm1;
    this->MakeSlotAvailable(&this->surface1ColorModeSlot);

    // Param for transparency scaling
    this->surf1AlphaScl = 1.0f;
    this->surf1AlphaSclSlot.SetParameter(new core::param::FloatParam(this->surf1AlphaScl, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surf1AlphaSclSlot);

    // Parameter for surface 2 render mode
    this->surface2RM = SURFACE_NONE;
    core::param::EnumParam *msrm2 = new core::param::EnumParam(int(this->surface2RM));
    msrm2->SetTypePair(SURFACE_NONE, "None");
    msrm2->SetTypePair(SURFACE_POINTS, "Points");
    msrm2->SetTypePair(SURFACE_WIREFRAME, "Wireframe");
    msrm2->SetTypePair(SURFACE_FILL, "Fill");
    this->surface2RMSlot << msrm2;
    this->MakeSlotAvailable(&this->surface2RMSlot);

    // Parameter for surface color 2 mode
    this->surface2ColorMode = SURFACE_UNI;
    core::param::EnumParam *mscm2 = new core::param::EnumParam(int(this->surface2ColorMode));
    mscm2->SetTypePair(SURFACE_UNI, "Uniform");
    mscm2->SetTypePair(SURFACE_NORMAL, "Normal");
    mscm2->SetTypePair(SURFACE_TEXCOORDS, "TexCoords");
    mscm2->SetTypePair(SURFACE_POTENTIAL, "Potential");
    this->surface2ColorModeSlot << mscm2;
    this->MakeSlotAvailable(&this->surface2ColorModeSlot);

    // Param for transparency scaling
    this->surf2AlphaScl = 1.0f;
    this->surf2AlphaSclSlot.SetParameter(new core::param::FloatParam(this->surf2AlphaScl, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surf2AlphaSclSlot);


    // Parameter for mapped surface render mode
    this->surfaceMappedRM = SURFACE_NONE;
    core::param::EnumParam *msrm = new core::param::EnumParam(int(this->surfaceMappedRM));
    msrm->SetTypePair(SURFACE_NONE, "None");
    msrm->SetTypePair(SURFACE_POINTS, "Points");
    msrm->SetTypePair(SURFACE_WIREFRAME, "Wireframe");
    msrm->SetTypePair(SURFACE_FILL, "Fill");
    this->surfaceMappedRMSlot << msrm;
    this->MakeSlotAvailable(&this->surfaceMappedRMSlot);

    // Parameter for mapped surface color mode
    this->surfaceMappedColorMode = SURFACE_UNI;
    core::param::EnumParam *mscm = new core::param::EnumParam(int(this->surfaceMappedColorMode));
    mscm->SetTypePair(SURFACE_UNI, "Uniform");
    mscm->SetTypePair(SURFACE_NORMAL, "Normal");
    mscm->SetTypePair(SURFACE_TEXCOORDS, "TexCoords");
    mscm->SetTypePair(SURFACE_DIST_TO_OLD_POS, "Dist to old pos");
    mscm->SetTypePair(SURFACE_POTENTIAL0, "Potential0");
    mscm->SetTypePair(SURFACE_POTENTIAL1, "Potential1");
    mscm->SetTypePair(SURFACE_POTENTIAL_DIFF, "Potential Diff");
    mscm->SetTypePair(SURFACE_POTENTIAL_SIGN, "Potential Sign");
    this->surfaceMappedColorModeSlot << mscm;
    this->MakeSlotAvailable(&this->surfaceMappedColorModeSlot);

    // Param for maximum value for euclidian distance // Not needed atm
    this->surfMaxPosDiff = 1.0f;
    this->surfMaxPosDiffSlot.SetParameter(new core::param::FloatParam(this->surfMaxPosDiff, 0.1f));
    this->MakeSlotAvailable(&this->surfMaxPosDiffSlot);

    // Param for transparency scaling
    this->surfMappedAlphaScl = 1.0f;
    this->surfMappedAlphaSclSlot.SetParameter(new core::param::FloatParam(this->surfMappedAlphaScl, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surfMappedAlphaSclSlot);

    /* DEBUG external forces */

    // Param for maximum x value
    this->posXMax = 10.0f;
    this->filterXMaxParam.SetParameter(new core::param::FloatParam(this->posXMax, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->filterXMaxParam);

    // Param for maximum y value
    this->posYMax = 10.0f;
    this->filterYMaxParam.SetParameter(new core::param::FloatParam(this->posYMax, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->filterYMaxParam);

    // Param for maximum z value
    this->posZMax = 10.0f;
    this->filterZMaxParam.SetParameter(new core::param::FloatParam(this->posZMax, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->filterZMaxParam);

    // Param for minimum x value
    this->posXMin = -10.0f;
    this->filterXMinParam.SetParameter(new core::param::FloatParam(this->posXMin, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->filterXMinParam);

    // Param for minimum y value
    this->posYMin = -10.0f;
    this->filterYMinParam.SetParameter(new core::param::FloatParam(this->posYMin, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->filterYMinParam);

    // Param for minimum z value
    this->posZMin = -10.0f;
    this->filterZMinParam.SetParameter(new core::param::FloatParam(this->posZMin, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->filterZMinParam);
}


/*
 * ComparativeMolSurfaceRenderer::~ComparativeMolSurfaceRenderer
 */
ComparativeMolSurfaceRenderer::~ComparativeMolSurfaceRenderer(void) {
    this->Release();
}


/*
 * ComparativeSurfacePotentialRenderer::computeVolumeTex
 */
bool ComparativeMolSurfaceRenderer::computeDensityMap(
        const MolecularDataCall *mol,
        CUDAQuickSurf *cqs,
        gridParams &gridDensMap,
        const Cubef &bboxParticles
        ) {

    using namespace vislib::sys;
    using namespace vislib::math;

    float gridXAxisLen, gridYAxisLen, gridZAxisLen;
    float padding;
    uint volSize;

    // (Re-)allocate memory for intermediate particle data
    this->gridDataPos.Validate(mol->AtomCount()*4);

    // Set particle radii and compute maximum particle radius
    this->maxAtomRad = 0.0f;
    this->minAtomRad = 100000.0f;

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
                            this->gridDataPos.Peek()[4*particleCnt+3] = mol->AtomTypes()[mol->AtomTypeIndices()[atomIdx]].Radius();
                        }
                        else {
                            this->gridDataPos.Peek()[4*particleCnt+3] = 1.0f;
                        }
                        this->maxAtomRad = std::max(this->maxAtomRad, this->gridDataPos.Peek()[4*particleCnt+3]);
                        this->minAtomRad = std::min(this->minAtomRad, this->gridDataPos.Peek()[4*particleCnt+3]);

                        particleCnt++;
                    }
                }
            }
        }
    }

    // Compute padding for the density map
    padding = this->maxAtomRad*this->qsParticleRad + this->qsGridSpacing*10; // TODO How much makes sense?

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
    volSize = gridDensMap.size[0]*gridDensMap.size[1]*gridDensMap.size[2];

    // Set particle positions
#pragma omp parallel for
    for (int cnt = 0; cnt < static_cast<int>(mol->AtomCount()); ++cnt) {
            this->gridDataPos.Peek()[4*cnt+0] -= gridDensMap.minC[0];
            this->gridDataPos.Peek()[4*cnt+1] -= gridDensMap.minC[1];
            this->gridDataPos.Peek()[4*cnt+2] -= gridDensMap.minC[2];
    }

//    printf("Grid dim %u %u %u, mol atom count %u, grid: %f, org %f %f %f\n",
//            gridDensMap.size[0], gridDensMap.size[1], gridDensMap.size[2],
//            mol->AtomCount(), this->gridDataPos.Peek()[0],
//            gridDensMap.minC[0], gridDensMap.minC[1], gridDensMap.minC[2]);

    // Compute uniform grid
    int rc = cqs->calc_map(
            mol->AtomCount(),
            &this->gridDataPos.Peek()[0],
            NULL,   // Pointer to 'color' array
            false,  // Do not use 'color' array
            (float*)&gridDensMap.minC,
            (int*)&gridDensMap.size,
            this->maxAtomRad,
            this->qsParticleRad, // Radius scaling
            this->qsGridSpacing,
            this->qsIsoVal,
            this->qsGaussLim);

    if (rc != 0) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Quicksurf class returned val != 0\n", this->ClassName());
        return false;
    }

//    this->externalPotentialBuff.Validate(volSize);
//    printf("volsize1 %i\n", volSize);
//    this->externalPotentialBuff.Set(0x00);
//    if (!CudaSafeCall(cudaMemcpy(this->externalPotentialBuff.Peek(),
//            cqs->getMap(), volSize*sizeof(float), cudaMemcpyDeviceToHost))) {
//        return false;
//    }

    return CheckForGLError();
}


/*
 * ComparativeMolSurfaceRenderer::create
 */
bool ComparativeMolSurfaceRenderer::create(void) {
    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    // Create quicksurf objects
    if (!this->cudaqsurf1) {
        this->cudaqsurf1 = new CUDAQuickSurf();
    }
    if (!this->cudaqsurf2) {
        this->cudaqsurf2 = new CUDAQuickSurf();
    }

    // Init extensions
    if (!glh_init_extensions("\
            GL_VERSION_2_0 \
            GL_EXT_texture3D \
            GL_EXT_framebuffer_object \
            GL_ARB_multitexture \
            GL_ARB_draw_buffers \
            GL_ARB_copy_buffer \
            GL_ARB_vertex_buffer_object")) {
        return false;
    }

    if (!DeformableGPUSurfaceMT::InitExtensions()) {
        return false;
    }

    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    // Load shader sources
    ShaderSource vertSrc, fragSrc, geomSrc;

    core::CoreInstance *ci = this->GetCoreInstance();
    if (!ci) {
        return false;
    }

    // Load shader for per pixel lighting of the surface
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("electrostatics::pplsurface::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    // Load ppl fragment shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("electrostatics::pplsurface::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    try {
        if(!this->pplSurfaceShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch(vislib::Exception &e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create the ppl shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Load shader for per pixel lighting of the surface
    if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("electrostatics::pplsurface::vertexMapped", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    // Load ppl fragment shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("electrostatics::pplsurface::fragmentMapped", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the ppl shader", this->ClassName());
        return false;
    }
    try {
        if(!this->pplMappedSurfaceShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch(vislib::Exception &e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create the ppl shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    return true;
}


/*
 * ComparativeSurfacePotentialRenderer::fitMoleculeRMS
 */
bool ComparativeMolSurfaceRenderer::fitMoleculeRMS(MolecularDataCall *mol0,
        MolecularDataCall *mol1) {
    using namespace vislib::sys;

    // Extract positions to be fitted
    vislib::Array<float> atomWeights;
    vislib::Array<int>  atomMask;
    uint posCnt=0;

//    printf("RMS frame idx %u and %u\n", mol0->FrameID(), mol1->FrameID());

    // No RMS fitting
    if (this->fittingMode == RMS_NONE) {
        this->rmsTranslation.Set(0.0f, 0.0f, 0.0f);
        this->rmsRotation.SetIdentity();
        this->rmsValue = 0.0f;
    } else {

        // Use all particles for RMS fitting
        if (this->fittingMode == RMS_ALL) {

            uint posCnt0=0, posCnt1=0;

            // (Re)-allocate memory if necessary
            this->rmsPosVec0.Validate(mol0->AtomCount()*3);
            this->rmsPosVec1.Validate(mol1->AtomCount()*3);

            // Extracting protein atoms from mol 0
            for (uint sec = 0; sec < mol0->SecondaryStructureCount(); ++sec) {
                for (uint acid = 0; acid < mol0->SecondaryStructures()[sec].AminoAcidCount(); ++acid) {
                    const MolecularDataCall::AminoAcid *aminoAcid =
                            dynamic_cast<const MolecularDataCall::AminoAcid*>(
                            (mol0->Residues()[mol0->SecondaryStructures()[sec].
                                 FirstAminoAcidIndex()+acid]));
                    if (aminoAcid == NULL) {
                        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                                "%s: Unable to perform RMSD fitting using all protein atoms (residue mislabeled as 'amino acid')", this->ClassName(),
        posCnt0, posCnt1);
                        return false;
                    }
                    for (uint at = 0; at < aminoAcid->AtomCount(); ++at) {
                        this->rmsPosVec0.Peek()[3*posCnt0+0] =
                                mol0->AtomPositions()[3*(aminoAcid->FirstAtomIndex()+at)+0];
                        this->rmsPosVec0.Peek()[3*posCnt0+1] =
                                mol0->AtomPositions()[3*(aminoAcid->FirstAtomIndex()+at)+1];
                        this->rmsPosVec0.Peek()[3*posCnt0+2] =
                                mol0->AtomPositions()[3*(aminoAcid->FirstAtomIndex()+at)+2];
                        posCnt0++;
                    }
                }
            }

            // Extracting protein atoms from mol 1
            for (uint sec = 0; sec < mol1->SecondaryStructureCount(); ++sec) {
                for (uint acid = 0; acid < mol1->SecondaryStructures()[sec].AminoAcidCount(); ++acid) {
                    const MolecularDataCall::AminoAcid *aminoAcid =
                            dynamic_cast<const MolecularDataCall::AminoAcid*>(
                            (mol1->Residues()[mol1->SecondaryStructures()[sec].
                                 FirstAminoAcidIndex()+acid]));
                    if (aminoAcid == NULL) {
                        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                                "%s: Unable to perform RMSD fitting using all protein atoms (residue mislabeled as 'amino acid')", this->ClassName(),
        posCnt0, posCnt1);
                        return false;
                    }
                    for (uint at = 0; at < aminoAcid->AtomCount(); ++at) {
                        this->rmsPosVec1.Peek()[3*posCnt1+0] =
                                mol1->AtomPositions()[3*(aminoAcid->FirstAtomIndex()+at)+0];
                        this->rmsPosVec1.Peek()[3*posCnt1+1] =
                                mol1->AtomPositions()[3*(aminoAcid->FirstAtomIndex()+at)+1];
                        this->rmsPosVec1.Peek()[3*posCnt1+2] =
                                mol1->AtomPositions()[3*(aminoAcid->FirstAtomIndex()+at)+2];
                        posCnt1++;
                    }
                }
            }

            if (posCnt0 != posCnt1) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: Unable to perform RMSD fitting using all protein atoms (non-equal atom count (%u vs. %u), try backbone instead)", this->ClassName(),
posCnt0, posCnt1);
                return false;
            }
            posCnt = posCnt0;

        } else if (this->fittingMode == RMS_BACKBONE) { // Use backbone atoms for RMS fitting
            // (Re)-allocate memory if necessary
            this->rmsPosVec0.Validate(mol0->AtomCount()*3);
            this->rmsPosVec1.Validate(mol1->AtomCount()*3);

            uint posCnt0=0, posCnt1=0;

            // Extracting backbone atoms from mol 0
            for (uint sec = 0; sec < mol0->SecondaryStructureCount(); ++sec) {

                for (uint acid = 0; acid < mol0->SecondaryStructures()[sec].AminoAcidCount(); ++acid) {

                    uint cAlphaIdx =
                            ((const MolecularDataCall::AminoAcid*)
                                (mol0->Residues()[mol0->SecondaryStructures()[sec].
                                     FirstAminoAcidIndex()+acid]))->CAlphaIndex();
                    uint cCarbIdx =
                            ((const MolecularDataCall::AminoAcid*)
                                (mol0->Residues()[mol0->SecondaryStructures()[sec].
                                     FirstAminoAcidIndex()+acid]))->CCarbIndex();
                    uint nIdx =
                            ((const MolecularDataCall::AminoAcid*)
                                (mol0->Residues()[mol0->SecondaryStructures()[sec].
                                     FirstAminoAcidIndex()+acid]))->NIndex();
                    uint oIdx =
                            ((const MolecularDataCall::AminoAcid*)
                                (mol0->Residues()[mol0->SecondaryStructures()[sec].
                                     FirstAminoAcidIndex()+acid]))->OIndex();

//                    printf("c alpha idx %u, cCarbIdx %u, o idx %u, n idx %u\n",
//                            cAlphaIdx, cCarbIdx, oIdx, nIdx); // DEBUG
                    this->rmsPosVec0.Peek()[3*posCnt0+0] = mol0->AtomPositions()[3*cAlphaIdx+0];
                    this->rmsPosVec0.Peek()[3*posCnt0+1] = mol0->AtomPositions()[3*cAlphaIdx+1];
                    this->rmsPosVec0.Peek()[3*posCnt0+2] = mol0->AtomPositions()[3*cAlphaIdx+2];
                    posCnt0++;
                    this->rmsPosVec0.Peek()[3*posCnt0+0] = mol0->AtomPositions()[3*cCarbIdx+0];
                    this->rmsPosVec0.Peek()[3*posCnt0+1] = mol0->AtomPositions()[3*cCarbIdx+1];
                    this->rmsPosVec0.Peek()[3*posCnt0+2] = mol0->AtomPositions()[3*cCarbIdx+2];
                    posCnt0++;
                    this->rmsPosVec0.Peek()[3*posCnt0+0] = mol0->AtomPositions()[3*oIdx+0];
                    this->rmsPosVec0.Peek()[3*posCnt0+1] = mol0->AtomPositions()[3*oIdx+1];
                    this->rmsPosVec0.Peek()[3*posCnt0+2] = mol0->AtomPositions()[3*oIdx+2];
                    posCnt0++;
                    this->rmsPosVec0.Peek()[3*posCnt0+0] = mol0->AtomPositions()[3*nIdx+0];
                    this->rmsPosVec0.Peek()[3*posCnt0+1] = mol0->AtomPositions()[3*nIdx+1];
                    this->rmsPosVec0.Peek()[3*posCnt0+2] = mol0->AtomPositions()[3*nIdx+2];
                    posCnt0++;
                }
            }

            // Extracting backbone atoms from mol 1
            for (uint sec = 0; sec < mol1->SecondaryStructureCount(); ++sec) {

                MolecularDataCall::SecStructure secStructure = mol1->SecondaryStructures()[sec];
                for (uint acid = 0; acid < secStructure.AminoAcidCount(); ++acid) {

                    uint cAlphaIdx =
                            ((const MolecularDataCall::AminoAcid*)
                                (mol1->Residues()[secStructure.
                                     FirstAminoAcidIndex()+acid]))->CAlphaIndex();
                    uint cCarbIdx =
                            ((const MolecularDataCall::AminoAcid*)
                                (mol1->Residues()[secStructure.
                                     FirstAminoAcidIndex()+acid]))->CCarbIndex();
                    uint nIdx =
                            ((const MolecularDataCall::AminoAcid*)
                                (mol1->Residues()[secStructure.
                                     FirstAminoAcidIndex()+acid]))->NIndex();
                    uint oIdx =
                            ((const MolecularDataCall::AminoAcid*)
                                (mol1->Residues()[secStructure.
                                     FirstAminoAcidIndex()+acid]))->OIndex();
//                    printf("amino acid idx %u, c alpha idx %u, cCarbIdx %u, o idx %u, n idx %u\n", secStructure.
//                            FirstAminoAcidIndex()+acid, cAlphaIdx, cCarbIdx, oIdx, nIdx);
                    this->rmsPosVec1.Peek()[3*posCnt1+0] = mol1->AtomPositions()[3*cAlphaIdx+0];
                    this->rmsPosVec1.Peek()[3*posCnt1+1] = mol1->AtomPositions()[3*cAlphaIdx+1];
                    this->rmsPosVec1.Peek()[3*posCnt1+2] = mol1->AtomPositions()[3*cAlphaIdx+2];
                    posCnt1++;
                    this->rmsPosVec1.Peek()[3*posCnt1+0] = mol1->AtomPositions()[3*cCarbIdx+0];
                    this->rmsPosVec1.Peek()[3*posCnt1+1] = mol1->AtomPositions()[3*cCarbIdx+1];
                    this->rmsPosVec1.Peek()[3*posCnt1+2] = mol1->AtomPositions()[3*cCarbIdx+2];
                    posCnt1++;
                    this->rmsPosVec1.Peek()[3*posCnt1+0] = mol1->AtomPositions()[3*oIdx+0];
                    this->rmsPosVec1.Peek()[3*posCnt1+1] = mol1->AtomPositions()[3*oIdx+1];
                    this->rmsPosVec1.Peek()[3*posCnt1+2] = mol1->AtomPositions()[3*oIdx+2];
                    posCnt1++;
                    this->rmsPosVec1.Peek()[3*posCnt1+0] = mol1->AtomPositions()[3*nIdx+0];
                    this->rmsPosVec1.Peek()[3*posCnt1+1] = mol1->AtomPositions()[3*nIdx+1];
                    this->rmsPosVec1.Peek()[3*posCnt1+2] = mol1->AtomPositions()[3*nIdx+2];
                    posCnt1++;
                }
            }

//            // DEBUG
//            printf("count0 = %u, count1 = %u\n", posCnt0, posCnt1);
//
//            // DEBUG
//            for (uint p = 0; p < posCnt0; p++) {
//                printf("%u: pos0 %f %f %f, pos1 %f %f %f\n", p,
//                        this->rmsPosVec0.Peek()[3*p+0],
//                        this->rmsPosVec0.Peek()[3*p+1],
//                        this->rmsPosVec0.Peek()[3*p+2],
//                        this->rmsPosVec1.Peek()[3*p+0],
//                        this->rmsPosVec1.Peek()[3*p+1],
//                        this->rmsPosVec1.Peek()[3*p+2]);
//            }

            if (posCnt0 != posCnt1) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: Unable to perform RMSD fitting using backbone \
atoms (non-equal atom count, %u vs. %u), try C alpha \
atoms instead.",
                        this->ClassName(), posCnt0, posCnt1);
                return false;
            }
            posCnt = posCnt0;
        } else if (this->fittingMode == RMS_C_ALPHA) { // Use C alpha atoms for RMS fitting
            // (Re)-allocate memory if necessary
            this->rmsPosVec0.Validate(mol0->AtomCount()*3);
            this->rmsPosVec1.Validate(mol1->AtomCount()*3);

            uint posCnt0=0, posCnt1=0;

            // Extracting C alpha atoms from mol 0
            for (uint sec = 0; sec < mol0->SecondaryStructureCount(); ++sec) {
                MolecularDataCall::SecStructure secStructure = mol0->SecondaryStructures()[sec];
                for (uint acid = 0; acid < secStructure.AminoAcidCount(); ++acid) {
                    uint cAlphaIdx =
                            ((const MolecularDataCall::AminoAcid*)
                                (mol0->Residues()[secStructure.
                                     FirstAminoAcidIndex()+acid]))->CAlphaIndex();
//                    printf("amino acid idx %u, c alpha idx %u\n", secStructure.
//                            FirstAminoAcidIndex()+acid, cAlphaIdx);
                    this->rmsPosVec0.Peek()[3*posCnt0+0] = mol0->AtomPositions()[3*cAlphaIdx+0];
                    this->rmsPosVec0.Peek()[3*posCnt0+1] = mol0->AtomPositions()[3*cAlphaIdx+1];
                    this->rmsPosVec0.Peek()[3*posCnt0+2] = mol0->AtomPositions()[3*cAlphaIdx+2];
                    posCnt0++;
                }
            }

            // Extracting C alpha atoms from mol 1
            for (uint sec = 0; sec < mol1->SecondaryStructureCount(); ++sec) {
                MolecularDataCall::SecStructure secStructure = mol1->SecondaryStructures()[sec];
                for (uint acid = 0; acid < secStructure.AminoAcidCount(); ++acid) {
                    uint cAlphaIdx =
                            ((const MolecularDataCall::AminoAcid*)
                                (mol1->Residues()[secStructure.
                                     FirstAminoAcidIndex()+acid]))->CAlphaIndex();
                    this->rmsPosVec1.Peek()[3*posCnt1+0] = mol1->AtomPositions()[3*cAlphaIdx+0];
                    this->rmsPosVec1.Peek()[3*posCnt1+1] = mol1->AtomPositions()[3*cAlphaIdx+1];
                    this->rmsPosVec1.Peek()[3*posCnt1+2] = mol1->AtomPositions()[3*cAlphaIdx+2];
                    posCnt1++;
                }
            }

            posCnt = std::min(posCnt0, posCnt1);
        }

        // Do actual RMSD calculations
        this->rmsMask.Validate(posCnt);
        this->rmsWeights.Validate(posCnt);
#pragma omp parallel for
        for (int a = 0; a < static_cast<int>(posCnt) ; ++a) {
            this->rmsMask.Peek()[a] = 1;
            this->rmsWeights.Peek()[a] = 1.0f;
        }

        float rotation[3][3], translation[3];

        this->rmsValue = CalculateRMS(
                posCnt,                     // Number of positions in each vector
                true,                       // Do fit positions
                2,                          // Save rotation/translation
                this->rmsWeights.Peek(),    // Weights for the particles
                this->rmsMask.Peek(),       // Which particles should be considered
                this->rmsPosVec1.Peek(),    // Vector to be fit
                this->rmsPosVec0.Peek(),    // Vector
                rotation,                   // Saves the rotation matrix
                translation                 // Saves the translation vector
        );

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

    // Check for sufficiently low rms value
    if (this->rmsValue > this->maxRMSVal) {
        return false;
    }

    return true;
}


/*
 * ComparativeMolSurfaceRenderer::GetCapabilities
 */
bool ComparativeMolSurfaceRenderer::GetCapabilities(core::Call& call) {
    core::view::AbstractCallRender3D *cr3d =
            dynamic_cast<core::view::AbstractCallRender3D *>(&call);

    if (cr3d == NULL) {
        return false;
    }

    cr3d->SetCapabilities(core::view::AbstractCallRender3D::CAP_RENDER |
                          core::view::AbstractCallRender3D::CAP_LIGHTING |
                          core::view::AbstractCallRender3D::CAP_ANIMATION);

    return true;
}


/*
 * ComparativeMolSurfaceRenderer::GetExtents
 */
bool ComparativeMolSurfaceRenderer::GetExtents(core::Call& call) {
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    // Get pointer to potential map data call
    protein::VTIDataCall *cmd0 =
            this->volDataSlot1.CallAs<protein::VTIDataCall>();
    if (cmd0 == NULL) {
        return false;
    }
    if (!(*cmd0)(VTIDataCall::CallForGetExtent)) {
        return false;
    }
    protein::VTIDataCall *cmd1 =
            this->volDataSlot2.CallAs<protein::VTIDataCall>();
    if (cmd1 == NULL) {
        return false;
    }
    if (!(*cmd1)(VTIDataCall::CallForGetExtent)) {
        return false;
    }

    // Get a pointer to particle data call
    MolecularDataCall *mol0 = this->molDataSlot1.CallAs<MolecularDataCall>();
    if (mol0 == NULL) {
        return false;
    }
    if (!(*mol0)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }
    MolecularDataCall *mol1 = this->molDataSlot2.CallAs<MolecularDataCall>();
    if (mol1 == NULL) {
        return false;
    }
    if (!(*mol1)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }

    // Get a pointer to the outgoing render call
    core::view::CallRender3D *ren = this->rendererCallerSlot.CallAs<core::view::CallRender3D>();
    if (ren != NULL) {
        if (!(*ren)(1)) {
            return false;
        }
    }

    this->bboxParticles1 = mol0->AccessBoundingBoxes();
    this->bboxParticles2 = mol1->AccessBoundingBoxes();
//    core::BoundingBoxes bboxPotential0 = cmd0->AccessBoundingBoxes();
    core::BoundingBoxes bboxPotential1 = cmd1->AccessBoundingBoxes();

    core::BoundingBoxes bbox_external;
    if (ren != NULL) {
        bbox_external = ren->AccessBoundingBoxes();
    }

    // Calc union of all bounding boxes
    vislib::math::Cuboid<float> bboxTmp;

    //bboxTmp = cmd0->AccessBoundingBoxes().ObjectSpaceBBox();
    //bboxTmp.Union(cmd1->AccessBoundingBoxes().ObjectSpaceBBox());
    bboxTmp = mol0->AccessBoundingBoxes().ObjectSpaceBBox();
    bboxTmp.Union(mol1->AccessBoundingBoxes().ObjectSpaceBBox());
    if (ren != NULL) {
        bboxTmp.Union(bbox_external.ObjectSpaceBBox());
    }
    this->bbox.SetObjectSpaceBBox(bboxTmp);

//    bboxTmp = cmd0->AccessBoundingBoxes().ObjectSpaceClipBox();
//    bboxTmp.Union(cmd1->AccessBoundingBoxes().ObjectSpaceClipBox());
    bboxTmp = mol0->AccessBoundingBoxes().ObjectSpaceClipBox();
    bboxTmp.Union(mol1->AccessBoundingBoxes().ObjectSpaceClipBox());
    if (ren != NULL) {
        bboxTmp.Union(bbox_external.ObjectSpaceClipBox());
    }
    this->bbox.SetObjectSpaceClipBox(bboxTmp);

//    bboxTmp = cmd0->AccessBoundingBoxes().WorldSpaceBBox();
//    bboxTmp.Union(cmd1->AccessBoundingBoxes().WorldSpaceBBox());
    bboxTmp = mol0->AccessBoundingBoxes().WorldSpaceBBox();
    bboxTmp.Union(mol1->AccessBoundingBoxes().WorldSpaceBBox());
    if (ren != NULL) {
        bboxTmp.Union(bbox_external.WorldSpaceBBox());
    }
    this->bbox.SetWorldSpaceBBox(bboxTmp);

//    bboxTmp = cmd0->AccessBoundingBoxes().WorldSpaceClipBox();
//    bboxTmp.Union(cmd1->AccessBoundingBoxes().WorldSpaceClipBox());
    bboxTmp = mol0->AccessBoundingBoxes().WorldSpaceClipBox();
    bboxTmp.Union(mol1->AccessBoundingBoxes().WorldSpaceClipBox());
    if (ren != NULL) {
        bboxTmp.Union(bbox_external.WorldSpaceClipBox());
    }
    this->bbox.SetWorldSpaceClipBox(bboxTmp);

    float scale;
    if(!vislib::math::IsEqual(this->bbox.ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / this->bbox.ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    this->bbox.MakeScaledWorld(scale);
    cr3d->AccessBoundingBoxes() = this->bbox;

    // The available frame count is determined by the 'compareFrames' parameter
    if (this->cmpMode == COMPARE_1_1) {
        // One by one frame comparison
        cr3d->SetTimeFramesCount(1);
    } else if (this->cmpMode == COMPARE_1_N) {
        // One frame of data set #0 is compared to all frames of data set #1
        cr3d->SetTimeFramesCount(std::min(cmd1->FrameCount(), mol1->FrameCount()));
    } else if (this->cmpMode == COMPARE_N_1) {
        // One frame of data set #1 is compared to all frames of data set #0
        cr3d->SetTimeFramesCount(std::min(cmd0->FrameCount(), mol0->FrameCount()));
    } else if (this->cmpMode == COMPARE_N_N) {
        // Frame by frame comparison
        // Note: The data set with more frames is truncated
        cr3d->SetTimeFramesCount(std::min(
                std::min(cmd0->FrameCount(), mol0->FrameCount()),
                std::min(cmd1->FrameCount(), mol1->FrameCount())));
    } else {
        return false; // Invalid compare mode
    }

//    printf("bbox %f %f %f %f %f %f\n",
//            this->bbox.ObjectSpaceBBox().Left(),
//            this->bbox.ObjectSpaceBBox().Bottom(),
//            this->bbox.ObjectSpaceBBox().Back(),
//            this->bbox.ObjectSpaceBBox().Right(),
//            this->bbox.ObjectSpaceBBox().Top(),
//            this->bbox.ObjectSpaceBBox().Front());

    return true;
}


/*
 * ComparativeSurfacePotentialRenderer::initPotentialMap
 */
bool ComparativeMolSurfaceRenderer::initPotentialMap(
        VTIDataCall *cmd,
        gridParams &gridPotentialMap,
        GLuint &potentialTex) {
    using namespace vislib::sys;

    // Setup grid parameters
    gridPotentialMap.minC[0] = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetLeft();
    gridPotentialMap.minC[1] = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetBottom();
    gridPotentialMap.minC[2] = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetBack();
    gridPotentialMap.maxC[0] = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetRight();
    gridPotentialMap.maxC[1] = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetTop();
    gridPotentialMap.maxC[2] = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetFront();
    gridPotentialMap.size[0] = cmd->GetGridsize().X();
    gridPotentialMap.size[1] = cmd->GetGridsize().Y();
    gridPotentialMap.size[2] = cmd->GetGridsize().Z();
    gridPotentialMap.delta[0] = cmd->GetSpacing().X();
    gridPotentialMap.delta[1] = cmd->GetSpacing().Y();
    gridPotentialMap.delta[2] = cmd->GetSpacing().Z();

//    printf("Init potential map, frame %u\n", cmd->FrameID());
//    printf("Min coord %f %f %f\n",
//            gridPotentialMap.minC[0], gridPotentialMap.minC[1], gridPotentialMap.minC[2]);
//    printf("Max coord %f %f %f\n",
//            gridPotentialMap.maxC[0], gridPotentialMap.maxC[1], gridPotentialMap.maxC[2]);
//    printf("Size %u %u %u\n", gridPotentialMap.size[0],
//            gridPotentialMap.size[1],
//            gridPotentialMap.size[2]);
//    printf("Delta %f %f %f\n", gridPotentialMap.delta[0],
//            gridPotentialMap.delta[1],
//            gridPotentialMap.delta[2]);

    //  Setup texture
    glEnable(GL_TEXTURE_3D);
    if (!glIsTexture(potentialTex)) {
        glGenTextures(1, &potentialTex);
    }
    glBindTexture(GL_TEXTURE_3D, potentialTex);
    glTexImage3DEXT(GL_TEXTURE_3D,
            0,
            GL_RGBA32F,
            cmd->GetGridsize().X(),
            cmd->GetGridsize().Y(),
            cmd->GetGridsize().Z(),
            0,
            GL_ALPHA,
            GL_FLOAT,
            cmd->GetPointDataByIdx(0, 0));
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

    glDisable(GL_TEXTURE_3D);

    return CheckForGLError();
}


/*
 * ComparativeMolSurfaceRenderer::release
 */
void ComparativeMolSurfaceRenderer::release(void) {
}


/*
 *  ComparativeMolSurfaceRenderer::Render
 */
bool ComparativeMolSurfaceRenderer::Render(core::Call& call) {
    using namespace vislib::sys;
    using namespace vislib::math;

#ifdef USE_TIMER
    time_t t;
#endif

    // Update parameters if necessary
    this->updateParams();

    // Get render call
    core::view::AbstractCallRender3D *cr3d =
            dynamic_cast<core::view::AbstractCallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    float calltime = cr3d->Time();
    int frameIdx1, frameIdx2;

    // Determine frame indices to be loaded based on 'compareFrames' parameter
    if (this->cmpMode == COMPARE_1_1) {
        // One by one frame comparison
        frameIdx1 = this->singleFrame1;
        frameIdx2 = this->singleFrame2;
    } else if (this->cmpMode == COMPARE_1_N) {
        // One frame of data set #0 is compared to all frames of data set #1
        frameIdx1 = this->singleFrame1;
        frameIdx2 = static_cast<int>(calltime);
    } else if (this->cmpMode == COMPARE_N_1) {
        // One frame of data set #1 is compared to all frames of data set #0
        frameIdx1 = static_cast<int>(calltime);
        frameIdx2 = this->singleFrame2;
    } else if (this->cmpMode == COMPARE_N_N) {
        // Frame by frame comparison
        // Note: The data set with more frames is truncated
        frameIdx1 = static_cast<int>(calltime);
        frameIdx2 = static_cast<int>(calltime);
    } else {
        return false; // Invalid compare mode
    }

#ifdef USE_TIMER
    t = clock();
#endif

    // Get surface texture of data set #1
    VTIDataCall *vti1 =
            this->volDataSlot1.CallAs<VTIDataCall>();
    if (vti1 == NULL) {
        return false;
    }
    vti1->SetCalltime(static_cast<float>(frameIdx1));  // Set call time
    vti1->SetFrameID(frameIdx1, true);  // Set frame ID and call data
    if (!(*vti1)(VTIDataCall::CallForGetData)) {
        return false;
    }

    VTIDataCall *vti2 =
            this->volDataSlot2.CallAs<VTIDataCall>();
    if (vti2 == NULL) {
        return false;
    }
    vti2->SetCalltime(static_cast<float>(frameIdx2));  // Set call time
    vti2->SetFrameID(frameIdx2, true);  // Set frame ID and call data
    if (!(*vti2)(VTIDataCall::CallForGetData)) {
        return false;
    }

    // Get the particle data calls
    MolecularDataCall *mol1 = this->molDataSlot1.CallAs<MolecularDataCall>();
    if (mol1 == NULL) {
        return false;
    }
    mol1->SetCalltime(static_cast<float>(frameIdx1));  // Set call time
    mol1->SetFrameID(frameIdx1, true);  // Set frame ID and call data
    if (!(*mol1)(MolecularDataCall::CallForGetData)) {
        return false;
    }

    MolecularDataCall *mol2 = this->molDataSlot2.CallAs<MolecularDataCall>();
    if (mol2 == NULL) {
        return false;
    }
    mol2->SetCalltime(static_cast<float>(frameIdx2));  // Set call time
    mol2->SetFrameID(frameIdx2, true);  // Set frame ID and call data
    if (!(*mol2)(MolecularDataCall::CallForGetData)) {
        return false;
    }

    // Init combined bounding boxes
    core::BoundingBoxes bboxPotential1 = vti1->AccessBoundingBoxes();
    core::BoundingBoxes bboxPotential2 = vti2->AccessBoundingBoxes();

    // Do RMS fitting if necessary
    if ((this->triggerRMSFit)
                || (mol1->DataHash() != this->datahashParticles1)
                || (mol2->DataHash() != this->datahashParticles2)
                || (calltime != this->calltimeOld)) {

        this->datahashParticles1 = mol1->DataHash();
        this->datahashParticles2 = mol2->DataHash();
        this->calltimeOld = calltime;

        if (!this->fitMoleculeRMS(mol2, mol1)) {
            return false;
        }
        this->triggerRMSFit = false;
        this->triggerComputeVolume = true;
#ifdef USE_TIMER
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "%s: Time for RMS fitting: %.6f s",
            this->ClassName(),
            (double(clock()-t)/double(CLOCKS_PER_SEC)));
#endif
    }

    // (Re-)compute volume texture if necessary
    if (this->triggerComputeVolume) {

        if (!this->computeDensityMap(
                mol1,
                (CUDAQuickSurf*)this->cudaqsurf1,
                this->gridDensMap1,
                this->bboxParticles1.ObjectSpaceBBox()
                )) {
            return false;
        }

        if (!this->computeDensityMap(
                mol2,
                (CUDAQuickSurf*)this->cudaqsurf2,
                this->gridDensMap2,
                this->bboxParticles2.ObjectSpaceBBox()
                )) {
            return false;
        }

        this->triggerComputeVolume = false;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }

    // (Re-)compute potential texture if necessary
    if ((this->triggerInitPotentialTex)
            ||(vti1->DataHash() != this->datahashPotential1)
            ||(vti2->DataHash() != this->datahashPotential2)
            ||(calltime != this->calltimeOld)) {

        this->datahashPotential1 = vti1->DataHash();
        this->datahashPotential2 = vti2->DataHash();
        if (!this->initPotentialMap(vti1, this->gridPotential1, this->surfAttribTex1)) {
            return false;
        }
        if (!this->initPotentialMap(vti2, this->gridPotential2, this->surfAttribTex2)) {
            return false;
        }
        this->triggerInitPotentialTex = false;
    }

    // (Re-)compute triangle mesh for surface #1 using Marching tetrahedra
    if (this->triggerComputeSurfacePoints1) {

        /* Surface #1 */

        // Get vertex positions based on the level set
        size_t volDim[3];
        volDim[0] = this->gridDensMap1.size[0];
        volDim[1] = this->gridDensMap1.size[1];
        volDim[2] = this->gridDensMap1.size[2];
        if (!this->deformSurf1.ComputeVertexPositions(
                ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                volDim,
                &this->gridDensMap1.minC[0],
                &this->gridDensMap1.delta[0],
                this->qsIsoVal)) {
            return false;
        }

        // Build triangle mesh from vertices
        if (!this->deformSurf1.ComputeTriangles(
                ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                                volDim,
                                &this->gridDensMap1.minC[0],
                                &this->gridDensMap1.delta[0],
                                this->qsIsoVal)) {
            return false;
        }

        // Compute vertex connectivity
        if (!this->deformSurf1.ComputeConnectivity(
                ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                volDim,
                &this->gridDensMap1.minC[0],
                &this->gridDensMap1.delta[0],
                this->qsIsoVal)) {
            return false;
        }

        // Regularize the mesh of surface #1
        if (!this->deformSurf1.MorphToVolume(
                ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                volDim,
                &this->gridDensMap1.minC[0],
                &this->gridDensMap1.delta[0],
                this->qsIsoVal,
                this->interpolMode,
                this->regMaxIt,
                0.0f, // minDisplScale // TODO ?
                this->regSpringStiffness,
                this->regForcesScl,
                this->regExternalForcesWeight)) {
            return false;
        }

        // Compute vertex normals
        if (!this->deformSurf1.ComputeNormals(
                ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                                volDim,
                                &this->gridDensMap1.minC[0],
                                &this->gridDensMap1.delta[0],
                                this->qsIsoVal)) {
            return false;
        }

        // Compute texture coordinates
        if (!this->deformSurf1.ComputeTexCoords(
                this->gridPotential1.minC,
                this->gridPotential1.maxC)) {
            return false;
        }

        this->triggerComputeSurfacePoints1 = false;
    }

    // (Re-)compute triangle mesh for surface #2 using Marching tetrahedra
    if (this->triggerComputeSurfacePoints2) {

        /* Surface #2 */

        // Get vertex positions based on the level set
        size_t volDim[3];
        volDim[0] = this->gridDensMap2.size[0];
        volDim[1] = this->gridDensMap2.size[1];
        volDim[2] = this->gridDensMap2.size[2];
        if (!this->deformSurf2.ComputeVertexPositions(
                ((CUDAQuickSurf*)this->cudaqsurf2)->getMap(),
                volDim,
                &this->gridDensMap2.minC[0],
                &this->gridDensMap2.delta[0],
                this->qsIsoVal)) {
            return false;
        }

        // Build triangle mesh from vertices
        if (!this->deformSurf2.ComputeTriangles(
                ((CUDAQuickSurf*)this->cudaqsurf2)->getMap(),
                                volDim,
                                &this->gridDensMap2.minC[0],
                                &this->gridDensMap2.delta[0],
                                this->qsIsoVal)) {
            return false;
        }

        // Compute vertex connectivity
        if (!this->deformSurf2.ComputeConnectivity(
                ((CUDAQuickSurf*)this->cudaqsurf2)->getMap(),
                volDim,
                &this->gridDensMap2.minC[0],
                &this->gridDensMap2.delta[0],
                this->qsIsoVal)) {
            return false;
        }

        // Regularize the mesh of surface #2
        if (!this->deformSurf2.MorphToVolume(
                ((CUDAQuickSurf*)this->cudaqsurf2)->getMap(),
                volDim,
                &this->gridDensMap2.minC[0],
                &this->gridDensMap2.delta[0],
                this->qsIsoVal,
                this->interpolMode,
                this->regMaxIt,
                0.0f, // minDisplScale // TODO ?
                this->regSpringStiffness,
                this->regForcesScl,
                this->regExternalForcesWeight)) {
            return false;
        }

        // Compute vertex normals
        if (!this->deformSurf2.ComputeNormals(
                ((CUDAQuickSurf*)this->cudaqsurf2)->getMap(),
                                volDim,
                                &this->gridDensMap2.minC[0],
                                &this->gridDensMap2.delta[0],
                                this->qsIsoVal)) {
            return false;
        }

        // Compute texture coordinates
        if (!this->deformSurf2.ComputeTexCoords(
                this->gridPotential2.minC,
                this->gridPotential2.maxC)) {
            return false;
        }

        this->triggerComputeSurfacePoints2 = false;
    }

    /* Map surface #2 to surface #1 */

    if (this->triggerSurfaceMapping) {

        // Make deep copy of regularized second surface
        this->deformSurfMapped = this->deformSurf2;

        // Transform vertices
        this->applyRMSFitting(mol2, &this->deformSurfMapped);

        size_t volDim1[3];
        volDim1[0] = static_cast<size_t>(this->gridDensMap1.size[0]);
        volDim1[1] = static_cast<size_t>(this->gridDensMap1.size[1]);
        volDim1[2] = static_cast<size_t>(this->gridDensMap1.size[2]);

        if (this->surfMappedExtForce == GVF) {

            // Morph surface #2 to shape #1 using GVF
            if (!this->deformSurfMapped.MorphToVolumeGVF(
                    ((CUDAQuickSurf*)this->cudaqsurf2)->getMap(),
                    ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                    this->deformSurf1.PeekCubeStates(),
                    volDim1,
                    &this->gridDensMap1.minC[0],
                    &this->gridDensMap1.delta[0],
                    this->qsIsoVal,
                    this->interpolMode,
                    this->surfaceMappingMaxIt,
                    this->surfMappedMinDisplScl,
                    this->surfMappedSpringStiffness,
                    this->surfaceMappingForcesScl,
                    this->surfaceMappingExternalForcesWeightScl,
                    this->surfMappedGVFScl,
                    this->surfMappedGVFIt)) {
                return false;
            }
        } else if (this->surfMappedExtForce == METABALLS) {
            // Morph surface #2 to shape #1 using implicit molecular surface
            if (!this->deformSurfMapped.MorphToVolume(
                    ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                    volDim1,
                    &this->gridDensMap1.minC[0],
                    &this->gridDensMap1.delta[0],
                    this->qsIsoVal,
                    this->interpolMode,
                    this->surfaceMappingMaxIt,
                    this->surfMappedMinDisplScl,
                    this->surfMappedSpringStiffness,
                    this->surfaceMappingForcesScl,
                    this->surfaceMappingExternalForcesWeightScl)) {
                return false;
            }
        } else if (this->surfMappedExtForce == METABALLS_DISTFIELD) {

        } else {
            return false;
        }

        size_t volDim2[3];
        volDim2[0] = static_cast<size_t>(this->gridDensMap2.size[0]);
        volDim2[1] = static_cast<size_t>(this->gridDensMap2.size[1]);
        volDim2[2] = static_cast<size_t>(this->gridDensMap2.size[2]);

        // Compute vertex normals
        if (!this->deformSurfMapped.ComputeNormals(
                ((CUDAQuickSurf*)this->cudaqsurf2)->getMap(),
                                volDim2,
                                &this->gridDensMap2.minC[0],
                                &this->gridDensMap2.delta[0],
                                this->qsIsoVal)) {
            return false;
        }

        // Compute texture coordinates
        if (!this->deformSurfMapped.ComputeTexCoords(
                this->gridPotential1.minC,
                this->gridPotential1.maxC)) {
            return false;
        }

        this->triggerSurfaceMapping = false;
    }

    // Get camera information
    this->cameraInfo =  dynamic_cast<core::view::AbstractCallRender3D*>(&call)->GetCameraParameters();

    /* Rendering of scene objects */

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    // Apply scaling based on combined bounding box
    float scaleCombined;
    if (!vislib::math::IsEqual(this->bbox.ObjectSpaceBBox().LongestEdge(), 0.0f)) {
        scaleCombined = 2.0f/this->bbox.ObjectSpaceBBox().LongestEdge();
    } else {
        scaleCombined = 1.0f;
    }
    glScalef(scaleCombined, scaleCombined, scaleCombined);
    //printf("scale master %f\n", scaleCombined);

    // Call external renderer if possible
    core::view::CallRender3D *ren = this->rendererCallerSlot.CallAs<core::view::CallRender3D>();
    if (ren != NULL) {
        // Call additional renderer
        ren->SetCameraParameters(this->cameraInfo);
        ren->SetTime(cr3d->Time());
        glPushMatrix();
        // Revert scaling done by external renderer in advance
        float scaleRevert;
        if (!vislib::math::IsEqual(ren->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
            scaleRevert = 2.0f/ren->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        } else {
            scaleRevert = 1.0f;
        }
        scaleRevert = 1.0f/scaleRevert;
        glScalef(scaleRevert, scaleRevert, scaleRevert);
        (*ren)(0); // Render call to external renderer
        glPopMatrix();
        CheckForGLError();
    }

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    //glLineWidth(2.0f);
    //glPointSize(2.0f);

    // Calculate cam pos using last column of inverse modelview matrix
    float camPos[3];
    GLfloat m[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, m);
    Mat4f modelMatrix(&m[0]);
    modelMatrix.Invert();
    camPos[0] = modelMatrix.GetAt(0, 3);
    camPos[1] = modelMatrix.GetAt(1, 3);
    camPos[2] = modelMatrix.GetAt(2, 3);

//    printf("volsize 1 %u %u %u (%u), volsize 2 %u %u %u\n",
//            this->gridDensMap1.size[0], this->gridDensMap1.size[1],
//            this->gridDensMap1.size[2],
//            this->gridDensMap1.size[0]*this->gridDensMap1.size[1]*this->gridDensMap1.size[2],
//            this->gridDensMap2.size[0],
//            this->gridDensMap2.size[1], this->gridDensMap2.size[2]);


    // DEBUG Render external forces as lines
    if (!this->renderExternalForces()) {
        return false;
    }
    // END DEBUG


    if (this->surface1RM != SURFACE_NONE) {

        // Sort triangles by camera distance
        if (!this->deformSurf1.SortTrianglesByCamDist(camPos)) {
            return false;
        }

        // Render surface #1
        if (!this->renderSurface(
                this->deformSurf1.GetVtxDataVBO(),
                this->deformSurf1.GetVertexCnt(),
                this->deformSurf1.GetTriangleIdxVBO(),
                this->deformSurf1.GetTriangleCnt()*3,
                this->surface1RM,
                this->surface1ColorMode,
                this->surfAttribTex1,
                this->uniformColorSurf1,
                this->surf1AlphaScl)) {
            return false;
        }
    }

    if (this->surface2RM != SURFACE_NONE) {
        // Sort triangles by camera distance
        if (!this->deformSurf2.SortTrianglesByCamDist(camPos)) {
            return false;
        }

        // Render surface #2
        if (!this->renderSurface(
                this->deformSurf2.GetVtxDataVBO(),
                this->deformSurf2.GetVertexCnt(),
                this->deformSurf2.GetTriangleIdxVBO(),
                this->deformSurf2.GetTriangleCnt()*3,
                this->surface2RM,
                this->surface2ColorMode,
                this->surfAttribTex2,
                this->uniformColorSurf2,
                this->surf2AlphaScl)) {
            return false;
        }
    }

    if (this->surfaceMappedRM != SURFACE_NONE) {

        // Sort triangles by camera distance
        if (!this->deformSurfMapped.SortTrianglesByCamDist(camPos)) {
            return false;
        }

        // Render mapped surface
        if (!this->renderMappedSurface(
                this->deformSurf2.GetVtxDataVBO(),
                this->deformSurfMapped.GetVtxDataVBO(),
                this->deformSurfMapped.GetVertexCnt(),
                this->deformSurfMapped.GetTriangleIdxVBO(),
                this->deformSurfMapped.GetTriangleCnt()*3,
                this->surfaceMappedRM,
                this->surfaceMappedColorMode)) {
            return false;
        }
    }

    glDisable(GL_TEXTURE_3D);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);

    glPopMatrix();

    // Unlock frames
    mol1->Unlock();
    mol2->Unlock();
    vti1->Unlock();
    vti2->Unlock();

    return CheckForGLError();
}


/*
 * ComparativeMolSurfaceRenderer::renderExternalForces
 */
bool ComparativeMolSurfaceRenderer::renderExternalForces() {

    using namespace vislib::math;

    size_t gridSize = this->gridDensMap1.size[0]*this->gridDensMap1.size[1]*this->gridDensMap1.size[2];
    if (this->triggerComputeLines) {

        this->gvf.Validate(gridSize*4);
        if (this->surfMappedExtForce == GVF) {
            if (!CudaSafeCall(cudaMemcpy(gvf.Peek(),
                    this->deformSurfMapped.PeekGVF(), sizeof(float)*gridSize*4,
                    cudaMemcpyDeviceToHost))) {
                return false;
            }
        } else if (this->surfMappedExtForce == METABALLS) {
            if (!CudaSafeCall(cudaMemcpy(gvf.Peek(),
                    this->deformSurfMapped.PeekVolGradient(), sizeof(float)*gridSize*4,
                    cudaMemcpyDeviceToHost))) {
                return false;
            }
        } else if (this->surfMappedExtForce == METABALLS_DISTFIELD) {
            if (!CudaSafeCall(cudaMemcpy(gvf.Peek(),
                    this->deformSurfMapped.PeekVolGradient(), sizeof(float)*gridSize*4,
                    cudaMemcpyDeviceToHost))) {
                return false;
            }
        } else {
            return false;
        }

        this->lines.SetCount(gridSize*6);
        this->lineColors.SetCount(gridSize*6);
        for (size_t x = 0; x < this->gridDensMap1.size[0]; ++x) {
            for (size_t y = 0; y < this->gridDensMap1.size[1]; ++y) {
                for (size_t z = 0; z < this->gridDensMap1.size[2]; ++z) {
                    unsigned int idx = this->gridDensMap1.size[0]*(this->gridDensMap1.size[1]*z + y) + x;

                    Vec3f pos(this->gridDensMap1.minC[0] + x*this->gridDensMap1.delta[0],
                            this->gridDensMap1.minC[1] + y*this->gridDensMap1.delta[1],
                            this->gridDensMap1.minC[2] + z*this->gridDensMap1.delta[2]);

                    Vec3f vec(gvf.Peek()[4*idx+0], gvf.Peek()[4*idx+1], gvf.Peek()[4*idx+2]);

                    if ((pos.X() <= this->posXMax)&&(pos.X() >= this->posXMin)
                      &&(pos.Y() <= this->posYMax)&&(pos.Y() >= this->posYMin)
                      &&(pos.Z() <= this->posZMax)&&(pos.Z() >= this->posZMin)) {

                        this->lines[6*idx+0] = pos.X() - vec.X()*0.5f;
                        this->lines[6*idx+1] = pos.Y() - vec.Y()*0.5f;
                        this->lines[6*idx+2] = pos.Z() - vec.Z()*0.5f;
                        this->lines[6*idx+3] = pos.X() + vec.X()*0.5f;
                        this->lines[6*idx+4] = pos.Y() + vec.Y()*0.5f;
                        this->lines[6*idx+5] = pos.Z() + vec.Z()*0.5f;
                    } else {
                        this->lines[6*idx+0] = 0.0f;
                        this->lines[6*idx+1] = 0.0f;
                        this->lines[6*idx+2] = 0.0f;
                        this->lines[6*idx+3] = 0.0f;
                        this->lines[6*idx+4] = 0.0f;
                        this->lines[6*idx+5] = 0.0f;
                    }

                    /// Color mode for south/north
                    this->lineColors[6*idx+0] = 0.0f;
                    this->lineColors[6*idx+1] = 0.0f;
                    this->lineColors[6*idx+2] = 1.0f;
                    this->lineColors[6*idx+3] = 1.0f;
                    this->lineColors[6*idx+4] = 1.0f;
                    this->lineColors[6*idx+5] = 0.0f;
                }
            }
        }
        this->triggerComputeLines = false;
    }

    // Draw lines
    glDisable(GL_LINE_SMOOTH);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, this->lines.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, this->lineColors.PeekElements());

    glLineWidth(4.0);
    glDrawArrays(GL_LINES, 0, gridSize*2);
    // deactivate vertex arrays after drawing
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glLineWidth(1.0);

    return CheckForGLError();
}


/*
 * ComparativeMolSurfaceRenderer::::renderSurface
 */
bool ComparativeMolSurfaceRenderer::renderSurface(
        GLuint vbo,
        uint vertexCnt,
        GLuint vboTriangleIdx,
        uint triangleVertexCnt,
        SurfaceRenderMode renderMode,
        SurfaceColorMode colorMode,
        GLuint potentialTex,
        Vec3f uniformColor,
        float alphaScl) {

    GLint attribLocPos, attribLocNormal, attribLocTexCoord;


    /* Get vertex attributes from vbo */

    glBindBufferARB(GL_ARRAY_BUFFER, vbo);
    CheckForGLError(); // OpenGL error check

    this->pplSurfaceShader.Enable();
    CheckForGLError(); // OpenGL error check

    // Note: glGetAttribLocation returnes -1 if the attribute if not used in
    // the shader code, because in this case the attribute is optimized out by
    // the compiler
    attribLocPos = glGetAttribLocationARB(this->pplSurfaceShader.ProgramHandle(), "pos");
    attribLocNormal = glGetAttribLocationARB(this->pplSurfaceShader.ProgramHandle(), "normal");
    attribLocTexCoord = glGetAttribLocationARB(this->pplSurfaceShader.ProgramHandle(), "texCoord");
    CheckForGLError(); // OpenGL error check

    glEnableVertexAttribArrayARB(attribLocPos);
    glEnableVertexAttribArrayARB(attribLocNormal);
    glEnableVertexAttribArrayARB(attribLocTexCoord);
    CheckForGLError(); // OpenGL error check

    glVertexAttribPointerARB(attribLocPos, 3, GL_FLOAT, GL_FALSE,
            DeformableGPUSurfaceMT::vertexDataStride*sizeof(float),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsPos*sizeof(float)));
    glVertexAttribPointerARB(attribLocNormal, 3, GL_FLOAT, GL_FALSE,
            DeformableGPUSurfaceMT::vertexDataStride*sizeof(float),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsNormal*sizeof(float)));
    glVertexAttribPointerARB(attribLocTexCoord, 3, GL_FLOAT, GL_FALSE,
            DeformableGPUSurfaceMT::vertexDataStride*sizeof(float),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsTexCoord*sizeof(float)));
    CheckForGLError(); // OpenGL error check


    /* Render */

    // Set uniform vars
    glUniform1iARB(this->pplSurfaceShader.ParameterLocation("potentialTex"), 0);
    glUniform1iARB(this->pplSurfaceShader.ParameterLocation("colorMode"), static_cast<int>(colorMode));
    glUniform1iARB(this->pplSurfaceShader.ParameterLocation("renderMode"), static_cast<int>(renderMode));
    glUniform3fvARB(this->pplSurfaceShader.ParameterLocation("colorMin"), 1, this->colorMinPotential.PeekComponents());
    glUniform3fvARB(this->pplSurfaceShader.ParameterLocation("colorMax"), 1, this->colorMaxPotential.PeekComponents());
    glUniform3fvARB(this->pplSurfaceShader.ParameterLocation("colorUniform"), 1, uniformColor.PeekComponents());
    glUniform1fARB(this->pplSurfaceShader.ParameterLocation("minPotential"), -50); // TODO param
    glUniform1fARB(this->pplSurfaceShader.ParameterLocation("maxPotential"), 50); // TODO param
    glUniform1fARB(this->pplSurfaceShader.ParameterLocation("alphaScl"), alphaScl);

    glActiveTextureARB(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, potentialTex);
    CheckForGLError(); // OpenGL error check

    if (renderMode == SURFACE_FILL) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    } else if (renderMode == SURFACE_WIREFRAME) {
        glCullFace(GL_BACK);
        glEnable(GL_CULL_FACE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else if (renderMode == SURFACE_POINTS){
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
    }

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, vboTriangleIdx);
    CheckForGLError(); // OpenGL error check

    glDrawElements(GL_TRIANGLES,
            triangleVertexCnt,
            GL_UNSIGNED_INT,
            reinterpret_cast<void*>(0));

//    glDrawArrays(GL_POINTS, 0, 3*vertexCnt); // DEBUG
    glDisable(GL_CULL_FACE);

    this->pplSurfaceShader.Disable();

    glDisableVertexAttribArrayARB(attribLocPos);
    glDisableVertexAttribArrayARB(attribLocNormal);
    glDisableVertexAttribArrayARB(attribLocTexCoord);
    CheckForGLError(); // OpenGL error check

    // Switch back to normal pointer operation by binding with 0
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    return CheckForGLError(); // OpenGL error check
}


/*
 * ComparativeMolSurfaceRenderer::::renderMappedSurface
 */
bool ComparativeMolSurfaceRenderer::renderMappedSurface(
        GLuint vboOld, GLuint vboNew,
        uint vertexCnt,
        GLuint vboTriangleIdx,
        uint triangleVertexCnt,
        SurfaceRenderMode renderMode,
        SurfaceColorMode colorMode) {

    GLint attribLocPosNew, attribLocPosOld;
    GLint attribLocNormal, attribLocCorruptTriangleFlag;
    GLint attribLocTexCoordNew, attribLocTexCoordOld;

    this->pplMappedSurfaceShader.Enable();
    CheckForGLError(); // OpenGL error check

    // Note: glGetAttribLocation returns -1 if the attribute if not used in
    // the shader code, because in this case the attribute is optimized out by
    // the compiler
    attribLocPosNew = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "posNew");
    attribLocPosOld = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "posOld");
    attribLocNormal = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "normal");
//    attribLocCorruptTriangleFlag = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "corruptTriangleFlag");
    attribLocTexCoordNew = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "texCoordNew");
    attribLocTexCoordOld = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "texCoordOld");
    CheckForGLError(); // OpenGL error check

    glEnableVertexAttribArrayARB(attribLocPosNew);
    glEnableVertexAttribArrayARB(attribLocPosOld);
    glEnableVertexAttribArrayARB(attribLocNormal);
//    glEnableVertexAttribArrayARB(attribLocCorruptTriangleFlag);
    glEnableVertexAttribArrayARB(attribLocTexCoordNew);
    glEnableVertexAttribArrayARB(attribLocTexCoordOld);
    CheckForGLError(); // OpenGL error check


    /* Get vertex attributes from vbos */

    // Attributes from new (mapped) surface

    glBindBufferARB(GL_ARRAY_BUFFER, vboNew);
    CheckForGLError(); // OpenGL error check

    glVertexAttribPointerARB(attribLocPosNew, 3, GL_FLOAT, GL_FALSE,
            DeformableGPUSurfaceMT::vertexDataStride*sizeof(float),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsPos*sizeof(float)));

    glVertexAttribPointerARB(attribLocTexCoordNew, 3, GL_FLOAT, GL_FALSE,
            DeformableGPUSurfaceMT::vertexDataStride*sizeof(float),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsTexCoord*sizeof(float)));

    glVertexAttribPointerARB(attribLocNormal, 3, GL_FLOAT, GL_FALSE,
            DeformableGPUSurfaceMT::vertexDataStride*sizeof(float),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsNormal*sizeof(float)));

    glBindBufferARB(GL_ARRAY_BUFFER, vboOld);
    CheckForGLError(); // OpenGL error check

    glVertexAttribPointerARB(attribLocPosOld, 3, GL_FLOAT, GL_FALSE,
            DeformableGPUSurfaceMT::vertexDataStride*sizeof(float),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsPos*sizeof(float)));

    glVertexAttribPointerARB(attribLocTexCoordOld, 3, GL_FLOAT, GL_FALSE,
            DeformableGPUSurfaceMT::vertexDataStride*sizeof(float),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsTexCoord*sizeof(float)));

    CheckForGLError(); // OpenGL error check


    /* Render */

    // Set uniform vars
    glUniform1iARB(this->pplMappedSurfaceShader.ParameterLocation("potentialTex0"), 0);
    glUniform1iARB(this->pplMappedSurfaceShader.ParameterLocation("potentialTex1"), 1);
    glUniform1iARB(this->pplMappedSurfaceShader.ParameterLocation("colorMode"), static_cast<int>(colorMode));
    glUniform1iARB(this->pplMappedSurfaceShader.ParameterLocation("renderMode"), static_cast<int>(renderMode));
    glUniform3fvARB(this->pplMappedSurfaceShader.ParameterLocation("colorMin"), 1, this->colorMinPotential.PeekComponents());
    glUniform3fvARB(this->pplMappedSurfaceShader.ParameterLocation("colorMax"), 1, this->colorMaxPotential.PeekComponents());
    glUniform3fvARB(this->pplMappedSurfaceShader.ParameterLocation("colorUniform"), 1, this->uniformColorSurfMapped.PeekComponents());
    glUniform1fARB(this->pplMappedSurfaceShader.ParameterLocation("minPotential"), -70);
    glUniform1fARB(this->pplMappedSurfaceShader.ParameterLocation("maxPotential"), 70);
    glUniform1fARB(this->pplMappedSurfaceShader.ParameterLocation("alphaScl"), this->surfMappedAlphaScl);
    glUniform1fARB(this->pplMappedSurfaceShader.ParameterLocation("maxPosDiff"), this->surfMaxPosDiff);

    glActiveTextureARB(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, this->surfAttribTex2);

    glActiveTextureARB(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, this->surfAttribTex1);
    CheckForGLError(); // OpenGL error check


    if (renderMode == SURFACE_FILL) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    } else if (renderMode == SURFACE_WIREFRAME) {
        glCullFace(GL_BACK);
        glEnable(GL_CULL_FACE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else if (renderMode == SURFACE_POINTS){
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
    }

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, vboTriangleIdx);
    CheckForGLError(); // OpenGL error check

    glDrawElements(GL_TRIANGLES,
            triangleVertexCnt,
            GL_UNSIGNED_INT,
            reinterpret_cast<void*>(0));

//    glDrawArrays(GL_POINTS, 0, 3*vertexCnt); // DEBUG

    this->pplMappedSurfaceShader.Disable();

    glDisableVertexAttribArrayARB(attribLocPosNew);
    glDisableVertexAttribArrayARB(attribLocPosOld);
    glDisableVertexAttribArrayARB(attribLocNormal);
//    glDisableVertexAttribArrayARB(attribLocCorruptTriangleFlag);
    glDisableVertexAttribArrayARB(attribLocTexCoordNew);
    glDisableVertexAttribArrayARB(attribLocTexCoordOld);
    CheckForGLError(); // OpenGL error check

    // Switch back to normal pointer operation by binding with 0
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    glDisable(GL_CULL_FACE);

    return CheckForGLError(); // OpenGL error check
}


/*
 * ComparativeMolSurfaceRenderer::updateParams
 */
void ComparativeMolSurfaceRenderer::updateParams() {
    /* Parameter for frame by frame comparison */

    // Param slot for compare mode
    if (this->cmpModeSlot.IsDirty()) {
        this->cmpMode = static_cast<CompareMode>(
                this->cmpModeSlot.Param<core::param::EnumParam>()->Value());
        this->cmpModeSlot.ResetDirty();
        this->triggerRMSFit = true;
        this->triggerComputeVolume = true;
        this->triggerInitPotentialTex = true;
    }

    // Param for single frame #1
    if (this->singleFrame1Slot.IsDirty()) {
        this->singleFrame1 = this->singleFrame1Slot.Param<core::param::IntParam>()->Value();
        this->singleFrame1Slot.ResetDirty();
        this->triggerRMSFit = true;
        this->triggerComputeVolume = true;
        this->triggerInitPotentialTex = true;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerSurfaceMapping = true;
    }

    // Param for single frame #2
    if (this->singleFrame2Slot.IsDirty()) {
        this->singleFrame2 = this->singleFrame2Slot.Param<core::param::IntParam>()->Value();
        this->singleFrame2Slot.ResetDirty();
        this->triggerRMSFit = true;
        this->triggerComputeVolume = true;
        this->triggerInitPotentialTex = true;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerSurfaceMapping = true;
    }


    /* Global mapping options */

    // Interpolation method used when computing external forces
    if (this->interpolModeSlot.IsDirty()) {
        this->interpolMode = static_cast<DeformableGPUSurfaceMT::InterpolationMode>(
                this->interpolModeSlot.Param<core::param::EnumParam>()->Value());
        this->interpolModeSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }


    /* Parameters for the mapped surface */

    // Parameter for the RMS fitting mode
    if (this->fittingModeSlot.IsDirty()) {
        this->fittingMode = static_cast<RMSFittingMode>(
                this->fittingModeSlot.Param<core::param::EnumParam>()->Value());
        this->fittingModeSlot.ResetDirty();
        this->triggerRMSFit = true;
        this->triggerSurfaceMapping = true;
    }

    // Parameter for the external forces
    if (this->surfMappedExtForceSlot.IsDirty()) {
        this->surfMappedExtForce = static_cast<ExternalForces>(
                this->surfMappedExtForceSlot.Param<core::param::EnumParam>()->Value());
        this->surfMappedExtForceSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
        this->triggerComputeLines = true;
    }

    // Weighting for external forces when mapping the surface
    if (this->surfaceMappingExternalForcesWeightSclSlot.IsDirty()) {
        this->surfaceMappingExternalForcesWeightScl =
                this->surfaceMappingExternalForcesWeightSclSlot.Param<core::param::FloatParam>()->Value();
        this->surfaceMappingExternalForcesWeightSclSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
    }

    // Overall scaling of the resulting forces
    if (this->surfaceMappingForcesSclSlot.IsDirty()) {
        this->surfaceMappingForcesScl = this->surfaceMappingForcesSclSlot.Param<core::param::FloatParam>()->Value();
        this->surfaceMappingForcesSclSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
    }

    // Maximum number of iterations when mapping the surface
    if (this->surfaceMappingMaxItSlot.IsDirty()) {
        this->surfaceMappingMaxIt = this->surfaceMappingMaxItSlot.Param<core::param::IntParam>()->Value();
        this->surfaceMappingMaxItSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
    }

    // Param for minimum vertex displacement
    if (this->surfMappedMinDisplSclSlot.IsDirty()) {
        this->surfMappedMinDisplScl = this->surfMappedMinDisplSclSlot.Param<core::param::FloatParam>()->Value()/1000.0f;
        this->surfMappedMinDisplSclSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
    }

    // Param for spring stiffness
    if (this->surfMappedSpringStiffnessSlot.IsDirty()) {
        this->surfMappedSpringStiffness =
                this->surfMappedSpringStiffnessSlot.Param<core::param::FloatParam>()->Value();
        this->surfMappedSpringStiffnessSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
    }


    /// GVF scale factor
    if (this->surfMappedGVFSclSlot.IsDirty()) {
        this->surfMappedGVFScl =
                this->surfMappedGVFSclSlot.Param<core::param::FloatParam>()->Value();
        this->surfMappedGVFSclSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
        this->triggerComputeLines = true;
    }

    /// GVF iterations
    if (this->surfMappedGVFItSlot.IsDirty()) {
        this->surfMappedGVFIt =
                this->surfMappedGVFItSlot.Param<core::param::IntParam>()->Value();
        this->surfMappedGVFItSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
        this->triggerComputeLines = true;
    }


    /* Parameters for surface regularization */

    // Maximum number of iterations when regularizing the mesh #0
    if (this->regMaxItSlot.IsDirty()) {
        this->regMaxIt = this->regMaxItSlot.Param<core::param::IntParam>()->Value();
        this->regMaxItSlot.ResetDirty();
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }

    // Stiffness of the springs defining the spring forces in surface #0
    if (this->regSpringStiffnessSlot.IsDirty()) {
        this->regSpringStiffness = this->regSpringStiffnessSlot.Param<core::param::FloatParam>()->Value();
        this->regSpringStiffnessSlot.ResetDirty();
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }

    // Weighting of the external forces in surface #0, note that the weight
    // of the internal forces is implicitely defined by
    // 1.0 - surf0ExternalForcesWeight
    if (this->regExternalForcesWeightSlot.IsDirty()) {
        this->regExternalForcesWeight =
                this->regExternalForcesWeightSlot.Param<core::param::FloatParam>()->Value();
        this->regExternalForcesWeightSlot.ResetDirty();
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }

    // Overall scaling for the forces acting upon surface #0
    if (this->regForcesSclSlot.IsDirty()) {
        this->regForcesScl = this->regForcesSclSlot.Param<core::param::FloatParam>()->Value();
        this->regForcesSclSlot.ResetDirty();
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }


    /* Rendering of surface #1 and #2 */

    // Parameter for surface #0 render mode
    if (this->surface1RMSlot.IsDirty()) {
        this->surface1RM = static_cast<SurfaceRenderMode>(
                this->surface1RMSlot.Param<core::param::EnumParam>()->Value());
        this->surface1RMSlot.ResetDirty();
    }

    // Parameter for surface #1 color mode
    if (this->surface1ColorModeSlot.IsDirty()) {
        this->surface1ColorMode = static_cast<SurfaceColorMode>(
                this->surface1ColorModeSlot.Param<core::param::EnumParam>()->Value());
        this->surface1ColorModeSlot.ResetDirty();
    }

    // Param for transparency scaling
    if (this->surf1AlphaSclSlot.IsDirty()) {
        this->surf1AlphaScl = this->surf1AlphaSclSlot.Param<core::param::FloatParam>()->Value();
        this->surf1AlphaSclSlot.ResetDirty();
    }

    // Parameter for surface #2 render mode
    if (this->surface2RMSlot.IsDirty()) {
        this->surface2RM = static_cast<SurfaceRenderMode>(
                this->surface2RMSlot.Param<core::param::EnumParam>()->Value());
        this->surface2RMSlot.ResetDirty();
    }

    // Parameter for surface #2 color mode
    if (this->surface2ColorModeSlot.IsDirty()) {
        this->surface2ColorMode = static_cast<SurfaceColorMode>(
                this->surface2ColorModeSlot.Param<core::param::EnumParam>()->Value());
        this->surface2ColorModeSlot.ResetDirty();
    }

    // Param for transparency scaling
    if (this->surf2AlphaSclSlot.IsDirty()) {
        this->surf2AlphaScl = this->surf2AlphaSclSlot.Param<core::param::FloatParam>()->Value();
        this->surf2AlphaSclSlot.ResetDirty();
    }


    /* Mapped surface rendering */

    // Parameter for mapped surface render mode
    if (this->surfaceMappedRMSlot.IsDirty()) {
        this->surfaceMappedRM = static_cast<SurfaceRenderMode>(
                this->surfaceMappedRMSlot.Param<core::param::EnumParam>()->Value());
        this->surfaceMappedRMSlot.ResetDirty();
    }

    // Parameter for mapped surface color mode
    if (this->surfaceMappedColorModeSlot.IsDirty()) {
        this->surfaceMappedColorMode = static_cast<SurfaceColorMode>(
                this->surfaceMappedColorModeSlot.Param<core::param::EnumParam>()->Value());
        this->surfaceMappedColorModeSlot.ResetDirty();
    }

    // Param for maximum value for euclidian distance
    if (this->surfMaxPosDiffSlot.IsDirty()) {
        this->surfMaxPosDiff = this->surfMaxPosDiffSlot.Param<core::param::FloatParam>()->Value();
        this->surfMaxPosDiffSlot.ResetDirty();
    }


    // Param for transparency scaling
    if (this->surfMappedAlphaSclSlot.IsDirty()) {
        this->surfMappedAlphaScl = this->surfMappedAlphaSclSlot.Param<core::param::FloatParam>()->Value();
        this->surfMappedAlphaSclSlot.ResetDirty();
    }

    /* DEBUG External forces rendering */

    // Max x
    if (this->filterXMaxParam.IsDirty()) {
        this->posXMax = this->filterXMaxParam.Param<core::param::FloatParam>()->Value();
        this->filterXMaxParam.ResetDirty();
        this->triggerComputeLines = true;
    }

    // Max y
    if (this->filterYMaxParam.IsDirty()) {
        this->posYMax = this->filterYMaxParam.Param<core::param::FloatParam>()->Value();
        this->filterYMaxParam.ResetDirty();
        this->triggerComputeLines = true;
    }

    // Max z
    if (this->filterZMaxParam.IsDirty()) {
        this->posZMax = this->filterZMaxParam.Param<core::param::FloatParam>()->Value();
        this->filterZMaxParam.ResetDirty();
        this->triggerComputeLines = true;
    }

    // Min x
    if (this->filterXMinParam.IsDirty()) {
        this->posXMin = this->filterXMinParam.Param<core::param::FloatParam>()->Value();
        this->filterXMinParam.ResetDirty();
        this->triggerComputeLines = true;
    }

    // Min y
    if (this->filterYMinParam.IsDirty()) {
        this->posYMin = this->filterYMinParam.Param<core::param::FloatParam>()->Value();
        this->filterYMinParam.ResetDirty();
        this->triggerComputeLines = true;
    }

    // Min z
    if (this->filterZMinParam.IsDirty()) {
        this->posZMin = this->filterZMinParam.Param<core::param::FloatParam>()->Value();
        this->filterZMinParam.ResetDirty();
        this->triggerComputeLines = true;
    }
}


#endif // WITH_CUDA


