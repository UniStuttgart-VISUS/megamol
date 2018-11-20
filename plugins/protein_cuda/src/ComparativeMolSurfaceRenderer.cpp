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

//#define USE_TIMER
//#define VERBOSE

#include "VBODataCall.h"
#include "protein_calls/VTIDataCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "RMS.h"
#include "ogl_error_check.h"
#include "DiffusionSolver.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/view/AbstractCallRender3D.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"

#include <algorithm>
// For profiling
#include <cuda_profiler_api.h>

#include <cmath>
#include <sstream>

#include "vislib/sys/ASCIIFileBuffer.h"

using namespace megamol;
using namespace megamol::protein_cuda;
using namespace megamol::protein_calls;

// 99 75 0
Vec3f dark_brown = Vec3f(0.388235294f, 0.294117647f, 0.0f);
Vec3f light_brown = Vec3f(0.654901961f, 0.494117647f, 0.0f);
Vec3f light_blue = Vec3f(0.0f, 0.458823529f, 0.650980392f);
Vec3f dark_blue = Vec3f(0.0f, 0.309803922f, 0.439215686f);

// Lighter brown for shading
// #DAB23A
// 218 178 58
Vec3f light_surf_brown = Vec3f(0.854901961f, 0.698039216f, 0.22745098f);

Vec3f white = Vec3f(1, 1, 1);

// Hardcoded parameters for 'quicksurf' class
//const float ComparativeMolSurfaceRenderer::qsParticleRad = 0.8f;
const float ComparativeMolSurfaceRenderer::qsGaussLim = 8.0f;
//const float ComparativeMolSurfaceRenderer::qsGridSpacing = 1.0f;
const bool ComparativeMolSurfaceRenderer::qsSclVanDerWaals = true;
//const float ComparativeMolSurfaceRenderer::qsIsoVal = 0.5f;

// Hardcoded colors for surface rendering
const Vec3f ComparativeMolSurfaceRenderer::uniformColorSurf2 = light_surf_brown;
const Vec3f ComparativeMolSurfaceRenderer::uniformColorSurf1 = light_blue;
const Vec3f ComparativeMolSurfaceRenderer::uniformColorSurfMapped =
        light_surf_brown;
const Vec3f ComparativeMolSurfaceRenderer::colorMaxPotential = Vec3f(0.0f, 0.0f,
        1.0f);
const Vec3f ComparativeMolSurfaceRenderer::colorMinPotential = Vec3f(1.0f, 0.0f,
        0.0f);

// Maximum RMS value to enable valid mapping
const float ComparativeMolSurfaceRenderer::maxRMSVal = 100.0f;

/*
 * ComparativeMolSurfaceRenderer::computeDensityBBox
 */
void ComparativeMolSurfaceRenderer::computeDensityBBox(const float *atomPos1,
        const float *atomPos2, size_t atomCnt1, size_t atomCnt2) {

    float3 minC, maxC;
    minC.x = maxC.x = atomPos1[0];
    minC.y = maxC.y = atomPos1[1];
    minC.z = maxC.z = atomPos1[2];
    for (size_t i = 0; i < atomCnt1; ++i) {
        minC.x = std::min(minC.x, atomPos1[3 * i + 0]);
        minC.y = std::min(minC.y, atomPos1[3 * i + 1]);
        minC.z = std::min(minC.z, atomPos1[3 * i + 2]);
        maxC.x = std::max(maxC.x, atomPos1[3 * i + 0]);
        maxC.y = std::max(maxC.y, atomPos1[3 * i + 1]);
        maxC.z = std::max(maxC.z, atomPos1[3 * i + 2]);
//        if (i < 10) {
//            printf("ATOM POS #0, %i: %f %f %f\n", i,
//                atomPos1[3*i+0],
//                atomPos1[3*i+1],
//                atomPos1[3*i+2]);
//        }
    }
    for (size_t i = 0; i < atomCnt2; ++i) {
        minC.x = std::min(minC.x, atomPos2[3 * i + 0]);
        minC.y = std::min(minC.y, atomPos2[3 * i + 1]);
        minC.z = std::min(minC.z, atomPos2[3 * i + 2]);
        maxC.x = std::max(maxC.x, atomPos2[3 * i + 0]);
        maxC.y = std::max(maxC.y, atomPos2[3 * i + 1]);
        maxC.z = std::max(maxC.z, atomPos2[3 * i + 2]);
//        if (i < 10) {
//            printf("ATOM POS #0, %i: %f %f %f\n", i,
//                atomPos1[3*i+0],
//                atomPos1[3*i+1],
//                atomPos1[3*i+2]);
//        }
    }

    this->bboxParticles.Set(minC.x, minC.y, minC.z, maxC.x, maxC.y, maxC.z);

#ifdef USE_PROCEDURAL_DATA
    this->bboxParticles.Set(-50, -50, -50, 50, 50, 50);
#endif

//    // DEBUG Print new bounding box
//    printf("bbboxParticles: %f %f %f %f %f %f\n", minC.x, minC.y, minC.z,
//            maxC.x, maxC.y, maxC.z);
//    printf("atomCnt0: %u\n",atomCnt1);
//    printf("atomCnt1: %u\n",atomCnt2);
//    // END DEBUG

}

/*
 * ComparativeMolSurfaceRenderer::ComparativeMolSurfaceRenderer
 */
ComparativeMolSurfaceRenderer::ComparativeMolSurfaceRenderer(void) :
        Renderer3DModuleDS(), vboSlaveSlot1("vboOut1",
                "Provides access to the vbo containing data for data set #1"), vboSlaveSlot2(
                "vboOut2",
                "Provides access to the vbo containing data for data set #2"), molDataSlot1(
                "molIn1", "Input molecule #1"), molDataSlot2("molIn2",
                "Input molecule #2"), volDataSlot1("volIn1",
                "Connects the rendering with data storage"), volDataSlot2(
                "volIn2", "Connects the rendering with data storage"), rendererCallerSlot1(
                "protren1", "Connects the renderer with another render module"), rendererCallerSlot2(
                "protren2", "Connects the renderer with another render module"),
        /* Parameters for frame by frame comparison */
        cmpModeSlot("cmpMode", "Param slot for compare mode"), singleFrame1Slot(
                "singleFrame1", "Param for single frame #1"), singleFrame2Slot(
                "singleFrame2", "Param for single frame #2"),
        /* Global mapping options */
        interpolModeSlot("interpolation", "Change interpolation method"), maxPotentialSlot(
                "maxPotential", "Maximum potential value for color mapping"), minPotentialSlot(
                "minPotential", "Minimum potential value for color mapping"),
        /* Parameters for mapped surface */
        fittingModeSlot("surfMap::RMSDfitting",
                "RMSD fitting for the mapped surface"), surfMappedExtForceSlot(
                "surfMap::externalForces",
                "External forces for surface mapping"), uncertaintyCriterionSlot(
                "surfMap::uncertainty", "Uncertainty criterion"), surfaceMappingExternalForcesWeightSclSlot(
                "surfMap::externalForcesScl",
                "Scale factor for the external forces weighting"), surfaceMappingForcesSclSlot(
                "surfMap::sclAll", "Overall scale factor for all forces"), surfaceMappingMaxItSlot(
                "surfMap::maxIt",
                "Maximum number of iterations for the surface mapping"), surfMappedMinDisplSclSlot(
                "surfMap::minDisplScl", "Minimum displacement for vertices"), surfMappedSpringStiffnessSlot(
                "surfMap::stiffness", "Spring stiffness"), surfMappedGVFSclSlot(
                "surfMap::GVFScl", "GVF scale factor"), surfMappedGVFItSlot(
                "surfMap::GVFIt", "GVF iterations"), distFieldThreshSlot(
                "surfMap::distFieldThresh", "..."),
        /* Surface regularization */
        regMaxItSlot("surfReg::regMaxIt",
                "Maximum number of iterations for regularization"), regSpringStiffnessSlot(
                "surfReg::stiffness", "Spring stiffness"), regExternalForcesWeightSlot(
                "surfReg::externalForcesWeight",
                "Weight of the external forces"), regForcesSclSlot(
                "surfReg::forcesScl", "Scaling of overall force"), surfregMinDisplSclSlot(
                "surfReg::minDisplScl", "Minimum displacement for vertices"),
        /* Surface rendering */
        surface1RMSlot("surf1::render", "Render mode for the first surface"), surface1ColorModeSlot(
                "surf1::color", "Color mode for the first surface"), surf1AlphaSclSlot(
                "surf1::alphaScl", "Transparency scale factor"), surface2RMSlot(
                "surf2::render", "Render mode for the second surface"), surface2ColorModeSlot(
                "surf2::color", "Render mode for the second surface"), surf2AlphaSclSlot(
                "surf2::alphaScl", "Transparency scale factor"), surfaceMappedRMSlot(
                "surfMap::render", "Render mode for the mapped surface"), surfaceMappedColorModeSlot(
                "surfMap::color", "Color mode for the mapped surface"), unmappedTrisColorSlot(
                "surfMap::unmapped", "Color mode for the unmapped triangles"), surfMaxPosDiffSlot(
                "surfMap::maxPosDiff", "Maximum value for euclidian distance"), surfMappedAlphaSclSlot(
                "surfMap::alphaScl", "Transparency scale factor"),
        /* DEBUG external forces */
        filterXMaxParam("posFilter::XMax",
                "The maximum position in x-direction"), filterYMaxParam(
                "posFilter::YMax", "The maximum position in y-direction"), filterZMaxParam(
                "posFilter::ZMax", "The maximum position in z-direction"), filterXMinParam(
                "posFilter::XMin", "The minimum position in x-direction"), filterYMinParam(
                "posFilter::YMin", "The minimum position in y-direction"), filterZMinParam(
                "posFilter::ZMin", "The minimum position in z-direction"),
        /* Quicksurf parameters */
        /// Parameter slot for atom radius scaling
        qsRadSclSlot("quicksurf::qsRadScl", "..."), qsGridDeltaSlot(
                "quicksurf::qsGridDelta", "..."), qsIsoValSlot(
                "quicksurf::qsIsoVal", "..."),
        /* Subdivision */
        showSubdivSlot("subdiv::show",
                "Parameter to toggle rendering of subdivided triangles"), maxSubdivLevelSlot(
                "subdiv::maxLevel", "Parameter for maximum subdivision level"), maxSubdivStepsSlot(
                "subdiv::maxSteps", "Parameter for maximum morphing steps"), subStepLevelSlot(
                "subdiv::initStepSize", "Initial stepsize"), triggerComputeSubdiv(
                true), cudaqsurf1(NULL), cudaqsurf2(NULL), triggerComputeVolume(
                true), triggerInitPotentialTex(true), triggerComputeSurfacePoints1(
                true), triggerComputeSurfacePoints2(true), triggerSurfaceMapping(
                true), triggerRMSFit(true), triggerComputeLines(true), oldCalltime(
                -1.0), qsGridDelta(2.0f) {

    /* Make data caller/callee slots available */

    this->vboSlaveSlot1.SetCallback(VBODataCall::ClassName(),
            VBODataCall::FunctionName(VBODataCall::CallForGetExtent),
            &ComparativeMolSurfaceRenderer::getVBOExtent);
    this->vboSlaveSlot1.SetCallback(VBODataCall::ClassName(),
            VBODataCall::FunctionName(VBODataCall::CallForGetData),
            &ComparativeMolSurfaceRenderer::getVBOData1);
    this->MakeSlotAvailable(&this->vboSlaveSlot1);

    this->vboSlaveSlot2.SetCallback(VBODataCall::ClassName(),
            VBODataCall::FunctionName(VBODataCall::CallForGetExtent),
            &ComparativeMolSurfaceRenderer::getVBOExtent);
    this->vboSlaveSlot2.SetCallback(VBODataCall::ClassName(),
            VBODataCall::FunctionName(VBODataCall::CallForGetData),
            &ComparativeMolSurfaceRenderer::getVBOData2);
    this->MakeSlotAvailable(&this->vboSlaveSlot2);

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

    // Renderer caller slot #1
    this->rendererCallerSlot1.SetCompatibleCall<
            core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rendererCallerSlot1);

    // Renderer caller slot #2
    this->rendererCallerSlot2.SetCompatibleCall<
            core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rendererCallerSlot2);

    /* Parameters for frame-by-frame comparison */

    // Param slot for compare mode
    this->cmpMode = COMPARE_1_1;
    core::param::EnumParam *cmpm = new core::param::EnumParam(
            int(this->cmpMode));
    cmpm->SetTypePair(COMPARE_1_1, "1-1");
    cmpm->SetTypePair(COMPARE_1_N, "1-n");
    cmpm->SetTypePair(COMPARE_N_1, "n-1");
    cmpm->SetTypePair(COMPARE_N_N, "n-n");
    cmpm->SetTypePair(COMPARE_1_MORPH, "1-Morph");
    cmpm->SetTypePair(COMPARE_1_ROT, "1-Rot");
    this->cmpModeSlot << cmpm;
    this->MakeSlotAvailable(&this->cmpModeSlot);

    // Param for single frame #1
    this->singleFrame1 = 0;
    this->singleFrame1Slot.SetParameter(
            new core::param::IntParam(this->singleFrame1, 0));
    this->MakeSlotAvailable(&this->singleFrame1Slot);

    // Param for single frame #2
    this->singleFrame2 = 0;
    this->singleFrame2Slot.SetParameter(
            new core::param::IntParam(this->singleFrame2, 0));
    this->MakeSlotAvailable(&this->singleFrame2Slot);

    /* Global mapping options */

    // Interpolation method used when computing external forces
    this->interpolMode = DeformableGPUSurfaceMT::INTERP_LINEAR;
    core::param::EnumParam *smi = new core::param::EnumParam(
            int(this->interpolMode));
    smi->SetTypePair(DeformableGPUSurfaceMT::INTERP_LINEAR, "Linear");
    smi->SetTypePair(DeformableGPUSurfaceMT::INTERP_CUBIC, "Cubic");
    this->interpolModeSlot << smi;
    this->MakeSlotAvailable(&this->interpolModeSlot);

    // Maximum potential
    this->maxPotential = 1.0f;
    this->maxPotentialSlot.SetParameter(
            new core::param::FloatParam(this->maxPotential));
    this->MakeSlotAvailable(&this->maxPotentialSlot);

    // Minimum potential
    this->minPotential = -1.0f;
    this->minPotentialSlot.SetParameter(
            new core::param::FloatParam(this->minPotential));
    this->MakeSlotAvailable(&this->minPotentialSlot);

    /* Parameters for mapped surface */

    // Parameter for the RMS fitting mode
    this->fittingMode = RMS_NONE;
    core::param::EnumParam *rms = new core::param::EnumParam(
            int(this->fittingMode));
    rms->SetTypePair(RMS_NONE, "None");
    rms->SetTypePair(RMS_ALL, "All");
    rms->SetTypePair(RMS_BACKBONE, "Backbone");
    rms->SetTypePair(RMS_C_ALPHA, "C alpha");
    this->fittingModeSlot << rms;
    this->MakeSlotAvailable(&this->fittingModeSlot);

    // Param slot for compare mode
    this->surfMappedExtForce = METABALLS;
    core::param::EnumParam *ext = new core::param::EnumParam(
            int(this->surfMappedExtForce));
    ext->SetTypePair(METABALLS, "Meta balls");
    ext->SetTypePair(METABALLS_DISTFIELD, "Meta balls + distance field");
    ext->SetTypePair(GVF, "GVF");
    ext->SetTypePair(TWO_WAY_GVF, "Two-Way-GVF");
    this->surfMappedExtForceSlot << ext;
    this->MakeSlotAvailable(&this->surfMappedExtForceSlot);

    // Param slot for uncertainty
    this->uncertaintyCriterion = EUCLIDEAN_DISTANCE;
    core::param::EnumParam *uc = new core::param::EnumParam(
            int(this->uncertaintyCriterion));
    uc->SetTypePair(EUCLIDEAN_DISTANCE, "Euclidean Distance");
    uc->SetTypePair(PATH_DISTANCE, "Path Distance");
    this->uncertaintyCriterionSlot << uc;
    this->MakeSlotAvailable(&this->uncertaintyCriterionSlot);

    // Weighting for external forces when mapping the surface
    this->surfaceMappingExternalForcesWeightScl = 0.5f;
    this->surfaceMappingExternalForcesWeightSclSlot.SetParameter(
            new core::param::FloatParam(
                    this->surfaceMappingExternalForcesWeightScl, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surfaceMappingExternalForcesWeightSclSlot);

    // Overall scaling of the resulting forces
    this->surfaceMappingForcesScl = 1.0f;
    this->surfaceMappingForcesSclSlot.SetParameter(
            new core::param::FloatParam(this->surfaceMappingForcesScl, 0.0f));
    this->MakeSlotAvailable(&this->surfaceMappingForcesSclSlot);

    // Maximum number of iterations when mapping the surface
    this->surfaceMappingMaxIt = 0;
    this->surfaceMappingMaxItSlot.SetParameter(
            new core::param::IntParam(this->surfaceMappingMaxIt, 0));
    this->MakeSlotAvailable(&this->surfaceMappingMaxItSlot);

    // Spring stiffness
    this->surfMappedSpringStiffness = 0.1f;
    this->surfMappedSpringStiffnessSlot.SetParameter(
            new core::param::FloatParam(this->surfMappedSpringStiffness,
                    0.01f));
    this->MakeSlotAvailable(&this->surfMappedSpringStiffnessSlot);

    // Minimum displacement
    this->surfMappedMinDisplScl = this->qsGridDelta / 100.0f
            * this->surfaceMappingForcesScl;
    //this->surfMappedMinDisplScl = ComparativeMolSurfaceRenderer::qsGridSpacing/100.0f;
    this->surfMappedMinDisplSclSlot.SetParameter(
            new core::param::FloatParam(this->surfMappedMinDisplScl, 0.0f));
    //this->MakeSlotAvailable(&this->surfMappedMinDisplSclSlot);

    /// GVF scale factor
    this->surfMappedGVFScl = 1.0f;
    this->surfMappedGVFSclSlot.SetParameter(
            new core::param::FloatParam(this->surfMappedGVFScl, 0.0f));
    this->MakeSlotAvailable(&this->surfMappedGVFSclSlot);

    /// GVF iterations
    this->surfMappedGVFIt = 0;
    this->surfMappedGVFItSlot.SetParameter(
            new core::param::IntParam(this->surfMappedGVFIt, 0));
    this->MakeSlotAvailable(&this->surfMappedGVFItSlot);

    /// Distance field threshold
    this->distFieldThresh = 1.0f;
    this->distFieldThreshSlot.SetParameter(
            new core::param::FloatParam(this->distFieldThresh, 0.0f));
    this->MakeSlotAvailable(&this->distFieldThreshSlot);

    /* Parameters for surface regularization */

    // Maximum number of iterations when regularizing the mesh #0
    this->regMaxIt = 0;
    this->regMaxItSlot.SetParameter(
            new core::param::IntParam(this->regMaxIt, 0));
    this->MakeSlotAvailable(&this->regMaxItSlot);

    // Stiffness of the springs defining the spring forces in surface #0
    this->regSpringStiffness = 1.0f;
    this->regSpringStiffnessSlot.SetParameter(
            new core::param::FloatParam(this->regSpringStiffness, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->regSpringStiffnessSlot);

    // Weighting of the external forces in surface #0, note that the weight
    // of the internal forces is implicitely defined by
    // 1.0 - surf0ExternalForcesWeight
    this->regExternalForcesWeight = 0.0f;
    this->regExternalForcesWeightSlot.SetParameter(
            new core::param::FloatParam(this->regExternalForcesWeight, 0.0f,
                    1.0f));
    this->MakeSlotAvailable(&this->regExternalForcesWeightSlot);

    // Overall scaling for the forces acting upon surface #0
    this->regForcesScl = 1.0f;
    this->regForcesSclSlot.SetParameter(
            new core::param::FloatParam(this->regForcesScl, 0.0f));
    this->MakeSlotAvailable(&this->regForcesSclSlot);

    // Minimum displacement
    this->surfregMinDisplScl = this->qsGridDelta / 100.0f * this->regForcesScl;
    this->surfregMinDisplSclSlot.SetParameter(
            new core::param::FloatParam(this->surfregMinDisplScl, 0.0f));
    //this->MakeSlotAvailable(&this->surfregMinDisplSclSlot);

    /* Rendering options of surface #1 and #2 */

    // Parameter for surface 1 render mode
    this->surface1RM = SURFACE_NONE;
    core::param::EnumParam *msrm1 = new core::param::EnumParam(
            int(this->surface1RM));
    msrm1->SetTypePair(SURFACE_NONE, "None");
    msrm1->SetTypePair(SURFACE_POINTS, "Points");
    msrm1->SetTypePair(SURFACE_WIREFRAME, "Wireframe");
    msrm1->SetTypePair(SURFACE_FILL, "Fill");
    this->surface1RMSlot << msrm1;
    this->MakeSlotAvailable(&this->surface1RMSlot);

    // Parameter for surface color 1 mode
    this->surface1ColorMode = SURFACE_UNI;
    core::param::EnumParam *mscm1 = new core::param::EnumParam(
            int(this->surface1ColorMode));
    mscm1->SetTypePair(SURFACE_UNI, "Uniform");
    mscm1->SetTypePair(SURFACE_NORMAL, "Normal");
    mscm1->SetTypePair(SURFACE_TEXCOORDS, "TexCoords");
    mscm1->SetTypePair(SURFACE_POTENTIAL, "Potential");
    this->surface1ColorModeSlot << mscm1;
    this->MakeSlotAvailable(&this->surface1ColorModeSlot);

    // Param for transparency scaling
    this->surf1AlphaScl = 1.0f;
    this->surf1AlphaSclSlot.SetParameter(
            new core::param::FloatParam(this->surf1AlphaScl, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surf1AlphaSclSlot);

    // Parameter for surface 2 render mode
    this->surface2RM = SURFACE_NONE;
    core::param::EnumParam *msrm2 = new core::param::EnumParam(
            int(this->surface2RM));
    msrm2->SetTypePair(SURFACE_NONE, "None");
    msrm2->SetTypePair(SURFACE_POINTS, "Points");
    msrm2->SetTypePair(SURFACE_WIREFRAME, "Wireframe");
    msrm2->SetTypePair(SURFACE_FILL, "Fill");
    this->surface2RMSlot << msrm2;
    this->MakeSlotAvailable(&this->surface2RMSlot);

    // Parameter for surface color 2 mode
    this->surface2ColorMode = SURFACE_UNI;
    core::param::EnumParam *mscm2 = new core::param::EnumParam(
            int(this->surface2ColorMode));
    mscm2->SetTypePair(SURFACE_UNI, "Uniform");
    mscm2->SetTypePair(SURFACE_NORMAL, "Normal");
    mscm2->SetTypePair(SURFACE_TEXCOORDS, "TexCoords");
    mscm2->SetTypePair(SURFACE_POTENTIAL, "Potential");
    this->surface2ColorModeSlot << mscm2;
    this->MakeSlotAvailable(&this->surface2ColorModeSlot);

    // Param for transparency scaling
    this->surf2AlphaScl = 1.0f;
    this->surf2AlphaSclSlot.SetParameter(
            new core::param::FloatParam(this->surf2AlphaScl, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surf2AlphaSclSlot);

    // Parameter for mapped surface render mode
    this->surfaceMappedRM = SURFACE_NONE;
    core::param::EnumParam *msrm = new core::param::EnumParam(
            int(this->surfaceMappedRM));
    msrm->SetTypePair(SURFACE_NONE, "None");
    msrm->SetTypePair(SURFACE_POINTS, "Points");
    msrm->SetTypePair(SURFACE_WIREFRAME, "Wireframe");
    msrm->SetTypePair(SURFACE_FILL, "Fill");
    this->surfaceMappedRMSlot << msrm;
    this->MakeSlotAvailable(&this->surfaceMappedRMSlot);

    // Parameter for mapped surface color mode
    this->surfaceMappedColorMode = SURFACE_UNI;
    core::param::EnumParam *mscm = new core::param::EnumParam(
            int(this->surfaceMappedColorMode));
    mscm->SetTypePair(SURFACE_UNI, "Uniform");
    mscm->SetTypePair(SURFACE_NORMAL, "Normal");
    mscm->SetTypePair(SURFACE_TEXCOORDS, "TexCoords");
    mscm->SetTypePair(SURFACE_DIST_TO_OLD_POS, "Dist to old pos");
    mscm->SetTypePair(SURFACE_POTENTIAL0, "Potential0");
    mscm->SetTypePair(SURFACE_POTENTIAL1, "Potential1");
    mscm->SetTypePair(SURFACE_POTENTIAL_DIFF, "Potential Diff");
    mscm->SetTypePair(SURFACE_POTENTIAL_SIGN, "Potential Sign");
    mscm->SetTypePair(SURFACE_LAPLACIAN, "Laplacian");
    this->surfaceMappedColorModeSlot << mscm;
    this->MakeSlotAvailable(&this->surfaceMappedColorModeSlot);

    // Parameter for unmapped triangles color mode
    this->theUnmappedTrisColor = UNMAPPEDTRIS_NONE;
    core::param::EnumParam *utcm = new core::param::EnumParam(
            int(this->surfaceMappedColorMode));
    utcm->SetTypePair(UNMAPPEDTRIS_NONE, "None");
    utcm->SetTypePair(UNMAPPEDTRIS_TRANS, "Transparent");
    utcm->SetTypePair(UNMAPPEDTRIS_COL, "Color");
    this->unmappedTrisColorSlot << utcm;
    this->MakeSlotAvailable(&this->unmappedTrisColorSlot);

    // Param for maximum value for euclidian distance // Not needed atm
    this->surfMaxPosDiff = 1.0f;
    this->surfMaxPosDiffSlot.SetParameter(
            new core::param::FloatParam(this->surfMaxPosDiff, 0.1f));
    this->MakeSlotAvailable(&this->surfMaxPosDiffSlot);

    // Param for transparency scaling
    this->surfMappedAlphaScl = 1.0f;
    this->surfMappedAlphaSclSlot.SetParameter(
            new core::param::FloatParam(this->surfMappedAlphaScl, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surfMappedAlphaSclSlot);

    /* DEBUG external forces */

    // Param for maximum x value
    this->posXMax = 10.0f;
    this->filterXMaxParam.SetParameter(
            new core::param::FloatParam(this->posXMax, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->filterXMaxParam);

    // Param for maximum y value
    this->posYMax = 10.0f;
    this->filterYMaxParam.SetParameter(
            new core::param::FloatParam(this->posYMax, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->filterYMaxParam);

    // Param for maximum z value
    this->posZMax = 10.0f;
    this->filterZMaxParam.SetParameter(
            new core::param::FloatParam(this->posZMax, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->filterZMaxParam);

    // Param for minimum x value
    this->posXMin = -10.0f;
    this->filterXMinParam.SetParameter(
            new core::param::FloatParam(this->posXMin, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->filterXMinParam);

    // Param for minimum y value
    this->posYMin = -10.0f;
    this->filterYMinParam.SetParameter(
            new core::param::FloatParam(this->posYMin, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->filterYMinParam);

    // Param for minimum z value
    this->posZMin = -10.0f;
    this->filterZMinParam.SetParameter(
            new core::param::FloatParam(this->posZMin, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->filterZMinParam);

    /* Quicksurf parameter */

    // Quicksurf radius scale
    this->qsRadScl = 1.0f;
    this->qsRadSclSlot.SetParameter(
            new core::param::FloatParam(this->qsRadScl));
    this->MakeSlotAvailable(&this->qsRadSclSlot);

    // Quicksurf grid scaling
    this->qsGridDeltaSlot.SetParameter(
            new core::param::FloatParam(this->qsGridDelta));
    this->MakeSlotAvailable(&this->qsGridDeltaSlot);

    // Quicksurf isovalue
    this->qsIsoVal = 0.5f;
    this->qsIsoValSlot.SetParameter(
            new core::param::FloatParam(this->qsIsoVal));
    this->MakeSlotAvailable(&this->qsIsoValSlot);

    /* Subdivision parameters */

    // Parameter to toggle rendering of subdivided triangles
    this->showSubdiv = false;
    this->showSubdivSlot.SetParameter(
            new core::param::BoolParam(this->showSubdiv));
    this->MakeSlotAvailable(&this->showSubdivSlot);

    // Parameter for maximum subdivision level
    this->maxSubdivLevel = 0;
    this->maxSubdivLevelSlot.SetParameter(
            new core::param::IntParam(this->maxSubdivLevel, 0));
    this->MakeSlotAvailable(&this->maxSubdivLevelSlot);

    // Parameter for maximum morphing steps
    this->maxSubdivSteps = 0;
    this->maxSubdivStepsSlot.SetParameter(
            new core::param::IntParam(this->maxSubdivSteps, 0));
    this->MakeSlotAvailable(&this->maxSubdivStepsSlot);

    // Parameter for initial stepsize
    this->subStepLevel = 1;
    this->subStepLevelSlot.SetParameter(
            new core::param::IntParam(this->subStepLevel));
    this->MakeSlotAvailable(&this->subStepLevelSlot);
}

/*
 * ComparativeMolSurfaceRenderer::~ComparativeMolSurfaceRenderer
 */
ComparativeMolSurfaceRenderer::~ComparativeMolSurfaceRenderer(void) {
    this->Release();
    CudaSafeCall(cudaDeviceReset());
}

/*
 * ComparativeSurfacePotentialRenderer::computeDensityMap
 */
bool ComparativeMolSurfaceRenderer::computeDensityMap(const float *atomPos,
        const MolecularDataCall *mol, CUDAQuickSurf *cqs) {

    using namespace vislib::sys;
    using namespace vislib::math;

    float gridXAxisLen, gridYAxisLen, gridZAxisLen;
    float padding;

    // (Re-)allocate memory for intermediate particle data
    this->gridDataPos.Validate(mol->AtomCount() * 4);

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

                MolecularDataCall::Molecule molTmp =
                        mol->Molecules()[chainTmp.FirstMoleculeIndex() + m];
                for (uint res = 0; res < molTmp.ResidueCount(); ++res) { // Loop through residues

                    const MolecularDataCall::Residue *resTmp =
                            mol->Residues()[molTmp.FirstResidueIndex() + res];

                    for (uint at = 0; at < resTmp->AtomCount(); ++at) { // Loop through atoms
                        uint atomIdx = resTmp->FirstAtomIndex() + at;

                        this->gridDataPos.Peek()[4 * particleCnt + 0] =
                                atomPos[3 * atomIdx + 0];
                        this->gridDataPos.Peek()[4 * particleCnt + 1] =
                                atomPos[3 * atomIdx + 1];
                        this->gridDataPos.Peek()[4 * particleCnt + 2] =
                                atomPos[3 * atomIdx + 2];
                        if (this->qsSclVanDerWaals) {
                            this->gridDataPos.Peek()[4 * particleCnt + 3] =
                                    mol->AtomTypes()[mol->AtomTypeIndices()[atomIdx]].Radius();
                        } else {
                            this->gridDataPos.Peek()[4 * particleCnt + 3] =
                                    1.0f;
                        }

                        this->maxAtomRad = std::max(this->maxAtomRad,
                                this->gridDataPos.Peek()[4 * particleCnt + 3]);
                        this->minAtomRad = std::min(this->minAtomRad,
                                this->gridDataPos.Peek()[4 * particleCnt + 3]);

                        particleCnt++;
                    }
                }
            }
        }
    }

//    printf("Compute density map particleCount %u,molcall %u\n", particleCnt,mol->AtomCount());

    // Compute padding for the density map
    padding = this->maxAtomRad * this->qsRadScl + this->qsGridDelta * 10;

    // Init grid parameters
    this->volOrg.x = this->bboxParticles.GetLeft() - padding;
    this->volOrg.y = this->bboxParticles.GetBottom() - padding;
    this->volOrg.z = this->bboxParticles.GetBack() - padding;
    this->volMaxC.x = this->bboxParticles.GetRight() + padding;
    this->volMaxC.y = this->bboxParticles.GetTop() + padding;
    this->volMaxC.z = this->bboxParticles.GetFront() + padding;
    gridXAxisLen = this->volMaxC.x - this->volOrg.x;
    gridYAxisLen = this->volMaxC.y - this->volOrg.y;
    gridZAxisLen = this->volMaxC.z - this->volOrg.z;
    this->volDim.x = (int) ceil(gridXAxisLen / this->qsGridDelta);
    this->volDim.y = (int) ceil(gridYAxisLen / this->qsGridDelta);
    this->volDim.z = (int) ceil(gridZAxisLen / this->qsGridDelta);
    gridXAxisLen = (this->volDim.x - 1) * this->qsGridDelta;
    gridYAxisLen = (this->volDim.y - 1) * this->qsGridDelta;
    gridZAxisLen = (this->volDim.z - 1) * this->qsGridDelta;
    this->volMaxC.x = this->volOrg.x + gridXAxisLen;
    this->volMaxC.y = this->volOrg.y + gridYAxisLen;
    this->volMaxC.z = this->volOrg.z + gridZAxisLen;
    this->volDelta.x = this->qsGridDelta;
    this->volDelta.y = this->qsGridDelta;
    this->volDelta.z = this->qsGridDelta;

    // Set particle positions
#pragma omp parallel for
    for (int cnt = 0; cnt < static_cast<int>(mol->AtomCount()); ++cnt) {
        this->gridDataPos.Peek()[4 * cnt + 0] -= this->volOrg.x;
        this->gridDataPos.Peek()[4 * cnt + 1] -= this->volOrg.y;
        this->gridDataPos.Peek()[4 * cnt + 2] -= this->volOrg.z;
    }

//    for (int cnt = 0; cnt < static_cast<int>(mol->AtomCount()); ++cnt) {
//        printf("data pos %i %f %f %f\n",cnt,
//                this->gridDataPos.Peek()[4*cnt+0],
//                this->gridDataPos.Peek()[4*cnt+1],
//                this->gridDataPos.Peek()[4*cnt+2]);
//    }

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif // USE_TIMER
    // Compute uniform grid
    int rc = cqs->calc_map(mol->AtomCount(), &this->gridDataPos.Peek()[0],
            NULL,   // Pointer to 'color' array
            false,  // Do not use 'color' array
            (float*) &this->volOrg, (int*) &this->volDim, this->maxAtomRad,
            this->qsRadScl, // Radius scaling
            this->qsGridDelta, this->qsIsoVal, this->qsGaussLim);

//    printf("QUICKSURF PARAMETERS");
//    printf("Particle count %u\n", mol->AtomCount());
//    printf("Last particle position %f %f %f\n",
//            this->gridDataPos.Peek()[(mol->AtomCount()-1)*4+0],
//            this->gridDataPos.Peek()[(mol->AtomCount()-1)*4+1],
//            this->gridDataPos.Peek()[(mol->AtomCount()-1)*4+2]);
//    printf("volOrg %f %f %f\n", this->volOrg.x,this->volOrg.y,this->volOrg.z);
//    printf("************* volDim %i %i %i **************\n", this->volDim.x,this->volDim.y,this->volDim.z);
//    printf("max atom rad %f\n", this->maxAtomRad);
//    printf("particle radius scaling %f\n",  this->qsParticleRad);
//    printf("gridSpacing %f\n", this->qsGridSpacing);
//    printf("isovalue %f \n", this->qsIsoVal);
//    printf("Gauss limit %f\n", this->qsGaussLim);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time for 'quicksurf':                             %.10f sec\n",
            dt_ms/1000.0f);
#endif // USE_TIMER
    if (rc != 0) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Quicksurf class returned val != 0\n", this->ClassName());
        return false;
    }

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
    if (!ogl_IsVersionGEQ(2,0) || !isExtAvailable("GL_EXT_texture3D") || !isExtAvailable("GL_EXT_framebuffer_object")
        || !isExtAvailable("GL_ARB_multitexture") || !isExtAvailable("GL_ARB_draw_buffers") || !isExtAvailable("GL_ARB_copy_buffer")
        || !isExtAvailable("GL_ARB_vertex_buffer_object")) {
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
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "electrostatics::pplsurface::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    // Load ppl fragment shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "electrostatics::pplsurface::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    try {
        if (!this->pplSurfaceShader.Create(vertSrc.Code(), vertSrc.Count(),
                fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__,
                    __LINE__);
        }
    } catch (vislib::Exception &e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to create the ppl shader: %s\n", this->ClassName(),
                e.GetMsgA());
        return false;
    }

    // Load shader for per pixel lighting of the surface
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "electrostatics::pplsurface::vertexMapped", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    // Load ppl fragment shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "electrostatics::pplsurface::fragmentMapped", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    try {
        if (!this->pplMappedSurfaceShader.Create(vertSrc.Code(),
                vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__,
                    __LINE__);
        }
    } catch (vislib::Exception &e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to create the ppl shader: %s\n", this->ClassName(),
                e.GetMsgA());
        return false;
    }

    // Load shader for per pixel lighting of the surface
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "electrostatics::pplsurface::vertexWithFlag", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    // Load ppl fragment shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "electrostatics::pplsurface::fragmentWithFlag", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    try {
        if (!this->pplSurfaceShaderVertexFlag.Create(vertSrc.Code(),
                vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__,
                    __LINE__);
        }
    } catch (vislib::Exception &e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to create the ppl shader (vertex flag): %s\n",
                this->ClassName(), e.GetMsgA());
        return false;
    }

    // Load shader for per pixel lighting of the surface
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "electrostatics::pplsurface::vertexUncertainty", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    // Load ppl fragment shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "electrostatics::pplsurface::fragmentUncertainty", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    try {
        if (!this->pplSurfaceShaderUncertainty.Create(vertSrc.Code(),
                vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__,
                    __LINE__);
        }
    } catch (vislib::Exception &e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to create the ppl shader (vertex flag): %s\n",
                this->ClassName(), e.GetMsgA());
        return false;
    }

    return true;
}

/*
 * ComparativeSurfacePotentialRenderer::fitMoleculeRMS
 */
bool ComparativeMolSurfaceRenderer::fitMoleculeRMS(MolecularDataCall *mol1,
        MolecularDataCall *mol2) {
    using namespace vislib::sys;

    // Extract positions to be fitted
    vislib::Array<float> atomWeights;
    vislib::Array<int> atomMask;
    uint posCnt = 0;

//    printf("RMS frame idx %u and %u\n", mol0->FrameID(), mol1->FrameID());

    // No RMS fitting
    if (this->fittingMode == RMS_NONE) {
        this->rmsTranslation.Set(0.0f, 0.0f, 0.0f);
        this->rmsRotation.SetIdentity();
        this->rmsValue = 0.0f;
    } else {
        // Use all particles for RMS fitting
        if (this->fittingMode == RMS_ALL) {

            uint posCnt1 = 0, posCnt2 = 0;

            // (Re)-allocate memory if necessary
            this->rmsPosVec1.Validate(mol1->AtomCount() * 3);
            this->rmsPosVec2.Validate(mol2->AtomCount() * 3);

            // Extracting protein atoms from mol 1
            for (uint sec = 0; sec < mol1->SecondaryStructureCount(); ++sec) {
                for (uint acid = 0;
                        acid < mol1->SecondaryStructures()[sec].AminoAcidCount();
                        ++acid) {
                    const MolecularDataCall::AminoAcid *aminoAcid =
                            dynamic_cast<const MolecularDataCall::AminoAcid*>((mol1->Residues()[mol1->SecondaryStructures()[sec].FirstAminoAcidIndex()
                                    + acid]));
                    if (aminoAcid == NULL) {
                        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                                "%s: Unable to perform RMSD fitting using all protein atoms (residue mislabeled as 'amino acid')",
                                this->ClassName(), posCnt1, posCnt2);
                        return false;
                    }
                    for (uint at = 0; at < aminoAcid->AtomCount(); ++at) {
                        this->rmsPosVec1.Peek()[3 * posCnt1 + 0] =
                                mol1->AtomPositions()[3
                                        * (aminoAcid->FirstAtomIndex() + at) + 0];
                        this->rmsPosVec1.Peek()[3 * posCnt1 + 1] =
                                mol1->AtomPositions()[3
                                        * (aminoAcid->FirstAtomIndex() + at) + 1];
                        this->rmsPosVec1.Peek()[3 * posCnt1 + 2] =
                                mol1->AtomPositions()[3
                                        * (aminoAcid->FirstAtomIndex() + at) + 2];
                        posCnt1++;
                    }
                }
            }

            // Extracting protein atoms from mol 1
            for (uint sec = 0; sec < mol2->SecondaryStructureCount(); ++sec) {
                for (uint acid = 0;
                        acid < mol2->SecondaryStructures()[sec].AminoAcidCount();
                        ++acid) {
                    const MolecularDataCall::AminoAcid *aminoAcid =
                            dynamic_cast<const MolecularDataCall::AminoAcid*>((mol2->Residues()[mol2->SecondaryStructures()[sec].FirstAminoAcidIndex()
                                    + acid]));
                    if (aminoAcid == NULL) {
                        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                                "%s: Unable to perform RMSD fitting using all protein atoms (residue mislabeled as 'amino acid')",
                                this->ClassName(), posCnt1, posCnt2);
                        return false;
                    }
                    for (uint at = 0; at < aminoAcid->AtomCount(); ++at) {
                        this->rmsPosVec2.Peek()[3 * posCnt2 + 0] =
                                mol2->AtomPositions()[3
                                        * (aminoAcid->FirstAtomIndex() + at) + 0];
                        this->rmsPosVec2.Peek()[3 * posCnt2 + 1] =
                                mol2->AtomPositions()[3
                                        * (aminoAcid->FirstAtomIndex() + at) + 1];
                        this->rmsPosVec2.Peek()[3 * posCnt2 + 2] =
                                mol2->AtomPositions()[3
                                        * (aminoAcid->FirstAtomIndex() + at) + 2];
                        posCnt2++;
                    }
                }
            }

            if (posCnt1 != posCnt2) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: Unable to perform RMSD fitting using all protein atoms (non-equal atom count (%u vs. %u), try backbone instead)",
                        this->ClassName(), posCnt1, posCnt2);
                return false;
            }
            posCnt = posCnt1;

        } else if (this->fittingMode == RMS_BACKBONE) { // Use backbone atoms for RMS fitting
            // (Re)-allocate memory if necessary
            this->rmsPosVec1.Validate(mol1->AtomCount() * 3);
            this->rmsPosVec2.Validate(mol2->AtomCount() * 3);

            uint posCnt1 = 0, posCnt2 = 0;

            // Extracting backbone atoms from mol 1
            for (uint sec = 0; sec < mol1->SecondaryStructureCount(); ++sec) {

                for (uint acid = 0;
                        acid < mol1->SecondaryStructures()[sec].AminoAcidCount();
                        ++acid) {

                    uint cAlphaIdx =
                            ((const MolecularDataCall::AminoAcid*) (mol1->Residues()[mol1->SecondaryStructures()[sec].FirstAminoAcidIndex()
                                    + acid]))->CAlphaIndex();
                    uint cCarbIdx =
                            ((const MolecularDataCall::AminoAcid*) (mol1->Residues()[mol1->SecondaryStructures()[sec].FirstAminoAcidIndex()
                                    + acid]))->CCarbIndex();
                    uint nIdx =
                            ((const MolecularDataCall::AminoAcid*) (mol1->Residues()[mol1->SecondaryStructures()[sec].FirstAminoAcidIndex()
                                    + acid]))->NIndex();
                    uint oIdx =
                            ((const MolecularDataCall::AminoAcid*) (mol1->Residues()[mol1->SecondaryStructures()[sec].FirstAminoAcidIndex()
                                    + acid]))->OIndex();

//                    printf("c alpha idx %u, cCarbIdx %u, o idx %u, n idx %u\n",
//                            cAlphaIdx, cCarbIdx, oIdx, nIdx); // DEBUG
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 0] =
                            mol1->AtomPositions()[3 * cAlphaIdx + 0];
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 1] =
                            mol1->AtomPositions()[3 * cAlphaIdx + 1];
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 2] =
                            mol1->AtomPositions()[3 * cAlphaIdx + 2];
                    posCnt1++;
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 0] =
                            mol1->AtomPositions()[3 * cCarbIdx + 0];
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 1] =
                            mol1->AtomPositions()[3 * cCarbIdx + 1];
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 2] =
                            mol1->AtomPositions()[3 * cCarbIdx + 2];
                    posCnt1++;
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 0] =
                            mol1->AtomPositions()[3 * oIdx + 0];
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 1] =
                            mol1->AtomPositions()[3 * oIdx + 1];
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 2] =
                            mol1->AtomPositions()[3 * oIdx + 2];
                    posCnt1++;
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 0] =
                            mol1->AtomPositions()[3 * nIdx + 0];
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 1] =
                            mol1->AtomPositions()[3 * nIdx + 1];
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 2] =
                            mol1->AtomPositions()[3 * nIdx + 2];
                    posCnt1++;
                }
            }

            // Extracting backbone atoms from mol 2
            for (uint sec = 0; sec < mol2->SecondaryStructureCount(); ++sec) {

                MolecularDataCall::SecStructure secStructure =
                        mol2->SecondaryStructures()[sec];
                for (uint acid = 0; acid < secStructure.AminoAcidCount();
                        ++acid) {

                    uint cAlphaIdx =
                            ((const MolecularDataCall::AminoAcid*) (mol2->Residues()[secStructure.FirstAminoAcidIndex()
                                    + acid]))->CAlphaIndex();
                    uint cCarbIdx =
                            ((const MolecularDataCall::AminoAcid*) (mol2->Residues()[secStructure.FirstAminoAcidIndex()
                                    + acid]))->CCarbIndex();
                    uint nIdx =
                            ((const MolecularDataCall::AminoAcid*) (mol2->Residues()[secStructure.FirstAminoAcidIndex()
                                    + acid]))->NIndex();
                    uint oIdx =
                            ((const MolecularDataCall::AminoAcid*) (mol2->Residues()[secStructure.FirstAminoAcidIndex()
                                    + acid]))->OIndex();
//                    printf("amino acid idx %u, c alpha idx %u, cCarbIdx %u, o idx %u, n idx %u\n", secStructure.
//                            FirstAminoAcidIndex()+acid, cAlphaIdx, cCarbIdx, oIdx, nIdx);
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 0] =
                            mol2->AtomPositions()[3 * cAlphaIdx + 0];
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 1] =
                            mol2->AtomPositions()[3 * cAlphaIdx + 1];
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 2] =
                            mol2->AtomPositions()[3 * cAlphaIdx + 2];
                    posCnt2++;
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 0] =
                            mol2->AtomPositions()[3 * cCarbIdx + 0];
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 1] =
                            mol2->AtomPositions()[3 * cCarbIdx + 1];
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 2] =
                            mol2->AtomPositions()[3 * cCarbIdx + 2];
                    posCnt2++;
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 0] =
                            mol2->AtomPositions()[3 * oIdx + 0];
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 1] =
                            mol2->AtomPositions()[3 * oIdx + 1];
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 2] =
                            mol2->AtomPositions()[3 * oIdx + 2];
                    posCnt2++;
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 0] =
                            mol2->AtomPositions()[3 * nIdx + 0];
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 1] =
                            mol2->AtomPositions()[3 * nIdx + 1];
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 2] =
                            mol2->AtomPositions()[3 * nIdx + 2];
                    posCnt2++;
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

            if (posCnt1 != posCnt2) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: Unable to perform RMSD fitting using backbone \
atoms (non-equal atom count, %u vs. %u), try C alpha \
atoms instead.",
                        this->ClassName(), posCnt1, posCnt2);
                return false;
            }
            posCnt = posCnt1;
        } else if (this->fittingMode == RMS_C_ALPHA) { // Use C alpha atoms for RMS fitting
            // (Re)-allocate memory if necessary
            this->rmsPosVec1.Validate(mol1->AtomCount() * 3);
            this->rmsPosVec2.Validate(mol2->AtomCount() * 3);

            uint posCnt1 = 0, posCnt2 = 0;

            // Extracting C alpha atoms from mol 1
//            for (uint sec = 0; sec < mol1->SecondaryStructureCount(); ++sec) {
//                printf("sec %u of %u\n", sec, mol1->SecondaryStructureCount());
//                MolecularDataCall::SecStructure secStructure = mol1->SecondaryStructures()[sec];
//                for (uint acid = 0; acid < secStructure.AminoAcidCount(); ++acid) {
//                    printf("acid %u of %u\n", acid, secStructure.AminoAcidCount());
//                    const MolecularDataCall::Residue *residue = mol1->Residues()[secStructure.FirstAminoAcidIndex()+acid];
//                    if (residue->Identifier() != MolecularDataCall::Residue::AMINOACID) {
//                        printf("No amino acid!!\n");
//                    }
//                    uint cAlphaIdx = ((const MolecularDataCall::AminoAcid*)(mol1->Residues()[secStructure.FirstAminoAcidIndex()+acid]))->CAlphaIndex();
////                    printf("amino acid idx %u, c alpha idx %u\n", secStructure.
////                            FirstAminoAcidIndex()+acid, cAlphaIdx);
//                    this->rmsPosVec1.Peek()[3*posCnt1+0] = mol1->AtomPositions()[3*cAlphaIdx+0];
//                    this->rmsPosVec1.Peek()[3*posCnt1+1] = mol1->AtomPositions()[3*cAlphaIdx+1];
//                    this->rmsPosVec1.Peek()[3*posCnt1+2] = mol1->AtomPositions()[3*cAlphaIdx+2];
//                    if (posCnt1 <= 20) {
//                        printf("ADDING ATOM POS 1 %f %f %f\n",
//                            this->rmsPosVec1.Peek()[3*posCnt1+0],
//                            this->rmsPosVec1.Peek()[3*posCnt1+1],
//                            this->rmsPosVec1.Peek()[3*posCnt1+2]);
//                        printf("amino acid idx %u, c alpha idx %u\n", secStructure.FirstAminoAcidIndex()+acid, cAlphaIdx);
//                    }
//                    posCnt1++;
//
//                }
//            }

            // Loop through all residues of the molecule
            for (size_t res = 0; res < mol1->ResidueCount(); ++res) {
                const MolecularDataCall::Residue *residue =
                        mol1->Residues()[res];
                if (residue->Identifier()
                        == MolecularDataCall::Residue::AMINOACID) {
                    uint cAlphaIdx =
                            ((const MolecularDataCall::AminoAcid*) (residue))->CAlphaIndex();
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 0] =
                            mol1->AtomPositions()[3 * cAlphaIdx + 0];
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 1] =
                            mol1->AtomPositions()[3 * cAlphaIdx + 1];
                    this->rmsPosVec1.Peek()[3 * posCnt1 + 2] =
                            mol1->AtomPositions()[3 * cAlphaIdx + 2];
//                    printf("ADDING ATOM POS 1 %f %f %f\n",
//                            this->rmsPosVec1.Peek()[3*posCnt1+0],
//                            this->rmsPosVec1.Peek()[3*posCnt1+1],
//                            this->rmsPosVec1.Peek()[3*posCnt1+2]);
                    posCnt1++;
                }
            }

//            // Extracting C alpha atoms from mol 2
//            for (uint sec = 0; sec < mol2->SecondaryStructureCount(); ++sec) {
//                MolecularDataCall::SecStructure secStructure = mol2->SecondaryStructures()[sec];
//                for (uint acid = 0; acid < secStructure.AminoAcidCount(); ++acid) {
//                    uint cAlphaIdx =
//                            ((const MolecularDataCall::AminoAcid*)
//                                (mol2->Residues()[secStructure.
//                                     FirstAminoAcidIndex()+acid]))->CAlphaIndex();
//                    this->rmsPosVec2.Peek()[3*posCnt2+0] = mol2->AtomPositions()[3*cAlphaIdx+0];
//                    this->rmsPosVec2.Peek()[3*posCnt2+1] = mol2->AtomPositions()[3*cAlphaIdx+1];
//                    this->rmsPosVec2.Peek()[3*posCnt2+2] = mol2->AtomPositions()[3*cAlphaIdx+2];
////                    if (posCnt2 < 10) {
////                        printf("ADDING ATOM POS 2 %f %f %f\n",
////                            this->rmsPosVec2.Peek()[3*posCnt2+0],
////                            this->rmsPosVec2.Peek()[3*posCnt2+1],
////                            this->rmsPosVec2.Peek()[3*posCnt2+2]);
////                    }
//                    posCnt2++;
//
//                }
//            }

            // Loop through all residues of the molecule
            for (size_t res = 0; res < mol2->ResidueCount(); ++res) {
                const MolecularDataCall::Residue *residue =
                        mol2->Residues()[res];
                if (residue->Identifier()
                        == MolecularDataCall::Residue::AMINOACID) {
                    uint cAlphaIdx =
                            ((const MolecularDataCall::AminoAcid*) (residue))->CAlphaIndex();
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 0] =
                            mol2->AtomPositions()[3 * cAlphaIdx + 0];
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 1] =
                            mol2->AtomPositions()[3 * cAlphaIdx + 1];
                    this->rmsPosVec2.Peek()[3 * posCnt2 + 2] =
                            mol2->AtomPositions()[3 * cAlphaIdx + 2];
//                    printf("ADDING ATOM POS 2 %f %f %f\n",
//                            this->rmsPosVec2.Peek()[3*posCnt2+0],
//                            this->rmsPosVec2.Peek()[3*posCnt2+1],
//                            this->rmsPosVec2.Peek()[3*posCnt2+2]);
                    posCnt2++;
                }
            }

            posCnt = std::min(posCnt1, posCnt2);
//            printf("poscount 1 %u, posCnt 2 %u\n", posCnt1, posCnt2);
        }

//        for (int i = 0; i < posCnt; ++i) {
//            printf("pos1 %f %f %f, pos2 %f %f %f\n",
//                    this->rmsPosVec1.Peek()[3*i+0],
//                    this->rmsPosVec1.Peek()[3*i+1],
//                    this->rmsPosVec1.Peek()[3*i+2],
//                    this->rmsPosVec2.Peek()[3*i+0],
//                    this->rmsPosVec2.Peek()[3*i+1],
//                    this->rmsPosVec2.Peek()[3*i+2]);
//        }

//        // Compute centroid 1
//        Vec3f centroid1;
//        centroid1.Set(0,0,0);
//        for (int cnt = 0; cnt < static_cast<int>(posCnt); ++cnt) {
//            centroid1 += Vec3f(
//                    this->rmsPosVec1.Peek()[cnt*3+0],
//                    this->rmsPosVec1.Peek()[cnt*3+1],
//                    this->rmsPosVec1.Peek()[cnt*3+2]);
//        }
//        centroid1 /= static_cast<float>(posCnt);
//
//        printf("centroid1 %.10f %.10f %.10f\n",
//                centroid1.PeekComponents()[0],
//                centroid1.PeekComponents()[1],
//                centroid1.PeekComponents()[2]);

        // Compute centroid
        for (int cnt = 0; cnt < static_cast<int>(posCnt); ++cnt) {
            this->rmsCentroid += Vec3f(this->rmsPosVec2.Peek()[cnt * 3 + 0],
                    this->rmsPosVec2.Peek()[cnt * 3 + 1],
                    this->rmsPosVec2.Peek()[cnt * 3 + 2]);
        }
        this->rmsCentroid /= static_cast<float>(posCnt);
//        printf("posCnt %u\n", posCnt);

//        for (int cnt = 0; cnt < static_cast<int>(mol2->AtomCount()); ++cnt) {
//            this->rmsCentroid += Vec3f(
//                    mol2->AtomPositions()[cnt*3+0],
//                    mol2->AtomPositions()[cnt*3+1],
//                    mol2->AtomPositions()[cnt*3+2]);
//        }
//        this->rmsCentroid /= static_cast<float>(mol2->AtomCount());

//        printf("centroid2   %.10f %.10f %.10f\n",
//                this->rmsCentroid.PeekComponents()[0],
//                this->rmsCentroid.PeekComponents()[1],
//                this->rmsCentroid.PeekComponents()[2]);

        // Do actual RMSD calculations
        this->rmsMask.Validate(posCnt);
        this->rmsWeights.Validate(posCnt);
#pragma omp parallel for
        for (int a = 0; a < static_cast<int>(posCnt); ++a) {
            this->rmsMask.Peek()[a] = 1;
            this->rmsWeights.Peek()[a] = 1.0f;
        }

        float rotation[3][3], translation[3];

        this->rmsValue = CalculateRMS(posCnt, // Number of positions in each vector
                true,                       // Do fit positions
                2,                          // Save rotation/translation
                this->rmsWeights.Peek(),    // Weights for the particles
                this->rmsMask.Peek(),    // Which particles should be considered
                this->rmsPosVec2.Peek(),    // Vector to be fitted
                this->rmsPosVec1.Peek(),    // Vector
                rotation,                   // Saves the rotation matrix
                translation                 // Saves the translation vector
                );

        this->rmsTranslation.Set(translation[0], translation[1],
                translation[2]);
        this->rmsRotation.SetAt(0, 0, rotation[0][0]);
        this->rmsRotation.SetAt(0, 1, rotation[0][1]);
        this->rmsRotation.SetAt(0, 2, rotation[0][2]);
        this->rmsRotation.SetAt(1, 0, rotation[1][0]);
        this->rmsRotation.SetAt(1, 1, rotation[1][1]);
        this->rmsRotation.SetAt(1, 2, rotation[1][2]);
        this->rmsRotation.SetAt(2, 0, rotation[2][0]);
        this->rmsRotation.SetAt(2, 1, rotation[2][1]);
        this->rmsRotation.SetAt(2, 2, rotation[2][2]);

//        printf("translation %.10f %.10f %.10f\n", translation[0],translation[1],translation[2]);
//        printf("rotation %.10f %.10f %.10f \n %.10f %.10f %.10f \n %.10f %.10f %.10f\n",
//                rotation[0][0], rotation[0][1], rotation[0][2],
//                rotation[1][0], rotation[1][1], rotation[1][2],
//                rotation[2][0], rotation[2][1], rotation[2][2]);

        this->rmsRotationMatrix.SetAt(0, 0, rotation[0][0]);
        this->rmsRotationMatrix.SetAt(0, 1, rotation[0][1]);
        this->rmsRotationMatrix.SetAt(0, 2, rotation[0][2]);
        this->rmsRotationMatrix.SetAt(0, 3, 0.0f);
        this->rmsRotationMatrix.SetAt(1, 0, rotation[1][0]);
        this->rmsRotationMatrix.SetAt(1, 1, rotation[1][1]);
        this->rmsRotationMatrix.SetAt(1, 2, rotation[1][2]);
        this->rmsRotationMatrix.SetAt(2, 3, 0.0f);
        this->rmsRotationMatrix.SetAt(2, 0, rotation[2][0]);
        this->rmsRotationMatrix.SetAt(2, 1, rotation[2][1]);
        this->rmsRotationMatrix.SetAt(2, 2, rotation[2][2]);
        this->rmsRotationMatrix.SetAt(2, 3, 0.0f);
        this->rmsRotationMatrix.SetAt(3, 3, 1.0f);
    }

    // Check for sufficiently low rms value
    if (this->rmsValue > this->maxRMSVal) {
        printf("RMSD value bigger then max value (%f)\n", this->rmsValue);
        return false;
    }

    /* Fit atom positions of source structure */
    this->atomPosFitted.Validate(mol2->AtomCount() * 3);

    if (this->fittingMode != RMS_NONE) {

        // Rotate/translate positions
#pragma omp parallel for
        for (int a = 0; a < static_cast<int>(mol2->AtomCount()); ++a) {
            Vec3f pos(&mol2->AtomPositions()[3 * a]);
            pos -= this->rmsCentroid;
            pos = this->rmsRotation * pos;
            pos += this->rmsTranslation;
            this->atomPosFitted.Peek()[3 * a + 0] = pos.X();
            this->atomPosFitted.Peek()[3 * a + 1] = pos.Y();
            this->atomPosFitted.Peek()[3 * a + 2] = pos.Z();
        }
    } else {
#pragma omp parallel for
        for (int a = 0; a < static_cast<int>(mol2->AtomCount()); ++a) {
            this->atomPosFitted.Peek()[3 * a + 0] = mol2->AtomPositions()[3 * a
                    + 0];
            this->atomPosFitted.Peek()[3 * a + 1] = mol2->AtomPositions()[3 * a
                    + 1];
            this->atomPosFitted.Peek()[3 * a + 2] = mol2->AtomPositions()[3 * a
                    + 2];
        }
    }

//    for (int a = 0; a <  mol1->AtomCount(); ++a) {
//    printf("pos1 %f %f %f, pos2 (fitted) %f %f %f\n",
//            mol1->AtomPositions()[3*a+0],
//            mol1->AtomPositions()[3*a+1],
//            mol1->AtomPositions()[3*a+2],
//            this->atomPosFitted.Peek()[3*a+0],
//            this->atomPosFitted.Peek()[3*a+1],
//            this->atomPosFitted.Peek()[3*a+2]);
//    }

    this->centroid.Set(0.0, 0.0, 0.0);
    for (int cnt = 0; cnt < static_cast<int>(posCnt); ++cnt) {
        this->centroid += Vec3f(this->atomPosFitted.Peek()[cnt * 3 + 0],
                this->atomPosFitted.Peek()[cnt * 3 + 1],
                this->atomPosFitted.Peek()[cnt * 3 + 2]);
    }
    this->centroid /= static_cast<float>(posCnt);

    return true;
}

/*
 * ComparativeMolSurfaceRenderer::GetExtents
 */
bool ComparativeMolSurfaceRenderer::GetExtents(core::Call& call) {
    core::view::CallRender3D *cr3d =
            dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    // Get pointer to potential map data call
    protein_calls::VTIDataCall *cmd0 =
            this->volDataSlot1.CallAs<protein_calls::VTIDataCall>();
    if (cmd0 == NULL) {
        return false;
    }
    if (!(*cmd0)(protein_calls::VTIDataCall::CallForGetExtent)) {
        return false;
    }
    protein_calls::VTIDataCall *cmd1 =
            this->volDataSlot2.CallAs<protein_calls::VTIDataCall>();
    if (cmd1 == NULL) {
        return false;
    }
    if (!(*cmd1)(protein_calls::VTIDataCall::CallForGetExtent)) {
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

    // Get a pointer to the outgoing render calls
    core::view::CallRender3D *ren1 = this->rendererCallerSlot1.CallAs<
            core::view::CallRender3D>();
    if (ren1 != NULL) {
        if (!(*ren1)(1)) {
            return false;
        }
    }
    core::view::CallRender3D *ren2 = this->rendererCallerSlot2.CallAs<
            core::view::CallRender3D>();
    if (ren2 != NULL) {
        if (!(*ren2)(1)) {
            return false;
        }
    }

//    core::BoundingBoxes bboxPotential0 = cmd0->AccessBoundingBoxes();
    core::BoundingBoxes bboxPotential1 = cmd1->AccessBoundingBoxes();

    core::BoundingBoxes bbox_external1, bbox_external2;
    if (ren1 != NULL) {
        bbox_external1 = ren1->AccessBoundingBoxes();
    }
    if (ren2 != NULL) {
        bbox_external2 = ren2->AccessBoundingBoxes();
    }

    // Calc union of all bounding boxes
    vislib::math::Cuboid<float> bboxTmp;

    //bboxTmp = cmd0->AccessBoundingBoxes().ObjectSpaceBBox();
    //bboxTmp.Union(cmd1->AccessBoundingBoxes().ObjectSpaceBBox());
    bboxTmp = mol0->AccessBoundingBoxes().ObjectSpaceBBox();
    bboxTmp.Union(mol1->AccessBoundingBoxes().ObjectSpaceBBox());
    if (ren1 != NULL) {
        bboxTmp.Union(bbox_external1.ObjectSpaceBBox());
    }
    if (ren2 != NULL) {
        bboxTmp.Union(bbox_external2.ObjectSpaceBBox());
    }
    this->bbox.SetObjectSpaceBBox(bboxTmp);

//    bboxTmp = cmd0->AccessBoundingBoxes().ObjectSpaceClipBox();
//    bboxTmp.Union(cmd1->AccessBoundingBoxes().ObjectSpaceClipBox());
    bboxTmp = mol0->AccessBoundingBoxes().ObjectSpaceClipBox();
    bboxTmp.Union(mol1->AccessBoundingBoxes().ObjectSpaceClipBox());
    if (ren1 != NULL) {
        bboxTmp.Union(bbox_external1.ObjectSpaceClipBox());
    }
    if (ren2 != NULL) {
        bboxTmp.Union(bbox_external2.ObjectSpaceClipBox());
    }
    this->bbox.SetObjectSpaceClipBox(bboxTmp);

//    bboxTmp = cmd0->AccessBoundingBoxes().WorldSpaceBBox();
//    bboxTmp.Union(cmd1->AccessBoundingBoxes().WorldSpaceBBox());
    bboxTmp = mol0->AccessBoundingBoxes().WorldSpaceBBox();
    bboxTmp.Union(mol1->AccessBoundingBoxes().WorldSpaceBBox());
    if (ren1 != NULL) {
        bboxTmp.Union(bbox_external1.WorldSpaceBBox());
    }
    if (ren2 != NULL) {
        bboxTmp.Union(bbox_external2.WorldSpaceBBox());
    }
    this->bbox.SetWorldSpaceBBox(bboxTmp);

//    bboxTmp = cmd0->AccessBoundingBoxes().WorldSpaceClipBox();
//    bboxTmp.Union(cmd1->AccessBoundingBoxes().WorldSpaceClipBox());
    bboxTmp = mol0->AccessBoundingBoxes().WorldSpaceClipBox();
    bboxTmp.Union(mol1->AccessBoundingBoxes().WorldSpaceClipBox());
    if (ren1 != NULL) {
        bboxTmp.Union(bbox_external1.WorldSpaceClipBox());
    }
    if (ren2 != NULL) {
        bboxTmp.Union(bbox_external2.WorldSpaceClipBox());
    }
    this->bbox.SetWorldSpaceClipBox(bboxTmp);

    float scale;
    if (!vislib::math::IsEqual(this->bbox.ObjectSpaceBBox().LongestEdge(),
            0.0f)) {
        scale = 2.0f / this->bbox.ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    float scale2;
    if (!vislib::math::IsEqual(
            mol0->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(),
            0.0f)) {
        scale2 = 2.0f
                / mol0->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale2 = 1.0f;
    }

    this->bbox.SetObjectSpaceBBox(
            mol0->AccessBoundingBoxes().ObjectSpaceBBox());
    this->bbox.MakeScaledWorld(scale2);
#ifndef USE_PROCEDURAL_DATA
    cr3d->AccessBoundingBoxes() = mol0->AccessBoundingBoxes();
#else // USE_PROCEDURAL DATA
    cr3d->AccessBoundingBoxes().SetObjectSpaceBBox(this->bboxParticles);
#endif // USE_PROCEDURAL DATA
    cr3d->AccessBoundingBoxes().MakeScaledWorld(scale2);

    // The available frame count is determined by the 'compareFrames' parameter
    if (this->cmpMode == COMPARE_1_1) {
        // One by one frame comparison
        cr3d->SetTimeFramesCount(1);
    } else if (this->cmpMode == COMPARE_1_N) {
        // One frame of data set #0 is compared to all frames of data set #1
        cr3d->SetTimeFramesCount(
                std::min(cmd1->FrameCount(), mol1->FrameCount()));
    } else if (this->cmpMode == COMPARE_N_1) {
        // One frame of data set #1 is compared to all frames of data set #0
        cr3d->SetTimeFramesCount(
                std::min(cmd0->FrameCount(), mol0->FrameCount()));
    } else if (this->cmpMode == COMPARE_N_N) {
        // Frame by frame comparison
        // Note: The data set with more frames is truncated
        cr3d->SetTimeFramesCount(
                std::min(std::min(cmd0->FrameCount(), mol0->FrameCount()),
                        std::min(cmd1->FrameCount(), mol1->FrameCount())));
    } else if (this->cmpMode == COMPARE_1_MORPH) {
        // Show morphing through time (good for videos)
        cr3d->SetTimeFramesCount(2000); // here, frames count is the maximum number of iterations
    } else if (this->cmpMode == COMPARE_1_ROT) {
        // Show rotating through time (good for videos)
        cr3d->SetTimeFramesCount(2000); // here, frames count is the maximum number of iterations
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
 * ComparativeMolSurfaceRenderer::getVBOExtent
 */
bool ComparativeMolSurfaceRenderer::getVBOExtent(core::Call& call) {

    VBODataCall *c = dynamic_cast<VBODataCall*>(&call);

    if (c == NULL) {
        return false;
    }

    c->SetBBox(this->bbox);

    // Get pointer to potential map data call
    protein_calls::VTIDataCall *cmd0 =
            this->volDataSlot1.CallAs<protein_calls::VTIDataCall>();
    if (cmd0 == NULL) {
        return false;
    }
    if (!(*cmd0)(protein_calls::VTIDataCall::CallForGetExtent)) {
        return false;
    }
    protein_calls::VTIDataCall *cmd1 =
            this->volDataSlot2.CallAs<protein_calls::VTIDataCall>();
    if (cmd1 == NULL) {
        return false;
    }
    if (!(*cmd1)(protein_calls::VTIDataCall::CallForGetExtent)) {
        return false;
    }

    // The available frame count is determined by the 'compareFrames' parameter
    if (this->cmpMode == COMPARE_1_1) {
        // One by one frame comparison
        c->SetFrameCnt(1);
    } else if (this->cmpMode == COMPARE_1_N) {
        // One frame of data set #0 is compared to all frames of data set #1
        c->SetFrameCnt(cmd1->FrameCount());
    } else if (this->cmpMode == COMPARE_N_1) {
        // One frame of data set #1 is compared to all frames of data set #0
        c->SetFrameCnt(cmd0->FrameCount());
    } else if (this->cmpMode == COMPARE_N_N) {
        // Frame by frame comparison
        // Note: The data set with lesser frames is truncated
        c->SetFrameCnt(std::min(cmd0->FrameCount(), cmd1->FrameCount()));
    } else if (this->cmpMode == COMPARE_1_MORPH) {
        // Frame by frame comparison
        // Note: The data set with lesser frames is truncated
        c->SetFrameCnt(1);
    } else if (this->cmpMode == COMPARE_1_ROT) {
        // Frame by frame comparison
        // Note: The data set with lesser frames is truncated
        c->SetFrameCnt(1);
    } else {
        return false; // Invalid compare mode
    }

    return true;
}

/*
 * ComparativeMolSurfaceRenderer::getVBOData1
 */
bool ComparativeMolSurfaceRenderer::getVBOData1(core::Call& call) {
    VBODataCall *c = dynamic_cast<VBODataCall*>(&call);

    if (c == NULL) {
        return false;
    }

    // Set vertex data
    c->SetVbo(this->deformSurf1.GetVtxDataVBO());
    c->SetDataStride(static_cast<int>(DeformableGPUSurfaceMT::vertexDataStride));
	c->SetDataOffs(static_cast<int>(DeformableGPUSurfaceMT::vertexDataOffsPos),
			static_cast<int>(DeformableGPUSurfaceMT::vertexDataOffsNormal),
			static_cast<int>(DeformableGPUSurfaceMT::vertexDataOffsTexCoord));
	c->SetVertexCnt(static_cast<unsigned int>(this->deformSurf1.GetVertexCnt()));

    // Set triangles
    c->SetVboTriangleIdx(this->deformSurf1.GetTriangleIdxVBO());
	c->SetTriangleCnt(static_cast<unsigned int>(this->deformSurf1.GetTriangleCnt()));

    // Set potential texture
    c->SetTex(this->surfAttribTex1);
    c->SetTexValRange(this->minPotential, this->maxPotential);

    return true;
}

/*
 * ComparativeMolSurfaceRenderer::getVBOData2
 */
bool ComparativeMolSurfaceRenderer::getVBOData2(core::Call& call) {

    VBODataCall *c = dynamic_cast<VBODataCall*>(&call);

    if (c == NULL) {
        return false;
    }

    // Set vertex data
    c->SetVbo(this->deformSurf2.GetVtxDataVBO());
	c->SetDataStride(static_cast<int>(DeformableGPUSurfaceMT::vertexDataStride));
	c->SetDataOffs(static_cast<int>(DeformableGPUSurfaceMT::vertexDataOffsPos),
			static_cast<int>(DeformableGPUSurfaceMT::vertexDataOffsNormal),
			static_cast<int>(DeformableGPUSurfaceMT::vertexDataOffsTexCoord));
	c->SetVertexCnt(static_cast<unsigned int>(this->deformSurf2.GetVertexCnt()));

    // Set triangles
    c->SetVboTriangleIdx(this->deformSurf2.GetTriangleIdxVBO());
	c->SetTriangleCnt(static_cast<unsigned int>(this->deformSurf2.GetTriangleCnt()));

    // Set potential texture
    c->SetTex(this->surfAttribTex2);
    c->SetTexValRange(this->minPotential, this->maxPotential);

    return true;
}

/*
 * ComparativeSurfacePotentialRenderer::initPotentialMap
 */
bool ComparativeMolSurfaceRenderer::initPotentialMap(VTIDataCall *cmd,
        gridParams &gridPotentialMap, GLuint &potentialTex,
        CudaDevArr<float> &tex_D, int3 &dim, float3 &org, float3 &delta) {
    using namespace vislib::sys;

    // Setup grid parameters
    gridPotentialMap.minC[0] =
            cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetLeft();
    gridPotentialMap.minC[1] =
            cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetBottom();
    gridPotentialMap.minC[2] =
            cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetBack();
    gridPotentialMap.maxC[0] =
            cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetRight();
    gridPotentialMap.maxC[1] =
            cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetTop();
    gridPotentialMap.maxC[2] =
            cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetFront();
    gridPotentialMap.size[0] = cmd->GetGridsize().X();
    gridPotentialMap.size[1] = cmd->GetGridsize().Y();
    gridPotentialMap.size[2] = cmd->GetGridsize().Z();
    gridPotentialMap.delta[0] = cmd->GetSpacing().X();
    gridPotentialMap.delta[1] = cmd->GetSpacing().Y();
    gridPotentialMap.delta[2] = cmd->GetSpacing().Z();

    dim.x = cmd->GetGridsize().X();
    dim.y = cmd->GetGridsize().Y();
    dim.z = cmd->GetGridsize().Z();

    org.x = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetLeft();
    org.y = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetBottom();
    org.z = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetBack();

    delta.x = cmd->GetSpacing().X();
    delta.y = cmd->GetSpacing().Y();
    delta.z = cmd->GetSpacing().Z();

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

//    printf("GRIDSIZE SHOULD BE %u\n",  gridPotentialMap.size[0]* gridPotentialMap.size[1]* gridPotentialMap.size[2]);

//    if (cmd->GetPointDataByIdx(0, 0) == NULL) {
//        return false;
//    } else {
//        for (int i = 0; i < cmd->GetGridsize().X()*cmd->GetGridsize().Y()*cmd->GetGridsize().Z(); ++i) {
//            if (i%1000 == 0)
//            printf("potential map %i: %f\n", i, ((float*)cmd->GetPointDataByIdx(0, 0))[i]);
//        }
//    }

    if (!CudaSafeCall(tex_D.Validate(cmd->GetGridsize().X()*cmd->GetGridsize().Y()*cmd->GetGridsize().Z()))) {
        return false;
    }
    if (!CudaSafeCall(cudaMemcpy(tex_D.Peek(), cmd->GetPointDataByIdx(0, 0),
                    sizeof(float)*tex_D.GetCount(), cudaMemcpyHostToDevice))) {
        return false;
    }

    //  Setup texture
    glEnable(GL_TEXTURE_3D);
    if (!glIsTexture(potentialTex)) {
        glGenTextures(1, &potentialTex);
    }

    glBindTexture(GL_TEXTURE_3D, potentialTex);

    glTexImage3DEXT(GL_TEXTURE_3D, 0, GL_RGBA32F, cmd->GetGridsize().X(),
            cmd->GetGridsize().Y(), cmd->GetGridsize().Z(), 0, GL_ALPHA,
            GL_FLOAT, cmd->GetPointDataByIdx(0, 0));

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
    if (this->cudaqsurf1 != NULL) {
        CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf1;
        delete cqs;
    }
    if (this->cudaqsurf2 != NULL) {
        CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf2;
        delete cqs;
    }
    this->deformSurf1.Release();
    this->deformSurf2.Release();
    this->deformSurfMapped.Release();

#ifdef USE_PROCEDURAL_DATA
    this->procField1D.Release();
    this->procField2D.Release();
    this->procField1.Release();
    this->procField2.Release();
#endif // USE_PROCEDURAL_DATA
    this->gridDataPos.Release();

    this->pplSurfaceShader.Release();
    this->pplSurfaceShaderVertexFlag.Release();
    this->pplSurfaceShaderUncertainty.Release();
    this->pplMappedSurfaceShader.Release();

    CudaSafeCall(this->surfAttribTex1_D.Release());
    CudaSafeCall(this->surfAttribTex2_D.Release());

    this->rmsPosVec1.Release();
    this->rmsPosVec2.Release();
    this->rmsWeights.Release();
    this->rmsMask.Release();

    this->atomPosFitted.Release();

    this->gvf.Release();
    this->mappedSurfVertexFlags.Release();
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

    if (this->oldCalltime != calltime) {
        this->triggerRMSFit = true;
        this->triggerComputeVolume = true;
        this->triggerInitPotentialTex = true;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerSurfaceMapping = true;
        this->triggerComputeLines = true;
    }
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
    } else if (this->cmpMode == COMPARE_1_MORPH) {
        // Show morphing
        frameIdx1 = this->singleFrame1;
        frameIdx2 = this->singleFrame2;
    } else if (this->cmpMode == COMPARE_1_ROT) {
        // Show morphing
        frameIdx1 = this->singleFrame1;
        frameIdx2 = this->singleFrame2;
    } else {
        return false; // Invalid compare mode
    }

    //printf("Frame1 %u, frame2 %u\n", frameIdx1, frameIdx2);

    // Get surface texture of data set #1
    VTIDataCall *vti1 = this->volDataSlot1.CallAs<VTIDataCall>();
    if (vti1 == NULL) {
        return false;
    }
    vti1->SetCalltime(static_cast<float>(frameIdx1));  // Set call time
    vti1->SetFrameID(frameIdx1, true);  // Set frame ID and call data
    if (!(*vti1)(VTIDataCall::CallForGetData)) {
        return false;
    }

    VTIDataCall *vti2 = this->volDataSlot2.CallAs<VTIDataCall>();
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
    if ((this->triggerRMSFit) || (mol1->DataHash() != this->datahashParticles1)
            || (mol2->DataHash() != this->datahashParticles2)
            || (calltime != this->calltimeOld)) {

#ifdef USE_TIMER
        t = clock();
#endif

        this->datahashParticles1 = mol1->DataHash();
        this->datahashParticles2 = mol2->DataHash();
        this->calltimeOld = calltime;

        if (!this->fitMoleculeRMS(mol1, mol2)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute RMSD fitting", this->ClassName());

            return false;
        }

#ifdef USE_TIMER
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: Time for RMS fitting: %.6f s",
                this->ClassName(),
                (double(clock()-t)/double(CLOCKS_PER_SEC)));
#endif

        // (Re)-compute bounding box
        this->computeDensityBBox(mol1->AtomPositions(),
                this->atomPosFitted.Peek(), mol1->AtomCount(),
                mol2->AtomCount());

        this->triggerRMSFit = false;
        this->triggerComputeVolume = true;
    }

    // (Re-)compute volume texture if necessary
    if (this->triggerComputeVolume) {

        if (!this->computeDensityMap(mol1->AtomPositions(), mol1,
                (CUDAQuickSurf*) this->cudaqsurf1)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute density map #1", this->ClassName());

            return false;
        }

        if (!this->computeDensityMap(this->atomPosFitted.Peek(), mol2,
                (CUDAQuickSurf*) this->cudaqsurf2)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute density map #2", this->ClassName());

            return false;
        }

#ifdef USE_PROCEDURAL_DATA
        if (!this->initProcFieldData()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute procedural field data",
                    this->ClassName());
            return false;
        }
#endif

        this->triggerComputeVolume = false;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
//
//        printf("DENSITY MAP DIM %u %u %u\n", this->volDim.x, this->volDim.y, this->volDim.z);
//        printf("DENSITY MAP ORG %f %f %f\n", this->volOrg.x, this->volOrg.y, this->volOrg.z);
//        printf("DENSITY MAP DELTA %f %f %f\n", this->volDelta.x, this->volDelta.y, this->volDelta.z);
    }

    // (Re-)compute potential texture if necessary
    if ((this->triggerInitPotentialTex)
            || (vti1->DataHash() != this->datahashPotential1)
            || (vti2->DataHash() != this->datahashPotential2)
            || (calltime != this->calltimeOld)) {

        this->datahashPotential1 = vti1->DataHash();
        this->datahashPotential2 = vti2->DataHash();

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: init potential texture #1",
                this->ClassName());
#endif // VERBOSE
        if (!this->initPotentialMap(vti1, this->gridPotential1,
                this->surfAttribTex1, this->surfAttribTex1_D, this->texDim1,
                this->texOrg1, this->texDelta1)) {

#ifdef VERBOSE
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not init potential map #1",
                    this->ClassName());
#endif // VERBOSE
            return false;
        }

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: init potential texture #2",
                this->ClassName());
#endif // VERBOSE
        if (!this->initPotentialMap(vti2, this->gridPotential2,
                this->surfAttribTex2, this->surfAttribTex2_D, this->texDim2,
                this->texOrg2, this->texDelta2)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not init potential map #2", this->ClassName());

            return false;
        }
        this->triggerInitPotentialTex = false;
    }

    // (Re-)compute triangle mesh for surface #1 using Marching tetrahedra
    if (this->triggerComputeSurfacePoints1) {

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: compute surface points #1",
                this->ClassName());
#endif // VERBOSE
        ::CheckForCudaErrorSync();

        /* Surface #1 */

        // Get vertex positions based on the level set
        if (!this->deformSurf1.ComputeVertexPositions(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf1)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField1D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->volDim, this->volOrg, this->volDelta, this->qsIsoVal)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute vertex positions #1",
                    this->ClassName());

            return false;
        }

////        this->deformSurf1.PrintCubeStates((this->volDim.x-1)*(this->volDim.y-1)*(this->volDim.z-1));
//        // DEBUG Print voltarget_D
//        HostArr<float> volTarget;
//        size_t gridSize = volDim.x*volDim.y*volDim.z;
//        volTarget.Validate(gridSize);
//        CudaSafeCall(cudaMemcpy(volTarget.Peek(),
//                ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
//                sizeof(float)*gridSize,
//                cudaMemcpyDeviceToHost));
//        for (int i = 0; i < gridSize; ++i) {
//            printf("VOL %.16f\n", volTarget.Peek()[i]);
//        }
//        volTarget.Release();
//        // END DEBUG

        ::CheckForCudaErrorSync();

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: compute triangles #1",
                this->ClassName());
#endif // VERBOSE
        // Build triangle mesh from vertices
        if (!this->deformSurf1.ComputeTriangles(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf1)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField1D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->volDim, this->volOrg, this->volDelta, this->qsIsoVal)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute vertex triangles #1",
                    this->ClassName());

            return false;
        }

        CheckForCudaErrorSync();

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: compute connectivity points #1",
                this->ClassName());
#endif // VERBOSE
        // Compute vertex connectivity
        if (!this->deformSurf1.ComputeConnectivity(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf1)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField1D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->volDim, this->volOrg, this->volDelta, this->qsIsoVal)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute vertex connectivity #1",
                    this->ClassName());

            return false;
        }

        CheckForCudaErrorSync();

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: regularize surface #1",
                this->ClassName());
#endif // VERBOSE
        // Regularize the mesh of surface #1
        if (!this->deformSurf1.MorphToVolumeGradient(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf1)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField1D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->volDim, this->volOrg, this->volDelta, this->qsIsoVal,
                this->interpolMode, this->regMaxIt, this->surfregMinDisplScl,
                this->regSpringStiffness, this->regForcesScl,
                this->regExternalForcesWeight)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not regularize surface #1", this->ClassName());

            return false;
        }

        CheckForCudaErrorSync();

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: compute normals #1",
                this->ClassName());
#endif // VERBOSE
        // Compute vertex normals
        if (!this->deformSurf1.ComputeNormals(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf1)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField1D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->volDim, this->volOrg, this->volDelta, this->qsIsoVal)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute normals #1", this->ClassName());

            return false;
        }

        CheckForCudaErrorSync();

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: compute texture coordinates #1",
                this->ClassName());
#endif // VERBOSE
        // Compute texture coordinates
        if (!this->deformSurf1.ComputeTexCoords(this->gridPotential1.minC,
                this->gridPotential1.maxC)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute tex coords #1", this->ClassName());

            return false;
        }

        this->triggerComputeSurfacePoints1 = false;
    }

    // (Re-)compute triangle mesh for surface #2 using Marching tetrahedra
    if (this->triggerComputeSurfacePoints2) {

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: compute vertex positions #2",
                this->ClassName());
#endif // VERBOSE
        /* Surface #2 */

        // Get vertex positions based on the level set
        if (!this->deformSurf2.ComputeVertexPositions(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf2)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField2D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->volDim, this->volOrg, this->volDelta, this->qsIsoVal)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute vertex positions #2",
                    this->ClassName());

            return false;
        }

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: compute triangles #2",
                this->ClassName());
#endif // VERBOSE
        // Build triangle mesh from vertices
        if (!this->deformSurf2.ComputeTriangles(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf2)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField2D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->volDim, this->volOrg, this->volDelta, this->qsIsoVal)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute triangles #2", this->ClassName());

            return false;
        }

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: compute vertex connectivity #2",
                this->ClassName());
#endif // VERBOSE
        // Compute vertex connectivity
        if (!this->deformSurf2.ComputeConnectivity(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf2)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField2D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->volDim, this->volOrg, this->volDelta, this->qsIsoVal)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute vertex connectivity #2",
                    this->ClassName());

            return false;
        }

        if (!this->deformSurf2.ComputeTriangleNeighbors(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf2)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField2D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->volDim, this->volOrg, this->volDelta, this->qsIsoVal)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute triangle neighborsd #2",
                    this->ClassName());
            return false;
        }

        // Compute edge list (this needs to be done before any deformation happens
        if (!this->deformSurf2.ComputeEdgeList(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf2)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField2D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->qsIsoVal, this->volDim, this->volOrg, this->volDelta)) {
#ifdef VERBOSE
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute edge list",
                    this->ClassName());
#endif // VERBOSE
            return false;
        }

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: regularize surface #2",
                this->ClassName());
#endif // VERBOSE
        // Regularize the mesh of surface #2
        if (!this->deformSurf2.MorphToVolumeGradient(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf2)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField2D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->volDim, this->volOrg, this->volDelta, this->qsIsoVal,
                this->interpolMode, this->regMaxIt, this->surfregMinDisplScl,
                this->regSpringStiffness, this->regForcesScl,
                this->regExternalForcesWeight)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not regularize surface #2", this->ClassName());

            return false;
        }

//        // DEBUG print positions of regularized surface
//        this->deformSurf2.PrintVertexBuffer(15);
//        // END DEBUG

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: compute normals #2",
                this->ClassName());
#endif // VERBOSE
        // Compute vertex normals
        if (!this->deformSurf2.ComputeNormals(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf2)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField2D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->volDim, this->volOrg, this->volDelta, this->qsIsoVal)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute vertex normals #2",
                    this->ClassName());

            return false;
        }

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: compute texture coordinates #2",
                this->ClassName());
#endif // VERBOSE
        // Compute texture coordinates
        Mat3f rmsRotInv(this->rmsRotation);
        if (!rmsRotInv.Invert()) {
            printf("Could not invert rotation matrix\n");
            return false;
        }
        if (!this->deformSurf2.ComputeTexCoordsOfRMSDFittedPositions(
                this->gridPotential2.minC, this->gridPotential2.maxC,
                this->rmsCentroid.PeekComponents(), rmsRotInv.PeekComponents(),
                this->rmsTranslation.PeekComponents())) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute tex coords #2", this->ClassName());

            return false;
        }

        this->triggerComputeSurfacePoints2 = false;
        this->triggerSurfaceMapping = true;
    }

    /* Map surface #2 to surface #1 */

    if (this->triggerSurfaceMapping) {

        // Make deep copy of regularized second surface
        this->deformSurfMapped = this->deformSurf2;

        if (this->surfMappedExtForce == GVF) {

#ifdef VERBOSE
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                    "%s: morph to volume gvf",
                    this->ClassName());
#endif // VERBOSE
            // Morph surface #2 to shape #1 using GVF
            if (!this->deformSurfMapped.MorphToVolumeGVF(
#ifndef  USE_PROCEDURAL_DATA
                    ((CUDAQuickSurf*) this->cudaqsurf2)->getMap(),
#else //  USE_PROCEDURAL_DATA
                    this->procField2D.Peek(),
#endif //  USE_PROCEDURAL_DATA
#ifndef  USE_PROCEDURAL_DATA
                    ((CUDAQuickSurf*) this->cudaqsurf1)->getMap(),
#else //  USE_PROCEDURAL_DATA
                    this->procField1D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                    this->deformSurf1.PeekCubeStates(), this->volDim,
                    this->volOrg, this->volDelta, this->qsIsoVal,
                    this->interpolMode,
                    ((this->cmpMode == COMPARE_1_MORPH) ?
                            (int) (calltime) : this->surfaceMappingMaxIt),
                    this->surfMappedMinDisplScl,
                    this->surfMappedSpringStiffness,
                    this->surfaceMappingForcesScl,
                    this->surfaceMappingExternalForcesWeightScl,
                    this->surfMappedGVFScl, this->surfMappedGVFIt)) {

                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: could not compute GVF deformation",
                        this->ClassName());

                return false;
            }
        } else if (this->surfMappedExtForce == METABALLS) {

#ifdef VERBOSE
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                    "%s: morph to volume meta balls",
                    this->ClassName());
#endif // VERBOSE
            // Morph surface #2 to shape #1 using implicit molecular surface
            if (!this->deformSurfMapped.MorphToVolumeGradient(
#ifndef  USE_PROCEDURAL_DATA
                    ((CUDAQuickSurf*) this->cudaqsurf1)->getMap(),
#else //  USE_PROCEDURAL_DATA
                    this->procField1D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                    this->volDim, this->volOrg, this->volDelta, this->qsIsoVal,
                    this->interpolMode,
                    ((this->cmpMode == COMPARE_1_MORPH) ?
                            (int) (calltime) : this->surfaceMappingMaxIt),
                    this->surfMappedMinDisplScl,
                    this->surfMappedSpringStiffness,
                    this->surfaceMappingForcesScl,
                    this->surfaceMappingExternalForcesWeightScl)) {

                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: could not compute metaballs deformation",
                        this->ClassName());

                return false;
            }
        } else if (this->surfMappedExtForce == METABALLS_DISTFIELD) {

#ifdef VERBOSE
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                    "%s: morph to volume distfield",
                    this->ClassName());
#endif // VERBOSE
            // Morph surface #2 to shape #1 using implicit molecular surface + distance field
            if (!this->deformSurfMapped.MorphToVolumeDistfield(
#ifndef  USE_PROCEDURAL_DATA
                    ((CUDAQuickSurf*) this->cudaqsurf1)->getMap(),
#else //  USE_PROCEDURAL_DATA
                    this->procField1D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                    this->volDim, this->volOrg, this->volDelta, this->qsIsoVal,
                    this->interpolMode,
                    ((this->cmpMode == COMPARE_1_MORPH) ?
                            (int) (calltime) : this->surfaceMappingMaxIt),
                    this->surfMappedMinDisplScl,
                    this->surfMappedSpringStiffness,
                    this->surfaceMappingForcesScl,
                    this->surfaceMappingExternalForcesWeightScl,
                    this->distFieldThresh)) {

                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: could not compute metaballs/distfield deformation",
                        this->ClassName());

                return false;
            }
        } else if (this->surfMappedExtForce == TWO_WAY_GVF) {

#ifdef VERBOSE
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                    "%s: morph to volume two-way-gvf",
                    this->ClassName());
#endif // VERBOSE
            // Morph surface #2 to shape #1 using Two-Way-GVF

//            // DEBUG Print voltarget_D
//            HostArr<float> volTarget;
//            size_t gridSize = volDim.x*volDim.y*volDim.z;
//            volTarget.Validate(gridSize);
//            CudaSafeCall(cudaMemcpy(volTarget.Peek(),
//                    ((CUDAQuickSurf*)this->cudaqsurf2)->getMap(),
//                    sizeof(float)*gridSize,
//                    cudaMemcpyDeviceToHost));
//
//            for (int i = 0; i < gridSize; ++i) {
//                printf("VOL %.16f\n", volTarget.Peek()[i]);
//            }
//            volTarget.Release();
//            // END DEBUG

//            // DEBUG print external forces
//            this->deformSurfMapped.PrintExternalForces(volDim.x*volDim.y*volDim.z);
//            // END DEBUG

//            this->deformSurf1.PrintCubeStates((volDim.x-1)*(volDim.y-1)*(volDim.z-1));

            if (!this->deformSurfMapped.MorphToVolumeTwoWayGVF(
#ifndef  USE_PROCEDURAL_DATA
                    ((CUDAQuickSurf*) this->cudaqsurf2)->getMap(),
#else //  USE_PROCEDURAL_DATA
                    this->procField2D.Peek(),
#endif //  USE_PROCEDURAL_DATA
#ifndef  USE_PROCEDURAL_DATA
                    ((CUDAQuickSurf*) this->cudaqsurf1)->getMap(),
#else //  USE_PROCEDURAL_DATA
                    this->procField1D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                    this->deformSurf2.PeekCubeStates(),
                    this->deformSurf1.PeekCubeStates(), this->volDim,
                    this->volOrg, this->volDelta, this->qsIsoVal,
                    this->interpolMode,
                    ((this->cmpMode == COMPARE_1_MORPH) ?
                            (int) (calltime) : this->surfaceMappingMaxIt),
                    this->surfMappedMinDisplScl,
                    this->surfMappedSpringStiffness,
                    this->surfaceMappingForcesScl,
                    this->surfaceMappingExternalForcesWeightScl,
                    this->surfMappedGVFScl, this->surfMappedGVFIt, true,
                    true)) {

                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: could not compute Two-Way-GVF deformation",
                        this->ClassName());

                return false;
            }

//            // DEBUG print positions of regularized surface
//            this->deformSurfMapped.PrintVertexBuffer(15);
//            // END DEBUG

//            printf("volOrg %f %f %f\n", volOrg.x, volOrg.y, volOrg.z);
//             printf("volDelta %f %f %f\n", volDelta.x, volDelta.y, volDelta.z);
//             printf("volDim %u %u %u\n", volDim.x, volDim.x, volDim.z);

        } else {
            return false;
        }

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: compute normals of mapped surface",
                this->ClassName());
#endif // VERBOSE
        // Perform subdivision with subsequent deformation to create a fine
        // target mesh enough
        if (((this->cmpMode == COMPARE_1_MORPH) ?
                (int) (calltime) : this->surfaceMappingMaxIt) > 0) {
            int newTris;
            for (int i = 0; i < this->maxSubdivLevel; ++i) {
                for (int j = 0; j < this->subStepLevel; ++j) {

//               printf("SUBDIV #%i\n", i);

                    newTris = this->deformSurfMapped.RefineMesh(1,
#ifndef  USE_PROCEDURAL_DATA
                            ((CUDAQuickSurf*) this->cudaqsurf2)->getMap(),
#else //  USE_PROCEDURAL_DATA
                            this->procField2D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                            this->volDim, this->volOrg, this->volDelta,
                            this->qsIsoVal, this->volDelta.x); // TODO Whats a good maximum edge length?

                    if (newTris < 0) {

                        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                                "%s: could not refine mesh", this->ClassName());

                        return false;
                    }
                    if (newTris == 0)
                        break;
                }

                // Perform morphing
                if (this->maxSubdivLevel > 0) {
                    // Morph surface #2 to shape #1 using Two-Way-GVF
                    if (!this->deformSurfMapped.MorphToVolumeTwoWayGVFSubdiv(
#ifndef  USE_PROCEDURAL_DATA
                            ((CUDAQuickSurf*) this->cudaqsurf2)->getMap(),
#else //  USE_PROCEDURAL_DATA
                            this->procField2D.Peek(),
#endif //  USE_PROCEDURAL_DATA
#ifndef  USE_PROCEDURAL_DATA
                            ((CUDAQuickSurf*) this->cudaqsurf1)->getMap(),
#else //  USE_PROCEDURAL_DATA
                            this->procField1D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                            this->deformSurf2.PeekCubeStates(),
                            this->deformSurf1.PeekCubeStates(), this->volDim,
                            this->volOrg, this->volDelta, this->qsIsoVal,
                            this->interpolMode, this->maxSubdivSteps,
                            this->surfMappedMinDisplScl,
                            this->surfMappedSpringStiffness,
                            this->surfaceMappingForcesScl,
                            this->surfaceMappingExternalForcesWeightScl,
                            this->surfMappedGVFScl, this->surfMappedGVFIt,
                            false, false)) {

                        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                                "%s: could not compute Two-Way-GVF deformation",
                                this->ClassName());

                        return false;
                    }
                }
            }
        }

        if (!this->deformSurfMapped.FlagCorruptTriangles(
#ifndef  USE_PROCEDURAL_DATA
                ((CUDAQuickSurf*) this->cudaqsurf1)->getMap(),
#else //  USE_PROCEDURAL_DATA
                this->procField1D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                this->deformSurf1.PeekCubeStates(), volDim, volOrg, volDelta,
                this->qsIsoVal)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not flag corrupt triangles", this->ClassName());

            return false;
        }

        // Copy vertex flags to host

        this->mappedSurfVertexFlags.Validate(
                this->deformSurfMapped.GetVertexCnt());
        this->mappedSurfVertexFlags.Set(0x00);
        if (this->maxSubdivLevel > 0) {
            if (!CudaSafeCall(cudaMemcpy(
                            this->mappedSurfVertexFlags.Peek(),
                            this->deformSurfMapped.PeekVertexFlagDevPt(),
                            sizeof(float)*this->deformSurfMapped.GetVertexCnt(),
                            cudaMemcpyDeviceToHost))) {
                return false;
            }
        }

        // Update vertex path VBO for new vertices
//        printf("BACKTRACKING\n");
        if (this->maxSubdivLevel > 0) {
            if (!this->deformSurfMapped.TrackPathSubdivVertices(
#ifndef  USE_PROCEDURAL_DATA
                    ((CUDAQuickSurf*) this->cudaqsurf2)->getMap(),
#else //  USE_PROCEDURAL_DATA
                    this->procField2D.Peek(),
#endif //  USE_PROCEDURAL_DATA
                    this->volDim, this->volOrg, this->volDelta,
                    this->surfaceMappingForcesScl, this->surfMappedMinDisplScl,
                    this->qsIsoVal,
                    ((this->cmpMode == COMPARE_1_MORPH) ?
                            (int) (calltime) : this->surfaceMappingMaxIt))) {
                return false;
            }
        }

        // Compute texture difference per vertex
        // Compute texture coordinates
        Mat3f rmsRotInv(this->rmsRotation);
        if (!rmsRotInv.Invert()) {
            printf("Could not invert rotation matrix\n");
            return false;
        }
        if (this->surfaceMappedColorMode == SURFACE_POTENTIAL_SIGN) {
            if (!this->deformSurfMapped.ComputeSurfAttribSignDiff(
                    this->deformSurf2,
                    this->rmsCentroid.PeekComponents(), // In case the start surface has been fitted using RMSD
                    rmsRotInv.PeekComponents(),
                    this->rmsTranslation.PeekComponents(),
                    this->surfAttribTex1_D.Peek(), this->texDim1, this->texOrg1,
                    this->texDelta1, this->surfAttribTex2_D.Peek(),
                    this->texDim2, this->texOrg2, this->texDelta2)) {
                return false;
            }
        } else if (this->surfaceMappedColorMode == SURFACE_LAPLACIAN) {
            // Compute mesh Laplacian
            if (!this->deformSurfMapped.ComputeMeshLaplacianDiff(
                    this->deformSurf2)) {
                printf("Could not compute mesh Laplacian\n");
                return false;
            }
        } else {
            if (!this->deformSurfMapped.ComputeSurfAttribDiff(this->deformSurf2,
                    this->rmsCentroid.PeekComponents(), // In case the start surface has been fitted using RMSD
                    rmsRotInv.PeekComponents(),
                    this->rmsTranslation.PeekComponents(),
                    this->surfAttribTex1_D.Peek(), this->texDim1, this->texOrg1,
                    this->texDelta1, this->surfAttribTex2_D.Peek(),
                    this->texDim2, this->texOrg2, this->texDelta2)) {
                return false;
            }
        }

#ifdef VERBOSE
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: compute texture coordinates of mapped surface",
                this->ClassName());
#endif // vERBOSE
        // Compute texture coordinates
        if (!this->deformSurfMapped.ComputeTexCoords(this->gridPotential1.minC,
                this->gridPotential1.maxC)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute tex coords of mapped surface",
                    this->ClassName());

            return false;
        }

        // Compute vertex normals
        if (!this->deformSurfMapped.ComputeNormalsSubdiv()) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not compute normals of mapped surface",
                    this->ClassName());

            return false;
        }

        this->triggerSurfaceMapping = false;
        this->triggerComputeSubdiv = true;
    }

    // Get camera information
    this->cameraInfo =
            dynamic_cast<core::view::AbstractCallRender3D*>(&call)->GetCameraParameters();

    /* Rendering of scene objects */

    glMatrixMode (GL_MODELVIEW);
    glPushMatrix();

    // Apply scaling based on combined bounding box
    float scaleCombined;
    if (!vislib::math::IsEqual(this->bbox.ObjectSpaceBBox().LongestEdge(),
            0.0f)) {
        scaleCombined = 2.0f / this->bbox.ObjectSpaceBBox().LongestEdge();
    } else {
        scaleCombined = 1.0f;
    }
    glScalef(scaleCombined, scaleCombined, scaleCombined);
    //printf("scale master %f\n", scaleCombined);

    if (this->cmpMode == COMPARE_1_ROT) {
        glTranslatef(this->centroid.X(),
                this->centroid.Y(),
                this->centroid.Z());
        glRotatef(calltime, 0.0, 1.0, 0.0);
        glTranslatef(-this->centroid.X(),
                -this->centroid.Y(),
                -this->centroid.Z());
    }

    // Call external renderer #1 if possible
    core::view::CallRender3D *ren1 = this->rendererCallerSlot1.CallAs<
            core::view::CallRender3D>();
    if (ren1 != NULL) {
        // Call additional renderer
        ren1->SetCameraParameters(this->cameraInfo);
        ren1->SetTime(static_cast<float>(frameIdx1));
        glPushMatrix();
        // Revert scaling done by external renderer in advance
        float scaleRevert;
        if (!vislib::math::IsEqual(
                ren1->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(),
                0.0f)) {
            scaleRevert =
                    2.0f
                            / ren1->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        } else {
            scaleRevert = 1.0f;
        }
        scaleRevert = 1.0f / scaleRevert;
        glScalef(scaleRevert, scaleRevert, scaleRevert);
        (*ren1)(0); // Render call to external renderer
        glPopMatrix();
        CheckForGLError();
    }

    // Call external renderer #2 if possible
    core::view::CallRender3D *ren2 = this->rendererCallerSlot2.CallAs<
            core::view::CallRender3D>();
    if (ren2 != NULL) {
        // Call additional renderer
        ren2->SetCameraParameters(this->cameraInfo);
        ren2->SetTime(static_cast<float>(frameIdx2));
        glPushMatrix();

        // Perform RMSD translation/rotation to fit the surface rendering
        if (this->fittingMode != RMS_NONE) {
            glTranslatef(this->rmsTranslation.X(), this->rmsTranslation.Y(),
                    this->rmsTranslation.Z());
            glMultMatrixf(this->rmsRotationMatrix.PeekComponents());
            glTranslatef(-this->rmsCentroid.X(), -this->rmsCentroid.Y(),
                    -this->rmsCentroid.Z());
        }

        // Revert scaling done by external renderer in advance
        float scaleRevert;
        if (!vislib::math::IsEqual(
                ren2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(),
                0.0f)) {
            scaleRevert =
                    2.0f
                            / ren2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        } else {
            scaleRevert = 1.0f;
        }
        scaleRevert = 1.0f / scaleRevert;
        glScalef(scaleRevert, scaleRevert, scaleRevert);

        (*ren2)(0); // Render call to external renderer
        glPopMatrix();
        CheckForGLError();
    }

    glEnable (GL_DEPTH_TEST);

    glEnable (GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable (GL_CULL_FACE);
    glDisable (GL_LIGHTING);
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

//    // DEBUG Render external forces as lines
//    if (!this->renderExternalForces()) {
//        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
//                "%s: could not render external forces",
//                this->ClassName());
//        return false;
//    }
//    // END DEBUG

    if (this->surface1RM != SURFACE_NONE) {

        // Sort triangles by camera distance
        if (!this->deformSurf1.SortTrianglesByCamDist(camPos)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not sort triangles #1", this->ClassName());
            return false;
        }

        // Render surface #1
        if (!this->renderSurface(this->deformSurf1.GetVtxDataVBO(),
                static_cast<uint>(this->deformSurf1.GetVertexCnt()),
                this->deformSurf1.GetTriangleIdxVBO(),
				static_cast<uint>(this->deformSurf1.GetTriangleCnt() * 3), this->surface1RM,
                this->surface1ColorMode, this->surfAttribTex1,
                this->uniformColorSurf1, this->surf1AlphaScl)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not render surface #1", this->ClassName());
            return false;
        }
    }

    if (this->surface2RM != SURFACE_NONE) {
        // Sort triangles by camera distance
        if (!this->deformSurf2.SortTrianglesByCamDist(camPos)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not sort triangles #2", this->ClassName());
            return false;
        }

        // Render surface #2
        if (!this->renderSurface(this->deformSurf2.GetVtxDataVBO(),
                static_cast<uint>(this->deformSurf2.GetVertexCnt()),
                this->deformSurf2.GetTriangleIdxVBO(),
				static_cast<uint>(this->deformSurf2.GetTriangleCnt() * 3), this->surface2RM,
                this->surface2ColorMode, this->surfAttribTex2,
                this->uniformColorSurf2, this->surf2AlphaScl)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not render surface #2", this->ClassName());
            return false;
        }

//        // Render mapped surface
//        if (!this->renderMappedSurface(
//                this->deformSurf2.GetVtxDataVBO(),
////                this->deformSurfMapped.GetVtxDataVBO(),
////                this->deformSurfMapped.GetVertexCnt(),
////                this->deformSurfMapped.GetTriangleIdxVBO(),
////                this->deformSurfMapped.GetTriangleCnt()*3,
//                this->deformSurf2.GetVtxDataVBO(),
//                 this->deformSurf2.GetVertexCnt(),
//                 this->deformSurf2.GetTriangleIdxVBO(),
//                 this->deformSurf2.GetTriangleCnt()*3,
//                this->surface2RM,
//                this->surfaceMappedColorMode)) {
//            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
//                    "%s: could not render mapped surface",
//                    this->ClassName());
//            return false;
//        }
    }

    if (this->surfaceMappedRM != SURFACE_NONE) {

        // Sort triangles by camera distance
        if (!this->deformSurfMapped.SortTrianglesByCamDist(camPos)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not sort triangles of mapped surface",
                    this->ClassName());
            return false;
        }

        // Render mapped surface
        if (!this->renderMappedSurface(this->deformSurf2.GetVtxDataVBO(),
                this->deformSurfMapped.GetVtxDataVBO(),
                static_cast<uint>(this->deformSurfMapped.GetVertexCnt()),
                this->deformSurfMapped.GetTriangleIdxVBO(),
				static_cast<uint>(this->deformSurfMapped.GetTriangleCnt() * 3),
                this->surfaceMappedRM, this->surfaceMappedColorMode)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: could not render mapped surface", this->ClassName());
            return false;
        }

//        // Render mapped surface as normal surface
//        if (!this->renderSurface(
//                this->deformSurfMapped.GetVtxDataVBO(),
//                this->deformSurfMapped.GetVertexCnt(),
//                this->deformSurfMapped.GetTriangleIdxVBO(),
//                this->deformSurfMapped.GetTriangleCnt()*3,
//                this->surfaceMappedRM,
//                this->surface2ColorMode,
//                this->surfAttribTex2,
//                this->uniformColorSurf2,
//                this->surf2AlphaScl)) {
//            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
//                    "%s: could not render mapped surface",
//                    this->ClassName());
//            return false;
//        }
//
//        if (!this->renderSurfaceWithSubdivFlag(this->deformSurfMapped)) {
//            return false;
//        }

//        if (!this->renderSurfaceWithUncertainty(this->deformSurfMapped)) {
//            return false;
//        }
    }

//
//    // DEBUG Render backwards mapped subdivision vertices
//    if (this->deformSurfMapped.PeekOldVertexDataDevPt() != NULL) {
//    glColor3f(0.0, 1.0, 0.0);
//    glDisable(GL_LIGHTING);
//    glDisable(GL_BLEND);
//    glPointSize(1.5);
//    HostArr<float> oldVertexData;
//    oldVertexData.Validate(9*this->deformSurfMapped.GetVertexCnt());
//    CudaSafeCall(cudaMemcpy(oldVertexData.Peek(),
//            this->deformSurfMapped.PeekOldVertexDataDevPt(),
//            sizeof(float)*9*this->deformSurfMapped.GetVertexCnt(),
//            cudaMemcpyDeviceToHost));
////    for (int i = 0; i < this->deformSurfMapped.GetVertexCnt(); ++i) {
////        printf("%i %f %f %f\n", i, oldVertexData.Peek()[9*i+0],
////                oldVertexData.Peek()[9*i+1],
////                oldVertexData.Peek()[9*i+2]);
////    }
//    glEnableClientState(GL_VERTEX_ARRAY); // Enable Vertex Arrays
//    glEnableClientState(GL_COLOR_ARRAY); // Enable Vertex Arrays
//    glVertexPointer(3, GL_FLOAT, 9*sizeof(float), oldVertexData.Peek()); // Set The Vertex Pointer To Vertex Data
//    glColorPointer(3, GL_FLOAT, 9*sizeof(float), oldVertexData.Peek() + 3);
//    glDrawArrays(GL_POINTS, 0, this->deformSurfMapped.GetVertexCnt()); //Draw the vertices
//    glDisableClientState(GL_VERTEX_ARRAY); // Enable Vertex Arrays
//    glDisableClientState(GL_COLOR_ARRAY); // Enable Vertex Arrays
//    glEnable(GL_LIGHTING);
//    oldVertexData.Release();
//    }
//    // END DEBUG

//    // DEBUG Compute mean vertex path of deformed surface
//    float surfArea = this->deformSurfMapped.GetTotalSurfArea();
//    float validSurfArea = this->deformSurfMapped.GetTotalValidSurfArea();
//    float meanVertexPath = this->deformSurfMapped.IntVtxPathOverSurfArea();
//    float meanValidVertexPath = this->deformSurfMapped.IntVtxPathOverValidSurfArea();
//    meanVertexPath /= surfArea;
//    meanValidVertexPath /= validSurfArea;
//    printf("SURFACE AREA %f\n", surfArea);
//    printf("VALID SURFACE AREA %f\n", validSurfArea);
//    printf("MEAN PATH    %f\n", meanVertexPath);
//    printf("MEAN VALID PATH    %f\n", meanValidVertexPath);
//    float3 minC, maxC;
//    //this->deformSurf2.ComputeMinMaxCoords(minC, maxC);
//    printf("min %f %f %f, max %f %f %f\n", minC.x, minC.y, minC.z,
//            maxC.x, maxC.y, maxC.z);
//    float radius = std::abs(maxC.x-minC.x)/2.0;
//    printf("RADIUS %f\n", radius);
//    radius = std::abs(maxC.y-minC.y)/2.0;
//    printf("RADIUS %f\n", radius);
//    radius = std::abs(maxC.z-minC.z)/2.0;
//    printf("RADIUS %f\n", radius);
//    float area = M_PI*radius*radius*4;
//    printf("SURFACE AREA should be %f\n", area);
    // END DEBUG

//
//    // DEBUG render grid
//    if (!this->renderGrid(this->deformSurfMapped)) {
//        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
//                "%s: could not render grid",
//                this->ClassName());
//        return false;
//    }

    glDisable(GL_TEXTURE_3D);
    glDisable (GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);

    glPopMatrix();

//    // Unlock frames
//    mol1->Unlock();
//    mol2->Unlock();
//    vti1->Unlock();
//    vti2->Unlock();

    this->oldCalltime = calltime;
    return CheckForGLError();
}

/*
 * ComparativeMolSurfaceRenderer::renderEdges
 */
bool ComparativeMolSurfaceRenderer::renderEdges() {

    unsigned int edgeCnt = static_cast<unsigned int>(this->deformSurfMapped.GetTriangleCnt() * 3 / 2);

    glDisable (GL_LIGHTING);
    glColor3f(0.0, 1.0, 0.0);

    // Create vertex buffer object for triangle indices
    glBindBufferARB(GL_ARRAY_BUFFER, this->deformSurfMapped.GetVtxDataVBO());

    //Establish array contains vertices (not normals, colours, texture coords etc)
    glEnableClientState (GL_VERTEX_ARRAY);
    //Draw Triangle from VBO - do each time window, view point or data changes
    //Establish its 3 coordinates per vertex with zero stride in this array; necessary here
    glVertexPointer(3, GL_FLOAT,
            static_cast<GLsizei>(this->deformSurfMapped.vertexDataStride * sizeof(float)),
            reinterpret_cast<void*>(0));

    glDrawElements(GL_LINES, edgeCnt * 2, GL_UNSIGNED_INT,
            this->deformSurfMapped.PeekEdges());

    //Force display to be drawn now
    glFlush();

    glDisableClientState(GL_VERTEX_ARRAY);

    glBindBufferARB(GL_ARRAY_BUFFER, 0);

    glEnable(GL_LIGHTING);

    return CheckForGLError();
}

/*
 * ComparativeMolSurfaceRenderer::renderExternalForces
 */
bool ComparativeMolSurfaceRenderer::renderExternalForces() {

    using namespace vislib::math;

    size_t gridSize = this->volDim.x * this->volDim.y * this->volDim.z;
    if (this->triggerComputeLines) {

        this->gvf.Validate(gridSize * 4);
        if (!CudaSafeCall(cudaMemcpy(gvf.Peek(),
                        this->deformSurfMapped.PeekExternalForces(), sizeof(float)*gridSize*4,
                        cudaMemcpyDeviceToHost))) {
            return false;
        }

        this->lines.SetCount(gridSize * 6);
        this->lineColors.SetCount(gridSize * 6);
        for (int x = 0; x < this->volDim.x; ++x) {
            for (int y = 0; y < this->volDim.y; ++y) {
                for (int z = 0; z < this->volDim.z; ++z) {
                    unsigned int idx = this->volDim.x * (this->volDim.y * z + y)
                            + x;

                    Vec3f pos(this->volOrg.x + x * this->volDelta.x,
                            this->volOrg.y + y * this->volDelta.y,
                            this->volOrg.z + z * this->volDelta.z);

                    Vec3f vec(gvf.Peek()[4 * idx + 0], gvf.Peek()[4 * idx + 1],
                            gvf.Peek()[4 * idx + 2]);
                    //vec *= this->surfaceMappingExternalForcesWeightScl;
                    //vec *= this->surfaceMappingForcesScl;
                    vec.Normalise();

                    if ((pos.X() <= this->posXMax) && (pos.X() >= this->posXMin)
                            && (pos.Y() <= this->posYMax)
                            && (pos.Y() >= this->posYMin)
                            && (pos.Z() <= this->posZMax)
                            && (pos.Z() >= this->posZMin)) {

                        this->lines[6 * idx + 0] = pos.X() - vec.X() * 0.5f;
                        this->lines[6 * idx + 1] = pos.Y() - vec.Y() * 0.5f;
                        this->lines[6 * idx + 2] = pos.Z() - vec.Z() * 0.5f;
                        this->lines[6 * idx + 3] = pos.X() + vec.X() * 0.5f;
                        this->lines[6 * idx + 4] = pos.Y() + vec.Y() * 0.5f;
                        this->lines[6 * idx + 5] = pos.Z() + vec.Z() * 0.5f;
                    } else {
                        this->lines[6 * idx + 0] = 0.0f;
                        this->lines[6 * idx + 1] = 0.0f;
                        this->lines[6 * idx + 2] = 0.0f;
                        this->lines[6 * idx + 3] = 0.0f;
                        this->lines[6 * idx + 4] = 0.0f;
                        this->lines[6 * idx + 5] = 0.0f;
                    }

                    /// Color mode for south/north
                    this->lineColors[6 * idx + 0] = 1.0f;
                    this->lineColors[6 * idx + 1] = 0.0f;
                    this->lineColors[6 * idx + 2] = 0.0f;
                    this->lineColors[6 * idx + 3] = 0.7f;
                    this->lineColors[6 * idx + 4] = 0.7f;
                    this->lineColors[6 * idx + 5] = 0.7f;
                }
            }
        }
        this->triggerComputeLines = false;
    }

    // Draw lines
    glDisable (GL_LINE_SMOOTH);
    glEnableClientState (GL_VERTEX_ARRAY);
    glEnableClientState (GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, this->lines.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, this->lineColors.PeekElements());

    glLineWidth(2.0);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(gridSize * 2));
    // deactivate vertex arrays after drawing
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glLineWidth(1.0);

    return CheckForGLError();
}

bool ComparativeMolSurfaceRenderer::renderGrid(
        DeformableGPUSurfaceMT &deformSurf) {
    using namespace vislib::math;

    if (this->triggerComputeLines) {

//        // Copy active cells
//        HostArr<unsigned int> activeCells;
//        activeCells.Validate(cellCnt);
//        CudaSafeCall(cudaMemcpy(activeCells.Peek(), deformSurf.PeekCubeStates(),
//                sizeof(uint)*cellCnt, cudaMemcpyDeviceToHost));

//        this->gvf.Validate(gridSize*4);
//        if (!CudaSafeCall(cudaMemcpy(gvf.Peek(),
//                this->deformSurfMapped.PeekExternalForces(), sizeof(float)*gridSize*4,
//                cudaMemcpyDeviceToHost))) {
//            return false;
        //        }

        this->lines.AssertCapacity(10000);

        for (int x = 0; x < this->volDim.x; ++x) {
            for (int y = 0; y < this->volDim.y; ++y) {

                Vec3f pos0(this->volOrg.x + x * this->volDelta.x,
                        this->volOrg.y + y * this->volDelta.y, this->volOrg.z);
                Vec3f pos1(this->volOrg.x + x * this->volDelta.x,
                        this->volOrg.y + y * this->volDelta.y, this->volMaxC.z);

                // Clip
                pos0.SetX(std::max(pos0.X(), this->posXMin));
                pos0.SetY(std::max(pos0.Y(), this->posYMin));
                pos0.SetZ(std::max(pos0.Z(), this->posZMin));
                pos0.SetX(std::min(pos0.X(), this->posXMax));
                pos0.SetY(std::min(pos0.Y(), this->posYMax));
                pos0.SetZ(std::min(pos0.Z(), this->posZMax));
                pos1.SetX(std::max(pos1.X(), this->posXMin));
                pos1.SetY(std::max(pos1.Y(), this->posYMin));
                pos1.SetZ(std::max(pos1.Z(), this->posZMin));
                pos1.SetX(std::min(pos1.X(), this->posXMax));
                pos1.SetY(std::min(pos1.Y(), this->posYMax));
                pos1.SetZ(std::min(pos1.Z(), this->posZMax));

                if ((pos0 - pos1).Length() > 0) {

                    this->lines.Add(pos0.X());
                    this->lines.Add(pos0.Y());
                    this->lines.Add(pos0.Z());
                    this->lines.Add(pos1.X());
                    this->lines.Add(pos1.Y());
                    this->lines.Add(pos1.Z());
                }
            }
        }

        for (int z = 0; z < this->volDim.z; ++z) {
            for (int y = 0; y < this->volDim.y; ++y) {

                Vec3f pos0(this->volOrg.x,
                        this->volOrg.y + y * this->volDelta.y,
                        this->volOrg.z + z * this->volDelta.z);
                Vec3f pos1(this->volMaxC.x,
                        this->volOrg.y + y * this->volDelta.y,
                        this->volOrg.z + z * this->volDelta.z);

                // Clip
                pos0.SetX(std::max(pos0.X(), this->posXMin));
                pos0.SetY(std::max(pos0.Y(), this->posYMin));
                pos0.SetZ(std::max(pos0.Z(), this->posZMin));
                pos0.SetX(std::min(pos0.X(), this->posXMax));
                pos0.SetY(std::min(pos0.Y(), this->posYMax));
                pos0.SetZ(std::min(pos0.Z(), this->posZMax));
                pos1.SetX(std::max(pos1.X(), this->posXMin));
                pos1.SetY(std::max(pos1.Y(), this->posYMin));
                pos1.SetZ(std::max(pos1.Z(), this->posZMin));
                pos1.SetX(std::min(pos1.X(), this->posXMax));
                pos1.SetY(std::min(pos1.Y(), this->posYMax));
                pos1.SetZ(std::min(pos1.Z(), this->posZMax));

                if ((pos0 - pos1).Length() > 0) {

                    this->lines.Add(pos0.X());
                    this->lines.Add(pos0.Y());
                    this->lines.Add(pos0.Z());
                    this->lines.Add(pos1.X());
                    this->lines.Add(pos1.Y());
                    this->lines.Add(pos1.Z());
                }
            }
        }

        for (int z = 0; z < this->volDim.z; ++z) {
            for (int x = 0; x < this->volDim.x; ++x) {

                Vec3f pos0(this->volOrg.x + x * this->volDelta.x,
                        this->volOrg.y, this->volOrg.z + z * this->volDelta.z);
                Vec3f pos1(this->volOrg.x + x * this->volDelta.x,
                        this->volMaxC.y, this->volOrg.z + z * this->volDelta.z);

                // Clip
                pos0.SetX(std::max(pos0.X(), this->posXMin));
                pos0.SetY(std::max(pos0.Y(), this->posYMin));
                pos0.SetZ(std::max(pos0.Z(), this->posZMin));
                pos0.SetX(std::min(pos0.X(), this->posXMax));
                pos0.SetY(std::min(pos0.Y(), this->posYMax));
                pos0.SetZ(std::min(pos0.Z(), this->posZMax));
                pos1.SetX(std::max(pos1.X(), this->posXMin));
                pos1.SetY(std::max(pos1.Y(), this->posYMin));
                pos1.SetZ(std::max(pos1.Z(), this->posZMin));
                pos1.SetX(std::min(pos1.X(), this->posXMax));
                pos1.SetY(std::min(pos1.Y(), this->posYMax));
                pos1.SetZ(std::min(pos1.Z(), this->posZMax));

                if ((pos0 - pos1).Length() > 0) {

                    this->lines.Add(pos0.X());
                    this->lines.Add(pos0.Y());
                    this->lines.Add(pos0.Z());
                    this->lines.Add(pos1.X());
                    this->lines.Add(pos1.Y());
                    this->lines.Add(pos1.Z());
                }
            }
        }

//        activeCells.Release();
        this->triggerComputeLines = false;
    }

    // Draw lines
    glDisable (GL_LINE_SMOOTH);
    glEnableClientState (GL_VERTEX_ARRAY);
//    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, this->lines.PeekElements());
//    glColorPointer(3, GL_FLOAT, 0, this->lineColors.PeekElements());

    glLineWidth(1.0);
    glColor3f(0.0, 0.0, 1.0);
	glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(this->lines.Count() / 3));
    // deactivate vertex arrays after drawing
    glDisableClientState(GL_VERTEX_ARRAY);
//    glDisableClientState(GL_COLOR_ARRAY);
    return ::CheckForGLError();
}

/*
 * ComparativeMolSurfaceRenderer::::renderSurface
 */
bool ComparativeMolSurfaceRenderer::renderSurface(GLuint vbo, uint vertexCnt,
        GLuint vboTriangleIdx, uint triangleVertexCnt,
        SurfaceRenderMode renderMode, SurfaceColorMode colorMode,
        GLuint potentialTex, Vec3f uniformColor, float alphaScl) {

    GLint attribLocPos, attribLocNormal, attribLocTexCoord;

    /* Get vertex attributes from vbo */

    glBindBufferARB(GL_ARRAY_BUFFER, vbo);
    CheckForGLError(); // OpenGL error check

    this->pplSurfaceShader.Enable();
    CheckForGLError(); // OpenGL error check

    // Note: glGetAttribLocation returnes -1 if the attribute if not used in
    // the shader code, because in this case the attribute is optimized out by
    // the compiler
    attribLocPos = glGetAttribLocationARB(
            this->pplSurfaceShader.ProgramHandle(), "pos");
    attribLocNormal = glGetAttribLocationARB(
            this->pplSurfaceShader.ProgramHandle(), "normal");
    attribLocTexCoord = glGetAttribLocationARB(
            this->pplSurfaceShader.ProgramHandle(), "texCoord");
    CheckForGLError(); // OpenGL error check

    glEnableVertexAttribArrayARB(attribLocPos);
    glEnableVertexAttribArrayARB(attribLocNormal);
    glEnableVertexAttribArrayARB(attribLocTexCoord);
    CheckForGLError(); // OpenGL error check

    glVertexAttribPointerARB(attribLocPos, 3, GL_FLOAT, GL_FALSE,
			static_cast<GLsizei>(DeformableGPUSurfaceMT::vertexDataStride * sizeof(float)),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsPos
                    * sizeof(float)));
    glVertexAttribPointerARB(attribLocNormal, 3, GL_FLOAT, GL_FALSE,
			static_cast<GLsizei>(DeformableGPUSurfaceMT::vertexDataStride * sizeof(float)),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsNormal
                    * sizeof(float)));
    glVertexAttribPointerARB(attribLocTexCoord, 3, GL_FLOAT, GL_FALSE,
			static_cast<GLsizei>(DeformableGPUSurfaceMT::vertexDataStride * sizeof(float)),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsTexCoord
                    * sizeof(float)));
    CheckForGLError(); // OpenGL error check

    /* Render */

    // Set uniform vars
    glUniform1iARB(this->pplSurfaceShader.ParameterLocation("potentialTex"), 0);
    glUniform1iARB(this->pplSurfaceShader.ParameterLocation("colorMode"),
            static_cast<int>(colorMode));
    glUniform1iARB(this->pplSurfaceShader.ParameterLocation("renderMode"),
            static_cast<int>(renderMode));
    glUniform3fvARB(this->pplSurfaceShader.ParameterLocation("colorMin"), 1,
            this->colorMinPotential.PeekComponents());
    glUniform3fvARB(this->pplSurfaceShader.ParameterLocation("colorMax"), 1,
            this->colorMaxPotential.PeekComponents());
    glUniform3fvARB(this->pplSurfaceShader.ParameterLocation("colorUniform"), 1,
            uniformColor.PeekComponents());
    glUniform1fARB(this->pplSurfaceShader.ParameterLocation("minPotential"),
            this->minPotential);
    glUniform1fARB(this->pplSurfaceShader.ParameterLocation("maxPotential"),
            this->maxPotential);
    glUniform1fARB(this->pplSurfaceShader.ParameterLocation("alphaScl"),
            alphaScl);

    glActiveTextureARB(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, potentialTex);
    CheckForGLError(); // OpenGL error check

    if (renderMode == SURFACE_FILL) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    } else if (renderMode == SURFACE_WIREFRAME) {
        glDisable (GL_LIGHTING);
        glLineWidth(1.0f);
        //glLineWidth(10.0f);
        glCullFace (GL_BACK);
        glEnable (GL_CULL_FACE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else if (renderMode == SURFACE_POINTS) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
    }

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, vboTriangleIdx);
    CheckForGLError(); // OpenGL error check

    glDrawElements(GL_TRIANGLES, triangleVertexCnt, GL_UNSIGNED_INT,
            reinterpret_cast<void*>(0));

//    glDrawArrays(GL_POINTS, 0, vertexCnt); // DEBUG
    glDisable (GL_CULL_FACE);

    this->pplSurfaceShader.Disable();

//    glColor3f(1.0, 0.0, 0.0);
//
//    glDrawElements(GL_TRIANGLES,
//            //triangleVertexCnt,
//            3*24,
//            GL_UNSIGNED_INT,
//            reinterpret_cast<void*>(36*sizeof(unsigned int)));

    glDisableVertexAttribArrayARB(attribLocPos);
    glDisableVertexAttribArrayARB(attribLocNormal);
    glDisableVertexAttribArrayARB(attribLocTexCoord);
    CheckForGLError(); // OpenGL error check

    // Switch back to normal pointer operation by binding with 0
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    glEnable (GL_LIGHTING);

    return CheckForGLError(); // OpenGL error check
}

/*
 * ComparativeMolSurfaceRenderer::renderSurfaceWithSubdivFlag
 */
bool ComparativeMolSurfaceRenderer::renderSurfaceWithSubdivFlag(
        DeformableGPUSurfaceMT &surf) {

    if (!surf.GetVtxDataVBO()) {
        printf("No vertex data VBO!!!\n");
        return false;
    }

    // Note: This method does not check whether the vertexFlag array is filled

    GLint attribLocPos, /*attribLocNormal,*/ attribLocFlag;

    /* Get vertex attributes from vbo */

    glBindBufferARB(GL_ARRAY_BUFFER, surf.GetVtxDataVBO());
    CheckForGLError(); // OpenGL error check

    this->pplSurfaceShaderVertexFlag.Enable();
    CheckForGLError(); // OpenGL error check

    // Note: glGetAttribLocation returnes -1 if the attribute if not used in
    // the shader code, because in this case the attribute is optimized out by
    // the compiler
    attribLocPos = glGetAttribLocationARB(
            this->pplSurfaceShaderVertexFlag.ProgramHandle(), "pos");
//    attribLocNormal = glGetAttribLocationARB(this->pplSurfaceShaderVertexFlag.ProgramHandle(), "normal");
    CheckForGLError(); // OpenGL error check

    glEnableVertexAttribArrayARB(attribLocPos);
//    glEnableVertexAttribArrayARB(attribLocNormal);
    CheckForGLError(); // OpenGL error check

    glVertexAttribPointerARB(attribLocPos, 3, GL_FLOAT, GL_FALSE,
			static_cast<GLsizei>(DeformableGPUSurfaceMT::vertexDataStride * sizeof(float)),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsPos
                    * sizeof(float)));
//    glVertexAttribPointerARB(attribLocNormal, 3, GL_FLOAT, GL_FALSE,
//            DeformableGPUSurfaceMT::vertexDataStride*sizeof(float),
//            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsNormal*sizeof(float)));
    CheckForGLError(); // OpenGL error check

    // Vertex flag

    glBindBufferARB(GL_ARRAY_BUFFER, 0);
    attribLocFlag = glGetAttribLocationARB(
            this->pplSurfaceShaderVertexFlag.ProgramHandle(), "flag");
    glEnableVertexAttribArrayARB(attribLocFlag);
    glVertexAttribPointerARB(attribLocFlag, 1, GL_FLOAT, GL_FALSE, 0,
            this->mappedSurfVertexFlags.Peek());
    CheckForGLError(); // OpenGL error check

//    for (uint i = 0; i < surf.GetVertexCnt(); ++i) {
//        printf("%i: %f\n", i, this->mappedSurfVertexFlags.Peek()[i]);
//    }

    /* Render */

    glDisable (GL_LIGHTING);
    glLineWidth(1.0f);
    glCullFace (GL_BACK);
    glEnable (GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, surf.GetTriangleIdxVBO());
    CheckForGLError(); // OpenGL error check

	glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(surf.GetTriangleCnt() * 3), GL_UNSIGNED_INT,
            reinterpret_cast<void*>(0));

    glDisable(GL_CULL_FACE);
    glEnable(GL_LIGHTING);

    this->pplSurfaceShaderVertexFlag.Disable();

    glDisableVertexAttribArrayARB(attribLocPos);
//    glDisableVertexAttribArrayARB(attribLocNormal);
    glDisableVertexAttribArrayARB(attribLocFlag);
    CheckForGLError(); // OpenGL error check

    // Switch back to normal pointer operation by binding with 0
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);

    return ::CheckForGLError();
}

/*
 * ComparativeMolSurfaceRenderer::renderSurfaceWithUncertainty
 */
bool ComparativeMolSurfaceRenderer::renderSurfaceWithUncertainty(
        DeformableGPUSurfaceMT &surf) {
    if (!surf.GetVtxDataVBO()) {
        printf("No vertex data VBO!!!\n");
        return false;
    }

    // Note: This method does not check whether the vertexFlag array is filled

    GLint attribLocPos, /*attribLocNormal,*/ attribLocUncertainty;

    /* Get vertex attributes from vbo */

    glBindBufferARB(GL_ARRAY_BUFFER, surf.GetVtxDataVBO());
    CheckForGLError(); // OpenGL error check

    this->pplSurfaceShaderUncertainty.Enable();
    glUniform1fARB(
            this->pplSurfaceShaderUncertainty.ParameterLocation(
                    "maxUncertainty"), 1.0f / this->surfMaxPosDiff);
    CheckForGLError(); // OpenGL error check

    // Note: glGetAttribLocation returnes -1 if the attribute if not used in
    // the shader code, because in this case the attribute is optimized out by
    // the compiler
    attribLocPos = glGetAttribLocationARB(
            this->pplSurfaceShaderUncertainty.ProgramHandle(), "pos");
//    attribLocNormal = glGetAttribLocationARB(this->pplSurfaceShaderUncertainty.ProgramHandle(), "normal");
    CheckForGLError(); // OpenGL error check

    glEnableVertexAttribArrayARB(attribLocPos);
//    glEnableVertexAttribArrayARB(attribLocNormal);
    CheckForGLError(); // OpenGL error check

    glVertexAttribPointerARB(attribLocPos, 3, GL_FLOAT, GL_FALSE,
			static_cast<GLsizei>(DeformableGPUSurfaceMT::vertexDataStride * sizeof(float)),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsPos
                    * sizeof(float)));
//    glVertexAttribPointerARB(attribLocNormal, 3, GL_FLOAT, GL_FALSE,
//            DeformableGPUSurfaceMT::vertexDataStride*sizeof(float),
//            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsNormal*sizeof(float)));
    CheckForGLError(); // OpenGL error check

    // Vertex flag

    glBindBufferARB(GL_ARRAY_BUFFER, surf.GetVtxPathVBO());
    attribLocUncertainty = glGetAttribLocationARB(
            this->pplSurfaceShaderUncertainty.ProgramHandle(), "uncertainty");
    glEnableVertexAttribArrayARB(attribLocUncertainty);
    glVertexAttribPointerARB(attribLocUncertainty, 1, GL_FLOAT, GL_FALSE, 0,
            reinterpret_cast<void*>(0));
    CheckForGLError(); // OpenGL error check

    /* Render */

    glDisable (GL_LIGHTING);
    glLineWidth(1.0f);
    glCullFace (GL_BACK);
    glEnable (GL_CULL_FACE);
    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, surf.GetTriangleIdxVBO());
    CheckForGLError(); // OpenGL error check

    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(surf.GetTriangleCnt() * 3), GL_UNSIGNED_INT,
            reinterpret_cast<void*>(0));

    glDisable(GL_CULL_FACE);
    glEnable(GL_LIGHTING);

    this->pplSurfaceShaderUncertainty.Disable();

    glDisableVertexAttribArrayARB(attribLocPos);
//    glDisableVertexAttribArrayARB(attribLocNormal);
    glDisableVertexAttribArrayARB(attribLocUncertainty);
    CheckForGLError(); // OpenGL error check

    // Switch back to normal pointer operation by binding with 0
    glBindBufferARB(GL_ARRAY_BUFFER, 0);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);

    return ::CheckForGLError();
}

/*
 * ComparativeMolSurfaceRenderer::::renderMappedSurface
 */
bool ComparativeMolSurfaceRenderer::renderMappedSurface(GLuint vboOld,
        GLuint vboNew, uint vertexCnt, GLuint vboTriangleIdx,
        uint triangleVertexCnt, SurfaceRenderMode renderMode,
        SurfaceColorMode colorMode) {

    GLint attribLocPosNew/*, attribLocPosOld*/;
    GLint attribLocNormal, attribLocCorruptTriangleFlag;
    GLint attribLocTexCoordNew, /*attribLocTexCoordOld,*/ attribLocPathLen,
            attribLocSurfAttrib;

    this->pplMappedSurfaceShader.Enable();
    CheckForGLError(); // OpenGL error check

    // Note: glGetAttribLocation returns -1 if an attribute if not used in
    // the shader code, because in this case the attribute is optimized out by
    // the compiler
    attribLocPosNew = glGetAttribLocationARB(
            this->pplMappedSurfaceShader.ProgramHandle(), "posNew");
//    attribLocPosOld = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "posOld");
    attribLocNormal = glGetAttribLocationARB(
            this->pplMappedSurfaceShader.ProgramHandle(), "normal");
    attribLocCorruptTriangleFlag = glGetAttribLocationARB(
            this->pplMappedSurfaceShader.ProgramHandle(),
            "corruptTriangleFlag");
    attribLocTexCoordNew = glGetAttribLocationARB(
            this->pplMappedSurfaceShader.ProgramHandle(), "texCoordNew");
//    attribLocTexCoordOld = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "texCoordOld");
    attribLocPathLen = glGetAttribLocationARB(
            this->pplMappedSurfaceShader.ProgramHandle(), "pathLen");
    attribLocSurfAttrib = glGetAttribLocationARB(
            this->pplMappedSurfaceShader.ProgramHandle(), "surfAttrib");
    CheckForGLError(); // OpenGL error check

    glEnableVertexAttribArrayARB(attribLocPosNew);
//    glEnableVertexAttribArrayARB(attribLocPosOld);
    glEnableVertexAttribArrayARB(attribLocNormal);
    glEnableVertexAttribArrayARB(attribLocCorruptTriangleFlag);
    glEnableVertexAttribArrayARB(attribLocTexCoordNew);
//    glEnableVertexAttribArrayARB(attribLocTexCoordOld);
    glEnableVertexAttribArrayARB(attribLocPathLen);
    glEnableVertexAttribArrayARB(attribLocSurfAttrib);
    CheckForGLError(); // OpenGL error check

    /* Get vertex attributes from vbos */

    // Attributes of new (mapped) surface
    glBindBufferARB(GL_ARRAY_BUFFER, vboNew);
    CheckForGLError(); // OpenGL error check

    glVertexAttribPointerARB(attribLocPosNew, 3, GL_FLOAT, GL_FALSE,
			static_cast<GLsizei>(DeformableGPUSurfaceMT::vertexDataStride * sizeof(float)),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsPos
                    * sizeof(float)));

    glVertexAttribPointerARB(attribLocTexCoordNew, 3, GL_FLOAT, GL_FALSE,
			static_cast<GLsizei>(DeformableGPUSurfaceMT::vertexDataStride * sizeof(float)),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsTexCoord
                    * sizeof(float)));

    glVertexAttribPointerARB(attribLocNormal, 3, GL_FLOAT, GL_FALSE,
            static_cast<GLsizei>(DeformableGPUSurfaceMT::vertexDataStride * sizeof(float)),
            reinterpret_cast<void*>(DeformableGPUSurfaceMT::vertexDataOffsNormal
                    * sizeof(float)));

    // Attribute for vertex path length
    glBindBufferARB(GL_ARRAY_BUFFER, this->deformSurfMapped.GetVtxPathVBO());
    CheckForGLError(); // OpenGL error check
    glVertexAttribPointerARB(attribLocPathLen, 1, GL_FLOAT, GL_FALSE, 0, 0);
    CheckForGLError(); // OpenGL error check

    // Attribute for surface property
    glBindBufferARB(GL_ARRAY_BUFFER, this->deformSurfMapped.GetVtxAttribVBO());
    CheckForGLError(); // OpenGL error check
    glVertexAttribPointerARB(attribLocSurfAttrib, 1, GL_FLOAT, GL_FALSE, 0, 0);
    CheckForGLError(); // OpenGL error check

    // Attribute for surface property
    glBindBufferARB(GL_ARRAY_BUFFER,
            this->deformSurfMapped.GetCorruptTriangleVtxFlagVBO());
    CheckForGLError(); // OpenGL error check
    glVertexAttribPointerARB(attribLocCorruptTriangleFlag, 1, GL_FLOAT,
            GL_FALSE, 0, 0);
    CheckForGLError(); // OpenGL error check

    /* Render */

    // Set uniform vars
    glUniform1iARB(
            this->pplMappedSurfaceShader.ParameterLocation("potentialTex0"), 0);
    glUniform1iARB(
            this->pplMappedSurfaceShader.ParameterLocation("potentialTex1"), 1);
    glUniform1iARB(this->pplMappedSurfaceShader.ParameterLocation("colorMode"),
            static_cast<int>(colorMode));
    glUniform1iARB(this->pplMappedSurfaceShader.ParameterLocation("renderMode"),
            static_cast<int>(renderMode));
    glUniform1iARB(
            this->pplMappedSurfaceShader.ParameterLocation(
                    "unmappedTrisColorMode"),
            static_cast<int>(this->theUnmappedTrisColor));
    glUniform3fvARB(this->pplMappedSurfaceShader.ParameterLocation("colorMin"),
            1, this->colorMinPotential.PeekComponents());
    glUniform3fvARB(this->pplMappedSurfaceShader.ParameterLocation("colorMax"),
            1, this->colorMaxPotential.PeekComponents());
    glUniform3fvARB(
            this->pplMappedSurfaceShader.ParameterLocation("colorUniform"), 1,
            this->uniformColorSurfMapped.PeekComponents());
    glUniform1fARB(
            this->pplMappedSurfaceShader.ParameterLocation("minPotential"),
            this->minPotential);
    glUniform1fARB(
            this->pplMappedSurfaceShader.ParameterLocation("maxPotential"),
            this->maxPotential);
    glUniform1fARB(this->pplMappedSurfaceShader.ParameterLocation("alphaScl"),
            this->surfMappedAlphaScl);
    glUniform1fARB(this->pplMappedSurfaceShader.ParameterLocation("maxPosDiff"),
            this->surfMaxPosDiff);
    glUniform1iARB(
            this->pplMappedSurfaceShader.ParameterLocation(
                    "uncertaintyMeasurement"),
            static_cast<int>(this->uncertaintyCriterion));

    glActiveTextureARB(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, this->surfAttribTex2);

    glActiveTextureARB(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, this->surfAttribTex1);
    CheckForGLError(); // OpenGL error check

    if (renderMode == SURFACE_FILL) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    } else if (renderMode == SURFACE_WIREFRAME) {
        glCullFace(GL_BACK);
        glDisable(GL_BLEND);
        glEnable (GL_CULL_FACE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else if (renderMode == SURFACE_POINTS) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
    }
    glLineWidth(1.0f);

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, vboTriangleIdx);
    CheckForGLError(); // OpenGL error check

    glDrawElements(GL_TRIANGLES, triangleVertexCnt, GL_UNSIGNED_INT,
            reinterpret_cast<void*>(0));

    this->pplMappedSurfaceShader.Disable();

    glDisableVertexAttribArrayARB(attribLocPosNew);
    glDisableVertexAttribArrayARB(attribLocNormal);
    glDisableVertexAttribArrayARB(attribLocTexCoordNew);
    glDisableVertexAttribArrayARB(attribLocPathLen);
    glDisableVertexAttribArrayARB(attribLocSurfAttrib);
    glDisableVertexAttribArrayARB(attribLocCorruptTriangleFlag);
    CheckForGLError(); // OpenGL error check

    // Switch back to normal pointer operation by binding with 0
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    glDisable (GL_CULL_FACE);
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    ::glDisable(GL_LIGHTING);
    ::glDisable(GL_TEXTURE_2D);
    ::glEnable(GL_LINE_SMOOTH);
    ::glLineWidth(1.5f);
    ::glDisable(GL_DEPTH_TEST);

    return CheckForGLError(); // OpenGL error check
}

/*
 * ComparativeMolSurfaceRenderer::updateParams
 */
void ComparativeMolSurfaceRenderer::updateParams() {
    /* Parameter for frame by frame comparison */

    // Param slot for compare mode
    if (this->cmpModeSlot.IsDirty()) {
        this->cmpMode = static_cast<CompareMode>(this->cmpModeSlot.Param<
                core::param::EnumParam>()->Value());
        this->cmpModeSlot.ResetDirty();
        this->triggerRMSFit = true;
        this->triggerComputeVolume = true;
        this->triggerInitPotentialTex = true;
    }

    // Param for single frame #1
    if (this->singleFrame1Slot.IsDirty()) {
        this->singleFrame1 =
                this->singleFrame1Slot.Param<core::param::IntParam>()->Value();
        this->singleFrame1Slot.ResetDirty();
        this->triggerRMSFit = true;
        this->triggerComputeVolume = true;
        this->triggerInitPotentialTex = true;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerSurfaceMapping = true;
        this->triggerComputeLines = true;
    }

    // Param for single frame #2
    if (this->singleFrame2Slot.IsDirty()) {
        this->singleFrame2 =
                this->singleFrame2Slot.Param<core::param::IntParam>()->Value();
        this->singleFrame2Slot.ResetDirty();
        this->triggerRMSFit = true;
        this->triggerComputeVolume = true;
        this->triggerInitPotentialTex = true;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerSurfaceMapping = true;
        this->triggerComputeLines = true;
    }

    /* Global mapping options */

    // Interpolation method used when computing external forces
    if (this->interpolModeSlot.IsDirty()) {
        this->interpolMode =
                static_cast<DeformableGPUSurfaceMT::InterpolationMode>(this->interpolModeSlot.Param<
                        core::param::EnumParam>()->Value());
        this->interpolModeSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }

    // Maximum potential
    if (this->maxPotentialSlot.IsDirty()) {
        this->maxPotential = this->maxPotentialSlot.Param<
                core::param::FloatParam>()->Value();
        this->maxPotentialSlot.ResetDirty();
    }

    // Minimum potential
    if (this->minPotentialSlot.IsDirty()) {
        this->minPotential = this->minPotentialSlot.Param<
                core::param::FloatParam>()->Value();
        this->minPotentialSlot.ResetDirty();
    }

    /* Parameters for the mapped surface */

    // Parameter for the RMS fitting mode
    if (this->fittingModeSlot.IsDirty()) {
        this->fittingMode =
                static_cast<RMSFittingMode>(this->fittingModeSlot.Param<
                        core::param::EnumParam>()->Value());
        this->fittingModeSlot.ResetDirty();
        this->triggerRMSFit = true;
        this->triggerSurfaceMapping = true;
        this->triggerComputeLines = true;
    }

    // Parameter for the external forces
    if (this->surfMappedExtForceSlot.IsDirty()) {
        this->surfMappedExtForce =
                static_cast<ExternalForces>(this->surfMappedExtForceSlot.Param<
                        core::param::EnumParam>()->Value());
        this->surfMappedExtForceSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
        this->triggerComputeLines = true;
    }

    // Parameter for uncertainty criterion
    if (this->uncertaintyCriterionSlot.IsDirty()) {
        this->uncertaintyCriterion =
                static_cast<UncertaintyCriterion>(this->uncertaintyCriterionSlot.Param<
                        core::param::EnumParam>()->Value());
        this->uncertaintyCriterionSlot.ResetDirty();
    }

    // Weighting for external forces when mapping the surface
    if (this->surfaceMappingExternalForcesWeightSclSlot.IsDirty()) {
        this->surfaceMappingExternalForcesWeightScl =
                this->surfaceMappingExternalForcesWeightSclSlot.Param<
                        core::param::FloatParam>()->Value();
        this->surfaceMappingExternalForcesWeightSclSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
    }

    // Overall scaling of the resulting forces
    if (this->surfaceMappingForcesSclSlot.IsDirty()) {
        this->surfaceMappingForcesScl = this->surfaceMappingForcesSclSlot.Param<
                core::param::FloatParam>()->Value();
        this->surfaceMappingForcesSclSlot.ResetDirty();
        this->surfMappedMinDisplScl = this->qsGridDelta / 100.0f
                * this->surfaceMappingForcesScl;
        this->triggerSurfaceMapping = true;
    }

    // Maximum number of iterations when mapping the surface
    if (this->surfaceMappingMaxItSlot.IsDirty()) {
        this->surfaceMappingMaxIt = this->surfaceMappingMaxItSlot.Param<
                core::param::IntParam>()->Value();
        this->surfaceMappingMaxItSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
    }

    // Param for minimum vertex displacement
    if (this->surfMappedMinDisplSclSlot.IsDirty()) {
        this->surfMappedMinDisplScl =
                1.0f
                        / this->surfMappedMinDisplSclSlot.Param<
                                core::param::FloatParam>()->Value();
        this->surfMappedMinDisplSclSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
    }

    // Param for spring stiffness
    if (this->surfMappedSpringStiffnessSlot.IsDirty()) {
        this->surfMappedSpringStiffness =
                this->surfMappedSpringStiffnessSlot.Param<
                        core::param::FloatParam>()->Value();
        this->surfMappedSpringStiffnessSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
    }

    // GVF scale factor
    if (this->surfMappedGVFSclSlot.IsDirty()) {
        this->surfMappedGVFScl = this->surfMappedGVFSclSlot.Param<
                core::param::FloatParam>()->Value();
        this->surfMappedGVFSclSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
        this->triggerComputeLines = true;
    }

    // GVF iterations
    if (this->surfMappedGVFItSlot.IsDirty()) {
        this->surfMappedGVFIt = this->surfMappedGVFItSlot.Param<
                core::param::IntParam>()->Value();
        this->surfMappedGVFItSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
        this->triggerComputeLines = true;
    }

    // Distance field threshold
    if (this->distFieldThreshSlot.IsDirty()) {
        this->distFieldThresh = this->distFieldThreshSlot.Param<
                core::param::FloatParam>()->Value();
        this->distFieldThreshSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
        this->triggerComputeLines = true;
    }

    /* Parameters for surface regularization */

    // Maximum number of iterations when regularizing the mesh #0
    if (this->regMaxItSlot.IsDirty()) {
        this->regMaxIt =
                this->regMaxItSlot.Param<core::param::IntParam>()->Value();
        this->regMaxItSlot.ResetDirty();
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }

    // Stiffness of the springs defining the spring forces in surface #0
    if (this->regSpringStiffnessSlot.IsDirty()) {
        this->regSpringStiffness = this->regSpringStiffnessSlot.Param<
                core::param::FloatParam>()->Value();
        this->regSpringStiffnessSlot.ResetDirty();
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }

    // Weighting of the external forces in surface #0, note that the weight
    // of the internal forces is implicitely defined by
    // 1.0 - surf0ExternalForcesWeight
    if (this->regExternalForcesWeightSlot.IsDirty()) {
        this->regExternalForcesWeight = this->regExternalForcesWeightSlot.Param<
                core::param::FloatParam>()->Value();
        this->regExternalForcesWeightSlot.ResetDirty();
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }

    // Overall scaling for the forces acting upon surface #0
    if (this->regForcesSclSlot.IsDirty()) {
        this->regForcesScl = this->regForcesSclSlot.Param<
                core::param::FloatParam>()->Value();
        this->regForcesSclSlot.ResetDirty();
        this->surfregMinDisplScl = this->qsGridDelta / 100.0f
                * this->regForcesScl;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }

    // Param for minimum vertex displacement
    if (this->surfregMinDisplSclSlot.IsDirty()) {
        this->surfregMinDisplScl =
                1.0f
                        / this->surfregMinDisplSclSlot.Param<
                                core::param::FloatParam>()->Value();
        this->surfregMinDisplSclSlot.ResetDirty();
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
        this->triggerSurfaceMapping = true;
    }

    /* Rendering of surface #1 and #2 */

    // Parameter for surface #0 render mode
    if (this->surface1RMSlot.IsDirty()) {
        this->surface1RM =
                static_cast<SurfaceRenderMode>(this->surface1RMSlot.Param<
                        core::param::EnumParam>()->Value());
        this->surface1RMSlot.ResetDirty();
    }

    // Parameter for surface #1 color mode
    if (this->surface1ColorModeSlot.IsDirty()) {
        this->surface1ColorMode =
                static_cast<SurfaceColorMode>(this->surface1ColorModeSlot.Param<
                        core::param::EnumParam>()->Value());
        this->surface1ColorModeSlot.ResetDirty();
    }

    // Param for transparency scaling
    if (this->surf1AlphaSclSlot.IsDirty()) {
        this->surf1AlphaScl = this->surf1AlphaSclSlot.Param<
                core::param::FloatParam>()->Value();
        this->surf1AlphaSclSlot.ResetDirty();
    }

    // Parameter for surface #2 render mode
    if (this->surface2RMSlot.IsDirty()) {
        this->surface2RM =
                static_cast<SurfaceRenderMode>(this->surface2RMSlot.Param<
                        core::param::EnumParam>()->Value());
        this->surface2RMSlot.ResetDirty();
    }

    // Parameter for surface #2 color mode
    if (this->surface2ColorModeSlot.IsDirty()) {
        this->surface2ColorMode =
                static_cast<SurfaceColorMode>(this->surface2ColorModeSlot.Param<
                        core::param::EnumParam>()->Value());
        this->surface2ColorModeSlot.ResetDirty();
    }

    // Param for transparency scaling
    if (this->surf2AlphaSclSlot.IsDirty()) {
        this->surf2AlphaScl = this->surf2AlphaSclSlot.Param<
                core::param::FloatParam>()->Value();
        this->surf2AlphaSclSlot.ResetDirty();
    }

    /* Mapped surface rendering */

    // Parameter for mapped surface render mode
    if (this->surfaceMappedRMSlot.IsDirty()) {
        this->surfaceMappedRM =
                static_cast<SurfaceRenderMode>(this->surfaceMappedRMSlot.Param<
                        core::param::EnumParam>()->Value());
        this->surfaceMappedRMSlot.ResetDirty();
    }

    // Parameter for mapped surface color mode
    if (this->surfaceMappedColorModeSlot.IsDirty()) {
        this->surfaceMappedColorMode =
                static_cast<SurfaceColorMode>(this->surfaceMappedColorModeSlot.Param<
                        core::param::EnumParam>()->Value());
        this->surfaceMappedColorModeSlot.ResetDirty();
    }

    // Parameter unmapped triangles color mode
    if (this->unmappedTrisColorSlot.IsDirty()) {
        this->theUnmappedTrisColor =
                static_cast<UnmappedTrisColor>(this->unmappedTrisColorSlot.Param<
                        core::param::EnumParam>()->Value());
        this->unmappedTrisColorSlot.ResetDirty();
    }

    // Param for maximum value for euclidian distance
    if (this->surfMaxPosDiffSlot.IsDirty()) {
        this->surfMaxPosDiff = this->surfMaxPosDiffSlot.Param<
                core::param::FloatParam>()->Value();
        this->surfMaxPosDiffSlot.ResetDirty();
    }

    // Param for transparency scaling
    if (this->surfMappedAlphaSclSlot.IsDirty()) {
        this->surfMappedAlphaScl = this->surfMappedAlphaSclSlot.Param<
                core::param::FloatParam>()->Value();
        this->surfMappedAlphaSclSlot.ResetDirty();
    }

    /* DEBUG External forces rendering */

    // Max x
    if (this->filterXMaxParam.IsDirty()) {
        this->posXMax =
                this->filterXMaxParam.Param<core::param::FloatParam>()->Value();
        this->filterXMaxParam.ResetDirty();
        this->triggerComputeLines = true;
    }

    // Max y
    if (this->filterYMaxParam.IsDirty()) {
        this->posYMax =
                this->filterYMaxParam.Param<core::param::FloatParam>()->Value();
        this->filterYMaxParam.ResetDirty();
        this->triggerComputeLines = true;
    }

    // Max z
    if (this->filterZMaxParam.IsDirty()) {
        this->posZMax =
                this->filterZMaxParam.Param<core::param::FloatParam>()->Value();
        this->filterZMaxParam.ResetDirty();
        this->triggerComputeLines = true;
    }

    // Min x
    if (this->filterXMinParam.IsDirty()) {
        this->posXMin =
                this->filterXMinParam.Param<core::param::FloatParam>()->Value();
        this->filterXMinParam.ResetDirty();
        this->triggerComputeLines = true;
    }

    // Min y
    if (this->filterYMinParam.IsDirty()) {
        this->posYMin =
                this->filterYMinParam.Param<core::param::FloatParam>()->Value();
        this->filterYMinParam.ResetDirty();
        this->triggerComputeLines = true;
    }

    // Min z
    if (this->filterZMinParam.IsDirty()) {
        this->posZMin =
                this->filterZMinParam.Param<core::param::FloatParam>()->Value();
        this->filterZMinParam.ResetDirty();
        this->triggerComputeLines = true;
    }

    /* Quicksurf parameter */

    // Quicksurf radius scale
    if (this->qsRadSclSlot.IsDirty()) {
        this->qsRadScl =
                this->qsRadSclSlot.Param<core::param::FloatParam>()->Value();
        this->qsRadSclSlot.ResetDirty();
        this->triggerComputeVolume = true;
    }

    // Quicksurf grid scaling
    if (this->qsGridDeltaSlot.IsDirty()) {
        this->qsGridDelta =
                this->qsGridDeltaSlot.Param<core::param::FloatParam>()->Value();
        this->qsGridDeltaSlot.ResetDirty();
        this->surfregMinDisplScl = this->qsGridDelta / 100.0f
                * this->regForcesScl;
        this->surfMappedMinDisplScl = this->qsGridDelta / 100.0f
                * this->surfaceMappingForcesScl;
        this->triggerComputeVolume = true;
    }

    // Quicksurf isovalue
    if (this->qsIsoValSlot.IsDirty()) {
        this->qsIsoVal =
                this->qsIsoValSlot.Param<core::param::FloatParam>()->Value();
        this->qsIsoValSlot.ResetDirty();
        this->triggerComputeVolume = true;
    }

    /* Subdivision */

    /// Parameter to toggle rendering of subdivided triangles
    if (this->showSubdivSlot.IsDirty()) {
        this->showSubdiv =
                this->showSubdivSlot.Param<core::param::BoolParam>()->Value();
        this->showSubdivSlot.ResetDirty();
        this->triggerComputeSubdiv = true;
    }

    /// Parameter for maximum subdivision level
    if (this->maxSubdivLevelSlot.IsDirty()) {
        this->maxSubdivLevel = this->maxSubdivLevelSlot.Param<
                core::param::IntParam>()->Value();
        this->maxSubdivLevelSlot.ResetDirty();
        this->triggerComputeSubdiv = true;
        this->triggerSurfaceMapping = true;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }

    /// Parameter for maximum morphing steps
    if (this->maxSubdivStepsSlot.IsDirty()) {
        this->maxSubdivSteps = this->maxSubdivStepsSlot.Param<
                core::param::IntParam>()->Value();
        this->maxSubdivStepsSlot.ResetDirty();
        this->triggerComputeSubdiv = true;
        this->triggerSurfaceMapping = true;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }

    /// Parameter for initial stepsize
    if (this->subStepLevelSlot.IsDirty()) {
        this->subStepLevel =
                this->subStepLevelSlot.Param<core::param::IntParam>()->Value();
        this->subStepLevelSlot.ResetDirty();
        this->triggerComputeSubdiv = true;
        this->triggerSurfaceMapping = true;
        this->triggerComputeSurfacePoints1 = true;
        this->triggerComputeSurfacePoints2 = true;
    }
}

#ifdef USE_PROCEDURAL_DATA
/*
 * ComparativeMolSurfaceRenderer::initProcFieldData
 */
bool ComparativeMolSurfaceRenderer::initProcFieldData() {
    typedef vislib::math::Vector<float, 3> Vec3f;

//    printf("vol dim %i %i %i\n", this->volDim.x, this->volDim.y, this->volDim.z);
    size_t fieldSize = this->volDim.x*this->volDim.y*this->volDim.z;
    // Allocate memory
    if (!CudaSafeCall(this->procField1D.Validate(fieldSize))) {
        return false;
    }
    if (!CudaSafeCall(this->procField2D.Validate(fieldSize))) {
        return false;
    }
    this->procField1.Validate(fieldSize);
    this->procField2.Validate(fieldSize);
    this->procField1.Set(0x00);
    this->procField2.Set(0x00);

    HostArr<float> testField1;
    HostArr<float> testField2;

    testField1.Validate(fieldSize);
    testField2.Validate(fieldSize);
    testField1.Set(0x00);
    testField2.Set(0x00);

    // Copy to device array
    if (!CudaSafeCall(cudaMemcpy(testField1.Peek(),
                            ((CUDAQuickSurf*)this->cudaqsurf1)->getMap(),
                            sizeof(float)*fieldSize,
                            cudaMemcpyDeviceToHost))) {
        return false;
    }
    // Copy to device array
    if (!CudaSafeCall(cudaMemcpy(testField2.Peek(),
                            ((CUDAQuickSurf*)this->cudaqsurf2)->getMap(),
                            sizeof(float)*fieldSize,
                            cudaMemcpyDeviceToHost))) {
        return false;
    }

//    /* Alternative 1: Generate two spheres with different radii */
//
//    const float rad1 = 15.0f;
//    const float rad2 = 20.0f;
//    Vec3f center(
//            this->volOrg.x + (this->volMaxC.x-this->volOrg.x)*0.5,
//            this->volOrg.y + (this->volMaxC.y-this->volOrg.y)*0.5,
//            this->volOrg.z + (this->volMaxC.z-this->volOrg.z)*0.5);
//    // Loop through all grid points
//    for (int x = 0; x < static_cast<int>(this->volDim.x); ++x) {
//        for (int y = 0; y < static_cast<int>(this->volDim.y); ++y) {
//            for (int z = 0; z < static_cast<int>(this->volDim.z); ++z) {
//                Vec3f pos(this->volOrg.x + x*this->volDelta.x,
//                          this->volOrg.y + y*this->volDelta.y,
//                          this->volOrg.z + z*this->volDelta.z);
//
//                float a1 = 1.0;
//                float c1 = 250.0;
//                float a2 = 1.0;
//                float c2 = 120.0;
//                float len = (pos-center).Length();
//                float lenSqrt = len*len;
////                this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = a1*std::exp(lenSqrt/(4.0*c1));
////                this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = a2*std::exp(lenSqrt/(4.0*c2));
//
//                this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = std::exp(-lenSqrt/(4.0*c1));
//                this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = std::exp(-lenSqrt/(4.0*c2));
//
////                printf("f1: %f, f2: %f\n",
////                        this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x],
////                        this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x]);
//
////                if ((pos-center).Length() <= rad1) {
////                    this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = this->qsIsoVal+1.0;
////                } else {
////                    this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = this->qsIsoVal-1.0;
////                }
////                if ((pos-center).Length() <= rad2) {
////                    this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = this->qsIsoVal+1.0;
////                } else {
////                    this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = this->qsIsoVal-1.0;
////                }
//            }
//        }
//    }
//

    /* Alternative 2: One sphere and one torus */

    const float rad1 = 18.5f;// sizediff 0
    //const float rad1 = 13.0f;  // sizediff 1
    //const float rad1 = 5.0f;  // sizediff 2
    const float rad2 = 20.5f;
    Vec3f center(
            this->volOrg.x + (this->volMaxC.x-this->volOrg.x)*0.5,
            this->volOrg.y + (this->volMaxC.y-this->volOrg.y)*0.5,
            this->volOrg.z + (this->volMaxC.z-this->volOrg.z)*0.5);
    // Loop through all grid points
    for (int x = 0; x < static_cast<int>(this->volDim.x); ++x) {
        for (int y = 0; y < static_cast<int>(this->volDim.y); ++y) {
            for (int z = 0; z < static_cast<int>(this->volDim.z); ++z) {
                Vec3f pos(this->volOrg.x + x*this->volDelta.x,
                        this->volOrg.y + y*this->volDelta.y,
                        this->volOrg.z + z*this->volDelta.z);

                float a1 = 1.0;
                float c1 = 250.0;
                float a2 = 1.0;
                float c2 = 120.0;
                float len = (pos-center).Length();
                float lenSqrt = len*len;
//                this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = a1*std::exp(lenSqrt/(4.0*c1));
//                this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = a2*std::exp(lenSqrt/(4.0*c2));

                //this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = std::exp(-lenSqrt/(4.0*c1));
                this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = (len - rad1)*-1.0;

                // Implicit equation for torus: (x^2+y^2+z^2+R^2-r^2)-4R^2(x^2+y^2) = 0
                //Vec3f offs = Vec3f(15.0, 15.0, 15.0);
                //Vec3f offs = Vec3f(0.0, 30.0, 0.0);
                Vec3f offs = Vec3f(0.0, 5.0, 0.0);
                Vec3f distVec = (pos-(center+offs));
                a2 = 1.0;
                c2 = 12.0;
                float R = 13.0f;
                float rad = 5.5;
                len = pow(distVec.X()*distVec.X()+
                        distVec.Y()*distVec.Y()+
                        distVec.Z()*distVec.Z()+
                        R*R-rad*rad, 2.0f)-
                4.0f*R*R*(distVec.X()*distVec.X()+
                        distVec.Y()*distVec.Y());
                lenSqrt = len*len;
                this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = len*-1.0;

//                printf("f1: %f, f2: %f\n",
//                        this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x],
//                        this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x]);

//                if ((pos-center).Length() <= rad1) {
//                    this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = this->qsIsoVal+1.0;
//                } else {
//                    this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = this->qsIsoVal-1.0;
//                }
//                if ((pos-center).Length() <= rad2) {
//                    this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = this->qsIsoVal+1.0;
//                } else {
//                    this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = this->qsIsoVal-1.0;
//                }
            }
        }
    }

    /* Alternative 3: One sphere and one sphere + enclosed cavity */

//    const float rad1 = 15.0f;
//    const float rad2 = 25.0f;
//    Vec3f center(
//            this->volOrg.x + (this->volMaxC.x-this->volOrg.x)*0.5,
//            this->volOrg.y + (this->volMaxC.y-this->volOrg.y)*0.5,
//            this->volOrg.z + (this->volMaxC.z-this->volOrg.z)*0.5);
//    // Loop through all grid points
//    for (int x = 0; x < static_cast<int>(this->volDim.x); ++x) {
//        for (int y = 0; y < static_cast<int>(this->volDim.y); ++y) {
//            for (int z = 0; z < static_cast<int>(this->volDim.z); ++z) {
//                Vec3f pos(this->volOrg.x + x*this->volDelta.x,
//                        this->volOrg.y + y*this->volDelta.y,
//                        this->volOrg.z + z*this->volDelta.z);
//
//                float a1 = 1.0;
//                float c1 = 250.0;
//                float a2 = 1.0;
//                float c2 = 120.0;
//                Vec3f offs(0.0, 25.0, 0.0);
//                float len = (pos-(center+offs)).Length();
//                float lenSqrt = len*len;
//                //                this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = a1*std::exp(lenSqrt/(4.0*c1));
//                //                this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = a2*std::exp(lenSqrt/(4.0*c2));
//
//                this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = std::exp(-lenSqrt/(4.0*c1));
//                Vec3f distToCenter = pos-center;
//                len = distToCenter.Length();
//                float diff1 = abs(len-rad1);
//                float diff2 = abs(len-rad2);
//                len = min(diff1, diff2);
//                lenSqrt = len*len;
////                this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = std::exp(-lenSqrt/(4.0*c2));
//                this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = (10.0-len)+this->qsIsoVal;
//
//                //                printf("f1: %f, f2: %f\n",
//                //                        this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x],
//                //                        this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x]);
//
//                //                if ((pos-center).Length() <= rad1) {
//                //                    this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = this->qsIsoVal+1.0;
//                //                } else {
//                //                    this->procField1.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = this->qsIsoVal-1.0;
//                //                }
//                //                if ((pos-center).Length() <= rad2) {
//                //                    this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = this->qsIsoVal+1.0;
//                //                } else {
//                //                    this->procField2.Peek()[this->volDim.x*(this->volDim.y*z+y)+x] = this->qsIsoVal-1.0;
//                //                }
//            }
//        }
//    }
    // Copy to device array
    if (!CudaSafeCall(cudaMemcpy(this->procField1D.Peek(),
                            this->procField1.Peek(),
                            sizeof(float)*fieldSize,
                            cudaMemcpyHostToDevice))) {
        return false;
    }
    if (!CudaSafeCall(cudaMemcpy(this->procField2D.Peek(),
                            this->procField2.Peek(),
                            sizeof(float)*fieldSize,
                            cudaMemcpyHostToDevice))) {
        return false;
    }

    testField1.Release();
    testField2.Release();
    return true;
}
#endif // USE_PROCEDUAL_DATA
