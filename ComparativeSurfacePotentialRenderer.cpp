//
// ComparativeSurfacePotentialRenderer.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 13, 2013
//     Author: scharnkn
//

#include <stdafx.h>
#include "ComparativeSurfacePotentialRenderer.h"

#if (defined(WITH_CUDA) && (WITH_CUDA))

#define POTENTIAL_VOLUME_RENDERER_CUDA_USE_TIMER

#include "VTIDataCall.h"
#include "MolecularDataCall.h"
#include "CUDAQuickSurf.h"
#include "ComparativeSurfacePotentialRenderer.cuh"
#include "Color.h"
#include "cuda_error_check.h"
#include "ogl_error_check.h"
#include "RMS.h"
#include "sort_triangles.cuh"
#include "VBODataCall.h"

#include "param/BoolParam.h"
#include "param/FloatParam.h"
#include "param/IntParam.h"
#include "param/EnumParam.h"
#include "param/StringParam.h"
#include "param/FilePathParam.h"
#include "CoreInstance.h"
#include "utility/ColourParser.h"
#include "view/CallRender3D.h"

#include "vislib/FramebufferObject.h"
#include "vislib/Matrix.h"
#include "vislib/Cuboid.h"

#include <glh/glh_genext.h>
#include <glh/glh_extensions.h>
#include <GL/glu.h>

#include <cuda_gl_interop.h>
#include "cuda_helper.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;

// Offsets and stride for vbos holding surface data
const uint ComparativeSurfacePotentialRenderer::vertexDataOffsPos = 0;
const uint ComparativeSurfacePotentialRenderer::vertexDataOffsNormal = 3;
const uint ComparativeSurfacePotentialRenderer::vertexDataOffsTexCoord = 6;
const uint ComparativeSurfacePotentialRenderer::vertexDataStride = 9;

// Offsets and stride for vbos holding mapped surface data
const uint ComparativeSurfacePotentialRenderer::vertexDataMappedOffsPosNew = 0;
const uint ComparativeSurfacePotentialRenderer::vertexDataMappedOffsPosOld = 3;
const uint ComparativeSurfacePotentialRenderer::vertexDataMappedOffsNormal = 6;
const uint ComparativeSurfacePotentialRenderer::vertexDataMappedOffsTexCoordNew = 9;
const uint ComparativeSurfacePotentialRenderer::vertexDataMappedOffsTexCoordOld = 12;
const uint ComparativeSurfacePotentialRenderer::vertexDataMappedOffsCorruptTriangleFlag = 15;
const uint ComparativeSurfacePotentialRenderer::vertexDataMappedStride = 16;

// Hardcoded parameters for 'quicksurf' class
const float ComparativeSurfacePotentialRenderer::qsParticleRad = 0.8f;
const float ComparativeSurfacePotentialRenderer::qsGaussLim = 10.0f;
const float ComparativeSurfacePotentialRenderer::qsGridSpacing = 1.0f;
const bool ComparativeSurfacePotentialRenderer::qsSclVanDerWaals = true;
const float ComparativeSurfacePotentialRenderer::qsIsoVal = 0.5f;

// Hardcoded colors for surface rendering
const Vec3f ComparativeSurfacePotentialRenderer::uniformColorSurf0 = Vec3f(0.36f, 0.36f, 1.0f);
const Vec3f ComparativeSurfacePotentialRenderer::uniformColorSurf1 = Vec3f(0.8f, 0.8f, 0.0f);
const Vec3f ComparativeSurfacePotentialRenderer::uniformColorSurfMapped = Vec3f(0.5f, 0.5f, 0.5f);
const Vec3f ComparativeSurfacePotentialRenderer::colorMaxPotential = Vec3f(0.0f, 0.0f, 1.0f);
const Vec3f ComparativeSurfacePotentialRenderer::colorMinPotential = Vec3f(1.0f, 0.0f, 0.0f);

// Maximum RMS value to enable valid mapping
const float ComparativeSurfacePotentialRenderer::maxRMSVal = 10.0f;

// TODO
// + Make CUDA implementation more efficient
//   -> compute gradient for the whole volume (put distance field and other volume together beforehand?)
// + Use CUDA resource for textures aswell?
// + Only (re)create vbos if memory is not sufficient
// + Make registering/unregistering ressources anjd such less chaotic

/*
 * ComparativeSurfacePotentialRenderer::ComparativeSurfacePotentialRenderer
 */
ComparativeSurfacePotentialRenderer::ComparativeSurfacePotentialRenderer(void) :
                Renderer3DModuleDS(),
                /* Callee slots */
                vboSlaveSlot0("vboOut0", "provides access to the vbo containing data for data set #0"),
                vboSlaveSlot1("vboOut1", "provides access to the vbo containing data for data set #1"),
                /* Caller slots */
                potentialDataCallerSlot0("getVolData0", "Connects the rendering with data storage"),
                potentialDataCallerSlot1("getVolData1", "Connects the rendering with data storage"),
                particleDataCallerSlot0("getParticleData0", "Connects the rendering with data storage"),
                particleDataCallerSlot1("getParticleData1", "Connects the rendering with data storage"),
                rendererCallerSlot("protren", "Connects the renderer with another render module"),
#if defined(USE_TEXTURE_SLICES)
                /* Parameters for slice rendering */
                sliceDataSetSlot("texslice::dataSet", "Choses the data set to be rendered"),
                sliceRMSlot("texslice::render", "The rendering mode for the slices"),
                xPlaneSlot("texslice::xPlanePos", "Change the position of the x-Plane"),
                yPlaneSlot("texslice::yPlanePos", "Change the position of the y-Plane"),
                zPlaneSlot("texslice::zPlanePos", "Change the position of the z-Plane"),
                toggleXPlaneSlot("texslice::showXPlane", "Change the position of the x-Plane"),
                toggleYPlaneSlot("texslice::showYPlane", "Change the position of the y-Plane"),
                toggleZPlaneSlot("texslice::showZPlane", "Change the position of the z-Plane"),
                sliceMinValSlot("texslice::minTex", "Minimum texture value"),
                sliceMaxValSlot("texslice::maxTex", "Maximum texture value"),
#endif // defined(USE_TEXTURE_SLICES)
                /* Parameters for frame by frame comparison */
                cmpModeSlot("cmpMode", "Param slot for compare mode"),
                singleFrame0Slot("singleFrame0", "Param for single frame #0"),
                singleFrame1Slot("singleFrame1", "Param for single frame #1"),
                /* Global rendering options */
                minPotentialSlot("minPotential", "Minimum potential value for the color map"),
                maxPotentialSlot("maxPotential", "Maximum potential value for the color map"),
                /* Global mapping options */
                interpolModeSlot("interpolation", "Change interpolation method"),
                /* Parameters for mapped surface */
                fittingModeSlot("surfMap::RMSDfitting", "RMSD fitting for the mapped surface"),
                surfaceMappingExternalForcesWeightSclSlot("surfMap::externalForces","Scale factor for the external forces weighting"),
                surfaceMappingForcesSclSlot("surfMap::sclAll","Overall scale factor for all forces"),
                surfaceMappingMaxItSlot("surfMap::maxIt","Maximum number of iterations for the surface mapping"),
                surfMappedMinDisplSclSlot("surfMap::minDisplScl", "Minimum displacement for vertices"),
#if defined(USE_DISTANCE_FIELD)
                surfMappedMaxDistSlot("surfMap::maxDist", "Maximum distance to use density map instead of distance field"),
#endif
                surfMappedSpringStiffnessSlot("surfMap::stiffness", "Spring stiffness"),
                surfaceMappedRMSlot("surfMap::render", "Render mode for the mapped surface"),
                surfaceMappedColorModeSlot("surfMap::color", "Color mode for the mapped surface"),
                surfMaxPosDiffSlot("surfMap::maxPosDiff", "Maximum value for euclidian distance"),
                surfMappedAlphaSclSlot("surfMap::alphaScl", "Transparency scale factor"),
                /* Surface regularization */
                regMaxItSlot("surfReg::regMaxIt", "Maximum number of iterations for regularization"),
                regSpringStiffnessSlot("surfReg::stiffness", "Spring stiffness"),
                regExternalForcesWeightSlot("surfReg::externalForcesWeight", "Weight of the external forces"),
                regForcesSclSlot("surfReg::forcesScl", "Scaling of overall force"),
                /* Surface rendering */
                surface0RMSlot("surf0::render", "Render mode for the first surface"),
                surface0ColorModeSlot("surf0::color", "Color mode for the first surface"),
                surf0AlphaSclSlot("surf0::alphaScl", "Transparency scale factor"),
                surface1RMSlot("surf1::render", "Render mode for the second surface"),
                surface1ColorModeSlot("surf1::color", "Render mode for the second surface"),
                surf1AlphaSclSlot("surf1::alphaScl", "Transparency scale factor"),
                /* Raycasting */
                potentialTex0(0), potentialTex1(0),
                /* Volume generation */
                cudaqsurf0(NULL), cudaqsurf1(NULL),
#if defined(USE_TEXTURE_SLICES)
                volumeTex0(0), volumeTex1(0),
#endif // defined(USE_TEXTURE_SLICES)
                minAtomRad(0), maxAtomRad(0),
                /* The data */
                datahashParticles0(0), datahashParticles1(0),
                datahashPotential0(0), datahashPotential1(0),
                calltimeOld(-1.0f),
                /* Boolean flags */
                triggerComputeVolume(true), triggerInitPotentialTex(true),
                triggerComputeSurfacePoints0(true), triggerComputeSurfacePoints1(true),
                triggerSurfaceMapping(true),
                /* RMS fitting */
                rmsValue(-1.0), toggleRMSFit(true),
#if defined(USE_DISTANCE_FIELD)
#if defined(USE_TEXTURE_SLICES)
                distFieldTex(0),
#endif // defined(USE_TEXTURE_SLICES)
                triggerComputeDistanceField(true),
#endif
                /* Vbo stuff and mapped memory */
                vbo0(0), vbo0Resource(NULL), vbo1(0), vbo1Resource(NULL),
                vboMapped(0), vboMappedResource(NULL),
                vboTriangleIdx0(0), vboTriangleIdx0Resource(NULL),
                vboTriangleIdx1(0), vboTriangleIdx1Resource(NULL),
                vboTriangleIdxMapped(0), vboTriangleIdxMappedResource(NULL),
                vertexCnt0(0), vertexCnt1(0), triangleCnt0(0), triangleCnt1(0) {

    this->vboSlaveSlot0.SetCallback(VBODataCall::ClassName(),
            VBODataCall::FunctionName(VBODataCall::CallForGetExtent),
            &ComparativeSurfacePotentialRenderer::getVBOExtent);
    this->vboSlaveSlot0.SetCallback(VBODataCall::ClassName(),
            VBODataCall::FunctionName(VBODataCall::CallForGetData),
            &ComparativeSurfacePotentialRenderer::getVBOData0);
    this->MakeSlotAvailable(&this->vboSlaveSlot0);

    this->vboSlaveSlot1.SetCallback(VBODataCall::ClassName(),
            VBODataCall::FunctionName(VBODataCall::CallForGetExtent),
            &ComparativeSurfacePotentialRenderer::getVBOExtent);
    this->vboSlaveSlot1.SetCallback(VBODataCall::ClassName(),
            VBODataCall::FunctionName(VBODataCall::CallForGetData),
            &ComparativeSurfacePotentialRenderer::getVBOData1);
    this->MakeSlotAvailable(&this->vboSlaveSlot1);

    // Data caller slot for the potential maps
    this->potentialDataCallerSlot0.SetCompatibleCall<VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->potentialDataCallerSlot0);
    this->potentialDataCallerSlot1.SetCompatibleCall<VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->potentialDataCallerSlot1);

    // Data caller slot for the particles
    this->particleDataCallerSlot0.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->particleDataCallerSlot0);
    this->particleDataCallerSlot1.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->particleDataCallerSlot1);

    // Renderer caller slot
    this->rendererCallerSlot.SetCompatibleCall<view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rendererCallerSlot);


#if defined(USE_TEXTURE_SLICES)
    /* Parameters for slice rendering */

    // Data set for slice rendering
    this->sliceDataSet = 0;
    param::EnumParam *sds = new core::param::EnumParam(this->sliceDataSet);
    sds->SetTypePair(0, "Data set #0");
    sds->SetTypePair(1, "Data set #1");
    this->sliceDataSetSlot << sds;
    this->MakeSlotAvailable(&this->sliceDataSetSlot);

    // Render modes for slices
    this->sliceRM = 0;
    param::EnumParam *srm = new core::param::EnumParam(this->sliceRM);
    srm->SetTypePair(0, "Potential");
    srm->SetTypePair(1, "Density");
#if defined(USE_DISTANCE_FIELD)
    srm->SetTypePair(2, "Distance field");
#endif // defined(USE_DISTANCE_FIELD)
    this->sliceRMSlot << srm;
    this->MakeSlotAvailable(&this->sliceRMSlot);

    // X-plane position
    this->xPlane = 0.0f;
    this->xPlaneSlot.SetParameter(new core::param::FloatParam(this->xPlane, -120.0f, 120.0f));
    this->MakeSlotAvailable(&this->xPlaneSlot);

    // X-plane visibility
    this->showXPlane = false;
    this->toggleXPlaneSlot.SetParameter(new core::param::BoolParam(this->showXPlane));
    this->MakeSlotAvailable(&this->toggleXPlaneSlot);

    // Y-plane position
    this->yPlane = 0.0f;
    this->yPlaneSlot.SetParameter(new core::param::FloatParam(this->yPlane, -120.0f, 120.0f));
    this->MakeSlotAvailable(&this->yPlaneSlot);

    // Y-plane visibility
    this->showYPlane = false;
    this->toggleYPlaneSlot.SetParameter(new core::param::BoolParam(this->showYPlane));
    this->MakeSlotAvailable(&this->toggleYPlaneSlot);

    // Z-plane position
    this->zPlane = 0.0f;
    this->zPlaneSlot.SetParameter(new core::param::FloatParam(this->zPlane, -120.0f, 120.0f));
    this->MakeSlotAvailable(&this->zPlaneSlot);

    // Z-plane visibility
    this->showZPlane = false;
    this->toggleZPlaneSlot.SetParameter(new core::param::BoolParam(this->showZPlane));
    this->MakeSlotAvailable(&this->toggleZPlaneSlot);

    // Minimum texture value
    this->sliceMinVal = -1.0f;
    this->sliceMinValSlot.SetParameter(new core::param::FloatParam(this->sliceMinVal));
    this->MakeSlotAvailable(&this->sliceMinValSlot);

    // Maximum texture value
    this->sliceMaxVal = 1.0f;
    this->sliceMaxValSlot.SetParameter(new core::param::FloatParam(this->sliceMaxVal));
    this->MakeSlotAvailable(&this->sliceMaxValSlot);

#endif // defined(USE_TEXTURE_SLICES)


    /* Parameters for frame-by-frame comparison */

    // Param slot for compare mode
    this->cmpMode = COMPARE_1_1;
    param::EnumParam *cmpm = new core::param::EnumParam(int(this->cmpMode));
    cmpm->SetTypePair(COMPARE_1_1, "1-1");
    cmpm->SetTypePair(COMPARE_1_N, "1-n");
    cmpm->SetTypePair(COMPARE_N_1, "n-1");
    cmpm->SetTypePair(COMPARE_N_N, "n-n");
    this->cmpModeSlot << cmpm;
    this->MakeSlotAvailable(&this->cmpModeSlot);

    // Param for single frame #0
    this->singleFrame0 = 0;
    this->singleFrame0Slot.SetParameter(new core::param::IntParam(this->singleFrame0, 0));
    this->MakeSlotAvailable(&this->singleFrame0Slot);

    // Param for single frame #1
    this->singleFrame1 = 0;
    this->singleFrame1Slot.SetParameter(new core::param::IntParam(this->singleFrame1, 0));
    this->MakeSlotAvailable(&this->singleFrame1Slot);


    /* Global rendering options */

    /// Parameter for minimum potential value for the color map
    this->minPotential = -1.0f;
    this->minPotentialSlot.SetParameter(new core::param::FloatParam(this->minPotential));
    this->MakeSlotAvailable(&this->minPotentialSlot);

    /// Parameter for maximum potential value for the color map
    this->maxPotential = 1.0f;
    this->maxPotentialSlot.SetParameter(new core::param::FloatParam(this->maxPotential));
    this->MakeSlotAvailable(&this->maxPotentialSlot);


    /* Global mapping options */

    // Interpolation method used when computing external forces
    this->interpolMode = INTERP_LINEAR;
    param::EnumParam *smi = new core::param::EnumParam(int(this->interpolMode));
    smi->SetTypePair(INTERP_LINEAR, "Linear");
    smi->SetTypePair(INTERP_CUBIC, "Cubic");
    this->interpolModeSlot << smi;
    this->MakeSlotAvailable(&this->interpolModeSlot);


    /* Parameters for mapped surface */

    // Parameter for the RMS fitting mode
    this->fittingMode = RMS_NONE;
    param::EnumParam *rms = new core::param::EnumParam(int(this->fittingMode));
    rms->SetTypePair(RMS_NONE, "None");
    rms->SetTypePair(RMS_ALL, "All");
    rms->SetTypePair(RMS_BACKBONE, "Backbone");
    rms->SetTypePair(RMS_C_ALPHA, "C alpha");
    this->fittingModeSlot << rms;
    this->MakeSlotAvailable(&this->fittingModeSlot);

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

    // Parameter for mapped surface render mode
    this->surfaceMappedRM = SURFACE_NONE;
    param::EnumParam *msrm = new core::param::EnumParam(int(this->surfaceMappedRM));
    msrm->SetTypePair(SURFACE_NONE, "None");
    msrm->SetTypePair(SURFACE_POINTS, "Points");
    msrm->SetTypePair(SURFACE_WIREFRAME, "Wireframe");
    msrm->SetTypePair(SURFACE_FILL, "Fill");
    this->surfaceMappedRMSlot << msrm;
    this->MakeSlotAvailable(&this->surfaceMappedRMSlot);

    // Parameter for mapped surface color mode
    this->surfaceMappedColorMode = SURFACE_UNI;
    param::EnumParam *mscm = new core::param::EnumParam(int(this->surfaceMappedColorMode));
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

    // Param for minimum value for surface coloring
    this->surfMappedMinDisplScl = 0.0f;
    this->surfMappedMinDisplSclSlot.SetParameter(new core::param::FloatParam(this->surfMappedMinDisplScl, 0.0f));
    this->MakeSlotAvailable(&this->surfMappedMinDisplSclSlot);

#if defined(USE_DISTANCE_FIELD)
    // Param for maximum distance to use density map instead of distance field
    this->surfMappedMaxDist = 10.0f;
    this->surfMappedMaxDistSlot.SetParameter(new core::param::FloatParam(this->surfMappedMaxDist, 0.0f));
    this->MakeSlotAvailable(&this->surfMappedMaxDistSlot);
#endif

    // Param for spring stiffness
    this->surfMappedSpringStiffness = 1.0f;
    this->surfMappedSpringStiffnessSlot.SetParameter(new core::param::FloatParam(this->surfMappedSpringStiffness, 0.0f));
    this->MakeSlotAvailable(&this->surfMappedSpringStiffnessSlot);

    // Param for maximum value for euclidian distance // Not needed atm
    this->surfMaxPosDiff = 1.0f;
    this->surfMaxPosDiffSlot.SetParameter(new core::param::FloatParam(this->surfMaxPosDiff, 0.1f));
    this->MakeSlotAvailable(&this->surfMaxPosDiffSlot);

    // Param for transparency scaling
    this->surfMappedAlphaScl = 1.0f;
    this->surfMappedAlphaSclSlot.SetParameter(new core::param::FloatParam(this->surfMappedAlphaScl, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surfMappedAlphaSclSlot);


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


    /* Rendering options of surface #0 and #1 */

    // Parameter for surface 0 render mode
    this->surface0RM = SURFACE_NONE;
    param::EnumParam *msrm0 = new core::param::EnumParam(int(this->surface0RM));
    msrm0->SetTypePair(SURFACE_NONE, "None");
    msrm0->SetTypePair(SURFACE_POINTS, "Points");
    msrm0->SetTypePair(SURFACE_WIREFRAME, "Wireframe");
    msrm0->SetTypePair(SURFACE_FILL, "Fill");
    this->surface0RMSlot << msrm0;
    this->MakeSlotAvailable(&this->surface0RMSlot);

    // Parameter for surface color 0 mode
    this->surface0ColorMode = SURFACE_UNI;
    param::EnumParam *mscm0 = new core::param::EnumParam(int(this->surface0ColorMode));
    mscm0->SetTypePair(SURFACE_UNI, "Uniform");
    mscm0->SetTypePair(SURFACE_NORMAL, "Normal");
    mscm0->SetTypePair(SURFACE_TEXCOORDS, "TexCoords");
    mscm0->SetTypePair(SURFACE_POTENTIAL, "Potential");
    this->surface0ColorModeSlot << mscm0;
    this->MakeSlotAvailable(&this->surface0ColorModeSlot);

    // Param for transparency scaling
    this->surf0AlphaScl = 1.0f;
    this->surf0AlphaSclSlot.SetParameter(new core::param::FloatParam(this->surf0AlphaScl, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surf0AlphaSclSlot);

    // Parameter for surface 1 render mode
    this->surface1RM = SURFACE_NONE;
    param::EnumParam *msrm1 = new core::param::EnumParam(int(this->surface1RM));
    msrm1->SetTypePair(SURFACE_NONE, "None");
    msrm1->SetTypePair(SURFACE_POINTS, "Points");
    msrm1->SetTypePair(SURFACE_WIREFRAME, "Wireframe");
    msrm1->SetTypePair(SURFACE_FILL, "Fill");
    this->surface1RMSlot << msrm1;
    this->MakeSlotAvailable(&this->surface1RMSlot);

    // Parameter for surface color 1 mode
    this->surface1ColorMode = SURFACE_UNI;
    param::EnumParam *mscm1 = new core::param::EnumParam(int(this->surface1ColorMode));
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


    /* Initialize grid parameters */

    this->gridDensMap0.minC[0] = -1.0f;
    this->gridDensMap0.minC[1] = -1.0f;
    this->gridDensMap0.minC[2] = -1.0f;
    this->gridDensMap0.maxC[0] = 1.0f;
    this->gridDensMap0.maxC[1] = 1.0f;
    this->gridDensMap0.maxC[2] = 1.0f;
    this->gridDensMap0.delta[0] = 1.0f;
    this->gridDensMap0.delta[1] = 1.0f;
    this->gridDensMap0.delta[2] = 1.0f;
    this->gridDensMap0.size[0] = 2;
    this->gridDensMap0.size[1] = 2;
    this->gridDensMap0.size[2] = 2;

    this->gridDensMap1.minC[0] = -1.0f;
    this->gridDensMap1.minC[1] = -1.0f;
    this->gridDensMap1.minC[2] = -1.0f;
    this->gridDensMap1.maxC[0] = 1.0f;
    this->gridDensMap1.maxC[1] = 1.0f;
    this->gridDensMap1.maxC[2] = 1.0f;
    this->gridDensMap1.delta[0] = 1.0f;
    this->gridDensMap1.delta[1] = 1.0f;
    this->gridDensMap1.delta[2] = 1.0f;
    this->gridDensMap1.size[0] = 2;
    this->gridDensMap1.size[1] = 2;
    this->gridDensMap1.size[2] = 2;

    this->gridPotential0.minC[0] = -1.0f;
    this->gridPotential0.minC[1] = -1.0f;
    this->gridPotential0.minC[2] = -1.0f;
    this->gridPotential0.maxC[0] = 1.0f;
    this->gridPotential0.maxC[1] = 1.0f;
    this->gridPotential0.maxC[2] = 1.0f;
    this->gridPotential0.delta[0] = 1.0f;
    this->gridPotential0.delta[1] = 1.0f;
    this->gridPotential0.delta[2] = 1.0f;
    this->gridPotential0.size[0] = 2;
    this->gridPotential0.size[1] = 2;
    this->gridPotential0.size[2] = 2;

    this->gridPotential1.minC[0] = -1.0f;
    this->gridPotential1.minC[1] = -1.0f;
    this->gridPotential1.minC[2] = -1.0f;
    this->gridPotential1.maxC[0] = 1.0f;
    this->gridPotential1.maxC[1] = 1.0f;
    this->gridPotential1.maxC[2] = 1.0f;
    this->gridPotential1.delta[0] = 1.0f;
    this->gridPotential1.delta[1] = 1.0f;
    this->gridPotential1.delta[2] = 1.0f;
    this->gridPotential1.size[0] = 2;
    this->gridPotential1.size[1] = 2;
    this->gridPotential1.size[2] = 2;

}


/*
 * ComparativeSurfacePotentialRenderer::~ComparativeSurfacePotentialRenderer
 */
ComparativeSurfacePotentialRenderer::~ComparativeSurfacePotentialRenderer(void) {
    this->Release();
}


/*
 * ComparativeSurfacePotentialRenderer::applyRMSFittingToPosArray
 */
bool ComparativeSurfacePotentialRenderer::applyRMSFittingToPosArray(
        MolecularDataCall *mol,
        cudaGraphicsResource **cudaTokenVboMapped,
        uint vertexCnt) {

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

        // Get mapped pointer to the vbo
        float *vboPt;
        size_t vboSize;
        if (!CudaSafeCall(cudaGraphicsMapResources(1, cudaTokenVboMapped, 0))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
                reinterpret_cast<void**>(&vboPt), // The mapped pointer
                &vboSize,             // The size of the accessible data
                *cudaTokenVboMapped))) {                   // The mapped resource
            return false;
        }

//        // DEBUG Print mapped positions
//        printf("Apply RMS,  positions before:\n");
//        HostArr<float> vertexPos;
//        vertexPos.Validate(vertexCnt*3);
//        cudaMemcpy(vertexPos.Peek(), vboPt,
//                sizeof(float)*this->vertexDataStride*3,
//                cudaMemcpyDeviceToHost);
//        for (int k = 0; k < 10; ++k) {
//            printf("%i: Vertex position (%f %f %f)\n", k, vertexPos.Peek()[this->vertexDataStride*k+0],
//                    vertexPos.Peek()[this->vertexDataStride*k+1], vertexPos.Peek()[this->vertexDataStride*k+2]);
//
//        }
//        // End DEBUG

        // Move vertex positions to origin (with respect to centroid)
        if (!CudaSafeCall(TranslatePos(
                vboPt,
                this->vertexDataStride,
                this->vertexDataOffsPos,
                make_float3(-centroid.X(), -centroid.Y(), -centroid.Z()),
                vertexCnt))) {
            return false;
        }

        // Rotate for best fit
        rotate_D.Validate(9);
        if (!CudaSafeCall(cudaMemcpy((void *)rotate_D.Peek(), this->rmsRotation.PeekComponents(),
                9*sizeof(float), cudaMemcpyHostToDevice))) {
            return false;
        }
        if (!CudaSafeCall(RotatePos(
                vboPt,
                this->vertexDataStride,
                this->vertexDataOffsPos,
                rotate_D.Peek(),
                vertexCnt))) {
            return false;
        }

        // Move vertex positions to centroid of second data set
        if (!CudaSafeCall(TranslatePos(
                vboPt,
                this->vertexDataStride,
                this->vertexDataOffsPos,
                make_float3(this->rmsTranslation.X(),
                            this->rmsTranslation.Y(),
                            this->rmsTranslation.Z()),
                vertexCnt))) {
            return false;
        }

        // Clean up
        rotate_D.Release();

//        // DEBUG
//        printf("RMS centroid %f %f %f\n", centroid.X(), centroid.Y(), centroid.Z());
//        printf("RMS translation %f %f %f\n", this->rmsTranslation.X(),
//                this->rmsTranslation.Y(), this->rmsTranslation.Z());
//        printf("RMS rotation \n %f %f %f\n%f %f %f\n%f %f %f\n",
//                this->rmsRotation.GetAt(0, 0), this->rmsRotation.GetAt(0, 1),
//                this->rmsRotation.GetAt(0, 2), this->rmsRotation.GetAt(1, 0),
//                this->rmsRotation.GetAt(1, 1), this->rmsRotation.GetAt(1, 2),
//                this->rmsRotation.GetAt(2, 0), this->rmsRotation.GetAt(2, 1),
//                this->rmsRotation.GetAt(2, 2));


//        // DEBUG Print mapped positions
//        printf("Apply RMS,  positions after:\n");
//        //HostArr<float> vertexPos;
//        vertexPos.Validate(vertexCnt*3);
//        cudaMemcpy(vertexPos.Peek(), vboPt,
//                sizeof(float)*this->vertexDataStride*3,
//                cudaMemcpyDeviceToHost);
//        for (int k = 0; k < 10; ++k) {
//            printf("%i: Vertex position (%f %f %f)\n", k, vertexPos.Peek()[this->vertexDataStride*k+0],
//                    vertexPos.Peek()[this->vertexDataStride*k+1], vertexPos.Peek()[this->vertexDataStride*k+2]);
//
//        }
//        // End DEBUG

        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, cudaTokenVboMapped, 0))) {
            return false;
        }
    }

    return true;
}


/*
 * ComparativeSurfacePotentialRenderer::computeVolumeTex
 */
bool ComparativeSurfacePotentialRenderer::computeDensityMap(
        const MolecularDataCall *mol,
        CUDAQuickSurf *cqs,
        gridParams &gridDensMap,
        const vislib::math::Cuboid<float> &bboxParticles
#if defined (USE_TEXTURE_SLICES)
        , HostArr<float> &volume, GLuint &volumeTex
#endif // defined (USE_TEXTURE_SLICES)
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
                        this->maxAtomRad = ::fmaxf(this->maxAtomRad, this->gridDataPos.Peek()[4*particleCnt+3]);
                        this->minAtomRad = ::fminf(this->minAtomRad, this->gridDataPos.Peek()[4*particleCnt+3]);

                        particleCnt++;
                    }
                }
            }
        }
    }

    // Compute padding for the density map
    padding = this->maxAtomRad*this->qsParticleRad + this->qsGridSpacing*10;

//        // DEBUG
//        printf("potential , org %f %f %f, maxCoord %f %f %f\n",
//                bboxParticles.GetOrigin().X(),
//                bboxParticles.GetOrigin().Y(),
//                bboxParticles.GetOrigin().Z(),
//                bboxParticles.GetRightTopFront().X(),
//                bboxParticles.GetRightTopFront().Y(),
//                bboxParticles.GetRightTopFront().Z());

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

#if defined (USE_TEXTURE_SLICES)

    // (Re-)allocate volume memory if necessary
    volume.Validate(volSize);

    CudaSafeCall(cudaMemcpy(volume.Peek(), cqs->getMap(),
            sizeof(float)*volume.GetCount(), cudaMemcpyDeviceToHost));

    //  Setup texture
    glEnable(GL_TEXTURE_3D);
    if(!glIsTexture(volumeTex)) {
        glGenTextures(1, &volumeTex);
    }
    glBindTexture(GL_TEXTURE_3D, volumeTex);
    glTexImage3DEXT(GL_TEXTURE_3D,
            0,
            GL_ALPHA,
            gridDensMap.size[0],
            gridDensMap.size[1],
            gridDensMap.size[2],
            0,
            GL_ALPHA,
            GL_FLOAT,
            volume.Peek());
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);

#endif // defined (USE_TEXTURE_SLICES)

    CheckForGLError();

    return true;
}


#if defined(USE_DISTANCE_FIELD)
/*
 * ComparativeSurfacePotentialRenderer::computeDistField
 */
bool ComparativeSurfacePotentialRenderer::computeDistField(
        const MolecularDataCall *mol,
        cudaGraphicsResource **vboResource,
        uint vertexCnt,
        CudaDevArr<float> &distField_D,
        float *volume_D,
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

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    CudaSafeCall(cudaGraphicsMapResources(1, vboResource, 0));
    CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,             // The size of the accessible data
            *vboResource));                   // The mapped resource

    // Compute distance field
    if (!CudaSafeCall(ComputeDistField(
            vboPt,
            distField_D.Peek(),
            volSize,
            vertexCnt,
            this->vertexDataOffsPos,
            this->vertexDataStride))) {
        return false;
    }

#if defined(USE_TEXTURE_SLICES)
    // Copy back to host
    this->distField.Validate(distField_D.GetSize());
    if (!CudaSafeCall(distField_D.CopyToHost(this->distField.Peek()))) {
        return false;
    }
#endif // defined(USE_TEXTURE_SLICES)

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, vboResource))) {
        return false;
    }

#if defined(USE_TEXTURE_SLICES)
    //  Setup texture
    glEnable(GL_TEXTURE_3D);
    if (!glIsTexture(this->distFieldTex)) {
        glGenTextures(1, &this->distFieldTex);
    }
    glBindTexture(GL_TEXTURE_3D, this->distFieldTex);
    glTexImage3DEXT(GL_TEXTURE_3D,
            0,
            GL_RGBA32F,
            gridDistField.size[0],
            gridDistField.size[1],
            gridDistField.size[2],
            0,
            GL_ALPHA,
            GL_FLOAT,
            this->distField.Peek());
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

    CheckForGLError();
#endif // defined(USE_TEXTURE_SLICES)

    return true;
}
#endif // defined(USE_DISTANCE_FIELD)


/*
 * ComparativeSurfacePotentialRenderer::create
 */
bool ComparativeSurfacePotentialRenderer::create() {
    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    // Create quicksurf objects
    if(!this->cudaqsurf0) {
        this->cudaqsurf0 = new CUDAQuickSurf();
    }
    if(!this->cudaqsurf1) {
        this->cudaqsurf1 = new CUDAQuickSurf();
    }

    // Init extensions
    if(!glh_init_extensions("\
            GL_VERSION_2_0 GL_EXT_texture3D \
            GL_EXT_framebuffer_object \
            GL_ARB_multitexture \
            GL_ARB_draw_buffers \
            GL_ARB_vertex_buffer_object")) {
        return false;
    }
    if(!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    // Load shader sources
    ShaderSource vertSrc, fragSrc, geomSrc;

    core::CoreInstance *ci = this->GetCoreInstance();
    if(!ci) return false;

    // Load shader for per pixel lighting of the surface
    if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("electrostatics::pplsurface::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    // Load ppl fragment shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("electrostatics::pplsurface::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the ppl shader", this->ClassName());
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

#if defined(USE_TEXTURE_SLICES)
    // Load slice shader
    if(!ci->ShaderSourceFactory().MakeShaderSource("electrostatics::slice::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source: slice shader", this->ClassName());
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("electrostatics::slice::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load fragment shader source:  slice shader", this->ClassName());
        return false;
    }
    try {
        if(!this->sliceShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
    catch(vislib::Exception &e){
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }
#endif // defined(USE_TEXTURE_SLICES)

    return true;
}


/*
 * ComparativeSurfacePotentialRenderer::createVbo
 */
bool ComparativeSurfacePotentialRenderer::createVbo(GLuint* vbo, size_t s, GLuint target) {
    glGenBuffersARB(1, vbo);
    glBindBufferARB(target, *vbo);
    glBufferDataARB(target, s, 0, GL_DYNAMIC_DRAW);
    glBindBufferARB(target, 0);
    return CheckForGLError();
}

/*
 * ComparativeSurfacePotentialRenderer::destroyVbo
 */
void ComparativeSurfacePotentialRenderer::destroyVbo(GLuint* vbo, GLuint target) {
    glBindBufferARB(target, *vbo);
    glDeleteBuffersARB(1, vbo);
    *vbo = 0;
    CheckForGLError();
}


/*
 * ComparativeSurfacePotentialRenderer::fitMoleculeRMS
 */
bool ComparativeSurfacePotentialRenderer::fitMoleculeRMS(MolecularDataCall *mol0,
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

//        printf("rmsPosVec0: %f %f %f %f %f %f\n",
//                this->rmsPosVec0.Peek()[0],
//                this->rmsPosVec0.Peek()[1],
//                this->rmsPosVec0.Peek()[2],
//                this->rmsPosVec0.Peek()[3],
//                this->rmsPosVec0.Peek()[4],
//                this->rmsPosVec0.Peek()[5]);
//
//        printf("rmsPosVec1: %f %f %f %f %f %f\n",
//                this->rmsPosVec1.Peek()[0],
//                this->rmsPosVec1.Peek()[1],
//                this->rmsPosVec1.Peek()[2],
//                this->rmsPosVec1.Peek()[3],
//                this->rmsPosVec1.Peek()[4],
//                this->rmsPosVec1.Peek()[5]);

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

//    // DEBUG
//    printf("RMS posCnt %u\n", posCnt);
//    printf("RMS value %f\n", this->rmsValue);
//    printf("RMS translation %f %f %f\n", this->rmsTranslation.X(),
//            this->rmsTranslation.Y(), this->rmsTranslation.Z());
//    printf("RMS rotation\n%f %f %f\n%f %f %f\n%f %f %f\n",
//            this->rmsRotation.GetAt(0, 0), this->rmsRotation.GetAt(0, 1),
//            this->rmsRotation.GetAt(0, 2), this->rmsRotation.GetAt(1, 0),
//            this->rmsRotation.GetAt(1, 1), this->rmsRotation.GetAt(1, 2),
//            this->rmsRotation.GetAt(2, 0), this->rmsRotation.GetAt(2, 1),
//            this->rmsRotation.GetAt(2, 2));

    // Check for sufficiently low rms value
    if (this->rmsValue > this->maxRMSVal) {
        return false;
    }

    return true;
}


/*
 * ComparativeSurfacePotentialRenderer::freeBuffers
 */
void ComparativeSurfacePotentialRenderer::freeBuffers() { // TODO is all this necessary?
    CudaSafeCall(this->cubeMap0_D.Release());
    CudaSafeCall(this->cubeMapInv0_D.Release());
    CudaSafeCall(this->cubeMap1_D.Release());
    CudaSafeCall(this->cubeMapInv1_D.Release());
    CudaSafeCall(this->cubeStates_D.Release());
    CudaSafeCall(this->cubeOffsets_D.Release());
    CudaSafeCall(this->vertexStates_D.Release());
    CudaSafeCall(this->activeVertexPos_D.Release());
    CudaSafeCall(this->vertexIdxOffs_D.Release());
    CudaSafeCall(this->vertexMap0_D.Release());
    CudaSafeCall(this->vertexMap1_D.Release());
    CudaSafeCall(this->vertexMapInv0_D.Release());
    CudaSafeCall(this->vertexMapInv1_D.Release());
    CudaSafeCall(this->vertexNeighbours0_D.Release());
    CudaSafeCall(this->vertexNeighbours1_D.Release());
    CudaSafeCall(this->verticesPerTetrahedron_D.Release());
    CudaSafeCall(this->tetrahedronVertexOffsets_D.Release());
    CudaSafeCall(this->vertexExternalForcesScl_D.Release());
    CudaSafeCall(this->triangleCamDistance_D.Release());
#if defined(USE_DISTANCE_FIELD)
    CudaSafeCall(this->distField_D.Release());
#endif // defined(USE_DISTANCE_FIELD)
#if defined(USE_TEXTURE_SLICES)
    this->volume0.Release();
    this->volume1.Release();
#endif // defined(USE_TEXTURE_SLICES)
    this->gridDataPos.Release();
    this->rmsPosVec0.Release();
    this->rmsPosVec1.Release();
    this->rmsWeights.Release();
    this->rmsMask.Release();
#if (defined(USE_DISTANCE_FIELD)&&defined(USE_TEXTURE_SLICES))
    this->distField.Release();
#endif // (defined(USE_DISTANCE_FIELD)&&defined(USE_TEXTURE_SLICES))
}


/*
 * ComparativeSurfacePotentialRenderer::GetCapabilities
 */
bool ComparativeSurfacePotentialRenderer::GetCapabilities(core::Call& call) {
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);

    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(core::view::AbstractCallRender3D::CAP_RENDER |
                          core::view::AbstractCallRender3D::CAP_LIGHTING |
                          core::view::AbstractCallRender3D::CAP_ANIMATION);

    return true;
}


/*
 * ComparativeSurfacePotentialRenderer::initPotentialMap
 */
bool ComparativeSurfacePotentialRenderer::initPotentialMap(
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
 * ComparativeSurfacePotentialRenderer::GetExtents
 */
bool ComparativeSurfacePotentialRenderer::GetExtents(core::Call& call) {

    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    // Get pointer to potential map data call
    protein::VTIDataCall *cmd0 =
            this->potentialDataCallerSlot0.CallAs<protein::VTIDataCall>();
    if (cmd0 == NULL) {
        return false;
    }
    if (!(*cmd0)(VTIDataCall::CallForGetExtent)) {
        return false;
    }
    protein::VTIDataCall *cmd1 =
            this->potentialDataCallerSlot1.CallAs<protein::VTIDataCall>();
    if (cmd1 == NULL) {
        return false;
    }
    if (!(*cmd1)(VTIDataCall::CallForGetExtent)) {
        return false;
    }

    // Get a pointer to particle data call
    MolecularDataCall *mol0 = this->particleDataCallerSlot0.CallAs<MolecularDataCall>();
    if (mol0 == NULL) {
        return false;
    }
    if (!(*mol0)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }
    MolecularDataCall *mol1 = this->particleDataCallerSlot1.CallAs<MolecularDataCall>();
    if (mol1 == NULL) {
        return false;
    }
    if (!(*mol1)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }

    // Get a pointer to the outgoing render call
    view::CallRender3D *ren = this->rendererCallerSlot.CallAs<view::CallRender3D>();
    if (ren != NULL) {
        if (!(*ren)(1)) {
            return false;
        }
    }

    this->bboxParticles0 = mol0->AccessBoundingBoxes();
    this->bboxParticles1 = mol1->AccessBoundingBoxes();
//    core::BoundingBoxes bboxPotential0 = cmd0->AccessBoundingBoxes();
    core::BoundingBoxes bboxPotential1 = cmd1->AccessBoundingBoxes();
//
////    // DEBUG
////    printf("potential obj bbox #0, org %f %f %f, maxCoord %f %f %f\n",
////            bboxPotential0.ObjectSpaceBBox().GetOrigin().X(),
////            bboxPotential0.ObjectSpaceBBox().GetOrigin().Y(),
////            bboxPotential0.ObjectSpaceBBox().GetOrigin().Z(),
////            bboxPotential0.ObjectSpaceBBox().GetRightTopFront().X(),
////            bboxPotential0.ObjectSpaceBBox().GetRightTopFront().Y(),
////            bboxPotential0.ObjectSpaceBBox().GetRightTopFront().Z());
////
//    // DEBUG
//    printf("potential obj bbox #1, org %f %f %f, maxCoord %f %f %f\n",
//            bboxPotential1.ObjectSpaceBBox().GetOrigin().X(),
//                bboxPotential1.ObjectSpaceBBox().GetOrigin().Y(),
//                bboxPotential1.ObjectSpaceBBox().GetOrigin().Z(),
//                bboxPotential1.ObjectSpaceBBox().GetRightTopFront().X(),
//                bboxPotential1.ObjectSpaceBBox().GetRightTopFront().Y(),
//                bboxPotential1.ObjectSpaceBBox().GetRightTopFront().Z());
//

// particles obj bbox #0, org -0.460000 18.689999 11.950000, maxCoord 58.299999 74.049995 73.309998
// particles obj bbox #1, org -0.460000 18.689999 11.950000, maxCoord 58.299999 74.049995 73.309998

//    // DEBUG
//    printf("particles obj bbox #0, org %f %f %f, maxCoord %f %f %f\n",
//            this->bboxParticles0.ObjectSpaceBBox().GetOrigin().X(),
//            this->bboxParticles0.ObjectSpaceBBox().GetOrigin().Y(),
//            this->bboxParticles0.ObjectSpaceBBox().GetOrigin().Z(),
//            this->bboxParticles0.ObjectSpaceBBox().GetRightTopFront().X(),
//            this->bboxParticles0.ObjectSpaceBBox().GetRightTopFront().Y(),
//            this->bboxParticles0.ObjectSpaceBBox().GetRightTopFront().Z());
//
//    // DEBUG
//    printf("particles obj bbox #1, org %f %f %f, maxCoord %f %f %f\n",
//            this->bboxParticles1.ObjectSpaceBBox().GetOrigin().X(),
//            this->bboxParticles1.ObjectSpaceBBox().GetOrigin().Y(),
//            this->bboxParticles1.ObjectSpaceBBox().GetOrigin().Z(),
//            this->bboxParticles1.ObjectSpaceBBox().GetRightTopFront().X(),
//            this->bboxParticles1.ObjectSpaceBBox().GetRightTopFront().Y(),
//            this->bboxParticles1.ObjectSpaceBBox().GetRightTopFront().Z());



    core::BoundingBoxes bbox_external;
    if (ren != NULL) {
        bbox_external = ren->AccessBoundingBoxes();

//        // DEBUG Print external bbox
//        printf("external bbox , org %f %f %f, maxCoord %f %f %f\n",
//                bbox_external.ObjectSpaceBBox().GetOrigin().X(),
//                bbox_external.ObjectSpaceBBox().GetOrigin().Y(),
//                bbox_external.ObjectSpaceBBox().GetOrigin().Z(),
//                bbox_external.ObjectSpaceBBox().GetRightTopFront().X(),
//                bbox_external.ObjectSpaceBBox().GetRightTopFront().Y(),
//                bbox_external.ObjectSpaceBBox().GetRightTopFront().Z());
//        // END DEBUG
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

//    // DEBUG
//    printf("Surface renderer scale %f, org %f %f %f, maxCoord %f %f %f\n",
//            scale,
//            this->bbox.ObjectSpaceBBox().GetOrigin().X(),
//            this->bbox.ObjectSpaceBBox().GetOrigin().Y(),
//            this->bbox.ObjectSpaceBBox().GetOrigin().Z(),
//            this->bbox.ObjectSpaceBBox().GetRightTopFront().X(),
//            this->bbox.ObjectSpaceBBox().GetRightTopFront().Y(),
//            this->bbox.ObjectSpaceBBox().GetRightTopFront().Z());

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

//    printf("Master Call3d Object Space BBOX %f %f %f, %f %f %f\n",
//            cr3d->AccessBoundingBoxes().ObjectSpaceBBox().Left(),
//            cr3d->AccessBoundingBoxes().ObjectSpaceBBox().Bottom(),
//            cr3d->AccessBoundingBoxes().ObjectSpaceBBox().Back(),
//            cr3d->AccessBoundingBoxes().ObjectSpaceBBox().Right(),
//            cr3d->AccessBoundingBoxes().ObjectSpaceBBox().Top(),
//            cr3d->AccessBoundingBoxes().ObjectSpaceBBox().Front());
//
//    printf("Master Call3d World Space BBOX %f %f %f, %f %f %f\n",
//            cr3d->AccessBoundingBoxes().WorldSpaceBBox().Left(),
//            cr3d->AccessBoundingBoxes().WorldSpaceBBox().Bottom(),
//            cr3d->AccessBoundingBoxes().WorldSpaceBBox().Back(),
//            cr3d->AccessBoundingBoxes().WorldSpaceBBox().Right(),
//            cr3d->AccessBoundingBoxes().WorldSpaceBBox().Top(),
//            cr3d->AccessBoundingBoxes().WorldSpaceBBox().Front());
//
//    printf("Master Call3d Object Space clip BBOX %f %f %f, %f %f %f\n",
//            cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().Left(),
//            cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().Bottom(),
//            cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().Back(),
//            cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().Right(),
//            cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().Top(),
//            cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().Front());
//
//    printf("Master Call3d World Space clip BBOX %f %f %f, %f %f %f\n",
//            cr3d->AccessBoundingBoxes().WorldSpaceClipBox().Left(),
//            cr3d->AccessBoundingBoxes().WorldSpaceClipBox().Bottom(),
//            cr3d->AccessBoundingBoxes().WorldSpaceClipBox().Back(),
//            cr3d->AccessBoundingBoxes().WorldSpaceClipBox().Right(),
//            cr3d->AccessBoundingBoxes().WorldSpaceClipBox().Top(),
//            cr3d->AccessBoundingBoxes().WorldSpaceClipBox().Front());

    return true;
}


/*
 * ComparativeSurfacePotentialRenderer::getVBOExtent
 */
bool ComparativeSurfacePotentialRenderer::getVBOExtent(core::Call& call) {

    VBODataCall *c = dynamic_cast<VBODataCall*>(&call);

    if (c == NULL) {
        return false;
    }

    c->SetBBox(this->bbox);

    // Get pointer to potential map data call
    protein::VTIDataCall *cmd0 =
            this->potentialDataCallerSlot0.CallAs<protein::VTIDataCall>();
    if (cmd0 == NULL) {
        return false;
    }
    if (!(*cmd0)(VTIDataCall::CallForGetExtent)) {
        return false;
    }
    protein::VTIDataCall *cmd1 =
            this->potentialDataCallerSlot1.CallAs<protein::VTIDataCall>();
    if (cmd1 == NULL) {
        return false;
    }
    if (!(*cmd1)(VTIDataCall::CallForGetExtent)) {
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
    } else {
        return false; // Invalid compare mode
    }

    return true;
}


/*
 * ComparativeSurfacePotentialRenderer::getVBOData0
 */
bool ComparativeSurfacePotentialRenderer::getVBOData0(core::Call& call) {
    VBODataCall *c = dynamic_cast<VBODataCall*>(&call);

    if (c == NULL) {
        return false;
    }

    // Set vertex data
    c->SetVbo(this->vbo0);
    c->SetDataStride(this->vertexDataStride);
    c->SetDataOffs(this->vertexDataOffsPos, this->vertexDataOffsNormal,
            this->vertexDataOffsTexCoord);
    c->SetVertexCnt(this->vertexCnt0);

    // Set triangles
    c->SetVboTriangleIdx(this->vboTriangleIdx0);
    c->SetTriangleCnt(this->triangleCnt0);

    // Set potential texture
    c->SetTex(this->potentialTex0);
    c->SetTexValRange(this->minPotential, this->maxPotential);

    // Set device pointer
    c->SetCudaRessourceHandle(&this->vbo0Resource);

    return true;
}

/*
 * ComparativeSurfacePotentialRenderer::getVBOData1
 */
bool ComparativeSurfacePotentialRenderer::getVBOData1(core::Call& call) {

    VBODataCall *c = dynamic_cast<VBODataCall*>(&call);

    if (c == NULL) {
        return false;
    }

    // Set vertex data
    c->SetVbo(this->vbo1);
    c->SetDataStride(this->vertexDataStride);
    c->SetDataOffs(this->vertexDataOffsPos, this->vertexDataOffsNormal,
            this->vertexDataOffsTexCoord);
    c->SetVertexCnt(this->vertexCnt1);

    // Set triangles
    c->SetVboTriangleIdx(this->vboTriangleIdx1);
    c->SetTriangleCnt(this->triangleCnt1);

    // Set potential texture
    c->SetTex(this->potentialTex1);
    c->SetTexValRange(this->minPotential, this->maxPotential);

    // Set device pointer
    c->SetCudaRessourceHandle(&this->vbo1Resource);

    return true;
}


/*
 * ComparativeSurfacePotentialRenderer::isosurfComputeNormals
 */
bool ComparativeSurfacePotentialRenderer::isosurfComputeNormals(
        float *volume_D,
        gridParams gridDensMap,
        cudaGraphicsResource **vboResource,
        CudaDevArr<uint> &vertexMap_D,
        CudaDevArr<uint> &vertexMapInv_D,
        CudaDevArr<uint> &cubeMap_D,
        CudaDevArr<uint> &cubeMapInv_D,
        uint activeVertexCnt,
        uint arrDataOffsPos,
        uint arrDataOffsNormals,
        uint arrDataSize) {

//    // DEBUG Print vertexMapInv
//    HostArr<uint> vertexMapInv;
//    vertexMapInv.Validate(vertexMapInv_D.GetSize());
//    vertexMapInv_D.CopyToHost(vertexMapInv.Peek());
//    for (int i = 0; i < vertexMapInv_D.GetSize(); ++i) {
//        if (vertexMapInv.Peek()[i] < 1000000) {
//        printf("%i --> %u (vertex count %u)\n", i, vertexMapInv.Peek()[i],
//                activeVertexCnt);
//        }
//    }
//    // END DEBUG

    CheckForCudaErrorSync(); // Error check with device sync

//    printf("Computing normals of %u vertices\n", activeVertexCnt);

    if (!CudaSafeCall(InitVolume(
            make_uint3(gridDensMap.size[0], gridDensMap.size[1], gridDensMap.size[2]),
            make_float3(gridDensMap.minC[0], gridDensMap.minC[1], gridDensMap.minC[2]),
            make_float3(gridDensMap.delta[0], gridDensMap.delta[1], gridDensMap.delta[2])))) {
        return false;
    }

    CheckForCudaErrorSync(); // Error check with device sync

    if (!CudaSafeCall(InitVolume_surface_generation(
            make_uint3(gridDensMap.size[0], gridDensMap.size[1], gridDensMap.size[2]),
            make_float3(gridDensMap.minC[0], gridDensMap.minC[1], gridDensMap.minC[2]),
            make_float3(gridDensMap.delta[0], gridDensMap.delta[1], gridDensMap.delta[2])))) {
        return false;
    }

//    printf("Init volume surface generation\n");
//    printf("grid size  %u %u %u\n", gridDensMap.size[0], gridDensMap.size[1], gridDensMap.size[2]);
//    printf("grid org   %f %f %f\n", gridDensMap.minC[0], gridDensMap.minC[1], gridDensMap.minC[2]);
//    printf("grid delta %f %f %f\n", gridDensMap.delta[0], gridDensMap.delta[1], gridDensMap.delta[2]);

    CheckForCudaErrorSync(); // Error check with device sync

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    CudaSafeCall(cudaGraphicsMapResources(1, vboResource, 0));
    CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,             // The size of the accessible data
            *vboResource));                   // The mapped resource

//    // DEBUG Print vertex data
//    HostArr<float> vertexBuff;
//    vertexBuff.Validate(activeVertexCnt*arrDataSize);
//    cudaMemcpy(vertexBuff.Peek(), vboPt, activeVertexCnt*arrDataSize*sizeof(float),
//            cudaMemcpyDeviceToHost);
//    printf("Before:\n");
//    for (int i = 0; i < 10; ++i) {
//        printf("%i: pos %f %f %f, normal %f %f %f\n",i,
//                vertexBuff.Peek()[i*arrDataSize+arrDataOffsPos],
//                vertexBuff.Peek()[i*arrDataSize+arrDataOffsPos+1],
//                vertexBuff.Peek()[i*arrDataSize+arrDataOffsPos+2],
//                vertexBuff.Peek()[i*arrDataSize+arrDataOffsNormals],
//                vertexBuff.Peek()[i*arrDataSize+arrDataOffsNormals+1],
//                vertexBuff.Peek()[i*arrDataSize+arrDataOffsNormals+2]);
//    }
//    // END DEBUG

    if (!CudaSafeCall(ComputeVertexNormals(
            vboPt,
            vertexMap_D.Peek(),
            vertexMapInv_D.Peek(),
            cubeMap_D.Peek(),
            cubeMapInv_D.Peek(),
            volume_D,
            this->qsIsoVal,
            activeVertexCnt,
            arrDataOffsPos,
            arrDataOffsNormals,
            arrDataSize))) {

        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, vboResource))) {
            return false;
        }

        return false;
    }

//    // DEBUG Print vertex data
//    HostArr<float> vertexBuff;
//    vertexBuff.Validate(activeVertexCnt*arrDataSize);
//    cudaMemcpy(vertexBuff.Peek(), vboPt, activeVertexCnt*arrDataSize*sizeof(float),
//            cudaMemcpyDeviceToHost);
//    for (uint i = 0; i < activeVertexCnt; ++i) {
//    //for (int i = 0; i < 10; ++i) {
//        if (vertexBuff.Peek()[i*arrDataSize+arrDataOffsNormals] > 7)
//        printf("%i: pos %f %f %f, normal %f %f %f\n",i,
//                vertexBuff.Peek()[i*arrDataSize+arrDataOffsPos],
//                vertexBuff.Peek()[i*arrDataSize+arrDataOffsPos+1],
//                vertexBuff.Peek()[i*arrDataSize+arrDataOffsPos+2],
//                vertexBuff.Peek()[i*arrDataSize+arrDataOffsNormals],
//                vertexBuff.Peek()[i*arrDataSize+arrDataOffsNormals+1],
//                vertexBuff.Peek()[i*arrDataSize+arrDataOffsNormals+2]);
//    }
//    // END DEBUG

    CheckForCudaErrorSync(); // Error check with device sync

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, vboResource))) {
        return false;
    }

    return CheckForCudaErrorSync(); // Error check with device sync
}


/*
 * ComparativeSurfacePotentialRenderer::isosurfComputeTexCoords
 */
bool ComparativeSurfacePotentialRenderer::isosurfComputeTexCoords(
        cudaGraphicsResource **vboResource,
        uint activeVertexCnt,
        float3 minC,
        float3 maxC,
        uint arrDataOffsPos,
        uint arrDataOffsTexCoords,
        uint arrDataSize) {

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    CudaSafeCall(cudaGraphicsMapResources(1, vboResource, 0));
    CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,             // The size of the accessible data
            *vboResource));                   // The mapped resource

    if (!CudaSafeCall(ComputeVertexTexCoords(
            vboPt,
            minC.x,
            minC.y,
            minC.z,
            maxC.x,
            maxC.y,
            maxC.z,
            activeVertexCnt,
            arrDataOffsPos,
            arrDataOffsTexCoords,
            arrDataSize))) {
        return false;
    }

//    // DEBUG Print vertex positions and respective texture coordinates
//    HostArr<float> buff;
//    buff.Validate(activeVertexCnt*arrDataSize);
//    CudaSafeCall(cudaMemcpy(buff.Peek(), vboPt,
//            activeVertexCnt*arrDataSize*sizeof(float), cudaMemcpyDeviceToHost));
//    for (int i = 0; i < 25; ++i) {
//        printf("%i: pos(%f %f %f), tex coords (%f %f %f), potential old %f , potential new %f\n", i,
//                buff.Peek()[i*arrDataSize + arrDataOffsPos+0],
//                buff.Peek()[i*arrDataSize + arrDataOffsPos+1],
//                buff.Peek()[i*arrDataSize + arrDataOffsPos+2],
//                buff.Peek()[i*arrDataSize + arrDataOffsTexCoords+0],
//                buff.Peek()[i*arrDataSize + arrDataOffsTexCoords+1],
//                buff.Peek()[i*arrDataSize + arrDataOffsTexCoords+2],
//                );
//    }
//    // END DEBUG

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, vboResource))) {
        return false;
    }

    return CheckForCudaErrorSync(); // Error check with device sync
}


/*
 * ComparativeSurfacePotentialRenderer::isosurfComputeVertices
 */
bool ComparativeSurfacePotentialRenderer::isosurfComputeVertices(
        float *volume_D,
        CudaDevArr<uint> &cubeMap_D,
        CudaDevArr<uint> &cubeMapInv_D,
        CudaDevArr<uint> &vertexMap_D,
        CudaDevArr<uint> &vertexMapInv_D,
        CudaDevArr<int> &vertexNeighbours_D,
        gridParams gridDensMap,
        uint &activeVertexCount,
        GLuint &vbo,
        cudaGraphicsResource **vboResource,
        uint &triangleCount,
        GLuint &vboTriangleIdx,
        cudaGraphicsResource **vboTriangleIdxResource) {

    using vislib::sys::Log;

    uint gridCellCnt = (gridDensMap.size[0]-1)*(gridDensMap.size[1]-1)*(gridDensMap.size[2]-1);
    uint activeCellCnt, triangleVtxCnt;

//    // DEBUG Print volume
//    HostArr<float> volume;
//    volume.Validate(gridDensMap.size[0]*gridDensMap.size[1]*gridDensMap.size[2]);
//    cudaMemcpy(volume.Peek(), volume_D,
//            sizeof(float)*gridDensMap.size[0]*gridDensMap.size[1]*gridDensMap.size[2],
//            cudaMemcpyDeviceToHost);
//    for (int i = 50000; i < 50010; ++i) {
//        printf("vol %f\n", volume.Peek()[i]);
//    }
//    // END DEBUG


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

//    printf("Grid dims %u %u %u\n", gridDensMap.size[0],gridDensMap.size[1], gridDensMap.size[2]);


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


//    printf("Active cell count %u\n", activeCellCnt); // DEBUG
//    printf("Reduction %f\n", 1.0 - static_cast<float>(activeCellCnt)/
//            static_cast<float>(gridCellCnt)); // DEBUG


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

//    printf("Vertex count %u\n", activeVertexCount);


    /* Create vertex buffer object and register with CUDA */

    // Create empty vbo to hold data for the surface
    if (!this->createVbo(&vbo,
            activeVertexCount*this->vertexDataStride*sizeof(float),
            GL_ARRAY_BUFFER)) {
        return false;
    }

//    printf("Create VBO of size %u\n", activeVertexCount*this->vertexDataStride*sizeof(float));
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(vboResource, vbo,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, vboResource, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,             // The size of the accessible data
            *vboResource))) {                   // The mapped resource
        return false;
    }

    // Init with zeros
    if (!CudaSafeCall(cudaMemset(vboPt, 0, vboSize))) {
        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, vboResource))) {
            return false;
        }
        return false;
    }

//    printf("Got VBO of size %u\n", vboSize);


    /* Compact list of vertex positions (keep only active vertices) */

    if (!CudaSafeCall(CompactActiveVertexPositions(
            vboPt,
            this->vertexStates_D.Peek(),
            this->vertexIdxOffs_D.Peek(),
            this->activeVertexPos_D.Peek(),
            activeCellCnt,
            this->vertexDataOffsPos,  // Array data byte offset
            this->vertexDataStride    // Array data element size
            ))) {
        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, vboResource))) {
            return false;
        }
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

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, vboResource))) {
        return false;
    }


    /* Calc vertex index map */

    if (!CudaSafeCall(vertexMap_D.Validate(activeVertexCount))) {
        return false;
    }
    if (!CudaSafeCall(vertexMapInv_D.Validate(7*activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(vertexMapInv_D.Set(0xff))) {
        return false;
    }
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


    /* Create vertex buffer object and register with CUDA */

    // Create empty vbo to hold the triangle indices
    if (!this->createVbo(&vboTriangleIdx,
            triangleVtxCnt*sizeof(uint),
            GL_ELEMENT_ARRAY_BUFFER)) {
        return false;
    }
    CheckForGLError();
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(vboTriangleIdxResource,
            vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    uint *vboTriangleIdxPt;
    size_t vboTriangleIdxSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, vboTriangleIdxResource, 0))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboTriangleIdxPt), // The mapped pointer
            &vboTriangleIdxSize,             // The size of the accessible data
            *vboTriangleIdxResource))) {                   // The mapped resource
        return false;
    }


    /* Generate triangles */

    if (!CudaSafeCall(cudaMemset(vboTriangleIdxPt, 0x00, vboTriangleIdxSize))) {
        return false;
    }

    if (!CudaSafeCall(GetTrianglesIdx(
            this->tetrahedronVertexOffsets_D.Peek(),
            cubeMap_D.Peek(),
            cubeMapInv_D.Peek(),
            this->qsIsoVal,
            activeCellCnt*6,
            activeCellCnt,
            vboTriangleIdxPt,
            vertexMapInv_D.Peek(),
            volume_D))) {
        return false;
    }

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, vboTriangleIdxResource))) {
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
//    vertexPos.Validate(activeVertexCount*this->vertexDataStride);
//    cudaMemcpy(vertexPos.Peek(), vboPt,
//            sizeof(float)*activeVertexCount*this->vertexDataStride,
//            cudaMemcpyDeviceToHost);
//    printf("Initial positions:\n");
//    for (int i = 0; i < 10; ++i) {
//        printf("%i: Vertex position (%f %f %f)\n", i,
//                vertexPos.Peek()[this->vertexDataStride*i+0],
//                vertexPos.Peek()[this->vertexDataStride*i+1],
//                vertexPos.Peek()[this->vertexDataStride*i+2]);
//    }
//    // End DEBUG

    return CheckForCudaErrorSync(); // Error check with sync
}


/*
 * ComparativeSurfacePotentialRenderer::regularizeSurface
 */
bool ComparativeSurfacePotentialRenderer::regularizeSurface(
        float *volume_D,
        gridParams gridDensMap,
        cudaGraphicsResource **vboCudaRes,
        uint vertexCnt,
        CudaDevArr<int> &vertexNeighbours_D,
        uint maxIt,
        float springStiffness,
        float forceScl,
        float externalForcesWeight,
        InterpolationMode interpMode) {

    using vislib::sys::Log;

    float *vboPt;
    size_t vboSize;

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

    if (!CudaSafeCall(cudaGraphicsMapResources(1, vboCudaRes, 0))) {
        return false;
    }

    // Get mapped pointer to the vbo
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,             // The size of the accessible data
            *vboCudaRes))) {                   // The mapped resource
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
            vboPt,
            static_cast<uint>(this->vertexExternalForcesScl_D.GetCount()),
            this->qsIsoVal,
            this->vertexDataOffsPos,
            this->vertexDataStride))) {
        return false;
    }

#ifdef POTENTIAL_VOLUME_RENDERER_CUDA_USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

//    // DEBUG Print mapped positions
//    HostArr<float> vertexPos;
//    vertexPos.Validate(vertexCnt*this->vertexDataStride);
//    cudaMemcpy(vertexPos.Peek(), vboPt,
//            sizeof(float)*vertexCnt*this->vertexDataStride,
//            cudaMemcpyDeviceToHost);
//    printf("Positions before:\n");
//    for (int i = 0; i < 10; ++i) {
//        printf("%i: Vertex position (%f %f %f)\n", i,
//                vertexPos.Peek()[this->vertexDataStride*i+0],
//                vertexPos.Peek()[this->vertexDataStride*i+1],
//                vertexPos.Peek()[this->vertexDataStride*i+2]);
//    }
//    // End DEBUG

    // Compute gradient
    if (!CudaSafeCall(this->volGradient_D.Validate(gridDensMap.size[0]*gridDensMap.size[1]*gridDensMap.size[2]))) {
        return false;
    }
    if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(), volume_D,
            this->volGradient_D.GetSize()))) {
        return false;
    }

    // Allocate memory for laplacian
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
                    vboPt,
                    this->vertexExternalForcesScl_D.Peek(),
                    vertexNeighbours_D.Peek(),
                    this->volGradient_D.Peek(),
                    this->laplacian_D.Peek(),
                    vertexCnt,
                    externalForcesWeight,
                    forceScl,
                    springStiffness,
                    this->qsIsoVal,
                    this->surfMappedMinDisplScl,
                    this->vertexDataOffsPos,
                    this->vertexDataStride))) {
                return false;
            }

        }
    } else {
        for (uint i = 0; i < maxIt; i += UPDATE_VTX_POS_ITERATIONS_PER_KERNEL) {

//            // Update position for all vertices using cubic interpolation
//            if (!CudaSafeCall(UpdateVertexPositionTricubic(
//                    volume_D,
//                    vboPt,
//                    this->vertexExternalForcesScl_D.Peek(),
//                    vertexNeighbours_D.Peek(),
//                    this->volGradient_D.Peek(),
//                    this->laplacian_D.Peek(),
//                    vertexCnt,
//                    externalForcesWeight,
//                    forceScl,
//                    springStiffness,
//                    this->qsIsoVal,
//                    this->surfMappedMinDisplScl,
//                    this->vertexDataOffsPos,
//                    this->vertexDataStride))) {
//                return false;
//            }
        }
    }
    printf("Parameters: vertex count %u, externalForcesWeight %f, forceScl %f, springStiffness %f, this->qsIsoVal %f minDispl %f\n",
            vertexCnt, externalForcesWeight, forceScl, springStiffness,
            this->qsIsoVal, this->surfMappedMinDisplScl);

//    // DEBUG Print mapped positions
//    HostArr<float> vertexPos;
//    vertexPos.Validate(vertexCnt*this->vertexDataStride);
//    cudaMemcpy(vertexPos.Peek(), vboPt,
//            sizeof(float)*vertexCnt*this->vertexDataStride,
//            cudaMemcpyDeviceToHost);
//    printf("Positions after:\n");
//    for (int i = 0; i < 100; ++i) {
//        printf("%i: Vertex position (%f %f %f)\n", i,
//                vertexPos.Peek()[this->vertexDataStride*i+0],
//                vertexPos.Peek()[this->vertexDataStride*i+1],
//                vertexPos.Peek()[this->vertexDataStride*i+2]);
//    }
//    // END DEBUG

#ifdef POTENTIAL_VOLUME_RENDERER_CUDA_USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "%s: Time for regularization (%u iterations): %f sec\n",
            this->ClassName(), maxIt, dt_ms/1000.0f);
#endif

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, vboCudaRes))) {
        return false;
    }

    return CheckForCudaErrorSync(); // Error check with sync
}


/*
 * ComparativeSurfacePotentialRenderer::mapIsosurfaceToVolume
 */
bool ComparativeSurfacePotentialRenderer::mapIsosurfaceToVolume(
        float *volume_D,
        gridParams gridDensMap,
        cudaGraphicsResource **vboCudaRes,
        cudaGraphicsResource **vboCudaResTriangleIdx,
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
            this->surfMappedMaxDist,
            this->qsIsoVal,
            this->volGradient_D.GetSize()))) {
        return false;
    }

#else
    if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(), volume_D,
            this->volGradient_D.GetSize()))) {
        return false;
    }
#endif


//    // DEBUG Print gradient field
//    HostArr<float4> gradFieldTest;
//    gradFieldTest.Validate(this->gradientDensMap0_D.GetSize());
//    if (!CudaSafeCall(this->gradientDensMap0_D.CopyToHost(gradFieldTest.Peek()))) {
//        return false;
//    }
//    for (int i = 0; i < 30000; ++i) {
//        printf("%i: Gradient: %f %f %f\n", i,
//                gradFieldTest.Peek()[i].x,
//                gradFieldTest.Peek()[i].y,
//                gradFieldTest.Peek()[i].z);
//    }
//    // END DEBUG

    // We need both cuda graphics resources to be mapped at the same time
    cudaGraphicsResource *cudaToken[2];
    cudaToken[0] = *vboCudaRes;
    cudaToken[1] = *vboCudaResTriangleIdx;
    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaToken, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data and the triangle indices
    float *vboPt;
    uint *vboTriangleIdxPt;
    size_t vboSize, vboTriangleIdxSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,                         // The size of the accessible data
            cudaToken[0]))) {                 // The mapped resource
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboTriangleIdxPt), // The mapped pointer
            &vboTriangleIdxSize,              // The size of the accessible data
            cudaToken[1]))) {                 // The mapped resource
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
            vboPt,
            static_cast<uint>(this->vertexExternalForcesScl_D.GetCount()),
            this->qsIsoVal,
            this->vertexDataMappedOffsPosNew,
            this->vertexDataMappedStride))) {
        return false;
    }

    if (!CudaSafeCall(this->laplacian_D.Validate(vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->laplacian_D.Set(0))) {
        return false;
    }

    if (!CudaSafeCall(this->displLen_D.Validate(vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Set(0xff))) {
        return false;
    }

#ifdef POTENTIAL_VOLUME_RENDERER_CUDA_USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

//    // DEBUG Print mapped positions
//    printf("Mapped positions before\n");
//    HostArr<float> vertexPos;
//    vertexPos.Validate(vertexCnt*this->vertexDataMappedStride);
//    cudaMemcpy(vertexPos.Peek(), vboPt, sizeof(float)*vertexCnt*this->vertexDataMappedStride, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < 10; ++i) {
//        printf("%i: Vertex position (%f %f %f)\n", i,
//                vertexPos.Peek()[this->vertexDataMappedStride*i+this->vertexDataMappedOffsPosNew+0],
//                vertexPos.Peek()[this->vertexDataMappedStride*i+this->vertexDataMappedOffsPosNew+1],
//                vertexPos.Peek()[this->vertexDataMappedStride*i+this->vertexDataMappedOffsPosNew+2]);
//
//    }
//    // End DEBUG

    if (interpMode == INTERP_LINEAR) {
        for (uint i = 0; i < maxIt; i += UPDATE_VTX_POS_ITERATIONS_PER_KERNEL) {

            // Update position for all vertices
            // Use no distance field
            if (!CudaSafeCall(UpdateVertexPositionTrilinear(
                    volume_D,
                    vboPt,
                    this->vertexExternalForcesScl_D.Peek(),
                    vertexNeighbours_D.Peek(),
                    this->volGradient_D.Peek(),
                    this->laplacian_D.Peek(),
                    vertexCnt,
                    externalForcesWeight,
                    forceScl,
                    springStiffness,
                    this->qsIsoVal,
                    this->surfMappedMinDisplScl,
                    this->vertexDataMappedOffsPosNew,
                    this->vertexDataMappedStride))) {
                return false;
            }
        }
    } else {
        for (uint i = 0; i < maxIt; i += UPDATE_VTX_POS_ITERATIONS_PER_KERNEL) {

//            // Update position for all vertices
//            if (!CudaSafeCall(UpdateVertexPositionTricubic(
//                    volume_D,
//                    vboPt,
//                    this->vertexExternalForcesScl_D.Peek(),
//                    vertexNeighbours_D.Peek(),
//                    this->volGradient_D.Peek(),
//                    this->laplacian_D.Peek(),
//                    vertexCnt,
//                    externalForcesWeight,
//                    forceScl,
//                    springStiffness,
//                    this->qsIsoVal,
//                    this->surfMappedMinDisplScl,
//                    this->vertexDataMappedOffsPosNew,
//                    this->vertexDataMappedStride))) {
//                return false;
//            }

//            // Update position for all vertices
//            if (!CudaSafeCall(UpdateVertexPositionTricubicWithDispl(
//                    volume_D,
//                    vboPt,
//                    this->vertexExternalForcesScl_D.Peek(),
//                    vertexNeighbours_D.Peek(),
//                    this->volGradient_D.Peek(),
//                    this->laplacian_D.Peek(),
//                    this->displLen_D.Peek(),
//                    vertexCnt,
//                    externalForcesWeight,
//                    forceScl,
//                    springStiffness,
//                    this->qsIsoVal,
//                    this->surfMappedMinDisplScl,
//                    this->vertexDataMappedOffsPosNew,
//                    this->vertexDataMappedStride))) {
//                return false;
//            }
        }
    }

//    // DEBUG Print mapped positions
//    printf("Mapped positions after\n");
//    //HostArr<float> vertexPos;
//    vertexPos.Validate(vertexCnt*this->vertexDataMappedStride);
//    cudaMemcpy(vertexPos.Peek(), vboPt, sizeof(float)*vertexCnt*this->vertexDataMappedStride, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < 10; ++i) {
//        printf("%i: Vertex position (%f %f %f)\n", i,
//                vertexPos.Peek()[this->vertexDataMappedStride*i+this->vertexDataMappedOffsPosNew+0],
//                vertexPos.Peek()[this->vertexDataMappedStride*i+this->vertexDataMappedOffsPosNew+1],
//                vertexPos.Peek()[this->vertexDataMappedStride*i+this->vertexDataMappedOffsPosNew+2]);
//
//    }
//    // End DEBUG

#ifdef POTENTIAL_VOLUME_RENDERER_CUDA_USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "%s: Time for mapping (%u iterations, %u vertices): %f sec\n",
            this->ClassName(),
            maxIt, vertexCnt, dt_ms/1000.0f);
#endif

    // Flag vertices adjacent to corrupt triangles
    if (!CudaSafeCall(FlagVerticesInCorruptTriangles(
            vboPt,
            this->vertexDataMappedStride,
            this->vertexDataMappedOffsPosNew,
            this->vertexDataMappedOffsCorruptTriangleFlag,
            vboTriangleIdxPt,
            volume_D,
            this->vertexExternalForcesScl_D.Peek(),
            triangleCnt,
            this->surfMappedMinDisplScl,
            this->qsIsoVal))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaToken, 0))) {
        return false;
    }

    return CheckForCudaErrorSync(); // Error check with sync
}


/*
 * ComparativeSurfacePotentialRenderer::release
 */
void ComparativeSurfacePotentialRenderer::release(void) {

    CheckForGLError();

    if (glIsTexture(this->potentialTex0)) {
        glDeleteTextures(1, &this->potentialTex0);
    }
    if (glIsTexture(this->potentialTex1)) {
        glDeleteTextures(1, &this->potentialTex1);
    }

#if defined(USE_TEXTURE_SLICES)
    if (glIsTexture(this->volumeTex0)) {
        glDeleteTextures(1, &this->volumeTex0);
    }
    if (glIsTexture(this->volumeTex1)) {
        glDeleteTextures(1, &this->volumeTex1);
    }
#endif // defined(USE_TEXTURE_SLICES)

#if (defined(USE_DISTANCE_FIELD)&&defined(USE_TEXTURE_SLICES))
    if (glIsTexture(this->distFieldTex)) {
        glDeleteTextures(1, &this->distFieldTex);
    }
#endif // (defined(USE_DISTANCE_FIELD)&&defined(USE_TEXTURE_SLICES))

    CheckForGLError();
#if defined(USE_TEXTURE_SLICES)
    this->sliceShader.Release();
#endif // defined(USE_TEXTURE_SLICES)
    this->pplSurfaceShader.Release();
    this->pplMappedSurfaceShader.Release();

    this->freeBuffers();

    CheckForGLError();

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

    if (this->vbo0Resource != NULL) {
        CudaSafeCall(cudaGraphicsUnregisterResource(this->vbo0Resource));
    }
    this->destroyVbo(&this->vbo0, GL_ARRAY_BUFFER);

    if (this->vbo1Resource != NULL) {
        CudaSafeCall(cudaGraphicsUnregisterResource(this->vbo1Resource));
    }
    this->destroyVbo(&this->vbo1, GL_ARRAY_BUFFER);

    if (this->vboMappedResource != NULL) {
        CudaSafeCall(cudaGraphicsUnregisterResource(this->vboMappedResource));
    }
    this->destroyVbo(&this->vboMapped, GL_ARRAY_BUFFER);

    if (this->vboTriangleIdx0Resource != NULL) {
        CudaSafeCall(cudaGraphicsUnregisterResource(this->vboTriangleIdx0Resource));
    }
    this->destroyVbo(&this->vboTriangleIdx0, GL_ELEMENT_ARRAY_BUFFER);

    if (this->vboTriangleIdx1Resource != NULL) {
        CudaSafeCall(cudaGraphicsUnregisterResource(this->vboTriangleIdx1Resource));
    }
    this->destroyVbo(&this->vboTriangleIdx1, GL_ELEMENT_ARRAY_BUFFER);

    if (this->vboTriangleIdxMappedResource != NULL) {
        CudaSafeCall(cudaGraphicsUnregisterResource(this->vboTriangleIdxMappedResource));
    }
    this->destroyVbo(&this->vboTriangleIdxMapped, GL_ELEMENT_ARRAY_BUFFER);

    CudaSafeCall(cudaDeviceReset());
}


/*
 * ComparativeSurfacePotentialRenderer::Render
 */
bool ComparativeSurfacePotentialRenderer::Render(core::Call& call) {
    using namespace vislib::sys;
    using namespace vislib::math;

#ifdef POTENTIAL_VOLUME_RENDERER_CUDA_USE_TIMER
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
    int frameIdx0, frameIdx1;

    // Determine frame indices to be loaded based on 'compareFrames' parameter
    if (this->cmpMode == COMPARE_1_1) {
        // One by one frame comparison
        frameIdx0 = this->singleFrame0;
        frameIdx1 = this->singleFrame1;
    } else if (this->cmpMode == COMPARE_1_N) {
        // One frame of data set #0 is compared to all frames of data set #1
        frameIdx0 = this->singleFrame0;
        frameIdx1 = static_cast<int>(calltime);
    } else if (this->cmpMode == COMPARE_N_1) {
        // One frame of data set #1 is compared to all frames of data set #0
        frameIdx0 = static_cast<int>(calltime);
        frameIdx1 = this->singleFrame1;
    } else if (this->cmpMode == COMPARE_N_N) {
        // Frame by frame comparison
        // Note: The data set with more frames is truncated
        frameIdx0 = static_cast<int>(calltime);
        frameIdx1 = static_cast<int>(calltime);
    } else {
        return false; // Invalid compare mode
    }

#ifdef POTENTIAL_VOLUME_RENDERER_CUDA_USE_TIMER
    t = clock();
#endif

    // Get potential map of data set #0
    VTIDataCall *cmd0 =
            this->potentialDataCallerSlot0.CallAs<protein::VTIDataCall>();
    if (cmd0 == NULL) {
        return false;
    }
    cmd0->SetCalltime(static_cast<float>(frameIdx0));  // Set call time
    cmd0->SetFrameID(frameIdx0, true);  // Set frame ID and call data
    if (!(*cmd0)(VTIDataCall::CallForGetData)) {
        return false;
    }

    protein::VTIDataCall *cmd1 =
            this->potentialDataCallerSlot1.CallAs<protein::VTIDataCall>();
    if (cmd1 == NULL) {
        return false;
    }
    cmd1->SetCalltime(static_cast<float>(frameIdx1));  // Set call time
    cmd1->SetFrameID(frameIdx1, true);  // Set frame ID and call data
    if (!(*cmd1)(VTIDataCall::CallForGetData)) {
        return false;
    }

    // Get the particle data calls
    MolecularDataCall *mol0 = this->particleDataCallerSlot0.CallAs<MolecularDataCall>();
    if (mol0 == NULL) {
        return false;
    }
    mol0->SetCalltime(static_cast<float>(frameIdx0));  // Set call time
    mol0->SetFrameID(frameIdx0, true);  // Set frame ID and call data
    if (!(*mol0)(MolecularDataCall::CallForGetData)) {
        return false;
    }

    MolecularDataCall *mol1 = this->particleDataCallerSlot1.CallAs<MolecularDataCall>();
    if (mol1 == NULL) {
        return false;
    }
    mol1->SetCalltime(static_cast<float>(frameIdx1));  // Set call time
    mol1->SetFrameID(frameIdx1, true);  // Set frame ID and call data
    if (!(*mol1)(MolecularDataCall::CallForGetData)) {
        return false;
    }

    // Init combined bounding boxes
    core::BoundingBoxes bboxPotential0 = cmd0->AccessBoundingBoxes();
    core::BoundingBoxes bboxPotential1 = cmd1->AccessBoundingBoxes();

//    // DEBUG
//    printf("potential obj bbox #0, org %f %f %f, maxCoord %f %f %f\n",
//            bboxPotential0.ObjectSpaceBBox().GetOrigin().X(),
//            bboxPotential0.ObjectSpaceBBox().GetOrigin().Y(),
//            bboxPotential0.ObjectSpaceBBox().GetOrigin().Z(),
//            bboxPotential0.ObjectSpaceBBox().GetRightTopFront().X(),
//            bboxPotential0.ObjectSpaceBBox().GetRightTopFront().Y(),
//            bboxPotential0.ObjectSpaceBBox().GetRightTopFront().Z());
//
//    // DEBUG
//    printf("potential obj bbox #1, org %f %f %f, maxCoord %f %f %f\n",
//            bboxPotential1.ObjectSpaceBBox().GetOrigin().X(),
//                bboxPotential1.ObjectSpaceBBox().GetOrigin().Y(),
//                bboxPotential1.ObjectSpaceBBox().GetOrigin().Z(),
//                bboxPotential1.ObjectSpaceBBox().GetRightTopFront().X(),
//                bboxPotential1.ObjectSpaceBBox().GetRightTopFront().Y(),
//                bboxPotential1.ObjectSpaceBBox().GetRightTopFront().Z());


    // Do RMS fitting if necessary
    if ((this->toggleRMSFit)
                || (mol0->DataHash() != this->datahashParticles0)
                || (mol1->DataHash() != this->datahashParticles1)
                || (calltime != this->calltimeOld)) {

        if (!this->fitMoleculeRMS(mol0, mol1)) {
            return false;
        }
        this->toggleRMSFit = false;

#ifdef POTENTIAL_VOLUME_RENDERER_CUDA_USE_TIMER
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "%s: Time for RMS fitting: %.6f s",
            this->ClassName(),
            (double(clock()-t)/double(CLOCKS_PER_SEC)));
#endif
    }

    // (Re-)compute volume texture if necessary
    if ((this->triggerComputeVolume)
            ||(mol0->DataHash() != this->datahashParticles0)
            ||(mol1->DataHash() != this->datahashParticles1)
            ||(calltime != this->calltimeOld)) {

        this->datahashParticles0 = mol0->DataHash();
        this->datahashParticles1 = mol1->DataHash();

        if (!this->computeDensityMap(mol0, (CUDAQuickSurf *)this->cudaqsurf0,
                this->gridDensMap0,
                this->bboxParticles0.ObjectSpaceBBox()
#if defined(USE_TEXTURE_SLICES)
                ,this->volume0, this->volumeTex0
#endif // defined(USE_TEXTURE_SLICES)
                )) {
            return false;
        }

        if (!this->computeDensityMap(mol1, (CUDAQuickSurf *)this->cudaqsurf1,
                this->gridDensMap1,
                this->bboxParticles1.ObjectSpaceBBox()
#if defined(USE_TEXTURE_SLICES)
                ,this->volume1, this->volumeTex1
#endif // defined(USE_TEXTURE_SLICES)
                )) {
            return false;
        }

        this->triggerComputeVolume = false;
        this->triggerComputeSurfacePoints0 = true;
        this->triggerComputeSurfacePoints1 = true;
#if defined(USE_DISTANCE_FIELD)
        this->triggerComputeDistanceField = true;
#endif // defined(USE_DISTANCE_FIELD)
    }

    // (Re-)compute potential texture if necessary
    if ((this->triggerInitPotentialTex)
            ||(cmd0->DataHash() != this->datahashPotential0)
            ||(cmd1->DataHash() != this->datahashPotential1)
            ||(calltime != this->calltimeOld)) {

        this->datahashPotential0 = cmd0->DataHash();
        this->datahashPotential1 = cmd1->DataHash();
        if (!this->initPotentialMap(cmd0, this->gridPotential0, this->potentialTex0)) {
            return false;
        }
        if (!this->initPotentialMap(cmd1, this->gridPotential1, this->potentialTex1)) {
            return false;
        }
        this->triggerInitPotentialTex = false;

//        printf("Reinititialized potential grid 0\n");
//        printf("Min coord %f %f %f\n", this->gridPotential0.minC[0],
//                this->gridPotential0.minC[1], this->gridPotential0.minC[2]);
//        printf("Max coord %f %f %f\n", this->gridPotential0.maxC[0],
//                this->gridPotential0.maxC[1], this->gridPotential0.maxC[2]);
//
//        printf("Reinititialized potential grid 1\n");
//        printf("Min coord %f %f %f\n", this->gridPotential1.minC[0],
//                this->gridPotential1.minC[1], this->gridPotential1.minC[2]);
//        printf("Max coord %f %f %f\n", this->gridPotential1.maxC[0],
//                this->gridPotential1.maxC[1], this->gridPotential1.maxC[2]);
    }

    // (Re)compute vertices by Marching tetrahedra
    if (this->triggerComputeSurfacePoints0) {

        /* Surface #0 */

        // Compute initial triangulation for surface #0
        if (!this->isosurfComputeVertices(
                ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                this->cubeMap0_D,
                this->cubeMapInv0_D,
                this->vertexMap0_D,
                this->vertexMapInv0_D,
                this->vertexNeighbours0_D,
                this->gridDensMap0,
                this->vertexCnt0,
                this->vbo0,
                &this->vbo0Resource,
                this->triangleCnt0,
                this->vboTriangleIdx0,
                &this->vboTriangleIdx0Resource)) {
            return false;
        }

        // Make mesh #0 more regular by using deformable model approach
        if (!this->regularizeSurface(
                ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                this->gridDensMap0,
                &this->vbo0Resource,
                this->vertexCnt0,
                this->vertexNeighbours0_D,
                this->regMaxIt,
                this->regSpringStiffness,
                this->regForcesScl,
                this->regExternalForcesWeight,
                this->interpolMode)) {
            return false;
        }

        // Compute normals for surface #0
        if (!this->isosurfComputeNormals(
                ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                this->gridDensMap0,
                &this->vbo0Resource,
                this->vertexMap0_D,
                this->vertexMapInv0_D,
                this->cubeMap0_D,
                this->cubeMapInv0_D,
                this->vertexCnt0,
                this->vertexDataOffsPos,
                this->vertexDataOffsNormal,
                this->vertexDataStride
                )) {
            return false;
        }

        // Compute texture coordinates for surface #0
        if (!this->isosurfComputeTexCoords(
                &this->vbo0Resource,
                this->vertexCnt0,
                make_float3(this->gridPotential0.minC[0],
                        this->gridPotential0.minC[1],
                        this->gridPotential0.minC[2]),
                make_float3(this->gridPotential0.maxC[0],
                        this->gridPotential0.maxC[1],
                        this->gridPotential0.maxC[2]),
                this->vertexDataOffsPos,
                this->vertexDataOffsTexCoord,
                this->vertexDataStride)) {
            return false;
        }

        this->triggerComputeSurfacePoints0 = false;
    }

    if (this->triggerComputeSurfacePoints1) {

        /* Surface #1 */

        // Compute initial triangulation for surface #1
        if (!this->isosurfComputeVertices(
                ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
                this->cubeMap1_D,
                this->cubeMapInv1_D,
                this->vertexMap1_D,
                this->vertexMapInv1_D,
                this->vertexNeighbours1_D,
                this->gridDensMap1,
                this->vertexCnt1,
                this->vbo1,
                &this->vbo1Resource,
                this->triangleCnt1,
                this->vboTriangleIdx1,
                &this->vboTriangleIdx1Resource)) {
            return false;
        }

        // Make mesh #1 more regular by using deformable model approach
        if (!this->regularizeSurface(
                ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
                this->gridDensMap1,
                &this->vbo1Resource,
                this->vertexCnt1,
                this->vertexNeighbours1_D,
                this->regMaxIt,
                this->regSpringStiffness,
                this->regForcesScl,
                this->regExternalForcesWeight,
                this->interpolMode)) {
            return false;
        }

        // Compute texture coordinates for surface #1
        if (!this->isosurfComputeTexCoords(
                &this->vbo1Resource,
                this->vertexCnt1,
                make_float3(this->gridPotential1.minC[0],
                        this->gridPotential1.minC[1],
                        this->gridPotential1.minC[2]),
                make_float3(this->gridPotential1.maxC[0],
                        this->gridPotential1.maxC[1],
                        this->gridPotential1.maxC[2]),
                this->vertexDataOffsPos,
                this->vertexDataOffsTexCoord,
                this->vertexDataStride)) {
            return false;
        }

        // Apply RMS transformation to starting positions
        if (!this->applyRMSFittingToPosArray(
                mol1,
                &this->vbo1Resource,
                this->vertexCnt1)) {
            return false;
        }

        // Compute normals for surface #1
        if (!this->isosurfComputeNormals(
                ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
                this->gridDensMap1,
                &this->vbo1Resource,
                this->vertexMap1_D,
                this->vertexMapInv1_D,
                this->cubeMap1_D,
                this->cubeMapInv1_D,
                this->vertexCnt1,
                this->vertexDataOffsPos,
                this->vertexDataOffsNormal,
                this->vertexDataStride
                )) {
            return false;
        }

        this->triggerComputeSurfacePoints1 = false;
        this->triggerSurfaceMapping = true;
    }

#if defined(USE_DISTANCE_FIELD)
    if (this->triggerComputeDistanceField ) {
        // Compute distance field based on regularized vertices of surface 0
        if (!this->computeDistField(mol0,
                &this->vbo0Resource,
                this->vertexCnt0,
                this->distField_D,
                ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
                this->gridDensMap0)) {
            return false;
        }
        this->triggerComputeDistanceField = false;
    }
#endif // defined(USE_DISTANCE_FIELD)

    /* Map surface #1 to surface #0 */

    if (this->triggerSurfaceMapping) {

        // Create vbos for vertex data and triangle indices for the mapped
        // surface and register them with CUDA
        if (!this->createVbo(&this->vboMapped,
                this->vertexCnt1*this->vertexDataMappedStride*sizeof(float),
                GL_ARRAY_BUFFER)) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
                &this->vboMappedResource,
                this->vboMapped,
                cudaGraphicsMapFlagsNone))) {
            return false;
        }
        if (!this->createVbo(&this->vboTriangleIdxMapped,
                this->triangleCnt1*sizeof(uint)*3,
                GL_ARRAY_BUFFER)) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
                &this->vboTriangleIdxMappedResource,
                this->vboTriangleIdxMapped,
                cudaGraphicsMapFlagsNone))) {
            return false;
        }

        // Get mapped pointer to vertex data of mapped surface
        // Get mapped pointer to the vbo
        float *vbo1Pt, *vboMappedPt;
        size_t vbo1Size, vboMappedSize;
        cudaGraphicsResource *cudaToken[2];
        cudaToken[0] = this->vbo1Resource;
        cudaToken[1] = this->vboMappedResource;
        if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaToken, 0))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
                reinterpret_cast<void**>(&vbo1Pt), // The mapped pointer
                &vbo1Size,             // The size of the accessible data
                cudaToken[0]))) {                   // The mapped resource
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
                reinterpret_cast<void**>(&vboMappedPt), // The mapped pointer
                &vboMappedSize,             // The size of the accessible data
                cudaToken[1]))) {                   // The mapped resource
            return false;
        }

        // Init buffer with zero
        if (!CudaSafeCall(cudaMemset(vboMappedPt, 0, vboMappedSize))) {
            return false;
        }

        // Init old pos with positions of surface #1
        if (!CudaSafeCall(InitVertexData3(vboMappedPt,
                this->vertexDataMappedStride,
                this->vertexDataMappedOffsPosOld,
                vbo1Pt,
                this->vertexDataStride,
                this->vertexDataOffsPos,
                this->vertexCnt1))) {
            return false;
        }

        // Copy old tex coordinates from surface #1
        if (!CudaSafeCall(InitVertexData3(vboMappedPt,
                this->vertexDataMappedStride,
                this->vertexDataMappedOffsTexCoordOld,
                vbo1Pt,
                this->vertexDataStride,
                this->vertexDataOffsTexCoord,
                this->vertexCnt1))) {
            return false;
        }

        if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaToken, 0))) {
            return false;
        }

        // Init triangle indices of mapped surface with the ones from surface #1
        // Get mapped pointer to vertex data of mapped surface
        // Get mapped pointer to the vbo
        float *vbo1TriangleIdxPt, *vboMappedTriangleIdxPt;
        size_t vbo1TriangleIdxSize, vboMappedTriangleIdxSize;
        cudaGraphicsResource *cudaTokenTriangleIdx[2];
        cudaTokenTriangleIdx[0] = this->vboTriangleIdx1Resource;
        cudaTokenTriangleIdx[1] = this->vboTriangleIdxMappedResource;
        if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaTokenTriangleIdx, 0))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
                reinterpret_cast<void**>(&vbo1TriangleIdxPt), // The mapped pointer
                &vbo1TriangleIdxSize,             // The size of the accessible data
                cudaTokenTriangleIdx[0]))) {                   // The mapped resource
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
                reinterpret_cast<void**>(&vboMappedTriangleIdxPt), // The mapped pointer
                &vboMappedTriangleIdxSize,             // The size of the accessible data
                cudaTokenTriangleIdx[1]))) {                   // The mapped resource
            return false;
        }


        if (!CudaSafeCall(InitVertexData3(
                vboMappedTriangleIdxPt,
                3, // Stride
                0, // Pos
                vbo1TriangleIdxPt,
                3, // Stride
                0,
                this->triangleCnt1))) {
            return false;
        }


        if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaTokenTriangleIdx, 0))) {
            return false;
        }

        // Compute texture coordinates for mapped surface for potential tex #1
        // This has to be done before the RMS fitting to match the potential
        // texture correctly (since it is not RMS transformed)
//        if (!this->isosurfComputeTexCoords(
//                &this->vboMappedResource,
//                this->vertexCnt1,
//                make_float3(this->gridPotential1.minC[0],
//                        this->gridPotential1.minC[1],
//                        this->gridPotential1.minC[2]),
//                make_float3(this->gridPotential1.maxC[0],
//                        this->gridPotential1.maxC[1],
//                        this->gridPotential1.maxC[2]),
//                this->vertexDataMappedOffsPosOld,
//                this->vertexDataMappedOffsTexCoordOld,
//                this->vertexDataMappedStride)) {
//            return false;
//        }

//        // Apply RMS transformation to starting positions
//        if (!this->applyRMSFittingToPosArray(
//                mol1,
//                &this->vboMappedResource,
//                this->vertexCnt1)) {
//            return false;
//        }

    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaToken, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vbo1Pt), // The mapped pointer
            &vbo1Size,             // The size of the accessible data
            cudaToken[0]))) {                   // The mapped resource
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboMappedPt), // The mapped pointer
            &vboMappedSize,             // The size of the accessible data
            cudaToken[1]))) {                   // The mapped resource
        return false;
    }

    // Init new pos with RMS transformed old pos
    if (!CudaSafeCall(InitVertexData3(vboMappedPt,
            this->vertexDataMappedStride,
            this->vertexDataMappedOffsPosNew,
            vboMappedPt,
            this->vertexDataMappedStride,
            this->vertexDataMappedOffsPosOld,
            this->vertexCnt1))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaToken, 0))) {
        return false;
    }

    if (!this->mapIsosurfaceToVolume(
            ((CUDAQuickSurf *)this->cudaqsurf0)->getMap(),
            this->gridDensMap0,
            &this->vboMappedResource,
            &this->vboTriangleIdxMappedResource,
            this->vertexCnt1,
            this->triangleCnt1,
            this->vertexNeighbours1_D,
            this->surfaceMappingMaxIt,
            this->surfMappedSpringStiffness,
            this->surfaceMappingForcesScl,
            this->surfaceMappingExternalForcesWeightScl,
            this->interpolMode)) {
        return false;
    }

    // Get normals of the final vertex positions
    if (!this->isosurfComputeNormals(
            ((CUDAQuickSurf *)this->cudaqsurf1)->getMap(),
            this->gridDensMap1,
            &this->vboMappedResource,
            this->vertexMap1_D,
            this->vertexMapInv1_D,
            this->cubeMap1_D,
            this->cubeMapInv1_D,
            this->vertexCnt1,
            this->vertexDataMappedOffsPosNew,
            this->vertexDataMappedOffsNormal,
            this->vertexDataMappedStride
            )) {
        return false;
    }

        // Compute texture coordinates for mapped surface for potential tex #0
        if (!this->isosurfComputeTexCoords(
                &this->vboMappedResource,
                this->vertexCnt1,
                make_float3(this->gridPotential0.minC[0],
                        this->gridPotential0.minC[1],
                        this->gridPotential0.minC[2]),
                make_float3(this->gridPotential0.maxC[0],
                        this->gridPotential0.maxC[1],
                        this->gridPotential0.maxC[2]),
                this->vertexDataMappedOffsPosNew,
                this->vertexDataMappedOffsTexCoordNew,
                this->vertexDataMappedStride)) {
            return false;
        }

//        // DEBUG Print resulting vertex positions, tex coords, etc.
//        float *vboPt; // Get mapped pointer to the vbo
//        size_t vboSize;
//        CudaSafeCall(cudaGraphicsMapResources(1, &this->vboMappedResource, 0));
//        CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
//                reinterpret_cast<void**>(&vboPt), // The mapped pointer
//                &vboSize,             // The size of the accessible data
//                this->vboMappedResource));                   // The mapped resource
//        HostArr<float> buff;
//        buff.Validate(this->vertexCnt1*this->vertexDataMappedStride);
//        CudaSafeCall(cudaMemcpy(buff.Peek(), vboPt,
//                this->vertexCnt1*this->vertexDataMappedStride*sizeof(float),
//                cudaMemcpyDeviceToHost));
//        for (int i = 0; i < 10; ++i) {
//            float3 posNew = make_float3(
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsPosNew + 0],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsPosNew + 1],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsPosNew + 2]);
//
//            float3 posOld = make_float3(
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsPosOld + 0],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsPosOld + 1],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsPosOld + 2]);
//
//            int3 posNewI;
//            posNewI.x = int((posNew.x - this->gridPotential0.minC[0])/this->gridPotential0.delta[0]);
//            posNewI.y = int((posNew.y - this->gridPotential0.minC[1])/this->gridPotential0.delta[1]);
//            posNewI.z = int((posNew.z - this->gridPotential0.minC[2])/this->gridPotential0.delta[2]);
//
//            int3 posOldI;
//            posOldI.x = int((posOld.x - this->gridPotential1.minC[0])/this->gridPotential1.delta[0]);
//            posOldI.y = int((posOld.y - this->gridPotential1.minC[1])/this->gridPotential1.delta[1]);
//            posOldI.z = int((posOld.z - this->gridPotential1.minC[2])/this->gridPotential1.delta[2]);
//
//            printf("posINew: %i %i %i\n", posNewI.x, posNewI.y, posNewI.z);
//            printf("posIOld: %i %i %i\n", posOldI.x, posOldI.y, posOldI.z);
//            printf("Grid");
//
//            float potOld = cmd1->GetScalarPointDataFloat(0)[
//                 this->gridPotential1.size[0]*(this->gridPotential1.size[1]*posOldI.z+posOldI.y)+posOldI.x];
//            float potNew = cmd0->GetScalarPointDataFloat(0)[
//                 this->gridPotential0.size[0]*(this->gridPotential0.size[1]*posNewI.z+posNewI.y)+posNewI.x];
//
//            printf("%i: pos new (%f %f %f), posOld (%f %f %f), \n     tex coords new (%f %f %f), tex coords old (%f %f %f),\n     potential old %f, potential new %f, difference %f\n", i,
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsPosNew + 0],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsPosNew + 1],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsPosNew + 2],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsPosOld + 0],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsPosOld + 1],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsPosOld + 2],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsTexCoordNew + 0],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsTexCoordNew + 1],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsTexCoordNew + 2],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsTexCoordOld + 0],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsTexCoordOld + 1],
//                    buff.Peek()[i*this->vertexDataMappedStride + this->vertexDataMappedOffsTexCoordOld + 2],
//                    potOld, potNew, fabs(potOld-potNew));
//        }
//        CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vboMappedResource, 0));
//        // END DEBUG

        this->triggerSurfaceMapping = false;
    }

    // Get camera information
    this->cameraInfo =  dynamic_cast<core::view::CallRender3D*>(&call)->GetCameraParameters();


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
    view::CallRender3D *ren = this->rendererCallerSlot.CallAs<view::CallRender3D>();
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
#if defined(USE_TEXTURE_SLICES)
    // Render slices
    if (this->sliceDataSet == 0) {
        if (!this->renderSlices(this->volumeTex0, this->potentialTex0,
                this->gridPotential0, this->gridDensMap0)) {
            return false;
        }
    } else {
        if (!this->renderSlices(this->volumeTex1, this->potentialTex1,
                this->gridPotential1, this->gridDensMap1)) {
            return false;
        }
    }
#endif // defined(USE_TEXTURE_SLICES)

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glLineWidth(2.0f);
    glPointSize(2.0f);

    if (this->surface0RM != SURFACE_NONE) {

        // Sort triangles of surface #0
        if (!this->sortTriangles(
                &this->vbo0Resource,
                this->vertexCnt0,
                &this->vboTriangleIdx0Resource,
                this->triangleCnt0,
                this->vertexDataStride,
                this->vertexDataOffsPos)) {
            return false;
        }

        // Render surface #0
        if (!this->renderSurface(
                this->vbo0,
                this->vertexCnt0,
                this->vboTriangleIdx0,
                this->triangleCnt0*3,
                this->surface0RM,
                this->surface0ColorMode,
                this->potentialTex0,
                this->uniformColorSurf0,
                this->surf0AlphaScl)) {
            return false;
        }
    }

    if (this->surface1RM != SURFACE_NONE) {

        // Sort triangles of surface #1
        if (!this->sortTriangles(
                &this->vbo1Resource,
                this->vertexCnt1,
                &this->vboTriangleIdx1Resource,
                this->triangleCnt1,
                this->vertexDataStride,
                this->vertexDataOffsPos)) {
            return false;
        }

        // Render surface #1
        if (!this->renderSurface(
                this->vbo1,
                this->vertexCnt1,
                this->vboTriangleIdx1,
                this->triangleCnt1*3,
                this->surface1RM,
                this->surface1ColorMode,
                this->potentialTex1,
                this->uniformColorSurf1,
                this->surf1AlphaScl)) {
            return false;
        }
    }

    if (this->surfaceMappedRM != SURFACE_NONE) {

        // Sort triangles of mapped surface
        if (!this->sortTriangles(
                &this->vboMappedResource,
                this->vertexCnt1,
                &this->vboTriangleIdxMappedResource,
                this->triangleCnt1,
                this->vertexDataMappedStride,
                this->vertexDataMappedOffsPosNew)) {
            return false;
        }

        // Render mapped surface
        if (!this->renderMappedSurface(
                this->vboMapped,
                this->vertexCnt1,
                this->vboTriangleIdxMapped,
                this->triangleCnt1*3,
                this->surfaceMappedRM,
                this->surfaceMappedColorMode
//                this->surfMappedColorMinPotential,
//                this->surfMappedColorZeroPotential,
//                this->surfMappedColorMaxPotential,
//                this->surfMappedUniformColor,
//                this->surfMappedColorMinPotentialScl,
//                this->surfMappedColorMidPotentialScl,
//                this->surfMappedColorMaxPotentialScl,
//                this->surfMappedAlphaScl
                )) {
            return false;
        }
    }

//    // DEBUG Print modelview matrix
//    GLfloat matrix[16];
//    printf("Modelview matrix:\n");
//    glGetFloatv (GL_MODELVIEW_MATRIX, matrix);
//    for (int i = 0; i < 4; ++i) {
//       for (int j = 0; j < 4; ++j)  {
//           printf("%.4f ", matrix[j*4+i]);
//       }
//       printf("\n");
//    }
//    // DEBUG END

    glDisable(GL_TEXTURE_3D);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    glEnable(GL_CULL_FACE);

    glPopMatrix();

    this->calltimeOld = calltime;
    mol0->Unlock();
    mol1->Unlock();
    cmd0->Unlock();
    cmd1->Unlock();

    return CheckForGLError();
}


#if defined(USE_TEXTURE_SLICES)
/*
 * ComparativeSurfacePotentialRenderer::renderSlices
 */
bool ComparativeSurfacePotentialRenderer::renderSlices(GLuint densityTex,
        GLuint potentialTex, gridParams potGrid, gridParams densGrid) {

    Vec3f gridMinCoord, gridMaxCoord;
    gridMinCoord[0] = this->bbox.ObjectSpaceBBox().Left();
    gridMinCoord[1] = this->bbox.ObjectSpaceBBox().Bottom();
    gridMinCoord[2] = this->bbox.ObjectSpaceBBox().Back();
    gridMaxCoord[0] = this->bbox.ObjectSpaceBBox().Right();
    gridMaxCoord[1] = this->bbox.ObjectSpaceBBox().Top();
    gridMaxCoord[2] = this->bbox.ObjectSpaceBBox().Front();

    // Texture coordinates for potential map
    float minXPotTC = (this->bbox.ObjectSpaceBBox().Left() - potGrid.minC[0])/(potGrid.maxC[0]-potGrid.minC[0]);
    float minYPotTC = (this->bbox.ObjectSpaceBBox().Bottom() - potGrid.minC[1])/(potGrid.maxC[1]-potGrid.minC[1]);
    float minZPotTC = (this->bbox.ObjectSpaceBBox().Back() - potGrid.minC[2])/(potGrid.maxC[2]-potGrid.minC[2]);
    float maxXPotTC = (this->bbox.ObjectSpaceBBox().Right() - potGrid.minC[0])/(potGrid.maxC[0]-potGrid.minC[0]);
    float maxYPotTC = (this->bbox.ObjectSpaceBBox().Top() - potGrid.minC[1])/(potGrid.maxC[1]-potGrid.minC[1]);
    float maxZPotTC = (this->bbox.ObjectSpaceBBox().Front() - potGrid.minC[2])/(potGrid.maxC[2]-potGrid.minC[2]);
    float planeXPotTC = (this->xPlane-potGrid.minC[0])/(potGrid.maxC[0]- potGrid.minC[0]);
    float planeYPotTC = (this->yPlane-potGrid.minC[1])/(potGrid.maxC[1]- potGrid.minC[1]);
    float planeZPotTC = (this->zPlane-potGrid.minC[2])/(potGrid.maxC[2]- potGrid.minC[2]);

    // Texture coordinates for the density grid
    float minXDensTC = (this->bbox.ObjectSpaceBBox().Left() - densGrid.minC[0])/(densGrid.maxC[0]-densGrid.minC[0]);
    float minYDensTC = (this->bbox.ObjectSpaceBBox().Bottom() - densGrid.minC[1])/(densGrid.maxC[1]-densGrid.minC[1]);
    float minZDensTC = (this->bbox.ObjectSpaceBBox().Back() - densGrid.minC[2])/(densGrid.maxC[2]-densGrid.minC[2]);
    float maxXDensTC = (this->bbox.ObjectSpaceBBox().Right() - densGrid.minC[0])/(densGrid.maxC[0]-densGrid.minC[0]);
    float maxYDensTC = (this->bbox.ObjectSpaceBBox().Top() - densGrid.minC[1])/(densGrid.maxC[1]-densGrid.minC[1]);
    float maxZDensTC = (this->bbox.ObjectSpaceBBox().Front() - densGrid.minC[2])/(densGrid.maxC[2]-densGrid.minC[2]);
    float planeXDensTC = (this->xPlane-densGrid.minC[0])/(densGrid.maxC[0]-densGrid.minC[0]);
    float planeYDensTC = (this->yPlane-densGrid.minC[1])/(densGrid.maxC[1]-densGrid.minC[1]);
    float planeZDensTC = (this->zPlane-densGrid.minC[2])/(densGrid.maxC[2]-densGrid.minC[2]);
//    printf("min %f %f %f, max %f %f %f\n", minXDensTC, minYDensTC, minZDensTC,
//            maxXDensTC, maxYDensTC, maxZDensTC);

    this->sliceShader.Enable();
    glUniform1iARB(this->sliceShader.ParameterLocation("potentialTex"), 0);
    glUniform1iARB(this->sliceShader.ParameterLocation("densityMap"), 1);
    glUniform1iARB(this->sliceShader.ParameterLocation("distField"), 2);
    glUniform1iARB(this->sliceShader.ParameterLocation("renderMode"), this->sliceRM);
    glUniform1fARB(this->sliceShader.ParameterLocation("colorMinVal"), this->sliceMinVal);
    glUniform1fARB(this->sliceShader.ParameterLocation("colorMaxVal"), this->sliceMaxVal);
    glUniform1fARB(this->sliceShader.ParameterLocation("minPotential"), this->minPotential);
    glUniform1fARB(this->sliceShader.ParameterLocation("maxPotential"), this->maxPotential);
    glUniform1fARB(this->sliceShader.ParameterLocation("isoval"), this->qsIsoVal);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glActiveTextureARB(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, densityTex);

#if defined(USE_DISTANCE_FIELD)
    glActiveTextureARB(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, this->distFieldTex);
#endif

    glActiveTextureARB(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, potentialTex);

    // X-plane
    if (this->showXPlane) {
        glBegin(GL_QUADS);
        glMultiTexCoord3fARB(GL_TEXTURE0, planeXPotTC, maxYPotTC, minZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, planeXDensTC, maxYDensTC, minZDensTC);
        glVertex3f(this->xPlane, gridMaxCoord[1], gridMinCoord[2]);
        glMultiTexCoord3fARB(GL_TEXTURE0, planeXPotTC, minYPotTC, minZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, planeXDensTC, minYDensTC, minZDensTC);
        glVertex3f(this->xPlane, gridMinCoord[1], gridMinCoord[2]);
        glMultiTexCoord3fARB(GL_TEXTURE0, planeXPotTC, minYPotTC, maxZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, planeXDensTC, minYDensTC, maxZDensTC);
        glVertex3f(this->xPlane, gridMinCoord[1], gridMaxCoord[2]);
        glMultiTexCoord3fARB(GL_TEXTURE0, planeXPotTC, maxYPotTC, maxZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, planeXDensTC, maxYDensTC, maxZDensTC);
        glVertex3f(this->xPlane, gridMaxCoord[1], gridMaxCoord[2]);
        glEnd();
    }

    // Y-plane
    if (this->showYPlane) {
        glBegin(GL_QUADS);
        glMultiTexCoord3fARB(GL_TEXTURE0, minXPotTC, planeYPotTC, maxZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, minXDensTC, planeYDensTC, maxZDensTC);
        glVertex3f(gridMinCoord[0], this->yPlane, gridMaxCoord[2]);
        glMultiTexCoord3fARB(GL_TEXTURE0, minXPotTC, planeYPotTC, minZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, minXDensTC, planeYDensTC, minZDensTC);
        glVertex3f(gridMinCoord[0], this->yPlane, gridMinCoord[2]);
        glMultiTexCoord3fARB(GL_TEXTURE0, maxXPotTC, planeYPotTC, minZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, maxXDensTC, planeYDensTC, minZDensTC);
        glVertex3f(gridMaxCoord[0], this->yPlane, gridMinCoord[2]);
        glMultiTexCoord3fARB(GL_TEXTURE0, maxXPotTC, planeYPotTC, maxZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, maxXDensTC, planeYDensTC, maxZDensTC);
        glVertex3f(gridMaxCoord[0], this->yPlane, gridMaxCoord[2]);
        glEnd();
    }

    // Z-plane
    if (this->showZPlane) {
        glBegin(GL_QUADS);
        glMultiTexCoord3fARB(GL_TEXTURE0, maxXPotTC, minYPotTC, planeZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, maxXDensTC, minYDensTC, planeZDensTC);
        glVertex3f(gridMaxCoord[0], gridMinCoord[1], this->zPlane);
        glMultiTexCoord3fARB(GL_TEXTURE0, maxXPotTC, maxYPotTC, planeZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, maxXDensTC, maxYDensTC, planeZDensTC);
        glVertex3f(gridMaxCoord[0], gridMaxCoord[1], this->zPlane);
        glMultiTexCoord3fARB(GL_TEXTURE0, minXPotTC, maxYPotTC, planeZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, minXDensTC, maxYDensTC, planeZDensTC);
        glVertex3f(gridMinCoord[0], gridMaxCoord[1], this->zPlane);
        glMultiTexCoord3fARB(GL_TEXTURE0, minXPotTC, minYPotTC, planeZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, minXDensTC, minYDensTC, planeZDensTC);
        glVertex3f(gridMinCoord[0], gridMinCoord[1], this->zPlane);
        glEnd();
    }

    glEnable(GL_CULL_FACE);

    this->sliceShader.Disable();

    return CheckForGLError();
}
#endif // defined(USE_TEXTURE_SLICES)


/*
 * ComparativeSurfacePotentialRenderer::renderSurface
 */
bool ComparativeSurfacePotentialRenderer::renderSurface(
        GLuint &vbo,
        uint vertexCnt,
        GLuint &vboTriangleIdx,
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
            this->vertexDataStride*sizeof(float),
            reinterpret_cast<void*>(this->vertexDataOffsPos*sizeof(float)));
    glVertexAttribPointerARB(attribLocNormal, 3, GL_FLOAT, GL_FALSE,
            this->vertexDataStride*sizeof(float),
            reinterpret_cast<void*>(this->vertexDataOffsNormal*sizeof(float)));
    glVertexAttribPointerARB(attribLocTexCoord, 3, GL_FLOAT, GL_FALSE,
            this->vertexDataStride*sizeof(float),
            reinterpret_cast<void*>(this->vertexDataOffsTexCoord*sizeof(float)));
    CheckForGLError(); // OpenGL error check


    /* Render */

    // Set uniform vars
    glUniform1iARB(this->pplSurfaceShader.ParameterLocation("potentialTex"), 0);
    glUniform1iARB(this->pplSurfaceShader.ParameterLocation("colorMode"), static_cast<int>(colorMode));
    glUniform1iARB(this->pplSurfaceShader.ParameterLocation("renderMode"), static_cast<int>(renderMode));
    glUniform3fvARB(this->pplSurfaceShader.ParameterLocation("colorMin"), 1, this->colorMinPotential.PeekComponents());
    glUniform3fvARB(this->pplSurfaceShader.ParameterLocation("colorMax"), 1, this->colorMaxPotential.PeekComponents());
    glUniform3fvARB(this->pplSurfaceShader.ParameterLocation("colorUniform"), 1, uniformColor.PeekComponents());
    glUniform1fARB(this->pplSurfaceShader.ParameterLocation("minPotential"), this->minPotential);
    glUniform1fARB(this->pplSurfaceShader.ParameterLocation("maxPotential"), this->maxPotential);
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
 * ComparativeSurfacePotentialRenderer::renderMappedSurface
 */
bool ComparativeSurfacePotentialRenderer::renderMappedSurface(
        GLuint &vbo,
        uint vertexCnt,
        GLuint &vboTriangleIdx,
        uint triangleVertexCnt,
        SurfaceRenderMode renderMode,
        SurfaceColorMode colorMode
        ) {


    GLint attribLocPosNew, attribLocPosOld;
    GLint attribLocNormal, attribLocCorruptTriangleFlag;
    GLint attribLocTexCoordNew, attribLocTexCoordOld;


    /* Get vertex attributes from vbo */

    glBindBufferARB(GL_ARRAY_BUFFER, vbo);
    CheckForGLError(); // OpenGL error check

    this->pplMappedSurfaceShader.Enable();
    CheckForGLError(); // OpenGL error check

    // Note: glGetAttribLocation returnes -1 if the attribute if not used in
    // the shader code, because in this case the attribute is optimized out by
    // the compiler
    attribLocPosNew = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "posNew");
    attribLocPosOld = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "posOld");
    attribLocNormal = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "normal");
    attribLocCorruptTriangleFlag = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "corruptTriangleFlag");
    attribLocTexCoordNew = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "texCoordNew");
    attribLocTexCoordOld = glGetAttribLocationARB(this->pplMappedSurfaceShader.ProgramHandle(), "texCoordOld");
    CheckForGLError(); // OpenGL error check

    glEnableVertexAttribArrayARB(attribLocPosNew);
    glEnableVertexAttribArrayARB(attribLocPosOld);
    glEnableVertexAttribArrayARB(attribLocNormal);
    glEnableVertexAttribArrayARB(attribLocCorruptTriangleFlag);
    glEnableVertexAttribArrayARB(attribLocTexCoordNew);
    glEnableVertexAttribArrayARB(attribLocTexCoordOld);
    CheckForGLError(); // OpenGL error check

    glVertexAttribPointerARB(attribLocPosNew, 3, GL_FLOAT, GL_FALSE,
            this->vertexDataMappedStride*sizeof(float),
            reinterpret_cast<void*>(this->vertexDataMappedOffsPosNew*sizeof(float)));
    glVertexAttribPointerARB(attribLocPosOld, 3, GL_FLOAT, GL_FALSE,
            this->vertexDataMappedStride*sizeof(float),
            reinterpret_cast<void*>(this->vertexDataMappedOffsPosOld*sizeof(float)));
    glVertexAttribPointerARB(attribLocNormal, 3, GL_FLOAT, GL_FALSE,
            this->vertexDataMappedStride*sizeof(float),
            reinterpret_cast<void*>(this->vertexDataMappedOffsNormal*sizeof(float)));
    glVertexAttribPointerARB(attribLocCorruptTriangleFlag, 1, GL_FLOAT, GL_FALSE,
            this->vertexDataMappedStride*sizeof(float),
            reinterpret_cast<void*>(this->vertexDataMappedOffsCorruptTriangleFlag*sizeof(float)));
    glVertexAttribPointerARB(attribLocTexCoordNew, 3, GL_FLOAT, GL_FALSE,
            this->vertexDataMappedStride*sizeof(float),
            reinterpret_cast<void*>(this->vertexDataMappedOffsTexCoordNew*sizeof(float)));
    glVertexAttribPointerARB(attribLocTexCoordOld, 3, GL_FLOAT, GL_FALSE,
            this->vertexDataMappedStride*sizeof(float),
            reinterpret_cast<void*>(this->vertexDataMappedOffsTexCoordOld*sizeof(float)));
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
    glUniform1fARB(this->pplMappedSurfaceShader.ParameterLocation("minPotential"), this->minPotential);
    glUniform1fARB(this->pplMappedSurfaceShader.ParameterLocation("maxPotential"), this->maxPotential);
    glUniform1fARB(this->pplMappedSurfaceShader.ParameterLocation("alphaScl"), this->surfMappedAlphaScl);
    glUniform1fARB(this->pplMappedSurfaceShader.ParameterLocation("maxPosDiff"), this->surfMaxPosDiff);

    glActiveTextureARB(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, this->potentialTex1);

    glActiveTextureARB(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, this->potentialTex0);
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
    glDisableVertexAttribArrayARB(attribLocCorruptTriangleFlag);
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
 * ComparativeSurfacePotentialRenderer::sortTriangles
 */
bool ComparativeSurfacePotentialRenderer::sortTriangles(
        cudaGraphicsResource **vboResource,
        uint vertexCount,
        cudaGraphicsResource **vboTriangleIdxResource,
        uint triangleCnt,
        uint dataBuffSize,
        uint dataBuffOffsPos) {

//    printf("Triangle Cnt %u\n", triangleCnt);

    using namespace vislib::math;

    // Calculate cam pos using last column of inverse modelview matrix
    float3 camPos;
    GLfloat m[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, m);
    Mat4f modelMatrix(&m[0]);
    modelMatrix.Invert();
    camPos.x = modelMatrix.GetAt(0, 3);
    camPos.y = modelMatrix.GetAt(1, 3);
    camPos.z = modelMatrix.GetAt(2, 3);

    if (!CudaSafeCall(this->triangleCamDistance_D.Validate(triangleCnt))) {
        return false;
    }

    // We need both cuda graphics resources to be mapped at the same time
    cudaGraphicsResource *cudaToken[2];
    cudaToken[0] = *vboResource;
    cudaToken[1] = *vboTriangleIdxResource;
    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaToken, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data and the triangle indices
    float *vboPt;
    uint *vboTriangleIdxPt;
    size_t vboSize, vboTriangleIdxSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,                         // The size of the accessible data
            cudaToken[0]))) {                 // The mapped resource
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboTriangleIdxPt), // The mapped pointer
            &vboTriangleIdxSize,              // The size of the accessible data
            cudaToken[1]))) {                 // The mapped resource
        return false;
    }

//    if (!CudaSafeCall(TrianglesCalcDistToCam(
//            vboPt,
//            dataBuffSize,
//            dataBuffOffsPos,
//            vboTriangleIdxPt,
//            this->triangleCamDistance_D.Peek(),
//            camPos,
//            triangleCnt))) {
//        return false;
//    }

//    // DEBUG print distance to cam
//    HostArr<float> triangleCamDistance;
//    triangleCamDistance.Validate(this->triangleCamDistance_D.GetSize());
//    this->triangleCamDistance_D.CopyToHost(triangleCamDistance.Peek());
//    for (int i = 0; i < this->triangleCamDistance_D.GetSize(); ++i) {
//        printf("Cam dist %f\n", triangleCamDistance.Peek()[i]);
//    }
//    // END DEBUG

    if (!CudaSafeCall(SortTrianglesByCamDistance(
            vboPt,
            dataBuffSize,
            dataBuffOffsPos,
            camPos,
            vboTriangleIdxPt,
            triangleCnt,
            this->triangleCamDistance_D.Peek()))) {
        return false;
    }

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaToken, 0))) {
        return false;
    }

    return true;
}


/*
 * ComparativeSurfacePotentialRenderer::updateParams
 */
void ComparativeSurfacePotentialRenderer::updateParams() {

#if defined(USE_TEXTURE_SLICES)
    /* Parameters for slice rendering */

    // Data set for slice rendering
    if (this->sliceDataSetSlot.IsDirty()) {
        this->sliceDataSet= this->sliceDataSetSlot.Param<core::param::EnumParam>()->Value();
        this->sliceDataSetSlot.ResetDirty();
    }

    // Render mode for slices
    if (this->sliceRMSlot.IsDirty()) {
        this->sliceRM= this->sliceRMSlot.Param<core::param::EnumParam>()->Value();
        this->sliceRMSlot.ResetDirty();
    }

    // X-plane position
    if (this->xPlaneSlot.IsDirty()) {
        this->xPlane = this->xPlaneSlot.Param<core::param::FloatParam>()->Value();
        this->xPlaneSlot.ResetDirty();
    }

    // X-plane visibility
    if (this->toggleXPlaneSlot.IsDirty()) {
        this->showXPlane = this->toggleXPlaneSlot.Param<core::param::BoolParam>()->Value();
        this->toggleXPlaneSlot.ResetDirty();
    }

    // Y-plane position
    if (this->yPlaneSlot.IsDirty()) {
        this->yPlane = this->yPlaneSlot.Param<core::param::FloatParam>()->Value();
        this->yPlaneSlot.ResetDirty();
    }

    // Y-plane visibility
    if (this->toggleYPlaneSlot.IsDirty()) {
        this->showYPlane = this->toggleYPlaneSlot.Param<core::param::BoolParam>()->Value();
        this->toggleYPlaneSlot.ResetDirty();
    }

    // Z-plane position
    if (this->zPlaneSlot.IsDirty()) {
        this->zPlane = this->zPlaneSlot.Param<core::param::FloatParam>()->Value();
        this->zPlaneSlot.ResetDirty();
    }

    // Z-plane visibility
    if (this->toggleZPlaneSlot.IsDirty()) {
        this->showZPlane = this->toggleZPlaneSlot.Param<core::param::BoolParam>()->Value();
        this->toggleZPlaneSlot.ResetDirty();
    }

    // Minimum texture value
    if (this->sliceMinValSlot.IsDirty()) {
        this->sliceMinVal = this->sliceMinValSlot.Param<core::param::FloatParam>()->Value();
        this->sliceMinValSlot.ResetDirty();
    }

    // Maximum texture value
    if (this->sliceMaxValSlot.IsDirty()) {
        this->sliceMaxVal = this->sliceMaxValSlot.Param<core::param::FloatParam>()->Value();
        this->sliceMaxValSlot.ResetDirty();
    }
#endif // defined(USE_TEXTURE_SLICES)


    /* Parameter for frame by frame comparison */

    // Param slot for compare mode
    if (this->cmpModeSlot.IsDirty()) {
        this->cmpMode = static_cast<CompareMode>(
                this->cmpModeSlot.Param<core::param::EnumParam>()->Value());
        this->cmpModeSlot.ResetDirty();
        this->toggleRMSFit = true;
        this->triggerComputeVolume = true;
        this->triggerInitPotentialTex = true;
    }

    // Param for single frame #0
    if (this->singleFrame0Slot.IsDirty()) {
        this->singleFrame0 = this->singleFrame0Slot.Param<core::param::IntParam>()->Value();
        this->singleFrame0Slot.ResetDirty();
        this->toggleRMSFit = true;
        this->triggerComputeVolume = true;
        this->triggerInitPotentialTex = true;
    }

    // Param for single frame #1
    if (this->singleFrame1Slot.IsDirty()) {
        this->singleFrame1 = this->singleFrame1Slot.Param<core::param::IntParam>()->Value();
        this->singleFrame1Slot.ResetDirty();
        this->toggleRMSFit = true;
        this->triggerComputeVolume = true;
        this->triggerInitPotentialTex = true;
    }


    /* Global rendering options */

    // Parameter for minimum potential value for the color map
    if (this->minPotentialSlot.IsDirty()) {
        this->minPotential = this->minPotentialSlot.Param<core::param::FloatParam>()->Value();
        this->minPotentialSlot.ResetDirty();
    }

    // Parameter for maximum potential value for the color map
    if (this->maxPotentialSlot.IsDirty()) {
        this->maxPotential = this->maxPotentialSlot.Param<core::param::FloatParam>()->Value();
        this->maxPotentialSlot.ResetDirty();
    }


    /* Global mapping options */

    // Interpolation method used when computing external forces
    if (this->interpolModeSlot.IsDirty()) {
        this->interpolMode = static_cast<InterpolationMode>(
                this->interpolModeSlot.Param<core::param::EnumParam>()->Value());
        this->interpolModeSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
        this->triggerComputeSurfacePoints0 = true;
        this->triggerComputeSurfacePoints1 = true;
    }


    /* Parameters for the mapped surface */

    // Parameter for the RMS fitting mode
    if (this->fittingModeSlot.IsDirty()) {
        this->fittingMode = static_cast<RMSFittingMode>(
                this->fittingModeSlot.Param<core::param::EnumParam>()->Value());
        this->fittingModeSlot.ResetDirty();
        this->toggleRMSFit = true;
        this->triggerSurfaceMapping = true;
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

#if defined(USE_DISTANCE_FIELD)
    // Param for minimum distance to use density map instead of distance field
    if (this->surfMappedMaxDistSlot.IsDirty()) {
        this->surfMappedMaxDist = this->surfMappedMaxDistSlot.Param<core::param::FloatParam>()->Value();
        this->surfMappedMaxDistSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
        this->triggerComputeDistanceField = true;
    }
#endif // defined(USE_DISTANCE_FIELD)

    // Param for spring stiffness
    if (this->surfMappedSpringStiffnessSlot.IsDirty()) {
        this->surfMappedSpringStiffness =
                this->surfMappedSpringStiffnessSlot.Param<core::param::FloatParam>()->Value();
        this->surfMappedSpringStiffnessSlot.ResetDirty();
        this->triggerSurfaceMapping = true;
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


    /* Parameters for surface regularization */

    // Maximum number of iterations when regularizing the mesh #0
    if (this->regMaxItSlot.IsDirty()) {
        this->regMaxIt = this->regMaxItSlot.Param<core::param::IntParam>()->Value();
        this->regMaxItSlot.ResetDirty();
        this->triggerComputeSurfacePoints0 = true;
        this->triggerComputeSurfacePoints1 = true;
    }

    // Stiffness of the springs defining the spring forces in surface #0
    if (this->regSpringStiffnessSlot.IsDirty()) {
        this->regSpringStiffness = this->regSpringStiffnessSlot.Param<core::param::FloatParam>()->Value();
        this->regSpringStiffnessSlot.ResetDirty();
        this->triggerComputeSurfacePoints0 = true;
        this->triggerComputeSurfacePoints1 = true;
    }

    // Weighting of the external forces in surface #0, note that the weight
    // of the internal forces is implicitely defined by
    // 1.0 - surf0ExternalForcesWeight
    if (this->regExternalForcesWeightSlot.IsDirty()) {
        this->regExternalForcesWeight =
                this->regExternalForcesWeightSlot.Param<core::param::FloatParam>()->Value();
        this->regExternalForcesWeightSlot.ResetDirty();
        this->triggerComputeSurfacePoints0 = true;
        this->triggerComputeSurfacePoints1 = true;
    }

    // Overall scaling for the forces acting upon surface #0
    if (this->regForcesSclSlot.IsDirty()) {
        this->regForcesScl = this->regForcesSclSlot.Param<core::param::FloatParam>()->Value();
        this->regForcesSclSlot.ResetDirty();
        this->triggerComputeSurfacePoints0 = true;
        this->triggerComputeSurfacePoints1 = true;
    }


    /* Rendering of surface #0 and #1 */

    // Parameter for surface #0 render mode
    if (this->surface0RMSlot.IsDirty()) {
        this->surface0RM = static_cast<SurfaceRenderMode>(
                this->surface0RMSlot.Param<core::param::EnumParam>()->Value());
        this->surface0RMSlot.ResetDirty();
    }

    // Parameter for surface #0 color mode
    if (this->surface0ColorModeSlot.IsDirty()) {
        this->surface0ColorMode = static_cast<SurfaceColorMode>(
                this->surface0ColorModeSlot.Param<core::param::EnumParam>()->Value());
        this->surface0ColorModeSlot.ResetDirty();
    }

    // Param for transparency scaling
    if (this->surf0AlphaSclSlot.IsDirty()) {
        this->surf0AlphaScl = this->surf0AlphaSclSlot.Param<core::param::FloatParam>()->Value();
        this->surf0AlphaSclSlot.ResetDirty();
    }

    // Parameter for surface #1 render mode
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

}

#endif // (defined(WITH_CUDA) && (WITH_CUDA))
