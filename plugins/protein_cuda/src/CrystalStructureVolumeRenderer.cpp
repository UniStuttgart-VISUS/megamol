/*
 * CrystalStructureVolumeRenderer.cpp
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#define SFB_DEMO // Disable everything that is unnecessary and slows down the rendering


#include "stdafx.h"
#include "CrystalStructureVolumeRenderer.h"

#define _USE_MATH_DEFINES 1

#define CALC_GRID 1
#define NOCLIP_ISOSURF 0
#define FILTER_BOUNDARY 0

#include "CritPoints.h"
#include "LIC.h"
#include "CUDAQuickSurf.h"
#include "CUDAMarchingCubes.h"
#include "CUDACurl.cuh"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/ColourParser.h"

#include "vislib/sys/Log.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Matrix.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

#include <thrust/version.h>
#include "cuda_gl_interop.h"

#include <GL/glu.h>

#include <cmath>
#include <fstream>
#include <vector>

using namespace megamol;


/*
 * protein_cuda::CrystalStructureVolumeRenderer::CrystalStructureVolumeRenderer
 */
protein_cuda::CrystalStructureVolumeRenderer::CrystalStructureVolumeRenderer(void):
        Renderer3DModuleDS(),
        dataCallerSlot("getData", "Connects the rendering with data storage"),
        // Atom/edge rendering
        interpolParam("atoms::posInterpol", "Toggle positional interpolation between frames"),
        atomRenderModeParam("atoms::atomRM", "The atom render mode "),
        sphereRadParam("atoms::atomSphereRad", "The sphere radius for atom rendering"),
        edgeBaRenderModeParam("atoms::edgeBaRM", "The render mode for BA edges"),
        baStickRadiusParam("atoms::stickRadBa", "The stick radius for BA edge rendering"),
        edgeTiRenderModeParam("atoms::edgeTiRM", "The render mode for TI edges"),
        tiStickRadiusParam("atoms::stickRadTi", "The stick radius for TI edge rendering"),
        // Positional filter
        filterAllParam("posFilter::all", "Positional filter for cells"),
        filterXMaxParam("posFilter::XMax", "The maximum position in x-direction"),
        filterYMaxParam("posFilter::YMax", "The maximum position in y-direction"),
        filterZMaxParam("posFilter::ZMax", "The maximum position in z-direction"),
        filterXMinParam("posFilter::XMin", "The minimum position in x-direction"),
        filterYMinParam("posFilter::YMin", "The minimum position in y-direction"),
        filterZMinParam("posFilter::ZMin", "The minimum position in z-direction"),
        // Vector field
        vecRMParam("vecField::vecRM", "The render mode of the vectors"),
        vecSclParam("vecField::vecScl", "Scale factor for vectors"),
        arrowRadParam("vecField::arrowRad", "Radius of arrows"),
        arrowUseFilterParam("vecField::arrowUseFilter", "Apply filters to arrow glyphs"),
        showBaAtomsParam("vecField::showBaAtoms", "..."),
        showTiAtomsParam("vecField::showTiAtoms", "..."),
        showOAtomsParam("vecField::showOAtoms", "..."),
        arrColorModeParam("vecField::arrowColor", "Change coloring of the arrow glyphes"),
        minVecMagParam("vecField::filterMinSclLen", "Minimum (scaled) length for vectors"),
        maxVecMagParam("vecField::filterMaxSclLen", "Maximum (scaled) length for vectors"),
        maxVecCurlParam("vecField::filterMaxCurl", "Maximum curl for vectors"),
        toggleNormVecParam("vecField::normalize", "Normalize vectors for arrow glyphs."),
        // Uniform grid
        gridSpacingParam("unigrid::gridSpacing", "Change grid spacing in uniform grid"),
        gridDataRadParam("unigrid::gridDataRad", "Change assumed radius for grid data"),
        gridQualityParam("unigrid::gridQuality", "Change quality of uniform grid calculation"),
        // Slice rendering
        sliceRenderModeParam("slice::sliceRM", "Render mode for slices"),
        xPlaneParam("slice::xPlanePos", "Change the position of the x-Plane"),
        yPlaneParam("slice::yPlanePos", "Change the position of the y-Plane"),
        zPlaneParam("slice::zPlanePos", "Change the position of the z-Plane"),
        toggleYPlaneParam("slice::showYPlane", "Change the position of the y-Plane"),
        toggleXPlaneParam("slice::showXPlane", "Change the position of the x-Plane"),
        toggleZPlaneParam("slice::showZPlane", "Change the position of the z-Plane"),
        licDirSclParam("slice::licVecScl", "LIC direction vector scale factor"),
        licStreamlineLengthParam("slice::licLen", "Length of LIC stream lines"),
        projectVec2DParam("slice::licProjVec2D", "Toggle 2D projection of vectors"),
        licRandBuffSizeParam("slice::licRandBuffSize", "Change the size of the LIC random buffer"),
        licContrastStretchingParam("slice::licContrast", "Change the contrast of the LIC output image"),
        licBrightParam("slice::licBright", "..."),
        licTCSclParam("slice::licTCScl", "Scale factor for texture coordinates."),
        sliceDataSclParam("slice::sliceDataScl", "Scale data visualized on slices."),
        // Rendering of critical points
        showCritPointsParam("critPoints::showCritPoints", "Show critical points of the vector field"),
        cpUsePosFilterParam("critPoints::usePosFilter", "..."),
        // Density grid
        densGridGaussLimParam("densGrid::densGaussLim", "..."),
        densGridRadParam("densGrid::densRad", "..."),
        densGridSpacingParam("densGrid::densSpacing", "..."),
        // Iso surface of density grid
        vColorModeParam("isosurf::color", "Change the coloring mode for the ray marching isosurface."),
        volDeltaParam("isosurf::volDelta", "Change step size for sampling the volume texture"),
        volAlphaSclParam("isosurf::volAlphaScl", "Alpha scale factor for volume rendering"),
        volIsoValParam("isosurf::volIsoVal", "Change iso value for sampling the volume texture"),
        volShowParam("isosurf::showIsoSurf", "..."),
        showIsoSurfParam("isosurf::showIsoSurfMC", "Toggle rendering of the iso surface."),
        volLicDirSclParam("isosurf::volLicDirScl", "..."),
        volLicLenParam("isosurf::volLicLen", "..."),
        volLicContrastStretchingParam("isosurf::volLicContrast", "Change the contrast of the LIC output image"),
        volLicBrightParam("isosurf::volLicBright", "..."),
        volLicTCSclParam("isosurf::volLicTCScl", "Scale factor for texture coordinates."),
        rmTexParam("isosurf::rmTex", "Texture to be used for raymarching."),
        // Fog
        fogStartParam("fog::fogStart", "The minimum z-value for fog"),
        fogEndParam("fog::fogEnd", "The maximum z-value for fog"),
        fogDensityParam("fog::fogDensity", "The density value for fog"),
        fogColourParam("fog::fogCol", "The fog color"),
        meshFileParam("ridges::meshFile", "VTK mesh file"),
        showRidgeParam("ridges::showRidge", "Render ridges"),
        toggleIsoSurfaceSlot("toggleIsoSurf", "..."),
        toggleCurlFilterSlot("toggleCurlFilter", "..."),
        // Flags
        recalcGrid(true), recalcCritPoints(true), recalcCurlMag(true),
        recalcArrowData(true), recalcPosInter(true),
        recalcVisibility(true), recalcDipole(true),
        recalcDensityGrid(true), filterVecField(true),
        // Arrays
        frame0(NULL), frame1(NULL),
        gridCurlMagD(NULL), gridCurlD(NULL), mcVertOut(NULL),
        mcVertOut_D(NULL), mcNormOut(NULL), mcNormOut_D(NULL),
        idxLastFrame(-1), cudaqsurf(NULL), atomCnt(0), visAtomCnt(0),
        edgeCntBa(0), edgeCntTi(0), callTimeOld(-1.0), fboDim(-1, -1),
        srcFboDim(-1, -1), cudaMC(NULL), nVerticesMCOld(0), frameOld(-1),
        setCUDAGLDevice(true) {


    // Data caller slot
	this->dataCallerSlot.SetCompatibleCall<protein_calls::CrystalStructureDataCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    // General params

    // Positional interpolation
    this->interpol = true;
    this->interpolParam.SetParameter(new core::param::BoolParam(this->interpol));
    this->MakeSlotAvailable(&this->interpolParam);

    // Param for minimal length of vectors
    this->minVecMag = 0.0f;
    this->minVecMagParam.SetParameter(
            new core::param::FloatParam(this->minVecMag, 0.0f));
    this->MakeSlotAvailable(&this->minVecMagParam);

    // Param for maximum length of vectors
    this->maxVecMag = 1000.0f;
    this->maxVecMagParam.SetParameter(
            new core::param::FloatParam(this->maxVecMag, 0.0f));
    this->MakeSlotAvailable(&this->maxVecMagParam);

    // Param for maximum curl
    this->maxVecCurl = 1.0f;
    this->maxVecCurlParam.SetParameter(
            new core::param::FloatParam(this->maxVecCurl, 0.0f));
    this->MakeSlotAvailable(&this->maxVecCurlParam);


    // Params for cell based rendering

    // Param for atom render mode
    this->atomRM = ATOM_NONE;
    core::param::EnumParam *rm_atoms = new core::param::EnumParam(this->atomRM);
    rm_atoms->SetTypePair(ATOM_SPHERES, "Spheres");
    rm_atoms->SetTypePair(ATOM_NONE, "None");
    this->atomRenderModeParam << rm_atoms;
    this->MakeSlotAvailable(&this->atomRenderModeParam);

    // Param for sphere scale radius
    this->sphereRad = 0.2f;
    this->sphereRadParam.SetParameter(new core::param::FloatParam(this->sphereRad, 0.0f, 10.0f));
    this->MakeSlotAvailable(&this->sphereRadParam);

    // Param for render mode for ba edges
    this->edgeBaRM = BA_EDGE_NONE;
    core::param::EnumParam *rm_edge_ba = new core::param::EnumParam(this->edgeBaRM);
    rm_edge_ba->SetTypePair(BA_EDGE_NONE, "None");
    rm_edge_ba->SetTypePair(BA_EDGE_LINES, "Lines");
    rm_edge_ba->SetTypePair(BA_EDGE_STICK, "Stick");
    this->edgeBaRenderModeParam << rm_edge_ba;
    this->MakeSlotAvailable(&this->edgeBaRenderModeParam);

    // Param for ba stick radius
    this->baStickRadius = 0.06f;
    this->baStickRadiusParam.SetParameter(new core::param::FloatParam(this->baStickRadius, 0.0f, 10.0f));
    this->MakeSlotAvailable(&this->baStickRadiusParam);

    // Param for render mode for ti edges
    this->edgeTiRM = TI_EDGE_NONE;
    core::param::EnumParam *rm_edge_ti = new core::param::EnumParam(this->edgeTiRM);
    rm_edge_ti->SetTypePair(TI_EDGE_NONE, "None");
    rm_edge_ti->SetTypePair(TI_EDGE_LINES, "Lines");
    rm_edge_ti->SetTypePair(TI_EDGE_STICK, "Stick");
    this->edgeTiRenderModeParam << rm_edge_ti;
    this->MakeSlotAvailable(&this->edgeTiRenderModeParam);

    // Param for ti stick radius
    this->tiStickRadius = 0.1f;
    this->tiStickRadiusParam.SetParameter(new core::param::FloatParam(this->tiStickRadius, 0.0f, 10.0f));
    this->MakeSlotAvailable(&this->tiStickRadiusParam);

    // Param for render mode of displacement vectors
    this->vecRM = VEC_NONE;
    core::param::EnumParam *rm_vec = new core::param::EnumParam(this->vecRM);
    rm_vec->SetTypePair(VEC_NONE, "None");
    rm_vec->SetTypePair(VEC_ARROWS, "Arrows");
    this->vecRMParam << rm_vec;
    this->MakeSlotAvailable(&this->vecRMParam);

    // Param for the radius for the displacement arrows
    this->arrowRad = 1.0f;
    this->arrowRadParam.SetParameter(new core::param::FloatParam(this->arrowRad, 0.0f));
    this->MakeSlotAvailable(&this->arrowRadParam);

    // Filter arrow glyphs
    this->arrowUseFilter = true;
    this->arrowUseFilterParam.SetParameter(new core::param::BoolParam(this->arrowUseFilter));
    this->MakeSlotAvailable(&this->arrowUseFilterParam);


    // Toggle vector normalization
    this->toggleNormVec = false;
    this->toggleNormVecParam.SetParameter(new core::param::BoolParam(this->toggleNormVec));
    this->MakeSlotAvailable(&this->toggleNormVecParam);

    // Show ba atoms
    this->showBaAtoms = true;
    this->showBaAtomsParam.SetParameter(new core::param::BoolParam(this->showBaAtoms));
    this->MakeSlotAvailable(&this->showBaAtomsParam);

    // Show ti atoms
    this->showTiAtoms = true;
    this->showTiAtomsParam.SetParameter(new core::param::BoolParam(this->showTiAtoms));
    this->MakeSlotAvailable(&this->showTiAtomsParam);

    // Show o atoms
    this->showOAtoms = true;
    this->showOAtomsParam.SetParameter(new core::param::BoolParam(this->showOAtoms));
    this->MakeSlotAvailable(&this->showOAtomsParam);


    // Show ridge
    this->showRidge = false;
    this->showRidgeParam.SetParameter(new core::param::BoolParam(this->showRidge));
    this->MakeSlotAvailable(&this->showRidgeParam);

    // Param for arrow coloring
    this->arrColorMode = ARRCOL_ORIENT;
    core::param::EnumParam *arr_cm = new core::param::EnumParam(this->arrColorMode);
    arr_cm->SetTypePair(ARRCOL_ORIENT, "Orientation");
    arr_cm->SetTypePair(ARRCOL_ELEMENT, "Element");
    arr_cm->SetTypePair(ARRCOL_MAGNITUDE, "Magnitude");
    this->arrColorModeParam << arr_cm;
    this->MakeSlotAvailable(&this->arrColorModeParam);

    // Param for displacement scale factor
    this->vecScl = 1.0f;
    this->vecSclParam.SetParameter(new core::param::FloatParam(this->vecScl, 0.0f));
    this->MakeSlotAvailable(&this->vecSclParam);

    // Param for positional cell filter
    this->posFilterAll = 10.0f;
    this->filterAllParam.SetParameter(new core::param::FloatParam(this->posFilterAll, 0.2f, 1000.0f));
    this->MakeSlotAvailable(&this->filterAllParam);

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

    // Params for grid based rendering

    // Param for grid render mode
    this->sliceRM = SLICE_NONE;
    core::param::EnumParam *rm = new core::param::EnumParam(this->sliceRM);
    rm->SetTypePair(VEC_MAG, "Vec Mag");
    rm->SetTypePair(VEC_DIR, "Vec Dir");
    rm->SetTypePair(LIC_GPU, "LIC (GPU)");
    rm->SetTypePair(ROT_MAG, "Curl Mag");
    rm->SetTypePair(SLICE_DENSITYMAP, "Density map");
    rm->SetTypePair(SLICE_DELTA_X, "Delta X");
    rm->SetTypePair(SLICE_DELTA_Y, "Delta Y");
    rm->SetTypePair(SLICE_DELTA_Z, "Delta Z");
    rm->SetTypePair(SLICE_COLORMAP, "Color tex");
    rm->SetTypePair(SLICE_NONE, "None");
    this->sliceRenderModeParam << rm;
    this->MakeSlotAvailable(&this->sliceRenderModeParam);

    // Param for grid spacing
    this->gridSpacing = 1.0f;
    this->gridSpacingParam.SetParameter(new core::param::FloatParam(this->gridSpacing, 0.0f, 5.0f));
    this->MakeSlotAvailable(&this->gridSpacingParam);

    // Param for dipole radius
    this->gridDataRad = 5.0f;
    this->gridDataRadParam.SetParameter(new core::param::FloatParam(this->gridDataRad, 0.0f, 100.0f));
    this->MakeSlotAvailable(&this->gridDataRadParam);

    // Gauss quality
    this->gridQuality = 0;
    core::param::EnumParam *gq = new core::param::EnumParam(this->gridQuality);
    gq->SetTypePair(0, "Low");
    gq->SetTypePair(1, "Medium");
    gq->SetTypePair(2, "High");
    gq->SetTypePair(3, "Maximum");
    this->gridQualityParam << gq;
    this->MakeSlotAvailable(&this->gridQualityParam);

    // X-plane position
    this->xPlane = 0.0f;
    this->xPlaneParam.SetParameter(new core::param::FloatParam(this->xPlane, -120.0f, 120.0f));
    this->MakeSlotAvailable(&this->xPlaneParam);

    // X-plane visibility
    this->showXPlane = false;
    this->toggleXPlaneParam.SetParameter(new core::param::BoolParam(this->showXPlane));
    this->MakeSlotAvailable(&this->toggleXPlaneParam);

    // Y-plane position
    this->yPlane = 0.0f;
    this->yPlaneParam.SetParameter(new core::param::FloatParam(this->yPlane, -120.0f, 120.0f));
    this->MakeSlotAvailable(&this->yPlaneParam);

    // Y-plane visibility
    this->showYPlane = false;
    this->toggleYPlaneParam.SetParameter(new core::param::BoolParam(this->showYPlane));
    this->MakeSlotAvailable(&this->toggleYPlaneParam);

    // Z-plane position
    this->zPlane = 0.0f;
    this->zPlaneParam.SetParameter(new core::param::FloatParam(this->zPlane, -120.0f, 120.0f));
    this->MakeSlotAvailable(&this->zPlaneParam);

    // Z-plane visibility
    this->showZPlane = true;
    this->toggleZPlaneParam.SetParameter(new core::param::BoolParam(this->showZPlane));
    this->MakeSlotAvailable(&this->toggleZPlaneParam);

    // Scale the direction vector when computing LIC
    this->licDirScl = 1.0f;
    this->licDirSclParam.SetParameter(new core::param::FloatParam(this->licDirScl, 0.0f));
    this->MakeSlotAvailable(&this->licDirSclParam);

    // Scale the direction vector when computing LIC
    this->licStreamlineLength = 10;
    this->licStreamlineLengthParam.SetParameter(new core::param::IntParam(this->licStreamlineLength, 1));
    this->MakeSlotAvailable(&this->licStreamlineLengthParam);

    // Toggle projecting the vectors
    this->projectVec2D = true;
    this->projectVec2DParam.SetParameter(new core::param::BoolParam(this->projectVec2D));
    this->MakeSlotAvailable(&this->projectVec2DParam);

    // Change LIC rand buff size
    this->licRandBuffSize = 64;
    this->licRandBuffSizeParam.SetParameter(new core::param::IntParam(this->licRandBuffSize, 16, 512));
    this->MakeSlotAvailable(&this->licRandBuffSizeParam);

    // Change LIC rand buff size
    this->sliceDataScl = 1.0f;
    this->sliceDataSclParam.SetParameter(new core::param::FloatParam(this->sliceDataScl, 0.0f, 100000.0f));
    this->MakeSlotAvailable(&this->sliceDataSclParam);

    // Change LIC contrast
    this->licContrastStretching = 0.0f;
    this->licContrastStretchingParam.SetParameter(new core::param::FloatParam(this->licContrastStretching, 0.0f, 0.5f));
    this->MakeSlotAvailable(&this->licContrastStretchingParam);

    // Change LIC brightness
    this->licBright = 0.0f;
    this->licBrightParam.SetParameter(new core::param::FloatParam(this->licBright, 0.0f));
    this->MakeSlotAvailable(&this->licBrightParam);

    // Change LIC TC scale factor
    this->licTCScl = 1.0f;
    this->licTCSclParam.SetParameter(new core::param::FloatParam(this->licTCScl, 0.0f));
    this->MakeSlotAvailable(&this->licTCSclParam);

    // Show critical points
    this->showCritPoints = false;
    this->showCritPointsParam.SetParameter(new core::param::BoolParam(this->showCritPoints));
    this->MakeSlotAvailable(&this->showCritPointsParam);

    // Quality for the density grid
    this->densGridGaussLim = 3.5f;
    this->densGridGaussLimParam.SetParameter(new core::param::FloatParam(this->densGridGaussLim, 0.0f, 200.0f));
    this->MakeSlotAvailable(&this->densGridGaussLimParam);

    // Density grid data radius
    this->densGridRad = 1.0f;
    this->densGridRadParam.SetParameter(new core::param::FloatParam(this->densGridRad, 0.0f, 200.0f));
    this->MakeSlotAvailable(&this->densGridRadParam);

    // Density grid data spacing
    this->densGridSpacing = 4.0f;
    this->densGridSpacingParam.SetParameter(new core::param::FloatParam(this->densGridSpacing, 0.0f, 10.0f));
    this->MakeSlotAvailable(&this->densGridSpacingParam);

    // Show volume texture
    this->volShow = false;
    this->volShowParam.SetParameter(new core::param::BoolParam(this->volShow));
    this->MakeSlotAvailable(&this->volShowParam);

    // Change step size for raycasting
    this->volDelta = 0.01f;
    this->volDeltaParam.SetParameter(new core::param::FloatParam(this->volDelta, 0.0f, 2.0f));
    this->MakeSlotAvailable(&this->volDeltaParam);

    // Change isovalue for raycasting
    this->volIsoVal = 0.7f;
    this->volIsoValParam.SetParameter(new core::param::FloatParam(this->volIsoVal, 0.0f, 100.0f));
    this->MakeSlotAvailable(&this->volIsoValParam);

    // Scale factor for volume alpha value
    this->volAlphaScl = 1.0f;
    this->volAlphaSclParam.SetParameter(new core::param::FloatParam(this->volAlphaScl, 0.0f, 1000000.0f));
    this->MakeSlotAvailable(&this->volAlphaSclParam);

    // Param for isosurface color mode
    this->vColorMode = VOL_UNI;
    core::param::EnumParam *cm = new core::param::EnumParam(this->vColorMode);
    cm->SetTypePair(VOL_UNI, "Uniform");
    cm->SetTypePair(VOL_DIR, "Vec dir");
    cm->SetTypePair(VOL_MAG, "Vec mag");
    cm->SetTypePair(VOL_LIC, "LIC");
    this->vColorModeParam << cm;
    this->MakeSlotAvailable(&this->vColorModeParam);

    // Param for raymarching texture
    this->rmTex = DENSITY;
    core::param::EnumParam *rm_tex = new core::param::EnumParam(this->rmTex);
    rm_tex->SetTypePair(DENSITY, "Density map");
    rm_tex->SetTypePair(DIR_MAG, "Dir magnitude");
    rm_tex->SetTypePair(CURL_MAG, "Curl magnitude");
    this->rmTexParam << rm_tex;
    this->MakeSlotAvailable(&this->rmTexParam);

    // Show iso surface
    this->showIsoSurf = false;
    this->showIsoSurfParam.SetParameter(new core::param::BoolParam(this->showIsoSurf));
  //  this->MakeSlotAvailable(&this->showIsoSurfParam);

    // Scale the direction vector when computing LIC on isosurface
    this->volLicDirScl = 1.0f;
    this->volLicDirSclParam.SetParameter(new core::param::FloatParam(this->volLicDirScl, 0.0f));
    this->MakeSlotAvailable(&this->volLicDirSclParam);

    // Scale the direction vector when computing LIC on isosurface
    this->volLicLen = 10;
    this->volLicLenParam.SetParameter(new core::param::IntParam(this->volLicLen, 1));
    this->MakeSlotAvailable(&this->volLicLenParam);

    // Change LIC contrast
    this->volLicContrastStretching = 0.0f;
    this->volLicContrastStretchingParam.SetParameter(new core::param::FloatParam(this->volLicContrastStretching, 0.0f, 0.5f));
    this->MakeSlotAvailable(&this->volLicContrastStretchingParam);

    // Change LIC brightness
    this->volLicBright = 0.0f;
    this->volLicBrightParam.SetParameter(new core::param::FloatParam(this->volLicBright, 0.0f));
    this->MakeSlotAvailable(&this->volLicBrightParam);

    // Volume LIC texture coordinates
    this->volLicTCScl = 1.0f;
    this->volLicTCSclParam.SetParameter(new core::param::FloatParam(this->volLicTCScl, 0.0f));
    this->MakeSlotAvailable(&this->volLicTCSclParam);

    // Fog parameters

    // Param for minimum z-value of fog
    this->fogStart = 1.0f;
    this->fogStartParam.SetParameter(new core::param::FloatParam(this->fogStart, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->fogStartParam);

    // Param for maximum z-value of fog
    this->fogEnd = 1.0f;
    this->fogEndParam.SetParameter(new core::param::FloatParam(this->fogEnd, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->fogEndParam);

    // Param for fog density
    this->fogDensity = 0.4f;
    this->fogDensityParam.SetParameter(
            new core::param::FloatParam(this->fogDensity, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->fogDensityParam);

    // Param for fog colour
    this->fogColour[0] = 0.0f;
    this->fogColour[1] = 0.0f;
    this->fogColour[2] = 0.13f;
    this->fogColour[3] = 1.0f;
    this->fogColourParam << new core::param::StringParam(
            core::utility::ColourParser::ToString(
                    this->fogColour[0],
                    this->fogColour[1],
                    this->fogColour[2]));
    this->MakeSlotAvailable(&this->fogColourParam);

    // Param to goggle pos filter for crit points
    this->cpUsePosFilter = true;
    this->cpUsePosFilterParam.SetParameter(new core::param::BoolParam(this->cpUsePosFilter));
    this->MakeSlotAvailable(&this->cpUsePosFilterParam);

    // VTK mesh file
    this->renderMesh = false;
    this->meshFileParam.SetParameter(new core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->meshFileParam);

    this->toggleIsoSurfaceSlot.SetParameter(new core::param::ButtonParam('I'));
    this->MakeSlotAvailable(&this->toggleIsoSurfaceSlot);

    this->toggleCurlFilterSlot.SetParameter(new core::param::ButtonParam('C'));
    this->MakeSlotAvailable(&this->toggleCurlFilterSlot);
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::~CrystalStructureRenderer
 */
protein_cuda::CrystalStructureVolumeRenderer::~CrystalStructureVolumeRenderer (void) {
    this->Release();
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::CalcDensityTex
 */
bool protein_cuda::CrystalStructureVolumeRenderer::CalcDensityTex(
		const protein_calls::CrystalStructureDataCall *dc,
        const float *atomPos) {

#if (defined(CALC_GRID) && (CALC_GRID))

    using namespace vislib;
    using namespace vislib::sys;
    using namespace vislib::math;

    if(this->recalcDensityGrid) {

        
//        printf("Calc density grid start\n"); // DEBUG

        Vector<float, 3> gridMinCoord, gridMaxCoord, gridXAxis, gridYAxis,
        gridZAxis, gridOrg;
        Vector<int, 3> gridDim;
        float gausslim;

        // Init uniform grid
        gridMinCoord[0] = this->bbox.ObjectSpaceBBox().Left();
        gridMinCoord[1] = this->bbox.ObjectSpaceBBox().Bottom();
        gridMinCoord[2] = this->bbox.ObjectSpaceBBox().Back();
        gridMaxCoord[0] = this->bbox.ObjectSpaceBBox().Right();
        gridMaxCoord[1] = this->bbox.ObjectSpaceBBox().Top();
        gridMaxCoord[2] = this->bbox.ObjectSpaceBBox().Front();
        gridXAxis[0] = gridMaxCoord[0]-gridMinCoord[0];
        gridYAxis[1] = gridMaxCoord[1]-gridMinCoord[1];
        gridZAxis[2] = gridMaxCoord[2]-gridMinCoord[2];
        gridDim[0] = (int) ceil(gridXAxis[0] / this->densGridSpacing);
        gridDim[1] = (int) ceil(gridYAxis[1] / this->densGridSpacing);
        gridDim[2] = (int) ceil(gridZAxis[2] / this->densGridSpacing);
        gridXAxis[0] = (gridDim[0]-1) * this->densGridSpacing;
        gridYAxis[1] = (gridDim[1]-1) * this->densGridSpacing;
        gridZAxis[2] = (gridDim[2]-1) * this->densGridSpacing;
        gridMaxCoord[0] = gridMinCoord[0] + gridXAxis[0];
        gridMaxCoord[1] = gridMinCoord[1] + gridYAxis[1];
        gridMaxCoord[2] = gridMinCoord[2] + gridZAxis[2];
        gridOrg[0] = gridMinCoord[0];
        gridOrg[1] = gridMinCoord[1];
        gridOrg[2] = gridMinCoord[2];

        // (Re)allocate memory if necessary
        if(this->uniGridDensity.GetGridDim().X()*
                this->uniGridDensity.GetGridDim().Y()*
                this->uniGridDensity.GetGridDim().Z() == 0) {
            this->uniGridDensity.Init(gridDim, gridOrg, this->densGridSpacing);
        }
        if(this->uniGridColor.GetGridDim().X()*
                this->uniGridColor.GetGridDim().Y()*
                this->uniGridColor.GetGridDim().Z() == 0) {
            this->uniGridColor.Init(gridDim, gridOrg, this->densGridSpacing);
        }

        /*switch (this->densGridQuality) {
        case 3: gausslim = 10.0f; break;  // max quality
        case 2: gausslim = 3.0f; break;  // high quality
        case 1: gausslim = 2.5f; break;  // medium quality
        case 0:
        default: gausslim = 2.0f; break; // low quality
        }*/
        gausslim = this->densGridGaussLim;

        Array<float> gridPos, gridCol;
        gridPos.SetCapacityIncrement(1000);
        gridCol.SetCapacityIncrement(1000);

        time_t t = clock(); // DEBUG

        unsigned int dipole_cnt = 0;

        this->maxLenDiff = 0.0f;
        for(int cnt = 0; cnt < static_cast<int>(dc->GetDipoleCnt()); cnt++) {

            if(this->arrowVis[cnt]) {

                // Set color according to coloring mode (LIC, Uni color are
                // set directly in the shader and therefore don't need a
                // texture

#if (defined(NOCLIP_ISOSURF) && (NOCLIP_ISOSURF))
                // Only add to density grid, if dipole is visible
                if(this->visDipole[cnt]) {
#endif
                float lenDiffVec;
                vislib::math::Vector<float, 3> diffVec;
                diffVec.Set(dc->GetDipole()[cnt*3+0],
                            dc->GetDipole()[cnt*3+1],
                            dc->GetDipole()[cnt*3+2]);
                lenDiffVec = diffVec.Length();

                if(this->toggleNormVec) {
                    this->maxLenDiff = 1.0f;
                }
                else {
                    this->maxLenDiff = vislib::math::Max(this->maxLenDiff, lenDiffVec);
                }

                gridPos.Add(dc->GetDipolePos()[3*cnt+0]-gridOrg[0]);
                gridPos.Add(dc->GetDipolePos()[3*cnt+1]-gridOrg[1]);
                gridPos.Add(dc->GetDipolePos()[3*cnt+2]-gridOrg[2]);
                if((this->toggleNormVec)&&(lenDiffVec > 0.0f)) {
                    gridPos.Add(1.0f);
                }
                else {
                    gridPos.Add(lenDiffVec);
                }

                vislib::math::Vector<float, 3> color;
                color = this->getColorGrad(lenDiffVec*this->vecScl);
                if(this->vColorMode == VOL_MAG) {
                    gridCol.Add(color.X());
                    gridCol.Add(color.Y());
                    gridCol.Add(color.Z());
                    gridCol.Add(1.0f);
                }
                else if(this->vColorMode == VOL_DIR) {
                    /*gridCol.Add((diffVec.X()/lenDiffVec+1.0)*0.5f);
                    gridCol.Add((diffVec.Y()/lenDiffVec+1.0)*0.5f);
                    gridCol.Add((diffVec.Z()/lenDiffVec+1.0)*0.5f);*/
                    /*gridCol.Add((diffVec.X()+1.0)*0.5f);
                    gridCol.Add((diffVec.Y()+1.0)*0.5f);
                    gridCol.Add((diffVec.Z()+1.0)*0.5f);*/

                    gridCol.Add(diffVec.X());
                    gridCol.Add(diffVec.Y());
                    gridCol.Add(diffVec.Z());
                    
                    gridCol.Add(1.0f);
                }
                else if(this->vColorMode == VOL_LIC) {
                    gridCol.Add(diffVec.X());
                    gridCol.Add(diffVec.Y());
                    gridCol.Add(diffVec.Z());
                    gridCol.Add(1.0f);
                }
                else {
                    gridCol.Add(0.0f);
                    gridCol.Add(0.0f);
                    gridCol.Add(0.0f);
                    gridCol.Add(0.0f);
                }

                dipole_cnt++;
#if (defined(NOCLIP_ISOSURF) && (NOCLIP_ISOSURF))
                }
#endif
            }
        }


        /*for(int i = 0; i < gridCol.Count()/4; i++) {
            printf("Grid color (%f %f %f %f)\n",
                    gridCol[4*i+0],
                    gridCol[4*i+1],
                    gridCol[4*i+2],
                    gridCol[4*i+3]);
        }*/ // DEBUG

//        printf("Vectors contained in density tex %u\n", dipole_cnt);

        // Compute uniform grid containing density map of the vectors
        CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf;

        int rc = cqs->calc_map(
                static_cast<long>(gridPos.Count()/4),
                gridPos.PeekElements(),
                gridCol.PeekElements(),
                true,                // Use 'color' array
                gridOrg.PeekComponents(),
                gridDim.PeekComponents(),
                maxLenDiff,          // Maximum radius
                this->densGridRad,   // Radius scaling
                this->densGridSpacing,
                this->volIsoVal,                // Iso value
                gausslim);

        if(rc != 0) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s::CalcDensityTex: Quicksurf class returned val (number of data points %u)!= 0\n",
                    this->ClassName(), gridPos.Count()/4);
            return false;
        }

        // Setup texture

        this->uniGridDensity.MemCpyFromDevice(cqs->getMap());
        this->uniGridColor.MemCpyFromDevice((float3*)cqs->getColorMap());

        /*for(int x = 0; x < gridDim.X(); x++) {
            for(int y = 0; y < gridDim.Y(); y++) {
                for(int z = 0; z < gridDim.Z(); z++) {
                    printf("(%i %i %i) (%f %f %f): %f\n", x, y, z,
                            this->uniGridColor.GetAt(x, y, z).x,
                            this->uniGridColor.GetAt(x, y, z).y,
                            this->uniGridColor.GetAt(x, y, z).z);
                }
            }
        }*/ // DEBUG

        // Setup density map
        if(!glIsTexture(this->uniGridDensityTex)) glGenTextures(1, &this->uniGridDensityTex);
        glBindTexture(GL_TEXTURE_3D, this->uniGridDensityTex);
        glTexImage3DEXT(GL_TEXTURE_3D,
                0,
                GL_ALPHA,
                gridDim[0],
                gridDim[1],
                gridDim[2],
                0,
                GL_ALPHA,
                GL_FLOAT,
                this->uniGridDensity.PeekBuffer());
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        // Setup color texture
        if(!glIsTexture(this->uniGridColorTex)) glGenTextures(1, &this->uniGridColorTex);
        glBindTexture(GL_TEXTURE_3D, this->uniGridColorTex);
        glTexImage3DEXT(GL_TEXTURE_3D,
                0,
                GL_RGB32F,
                gridDim[0],
                gridDim[1],
                gridDim[2],
                0,
                GL_RGB,
                GL_FLOAT,
                this->uniGridColor.PeekBuffer());
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);


        // Check for opengl error
        GLenum err = glGetError();
        if(err != GL_NO_ERROR) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s::CalcDensityMap: glError %s\n",
                    this->ClassName(),
                    gluErrorString(err));
            return false;
        }

//        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: time for computing density map: %f",
//                this->ClassName(),
//                (double(clock()-t)/double(CLOCKS_PER_SEC) )); // DEBUG

        this->recalcDensityGrid = false;
    }

    //printf("Calc density grid stop\n"); // DEBUG
#endif

    return true;
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::CalcMagCurlTex
 */
bool protein_cuda::CrystalStructureVolumeRenderer::CalcMagCurlTex() {
    using namespace vislib::sys;
    using namespace vislib::math;

#if (defined(CALC_GRID) && (CALC_GRID))

    Vector<float, 3> gridMinCoord, gridMaxCoord, gridXAxis, gridYAxis,
                     gridZAxis, gridOrg;
    Vector<int, 3> gridDim;

    gridMinCoord[0] = this->bbox.ObjectSpaceBBox().Left();
    gridMinCoord[1] = this->bbox.ObjectSpaceBBox().Bottom();
    gridMinCoord[2] = this->bbox.ObjectSpaceBBox().Back();
    gridMaxCoord[0] = this->bbox.ObjectSpaceBBox().Right();
    gridMaxCoord[1] = this->bbox.ObjectSpaceBBox().Top();
    gridMaxCoord[2] = this->bbox.ObjectSpaceBBox().Front();
    gridXAxis[0] = gridMaxCoord[0]-gridMinCoord[0];
    gridYAxis[1] = gridMaxCoord[1]-gridMinCoord[1];
    gridZAxis[2] = gridMaxCoord[2]-gridMinCoord[2];
    gridDim[0] = (int) ceil(gridXAxis[0] / this->gridSpacing);
    gridDim[1] = (int) ceil(gridYAxis[1] / this->gridSpacing);
    gridDim[2] = (int) ceil(gridZAxis[2] / this->gridSpacing);
    gridXAxis[0] = (gridDim[0]-1) * this->gridSpacing;
    gridYAxis[1] = (gridDim[1]-1) * this->gridSpacing;
    gridZAxis[2] = (gridDim[2]-1) * this->gridSpacing;
    gridMaxCoord[0] = gridMinCoord[0] + gridXAxis[0];
    gridMaxCoord[1] = gridMinCoord[1] + gridYAxis[1];
    gridMaxCoord[2] = gridMinCoord[2] + gridZAxis[2];
    gridOrg[0] = gridMinCoord[0];
    gridOrg[1] = gridMinCoord[1];
    gridOrg[2] = gridMinCoord[2];

    // (Re)allocate memory if necessary
    if(this->uniGridCurlMag.GetGridDim().X()*
            this->uniGridCurlMag.GetGridDim().Y()*
            this->uniGridCurlMag.GetGridDim().Z() == 0) {
        this->uniGridCurlMag.Init(gridDim, gridOrg, this->gridSpacing);
    }

    unsigned int nVoxels = gridDim.X()*gridDim.Y()*gridDim.Z();
    cudaError_t cudaErr;
    GLenum glErr;

    time_t t = clock(); // DEBUG

    // Allocate device memory if necessary

    if(this->gridCurlD == NULL) {
        checkCudaErrors(cudaMalloc((void **)&this->gridCurlD, sizeof(float)*nVoxels*3));
    }
    if(this->gridCurlMagD == NULL) {
        checkCudaErrors(cudaMalloc((void **)&this->gridCurlMagD, sizeof(float)*nVoxels));
    }

    // Copy grid parameters to constant device memory

    this->params.gridDim.x = gridDim.X();
    this->params.gridDim.y = gridDim.Y();
    this->params.gridDim.z = gridDim.Z();

    this->params.gridOrg.x = gridOrg.X();
    this->params.gridOrg.y = gridOrg.Y();
    this->params.gridOrg.z = gridOrg.Z();

    this->params.gridStep.x = this->gridSpacing;
    this->params.gridStep.y = this->gridSpacing;
    this->params.gridStep.z = this->gridSpacing;

    this->params.gridXAxis.x = gridXAxis.X();
    this->params.gridXAxis.y = gridXAxis.Y();
    this->params.gridXAxis.z = gridXAxis.Z();

    this->params.gridYAxis.x = gridYAxis.X();
    this->params.gridYAxis.y = gridYAxis.Y();
    this->params.gridYAxis.z = gridYAxis.Z();

    this->params.gridZAxis.x = gridZAxis.X();
    this->params.gridZAxis.y = gridZAxis.Y();
    this->params.gridZAxis.z = gridZAxis.Z();

    cudaErr = protein_cuda::CUDASetCurlParams(&this->params);
    if(cudaErr != cudaSuccess) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s::CalcMagCurlTex: unable to copy grid params to device memory (%s)\n",
                this->ClassName(), cudaGetErrorString(cudaErr));
        return false;
    }

    // Compute curl magnitude

    CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf;

    cudaErr = protein_cuda::CudaGetCurlMagnitude(cqs->getColorMap(),
                                            this->gridCurlD,
                                            this->gridCurlMagD,
                                            nVoxels,
                                            this->gridSpacing);

    if(cudaErr != cudaSuccess) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s::CalcMagCurlTex: unable to compute curl vector magnitude (%s)\n",
                this->ClassName(), cudaGetErrorString(cudaErr));
        return false;
    }

    this->uniGridCurlMag.MemCpyFromDevice(this->gridCurlMagD);

    //for(int cnt = 0; cnt < this->gridDim.X()*this->gridDim.Y()*this->gridDim.Z(); cnt++) {
    //  printf("Curl mag %f\n", this->gridCurlMag[cnt]);
    //} // DEBUG

    // Setup texture

    if(!glIsTexture(this->curlMagTex)) glGenTextures(1, &this->curlMagTex);

    glEnable(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_3D, this->curlMagTex);
    glTexImage3DEXT(GL_TEXTURE_3D, 0, GL_ALPHA, gridDim.X(),
            gridDim.Y(), gridDim.Z(), 0, GL_ALPHA,
            GL_FLOAT, this->uniGridCurlMag.PeekBuffer());

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

    glBindTexture(GL_TEXTURE_3D, 0);
    glDisable(GL_TEXTURE_3D);

//    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: time for computing curl: %f",
//            this->ClassName(),
//            (double(clock()-t)/double(CLOCKS_PER_SEC) )); // DEBUG

    // Check for opengl error
    glErr = glGetError();
    if(glErr != GL_NO_ERROR) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s::CalcMagCurlTex: unable to setup curl texture (glError %s)\n",
                this->ClassName(), gluErrorString(glErr));
        return false;
    }

#endif

    return true;
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::CalcUniGrid
 */
bool protein_cuda::CrystalStructureVolumeRenderer::CalcUniGrid (
		const protein_calls::CrystalStructureDataCall *dc,
        const float *atomPos,
        const float *col) {

#if (defined(CALC_GRID) && (CALC_GRID))

    //printf("CalcUnigrid start\n"); // DEBUG

    using namespace vislib::sys;
    using namespace vislib::math;

    Vector<float, 3> gridMinCoord, gridMaxCoord, gridXAxis, gridYAxis,
                     gridZAxis, gridOrg;
    Vector<int, 3> gridDim;
    float gausslim;

    // Init uniform grid
    gridMinCoord[0] = this->bbox.ObjectSpaceBBox().Left();
    gridMinCoord[1] = this->bbox.ObjectSpaceBBox().Bottom();
    gridMinCoord[2] = this->bbox.ObjectSpaceBBox().Back();
    gridMaxCoord[0] = this->bbox.ObjectSpaceBBox().Right();
    gridMaxCoord[1] = this->bbox.ObjectSpaceBBox().Top();
    gridMaxCoord[2] = this->bbox.ObjectSpaceBBox().Front();
    gridXAxis[0] = gridMaxCoord[0]-gridMinCoord[0];
    gridYAxis[1] = gridMaxCoord[1]-gridMinCoord[1];
    gridZAxis[2] = gridMaxCoord[2]-gridMinCoord[2];
    gridDim[0] = (int) ceil(gridXAxis[0] / this->gridSpacing);
    gridDim[1] = (int) ceil(gridYAxis[1] / this->gridSpacing);
    gridDim[2] = (int) ceil(gridZAxis[2] / this->gridSpacing);
    gridXAxis[0] = (gridDim[0]-1) * this->gridSpacing;
    gridYAxis[1] = (gridDim[1]-1) * this->gridSpacing;
    gridZAxis[2] = (gridDim[2]-1) * this->gridSpacing;
    gridMaxCoord[0] = gridMinCoord[0] + gridXAxis[0];
    gridMaxCoord[1] = gridMinCoord[1] + gridYAxis[1];
    gridMaxCoord[2] = gridMinCoord[2] + gridZAxis[2];
    gridOrg[0] = gridMinCoord[0];
    gridOrg[1] = gridMinCoord[1];
    gridOrg[2] = gridMinCoord[2];

    //printf("gridDim min(%f %f %f), max(%f %f %f), dim(%u %u %u), spacing = %f\n",
    //      gridMinCoord[0], gridMinCoord[1], gridMinCoord[2],
    //      gridMaxCoord[0], gridMaxCoord[1], gridMaxCoord[2],
    //      gridDim[0], gridDim[1], gridDim[2], this->gridSpacing); // DEBUG

    // (Re)allocate if necessary
    if(this->uniGridVecField.GetGridDim().X()*
            this->uniGridVecField.GetGridDim().Y()*
            this->uniGridVecField.GetGridDim().Z() == 0) {
        this->uniGridVecField.Init(gridDim, gridOrg, this->gridSpacing);
    }

    switch (this->gridQuality) {
        case 3: gausslim = 4.0f; break;  // max quality
        case 2: gausslim = 3.0f; break;  // high quality
        case 1: gausslim = 2.5f; break;  // medium quality
        case 0:
        default: gausslim = 2.0f; break; // low quality
    }

    float *gridDataPos;
    float *gridData;
    unsigned int dataCnt = 0;

    time_t t = clock();

    gridDataPos = new float[dc->GetDipoleCnt()*4]; // TODO Avoid memory allocation in every frame
    gridData    = new float[dc->GetDipoleCnt()*4];
    dataCnt = dc->GetDipoleCnt();

    /*for(int cnt = 0; cnt < static_cast<int>(dc->GetDipoleCnt()); cnt++) {
        Vector<float, 3> vec(&dc->GetDipole()[3*cnt]);
        printf("%i (%f %f %f)\n", cnt, vec[0], vec[1], vec[2]);
    }*/

#pragma omp parallel for
    for(int cnt = 0; cnt < static_cast<int>(dc->GetDipoleCnt()); cnt++) {
        Vector<float, 3> vec(&dc->GetDipole()[3*cnt]);
        gridDataPos[4*cnt+0] = dc->GetDipolePos()[3*cnt+0]-gridOrg[0];
        gridDataPos[4*cnt+1] = dc->GetDipolePos()[3*cnt+1]-gridOrg[1];
        gridDataPos[4*cnt+2] = dc->GetDipolePos()[3*cnt+2]-gridOrg[2];
        gridDataPos[4*cnt+3] = this->gridDataRad;
        gridData[4*cnt+0] = vec[0];
        gridData[4*cnt+1] = vec[1];
        gridData[4*cnt+2] = vec[2];
        gridData[4*cnt+3] = 1.0f;
        //printf("%i (%f %f %f)\n", cnt, vec[0], vec[1], vec[2]);
    }

    // Compute uniform grid (vector field and density map)

    CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf;
    int rc = cqs->calc_map(
            dataCnt,
            &gridDataPos[0],
            &gridData[0],
            true,                // Use seperate 'color' array
            gridOrg.PeekComponents(),
            gridDim.PeekComponents(),
            this->gridDataRad,   // Maximum radius
            1.0f,                // Radius scaling
            this->gridSpacing,
            1.0f,                // Iso value TODO ?
            gausslim);


    if(rc != 0) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s::CalcUniGrid: Quicksurf class returned val != 0\n", this->ClassName());
        this->recalcGrid = false;
        return false;
    }

    // Copy data from device to host

    this->uniGridVecField.MemCpyFromDevice((float3 *)(cqs->getColorMap()));

    // DEBUG
    /*for(int x = 0; x < this->uniGridVecField.GetGridDim().X(); x++) {
        for(int y = 0; y < this->uniGridVecField.GetGridDim().Y(); y++) {
            for(int z = 0; z < this->uniGridVecField.GetGridDim().Z(); z++) {
                //this->uniGridVecField.SetAt(x, y, z, make_float3(1.0, 1.0, 0.0));
                float3 vec = this->uniGridVecField.GetAt(x,y,z);
                printf("(%i %i %i) (%f %f %f)\n", x, y, z, vec.x, vec.y, vec.z);
            }
        }
    }*/

//    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
//            "%s: time for computing uni grid %f",
//            this->ClassName(),
//            (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG

    //  Setup textures


    glEnable(GL_TEXTURE_3D);
    if(!glIsTexture(this->uniGridTex)) {
        glGenTextures(1, &this->uniGridTex);
    }
    glBindTexture(GL_TEXTURE_3D, this->uniGridTex);
    glTexImage3DEXT(GL_TEXTURE_3D,
            0,
            GL_RGB32F,
            gridDim[0],
            gridDim[1],
            gridDim[2],
            0,
            GL_RGB,
            GL_FLOAT,
            this->uniGridVecField.PeekBuffer());
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);

    glDisable(GL_TEXTURE_3D);

    // Check for opengl error
    GLenum err = glGetError();
    if(err != GL_NO_ERROR) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s::CalcUniGrid: glError %s\n",
                this->ClassName(),
                gluErrorString(err));

        delete[] gridDataPos;
        delete[] gridData;
        return false;
    }
    delete[] gridDataPos;
    delete[] gridData;

    //printf("CalcUnigrid done\n"); // DEBUG
#endif

    return true;
}


/*
 * protein_cuda::CrystalStructureRenderer::ApplyPosFilter
 */
void protein_cuda::CrystalStructureVolumeRenderer::ApplyPosFilter(
		const protein_calls::CrystalStructureDataCall *dc) {

    using namespace vislib::sys;

#ifndef SFB_DEMO

    // Calculate atom visibility based on interpolated atom positions
    this->visAtom.SetCount(dc->GetAtomCnt());
#pragma omp parallel for
    for(int cnt = 0; cnt < int(dc->GetAtomCnt()); cnt++) {
        this->visAtom[cnt] = false;
        if((this->posInter[4*cnt+0] > this->posXMax)||
                (this->posInter[4*cnt+0] < this->posXMin)) continue;
        if((this->posInter[4*cnt+1] > this->posYMax)||
                (this->posInter[4*cnt+1] < this->posYMin)) continue;
        if((this->posInter[4*cnt+2] > this->posZMax)||
                (this->posInter[4*cnt+2] < this->posZMin)) continue;

        this->visAtom[cnt] = true;
    }
    /*unsigned int cellIdx = this->showCellIdxParam.Param<core::param::IntParam>()->Value();
    for(int i = 0; i < 15; i++) {
        this->visAtom[dc->GetCells()[15*cellIdx+i]] = true;
    }*/ // DEBUG

    // Setup array with indices of all visible atoms
    this->visAtomIdx.Clear();
    this->visAtomIdx.SetCapacityIncrement(1000);
    for(int cnt = 0; cnt < int(dc->GetAtomCnt()); cnt++) {
        if(this->visAtom[cnt]) {
            this->visAtomIdx.Add(static_cast<int>(cnt));
        }
    }

#endif // SFB_DEMO

    // Calculate dipole visibility based on positions
    this->visDipole.SetCount(dc->GetDipoleCnt());
#pragma omp parallel for
    for(int cnt = 0; cnt < int(dc->GetDipoleCnt()); cnt++) {
        this->visDipole[cnt] = false;
        if((dc->GetDipolePos()[3*cnt+0] > this->posXMax)||
                (dc->GetDipolePos()[3*cnt+0] < this->posXMin)) continue;
        if((dc->GetDipolePos()[3*cnt+1] > this->posYMax)||
                (dc->GetDipolePos()[3*cnt+1] < this->posYMin)) continue;
        if((dc->GetDipolePos()[3*cnt+2] > this->posZMax)||
                (dc->GetDipolePos()[3*cnt+2] < this->posZMin)) continue;

        this->visDipole[cnt] = true;
    }

#ifndef SFB_DEMO

    // Setup array with atom indices of all visible Ba edges
    this->edgeIdxBa.Clear();
    this->edgeIdxTi.Clear();
    this->edgeIdxBa.SetCapacityIncrement(1000);
    this->edgeIdxTi.SetCapacityIncrement(1000);
    for(int cnt = 0; cnt < static_cast<int>(dc->GetAtomCnt()); cnt++) {
        if(this->visAtom[cnt] == false) continue;
        for(int n = 0; n < 6; n++) { // Examine all six potential edges
            if(dc->GetAtomCon()[cnt*6+n] != -1) {
                if(this->visAtom[dc->GetAtomCon()[cnt*6+n]]) {
					if(dc->GetAtomType()[cnt] == protein_calls::CrystalStructureDataCall::BA) {
                        this->edgeIdxBa.Add(cnt);
                        this->edgeIdxBa.Add(dc->GetAtomCon()[cnt*6+n]);
                    }
					if(dc->GetAtomType()[cnt] == protein_calls::CrystalStructureDataCall::TI) {
                        this->edgeIdxTi.Add(cnt);
                        this->edgeIdxTi.Add(dc->GetAtomCon()[cnt*6+n]);
                    }
                }
            }
        }
    }

#endif // SFB_DEMO

    this->recalcArrowData = true;

//    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
//            "%s: done filtering - #atoms %u, #edges %u",
//            this->ClassName(),
//            this->visAtomIdx.Count(),
//            this->edgeIdxTi.Count()+this->edgeIdxBa.Count());
}



/*
 * protein_cuda::CrystalStructureVolumeRenderer::create
 */
bool protein_cuda::CrystalStructureVolumeRenderer::create (void) {
    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    // Init random number generator
    srand((unsigned)time(0));

    // Create quicksurf object
    if(!this->cudaqsurf) {
        this->cudaqsurf = new CUDAQuickSurf();
    }

    // Init extensions
    if(!ogl_IsVersionGEQ(2,0) || !isExtAvailable("GL_EXT_texture3D")
        || !isExtAvailable("GL_EXT_framebuffer_object") || !isExtAvailable("GL_ARB_multitexture")
        || !isExtAvailable("GL_ARB_draw_buffers")) {
        return false;
    }
    if(!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    // Load shader sources
    ShaderSource vertSrc, fragSrc, geomSrc;

    core::CoreInstance *ci = this->GetCoreInstance();
    if(!ci) return false;

    if(!ci->ShaderSourceFactory().MakeShaderSource("scivis::slice::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source", this->ClassName());
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("scivis::slice::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load fragment shader source", this->ClassName() );
        return false;
    }
    try {
        if(!this->vrShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__);
    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    // Load sphere vertex shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().
            MakeShaderSource("protein_cuda::std::sphereVertex", vertSrc)) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for sphere shader",
                this->ClassName());
        return false;
    }
    // Load sphere fragment shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().
            //MakeShaderSource("protein_cuda::std::sphereFragmentFog", fragSrc)) {
            MakeShaderSource("protein_cuda::std::sphereFragment", fragSrc)) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for sphere shader",
                this->ClassName());
        return false;
    }
    try {
        if(!this->sphereShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create sphere shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Load raycasting vertex shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("scivis::raycasting::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the raycasting shader",
                this->ClassName());
        return false;
    }
    // Load raycasting fragment shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("scivis::raycasting::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the raycasting shader", this->ClassName());
        return false;
    }
    try {
        if(!this->rcShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create the raycasting shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Load raycasting vertex shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("scivis::raycasting::vertexDebug", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the raycasting shader",
                this->ClassName());
        return false;
    }
    // Load raycasting fragment shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("scivis::raycasting::fragmentDebug", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the raycasting shader", this->ClassName());
        return false;
    }
    try {
        if(!this->rcShaderDebug.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create the raycasting shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Load alternative arrow shader (uses geometry shader)
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::arrowVertexGeom", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for arrow shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::arrowGeom", geomSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load geometry shader source for arrow shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::arrowFragmentGeom", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for arrow shader");
        return false;
    }
    this->arrowShader.Compile( vertSrc.Code(), vertSrc.Count(), geomSrc.Code(), geomSrc.Count(), fragSrc.Code(), fragSrc.Count());
    this->arrowShader.Link();

    // Load cylinder vertex shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::cylinderVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for cylinder shader");
        return false;
    }
    // Load cylinder fragment shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::cylinderFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for cylinder shader");
        return false;
    }
    try {
        if (!this->cylinderShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create cylinder shader: %s\n", e.GetMsgA());
        return false;
    }

    // Load per pixel lighting shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().
            MakeShaderSource("protein_cuda::std::perpixellightVertex", vertSrc)) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for per pixel lighting",
                this->ClassName());
        return false;
    }

    if(!this->GetCoreInstance()->ShaderSourceFactory().
            MakeShaderSource("protein_cuda::std::perpixellightFragment", fragSrc)) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for per pixel lighting",
                this->ClassName());
        return false;
    }
    try {
        if(!this->pplShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create shader for per pixel lighting: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }



        // TODO
    if(!this->GetCoreInstance()->ShaderSourceFactory().
            MakeShaderSource("scivis::ppl::perpixellightVertex", vertSrc)) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for per pixel lighting",
                this->ClassName());
        return false;
    }

    if(!this->GetCoreInstance()->ShaderSourceFactory().
            MakeShaderSource("scivis::ppl::perpixellightFragment", fragSrc)) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source for per pixel lighting",
                this->ClassName());
        return false;
    }
    try {
        if(!this->pplShaderClip.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create shader for per pixel lighting: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }




    return true;
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::createFbo
 */
bool protein_cuda::CrystalStructureVolumeRenderer::CreateFbo(UINT width, UINT height) {

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
               "%s: (re)creating raycasting fbo.", this->ClassName());

    glEnable(GL_TEXTURE_2D);

    FramebufferObject::ColourAttachParams cap[3];
    cap[0].internalFormat = GL_RGBA32F;
    cap[0].format = GL_RGBA;
    cap[0].type = GL_FLOAT;
    cap[1].internalFormat = GL_RGBA32F;
    cap[1].format = GL_RGBA;
    cap[1].type = GL_FLOAT;
    cap[2].internalFormat = GL_RGBA32F;
    cap[2].format = GL_RGBA;
    cap[2].type = GL_FLOAT;

    FramebufferObject::DepthAttachParams dap;
    dap.format = GL_DEPTH_COMPONENT24;
    dap.state = FramebufferObject::ATTACHMENT_DISABLED;

    FramebufferObject::StencilAttachParams sap;
    sap.format = GL_STENCIL_INDEX;
    sap.state = FramebufferObject::ATTACHMENT_DISABLED;

    return(this->rcFbo.Create(width, height, 3, cap, dap, sap));
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::createSrcFbo
 */
bool protein_cuda::CrystalStructureVolumeRenderer::CreateSrcFbo(size_t width, size_t height) {

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
               "%s: (re)creating source fbo.", this->ClassName());

    glEnable(GL_TEXTURE_2D);

    FramebufferObject::ColourAttachParams cap[3];
    cap[0].internalFormat = GL_RGBA32F;
    cap[0].format = GL_RGBA;
    cap[0].type = GL_FLOAT;

    FramebufferObject::DepthAttachParams dap;
    dap.format = GL_DEPTH_COMPONENT32;
    dap.state = FramebufferObject::ATTACHMENT_TEXTURE;

    FramebufferObject::StencilAttachParams sap;
    sap.format = GL_STENCIL_INDEX;
    sap.state = FramebufferObject::ATTACHMENT_DISABLED;

	return this->srcFbo.Create(static_cast<UINT>(width), static_cast<UINT>(height), 1, cap, dap, sap);
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::FreeBuffs
 */
void protein_cuda::CrystalStructureVolumeRenderer::FreeBuffs() {

    if(this->frame0 != NULL) { delete[] this->frame0; this->frame0 = NULL; }
    if(this->frame1 != NULL){ delete[] this->frame1; this->frame1 = NULL; }

    if(this->gridCurlD != NULL) { checkCudaErrors(cudaFree(this->gridCurlD)); this->gridCurlD = NULL; }
    if(this->gridCurlMagD != NULL) { checkCudaErrors(cudaFree(this->gridCurlMagD)); this->gridCurlMagD = NULL; }
    if(this->mcVertOut_D != NULL) { checkCudaErrors(cudaFree(this->mcVertOut_D)); this->mcVertOut_D = NULL; }
    if(this->mcNormOut_D != NULL) { checkCudaErrors(cudaFree(this->mcNormOut_D)); this->mcNormOut_D = NULL; }

    if(glIsTexture(this->uniGridTex)) glDeleteTextures(1, &this->uniGridTex);
    if(glIsTexture(this->curlMagTex)) glDeleteTextures(1, &this->curlMagTex);
    if(glIsTexture(this->randNoiseTex)) glDeleteTextures(1, &this->randNoiseTex);
    if(glIsTexture(this->uniGridDensityTex)) glDeleteTextures(1, &this->uniGridDensityTex);
    if(glIsTexture(this->uniGridColorTex)) glDeleteTextures(1, &this->uniGridColorTex);

    this->uniGridDensity.Clear();
    this->uniGridVecField.Clear();
    this->uniGridColor.Clear();
    this->licRandBuff.Clear();
    this->uniGridCurlMag.Clear();

}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::FilterVecField
 */
void protein_cuda::CrystalStructureVolumeRenderer::FilterVecField(
		const protein_calls::CrystalStructureDataCall *dc,
        const float *atomPos) {

    using namespace vislib::math;

    Vector<float, 3> gridMinCoord, gridMaxCoord, gridXAxis, gridYAxis,
                     gridZAxis, gridOrg;
    Vector<int, 3> gridDim;

    // Init uniform grid
    gridMinCoord[0] = this->bbox.ObjectSpaceBBox().Left();
    gridMinCoord[1] = this->bbox.ObjectSpaceBBox().Bottom();
    gridMinCoord[2] = this->bbox.ObjectSpaceBBox().Back();
    gridMaxCoord[0] = this->bbox.ObjectSpaceBBox().Right();
    gridMaxCoord[1] = this->bbox.ObjectSpaceBBox().Top();
    gridMaxCoord[2] = this->bbox.ObjectSpaceBBox().Front();
    gridXAxis[0] = gridMaxCoord[0]-gridMinCoord[0];
    gridYAxis[1] = gridMaxCoord[1]-gridMinCoord[1];
    gridZAxis[2] = gridMaxCoord[2]-gridMinCoord[2];
    gridDim[0] = (int) ceil(gridXAxis[0] / this->gridSpacing);
    gridDim[1] = (int) ceil(gridYAxis[1] / this->gridSpacing);
    gridDim[2] = (int) ceil(gridZAxis[2] / this->gridSpacing);
    gridXAxis[0] = (gridDim[0]-1) * this->gridSpacing;
    gridYAxis[1] = (gridDim[1]-1) * this->gridSpacing;
    gridZAxis[2] = (gridDim[2]-1) * this->gridSpacing;
    gridMaxCoord[0] = gridMinCoord[0] + gridXAxis[0];
    gridMaxCoord[1] = gridMinCoord[1] + gridYAxis[1];
    gridMaxCoord[2] = gridMinCoord[2] + gridZAxis[2];
    gridOrg[0] = gridMinCoord[0];
    gridOrg[1] = gridMinCoord[1];
    gridOrg[2] = gridMinCoord[2];

    //if(this->filterVecField) {

        // Keep track of min/max magnitudes (necessary for color gradient)
        this->vecMagSclMax = -100.0f;
        this->vecMagSclMin = 100.0f;
        float min=1000.0, max=0.0;
        for(int cnt = 0; cnt < int(dc->GetDipoleCnt()); ++cnt) {
            Vector<float, 3> vec(&dc->GetDipole()[cnt*3]);
            if(vec.Length()*this->vecScl > this->vecMagSclMax) {
                this->vecMagSclMax = vec.Length()*this->vecScl;
            }
            if(vec.Length()*this->vecScl < this->vecMagSclMin) {
                this->vecMagSclMin = vec.Length()*this->vecScl;
            }

            if(vec.Length() > max) {
                max = vec.Length();
            }
            if(vec.Length() < min) {
                min = vec.Length();
            }
        }
        this->arrowVis.SetCount(dc->GetDipoleCnt());

       // printf("Vector min: %f max: %f\n", min, max); // DEBUG

#pragma omp parallel for
        for(int cnt = 0; cnt < int(dc->GetDipoleCnt()); ++cnt) {
            Vector<float, 3> vec(&dc->GetDipole()[cnt*3]);
            this->arrowVis[cnt] = true;

#if(FILTER_BOUNDARY)
            // Filter boundary areas
            if(dc->GetDipolePos()[3*cnt+1] < 4.0f) {
                this->arrowVis[cnt] = false;
                continue;
            }
            if((dc->GetDipolePos()[3*cnt+1] > 34.0f)&&(dc->GetDipolePos()[3*cnt+1] < 42.0f)) {
                this->arrowVis[cnt] = false;
                continue;
            }
            if(dc->GetDipolePos()[3*cnt+1] > 73.0f) {
                this->arrowVis[cnt] = false;
                continue;
            }
            if((dc->GetDipolePos()[3*cnt+1] > 71.0f)&&(dc->GetDipolePos()[3*cnt+2] < 58.0f)) {
                this->arrowVis[cnt] = false;
                continue;
            }
            if((dc->GetDipolePos()[3*cnt+2] > 60.0f)&&(dc->GetDipolePos()[3*cnt+2] < 65.0f)) {
                this->arrowVis[cnt] = false;
                continue;
            }
            if(dc->GetDipolePos()[3*cnt+2] < 3.0f) {
                this->arrowVis[cnt] = false;
                continue;
            }
#endif

            // Apply length filter
            if(vec.Length()*this->vecScl < this->minVecMag/100.0f) {
                this->arrowVis[cnt] = false;
                continue;
            }
            if(vec.Length()*this->vecScl > this->maxVecMag/100.0f) {
                this->arrowVis[cnt] = false;
                continue;
            }

#if (defined(CALC_GRID) && (CALC_GRID))

            // Apply curl filter
            float curl = this->uniGridCurlMag.SampleNearest(
                    dc->GetDipolePos()[3*cnt+0],
                    dc->GetDipolePos()[3*cnt+1],
                    dc->GetDipolePos()[3*cnt+2]);
            if(curl > this->maxVecCurl) {
                this->arrowVis[cnt] = false;
                continue;
            }
#endif

            if(dc->GetDipoleCnt() == 625000) { // Filter by atom type
				if ((!this->showBaAtoms) && (dc->GetAtomType()[cnt] == protein_calls::CrystalStructureDataCall::BA)) {
                    this->arrowVis[cnt] = false;
                    continue;
            	}
				if ((!this->showOAtoms) && (dc->GetAtomType()[cnt] == protein_calls::CrystalStructureDataCall::O)) {
                    this->arrowVis[cnt] = false;
                    continue;
            	}
				if ((!this->showTiAtoms) && (dc->GetAtomType()[cnt] == protein_calls::CrystalStructureDataCall::TI)) {
                    this->arrowVis[cnt] = false;
                    continue;
            	}
            }
        }

      //  this->filterVecField = false;
    //}
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::GetExtents
 */
bool protein_cuda::CrystalStructureVolumeRenderer::GetExtents(core::Call& call) {

    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

	protein_calls::CrystalStructureDataCall *dc =
			this->dataCallerSlot.CallAs<protein_calls::CrystalStructureDataCall>();
    if(dc == NULL) return false;
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) return false;

    float scale;
    if(!vislib::math::IsEqual(dc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / dc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    cr3d->AccessBoundingBoxes() = dc->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
    cr3d->SetTimeFramesCount(dc->FrameCount());
    this->bbox = dc->AccessBoundingBoxes();

    /*cr3d->AccessBoundingBoxes() = dc->AccessBoundingBoxes();
    cr3d->SetTimeFramesCount(dc->FrameCount());
    this->bbox = dc->AccessBoundingBoxes();*/ // No scaling

    return true;
}


/*
 *  protein_cuda::CrystalStructureVolumeRenderer::initLIC
 */
bool protein_cuda::CrystalStructureVolumeRenderer::InitLIC() {
    using namespace vislib::sys;

    // Create randbuffer
    vislib::math::Vector<unsigned int, 3> buffDim(this->licRandBuffSize,
            this->licRandBuffSize, this->licRandBuffSize);
    vislib::math::Vector<float, 3> buffOrg(0.0f, 0.0f, 0.0f);
    this->licRandBuff.Init(buffDim, buffOrg, 1.0f);
    for(int x = 0; x < this->licRandBuffSize; x++) {
        for(int y = 0; y < this->licRandBuffSize; y++) {
            for(int z = 0; z < this->licRandBuffSize; z++) {
                float randVal = (float)rand()/float(RAND_MAX);
                /*if(randVal > 0.5f)
                    this->licRandBuff.SetAt(x, y, z, 1.0f);
                else
                    this->licRandBuff.SetAt(x, y, z, 0.0f);*/
                this->licRandBuff.SetAt(x, y, z, randVal);
            }
        }
    }

    // Setup random noise texture
    glEnable(GL_TEXTURE_3D);
    if(glIsTexture(this->randNoiseTex)) glDeleteTextures(1, &this->randNoiseTex);
    glGenTextures(1, &this->randNoiseTex);
    glBindTexture(GL_TEXTURE_3D, this->randNoiseTex);

    glTexImage3DEXT(GL_TEXTURE_3D,
            0,
            GL_ALPHA,
            this->licRandBuffSize,
            this->licRandBuffSize,
            this->licRandBuffSize,
            0,
            GL_ALPHA,
            GL_FLOAT,
            this->licRandBuff.PeekBuffer());

    //glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    //glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glBindTexture(GL_TEXTURE_3D, 0);
    glDisable(GL_TEXTURE_3D);

    // Check for opengl error
    GLenum err = glGetError();
    if(err != GL_NO_ERROR) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s::InitLIC: glError %s\n", this->ClassName(), gluErrorString(err));
        return false;
    }

    return true;
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::release
 */
void protein_cuda::CrystalStructureVolumeRenderer::release(void) {
    this->FreeBuffs();
    this->vrShader.Release();
    this->sphereShader.Release();
    this->arrowShader.Release();
    this->cylinderShader.Release();
    this->rcShader.Release();
    this->rcShaderDebug.Release();
    this->pplShader.Release();
    if(this->cudaqsurf != NULL) {
        CUDAQuickSurf *cqs = (CUDAQuickSurf *)this->cudaqsurf;
        delete cqs;
    }
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::Render
 */
bool protein_cuda::CrystalStructureVolumeRenderer::Render(core::Call& call) {
    using namespace vislib::sys;
    using namespace vislib::math;

    GLenum err;

    core::view::AbstractCallRender3D *cr3d =
            dynamic_cast<core::view::AbstractCallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }
    
#ifdef _WIN32
    if( setCUDAGLDevice ) {
        if( cr3d->IsGpuAffinity() ) {
            HGPUNV gpuId = cr3d->GpuAffinity<HGPUNV>();
            int devId;
            cudaWGLGetDevice( &devId, gpuId);
            cudaGLSetGLDevice( devId);
            printf( "cudaGLSetGLDevice: %s\n", cudaGetErrorString( cudaGetLastError()));
        }
        setCUDAGLDevice = false;
    }
#endif

	protein_calls::CrystalStructureDataCall *dc =
			this->dataCallerSlot.CallAs<protein_calls::CrystalStructureDataCall>();
    if (dc == NULL) {
        return false;
    }

    // Update parameters if necessary
    if(!this->UpdateParams(dc)) {
        Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "%s: Unable to update parameters",
                this->ClassName());
        return false;
    }

    // Check button params
    if (this->toggleIsoSurfaceSlot.IsDirty()) {
        this->volShow = !this->volShow;
        this->volShowParam.Param<core::param::BoolParam>()->SetValue( this->volShow);
        this->toggleIsoSurfaceSlot.ResetDirty();
    }
    if (this->toggleCurlFilterSlot.IsDirty()) {
        this->filterVecField = !this->filterVecField;
        this->arrowUseFilterParam.Param<core::param::BoolParam>()->SetValue( this->filterVecField);
        this->toggleCurlFilterSlot.ResetDirty();
    }

    // Get camera information
    this->cameraInfo =  dynamic_cast<core::view::CallRender3D*>(&call)->GetCameraParameters();

    float callTime = cr3d->Time();
    dc->SetCalltime(callTime);                             // Set call time
    dc->SetFrameID(static_cast<int>(callTime), true);      // Set frame ID and force flag

    // Don't remove this !!
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        return false;
    }

#ifndef SFB_DEMO
    if(this->callTimeOld != callTime) {
        this->recalcGrid = true;     // Recalc displacement if position changed
        this->recalcPosInter = true;
    }
#else
    if(this->frameOld != dc->FrameID()) {
        this->recalcGrid = true;     // Recalc displacement if position changed
        this->recalcPosInter = true;
    }
#endif
    this->callTimeOld = callTime;

    // Current frame
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
        return false;
    }

#ifndef SFB_DEMO
    // Copy frame data and interpolate positions
    if(this->idxLastFrame != static_cast<int>(callTime)) {

        // Allocate memory if necessary
        if(this->frame0 == NULL) {
            this->frame0 = new float[dc->GetAtomCnt()*3];
        }
        if(this->frame1 == NULL) {
            this->frame1 = new float[dc->GetAtomCnt()*3];
        }

        memcpy(this->frame0, dc->GetAtomPos(), dc->GetAtomCnt()*3*sizeof(float));

        // Get next frame if interpolation is enabled (otherwise use the same
        // frame twice)
        if(this->interpol) {
            if ((static_cast<int>(callTime)+1) < dc->FrameCount()) {
                dc->SetFrameID(static_cast<int>(callTime)+1, true);
                dc->SetCalltime(static_cast<float>(dc->FrameID()));
            }
            // Current frame
			if(!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
                return false;
            }
            memcpy(this->frame1, dc->GetAtomPos(), dc->GetAtomCnt()*3*sizeof(float));
        }
        else {
            memcpy(this->frame1, this->frame0, dc->GetAtomCnt()*3*sizeof(float));
        }
        dc->Unlock();

        this->recalcArrowData = true;
        this->recalcGrid = true;
        this->recalcCurlMag = true;
    }
    this->idxLastFrame = static_cast<int>(cr3d->Time());

    if(this->recalcPosInter) {
        // Interpolate atom positions between frames
        this->posInter.SetCount(dc->GetAtomCnt()*4);
        float inter = callTime - static_cast<float>(static_cast<int>(callTime));
        float threshold = vislib::math::Min(dc->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
                            vislib::math::Min(dc->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
                                    dc->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
        for(int cnt = 0; cnt < int(dc->GetAtomCnt()); ++cnt) {
            if(std::sqrt(std::pow(this->frame1[3*cnt+0] - this->frame0[3*cnt+0], 2) +
                         std::pow(this->frame1[3*cnt+1] - this->frame0[3*cnt+1], 2) +
                         std::pow(this->frame1[3*cnt+2] - this->frame0[3*cnt+2], 2)) < threshold ) {

                this->posInter[4*cnt+0] = (1.0f - inter) * this->frame0[3*cnt+0] + inter * this->frame1[3*cnt+0];
                this->posInter[4*cnt+1] = (1.0f - inter) * this->frame0[3*cnt+1] + inter * this->frame1[3*cnt+1];
                this->posInter[4*cnt+2] = (1.0f - inter) * this->frame0[3*cnt+2] + inter * this->frame1[3*cnt+2];

            } else if(inter < 0.5f) {
                this->posInter[4*cnt+0] = this->frame0[3*cnt+0];
                this->posInter[4*cnt+1] = this->frame0[3*cnt+1];
                this->posInter[4*cnt+2] = this->frame0[3*cnt+2];
            } else {
                this->posInter[4*cnt+0] = this->frame1[3*cnt+0];
                this->posInter[4*cnt+1] = this->frame1[3*cnt+1];
                this->posInter[4*cnt+2] = this->frame1[3*cnt+2];
            }

            // Van der waals radii
			if(dc->GetAtomType()[cnt] == protein_calls::CrystalStructureDataCall::BA)
                this->posInter[4*cnt+3] = 2.68f*this->sphereRad;
			else if(dc->GetAtomType()[cnt] == protein_calls::CrystalStructureDataCall::O)
                this->posInter[4*cnt+3] = 1.20f*this->sphereRad;
			else if(dc->GetAtomType()[cnt] == protein_calls::CrystalStructureDataCall::TI)
                this->posInter[4*cnt+3] = 1.47f*this->sphereRad;
        }
        this->recalcPosInter = false;
        this->recalcDipole = true;
    }

#endif // SFB_DEMO
    // Setup atom color array
    if(this->atomColor.Count() != dc->GetAtomCnt()*3) {
        this->SetupAtomColors(dc); // TODO write radii also here
    }


    // (Re)calculate visibility based on interpolated atom positions
    if(this->recalcVisibility) {
        this->ApplyPosFilter(dc);
        this->recalcVisibility = false;
    }


    if(this->recalcGrid) {
        if(!this->CalcUniGrid(dc, this->posInter.PeekElements(), NULL)) {
            return false;
        }
        this->recalcGrid = false;
        this->recalcCritPoints = true;
        this->filterVecField = true;
        this->recalcCurlMag = true;
    }

    // (Re)calc curl
    if(this->recalcCurlMag) {
        if(!this->CalcMagCurlTex()) {
            return false;
        }

        this->recalcCurlMag = false;
        this->filterVecField = true;
        this->recalcDensityGrid = true;
    }

    // Apply filter to vector field
    this->FilterVecField(dc, this->posInter.PeekElements());

    // (Re)calc density tex
    if(!this->CalcDensityTex(dc, this->posInter.PeekElements())) {
        return false;
    }

    // (Re)create random texture for LIC
    if(this->licRandBuff.GetGridDim().X()*
            this->licRandBuff.GetGridDim().Y()*
            this->licRandBuff.GetGridDim().Z() == 0) {

        if(!this->InitLIC()) {
            Log::DefaultLog.WriteMsg(
                    Log::LEVEL_ERROR, "%s: Unable to setup random texture",
                    this->ClassName());
            return false;
        }
    }

    // Get current viewport and recreate fbo if necessary
    float curVP[4];
    glGetFloatv(GL_VIEWPORT, curVP);
    if((curVP[2] != this->srcFboDim.X()) || (curVP[3] != srcFboDim.Y())) {
        this->srcFboDim.SetX(static_cast<int>(curVP[2]));
        this->srcFboDim.SetY(static_cast<int>(curVP[3]));
		if (!this->CreateSrcFbo(static_cast<size_t>(curVP[2]), static_cast<size_t>(curVP[3]))) return false;
    }

    this->srcFbo.Enable(0);



    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render cell based data

    // compute scale factor and scale world
    float scale;
    if(!vislib::math::IsEqual(dc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / dc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef( scale, scale, scale);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

#ifndef SFB_DEMO
    // Render atoms
    if(this->atomRM == ATOM_SPHERES) {
        this->RenderAtomsSpheres(dc);
    }


    // Render ba edges
    if(this->edgeBaRM == BA_EDGE_STICK) {
        this->RenderEdgesBaStick(dc, this->posInter.PeekElements(), NULL);
    }
    else if(this->edgeBaRM == BA_EDGE_LINES) {
        this->RenderEdgesBaLines(dc, this->posInter.PeekElements(), this->atomColor.PeekElements());
    }

    // Render ti edges
    if(this->edgeTiRM == TI_EDGE_STICK) {
        this->RenderEdgesTiStick(dc, this->posInter.PeekElements(), this->atomColor.PeekElements());
    }
    else if(this->edgeTiRM == TI_EDGE_LINES) {
        this->RenderEdgesTiLines(dc, this->posInter.PeekElements(), this->atomColor.PeekElements());
    }

#endif // SFB_DEMO

    // Render arrow glyphs representing the vector field
    if(this->vecRM == VEC_ARROWS) {
        if(!this->RenderVecFieldArrows(dc, this->posInter.PeekElements(), this->atomColor.PeekElements())) {
            Log::DefaultLog.WriteMsg(
                    Log::LEVEL_ERROR, "%s: Unable to render arrow glyphs",
                    this->ClassName());
        }
    }



    Vector<float, 3> gridMinCoord, gridMaxCoord, gridXAxis,gridYAxis, gridZAxis;
    Vector<int, 3> gridDim;
    gridMinCoord[0] = this->bbox.ObjectSpaceBBox().Left();
    gridMinCoord[1] = this->bbox.ObjectSpaceBBox().Bottom();
    gridMinCoord[2] = this->bbox.ObjectSpaceBBox().Back();
    gridMaxCoord[0] = this->bbox.ObjectSpaceBBox().Right();
    gridMaxCoord[1] = this->bbox.ObjectSpaceBBox().Top();
    gridMaxCoord[2] = this->bbox.ObjectSpaceBBox().Front();
    gridXAxis[0] = gridMaxCoord[0]-gridMinCoord[0];
    gridYAxis[1] = gridMaxCoord[1]-gridMinCoord[1];
    gridZAxis[2] = gridMaxCoord[2]-gridMinCoord[2];
    gridDim[0] = (int) ceil(gridXAxis[0] / this->gridSpacing);
    gridDim[1] = (int) ceil(gridYAxis[1] / this->gridSpacing);
    gridDim[2] = (int) ceil(gridZAxis[2] / this->gridSpacing);
    gridXAxis[0] = (gridDim[0]-1) * this->gridSpacing;
    gridYAxis[1] = (gridDim[1]-1) * this->gridSpacing;
    gridZAxis[2] = (gridDim[2]-1) * this->gridSpacing;
    gridMaxCoord[0] = gridMinCoord[0] + gridXAxis[0];
    gridMaxCoord[1] = gridMinCoord[1] + gridYAxis[1];
    gridMaxCoord[2] = gridMinCoord[2] + gridZAxis[2];

#ifndef SFB_DEMO
    ////////////////////////////////////////////////////////////////////////////
    if(this->showRidge) {

        // TODO VTK mesh rendering ...
        this->pplShaderClip.Enable();

        glUniform1fARB(this->pplShaderClip.ParameterLocation("posXMax"), this->posXMax);
        glUniform1fARB(this->pplShaderClip.ParameterLocation("posYMax"), this->posYMax);
        glUniform1fARB(this->pplShaderClip.ParameterLocation("posZMax"), this->posZMax);
        glUniform1fARB(this->pplShaderClip.ParameterLocation("posXMin"), this->posXMin);
        glUniform1fARB(this->pplShaderClip.ParameterLocation("posYMin"), this->posYMin);
        glUniform1fARB(this->pplShaderClip.ParameterLocation("posZMin"), this->posZMin);
        glUniform1iARB(this->pplShaderClip.ParameterLocation("curlMagTex"), 0);

        glEnable(GL_TEXTURE_3D);
        glActiveTextureARB(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, this->curlMagTex);

        glDisable(GL_CULL_FACE);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glColor4f(1.0, 1.0, 1.0, 1.0);
        if( this->renderMesh ) {
            glColor3f( 1.0f, 0.75f, 0.0f);
            glBegin(GL_TRIANGLES);
            for (int p=0; p< static_cast<int>(meshFaces.Count()) / 3; p++) {
                for (int e=0; e < 3; e++) {

                    // Calc tex coords
                    float texCoordX = (meshVertices[meshFaces[p*3+e]*3+0] - gridMinCoord[0])/(gridMaxCoord[0] - gridMinCoord[0]);
                    float texCoordY = (meshVertices[meshFaces[p*3+e]*3+1] - gridMinCoord[1])/(gridMaxCoord[1] - gridMinCoord[1]);
                    float texCoordZ = (meshVertices[meshFaces[p*3+e]*3+2] - gridMinCoord[2])/(gridMaxCoord[2] - gridMinCoord[2]);

                    glTexCoord3f(texCoordX, texCoordY, texCoordZ);
                    glNormal3f(meshNormals[meshFaces[p*3+e]*3+0],
                            meshNormals[meshFaces[p*3+e]*3+1],
                            meshNormals[meshFaces[p*3+e]*3+2]);
                    glVertex3f(meshVertices[meshFaces[p*3+e]*3+0],
                            meshVertices[meshFaces[p*3+e]*3+1],
                            meshVertices[meshFaces[p*3+e]*3+2]);
                }
            }
            glEnd();
        }
        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_3D);
        // ... VTK mesh rendering
        this->pplShaderClip.Disable();
    }
    ///////////////////////////////////////////////////////////////////////////

    // Compute/render critical points
    if(this->showCritPoints) {
        this->RenderCritPointsSpheres(dc);
    }

    if(this->showIsoSurf) {
        this->RenderIsoSurfMC();
    }
#endif // SFB_DEMO

    // Render slices

    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glColor4f(1.0, 1.0, 1.0, 1.0);

    // Slice rendering
    if(this->sliceRM != SLICE_NONE) {

        // Calc tex coords for planes
        float texCoordX = (this->xPlane - gridMinCoord[0])/(gridMaxCoord[0] - gridMinCoord[0]);
        float texCoordY = (this->yPlane - gridMinCoord[1])/(gridMaxCoord[1] - gridMinCoord[1]);
        float texCoordZ = (this->zPlane - gridMinCoord[2])/(gridMaxCoord[2] - gridMinCoord[2]);

        this->vrShader.Enable();
        glUniform1iARB(this->vrShader.ParameterLocation("uniGridTex"), 0);
        glUniform1iARB(this->vrShader.ParameterLocation("curlMagTex"), 1);
        glUniform1iARB(this->vrShader.ParameterLocation("randNoiseTex"), 2);
        glUniform1iARB(this->vrShader.ParameterLocation("densityMapTex"), 3);
        glUniform1iARB(this->vrShader.ParameterLocation("colorTex"), 4);
        glUniform1iARB(this->vrShader.ParameterLocation("sliceRM"), static_cast<int>(this->sliceRM));
        glUniform1fARB(this->vrShader.ParameterLocation("licDirScl"), this->licDirScl);
        glUniform1iARB(this->vrShader.ParameterLocation("licLen"), this->licStreamlineLength);
        glUniform1fARB(this->vrShader.ParameterLocation("dataScl"), this->sliceDataScl);
        glUniform1fARB(this->vrShader.ParameterLocation("licContrast"), this->licContrastStretching);
        glUniform1fARB(this->vrShader.ParameterLocation("licBrightness"), this->licBright);
        glUniform1fARB(this->vrShader.ParameterLocation("licTCScl"), this->licTCScl);
        if(this->projectVec2D)
            glUniform1iARB(this->vrShader.ParameterLocation("licProj2D"), 1);
        else
            glUniform1iARB(this->vrShader.ParameterLocation("licProj2D"), 0);


        glEnable(GL_TEXTURE_3D);
        glEnable(GL_TEXTURE_2D);

        glActiveTextureARB(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_3D, this->randNoiseTex);

        glActiveTextureARB(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, this->curlMagTex);

        glActiveTextureARB(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_3D, this->uniGridDensityTex);

        glActiveTextureARB(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_3D, this->uniGridColorTex);

        glActiveTextureARB(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, this->uniGridTex);

        if(this->showXPlane) { // Render x plane
            glUniform1iARB(this->vrShader.ParameterLocation("plane"), 0);
            glBegin(GL_QUADS);

            glMultiTexCoord3fARB(GL_TEXTURE0, texCoordX, 1.0f, 0.0f);
            glMultiTexCoord2fARB(GL_TEXTURE1, 1.0f, 0.0f);
            glVertex3f(this->xPlane, gridMaxCoord[1], gridMinCoord[2]);

            glMultiTexCoord3fARB(GL_TEXTURE0, texCoordX, 0.0f, 0.0f);
            glMultiTexCoord2fARB(GL_TEXTURE1, 0.0f, 0.0f);
            glVertex3f(this->xPlane, gridMinCoord[1], gridMinCoord[2]);

            glMultiTexCoord3fARB(GL_TEXTURE0, texCoordX, 0.0f, 1.0f);
            glMultiTexCoord2fARB(GL_TEXTURE1, 0.0f, 1.0f);
            glVertex3f(this->xPlane, gridMinCoord[1], gridMaxCoord[2]);

            glMultiTexCoord3fARB(GL_TEXTURE0, texCoordX, 1.0f, 1.0f);
            glMultiTexCoord2fARB(GL_TEXTURE1, 1.0f, 1.0f);
            glVertex3f(this->xPlane, gridMaxCoord[1], gridMaxCoord[2]);

            glEnd();

        }
        if(this->showYPlane) { // Render y plane
            glUniform1iARB(this->vrShader.ParameterLocation("plane"), 1);
            glBegin(GL_QUADS);

            glMultiTexCoord3fARB(GL_TEXTURE0,0.0f, texCoordY, 1.0f);
            glMultiTexCoord2fARB(GL_TEXTURE1, 0.0f, 1.0f);
            glVertex3f(gridMinCoord[0], this->yPlane, gridMaxCoord[2]);

            glMultiTexCoord3fARB(GL_TEXTURE0, 0.0f, texCoordY, 0.0f);
            glMultiTexCoord2fARB(GL_TEXTURE1, 0.0f, 0.0f);
            glVertex3f(gridMinCoord[0], this->yPlane, gridMinCoord[2]);

            glMultiTexCoord3fARB(GL_TEXTURE0, 1.0f, texCoordY, 0.0f);
            glMultiTexCoord2fARB(GL_TEXTURE1, 1.0f, 0.0f);
            glVertex3f(gridMaxCoord[0], this->yPlane, gridMinCoord[2]);

            glMultiTexCoord3fARB(GL_TEXTURE0, 1.0f, texCoordY, 1.0f);
            glMultiTexCoord2fARB(GL_TEXTURE1, 1.0f, 1.0f);
            glVertex3f(gridMaxCoord[0], this->yPlane, gridMaxCoord[2]);

            glEnd();
        }
        if(this->showZPlane) { // Render z plane
            glUniform1iARB(this->vrShader.ParameterLocation("plane"), 2);
            glBegin(GL_QUADS);

            glMultiTexCoord3fARB(GL_TEXTURE0, 0.0f, 1.0f, texCoordZ);
            glMultiTexCoord2fARB(GL_TEXTURE1, 0.0f, 1.0f);
            glVertex3f(gridMinCoord[0], gridMaxCoord[1], this->zPlane);

            glMultiTexCoord3fARB(GL_TEXTURE0, 0.0f, 0.0f, texCoordZ);
            glMultiTexCoord2fARB(GL_TEXTURE1, 0.0f, 0.0f);
            glVertex3f(gridMinCoord[0], gridMinCoord[1], this->zPlane);

            glMultiTexCoord3fARB(GL_TEXTURE0, 1.0f, 0.0f, texCoordZ);
            glMultiTexCoord2fARB(GL_TEXTURE1, 1.0f, 0.0f);
            glVertex3f(gridMaxCoord[0], gridMinCoord[1], this->zPlane);

            glMultiTexCoord3fARB(GL_TEXTURE0, 1.0f, 1.0f, texCoordZ);
            glMultiTexCoord2fARB(GL_TEXTURE1, 1.0f, 1.0f);
            glVertex3f(gridMaxCoord[0], gridMaxCoord[1], this->zPlane);

            glEnd();
        }
        glActiveTextureARB(GL_TEXTURE0);
        glDisable(GL_TEXTURE_3D);
        glDisable(GL_TEXTURE_2D);
        this->vrShader.Disable();
    }

    // Disable rendering to framebuffer
    this->srcFbo.Disable();

    // Render screen quad for opaque frame buffer content
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);

    //glBindTexture(GL_TEXTURE_2D, this->srcColorBuff);
    this->srcFbo.BindColourTexture(0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1, 0); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(1, 1); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0, 1); glVertex2f(-1.0f, 1.0f);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);

    // Render volume
    // Note: uses depth and color buffer the rest of the scene has been rendered to
    if(this->volShow) {
        if(!this->RenderVolume()) {
            Log::DefaultLog.WriteMsg(
                    Log::LEVEL_ERROR, "%s: Unable to render volume\n",
                    this->ClassName());
            return false;
        }
    }

    // Check for opengl error
    err = glGetError();
    if(err != GL_NO_ERROR) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s::Render: glError %s\n", this->ClassName(),
                gluErrorString(err));
        return false;
    }

    dc->Unlock();

    this->frameOld = dc->FrameID();

    return true;
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::RenderVecFieldArrows
 */
bool protein_cuda::CrystalStructureVolumeRenderer::RenderVecFieldArrows(
		const protein_calls::CrystalStructureDataCall *dc,
        const float *atomPos,
        const float *col) {

    using namespace vislib::math;



    // Check whether filter has been applied
    if(this->arrowVis.Count() != dc->GetDipoleCnt()) return false;

    // Setup necessary data arrays and apply filtering

    this->arrowData.SetCount(dc->GetDipoleCnt()*4);
    this->arrowDataPos.SetCount(dc->GetDipoleCnt()*3);
    this->arrowCol.SetCount(dc->GetDipoleCnt()*3);

#pragma omp parallel for
    for(int cnt = 0; cnt < int(dc->GetDipoleCnt()); ++cnt) {

        Vector<float, 3> vec(&dc->GetDipole()[3*cnt]);

        if(this->arrColorMode == ARRCOL_ELEMENT) {
            // Set color based on atom color array
            this->arrowCol[cnt*3+0] = this->atomColor.PeekElements()[3*cnt+0];
            this->arrowCol[cnt*3+1] = this->atomColor.PeekElements()[3*cnt+1];
            this->arrowCol[cnt*3+2] = this->atomColor.PeekElements()[3*cnt+2];
        }
        else if(this->arrColorMode == ARRCOL_ORIENT){
            float len = vec.Length();
            // Set color based on orientation
            this->arrowCol[cnt*3+0] = (vec.X()/len+1.0f)*0.5f;
            this->arrowCol[cnt*3+1] = (vec.Y()/len+1.0f)*0.5f;
            this->arrowCol[cnt*3+2] = (vec.Z()/len+1.0f)*0.5f;
        }
        else if(this->arrColorMode == ARRCOL_MAGNITUDE) {
            // Use gradient color based on magnitude
            Vector<float, 3> color = this->getColorGrad(vec.Length()*this->vecScl);
            this->arrowCol[cnt*3+0] = color.X();
            this->arrowCol[cnt*3+1] = color.Y();
            this->arrowCol[cnt*3+2] = color.Z();
        }
        else { // Invalid color mode
            this->arrowCol[cnt*3+0] = 0.0f;
            this->arrowCol[cnt*3+1] = 1.0f;
            this->arrowCol[cnt*3+2] = 1.0f;
        }

        //if((this->arrowVis[cnt])&&(this->visDipole[cnt])) {
        //    printf("Vec #%i (%f %f %f)\n", cnt, vec.X(), vec.Y(), vec.Z()); 
        //} // DEBUG

        if(this->toggleNormVec) {
            vec.Normalise();
        }

        // Setup vector
        this->arrowData[4*cnt+0] = dc->GetDipolePos()[cnt*3+0] + vec.X()*this->vecScl*0.5f;
        this->arrowData[4*cnt+1] = dc->GetDipolePos()[cnt*3+1] + vec.Y()*this->vecScl*0.5f;
        this->arrowData[4*cnt+2] = dc->GetDipolePos()[cnt*3+2] + vec.Z()*this->vecScl*0.5f;
        this->arrowData[4*cnt+3] = 1.0f;

        // Set position of the arrow
        this->arrowDataPos[3*cnt+0] = dc->GetDipolePos()[cnt*3+0] - vec.X()*this->vecScl*0.5f;
        this->arrowDataPos[3*cnt+1] = dc->GetDipolePos()[cnt*3+1] - vec.Y()*this->vecScl*0.5f;
        this->arrowDataPos[3*cnt+2] = dc->GetDipolePos()[cnt*3+2] - vec.Z()*this->vecScl*0.5f;
    }


    // Write idx array
    this->arrowVisIdx.Clear();
    this->arrowVisIdx.SetCapacityIncrement(1000);
    if(this->arrowUseFilter) {
        for(int cnt = 0; cnt < int(dc->GetDipoleCnt()); cnt++) {
            if((this->arrowVis[cnt])&&(this->visDipole[cnt])) {
                this->arrowVisIdx.Add(static_cast<int>(cnt));
            }
        }
    }
    else {
        for(int cnt = 0; cnt < int(dc->GetDipoleCnt()); cnt++) {
            if(this->visDipole[cnt]) { // Only positional filtering
                this->arrowVisIdx.Add(static_cast<int>(cnt));
            }
        }
    }
    // printf("Number of visible arrows %u\n", this->arrowVisIdx.Count());

    // Actual rendering

    float viewportStuff[4] = {
            cameraInfo->TileRect().Left(),
            cameraInfo->TileRect().Bottom(),
            cameraInfo->TileRect().Width(),
            cameraInfo->TileRect().Height()
    };

    if(viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if(viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable(GL_BLEND);

    /*this->arrowShader.Enable();

    glUniform4fvARB(this->arrowShader.ParameterLocation("viewAttr"), 1, viewportStuff );
    glUniform3fvARB(this->arrowShader.ParameterLocation("camIn"), 1, this->cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->arrowShader.ParameterLocation("camRight"), 1, this->cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->arrowShader.ParameterLocation("camUp"), 1, this->cameraInfo->Up().PeekComponents());
    glUniform1fARB(this->arrowShader.ParameterLocation("radScale"), this->arrowRad);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glTexCoordPointer(3, GL_FLOAT, 0, this->arrowDataPos.PeekElements());
    glVertexPointer(4, GL_FLOAT, 0, this->arrowData.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, this->arrowCol.PeekElements());

    glDrawElements(GL_POINTS, static_cast<GLsizei>(this->arrowVisIdx.Count()),
        GL_UNSIGNED_INT, this->arrowVisIdx.PeekElements());

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    this->arrowShader.Disable();*/


    using namespace vislib;
    using namespace vislib::math;

    // TODO: Make these class members and retrieve only once per frame
    // Get GL_MODELVIEW matrix
    GLfloat modelMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelMatrix_column);
    Matrix<GLfloat, 4, COLUMN_MAJOR> modelMatrix(&modelMatrix_column[0]);
    // Get GL_PROJECTION matrix
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    Matrix<GLfloat, 4, COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);
    // Get light position
    GLfloat lightPos[4];
    glGetLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    // Enable geometry shader
    this->arrowShader.Enable();
    glUniform4fvARB(this->arrowShader.ParameterLocation("viewAttr"),
            1, viewportStuff);
    glUniform3fvARB(this->arrowShader.ParameterLocation("camIn"),
            1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->arrowShader.ParameterLocation("camRight"),
            1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->arrowShader.ParameterLocation("camUp" ),
            1, cameraInfo->Up().PeekComponents());
    glUniform1fARB(this->arrowShader.ParameterLocation("radScale"),
            this->arrowRad);
    glUniformMatrix4fvARB(this->arrowShader.ParameterLocation("modelview"),
            1, false, modelMatrix_column);
    glUniformMatrix4fvARB(this->arrowShader.ParameterLocation("proj"),
            1, false, projMatrix_column);
    glUniform4fvARB(this->arrowShader.ParameterLocation("lightPos"),
            1, lightPos);

    // Get attribute locations
    GLint attribPos0 = glGetAttribLocationARB(this->arrowShader, "pos0");
    GLint attribPos1 = glGetAttribLocationARB(this->arrowShader, "pos1");
    GLint attribColor = glGetAttribLocationARB(this->arrowShader, "color");

    // Enable arrays for attributes
    glEnableVertexAttribArrayARB(attribPos0);
    glEnableVertexAttribArrayARB(attribPos1);
    glEnableVertexAttribArrayARB(attribColor);

    // Set attribute pointers
    glVertexAttribPointerARB(attribPos0, 4, GL_FLOAT, GL_FALSE, 0, this->arrowData.PeekElements());
    glVertexAttribPointerARB(attribPos1, 3, GL_FLOAT, GL_FALSE, 0, this->arrowDataPos.PeekElements());
    glVertexAttribPointerARB(attribColor, 3, GL_FLOAT, GL_FALSE, 0, this->arrowCol.PeekElements());

    // Draw points
    glDrawElements(GL_POINTS, static_cast<GLsizei>(this->arrowVisIdx.Count()),
        GL_UNSIGNED_INT, this->arrowVisIdx.PeekElements());

    // Disable arrays for attributes
    glDisableVertexAttribArrayARB(attribPos0);
    glDisableVertexAttribArrayARB(attribPos1);
    glDisableVertexAttribArrayARB(attribColor);

    this->arrowShader.Disable();

    return true;
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::RenderAtomsSpheres
 */
void protein_cuda::CrystalStructureVolumeRenderer::RenderAtomsSpheres (
		const protein_calls::CrystalStructureDataCall *dc) {

    float viewportStuff[4] = {
            cameraInfo->TileRect().Left(),
            cameraInfo->TileRect().Bottom(),
            cameraInfo->TileRect().Width(),
            cameraInfo->TileRect().Height()
    };
    if(viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if(viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable(GL_BLEND);

    // Render spheres for all visible atoms
    this->sphereShader.Enable();

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // Set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, this->cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, this->cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, this->cameraInfo->Up().PeekComponents());
    glUniform2fARB(this->sphereShader.ParameterLocation("zValues"), this->cameraInfo->NearClip(), this->cameraInfo->FarClip());

    // Set vertex and color pointers and draw them
    glVertexPointer(4, GL_FLOAT, 0, this->posInter.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, this->atomColor.PeekElements());
    //glDrawArrays(GL_POINTS, 0, dc->GetAtomCnt());
    glDrawElements(GL_POINTS, static_cast<GLsizei>(this->visAtomIdx.Count()),
            GL_UNSIGNED_INT, this->visAtomIdx.PeekElements());

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    this->sphereShader.Disable();
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::RenderCritPointsSpheres
 */
void protein_cuda::CrystalStructureVolumeRenderer::RenderCritPointsSpheres(
		const protein_calls::CrystalStructureDataCall *dc) {
    using namespace vislib::math;

    glDisable(GL_BLEND);

    Vector<float, 3> gridMinCoord, gridMaxCoord;
    gridMinCoord[0] = this->bbox.ObjectSpaceBBox().Left();
    gridMinCoord[1] = this->bbox.ObjectSpaceBBox().Bottom();
    gridMinCoord[2] = this->bbox.ObjectSpaceBBox().Back();
    gridMaxCoord[0] = this->bbox.ObjectSpaceBBox().Right();
    gridMaxCoord[1] = this->bbox.ObjectSpaceBBox().Top();
    gridMaxCoord[2] = this->bbox.ObjectSpaceBBox().Front();

    // Recalculate critical points if necessary
    Vector<float, 3> offs(0.5f, 0.5f, 0.5f);
    if(this->recalcCritPoints) {
        this->critPoints = CritPoints::GetCritPoints(
                this->uniGridVecField,
                gridMinCoord + offs,
                gridMaxCoord - offs);
        this->recalcCritPoints = false;
    }

    // Prepare sphere rendering
    vislib::Array<float> sphColor, sphPos;
    sphColor.SetCount(this->critPoints.Count());
    sphPos.SetCount(this->critPoints.Count()/3*4);

#pragma omp parallel for
    for(int cnt = 0; cnt < static_cast<int>(this->critPoints.Count()/3); cnt++) {
            sphPos[cnt*4+0] = this->critPoints[3*cnt+0];
            sphPos[cnt*4+1] = this->critPoints[3*cnt+1];
            sphPos[cnt*4+2] = this->critPoints[3*cnt+2];
            sphPos[cnt*4+3] = 1.2f;
            sphColor[cnt*3+0] = 1.0f; // TODO Different colors for sink/source?
            sphColor[cnt*3+1] = 0.5f;
            sphColor[cnt*3+2] = 0.0f;
        if(this->cpUsePosFilter) {
            if((this->critPoints[3*cnt+0] > this->posXMax) ||
                (this->critPoints[3*cnt+0] < this->posXMin) ||
                (this->critPoints[3*cnt+1] > this->posYMax) ||
                (this->critPoints[3*cnt+1] < this->posYMin) ||
                (this->critPoints[3*cnt+2] > this->posZMax) ||
                (this->critPoints[3*cnt+2] < this->posZMin)) {
                    sphPos[cnt*4+3] = 0.0f; // Set radius to zero if sphere is filtered out
            }
        }
    }

    float viewportStuff[4] = {
            this->cameraInfo->TileRect().Left(),
            this->cameraInfo->TileRect().Bottom(),
            this->cameraInfo->TileRect().Width(),
            this->cameraInfo->TileRect().Height()
    };
    if(viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if(viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];


    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // Render spheres for all visible atoms
    this->sphereShader.Enable();

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // Set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, this->cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, this->cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, this->cameraInfo->Up().PeekComponents());
    glUniform2fARB(this->sphereShader.ParameterLocation("zValues"), this->cameraInfo->NearClip(), this->cameraInfo->FarClip());

    // Set vertex and color pointers and draw them
    glVertexPointer(4, GL_FLOAT, 0, sphPos.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, sphColor.PeekElements());
	glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(this->critPoints.Count() / 3));

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    this->sphereShader.Disable();

}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::RenderEdgesBaLines
 */
void protein_cuda::CrystalStructureVolumeRenderer::RenderEdgesBaLines(
		const protein_calls::CrystalStructureDataCall *dc, const float *atomPos,
        const float *atomCol) {

    glDisable(GL_BLEND);
    glDisable(GL_LIGHTING);

    // Render lines for all atoms
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glVertexPointer(3, GL_FLOAT, 16, atomPos);
    glColorPointer(3, GL_FLOAT, 0, atomCol);
	glDrawElements(GL_LINES, static_cast<GLsizei>(this->edgeIdxBa.Count()), GL_UNSIGNED_INT,
            this->edgeIdxBa.PeekElements());

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::RenderEdgesBaStick
 */
void protein_cuda::CrystalStructureVolumeRenderer::RenderEdgesBaStick(
		const protein_calls::CrystalStructureDataCall *dc, const float *atomPos,
        const float *atomCol) {

    int cnt, idx0, idx1;
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Vector<float, 3> tmpVec, ortho, dir, position;
    vislib::math::Quaternion<float> quatC(0, 0, 0, 1);
    float angle;

    // Allocate memory for stick raycasting
    this->vertCylinders.SetCount((this->edgeIdxBa.Count()/2)*4);
    this->quatCylinders.SetCount((this->edgeIdxBa.Count()/2)*4);
    this->inParaCylinders.SetCount((this->edgeIdxBa.Count()/2)*2);
    this->color1Cylinders.SetCount((this->edgeIdxBa.Count()/2)*3);
    this->color2Cylinders.SetCount((this->edgeIdxBa.Count()/2)*3);

    // Loop over all connections and compute cylinder parameters
#pragma omp parallel for private(idx0, idx1, firstAtomPos, secondAtomPos, quatC, tmpVec, ortho, dir, position, angle)
    for(cnt = 0; cnt < static_cast<int>(this->edgeIdxBa.Count()/2); cnt++) {
        idx0 = this->edgeIdxBa[2*cnt+0];
        idx1 = this->edgeIdxBa[2*cnt+1];

        firstAtomPos.SetX(atomPos[4*idx0+0]);
        firstAtomPos.SetY(atomPos[4*idx0+1]);
        firstAtomPos.SetZ(atomPos[4*idx0+2]);

        secondAtomPos.SetX(atomPos[4*idx1+0]);
        secondAtomPos.SetY(atomPos[4*idx1+1]);
        secondAtomPos.SetZ(atomPos[4*idx1+2]);

        // compute the quaternion for the rotation of the cylinder
        dir = secondAtomPos - firstAtomPos;
        tmpVec.Set( 1.0f, 0.0f, 0.0f);
        angle = - tmpVec.Angle( dir);
        ortho = tmpVec.Cross( dir);
        ortho.Normalise();
        quatC.Set( angle, ortho);
        // compute the absolute position 'position' of the cylinder (center point)
        position = firstAtomPos + (dir/2.0f);

        this->inParaCylinders[2*cnt+0] = this->baStickRadius;
        this->inParaCylinders[2*cnt+1] = (firstAtomPos-secondAtomPos).Length();

        this->quatCylinders[4*cnt+0] = quatC.GetX();
        this->quatCylinders[4*cnt+1] = quatC.GetY();
        this->quatCylinders[4*cnt+2] = quatC.GetZ();
        this->quatCylinders[4*cnt+3] = quatC.GetW();

        if(atomCol != NULL) {
            this->color1Cylinders[3*cnt+0] = atomCol[3*idx0+0];
            this->color1Cylinders[3*cnt+1] = atomCol[3*idx0+1];
            this->color1Cylinders[3*cnt+2] = atomCol[3*idx0+2];
            this->color2Cylinders[3*cnt+0] = atomCol[3*idx1+0];
            this->color2Cylinders[3*cnt+1] = atomCol[3*idx1+1];
            this->color2Cylinders[3*cnt+2] = atomCol[3*idx1+2];
        }
        else {
            this->color1Cylinders[3*cnt+0] = 0.0f;
            this->color1Cylinders[3*cnt+1] = 0.0f;
            this->color1Cylinders[3*cnt+2] = 0.0f;
            this->color2Cylinders[3*cnt+0] = 0.0f;
            this->color2Cylinders[3*cnt+1] = 0.0f;
            this->color2Cylinders[3*cnt+2] = 0.0f;
        }

        this->vertCylinders[4*cnt+0] = position.X();
        this->vertCylinders[4*cnt+1] = position.Y();
        this->vertCylinders[4*cnt+2] = position.Z();
        this->vertCylinders[4*cnt+3] = 0.0f;

    }


    // Get viewpoint parameters for raycasting
    float viewportStuff[4] = {
            this->cameraInfo->TileRect().Left(),
            this->cameraInfo->TileRect().Bottom(),
            this->cameraInfo->TileRect().Width(),
            this->cameraInfo->TileRect().Height()};
    if(viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if(viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable(GL_BLEND);

    // Enable cylinder shader
    this->cylinderShader.Enable();
    glUniform4fvARB(this->cylinderShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->cylinderShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->cylinderShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->cylinderShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    this->attribLocInParams = glGetAttribLocationARB(this->cylinderShader, "inParams");
    this->attribLocQuatC    = glGetAttribLocationARB(this->cylinderShader, "quatC");
    this->attribLocColor1   = glGetAttribLocationARB(this->cylinderShader, "color1");
    this->attribLocColor2   = glGetAttribLocationARB(this->cylinderShader, "color2");

    glEnableClientState(GL_VERTEX_ARRAY);

    glEnableVertexAttribArrayARB(this->attribLocInParams);
    glEnableVertexAttribArrayARB(this->attribLocQuatC);
    glEnableVertexAttribArrayARB(this->attribLocColor1);
    glEnableVertexAttribArrayARB(this->attribLocColor2);

    // Set vertex and attribute pointers and draw them
    glVertexPointer(4, GL_FLOAT, 0, this->vertCylinders.PeekElements());
    glVertexAttribPointerARB(this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylinders.PeekElements());
    glVertexAttribPointerARB(this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinders.PeekElements());
    glVertexAttribPointerARB(this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1Cylinders.PeekElements());
    glVertexAttribPointerARB(this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2Cylinders.PeekElements());

	glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(this->edgeIdxBa.Count() / 2));

    glDisableVertexAttribArrayARB(this->attribLocInParams);
    glDisableVertexAttribArrayARB(this->attribLocQuatC);
    glDisableVertexAttribArrayARB(this->attribLocColor1);
    glDisableVertexAttribArrayARB(this->attribLocColor2);

    glDisableClientState(GL_VERTEX_ARRAY);

    // disable cylinder shader
    this->cylinderShader.Disable();
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::RenderEdgesTiLines
 */
void protein_cuda::CrystalStructureVolumeRenderer::RenderEdgesTiLines(
		const protein_calls::CrystalStructureDataCall *dc, const float *atomPos,
        const float *atomCol) {

    glDisable(GL_BLEND);
    glDisable(GL_LIGHTING);

    // Render lines for all atoms
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glVertexPointer(3, GL_FLOAT, 16, atomPos);
    glColorPointer(3, GL_FLOAT, 0, atomCol);
	glDrawElements(GL_LINES, static_cast<GLsizei>(this->edgeIdxTi.Count()), GL_UNSIGNED_INT,
            this->edgeIdxTi.PeekElements());

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::RenderEdgesTiStick
 */
void protein_cuda::CrystalStructureVolumeRenderer::RenderEdgesTiStick(
		const protein_calls::CrystalStructureDataCall *dc, const float *atomPos,
        const float *atomCol) {

    int cnt, idx0, idx1;
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Vector<float, 3> tmpVec, ortho, dir, position;
    vislib::math::Quaternion<float> quatC(0, 0, 0, 1);
    float angle;

    // Allocate memory for stick raycasting
    this->vertCylinders.SetCount((this->edgeIdxTi.Count()/2)*4);
    this->quatCylinders.SetCount((this->edgeIdxTi.Count()/2)*4);
    this->inParaCylinders.SetCount((this->edgeIdxTi.Count()/2)*2);
    this->color1Cylinders.SetCount((this->edgeIdxTi.Count()/2)*3);
    this->color2Cylinders.SetCount((this->edgeIdxTi.Count()/2)*3);


    // Loop over all connections and compute cylinder parameters
#pragma omp parallel for private(idx0, idx1, firstAtomPos, secondAtomPos, quatC, tmpVec, ortho, dir, position, angle)
    for(cnt = 0; cnt < static_cast<int>(this->edgeIdxTi.Count()/2); cnt++) {
        idx0 = this->edgeIdxTi[2*cnt+0];
        idx1 = this->edgeIdxTi[2*cnt+1];

        firstAtomPos.SetX(atomPos[4*idx0+0]);
        firstAtomPos.SetY(atomPos[4*idx0+1]);
        firstAtomPos.SetZ(atomPos[4*idx0+2]);

        secondAtomPos.SetX(atomPos[4*idx1+0]);
        secondAtomPos.SetY(atomPos[4*idx1+1]);
        secondAtomPos.SetZ(atomPos[4*idx1+2]);

        // compute the quaternion for the rotation of the cylinder
        dir = secondAtomPos - firstAtomPos;
        tmpVec.Set( 1.0f, 0.0f, 0.0f);
        angle = - tmpVec.Angle( dir);
        ortho = tmpVec.Cross( dir);
        ortho.Normalise();
        quatC.Set( angle, ortho);
        // compute the absolute position 'position' of the cylinder (center point)
        position = firstAtomPos + (dir/2.0f);

        this->inParaCylinders[2*cnt+0] = this->tiStickRadius;
        this->inParaCylinders[2*cnt+1] = (firstAtomPos-secondAtomPos).Length();

        this->quatCylinders[4*cnt+0] = quatC.GetX();
        this->quatCylinders[4*cnt+1] = quatC.GetY();
        this->quatCylinders[4*cnt+2] = quatC.GetZ();
        this->quatCylinders[4*cnt+3] = quatC.GetW();

        this->color1Cylinders[3*cnt+0] = atomCol[3*idx0+0];
        this->color1Cylinders[3*cnt+1] = atomCol[3*idx0+1];
        this->color1Cylinders[3*cnt+2] = atomCol[3*idx0+2];
        this->color2Cylinders[3*cnt+0] = atomCol[3*idx1+0];
        this->color2Cylinders[3*cnt+1] = atomCol[3*idx1+1];
        this->color2Cylinders[3*cnt+2] = atomCol[3*idx1+2];

        this->vertCylinders[4*cnt+0] = position.X();
        this->vertCylinders[4*cnt+1] = position.Y();
        this->vertCylinders[4*cnt+2] = position.Z();
        this->vertCylinders[4*cnt+3] = 0.0f;

    }


    // Get viewpoint parameters for raycasting
    float viewportStuff[4] = {
            this->cameraInfo->TileRect().Left(),
            this->cameraInfo->TileRect().Bottom(),
            this->cameraInfo->TileRect().Width(),
            this->cameraInfo->TileRect().Height()};
    if(viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if(viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable(GL_BLEND);

    // Enable cylinder shader
    this->cylinderShader.Enable();
    glUniform4fvARB(this->cylinderShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->cylinderShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->cylinderShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->cylinderShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    this->attribLocInParams = glGetAttribLocationARB(this->cylinderShader, "inParams");
    this->attribLocQuatC    = glGetAttribLocationARB(this->cylinderShader, "quatC");
    this->attribLocColor1   = glGetAttribLocationARB(this->cylinderShader, "color1");
    this->attribLocColor2   = glGetAttribLocationARB(this->cylinderShader, "color2");

    glEnableClientState(GL_VERTEX_ARRAY);

    glEnableVertexAttribArrayARB(this->attribLocInParams);
    glEnableVertexAttribArrayARB(this->attribLocQuatC);
    glEnableVertexAttribArrayARB(this->attribLocColor1);
    glEnableVertexAttribArrayARB(this->attribLocColor2);

    // Set vertex and attribute pointers and draw them
    glVertexPointer(4, GL_FLOAT, 0, this->vertCylinders.PeekElements());
    glVertexAttribPointerARB(this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylinders.PeekElements());
    glVertexAttribPointerARB(this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinders.PeekElements());
    glVertexAttribPointerARB(this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1Cylinders.PeekElements());
    glVertexAttribPointerARB(this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2Cylinders.PeekElements());

	glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>((this->edgeIdxTi.Count() / 2)));

    glDisableVertexAttribArrayARB(this->attribLocInParams);
    glDisableVertexAttribArrayARB(this->attribLocQuatC);
    glDisableVertexAttribArrayARB(this->attribLocColor1);
    glDisableVertexAttribArrayARB(this->attribLocColor2);

    glDisableClientState(GL_VERTEX_ARRAY);

    // disable cylinder shader
    this->cylinderShader.Disable();
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::renderIsoSurfMC
 */
bool protein_cuda::CrystalStructureVolumeRenderer::RenderIsoSurfMC() {
    using namespace vislib::sys;
    using namespace vislib::math;

    // Calculate isosurface

    //time_t t = clock();

    Vector<float, 3> gridMinCoord, gridMaxCoord, gridXAxis, gridYAxis,
    gridZAxis, gridOrg;
    Vector<int, 3> gridDim;

    // Init uniform grid
    gridMinCoord[0] = this->bbox.ObjectSpaceBBox().Left();
    gridMinCoord[1] = this->bbox.ObjectSpaceBBox().Bottom();
    gridMinCoord[2] = this->bbox.ObjectSpaceBBox().Back();
    gridMaxCoord[0] = this->bbox.ObjectSpaceBBox().Right();
    gridMaxCoord[1] = this->bbox.ObjectSpaceBBox().Top();
    gridMaxCoord[2] = this->bbox.ObjectSpaceBBox().Front();
    gridXAxis[0] = gridMaxCoord[0]-gridMinCoord[0];
    gridYAxis[1] = gridMaxCoord[1]-gridMinCoord[1];
    gridZAxis[2] = gridMaxCoord[2]-gridMinCoord[2];
    gridDim[0] = (int) ceil(gridXAxis[0] / this->densGridSpacing);
    gridDim[1] = (int) ceil(gridYAxis[1] / this->densGridSpacing);
    gridDim[2] = (int) ceil(gridZAxis[2] / this->densGridSpacing);
    gridXAxis[0] = (gridDim[0]-1) * this->densGridSpacing;
    gridYAxis[1] = (gridDim[1]-1) * this->densGridSpacing;
    gridZAxis[2] = (gridDim[2]-1) * this->densGridSpacing;
    gridMaxCoord[0] = gridMinCoord[0] + gridXAxis[0];
    gridMaxCoord[1] = gridMinCoord[1] + gridYAxis[1];
    gridMaxCoord[2] = gridMinCoord[2] + gridZAxis[2];
    gridOrg[0] = gridMinCoord[0];
    gridOrg[1] = gridMinCoord[1];
    gridOrg[2] = gridMinCoord[2];

    if(!this->cudaMC) {
        this->cudaMC = new CUDAMarchingCubes();
    }

    uint3 gridDimAlt = make_uint3(gridDim.X(), gridDim.Y(), gridDim.Z());
    float3 gridOrgAlt = make_float3(gridOrg.X(),gridOrg.Y(), gridOrg.Z());
    float3 gridBBox = make_float3(
            gridMaxCoord[0] - gridMinCoord[0],
            gridMaxCoord[1] - gridMinCoord[1],
            gridMaxCoord[2] - gridMinCoord[2]);

    uint3 subVolStart;
    subVolStart.x = static_cast<unsigned int>(((this->posXMin - gridOrg[0])/gridXAxis[0])*gridDim.X());
    subVolStart.y = static_cast<unsigned int>(((this->posYMin - gridOrg[1])/gridYAxis[1])*gridDim.Y());
    subVolStart.z = static_cast<unsigned int>(((this->posZMin - gridOrg[2])/gridZAxis[2])*gridDim.Z());

    uint3 subVolEnd;
    subVolEnd.x = static_cast<unsigned int>(((this->posXMax - gridOrg[0])/gridXAxis[0])*gridDim.X());
    subVolEnd.y = static_cast<unsigned int>(((this->posYMax - gridOrg[1])/gridYAxis[1])*gridDim.Y());
    subVolEnd.z = static_cast<unsigned int>(((this->posZMax - gridOrg[2])/gridZAxis[2])*gridDim.Z());

    if(this->posXMax > gridMaxCoord[0]) this->posXMax = gridMaxCoord[0];
    if(this->posYMax > gridMaxCoord[1]) this->posYMax = gridMaxCoord[1];
    if(this->posZMax > gridMaxCoord[2]) this->posZMax = gridMaxCoord[2];
    if(this->posXMin < gridMinCoord[0]) this->posXMin = gridMinCoord[0];
    if(this->posYMin < gridMinCoord[1]) this->posYMin = gridMinCoord[1];
    if(this->posZMin < gridMinCoord[2]) this->posZMin = gridMinCoord[2];

    // TODO Make NVertices dependent of actual subvolume
    unsigned int nVerticesMC = (gridDim.X()*gridDim.Y()*gridDim.Z())/2*3*6;

    //printf("Subvolume start (%f %f %f) (%u %u %u)\n", this->volXMin, this->volYMin,
    //      this->volZMin, subVolStart.x, subVolStart.y, subVolStart.z); // DEBUG

    //printf("Subvolume end (%f %f %f) (%u %u %u)\n", this->volXMax, this->volYMax,
    //this->volZMax, subVolEnd.x, subVolEnd.y, subVolEnd.z); // DEBUG

    if(nVerticesMC != this->nVerticesMCOld) {
        // Free if necessary
        if(this->mcVertOut_D != NULL) checkCudaErrors(cudaFree(this->mcVertOut_D));
        if(this->mcNormOut_D != NULL) checkCudaErrors(cudaFree(this->mcNormOut_D));
        if(this->mcVertOut != NULL) delete[] this->mcVertOut;
        if(this->mcNormOut != NULL) delete[] this->mcNormOut;
        // Allocate memory
        checkCudaErrors(cudaMalloc((void **)&this->mcVertOut_D, nVerticesMC*sizeof(float3)));
        checkCudaErrors(cudaMalloc((void **)&this->mcNormOut_D, nVerticesMC*sizeof(float3)));
        this->mcVertOut = new float[nVerticesMC*3];
        this->mcNormOut = new float[nVerticesMC*3];
        printf("(Re)allocating of memory done.");
    }

    CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf;

    // Setup
    if (!this->cudaMC->Initialize(gridDimAlt)) {
        return false;
    }
    if (!this->cudaMC->SetVolumeData(
            //this->gridCurlMagD,
            cqs->getMap(),
            NULL, gridDimAlt, gridOrgAlt, gridBBox, true)) {
        return false;
    }
    this->cudaMC->SetSubVolume(subVolStart, subVolEnd);
    this->cudaMC->SetIsovalue(this->volIsoVal);
    this->cudaMC->computeIsosurface(this->mcVertOut_D,
            this->mcNormOut_D, NULL, nVerticesMC);
    this->cudaMC->Cleanup();

    //printf("Number of vertices %u\n", this->cudaMC->GetVertexCount());

    /*cudaMC->computeIsosurface(
            this->gridCurlMagD,
            NULL,       // Color array
            gridDim,
            gridOrg,
            gridBBox,
            true,      // Use device memory
            this->mcVertOut_D,  // Output
            this->mcNormOut_D,  // Output
            NULL,   // Output
            nVerticesMC); // Maximum number of vertices*/

    //Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
    //      "Time for computing isosurface by CUDA marching cubes %f",
    //          (double(clock()-t)/double(CLOCKS_PER_SEC) )); // DEBUG
    //t = clock();


    // Copy data to host

    checkCudaErrors(cudaMemcpy(
            this->mcVertOut,
            this->mcVertOut_D,
            this->cudaMC->GetVertexCount()*3*sizeof(float),
            cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(
            this->mcNormOut,
            this->mcNormOut_D,
            this->cudaMC->GetVertexCount()*3*sizeof(float),
            cudaMemcpyDeviceToHost));

   /* Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "Time for CUDA memcopy %f",
            (double(clock()-t)/double(CLOCKS_PER_SEC) )); // DEBUG
    t = clock();*/


    // Render

    this->pplShader.Enable();
    glEnable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_3D);
    glDisable(GL_BLEND);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glColor4f(1.0, 0.0f, 0.0f, 1.0f);

    glVertexPointer(3, GL_FLOAT, 0, this->mcVertOut);
    glNormalPointer(GL_FLOAT, 0, this->mcNormOut);
    glDrawArrays(GL_TRIANGLES, 0, this->cudaMC->GetVertexCount());

    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glDisable(GL_LIGHTING);

    this->pplShader.Disable();

    glColor4f(1.0, 1.0f, 1.0f, 1.0f);

    /*Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "Time for rendering marching cubes triangles %f",
            (double(clock()-t)/double(CLOCKS_PER_SEC) )); // DEBUG*/

    this->nVerticesMCOld = nVerticesMC;

    return true;
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::RenderVolCube
 */
void protein_cuda::CrystalStructureVolumeRenderer::RenderVolCube() {

    using namespace vislib::math;
    Vector<float, 3> gridMinCoord, gridMaxCoord;

    gridMinCoord[0] = this->bbox.ObjectSpaceBBox().Left();
    gridMinCoord[1] = this->bbox.ObjectSpaceBBox().Bottom();
    gridMinCoord[2] = this->bbox.ObjectSpaceBBox().Back();
    gridMaxCoord[0] = this->bbox.ObjectSpaceBBox().Right();
    gridMaxCoord[1] = this->bbox.ObjectSpaceBBox().Top();
    gridMaxCoord[2] = this->bbox.ObjectSpaceBBox().Front();

#if (defined(NOCLIP_ISOSURF) && (NOCLIP_ISOSURF))
    this->posXMin -= this->maxLenDiff*this->densGridRad;
    this->posYMin -= this->maxLenDiff*this->densGridRad;
    this->posZMin -= this->maxLenDiff*this->densGridRad;
    this->posXMax += this->maxLenDiff*this->densGridRad;
    this->posYMax += this->maxLenDiff*this->densGridRad;
    this->posZMax += this->maxLenDiff*this->densGridRad;
#endif

    if(this->posXMin < gridMinCoord[0])this->posXMin = gridMinCoord[0];
    if(this->posYMin < gridMinCoord[1])this->posYMin = gridMinCoord[1];
    if(this->posZMin < gridMinCoord[2])this->posZMin = gridMinCoord[2];
    if(this->posXMax > gridMaxCoord[0])this->posXMax = gridMaxCoord[0];
    if(this->posYMax > gridMaxCoord[1])this->posYMax = gridMaxCoord[1];
    if(this->posZMax > gridMaxCoord[2])this->posZMax = gridMaxCoord[2];

    /*printf("Bounding box:\n");
    printf("    LEFT %f, RIGHT %f\n", this->bbox.ObjectSpaceBBox().Left(), this->bbox.ObjectSpaceBBox().Right());
    printf("    BOTTOM %f, TOP %f\n", this->bbox.ObjectSpaceBBox().Bottom(), this->bbox.ObjectSpaceBBox().Top());
    printf("    BACK %f, FRONT %f\n", this->bbox.ObjectSpaceBBox().Back(), this->bbox.ObjectSpaceBBox().Front());*/

    // Cut off border areas according to filter parameters
    float minXTex = (this->posXMin-gridMinCoord.X())/(gridMaxCoord.X()-gridMinCoord.X());
    float minYTex = (this->posYMin-gridMinCoord.Y())/(gridMaxCoord.Y()-gridMinCoord.Y());
    float minZTex = (this->posZMin-gridMinCoord.Z())/(gridMaxCoord.Z()-gridMinCoord.Z());
    float maxXTex = (this->posXMax-gridMinCoord.X())/(gridMaxCoord.X()-gridMinCoord.X());
    float maxYTex = (this->posYMax-gridMinCoord.Y())/(gridMaxCoord.Y()-gridMinCoord.Y());
    float maxZTex = (this->posZMax-gridMinCoord.Z())/(gridMaxCoord.Z()-gridMinCoord.Z());


    glBegin(GL_QUADS);

    // Front

    glColor3f(maxXTex, minYTex, maxZTex);
    glTexCoord3f(maxXTex, minYTex, maxZTex);
    glVertex3f(this->posXMax, this->posYMin, this->posZMax);

    glColor3f(maxXTex, maxYTex, maxZTex);
    glTexCoord3f(maxXTex, maxYTex, maxZTex);
    glVertex3f(this->posXMax, this->posYMax, this->posZMax);

    glColor3f(minXTex, maxYTex, maxZTex);
    glTexCoord3f(minXTex, maxYTex, maxZTex);
    glVertex3f(this->posXMin, this->posYMax, this->posZMax);

    glColor3f(minXTex, minYTex, maxZTex);
    glTexCoord3f(minXTex, minYTex, maxZTex);
    glVertex3f(this->posXMin, this->posYMin, this->posZMax);

    // Back

    glColor3f(minXTex, maxYTex, minZTex);
    glTexCoord3f(minXTex, maxYTex, minZTex);
    glVertex3f(this->posXMin, this->posYMax, this->posZMin);

    glColor3f(maxXTex, maxYTex, minZTex);
    glTexCoord3f(maxXTex, maxYTex, minZTex);
    glVertex3f(this->posXMax, this->posYMax, this->posZMin);

    glColor3f(maxXTex, minYTex, minZTex);
    glTexCoord3f(maxXTex, minYTex, minZTex);
    glVertex3f(this->posXMax, this->posYMin, this->posZMin);

    glColor3f(minXTex, minYTex, minZTex);
    glTexCoord3f(minXTex, minYTex, minZTex);
    glVertex3f(this->posXMin, this->posYMin, this->posZMin);

    // Left

    glColor3f(minXTex, minYTex, maxZTex);
    glTexCoord3f(minXTex, minYTex, maxZTex);
    glVertex3f(this->posXMin, this->posYMin, this->posZMax);

    glColor3f(minXTex, maxYTex, maxZTex);
    glTexCoord3f(minXTex, maxYTex, maxZTex);
    glVertex3f(this->posXMin, this->posYMax, this->posZMax);

    glColor3f(minXTex, maxYTex, minZTex);
    glTexCoord3f(minXTex, maxYTex, minZTex);
    glVertex3f(this->posXMin, this->posYMax, this->posZMin);

    glColor3f(minXTex, minYTex, minZTex);
    glTexCoord3f(minXTex, minYTex, minZTex);
    glVertex3f(this->posXMin, this->posYMin, this->posZMin);

    // Right

    glColor3f(maxXTex, maxYTex, minZTex);
    glTexCoord3f(maxXTex, maxYTex, minZTex);
    glVertex3f(this->posXMax, this->posYMax, this->posZMin);

    glColor3f(maxXTex, maxYTex, maxZTex);
    glTexCoord3f(maxXTex, maxYTex, maxZTex);
    glVertex3f(this->posXMax, this->posYMax, this->posZMax);

    glColor3f(maxXTex, minYTex, maxZTex);
    glTexCoord3f(maxXTex, minYTex, maxZTex);
    glVertex3f(this->posXMax, this->posYMin, this->posZMax);

    glColor3f(maxXTex, minYTex, minZTex);
    glTexCoord3f(maxXTex, minYTex, minZTex);
    glVertex3f(this->posXMax, this->posYMin, this->posZMin);

    // Top

    glColor3f(minXTex, maxYTex, maxZTex);
    glTexCoord3f(minXTex, maxYTex, maxZTex);
    glVertex3f(this->posXMin, this->posYMax, this->posZMax);

    glColor3f(maxXTex, maxYTex, maxZTex);
    glTexCoord3f(maxXTex, maxYTex, maxZTex);
    glVertex3f(this->posXMax, this->posYMax, this->posZMax);

    glColor3f(maxXTex, maxYTex, minZTex);
    glTexCoord3f(maxXTex, maxYTex, minZTex);
    glVertex3f(this->posXMax, this->posYMax, this->posZMin);

    glColor3f(minXTex, maxYTex, minZTex);
    glTexCoord3f(minXTex, maxYTex, minZTex);
    glVertex3f(this->posXMin, this->posYMax, this->posZMin);

    // Bottom

    glColor3f(maxXTex, minYTex, minZTex);
    glTexCoord3f(maxXTex, minYTex, minZTex);
    glVertex3f(this->posXMax, this->posYMin, this->posZMin);

    glColor3f(maxXTex, minYTex, maxZTex);
    glTexCoord3f(maxXTex, minYTex, maxZTex);
    glVertex3f(this->posXMax, this->posYMin, this->posZMax);

    glColor3f(minXTex, minYTex, maxZTex);
    glTexCoord3f(minXTex, minYTex, maxZTex);
    glVertex3f(this->posXMin, this->posYMin, this->posZMax);

    glColor3f(minXTex, minYTex,minZTex);
    glTexCoord3f(minXTex, minYTex, minZTex);
    glVertex3f(this->posXMin, this->posYMin, this->posZMin);

    glEnd();

}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::RenderVolume
 */
bool protein_cuda::CrystalStructureVolumeRenderer::RenderVolume() {
    using namespace vislib::sys;
    GLenum err;

    if(!glIsTexture(this->uniGridTex)||!glIsTexture(this->uniGridDensityTex)) {
        return false;
    }

    // Get current viewport and recreate fbo if necessary
    float curVP[4];
    glGetFloatv(GL_VIEWPORT, curVP);
    if((curVP[2] != this->fboDim.X()) || (curVP[3] != this->fboDim.Y())) {
        this->fboDim.SetX(static_cast<int>(curVP[2]));
        this->fboDim.SetY(static_cast<int>(curVP[3]));
        if(!this->CreateFbo(static_cast<UINT>(curVP[2]), static_cast<UINT>(curVP[3])))
            return false;
    }

    // Render back of the cube to fbo

    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    glDisable(GL_LIGHTING);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_3D);

    this->rcFbo.EnableMultiple(3, GL_COLOR_ATTACHMENT0_EXT,
            GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    this->rcShaderDebug.Enable();
    this->RenderVolCube(); // Render back of the cube and store depth values in fbo texture
    this->rcShaderDebug.Disable();

    this->rcFbo.Disable();

    // Render the front of the cube

    glCullFace(GL_BACK);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

    this->rcShader.Enable();
    glUniform1iARB(this->rcShader.ParameterLocation("tcBuff"), 0);
    glUniform1iARB(this->rcShader.ParameterLocation("densityTex"), 1);
    glUniform1iARB(this->rcShader.ParameterLocation("uniGridTex"), 2);
    glUniform1iARB(this->rcShader.ParameterLocation("srcColorBuff"), 3);
    glUniform1iARB(this->rcShader.ParameterLocation("srcDepthBuff"), 4);
    glUniform1iARB(this->rcShader.ParameterLocation("posESBuff"), 5);
    glUniform1iARB(this->rcShader.ParameterLocation("posWinSBuff"), 6);
    glUniform1iARB(this->rcShader.ParameterLocation("randNoiseTex"), 7);
    glUniform1iARB(this->rcShader.ParameterLocation("curlMagTex"), 8);
    glUniform1iARB(this->rcShader.ParameterLocation("colorTex"), 9);
    glUniform1fARB(this->rcShader.ParameterLocation("delta"), this->volDelta*0.01f);
    glUniform1fARB(this->rcShader.ParameterLocation("isoVal"), this->volIsoVal);
    glUniform1fARB(this->rcShader.ParameterLocation("alphaScl"), this->volAlphaScl);
    glUniform1fARB(this->rcShader.ParameterLocation("licDirScl"), this->volLicDirScl);
    glUniform1iARB(this->rcShader.ParameterLocation("licLen"), this->volLicLen);
    glUniform1fARB(this->rcShader.ParameterLocation("licContrast"), this->volLicContrastStretching);
    glUniform1fARB(this->rcShader.ParameterLocation("licBright"), this->volLicBright);
    glUniform1fARB(this->rcShader.ParameterLocation("licTCScl"), this->volLicTCScl);
	glUniform4fARB(this->rcShader.ParameterLocation("viewportDim"), static_cast<float>(fboDim.X()), static_cast<float>(fboDim.Y()),
            this->cameraInfo->NearClip(), this->cameraInfo->FarClip());
    glUniform1iARB(this->rcShader.ParameterLocation("vColorMode"), this->vColorMode);
    glUniform1iARB(this->rcShader.ParameterLocation("rayMarchTex"), this->rmTex);

    glActiveTextureARB(GL_TEXTURE1_ARB);
    glBindTexture(GL_TEXTURE_3D, this->uniGridDensityTex);

    glActiveTextureARB(GL_TEXTURE2_ARB);
    glBindTexture(GL_TEXTURE_3D, this->uniGridTex);

    glActiveTextureARB(GL_TEXTURE3_ARB);
    this->srcFbo.BindColourTexture(0);

    glActiveTextureARB(GL_TEXTURE4_ARB);
    this->srcFbo.BindDepthTexture();

    glActiveTextureARB(GL_TEXTURE5_ARB);
    this->rcFbo.BindColourTexture(1);

    glActiveTextureARB(GL_TEXTURE6_ARB);
    this->rcFbo.BindColourTexture(2);

    glActiveTextureARB(GL_TEXTURE7_ARB);
    glBindTexture(GL_TEXTURE_3D, this->randNoiseTex);

    glActiveTextureARB(GL_TEXTURE8_ARB);
    glBindTexture(GL_TEXTURE_3D, this->curlMagTex);

    glActiveTextureARB(GL_TEXTURE9_ARB);
    glBindTexture(GL_TEXTURE_3D, this->uniGridColorTex);

    glActiveTextureARB(GL_TEXTURE0_ARB);
    this->rcFbo.BindColourTexture(0);

    glEnable(GL_TEXTURE_2D);
    glEnable(GL_TEXTURE_3D);

    this->RenderVolCube();

    glDisable(GL_TEXTURE_3D);
    glDisable(GL_TEXTURE_2D);

    this->rcShader.Disable();

    // Check for opengl error
    err = glGetError();
    if(err != GL_NO_ERROR) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s:RenderVolume:: glError %s \n", this->ClassName(),
                gluErrorString(err));
        return false;
    }

    return true;
}


bool protein_cuda::CrystalStructureVolumeRenderer::SetupAtomColors(
		const protein_calls::CrystalStructureDataCall *dc) {

    this->atomColor.SetCount(dc->GetAtomCnt()*3);
#pragma omp parallel for
	for (int at = 0; at < (int)dc->GetAtomCnt(); at++) {
		if (dc->GetAtomType()[at] == protein_calls::CrystalStructureDataCall::BA) { // Green
            this->atomColor[at*3+0] = 0.0f;
            this->atomColor[at*3+1] = 0.6f;
            this->atomColor[at*3+2] = 0.4f;
        }
		else if (dc->GetAtomType()[at] == protein_calls::CrystalStructureDataCall::TI) { // Light grey
            this->atomColor[at*3+0] = 0.8f;
            this->atomColor[at*3+1] = 0.8f;
            this->atomColor[at*3+2] = 0.8f;
        }
		else if (dc->GetAtomType()[at] == protein_calls::CrystalStructureDataCall::O) { // Red
            this->atomColor[at*3+0] = 1.0f;
            this->atomColor[at*3+1] = 0.0f;
            this->atomColor[at*3+2] = 0.0f;
        }
        else { // White
            this->atomColor[at*3+0] = 0.0f;
            this->atomColor[at*3+1] = 0.0f;
            this->atomColor[at*3+2] = 0.0f;
        }
    }

    return true;
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::updateParams
 */
bool protein_cuda::CrystalStructureVolumeRenderer::UpdateParams(
		const protein_calls::CrystalStructureDataCall *dc) {

    // General params

    // Toggle positional interpolation
    if (this->interpolParam.IsDirty()) {
        this->interpol = this->interpolParam.Param<core::param::BoolParam>()->Value();
        this->interpolParam.ResetDirty();
    }

    // Minimal displacement param
    if (this->minVecMagParam.IsDirty()) {
        this->minVecMag = this->minVecMagParam.Param<core::param::FloatParam>()->Value();
        this->minVecMagParam.ResetDirty();
        this->recalcArrowData = true;
        this->recalcGrid = true;
        this->recalcCurlMag = true;
        this->recalcDensityGrid = true;
        this->filterVecField = true;
    }

    // Maximum displacement param
    if (this->maxVecMagParam.IsDirty()) {
        this->maxVecMag = this->maxVecMagParam.Param<core::param::FloatParam>()->Value();
        this->maxVecMagParam.ResetDirty();
        this->recalcArrowData = true;
        this->recalcGrid = true;
        this->recalcCurlMag = true;
        this->recalcDensityGrid = true;
        this->filterVecField = true;
    }

    // Maximum curl param
    if (this->maxVecCurlParam.IsDirty()) {
        this->maxVecCurl = this->maxVecCurlParam.Param<core::param::FloatParam>()->Value();
        this->maxVecCurlParam.ResetDirty();
        this->recalcArrowData = true;
        this->recalcGrid = true;
        this->recalcCurlMag = true;
        this->recalcDensityGrid = true;
        this->filterVecField = true;
    }


    // Cell based rendering modes

    // Atom render mode
    if (this->atomRenderModeParam.IsDirty()) {
        this->atomRenderModeParam.ResetDirty();
        this->atomRM = static_cast<AtomRenderMode>(this->atomRenderModeParam.Param<core::param::EnumParam>()->Value());
    }

    // Sphere radius
    if (this->sphereRadParam.IsDirty()) {
        this->sphereRad = this->sphereRadParam.Param<core::param::FloatParam>()->Value();
        this->sphereRadParam.ResetDirty();
        this->recalcPosInter = true;
    }

    // Ba edge render mode
    if (this->edgeBaRenderModeParam.IsDirty()) {
        this->edgeBaRenderModeParam.ResetDirty();
        this->edgeBaRM = static_cast<EdgeBaRenderMode>(this->edgeBaRenderModeParam.Param<core::param::EnumParam>()->Value());
    }

    // Ba Stick radius
    if (this->baStickRadiusParam.IsDirty()) {
        this->baStickRadius = this->baStickRadiusParam.Param<core::param::FloatParam>()->Value();
        this->baStickRadiusParam.ResetDirty();
    }

    // Ti edge render mode
    if (this->edgeTiRenderModeParam.IsDirty()) {
        this->edgeTiRenderModeParam.ResetDirty();
        this->edgeTiRM = static_cast<EdgeTiRenderMode>(this->edgeTiRenderModeParam.Param<core::param::EnumParam>()->Value());
    }

    // Ti stick radius
    if (this->tiStickRadiusParam.IsDirty()) {
        this->tiStickRadius = this->tiStickRadiusParam.Param<core::param::FloatParam>()->Value();
        this->tiStickRadiusParam.ResetDirty();
    }

    // Displacement render mode
    if (this->vecRMParam.IsDirty()) {
        this->vecRMParam.ResetDirty();
        this->vecRM = static_cast<VecRenderMode>(vecRMParam.Param<core::param::EnumParam>()->Value());
    }

    // Displacement arrow radius param
    if (this->arrowRadParam.IsDirty()) {
        this->arrowRad = this->arrowRadParam.Param<core::param::FloatParam>()->Value();
        this->arrowRadParam.ResetDirty();
        this->recalcArrowData = true;
    }

    // Toggle filtering of arrows
    if (this->arrowUseFilterParam.IsDirty()) {
        this->arrowUseFilter = this->arrowUseFilterParam.Param<core::param::BoolParam>()->Value();
        this->arrowUseFilterParam.ResetDirty();
        this->recalcArrowData = true;
    }

    // Toggle normalization of arrows
    if (this->toggleNormVecParam.IsDirty()) {
        this->toggleNormVec = this->toggleNormVecParam.Param<core::param::BoolParam>()->Value();
        this->toggleNormVecParam.ResetDirty();
        this->recalcArrowData = true;
    }

    // Show ba atoms
    if (this->showBaAtomsParam.IsDirty()) {
        this->showBaAtoms = this->showBaAtomsParam.Param<core::param::BoolParam>()->Value();
        this->showBaAtomsParam.ResetDirty();
        this->recalcArrowData = true;
    }

    // Show ti atoms
    if (this->showTiAtomsParam.IsDirty()) {
        this->showTiAtoms = this->showTiAtomsParam.Param<core::param::BoolParam>()->Value();
        this->showTiAtomsParam.ResetDirty();
        this->recalcArrowData = true;
    }

    // Show o atoms
    if (this->showOAtomsParam.IsDirty()) {
        this->showOAtoms = this->showOAtomsParam.Param<core::param::BoolParam>()->Value();
        this->showOAtomsParam.ResetDirty();
        this->recalcArrowData = true;
    }

    // Show ridge
    if (this->showRidgeParam.IsDirty()) {
        this->showRidge = this->showRidgeParam.Param<core::param::BoolParam>()->Value();
        this->showRidgeParam.ResetDirty();
    }

    // Displacement scaling
    if (this->vecSclParam.IsDirty()) {
        this->vecScl = this->vecSclParam.Param<core::param::FloatParam>()->Value();
        this->vecSclParam.ResetDirty();
        this->recalcArrowData = true;
        this->filterVecField = true;
    }

    // Positional filter
    if (this->filterAllParam.IsDirty()) {
        this->posFilterAll = this->filterAllParam.Param<core::param::FloatParam>()->Value();
        this->filterAllParam.ResetDirty();
        this->posXMax = this->posFilterAll;
        this->posYMax = this->posFilterAll;
        this->posZMax = this->posFilterAll;
        this->posXMin = -this->posFilterAll;
        this->posYMin = -this->posFilterAll;
        this->posZMin = -this->posFilterAll;
        this->recalcVisibility = true;
    }

    // Max x
    if (this->filterXMaxParam.IsDirty()) {
        this->posXMax = this->filterXMaxParam.Param<core::param::FloatParam>()->Value();
        this->filterXMaxParam.ResetDirty();
        this->recalcVisibility = true;
    }

    // Max y
    if (this->filterYMaxParam.IsDirty()) {
        this->posYMax = this->filterYMaxParam.Param<core::param::FloatParam>()->Value();
        this->filterYMaxParam.ResetDirty();
        this->recalcVisibility = true;
    }

    // Max z
    if (this->filterZMaxParam.IsDirty()) {
        this->posZMax = this->filterZMaxParam.Param<core::param::FloatParam>()->Value();
        this->filterZMaxParam.ResetDirty();
        this->recalcVisibility = true;
    }

    // Min x
    if (this->filterXMinParam.IsDirty()) {
        this->posXMin = this->filterXMinParam.Param<core::param::FloatParam>()->Value();
        this->filterXMinParam.ResetDirty();
        this->recalcVisibility = true;
    }

    // Min y
    if (this->filterYMinParam.IsDirty()) {
        this->posYMin = this->filterYMinParam.Param<core::param::FloatParam>()->Value();
        this->filterYMinParam.ResetDirty();
        this->recalcVisibility = true;
    }

    // Min z
    if (this->filterZMinParam.IsDirty()) {
        this->posZMin = this->filterZMinParam.Param<core::param::FloatParam>()->Value();
        this->filterZMinParam.ResetDirty();
        this->recalcVisibility = true;
    }

    // Arrow color mode
    if (this->arrColorModeParam.IsDirty()) {
        this->arrColorModeParam.ResetDirty();
        this->arrColorMode = static_cast<ArrowColorMode>(this->arrColorModeParam.Param<core::param::EnumParam>()->Value());
        this->recalcArrowData = true;
    }



    // Arrow color mode
    if (this->cpUsePosFilterParam.IsDirty()) {
        this->cpUsePosFilterParam.ResetDirty();
        this->cpUsePosFilter = this->cpUsePosFilterParam.Param<core::param::BoolParam>()->Value();
        //this->recalcCritPoints = true;
    }

    // Grid based rendering

    // Grid render mode
    if (this->sliceRenderModeParam.IsDirty()) {
        this->sliceRenderModeParam.ResetDirty();
        this->sliceRM = static_cast<SliceRenderMode>(this->sliceRenderModeParam.Param<core::param::EnumParam>()->Value());
    }

    // Grid spacing
    if (this->gridSpacingParam.IsDirty()) {
        this->gridSpacing = this->gridSpacingParam.Param<core::param::FloatParam>()->Value();
        this->gridSpacingParam.ResetDirty();
        this->recalcGrid = true;
        this->FreeBuffs(); // Free buffers and set all pointers to NULL
    }

    // Grid data radius radius
    if (this->gridDataRadParam.IsDirty()) {
        this->gridDataRad = this->gridDataRadParam.Param<core::param::FloatParam>()->Value();
        this->gridDataRadParam.ResetDirty();
        this->recalcGrid = true;
    }

    // Density grid radius
    if (this->densGridRadParam.IsDirty()) {
        this->densGridRad = this->densGridRadParam.Param<core::param::FloatParam>()->Value();
        this->densGridRadParam.ResetDirty();
        this->recalcDensityGrid = true;
    }

    // Density grid spacing
    if (this->densGridSpacingParam.IsDirty()) {
        this->densGridSpacing = this->densGridSpacingParam.Param<core::param::FloatParam>()->Value();
        this->densGridSpacingParam.ResetDirty();
        this->recalcDensityGrid = true;
        this->FreeBuffs();
        this->recalcGrid = true;
        this->recalcCurlMag = true;
    }

    // Density grid quality
    if (this->densGridGaussLimParam.IsDirty()) {
        this->densGridGaussLim = this->densGridGaussLimParam.Param<core::param::FloatParam>()->Value();
        this->densGridGaussLimParam.ResetDirty();
        this->recalcDensityGrid = true;
    }

    // Grid interpolation quality
    if (this->gridQualityParam.IsDirty()) {
        this->gridQuality = static_cast<int>(this->gridQualityParam.Param<core::param::EnumParam>()->Value());
        this->gridQualityParam.ResetDirty();
        this->recalcGrid = true;
    }

    // X-plane position
    if (this->xPlaneParam.IsDirty()) {
        this->xPlane = this->xPlaneParam.Param<core::param::FloatParam>()->Value();
        this->xPlaneParam.ResetDirty();
    }

    // X-plane visibility
    if (this->toggleXPlaneParam.IsDirty()) {
        this->showXPlane = this->toggleXPlaneParam.Param<core::param::BoolParam>()->Value();
        this->toggleXPlaneParam.ResetDirty();
    }

    // Y-plane position
    if (this->yPlaneParam.IsDirty()) {
        this->yPlane = this->yPlaneParam.Param<core::param::FloatParam>()->Value();
        this->yPlaneParam.ResetDirty();
    }

    // Y-plane visibility
    if (this->toggleYPlaneParam.IsDirty()) {
        this->showYPlane = this->toggleYPlaneParam.Param<core::param::BoolParam>()->Value();
        this->toggleYPlaneParam.ResetDirty();
    }

    // Z-plane position
    if (this->zPlaneParam.IsDirty()) {
        this->zPlane = this->zPlaneParam.Param<core::param::FloatParam>()->Value();
        this->zPlaneParam.ResetDirty();
    }

    // Z-plane visibility
    if (this->toggleZPlaneParam.IsDirty()) {
        this->showZPlane = this->toggleZPlaneParam.Param<core::param::BoolParam>()->Value();
        this->toggleZPlaneParam.ResetDirty();
    }

    // LIC direction vector scale param
    if (this->licDirSclParam.IsDirty()) {
        this->licDirScl = this->licDirSclParam.Param<core::param::FloatParam>()->Value();
        this->licDirSclParam.ResetDirty();
    }

    // LIC stream line length
    if (this->licStreamlineLengthParam.IsDirty()) {
        this->licStreamlineLength = this->licStreamlineLengthParam.Param<core::param::IntParam>()->Value();
        this->licStreamlineLengthParam.ResetDirty();
    }

    // LIC 2D projection
    if (this->projectVec2DParam.IsDirty()) {
        this->projectVec2D = this->projectVec2DParam.Param<core::param::BoolParam>()->Value();
        this->projectVec2DParam.ResetDirty();
    }

    // LIC random buffer size
    if (this->licRandBuffSizeParam.IsDirty()) {
        this->licRandBuffSize = this->licRandBuffSizeParam.Param<core::param::IntParam>()->Value();
        this->licRandBuffSizeParam.ResetDirty();
        // (Re)create random noise texture for LIC
        if(!this->InitLIC()) return false;
    }

    // LIC contrast stretching
    if (this->licContrastStretchingParam.IsDirty()) {
        this->licContrastStretching = this->licContrastStretchingParam.Param<core::param::FloatParam>()->Value();
        this->licContrastStretchingParam.ResetDirty();
    }

    // LIC brightness
    if (this->licBrightParam.IsDirty()) {
        this->licBright = this->licBrightParam.Param<core::param::FloatParam>()->Value();
        this->licBrightParam.ResetDirty();
    }

    // LIC TC scale factor
    if (this->licTCSclParam.IsDirty()) {
        this->licTCScl = this->licTCSclParam.Param<core::param::FloatParam>()->Value();
        this->licTCSclParam.ResetDirty();
    }

    // Slice data scale
    if (this->sliceDataSclParam.IsDirty()) {
        this->sliceDataScl = this->sliceDataSclParam.Param<core::param::FloatParam>()->Value();
        this->sliceDataSclParam.ResetDirty();
    }

    // Show crit points
    if (this->showCritPointsParam.IsDirty()) {
        this->showCritPoints = this->showCritPointsParam.Param<core::param::BoolParam>()->Value();
        this->showCritPointsParam.ResetDirty();
    }

    // Color mode for isosurface
    if (this->vColorModeParam.IsDirty()) {
        this->vColorModeParam.ResetDirty();
        this->vColorMode = static_cast<VolColorMode>(vColorModeParam.Param<core::param::EnumParam>()->Value());
        this->recalcDensityGrid = true;
    }

    // Show volume texture
    if (this->volShowParam.IsDirty()) {
        this->volShow = this->volShowParam.Param<core::param::BoolParam>()->Value();
        this->volShowParam.ResetDirty();
    }

    // Uni grid data base
    if (this->rmTexParam.IsDirty()) {
        this->rmTexParam.ResetDirty();
        this->rmTex = static_cast<RayMarchTex>(this->rmTexParam.Param<core::param::EnumParam>()->Value());
    }

    // Change step size for raycasting
    if (this->volDeltaParam.IsDirty()) {
        this->volDelta = this->volDeltaParam.Param<core::param::FloatParam>()->Value();
        this->volDeltaParam.ResetDirty();
    }

    // Change isovalue for raycasting
    if (this->volIsoValParam.IsDirty()) {
        this->volIsoVal = this->volIsoValParam.Param<core::param::FloatParam>()->Value();
        this->volIsoValParam.ResetDirty();
        if(this->rmTex == DENSITY) {
            this->recalcDensityGrid = true;
        }
    }

    // Scale factor for volume alpha value
    if (this->volAlphaSclParam.IsDirty()) {
        this->volAlphaScl = this->volAlphaSclParam.Param<core::param::FloatParam>()->Value();
        this->volAlphaSclParam.ResetDirty();
    }

    // Show iso surface
    if (this->showIsoSurfParam.IsDirty()) {
        this->showIsoSurf = this->showIsoSurfParam.Param<core::param::BoolParam>()->Value();
        this->showIsoSurfParam.ResetDirty();
    }

    // LIC direction vector scale param for isosurface
    if (this->volLicDirSclParam.IsDirty()) {
        this->volLicDirScl = this->volLicDirSclParam.Param<core::param::FloatParam>()->Value();
        this->volLicDirSclParam.ResetDirty();
    }

    // LIC stream line length for isosurface
    if (this->volLicLenParam.IsDirty()) {
        this->volLicLen = this->volLicLenParam.Param<core::param::IntParam>()->Value();
        this->volLicLenParam.ResetDirty();
    }

    // LIC contrast stretching
    if (this->volLicContrastStretchingParam.IsDirty()) {
        this->volLicContrastStretching = this->volLicContrastStretchingParam.Param<core::param::FloatParam>()->Value();
        this->volLicContrastStretchingParam.ResetDirty();
    }

    // LIC brightness
    if (this->volLicBrightParam.IsDirty()) {
        this->volLicBright = this->volLicBrightParam.Param<core::param::FloatParam>()->Value();
        this->volLicBrightParam.ResetDirty();
    }

    // LIC texture coordinates
    if (this->volLicTCSclParam.IsDirty()) {
        this->volLicTCScl = this->volLicTCSclParam.Param<core::param::FloatParam>()->Value();
        this->volLicTCSclParam.ResetDirty();
    }


    // Fog

    // Fog start
    if (this->fogStartParam.IsDirty()) {
        this->fogStart = this->fogStartParam.Param<core::param::FloatParam>()->Value();
        this->fogStartParam.ResetDirty();
        glFogf(GL_FOG_START, this->fogStart);
    }

    // Fog end
    if (this->fogEndParam.IsDirty()) {
        this->fogEnd = this->fogEndParam.Param<core::param::FloatParam>()->Value();
        this->fogEndParam.ResetDirty();
        glFogf(GL_FOG_END, this->fogEnd);
    }

    // Fog density
    if (this->fogDensityParam.IsDirty()) {
        this->fogDensity = this->fogDensityParam.Param<core::param::FloatParam>()->Value();
        this->fogDensityParam.ResetDirty();
        glFogf(GL_FOG_DENSITY, this->fogDensity);
    }

    // Fog colour
    if (this->fogColourParam.IsDirty()) {
        this->fogColourParam.ResetDirty();
        core::utility::ColourParser::FromString(this->fogColourParam.Param<core::param::StringParam>()->Value(),
            this->fogColour[0], fogColour[1], fogColour[2]);
        glFogfv(GL_FOG_COLOR, this->fogColour);
    }

    // VTK mesh file
    if( this->meshFileParam.IsDirty() ) {
        this->renderMesh = this->loadVTKMesh( this->meshFileParam.Param<core::param::FilePathParam>()->Value());
        this->meshFileParam.ResetDirty();
    }

    return true;
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::calcCellVolume
 */
float protein_cuda::CrystalStructureVolumeRenderer::calcCellVolume(
        vislib::math::Vector<float, 3> A,
        vislib::math::Vector<float, 3> B,
        vislib::math::Vector<float, 3> C,
        vislib::math::Vector<float, 3> D,
        vislib::math::Vector<float, 3> E,
        vislib::math::Vector<float, 3> F,
        vislib::math::Vector<float, 3> G,
        vislib::math::Vector<float, 3> H) {

    float res = 0.0f;

    res += this->calcVolTetrahedron(A, B, C, E);
    res += this->calcVolTetrahedron(D, C, B, H);
    res += this->calcVolTetrahedron(G, C, H, E);
    res += this->calcVolTetrahedron(F, B, H, E);
    res += this->calcVolTetrahedron(H, C, B, E);

    return res;
}


/*
 * protein_cuda::CrystalStructureVolumeRenderer::calcCellTetrahedron
 */
float protein_cuda::CrystalStructureVolumeRenderer::calcVolTetrahedron(
        vislib::math::Vector<float, 3> A,
        vislib::math::Vector<float, 3> B,
        vislib::math::Vector<float, 3> C,
        vislib::math::Vector<float, 3> D) {

    using namespace vislib;
    using namespace vislib::math;

    Vector <float, 3> vecA = B-A;
    Vector <float, 3> vecB = C-A;
    Vector <float, 3> vecC = D-A;

    float res = fabs(vecA.Cross(vecB).Dot(vecC));
    res /= 6.0f;
    return res;
}

/*
 * protein_cuda::CrystalStructureVolumeRenderer::getColorGrad
 */
vislib::math::Vector<float, 3> protein_cuda::CrystalStructureVolumeRenderer::getColorGrad(float val) {
    using namespace vislib::math;

    Vector<float, 3> colBlue(0.23f, 0.29f, 0.75f); // blue
    Vector<float, 3> colWhite(1.0f, 1.0f, 1.0f);   // white
    Vector<float, 3> colRed(0.75f, 0.01f, 0.15f);  // red

    //printf("min %f, max %f\n", this->vecMagSclMin, vecMagSclMax); // DEBUG
    //printf("val input %f, ", val); // DEBUG

    val -= this->vecMagSclMin;
    val /= (this->vecMagSclMax - this->vecMagSclMin);

    //printf("val output %f, ", val); // DEBUG

    // Calc color
    Vector<float, 3> colRes;
    if(val < 0.5f) {
        val *= 2.0f;
        colRes = val*colWhite + (1.0f - val)*colBlue;
    }
    else {
        val -= 0.5;
        val *= 2.0f;
        colRes = val*colRed + (1.0f - val)*colWhite;
    }
    //printf("color output (%f %f %f)\n ", colRes.X(), colRes.Y(), colRes.Z()); // DEBUG

    return colRes;
}

bool protein_cuda::CrystalStructureVolumeRenderer::loadVTKMesh( vislib::StringA filename ) {
    // double colR, double colG, double colB, bool flipNormals, bool smoothNormals)
    //
    // reads VTK format, from Sadlo's read_geom_VTK()
    double scale = 1.0;
    double transX = 0.0;
    double transY = 0.0;
    double transZ = 0.0;

    int stride = 1;
    int first = 0;
    int last = -1;

    double colR = 0.75;
    double colG = 0.75;
    double colB = 0.75;
    bool flipNormals = false;
    bool smoothNormals = true;

    FILE *fp = fopen( filename.PeekBuffer(), "r");
    if (!fp) {
        printf("could not open input file %s\n", filename.PeekBuffer());
        return 1; // fail
    }

    const int bufSize = 4096;
    char buf1[bufSize], buf2[bufSize], buf3[bufSize], buf4[bufSize], buf5[bufSize], buf6[bufSize];
    fgets(buf1, bufSize, fp);
    sscanf(buf1,"%s%s%s%s%s", buf2, buf3, buf4, buf5, buf6);

    if (strcmp(buf2, "#") != 0 ||
            strcmp(buf3, "vtk") != 0 ||
            strcmp(buf4, "DataFile") != 0 ||
            strcmp(buf5, "Version") != 0 ||
            (strcmp(buf6, "1.0") != 0 && strcmp(buf6, "2.0"))) {
        printf("unsupported file format\n");
        fclose(fp);
        return 1; // fail
    }

    // skip description // ### this fails if description longer than bufSize
    fgets(buf1, bufSize, fp);
    if (strncmp(buf1,"Description", strlen("Description")) != 0) {
        printf("missing Description entry, aborting\n");
        fclose(fp);
        return 1; // fail
    }

    fgets(buf1, bufSize, fp);
    if (strncmp(buf1, "ASCII", strlen("ASCII") != 0)) {
        printf("not ASCII type, aborting\n");
        fclose(fp);
        return 1; // fail
    }

    fgets(buf1, bufSize, fp);
    // ### HACK, skipping empty lines only here
    while (strlen(buf1) == 1) fgets(buf1, bufSize, fp);
    if (strncmp(buf1, "DATASET POLYDATA", strlen("DATASET POLYDATA") != 0)) {
        printf("not \"DATASET POLYDATA\" type, aborting\n");
        fclose(fp);
        return 1; // fail
    }

    int numVertices;
    fgets(buf1, bufSize, fp);
    sscanf(buf1, "%s%d%s", buf2, &numVertices, buf3);
    if (strncmp(buf2, "POINTS", strlen("POINTS") != 0) ||
            strncmp(buf3, "float", strlen("float") != 0)) {
        printf("expecting POINTS of type \"float\", aborting\n");
        fclose(fp);
        return 1; // fail
    }

    // read vertices
    meshVertices.SetCount( 0);
    meshVertices.SetCapacityIncrement( 1000);
    meshVertices.Resize( numVertices * 3);

    for (int v=0; v<numVertices; v++) {
        fgets(buf1, bufSize, fp);
        double verts[3];
        sscanf(buf1, "%lf%lf%lf", verts+0, verts+1, verts+2);
        meshVertices.Add(verts[0]);
        meshVertices.Add(verts[1]);
        meshVertices.Add(verts[2]);
    }

    //printf("read %d vertices\n", numVertices);


    // read lines, if any
    //std::vector<std::vector<int> > lines;
    int numLines = 0;
    int linesEntries = 0; // this is the complete connectivity info of lines, incl. the counts of vertices per line
    fgets(buf1, bufSize, fp);
    // ### HACK, skipping empty lines only here
    while (strlen(buf1) == 1) fgets(buf1, bufSize, fp);
    sscanf(buf1, "%s%d%d", buf2, &numLines, &linesEntries);
    if (strncmp(buf2, "LINES", strlen("LINES")) != 0) {
        numLines = 0;
    }

    if (numLines > 0) {
        for (int l=0; l<numLines; l++) {
            //std::vector<int> line;
            int lineSize;
            fscanf(fp, "%d", &lineSize);
            for (int e=0; e<lineSize; e++) {
                int w;
                fscanf(fp, "%d", &w);
                //line.push_back(w);
            }
            if (l % stride == 0 && (first < 0 || l >= first) && (last < 0 || l <= last)) {
                //lines.push_back(line);
            }
        }
    }

    // read polygons, if any
    //std::vector<std::vector<int> > polygons;
    meshFaces.SetCount( 0);
    meshFaces.SetCapacityIncrement( 1000);

    int numPolygons = 0;
    int polygonsEntries = 0; // this is the complete connectivity info of polygons, incl. the counts of vertices per polygon
    if (numLines > 0) { // if no lines read, buf1 already contains line
        buf1[0] = '\0';
        fgets(buf1, bufSize, fp);
        // ### HACK, skipping empty lines only here
        while (strlen(buf1) == 1) {
            buf1[0] = '\0';
            fgets(buf1, bufSize, fp);
        }
    }
    sscanf(buf1, "%s%d%d", buf2, &numPolygons, &polygonsEntries);
    if (strncmp(buf2, "POLYGONS", strlen("POLYGONS")) != 0) {
        numPolygons = 0;
    }

    meshFaces.Resize( numPolygons * 3);

    if (numPolygons > 0) {
        for (int p=0; p<numPolygons; p++) {
            std::vector<int> polygon;
            int polygonSize;
            fscanf(fp, "%d", &polygonSize);
            for (int e=0; e<polygonSize; e++) {
                if( e > 2 ) continue;
                int w;
                fscanf(fp, "%d", &w);
                //polygon.push_back(w);
                meshFaces.Add( w);
            }
            //polygons.push_back(polygon);
        }
    }

    fclose(fp);

    // skipped lines

    // polygons
    if (numPolygons > 0) {

        //float *normals = new float[vertices.size()];
        this->meshNormals.SetCount( meshVertices.Count());

        for (int i=0; i<meshVertices.Count(); i++) {
            meshNormals[i] = 0.0;
        }


        for (int p=0; p<numPolygons; p++) {

            float normVerts[9];
            normVerts[0] = static_cast<float>(transX + scale * meshVertices[meshFaces[p*3+0]*3+0]);
            normVerts[1] = static_cast<float>(transY + scale * meshVertices[meshFaces[p*3+0]*3+1]);
            normVerts[2] = static_cast<float>(transZ + scale * meshVertices[meshFaces[p*3+0]*3+2]);
            normVerts[3] = static_cast<float>(transX + scale * meshVertices[meshFaces[p*3+1]*3+0]);
            normVerts[4] = static_cast<float>(transY + scale * meshVertices[meshFaces[p*3+1]*3+1]);
            normVerts[5] = static_cast<float>(transZ + scale * meshVertices[meshFaces[p*3+1]*3+2]);
            normVerts[6] = static_cast<float>(transX + scale * meshVertices[meshFaces[p*3+2]*3+0]);
            normVerts[7] = static_cast<float>(transY + scale * meshVertices[meshFaces[p*3+2]*3+1]);
            normVerts[8] = static_cast<float>(transZ + scale * meshVertices[meshFaces[p*3+2]*3+2]);

            float norm[3];
            {
                float v1[3], v2[3];
                v1[0] = normVerts[1*3 + 0] - normVerts[0*3 + 0];
                v1[1] = normVerts[1*3 + 1] - normVerts[0*3 + 1];
                v1[2] = normVerts[1*3 + 2] - normVerts[0*3 + 2];

                v2[0] = normVerts[2*3 + 0] - normVerts[0*3 + 0];
                v2[1] = normVerts[2*3 + 1] - normVerts[0*3 + 1];
                v2[2] = normVerts[2*3 + 2] - normVerts[0*3 + 2];

                norm[0] = (v1[1] * v2[2]) - (v1[2] * v2[1]);
                norm[1] = (v1[2] * v2[0]) - (v1[0] * v2[2]);
                norm[2] = (v1[0] * v2[1]) - (v1[1] * v2[0]);

                double len = sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);

                if (flipNormals) {
                    len *= -1.0;
                }

                norm[0] /= static_cast<float>(len);
                norm[1] /= static_cast<float>(len);
                norm[2] /= static_cast<float>(len);
            }

            // ###### HACK neglecting normals of non-triangles
            meshNormals[meshFaces[p*3+0]*3+0] += norm[0];
            meshNormals[meshFaces[p*3+0]*3+1] += norm[1];
            meshNormals[meshFaces[p*3+0]*3+2] += norm[2];

            meshNormals[meshFaces[p*3+1]*3+0] += norm[0];
            meshNormals[meshFaces[p*3+1]*3+1] += norm[1];
            meshNormals[meshFaces[p*3+1]*3+2] += norm[2];

            meshNormals[meshFaces[p*3+2]*3+0] += norm[0];
            meshNormals[meshFaces[p*3+2]*3+1] += norm[1];
            meshNormals[meshFaces[p*3+2]*3+2] += norm[2];
        }

        for (int p=0; p<numPolygons; p++) {
            for (int i=0; i<3; i++) { // ### hack
                double len = sqrt(
                    meshNormals[meshFaces[p*3+i]*3+0] * meshNormals[meshFaces[p*3+i]*3+0] +
                    meshNormals[meshFaces[p*3+i]*3+1] * meshNormals[meshFaces[p*3+i]*3+1] +
                    meshNormals[meshFaces[p*3+i]*3+2] * meshNormals[meshFaces[p*3+i]*3+2]);

                meshNormals[meshFaces[p*3+i]*3+0] /= len;
                meshNormals[meshFaces[p*3+i]*3+1] /= len;
                meshNormals[meshFaces[p*3+i]*3+2] /= len;
            }

        }

        /*
        glColor3f(colR, colG, colB);
        glBegin(GL_TRIANGLES);
        for (int p=0; p<meshFaces.Count() / 3; p++) {
            for (int e=0; e < 3; e++) {
                glNormal3f(meshNormals[meshFaces[p*3+e]*3+0],
                    meshNormals[meshFaces[p*3+e]*3+1],
                    meshNormals[meshFaces[p*3+e]*3+2]);
                glVertex3f(meshVertices[meshFaces[p*3+e]*3+0],
                    meshVertices[meshFaces[p*3+e]*3+1],
                    meshVertices[meshFaces[p*3+e]*3+2]);
            }
        }
        glEnd();
        */
    }

    return true;
}
