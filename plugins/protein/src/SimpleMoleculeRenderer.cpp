/*
 * SimpleMoleculeRenderer.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "vislib/graphics/gl/IncludeAllGL.h"

#include <GL/glu.h>
#include <omp.h>
#include "Color.h"
#include "SimpleMoleculeRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/utility/ShaderFactory.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/utility/sys/ASCIIFileBuffer.h"
#include "mmcore/view/light/DistantLight.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/Trace.h"
#include "vislib/assert.h"
#include "vislib/graphics/gl/AbstractOpenGLShader.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/Quaternion.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;
using namespace megamol::core::utility::log;

//#define OLD_SHADERS

/*
 * protein::SimpleMoleculeRenderer::SimpleMoleculeRenderer (CTOR)
 */
SimpleMoleculeRenderer::SimpleMoleculeRenderer(void)
        : view::Renderer3DModuleGL()
        , molDataCallerSlot("getData", "Connects the molecule rendering with molecule data storage")
        , bsDataCallerSlot("getBindingSites", "Connects the molecule rendering with binding site data storage")
        , getLightsSlot("getLights", "Connects the molecule rendering with availabel light sources")
        , colorTableFileParam("color::colorTableFilename", "The filename of the color table.")
        , coloringModeParam0("color::coloringMode0", "The first coloring mode.")
        , coloringModeParam1("color::coloringMode1", "The second coloring mode.")
        , cmWeightParam("color::colorWeighting", "The weighting of the two coloring modes.")
        , renderModeParam("renderMode", "The rendering mode.")
        , stickRadiusParam("stickRadius", "The radius for stick rendering")
        , probeRadiusParam("probeRadius", "The probe radius for SAS rendering")
        , minGradColorParam("color::minGradColor", "The color for the minimum value for gradient coloring")
        , midGradColorParam("color::midGradColor", "The color for the middle value for gradient coloring")
        , maxGradColorParam("color::maxGradColor", "The color for the maximum value for gradient coloring")
        , molIdxListParam("molIdxList", "The list of molecule indices for RS computation:")
        , specialColorParam("color::specialColor", "The color for the specified molecules")
        , interpolParam("posInterpolation", "Enable positional interpolation between frames")
        , offscreenRenderingParam("offscreenRendering", "Toggle offscreenRendering")
        , toggleZClippingParam("toggleZClip", "...")
        , clipPlaneTimeOffsetParam("clipPlane::timeOffset", "...")
        , clipPlaneDurationParam("clipPlane::Duration", "...")
        , useNeighborColors("color::neighborhood", "Add the color of the neighborhood to the own")
        , currentZClipPos(-20) {
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->molDataCallerSlot.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->molDataCallerSlot);
    this->getLightsSlot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->getLightsSlot.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->getLightsSlot);
    this->bsDataCallerSlot.SetCompatibleCall<BindingSiteCallDescription>();
    this->MakeSlotAvailable(&this->bsDataCallerSlot);

    // fill color table with default values and set the filename param
    std::string filename("colors.txt");
    Color::ReadColorTableFromFile(filename, this->colorLookupTable);
    this->colorTableFileParam.SetParameter(
        new param::FilePathParam(filename, core::param::FilePathParam::FilePathFlags_::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&this->colorTableFileParam);

    // coloring modes
    this->currentColoringMode0 = Color::CHAIN;
    this->currentColoringMode1 = Color::ELEMENT;
    param::EnumParam* cm0 = new param::EnumParam(int(this->currentColoringMode0));
    param::EnumParam* cm1 = new param::EnumParam(int(this->currentColoringMode1));
    MolecularDataCall* mol = new MolecularDataCall();
    BindingSiteCall* bs = new BindingSiteCall();
    unsigned int cCnt;
    Color::ColoringMode cMode;
    for (cCnt = 0; cCnt < Color::GetNumOfColoringModes(mol, bs); ++cCnt) {
        cMode = Color::GetModeByIndex(mol, bs, cCnt);
        cm0->SetTypePair(cMode, Color::GetName(cMode).c_str());
        cm1->SetTypePair(cMode, Color::GetName(cMode).c_str());
    }
    delete mol;
    delete bs;
    this->coloringModeParam0 << cm0;
    this->coloringModeParam1 << cm1;
    this->MakeSlotAvailable(&this->coloringModeParam0);
    this->MakeSlotAvailable(&this->coloringModeParam1);

    // Color weighting parameter
    this->cmWeightParam.SetParameter(new param::FloatParam(0.5f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->cmWeightParam);

    this->useNeighborColors.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->useNeighborColors);

    // rendering mode
    // this->currentRenderMode = LINES;
    // this->currentRenderMode = STICK;
    // this->currentRenderMode = BALL_AND_STICK;
    this->currentRenderMode = SPACEFILLING;
    // this->currentRenderMode = SAS;
    param::EnumParam* rm = new param::EnumParam(int(this->currentRenderMode));
    rm->SetTypePair(LINES, "Lines");
    rm->SetTypePair(LINES_FILTER, "Lines-Filter");
    rm->SetTypePair(STICK, "Stick");
    rm->SetTypePair(STICK_FILTER, "Stick-Filter");
    rm->SetTypePair(BALL_AND_STICK, "Ball-and-Stick");
    rm->SetTypePair(SPACEFILLING, "Spacefilling");
    rm->SetTypePair(SPACEFILL_FILTER, "Spacefilling-Filter");
    rm->SetTypePair(SAS, "SAS");
    this->renderModeParam << rm;
    this->MakeSlotAvailable(&this->renderModeParam);

    // fill color table with default values and set the filename param
    this->stickRadiusParam.SetParameter(new param::FloatParam(0.3f, 0.0f));
    this->MakeSlotAvailable(&this->stickRadiusParam);

    // fill color table with default values and set the filename param
    this->probeRadiusParam.SetParameter(new param::FloatParam(1.4f));
    this->MakeSlotAvailable(&this->probeRadiusParam);

    // the color for the minimum value (gradient coloring
    this->minGradColorParam.SetParameter(new param::ColorParam("#146496"));
    this->MakeSlotAvailable(&this->minGradColorParam);

    // the color for the middle value (gradient coloring
    this->midGradColorParam.SetParameter(new param::ColorParam("#f0f0f0"));
    this->MakeSlotAvailable(&this->midGradColorParam);

    // the color for the maximum value (gradient coloring
    this->maxGradColorParam.SetParameter(new param::ColorParam("#ae3b32"));
    this->MakeSlotAvailable(&this->maxGradColorParam);

    // molecular indices list param
    this->molIdxList.Clear();
    this->molIdxListParam.SetParameter(new param::StringParam(""));
    this->MakeSlotAvailable(&this->molIdxListParam);

    // the color for the maximum value (gradient coloring
    this->specialColorParam.SetParameter(new param::ColorParam("#228B22"));
    this->MakeSlotAvailable(&this->specialColorParam);

    // en-/disable positional interpolation
    this->interpolParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->interpolParam);

    // make the rainbow color table
    Color::MakeRainbowColorTable(100, this->rainbowColors);

    // Toggle offscreen rendering
    this->offscreenRenderingParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->offscreenRenderingParam);

    // Toggle Z-Clipping
    this->toggleZClippingParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->toggleZClippingParam);


    this->clipPlaneTimeOffsetParam.SetParameter(new param::FloatParam(100.0f));
    this->MakeSlotAvailable(&this->clipPlaneTimeOffsetParam);

    this->clipPlaneDurationParam.SetParameter(new param::FloatParam(40.0f));
    this->MakeSlotAvailable(&this->clipPlaneDurationParam);

    this->lastDataHash = 0;
}

/*
 * protein::SimpleMoleculeRenderer::~SimpleMoleculeRenderer (DTOR)
 */
SimpleMoleculeRenderer::~SimpleMoleculeRenderer(void) {
    this->Release();
}

/*
 * protein::SimpleMoleculeRenderer::release
 */
void SimpleMoleculeRenderer::release(void) {}

/*
 * protein::SimpleMoleculeRenderer::create
 */
bool SimpleMoleculeRenderer::create(void) {
    if (!ogl_IsVersionGEQ(2, 0))
        return false;

    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions())
        return false;

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertSrc;
    ShaderSource fragSrc;

    // Load sphere shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    try {
        if (!this->sphereShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    // Load sphere shader for offscreen rendering
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereFragmentOR", fragSrc)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to load vertex shader source for offscreen rendering sphere shader");
        return false;
    }
    try {
        if (!this->sphereShaderOR.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create offscreen sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    // Load filter sphere shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::filterSphereVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for filter sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for filter sphere shader");
        return false;
    }
    try {
        if (!this->filterSphereShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create filter sphere shader: %s\n", e.GetMsgA());
        return false;
    }
    // Load filter sphere shader for offscreen rendering
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereFragmentOR", fragSrc)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to load vertex shader source for offscreen filter sphere shader (OR)");
        return false;
    }
    try {
        if (!this->filterSphereShaderOR.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create filter sphere shader (OR): %s\n", e.GetMsgA());
        return false;
    }

    // Load clip plane sphere shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "protein::std::sphereClipPlaneVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for clip plane sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "protein::std::sphereClipPlaneFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for filter sphere shader");
        return false;
    }
    try {
        if (!this->sphereClipPlaneShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create clip plane sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    // Load cylinder shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::cylinderVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for cylinder shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::cylinderFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for cylinder shader");
        return false;
    }
    try {
        if (!this->cylinderShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create cylinder shader: %s\n", e.GetMsgA());
        return false;
    }

    // Load cylinder shader for offscreen rendering
    fragSrc.Clear();
    vertSrc.Clear();
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::cylinderVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for offscreen cylinder shader");
        return false;
    }
    ///

    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::cylinderFragmentOR", fragSrc)) {
        //        "protein::std::cylinderFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to load fragment shader source for offscreen cylinder shader");
        return false;
    }
    try {
        if (!this->cylinderShaderOR.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create offscreen cylinder shader: %s\n", e.GetMsgA());
        return false;
    }

    // Load filter cylinder shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "protein::std::filterCylinderVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for filter cylinder shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::cylinderFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for filter cylinder shader");
        return false;
    }
    try {
        if (!this->filterCylinderShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create filter cylinder shader: %s\n", e.GetMsgA());
        return false;
    }
    // Load filter cylinder shader offscreen rendering
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::cylinderFragmentOR", fragSrc)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to load vertex shader source for filter cylinder shader (OR)");
        return false;
    }
    try {
        if (!this->filterCylinderShaderOR.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create filter cylinder shader (OR): %s\n", e.GetMsgA());
        return false;
    }

    // Load clip plane sphere shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::cylinderVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to load vertex shader source for clip plane cylinder shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "protein::std::cylinderClipPlaneFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to load vertex shader source for clip plane cylinder shader");
        return false;
    }
    try {
        if (!this->cylinderClipPlaneShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create clip plane cylinder shader: %s\n", e.GetMsgA());
        return false;
    }

    // new shaders
    try {
        auto const shdr_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

        sphereShader_ = core::utility::make_shared_glowl_shader("sphere", shdr_options,
            std::filesystem::path("simplemolecule/sm_sphere.vert.glsl"),
            std::filesystem::path("simplemolecule/sm_sphere.frag.glsl"));

        sphereClipPlaneShader_ = core::utility::make_shared_glowl_shader("sphereClipPlane", shdr_options,
            std::filesystem::path("simplemolecule/sm_sphere.vert.glsl"),
            std::filesystem::path("simplemolecule/sm_sphere_clipplane.frag.glsl"));

        filterSphereShader_ = core::utility::make_shared_glowl_shader("sphereFilter", shdr_options,
            std::filesystem::path("simplemolecule/sm_sphere_filter.vert.glsl"),
            std::filesystem::path("simplemolecule/sm_sphere.frag.glsl"));

        cylinderShader_ = core::utility::make_shared_glowl_shader("cylinder", shdr_options,
            std::filesystem::path("simplemolecule/sm_cylinder.vert.glsl"),
            std::filesystem::path("simplemolecule/sm_cylinder.frag.glsl"));

        cylinderClipPlaneShader_ = core::utility::make_shared_glowl_shader("cylinderClipPlane", shdr_options,
            std::filesystem::path("simplemolecule/sm_cylinder.vert.glsl"),
            std::filesystem::path("simplemolecule/sm_cylinder_clipplane.frag.glsl"));

        filterCylinderShader_ = core::utility::make_shared_glowl_shader("cylinderFilter", shdr_options,
            std::filesystem::path("simplemolecule/sm_cylinder_filter.vert.glsl"),
            std::filesystem::path("simplemolecule/sm_cylinder.frag.glsl"));

        // TODO shaders for the second render pass

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "[SimpleMoleculeRenderer] %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[SimpleMoleculeRenderer] Unable to compile shader: Unknown exception");
        return false;
    }

    return true;
}


/*
 * protein::SimpleMoleculeRenderer::GetExtents
 */
bool SimpleMoleculeRenderer::GetExtents(core::view::CallRender3DGL& call) {
    MolecularDataCall* mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == NULL)
        return false;
    if (!(*mol)(MolecularDataCall::CallForGetExtent))
        return false;

    call.AccessBoundingBoxes().SetBoundingBox(mol->AccessBoundingBoxes().ObjectSpaceBBox());
    call.AccessBoundingBoxes().SetBoundingBox(mol->AccessBoundingBoxes().ObjectSpaceClipBox());

    call.SetTimeFramesCount(mol->FrameCount());

    return true;
}

/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * protein::SimpleMoleculeRenderer::Render
 */
bool SimpleMoleculeRenderer::Render(core::view::CallRender3DGL& call) {

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    float callTime = call.Time();

    cam = call.GetCamera();

    // matrices
    view = cam.getViewMatrix();
    proj = cam.getProjectionMatrix();
    MVinv = glm::inverse(view);
    NormalM = glm::transpose(MVinv);
    MVP = proj * view;
    MVPinv = glm::inverse(MVP);
    MVPtransp = glm::transpose(MVP);

    // get viewpoint parameters for raycasting
    this->viewportStuff[0] = 0.0f;
    this->viewportStuff[1] = 0.0f;
    this->viewportStuff[2] = call.GetFramebuffer()->getWidth();
    this->viewportStuff[3] = call.GetFramebuffer()->getHeight();
    if (this->viewportStuff[2] < 1.0f)
        this->viewportStuff[2] = 1.0f;
    if (this->viewportStuff[3] < 1.0f)
        this->viewportStuff[3] = 1.0f;
    this->viewportStuff[2] = 2.0f / this->viewportStuff[2];
    this->viewportStuff[3] = 2.0f / this->viewportStuff[3];

    // get pointer to MolecularDataCall
    MolecularDataCall* mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == NULL)
        return false;

    // get pointer to BindingSiteCall
    BindingSiteCall* bs = this->bsDataCallerSlot.CallAs<BindingSiteCall>();
    if (bs) {
        (*bs)(BindingSiteCall::CallForGetData);
    }

    int cnt;

    this->currentZClipPos = mol->AccessBoundingBoxes().ObjectSpaceBBox().Back() +
                            (mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth()) *
                                ((callTime - this->clipPlaneTimeOffsetParam.Param<param::FloatParam>()->Value()) /
                                    this->clipPlaneDurationParam.Param<param::FloatParam>()->Value());

    // set call time
    mol->SetCalltime(callTime);
    // set frame ID and call data
    mol->SetFrameID(static_cast<int>(callTime));

    if (!(*mol)(MolecularDataCall::CallForGetData))
        return false;
    // check if atom count is zero
    if (mol->AtomCount() == 0)
        return true;
    // get positions of the first frame
    float* pos0 = new float[mol->AtomCount() * 3];
    memcpy(pos0, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof(float));
    // set next frame ID and get positions of the second frame
    if (((static_cast<int>(callTime) + 1) < int(mol->FrameCount())) &&
        this->interpolParam.Param<param::BoolParam>()->Value())
        mol->SetFrameID(static_cast<int>(callTime) + 1);
    else
        mol->SetFrameID(static_cast<int>(callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData)) {
        delete[] pos0;
        return false;
    }
    float* pos1 = new float[mol->AtomCount() * 3];
    memcpy(pos1, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof(float));

    // interpolate atom positions between frames
    float* posInter = new float[mol->AtomCount() * 3];
    float inter = callTime - static_cast<float>(static_cast<int>(callTime));
    float threshold = vislib::math::Min(mol->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
                          vislib::math::Min(mol->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
                              mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) *
                      0.75f;
#pragma omp parallel for
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        if (std::sqrt(std::pow(pos0[3 * cnt + 0] - pos1[3 * cnt + 0], 2) +
                      std::pow(pos0[3 * cnt + 1] - pos1[3 * cnt + 1], 2) +
                      std::pow(pos0[3 * cnt + 2] - pos1[3 * cnt + 2], 2)) < threshold) {
            posInter[3 * cnt + 0] = (1.0f - inter) * pos0[3 * cnt + 0] + inter * pos1[3 * cnt + 0];
            posInter[3 * cnt + 1] = (1.0f - inter) * pos0[3 * cnt + 1] + inter * pos1[3 * cnt + 1];
            posInter[3 * cnt + 2] = (1.0f - inter) * pos0[3 * cnt + 2] + inter * pos1[3 * cnt + 2];
        } else if (inter < 0.5f) {
            posInter[3 * cnt + 0] = pos0[3 * cnt + 0];
            posInter[3 * cnt + 1] = pos0[3 * cnt + 1];
            posInter[3 * cnt + 2] = pos0[3 * cnt + 2];
        } else {
            posInter[3 * cnt + 0] = pos1[3 * cnt + 0];
            posInter[3 * cnt + 1] = pos1[3 * cnt + 1];
            posInter[3 * cnt + 2] = pos1[3 * cnt + 2];
        }
    }

    // ---------- update parameters ----------
    this->UpdateParameters(mol, bs);

    // recompute color table, if necessary
    if (this->atomColorTable.Count() / 3 < mol->AtomCount()) {

        // Mix two coloring modes
        Color::MakeColorTable(mol, this->currentColoringMode0, this->currentColoringMode1,
            cmWeightParam.Param<param::FloatParam>()->Value(),        // weight for the first cm
            1.0f - cmWeightParam.Param<param::FloatParam>()->Value(), // weight for the second cm
            this->atomColorTable, this->colorLookupTable, this->rainbowColors,
            this->minGradColorParam.Param<param::ColorParam>()->Value(),
            this->midGradColorParam.Param<param::ColorParam>()->Value(),
            this->maxGradColorParam.Param<param::ColorParam>()->Value(), true, bs,
            this->useNeighborColors.Param<param::BoolParam>()->Value());
    }

    // ---------- special color handling ... -----------
    unsigned int midx, ridx, rcnt, aidx, acnt;
    auto specCol = this->specialColorParam.Param<param::ColorParam>()->Value();
    for (unsigned int mi = 0; mi < this->molIdxList.Count(); ++mi) {
        midx = atoi(this->molIdxList[mi]);
        ridx = mol->Molecules()[midx].FirstResidueIndex();
        rcnt = ridx + mol->Molecules()[midx].ResidueCount();
        for (unsigned int ri = ridx; ri < rcnt; ++ri) {
            aidx = mol->Residues()[ri]->FirstAtomIndex();
            acnt = aidx + mol->Residues()[ri]->AtomCount();
            for (unsigned int ai = aidx; ai < acnt; ++ai) {
                this->atomColorTable[3 * ai + 0] = specCol[0];
                this->atomColorTable[3 * ai + 1] = specCol[1];
                this->atomColorTable[3 * ai + 2] = specCol[2];
            }
        }
    }
    // ---------- ... special color handling -----------

    // TODO: ---------- render ----------

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // render data using the current rendering mode
    if (this->currentRenderMode == LINES) {
        this->RenderLines(mol, posInter);
    } else if (this->currentRenderMode == STICK) {
        if (this->toggleZClippingParam.Param<param::BoolParam>()->Value()) {
            this->RenderStickClipPlane(mol, posInter);
        } else {
            this->RenderStick(mol, posInter);
        }
    } else if (this->currentRenderMode == BALL_AND_STICK) {
        this->RenderBallAndStick(mol, posInter);
    } else if (this->currentRenderMode == SPACEFILLING) {
        if (this->toggleZClippingParam.Param<param::BoolParam>()->Value()) {
            this->RenderSpacefillingClipPlane(mol, posInter);
        } else {
            this->RenderSpacefilling(mol, posInter);
        }
    } else if (this->currentRenderMode == SPACEFILL_FILTER) {
        this->RenderSpacefillingFilter(mol, posInter);
    } else if (this->currentRenderMode == SAS) {
        this->RenderSAS(mol, posInter);
    } else if (this->currentRenderMode == LINES_FILTER) {
        this->RenderLinesFilter(mol, posInter);
    } else if (this->currentRenderMode == STICK_FILTER) {
        this->RenderStickFilter(mol, posInter);
    }

    delete[] pos0;
    delete[] pos1;
    delete[] posInter;

    // unlock the current frame
    mol->Unlock();

    return true;
}

/*
 * render the atom using lines and points
 */
void SimpleMoleculeRenderer::RenderLines(const MolecularDataCall* mol, const float* atomPos) {
    glDisable(GL_LIGHTING);
    glLineWidth(2.0f);
    // ----- draw atoms as points -----
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    // set vertex and color pointers and draw them
    glVertexPointer(3, GL_FLOAT, 0, atomPos);
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    glDrawArrays(GL_POINTS, 0, mol->AtomCount());
    // disable sphere shader
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    // ----- draw bonds as lines -----
    unsigned int cnt, atomIdx0, atomIdx1;
    glBegin(GL_LINES);
    for (cnt = 0; cnt < mol->ConnectionCount(); ++cnt) {
        // get atom indices
        atomIdx0 = mol->Connection()[2 * cnt + 0];
        atomIdx1 = mol->Connection()[2 * cnt + 1];
        // distance check
        if ((vislib::math::Vector<float, 3>(&atomPos[atomIdx0 * 3]) -
                vislib::math::Vector<float, 3>(&atomPos[atomIdx1 * 3]))
                .Length() > 3.0f)
            continue;
        // set colors and vertices of first atom
        glColor3fv(&this->atomColorTable[atomIdx0 * 3]);
        glVertex3f(atomPos[atomIdx0 * 3 + 0], atomPos[atomIdx0 * 3 + 1], atomPos[atomIdx0 * 3 + 2]);
        // set colors and vertices of second atom
        glColor3fv(&this->atomColorTable[atomIdx1 * 3]);
        glVertex3f(atomPos[atomIdx1 * 3 + 0], atomPos[atomIdx1 * 3 + 1], atomPos[atomIdx1 * 3 + 2]);
    }
    glEnd(); // GL_LINES
    glEnable(GL_LIGHTING);
}

/*
 * Render the molecular data in stick mode.
 */
void SimpleMoleculeRenderer::RenderStick(const MolecularDataCall* mol, const float* atomPos) {
    // ----- prepare stick raycasting -----
    this->vertSpheres.SetCount(mol->AtomCount() * 4);
    this->vertCylinders.SetCount(mol->ConnectionCount() * 4);
    this->quatCylinders.SetCount(mol->ConnectionCount() * 4);
    this->inParaCylinders.SetCount(mol->ConnectionCount() * 2);
    this->color1Cylinders.SetCount(mol->ConnectionCount() * 3);
    this->color2Cylinders.SetCount(mol->ConnectionCount() * 3);

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        this->vertSpheres[4 * cnt + 0] = atomPos[3 * cnt + 0];
        this->vertSpheres[4 * cnt + 1] = atomPos[3 * cnt + 1];
        this->vertSpheres[4 * cnt + 2] = atomPos[3 * cnt + 2];
        this->vertSpheres[4 * cnt + 3] = this->stickRadiusParam.Param<param::FloatParam>()->Value();
    }

    unsigned int idx0, idx1;
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Quaternion<float> quatC(0, 0, 0, 1);
    vislib::math::Vector<float, 3> tmpVec, ortho, dir, position;
    float angle;
    // loop over all connections and compute cylinder parameters
#pragma omp parallel for private(idx0, idx1, firstAtomPos, secondAtomPos, quatC, tmpVec, ortho, dir, position, angle)
    for (cnt = 0; cnt < int(mol->ConnectionCount()); ++cnt) {
        idx0 = mol->Connection()[2 * cnt];
        idx1 = mol->Connection()[2 * cnt + 1];

        firstAtomPos.SetX(atomPos[3 * idx0 + 0]);
        firstAtomPos.SetY(atomPos[3 * idx0 + 1]);
        firstAtomPos.SetZ(atomPos[3 * idx0 + 2]);

        secondAtomPos.SetX(atomPos[3 * idx1 + 0]);
        secondAtomPos.SetY(atomPos[3 * idx1 + 1]);
        secondAtomPos.SetZ(atomPos[3 * idx1 + 2]);

        // compute the quaternion for the rotation of the cylinder
        dir = secondAtomPos - firstAtomPos;
        tmpVec.Set(1.0f, 0.0f, 0.0f);
        angle = -tmpVec.Angle(dir);
        ortho = tmpVec.Cross(dir);
        ortho.Normalise();
        quatC.Set(angle, ortho);
        // compute the absolute position 'position' of the cylinder (center point)
        position = firstAtomPos + (dir / 2.0f);

        this->inParaCylinders[2 * cnt] = this->stickRadiusParam.Param<param::FloatParam>()->Value();

        this->inParaCylinders[2 * cnt + 1] = (firstAtomPos - secondAtomPos).Length();

        // thomasbm: hotfix for jumping molecules near bounding box
        if (this->inParaCylinders[2 * cnt + 1] > mol->AtomTypes()[mol->AtomTypeIndices()[idx0]].Radius() +
                                                     mol->AtomTypes()[mol->AtomTypeIndices()[idx1]].Radius()) {
            this->inParaCylinders[2 * cnt + 1] = 0;
        }

        this->quatCylinders[4 * cnt + 0] = quatC.GetX();
        this->quatCylinders[4 * cnt + 1] = quatC.GetY();
        this->quatCylinders[4 * cnt + 2] = quatC.GetZ();
        this->quatCylinders[4 * cnt + 3] = quatC.GetW();

        this->color1Cylinders[3 * cnt + 0] = this->atomColorTable[3 * idx0 + 0];
        this->color1Cylinders[3 * cnt + 1] = this->atomColorTable[3 * idx0 + 1];
        this->color1Cylinders[3 * cnt + 2] = this->atomColorTable[3 * idx0 + 2];

        this->color2Cylinders[3 * cnt + 0] = this->atomColorTable[3 * idx1 + 0];
        this->color2Cylinders[3 * cnt + 1] = this->atomColorTable[3 * idx1 + 1];
        this->color2Cylinders[3 * cnt + 2] = this->atomColorTable[3 * idx1 + 2];

        this->vertCylinders[4 * cnt + 0] = position.X();
        this->vertCylinders[4 * cnt + 1] = position.Y();
        this->vertCylinders[4 * cnt + 2] = position.Z();
        this->vertCylinders[4 * cnt + 3] = 0.0f;
    }

    // ---------- actual rendering ----------

    auto cam_pose = cam.get<megamol::core::view::Camera::Pose>();

    float near_plane = 0.0;
    float far_plane = 0.0;
    try {
        auto cam_intrinsics = cam.get<megamol::core::view::Camera::PerspectiveParameters>();
        near_plane = cam_intrinsics.near_plane;
        far_plane = cam_intrinsics.far_plane;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SimpleMoleculeRenderer - Error when getting perspective camera intrinsics");
    }

    // enable sphere shader
    if (!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
#ifdef OLD_SHADERS
        this->sphereShader.Enable();
        // set shader variables
        glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->sphereShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->sphereShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVP"), 1, false, glm::value_ptr(MVP));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVinv"), 1, false, glm::value_ptr(MVinv));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPinv"), 1, false, glm::value_ptr(MVPinv));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPtransp"), 1, false, glm::value_ptr(MVPtransp));
#else
        sphereShader_->use();
        sphereShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
        sphereShader_->setUniform("camIn", cam_pose.direction);
        sphereShader_->setUniform("camRight", cam_pose.right);
        sphereShader_->setUniform("camUp", cam_pose.up);
        sphereShader_->setUniform("MVP", MVP);
        sphereShader_->setUniform("MVinv", MVinv);
        sphereShader_->setUniform("MVPinv", MVPinv);
        sphereShader_->setUniform("MVPtransp", MVPtransp);
#endif
    } else {
        this->sphereShaderOR.Enable();
        // set shader variables
        glUniform4fv(this->sphereShaderOR.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->sphereShaderOR.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->sphereShaderOR.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->sphereShaderOR.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniform2f(this->sphereShaderOR.ParameterLocation("zValues"), near_plane, far_plane);
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVP"), 1, false, glm::value_ptr(MVP));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVinv"), 1, false, glm::value_ptr(MVinv));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVPinv"), 1, false, glm::value_ptr(MVPinv));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVPtransp"), 1, false, glm::value_ptr(MVPtransp));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("NormalM"), 1, false, glm::value_ptr(NormalM));
    }

    // set vertex and color pointers and draw them
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    glDrawArrays(GL_POINTS, 0, mol->AtomCount());

    // disable sphere shader
    glUseProgram(0);

    // enable cylinder shader
    if (!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
#ifdef OLD_SHADERS
        this->cylinderShader.Enable();
        // set shader variables
        glUniform4fv(this->cylinderShader.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->cylinderShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->cylinderShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->cylinderShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniformMatrix4fv(this->cylinderShader.ParameterLocation("MVP"), 1, false, glm::value_ptr(MVP));
        glUniformMatrix4fv(this->cylinderShader.ParameterLocation("MVinv"), 1, false, glm::value_ptr(MVinv));
        glUniformMatrix4fv(this->cylinderShader.ParameterLocation("MVPinv"), 1, false, glm::value_ptr(MVPinv));
        glUniformMatrix4fv(this->cylinderShader.ParameterLocation("MVPtransp"), 1, false, glm::value_ptr(MVPtransp));
        // get the attribute locations
        attribLocInParams = glGetAttribLocation(this->cylinderShader, "inParams");
        attribLocQuatC = glGetAttribLocation(this->cylinderShader, "quatC");
        attribLocColor1 = glGetAttribLocation(this->cylinderShader, "color1");
        attribLocColor2 = glGetAttribLocation(this->cylinderShader, "color2");
#else
        cylinderShader_->use();
        cylinderShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
        cylinderShader_->setUniform("camIn", cam_pose.direction);
        cylinderShader_->setUniform("camRight", cam_pose.right);
        cylinderShader_->setUniform("camUp", cam_pose.up);
        cylinderShader_->setUniform("MVP", MVP);
        cylinderShader_->setUniform("MVinv", MVinv);
        cylinderShader_->setUniform("MVPinv", MVPinv);
        cylinderShader_->setUniform("MVPtransp", MVPtransp);
        attribLocInParams = glGetAttribLocation(cylinderShader_->getHandle(), "inParams");
        attribLocQuatC = glGetAttribLocation(cylinderShader_->getHandle(), "quatC");
        attribLocColor1 = glGetAttribLocation(cylinderShader_->getHandle(), "color1");
        attribLocColor2 = glGetAttribLocation(cylinderShader_->getHandle(), "color2");
#endif
    } else {
        this->cylinderShaderOR.Enable();
        // set shader variables
        glUniform4fv(this->cylinderShaderOR.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->cylinderShaderOR.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->cylinderShaderOR.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->cylinderShaderOR.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniform2f(this->cylinderShaderOR.ParameterLocation("zValues"), near_plane, far_plane);
        glUniformMatrix4fv(this->cylinderShaderOR.ParameterLocation("MVP"), 1, false, glm::value_ptr(MVP));
        glUniformMatrix4fv(this->cylinderShaderOR.ParameterLocation("MVinv"), 1, false, glm::value_ptr(MVinv));
        glUniformMatrix4fv(this->cylinderShaderOR.ParameterLocation("MVPinv"), 1, false, glm::value_ptr(MVPinv));
        glUniformMatrix4fv(this->cylinderShaderOR.ParameterLocation("MVPtransp"), 1, false, glm::value_ptr(MVPtransp));
        glUniformMatrix4fv(this->cylinderShaderOR.ParameterLocation("NormalM"), 1, false, glm::value_ptr(NormalM));
        // get the attribute locations
        attribLocInParams = glGetAttribLocation(this->cylinderShaderOR, "inParams");
        attribLocQuatC = glGetAttribLocation(this->cylinderShaderOR, "quatC");
        attribLocColor1 = glGetAttribLocation(this->cylinderShaderOR, "color1");
        attribLocColor2 = glGetAttribLocation(this->cylinderShaderOR, "color2");
    }

    // enable vertex attribute arrays for the attribute locations
    glDisableClientState(GL_COLOR_ARRAY);
    glEnableVertexAttribArray(this->attribLocInParams);
    glEnableVertexAttribArray(this->attribLocQuatC);
    glEnableVertexAttribArray(this->attribLocColor1);
    glEnableVertexAttribArray(this->attribLocColor2);
    // set vertex and attribute pointers and draw them
    glVertexPointer(4, GL_FLOAT, 0, this->vertCylinders.PeekElements());
    glVertexAttribPointer(this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylinders.PeekElements());
    glVertexAttribPointer(this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinders.PeekElements());
    glVertexAttribPointer(this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1Cylinders.PeekElements());
    glVertexAttribPointer(this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2Cylinders.PeekElements());
    glDrawArrays(GL_POINTS, 0, mol->ConnectionCount());
    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArray(this->attribLocInParams);
    glDisableVertexAttribArray(this->attribLocQuatC);
    glDisableVertexAttribArray(this->attribLocColor1);
    glDisableVertexAttribArray(this->attribLocColor2);
    glDisableClientState(GL_VERTEX_ARRAY);

    // disable cylinder shader
    glUseProgram(0);
}

/*
 * Render the molecular data in ball-and-stick mode.
 */
void SimpleMoleculeRenderer::RenderBallAndStick(const MolecularDataCall* mol, const float* atomPos) {
    // ----- prepare stick raycasting -----
    this->vertSpheres.SetCount(mol->AtomCount() * 4);
    this->vertCylinders.SetCount(mol->ConnectionCount() * 4);
    this->quatCylinders.SetCount(mol->ConnectionCount() * 4);
    this->inParaCylinders.SetCount(mol->ConnectionCount() * 2);
    this->color1Cylinders.SetCount(mol->ConnectionCount() * 3);
    this->color2Cylinders.SetCount(mol->ConnectionCount() * 3);

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        this->vertSpheres[4 * cnt + 0] = atomPos[3 * cnt + 0];
        this->vertSpheres[4 * cnt + 1] = atomPos[3 * cnt + 1];
        this->vertSpheres[4 * cnt + 2] = atomPos[3 * cnt + 2];
        this->vertSpheres[4 * cnt + 3] = this->stickRadiusParam.Param<param::FloatParam>()->Value();
    }

    unsigned int idx0, idx1;
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Quaternion<float> quatC(0, 0, 0, 1);
    vislib::math::Vector<float, 3> tmpVec, ortho, dir, position;
    float angle;
    // loop over all connections and compute cylinder parameters
#pragma omp parallel for private(idx0, idx1, firstAtomPos, secondAtomPos, quatC, tmpVec, ortho, dir, position, angle)
    for (cnt = 0; cnt < int(mol->ConnectionCount()); ++cnt) {
        idx0 = mol->Connection()[2 * cnt];
        idx1 = mol->Connection()[2 * cnt + 1];

        firstAtomPos.SetX(atomPos[3 * idx0 + 0]);
        firstAtomPos.SetY(atomPos[3 * idx0 + 1]);
        firstAtomPos.SetZ(atomPos[3 * idx0 + 2]);

        secondAtomPos.SetX(atomPos[3 * idx1 + 0]);
        secondAtomPos.SetY(atomPos[3 * idx1 + 1]);
        secondAtomPos.SetZ(atomPos[3 * idx1 + 2]);

        // compute the quaternion for the rotation of the cylinder
        dir = secondAtomPos - firstAtomPos;
        tmpVec.Set(1.0f, 0.0f, 0.0f);
        angle = -tmpVec.Angle(dir);
        ortho = tmpVec.Cross(dir);
        ortho.Normalise();
        quatC.Set(angle, ortho);
        // compute the absolute position 'position' of the cylinder (center point)
        position = firstAtomPos + (dir / 2.0f);

        this->inParaCylinders[2 * cnt] = this->stickRadiusParam.Param<param::FloatParam>()->Value() / 3.0f;
        this->inParaCylinders[2 * cnt + 1] = (firstAtomPos - secondAtomPos).Length();

        // thomasbm: hotfix for jumping molecules near bounding box
        if (this->inParaCylinders[2 * cnt + 1] > mol->AtomTypes()[mol->AtomTypeIndices()[idx0]].Radius() +
                                                     mol->AtomTypes()[mol->AtomTypeIndices()[idx1]].Radius()) {
            this->inParaCylinders[2 * cnt + 1] = 0;
        }

        this->quatCylinders[4 * cnt + 0] = quatC.GetX();
        this->quatCylinders[4 * cnt + 1] = quatC.GetY();
        this->quatCylinders[4 * cnt + 2] = quatC.GetZ();
        this->quatCylinders[4 * cnt + 3] = quatC.GetW();

        this->color1Cylinders[3 * cnt + 0] = this->atomColorTable[3 * idx0 + 0];
        this->color1Cylinders[3 * cnt + 1] = this->atomColorTable[3 * idx0 + 1];
        this->color1Cylinders[3 * cnt + 2] = this->atomColorTable[3 * idx0 + 2];

        this->color2Cylinders[3 * cnt + 0] = this->atomColorTable[3 * idx1 + 0];
        this->color2Cylinders[3 * cnt + 1] = this->atomColorTable[3 * idx1 + 1];
        this->color2Cylinders[3 * cnt + 2] = this->atomColorTable[3 * idx1 + 2];

        this->vertCylinders[4 * cnt + 0] = position.X();
        this->vertCylinders[4 * cnt + 1] = position.Y();
        this->vertCylinders[4 * cnt + 2] = position.Z();
        this->vertCylinders[4 * cnt + 3] = 0.0f;
    }

    // ---------- actual rendering ----------

    auto cam_pose = cam.get<megamol::core::view::Camera::Pose>();

    float near_plane = 0.0;
    float far_plane = 0.0;
    try {
        auto cam_intrinsics = cam.get<megamol::core::view::Camera::PerspectiveParameters>();
        near_plane = cam_intrinsics.near_plane;
        far_plane = cam_intrinsics.far_plane;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SimpleMoleculeRenderer - Error when getting perspective camera intrinsics");
    }

    // enable sphere shader
    if (!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
#ifdef OLD_SHADERS
        this->sphereShader.Enable();
        // set shader variables
        glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->sphereShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->sphereShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVP"), 1, false, glm::value_ptr(MVP));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVinv"), 1, false, glm::value_ptr(MVinv));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPinv"), 1, false, glm::value_ptr(MVPinv));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPtransp"), 1, false, glm::value_ptr(MVPtransp));
#else
        sphereShader_->use();
        sphereShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
        sphereShader_->setUniform("camIn", cam_pose.direction);
        sphereShader_->setUniform("camRight", cam_pose.right);
        sphereShader_->setUniform("camUp", cam_pose.up);
        sphereShader_->setUniform("MVP", MVP);
        sphereShader_->setUniform("MVinv", MVinv);
        sphereShader_->setUniform("MVPinv", MVPinv);
        sphereShader_->setUniform("MVPtransp", MVPtransp);
#endif
    } else {
        this->sphereShaderOR.Enable();
        // set shader variables
        glUniform4fv(this->sphereShaderOR.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->sphereShaderOR.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->sphereShaderOR.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->sphereShaderOR.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniform2f(this->sphereShaderOR.ParameterLocation("zValues"), near_plane, far_plane);
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVP"), 1, false, glm::value_ptr(MVP));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVinv"), 1, false, glm::value_ptr(MVinv));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVPinv"), 1, false, glm::value_ptr(MVPinv));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVPtransp"), 1, false, glm::value_ptr(MVPtransp));
    }

    // set vertex and color pointers and draw them
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    glDrawArrays(GL_POINTS, 0, mol->AtomCount());

    // disable sphere shader
    glUseProgram(0);

    // enable cylinder shader
    if (!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
#ifdef OLD_SHADERS
        this->cylinderShader.Enable();
        // set shader variables
        glUniform4fv(this->cylinderShader.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->cylinderShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->cylinderShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->cylinderShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniformMatrix4fv(this->cylinderShader.ParameterLocation("MVP"), 1, false, glm::value_ptr(MVP));
        glUniformMatrix4fv(this->cylinderShader.ParameterLocation("MVinv"), 1, false, glm::value_ptr(MVinv));
        glUniformMatrix4fv(this->cylinderShader.ParameterLocation("MVPinv"), 1, false, glm::value_ptr(MVPinv));
        glUniformMatrix4fv(this->cylinderShader.ParameterLocation("MVPtransp"), 1, false, glm::value_ptr(MVPtransp));
        // get the attribute locations
        attribLocInParams = glGetAttribLocation(this->cylinderShader, "inParams");
        attribLocQuatC = glGetAttribLocation(this->cylinderShader, "quatC");
        attribLocColor1 = glGetAttribLocation(this->cylinderShader, "color1");
        attribLocColor2 = glGetAttribLocation(this->cylinderShader, "color2");
#else
        cylinderShader_->use();
        cylinderShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
        cylinderShader_->setUniform("camIn", cam_pose.direction);
        cylinderShader_->setUniform("camRight", cam_pose.right);
        cylinderShader_->setUniform("camUp", cam_pose.up);
        cylinderShader_->setUniform("MVP", MVP);
        cylinderShader_->setUniform("MVinv", MVinv);
        cylinderShader_->setUniform("MVPinv", MVPinv);
        cylinderShader_->setUniform("MVPtransp", MVPtransp);
        attribLocInParams = glGetAttribLocation(cylinderShader_->getHandle(), "inParams");
        attribLocQuatC = glGetAttribLocation(cylinderShader_->getHandle(), "quatC");
        attribLocColor1 = glGetAttribLocation(cylinderShader_->getHandle(), "color1");
        attribLocColor2 = glGetAttribLocation(cylinderShader_->getHandle(), "color2");
#endif
    } else {
        this->cylinderShaderOR.Enable();
        // set shader variables
        glUniform4fv(this->cylinderShaderOR.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->cylinderShaderOR.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->cylinderShaderOR.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->cylinderShaderOR.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniform2f(this->cylinderShaderOR.ParameterLocation("zValues"), near_plane, far_plane);
        glUniformMatrix4fv(this->cylinderShaderOR.ParameterLocation("MVP"), 1, false, glm::value_ptr(MVP));
        glUniformMatrix4fv(this->cylinderShaderOR.ParameterLocation("MVinv"), 1, false, glm::value_ptr(MVinv));
        glUniformMatrix4fv(this->cylinderShaderOR.ParameterLocation("MVPinv"), 1, false, glm::value_ptr(MVPinv));
        glUniformMatrix4fv(this->cylinderShaderOR.ParameterLocation("MVPtransp"), 1, false, glm::value_ptr(MVPtransp));
        // get the attribute locations
        attribLocInParams = glGetAttribLocation(this->cylinderShaderOR, "inParams");
        attribLocQuatC = glGetAttribLocation(this->cylinderShaderOR, "quatC");
        attribLocColor1 = glGetAttribLocation(this->cylinderShaderOR, "color1");
        attribLocColor2 = glGetAttribLocation(this->cylinderShaderOR, "color2");
    }

    // enable vertex attribute arrays for the attribute locations
    glDisableClientState(GL_COLOR_ARRAY);
    glEnableVertexAttribArray(this->attribLocInParams);
    glEnableVertexAttribArray(this->attribLocQuatC);
    glEnableVertexAttribArray(this->attribLocColor1);
    glEnableVertexAttribArray(this->attribLocColor2);
    // set vertex and attribute pointers and draw them
    glVertexPointer(4, GL_FLOAT, 0, this->vertCylinders.PeekElements());
    glVertexAttribPointer(this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylinders.PeekElements());
    glVertexAttribPointer(this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinders.PeekElements());
    glVertexAttribPointer(this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1Cylinders.PeekElements());
    glVertexAttribPointer(this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2Cylinders.PeekElements());
    glDrawArrays(GL_POINTS, 0, mol->ConnectionCount());
    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArray(this->attribLocInParams);
    glDisableVertexAttribArray(this->attribLocQuatC);
    glDisableVertexAttribArray(this->attribLocColor1);
    glDisableVertexAttribArray(this->attribLocColor2);
    glDisableClientState(GL_VERTEX_ARRAY);

    // disable cylinder shader
    glUseProgram(0);
}

/*
 * Render the molecular data in ball-and-stick mode.
 */
void SimpleMoleculeRenderer::RenderStickClipPlane(MolecularDataCall* mol, const float* atomPos) {

    // ----- prepare stick raycasting -----
    this->vertSpheres.SetCount(mol->AtomCount() * 4);
    this->vertCylinders.SetCount(mol->ConnectionCount() * 4);
    this->quatCylinders.SetCount(mol->ConnectionCount() * 4);
    this->inParaCylinders.SetCount(mol->ConnectionCount() * 2);
    this->color1Cylinders.SetCount(mol->ConnectionCount() * 3);
    this->color2Cylinders.SetCount(mol->ConnectionCount() * 3);

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        this->vertSpheres[4 * cnt + 0] = atomPos[3 * cnt + 0];
        this->vertSpheres[4 * cnt + 1] = atomPos[3 * cnt + 1];
        this->vertSpheres[4 * cnt + 2] = atomPos[3 * cnt + 2];
        this->vertSpheres[4 * cnt + 3] = this->stickRadiusParam.Param<param::FloatParam>()->Value();
    }

    unsigned int idx0, idx1;
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Quaternion<float> quatC(0, 0, 0, 1);
    vislib::math::Vector<float, 3> tmpVec, ortho, dir, position;
    float angle;
    // loop over all connections and compute cylinder parameters
#pragma omp parallel for private(idx0, idx1, firstAtomPos, secondAtomPos, quatC, tmpVec, ortho, dir, position, angle)
    for (cnt = 0; cnt < int(mol->ConnectionCount()); ++cnt) {
        idx0 = mol->Connection()[2 * cnt];
        idx1 = mol->Connection()[2 * cnt + 1];

        firstAtomPos.SetX(atomPos[3 * idx0 + 0]);
        firstAtomPos.SetY(atomPos[3 * idx0 + 1]);
        firstAtomPos.SetZ(atomPos[3 * idx0 + 2]);

        secondAtomPos.SetX(atomPos[3 * idx1 + 0]);
        secondAtomPos.SetY(atomPos[3 * idx1 + 1]);
        secondAtomPos.SetZ(atomPos[3 * idx1 + 2]);

        // compute the quaternion for the rotation of the cylinder
        dir = secondAtomPos - firstAtomPos;
        tmpVec.Set(1.0f, 0.0f, 0.0f);
        angle = -tmpVec.Angle(dir);
        ortho = tmpVec.Cross(dir);
        ortho.Normalise();
        quatC.Set(angle, ortho);
        // compute the absolute position 'position' of the cylinder (center point)
        position = firstAtomPos + (dir / 2.0f);

        this->inParaCylinders[2 * cnt] = this->stickRadiusParam.Param<param::FloatParam>()->Value();

        this->inParaCylinders[2 * cnt + 1] = (firstAtomPos - secondAtomPos).Length();

        // thomasbm: hotfix for jumping molecules near bounding box
        if (this->inParaCylinders[2 * cnt + 1] > mol->AtomTypes()[mol->AtomTypeIndices()[idx0]].Radius() +
                                                     mol->AtomTypes()[mol->AtomTypeIndices()[idx1]].Radius()) {
            this->inParaCylinders[2 * cnt + 1] = 0;
        }

        this->quatCylinders[4 * cnt + 0] = quatC.GetX();
        this->quatCylinders[4 * cnt + 1] = quatC.GetY();
        this->quatCylinders[4 * cnt + 2] = quatC.GetZ();
        this->quatCylinders[4 * cnt + 3] = quatC.GetW();

        this->color1Cylinders[3 * cnt + 0] = this->atomColorTable[3 * idx0 + 0];
        this->color1Cylinders[3 * cnt + 1] = this->atomColorTable[3 * idx0 + 1];
        this->color1Cylinders[3 * cnt + 2] = this->atomColorTable[3 * idx0 + 2];

        this->color2Cylinders[3 * cnt + 0] = this->atomColorTable[3 * idx1 + 0];
        this->color2Cylinders[3 * cnt + 1] = this->atomColorTable[3 * idx1 + 1];
        this->color2Cylinders[3 * cnt + 2] = this->atomColorTable[3 * idx1 + 2];

        this->vertCylinders[4 * cnt + 0] = position.X();
        this->vertCylinders[4 * cnt + 1] = position.Y();
        this->vertCylinders[4 * cnt + 2] = position.Z();
        this->vertCylinders[4 * cnt + 3] = 0.0f;
    }

    // ---------- actual rendering ----------

    auto cam_pose = cam.get<megamol::core::view::Camera::Pose>();

    float near_plane = 0.0;
    float far_plane = 0.0;
    try {
        auto cam_intrinsics = cam.get<megamol::core::view::Camera::PerspectiveParameters>();
        near_plane = cam_intrinsics.near_plane;
        far_plane = cam_intrinsics.far_plane;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SimpleMoleculeRenderer - Error when getting perspective camera intrinsics");
    }

    /// Draw spheres ///

#ifdef OLD_SHADERS
    this->sphereClipPlaneShader.Enable();
    // set shader variables
    glUniform4fv(this->sphereClipPlaneShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv(this->sphereClipPlaneShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
    glUniform3fv(this->sphereClipPlaneShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
    glUniform3fv(this->sphereClipPlaneShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
    glUniform3f(this->sphereClipPlaneShader.ParameterLocation("clipPlaneDir"), 0.0, 0.0,
        mol->AccessBoundingBoxes().ObjectSpaceBBox().Back());
    glUniform3f(this->sphereClipPlaneShader.ParameterLocation("clipPlaneBase"), 0.0, 0.0, this->currentZClipPos);
#else
    sphereClipPlaneShader_->use();
    sphereClipPlaneShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
    sphereClipPlaneShader_->setUniform("camIn", cam_pose.direction);
    sphereClipPlaneShader_->setUniform("camRight", cam_pose.right);
    sphereClipPlaneShader_->setUniform("camUp", cam_pose.up);
    sphereClipPlaneShader_->setUniform(
        "clipPlaneDir", glm::vec3(0.0f, 0.0f, mol->AccessBoundingBoxes().ObjectSpaceBBox().Back()));
    sphereClipPlaneShader_->setUniform("clipPlaneBase", glm::vec3(0.0f, 0.0f, this->currentZClipPos));
#endif
    // set vertex and color pointers and draw them
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    glDrawArrays(GL_POINTS, 0, mol->AtomCount());
    glUseProgram(0);

    /// Draw cylinders ///

#ifdef OLD_SHADERS
    this->cylinderClipPlaneShader.Enable();
    glUniform4fv(this->cylinderClipPlaneShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv(this->cylinderClipPlaneShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
    glUniform3fv(this->cylinderClipPlaneShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
    glUniform3fv(this->cylinderClipPlaneShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
    // get the attribute locations
    attribLocInParams = glGetAttribLocation(this->cylinderClipPlaneShader, "inParams");
    attribLocQuatC = glGetAttribLocation(this->cylinderClipPlaneShader, "quatC");
    attribLocColor1 = glGetAttribLocation(this->cylinderClipPlaneShader, "color1");
    attribLocColor2 = glGetAttribLocation(this->cylinderClipPlaneShader, "color2");
    glUniform3f(this->cylinderClipPlaneShader.ParameterLocation("clipPlaneDir"), 0.0, 0.0,
        mol->AccessBoundingBoxes().ObjectSpaceBBox().Back());
    glUniform3f(this->cylinderClipPlaneShader.ParameterLocation("clipPlaneBase"), 0.0, 0.0, this->currentZClipPos);
#else
    cylinderClipPlaneShader_->use();
    cylinderClipPlaneShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
    cylinderClipPlaneShader_->setUniform("camIn", cam_pose.direction);
    cylinderClipPlaneShader_->setUniform("camRight", cam_pose.right);
    cylinderClipPlaneShader_->setUniform("camUp", cam_pose.up);
    attribLocInParams = glGetAttribLocation(cylinderClipPlaneShader_->getHandle(), "inParams");
    attribLocQuatC = glGetAttribLocation(cylinderClipPlaneShader_->getHandle(), "quatC");
    attribLocColor1 = glGetAttribLocation(cylinderClipPlaneShader_->getHandle(), "color1");
    attribLocColor2 = glGetAttribLocation(cylinderClipPlaneShader_->getHandle(), "color2");
    cylinderClipPlaneShader_->setUniform(
        "clipPlaneDir", glm::vec3(0.0f, 0.0f, mol->AccessBoundingBoxes().ObjectSpaceBBox().Back()));
    cylinderClipPlaneShader_->setUniform("clipPlaneBase", glm::vec3(0.0f, 0.0f, this->currentZClipPos));
#endif

    // enable vertex attribute arrays for the attribute locations
    glDisableClientState(GL_COLOR_ARRAY);
    glEnableVertexAttribArray(this->attribLocInParams);
    glEnableVertexAttribArray(this->attribLocQuatC);
    glEnableVertexAttribArray(this->attribLocColor1);
    glEnableVertexAttribArray(this->attribLocColor2);
    // set vertex and attribute pointers and draw them
    glVertexPointer(4, GL_FLOAT, 0, this->vertCylinders.PeekElements());
    glVertexAttribPointer(this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylinders.PeekElements());
    glVertexAttribPointer(this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinders.PeekElements());
    glVertexAttribPointer(this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1Cylinders.PeekElements());
    glVertexAttribPointer(this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2Cylinders.PeekElements());
    glDrawArrays(GL_POINTS, 0, mol->ConnectionCount());
    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArray(this->attribLocInParams);
    glDisableVertexAttribArray(this->attribLocQuatC);
    glDisableVertexAttribArray(this->attribLocColor1);
    glDisableVertexAttribArray(this->attribLocColor2);
    glDisableClientState(GL_VERTEX_ARRAY);
    glUseProgram(0);
}

/*
 * Render the molecular data in spacefilling mode.
 */
void SimpleMoleculeRenderer::RenderSpacefilling(const MolecularDataCall* mol, const float* atomPos) {

    this->vertSpheres.SetCount(mol->AtomCount() * 4);

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        this->vertSpheres[4 * cnt + 0] = atomPos[3 * cnt + 0];
        this->vertSpheres[4 * cnt + 1] = atomPos[3 * cnt + 1];
        this->vertSpheres[4 * cnt + 2] = atomPos[3 * cnt + 2];
        this->vertSpheres[4 * cnt + 3] = mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius();
    }

    // ---------- actual rendering ----------

    auto cam_pose = cam.get<megamol::core::view::Camera::Pose>();

    float near_plane = 0.0;
    float far_plane = 0.0;
    try {
        auto cam_intrinsics = cam.get<megamol::core::view::Camera::PerspectiveParameters>();
        near_plane = cam_intrinsics.near_plane;
        far_plane = cam_intrinsics.far_plane;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SimpleMoleculeRenderer - Error when getting perspective camera intrinsics");
    }

    // enable sphere shader
    if (!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
#ifdef OLD_SHADERS
        this->sphereShader.Enable();
        // set shader variables
        glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->sphereShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->sphereShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVP"), 1, false, glm::value_ptr(MVP));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVinv"), 1, false, glm::value_ptr(MVinv));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPinv"), 1, false, glm::value_ptr(MVPinv));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPtransp"), 1, false, glm::value_ptr(MVPtransp));
#else
        sphereShader_->use();
        sphereShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
        sphereShader_->setUniform("camIn", cam_pose.direction);
        sphereShader_->setUniform("camRight", cam_pose.right);
        sphereShader_->setUniform("camUp", cam_pose.up);
        sphereShader_->setUniform("MVP", MVP);
        sphereShader_->setUniform("MVinv", MVinv);
        sphereShader_->setUniform("MVPinv", MVPinv);
        sphereShader_->setUniform("MVPtransp", MVPtransp);
#endif
    } else {
        this->sphereShaderOR.Enable();
        // set shader variables
        glUniform4fv(this->sphereShaderOR.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->sphereShaderOR.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->sphereShaderOR.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->sphereShaderOR.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniform2f(this->sphereShaderOR.ParameterLocation("zValues"), near_plane, far_plane);
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVP"), 1, false, glm::value_ptr(MVP));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVinv"), 1, false, glm::value_ptr(MVinv));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVPinv"), 1, false, glm::value_ptr(MVPinv));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVPtransp"), 1, false, glm::value_ptr(MVPtransp));
    }

    // set vertex and color pointers and draw them
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    glDrawArrays(GL_POINTS, 0, mol->AtomCount());

    // disable sphere shader
    glUseProgram(0);
}

/*
 * Render the molecular data in spacefilling mode.
 */
void SimpleMoleculeRenderer::RenderSpacefillingClipPlane(MolecularDataCall* mol, const float* atomPos) {

    this->vertSpheres.SetCount(mol->AtomCount() * 4);

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        this->vertSpheres[4 * cnt + 0] = atomPos[3 * cnt + 0];
        this->vertSpheres[4 * cnt + 1] = atomPos[3 * cnt + 1];
        this->vertSpheres[4 * cnt + 2] = atomPos[3 * cnt + 2];
        this->vertSpheres[4 * cnt + 3] = mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius();
    }

    // ---------- actual rendering ----------

    auto cam_pose = cam.get<megamol::core::view::Camera::Pose>();

#ifdef OLD_SHADERS
    this->sphereClipPlaneShader.Enable();
    // set shader variables
    glUniform4fv(this->sphereClipPlaneShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv(this->sphereClipPlaneShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
    glUniform3fv(this->sphereClipPlaneShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
    glUniform3fv(this->sphereClipPlaneShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
    glUniform3f(this->sphereClipPlaneShader.ParameterLocation("clipPlaneDir"), 0.0, 0.0,
        mol->AccessBoundingBoxes().ObjectSpaceBBox().Back());
    glUniform3f(this->sphereClipPlaneShader.ParameterLocation("clipPlaneBase"), 0.0, 0.0, this->currentZClipPos);
#else
    sphereClipPlaneShader_->use();
    sphereClipPlaneShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
    sphereClipPlaneShader_->setUniform("camIn", cam_pose.direction);
    sphereClipPlaneShader_->setUniform("camRight", cam_pose.right);
    sphereClipPlaneShader_->setUniform("camUp", cam_pose.up);
    sphereClipPlaneShader_->setUniform(
        "clipPlaneDir", glm::vec3(0.0f, 0.0f, mol->AccessBoundingBoxes().ObjectSpaceBBox().Back()));
    sphereClipPlaneShader_->setUniform("clipPlaneBase", glm::vec3(0.0f, 0.0f, this->currentZClipPos));
#endif

    // set vertex and color pointers and draw them
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    glDrawArrays(GL_POINTS, 0, mol->AtomCount());

    // disable sphere shader
    glUseProgram(0);
}

/*
 * Render the molecular data in solvent accessible surface mode.
 */
void SimpleMoleculeRenderer::RenderSAS(const MolecularDataCall* mol, const float* atomPos) {
    // ----- prepare stick raycasting -----
    this->vertSpheres.SetCount(mol->AtomCount() * 4);

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        this->vertSpheres[4 * cnt + 0] = atomPos[3 * cnt + 0];
        this->vertSpheres[4 * cnt + 1] = atomPos[3 * cnt + 1];
        this->vertSpheres[4 * cnt + 2] = atomPos[3 * cnt + 2];
        this->vertSpheres[4 * cnt + 3] = mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius() +
                                         this->probeRadiusParam.Param<param::FloatParam>()->Value();
    }


    // ---------- actual rendering ----------

    auto cam_pose = cam.get<megamol::core::view::Camera::Pose>();

    float near_plane = 0.0;
    float far_plane = 0.0;
    try {
        auto cam_intrinsics = cam.get<megamol::core::view::Camera::PerspectiveParameters>();
        near_plane = cam_intrinsics.near_plane;
        far_plane = cam_intrinsics.far_plane;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SimpleMoleculeRenderer - Error when getting perspective camera intrinsics");
    }

    // enable sphere shader
    if (!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
#ifdef OLD_SHADERS
        this->sphereShader.Enable();
        // set shader variables
        glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->sphereShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->sphereShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVP"), 1, false, glm::value_ptr(MVP));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVinv"), 1, false, glm::value_ptr(MVinv));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPinv"), 1, false, glm::value_ptr(MVPinv));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPtransp"), 1, false, glm::value_ptr(MVPtransp));
#else
        sphereShader_->use();
        sphereShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
        sphereShader_->setUniform("camIn", cam_pose.direction);
        sphereShader_->setUniform("camRight", cam_pose.right);
        sphereShader_->setUniform("camUp", cam_pose.up);
        sphereShader_->setUniform("MVP", MVP);
        sphereShader_->setUniform("MVinv", MVinv);
        sphereShader_->setUniform("MVPinv", MVPinv);
        sphereShader_->setUniform("MVPtransp", MVPtransp);
#endif
    } else {
        this->sphereShaderOR.Enable();
        // set shader variables
        glUniform4fv(this->sphereShaderOR.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->sphereShaderOR.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->sphereShaderOR.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->sphereShaderOR.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniform2f(this->sphereShaderOR.ParameterLocation("zValues"), near_plane, far_plane);
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVP"), 1, false, glm::value_ptr(MVP));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVinv"), 1, false, glm::value_ptr(MVinv));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVPinv"), 1, false, glm::value_ptr(MVPinv));
        glUniformMatrix4fv(this->sphereShaderOR.ParameterLocation("MVPtransp"), 1, false, glm::value_ptr(MVPtransp));
    }

    // set vertex and color pointers and draw them
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    glDrawArrays(GL_POINTS, 0, mol->AtomCount());

    // disable sphere shader
    glUseProgram(0);
}

/*
 * renderPointsFilter
 *
 * Helper function to test the filter module.
 */
void SimpleMoleculeRenderer::RenderPointsFilter(const MolecularDataCall* mol, const float* atomPos) {

    vislib::Array<unsigned int> idx;
    unsigned int i, visAtmCnt = 0;

    idx.SetCapacityIncrement(1000);

    // Get indices of visible atoms
    for (i = 0; i < mol->AtomCount(); i++) {
        if (mol->Filter()[i] == 1) {
            idx.Add(i);
            visAtmCnt++;
        }
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glVertexPointer(3, GL_FLOAT, 0, atomPos);
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());

    glDrawElements(GL_POINTS, visAtmCnt, GL_UNSIGNED_INT, idx.PeekElements());

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

/*
 * renderLinesFilter
 *
 * Helper function to test the filter module.
 */
void SimpleMoleculeRenderer::RenderLinesFilter(const MolecularDataCall* mol, const float* atomPos) {

    vislib::Array<unsigned int> visAtmIdx;
    vislib::Array<unsigned int> visConIdx;

    unsigned int visAtmCnt = 0, visConCnt = 0;
    unsigned int m, at, c;
    unsigned int firstAtmIdx, lastAtmIdx, firstConIdx, lastConIdx;

    visAtmIdx.SetCapacityIncrement(2000);
    visConIdx.SetCapacityIncrement(2000);

    // Loop through all molecules
    for (m = 0; m < mol->MoleculeCount(); m++) {

        // If the molecule is visible
        if (mol->Molecules()[m].Filter() == 1) {

            // Get indices of all atoms in this molecule
            firstAtmIdx = mol->Residues()[mol->Molecules()[m].FirstResidueIndex()]->FirstAtomIndex();

            lastAtmIdx =
                mol->Residues()[mol->Molecules()[m].FirstResidueIndex() + mol->Molecules()[m].ResidueCount() - 1]
                    ->FirstAtomIndex() +
                mol->Residues()[mol->Molecules()[m].FirstResidueIndex() + mol->Molecules()[m].ResidueCount() - 1]
                    ->AtomCount() -
                1;

            for (at = firstAtmIdx; at <= lastAtmIdx; at++) {
                visAtmIdx.Add(at);
            }

            visAtmCnt += (lastAtmIdx - firstAtmIdx + 1);

            // Get indices of all connections in this molecule
            if (mol->Molecules()[m].ConnectionCount() > 0) {

                firstConIdx = mol->Molecules()[m].FirstConnectionIndex();

                lastConIdx =
                    mol->Molecules()[m].FirstConnectionIndex() + (mol->Molecules()[m].ConnectionCount() - 1) * 2;

                for (c = firstConIdx; c <= lastConIdx; c += 2) {

                    visConIdx.Add(mol->Connection()[c]);
                    visConIdx.Add(mol->Connection()[c + 1]);
                }
                visConCnt += (lastConIdx - firstConIdx + 2);
            }
        }
    }

    glDisable(GL_LIGHTING);
    glLineWidth(2.0f);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glVertexPointer(3, GL_FLOAT, 0, atomPos);
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());

    // Draw visible atoms
    glDrawElements(GL_POINTS, visAtmCnt, GL_UNSIGNED_INT, visAtmIdx.PeekElements());
    // Draw vivisble bonds
    glDrawElements(GL_LINES, visConCnt, GL_UNSIGNED_INT, visConIdx.PeekElements());

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    glEnable(GL_LIGHTING);
}

/*
 * Render the molecular data in stick mode.
 */
void SimpleMoleculeRenderer::RenderStickFilter(const MolecularDataCall* mol, const float* atomPos) {

    // int n;
    // glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &n);
    // megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
    //  "Maximum num of generic vertex attributes: %i\n", n);

    // ----- prepare stick raycasting -----
    this->vertSpheres.SetCount(mol->AtomCount() * 4);
    this->vertCylinders.SetCount(mol->ConnectionCount() * 4);
    this->quatCylinders.SetCount(mol->ConnectionCount() * 4);
    this->inParaCylinders.SetCount(mol->ConnectionCount() * 2);
    this->color1Cylinders.SetCount(mol->ConnectionCount() * 3);
    this->color2Cylinders.SetCount(mol->ConnectionCount() * 3);
    this->conFilter.SetCount(mol->ConnectionCount());

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        this->vertSpheres[4 * cnt + 0] = atomPos[3 * cnt + 0];
        this->vertSpheres[4 * cnt + 1] = atomPos[3 * cnt + 1];
        this->vertSpheres[4 * cnt + 2] = atomPos[3 * cnt + 2];
        this->vertSpheres[4 * cnt + 3] = this->stickRadiusParam.Param<param::FloatParam>()->Value();
    }

    unsigned int idx0, idx1;
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Quaternion<float> quatC(0, 0, 0, 1);
    vislib::math::Vector<float, 3> tmpVec, ortho, dir, position;
    float angle;
    // loop over all connections and compute cylinder parameters
#pragma omp parallel for private(idx0, idx1, firstAtomPos, secondAtomPos, quatC, tmpVec, ortho, dir, position, angle)
    for (cnt = 0; cnt < int(mol->ConnectionCount()); ++cnt) {
        idx0 = mol->Connection()[2 * cnt];
        idx1 = mol->Connection()[2 * cnt + 1];

        firstAtomPos.SetX(atomPos[3 * idx0 + 0]);
        firstAtomPos.SetY(atomPos[3 * idx0 + 1]);
        firstAtomPos.SetZ(atomPos[3 * idx0 + 2]);

        secondAtomPos.SetX(atomPos[3 * idx1 + 0]);
        secondAtomPos.SetY(atomPos[3 * idx1 + 1]);
        secondAtomPos.SetZ(atomPos[3 * idx1 + 2]);

        // Set filter information for this connection
        if ((mol->Filter()[idx0] == 1) && (mol->Filter()[idx1] == 1))
            this->conFilter[cnt] = 1.0f;
        else
            this->conFilter[cnt] = 0.0f;

        // compute the quaternion for the rotation of the cylinder
        dir = secondAtomPos - firstAtomPos;
        tmpVec.Set(1.0f, 0.0f, 0.0f);
        angle = -tmpVec.Angle(dir);
        ortho = tmpVec.Cross(dir);
        ortho.Normalise();
        quatC.Set(angle, ortho);
        // compute the absolute position 'position' of the cylinder (center point)
        position = firstAtomPos + (dir / 2.0f);

        this->inParaCylinders[2 * cnt] = this->stickRadiusParam.Param<param::FloatParam>()->Value();
        this->inParaCylinders[2 * cnt + 1] = (firstAtomPos - secondAtomPos).Length();

        // thomasbm: hotfix for jumping molecules near bounding box
        if (this->inParaCylinders[2 * cnt + 1] > mol->AtomTypes()[mol->AtomTypeIndices()[idx0]].Radius() +
                                                     mol->AtomTypes()[mol->AtomTypeIndices()[idx1]].Radius()) {
            this->inParaCylinders[2 * cnt + 1] = 0;
        }

        this->quatCylinders[4 * cnt + 0] = quatC.GetX();
        this->quatCylinders[4 * cnt + 1] = quatC.GetY();
        this->quatCylinders[4 * cnt + 2] = quatC.GetZ();
        this->quatCylinders[4 * cnt + 3] = quatC.GetW();

        this->color1Cylinders[3 * cnt + 0] = this->atomColorTable[3 * idx0 + 0];
        this->color1Cylinders[3 * cnt + 1] = this->atomColorTable[3 * idx0 + 1];
        this->color1Cylinders[3 * cnt + 2] = this->atomColorTable[3 * idx0 + 2];

        this->color2Cylinders[3 * cnt + 0] = this->atomColorTable[3 * idx1 + 0];
        this->color2Cylinders[3 * cnt + 1] = this->atomColorTable[3 * idx1 + 1];
        this->color2Cylinders[3 * cnt + 2] = this->atomColorTable[3 * idx1 + 2];

        this->vertCylinders[4 * cnt + 0] = position.X();
        this->vertCylinders[4 * cnt + 1] = position.Y();
        this->vertCylinders[4 * cnt + 2] = position.Z();
        this->vertCylinders[4 * cnt + 3] = 0.0f;
    }

    // Set filter information of connections according to molecules
    /*
     unsigned int c, m, firstConIdx, lastConIdx;

     for(m = 0; m < mol->MoleculeCount(); m++) {
     if(mol->Molecules()[m].ConnectionCount() > 0) {

     firstConIdx =
     mol->Molecules()[m].FirstConnectionIndex();

     lastConIdx =
     mol->Molecules()[m].FirstConnectionIndex()
     + mol->Molecules()[m].ConnectionCount() - 1;

     if(mol->Molecules()[m].Filter() == 1) {
     for(c = firstConIdx; c <= lastConIdx; c ++) {
     //conFilter[c] = 1.0;
     conFilter.Add(1.0);
     }
     }
     else {
     for(c = firstConIdx; c <= lastConIdx; c ++) {
     conFilter.Add(0.0);
     //conFilter[c] = 0.0;
     }
     }
     }
     }
     */

    // ---------- actual rendering ----------

    auto cam_pose = cam.get<megamol::core::view::Camera::Pose>();

    float near_plane = 0.0;
    float far_plane = 0.0;
    try {
        auto cam_intrinsics = cam.get<megamol::core::view::Camera::PerspectiveParameters>();
        near_plane = cam_intrinsics.near_plane;
        far_plane = cam_intrinsics.far_plane;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SimpleMoleculeRenderer - Error when getting perspective camera intrinsics");
    }

    // enable sphere shader
    if (!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
#ifdef OLD_SHADERS
        this->filterSphereShader.Enable();
        // set shader variables
        glUniform4fv(this->filterSphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->filterSphereShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->filterSphereShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->filterSphereShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        // Set filter attribute
        this->attribLocAtomFilter = glGetAttribLocation(this->filterSphereShader.ProgramHandle(), "filter");
#else
        filterSphereShader_->use();
        filterSphereShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
        filterSphereShader_->setUniform("camIn", cam_pose.direction);
        filterSphereShader_->setUniform("camRight", cam_pose.right);
        filterSphereShader_->setUniform("camUp", cam_pose.up);
        this->attribLocAtomFilter = glGetAttribLocation(filterSphereShader_->getHandle(), "filter");
#endif
    } else {
        this->filterSphereShaderOR.Enable();
        // set shader variables
        glUniform4fv(this->filterSphereShaderOR.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->filterSphereShaderOR.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->filterSphereShaderOR.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->filterSphereShaderOR.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniform2f(this->filterSphereShaderOR.ParameterLocation("zValues"), near_plane, far_plane);
        // Set filter attribute
        this->attribLocAtomFilter = glGetAttribLocation(this->filterSphereShaderOR.ProgramHandle(), "filter");
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // Set vertex and color pointers and draw them
    glVertexPointer(4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());

    // Set attribute pointer
    glEnableVertexAttribArray(this->attribLocAtomFilter);
    glVertexAttribPointer(this->attribLocAtomFilter, 1, GL_INT, 0, 0, mol->Filter());

    glDrawArrays(GL_POINTS, 0, mol->AtomCount());

    glDisableVertexAttribArray(this->attribLocAtomFilter);

    // disable sphere shader
    glUseProgram(0);

    // enable cylinder shader
    // enable cylinder shader
    if (!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
#ifdef OLD_SHADERS
        this->filterCylinderShader.Enable();
        // set shader variables
        glUniform4fv(this->filterCylinderShader.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->filterCylinderShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->filterCylinderShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->filterCylinderShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        // get the attribute locations
        attribLocInParams = glGetAttribLocation(this->filterCylinderShader, "inParams");
        attribLocQuatC = glGetAttribLocation(this->filterCylinderShader, "quatC");
        attribLocColor1 = glGetAttribLocation(this->filterCylinderShader, "color1");
        attribLocColor2 = glGetAttribLocation(this->filterCylinderShader, "color2");
        this->attribLocConFilter = glGetAttribLocation(this->filterCylinderShader, "filter");
#else
        filterCylinderShader_->use();
        filterCylinderShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
        filterCylinderShader_->setUniform("camIn", cam_pose.direction);
        filterCylinderShader_->setUniform("camRight", cam_pose.right);
        filterCylinderShader_->setUniform("camUp", cam_pose.up);
        attribLocInParams = glGetAttribLocation(filterCylinderShader_->getHandle(), "inParams");
        attribLocQuatC = glGetAttribLocation(filterCylinderShader_->getHandle(), "quatC");
        attribLocColor1 = glGetAttribLocation(filterCylinderShader_->getHandle(), "color1");
        attribLocColor2 = glGetAttribLocation(filterCylinderShader_->getHandle(), "color2");
        this->attribLocConFilter = glGetAttribLocation(filterCylinderShader_->getHandle(), "filter");
#endif
    } else {
        this->filterCylinderShaderOR.Enable();
        // set shader variables
        glUniform4fv(this->filterCylinderShaderOR.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->filterCylinderShaderOR.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->filterCylinderShaderOR.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->filterCylinderShaderOR.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniform2f(this->filterCylinderShaderOR.ParameterLocation("zValues"), near_plane, far_plane);
        // get the attribute locations
        attribLocInParams = glGetAttribLocation(this->filterCylinderShaderOR, "inParams");
        attribLocQuatC = glGetAttribLocation(this->filterCylinderShaderOR, "quatC");
        attribLocColor1 = glGetAttribLocation(this->filterCylinderShaderOR, "color1");
        attribLocColor2 = glGetAttribLocation(this->filterCylinderShaderOR, "color2");
        this->attribLocConFilter = glGetAttribLocation(this->filterCylinderShaderOR, "filter");
    }

    // enable vertex attribute arrays for the attribute locations
    glDisableClientState(GL_COLOR_ARRAY);
    glEnableVertexAttribArray(this->attribLocInParams);
    glEnableVertexAttribArray(this->attribLocQuatC);
    glEnableVertexAttribArray(this->attribLocColor1);
    glEnableVertexAttribArray(this->attribLocColor2);
    glEnableVertexAttribArray(this->attribLocConFilter);
    // set vertex and attribute pointers and draw them
    glVertexPointer(4, GL_FLOAT, 0, this->vertCylinders.PeekElements());
    glVertexAttribPointer(this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylinders.PeekElements());
    glVertexAttribPointer(this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinders.PeekElements());
    glVertexAttribPointer(this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1Cylinders.PeekElements());
    glVertexAttribPointer(this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2Cylinders.PeekElements());
    glVertexAttribPointer(this->attribLocConFilter, 1, GL_FLOAT, 0, 0, this->conFilter.PeekElements());

    glDrawArrays(GL_POINTS, 0, mol->ConnectionCount());
    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArray(this->attribLocInParams);
    glDisableVertexAttribArray(this->attribLocQuatC);
    glDisableVertexAttribArray(this->attribLocColor1);
    glDisableVertexAttribArray(this->attribLocColor2);
    glDisableVertexAttribArray(this->attribLocConFilter);
    glDisableClientState(GL_VERTEX_ARRAY);

    // disable cylinder shader
    glUseProgram(0);

    /* GLenum errCode;
     const GLubyte *errString;

     if ((errCode = glGetError()) != GL_NO_ERROR) {
     errString = gluErrorString(errCode);
     vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
     "OpenGL Error: %s\n", errString);
     //fprintf (stderr, "OpenGL Error: %s\n", errString);
     }*/
}

/*
 * Render the molecular data in stick mode.
 */
void SimpleMoleculeRenderer::RenderSpacefillingFilter(const MolecularDataCall* mol, const float* atomPos) {

    // ----- prepare stick raycasting -----
    this->vertSpheres.SetCount(mol->AtomCount() * 4);

    int cnt;

    // copy atom pos and radius to vertex array
#pragma omp parallel for
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        this->vertSpheres[4 * cnt + 0] = atomPos[3 * cnt + 0];
        this->vertSpheres[4 * cnt + 1] = atomPos[3 * cnt + 1];
        this->vertSpheres[4 * cnt + 2] = atomPos[3 * cnt + 2];
        this->vertSpheres[4 * cnt + 3] = mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius();
    }

    // ---------- actual rendering ----------

    auto cam_pose = cam.get<megamol::core::view::Camera::Pose>();

    float near_plane = 0.0;
    float far_plane = 0.0;
    try {
        auto cam_intrinsics = cam.get<megamol::core::view::Camera::PerspectiveParameters>();
        near_plane = cam_intrinsics.near_plane;
        far_plane = cam_intrinsics.far_plane;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SimpleMoleculeRenderer - Error when getting perspective camera intrinsics");
    }

    // Enable sphere shader
    if (!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
#ifdef OLD_SHADERS
        this->filterSphereShader.Enable();
        // set shader variables
        glUniform4fv(this->filterSphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->filterSphereShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->filterSphereShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->filterSphereShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        // Set filter attribute
        this->attribLocAtomFilter = glGetAttribLocation(this->filterSphereShader.ProgramHandle(), "filter");
#else
        filterSphereShader_->use();
        filterSphereShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
        filterSphereShader_->setUniform("camIn", cam_pose.direction);
        filterSphereShader_->setUniform("camRight", cam_pose.right);
        filterSphereShader_->setUniform("camUp", cam_pose.up);
        this->attribLocAtomFilter = glGetAttribLocation(filterSphereShader_->getHandle(), "filter");
#endif
    } else {
        this->filterSphereShaderOR.Enable();
        // set shader variables
        glUniform4fv(this->filterSphereShaderOR.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(this->filterSphereShaderOR.ParameterLocation("camIn"), 1, glm::value_ptr(cam_pose.direction));
        glUniform3fv(this->filterSphereShaderOR.ParameterLocation("camRight"), 1, glm::value_ptr(cam_pose.right));
        glUniform3fv(this->filterSphereShaderOR.ParameterLocation("camUp"), 1, glm::value_ptr(cam_pose.up));
        glUniform2f(this->filterSphereShaderOR.ParameterLocation("zValues"), near_plane, far_plane);
        // Set filter attribute
        this->attribLocAtomFilter = glGetAttribLocation(this->filterSphereShaderOR.ProgramHandle(), "filter");
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableVertexAttribArray(this->attribLocAtomFilter);
    glVertexPointer(4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
    glColorPointer(3, GL_FLOAT, 0, this->atomColorTable.PeekElements());

    glVertexAttribPointer(this->attribLocAtomFilter, 1, GL_INT, 0, 0, mol->Filter());
    glDrawArrays(GL_POINTS, 0, mol->AtomCount());
    glDisableVertexAttribArray(this->attribLocAtomFilter);

    // disable sphere shader
    glUseProgram(0);
}

/*
 * update parameters
 */
void SimpleMoleculeRenderer::UpdateParameters(const MolecularDataCall* mol, const protein_calls::BindingSiteCall* bs) {
    // color table param
    bool updatedColorTable = false;
    if (this->colorTableFileParam.IsDirty()) {
        Color::ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::FilePathParam>()->Value(), this->colorLookupTable);
        this->colorTableFileParam.ResetDirty();
        updatedColorTable = true;
    }
    // Recompute color table
    if ((this->coloringModeParam0.IsDirty()) || (this->coloringModeParam1.IsDirty()) ||
        (this->cmWeightParam.IsDirty()) || (this->useNeighborColors.IsDirty()) || lastDataHash != mol->DataHash() ||
        updatedColorTable || this->minGradColorParam.IsDirty() || this->midGradColorParam.IsDirty() ||
        this->maxGradColorParam.IsDirty() || this->specialColorParam.IsDirty()) {

        lastDataHash = mol->DataHash();

        this->currentColoringMode0 =
            static_cast<Color::ColoringMode>(int(this->coloringModeParam0.Param<param::EnumParam>()->Value()));

        this->currentColoringMode1 =
            static_cast<Color::ColoringMode>(int(this->coloringModeParam1.Param<param::EnumParam>()->Value()));

        // Mix two coloring modes
        Color::MakeColorTable(mol, this->currentColoringMode0, this->currentColoringMode1,
            cmWeightParam.Param<param::FloatParam>()->Value(),        // weight for the first cm
            1.0f - cmWeightParam.Param<param::FloatParam>()->Value(), // weight for the second cm
            this->atomColorTable, this->colorLookupTable, this->rainbowColors,
            this->minGradColorParam.Param<param::ColorParam>()->Value(),
            this->midGradColorParam.Param<param::ColorParam>()->Value(),
            this->maxGradColorParam.Param<param::ColorParam>()->Value(), true, bs,
            this->useNeighborColors.Param<param::BoolParam>()->Value());

        // Use one coloring mode
        /*Color::MakeColorTable( mol,
         this->currentColoringMode0,
         this->atomColorTable, this->colorLookupTable, this->rainbowColors,
         this->minGradColorParam.Param<param::StringParam>()->Value(),
         this->midGradColorParam.Param<param::StringParam>()->Value(),
         this->maxGradColorParam.Param<param::StringParam>()->Value(),
         true, nullptr,
         this->useNeighborColors.Param<param::BoolParam>()->Value());*/

        this->coloringModeParam0.ResetDirty();
        this->coloringModeParam1.ResetDirty();
        this->cmWeightParam.ResetDirty();
        this->useNeighborColors.ResetDirty();
        this->minGradColorParam.ResetDirty();
        this->midGradColorParam.ResetDirty();
        this->maxGradColorParam.ResetDirty();
        this->specialColorParam.ResetDirty();
    }
    // rendering mode param
    if (this->renderModeParam.IsDirty()) {
        this->currentRenderMode =
            static_cast<RenderMode>(int(this->renderModeParam.Param<param::EnumParam>()->Value()));
    }
    // get molecule lust
    if (this->molIdxListParam.IsDirty()) {
        vislib::StringA tmpStr(this->molIdxListParam.Param<param::StringParam>()->Value());
        this->molIdxList = vislib::StringTokeniser<vislib::CharTraitsA>::Split(tmpStr, ';', true);
        this->molIdxListParam.ResetDirty();
    }
}
