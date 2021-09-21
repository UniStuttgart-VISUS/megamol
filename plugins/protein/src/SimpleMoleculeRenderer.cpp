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
#include "compositing/CompositingCalls.h"
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
#include "mmcore/view/light/PointLight.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/Trace.h"
#include "vislib/assert.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/Quaternion.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;
using namespace megamol::core::utility::log;

/*
 * protein::SimpleMoleculeRenderer::SimpleMoleculeRenderer (CTOR)
 */
SimpleMoleculeRenderer::SimpleMoleculeRenderer(void)
        : view::Renderer3DModuleGL()
        , molDataCallerSlot("getData", "Connects the molecule rendering with molecule data storage")
        , bsDataCallerSlot("getBindingSites", "Connects the molecule rendering with binding site data storage")
        , getLightsSlot("getLights", "Connects the molecule rendering with availabel light sources")
        , getFramebufferSlot("getFramebuffer", "Connects the molecule rendering to an optional external framebuffer")
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
        , toggleZClippingParam("toggleZClip", "...")
        , clipPlaneTimeOffsetParam("clipPlane::timeOffset", "...")
        , clipPlaneDurationParam("clipPlane::Duration", "...")
        , useNeighborColors("color::neighborhood", "Add the color of the neighborhood to the own")
        , ambientColorParam("lighting::ambientColor", "...")
        , diffuseColorParam("lighting::diffuseColor", "...")
        , specularColorParam("lighting::specularColor", "...")
        , ambientFactorParam("lighting::ambientFactor", "...")
        , diffuseFactorParam("lighting::diffuseFactor", "...")
        , specularFactorParam("lighting::specularFactor", "...")
        , exponentFactorParam("lighting::specularExponent", "...")
        , useLambertParam("lighting::lambertShading", "If turned on, the local lighting uses lambert instead of Blinn-Phong.")
        , currentZClipPos(-20)
        , fbo_version_(0)
        , vertex_array_(0)
        , localFramebufferObj_(nullptr)
        , usedFramebufferObj_(nullptr) {
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->molDataCallerSlot.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->molDataCallerSlot);
    this->getLightsSlot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->getLightsSlot.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->getLightsSlot);
    this->bsDataCallerSlot.SetCompatibleCall<BindingSiteCallDescription>();
    this->MakeSlotAvailable(&this->bsDataCallerSlot);
    this->getFramebufferSlot.SetCompatibleCall<compositing::CallFramebufferGLDescription>();
    this->MakeSlotAvailable(&this->getFramebufferSlot);

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

    // Toggle Z-Clipping
    this->toggleZClippingParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->toggleZClippingParam);


    this->clipPlaneTimeOffsetParam.SetParameter(new param::FloatParam(100.0f));
    this->MakeSlotAvailable(&this->clipPlaneTimeOffsetParam);

    this->clipPlaneDurationParam.SetParameter(new param::FloatParam(40.0f));
    this->MakeSlotAvailable(&this->clipPlaneDurationParam);

    this->useLambertParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->useLambertParam);

    this->ambientColorParam.SetParameter(new param::ColorParam("#ffffff"));
    this->MakeSlotAvailable(&this->ambientColorParam);

    this->diffuseColorParam.SetParameter(new param::ColorParam("#ffffff"));
    this->MakeSlotAvailable(&this->diffuseColorParam);

    this->specularColorParam.SetParameter(new param::ColorParam("#ffffff"));
    this->MakeSlotAvailable(&this->specularColorParam);

    this->ambientFactorParam.SetParameter(new param::FloatParam(0.2f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->ambientFactorParam);

    this->diffuseFactorParam.SetParameter(new param::FloatParam(0.798f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->diffuseFactorParam);

    this->specularFactorParam.SetParameter(new param::FloatParam(0.02f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->specularFactorParam);

    this->exponentFactorParam.SetParameter(new param::FloatParam(120.0f, 1.0f, 1000.0f));
    this->MakeSlotAvailable(&this->exponentFactorParam);

    this->lastDataHash = 0;
}

/*
 * protein::SimpleMoleculeRenderer::~SimpleMoleculeRenderer (DTOR)
 */
SimpleMoleculeRenderer::~SimpleMoleculeRenderer(void) {
    localFramebufferObj_.reset();
    usedFramebufferObj_.reset();
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

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // new shaders
    try {
        auto const shdr_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

        sphereShader_ = core::utility::make_shared_glowl_shader("sphere", shdr_options,
            std::filesystem::path("simplemolecule/sm_sphere.vert.glsl"),
            std::filesystem::path("simplemolecule/sm_sphere.frag.glsl"));

        cylinderShader_ = core::utility::make_shared_glowl_shader("cylinder", shdr_options,
            std::filesystem::path("simplemolecule/sm_cylinder.vert.glsl"),
            std::filesystem::path("simplemolecule/sm_cylinder.frag.glsl"));

        lightingShader_ = core::utility::make_shared_glowl_shader("lighting", shdr_options,
            std::filesystem::path("simplemolecule/sm_common_lighting.vert.glsl"),
            std::filesystem::path("simplemolecule/sm_common_lighting.frag.glsl"));

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "[SimpleMoleculeRenderer] %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[SimpleMoleculeRenderer] Unable to compile shader: Unknown exception");
        return false;
    }

    // generate local framebuffer
    localFramebufferObj_ = std::make_shared<glowl::FramebufferObject>(1, 1);
    localFramebufferObj_->createColorAttachment(GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT); // surface albedo
    localFramebufferObj_->createColorAttachment(GL_RGB16F, GL_RGB, GL_HALF_FLOAT);   // normals
    localFramebufferObj_->createColorAttachment(GL_R32F, GL_RED, GL_FLOAT);          // clip space depth

    // generate data buffers
    buffers_[Buffers::POSITION] = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[Buffers::COLOR] = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[Buffers::CYL_PARAMS] = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[Buffers::CYL_QUAT] = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[Buffers::CYL_COL1] = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[Buffers::CYL_COL2] = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[Buffers::FILTER] = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[Buffers::LIGHT_POSITIONAL] =
        std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[Buffers::LIGHT_DIRECTIONAL] =
        std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &vertex_array_);
    glBindVertexArray(vertex_array_);

    buffers_[Buffers::POSITION]->bind();
    glEnableVertexAttribArray(Buffers::POSITION);
    glVertexAttribPointer(Buffers::POSITION, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[Buffers::COLOR]->bind();
    glEnableVertexAttribArray(Buffers::COLOR);
    glVertexAttribPointer(Buffers::COLOR, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[Buffers::CYL_PARAMS]->bind();
    glEnableVertexAttribArray(Buffers::CYL_PARAMS);
    glVertexAttribPointer(Buffers::CYL_PARAMS, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[Buffers::CYL_QUAT]->bind();
    glEnableVertexAttribArray(Buffers::CYL_QUAT);
    glVertexAttribPointer(Buffers::CYL_QUAT, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[Buffers::CYL_COL1]->bind();
    glEnableVertexAttribArray(Buffers::CYL_COL1);
    glVertexAttribPointer(Buffers::CYL_COL1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[Buffers::CYL_COL2]->bind();
    glEnableVertexAttribArray(Buffers::CYL_COL2);
    glVertexAttribPointer(Buffers::CYL_COL2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[Buffers::FILTER]->bind();
    glEnableVertexAttribArray(Buffers::FILTER);
    glVertexAttribIPointer(Buffers::FILTER, 1, GL_INT, 0, nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(Buffers::POSITION);
    glDisableVertexAttribArray(Buffers::COLOR);
    glDisableVertexAttribArray(Buffers::CYL_PARAMS);
    glDisableVertexAttribArray(Buffers::CYL_QUAT);
    glDisableVertexAttribArray(Buffers::CYL_COL1);
    glDisableVertexAttribArray(Buffers::CYL_COL2);
    glDisableVertexAttribArray(Buffers::FILTER);

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
    ++fbo_version_;
    auto call_fbo = call.GetFramebuffer();

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    float callTime = call.Time();

    cam = call.GetCamera();

    if (localFramebufferObj_->getWidth() != call_fbo->getWidth() ||
        localFramebufferObj_->getHeight() != call_fbo->getHeight()) {
        localFramebufferObj_->resize(call_fbo->getWidth(), call_fbo->getHeight());
    }

    localFramebufferObj_->bind();
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    call_fbo->bind(); // reset to default behavior

    // if there is a fbo connected, use the connected one
    usedFramebufferObj_ = localFramebufferObj_;
    auto cfbo = getFramebufferSlot.CallAs<compositing::CallFramebufferGL>();
    if (cfbo != nullptr) {
        cfbo->operator()(compositing::CallFramebufferGL::CallGetMetaData);
        cfbo->operator()(compositing::CallFramebufferGL::CallGetData);
        auto fbo = cfbo->getData();
        if (fbo != nullptr) {
            usedFramebufferObj_ = fbo;
        }
    }
    usedFramebufferObj_->bind();

    // matrices
    view = cam.getViewMatrix();
    proj = cam.getProjectionMatrix();
    MVinv = glm::inverse(view);
    invProj = glm::inverse(proj);
    NormalM = glm::transpose(MVinv);
    MVP = proj * view;
    MVPinv = glm::inverse(MVP);
    MVPtransp = glm::transpose(MVP);

    // get viewpoint parameters for raycasting
    this->viewportStuff[0] = 0.0f;
    this->viewportStuff[1] = 0.0f;
    this->viewportStuff[2] = call_fbo->getWidth();
    this->viewportStuff[3] = call_fbo->getHeight();
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
        this->RenderStick(mol, posInter, false, this->toggleZClippingParam.Param<param::BoolParam>()->Value());
    } else if (this->currentRenderMode == BALL_AND_STICK) {
        this->RenderBallAndStick(mol, posInter);
    } else if (this->currentRenderMode == SPACEFILLING) {
        this->RenderSpacefilling(mol, posInter, false, this->toggleZClippingParam.Param<param::BoolParam>()->Value());
    } else if (this->currentRenderMode == SPACEFILL_FILTER) {
        this->RenderSpacefilling(mol, posInter, true);
    } else if (this->currentRenderMode == SAS) {
        this->RenderSAS(mol, posInter);
    } else if (this->currentRenderMode == LINES_FILTER) {
        this->RenderLinesFilter(mol, posInter);
    } else if (this->currentRenderMode == STICK_FILTER) {
        this->RenderStick(mol, posInter, true);
    }

    delete[] pos0;
    delete[] pos1;
    delete[] posInter;

    // swap back to normal fbo
    call_fbo->bind();

    // perform the lighing pass only if no framebuffer is attached
    if (cfbo == nullptr) {
        this->RenderLighting();
    }

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
void SimpleMoleculeRenderer::RenderStick(
    const MolecularDataCall* mol, const float* atomPos, bool useFiltering, bool useClipplane) {
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
            this->conFilter[cnt] = 1;
        else
            this->conFilter[cnt] = 0;

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

    // ---------- upload lists --------------
    // We already upload all necessary cylinder information although they are needed after the sphere rendering pass.
    // The only information not uploaded are the cylinder vertex positions, as the sphere vertex positions use the same
    // buffer
    buffers_[Buffers::POSITION]->rebuffer(this->vertSpheres.PeekElements(), this->vertSpheres.Count() * sizeof(float));
    buffers_[Buffers::COLOR]->rebuffer(
        this->atomColorTable.PeekElements(), this->atomColorTable.Count() * sizeof(float));
    buffers_[Buffers::CYL_PARAMS]->rebuffer(
        this->inParaCylinders.PeekElements(), this->inParaCylinders.Count() * sizeof(float));
    buffers_[Buffers::CYL_QUAT]->rebuffer(
        this->quatCylinders.PeekElements(), this->quatCylinders.Count() * sizeof(float));
    buffers_[Buffers::CYL_COL1]->rebuffer(
        this->color1Cylinders.PeekElements(), this->color1Cylinders.Count() * sizeof(float));
    buffers_[Buffers::CYL_COL2]->rebuffer(
        this->color2Cylinders.PeekElements(), this->color2Cylinders.Count() * sizeof(float));
    buffers_[Buffers::FILTER]->rebuffer(mol->Filter(), mol->AtomCount() * sizeof(int));

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

    glBindVertexArray(vertex_array_);

    // enable sphere shader
    sphereShader_->use();
    sphereShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
    sphereShader_->setUniform("camIn", cam_pose.direction);
    sphereShader_->setUniform("camRight", cam_pose.right);
    sphereShader_->setUniform("camUp", cam_pose.up);
    sphereShader_->setUniform("MVP", MVP);
    sphereShader_->setUniform("MVinv", MVinv);
    sphereShader_->setUniform("MVPinv", MVPinv);
    sphereShader_->setUniform("MVPtransp", MVPtransp);
    sphereShader_->setUniform("NormalM", NormalM);
    sphereShader_->setUniform("planes", glm::vec2(near_plane, far_plane));
    sphereShader_->setUniform("clipPlaneDir", glm::vec3(0.0f, 0.0f, mol->GetBoundingBoxes().ObjectSpaceBBox().Back()));
    sphereShader_->setUniform("clipPlaneBase", glm::vec3(0.0f, 0.0f, this->currentZClipPos));
    sphereShader_->setUniform("applyFiltering", useFiltering);
    sphereShader_->setUniform("useClipPlane", useClipplane);

    // set vertex and color pointers and draw them
    glDrawArrays(GL_POINTS, 0, mol->AtomCount());

    // disable sphere shader
    glUseProgram(0);
    glBindVertexArray(0);

    // upload cylinder vertices
    buffers_[Buffers::POSITION]->rebuffer(
        this->vertCylinders.PeekElements(), this->vertCylinders.Count() * sizeof(float));
    buffers_[Buffers::FILTER]->rebuffer(this->conFilter.PeekElements(), this->conFilter.Count() * sizeof(int));
    glBindVertexArray(vertex_array_);

    // enable cylinder shader
    cylinderShader_->use();
    cylinderShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
    cylinderShader_->setUniform("camIn", cam_pose.direction);
    cylinderShader_->setUniform("camRight", cam_pose.right);
    cylinderShader_->setUniform("camUp", cam_pose.up);
    cylinderShader_->setUniform("MVP", MVP);
    cylinderShader_->setUniform("MVinv", MVinv);
    cylinderShader_->setUniform("MVPinv", MVPinv);
    cylinderShader_->setUniform("MVPtransp", MVPtransp);
    cylinderShader_->setUniform("NormalM", NormalM);
    cylinderShader_->setUniform("planes", glm::vec2(near_plane, far_plane));
    cylinderShader_->setUniform(
        "clipPlaneDir", glm::vec3(0.0f, 0.0f, mol->GetBoundingBoxes().ObjectSpaceBBox().Back()));
    cylinderShader_->setUniform("clipPlaneBase", glm::vec3(0.0f, 0.0f, this->currentZClipPos));
    cylinderShader_->setUniform("applyFiltering", useFiltering);
    cylinderShader_->setUniform("useClipPlane", useClipplane);

    // draw everything
    glDrawArrays(GL_POINTS, 0, mol->ConnectionCount());

    // disable cylinder shader
    glUseProgram(0);
    glBindVertexArray(0);
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

    // ---------- upload lists --------------
    // We already upload all necessary cylinder information although they are needed after the sphere rendering pass.
    // The only information not uploaded are the cylinder vertex positions, as the sphere vertex positions use the same
    // buffer
    buffers_[Buffers::POSITION]->rebuffer(this->vertSpheres.PeekElements(), this->vertSpheres.Count() * sizeof(float));
    buffers_[Buffers::COLOR]->rebuffer(
        this->atomColorTable.PeekElements(), this->atomColorTable.Count() * sizeof(float));
    buffers_[Buffers::CYL_PARAMS]->rebuffer(
        this->inParaCylinders.PeekElements(), this->inParaCylinders.Count() * sizeof(float));
    buffers_[Buffers::CYL_QUAT]->rebuffer(
        this->quatCylinders.PeekElements(), this->quatCylinders.Count() * sizeof(float));
    buffers_[Buffers::CYL_COL1]->rebuffer(
        this->color1Cylinders.PeekElements(), this->color1Cylinders.Count() * sizeof(float));
    buffers_[Buffers::CYL_COL2]->rebuffer(
        this->color2Cylinders.PeekElements(), this->color2Cylinders.Count() * sizeof(float));

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

    glBindVertexArray(vertex_array_);

    // enable sphere shader
    sphereShader_->use();
    sphereShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
    sphereShader_->setUniform("camIn", cam_pose.direction);
    sphereShader_->setUniform("camRight", cam_pose.right);
    sphereShader_->setUniform("camUp", cam_pose.up);
    sphereShader_->setUniform("MVP", MVP);
    sphereShader_->setUniform("MVinv", MVinv);
    sphereShader_->setUniform("MVPinv", MVPinv);
    sphereShader_->setUniform("MVPtransp", MVPtransp);
    sphereShader_->setUniform("NormalM", NormalM);
    sphereShader_->setUniform("planes", glm::vec2(near_plane, far_plane));

    // set vertex and color pointers and draw them
    glDrawArrays(GL_POINTS, 0, mol->AtomCount());

    // disable sphere shader
    glUseProgram(0);
    glBindVertexArray(0);

    // upload cylinder vertices
    buffers_[Buffers::POSITION]->rebuffer(
        this->vertCylinders.PeekElements(), this->vertCylinders.Count() * sizeof(float));
    glBindVertexArray(vertex_array_);

    // enable cylinder shader
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

    // draw everything
    glDrawArrays(GL_POINTS, 0, mol->ConnectionCount());

    // disable cylinder shader
    glUseProgram(0);
    glBindVertexArray(0);
}

/*
 * Render the molecular data in spacefilling mode.
 */
void SimpleMoleculeRenderer::RenderSpacefilling(
    const MolecularDataCall* mol, const float* atomPos, bool useFiltering, bool useClipplane) {

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

    buffers_[Buffers::POSITION]->rebuffer(this->vertSpheres.PeekElements(), this->vertSpheres.Count() * sizeof(float));
    buffers_[Buffers::COLOR]->rebuffer(
        this->atomColorTable.PeekElements(), this->atomColorTable.Count() * sizeof(float));
    buffers_[Buffers::FILTER]->rebuffer(mol->Filter(), mol->AtomCount() * sizeof(int));

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

    glBindVertexArray(vertex_array_);

    // enable sphere shader
    sphereShader_->use();
    sphereShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
    sphereShader_->setUniform("camIn", cam_pose.direction);
    sphereShader_->setUniform("camRight", cam_pose.right);
    sphereShader_->setUniform("camUp", cam_pose.up);
    sphereShader_->setUniform("MVP", MVP);
    sphereShader_->setUniform("MVinv", MVinv);
    sphereShader_->setUniform("MVPinv", MVPinv);
    sphereShader_->setUniform("MVPtransp", MVPtransp);
    sphereShader_->setUniform("NormalM", NormalM);
    sphereShader_->setUniform("planes", glm::vec2(near_plane, far_plane));
    sphereShader_->setUniform("clipPlaneDir", glm::vec3(0.0f, 0.0f, mol->GetBoundingBoxes().ObjectSpaceBBox().Back()));
    sphereShader_->setUniform("clipPlaneBase", glm::vec3(0.0f, 0.0f, this->currentZClipPos));
    sphereShader_->setUniform("applyFiltering", useFiltering);
    sphereShader_->setUniform("useClipPlane", useClipplane);

    // draw everything
    glDrawArrays(GL_POINTS, 0, mol->AtomCount());

    // disable sphere shader
    glUseProgram(0);
    glBindVertexArray(0);
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

    buffers_[Buffers::POSITION]->rebuffer(this->vertSpheres.PeekElements(), this->vertSpheres.Count() * sizeof(float));
    buffers_[Buffers::COLOR]->rebuffer(
        this->atomColorTable.PeekElements(), this->atomColorTable.Count() * sizeof(float));

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

    glBindVertexArray(vertex_array_);

    // enable sphere shader
    sphereShader_->use();
    sphereShader_->setUniform("viewAttr", glm::make_vec4(viewportStuff));
    sphereShader_->setUniform("camIn", cam_pose.direction);
    sphereShader_->setUniform("camRight", cam_pose.right);
    sphereShader_->setUniform("camUp", cam_pose.up);
    sphereShader_->setUniform("MVP", MVP);
    sphereShader_->setUniform("MVinv", MVinv);
    sphereShader_->setUniform("MVPinv", MVPinv);
    sphereShader_->setUniform("MVPtransp", MVPtransp);
    sphereShader_->setUniform("NormalM", NormalM);
    sphereShader_->setUniform("planes", glm::vec2(near_plane, far_plane));

    glDrawArrays(GL_POINTS, 0, mol->AtomCount());

    // disable sphere shader
    glUseProgram(0);
    glBindVertexArray(0);
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

void SimpleMoleculeRenderer::RenderLighting(void) {

    auto call_light = this->getLightsSlot.CallAs<core::view::light::CallLight>();
    if (call_light != nullptr) {
        if (!(*call_light)(0)) {
            return;
        }
    } else {
        pointLights_.clear();
        directionalLights_.clear();
    }

    if (call_light != nullptr && call_light->hasUpdate()) {
        auto& lights = call_light->getData();

        pointLights_.clear();
        directionalLights_.clear();

        auto point_lights = lights.get<core::view::light::PointLightType>();
        auto distant_lights = lights.get<core::view::light::DistantLightType>();

        for (auto& pl : point_lights) {
            pointLights_.push_back({pl.position[0], pl.position[1], pl.position[2], pl.intensity});
        }

        for (auto& dl : distant_lights) {
            if (dl.eye_direction) {
                auto cam_dir = glm::normalize(cam.getPose().direction);
                directionalLights_.push_back({cam_dir.x, cam_dir.y, cam_dir.z, dl.intensity});
            } else {
                directionalLights_.push_back({dl.direction[0], dl.direction[1], dl.direction[2], dl.intensity});
            }
        }
    }
    buffers_[Buffers::LIGHT_POSITIONAL]->rebuffer(pointLights_);
    buffers_[Buffers::LIGHT_DIRECTIONAL]->rebuffer(directionalLights_);

    if (pointLights_.empty() && directionalLights_.empty()) {
        core::utility::log::Log::DefaultLog.WriteWarn("[SimpleMoleculeRenderer]: There are no directional or "
                                                      "positional lights connected. Lighting not available.");
    }

    lightingShader_->use();

    buffers_[Buffers::LIGHT_POSITIONAL]->bind(1);
    lightingShader_->setUniform("point_light_cnt", static_cast<GLint>(pointLights_.size()));

    buffers_[Buffers::LIGHT_DIRECTIONAL]->bind(2);
    lightingShader_->setUniform("distant_light_cnt", static_cast<GLint>(directionalLights_.size()));

    glActiveTexture(GL_TEXTURE0);
    usedFramebufferObj_->bindColorbuffer(0);
    lightingShader_->setUniform("albedo_tx2D", 0);

    glActiveTexture(GL_TEXTURE1);
    usedFramebufferObj_->bindColorbuffer(1);
    lightingShader_->setUniform("normal_tx2D", 1);

    glActiveTexture(GL_TEXTURE2);
    usedFramebufferObj_->bindColorbuffer(2);
    lightingShader_->setUniform("depth_tx2D", 2);

    lightingShader_->setUniform("camPos", cam.getPose().position);
    lightingShader_->setUniform("inv_view_mx", this->MVinv);
    lightingShader_->setUniform("inv_proj_mx", this->invProj);
    lightingShader_->setUniform("use_lambert", this->useLambertParam.Param<param::BoolParam>()->Value());

    lightingShader_->setUniform(
        "ambientColor", glm::make_vec4(this->ambientColorParam.Param<param::ColorParam>()->Value().data()));
    lightingShader_->setUniform(
        "diffuseColor", glm::make_vec4(this->diffuseColorParam.Param<param::ColorParam>()->Value().data()));
    lightingShader_->setUniform(
        "specularColor", glm::make_vec4(this->specularColorParam.Param<param::ColorParam>()->Value().data()));
    lightingShader_->setUniform("k_amb", this->ambientFactorParam.Param<param::FloatParam>()->Value());
    lightingShader_->setUniform("k_diff", this->diffuseFactorParam.Param<param::FloatParam>()->Value());
    lightingShader_->setUniform("k_spec", this->specularFactorParam.Param<param::FloatParam>()->Value());
    lightingShader_->setUniform("k_exp", this->exponentFactorParam.Param<param::FloatParam>()->Value());

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glUseProgram(0);
}
