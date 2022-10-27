/*
 * SimpleMoleculeRenderer.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#define _USE_MATH_DEFINES 1

#include "vislib_gl/graphics/gl/IncludeAllGL.h"

#include "SimpleMoleculeRenderer.h"
#include "compositing_gl/CompositingCalls.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "protein_calls/ProteinColor.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/Trace.h"
#include "vislib/assert.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include <GL/glu.h>
#include <omp.h>

#include "vislib/math/Matrix.h"
#include "vislib/math/Quaternion.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_gl;
using namespace megamol::protein_calls;
using namespace megamol::core::utility::log;

/*
 * protein::SimpleMoleculeRenderer::SimpleMoleculeRenderer (CTOR)
 */
SimpleMoleculeRenderer::SimpleMoleculeRenderer(void)
        : mmstd_gl::Renderer3DModuleGL()
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
        , toggleZClippingParam("toggleZClip", "...")
        , clipPlaneTimeOffsetParam("clipPlane::timeOffset", "...")
        , clipPlaneDurationParam("clipPlane::Duration", "...")
        , useNeighborColors("color::neighborhood", "Add the color of the neighborhood to the own")
        , currentZClipPos(-20)
        , vertex_array_(0)
        , tableFromFile_(false) {
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
    tableFromFile_ = ProteinColor::ReadColorTableFromFile(filename, this->fileColorTable_);
    this->colorTableFileParam.SetParameter(
        new param::FilePathParam(filename, core::param::FilePathParam::FilePathFlags_::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&this->colorTableFileParam);

    // coloring modes
    this->currentColoringMode0 = ProteinColor::ColoringMode::CHAIN;
    this->currentColoringMode1 = ProteinColor::ColoringMode::ELEMENT;
    param::EnumParam* cm0 = new param::EnumParam(int(this->currentColoringMode0));
    param::EnumParam* cm1 = new param::EnumParam(int(this->currentColoringMode1));
    unsigned int cCnt;
    ProteinColor::ColoringMode cMode;
    for (cCnt = 0; cCnt < static_cast<unsigned int>(ProteinColor::ColoringMode::MODE_COUNT); ++cCnt) {
        cMode = static_cast<ProteinColor::ColoringMode>(cCnt);
        cm0->SetTypePair(static_cast<int>(cMode), ProteinColor::GetName(cMode).c_str());
        cm1->SetTypePair(static_cast<int>(cMode), ProteinColor::GetName(cMode).c_str());
    }
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
    ProteinColor::MakeRainbowColorTable(100, this->rainbowColors_);

    // Toggle Z-Clipping
    this->toggleZClippingParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->toggleZClippingParam);


    this->clipPlaneTimeOffsetParam.SetParameter(new param::FloatParam(100.0f));
    this->MakeSlotAvailable(&this->clipPlaneTimeOffsetParam);

    this->clipPlaneDurationParam.SetParameter(new param::FloatParam(40.0f));
    this->MakeSlotAvailable(&this->clipPlaneDurationParam);

    // make all the deferred slots from the deferred class available
    auto defparams = deferredProvider_.getUsedParamSlots();
    for (const auto& param : defparams) {
        this->MakeSlotAvailable(param);
    }

    this->colorLookupTable_.clear();
    this->colorLookupTable_ = {glm::make_vec3(this->minGradColorParam.Param<param::ColorParam>()->Value().data()),
        glm::make_vec3(this->midGradColorParam.Param<param::ColorParam>()->Value().data()),
        glm::make_vec3(this->maxGradColorParam.Param<param::ColorParam>()->Value().data())};

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

    glEnable(GL_DEPTH_TEST);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // new shaders
    try {
        auto const shdr_options = core::utility::make_path_shader_options(
            frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

        sphereShader_ = core::utility::make_shared_glowl_shader("sphere", shdr_options,
            std::filesystem::path("protein_gl/simplemolecule/sm_sphere.vert.glsl"),
            std::filesystem::path("protein_gl/simplemolecule/sm_sphere.frag.glsl"));

        cylinderShader_ = core::utility::make_shared_glowl_shader("cylinder", shdr_options,
            std::filesystem::path("protein_gl/simplemolecule/sm_cylinder.vert.glsl"),
            std::filesystem::path("protein_gl/simplemolecule/sm_cylinder.frag.glsl"));

        lineShader_ = core::utility::make_shared_glowl_shader("line", shdr_options,
            std::filesystem::path("protein_gl/simplemolecule/sm_line.vert.glsl"),
            std::filesystem::path("protein_gl/simplemolecule/sm_line.frag.glsl"));

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[SimpleMoleculeRenderer] %s", ex.what());
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[SimpleMoleculeRenderer] Unable to compile shader: Unknown exception: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[SimpleMoleculeRenderer] Unable to compile shader: Unknown exception.");
    }

    // generate data buffers
    buffers_[static_cast<int>(Buffers::POSITION)] =
        std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[static_cast<int>(Buffers::COLOR)] =
        std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[static_cast<int>(Buffers::CYL_PARAMS)] =
        std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[static_cast<int>(Buffers::CYL_QUAT)] =
        std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[static_cast<int>(Buffers::CYL_COL1)] =
        std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[static_cast<int>(Buffers::CYL_COL2)] =
        std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    buffers_[static_cast<int>(Buffers::FILTER)] =
        std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &vertex_array_);
    glBindVertexArray(vertex_array_);

    buffers_[static_cast<int>(Buffers::POSITION)]->bind();
    glEnableVertexAttribArray(static_cast<int>(Buffers::POSITION));
    glVertexAttribPointer(static_cast<int>(Buffers::POSITION), 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[static_cast<int>(Buffers::COLOR)]->bind();
    glEnableVertexAttribArray(static_cast<int>(Buffers::COLOR));
    glVertexAttribPointer(static_cast<int>(Buffers::COLOR), 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[static_cast<int>(Buffers::CYL_PARAMS)]->bind();
    glEnableVertexAttribArray(static_cast<int>(Buffers::CYL_PARAMS));
    glVertexAttribPointer(static_cast<int>(Buffers::CYL_PARAMS), 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[static_cast<int>(Buffers::CYL_QUAT)]->bind();
    glEnableVertexAttribArray(static_cast<int>(Buffers::CYL_QUAT));
    glVertexAttribPointer(static_cast<int>(Buffers::CYL_QUAT), 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[static_cast<int>(Buffers::CYL_COL1)]->bind();
    glEnableVertexAttribArray(static_cast<int>(Buffers::CYL_COL1));
    glVertexAttribPointer(static_cast<int>(Buffers::CYL_COL1), 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[static_cast<int>(Buffers::CYL_COL2)]->bind();
    glEnableVertexAttribArray(static_cast<int>(Buffers::CYL_COL2));
    glVertexAttribPointer(static_cast<int>(Buffers::CYL_COL2), 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[static_cast<int>(Buffers::FILTER)]->bind();
    glEnableVertexAttribArray(static_cast<int>(Buffers::FILTER));
    glVertexAttribIPointer(static_cast<int>(Buffers::FILTER), 1, GL_INT, 0, nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(static_cast<int>(Buffers::POSITION));
    glDisableVertexAttribArray(static_cast<int>(Buffers::COLOR));
    glDisableVertexAttribArray(static_cast<int>(Buffers::CYL_PARAMS));
    glDisableVertexAttribArray(static_cast<int>(Buffers::CYL_QUAT));
    glDisableVertexAttribArray(static_cast<int>(Buffers::CYL_COL1));
    glDisableVertexAttribArray(static_cast<int>(Buffers::CYL_COL2));
    glDisableVertexAttribArray(static_cast<int>(Buffers::FILTER));

    // setup all the deferred stuff
    deferredProvider_.setup(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    return true;
}


/*
 * protein::SimpleMoleculeRenderer::GetExtents
 */
bool SimpleMoleculeRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
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
bool SimpleMoleculeRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    auto call_fbo = call.GetFramebuffer();

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    float callTime = call.Time();

    cam = call.GetCamera();

    deferredProvider_.setFramebufferExtents(call_fbo->getWidth(), call_fbo->getHeight());

    call_fbo->bind(); // set to default behavior

    // if there is a fbo of appropriate size connected, draw to it.
    // if not, we handle the lighting
    bool externalfbo = false;
    auto cfbo = call.GetFramebuffer();
    if (cfbo != nullptr && cfbo->getNumColorAttachments() == 3) {
        externalfbo = true;
    } else {
        deferredProvider_.bindDeferredFramebufferToDraw();
    }

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
    if (this->atomColorTable_.size() < mol->AtomCount()) {

        // Mix two coloring modes
        ProteinColor::MakeWeightedColorTable(*mol, this->currentColoringMode0, this->currentColoringMode1,
            cmWeightParam.Param<param::FloatParam>()->Value(), 1.0f - cmWeightParam.Param<param::FloatParam>()->Value(),
            this->atomColorTable_, this->colorLookupTable_, this->fileColorTable_, this->rainbowColors_, bs, nullptr,
            true, this->useNeighborColors.Param<param::BoolParam>()->Value());
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
                this->atomColorTable_[ai] = glm::make_vec3(specCol.data());
            }
        }
    }
    // ---------- ... special color handling -----------

    // TODO: ---------- render ----------

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // render data using the current rendering mode
    if (this->currentRenderMode == LINES) {
        this->RenderLines(mol, posInter, false, this->toggleZClippingParam.Param<param::BoolParam>()->Value());
    } else if (this->currentRenderMode == STICK) {
        this->RenderStick(mol, posInter, false, this->toggleZClippingParam.Param<param::BoolParam>()->Value());
    } else if (this->currentRenderMode == BALL_AND_STICK) {
        this->RenderBallAndStick(mol, posInter);
    } else if (this->currentRenderMode == SPACEFILLING) {
        this->RenderSpacefilling(mol, posInter, false, this->toggleZClippingParam.Param<param::BoolParam>()->Value());
    } else if (this->currentRenderMode == SPACEFILL_FILTER) {
        this->RenderSpacefilling(mol, posInter, true, this->toggleZClippingParam.Param<param::BoolParam>()->Value());
    } else if (this->currentRenderMode == SAS) {
        this->RenderSAS(mol, posInter);
    } else if (this->currentRenderMode == LINES_FILTER) {
        this->RenderLines(mol, posInter, true, this->toggleZClippingParam.Param<param::BoolParam>()->Value());
    } else if (this->currentRenderMode == STICK_FILTER) {
        this->RenderStick(mol, posInter, true, this->toggleZClippingParam.Param<param::BoolParam>()->Value());
    }

    delete[] pos0;
    delete[] pos1;
    delete[] posInter;

    // swap back to normal fbo
    if (externalfbo) {
        call_fbo->bind();
    } else {
        deferredProvider_.resetToPreviousFramebuffer();
    }

    // perform the lighing pass only if no framebuffer is attached
    if (!externalfbo) {
        deferredProvider_.draw(call, this->getLightsSlot.CallAs<core::view::light::CallLight>(),
            this->currentRenderMode == LINES || this->currentRenderMode == LINES_FILTER);
    }

    // unlock the current frame
    mol->Unlock();

    return true;
}

/*
 * render the atom using lines and points
 */
void SimpleMoleculeRenderer::RenderLines(
    const MolecularDataCall* mol, const float* atomPos, bool useFiltering, bool useClipplane) {
    vertPoints_.resize(mol->AtomCount());
    // will be drawn as GL_LINES so each line needs 2*4 input coordinates and 2 * 3 input color values
    vertLines_.resize(mol->ConnectionCount() * 2);
    colorLines_.resize(mol->ConnectionCount() * 2);

    for (int cnt = 0; cnt < mol->AtomCount(); ++cnt) {
        vertPoints_[cnt] = glm::vec4(glm::make_vec3(&atomPos[3 * cnt]), 1.0f);
    }

    glPointSize(5.0f);
    glLineWidth(2.0f);
    // ----- draw atoms as points -----
    buffers_[static_cast<int>(Buffers::POSITION)]->rebuffer(vertPoints_.data(), vertPoints_.size() * sizeof(glm::vec4));
    buffers_[static_cast<int>(Buffers::COLOR)]->rebuffer(
        atomColorTable_.data(), atomColorTable_.size() * 3 * sizeof(float));

    glBindVertexArray(vertex_array_);
    lineShader_->use();

    lineShader_->setUniform("MVP", this->MVP);
    lineShader_->setUniform("MVPtransp", this->MVPtransp);
    lineShader_->setUniform("applyFiltering", useFiltering);

    glDrawArrays(GL_POINTS, 0, mol->AtomCount());


    // ----- draw bonds as lines -----
    unsigned int cnt, atomIdx0, atomIdx1;
    for (cnt = 0; cnt < mol->ConnectionCount(); ++cnt) {
        // get atom indices
        atomIdx0 = mol->Connection()[2 * cnt + 0];
        atomIdx1 = mol->Connection()[2 * cnt + 1];
        // distance check
        if ((vislib::math::Vector<float, 3>(&atomPos[atomIdx0 * 3]) -
                vislib::math::Vector<float, 3>(&atomPos[atomIdx1 * 3]))
                .Length() > 3.0f)
            continue;

        vertLines_[2 * cnt + 0] = glm::vec4(glm::make_vec3(&atomPos[atomIdx0 * 3]), 1.0);
        vertLines_[2 * cnt + 1] = glm::vec4(glm::make_vec3(&atomPos[atomIdx1 * 3]), 1.0);
        colorLines_[2 * cnt + 0] = atomColorTable_[atomIdx0];
        colorLines_[2 * cnt + 1] = atomColorTable_[atomIdx1];
    }
    buffers_[static_cast<int>(Buffers::POSITION)]->rebuffer(vertLines_.data(), vertLines_.size() * sizeof(glm::vec4));
    buffers_[static_cast<int>(Buffers::COLOR)]->rebuffer(colorLines_.data(), colorLines_.size() * sizeof(glm::vec3));

    glDrawArrays(GL_LINES, 0, mol->ConnectionCount() * 2);

    glUseProgram(0);
    glBindVertexArray(0);

    glEnable(GL_LIGHTING);
}

/*
 * Render the molecular data in stick mode.
 */
void SimpleMoleculeRenderer::RenderStick(
    const MolecularDataCall* mol, const float* atomPos, bool useFiltering, bool useClipplane) {
    // ----- prepare stick raycasting -----
    vertSpheres_.resize(mol->AtomCount());
    vertCylinders_.resize(mol->ConnectionCount());
    quatCylinders_.resize(mol->ConnectionCount());
    inParaCylinders_.resize(mol->ConnectionCount());
    color1Cylinders_.resize(mol->ConnectionCount());
    color2Cylinders_.resize(mol->ConnectionCount());
    conFilter_.resize(mol->ConnectionCount());

    int cnt;

    // copy atom pos and radius to vertex array
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        vertSpheres_[cnt] =
            glm::vec4(glm::make_vec3(&atomPos[3 * cnt]), this->stickRadiusParam.Param<param::FloatParam>()->Value());
    }

    unsigned int idx0, idx1;
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Quaternion<float> quatC(0, 0, 0, 1);
    vislib::math::Vector<float, 3> tmpVec, ortho, dir, position;
    float angle;
    // loop over all connections and compute cylinder parameters
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
        if ((mol->Filter() != nullptr) && (mol->Filter()[idx0] == 1) && (mol->Filter()[idx1] == 1))
            conFilter_[cnt] = 1;
        else
            conFilter_[cnt] = 0;

        // compute the quaternion for the rotation of the cylinder
        dir = secondAtomPos - firstAtomPos;
        tmpVec.Set(1.0f, 0.0f, 0.0f);
        angle = -tmpVec.Angle(dir);
        ortho = tmpVec.Cross(dir);
        ortho.Normalise();
        quatC.Set(angle, ortho);
        // compute the absolute position 'position' of the cylinder (center point)
        position = firstAtomPos + (dir / 2.0f);

        inParaCylinders_[cnt].x = this->stickRadiusParam.Param<param::FloatParam>()->Value();
        inParaCylinders_[cnt].y = (firstAtomPos - secondAtomPos).Length();

        // thomasbm: hotfix for jumping molecules near bounding box
        if (inParaCylinders_[cnt].y > mol->AtomTypes()[mol->AtomTypeIndices()[idx0]].Radius() +
                                          mol->AtomTypes()[mol->AtomTypeIndices()[idx1]].Radius()) {
            inParaCylinders_[cnt].y = 0;
        }

        quatCylinders_[cnt] = glm::vec4(quatC.GetX(), quatC.GetY(), quatC.GetZ(), quatC.GetW());
        color1Cylinders_[cnt] = this->atomColorTable_[idx0];
        color2Cylinders_[cnt] = this->atomColorTable_[idx1];
        vertCylinders_[cnt] = glm::vec4(position.X(), position.Y(), position.Z(), 0.0f);
    }

    // ---------- upload lists --------------
    // We already upload all necessary cylinder information although they are needed after the sphere rendering pass.
    // The only information not uploaded are the cylinder vertex positions, as the sphere vertex positions use the same
    // buffer
    buffers_[static_cast<int>(Buffers::POSITION)]->rebuffer(
        vertSpheres_.data(), vertSpheres_.size() * sizeof(glm::vec4));
    buffers_[static_cast<int>(Buffers::COLOR)]->rebuffer(
        this->atomColorTable_.data(), this->atomColorTable_.size() * sizeof(glm::vec3));
    buffers_[static_cast<int>(Buffers::CYL_PARAMS)]->rebuffer(
        inParaCylinders_.data(), inParaCylinders_.size() * sizeof(glm::vec2));
    buffers_[static_cast<int>(Buffers::CYL_QUAT)]->rebuffer(
        quatCylinders_.data(), quatCylinders_.size() * sizeof(glm::vec4));
    buffers_[static_cast<int>(Buffers::CYL_COL1)]->rebuffer(
        color1Cylinders_.data(), color1Cylinders_.size() * sizeof(glm::vec3));
    buffers_[static_cast<int>(Buffers::CYL_COL2)]->rebuffer(
        color2Cylinders_.data(), color2Cylinders_.size() * sizeof(glm::vec3));
    buffers_[static_cast<int>(Buffers::FILTER)]->rebuffer(mol->Filter(), mol->AtomCount() * sizeof(int));

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
    buffers_[static_cast<int>(Buffers::POSITION)]->rebuffer(
        vertCylinders_.data(), vertCylinders_.size() * sizeof(glm::vec4));
    buffers_[static_cast<int>(Buffers::FILTER)]->rebuffer(conFilter_.data(), conFilter_.size() * sizeof(int));
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
    vertSpheres_.resize(mol->AtomCount());
    vertCylinders_.resize(mol->ConnectionCount());
    quatCylinders_.resize(mol->ConnectionCount());
    inParaCylinders_.resize(mol->ConnectionCount());
    color1Cylinders_.resize(mol->ConnectionCount());
    color2Cylinders_.resize(mol->ConnectionCount());

    int cnt;

    // copy atom pos and radius to vertex array
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        vertSpheres_[cnt] =
            glm::vec4(glm::make_vec3(&atomPos[3 * cnt]), this->stickRadiusParam.Param<param::FloatParam>()->Value());
    }

    unsigned int idx0, idx1;
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Quaternion<float> quatC(0, 0, 0, 1);
    vislib::math::Vector<float, 3> tmpVec, ortho, dir, position;
    float angle;
    // loop over all connections and compute cylinder parameters
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

        inParaCylinders_[cnt].x = this->stickRadiusParam.Param<param::FloatParam>()->Value() / 3.0f;
        inParaCylinders_[cnt].y = (firstAtomPos - secondAtomPos).Length();

        // thomasbm: hotfix for jumping molecules near bounding box
        if (inParaCylinders_[cnt].y > mol->AtomTypes()[mol->AtomTypeIndices()[idx0]].Radius() +
                                          mol->AtomTypes()[mol->AtomTypeIndices()[idx1]].Radius()) {
            inParaCylinders_[cnt].y = 0;
        }

        quatCylinders_[cnt] = glm::vec4(quatC.GetX(), quatC.GetY(), quatC.GetZ(), quatC.GetW());
        color1Cylinders_[cnt] = atomColorTable_[idx0];
        color2Cylinders_[cnt] = atomColorTable_[idx1];
        vertCylinders_[cnt] = glm::vec4(position.X(), position.Y(), position.Z(), 0.0f);
    }

    // ---------- upload lists --------------
    // We already upload all necessary cylinder information although they are needed after the sphere rendering pass.
    // The only information not uploaded are the cylinder vertex positions, as the sphere vertex positions use the same
    // buffer
    buffers_[static_cast<int>(Buffers::POSITION)]->rebuffer(
        vertSpheres_.data(), vertSpheres_.size() * sizeof(glm::vec4));
    buffers_[static_cast<int>(Buffers::COLOR)]->rebuffer(
        this->atomColorTable_.data(), this->atomColorTable_.size() * sizeof(glm::vec3));
    buffers_[static_cast<int>(Buffers::CYL_PARAMS)]->rebuffer(
        inParaCylinders_.data(), inParaCylinders_.size() * sizeof(glm::vec2));
    buffers_[static_cast<int>(Buffers::CYL_QUAT)]->rebuffer(
        quatCylinders_.data(), quatCylinders_.size() * sizeof(glm::vec4));
    buffers_[static_cast<int>(Buffers::CYL_COL1)]->rebuffer(
        color1Cylinders_.data(), color1Cylinders_.size() * sizeof(glm::vec3));
    buffers_[static_cast<int>(Buffers::CYL_COL2)]->rebuffer(
        color2Cylinders_.data(), color2Cylinders_.size() * sizeof(glm::vec3));

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
    buffers_[static_cast<int>(Buffers::POSITION)]->rebuffer(
        vertCylinders_.data(), vertCylinders_.size() * sizeof(glm::vec4));
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

    vertSpheres_.resize(mol->AtomCount());

    int cnt;

    // copy atom pos and radius to vertex array
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        vertSpheres_[cnt] =
            glm::vec4(glm::make_vec3(&atomPos[3 * cnt]), mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius());
    }

    buffers_[static_cast<int>(Buffers::POSITION)]->rebuffer(
        vertSpheres_.data(), vertSpheres_.size() * sizeof(glm::vec4));
    buffers_[static_cast<int>(Buffers::COLOR)]->rebuffer(
        this->atomColorTable_.data(), this->atomColorTable_.size() * sizeof(glm::vec3));
    buffers_[static_cast<int>(Buffers::FILTER)]->rebuffer(mol->Filter(), mol->AtomCount() * sizeof(int));

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
    vertSpheres_.resize(mol->AtomCount());

    int cnt;

    // copy atom pos and radius to vertex array
    for (cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        vertSpheres_[cnt] = glm::vec4(
            glm::make_vec3(&atomPos[3 * cnt]), mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius() +
                                                   this->probeRadiusParam.Param<param::FloatParam>()->Value());
    }

    buffers_[static_cast<int>(Buffers::POSITION)]->rebuffer(
        vertSpheres_.data(), vertSpheres_.size() * sizeof(glm::vec4));
    buffers_[static_cast<int>(Buffers::COLOR)]->rebuffer(
        this->atomColorTable_.data(), this->atomColorTable_.size() * 3 * sizeof(float));

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
 * update parameters
 */
void SimpleMoleculeRenderer::UpdateParameters(const MolecularDataCall* mol, const protein_calls::BindingSiteCall* bs) {
    // color table param
    bool updatedColorTable = false;
    if (this->colorTableFileParam.IsDirty()) {
        this->tableFromFile_ = ProteinColor::ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::FilePathParam>()->Value(), this->fileColorTable_);
        this->colorTableFileParam.ResetDirty();
        updatedColorTable = true;
    }
    // Recompute color table
    if ((this->coloringModeParam0.IsDirty()) || (this->coloringModeParam1.IsDirty()) ||
        (this->cmWeightParam.IsDirty()) || (this->useNeighborColors.IsDirty()) || lastDataHash != mol->DataHash() ||
        updatedColorTable || this->minGradColorParam.IsDirty() || this->midGradColorParam.IsDirty() ||
        this->maxGradColorParam.IsDirty() || this->specialColorParam.IsDirty()) {

        this->colorLookupTable_.clear();
        this->colorLookupTable_ = {glm::make_vec3(this->minGradColorParam.Param<param::ColorParam>()->Value().data()),
            glm::make_vec3(this->midGradColorParam.Param<param::ColorParam>()->Value().data()),
            glm::make_vec3(this->maxGradColorParam.Param<param::ColorParam>()->Value().data())};

        lastDataHash = mol->DataHash();

        this->currentColoringMode0 =
            static_cast<ProteinColor::ColoringMode>(int(this->coloringModeParam0.Param<param::EnumParam>()->Value()));

        this->currentColoringMode1 =
            static_cast<ProteinColor::ColoringMode>(int(this->coloringModeParam1.Param<param::EnumParam>()->Value()));

        // Mix two coloring modes
        ProteinColor::MakeWeightedColorTable(*mol, this->currentColoringMode0, this->currentColoringMode1,
            cmWeightParam.Param<param::FloatParam>()->Value(), 1.0f - cmWeightParam.Param<param::FloatParam>()->Value(),
            this->atomColorTable_, this->colorLookupTable_, this->fileColorTable_, this->rainbowColors_, bs, nullptr,
            true, this->useNeighborColors.Param<param::BoolParam>()->Value());

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
        std::string tmpStr(this->molIdxListParam.Param<param::StringParam>()->Value());
        this->molIdxList = vislib::StringTokeniser<vislib::CharTraitsA>::Split(tmpStr.c_str(), ';', true);
        this->molIdxListParam.ResetDirty();
    }
}
