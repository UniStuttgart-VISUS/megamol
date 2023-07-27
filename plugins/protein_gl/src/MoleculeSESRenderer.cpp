/*
 * MoleculeSESRenderer.cpp
 *
 * Copyright (C) 2009-2021 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#define _USE_MATH_DEFINES 1

#include "MoleculeSESRenderer.h"

#include <math.h>

#include <ctime>
#include <fstream>
#include <iostream>

#include <glm/gtx/string_cast.hpp>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd/light/DistantLight.h"
#include "mmstd/light/PointLight.h"
#include "protein_calls/ProteinColor.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/Trace.h"
#include "vislib/assert.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/sys/File.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;
using namespace megamol::protein_gl;
using namespace megamol::core::utility::log;

/*
 * MoleculeSESRenderer::MoleculeSESRenderer
 */
MoleculeSESRenderer::MoleculeSESRenderer()
        : Renderer3DModuleGL()
        , molDataCallerSlot("getData", "Connects the protein SES rendering with protein data storage")
        , getLightsSlot("getLights", "Connects the protein SES rendering with light sources")
        , bsDataCallerSlot("getBindingSites", "Connects the molecule rendering with binding site data storage")
        , coloringModeParam0("color::coloringMode0", "The first coloring mode.")
        , coloringModeParam1("color::coloringMode1", "The second coloring mode.")
        , cmWeightParam("color::colorWeighting", "The weighting of the two coloring modes.")
        , minGradColorParam("color::minGradColor", "The color for the minimum value for gradient coloring")
        , midGradColorParam("color::midGradColor", "The color for the middle value for gradient coloring")
        , maxGradColorParam("color::maxGradColor", "The color for the maximum value for gradient coloring")
        , drawSESParam("drawSES", "Draw the SES: ")
        , drawSASParam("drawSAS", "Draw the SAS: ")
        , molIdxListParam("molIdxList", "The list of molecule indices for RS computation:")
        , colorTableFileParam("color::colorTableFilename", "The filename of the color table.")
        , probeRadiusSlot("probeRadius", "The probe radius for the surface computation")
        , computeSesPerMolecule(false)
        , vertexArraySphere_(0)
        , vertexArrayTorus_(0)
        , vertexArrayTria_(0)
        , sphereColorBuffer_(nullptr)
        , sphereVertexBuffer_(nullptr)
        , torusColorBuffer_(nullptr)
        , torusVertexBuffer_(nullptr)
        , torusParamsBuffer_(nullptr)
        , torusQuaternionBuffer_(nullptr)
        , torusSphereBuffer_(nullptr)
        , torusCuttingPlaneBuffer_(nullptr)
        , triaColorBuffer_(nullptr)
        , triaVertexBuffer_(nullptr)
        , triaAttrib1Buffer_(nullptr)
        , triaAttrib2Buffer_(nullptr)
        , triaAttrib3Buffer_(nullptr)
        , triaAttribTexCoord1Buffer_(nullptr)
        , triaAttribTexCoord2Buffer_(nullptr)
        , triaAttribTexCoord3Buffer_(nullptr)
        , pointLightBuffer_(nullptr)
        , directionalLightBuffer_(nullptr)
        , atomCount_(0) {
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->molDataCallerSlot.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->molDataCallerSlot);
    this->getLightsSlot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->getLightsSlot.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->getLightsSlot);
    this->bsDataCallerSlot.SetCompatibleCall<BindingSiteCallDescription>();
    this->MakeSlotAvailable(&this->bsDataCallerSlot);

    // set epsilon value for float-comparison
    this->epsilon = vislib::math::FLOAT_EPSILON;
    // set probe radius
    this->probeRadius = 1.4f;

    this->probeRadiusSlot.SetParameter(new param::FloatParam(1.4f, 0.1f));
    this->MakeSlotAvailable(&this->probeRadiusSlot);

    // coloring modes
    this->currentColoringMode0 = ProteinColor::ColoringMode::CHAIN;
    this->currentColoringMode1 = ProteinColor::ColoringMode::ELEMENT;
    param::EnumParam* cm0 = new param::EnumParam(int(this->currentColoringMode0));
    param::EnumParam* cm1 = new param::EnumParam(int(this->currentColoringMode1));
    MolecularDataCall* mol = new MolecularDataCall();
    BindingSiteCall* bs = new BindingSiteCall();
    unsigned int cCnt;
    ProteinColor::ColoringMode cMode;
    for (cCnt = 0; cCnt < static_cast<uint32_t>(ProteinColor::ColoringMode::MODE_COUNT); ++cCnt) {
        cMode = static_cast<ProteinColor::ColoringMode>(cCnt);
        cm0->SetTypePair(static_cast<int>(cMode), ProteinColor::GetName(cMode).c_str());
        cm1->SetTypePair(static_cast<int>(cMode), ProteinColor::GetName(cMode).c_str());
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

    // the color for the minimum value (gradient coloring
    this->minGradColorParam.SetParameter(new param::ColorParam("#146496"));
    this->MakeSlotAvailable(&this->minGradColorParam);

    // the color for the middle value (gradient coloring
    this->midGradColorParam.SetParameter(new param::ColorParam("#f0f0f0"));
    this->MakeSlotAvailable(&this->midGradColorParam);

    // the color for the maximum value (gradient coloring
    this->maxGradColorParam.SetParameter(new param::ColorParam("#ae3b32"));
    this->MakeSlotAvailable(&this->maxGradColorParam);

    // ----- draw SES param -----
    this->drawSES = true;
    param::BoolParam* sespm = new param::BoolParam(this->drawSES);
    this->drawSESParam << sespm;

    // ----- draw SAS param -----
    this->drawSAS = false;
    param::BoolParam* saspm = new param::BoolParam(this->drawSAS);
    this->drawSASParam << saspm;

    // ----- molecular indices list param -----
    this->molIdxList.Add("0");
    this->molIdxListParam.SetParameter(new param::StringParam("0"));
    this->MakeSlotAvailable(&this->molIdxListParam);

    // fill color table with default values and set the filename param
    std::string filename("colors.txt");
    ProteinColor::ReadColorTableFromFile(filename, this->fileLookupTable);
    this->colorTableFileParam.SetParameter(
        new param::FilePathParam(filename, core::param::FilePathParam::FilePathFlags_::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&this->colorTableFileParam);

    // fill rainbow color table
    ProteinColor::MakeRainbowColorTable(100, this->rainbowColors);

    // width and height of the screen
    this->width = 0;
    this->height = 0;

    // clear singularity texture
    singularityTexture.clear();
    // set singTexData to 0
    this->singTexData = 0;

    this->preComputationDone = false;

    // export parameters
    this->MakeSlotAvailable(&this->drawSESParam);
    this->MakeSlotAvailable(&this->drawSASParam);

    auto defparams = deferredProvider_.getUsedParamSlots();
    for (const auto& param : defparams) {
        this->MakeSlotAvailable(param);
    }
}


/*
 * MoleculeSESRenderer::~MoleculeSESRenderer
 */
MoleculeSESRenderer::~MoleculeSESRenderer() {
    // delete singularity texture
    for (unsigned int i = 0; i < singularityTexture.size(); ++i)
        glDeleteTextures(1, &singularityTexture[i]);

    this->Release();
}


/*
 * protein::MoleculeSESRenderer::release
 */
void MoleculeSESRenderer::release() {}


/*
 * MoleculeSESRenderer::create
 */
bool MoleculeSESRenderer::create() {

    // glEnable( GL_NORMALIZE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);

    try {
        auto const shdr_options = core::utility::make_path_shader_options(
            frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

        sphereShader_ = core::utility::make_shared_glowl_shader("sphere", shdr_options,
            std::filesystem::path("protein_gl/moleculeses/mses_sphere.vert.glsl"),
            std::filesystem::path("protein_gl/moleculeses/mses_sphere.frag.glsl"));

        torusShader_ = core::utility::make_shared_glowl_shader("torus", shdr_options,
            std::filesystem::path("protein_gl/moleculeses/mses_torus.vert.glsl"),
            std::filesystem::path("protein_gl/moleculeses/mses_torus.frag.glsl"));

        sphericalTriangleShader_ = core::utility::make_shared_glowl_shader("sphericaltriangle", shdr_options,
            std::filesystem::path("protein_gl/moleculeses/mses_spherical_triangle.vert.glsl"),
            std::filesystem::path("protein_gl/moleculeses/mses_spherical_triangle.frag.glsl"));
    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[MoleculeSESRenderer] %s", ex.what());
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[MoleculeSESRenderer] Unable to compile shader: Unknown exception: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[MoleculeSESRenderer] Unable to compile shader: Unknown exception.");
    }

    // create the buffer objects
    sphereVertexBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    sphereColorBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    torusVertexBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    torusColorBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    torusParamsBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    torusQuaternionBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    torusSphereBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    torusCuttingPlaneBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    triaVertexBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    triaColorBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    triaAttrib1Buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    triaAttrib2Buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    triaAttrib3Buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    triaAttribTexCoord1Buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    triaAttribTexCoord2Buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    triaAttribTexCoord3Buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    pointLightBuffer_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    directionalLightBuffer_ =
        std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &vertexArraySphere_);
    glBindVertexArray(vertexArraySphere_);

    sphereVertexBuffer_->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    sphereColorBuffer_->bind();
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glGenVertexArrays(1, &vertexArrayTorus_);
    glBindVertexArray(vertexArrayTorus_);

    torusVertexBuffer_->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    torusColorBuffer_->bind();
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    torusParamsBuffer_->bind();
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    torusQuaternionBuffer_->bind();
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    torusSphereBuffer_->bind();
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    torusCuttingPlaneBuffer_->bind();
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    for (int i = 0; i < 6; ++i) {
        glDisableVertexAttribArray(i);
    }

    glGenVertexArrays(1, &vertexArrayTria_);
    glBindVertexArray(vertexArrayTria_);

    triaVertexBuffer_->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    triaColorBuffer_->bind();
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    triaAttrib1Buffer_->bind();
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    triaAttrib2Buffer_->bind();
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    triaAttrib3Buffer_->bind();
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    triaAttribTexCoord1Buffer_->bind();
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    triaAttribTexCoord2Buffer_->bind();
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    triaAttribTexCoord3Buffer_->bind();
    glEnableVertexAttribArray(7);
    glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    for (int i = 0; i < 8; ++i) {
        glDisableVertexAttribArray(i);
    }

    deferredProvider_.setup(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    return true;
}

/*
 * MoleculeSESRenderer::GetExtents
 */
bool MoleculeSESRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {

    MolecularDataCall* mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == NULL)
        return false;
    if (!(*mol)(1))
        return false;

    call.AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    call.SetTimeFramesCount(mol->FrameCount());

    return true;
}

/*
 * MoleculeSESRenderer::Render
 */
bool MoleculeSESRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    // temporary variables
    unsigned int cntRS = 0;

    // get camera information
    this->camera = call.GetCamera();
    view_ = this->camera.getViewMatrix();
    proj_ = this->camera.getProjectionMatrix();
    invview_ = glm::inverse(view_);
    transview_ = glm::transpose(view_);
    invproj_ = glm::inverse(proj_);
    invtransview_ = glm::transpose(invview_);
    mvp_ = proj_ * view_;
    mvpinverse_ = glm::inverse(mvp_);
    mvptranspose_ = glm::transpose(mvp_);

    fbo_ = call.GetFramebuffer();
    deferredProvider_.setFramebufferExtents(fbo_->getWidth(), fbo_->getHeight());

    std::array<int, 2> resolution = {fbo_->getWidth(), fbo_->getHeight()};

    float callTime = call.Time();

    // get pointer to CallProteinData
    MolecularDataCall* mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    // if something went wrong --> return
    if (!mol)
        return false;

    // execute the call
    mol->SetFrameID(static_cast<int>(callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData))
        return false;

    // get pointer to BindingSiteCall
    BindingSiteCall* bs = this->bsDataCallerSlot.CallAs<BindingSiteCall>();
    if (bs) {
        (*bs)(BindingSiteCall::CallForGetData);
    }

    fbo_->bind();

    bool externalfbo = false;
    if (fbo_->getNumColorAttachments() == 3) {
        externalfbo = true;
    } else {
        deferredProvider_.bindDeferredFramebufferToDraw();
    }

    // ==================== check parameters ====================
    this->UpdateParameters(mol, bs);

    // ==================== Precomputations ====================
    this->probeRadius = this->probeRadiusSlot.Param<param::FloatParam>()->Value();

    // init the reduced surfaces
    if (this->reducedSurface.empty()) {
        time_t t = clock();
        // create the reduced surface
        unsigned int chainIds;
        if (!this->computeSesPerMolecule) {
            this->reducedSurface.push_back(new ReducedSurface(mol, this->probeRadius));
            this->reducedSurface.back()->ComputeReducedSurface();
        } else {
            // if no molecule indices are given, compute the SES for all molecules
            if (this->molIdxList.IsEmpty()) {
                for (chainIds = 0; chainIds < mol->MoleculeCount(); ++chainIds) {
                    this->reducedSurface.push_back(new ReducedSurface(chainIds, mol, this->probeRadius));
                    this->reducedSurface.back()->ComputeReducedSurface();
                }
            } else {
                // else compute the SES for all selected molecules
                for (chainIds = 0; chainIds < this->molIdxList.Count(); ++chainIds) {
                    this->reducedSurface.push_back(
                        new ReducedSurface(atoi(this->molIdxList[chainIds]), mol, this->probeRadius));
                    this->reducedSurface.back()->ComputeReducedSurface();
                }
            }
        }
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "%s: RS computed in: %f s\n", this->ClassName(), (double(clock() - t) / double(CLOCKS_PER_SEC)));
    }
    // update the data / the RS
    for (cntRS = 0; cntRS < this->reducedSurface.size(); ++cntRS) {
        if (this->reducedSurface[cntRS]->UpdateData(1.0f, 5.0f)) {
            this->ComputeRaycastingArrays(cntRS);
        }
    }

    if (!this->preComputationDone) {
        this->colorLookupTable = {glm::make_vec3(this->minGradColorParam.Param<param::ColorParam>()->Value().data()),
            glm::make_vec3(this->midGradColorParam.Param<param::ColorParam>()->Value().data()),
            glm::make_vec3(this->maxGradColorParam.Param<param::ColorParam>()->Value().data())};

        // compute the color table
        ProteinColor::MakeWeightedColorTable(*mol, this->currentColoringMode0, this->currentColoringMode1,
            this->cmWeightParam.Param<param::FloatParam>()->Value(), // weight for the first cm
            1.0f - this->cmWeightParam.Param<param::FloatParam>()->Value(), this->atomColorTable,
            this->colorLookupTable, this->fileLookupTable, this->rainbowColors, nullptr, nullptr, true);

        // compute the data needed for the current render mode
        this->ComputeRaycastingArrays();
        // set the precomputation of the data as done
        this->preComputationDone = true;
    }

    bool virtualViewportChanged = false;
    if (static_cast<unsigned int>(std::get<0>(resolution)) != this->width ||
        static_cast<unsigned int>(std::get<1>(resolution)) != this->height) {
        this->width = static_cast<unsigned int>(std::get<0>(resolution));
        this->height = static_cast<unsigned int>(std::get<1>(resolution));
        virtualViewportChanged = true;
    }

    // ==================== Start actual rendering ====================

    this->RenderSESGpuRaycasting(mol);

    if (externalfbo) {
        fbo_->bind();
    } else {
        deferredProvider_.resetToPreviousFramebuffer();
        deferredProvider_.draw(call, this->getLightsSlot.CallAs<core::view::light::CallLight>());
    }

    // unlock the current frame
    mol->Unlock();

    return true;
}


/*
 * update parameters
 */
void MoleculeSESRenderer::UpdateParameters(const MolecularDataCall* mol, const BindingSiteCall* bs) {
    // variables
    bool recomputeColors = false;

    if (atomCount_ != mol->AtomCount()) {
        atomCount_ = mol->AtomCount();
        reducedSurface.clear();
        this->preComputationDone = false;
    }

    // ==================== check parameters ====================
    if (this->coloringModeParam0.IsDirty() || this->coloringModeParam1.IsDirty() || this->cmWeightParam.IsDirty()) {
        this->currentColoringMode0 =
            static_cast<ProteinColor::ColoringMode>(this->coloringModeParam0.Param<param::EnumParam>()->Value());
        this->currentColoringMode1 =
            static_cast<ProteinColor::ColoringMode>(this->coloringModeParam1.Param<param::EnumParam>()->Value());

        this->colorLookupTable = {glm::make_vec3(this->minGradColorParam.Param<param::ColorParam>()->Value().data()),
            glm::make_vec3(this->midGradColorParam.Param<param::ColorParam>()->Value().data()),
            glm::make_vec3(this->maxGradColorParam.Param<param::ColorParam>()->Value().data())};

        ProteinColor::MakeWeightedColorTable(*mol, this->currentColoringMode0, this->currentColoringMode1,
            this->cmWeightParam.Param<param::FloatParam>()->Value(), // weight for the first cm
            1.0f - this->cmWeightParam.Param<param::FloatParam>()->Value(), this->atomColorTable,
            this->colorLookupTable, this->fileLookupTable, this->rainbowColors, nullptr, nullptr, true);

        this->preComputationDone = false;
        this->coloringModeParam0.ResetDirty();
        this->coloringModeParam1.ResetDirty();
        this->cmWeightParam.ResetDirty();
    }
    if (this->drawSESParam.IsDirty()) {
        this->drawSES = this->drawSESParam.Param<param::BoolParam>()->Value();
        this->drawSESParam.ResetDirty();
    }
    if (this->drawSASParam.IsDirty()) {
        this->drawSAS = this->drawSASParam.Param<param::BoolParam>()->Value();
        this->drawSASParam.ResetDirty();
        this->preComputationDone = false;
    }
    if (this->molIdxListParam.IsDirty()) {
        std::string tmpStr(this->molIdxListParam.Param<param::StringParam>()->Value());
        this->molIdxList = vislib::StringTokeniser<vislib::CharTraitsA>::Split(tmpStr.c_str(), ';', true);
        this->molIdxListParam.ResetDirty();
    }
    // color table param
    if (this->colorTableFileParam.IsDirty()) {
        ProteinColor::ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::FilePathParam>()->Value(), this->fileLookupTable);
        this->colorTableFileParam.ResetDirty();
        recomputeColors = true;
    }
    if (this->probeRadiusSlot.IsDirty()) {
        this->probeRadius = this->probeRadiusSlot.Param<param::FloatParam>()->Value();
        this->reducedSurface.clear();
        this->preComputationDone = false;
        this->probeRadiusSlot.ResetDirty();
    }

    if (recomputeColors) {
        this->preComputationDone = false;
    }
}

/*
 * Render the molecular surface using GPU raycasting
 */
void MoleculeSESRenderer::RenderSESGpuRaycasting(const MolecularDataCall* mol) {
    // TODO: attribute locations nicht jedes mal neu abfragen!

    bool virtualViewportChanged = false;
    if (static_cast<unsigned int>(fbo_->getWidth()) != this->width ||
        static_cast<unsigned int>(fbo_->getHeight()) != this->height) {
        this->width = static_cast<unsigned int>(fbo_->getWidth());
        this->height = static_cast<unsigned int>(fbo_->getHeight());
        virtualViewportChanged = true;
    }

    // set viewport
    glm::vec4 viewportStuff;
    viewportStuff[0] = 0.0f;
    viewportStuff[1] = 0.0f;
    viewportStuff[2] = static_cast<float>(fbo_->getWidth());
    viewportStuff[3] = static_cast<float>(fbo_->getHeight());
    if (viewportStuff[2] < 1.0f)
        viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f)
        viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glm::vec3 camdir = camera.get<core::view::Camera::Pose>().direction;
    glm::vec3 right = camera.get<core::view::Camera::Pose>().right;
    glm::vec3 up = camera.get<core::view::Camera::Pose>().up;
    float nearplane = camera.get<core::view::Camera::NearPlane>();
    float farplane = camera.get<core::view::Camera::FarPlane>();

    unsigned int cntRS;

    for (cntRS = 0; cntRS < this->reducedSurface.size(); ++cntRS) {
        //////////////////////////////////
        // ray cast the tori on the GPU //
        //////////////////////////////////
        GLuint attribInParams;
        GLuint attribQuatC;
        GLuint attribInSphere;
        GLuint attribInColors;
        GLuint attribInCuttingPlane;

        if (this->drawSES) {

            torusVertexBuffer_->rebuffer(
                torusVertexArray[cntRS].PeekElements(), torusVertexArray[cntRS].Count() * sizeof(float));
            torusColorBuffer_->rebuffer(torusColors[cntRS].PeekElements(), torusColors[cntRS].Count() * sizeof(float));
            torusParamsBuffer_->rebuffer(
                torusInParamArray[cntRS].PeekElements(), torusInParamArray[cntRS].Count() * sizeof(float));
            torusQuaternionBuffer_->rebuffer(
                torusQuatCArray[cntRS].PeekElements(), torusQuatCArray[cntRS].Count() * sizeof(float));
            torusSphereBuffer_->rebuffer(
                torusInSphereArray[cntRS].PeekElements(), torusInSphereArray[cntRS].Count() * sizeof(float));
            torusCuttingPlaneBuffer_->rebuffer(torusInCuttingPlaneArray[cntRS].PeekElements(),
                torusInCuttingPlaneArray[cntRS].Count() * sizeof(float));

            glBindVertexArray(vertexArrayTorus_);
            // enable torus shader
            torusShader_->use();
            // set shader variables
            torusShader_->setUniform("viewAttr", viewportStuff);
            torusShader_->setUniform("camIn", camdir);
            torusShader_->setUniform("camRight", right);
            torusShader_->setUniform("camUp", up);
            torusShader_->setUniform("zValues", nearplane, farplane);
            torusShader_->setUniform("view", view_);
            torusShader_->setUniform("proj", proj_);
            torusShader_->setUniform("viewInverse", invview_);
            torusShader_->setUniform("mvp", mvp_);
            torusShader_->setUniform("mvpinverse", mvpinverse_);
            torusShader_->setUniform("mvptransposed", mvptranspose_);

            glDrawArrays(GL_POINTS, 0, ((unsigned int)this->torusVertexArray[cntRS].Count()) / 3);

            glUseProgram(0);
            glBindVertexArray(0);

            /////////////////////////////////////////////////
            // ray cast the spherical triangles on the GPU //
            /////////////////////////////////////////////////
            GLuint attribVec1;
            GLuint attribVec2;
            GLuint attribVec3;
            GLuint attribTexCoord1;
            GLuint attribTexCoord2;
            GLuint attribTexCoord3;
            GLuint attribColors;

            triaVertexBuffer_->rebuffer(
                sphericTriaVertexArray[cntRS].PeekElements(), sphericTriaVertexArray[cntRS].Count() * sizeof(float));
            triaColorBuffer_->rebuffer(
                sphericTriaColors[cntRS].PeekElements(), sphericTriaColors[cntRS].Count() * sizeof(float));
            triaAttrib1Buffer_->rebuffer(
                sphericTriaVec1[cntRS].PeekElements(), sphericTriaVec1[cntRS].Count() * sizeof(float));
            triaAttrib2Buffer_->rebuffer(
                sphericTriaVec2[cntRS].PeekElements(), sphericTriaVec2[cntRS].Count() * sizeof(float));
            triaAttrib3Buffer_->rebuffer(
                sphericTriaVec3[cntRS].PeekElements(), sphericTriaVec3[cntRS].Count() * sizeof(float));
            triaAttribTexCoord1Buffer_->rebuffer(
                sphericTriaTexCoord1[cntRS].PeekElements(), sphericTriaTexCoord1[cntRS].Count() * sizeof(float));
            triaAttribTexCoord2Buffer_->rebuffer(
                sphericTriaTexCoord2[cntRS].PeekElements(), sphericTriaTexCoord2[cntRS].Count() * sizeof(float));
            triaAttribTexCoord3Buffer_->rebuffer(
                sphericTriaTexCoord3[cntRS].PeekElements(), sphericTriaTexCoord3[cntRS].Count() * sizeof(float));

            // bind texture
            glBindTexture(GL_TEXTURE_2D, singularityTexture[cntRS]);
            glBindVertexArray(vertexArrayTria_);
            // enable spherical triangle shader
            sphericalTriangleShader_->use();

            sphericalTriangleShader_->setUniform("viewAttr", viewportStuff);
            sphericalTriangleShader_->setUniform("camIn", camdir);
            sphericalTriangleShader_->setUniform("camRight", right);
            sphericalTriangleShader_->setUniform("camUp", up);
            sphericalTriangleShader_->setUniform("zValues", nearplane, farplane);
            sphericalTriangleShader_->setUniform(
                "texOffset", 1.0f / (float)this->singTexWidth[cntRS], 1.0f / (float)this->singTexHeight[cntRS]);
            sphericalTriangleShader_->setUniform("view", view_);
            sphericalTriangleShader_->setUniform("proj", proj_);
            sphericalTriangleShader_->setUniform("viewInverse", invview_);
            sphericalTriangleShader_->setUniform("mvp", mvp_);
            sphericalTriangleShader_->setUniform("mvpinverse", mvpinverse_);
            sphericalTriangleShader_->setUniform("mvptransposed", mvptranspose_);

            glDrawArrays(GL_POINTS, 0, ((unsigned int)this->sphericTriaVertexArray[cntRS].Count()) / 4);

            // disable spherical triangle shader
            glUseProgram(0);
            // unbind texture
            glBindVertexArray(0);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        /////////////////////////////////////
        // ray cast the spheres on the GPU //
        /////////////////////////////////////
        this->sphereVertexBuffer_->rebuffer(
            this->sphereVertexArray[cntRS].PeekElements(), this->sphereVertexArray[cntRS].Count() * sizeof(float));
        this->sphereColorBuffer_->rebuffer(
            this->sphereColors[cntRS].PeekElements(), this->sphereColors[cntRS].Count() * sizeof(float));
        glBindVertexArray(vertexArraySphere_);

        sphereShader_->use();

        // set shader variables
        sphereShader_->setUniform("viewAttr", viewportStuff);
        sphereShader_->setUniform("camIn", camdir);
        sphereShader_->setUniform("camRight", right);
        sphereShader_->setUniform("camUp", up);
        sphereShader_->setUniform("zValues", nearplane, farplane);
        sphereShader_->setUniform("view", view_);
        sphereShader_->setUniform("proj", proj_);
        sphereShader_->setUniform("viewInverse", invview_);
        sphereShader_->setUniform("mvp", mvp_);
        sphereShader_->setUniform("mvpinverse", mvpinverse_);
        sphereShader_->setUniform("mvptransposed", mvptranspose_);

        glDrawArrays(GL_POINTS, 0, ((unsigned int)this->sphereVertexArray[cntRS].Count()) / 4);

        // disable sphere shader
        glUseProgram(0);
        glBindVertexArray(0);
    }
}

/*
 * Compute the vertex and attribute arrays for the raycasting shaders
 * (spheres, spherical triangles & tori)
 */
void MoleculeSESRenderer::ComputeRaycastingArrays() {
    // time_t t = clock();

    unsigned int cntRS;
    unsigned int i;

    // resize lists of vertex, attribute and color arrays
    this->sphericTriaVertexArray.resize(this->reducedSurface.size());
    this->sphericTriaVec1.resize(this->reducedSurface.size());
    this->sphericTriaVec2.resize(this->reducedSurface.size());
    this->sphericTriaVec3.resize(this->reducedSurface.size());
    this->sphericTriaTexCoord1.resize(this->reducedSurface.size());
    this->sphericTriaTexCoord2.resize(this->reducedSurface.size());
    this->sphericTriaTexCoord3.resize(this->reducedSurface.size());
    this->sphericTriaColors.resize(this->reducedSurface.size());
    this->torusVertexArray.resize(this->reducedSurface.size());
    this->torusInParamArray.resize(this->reducedSurface.size());
    this->torusQuatCArray.resize(this->reducedSurface.size());
    this->torusInSphereArray.resize(this->reducedSurface.size());
    this->torusColors.resize(this->reducedSurface.size());
    this->torusInCuttingPlaneArray.resize(this->reducedSurface.size());
    this->sphereVertexArray.resize(this->reducedSurface.size());
    this->sphereColors.resize(this->reducedSurface.size());


    // compute singulatity textures
    this->CreateSingularityTextures();

    for (cntRS = 0; cntRS < this->reducedSurface.size(); ++cntRS) {
        ///////////////////////////////////////////////////////////////////////
        // compute arrays for ray casting the spherical triangles on the GPU //
        ///////////////////////////////////////////////////////////////////////
        vislib::math::Vector<float, 3> tmpVec;
        vislib::math::Vector<float, 3> tmpDualProbe(1.0f, 1.0f, 1.0f);
        float dualProbeRad = 0.0f;

        this->sphericTriaVertexArray[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSFaceCount() * 4);
        this->sphericTriaVec1[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSFaceCount() * 4);
        this->sphericTriaVec2[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSFaceCount() * 4);
        this->sphericTriaVec3[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSFaceCount() * 4);
        this->sphericTriaTexCoord1[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSFaceCount() * 3);
        this->sphericTriaTexCoord2[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSFaceCount() * 3);
        this->sphericTriaTexCoord3[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSFaceCount() * 3);
        this->sphericTriaColors[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSFaceCount() * 3);

        this->probeRadius = this->probeRadiusSlot.Param<param::FloatParam>()->Value();

        // loop over all RS-faces
        for (i = 0; i < this->reducedSurface[cntRS]->GetRSFaceCount(); ++i) {
            // if the face has a dual face --> store the probe of this face
            if (this->reducedSurface[cntRS]->GetRSFace(i)->GetDualFace() != NULL) {
                tmpDualProbe = this->reducedSurface[cntRS]->GetRSFace(i)->GetDualFace()->GetProbeCenter();
                dualProbeRad = this->probeRadius;
            }
            // first RS-vertex
            tmpVec = this->reducedSurface[cntRS]->GetRSFace(i)->GetVertex1()->GetPosition() -
                     this->reducedSurface[cntRS]->GetRSFace(i)->GetProbeCenter();
            this->sphericTriaVec1[cntRS][i * 4 + 0] = tmpVec.GetX();
            this->sphericTriaVec1[cntRS][i * 4 + 1] = tmpVec.GetY();
            this->sphericTriaVec1[cntRS][i * 4 + 2] = tmpVec.GetZ();
            this->sphericTriaVec1[cntRS][i * 4 + 3] = 1.0f;
            // second RS-vertex
            tmpVec = this->reducedSurface[cntRS]->GetRSFace(i)->GetVertex2()->GetPosition() -
                     this->reducedSurface[cntRS]->GetRSFace(i)->GetProbeCenter();
            this->sphericTriaVec2[cntRS][i * 4 + 0] = tmpVec.GetX();
            this->sphericTriaVec2[cntRS][i * 4 + 1] = tmpVec.GetY();
            this->sphericTriaVec2[cntRS][i * 4 + 2] = tmpVec.GetZ();
            this->sphericTriaVec2[cntRS][i * 4 + 3] = 1.0f;
            // third RS-vertex
            tmpVec = this->reducedSurface[cntRS]->GetRSFace(i)->GetVertex3()->GetPosition() -
                     this->reducedSurface[cntRS]->GetRSFace(i)->GetProbeCenter();
            this->sphericTriaVec3[cntRS][i * 4 + 0] = tmpVec.GetX();
            this->sphericTriaVec3[cntRS][i * 4 + 1] = tmpVec.GetY();
            this->sphericTriaVec3[cntRS][i * 4 + 2] = tmpVec.GetZ();
            this->sphericTriaVec3[cntRS][i * 4 + 3] = dualProbeRad * dualProbeRad;
            // store number of cutting probes and texture coordinates for each edge
            this->sphericTriaTexCoord1[cntRS][i * 3 + 0] =
                (float)this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge1()->cuttingProbes.size();
            this->sphericTriaTexCoord1[cntRS][i * 3 + 1] =
                (float)this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge1()->GetTexCoordX();
            this->sphericTriaTexCoord1[cntRS][i * 3 + 2] =
                (float)this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge1()->GetTexCoordY();
            this->sphericTriaTexCoord2[cntRS][i * 3 + 0] =
                (float)this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge2()->cuttingProbes.size();
            this->sphericTriaTexCoord2[cntRS][i * 3 + 1] =
                (float)this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge2()->GetTexCoordX();
            this->sphericTriaTexCoord2[cntRS][i * 3 + 2] =
                (float)this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge2()->GetTexCoordY();
            this->sphericTriaTexCoord3[cntRS][i * 3 + 0] =
                (float)this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge3()->cuttingProbes.size();
            this->sphericTriaTexCoord3[cntRS][i * 3 + 1] =
                (float)this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge3()->GetTexCoordX();
            this->sphericTriaTexCoord3[cntRS][i * 3 + 2] =
                (float)this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge3()->GetTexCoordY();
            // colors
            this->sphericTriaColors[cntRS][i * 3 + 0] =
                CodeColor(&this->atomColorTable[this->reducedSurface[cntRS]->GetRSFace(i)->GetVertex1()->GetIndex()].x);
            this->sphericTriaColors[cntRS][i * 3 + 1] =
                CodeColor(&this->atomColorTable[this->reducedSurface[cntRS]->GetRSFace(i)->GetVertex2()->GetIndex()].x);
            this->sphericTriaColors[cntRS][i * 3 + 2] =
                CodeColor(&this->atomColorTable[this->reducedSurface[cntRS]->GetRSFace(i)->GetVertex3()->GetIndex()].x);
            // sphere center
            this->sphericTriaVertexArray[cntRS][i * 4 + 0] =
                this->reducedSurface[cntRS]->GetRSFace(i)->GetProbeCenter().GetX();
            this->sphericTriaVertexArray[cntRS][i * 4 + 1] =
                this->reducedSurface[cntRS]->GetRSFace(i)->GetProbeCenter().GetY();
            this->sphericTriaVertexArray[cntRS][i * 4 + 2] =
                this->reducedSurface[cntRS]->GetRSFace(i)->GetProbeCenter().GetZ();
            this->sphericTriaVertexArray[cntRS][i * 4 + 3] = this->GetProbeRadius();
        }

        ////////////////////////////////////////////////////////
        // compute arrays for ray casting the tori on the GPU //
        ////////////////////////////////////////////////////////
        vislib::math::Quaternion<float> quatC;
        vislib::math::Vector<float, 3> zAxis, torusAxis, rotAxis, P, X1, X2, C, planeNormal;
        zAxis.Set(0.0f, 0.0f, 1.0f);
        float distance, d;
        vislib::math::Vector<float, 3> tmpDir1, tmpDir2, tmpDir3, cutPlaneNorm;

        this->torusVertexArray[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSEdgeCount() * 3);
        this->torusInParamArray[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSEdgeCount() * 3);
        this->torusQuatCArray[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSEdgeCount() * 4);
        this->torusInSphereArray[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSEdgeCount() * 4);
        this->torusColors[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSEdgeCount() * 4);
        this->torusInCuttingPlaneArray[cntRS].SetCount(this->reducedSurface[cntRS]->GetRSEdgeCount() * 3);

        // loop over all RS-edges
        for (i = 0; i < this->reducedSurface[cntRS]->GetRSEdgeCount(); ++i) {
            // get the rotation axis of the torus
            torusAxis = this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex1()->GetPosition() -
                        this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusCenter();
            torusAxis.Normalise();
            // get the axis for rotating the torus rotations axis on the z-axis
            rotAxis = torusAxis.Cross(zAxis);
            rotAxis.Normalise();
            // compute quaternion
            quatC.Set(torusAxis.Angle(zAxis), rotAxis);
            // compute the tangential point X2 of the spheres
            P = this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusCenter() +
                rotAxis * this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusRadius();

            X1 = P - this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex1()->GetPosition();
            X1.Normalise();
            X1 *= this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex1()->GetRadius();
            X2 = P - this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex2()->GetPosition();
            X2.Normalise();
            X2 *= this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex2()->GetRadius();
            d = (X1 + this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex1()->GetPosition() -
                 this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusCenter())
                    .Dot(torusAxis);

            C = this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex1()->GetPosition() -
                this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex2()->GetPosition();
            C = ((P - this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex2()->GetPosition()).Length() /
                    ((P - this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex1()->GetPosition()).Length() +
                        (P - this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex2()->GetPosition()).Length())) *
                C;
            distance = (X2 - C).Length();
            C = (C + this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex2()->GetPosition()) -
                this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusCenter();

            // compute normal of the cutting plane
            tmpDir1 = this->reducedSurface[cntRS]->GetRSEdge(i)->GetFace1()->GetProbeCenter();
            tmpDir2 = this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex2()->GetPosition() - tmpDir1;
            tmpDir2.Normalise();
            tmpDir2 *= this->probeRadius;
            tmpDir2 = tmpDir2 + tmpDir1 - this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusCenter();
            tmpDir3 = this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex1()->GetPosition() - tmpDir1;
            tmpDir3.Normalise();
            tmpDir3 *= this->probeRadius;
            tmpDir3 = tmpDir3 + tmpDir1 - this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusCenter();
            // tmpDir2 and tmpDir3 now store the position of the intersection points for face 1
            cutPlaneNorm = tmpDir2 - tmpDir3;
            // cutPlaneNorm now stores the vector between the two intersection points for face 1
            tmpDir1 = this->reducedSurface[cntRS]->GetRSEdge(i)->GetFace2()->GetProbeCenter();
            tmpDir2 = this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex1()->GetPosition() - tmpDir1;
            tmpDir2.Normalise();
            tmpDir2 *= this->probeRadius;
            tmpDir2 = tmpDir2 + tmpDir1 - this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusCenter();
            // tmpDir2 now stores the position of the intersection point 1 for face 2
            tmpDir2 = tmpDir2 - tmpDir3;
            // tmpDir2 and tmpDir3 now span the plane containing the four intersection points
            cutPlaneNorm = cutPlaneNorm.Cross(tmpDir2);
            cutPlaneNorm = torusAxis.Cross(cutPlaneNorm);
            cutPlaneNorm.Normalise();

            // attributes
            this->torusInParamArray[cntRS][i * 3 + 0] = this->probeRadius;
            this->torusInParamArray[cntRS][i * 3 + 1] = this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusRadius();
            this->torusInParamArray[cntRS][i * 3 + 2] = this->reducedSurface[cntRS]->GetRSEdge(i)->GetRotationAngle();
            this->torusQuatCArray[cntRS][i * 4 + 0] = quatC.GetX();
            this->torusQuatCArray[cntRS][i * 4 + 1] = quatC.GetY();
            this->torusQuatCArray[cntRS][i * 4 + 2] = quatC.GetZ();
            this->torusQuatCArray[cntRS][i * 4 + 3] = quatC.GetW();
            this->torusInSphereArray[cntRS][i * 4 + 0] = C.GetX();
            this->torusInSphereArray[cntRS][i * 4 + 1] = C.GetY();
            this->torusInSphereArray[cntRS][i * 4 + 2] = C.GetZ();
            this->torusInSphereArray[cntRS][i * 4 + 3] = distance;
            // colors
            this->torusColors[cntRS][i * 4 + 0] =
                CodeColor(&this->atomColorTable[this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex1()->GetIndex()].x);
            this->torusColors[cntRS][i * 4 + 1] =
                CodeColor(&this->atomColorTable[this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex2()->GetIndex()].x);
            this->torusColors[cntRS][i * 4 + 2] = d;
            // this->torusColors[cntRS][i*4+3] = ( X2 - X1).Length();
            this->torusColors[cntRS][i * 4 + 3] =
                (X2 + this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex2()->GetPosition() -
                    this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusCenter())
                    .Dot(torusAxis) -
                d;
            // cutting plane
            this->torusInCuttingPlaneArray[cntRS][i * 3 + 0] = cutPlaneNorm.GetX();
            this->torusInCuttingPlaneArray[cntRS][i * 3 + 1] = cutPlaneNorm.GetY();
            this->torusInCuttingPlaneArray[cntRS][i * 3 + 2] = cutPlaneNorm.GetZ();
            // torus center
            this->torusVertexArray[cntRS][i * 3 + 0] =
                this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusCenter().GetX();
            this->torusVertexArray[cntRS][i * 3 + 1] =
                this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusCenter().GetY();
            this->torusVertexArray[cntRS][i * 3 + 2] =
                this->reducedSurface[cntRS]->GetRSEdge(i)->GetTorusCenter().GetZ();
        }

        ///////////////////////////////////////////////////////////
        // compute arrays for ray casting the spheres on the GPU //
        ///////////////////////////////////////////////////////////
        /*
        this->sphereVertexArray[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSVertexCount() * 4);
        this->sphereColors[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSVertexCount() * 3);
        */
        this->sphereVertexArray[cntRS].AssertCapacity(this->reducedSurface[cntRS]->GetRSVertexCount() * 4);
        this->sphereVertexArray[cntRS].Clear();
        this->sphereColors[cntRS].AssertCapacity(this->reducedSurface[cntRS]->GetRSVertexCount() * 3);
        this->sphereColors[cntRS].Clear();

        // loop over all RS-vertices (i.e. all protein atoms)
        for (i = 0; i < this->reducedSurface[cntRS]->GetRSVertexCount(); ++i) {
            // add only surface atoms (i.e. with not buried RS-vertices)
            if (this->reducedSurface[cntRS]->GetRSVertex(i)->IsBuried())
                continue;
            // set vertex color
            this->sphereColors[cntRS].Append(
                this->atomColorTable[this->reducedSurface[cntRS]->GetRSVertex(i)->GetIndex()].x);
            this->sphereColors[cntRS].Append(
                this->atomColorTable[this->reducedSurface[cntRS]->GetRSVertex(i)->GetIndex()].y);
            this->sphereColors[cntRS].Append(
                this->atomColorTable[this->reducedSurface[cntRS]->GetRSVertex(i)->GetIndex()].z);
            // set vertex position
            this->sphereVertexArray[cntRS].Append(this->reducedSurface[cntRS]->GetRSVertex(i)->GetPosition().GetX());
            this->sphereVertexArray[cntRS].Append(this->reducedSurface[cntRS]->GetRSVertex(i)->GetPosition().GetY());
            this->sphereVertexArray[cntRS].Append(this->reducedSurface[cntRS]->GetRSVertex(i)->GetPosition().GetZ());
            if (this->drawSAS) {
                this->sphereVertexArray[cntRS].Append(
                    this->reducedSurface[cntRS]->GetRSVertex(i)->GetRadius() + this->probeRadius);
            } else {
                this->sphereVertexArray[cntRS].Append(this->reducedSurface[cntRS]->GetRSVertex(i)->GetRadius());
            }
        }
    }
    // print the time of the computation
    // std::cout << "computation of arrays for GPU ray casting finished:" << ( double( clock() - t) / double(
    // CLOCKS_PER_SEC) ) << std::endl;
}


/*
 * Compute the vertex and attribute arrays for the raycasting shaders
 * (spheres, spherical triangles & tori)
 */
void MoleculeSESRenderer::ComputeRaycastingArrays(unsigned int idxRS) {
    // do nothing if the given index is out of bounds
    if (idxRS > this->reducedSurface.size())
        return;

    this->probeRadius = this->probeRadiusSlot.Param<param::FloatParam>()->Value();

    // check if all arrays have the correct size
    if (this->sphericTriaVertexArray.size() != this->reducedSurface.size() ||
        this->sphericTriaVec1.size() != this->reducedSurface.size() ||
        this->sphericTriaVec2.size() != this->reducedSurface.size() ||
        this->sphericTriaVec3.size() != this->reducedSurface.size() ||
        this->sphericTriaTexCoord1.size() != this->reducedSurface.size() ||
        this->sphericTriaTexCoord2.size() != this->reducedSurface.size() ||
        this->sphericTriaTexCoord3.size() != this->reducedSurface.size() ||
        this->sphericTriaColors.size() != this->reducedSurface.size() ||
        this->torusVertexArray.size() != this->reducedSurface.size() ||
        this->torusInParamArray.size() != this->reducedSurface.size() ||
        this->torusQuatCArray.size() != this->reducedSurface.size() ||
        this->torusInSphereArray.size() != this->reducedSurface.size() ||
        this->torusColors.size() != this->reducedSurface.size() ||
        this->torusInCuttingPlaneArray.size() != this->reducedSurface.size() ||
        this->sphereVertexArray.size() != this->reducedSurface.size() ||
        this->sphereColors.size() != this->reducedSurface.size()) {
        // recompute everything if one of the arrays has the wrong size
        // ComputeRaycastingArrays();
        this->preComputationDone = false;
        return;
    }

    unsigned int i;

    // compute singulatity textures
    this->CreateSingularityTexture(idxRS);

    ///////////////////////////////////////////////////////////////////////
    // compute arrays for ray casting the spherical triangles on the GPU //
    ///////////////////////////////////////////////////////////////////////
    vislib::math::Vector<float, 3> tmpVec;
    vislib::math::Vector<float, 3> tmpDualProbe(1.0f, 1.0f, 1.0f);
    float dualProbeRad = 0.0f;

    this->sphericTriaVertexArray[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSFaceCount() * 4);
    this->sphericTriaVec1[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSFaceCount() * 4);
    this->sphericTriaVec2[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSFaceCount() * 4);
    this->sphericTriaVec3[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSFaceCount() * 4);
    this->sphericTriaTexCoord1[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSFaceCount() * 3);
    this->sphericTriaTexCoord2[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSFaceCount() * 3);
    this->sphericTriaTexCoord3[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSFaceCount() * 3);
    this->sphericTriaColors[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSFaceCount() * 3);

    // loop over all RS-faces
    for (i = 0; i < this->reducedSurface[idxRS]->GetRSFaceCount(); ++i) {
        // if the face has a dual face --> store the probe of this face
        if (this->reducedSurface[idxRS]->GetRSFace(i)->GetDualFace() != NULL) {
            tmpDualProbe = this->reducedSurface[idxRS]->GetRSFace(i)->GetDualFace()->GetProbeCenter();
            dualProbeRad = this->probeRadius;
        }
        // first RS-vertex
        tmpVec = this->reducedSurface[idxRS]->GetRSFace(i)->GetVertex1()->GetPosition() -
                 this->reducedSurface[idxRS]->GetRSFace(i)->GetProbeCenter();
        this->sphericTriaVec1[idxRS][i * 4 + 0] = tmpVec.GetX();
        this->sphericTriaVec1[idxRS][i * 4 + 1] = tmpVec.GetY();
        this->sphericTriaVec1[idxRS][i * 4 + 2] = tmpVec.GetZ();
        this->sphericTriaVec1[idxRS][i * 4 + 3] = 1.0f;
        // second RS-vertex
        tmpVec = this->reducedSurface[idxRS]->GetRSFace(i)->GetVertex2()->GetPosition() -
                 this->reducedSurface[idxRS]->GetRSFace(i)->GetProbeCenter();
        this->sphericTriaVec2[idxRS][i * 4 + 0] = tmpVec.GetX();
        this->sphericTriaVec2[idxRS][i * 4 + 1] = tmpVec.GetY();
        this->sphericTriaVec2[idxRS][i * 4 + 2] = tmpVec.GetZ();
        this->sphericTriaVec2[idxRS][i * 4 + 3] = 1.0f;
        // third RS-vertex
        tmpVec = this->reducedSurface[idxRS]->GetRSFace(i)->GetVertex3()->GetPosition() -
                 this->reducedSurface[idxRS]->GetRSFace(i)->GetProbeCenter();
        this->sphericTriaVec3[idxRS][i * 4 + 0] = tmpVec.GetX();
        this->sphericTriaVec3[idxRS][i * 4 + 1] = tmpVec.GetY();
        this->sphericTriaVec3[idxRS][i * 4 + 2] = tmpVec.GetZ();
        this->sphericTriaVec3[idxRS][i * 4 + 3] = dualProbeRad * dualProbeRad;
        // store number of cutting probes and texture coordinates for each edge
        this->sphericTriaTexCoord1[idxRS][i * 3 + 0] =
            (float)this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge1()->cuttingProbes.size();
        this->sphericTriaTexCoord1[idxRS][i * 3 + 1] =
            (float)this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge1()->GetTexCoordX();
        this->sphericTriaTexCoord1[idxRS][i * 3 + 2] =
            (float)this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge1()->GetTexCoordY();
        this->sphericTriaTexCoord2[idxRS][i * 3 + 0] =
            (float)this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge2()->cuttingProbes.size();
        this->sphericTriaTexCoord2[idxRS][i * 3 + 1] =
            (float)this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge2()->GetTexCoordX();
        this->sphericTriaTexCoord2[idxRS][i * 3 + 2] =
            (float)this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge2()->GetTexCoordY();
        this->sphericTriaTexCoord3[idxRS][i * 3 + 0] =
            (float)this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge3()->cuttingProbes.size();
        this->sphericTriaTexCoord3[idxRS][i * 3 + 1] =
            (float)this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge3()->GetTexCoordX();
        this->sphericTriaTexCoord3[idxRS][i * 3 + 2] =
            (float)this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge3()->GetTexCoordY();
        // colors
        this->sphericTriaColors[idxRS][i * 3 + 0] =
            CodeColor(&this->atomColorTable[this->reducedSurface[idxRS]->GetRSFace(i)->GetVertex1()->GetIndex()].x);
        this->sphericTriaColors[idxRS][i * 3 + 1] =
            CodeColor(&this->atomColorTable[this->reducedSurface[idxRS]->GetRSFace(i)->GetVertex2()->GetIndex()].x);
        this->sphericTriaColors[idxRS][i * 3 + 2] =
            CodeColor(&this->atomColorTable[this->reducedSurface[idxRS]->GetRSFace(i)->GetVertex3()->GetIndex()].x);
        // sphere center
        this->sphericTriaVertexArray[idxRS][i * 4 + 0] =
            this->reducedSurface[idxRS]->GetRSFace(i)->GetProbeCenter().GetX();
        this->sphericTriaVertexArray[idxRS][i * 4 + 1] =
            this->reducedSurface[idxRS]->GetRSFace(i)->GetProbeCenter().GetY();
        this->sphericTriaVertexArray[idxRS][i * 4 + 2] =
            this->reducedSurface[idxRS]->GetRSFace(i)->GetProbeCenter().GetZ();
        this->sphericTriaVertexArray[idxRS][i * 4 + 3] = this->GetProbeRadius();
    }

    ////////////////////////////////////////////////////////
    // compute arrays for ray casting the tori on the GPU //
    ////////////////////////////////////////////////////////
    vislib::math::Quaternion<float> quatC;
    vislib::math::Vector<float, 3> zAxis, torusAxis, rotAxis, P, X1, X2, C, planeNormal;
    zAxis.Set(0.0f, 0.0f, 1.0f);
    float distance, d;
    vislib::math::Vector<float, 3> tmpDir1, tmpDir2, tmpDir3, cutPlaneNorm;

    this->torusVertexArray[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSEdgeCount() * 3);
    this->torusInParamArray[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSEdgeCount() * 3);
    this->torusQuatCArray[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSEdgeCount() * 4);
    this->torusInSphereArray[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSEdgeCount() * 4);
    this->torusColors[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSEdgeCount() * 4);
    this->torusInCuttingPlaneArray[idxRS].SetCount(this->reducedSurface[idxRS]->GetRSEdgeCount() * 3);

    // loop over all RS-edges
    for (i = 0; i < this->reducedSurface[idxRS]->GetRSEdgeCount(); ++i) {
        // get the rotation axis of the torus
        torusAxis = this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex1()->GetPosition() -
                    this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusCenter();
        torusAxis.Normalise();
        // get the axis for rotating the torus rotations axis on the z-axis
        rotAxis = torusAxis.Cross(zAxis);
        rotAxis.Normalise();
        // compute quaternion
        quatC.Set(torusAxis.Angle(zAxis), rotAxis);
        // compute the tangential point X2 of the spheres
        P = this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusCenter() +
            rotAxis * this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusRadius();

        X1 = P - this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex1()->GetPosition();
        X1.Normalise();
        X1 *= this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex1()->GetRadius();
        X2 = P - this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex2()->GetPosition();
        X2.Normalise();
        X2 *= this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex2()->GetRadius();
        d = (X1 + this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex1()->GetPosition() -
             this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusCenter())
                .Dot(torusAxis);

        C = this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex1()->GetPosition() -
            this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex2()->GetPosition();
        C = ((P - this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex2()->GetPosition()).Length() /
                ((P - this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex1()->GetPosition()).Length() +
                    (P - this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex2()->GetPosition()).Length())) *
            C;
        distance = (X2 - C).Length();
        C = (C + this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex2()->GetPosition()) -
            this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusCenter();

        // compute normal of the cutting plane
        tmpDir1 = this->reducedSurface[idxRS]->GetRSEdge(i)->GetFace1()->GetProbeCenter();
        tmpDir2 = this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex2()->GetPosition() - tmpDir1;
        tmpDir2.Normalise();
        tmpDir2 *= this->probeRadius;
        tmpDir2 = tmpDir2 + tmpDir1 - this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusCenter();
        tmpDir3 = this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex1()->GetPosition() - tmpDir1;
        tmpDir3.Normalise();
        tmpDir3 *= this->probeRadius;
        tmpDir3 = tmpDir3 + tmpDir1 - this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusCenter();
        // tmpDir2 and tmpDir3 now store the position of the intersection points for face 1
        cutPlaneNorm = tmpDir2 - tmpDir3;
        // cutPlaneNorm now stores the vector between the two intersection points for face 1
        tmpDir1 = this->reducedSurface[idxRS]->GetRSEdge(i)->GetFace2()->GetProbeCenter();
        tmpDir2 = this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex1()->GetPosition() - tmpDir1;
        tmpDir2.Normalise();
        tmpDir2 *= this->probeRadius;
        tmpDir2 = tmpDir2 + tmpDir1 - this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusCenter();
        // tmpDir2 now stores the position of the intersection point 1 for face 2
        tmpDir2 = tmpDir2 - tmpDir3;
        // tmpDir2 and tmpDir3 now span the plane containing the four intersection points
        cutPlaneNorm = cutPlaneNorm.Cross(tmpDir2);
        cutPlaneNorm = torusAxis.Cross(cutPlaneNorm);
        cutPlaneNorm.Normalise();

        // attributes
        this->torusInParamArray[idxRS][i * 3 + 0] = this->probeRadius;
        this->torusInParamArray[idxRS][i * 3 + 1] = this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusRadius();
        this->torusInParamArray[idxRS][i * 3 + 2] = this->reducedSurface[idxRS]->GetRSEdge(i)->GetRotationAngle();
        this->torusQuatCArray[idxRS][i * 4 + 0] = quatC.GetX();
        this->torusQuatCArray[idxRS][i * 4 + 1] = quatC.GetY();
        this->torusQuatCArray[idxRS][i * 4 + 2] = quatC.GetZ();
        this->torusQuatCArray[idxRS][i * 4 + 3] = quatC.GetW();
        this->torusInSphereArray[idxRS][i * 4 + 0] = C.GetX();
        this->torusInSphereArray[idxRS][i * 4 + 1] = C.GetY();
        this->torusInSphereArray[idxRS][i * 4 + 2] = C.GetZ();
        this->torusInSphereArray[idxRS][i * 4 + 3] = distance;
        // colors
        this->torusColors[idxRS][i * 4 + 0] =
            CodeColor(&this->atomColorTable[this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex1()->GetIndex()].x);
        this->torusColors[idxRS][i * 4 + 1] =
            CodeColor(&this->atomColorTable[this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex2()->GetIndex()].x);
        this->torusColors[idxRS][i * 4 + 2] = d;
        this->torusColors[idxRS][i * 4 + 3] =
            (X2 + this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex2()->GetPosition() -
                this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusCenter())
                .Dot(torusAxis) -
            d;
        // cutting plane
        this->torusInCuttingPlaneArray[idxRS][i * 3 + 0] = cutPlaneNorm.GetX();
        this->torusInCuttingPlaneArray[idxRS][i * 3 + 1] = cutPlaneNorm.GetY();
        this->torusInCuttingPlaneArray[idxRS][i * 3 + 2] = cutPlaneNorm.GetZ();
        // torus center
        this->torusVertexArray[idxRS][i * 3 + 0] = this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusCenter().GetX();
        this->torusVertexArray[idxRS][i * 3 + 1] = this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusCenter().GetY();
        this->torusVertexArray[idxRS][i * 3 + 2] = this->reducedSurface[idxRS]->GetRSEdge(i)->GetTorusCenter().GetZ();
    }

    ///////////////////////////////////////////////////////////
    // compute arrays for ray casting the spheres on the GPU //
    ///////////////////////////////////////////////////////////
    this->sphereVertexArray[idxRS].AssertCapacity(this->reducedSurface[idxRS]->GetRSVertexCount() * 4);
    this->sphereVertexArray[idxRS].Clear();
    this->sphereColors[idxRS].AssertCapacity(this->reducedSurface[idxRS]->GetRSVertexCount() * 3);
    this->sphereColors[idxRS].Clear();

    // loop over all RS-vertices (i.e. all protein atoms)
    for (i = 0; i < this->reducedSurface[idxRS]->GetRSVertexCount(); ++i) {
        // add only surface atoms (i.e. with not buried RS-vertices)
        if (this->reducedSurface[idxRS]->GetRSVertex(i)->IsBuried())
            continue;
        // set vertex color
        this->sphereColors[idxRS].Append(
            this->atomColorTable[this->reducedSurface[idxRS]->GetRSVertex(i)->GetIndex()].x);
        this->sphereColors[idxRS].Append(
            this->atomColorTable[this->reducedSurface[idxRS]->GetRSVertex(i)->GetIndex()].y);
        this->sphereColors[idxRS].Append(
            this->atomColorTable[this->reducedSurface[idxRS]->GetRSVertex(i)->GetIndex()].z);
        // set vertex position
        this->sphereVertexArray[idxRS].Append(this->reducedSurface[idxRS]->GetRSVertex(i)->GetPosition().GetX());
        this->sphereVertexArray[idxRS].Append(this->reducedSurface[idxRS]->GetRSVertex(i)->GetPosition().GetY());
        this->sphereVertexArray[idxRS].Append(this->reducedSurface[idxRS]->GetRSVertex(i)->GetPosition().GetZ());
        if (this->drawSAS) {
            this->sphereVertexArray[idxRS].Append(
                this->reducedSurface[idxRS]->GetRSVertex(i)->GetRadius() + this->probeRadius);
        } else {
            this->sphereVertexArray[idxRS].Append(this->reducedSurface[idxRS]->GetRSVertex(i)->GetRadius());
        }
    }
}


/*
 * code a rgb-color into one float
 */
float MoleculeSESRenderer::CodeColor(const float* col) const {
    return float((int)(col[0] * 255.0f) * 1000000 // red
                 + (int)(col[1] * 255.0f) * 1000  // green
                 + (int)(col[2] * 255.0f));       // blue
}


/*
 * decode a coded color to the original rgb-color
 */
vislib::math::Vector<float, 3> MoleculeSESRenderer::DecodeColor(int codedColor) const {
    int col = codedColor;
    vislib::math::Vector<float, 3> color;
    float red, green;
    if (col >= 1000000)
        red = floor((float)col / 1000000.0f);
    else
        red = 0.0;
    col = col - int(red * 1000000.0f);
    if (col > 1000)
        green = floor((float)col / 1000.0f);
    else
        green = 0.0;
    col = col - int(green * 1000.0f);
    // color.Set( red / 255.0f, green / 255.0f, float(col) / 255.0f);
    color.Set(std::min(1.0f, std::max(0.0f, red / 255.0f)), std::min(1.0f, std::max(0.0f, green / 255.0f)),
        std::min(1.0f, std::max(0.0f, col / 255.0f)));
    return color;
}


/*
 * Creates the texture for singularity handling.
 */
void MoleculeSESRenderer::CreateSingularityTextures() {
    // time_t t = clock();
    unsigned int cnt1, cnt2, cntRS;

    // delete old singularity textures
    for (cnt1 = 0; cnt1 < this->singularityTexture.size(); ++cnt1) {
        glDeleteTextures(1, &singularityTexture[cnt1]);
    }
    // check if the singularity texture has the right size
    if (this->reducedSurface.size() != this->singularityTexture.size()) {
        // store old singularity texture size
        unsigned int singTexSizeOld = (unsigned int)this->singularityTexture.size();
        // resize singularity texture to fit the number of reduced surfaces
        this->singularityTexture.resize(this->reducedSurface.size());
        // generate a new texture for each new singularity texture
        for (cnt1 = singTexSizeOld; cnt1 < singularityTexture.size(); ++cnt1) {
            glGenTextures(1, &singularityTexture[cnt1]);
        }
    }
    // resize singularity texture dimension arrays
    this->singTexWidth.resize(this->reducedSurface.size());
    this->singTexHeight.resize(this->reducedSurface.size());

    // get maximum texture size
    GLint texSize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &texSize);

    // TODO: compute proper maximum number of cutting probes
    unsigned int numProbes = 16;

    for (cntRS = 0; cntRS < this->reducedSurface.size(); ++cntRS) {
        // set width and height of texture
        if ((unsigned int)texSize < this->reducedSurface[cntRS]->GetCutRSEdgesCount()) {
            this->singTexHeight[cntRS] = texSize;
            this->singTexWidth[cntRS] =
                numProbes * (int)ceil(double(this->reducedSurface[cntRS]->GetCutRSEdgesCount()) / (double)texSize);
        } else {
            this->singTexHeight[cntRS] = this->reducedSurface[cntRS]->GetCutRSEdgesCount();
            this->singTexWidth[cntRS] = numProbes;
        }
        // generate float-array for texture with the appropriate dimension
        if (this->singTexData)
            delete[] this->singTexData;
        this->singTexData = new float[this->singTexWidth[cntRS] * this->singTexHeight[cntRS] * 3];
        // write probes to singularity texture
        unsigned int coordX = 0;
        unsigned int coordY = 0;
        unsigned int counter = 0;
        for (cnt1 = 0; cnt1 < this->reducedSurface[cntRS]->GetRSEdgeCount(); ++cnt1) {
            if (this->reducedSurface[cntRS]->GetRSEdge(cnt1)->cuttingProbes.empty()) {
                this->reducedSurface[cntRS]->GetRSEdge(cnt1)->SetTexCoord(0, 0);
            } else {
                // set texture coordinates
                this->reducedSurface[cntRS]->GetRSEdge(cnt1)->SetTexCoord(coordX, coordY);
                // compute texture coordinates for next entry
                coordY++;
                if (coordY == this->singTexHeight[cntRS]) {
                    coordY = 0;
                    coordX = coordX + numProbes;
                }
                // write probes to texture
                for (cnt2 = 0; cnt2 < numProbes; ++cnt2) {
                    if (cnt2 < this->reducedSurface[cntRS]->GetRSEdge(cnt1)->cuttingProbes.size()) {
                        singTexData[counter] =
                            this->reducedSurface[cntRS]->GetRSEdge(cnt1)->cuttingProbes[cnt2]->GetProbeCenter().GetX();
                        counter++;
                        singTexData[counter] =
                            this->reducedSurface[cntRS]->GetRSEdge(cnt1)->cuttingProbes[cnt2]->GetProbeCenter().GetY();
                        counter++;
                        singTexData[counter] =
                            this->reducedSurface[cntRS]->GetRSEdge(cnt1)->cuttingProbes[cnt2]->GetProbeCenter().GetZ();
                        counter++;
                    } else {
                        singTexData[counter] = 0.0f;
                        counter++;
                        singTexData[counter] = 0.0f;
                        counter++;
                        singTexData[counter] = 0.0f;
                        counter++;
                    }
                }
            }
        }
        // texture generation
        glBindTexture(GL_TEXTURE_2D, singularityTexture[cntRS]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, this->singTexWidth[cntRS], this->singTexHeight[cntRS], 0, GL_RGB,
            GL_FLOAT, this->singTexData);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    // std::cout << "Create texture: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
}


/*
 * Creates the texture for singularity handling.
 */
void MoleculeSESRenderer::CreateSingularityTexture(unsigned int idxRS) {
    // do nothing if the index is out of bounds
    if (idxRS > this->reducedSurface.size())
        return;

    // check if all arrays have the appropriate size
    if (this->singularityTexture.size() != this->reducedSurface.size() ||
        this->singTexWidth.size() != this->reducedSurface.size() ||
        this->singTexHeight.size() != this->reducedSurface.size()) {
        // create all singularity textures
        CreateSingularityTextures();
        return;
    }

    unsigned int cnt1, cnt2;

    // delete old singularity texture
    glDeleteTextures(1, &singularityTexture[idxRS]);

    // get maximum texture size
    GLint texSize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &texSize);

    // TODO: compute proper maximum number of cutting probes
    unsigned int numProbes = 16;

    // set width and height of texture
    if ((unsigned int)texSize < this->reducedSurface[idxRS]->GetCutRSEdgesCount()) {
        this->singTexHeight[idxRS] = texSize;
        this->singTexWidth[idxRS] =
            numProbes * (int)ceil(double(this->reducedSurface[idxRS]->GetCutRSEdgesCount()) / (double)texSize);
    } else {
        this->singTexHeight[idxRS] = this->reducedSurface[idxRS]->GetCutRSEdgesCount();
        this->singTexWidth[idxRS] = numProbes;
    }
    // generate float-array for texture with the appropriate dimension
    if (this->singTexData)
        delete[] this->singTexData;
    this->singTexData = new float[this->singTexWidth[idxRS] * this->singTexHeight[idxRS] * 3];
    // write probes to singularity texture
    unsigned int coordX = 0;
    unsigned int coordY = 0;
    unsigned int counter = 0;
    for (cnt1 = 0; cnt1 < this->reducedSurface[idxRS]->GetRSEdgeCount(); ++cnt1) {
        if (this->reducedSurface[idxRS]->GetRSEdge(cnt1)->cuttingProbes.empty()) {
            this->reducedSurface[idxRS]->GetRSEdge(cnt1)->SetTexCoord(0, 0);
        } else {
            // set texture coordinates
            this->reducedSurface[idxRS]->GetRSEdge(cnt1)->SetTexCoord(coordX, coordY);
            // compute texture coordinates for next entry
            coordY++;
            if (coordY == this->singTexHeight[idxRS]) {
                coordY = 0;
                coordX = coordX + numProbes;
            }
            // write probes to texture
            for (cnt2 = 0; cnt2 < numProbes; ++cnt2) {
                if (cnt2 < this->reducedSurface[idxRS]->GetRSEdge(cnt1)->cuttingProbes.size()) {
                    singTexData[counter] =
                        this->reducedSurface[idxRS]->GetRSEdge(cnt1)->cuttingProbes[cnt2]->GetProbeCenter().GetX();
                    counter++;
                    singTexData[counter] =
                        this->reducedSurface[idxRS]->GetRSEdge(cnt1)->cuttingProbes[cnt2]->GetProbeCenter().GetY();
                    counter++;
                    singTexData[counter] =
                        this->reducedSurface[idxRS]->GetRSEdge(cnt1)->cuttingProbes[cnt2]->GetProbeCenter().GetZ();
                    counter++;
                } else {
                    singTexData[counter] = 0.0f;
                    counter++;
                    singTexData[counter] = 0.0f;
                    counter++;
                    singTexData[counter] = 0.0f;
                    counter++;
                }
            }
        }
    }
    // texture generation
    glBindTexture(GL_TEXTURE_2D, singularityTexture[idxRS]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, this->singTexWidth[idxRS], this->singTexHeight[idxRS], 0, GL_RGB,
        GL_FLOAT, this->singTexData);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

/*
 * MoleculeSESRenderer::deinitialise
 */
void MoleculeSESRenderer::deinitialise() {
    // delete singularity texture
    for (unsigned int i = 0; i < singularityTexture.size(); ++i)
        glDeleteTextures(1, &singularityTexture[i]);
}
