/*
 * MoleculeSESMeshRenderer.cpp
 *
 * Copyright (C) 2009-2021 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#define _USE_MATH_DEFINES 1

#include "MoleculeSESMeshRenderer.h"
#include "glm/gtx/string_cast.hpp"
#include "mmcore/CoreInstance.h"
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
#include <ctime>
#include <fstream>
#include <iostream>
#include <cmath>


#include "protein/Icosphere.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::geocalls_gl;
using namespace megamol::protein;
using namespace megamol::protein_calls;
using namespace megamol::protein_gl;
using namespace megamol::core::utility::log;

/*
 * MoleculeSESMeshRenderer::MoleculeSESMeshRenderer
 */
MoleculeSESMeshRenderer::MoleculeSESMeshRenderer()
        : Renderer3DModuleGL()
        , molDataCallerSlot("getData", "Connects the protein SES rendering with protein data storage")
        , getLightsSlot("getLights", "Connects the protein SES rendering with light sources")
        , bsDataCallerSlot("getBindingSites", "Connects the molecule rendering with binding site data storage")
        , getTriangleDataSlot("gettriadata", "The slot publishing the generated triangle data")
        , coloringModeParam0("color::coloringMode0", "The first coloring mode.")
        , coloringModeParam1("color::coloringMode1", "The second coloring mode.")
        , cmWeightParam("color::colorWeighting", "The weighting of the two coloring modes.")
        , minGradColorParam("color::minGradColor", "The color for the minimum value for gradient coloring")
        , midGradColorParam("color::midGradColor", "The color for the middle value for gradient coloring")
        , maxGradColorParam("color::maxGradColor", "The color for the maximum value for gradient coloring")
        , colorTableFileParam("color::colorTableFilename", "The filename of the color table.")
        , probeRadiusSlot("probeRadius", "The probe radius for the surface computation")
        , datahash(0)
        , computeSesPerMolecule(false)
        , sphereColorBuffer_(nullptr)
        , sphereVertexBuffer_(nullptr)
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

    // the triangle data out slot
    this->getTriangleDataSlot.SetCallback(
        CallTriMeshDataGL::ClassName(), "GetData", &MoleculeSESMeshRenderer::getTriangleDataCallback);
    this->getTriangleDataSlot.SetCallback(
        CallTriMeshDataGL::ClassName(), "GetExtent", &MoleculeSESMeshRenderer::getExtentCallback);
    this->MakeSlotAvailable(&this->getTriangleDataSlot);

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

    this->preComputationDone = false;

    auto defparams = deferredProvider_.getUsedParamSlots();
    for (const auto& param : defparams) {
        this->MakeSlotAvailable(param);
    }
}


/*
 * MoleculeSESMeshRenderer::~MoleculeSESMeshRenderer
 */
MoleculeSESMeshRenderer::~MoleculeSESMeshRenderer(void) {
    this->Release();
}


/*
 * protein::MoleculeSESMeshRenderer::release
 */
void MoleculeSESMeshRenderer::release(void) {}


/*
 * MoleculeSESMeshRenderer::create
 */
bool MoleculeSESMeshRenderer::create(void) {

    // glEnable( GL_NORMALIZE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);

    CoreInstance* ci = this->GetCoreInstance();
    if (!ci)
        return false;

    try {
        auto const shdr_options = msf::ShaderFactoryOptionsOpenGL(ci->GetShaderPaths());

        sphereShader_ = core::utility::make_shared_glowl_shader("sphere", shdr_options,
            std::filesystem::path("protein_gl/moleculeses/mses_sphere.vert.glsl"),
            std::filesystem::path("protein_gl/moleculeses/mses_sphere.frag.glsl"));

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[MoleculeSESMeshRenderer] %s", ex.what());
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[MoleculeSESMeshRenderer] Unable to compile shader: Unknown exception: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[MoleculeSESMeshRenderer] Unable to compile shader: Unknown exception.");
    }

    // create the buffer objects
    sphereVertexBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    sphereColorBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

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

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    for (int i = 0; i < 6; ++i) {
        glDisableVertexAttribArray(i);
    }

    deferredProvider_.setup(this->GetCoreInstance());

    return true;
}

/*
 * MoleculeSESMeshRenderer::GetExtents
 */
bool MoleculeSESMeshRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {

    MolecularDataCall* mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == nullptr)
        return false;
    if (!(*mol)(1))
        return false;

    call.AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    call.SetTimeFramesCount(mol->FrameCount());

    return true;
}

/*
 * MoleculeSESMeshRenderer::Render
 */
bool MoleculeSESMeshRenderer::Render(mmstd_gl::CallRender3DGL& call) {
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

    // framebuffer object for rendering / deferred shading
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
        unsigned int molIds;
        if (!this->computeSesPerMolecule) {
            this->reducedSurface.push_back(new ReducedSurface(mol, this->probeRadius));
            this->reducedSurface.back()->ComputeReducedSurface();
        } else {
            for (molIds = 0; molIds < mol->MoleculeCount(); ++molIds) {
                this->reducedSurface.push_back(new ReducedSurface(molIds, mol, this->probeRadius));
                this->reducedSurface.back()->ComputeReducedSurface();
            }
        }
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "%s: RS computed in: %f s\n", this->ClassName(), (double(clock() - t) / double(CLOCKS_PER_SEC)));
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

    this->RenderAtoms(mol);
    if (!this->reducedSurface.empty())
        this->RenderReducedSurface(this->reducedSurface[0]);

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
void MoleculeSESMeshRenderer::UpdateParameters(const MolecularDataCall* mol, const BindingSiteCall* bs) {
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
    // color table param
    if (this->colorTableFileParam.IsDirty()) {
        ProteinColor::ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::FilePathParam>()->Value(), this->fileLookupTable);
        this->colorTableFileParam.ResetDirty();
        recomputeColors = true;
    }
    // check probe radius parameter slot
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
void MoleculeSESMeshRenderer::RenderAtoms(const MolecularDataCall* mol) {

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

    /////////////////////////////////////
    // ray cast the spheres on the GPU //
    /////////////////////////////////////
    std::vector<float> tmpAtoms;
    tmpAtoms.resize(mol->AtomCount() * 4);
    for (auto i = 0; i < mol->AtomCount(); i++) {
        tmpAtoms[i * 4 + 0] = mol->AtomPositions()[i * 3 + 0];
        tmpAtoms[i * 4 + 1] = mol->AtomPositions()[i * 3 + 1];
        tmpAtoms[i * 4 + 2] = mol->AtomPositions()[i * 3 + 2];
        //tmpAtoms[i * 4 + 3] = mol->AtomTypes()[mol->AtomTypeIndices()[i]].Radius();
        tmpAtoms[i * 4 + 3] = 0.2f;
    }
    this->sphereVertexBuffer_->rebuffer(tmpAtoms.data(), tmpAtoms.size() * sizeof(float));
    this->sphereColorBuffer_->rebuffer(this->atomColorTable.data(), this->atomColorTable.size() * 3 * sizeof(float));
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

    glDrawArrays(GL_POINTS, 0, (unsigned int)mol->AtomCount());

    // disable sphere shader
    glUseProgram(0);
    glBindVertexArray(0);

}

/*
 * Render the reduced surface for debugging/testing
 */
void MoleculeSESMeshRenderer::RenderReducedSurface(protein::ReducedSurface* rs) {
    glBegin(GL_TRIANGLES);
    for (auto faceIdx = 0; faceIdx < rs->GetRSFaceCount(); faceIdx++) {
        glVertex3fv(rs->GetRSFace(faceIdx)->GetVertex1()->GetPosition().PeekComponents());
        glVertex3fv(rs->GetRSFace(faceIdx)->GetVertex2()->GetPosition().PeekComponents());
        glVertex3fv(rs->GetRSFace(faceIdx)->GetVertex3()->GetPosition().PeekComponents());
    }
    glEnd();
}


/*
 * MoleculeSESMeshRenderer::deinitialise
 */
void MoleculeSESMeshRenderer::deinitialise() {

}

/*
 * MoleculeSESMeshRenderer::getDataCallback
 */
bool MoleculeSESMeshRenderer::getTriangleDataCallback(core::Call& caller) {
    auto* ctmd = dynamic_cast<CallTriMeshDataGL*>(&caller);
    if (ctmd == nullptr)
        return false;

    auto* mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == nullptr)
        return false;
    if (!(*mol)(1))
        return false;

    ctmd->SetFrameCount(1);
    ctmd->AccessBoundingBoxes() = mol->AccessBoundingBoxes();

    // if no mesh is available, generate a new mesh object
    if (this->triaMesh.empty()) {
        this->triaMesh.push_back(new CallTriMeshDataGL::Mesh());
    }

    // fill the triangle mesh with data
    if (this->reducedSurface.empty())
        return false;
    int subdivision_level = 2;
    int vertex_counter = 5 + (subdivision_level + 4) * 22*55;
    auto* vertex = new float[vertex_counter * 3]{};
    auto* normal = new float[vertex_counter * 3]{};
    auto* color = new float[vertex_counter * 3]{};


    /*
     *
     *
     *
     *                                A
     *
     *                                D
     *                         B              C
     *
     *                         Notmale: AB X AC
     *                         AB = punkt A - B
     *                         3   7   4    b-a = von b nach a
     *                         2   2   0    a-b = von a nach b
     *                       C=  (A-B)+A  und  D= (A-C)+A ->
     *                                                  C2*D3 - C3*D2
     *                                                  C3*D1 - C1*D3
     *                                                  C1*D2 - C2*D1
     *
     *                                                  C1: (A1-B1)+A1
     *                                                  C2: (A2-B2)+A2
     *                                                  C3: (A3-B3)+A3
     *                                                  D1: (A1-C1)+A1
     *                                                  D2: (A2-C2)+A2
     *                                                  D3  (A3-C3)+A3
     *
     *
<                                     (((A2-B2)+A2)*((A3-C3)+A3) - (((A3-B3)+A3)*((A2-C2)+A2)))
>                                     (((A3-B3)+A3)*((A1-C1)+A1) - (((A1-B1)+A1)*((A3-C3)+A3)))
                                      (((A1-B1)+A1)*((A2-C2)+A2) - (((A2-B2)+A2)*((A1-C1)+A1)))
     *                         ......
     *                          sqrt((((A2-B2)+A2)*((A3-C3)+A3) - (((A3-B3)+A3)*((A2-C2)+A2)))*(((A2-B2)+A2)*((A3-C3)+A3)
     *                          - (((A3-B3)+A3)*((A2-C2)+A2)))  (((A3-B3)+A3)*((A1-C1)+A1) - (((A1-B1)+A1)*((A3-C3)+A3)))*(((A3-B3)+A3)*((A1-C1)+A1) -
     *                          (((A1-B1)+A1)*((A3-C3)+A3)))  (((A1-B1)+A1)*((A2-C2)+A2) - (((A2-B2)+A2)*((A1-C1)+A1)))*(((A1-B1)+A1)*((A2-C2)+A2) - (((A2-B2)+A2)*((A1-C1)+A1))))
     *
     *
     *
     *
     *
     *
     *
     *
     */


    vertex[0 + 0] = -0.944;   //Punkt 1     A
    float A1 = vertex[0];
    vertex[0 + 1] = 20.730;   //Punkt 1
    float A2 = vertex[1];
    vertex[0 + 2] = -12.605;  //Punkt 1
    float A3 = vertex[2];
    vertex[3 + 0] = -1.143;   //Punkt 2     B
    float B1 = vertex[3];
    vertex[3 + 1] = 19.976;   //Punkt 2
    float B2 = vertex[4];
    vertex[3 + 2] = -11.390;  //Punkt 2
    float B3 = vertex[5];
    vertex[6 + 0] = -0.115;   //Punkt 3     C
    float C1 = vertex[6];
    vertex[6 + 1] = 18.874;   //Punkt 3
    float C2 = vertex[7];
    vertex[6 + 2] = -11.198;  //Punkt 3
    float C3 = vertex[8];
    vertex[9 + 0] = -0.734;   //Punkt 4    Mittelpunkt
    float D1 = vertex[9];
    vertex[9 + 1] = 19.86;    //Punkt 4
    float D2 = vertex[10];
    vertex[9 + 2] = -11.731;  //Punkt 4
    float D3 = vertex[11];


    float normierung = std::sqrt((((A2-B2)+A2)*((A3-C3)+A3) -
                                     (((A3-B3)+A3)*((A2-C2)+A2)))*(((A2-B2)+A2)*((A3-C3)+A3) -
                                         (((A3-B3)+A3)*((A2-C2)+A2))) + (((A3-B3)+A3)*((A1-C1)+A1) -
                                     (((A1-B1)+A1)*((A3-C3)+A3)))*(((A3-B3)+A3)*((A1-C1)+A1) -
                                         (((A1-B1)+A1)*((A3-C3)+A3))) +  (((A1-B1)+A1)*((A2-C2)+A2) -
                                     (((A2-B2)+A2)*((A1-C1)+A1)))*(((A1-B1)+A1)*((A2-C2)+A2) - (((A2-B2)+A2)*((A1-C1)+A1))));
    vertex[12 + 0] = (((A2-B2)+A2)*((A3-C3)+A3) - (((A3-B3)+A3)*((A2-C2)+A2)))/(normierung/2)+vertex[9];
    vertex[12 + 1] = (((A3-B3)+A3)*((A1-C1)+A1) - (((A1-B1)+A1)*((A3-C3)+A3)))/(normierung/2)+vertex[10];
    vertex[12 + 2] = (((A1-B1)+A1)*((A2-C2)+A2) - (((A2-B2)+A2)*((A1-C1)+A1)))/(normierung/2)+vertex[11];


    normal[0 + 0] = -0.944;   //Punkt 1
    normal[0 + 1] = 20.730;   //Punkt 1
    normal[0 + 2] = -12.605;  //Punkt 1
    normal[3 + 0] = -1.143;   //Punkt 2
    normal[3 + 1] = 19.976;   //Punkt 2
    normal[3 + 2] = -11.390;  //Punkt 2
    normal[6 + 0] = -0.115;   //Punkt 3
    normal[6 + 1] = 18.874;   //Punkt 3
    normal[6 + 2] = -11.198;  //Punkt 3
    normal[9 + 0] = -1.043;   //Punkt 4
    normal[9 + 1] = 20.353;   //Punkt 4
    normal[9 + 2] = -11.9975; //Punkt 4
    normal[12 + 0] = float(1.0f/10* (1 / 3.2653051176227024) * -2.7703830000000016 + normal[9 + 0]); //Punkt 5
    normal[12 + 1] = float(1.0f/10* (1/3.2653051176227024) * 1.399128  + normal[9 + 1]);   //Punkt 5
    normal[12 + 2] = float(1.0f/10*  (1/3.2653051176227024) * 1.0147100000000016 + normal[9 + 2]); //Punkt 5

    // 0.14900000000000002
    // 0.7540000000000013
    // -1.2149999999999999
    //Von A -> B


    // -0.979
    // 1.8560000000000016
    // -1.407
    //Von A -> C


    color[0 + 0] = 1.0f;    //Punkt 1
    color[0 + 1] = 0.0f;    //Punkt 1
    color[0 + 2] = 1.0f;    //Punkt 1
    color[3 + 0] = 0.3f;    //Punkt 2
    color[3 + 1] = 1.0f;    //Punkt 2
    color[3 + 2] = 0.6f;    //Punkt 2
    color[6 + 0] = 0.07f;    //Punkt 3
    color[6 + 1] = 1.0f;    //Punkt 3
    color[6 + 2] = 0.79f;    //Punkt 3
    color[9 + 0] = 0.0f;    //Punkt 4
    color[9 + 1] = 0.7544f; //Punkt 4
    color[9 + 2] = 0.10545f;    //Punkt 4
    color[12 + 0] = 1.0f;    //Punkt 5
    color[12 + 1] = 1.0; //Punkt 5
    color[12 + 2] = 0.0f;    //Punkt 5

    auto ico = new Icosphere(0.15, subdivision_level, true);
    for (int i = 0; i < ico->getVertexCount() *3; ++i) {
        if ((i % 3) == 0) {
            vertex[15 + i] = ico->getVertices()[i] + A1;
            normal[15 + i] = ico->getNormals()[i];
        }
        if ((i % 3) == 1) {
            vertex[15 + i] = ico->getVertices()[i] + A2;
            normal[15 + i] = ico->getNormals()[i];
        }
        if ((i % 3) == 2) {
            vertex[15 + i] = ico->getVertices()[i] + A3;
            normal[15 + i] = ico->getNormals()[i];
        }
        color[15 + i] = 0.5f;
    }


    /*for (auto i = 0; i < mol->AtomCount() + 1; i++) {
        vertex[3 * i + 0] = mol->AtomPositions()[3 * i];
        vertex[3 * i + 1] = mol->AtomPositions()[3 * i + 1];
        vertex[3 * i + 2] = mol->AtomPositions()[3 * i + 2];
        normal[3 * i + 0] = mol->AtomPositions()[3 * i];
        normal[3 * i + 1] = mol->AtomPositions()[3 * i + 1];
        normal[3 * i + 2] = mol->AtomPositions()[3 * i + 2];
        color[3 * i + 0] = 1.0f;
        color[3 * i + 1] = 0.5f;
        color[3 * i + 2] = 0.0f;
    }*/
    //int face_counter = (subdivision_level + 1) * 22;
    int face_counter = 1088;
    auto* face = new unsigned int[ico->getIndexCount() * 3]{};
    // Es hat 2 faces
    // Vorder und Rückseite jeweils


    /*
     *
     Punkte aus orginalem pdb
     Nur diese Atome ausgeführt: Bild 1
 ATOM      1  N   GLY A   1      -0.944  20.730 -12.605  1.00 19.62           N
 ATOM      2  CA  GLY A   1      -1.143  19.976 -11.390  1.00 18.58           C
 ATOM      3  C   GLY A   1      -0.115  18.874 -11.198  1.00 16.99           C

     Selbst definierter Punkt:
     Mit dem Punkt hier: Bild 2
 ATOM      4  C   GLY A   1      -0.734  19.5853 -11.731  1.00 16.99           C
     *
     */
    /*face[0] = 0;
    face[1] = 1;
    face[2] = 5 + 10; //5 nicht 15, da hier die vertices und nicht vertices indexes gelesen werden

    face[3] = 0;
    face[4] = 1;
    face[5] = 5 + 14;

    face[6] = 0;
    face[7] = 1;
    face[8] = 5 + 0; */



    for (int i = 0; i <  ico->getIndexCount(); ++i) {
        face[i] = ico->getIndices()[i] + 5;
    }


/*
    for (auto i = 0; i < this->reducedSurface[0]->GetRSFaceCount(); i++) {
        face[3 * i + 0] = this->reducedSurface[0]->GetRSFace(i)->GetVertex1()->GetIndex();
        face[3 * i + 1] = this->reducedSurface[0]->GetRSFace(i)->GetVertex2()->GetIndex();
        face[3 * i + 2] = this->reducedSurface[0]->GetRSFace(i)->GetVertex3()->GetIndex();
        // check/correct winding order of triangle vertices
        vislib::math::Vector<float, 3> tmpVec1(this->reducedSurface[0]->GetRSFace(i)->GetVertex1()->GetPosition());
        vislib::math::Vector<float, 3> tmpVec2(this->reducedSurface[0]->GetRSFace(i)->GetVertex2()->GetPosition());
        vislib::math::Vector<float, 3> tmpVec3(this->reducedSurface[0]->GetRSFace(i)->GetVertex3()->GetPosition());
        vislib::math::Vector<float, 3> tmpVecProbe(this->reducedSurface[0]->GetRSFace(i)->GetProbeCenter());
        vislib::math::Vector<float, 3> tmpVecN = (tmpVec2 - tmpVec1).Cross(tmpVec3 - tmpVec1);
        tmpVecN.Normalise();
        if (tmpVecN.Dot(tmpVecProbe - tmpVec1) < 0.0f) {
            std::swap(face[3 * i + 0], face[3 * i + 1]);
        }
    }
    this->triaMesh[0]->SetVertexData(mol->AtomCount(), vertex, normal, color, NULL, true);
    this->triaMesh[0]->SetTriangleData(this->reducedSurface[0]->GetRSFaceCount(), face, true); */
    this->triaMesh[0]->SetVertexData( vertex_counter * 3, vertex, normal, color, nullptr, true);
    this->triaMesh[0]->SetTriangleData(face_counter, face, true);
    this->triaMesh[0]->SetMaterial(nullptr);

    // set triangle mesh to caller
    if (this->triaMesh[0]->GetVertexCount() > 0) {
        ctmd->SetDataHash(this->datahash);
        ctmd->SetObjects(1, this->triaMesh[0]);
        ctmd->SetUnlocker(nullptr);
        return true;
    } else {
        return false;
    }
}

/*
 * MoleculeSESMeshRenderer::getExtentCallback
 */
bool MoleculeSESMeshRenderer::getExtentCallback(core::Call& caller) {
    auto* ctmd = dynamic_cast<CallTriMeshDataGL*>(&caller);
    if (ctmd == nullptr)
        return false;

    auto* mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == nullptr)
        return false;
    if (!(*mol)(1))
        return false;

    ctmd->SetFrameCount(1);
    ctmd->AccessBoundingBoxes() = mol->AccessBoundingBoxes();

    return true;
}
