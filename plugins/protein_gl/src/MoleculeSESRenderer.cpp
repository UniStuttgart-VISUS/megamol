/*
 * MoleculeSESRenderer.cpp
 *
 * Copyright (C) 2009-2021 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include <ctime>
#include <fstream>
#include <iostream>
#include <math.h>
#include "MoleculeSESRenderer.h"
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
#include "mmcore/utility/sys/ASCIIFileBuffer.h"
#include "mmcore/view/light/DistantLight.h"
#include "mmcore/view/light/PointLight.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"
#include "protein/Color.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/Trace.h"
#include "vislib/assert.h"
#include "vislib/sys/File.h"
#include "vislib_gl/graphics/gl/AbstractOpenGLShader.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"
#include "vislib_gl/graphics/gl/ShaderSource.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core_gl;
using namespace megamol::protein;
using namespace megamol::protein_calls;
using namespace megamol::protein_gl;
using namespace megamol::core::utility::log;

/*
 * MoleculeSESRenderer::MoleculeSESRenderer
 */
MoleculeSESRenderer::MoleculeSESRenderer(void)
        : Renderer3DModuleGL()
        , molDataCallerSlot("getData", "Connects the protein SES rendering with protein data storage")
        , getLightsSlot("getLights", "Connects the protein SES rendering with light sources")
        , bsDataCallerSlot("getBindingSites", "Connects the molecule rendering with binding site data storage")
        , postprocessingParam("postProcessingMode", "Enable Postprocessing Mode: ")
        , coloringModeParam0("color::coloringMode0", "The first coloring mode.")
        , coloringModeParam1("color::coloringMode1", "The second coloring mode.")
        , cmWeightParam("color::colorWeighting", "The weighting of the two coloring modes.")
        , silhouettecolorParam("silhouetteColor", "Silhouette Color: ")
        , sigmaParam("SSAOsigma", "Sigma value for SSAO: ")
        , lambdaParam("SSAOlambda", "Lambda value for SSAO: ")
        , minGradColorParam("color::minGradColor", "The color for the minimum value for gradient coloring")
        , midGradColorParam("color::midGradColor", "The color for the middle value for gradient coloring")
        , maxGradColorParam("color::maxGradColor", "The color for the maximum value for gradient coloring")
        , debugParam("drawRS", "Draw the Reduced Surface: ")
        , drawSESParam("drawSES", "Draw the SES: ")
        , drawSASParam("drawSAS", "Draw the SAS: ")
        , fogstartParam("fogStart", "Fog Start: ")
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
        , directionalLightBuffer_(nullptr) {
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
    // set transparency
    this->transparency = 0.5f;

    this->probeRadiusSlot.SetParameter(new param::FloatParam(1.4f, 0.1f));
    this->MakeSlotAvailable(&this->probeRadiusSlot);

    // ----- en-/disable postprocessing -----
    this->postprocessing = NONE;
    // this->postprocessing = AMBIENT_OCCLUSION;
    // this->postprocessing = SILHOUETTE;
    // this->postprocessing = TRANSPARENCY;
    param::EnumParam* ppm = new param::EnumParam(int(this->postprocessing));
    ppm->SetTypePair(NONE, "None");
    ppm->SetTypePair(AMBIENT_OCCLUSION, "Screen Space Ambient Occlusion");
    ppm->SetTypePair(SILHOUETTE, "Silhouette");
    // ppm->SetTypePair( TRANSPARENCY, "Transparency");
    this->postprocessingParam << ppm;

    // ----- set the default color for the silhouette -----
    this->SetSilhouetteColor(1.0f, 1.0f, 1.0f);
    param::IntParam* sc = new param::IntParam(this->codedSilhouetteColor, 0, 255255255);
    this->silhouettecolorParam << sc;

    // ----- set sigma for screen space ambient occlusion (SSAO) -----
    this->sigma = 5.0f;
    param::FloatParam* ssaos = new param::FloatParam(this->sigma);
    this->sigmaParam << ssaos;

    // ----- set lambda for screen space ambient occlusion (SSAO) -----
    this->lambda = 10.0f;
    param::FloatParam* ssaol = new param::FloatParam(this->lambda);
    this->lambdaParam << ssaol;

    // ----- set start value for fogging -----
    this->fogStart = 0.5f;
    param::FloatParam* fs = new param::FloatParam(this->fogStart, 0.0f);
    this->fogstartParam << fs;

    // coloring modes
    this->currentColoringMode0 = Color::ColoringMode::CHAIN;
    this->currentColoringMode1 = Color::ColoringMode::ELEMENT;
    param::EnumParam* cm0 = new param::EnumParam(int(this->currentColoringMode0));
    param::EnumParam* cm1 = new param::EnumParam(int(this->currentColoringMode1));
    MolecularDataCall* mol = new MolecularDataCall();
    BindingSiteCall* bs = new BindingSiteCall();
    unsigned int cCnt;
    Color::ColoringMode cMode;
    for (cCnt = 0; cCnt < Color::GetNumOfColoringModes(mol, bs); ++cCnt) {
        cMode = Color::GetModeByIndex(mol, bs, cCnt);
        cm0->SetTypePair(static_cast<int>(cMode), Color::GetName(cMode).c_str());
        cm1->SetTypePair(static_cast<int>(cMode), Color::GetName(cMode).c_str());
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

    // ----- draw RS param -----
    this->drawRS = false;
    param::BoolParam* bpm = new param::BoolParam(this->drawRS);
    this->debugParam << bpm;

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
    Color::ReadColorTableFromFile(filename, this->colorLookupTable);
    this->colorTableFileParam.SetParameter(
        new param::FilePathParam(filename, core::param::FilePathParam::FilePathFlags_::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&this->colorTableFileParam);

    // fill rainbow color table
    Color::MakeRainbowColorTable(100, this->rainbowColors);

    // set the FBOs and textures for post processing
    this->colorFBO = 0;
    this->blendFBO = 0;
    this->horizontalFilterFBO = 0;
    this->verticalFilterFBO = 0;
    this->texture0 = 0;
    this->depthTex0 = 0;
    this->hFilter = 0;
    this->vFilter = 0;
    // width and height of the screen
    this->width = 0;
    this->height = 0;

    // clear singularity texture
    singularityTexture.clear();
    // set singTexData to 0
    this->singTexData = 0;

    this->preComputationDone = false;

    // export parameters
    this->MakeSlotAvailable(&this->postprocessingParam);
    this->MakeSlotAvailable(&this->silhouettecolorParam);
    this->MakeSlotAvailable(&this->sigmaParam);
    this->MakeSlotAvailable(&this->lambdaParam);
    this->MakeSlotAvailable(&this->fogstartParam);
    this->MakeSlotAvailable(&this->debugParam);
    this->MakeSlotAvailable(&this->drawSESParam);
    this->MakeSlotAvailable(&this->drawSASParam);
}


/*
 * MoleculeSESRenderer::~MoleculeSESRenderer
 */
MoleculeSESRenderer::~MoleculeSESRenderer(void) {
    if (colorFBO) {
        glDeleteFramebuffersEXT(1, &colorFBO);
        glDeleteFramebuffersEXT(1, &blendFBO);
        glDeleteFramebuffersEXT(1, &horizontalFilterFBO);
        glDeleteFramebuffersEXT(1, &verticalFilterFBO);
        glDeleteTextures(1, &texture0);
        glDeleteTextures(1, &depthTex0);
        glDeleteTextures(1, &texture1);
        glDeleteTextures(1, &depthTex1);
        glDeleteTextures(1, &hFilter);
        glDeleteTextures(1, &vFilter);
    }
    // delete singularity texture
    for (unsigned int i = 0; i < singularityTexture.size(); ++i)
        glDeleteTextures(1, &singularityTexture[i]);
    // release
    this->cylinderShader.Release();
    this->sphereShader.Release();
    this->sphereClipInteriorShader.Release();
    this->lightShader.Release();
    this->hfilterShader.Release();
    this->vfilterShader.Release();
    this->silhouetteShader.Release();
    this->transparencyShader.Release();

    this->Release();
}


/*
 * protein::MoleculeSESRenderer::release
 */
void MoleculeSESRenderer::release(void) {}


/*
 * MoleculeSESRenderer::create
 */
bool MoleculeSESRenderer::create(void) {

    // glEnable( GL_NORMALIZE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);

    float spec[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, spec);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0f);

    using namespace vislib_gl::graphics::gl;

    ShaderSource compSrc;
    ShaderSource vertSrc;
    ShaderSource geomSrc;
    ShaderSource fragSrc;

    CoreInstance* ci = this->GetCoreInstance();
    if (!ci)
        return false;

    try {
        auto const shdr_options = msf::ShaderFactoryOptionsOpenGL(ci->GetShaderPaths());

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
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "[SimpleMoleculeRenderer] %s", ex.what());
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[SimpleMoleculeRenderer] Unable to compile shader: Unknown exception: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[SimpleMoleculeRenderer] Unable to compile shader: Unknown exception.");
    }

    ////////////////////////////////////////////////////
    // load the shader source for the sphere renderer //
    ////////////////////////////////////////////////////
    auto ssf = std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());
    if (!ssf->MakeShaderSource("protein::ses::sphereVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for sphere shader", this->ClassName());
        return false;
    }
    if (!ssf->MakeShaderSource("protein::ses::sphereFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for sphere shader", this->ClassName());
        return false;
    }

    try {
        if (!this->sphereShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "%s: Unable to create sphere shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    //////////////////////////////////////////////////////
    // load the shader files for the per pixel lighting //
    //////////////////////////////////////////////////////
    if (!ssf->MakeShaderSource("protein::std::perpixellightVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "%s: Unable to load vertex shader source for per pixel lighting shader", this->ClassName());
        return false;
    }
    if (!ssf->MakeShaderSource("protein::std::perpixellightFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "%s: Unable to load fragment shader source for per pixel lighting shader", this->ClassName());
        return false;
    }
    try {
        if (!this->lightShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "%s: Unable to create per pixel lighting shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    /////////////////////////////////////////////////////////////////
    // load the shader files for horizontal 1D gaussian filtering  //
    /////////////////////////////////////////////////////////////////
    if (!ssf->MakeShaderSource("protein::std::hfilterVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "%s: Unable to load vertex shader source for horizontal 1D gaussian filter shader", this->ClassName());
        return false;
    }
    if (!ssf->MakeShaderSource("protein::std::hfilterFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "%s: Unable to load fragment shader source for horizontal 1D gaussian filter shader", this->ClassName());
        return false;
    }
    try {
        if (!this->hfilterShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create horizontal 1D gaussian filter shader: %s\n",
            this->ClassName(), e.GetMsgA());
        return false;
    }

    ///////////////////////////////////////////////////////////////
    // load the shader files for vertical 1D gaussian filtering  //
    ///////////////////////////////////////////////////////////////
    if (!ssf->MakeShaderSource("protein::std::vfilterVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "%s: Unable to load vertex shader source for vertical 1D gaussian filter shader", this->ClassName());
        return false;
    }
    if (!ssf->MakeShaderSource("protein::std::vfilterFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "%s: Unable to load fragment shader source for vertical 1D gaussian filter shader", this->ClassName());
        return false;
    }
    try {
        if (!this->vfilterShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create vertical 1D gaussian filter shader: %s\n",
            this->ClassName(), e.GetMsgA());
        return false;
    }

    //////////////////////////////////////////////////////
    // load the shader files for silhouette drawing     //
    //////////////////////////////////////////////////////
    if (!ssf->MakeShaderSource("protein::std::silhouetteVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "%s: Unable to load vertex shader source for silhouette drawing shader", this->ClassName());
        return false;
    }
    if (!ssf->MakeShaderSource("protein::std::silhouetteFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "%s: Unable to load fragment shader source for silhouette drawing shader", this->ClassName());
        return false;
    }
    try {
        if (!this->silhouetteShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create vertical 1D gaussian filter shader: %s\n",
            this->ClassName(), e.GetMsgA());
        return false;
    }

    //////////////////////////////////////////////////////
    // load the shader source for the cylinder renderer //
    //////////////////////////////////////////////////////
    if (!ssf->MakeShaderSource("protein::std::cylinderVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "%: Unable to load vertex shader source for cylinder shader", this->ClassName());
        return false;
    }
    if (!ssf->MakeShaderSource("protein::std::cylinderFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for cylinder shader", this->ClassName());
        return false;
    }
    try {
        if (!this->cylinderShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "%s: Unable to create cylinder shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
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
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

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

    return true;
}

/*
 * MoleculeSESRenderer::GetExtents
 */
bool MoleculeSESRenderer::GetExtents(core_gl::view::CallRender3DGL& call) {

    MolecularDataCall* mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == NULL)
        return false;
    if (!(*mol)(1))
        return false;

    call.AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    call.SetTimeFramesCount(mol->FrameCount());

    return true;
}

void MoleculeSESRenderer::UpdateLights(void) {
    auto call_light = this->getLightsSlot.CallAs<core::view::light::CallLight>();
    bool lighting_available = false;
    if (call_light != nullptr) {
        if ((*call_light)(0)) {
            lighting_available = true;
        }
    }

    if (!lighting_available) {
        pointLights_.clear();
        directionalLights_.clear();
    }

    if (lighting_available && call_light->hasUpdate()) {
        auto& lights = call_light->getData();
        pointLights_.clear();
        directionalLights_.clear();

        auto point_lights = lights.get<core::view::light::PointLightType>();
        auto directional_lights = lights.get<core::view::light::DistantLightType>();

        for (auto& pl : point_lights) {
            pointLights_.push_back({pl.position[0], pl.position[1], pl.position[2], pl.intensity});
        }

        for (auto& dl : directional_lights) {
            if (dl.eye_direction) {
                auto cam_dir = glm::normalize(this->camera.getPose().direction);
                directionalLights_.push_back({cam_dir.x, cam_dir.y, cam_dir.z, dl.intensity});
            } else {
                directionalLights_.push_back({dl.direction[0], dl.direction[1], dl.direction[2], dl.intensity});
            }
        }
    }
    pointLightBuffer_->rebuffer(pointLights_);
    directionalLightBuffer_->rebuffer(directionalLights_);
}

/*
 * MoleculeSESRenderer::Render
 */
bool MoleculeSESRenderer::Render(core_gl::view::CallRender3DGL& call) {
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

    std::array<int, 2> resolution = {call.GetFramebuffer()->getWidth(), call.GetFramebuffer()->getHeight()};

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glLoadMatrixf(glm::value_ptr(proj_));

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glLoadMatrixf(glm::value_ptr(view_));

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

    fbo = call.GetFramebuffer();

    // ==================== check parameters ====================
    this->UpdateParameters(mol, bs);
    this->UpdateLights();

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
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
            "%s: RS computed in: %f s\n", this->ClassName(), (double(clock() - t) / double(CLOCKS_PER_SEC)));
    }
    // update the data / the RS
    for (cntRS = 0; cntRS < this->reducedSurface.size(); ++cntRS) {
        if (this->reducedSurface[cntRS]->UpdateData(1.0f, 5.0f)) {
            this->ComputeRaycastingArrays(cntRS);
        }
    }

    if (!this->preComputationDone) {
        // compute the color table
        Color::MakeColorTable(mol, this->currentColoringMode0, this->currentColoringMode1,
            this->cmWeightParam.Param<param::FloatParam>()->Value(),        // weight for the first cm
            1.0f - this->cmWeightParam.Param<param::FloatParam>()->Value(), // weight for the second cm
            this->atomColorTable, this->colorLookupTable, this->rainbowColors,
            this->minGradColorParam.Param<param::ColorParam>()->Value(),
            this->midGradColorParam.Param<param::ColorParam>()->Value(),
            this->maxGradColorParam.Param<param::ColorParam>()->Value(), true, bs);
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

    if (this->postprocessing != NONE && virtualViewportChanged)
        this->CreateFBO();

    // ==================== Scale & Translate ====================

    glPushMatrix();

    // ==================== Start actual rendering ====================

    glDisable(GL_BLEND);
    // glEnable( GL_NORMALIZE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);

    if (this->postprocessing == TRANSPARENCY) {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->blendFBO);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (this->drawRS)
            this->RenderDebugStuff(mol);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    } else {
        if (this->drawRS) {
            this->RenderDebugStuff(mol);
            // DEMO
            glPopMatrix();
            return true;
        }
    }

    // start rendering to frame buffer object
    if (this->postprocessing != NONE) {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->colorFBO);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    // render the SES
    this->RenderSESGpuRaycasting(mol);

    //////////////////////////////////
    // apply postprocessing effects //
    //////////////////////////////////
    if (this->postprocessing != NONE) {
        // stop rendering to frame buffer object
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

        if (this->postprocessing == AMBIENT_OCCLUSION)
            this->PostprocessingSSAO();
        else if (this->postprocessing == SILHOUETTE)
            this->PostprocessingSilhouette();
        else if (this->postprocessing == TRANSPARENCY)
            this->PostprocessingTransparency(0.5f);
    }

    glPopMatrix();

    // unlock the current frame
    mol->Unlock();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glLoadIdentity();

    return true;
}


/*
 * update parameters
 */
void MoleculeSESRenderer::UpdateParameters(const MolecularDataCall* mol, const BindingSiteCall* bs) {
    // variables
    bool recomputeColors = false;
    // ==================== check parameters ====================
    if (this->postprocessingParam.IsDirty()) {
        this->postprocessing =
            static_cast<PostprocessingMode>(this->postprocessingParam.Param<param::EnumParam>()->Value());
        this->postprocessingParam.ResetDirty();
    }
    if (this->coloringModeParam0.IsDirty() || this->coloringModeParam1.IsDirty() || this->cmWeightParam.IsDirty()) {
        this->currentColoringMode0 =
            static_cast<Color::ColoringMode>(this->coloringModeParam0.Param<param::EnumParam>()->Value());
        this->currentColoringMode1 =
            static_cast<Color::ColoringMode>(this->coloringModeParam1.Param<param::EnumParam>()->Value());

        Color::MakeColorTable(mol, this->currentColoringMode0, this->currentColoringMode1,
            this->cmWeightParam.Param<param::FloatParam>()->Value(),        // weight for the first cm
            1.0f - this->cmWeightParam.Param<param::FloatParam>()->Value(), // weight for the second cm
            this->atomColorTable, this->colorLookupTable, this->rainbowColors,
            this->minGradColorParam.Param<param::ColorParam>()->Value(),
            this->midGradColorParam.Param<param::ColorParam>()->Value(),
            this->maxGradColorParam.Param<param::ColorParam>()->Value(), true, bs);

        this->preComputationDone = false;
        this->coloringModeParam0.ResetDirty();
        this->coloringModeParam1.ResetDirty();
        this->cmWeightParam.ResetDirty();
    }
    if (this->silhouettecolorParam.IsDirty()) {
        this->SetSilhouetteColor(this->DecodeColor(this->silhouettecolorParam.Param<param::IntParam>()->Value()));
        this->silhouettecolorParam.ResetDirty();
    }
    if (this->sigmaParam.IsDirty()) {
        this->sigma = this->sigmaParam.Param<param::FloatParam>()->Value();
        this->sigmaParam.ResetDirty();
    }
    if (this->lambdaParam.IsDirty()) {
        this->lambda = this->lambdaParam.Param<param::FloatParam>()->Value();
        this->lambdaParam.ResetDirty();
    }
    if (this->fogstartParam.IsDirty()) {
        this->fogStart = this->fogstartParam.Param<param::FloatParam>()->Value();
        this->fogstartParam.ResetDirty();
    }
    if (this->debugParam.IsDirty()) {
        this->drawRS = this->debugParam.Param<param::BoolParam>()->Value();
        this->debugParam.ResetDirty();
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
        Color::ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::FilePathParam>()->Value(), this->colorLookupTable);
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
 * postprocessing: use screen space ambient occlusion
 */
void MoleculeSESRenderer::PostprocessingSSAO() {
    // START draw overlay
    glBindTexture(GL_TEXTURE_2D, this->depthTex0);
    // --> this seems to be unnecessary since no mipmap but the original resolution is used
    // glGenerateMipmapEXT( GL_TEXTURE_2D);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);

    // ----- START gaussian filtering + SSAO -----
    // apply horizontal filter
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->horizontalFilterFBO);

    this->hfilterShader.Enable();

    glUniform1fARB(this->hfilterShader.ParameterLocation("sigma"), this->sigma);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glVertex2f(0.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glVertex2f(1.0f, 1.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    this->hfilterShader.Disable();

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    // apply vertical filter to horizontally filtered image and compute colors
    glBindTexture(GL_TEXTURE_2D, this->hFilter);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->texture0);

    this->vfilterShader.Enable();

    glUniform1iARB(this->vfilterShader.ParameterLocation("tex"), 0);
    glUniform1iARB(this->vfilterShader.ParameterLocation("colorTex"), 1);
    glUniform1fARB(this->vfilterShader.ParameterLocation("sigma"), this->sigma);
    glUniform1fARB(this->vfilterShader.ParameterLocation("lambda"), this->lambda);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glVertex2f(0.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glVertex2f(1.0f, 1.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    this->vfilterShader.Disable();
    // ----- END gaussian filtering + SSAO -----

    glPopAttrib();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    // END draw overlay
}


/*
 * postprocessing: use silhouette shader
 */
void MoleculeSESRenderer::PostprocessingSilhouette() {
    // START draw overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);

    // ----- START -----
    glBindTexture(GL_TEXTURE_2D, this->depthTex0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->texture0);

    this->silhouetteShader.Enable();

    glUniform1iARB(this->silhouetteShader.ParameterLocation("tex"), 0);
    glUniform1iARB(this->silhouetteShader.ParameterLocation("colorTex"), 1);
    glUniform1fARB(this->silhouetteShader.ParameterLocation("difference"), 0.025f);
    glColor4f(this->silhouetteColor.GetX(), this->silhouetteColor.GetY(), this->silhouetteColor.GetZ(), 1.0f);
    glBegin(GL_QUADS);
    glVertex2f(0.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glVertex2f(1.0f, 1.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    this->silhouetteShader.Disable();
    // ----- END -----

    glPopAttrib();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    // END draw overlay
}


/*
 * postprocessing: transparency (blend two images)
 */
void MoleculeSESRenderer::PostprocessingTransparency(float transparency) {
    // START draw overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);

    // ----- START -----
    glBindTexture(GL_TEXTURE_2D, this->depthTex0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->texture0);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, this->depthTex1);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, this->texture1);

    this->transparencyShader.Enable();

    glUniform1iARB(this->transparencyShader.ParameterLocation("depthTex0"), 0);
    glUniform1iARB(this->transparencyShader.ParameterLocation("colorTex0"), 1);
    glUniform1iARB(this->transparencyShader.ParameterLocation("depthTex1"), 2);
    glUniform1iARB(this->transparencyShader.ParameterLocation("colorTex1"), 3);
    glUniform1fARB(this->transparencyShader.ParameterLocation("transparency"), transparency);
    glBegin(GL_QUADS);
    glVertex2f(0.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glVertex2f(1.0f, 1.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    this->transparencyShader.Disable();
    // ----- END -----

    glPopAttrib();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    // END draw overlay
}


/*
 * Create the fbo and texture needed for offscreen rendering
 */
void MoleculeSESRenderer::CreateFBO() {
    if (colorFBO) {
        glDeleteFramebuffersEXT(1, &colorFBO);
        glDeleteFramebuffersEXT(1, &blendFBO);
        glDeleteFramebuffersEXT(1, &horizontalFilterFBO);
        glDeleteFramebuffersEXT(1, &verticalFilterFBO);
        glDeleteTextures(1, &texture0);
        glDeleteTextures(1, &depthTex0);
        glDeleteTextures(1, &texture1);
        glDeleteTextures(1, &depthTex1);
        glDeleteTextures(1, &hFilter);
        glDeleteTextures(1, &vFilter);
    }
    glGenFramebuffersEXT(1, &colorFBO);
    glGenFramebuffersEXT(1, &blendFBO);
    glGenFramebuffersEXT(1, &horizontalFilterFBO);
    glGenFramebuffersEXT(1, &verticalFilterFBO);
    glGenTextures(1, &texture0);
    glGenTextures(1, &depthTex0);
    glGenTextures(1, &texture1);
    glGenTextures(1, &depthTex1);
    glGenTextures(1, &hFilter);
    glGenTextures(1, &vFilter);

    // color and depth FBO
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->colorFBO);
    // init texture0 (color)
    glBindTexture(GL_TEXTURE_2D, texture0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16_EXT, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texture0, 0);
    // init depth texture
    glBindTexture(GL_TEXTURE_2D, depthTex0);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, this->width, this->height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->depthTex0, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // color and depth FBO for blending (transparency)
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->blendFBO);
    // init texture1 (color)
    glBindTexture(GL_TEXTURE_2D, texture1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16_EXT, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texture1, 0);
    // init depth texture
    glBindTexture(GL_TEXTURE_2D, depthTex1);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, this->width, this->height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->depthTex1, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // horizontal filter FBO
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->horizontalFilterFBO);
    glBindTexture(GL_TEXTURE_2D, this->hFilter);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16_EXT, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, hFilter, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // vertical filter FBO
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->verticalFilterFBO);
    glBindTexture(GL_TEXTURE_2D, this->vFilter);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16_EXT, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, vFilter, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}


/*
 * Render the molecular surface using GPU raycasting
 */
void MoleculeSESRenderer::RenderSESGpuRaycasting(const MolecularDataCall* mol) {
    // TODO: attribute locations nicht jedes mal neu abfragen!

    bool virtualViewportChanged = false;
    if (static_cast<unsigned int>(fbo->getWidth()) != this->width ||
        static_cast<unsigned int>(fbo->getHeight()) != this->height) {
        this->width = static_cast<unsigned int>(fbo->getWidth());
        this->height = static_cast<unsigned int>(fbo->getHeight());
        virtualViewportChanged = true;
    }

    // set viewport
    glm::vec4 viewportStuff;
    viewportStuff[0] = 0.0f;
    viewportStuff[1] = 0.0f;
    viewportStuff[2] = static_cast<float>(fbo->getWidth());
    viewportStuff[3] = static_cast<float>(fbo->getHeight());
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

    // get clear color (i.e. background color) for fogging
    float* clearColor = new float[4];
    glGetFloatv(GL_COLOR_CLEAR_VALUE, clearColor);
    vislib::math::Vector<float, 3> fogCol(clearColor[0], clearColor[1], clearColor[2]);

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
            torusShader_->setUniform("zValues", fogStart, nearplane, farplane);
            torusShader_->setUniform("fogCol", fogCol.GetX(), fogCol.GetY(), fogCol.GetZ());
            torusShader_->setUniform("alpha", this->transparency);
            torusShader_->setUniform("view", view_);
            torusShader_->setUniform("proj", proj_);
            torusShader_->setUniform("viewInverse", invview_);
            torusShader_->setUniform("mvp", mvp_);
            torusShader_->setUniform("mvpinverse", mvpinverse_);
            torusShader_->setUniform("mvptransposed", mvptranspose_);

            glDrawArrays(GL_POINTS, 0, ((unsigned int) this->torusVertexArray[cntRS].Count()) / 3);

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
            sphericalTriangleShader_->setUniform("zValues", fogStart, nearplane, farplane);
            sphericalTriangleShader_->setUniform("fogCol", fogCol.GetX(), fogCol.GetY(), fogCol.GetZ());
            sphericalTriangleShader_->setUniform(
                "texOffset", 1.0f / (float) this->singTexWidth[cntRS], 1.0f / (float) this->singTexHeight[cntRS]);
            sphericalTriangleShader_->setUniform("alpha", this->transparency);
            sphericalTriangleShader_->setUniform("view", view_);
            sphericalTriangleShader_->setUniform("proj", proj_);
            sphericalTriangleShader_->setUniform("viewInverse", invview_);
            sphericalTriangleShader_->setUniform("mvp", mvp_);
            sphericalTriangleShader_->setUniform("mvpinverse", mvpinverse_);
            sphericalTriangleShader_->setUniform("mvptransposed", mvptranspose_);

            glDrawArrays(GL_POINTS, 0, ((unsigned int) this->sphericTriaVertexArray[cntRS].Count()) / 4);

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
        sphereShader_->setUniform("zValues", fogStart, nearplane, farplane);
        sphereShader_->setUniform("fogCol", fogCol.GetX(), fogCol.GetY(), fogCol.GetZ());
        sphereShader_->setUniform("alpha", this->transparency);
        sphereShader_->setUniform("view", view_);
        sphereShader_->setUniform("proj", proj_);
        sphereShader_->setUniform("viewInverse", invview_);
        sphereShader_->setUniform("mvp", mvp_);
        sphereShader_->setUniform("mvpinverse", mvpinverse_);
        sphereShader_->setUniform("mvptransposed", mvptranspose_);

        glDrawArrays(GL_POINTS, 0, ((unsigned int) this->sphereVertexArray[cntRS].Count()) / 4);

        // disable sphere shader
        glUseProgram(0);
        glBindVertexArray(0);
    }

    // delete pointers
    delete[] clearColor;
}


/*
 * Render debug stuff
 */
void MoleculeSESRenderer::RenderDebugStuff(const MolecularDataCall* mol) {
    // --> USAGE: UNCOMMENT THE NEEDED PARTS

    // temporary variables
    unsigned int max1, max2;
    max1 = max2 = 0;
    vislib::math::Vector<float, 3> v1, v2, v3, n1;
    v1.Set(0, 0, 0);
    v2 = v3 = n1 = v1;

    //////////////////////////////////////////////////////////////////////////
    // Draw reduced surface
    //////////////////////////////////////////////////////////////////////////
    this->RenderAtomsGPU(mol, 0.2f);
    vislib::math::Quaternion<float> quatC;
    quatC.Set(0, 0, 0, 1);
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Vector<float, 3> tmpVec, ortho, dir, position;
    float angle;
    // set viewport
    glm::vec4 viewportStuff;
    viewportStuff[0] = 0.0f;
    viewportStuff[1] = 0.0f;
    viewportStuff[2] = static_cast<float>(fbo->getWidth());
    viewportStuff[3] = static_cast<float>(fbo->getHeight());
    if (viewportStuff[2] < 1.0f)
        viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f)
        viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glm::vec3 camdir = camera.get<core::view::Camera::Pose>().direction;
    glm::vec3 right = camera.get<core::view::Camera::Pose>().right;
    glm::vec3 up = camera.get<core::view::Camera::Pose>().up;
    // enable cylinder shader
    this->cylinderShader.Enable();
    // set shader variables
    glUniform4fvARB(this->cylinderShader.ParameterLocation("viewAttr"), 1, glm::value_ptr(viewportStuff));
    glUniform3fvARB(this->cylinderShader.ParameterLocation("camIn"), 1, glm::value_ptr(camdir));
    glUniform3fvARB(this->cylinderShader.ParameterLocation("camRight"), 1, glm::value_ptr(right));
    glUniform3fvARB(this->cylinderShader.ParameterLocation("camUp"), 1, glm::value_ptr(up));
    // get the attribute locations
    GLint attribLocInParams = glGetAttribLocation(this->cylinderShader, "inParams");
    GLint attribLocQuatC = glGetAttribLocation(this->cylinderShader, "quatC");
    GLint attribLocColor1 = glGetAttribLocation(this->cylinderShader, "color1");
    GLint attribLocColor2 = glGetAttribLocation(this->cylinderShader, "color2");
    glBegin(GL_POINTS);
    max1 = (unsigned int) this->reducedSurface.size();
    for (unsigned int cntRS = 0; cntRS < max1; ++cntRS) {
        max2 = this->reducedSurface[cntRS]->GetRSEdgeCount();
        for (unsigned int j = 0; j < max2; ++j) {
            firstAtomPos = this->reducedSurface[cntRS]->GetRSEdge(j)->GetVertex1()->GetPosition();
            secondAtomPos = this->reducedSurface[cntRS]->GetRSEdge(j)->GetVertex2()->GetPosition();

            // compute the quaternion for the rotation of the cylinder
            dir = secondAtomPos - firstAtomPos;
            tmpVec.Set(1.0f, 0.0f, 0.0f);
            angle = -tmpVec.Angle(dir);
            ortho = tmpVec.Cross(dir);
            ortho.Normalise();
            quatC.Set(angle, ortho);
            // compute the absolute position 'position' of the cylinder (center point)
            position = firstAtomPos + (dir / 2.0f);

            // draw vertex and attributes
            glVertexAttrib2f(attribLocInParams, 0.12f, (firstAtomPos - secondAtomPos).Length());
            glVertexAttrib4fv(attribLocQuatC, quatC.PeekComponents());
            glVertexAttrib3f(attribLocColor1, 1.0f, 0.5f, 0.0f);
            glVertexAttrib3f(attribLocColor2, 1.0f, 0.5f, 0.0f);
            glVertex4f(position.GetX(), position.GetY(), position.GetZ(), 1.0f);
        }
    }
    glEnd(); // GL_POINTS
    // disable cylinder shader
    this->cylinderShader.Disable();

    glEnable(GL_COLOR_MATERIAL);
    glPolygonMode(GL_FRONT_AND_BACK, GL_TRIANGLES);
    glDisable(GL_CULL_FACE);
    this->lightShader.Enable();
    unsigned int i;
    for (unsigned int cntRS = 0; cntRS < max1; ++cntRS) {
        max2 = this->reducedSurface[cntRS]->GetRSFaceCount();
        for (i = 0; i < max2; ++i) {
            n1 = this->reducedSurface[cntRS]->GetRSFace(i)->GetFaceNormal();
            v1 = this->reducedSurface[cntRS]->GetRSFace(i)->GetVertex1()->GetPosition();
            v2 = this->reducedSurface[cntRS]->GetRSFace(i)->GetVertex2()->GetPosition();
            v3 = this->reducedSurface[cntRS]->GetRSFace(i)->GetVertex3()->GetPosition();

            glBegin(GL_TRIANGLES);
            glNormal3fv(n1.PeekComponents());
            glColor3f(1.0f, 0.8f, 0.0f);
            glVertex3fv(v1.PeekComponents());
            // glColor3f( 0.0f, 0.7f, 0.7f);
            glVertex3fv(v2.PeekComponents());
            // glColor3f( 0.7f, 0.0f, 0.7f);
            glVertex3fv(v3.PeekComponents());
            glEnd(); // GL_TRIANGLES
        }
    }
    this->lightShader.Disable();
    glDisable(GL_COLOR_MATERIAL);
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
                (float) this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge1()->cuttingProbes.size();
            this->sphericTriaTexCoord1[cntRS][i * 3 + 1] =
                (float) this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge1()->GetTexCoordX();
            this->sphericTriaTexCoord1[cntRS][i * 3 + 2] =
                (float) this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge1()->GetTexCoordY();
            this->sphericTriaTexCoord2[cntRS][i * 3 + 0] =
                (float) this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge2()->cuttingProbes.size();
            this->sphericTriaTexCoord2[cntRS][i * 3 + 1] =
                (float) this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge2()->GetTexCoordX();
            this->sphericTriaTexCoord2[cntRS][i * 3 + 2] =
                (float) this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge2()->GetTexCoordY();
            this->sphericTriaTexCoord3[cntRS][i * 3 + 0] =
                (float) this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge3()->cuttingProbes.size();
            this->sphericTriaTexCoord3[cntRS][i * 3 + 1] =
                (float) this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge3()->GetTexCoordX();
            this->sphericTriaTexCoord3[cntRS][i * 3 + 2] =
                (float) this->reducedSurface[cntRS]->GetRSFace(i)->GetEdge3()->GetTexCoordY();
            // colors
            this->sphericTriaColors[cntRS][i * 3 + 0] = CodeColor(
                &this->atomColorTable[this->reducedSurface[cntRS]->GetRSFace(i)->GetVertex1()->GetIndex() * 3]);
            this->sphericTriaColors[cntRS][i * 3 + 1] = CodeColor(
                &this->atomColorTable[this->reducedSurface[cntRS]->GetRSFace(i)->GetVertex2()->GetIndex() * 3]);
            this->sphericTriaColors[cntRS][i * 3 + 2] = CodeColor(
                &this->atomColorTable[this->reducedSurface[cntRS]->GetRSFace(i)->GetVertex3()->GetIndex() * 3]);
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
            this->torusColors[cntRS][i * 4 + 0] = CodeColor(
                &this->atomColorTable[this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex1()->GetIndex() * 3]);
            this->torusColors[cntRS][i * 4 + 1] = CodeColor(
                &this->atomColorTable[this->reducedSurface[cntRS]->GetRSEdge(i)->GetVertex2()->GetIndex() * 3]);
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
                this->atomColorTable[this->reducedSurface[cntRS]->GetRSVertex(i)->GetIndex() * 3 + 0]);
            this->sphereColors[cntRS].Append(
                this->atomColorTable[this->reducedSurface[cntRS]->GetRSVertex(i)->GetIndex() * 3 + 1]);
            this->sphereColors[cntRS].Append(
                this->atomColorTable[this->reducedSurface[cntRS]->GetRSVertex(i)->GetIndex() * 3 + 2]);
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
            (float) this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge1()->cuttingProbes.size();
        this->sphericTriaTexCoord1[idxRS][i * 3 + 1] =
            (float) this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge1()->GetTexCoordX();
        this->sphericTriaTexCoord1[idxRS][i * 3 + 2] =
            (float) this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge1()->GetTexCoordY();
        this->sphericTriaTexCoord2[idxRS][i * 3 + 0] =
            (float) this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge2()->cuttingProbes.size();
        this->sphericTriaTexCoord2[idxRS][i * 3 + 1] =
            (float) this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge2()->GetTexCoordX();
        this->sphericTriaTexCoord2[idxRS][i * 3 + 2] =
            (float) this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge2()->GetTexCoordY();
        this->sphericTriaTexCoord3[idxRS][i * 3 + 0] =
            (float) this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge3()->cuttingProbes.size();
        this->sphericTriaTexCoord3[idxRS][i * 3 + 1] =
            (float) this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge3()->GetTexCoordX();
        this->sphericTriaTexCoord3[idxRS][i * 3 + 2] =
            (float) this->reducedSurface[idxRS]->GetRSFace(i)->GetEdge3()->GetTexCoordY();
        // colors
        this->sphericTriaColors[idxRS][i * 3 + 0] =
            CodeColor(&this->atomColorTable[this->reducedSurface[idxRS]->GetRSFace(i)->GetVertex1()->GetIndex() * 3]);
        this->sphericTriaColors[idxRS][i * 3 + 1] =
            CodeColor(&this->atomColorTable[this->reducedSurface[idxRS]->GetRSFace(i)->GetVertex2()->GetIndex() * 3]);
        this->sphericTriaColors[idxRS][i * 3 + 2] =
            CodeColor(&this->atomColorTable[this->reducedSurface[idxRS]->GetRSFace(i)->GetVertex3()->GetIndex() * 3]);
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
            CodeColor(&this->atomColorTable[this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex1()->GetIndex() * 3]);
        this->torusColors[idxRS][i * 4 + 1] =
            CodeColor(&this->atomColorTable[this->reducedSurface[idxRS]->GetRSEdge(i)->GetVertex2()->GetIndex() * 3]);
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
            this->atomColorTable[this->reducedSurface[idxRS]->GetRSVertex(i)->GetIndex() * 3 + 0]);
        this->sphereColors[idxRS].Append(
            this->atomColorTable[this->reducedSurface[idxRS]->GetRSVertex(i)->GetIndex() * 3 + 1]);
        this->sphereColors[idxRS].Append(
            this->atomColorTable[this->reducedSurface[idxRS]->GetRSVertex(i)->GetIndex() * 3 + 2]);
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
    return float((int) (col[0] * 255.0f) * 1000000 // red
                 + (int) (col[1] * 255.0f) * 1000  // green
                 + (int) (col[2] * 255.0f));       // blue
}


/*
 * decode a coded color to the original rgb-color
 */
vislib::math::Vector<float, 3> MoleculeSESRenderer::DecodeColor(int codedColor) const {
    int col = codedColor;
    vislib::math::Vector<float, 3> color;
    float red, green;
    if (col >= 1000000)
        red = floor((float) col / 1000000.0f);
    else
        red = 0.0;
    col = col - int(red * 1000000.0f);
    if (col > 1000)
        green = floor((float) col / 1000.0f);
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
        unsigned int singTexSizeOld = (unsigned int) this->singularityTexture.size();
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
        if ((unsigned int) texSize < this->reducedSurface[cntRS]->GetCutRSEdgesCount()) {
            this->singTexHeight[cntRS] = texSize;
            this->singTexWidth[cntRS] =
                numProbes * (int) ceil(double(this->reducedSurface[cntRS]->GetCutRSEdgesCount()) / (double) texSize);
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
    if ((unsigned int) texSize < this->reducedSurface[idxRS]->GetCutRSEdgesCount()) {
        this->singTexHeight[idxRS] = texSize;
        this->singTexWidth[idxRS] =
            numProbes * (int) ceil(double(this->reducedSurface[idxRS]->GetCutRSEdgesCount()) / (double) texSize);
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
 * Render all atoms
 */
void MoleculeSESRenderer::RenderAtomsGPU(const MolecularDataCall* mol, const float scale) {
    unsigned int cnt, cntRS, max1, max2;

    // set viewport
    glm::vec4 viewportStuff;
    viewportStuff[0] = 0.0f;
    viewportStuff[1] = 0.0f;
    viewportStuff[2] = static_cast<float>(fbo->getWidth());
    viewportStuff[3] = static_cast<float>(fbo->getHeight());
    if (viewportStuff[2] < 1.0f)
        viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f)
        viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glm::vec3 camdir = camera.get<core::view::Camera::Pose>().direction;
    glm::vec3 right = camera.get<core::view::Camera::Pose>().right;
    glm::vec3 up = camera.get<core::view::Camera::Pose>().up;

    // enable sphere shader
    this->sphereShader.Enable();
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, glm::value_ptr(viewportStuff));
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, glm::value_ptr(camdir));
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, glm::value_ptr(right));
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, glm::value_ptr(up));

    glBegin(GL_POINTS);

    glColor3f(1.0f, 0.0f, 0.0f);
    max1 = (unsigned int) this->reducedSurface.size();
    for (cntRS = 0; cntRS < max1; ++cntRS) {
        max2 = this->reducedSurface[cntRS]->GetRSVertexCount();
        // loop over all protein atoms
        for (cnt = 0; cnt < max2; ++cnt) {
            if (this->reducedSurface[cntRS]->GetRSVertex(cnt)->IsBuried())
                continue;
            // glColor3ubv( protein->AtomTypes()[protein->ProteinAtomData()[this->reducedSurface[cntRS]->GetRSVertex(
            // cnt)->GetIndex()].TypeIndex()].Colour());
            glColor3f(1.0f, 0.0f, 0.0f);
            glVertex4f(this->reducedSurface[cntRS]->GetRSVertex(cnt)->GetPosition().GetX(),
                this->reducedSurface[cntRS]->GetRSVertex(cnt)->GetPosition().GetY(),
                this->reducedSurface[cntRS]->GetRSVertex(cnt)->GetPosition().GetZ(),
                this->reducedSurface[cntRS]->GetRSVertex(cnt)->GetRadius() * scale);
        }
    }

    glEnd(); // GL_POINTS

    // disable sphere shader
    this->sphereShader.Disable();
}


/*
 * Renders the probe at postion 'm'
 */
/*
void MoleculeSESRenderer::RenderProbe(const vislib::math::Vector<float, 3> m) {
    GLUquadricObj* sphere = gluNewQuadric();
    gluQuadricNormals(sphere, GL_SMOOTH);

    this->probeRadius = this->probeRadiusSlot.Param<param::FloatParam>()->Value();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glPushMatrix();
    glTranslatef(m.GetX(), m.GetY(), m.GetZ());
    glColor4f(1.0f, 1.0f, 1.0f, 0.6f);
    gluSphere(sphere, probeRadius, 16, 8);
    glPopMatrix();

    glDisable(GL_BLEND);
}
*/


/*
 * Renders the probe at postion 'm'
 */
void MoleculeSESRenderer::RenderProbeGPU(const vislib::math::Vector<float, 3> m) {
    // set viewport
    glm::vec4 viewportStuff;
    viewportStuff[0] = 0.0f;
    viewportStuff[1] = 0.0f;
    viewportStuff[2] = static_cast<float>(fbo->getWidth());
    viewportStuff[3] = static_cast<float>(fbo->getHeight());
    if (viewportStuff[2] < 1.0f)
        viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f)
        viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glm::vec3 camdir = camera.get<core::view::Camera::Pose>().direction;
    glm::vec3 right = camera.get<core::view::Camera::Pose>().right;
    glm::vec3 up = camera.get<core::view::Camera::Pose>().up;

    // enable sphere shader
    this->sphereShader.Enable();
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, glm::value_ptr(viewportStuff));
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, glm::value_ptr(camdir));
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, glm::value_ptr(right));
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, glm::value_ptr(up));

    this->probeRadius = this->probeRadiusSlot.Param<param::FloatParam>()->Value();

    glBegin(GL_POINTS);
    glColor3f(1.0f, 1.0f, 1.0f);
    glVertex4f(m.GetX(), m.GetY(), m.GetZ(), probeRadius);
    glEnd();

    // disable sphere shader
    this->sphereShader.Disable();
}


/*
 * MoleculeSESRenderer::deinitialise
 */
void MoleculeSESRenderer::deinitialise(void) {
    if (colorFBO) {
        glDeleteFramebuffersEXT(1, &colorFBO);
        glDeleteFramebuffersEXT(1, &blendFBO);
        glDeleteFramebuffersEXT(1, &horizontalFilterFBO);
        glDeleteFramebuffersEXT(1, &verticalFilterFBO);
        glDeleteTextures(1, &texture0);
        glDeleteTextures(1, &depthTex0);
        glDeleteTextures(1, &texture1);
        glDeleteTextures(1, &depthTex1);
        glDeleteTextures(1, &hFilter);
        glDeleteTextures(1, &vFilter);
    }
    // delete singularity texture
    for (unsigned int i = 0; i < singularityTexture.size(); ++i)
        glDeleteTextures(1, &singularityTexture[i]);
    // release shaders
    this->cylinderShader.Release();
    this->sphereShader.Release();
    this->sphereClipInteriorShader.Release();
    this->lightShader.Release();
    this->hfilterShader.Release();
    this->vfilterShader.Release();
    this->silhouetteShader.Release();
    this->transparencyShader.Release();
}


/*
 * returns the color of the atom 'idx' for the current coloring mode
 */
vislib::math::Vector<float, 3> MoleculeSESRenderer::GetProteinAtomColor(unsigned int idx) {
    if (idx < this->atomColorTable.Count() / 3)
        // return this->atomColorTable[idx];
        return vislib::math::Vector<float, 3>(
            this->atomColorTable[idx * 3 + 0], this->atomColorTable[idx * 3 + 1], this->atomColorTable[idx * 3 + 0]);
    else
        return vislib::math::Vector<float, 3>(0.5f, 0.5f, 0.5f);
}
