/*/*
 * SphereRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 *
 */

#include "stdafx.h"
#include "SphereRenderer.h"


using namespace megamol::core;
using namespace megamol::core::moldyn;
using namespace megamol::stdplugin::moldyn::rendering;
using namespace vislib::graphics::gl;


//#define CHRONOTIMING

#define SSBO_GENERATED_SHADER_INSTANCE "gl_VertexID" // or "gl_InstanceID"
#define SSBO_GENERATED_SHADER_ALIGNMENT "packed"     // "std430"

const GLuint SSBOvertexBindingPoint = 2;
const GLuint SSBOcolorBindingPoint = 3;
const GLuint SSBOflagsBindingPoint = 4;

SphereRenderer::SphereRenderer(void) : view::Renderer3DModule_2()
    , getDataSlot("getdata", "Connects to the data source")
    , getTFSlot("gettransferfunction", "Connects to the transfer function module")
    , getClipPlaneSlot("getclipplane", "Connects to a clipping plane module")
    , getFlagsSlot("getflags", "Connects to a flag storage module")
    , curViewAttrib()
    , curClipDat()
    , oldClipDat()
    , curClipCol()
    , curlightDir()
    , curVpWidth(-1)
    , curVpHeight(-1)
    , lastVpWidth(0)
    , lastVpHeight(0)
    , curMVinv()
    , curMVtransp()
    , curMVP()
    , curMVPinv()
    , curMVPtransp()
    , renderMode(RenderMode::SIMPLE)
    , greyTF(0)
    , flagsEnabled(false)
    , flagsBuffer(0)
    , flagsUseSSBO(false)
    , flagsCurrentVersion(0xFFFFFFFF)
    , flagsData(nullptr)
    , sphereShader()
    , sphereGeometryShader()
    , lightingShader()
    , vertShader(nullptr)
    , fragShader(nullptr)
    , geoShader(nullptr)
    , vertArray()
    , colType(SimpleSphericalParticles::ColourDataType::COLDATA_NONE)
    , vertType(SimpleSphericalParticles::VertexDataType::VERTDATA_NONE)
    , newShader(nullptr)
    , theShaders()
    , theSingleBuffer()
    , currBuf(0)
    , bufSize(32 * 1024 * 1024)
    , numBuffers(3)
    , theSingleMappedMem(nullptr)
    , gpuData()
    , gBuffer()
    , oldHash(-1)
    , oldFrameID(0)
    , ambConeConstants()
    , volGen(nullptr)
    , triggerRebuildGBuffer(false)
// , timer()
#if defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)
    /// This variant should not need the fence (?)
    // ,singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT);
    // ,singleBufferMappingBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT);
    , singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT)
    , singleBufferMappingBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT)
    , fences()
#endif // defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)
#ifdef SPHERE_MIN_OGL_SSBO_STREAM
    , streamer()
    , colStreamer()
    , bufArray()
    , colBufArray()
#endif // SPHERE_MIN_OGL_SSBO_STREAM
    , renderModeParam("renderMode", "The sphere render mode.")
    , radiusScalingParam("scaling", "Scaling factor for particle radii.")
    , forceTimeSlot(
          "forceTime", "Flag to force the time code to the specified value. Set to true when rendering a video.")
    , useLocalBBoxParam("useLocalBbox", "Enforce usage of local bbox for camera setup")
    , colIdxRangeInfoParam(
          "transfer function::colorIndexRange", "The current color index range. Use as range in transfer function.")
    , selectColorParam("flag storage::selectedColor", "Color for selected spheres in flag storage.")
    , softSelectColorParam("flag storage::softSelectedColor", "Color for soft selected spheres in flag storage.")
    , alphaScalingParam("splat::alphaScaling", "Splat: Scaling factor for particle alpha.")
    , attenuateSubpixelParam(
          "splat::attenuateSubpixel", "Splat: Attenuate alpha of points that should have subpixel size.")
    , useStaticDataParam("ssbo::staticData", "SSBO: Upload data only once per hash change and keep data static on GPU")
    , enableLightingSlot("ambient occlusion::enableLighting", "Ambient Occlusion: Enable Lighting")
    , enableGeometryShader("ambient occlusion::useGsProxies",
          "Ambient Occlusion: Enables rendering using triangle strips from the geometry shader")
    , aoVolSizeSlot("ambient occlusion::volumeSize", "Ambient Occlusion: Longest volume edge")
    , aoConeApexSlot("ambient occlusion::apex", "Ambient Occlusion: Cone Apex Angle")
    , aoOffsetSlot("ambient occlusion::offset", "Ambient Occlusion: Offset from Surface")
    , aoStrengthSlot("ambient occlusion::strength", "Ambient Occlusion: Strength")
    , aoConeLengthSlot("ambient occlusion::coneLength", "Ambient Occlusion: Cone length")
    , useHPTexturesSlot("ambient occlusion::highPrecisionTexture", "Ambient Occlusion: Use high precision textures")
    , outlineSizeSlot("outline::width", "Width of the outline") {

    this->getDataSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getClipPlaneSlot.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);

    this->getFlagsSlot.SetCompatibleCall<FlagCallDescription>();
    this->MakeSlotAvailable(&this->getFlagsSlot);

    // Initialising enum param with all possible modes (needed for configurator)
    // (Removing not available render modes later in create function)
    param::EnumParam* rmp = new param::EnumParam(this->renderMode);
    rmp->SetTypePair(RenderMode::SIMPLE, "Simple");
    rmp->SetTypePair(RenderMode::SIMPLE_CLUSTERED, "Simple_Clustered");
    rmp->SetTypePair(RenderMode::GEOMETRY_SHADER, "Geometry_Shader");
    rmp->SetTypePair(RenderMode::SSBO_STREAM, "SSBO_Stream");
    rmp->SetTypePair(RenderMode::BUFFER_ARRAY, "Buffer_Array");
    rmp->SetTypePair(RenderMode::SPLAT, "Splat");
    rmp->SetTypePair(RenderMode::AMBIENT_OCCLUSION, "Ambient_Occlusion");
    rmp->SetTypePair(RenderMode::OUTLINE, "Outline");
    this->renderModeParam << rmp;
    this->MakeSlotAvailable(&this->renderModeParam);

    this->radiusScalingParam << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->radiusScalingParam);

    this->forceTimeSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->forceTimeSlot);

    this->useLocalBBoxParam << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->useLocalBBoxParam);

    this->colIdxRangeInfoParam << new param::Vector2fParam(vislib::math::Vector<float, 2>(0.0f, 0.0f));
    this->MakeSlotAvailable(&this->colIdxRangeInfoParam);
    this->colIdxRangeInfoParam.Param<param::Vector2fParam>()->SetGUIReadOnly(true);

    this->selectColorParam << new param::ColorParam(1.0f, 0.0f, 0.0f, 1.0f);
    this->MakeSlotAvailable(&this->selectColorParam);

    this->softSelectColorParam << new param::ColorParam(1.0f, 0.5f, 0.5f, 1.0f);
    this->MakeSlotAvailable(&this->softSelectColorParam);

    this->alphaScalingParam << new param::FloatParam(5.0f);
    this->MakeSlotAvailable(&this->alphaScalingParam);

    this->attenuateSubpixelParam << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->attenuateSubpixelParam);

    this->useStaticDataParam << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->useStaticDataParam);

    this->enableLightingSlot << (new param::BoolParam(false));
    this->MakeSlotAvailable(&this->enableLightingSlot);

    this->enableGeometryShader << (new param::BoolParam(false));
    this->MakeSlotAvailable(&this->enableGeometryShader);

    this->aoVolSizeSlot << (new param::IntParam(128, 1, 1024));
    this->MakeSlotAvailable(&this->aoVolSizeSlot);

    this->aoConeApexSlot << (new param::FloatParam(50.0f, 1.0f, 90.0f));
    this->MakeSlotAvailable(&this->aoConeApexSlot);

    this->aoOffsetSlot << (new param::FloatParam(0.01f, 0.0f, 0.2f));
    this->MakeSlotAvailable(&this->aoOffsetSlot);

    this->aoStrengthSlot << (new param::FloatParam(1.0f, 0.1f, 20.0f));
    this->MakeSlotAvailable(&this->aoStrengthSlot);

    this->aoConeLengthSlot << (new param::FloatParam(0.8f, 0.01f, 1.0f));
    this->MakeSlotAvailable(&this->aoConeLengthSlot);

    this->useHPTexturesSlot << (new param::BoolParam(false));
    this->MakeSlotAvailable(&this->useHPTexturesSlot);

    this->outlineSizeSlot << (new core::param::FloatParam(2.0f, 0.0f));
    this->MakeSlotAvailable(&this->outlineSizeSlot);
}


SphereRenderer::~SphereRenderer(void) { this->Release(); }


bool SphereRenderer::GetExtents(view::CallRender3D_2& call) {

    auto cr = &call;
    if (cr == nullptr) return false;

    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if ((c2 != nullptr)) {
        c2->SetFrameID(static_cast<unsigned int>(cr->Time()), this->forceTimeSlot.Param<param::BoolParam>()->Value());
        if (!(*c2)(1)) return false;
        cr->SetTimeFramesCount(c2->FrameCount());
        auto const plcount = c2->GetParticleListCount();
        if (this->useLocalBBoxParam.Param<param::BoolParam>()->Value() && plcount > 0) {
            auto bbox = c2->AccessParticles(0).GetBBox();
            auto cbbox = bbox;
            cbbox.Grow(c2->AccessParticles(0).GetGlobalRadius());
            for (unsigned pidx = 1; pidx < plcount; ++pidx) {
                auto temp = c2->AccessParticles(pidx).GetBBox();
                bbox.Union(temp);
                temp.Grow(c2->AccessParticles(pidx).GetGlobalRadius());
                cbbox.Union(temp);
            }
            cr->AccessBoundingBoxes().SetBoundingBox(bbox);
            cr->AccessBoundingBoxes().SetClipBox(cbbox);
        }
        else {
            cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();
        }

    }
    else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }
    this->curClipBox = cr->AccessBoundingBoxes().ClipBox();

    return true;
}


bool SphereRenderer::create(void) {

    ASSERT(IsAvailable());
    // At least the simple render mode must be available
    ASSERT(this->isRenderModeAvailable(RenderMode::SIMPLE));

    // Reduce to available render modes
    this->SetSlotUnavailable(&this->renderModeParam);
    this->renderModeParam.Param<param::EnumParam>()->ClearTypePairs();
    this->renderModeParam.Param<param::EnumParam>()->SetTypePair(RenderMode::SIMPLE, "Simple");
    if (this->isRenderModeAvailable(RenderMode::SIMPLE_CLUSTERED)) {
        this->renderModeParam.Param<param::EnumParam>()->SetTypePair(RenderMode::SIMPLE_CLUSTERED, "Simple_Clustered");
    }
    if (this->isRenderModeAvailable(RenderMode::GEOMETRY_SHADER)) {
        this->renderModeParam.Param<param::EnumParam>()->SetTypePair(RenderMode::GEOMETRY_SHADER, "Geometry_Shader");
    }
    if (this->isRenderModeAvailable(RenderMode::SSBO_STREAM)) {
        this->renderModeParam.Param<param::EnumParam>()->SetTypePair(RenderMode::SSBO_STREAM, "SSBO_Stream");
    }
    if (this->isRenderModeAvailable(RenderMode::SPLAT)) {
        this->renderModeParam.Param<param::EnumParam>()->SetTypePair(RenderMode::SPLAT, "Splat");
    }
    if (this->isRenderModeAvailable(RenderMode::BUFFER_ARRAY)) {
        this->renderModeParam.Param<param::EnumParam>()->SetTypePair(RenderMode::BUFFER_ARRAY, "Buffer_Array");
    }
    if (this->isRenderModeAvailable(RenderMode::AMBIENT_OCCLUSION)) {
        this->renderModeParam.Param<param::EnumParam>()->SetTypePair(
            RenderMode::AMBIENT_OCCLUSION, "Ambient_Occlusion");
    }
    if (this->isRenderModeAvailable(RenderMode::OUTLINE)) {
        this->renderModeParam.Param<param::EnumParam>()->SetTypePair(RenderMode::OUTLINE, "Outline");
    }
    this->MakeSlotAvailable(&this->renderModeParam);

    // Check initial render mode
    if (!this->isRenderModeAvailable(this->renderMode)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
            "[SphereRenderer] Render mode: %s is not available - falling back to SIMPLE render mode.",
            (this->getRenderModeString(this->renderMode)).c_str());
        // Always available fallback render mode
        this->renderMode = RenderMode::SIMPLE;
    }

    // Create resources for initial render mode
    if (!this->createResources()) {
        return false;
    }

    // timer.SetNumRegions(4);
    // const char *regions[4] = {"Upload1", "Upload2", "Upload3", "Rendering"};megamol::
    // timer.SetRegionNames(4, regions);
    // timer.SetStatisticsFileName("fullstats.csv");
    // timer.SetSummaryFileName("summary.csv");
    // timer.SetMaximumFrames(20, 100);

    return true;
}


void SphereRenderer::release(void) {

    this->resetResources();
}


bool SphereRenderer::resetResources(void) {

    this->selectColorParam.Param<param::ColorParam>()->SetGUIVisible(false);
    this->softSelectColorParam.Param<param::ColorParam>()->SetGUIVisible(false);
    this->colIdxRangeInfoParam.Param<param::Vector2fParam>()->SetGUIVisible(false);

    // Set all render mode dependent parameter to GUI invisible
    // SPLAT
    this->alphaScalingParam.Param<param::FloatParam>()->SetGUIVisible(false);
    this->attenuateSubpixelParam.Param<param::BoolParam>()->SetGUIVisible(false);
    // SSBO
    this->useStaticDataParam.Param<param::BoolParam>()->SetGUIVisible(false);
    // Ambient Occlusion
    this->enableLightingSlot.Param<param::BoolParam>()->SetGUIVisible(false);
    this->enableGeometryShader.Param<param::BoolParam>()->SetGUIVisible(false);
    this->aoVolSizeSlot.Param<param::IntParam>()->SetGUIVisible(false);
    this->aoConeApexSlot.Param<param::FloatParam>()->SetGUIVisible(false);
    this->aoOffsetSlot.Param<param::FloatParam>()->SetGUIVisible(false);
    this->aoStrengthSlot.Param<param::FloatParam>()->SetGUIVisible(false);
    this->aoConeLengthSlot.Param<param::FloatParam>()->SetGUIVisible(false);
    this->useHPTexturesSlot.Param<param::BoolParam>()->SetGUIVisible(false);
    // Outlining
    this->outlineSizeSlot.Param<param::FloatParam>()->SetGUIVisible(false);

    this->flagsCurrentVersion = (0xFFFFFFFF);
    this->flagsEnabled = false;
    this->flagsUseSSBO = false;
    this->flagsData = nullptr;
    if (this->flagsUseSSBO) {
        glDeleteBuffers(1, &this->flagsBuffer);
    }

    if (this->greyTF != 0) {
        glDeleteTextures(1, &this->greyTF);
    }
    this->greyTF = 0;

    this->sphereShader.Release();
    this->sphereGeometryShader.Release();
    this->lightingShader.Release();

    this->vertShader.reset();
    this->fragShader.reset();
    this->geoShader.reset();

    this->theSingleMappedMem = nullptr;

    if (this->newShader != nullptr) {
        this->newShader->Release();
        this->newShader.reset();
    }
    this->theShaders.clear();

    if (this->volGen != nullptr) {
        delete this->volGen;
        this->volGen = nullptr;
    }

    this->currBuf = 0;
    this->bufSize = (32 * 1024 * 1024);
    this->numBuffers = 3;

    this->oldHash = -1;
    this->oldFrameID = -1;

    this->colType = SimpleSphericalParticles::ColourDataType::COLDATA_NONE;
    this->vertType = SimpleSphericalParticles::VertexDataType::VERTDATA_NONE;

    // AMBIENT OCCLUSION
    if (this->isRenderModeAvailable(RenderMode::AMBIENT_OCCLUSION, true)) {
        for (unsigned int i = 0; i < this->gpuData.size(); ++i) {
            glDeleteVertexArrays(3, reinterpret_cast<GLuint*>(&(this->gpuData[i])));
        }
        this->gpuData.clear();

        if (this->gBuffer.color != 0) {
            glDeleteTextures(1, &this->gBuffer.color);
        }
        this->gBuffer.color = 0;
        if (this->gBuffer.depth != 0) {
            glDeleteTextures(1, &this->gBuffer.depth);
        }
        this->gBuffer.depth = 0;
        if (this->gBuffer.normals != 0) {
            glDeleteTextures(1, &this->gBuffer.normals);
        }
        this->gBuffer.normals = 0;

        glDeleteFramebuffers(1, &this->gBuffer.fbo);
    }

    // SPLAT or BUFFER_ARRAY
    if (this->isRenderModeAvailable(RenderMode::SPLAT, true) ||
        this->isRenderModeAvailable(RenderMode::BUFFER_ARRAY, true)) {

        glUnmapNamedBuffer(this->theSingleBuffer);

        for (auto& x : fences) {
            if (x) {
                glDeleteSync(x);
            }
        }
        this->fences.clear();
        this->fences.resize(numBuffers);

        this->singleBufferCreationBits = (GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT);
        this->singleBufferMappingBits = (GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);

        glDeleteBuffers(1, &(this->theSingleBuffer));
    }

    // SSBO or SPLAT or BUFFER_ARRAY
    if (this->isRenderModeAvailable(RenderMode::SSBO_STREAM) || this->isRenderModeAvailable(RenderMode::SPLAT) ||
        this->isRenderModeAvailable(RenderMode::BUFFER_ARRAY)) {
        glDeleteVertexArrays(1, &(this->vertArray));
    }

    return true;
}


bool SphereRenderer::createResources() {

    this->resetResources();

    this->vertShader = std::make_shared<ShaderSource>();
    this->fragShader = std::make_shared<ShaderSource>();

    vislib::StringA vertShaderName;
    vislib::StringA fragShaderName;
    vislib::StringA geoShaderName;

    if (!this->isRenderModeAvailable(this->renderMode)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
            "[SphereRenderer] Render mode: %s is not available - falling back to SIMPLE render mode.",
            (this->getRenderModeString(this->renderMode)).c_str());
        this->renderMode = RenderMode::SIMPLE; // Fallback render mode ...
        return false;
    } else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "[SphereRenderer] Using render mode: %s",
            (this->getRenderModeString(this->renderMode)).c_str());
    }

    // Check availbility of ssbo flag storage
    int major = -1;
    int minor = -1;
    this->flagsUseSSBO = false;
    this->getGLSLVersion(major, minor);
    if ((major == 4) && (minor >= 3) || (major > 4)) {
        this->flagsUseSSBO = true;
    }
    vislib::SmartPtr<ShaderSource::Snippet> flagSnippet;
    std::string flagDefine = "\n#define FLAG_STORAGE_SSBO\n\n";
    flagSnippet = new ShaderSource::StringSnippet(flagDefine.c_str());

    // Fallback transfer function texture
    glGenTextures(1, &this->greyTF);
    unsigned char tex[6] = {0, 0, 0, 255, 255, 255};
    glBindTexture(GL_TEXTURE_1D, this->greyTF);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);

    try {
        switch (this->renderMode) {

        case (RenderMode::SIMPLE):
        case (RenderMode::SIMPLE_CLUSTERED): {
            vertShaderName = "sphere_simple::vertex";
            fragShaderName = "sphere_simple::fragment";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), (*this->vertShader))) {
                return false;
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), *this->fragShader)) {
                return false;
            }
            if (!this->sphereShader.Create(this->vertShader->Code(), this->vertShader->Count(),
                    this->fragShader->Code(), this->fragShader->Count())) {
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Unable to compile sphere shader: Unknown error\n");
                return false;
            }
            this->flagsUseSSBO = false;
        } break;

        case (RenderMode::GEOMETRY_SHADER): {
            this->geoShader = std::make_shared<ShaderSource>();
            vertShaderName = "sphere_geo::vertex";
            fragShaderName = "sphere_geo::fragment";
            geoShaderName = "sphere_geo::geometry";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), *this->vertShader)) {
                return false;
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), *this->fragShader)) {
                return false;
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(geoShaderName.PeekBuffer(), *this->geoShader)) {
                return false;
            }
            if (!this->sphereGeometryShader.Compile(this->vertShader->Code(), this->vertShader->Count(),
                    this->geoShader->Code(), this->geoShader->Count(), this->fragShader->Code(),
                    this->fragShader->Count())) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "[SphereRenderer] Unable to compile sphere geometry shader: Unknown error\n");
                return false;
            }
            if (!this->sphereGeometryShader.Link()) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "[SphereRenderer] Unable to link sphere geometry shader: Unknown error\n");
                return false;
            }
            this->flagsUseSSBO = false;
        } break;

        case (RenderMode::SSBO_STREAM): {
            this->useStaticDataParam.Param<param::BoolParam>()->SetGUIVisible(true);
            vertShaderName = "sphere_ssbo::vertex";
            fragShaderName = "sphere_ssbo::fragment";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), *this->vertShader)) {
                return false;
            }
            if (this->flagsUseSSBO) {
                this->vertShader->Insert(1, flagSnippet);
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), *this->fragShader)) {
                return false;
            }
            glGenVertexArrays(1, &this->vertArray);
            glBindVertexArray(this->vertArray);
            glBindVertexArray(0);
        } break;

        case (RenderMode::SPLAT): {
            this->alphaScalingParam.Param<param::FloatParam>()->SetGUIVisible(true);
            this->attenuateSubpixelParam.Param<param::BoolParam>()->SetGUIVisible(true);
            vertShaderName = "sphere_splat::vertex";
            fragShaderName = "sphere_splat::fragment";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), *this->vertShader)) {
                return false;
            }
            if (this->flagsUseSSBO) {
                this->vertShader->Insert(1, flagSnippet);
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), *this->fragShader)) {
                return false;
            }
            glGenVertexArrays(1, &this->vertArray);
            glBindVertexArray(this->vertArray);
            glGenBuffers(1, &this->theSingleBuffer);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSingleBuffer);
            glBufferStorage(
                GL_SHADER_STORAGE_BUFFER, this->bufSize * this->numBuffers, nullptr, singleBufferCreationBits);
            this->theSingleMappedMem = glMapNamedBufferRange(
                this->theSingleBuffer, 0, this->bufSize * this->numBuffers, singleBufferMappingBits);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            glBindVertexArray(0);
        } break;

        case (RenderMode::BUFFER_ARRAY): {
            vertShaderName = "sphere_bufferarray::vertex";
            fragShaderName = "sphere_bufferarray::fragment";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), *this->vertShader)) {
                return false;
            }
            if (this->flagsUseSSBO) {
                this->vertShader->Insert(1, flagSnippet);
            }
            // this->flagsUseSSBO = false;
            if (!instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), *this->fragShader)) {
                return false;
            }
            if (!this->sphereShader.Create(this->vertShader->Code(), this->vertShader->Count(),
                    this->fragShader->Code(), this->fragShader->Count())) {
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Unable to compile sphere shader: Unknown error\n");
                return false;
            }
            glGenVertexArrays(1, &this->vertArray);
            glBindVertexArray(this->vertArray);
            glGenBuffers(1, &this->theSingleBuffer);
            glBindBuffer(GL_ARRAY_BUFFER, this->theSingleBuffer);
            glBufferStorage(GL_ARRAY_BUFFER, this->bufSize * this->numBuffers, nullptr, singleBufferCreationBits);
            this->theSingleMappedMem = glMapNamedBufferRange(
                this->theSingleBuffer, 0, this->bufSize * this->numBuffers, singleBufferMappingBits);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
        } break;

        case (RenderMode::AMBIENT_OCCLUSION): {
            this->enableLightingSlot.Param<param::BoolParam>()->SetGUIVisible(true);
            this->enableGeometryShader.Param<param::BoolParam>()->SetGUIVisible(true);
            this->aoVolSizeSlot.Param<param::IntParam>()->SetGUIVisible(true);
            this->aoConeApexSlot.Param<param::FloatParam>()->SetGUIVisible(true);
            this->aoOffsetSlot.Param<param::FloatParam>()->SetGUIVisible(true);
            this->aoStrengthSlot.Param<param::FloatParam>()->SetGUIVisible(true);
            this->aoConeLengthSlot.Param<param::FloatParam>()->SetGUIVisible(true);
            this->useHPTexturesSlot.Param<param::BoolParam>()->SetGUIVisible(true);

            this->aoConeApexSlot.ResetDirty();
            this->enableLightingSlot.ResetDirty();

            // Generate texture and frame buffer handles
            glGenTextures(1, &this->gBuffer.color);
            glGenTextures(1, &this->gBuffer.normals);
            glGenTextures(1, &this->gBuffer.depth);
            glGenFramebuffers(1, &(this->gBuffer.fbo));

            // Create the sphere shader
            vertShaderName = "sphere_mdao::vertex";
            fragShaderName = "sphere_mdao::fragment";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), *this->vertShader)) {
                return false;
            }
            if (this->flagsUseSSBO) {
                this->vertShader->Insert(1, flagSnippet);
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), *this->fragShader)) {
                return false;
            }
            if (!this->sphereShader.Create(this->vertShader->Code(), this->vertShader->Count(),
                    this->fragShader->Code(), this->fragShader->Count())) {
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Unable to compile sphere shader: Unknown error\n");
                return false;
            }

            // Create the geometry shader
            this->geoShader = std::make_shared<ShaderSource>();
            this->vertShader->Clear();
            this->fragShader->Clear();
            vertShaderName = "sphere_mdao::geometry::vertex";
            geoShaderName = "sphere_mdao::geometry::geometry";
            fragShaderName = "sphere_mdao::geometry::fragment";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), *this->vertShader)) {
                return false;
            }
            if (this->flagsUseSSBO) {
                this->vertShader->Insert(1, flagSnippet);
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(geoShaderName.PeekBuffer(), *this->geoShader)) {
                return false;
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), *this->fragShader)) {
                return false;
            }
            if (!this->sphereGeometryShader.Compile(this->vertShader->Code(), this->vertShader->Count(),
                    this->geoShader->Code(), this->geoShader->Count(), this->fragShader->Code(),
                    this->fragShader->Count())) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "[SphereRenderer] Unable to compile sphere geometry shader: Unknown error\n");
                return false;
            }
            if (!this->sphereGeometryShader.Link()) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "[SphereRenderer] Unable to link sphere geometry shader: Unknown error\n");
                return false;
            }

            // Create the deferred shader
            this->vertShader->Clear();
            this->fragShader->Clear();
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "sphere_mdao::deferred::vertex", *this->vertShader)) {
                return false;
            }
            bool enableLighting = this->enableLightingSlot.Param<param::BoolParam>()->Value();
            this->fragShader->Append(
                instance()->ShaderSourceFactory().MakeShaderSnippet("sphere_mdao::deferred::fragment::Main"));
            if (enableLighting) {
                this->fragShader->Append(
                    instance()->ShaderSourceFactory().MakeShaderSnippet("sphere_mdao::deferred::fragment::Lighting"));
            } else {
                this->fragShader->Append(instance()->ShaderSourceFactory().MakeShaderSnippet(
                    "sphere_mdao::deferred::fragment::LightingStub"));
            }
            float apex = this->aoConeApexSlot.Param<param::FloatParam>()->Value();
            std::vector<glm::vec4> directions;
            this->generate3ConeDirections(directions, apex * static_cast<float>(M_PI) / 180.0f);
            std::string directionsCode = this->generateDirectionShaderArrayString(directions, "coneDirs");
            vislib::graphics::gl::ShaderSource::StringSnippet* dirSnippet =
                new vislib::graphics::gl::ShaderSource::StringSnippet(directionsCode.c_str());
            this->fragShader->Append(dirSnippet);
            this->fragShader->Append(instance()->ShaderSourceFactory().MakeShaderSnippet(
                "sphere_mdao::deferred::fragment::AmbientOcclusion"));
            if (!this->lightingShader.Create(this->vertShader->Code(), this->vertShader->Count(),
                    this->fragShader->Code(), this->fragShader->Count())) {
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Unable to compile mdao lightning shader.");
                return false;
            }

            // Init volume generator
            this->volGen = new misc::MDAOVolumeGenerator();
            this->volGen->SetShaderSourceFactory(&this->GetCoreInstance()->ShaderSourceFactory());
            if (!this->volGen->Init()) {
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR, "Error initializing volume generator!\n");
                return false;
            }

            this->triggerRebuildGBuffer = true;
        } break;

        case RenderMode::OUTLINE: {
            this->outlineSizeSlot.Param<param::FloatParam>()->SetGUIVisible(true);
            vertShaderName = "sphere_outline::vertex";
            fragShaderName = "sphere_outline::fragment";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), *this->vertShader)) {
                return false;
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), *this->fragShader)) {
                return false;
            }
            if (!this->sphereShader.Create(this->vertShader->Code(), this->vertShader->Count(),
                this->fragShader->Code(), this->fragShader->Count())) {
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Unable to compile sphere shader: Unknown error\n");
                return false;
            }
            this->flagsUseSSBO = false;
        } break;

        default:
            return false;
        }
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "[SphereRenderer] Unable to compile sphere shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Unable to compile sphere shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Unable to compile sphere shader: Unknown exception\n");
        return false;
    }

    if (this->flagsUseSSBO) {
        glGenBuffers(1, &this->flagsBuffer);
    }

    return true;
}



MultiParticleDataCall *SphereRenderer::getData(unsigned int t, float& outScaling) {

    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    outScaling = 1.0f;
    if (c2 != nullptr) {
        c2->SetFrameID(t, this->forceTimeSlot.Param<param::BoolParam>()->Value());
        if (!(*c2)(1)) return nullptr;

        // calculate scaling
        auto const plcount = c2->GetParticleListCount();
        if (this->useLocalBBoxParam.Param<param::BoolParam>()->Value() && plcount > 0) {
            outScaling = c2->AccessParticles(0).GetBBox().LongestEdge();
            for (unsigned pidx = 0; pidx < plcount; ++pidx) {
                auto const temp = c2->AccessParticles(pidx).GetBBox().LongestEdge();
                if (outScaling < temp) {
                    outScaling = temp;
                }
            }
        } else {
            outScaling = c2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        }
        if (outScaling > 0.0000001) {
            outScaling = 10.0f / outScaling;
        } else {
            outScaling = 1.0f;
        }

        c2->SetFrameID(t, this->forceTimeSlot.Param<param::BoolParam>()->Value());
        if (!(*c2)(0)) return nullptr;

        return c2;
    }
    else {
        return nullptr;
    }
}


void SphereRenderer::getClipData(glm::vec4& out_clipDat, glm::vec4& out_clipCol) {
    
    view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<view::CallClipPlane>();
    if ((ccp != nullptr) && (*ccp)()) {
        out_clipDat[0] = ccp->GetPlane().Normal().X();
        out_clipDat[1] = ccp->GetPlane().Normal().Y();
        out_clipDat[2] = ccp->GetPlane().Normal().Z();

        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        out_clipDat[3] = grr.Dot(ccp->GetPlane().Normal());

        out_clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
        out_clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
        out_clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;
        out_clipCol[3] = static_cast<float>(ccp->GetColour()[3]) / 255.0f;
    }
    else {
        out_clipDat[0] = out_clipDat[1] = out_clipDat[2] = out_clipDat[3] = 0.0f;

        out_clipCol[0] = out_clipCol[1] = out_clipCol[2] = 0.75f;
        out_clipCol[3] = 1.0f;
    }
}


bool SphereRenderer::isRenderModeAvailable(RenderMode rm, bool silent) {

    std::string warnstr;

    // Check additonal requirements for each render mode separatly
    switch (rm) {
    case (RenderMode::SIMPLE):
        if (ogl_IsVersionGEQ(1, 4) == 0) {
            warnstr +=
                "[SphereRenderer] Render Mode 'SIMPLE' is not available. OpenGL version 1.4 or greater is required.\n";
        }
        break;
    case (RenderMode::SIMPLE_CLUSTERED):
        if (ogl_IsVersionGEQ(1, 4) == 0) {
            warnstr += "[SphereRenderer] Render Mode 'SIMPLE_CLUSTERED' is not available. OpenGL version 1.4 or "
                       "greater is required.\n";
        }
        break;
    case (RenderMode::GEOMETRY_SHADER):
        if (ogl_IsVersionGEQ(3, 2) == 0) {
            warnstr += "[SphereRenderer] Render Mode 'GEOMETRY_SHADER' is not available. OpenGL version 3.2 or greater "
                       "is required.\n";
        }
        if (!vislib::graphics::gl::GLSLGeometryShader::AreExtensionsAvailable()) {
            warnstr += "[SphereRenderer] Render Mode 'GEOMETRY_SHADER' is not available. Geometry shader extensions "
                       "are required. \n";
        }
        if (!isExtAvailable("GL_EXT_geometry_shader4")) {
            warnstr += "[SphereRenderer] Render Mode 'GEOMETRY_SHADER' is not available. Extension "
                       "GL_EXT_geometry_shader4 is required. \n";
        }
        if (!isExtAvailable("GL_EXT_gpu_shader4")) {
            warnstr += "[SphereRenderer] Render Mode 'GEOMETRY_SHADER' is not available. Extension GL_EXT_gpu_shader4 "
                       "is required. \n";
        }
        if (!isExtAvailable("GL_EXT_bindable_uniform")) {
            warnstr += "[SphereRenderer] Render Mode 'GEOMETRY_SHADER' is not available. Extension "
                       "GL_EXT_bindable_uniform is required. \n";
        }
        if (!isExtAvailable("GL_ARB_shader_objects")) {
            warnstr += "[SphereRenderer] Render Mode 'GEOMETRY_SHADER' is not available. Extension "
                       "GL_ARB_shader_objects is required. \n";
        }
        break;
    case (RenderMode::SSBO_STREAM):
        if (ogl_IsVersionGEQ(4, 2) == 0) {
            warnstr += "[SphereRenderer] Render Mode 'SSBO_STREAM' is not available. OpenGL version 4.2 or greater is "
                       "required. \n";
        }
        if (!isExtAvailable("GL_ARB_shader_storage_buffer_object")) {
            warnstr += "[SphereRenderer] Render Mode 'SSBO_STREAM' is not available. Extension "
                       "GL_ARB_shader_storage_buffer_object is required. \n";
        }
        if (!isExtAvailable("GL_ARB_gpu_shader5")) {
            warnstr += "[SphereRenderer] Render Mode 'SSBO_STREAM' is not available. Extension GL_ARB_gpu_shader5 is "
                       "required. \n";
        }
        if (!isExtAvailable("GL_ARB_gpu_shader_fp64")) {
            warnstr += "[SphereRenderer] Render Mode 'SSBO_STREAM' is not available. Extension GL_ARB_gpu_shader_fp64 "
                       "is required. \n";
        }
        break;
    case (RenderMode::SPLAT):
        if (ogl_IsVersionGEQ(4, 5) == 0) {
            warnstr +=
                "[SphereRenderer] Render Mode 'SPLAT' is not available. OpenGL version 4.5 or greater is required. \n";
        }
        if (!isExtAvailable("GL_ARB_shader_storage_buffer_object")) {
            warnstr += "[SphereRenderer] Render Mode 'SPLAT' is not available. Extension "
                       "GL_ARB_shader_storage_buffer_object is required. \n";
        }
        if (!isExtAvailable("GL_EXT_gpu_shader4")) {
            warnstr +=
                "[SphereRenderer] Render Mode 'SPLAT' is not available. Extension GL_EXT_gpu_shader4 is required. \n";
        }
        break;
    case (RenderMode::BUFFER_ARRAY):
        if (ogl_IsVersionGEQ(4, 5) == 0) {
            warnstr += "[SphereRenderer] Render Mode 'BUFFER_ARRAY' is not available. OpenGL version 4.5 or greater is "
                       "required. \n";
        }
        break;
    case (RenderMode::AMBIENT_OCCLUSION):
        if (ogl_IsVersionGEQ(4, 5) == 0) {
            warnstr += "[SphereRenderer] Render Mode 'AMBIENT_OCCLUSION' is not available. OpenGL version 4.5 or "
                       "greater is required. \n";
        }
        if (!vislib::graphics::gl::GLSLGeometryShader::AreExtensionsAvailable()) {
            warnstr += "[SphereRenderer] Render Mode 'AMBIENT_OCCLUSION' is not available. Geometry shader extensions "
                       "are required. \n";
        }
        if (!isExtAvailable("GL_EXT_geometry_shader4")) {
            warnstr += "[SphereRenderer] Render Mode 'AMBIENT_OCCLUSION' is not available. Extension GL_EXT_geometry_shader4 is required. \n";
        }
        if (!isExtAvailable("GL_ARB_gpu_shader_fp64")) {
            warnstr += "[SphereRenderer] Render Mode 'AMBIENT_OCCLUSION' is not available. Extension "
                       "GL_ARB_gpu_shader_fp64 is required. \n";
        }
        if (!isExtAvailable("GL_ARB_compute_shader")) {
            warnstr += "[SphereRenderer] Render Mode 'AMBIENT_OCCLUSION' is not available. Extension GL_ARB_compute_shader is required. \n";
        }
        break;
    case (RenderMode::OUTLINE):
        if (ogl_IsVersionGEQ(1, 4) == 0) { 
            warnstr += "[SphereRenderer] Render Mode 'OUTLINE' is not available. Minimum OpenGL version is 1.4 \n";
        }
        break;
    default:
        warnstr += "[SphereRenderer] Unknown render mode.\n";
        break;
    }

    if (!silent && !warnstr.empty()) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, warnstr.c_str());
    }

    return (warnstr.empty());
}


std::string SphereRenderer::getRenderModeString(RenderMode rm) {

    std::string mode;

    switch (rm) {
    case (RenderMode::SIMPLE):
        mode = "SIMPLE";
        break;
    case (RenderMode::SIMPLE_CLUSTERED):
        mode = "SIMPLE CLUSTERED";
        break;
    case (RenderMode::GEOMETRY_SHADER):
        mode = "SIMPLE GEOMETRY SHADER";
        break;
    case (RenderMode::SSBO_STREAM):
        mode = "SSBO STREAM";
        break;
    case (RenderMode::SPLAT):
        mode = "SPLAT";
        break;
    case (RenderMode::BUFFER_ARRAY):
        mode = "BUFFER ARRAY";
        break;
    case (RenderMode::AMBIENT_OCCLUSION):
        mode = "AMBIENT OCCLUSION";
        break;
    case (RenderMode::OUTLINE):
        mode = "OUTLINE";
        break;
    default:
        mode = "unknown";
        break;
    }

    return mode;
}



bool SphereRenderer::Render(view::CallRender3D_2& call) {

    // timer.BeginFrame();

    auto cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
    auto flagc = this->getFlagsSlot.CallAs<FlagCall>();

    // Checking for changed render mode
    auto currentRenderMode = static_cast<RenderMode>(this->renderModeParam.Param<param::EnumParam>()->Value());
    if (currentRenderMode != this->renderMode) {
        this->renderMode = currentRenderMode;
        if (!this->createResources()) {
            return false;
        }
    }

    // Get data
    float scaling = 1.0f;
    MultiParticleDataCall* mpdc = this->getData(static_cast<unsigned int>(call.Time()), scaling);
    if (mpdc == nullptr) return false;

    // Update parameter visibility
    this->colIdxRangeInfoParam.Param<param::Vector2fParam>()->SetGUIVisible((bool)(cgtf != nullptr));
    this->selectColorParam.Param<param::ColorParam>()->SetGUIVisible((bool)(flagc != nullptr));
    this->softSelectColorParam.Param<param::ColorParam>()->SetGUIVisible((bool)(flagc != nullptr));

    // Update current state variables -----------------------------------------

    // Check if we got a new data set
    const SIZE_T hash = mpdc->DataHash();
    const unsigned int frameID = mpdc->FrameID();
    this->stateInvalid = ((hash != this->oldHash) || (frameID != this->oldFrameID));

    // Update read only parameter values of color index range to be set manually in  transfer function
    if (this->stateInvalid) {
        std::array<float, 2> range;
        range[0] = std::numeric_limits<float>::max(); // min
        range[1] = std::numeric_limits<float>::min(); // max
        for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
            MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);
            range[0] = std::min(parts.GetMinColourIndexValue(), range[0]);
            range[1] = std::max(parts.GetMaxColourIndexValue(), range[1]);
        }
        this->colIdxRangeInfoParam.Param<param::Vector2fParam>()->SetValue(
            vislib::math::Vector<float, 2>(range[0], range[1]));
    }

    this->oldHash = hash;
    this->oldFrameID = frameID;

    // Clipping
    this->getClipData(this->curClipDat, this->curClipCol);

    // Camera 
    view::Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;
    // Generate complete snapshot and calculate matrices
    //auto snap_content = thecam::snapshot_content::camera_coordinate_system; /// Will be overwritten in camera.inl calc_matrices() line 216ff ...
    cam.calc_matrices(snapshot, viewTemp, projTemp, thecam::snapshot_content::all);
    this->curCamPos = snapshot.position;
    this->curCamView = snapshot.view_vector;
    this->curCamRight = snapshot.right_vector;
    this->curCamUp = snapshot.up_vector;
    this->curCamNearClip = snapshot.frustum_near;

    glm::mat4 view = viewTemp;
    glm::mat4 proj = projTemp;
    this->curMVinv = glm::inverse(view);
    this->curMVtransp = glm::transpose(view);
    this->curMVP = proj * view;
    this->curMVPinv = glm::inverse(this->curMVP);
    this->curMVPtransp = glm::transpose(this->curMVP);

    // Lights
    this->GetLights();
    this->curlightDir = { 0.0f, 0.0f, 0.0f, 1.0f };
    if (this->lightMap.size() > 1) {
        vislib::sys::Log::DefaultLog.WriteWarn("[SphereRenderer] Only one single 'Distant Light' source is supported by this renderer");
    }
    for (auto light : this->lightMap) {
        if (light.second.lightType != core::view::light::DISTANTLIGHT) {
            vislib::sys::Log::DefaultLog.WriteWarn("[SphereRenderer] Only single 'Distant Light' source is supported by this renderer");
        }
        else {
            auto use_eyedir = light.second.dl_eye_direction;
            if (use_eyedir) {
                this->curlightDir = -this->curCamView;
            }
            else {
                auto lightDir = light.second.dl_direction;
                if (lightDir.size() == 3) {
                    this->curlightDir[0] = lightDir[0];
                    this->curlightDir[1] = lightDir[1];
                    this->curlightDir[2] = lightDir[2];
                }
                if (lightDir.size() == 4) {
                    this->curlightDir[3] = lightDir[3];
                }
                /// View Space Lighting. Comment line to change to Object Space Lighting.
                //this->curlightDir = this->curMVtransp * this->curlightDir;
            }
/// TODO Implement missing distant light parameters:
            //light.second.dl_angularDiameter;
            //light.second.lightColor;
            //light.second.lightIntensity;
        }
    }

    // Viewport
    auto viewport = call.GetViewport();
    this->curVpWidth = viewport.Width();
    this->curVpHeight = viewport.Height();
    this->curViewAttrib[0] = 0.0f;
    this->curViewAttrib[1] = 0.0f;
    this->curViewAttrib[2] = static_cast<float>(viewport.Width());
    this->curViewAttrib[3] = static_cast<float>(viewport.Height());
    if (this->curViewAttrib[2] < 1.0f) this->curViewAttrib[2] = 1.0f;
    if (this->curViewAttrib[3] < 1.0f) this->curViewAttrib[3] = 1.0f;
    this->curViewAttrib[2] = 2.0f / this->curViewAttrib[2];
    this->curViewAttrib[3] = 2.0f / this->curViewAttrib[3];

    // ------------------------------------------------------------------------

    // Set OpenGL state
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS); /// Necessary for early depth test in fragment shader (default)
    glEnable(GL_CLIP_DISTANCE0);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    bool retval = false;
    switch (currentRenderMode) {
    case (RenderMode::SIMPLE):
        retval = this->renderSimple(call, mpdc);
        break;
    case (RenderMode::SIMPLE_CLUSTERED):
        retval = this->renderSimple(call, mpdc);
        break;
    case (RenderMode::GEOMETRY_SHADER):
        retval = this->renderGeometryShader(call, mpdc);
        break;
    case (RenderMode::SSBO_STREAM):
        retval = this->renderSSBO(call, mpdc);
        break;
    case (RenderMode::SPLAT):
        retval = this->renderSplat(call, mpdc);
        break;
    case (RenderMode::BUFFER_ARRAY):
        retval = this->renderBufferArray(call, mpdc);
        break;
    case (RenderMode::AMBIENT_OCCLUSION):
        retval = this->renderAmbientOcclusion(call, mpdc); break;
    case (RenderMode::OUTLINE):
        retval = this->renderOutline(call, mpdc); break;
    default:
        break;
    }

    // Reset OpenGl state
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CLIP_DISTANCE0);

    // Save some current data
    this->lastVpHeight = this->curVpHeight;
    this->lastVpWidth = this->curVpWidth;
    for (size_t i = 0; i < 4; ++i) {
        this->oldClipDat[i] = this->curClipDat[i];
    }

    // timer.EndFrame();

    return retval;
}


bool SphereRenderer::renderSimple(view::CallRender3D_2& call, MultiParticleDataCall* mpdc) {

    this->sphereShader.Enable();

    this->setFlagStorage(this->sphereShader, mpdc);

    glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, glm::value_ptr(this->curViewAttrib));
    glUniform3fv(this->sphereShader.ParameterLocation("camIn"), 1, glm::value_ptr(this->curCamView));
    glUniform3fv(this->sphereShader.ParameterLocation("camRight"), 1, glm::value_ptr(this->curCamRight));
    glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, glm::value_ptr(this->curCamUp));
    glUniform1f(this->sphereShader.ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphereShader.ParameterLocation("clipDat"), 1, glm::value_ptr(this->curClipDat));
    glUniform4fv(this->sphereShader.ParameterLocation("clipCol"), 1, glm::value_ptr(this->curClipCol));
    glUniform4fv(this->sphereShader.ParameterLocation("lightDir"), 1, glm::value_ptr(this->curlightDir));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->curMVinv));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVtransp));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->curMVP));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->curMVPinv));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVPtransp));

    GLuint flagPartsCount = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (!this->setShaderData(this->sphereShader, parts)) {
            continue;
        }

        glUniform1ui(this->sphereShader.ParameterLocation("flagsAvailable"), GLuint(this->flagsEnabled));
        if (this->flagsEnabled) {
            glUniform1ui(this->sphereShader.ParameterLocation("flagOffset"), flagPartsCount);
            glUniform4fv(this->sphereShader.ParameterLocation("flagSelectedCol"), 1,
                this->selectColorParam.Param<param::ColorParam>()->Value().data());
            glUniform4fv(this->sphereShader.ParameterLocation("flagSoftSelectedCol"), 1,
                this->softSelectColorParam.Param<param::ColorParam>()->Value().data());
        }

        GLuint vao, vb, cb;
        if (this->renderMode == RenderMode::SIMPLE_CLUSTERED) {
            parts.GetVAOs(vao, vb, cb);
            if (parts.IsVAO()) {
                glBindVertexArray(vao);
                this->setBufferData(
                    this->sphereShader, parts, vb, parts.GetVertexData(), cb, parts.GetColourData(), true); // or false?
            }
        }
        if ((this->renderMode == RenderMode::SIMPLE) || (!parts.IsVAO())) {
            this->setBufferData(this->sphereShader, parts, 0, parts.GetVertexData(), 0, parts.GetColourData());
        }

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

        if (this->renderMode == RenderMode::SIMPLE_CLUSTERED) {
            if (parts.IsVAO()) {
                glBindVertexArray(0);
            }
        }
        this->unsetBufferData(this->sphereShader);
        this->unsetShaderData();
        flagPartsCount += parts.GetCount();
    }

    this->unsetFlagStorage(this->sphereShader);
    this->sphereShader.Disable();

    mpdc->Unlock();

    return true;
}


bool SphereRenderer::renderSSBO(view::CallRender3D_2& call, MultiParticleDataCall* mpdc) {

#ifdef CHRONOTIMING
    std::vector<std::chrono::steady_clock::time_point> deltas;
    std::chrono::steady_clock::time_point before, after;
#endif

    // this->currBuf = 0;
    GLuint flagPartsCount = 0;
    if (this->stateInvalid) {
        this->bufArray.resize(mpdc->GetParticleListCount());
        this->colBufArray.resize(mpdc->GetParticleListCount());
    }
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (colType != parts.GetColourDataType() || vertType != parts.GetVertexDataType()) {
            this->newShader = this->generateShader(parts);
        }

        this->newShader->Enable();

        if (!this->setShaderData(*this->newShader, parts)) {
            continue;
        }

        this->setFlagStorage(*this->newShader, mpdc);
        glUniform1ui(this->newShader->ParameterLocation("flagsAvailable"), GLuint(this->flagsEnabled));
        if (this->flagsEnabled) {
            glUniform1ui(this->newShader->ParameterLocation("flagOffset"), flagPartsCount);
            glUniform4fv(this->newShader->ParameterLocation("flagSelectedCol"), 1,
                this->selectColorParam.Param<param::ColorParam>()->Value().data());
            glUniform4fv(this->newShader->ParameterLocation("flagSoftSelectedCol"), 1,
                this->softSelectColorParam.Param<param::ColorParam>()->Value().data());
        }

        glUniform4fv(this->newShader->ParameterLocation("viewAttr"), 1, glm::value_ptr(this->curViewAttrib));
        glUniform3fv(this->newShader->ParameterLocation("camIn"), 1, glm::value_ptr(this->curCamView));
        glUniform3fv(this->newShader->ParameterLocation("camRight"), 1, glm::value_ptr(this->curCamRight));
        glUniform3fv(this->newShader->ParameterLocation("camUp"), 1, glm::value_ptr(this->curCamUp));
        glUniform1f(this->newShader->ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
        glUniform4fv(this->newShader->ParameterLocation("clipDat"), 1, glm::value_ptr(this->curClipDat));
        glUniform4fv(this->newShader->ParameterLocation("clipCol"), 1, glm::value_ptr(this->curClipCol));
        glUniform4fv(this->newShader->ParameterLocation("lightDir"), 1, glm::value_ptr(this->curlightDir));
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->curMVinv));
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVtransp));
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->curMVP));
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->curMVPinv));
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVPtransp));

        unsigned int colBytes, vertBytes, colStride, vertStride;
        bool interleaved;
        const bool staticData = this->useStaticDataParam.Param<param::BoolParam>()->Value();
        this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride, interleaved);

        // does all data reside interleaved in the same memory?
        if (interleaved) {
            if (staticData) {
                auto& bufA = this->bufArray[i];
                if (this->stateInvalid || (bufA.GetNumChunks() == 0)) {
                    bufA.SetDataWithSize(parts.GetVertexData(), vertStride, vertStride, parts.GetCount(),
                        (GLuint)(2 * 1024 * 1024 * 1024 - 1));
                    // 2 GB - khronos: Most implementations will let you allocate a size up to the limit of GPU memory.
                }
                const GLuint numChunks = bufA.GetNumChunks();

                for (GLuint x = 0; x < numChunks; ++x) {
                    glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);
                    auto actualItems = bufA.GetNumItems(x);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufA.GetHandle(x));
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOvertexBindingPoint, bufA.GetHandle(x));
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBOvertexBindingPoint, bufA.GetHandle(x), 0,
                        bufA.GetMaxNumItemsPerChunk() * vertStride);
                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(actualItems));
                    //bufA.SignalCompletion();
                }
            } else {
                const GLuint numChunks = this->streamer.SetDataWithSize(
                    parts.GetVertexData(), vertStride, vertStride, parts.GetCount(), 3, (GLuint)(32 * 1024 * 1024));
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->streamer.GetHandle());
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOvertexBindingPoint, this->streamer.GetHandle());

                for (GLuint x = 0; x < numChunks; ++x) {
                    GLuint numItems, sync;
                    GLsizeiptr dstOff, dstLen;
                    this->streamer.UploadChunk(x, numItems, sync, dstOff, dstLen);
                    // streamer.UploadChunk<float, float>(x, [](float f) -> float { return f + 100.0; },
                    //    numItems, sync, dstOff, dstLen);
                    // vislib::sys::Log::DefaultLog.WriteInfo("[SphereRenderer] Uploading chunk %u at %lu len %lu", x,
                    // dstOff, dstLen);
                    glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    glBindBufferRange(
                        GL_SHADER_STORAGE_BUFFER, SSBOvertexBindingPoint, this->streamer.GetHandle(), dstOff, dstLen);
                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(numItems));
                    this->streamer.SignalCompletion(sync);
                }
            }
        } else {
            if (staticData) {
                auto& bufA = this->bufArray[i];
                auto& colA = this->colBufArray[i];
                if (this->stateInvalid || (bufA.GetNumChunks() == 0)) {
                    bufA.SetDataWithSize(parts.GetVertexData(), vertStride, vertStride, parts.GetCount(),
                        (GLuint)(2 * 1024 * 1024 * 1024));
                    // 2 GB - khronos: Most implementations will let you allocate a size up to the limit of GPU memory.
                    colA.SetDataWithItems(parts.GetColourData(), colStride, colStride, parts.GetCount(),
                        bufA.GetMaxNumItemsPerChunk());
                }
                const GLuint numChunks = bufA.GetNumChunks();

                for (GLuint x = 0; x < numChunks; ++x) {
                    glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);
                    auto actualItems = bufA.GetNumItems(x);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufA.GetHandle(x));
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOvertexBindingPoint, bufA.GetHandle(x));
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBOvertexBindingPoint, bufA.GetHandle(x), 0,
                        bufA.GetMaxNumItemsPerChunk() * vertStride);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, colA.GetHandle(x));
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOcolorBindingPoint, colA.GetHandle(x));
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBOcolorBindingPoint, colA.GetHandle(x),
                        0, colA.GetMaxNumItemsPerChunk() * colStride);
                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(actualItems));
                    //bufA.SignalCompletion();
                    //colA.SignalCompletion();
                }
            } else {
                const GLuint numChunks = this->streamer.SetDataWithSize(
                    parts.GetVertexData(), vertStride, vertStride, parts.GetCount(), 3, (GLuint)(32 * 1024 * 1024));
                const GLuint colSize = this->colStreamer.SetDataWithItems(parts.GetColourData(), colStride, colStride,
                    parts.GetCount(), 3, this->streamer.GetMaxNumItemsPerChunk());
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->streamer.GetHandle());
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOvertexBindingPoint, this->streamer.GetHandle());
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->colStreamer.GetHandle());
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOcolorBindingPoint, this->colStreamer.GetHandle());

                for (GLuint x = 0; x < numChunks; ++x) {
                    GLuint numItems, numItems2, sync, sync2;
                    GLsizeiptr dstOff, dstLen, dstOff2, dstLen2;
                    this->streamer.UploadChunk(x, numItems, sync, dstOff, dstLen);
                    this->colStreamer.UploadChunk(x, numItems2, sync2, dstOff2, dstLen2);
                    ASSERT(numItems == numItems2);
                    glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    glBindBufferRange(
                        GL_SHADER_STORAGE_BUFFER, SSBOvertexBindingPoint, this->streamer.GetHandle(), dstOff, dstLen);
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBOcolorBindingPoint, this->colStreamer.GetHandle(),
                        dstOff2, dstLen2);
                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(numItems));
                    this->streamer.SignalCompletion(sync);
                    this->colStreamer.SignalCompletion(sync2);
                }
            }
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        this->unsetShaderData();
        this->unsetFlagStorage(*this->newShader);
        flagPartsCount += parts.GetCount();
        this->newShader->Disable();

#ifdef CHRONOTIMING
        printf("waitSignal times:\n");
        for (auto d : deltas) {
            printf("%u, ", d);
        }
        printf("\n");
#endif
    }

    mpdc->Unlock();

    return true;
}


bool SphereRenderer::renderSplat(view::CallRender3D_2& call, MultiParticleDataCall* mpdc) {

    glDisable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);

#if 1
    // Should be default for splat rendering (Hint: Background colour should not be WHITE)
    glBlendFunc(GL_ONE, GL_ONE);
#else
    // Maybe for blending against white, remove pre-mult alpha and use this:
    // @gl.blendFuncSeparate @gl.SRC_ALPHA, @gl.ONE_MINUS_SRC_ALPHA, @gl.ONE, @gl.ONE_MINUS_SRC_ALPHA
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    // glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
#endif

    glEnable(GL_POINT_SPRITE);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSingleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOvertexBindingPoint, this->theSingleBuffer);

    // this->currBuf = 0;
    GLuint flagPartsCount = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (colType != parts.GetColourDataType() || vertType != parts.GetVertexDataType()) {
            this->newShader = this->generateShader(parts);
        }

        this->newShader->Enable();

        if (!this->setShaderData(*this->newShader, parts)) {
            continue;
        }

        this->setFlagStorage(*this->newShader, mpdc);
        glUniform1ui(this->newShader->ParameterLocation("flagsAvailable"), GLuint(this->flagsEnabled));
        if (this->flagsEnabled) {
            glUniform1ui(this->newShader->ParameterLocation("flagOffset"), flagPartsCount);
            glUniform4fv(this->newShader->ParameterLocation("flagSelectedCol"), 1,
                this->selectColorParam.Param<param::ColorParam>()->Value().data());
            glUniform4fv(this->newShader->ParameterLocation("flagSoftSelectedCol"), 1,
                this->softSelectColorParam.Param<param::ColorParam>()->Value().data());
        }
        glUniform4fv(this->newShader->ParameterLocation("viewAttr"), 1, glm::value_ptr(this->curViewAttrib));
        glUniform3fv(this->newShader->ParameterLocation("camIn"), 1, glm::value_ptr(this->curCamView));
        glUniform3fv(this->newShader->ParameterLocation("camRight"), 1, glm::value_ptr(this->curCamRight));
        glUniform3fv(this->newShader->ParameterLocation("camUp"), 1, glm::value_ptr(this->curCamUp));
        glUniform1f(this->newShader->ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
        glUniform4fv(this->newShader->ParameterLocation("clipDat"), 1, glm::value_ptr(this->curClipDat));
        glUniform4fv(this->newShader->ParameterLocation("clipCol"), 1, glm::value_ptr(this->curClipCol));
        glUniform4fv(this->newShader->ParameterLocation("lightDir"), 1, glm::value_ptr(this->curlightDir));
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->curMVinv));
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVtransp));
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->curMVP));
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->curMVPinv));
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVPtransp));
        glUniform1f(this->newShader->ParameterLocation("alphaScaling"), this->alphaScalingParam.Param<param::FloatParam>()->Value());
        glUniform1i(this->newShader->ParameterLocation("attenuateSubpixel"), this->attenuateSubpixelParam.Param<param::BoolParam>()->Value() ? 1 : 0);

        unsigned int colBytes, vertBytes, colStride, vertStride;
        bool interleaved;
        this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride, interleaved);

        // currBuf = 0;
        UINT64 numVerts, vertCounter;
        // does all data reside interleaved in the same memory?
        if (interleaved) {

            numVerts = this->bufSize / vertStride;
            const char* currVert = static_cast<const char*>(parts.GetVertexData());
            const char* currCol = static_cast<const char*>(parts.GetColourData());
            vertCounter = 0;
            while (vertCounter < parts.GetCount()) {
                // GLuint vb = this->theBuffers[currBuf];
                void* mem = static_cast<char*>(theSingleMappedMem) + bufSize * this->currBuf;
                currCol = colStride == 0 ? currVert : currCol;
                // currCol = currCol == 0 ? currVert : currCol;
                const char* whence = currVert < currCol ? currVert : currCol;
                UINT64 vertsThisTime = vislib::math::Min(parts.GetCount() - vertCounter, numVerts);
                this->waitSingle(this->fences[this->currBuf]);
                // vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Memcopying %u
                // bytes from %016" PRIxPTR " to %016" PRIxPTR "\n", vertsThisTime * vertStride, whence, mem);
                memcpy(mem, whence, vertsThisTime * vertStride);
                glFlushMappedNamedBufferRange(
                    this->theSingleBuffer, bufSize * this->currBuf, vertsThisTime * vertStride);
                // glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
                // glUniform1i(this->newShader->ParameterLocation("instanceOffset"), numVerts * currBuf);
                glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);

                // this->theSingleBuffer, reinterpret_cast<const void *>(currCol - whence));
                // glBindBuffer(GL_ARRAY_BUFFER, 0);
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBOvertexBindingPoint, this->theSingleBuffer,
                    bufSize * this->currBuf, bufSize);
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vertsThisTime));
                // glDrawArraysInstanced(GL_POINTS, 0, 1, vertsThisTime);
                this->lockSingle(this->fences[this->currBuf]);

                this->currBuf = (this->currBuf + 1) % this->numBuffers;
                vertCounter += vertsThisTime;
                currVert += vertsThisTime * vertStride;
                currCol += vertsThisTime * colStride;
            }
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                "[SphereRenderer] Splat mode does not support not interleaved data so far ...");
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        this->unsetShaderData();
        this->unsetFlagStorage(*this->newShader);
        flagPartsCount += parts.GetCount();
        newShader->Disable();
    }

    glDisable(GL_POINT_SPRITE);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    mpdc->Unlock();

    return true;
}


bool SphereRenderer::renderBufferArray(view::CallRender3D_2& call, MultiParticleDataCall* mpdc) {

    this->sphereShader.Enable();

    this->setFlagStorage(this->sphereShader, mpdc);

    glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, glm::value_ptr(this->curViewAttrib));
    glUniform3fv(this->sphereShader.ParameterLocation("camIn"), 1, glm::value_ptr(this->curCamView));
    glUniform3fv(this->sphereShader.ParameterLocation("camRight"), 1, glm::value_ptr(this->curCamRight));
    glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, glm::value_ptr(this->curCamUp));
    glUniform1f(this->sphereShader.ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphereShader.ParameterLocation("clipDat"), 1, glm::value_ptr(this->curClipDat));
    glUniform4fv(this->sphereShader.ParameterLocation("clipCol"), 1, glm::value_ptr(this->curClipCol));
    glUniform4fv(this->sphereShader.ParameterLocation("lightDir"), 1, glm::value_ptr(this->curlightDir));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->curMVinv));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVtransp));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->curMVP));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->curMVPinv));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVPtransp));

    //this->currBuf = 0;
    GLuint flagPartsCount = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (!this->setShaderData(this->sphereShader, parts)) {
            continue;
        }

        glUniform1ui(this->sphereShader.ParameterLocation("flagsAvailable"), GLuint(this->flagsEnabled));
        if (this->flagsEnabled) {
            glUniform4fv(this->sphereShader.ParameterLocation("flagSelectedCol"), 1, this->selectColorParam.Param<param::ColorParam>()->Value().data());
            glUniform4fv(this->sphereShader.ParameterLocation("flagSoftSelectedCol"), 1, this->softSelectColorParam.Param<param::ColorParam>()->Value().data());
        }

        unsigned int colBytes, vertBytes, colStride, vertStride;
        bool interleaved;
        this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride, interleaved);

        UINT64 numVerts, vertCounter;
        // does all data reside interleaved in the same memory?
        if (interleaved) {

            numVerts = this->bufSize / vertStride;
            const char* currVert = static_cast<const char*>(parts.GetVertexData());
            const char* currCol = static_cast<const char*>(parts.GetColourData());
            vertCounter = 0;
            while (vertCounter < parts.GetCount()) {
                // GLuint vb = this->theBuffers[currBuf];
                void* mem = static_cast<char*>(this->theSingleMappedMem) + numVerts * vertStride * this->currBuf;
                currCol = colStride == 0 ? currVert : currCol;
                // currCol = currCol == 0 ? currVert : currCol;
                const char* whence = currVert < currCol ? currVert : currCol;
                UINT64 vertsThisTime = vislib::math::Min(parts.GetCount() - vertCounter, numVerts);
                this->waitSingle(this->fences[this->currBuf]);
                // vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Memcopying %u
                // bytes from %016" PRIxPTR " to %016" PRIxPTR "\n", vertsThisTime * vertStride, whence, mem);
                memcpy(mem, whence, vertsThisTime * vertStride);
                glFlushMappedNamedBufferRange(
                    this->theSingleBuffer, numVerts * this->currBuf, vertsThisTime * vertStride);
                // glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);

                if (this->flagsEnabled) {
                    // Adapting flag offset to ring buffer gl_VertexID
                    glUniform1ui(this->sphereShader.ParameterLocation("flagOffset"),
                        flagPartsCount - static_cast<GLuint>(numVerts * this->currBuf));
                }
                this->setBufferData(this->sphereShader, parts, this->theSingleBuffer,
                    reinterpret_cast<const void*>(currVert - whence), this->theSingleBuffer,
                    reinterpret_cast<const void*>(currCol - whence));

                glDrawArrays(
                    GL_POINTS, static_cast<GLint>(numVerts * this->currBuf), static_cast<GLsizei>(vertsThisTime));
                this->lockSingle(this->fences[this->currBuf]);

                this->currBuf = (this->currBuf + 1) % this->numBuffers;
                vertCounter += vertsThisTime;
                currVert += vertsThisTime * vertStride;
                currCol += vertsThisTime * colStride;
            }
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                "[SphereRenderer] BufferArray mode does not support not interleaved data so far ...");
        }

        this->unsetBufferData(this->sphereShader);
        this->unsetShaderData();
        flagPartsCount += parts.GetCount();
    }

    this->unsetFlagStorage(this->sphereShader);
    this->sphereShader.Disable();

    mpdc->Unlock();

    return true;
}


bool SphereRenderer::renderGeometryShader(view::CallRender3D_2& call, MultiParticleDataCall* mpdc) {

    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    /// Default GL_LESS works, too?
    // glDepthFunc(GL_LEQUAL);

    /// If enabled and a vertex shader is active, it specifies that the GL will choose between front and
    /// back colors based on the polygon's face direction of which the vertex being shaded is a part.
    /// It has no effect on points or lines and has significant negative performance impact.
    // glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);

    this->sphereGeometryShader.Enable();

    this->setFlagStorage(this->sphereGeometryShader, mpdc);

    // Set shader variables
    glUniform4fv(this->sphereGeometryShader.ParameterLocation("viewAttr"), 1, glm::value_ptr(this->curViewAttrib));
    glUniform3fv(this->sphereGeometryShader.ParameterLocation("camIn"), 1, glm::value_ptr(this->curCamView));
    glUniform3fv(this->sphereGeometryShader.ParameterLocation("camRight"), 1, glm::value_ptr(this->curCamRight));
    glUniform3fv(this->sphereGeometryShader.ParameterLocation("camUp"), 1, glm::value_ptr(this->curCamUp));
    glUniform1f(this->sphereGeometryShader.ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphereGeometryShader.ParameterLocation("clipDat"), 1, glm::value_ptr(this->curClipDat));
    glUniform4fv(this->sphereGeometryShader.ParameterLocation("clipCol"), 1, glm::value_ptr(this->curClipCol));
    glUniform4fv(this->sphereGeometryShader.ParameterLocation("lightDir"), 1, glm::value_ptr(this->curlightDir));
    glUniformMatrix4fv(this->sphereGeometryShader.ParameterLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->curMVinv));
    glUniformMatrix4fv(this->sphereGeometryShader.ParameterLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVtransp));
    glUniformMatrix4fv(this->sphereGeometryShader.ParameterLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->curMVP));
    glUniformMatrix4fv(this->sphereGeometryShader.ParameterLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->curMVPinv));
    glUniformMatrix4fv(this->sphereGeometryShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVPtransp));

    GLuint flagPartsCount = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (!this->setShaderData(this->sphereGeometryShader, parts)) {
            continue;
        }

        glUniform1ui(this->sphereGeometryShader.ParameterLocation("flagsAvailable"), GLuint(this->flagsEnabled));
        if (this->flagsEnabled) {
            glUniform1ui(this->sphereGeometryShader.ParameterLocation("flagOffset"), flagPartsCount);
            glUniform4fv(this->sphereGeometryShader.ParameterLocation("flagSelectedCol"), 1,
                this->selectColorParam.Param<param::ColorParam>()->Value().data());
            glUniform4fv(this->sphereGeometryShader.ParameterLocation("flagSoftSelectedCol"), 1,
                this->softSelectColorParam.Param<param::ColorParam>()->Value().data());
        }

        this->setBufferData(this->sphereGeometryShader, parts, 0, parts.GetVertexData(), 0, parts.GetColourData());

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

        this->unsetBufferData(this->sphereGeometryShader);
        this->unsetShaderData();
        flagPartsCount += parts.GetCount();
    }

    this->unsetFlagStorage(this->sphereGeometryShader);
    this->sphereGeometryShader.Disable();

    // glDisable(GL_VERTEX_PROGRAM_TWO_SIDE);
    // glDepthFunc(GL_LESS);

    mpdc->Unlock();

    return true;
}


bool SphereRenderer::renderAmbientOcclusion(view::CallRender3D_2& call, MultiParticleDataCall* mpdc) {

    // We need to regenerate the shader if certain settings are changed
    if (this->enableLightingSlot.IsDirty() || this->aoConeApexSlot.IsDirty()) {
        this->aoConeApexSlot.ResetDirty();
        this->enableLightingSlot.ResetDirty();

        this->createResources();
    }

    // Rebuild the GBuffer if neccessary
    this->rebuildGBuffer();

    // Render the particles' geometry
    bool highPrecision = this->useHPTexturesSlot.Param<param::BoolParam>()->Value();

    // Choose shader
    bool useGeo = this->enableGeometryShader.Param<param::BoolParam>()->Value();
    vislib::graphics::gl::GLSLShader& theShader = useGeo ? this->sphereGeometryShader : this->sphereShader;

    // Rebuild and reupload working data if neccessary
    this->rebuildWorkingData(call, mpdc, theShader);

    GLint prevFBO;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, this->gBuffer.fbo);
    GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindFragDataLocation(theShader.ProgramHandle(), 0, "outColor");
    glBindFragDataLocation(theShader.ProgramHandle(), 1, "outNormal");

    theShader.Enable();

    this->setFlagStorage(theShader, mpdc);

    glUniformMatrix4fv(theShader.ParameterLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->curMVP));
    glUniformMatrix4fv(theShader.ParameterLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->curMVPinv));
    glUniformMatrix4fv(theShader.ParameterLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->curMVinv));
    glUniformMatrix4fv(theShader.ParameterLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVtransp));
    glUniformMatrix4fv(theShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVPtransp));
    glUniform1f(theShader.ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
    glUniform4fv(theShader.ParameterLocation("viewAttr"), 1, glm::value_ptr(this->curViewAttrib));
    glUniform3fv(theShader.ParameterLocation("camRight"), 1, glm::value_ptr(this->curCamRight));
    glUniform3fv(theShader.ParameterLocation("camUp"), 1, glm::value_ptr(this->curCamUp));
    glUniform3fv(theShader.ParameterLocation("camIn"), 1, glm::value_ptr(this->curCamView));
    glUniform4fv(theShader.ParameterLocation("clipDat"), 1, glm::value_ptr(this->curClipDat));
    glUniform4fv(theShader.ParameterLocation("clipCol"), 1, glm::value_ptr(this->curClipCol));
    glUniform1i(theShader.ParameterLocation("inUseHighPrecision"), (int)highPrecision);

    GLuint flagPartsCount = 0;
    for (unsigned int i = 0; i < this->gpuData.size(); ++i) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (!this->setShaderData(theShader, parts)) {
            continue;
        }

        glUniform1ui(theShader.ParameterLocation("flagsAvailable"), GLuint(this->flagsEnabled));
        if (this->flagsEnabled) {
            glUniform1ui(theShader.ParameterLocation("flagOffset"), flagPartsCount);
            glUniform4fv(theShader.ParameterLocation("flagSelectedCol"), 1,
                this->selectColorParam.Param<param::ColorParam>()->Value().data());
            glUniform4fv(theShader.ParameterLocation("flagSoftSelectedCol"), 1,
                this->softSelectColorParam.Param<param::ColorParam>()->Value().data());
        }

        glBindVertexArray(this->gpuData[i].vertexArray);

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(mpdc->AccessParticles(i).GetCount()));

        this->unsetShaderData();
        flagPartsCount += parts.GetCount();
    }

    glBindVertexArray(0);

    this->unsetFlagStorage(theShader);
    theShader.Disable();

    glBindFramebuffer(GL_FRAMEBUFFER, prevFBO);

    // Deferred rendering pass
    this->renderDeferredPass(call);

    return true;
}


bool SphereRenderer::renderOutline(view::CallRender3D_2& call, MultiParticleDataCall* mpdc) {

    this->sphereShader.Enable();

    this->setFlagStorage(this->sphereShader, mpdc);

    glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, glm::value_ptr(this->curViewAttrib));
    glUniform3fv(this->sphereShader.ParameterLocation("camIn"), 1, glm::value_ptr(this->curCamView));
    glUniform3fv(this->sphereShader.ParameterLocation("camRight"), 1, glm::value_ptr(this->curCamRight));
    glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, glm::value_ptr(this->curCamUp));
    glUniform1f(this->sphereShader.ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphereShader.ParameterLocation("clipDat"), 1, glm::value_ptr(this->curClipDat));
    glUniform4fv(this->sphereShader.ParameterLocation("clipCol"), 1, glm::value_ptr(this->curClipCol));
    glUniform4fv(this->sphereShader.ParameterLocation("lightDir"), 1, glm::value_ptr(this->curlightDir));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->curMVinv));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVtransp));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->curMVP));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->curMVPinv));
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->curMVPtransp));

    glUniform1f(this->sphereShader.ParameterLocation("outlineSize"), this->outlineSizeSlot.Param<param::FloatParam>()->Value());

    GLuint flagPartsCount = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (!this->setShaderData(this->sphereShader, parts)) {
            continue;
        }

        glUniform1ui(this->sphereShader.ParameterLocation("flagsAvailable"), GLuint(this->flagsEnabled));
        if (this->flagsEnabled) {
            glUniform1ui(this->sphereShader.ParameterLocation("flagOffset"), flagPartsCount);
            glUniform4fv(this->sphereShader.ParameterLocation("flagSelectedCol"), 1, this->selectColorParam.Param<param::ColorParam>()->Value().data());
            glUniform4fv(this->sphereShader.ParameterLocation("flagSoftSelectedCol"), 1, this->softSelectColorParam.Param<param::ColorParam>()->Value().data());
        }

        this->setBufferData(this->sphereShader, parts, 0, parts.GetVertexData(), 0, parts.GetColourData());
    
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

        this->unsetBufferData(this->sphereShader);
        this->unsetShaderData();
        flagPartsCount += parts.GetCount();
    }

    this->unsetFlagStorage(this->sphereShader);
    this->sphereShader.Disable();

    mpdc->Unlock();

    return true;
}


bool SphereRenderer::setBufferData(const vislib::graphics::gl::GLSLShader& shader, const MultiParticleDataCall::Particles &parts,
    GLuint vertBuf, const void *vertPtr, GLuint colBuf, const void *colPtr, bool createBufferData) {

    GLuint vertAttribLoc = glGetAttribLocation(shader, "inPosition");
    GLuint colAttribLoc = glGetAttribLocation(shader, "inColor");
    GLuint colIdxAttribLoc = glGetAttribLocation(shader, "inColIdx");

    const void* colorPtr = colPtr;
    const void* vertexPtr = vertPtr;
    if (createBufferData) {
        colorPtr = nullptr;
        vertexPtr = nullptr;
    }

    unsigned int partCount = static_cast<unsigned int>(parts.GetCount());

    // colour
    glBindBuffer(GL_ARRAY_BUFFER, colBuf);
    switch (parts.GetColourDataType()) {
    case MultiParticleDataCall::Particles::COLDATA_NONE:
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
        if (createBufferData) {
            glBufferData(GL_ARRAY_BUFFER, partCount * (std::max)(parts.GetColourDataStride(), 3u),
                parts.GetColourData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(colAttribLoc);
        glVertexAttribPointer(colAttribLoc, 3, GL_UNSIGNED_BYTE, GL_TRUE, parts.GetColourDataStride(), colorPtr);
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        if (createBufferData) {
            glBufferData(GL_ARRAY_BUFFER, partCount * (std::max)(parts.GetColourDataStride(), 4u),
                parts.GetColourData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(colAttribLoc);
        glVertexAttribPointer(colAttribLoc, 4, GL_UNSIGNED_BYTE, GL_TRUE, parts.GetColourDataStride(), colorPtr);
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
        if (createBufferData) {
            glBufferData(GL_ARRAY_BUFFER,
                partCount * (std::max)(parts.GetColourDataStride(), static_cast<unsigned int>(3 * sizeof(float))),
                parts.GetColourData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(colAttribLoc);
        glVertexAttribPointer(colAttribLoc, 3, GL_FLOAT, GL_TRUE, parts.GetColourDataStride(), colorPtr);
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
        if (createBufferData) {
            glBufferData(GL_ARRAY_BUFFER,
                partCount * (std::max)(parts.GetColourDataStride(), static_cast<unsigned int>(4 * sizeof(float))),
                parts.GetColourData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(colAttribLoc);
        glVertexAttribPointer(colAttribLoc, 4, GL_FLOAT, GL_TRUE, parts.GetColourDataStride(), colorPtr);
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
    case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
        if (createBufferData) {
            glBufferData(GL_ARRAY_BUFFER,
                partCount * (std::max)(parts.GetColourDataStride(), static_cast<unsigned int>(1 * sizeof(float))),
                parts.GetColourData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(colIdxAttribLoc);
        if (parts.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
            glVertexAttribPointer(colIdxAttribLoc, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), colorPtr);
        } else {
            glVertexAttribPointer(colIdxAttribLoc, 1, GL_DOUBLE, GL_FALSE, parts.GetColourDataStride(), colorPtr);
        }
    } break;
    case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA:
        if (createBufferData) {
            glBufferData(GL_ARRAY_BUFFER,
                partCount *
                    (std::max)(parts.GetColourDataStride(), static_cast<unsigned int>(4 * sizeof(unsigned short))),
                parts.GetColourData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(colAttribLoc);
        glVertexAttribPointer(colAttribLoc, 4, GL_UNSIGNED_SHORT, GL_TRUE, parts.GetColourDataStride(), colorPtr);
        break;
    default:
        break;
    }

    // radius and position
    glBindBuffer(GL_ARRAY_BUFFER, vertBuf);
    switch (parts.GetVertexDataType()) {
    case MultiParticleDataCall::Particles::VERTDATA_NONE:
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        if (createBufferData) {
            glBufferData(GL_ARRAY_BUFFER,
                partCount * (std::max)(parts.GetVertexDataStride(), static_cast<unsigned int>(3 * sizeof(float))),
                parts.GetVertexData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(vertAttribLoc);
        glVertexAttribPointer(vertAttribLoc, 3, GL_FLOAT, GL_FALSE, parts.GetVertexDataStride(), vertexPtr);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        if (createBufferData) {
            glBufferData(GL_ARRAY_BUFFER,
                partCount * (std::max)(parts.GetVertexDataStride(), static_cast<unsigned int>(4 * sizeof(float))),
                parts.GetVertexData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(vertAttribLoc);
        glVertexAttribPointer(vertAttribLoc, 4, GL_FLOAT, GL_FALSE, parts.GetVertexDataStride(), vertexPtr);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
        if (createBufferData) {
            glBufferData(GL_ARRAY_BUFFER,
                partCount * (std::max)(parts.GetVertexDataStride(), static_cast<unsigned int>(3 * sizeof(double))),
                parts.GetVertexData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(vertAttribLoc);
        glVertexAttribPointer(vertAttribLoc, 3, GL_DOUBLE, GL_FALSE, parts.GetVertexDataStride(), vertexPtr);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
        if (createBufferData) {
            glBufferData(GL_ARRAY_BUFFER,
                partCount * (std::max)(parts.GetVertexDataStride(), static_cast<unsigned int>(3 * sizeof(short))),
                parts.GetVertexData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(vertAttribLoc);
        glVertexAttribPointer(vertAttribLoc, 3, GL_SHORT, GL_FALSE, parts.GetVertexDataStride(), vertexPtr);
        break;
    default:
        break;
    }

    return true;
}


bool SphereRenderer::unsetBufferData(const vislib::graphics::gl::GLSLShader& shader) {

    GLuint vertAttribLoc = glGetAttribLocation(shader, "inPosition");
    GLuint colAttribLoc = glGetAttribLocation(shader, "inColor");
    GLuint colIdxAttribLoc = glGetAttribLocation(shader, "inColIdx");

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(vertAttribLoc);
    glDisableVertexAttribArray(colAttribLoc);
    glDisableVertexAttribArray(colIdxAttribLoc);

    return true;
}


bool SphereRenderer::setShaderData(vislib::graphics::gl::GLSLShader& shader, const MultiParticleDataCall::Particles &parts) {

    // colour
    bool useGlobalColor = false;
    bool useTf = false;
    switch (parts.GetColourDataType()) {
    case MultiParticleDataCall::Particles::COLDATA_NONE: {
        glUniform4f(shader.ParameterLocation("globalCol"), static_cast<float>(parts.GetGlobalColour()[0]) / 255.0f,
            static_cast<float>(parts.GetGlobalColour()[1]) / 255.0f,
            static_cast<float>(parts.GetGlobalColour()[2]) / 255.0f, 1.0f);
        useGlobalColor = true;
    } break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
    case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
        this->setTransferFunctionTexture(shader);
        useTf = true;
    } break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
    case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA: {
    } break;
    default: {
        glUniform4f(shader.ParameterLocation("globalCol"), 0.5f, 0.5f, 0.5f, 1.0f);
        useGlobalColor = true;
    } break;
    }
    glUniform1i(shader.ParameterLocation("useGlobalCol"), static_cast<GLint>(useGlobalColor));
    glUniform1i(shader.ParameterLocation("useTf"), static_cast<GLint>(useTf));

    // radius and position
    switch (parts.GetVertexDataType()) {
    case MultiParticleDataCall::Particles::VERTDATA_NONE:
        return false;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
    case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ: {
        glUniform1f(shader.ParameterLocation("constRad"), parts.GetGlobalRadius());
    } break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        glUniform1f(shader.ParameterLocation("constRad"), -1.0f);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
        glUniform1f(shader.ParameterLocation("constRad"), parts.GetGlobalRadius());
    default:
        return false;
    }

    return true;
}


bool SphereRenderer::unsetShaderData(void) {

    return this->unsetTransferFunctionTexture();
}


bool SphereRenderer::setTransferFunctionTexture(vislib::graphics::gl::GLSLShader& shader) {

    view::CallGetTransferFunction* cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
    if ((cgtf != nullptr) && (*cgtf)(0)) {
        cgtf->BindConvenience(shader, GL_TEXTURE0, 0);
    } else {
        glEnable(GL_TEXTURE_1D);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_1D, this->greyTF);
        glUniform1i(shader.ParameterLocation("tfTexture"), 0);
        auto range = this->colIdxRangeInfoParam.Param<param::Vector2fParam>()->Value();
        glUniform2fv(shader.ParameterLocation("tfRange"), 1, static_cast<GLfloat*>(range.PeekComponents()));
    }
    return true;
}


bool SphereRenderer::unsetTransferFunctionTexture(void) {

    view::CallGetTransferFunction* cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
    if (cgtf != nullptr) {
        cgtf->UnbindConvenience();
    } else {
        glBindTexture(GL_TEXTURE_1D, 0);
        glDisable(GL_TEXTURE_1D);
    }
    return true;
}


bool SphereRenderer::setFlagStorage(const vislib::graphics::gl::GLSLShader& shader, MultiParticleDataCall* mpdc) {

    this->flagsEnabled = false;

    auto flagc = this->getFlagsSlot.CallAs<FlagCall>();
    if (flagc == nullptr) {
        return false;
    }

    if (mpdc == nullptr) {
        return false;
    }
    unsigned int partsCount = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);
        partsCount += parts.GetCount();
    }

    ((*flagc)(FlagCall::CallMapFlags));
    flagc->validateFlagsCount(partsCount);

    this->flagsData = nullptr;
    this->flagsData = flagc->GetFlags();

    if (this->flagsUseSSBO) {
        auto version = flagc->GetVersion();
        if ((version != this->flagsCurrentVersion) || (version == 0)) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->flagsBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, partsCount * sizeof(FlagStorage::FlagItemType),
                this->flagsData.get()->data(), GL_STATIC_DRAW);
            this->flagsCurrentVersion = flagc->GetVersion();
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOflagsBindingPoint, this->flagsBuffer);
    } else {
        GLuint flagAttrib = glGetAttribLocation(shader, "inFlags");
        glEnableVertexAttribArray(flagAttrib);
        glVertexAttribIPointer(
            flagAttrib, 1, GL_UNSIGNED_INT, sizeof(FlagStorage::FlagItemType), this->flagsData.get()->data());
    }

    this->flagsEnabled = true;

    return true;
}


bool SphereRenderer::unsetFlagStorage(const vislib::graphics::gl::GLSLShader& shader) {

    auto flagc = this->getFlagsSlot.CallAs<FlagCall>();
    if (flagc == nullptr) {
        return false;
    }

    if (this->flagsUseSSBO) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    } else {
        GLuint flagAttrib = glGetAttribLocation(shader, "inFlags");
        glDisableVertexAttribArray(flagAttrib);
    }

    flagc->SetFlags(this->flagsData);
    (*flagc)(FlagCall::CallUnmapFlags);

    this->flagsEnabled = false;

    return true;
}


bool SphereRenderer::makeColorString(
    const MultiParticleDataCall::Particles& parts, std::string& outCode, std::string& outDeclaration, bool interleaved) {

    bool ret = true;

    switch (parts.GetColourDataType()) {
    case MultiParticleDataCall::Particles::COLDATA_NONE:
        outDeclaration = "";
        outCode = "    inColor = globalCol;\n";
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
        vislib::sys::Log::DefaultLog.WriteError(
            "[SphereRenderer] Cannot pack an unaligned RGB color into an SSBO! Giving up.");
        ret = false;
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        outDeclaration = "    uint color;\n";
        if (interleaved) {
            outCode =
                "    inColor = unpackUnorm4x8(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE "+ instanceOffset].color);\n";
        } else {
            outCode = "    inColor = unpackUnorm4x8(theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                      "+ instanceOffset].color);\n";
        }
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
        outDeclaration = "    float r; float g; float b;\n";
        if (interleaved) {
            outCode =
                "    inColor = vec4(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].r,\n"
                "                       theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].g,\n"
                "                       theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].b, 1.0); \n";
        } else {
            outCode =
                "    inColor = vec4(theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].r,\n"
                "                       theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].g,\n"
                "                       theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].b, 1.0); \n";
        }
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
        outDeclaration = "    float r; float g; float b; float a;\n";
        if (interleaved) {
            outCode = "    inColor = vec4(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].r,\n"
                      "                       theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].g,\n"
                      "                       theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].b,\n"
                      "                       theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].a); \n";
        } else {
            outCode = "    inColor = vec4(theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].r,\n"
                      "                       theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].g,\n"
                      "                       theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].b,\n"
                      "                       theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].a); \n";
        }
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
        outDeclaration = "    float colorIndex;\n";
        if (interleaved) {
            outCode = "    inColIdx = theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].colorIndex; \n";
        } else {
            outCode = "    inColIdx = theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].colorIndex; \n";
        }
    } break;
    case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
        outDeclaration = "    double colorIndex;\n";
        if (interleaved) {
            outCode =
                "    inColIdx = float(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].colorIndex); \n";
        } else {
            outCode = "    inColIdx = float(theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                      " + instanceOffset].colorIndex); \n";
        }
    } break;
    case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA: {
        outDeclaration = "    uint col1; uint col2;\n";
        if (interleaved) {
            outCode = "    inColor.xy = unpackUnorm2x16(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                      "+ instanceOffset].col1);\n"
                      "    inColor.zw = unpackUnorm2x16(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                      "+ instanceOffset].col2);\n";
        } else {
            outCode = "    inColor.xy = unpackUnorm2x16(theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                      "+ instanceOffset].col1);\n"
                      "    inColor.zw = unpackUnorm2x16(theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                      "+ instanceOffset].col2);\n";
        }
    } break;
    default:
        outDeclaration = "";
        outCode = "    inColor = globalCol;\n";
        break;
    }
    // outCode = "    inColor = vec4(0.2, 0.7, 1.0, 1.0);";

    return ret;
}


bool SphereRenderer::makeVertexString(
    const MultiParticleDataCall::Particles& parts, std::string& outCode, std::string& outDeclaration, bool interleaved) {

    bool ret = true;

    switch (parts.GetVertexDataType()) {
    case MultiParticleDataCall::Particles::VERTDATA_NONE:
        outDeclaration = "";
        outCode = "";
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        outDeclaration = "    float posX; float posY; float posZ;\n";
        if (interleaved) {
            outCode = "    inPosition = vec4(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX,\n"
                      "                 theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY,\n"
                      "                 theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ, 1.0); \n"
                      "    rad = constRad;";
        } else {
            outCode =
                "    inPosition = vec4(thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX,\n"
                "                 thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY,\n"
                "                 thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ, 1.0); \n"
                "    rad = constRad;";
        }
        break;
    case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
        outDeclaration = "    double posX; double posY; double posZ;\n";
        if (interleaved) {
            outCode =
                "    inPosition = vec4(float(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX),\n"
                "                 float(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY),\n"
                "                 float(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ), 1.0); \n"
                "    rad = constRad;";
        } else {
            outCode =
                "    inPosition = vec4(float(thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX),\n"
                "                 float(thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY),\n"
                "                 float(thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                " + instanceOffset].posZ), 1.0); \n"
                "    rad = constRad;";
        }
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        outDeclaration = "    float posX; float posY; float posZ; float posR;\n";
        if (interleaved) {
            outCode = "    inPosition = vec4(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX,\n"
                      "                 theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY,\n"
                      "                 theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ, 1.0); \n"
                      "    rad = theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posR;";
        } else {
            outCode =
                "    inPosition = vec4(thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX,\n"
                "                 thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY,\n"
                "                 thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ, 1.0); \n"
                "    rad = thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posR;";
        }
        break;
    default:
        outDeclaration = "";
        outCode = "";
        break;
    }

    return ret;
}


std::shared_ptr<GLSLShader> SphereRenderer::makeShader(std::shared_ptr<ShaderSource> vert, std::shared_ptr<ShaderSource> frag) {

    std::shared_ptr<GLSLShader> sh = std::make_shared<GLSLShader>(GLSLShader());
    try {
        if (!sh->Create(vert->Code(), vert->Count(), frag->Code(), frag->Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Unable to compile sphere shader: Unknown error\n");
            return nullptr;
        }

    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "[SphereRenderer] Unable to compile sphere shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return nullptr;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Unable to compile sphere shader: %s\n", e.GetMsgA());
        return nullptr;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Unable to compile sphere shader: Unknown exception\n");
        return nullptr;
    }
    return sh;
}


std::shared_ptr<vislib::graphics::gl::GLSLShader> SphereRenderer::generateShader(const MultiParticleDataCall::Particles& parts) {

    int c = parts.GetColourDataType();
    int p = parts.GetVertexDataType();

    unsigned int colBytes, vertBytes, colStride, vertStride;
    bool interleaved;
    this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride, interleaved);

    shaderMap::iterator i = this->theShaders.find(std::make_tuple(c, p, interleaved));
    if (i == this->theShaders.end()) {
        std::shared_ptr<ShaderSource> v2 = std::make_shared<ShaderSource>(*this->vertShader);
        vislib::SmartPtr<ShaderSource::Snippet> codeSnip, declarationSnip;
        std::string vertCode, colCode, vertDecl, colDecl, decl;
        makeVertexString(parts, vertCode, vertDecl, interleaved);
        makeColorString(parts, colCode, colDecl, interleaved);

        if (interleaved) {

            decl = "\nstruct SphereParams {\n";

            if (parts.GetColourData() < parts.GetVertexData()) {
                decl += colDecl;
                decl += vertDecl;
            } else {
                decl += vertDecl;
                decl += colDecl;
            }
            decl += "};\n";

            decl += "layout(" SSBO_GENERATED_SHADER_ALIGNMENT ", binding = " + std::to_string(SSBOvertexBindingPoint) +
                    ") buffer shader_data {\n"
                    "    SphereParams theBuffer[];\n"
                    // flat float version
                    //"    float theBuffer[];\n"
                    "};\n";

        } else {
            // we seem to have separate buffers for vertex and color data

            decl = "\nstruct SpherePosParams {\n" + vertDecl;
            decl += "};\n";

            decl += "\nstruct SphereColParams {\n" + colDecl;
            decl += "};\n";

            decl += "layout(" SSBO_GENERATED_SHADER_ALIGNMENT ", binding = " + std::to_string(SSBOvertexBindingPoint) +
                    ") buffer shader_data {\n"
                    "    SpherePosParams thePosBuffer[];\n"
                    "};\n";
            decl += "layout(" SSBO_GENERATED_SHADER_ALIGNMENT ", binding = " + std::to_string(SSBOcolorBindingPoint) +
                    ") buffer shader_data2 {\n"
                    "    SphereColParams theColBuffer[];\n"
                    "};\n";
        }
        std::string code = "\n";
        code += colCode;
        code += vertCode;
        declarationSnip = new ShaderSource::StringSnippet(decl.c_str());
        codeSnip = new ShaderSource::StringSnippet(code.c_str());

        // Generated shader declaration snippet is inserted after ssbo_vert_attributes.glsl
        v2->Insert(8, declarationSnip);
        // Generated shader code snippet is inserted after ssbo_vert_mainstart.glsl (Consider new index through first
        // insertion)
        v2->Insert(10, codeSnip);

        std::shared_ptr<ShaderSource> vss(v2);
        this->theShaders.emplace(std::make_pair(std::make_tuple(c, p, interleaved), makeShader(v2, this->fragShader)));
        i = this->theShaders.find(std::make_tuple(c, p, interleaved));
    }
    return i->second;
}


void SphereRenderer::getBytesAndStride(const MultiParticleDataCall::Particles &parts, unsigned int &outColBytes, unsigned int &outVertBytes,
    unsigned int &outColStride, unsigned int &outVertStride, bool &outInterleaved) {

    outVertBytes = MultiParticleDataCall::Particles::VertexDataSize[parts.GetVertexDataType()];
    outColBytes = MultiParticleDataCall::Particles::ColorDataSize[parts.GetColourDataType()];

    outColStride = parts.GetColourDataStride();
    outColStride = outColStride < outColBytes ? outColBytes : outColStride;
    outVertStride = parts.GetVertexDataStride();
    outVertStride = outVertStride < outVertBytes ? outVertBytes : outVertStride;

    outInterleaved = (std::abs(reinterpret_cast<const ptrdiff_t>(parts.GetColourData()) -
                               reinterpret_cast<const ptrdiff_t>(parts.GetVertexData())) <= outVertStride &&
                         outVertStride == outColStride) ||
                     outColStride == 0;
}


void SphereRenderer::getGLSLVersion(int &outMajor, int &outMinor) const {
    
    outMajor = -1;
    outMinor = -1;
    std::string glslVerStr((char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
    std::size_t found = glslVerStr.find(".");
    if (found != std::string::npos) {
        outMajor = std::atoi(glslVerStr.substr(0, 1).c_str());
        outMinor = std::atoi(glslVerStr.substr(found + 1, 1).c_str());
    } else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
            "[SphereRenderer] No valid GL_SHADING_LANGUAGE_VERSION string found: %s", glslVerStr.c_str());
    }
}


#if defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)

void SphereRenderer::lockSingle(GLsync& outSyncObj) {

    if (outSyncObj) {
        glDeleteSync(outSyncObj);
    }
    outSyncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}


void SphereRenderer::waitSingle(const GLsync& syncObj) {

    if (syncObj) {
        while (1) {
            GLenum wait = glClientWaitSync(syncObj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                return;
            }
        }
    }
}

#endif // defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)


// ##### Ambient Occlusion ################################################# //

bool SphereRenderer::rebuildGBuffer() {

    if (!this->triggerRebuildGBuffer && (this->curVpWidth == this->lastVpWidth) &&
        (this->curVpHeight == this->lastVpHeight) && !this->useHPTexturesSlot.IsDirty()) {
        return true;
    }

    this->useHPTexturesSlot.ResetDirty();

    bool highPrecision = this->useHPTexturesSlot.Param<param::BoolParam>()->Value();

    glBindTexture(GL_TEXTURE_2D, this->gBuffer.color);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, this->curVpWidth, this->curVpHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    glBindTexture(GL_TEXTURE_2D, this->gBuffer.normals);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, highPrecision ? GL_RGBA32F : GL_RGBA, this->curVpWidth, this->curVpHeight, 0, GL_RGB,
        GL_UNSIGNED_BYTE, nullptr);

    glBindTexture(GL_TEXTURE_2D, this->gBuffer.depth);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, this->curVpWidth, this->curVpHeight, 0, GL_DEPTH_COMPONENT,
        GL_UNSIGNED_BYTE, nullptr);

    glBindTexture(GL_TEXTURE_2D, 0);

    // Configure the framebuffer object
    GLint prevFBO;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFBO);

    glBindFramebuffer(GL_FRAMEBUFFER, this->gBuffer.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->gBuffer.color, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, this->gBuffer.normals, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->gBuffer.depth, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "[SphereRenderer] Framebuffer NOT complete!");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, prevFBO);

    if (this->triggerRebuildGBuffer) {
        this->triggerRebuildGBuffer = false;
    }

    return true;
}


void SphereRenderer::rebuildWorkingData(view::CallRender3D_2& call, MultiParticleDataCall* mpdc, const vislib::graphics::gl::GLSLShader& shader) {

    // Upload new data if neccessary
    if (stateInvalid) {
        unsigned int partsCount = mpdc->GetParticleListCount();

        // Add buffers if neccessary
        for (unsigned int i = static_cast<unsigned int>(this->gpuData.size()); i < partsCount; ++i) {
            gpuParticleDataType data;

            glGenVertexArrays(1, &(data.vertexArray));
            glGenBuffers(1, &(data.vertexVBO));
            glGenBuffers(1, &(data.colorVBO));

            this->gpuData.push_back(data);
        }

        // Remove buffers if neccessary
        while (this->gpuData.size() > partsCount) {
            gpuParticleDataType& data = this->gpuData.back();
            glDeleteVertexArrays(1, &(data.vertexArray));
            glDeleteBuffers(1, &(data.vertexVBO));
            glDeleteBuffers(1, &(data.colorVBO));
            this->gpuData.pop_back();
        }

        // Reupload buffers
        for (unsigned int i = 0; i < partsCount; ++i) {
            MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

            glBindVertexArray(this->gpuData[i].vertexArray);
            this->setBufferData(shader, parts, this->gpuData[i].vertexVBO, parts.GetVertexData(),
                this->gpuData[i].colorVBO, parts.GetColourData(), true);
            glBindVertexArray(0);
            this->unsetBufferData(shader);
        }
    }

    // Check if voxelization is even needed
    if (this->volGen == nullptr) {
        this->volGen = new misc::MDAOVolumeGenerator();
        this->volGen->SetShaderSourceFactory(&instance()->ShaderSourceFactory());
        this->volGen->Init();
    }

    // Recreate the volume if neccessary
    bool equalClipData = true;
    for (size_t i = 0; i < 4; ++i) {
        if (this->oldClipDat[i] != this->curClipDat[i]) {
            equalClipData = false;
            break;
        }
    }
    if (volGen != nullptr && (stateInvalid || this->aoVolSizeSlot.IsDirty() || !equalClipData)) {
        int volSize = this->aoVolSizeSlot.Param<param::IntParam>()->Value();

        vislib::math::Dimension<float, 3> dims = this->curClipBox.GetSize();

        // Calculate the extensions of the volume by using the specified number of voxels for the longest edge
        float longestEdge = this->curClipBox.LongestEdge();
        dims.Scale(static_cast<float>(volSize) / longestEdge);

        // The X size must be a multiple of 4, so we might have to correct that a little
        dims.SetWidth(ceil(dims.GetWidth() / 4.0f) * 4.0f);

        dims.SetHeight(ceil(dims.GetHeight()));
        dims.SetDepth(ceil(dims.GetDepth()));
        ambConeConstants[0] = (std::min)(dims.Width(), (std::min)(dims.Height(), dims.Depth()));
        ambConeConstants[1] = ceil(std::log2(static_cast<float>(volSize))) - 1.0f;

        // Set resolution accordingly
        this->volGen->SetResolution(ceil(dims.GetWidth()), ceil(dims.GetHeight()), ceil(dims.GetDepth()));

        // Insert all particle lists
        this->volGen->ClearVolume();

        this->volGen->StartInsertion(this->curClipBox, glm::vec4(this->curClipDat[0], this->curClipDat[1],
            this->curClipDat[2], this->curClipDat[3]));
        for (unsigned int i = 0; i < this->gpuData.size(); ++i) {
            float globalRadius = 0.0f;
            if (mpdc->AccessParticles(i).GetVertexDataType() !=
                MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)
                globalRadius = mpdc->AccessParticles(i).GetGlobalRadius();
            this->volGen->InsertParticles(static_cast<unsigned int>(mpdc->AccessParticles(i).GetCount()), globalRadius,
                this->gpuData[i].vertexArray);
        }
        this->volGen->EndInsertion();

        this->volGen->RecreateMipmap();
    }

    this->aoVolSizeSlot.ResetDirty();
}



void SphereRenderer::renderDeferredPass(view::CallRender3D_2& call) {

    bool enableLighting = this->enableLightingSlot.Param<param::BoolParam>()->Value();
    bool highPrecision = this->useHPTexturesSlot.Param<param::BoolParam>()->Value();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, this->gBuffer.depth);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->gBuffer.normals);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->gBuffer.color);
    if (volGen != nullptr) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_3D, this->volGen->GetVolumeTextureHandle());
        glActiveTexture(GL_TEXTURE0);
    }

    this->lightingShader.Enable();

    this->lightingShader.SetParameter("inColorTex", static_cast<int>(0));
    this->lightingShader.SetParameter("inNormalsTex", static_cast<int>(1));
    this->lightingShader.SetParameter("inDepthTex", static_cast<int>(2));
    this->lightingShader.SetParameter("inDensityTex", static_cast<int>(3));

    this->lightingShader.SetParameter("inWidth", static_cast<float>(this->curVpWidth));
    this->lightingShader.SetParameter("inHeight", static_cast<float>(this->curVpHeight));
    glUniformMatrix4fv(this->lightingShader.ParameterLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->curMVPinv));
    this->lightingShader.SetParameter("inUseHighPrecision", highPrecision);
    if (enableLighting) {
        this->lightingShader.SetParameterArray3("inObjLightDir", 1, glm::value_ptr(this->curlightDir));
        this->lightingShader.SetParameterArray3("inObjCamPos", 1, glm::value_ptr(this->curCamPos));
    }
    this->lightingShader.SetParameter("inAOOffset", this->aoOffsetSlot.Param<param::FloatParam>()->Value());
    this->lightingShader.SetParameter("inAOStrength", this->aoStrengthSlot.Param<param::FloatParam>()->Value());
    this->lightingShader.SetParameter("inAOConeLength", this->aoConeLengthSlot.Param<param::FloatParam>()->Value());
    this->lightingShader.SetParameter("inAmbVolShortestEdge", this->ambConeConstants[0]);
    this->lightingShader.SetParameter("inAmbVolMaxLod", this->ambConeConstants[1]);
    this->lightingShader.SetParameterArray3("inBoundsMin", 1, this->curClipBox.GetLeftBottomBack().PeekCoordinates());
    this->lightingShader.SetParameterArray3("inBoundsSize", 1, this->curClipBox.GetSize().PeekDimension());
  
    // Draw screen filling 'quad' (2 triangle, front facing: CCW)
    std::vector<GLfloat> vertices = {-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f};
    GLuint vertAttribLoc = glGetAttribLocation(this->lightingShader, "inPosition");
    glEnableVertexAttribArray(vertAttribLoc);
    glVertexAttribPointer(vertAttribLoc, 2, GL_FLOAT, GL_TRUE, 0, vertices.data());
    glDrawArrays(GL_TRIANGLES, static_cast<GLint>(0), static_cast<GLsizei>(vertices.size() / 2));
    glDisableVertexAttribArray(vertAttribLoc);

    this->lightingShader.Disable();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindTexture(GL_TEXTURE_3D, 0);

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_3D);
}


void SphereRenderer::generate3ConeDirections(std::vector<glm::vec4>& directions, float apex) {

    directions.clear();

    float edge_length = 2.0f * tan(0.5f * apex);
    float height = sqrt(1.0f - edge_length * edge_length / 12.0f);
    float radius = sqrt(3.0f) / 3.0f * edge_length;

    for (int i = 0; i < 3; ++i) {
        float angle = static_cast<float>(i) / 3.0f * 2.0f * static_cast<float>(M_PI);

        glm::vec3 center(cos(angle) * radius, height, sin(angle) * radius);
        center = glm::normalize(center);
        directions.push_back(glm::vec4(center.x, center.y, center.z, edge_length));
    }
}


std::string SphereRenderer::generateDirectionShaderArrayString(const std::vector<glm::vec4>& directions, const std::string& directionsName) {

    std::stringstream result;

    std::string upperDirName = directionsName;
    std::transform(upperDirName.begin(), upperDirName.end(), upperDirName.begin(), ::toupper);

    result << "#define NUM_" << upperDirName << " " << directions.size() << std::endl;
    result << "const vec4 " << directionsName << "[NUM_" << upperDirName << "] = vec4[NUM_" << upperDirName << "]("
           << std::endl;

    for (auto iter = directions.begin(); iter != directions.end(); ++iter) {
        result << "\tvec4(" << (*iter)[0] << ", " << (*iter)[1] << ", " << (*iter)[2] << ", " << (*iter)[3] << ")";
        if (iter + 1 != directions.end()) result << ",";
        result << std::endl;
    }
    result << ");" << std::endl;

    return result.str();
}
