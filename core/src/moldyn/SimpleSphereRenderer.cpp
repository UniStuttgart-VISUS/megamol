/*
 * SimpleSphereRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 *
 */


#include "stdafx.h"
#include "mmcore/moldyn/SimpleSphereRenderer.h"


using namespace megamol::core;
using namespace vislib::graphics::gl;


#define MAP_BUFFER_LOCALLY
#define DEBUG_GL_CALLBACK
//#define CHRONOTIMING

#define NGS_THE_INSTANCE "gl_VertexID" // "gl_InstanceID"
#define NGS_THE_ALIGNMENT "packed"

const GLuint SSBObindingPoint = 2;
const GLuint SSBOcolorBindingPoint = 3;


#define checkGLError                                                                                                   \
    {                                                                                                                  \
        GLenum errCode = glGetError();                                                                                 \
        if (errCode != GL_NO_ERROR)                                                                                    \
            std::cout << "Error in line " << __LINE__ << ": " << gluErrorString(errCode) << std::endl;                 \
    }


// typedef void (APIENTRY *GLDEBUGPROC)(GLenum source,GLenum type,GLuint id,GLenum severity,GLsizei length,const GLchar
// *message,const void *userParam);
void APIENTRY DebugGLCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
    const GLchar* message, const GLvoid* userParam) {
    const char *sourceText, *typeText, *severityText;
    switch (source) {
    case GL_DEBUG_SOURCE_API:
        sourceText = "API";
        break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        sourceText = "Window System";
        break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
        sourceText = "Shader Compiler";
        break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:
        sourceText = "Third Party";
        break;
    case GL_DEBUG_SOURCE_APPLICATION:
        sourceText = "Application";
        break;
    case GL_DEBUG_SOURCE_OTHER:
        sourceText = "Other";
        break;
    default:
        sourceText = "Unknown";
        break;
    }
    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
        typeText = "Error";
        break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        typeText = "Deprecated Behavior";
        break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        typeText = "Undefined Behavior";
        break;
    case GL_DEBUG_TYPE_PORTABILITY:
        typeText = "Portability";
        break;
    case GL_DEBUG_TYPE_PERFORMANCE:
        typeText = "Performance";
        break;
    case GL_DEBUG_TYPE_OTHER:
        typeText = "Other";
        break;
    case GL_DEBUG_TYPE_MARKER:
        typeText = "Marker";
        break;
    default:
        typeText = "Unknown";
        break;
    }
    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        severityText = "High";
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        severityText = "Medium";
        break;
    case GL_DEBUG_SEVERITY_LOW:
        severityText = "Low";
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        severityText = "Notification";
        break;
    default:
        severityText = "Unknown";
        break;
    }
    vislib::sys::Log::DefaultLog.WriteMsg(
        vislib::sys::Log::LEVEL_ERROR, "[%s %s] (%s %u) %s\n", sourceText, severityText, typeText, id, message);
}


/******************************************************************************
 * Known DebugGLCallback Errors (CAN BE IGNORED):
 *
 * >>> glUnmapNamedBuffe (only if named buffer wasn't mapped before):
 *      [API High] (Error 1282) GL_INVALID_OPERATION error generated. Buffer is unbound or is already unmapped.
 *
 * >>> glMapNamedBufferRange:
 *      [API Notification] (Other 131185) Buffer detailed info: Buffer object 1 (bound to GL_SHADER_STORAGE_BUFFER,
 *usage hint is GL_DYNAMIC_DRAW) will use SYSTEM HEAP memory as the source for buffer object operations. [API
 *Notification] (Other 131185) Buffer detailed info: Buffer object 1 (bound to GL_SHADER_STORAGE_BUFFER, usage hint is
 *GL_DYNAMIC_DRAW) has been mapped WRITE_ONLY in SYSTEM HEAP memory(fast).
 *
 ******************************************************************************/


 /*
  * moldyn::SimpleSphereRenderer::SimpleSphereRenderer
  */
moldyn::SimpleSphereRenderer::SimpleSphereRenderer(void)
    : AbstractSimpleSphereRenderer()
    , curViewAttrib()
    , curClipDat()
    , oldClipDat()
    , curClipCol()
    , curLightPos()
    , curVpWidth(0)
    , curVpHeight(0)
    , lastVpWidth(0)
    , lastVpHeight(0)
    , curMVinv()
    , curMVP()
    , curMVPinv()
    , curMVPtransp()
    , renderMode(RenderMode::NG)
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
    , theShaders_splat()
    , streamer()
    , colStreamer()
    , bufArray()
    , colBufArray()
    , fences()
    , theSingleBuffer()
    , currBuf(0)
    , bufSize(32 * 1024 * 1024)
    , numBuffers(3)
    , theSingleMappedMem(nullptr)
    , singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT)
    , singleBufferMappingBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT)
    , gpuData()
    , gBuffer()
    , oldHash(0)
    , oldFrameID(0)
    , ambConeConstants()
    , tfFallbackHandle(0)
    , volGen(nullptr)
    ,
    // timer(),
    renderModeParam("renderMode", "The sphere render mode.")
    , toggleModeParam("renderModeButton", "Toggle sphere render modes.")
    , radiusScalingParam("scaling", "Scaling factor for particle radii.")
    , alphaScalingParam("splat::alphaScaling", "NG Splat: Scaling factor for particle alpha.")
    , attenuateSubpixelParam(
        "splat::attenuateSubpixel", "NG Splat: Attenuate alpha of points that should have subpixel size.")
    , useStaticDataParam("ng::staticData", "NG: Upload data only once per hash change and keep data static on GPU")
    , enableLightingSlot("ao::enable_lighting", "Ambient Occlusion: Enable Lighting")
    , enableAOSlot("ao::enable_ao", "Ambient Occlusion: Enable Ambient Occlusion")
    , enableGeometryShader(
        "ao::use_gs_proxies", "Ambient Occlusion: Enables rendering using triangle strips from the geometry shader")
    , aoVolSizeSlot("ao::volsize", "Ambient Occlusion: Longest volume edge")
    , aoConeApexSlot("ao::apex", "Ambient Occlusion: Cone Apex Angle")
    , aoOffsetSlot("ao::offset", "Ambient Occlusion: Offset from Surface")
    , aoStrengthSlot("ao::strength", "Ambient Occlusion: Strength")
    , aoConeLengthSlot("ao::conelen", "Ambient Occlusion: Cone length")
    , useHPTexturesSlot("ao::high_prec_tex", "Ambient Occlusion: Use high precision textures") {

    param::EnumParam* rmp = new param::EnumParam(this->renderMode);
    rmp->SetTypePair(RenderMode::SIMPLE, "Simple");
    rmp->SetTypePair(RenderMode::SIMPLE_CLUSTERED, "Simple_Clustered");
    rmp->SetTypePair(RenderMode::SIMPLE_GEO, "Simple_Geometry_Shader");
    rmp->SetTypePair(RenderMode::NG, "NG");
    rmp->SetTypePair(RenderMode::NG_SPLAT, "NG_Splat");
    rmp->SetTypePair(RenderMode::NG_BUFFER_ARRAY, "NG_Buffer_Array");
    rmp->SetTypePair(RenderMode::AMBIENT_OCCLUSION, "Ambient_Occlusion");

    this->renderModeParam << rmp;
    this->MakeSlotAvailable(&this->renderModeParam);

    this->toggleModeParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_R));
    this->toggleModeParam.SetUpdateCallback(&SimpleSphereRenderer::toggleRenderMode);
    this->MakeSlotAvailable(&this->toggleModeParam);

    this->radiusScalingParam << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->radiusScalingParam);

    this->alphaScalingParam << new core::param::FloatParam(5.0f);
    this->MakeSlotAvailable(&this->alphaScalingParam);

    this->attenuateSubpixelParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->attenuateSubpixelParam);

    this->useStaticDataParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->useStaticDataParam);

    this->enableLightingSlot << (new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->enableLightingSlot);

    this->enableAOSlot << (new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->enableAOSlot);

    this->enableGeometryShader << (new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->enableGeometryShader);

    this->aoVolSizeSlot << (new core::param::IntParam(128, 1, 1024));
    this->MakeSlotAvailable(&this->aoVolSizeSlot);

    this->aoConeApexSlot << (new core::param::FloatParam(50.0f, 1.0f, 90.0f));
    this->MakeSlotAvailable(&this->aoConeApexSlot);

    this->aoOffsetSlot << (new core::param::FloatParam(0.01f, 0.0f, 0.2f));
    this->MakeSlotAvailable(&this->aoOffsetSlot);

    this->aoStrengthSlot << (new core::param::FloatParam(1.0f, 0.1f, 20.0f));
    this->MakeSlotAvailable(&this->aoStrengthSlot);

    this->aoConeLengthSlot << (new core::param::FloatParam(0.8f, 0.01f, 1.0f));
    this->MakeSlotAvailable(&this->aoConeLengthSlot);

    this->useHPTexturesSlot << (new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->useHPTexturesSlot);

    // this->forceTimeSlot.SetParameter(new core::param::BoolParam(false));
    // this->MakeSlotAvailable(&this->forceTimeSlot);

    // this->resetResources();

    // Ambient Occlusion ------------------------------------------------------
    oldHash = -1;
    curVpWidth = -1;
    curVpHeight = -1;
    this->volGen = nullptr;
}


/*
 * moldyn::SimpleSphereRenderer::~SimpleSphereRenderer
 */
moldyn::SimpleSphereRenderer::~SimpleSphereRenderer(void) { this->Release(); }


/*
 * moldyn::SimpleSphereRenderer::create
 */
bool moldyn::SimpleSphereRenderer::create(void) {

    ASSERT(IsAvailable());

#ifdef DEBUG_GL_CALLBACK
    glDebugMessageCallback(DebugGLCallback, nullptr);
#endif

    this->renderMode = static_cast<RenderMode>(this->renderModeParam.Param<param::EnumParam>()->Value());
    if (!this->createResources()) {
        return false;
    }

    // timer.SetNumRegions(4);
    // const char *regions[4] = {"Upload1", "Upload2", "Upload3", "Rendering"};
    // timer.SetRegionNames(4, regions);
    // timer.SetStatisticsFileName("fullstats.csv");
    // timer.SetSummaryFileName("summary.csv");
    // timer.SetMaximumFrames(20, 100);

    return (AbstractSimpleSphereRenderer::create());
}


/*
 * moldyn::SimpleSphereRenderer::release
 */
void moldyn::SimpleSphereRenderer::release(void) {

    this->resetResources();
    AbstractSimpleSphereRenderer::release();
}


/*
 * moldyn::SimpleSphereRenderer::toggleRenderMode
 */
bool moldyn::SimpleSphereRenderer::toggleRenderMode(param::ParamSlot& slot) {

    ASSERT((&slot == &this->toggleModeParam));

    // Only changing value of parameter.
    auto currentRenderMode = this->renderModeParam.Param<param::EnumParam>()->Value();
    currentRenderMode = (currentRenderMode + 1) % (static_cast<int>(RenderMode::__MODE_COUNT__));
    this->renderModeParam.Param<param::EnumParam>()->SetValue(currentRenderMode);

    return true;
}


/*
 * moldyn::SimpleSphereRenderer::resetResources
 */
bool moldyn::SimpleSphereRenderer::resetResources(void) {

    this->sphereShader.Release();
    this->sphereGeometryShader.Release();
    this->lightingShader.Release();

    this->vertShader = nullptr;
    this->fragShader = nullptr;
    this->geoShader = nullptr;

    this->theSingleMappedMem = nullptr;

    if (this->newShader != nullptr) {
        this->newShader->Release();
        this->newShader.reset();
    }
    this->theShaders.clear();
    this->theShaders_splat.clear();

    if (this->volGen != nullptr) {
        delete this->volGen;
        this->volGen = nullptr;
    }

    glDeleteTextures(3, reinterpret_cast<GLuint*>(&this->gBuffer));
    glDeleteFramebuffers(1, &(this->gBuffer.fbo));

    glDeleteTextures(1, &(this->tfFallbackHandle));

    for (int i = 0; i < this->gpuData.size(); ++i) {
        glDeleteVertexArrays(3, reinterpret_cast<GLuint*>(&(this->gpuData[i])));
    }
    this->gpuData.clear();

    glUnmapNamedBufferEXT(this->theSingleBuffer);

    for (auto& x : fences) {
        if (x) {
            glDeleteSync(x);
        }
    }

    glDeleteBuffers(1, &(this->theSingleBuffer));
    glDeleteVertexArrays(1, &(this->vertArray));

    this->currBuf = 0;
    this->bufSize = (32 * 1024 * 1024);
    this->numBuffers = 3;
    this->fences.clear();
    this->fences.resize(numBuffers);

    this->oldHash = 0;
    this->oldFrameID = 0;

    this->colType = SimpleSphericalParticles::ColourDataType::COLDATA_NONE;
    this->vertType = SimpleSphericalParticles::VertexDataType::VERTDATA_NONE;

    /// This variant should not need the fence (?)
    // singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT);
    // singleBufferMappingBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT);
    this->singleBufferCreationBits = (GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT);
    this->singleBufferMappingBits = (GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);

    return true;
}


/*
 * moldyn::SimpleSphereRenderer::createResources
 */
bool moldyn::SimpleSphereRenderer::createResources() {

    this->resetResources();

    this->vertShader = new ShaderSource();
    this->fragShader = new ShaderSource();

    vislib::StringA vertShaderName;
    vislib::StringA fragShaderName;
    vislib::StringA geoShaderName;

    vislib::StringA mode;
    switch (this->renderMode) {
    case (RenderMode::SIMPLE):
        mode = "SIMPLE";
        break;
    case (RenderMode::SIMPLE_CLUSTERED):
        mode = "SIMPLE CLUSTERED";
        break;
    case (RenderMode::SIMPLE_GEO):
        mode = "SIMPLE GEOMETRY SHADER";
        break;
    case (RenderMode::NG):
        mode = "NG";
        break;
    case (RenderMode::NG_SPLAT):
        mode = "NG SPLAT";
        break;
    case (RenderMode::NG_BUFFER_ARRAY):
        mode = "NG BUFFER ARRAY";
        break;
    case (RenderMode::AMBIENT_OCCLUSION):
        mode = "AMBIENT OCCLUSION";
        break;
    default:
        break;
    }
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, ">>>>> Using render mode: %s (%d)",
        mode.PeekBuffer(), static_cast<int>(this->renderMode));

    try {
        switch (this->renderMode) {

        case (RenderMode::SIMPLE):
        case (RenderMode::SIMPLE_CLUSTERED): {
            vertShaderName = "simplesphere::vertex";
            fragShaderName = "simplesphere::fragment";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), *this->vertShader)) {
                return false;
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), *this->fragShader)) {
                return false;
            }
            if (!this->sphereShader.Create(this->vertShader->Code(), this->vertShader->Count(),
                this->fragShader->Code(), this->fragShader->Count())) {
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR, "Unable to compile sphere shader: Unknown error\n");
                return false;
            }
        } break;

        case (RenderMode::SIMPLE_GEO):
            this->geoShader = new ShaderSource();
            vertShaderName = "geosphere::vertex";
            fragShaderName = "geosphere::fragment";
            geoShaderName = "geosphere::geometry";
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
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR, "Unable to compile sphere geometry shader: Unknown error\n");
                return false;
            }
            if (!this->sphereGeometryShader.Link()) {
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR, "Unable to link sphere geometry shader: Unknown error\n");
                return false;
            }
            break;

        case (RenderMode::NG):
            vertShaderName = "ngsphere::vertex";
            fragShaderName = "ngsphere::fragment";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), *this->vertShader)) {
                return false;
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), *this->fragShader)) {
                return false;
            }
            glGenVertexArrays(1, &this->vertArray);
            glBindVertexArray(this->vertArray);
            glBindVertexArray(0);
            break;

        case (RenderMode::NG_SPLAT):
            vertShaderName = "ngsplat::vertex";
            fragShaderName = "ngsplat::fragment";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), *this->vertShader)) {
                return false;
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
            this->theSingleMappedMem = glMapNamedBufferRangeEXT(
                this->theSingleBuffer, 0, this->bufSize * this->numBuffers, singleBufferMappingBits);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            glBindVertexArray(0);
            break;

        case (RenderMode::NG_BUFFER_ARRAY):
            vertShaderName = "ngbufferarray::vertex";
            fragShaderName = "ngbufferarray::fragment";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), *this->vertShader)) {
                return false;
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), *this->fragShader)) {
                return false;
            }
            if (!this->sphereShader.Create(this->vertShader->Code(), this->vertShader->Count(),
                this->fragShader->Code(), this->fragShader->Count())) {
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR, "Unable to compile sphere shader: Unknown error\n");
                return false;
            }
            glGenVertexArrays(1, &this->vertArray);
            glBindVertexArray(this->vertArray);
            glGenBuffers(1, &this->theSingleBuffer);
            glBindBuffer(GL_ARRAY_BUFFER, this->theSingleBuffer);
            glBufferStorage(GL_ARRAY_BUFFER, this->bufSize * this->numBuffers, nullptr, singleBufferCreationBits);
            this->theSingleMappedMem = glMapNamedBufferRangeEXT(
                this->theSingleBuffer, 0, this->bufSize * this->numBuffers, singleBufferMappingBits);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
            break;

        case (RenderMode::AMBIENT_OCCLUSION): {
            // Try to initialize OPENGL extensions
            if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
                return false;
            }

            // Generate texture and frame buffer handles
            glGenTextures(3, reinterpret_cast<GLuint*>(&this->gBuffer));
            glGenFramebuffers(1, &(this->gBuffer.fbo));

            // Build the sphere shader
            this->rebuildShader();

            bool enableAO = this->enableAOSlot.Param<megamol::core::param::BoolParam>()->Value();

            if (enableAO) {
                this->volGen = new megamol::core::utility::MDAO2VolumeGenerator();
                this->volGen->SetShaderSourceFactory(&this->GetCoreInstance()->ShaderSourceFactory());
                if (!this->volGen->Init()) {
                    vislib::sys::Log::DefaultLog.WriteMsg(
                        vislib::sys::Log::LEVEL_ERROR, "Error initializing volume generator!\n");
                    return false;
                }
            }

            glGenTextures(1, &this->tfFallbackHandle);
            unsigned char tex[6] = { 0, 0, 0, 255, 255, 255 };
            glBindTexture(GL_TEXTURE_1D, this->tfFallbackHandle);
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glBindTexture(GL_TEXTURE_1D, 0);
        } break;

        default:
            return false;
        }
    }
    catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    }
    catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile sphere shader: %s\n", e.GetMsgA());
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile sphere shader: Unknown exception\n");
        return false;
    }

    return true;
}


/*
 * moldyn::SimpleSphereRenderer::Render
 */
bool moldyn::SimpleSphereRenderer::Render(view::CallRender3D& call) {

#ifdef DEBUG_GL_CALLBACK
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif

    // Checking for changed render mode
    auto currentRenderMode = static_cast<RenderMode>(this->renderModeParam.Param<param::EnumParam>()->Value());
    if (currentRenderMode != this->renderMode) {
        this->renderMode = currentRenderMode;
        if (!this->createResources()) {
            return false;
        }
    }

    // timer.BeginFrame();

    view::CallRender3D* cr3d = dynamic_cast<view::CallRender3D*>(&call);
    if (cr3d == nullptr) return false;

    float scaling = 1.0f;
    MultiParticleDataCall* mpdc = this->getData(static_cast<unsigned int>(cr3d->Time()), scaling);
    if (mpdc == nullptr) return false;

    // Update current state variables -----------------------------------------
    glGetFloatv(GL_VIEWPORT, this->curViewAttrib);
    this->curVpWidth = static_cast<int>(this->curViewAttrib[2]);
    this->curVpHeight = static_cast<int>(this->curViewAttrib[3]);
    if (this->curViewAttrib[2] < 1.0f) this->curViewAttrib[2] = 1.0f;
    if (this->curViewAttrib[3] < 1.0f) this->curViewAttrib[3] = 1.0f;
    this->curViewAttrib[2] = 2.0f / this->curViewAttrib[2];
    this->curViewAttrib[3] = 2.0f / this->curViewAttrib[3];

    const SIZE_T hash = mpdc->DataHash();
    const unsigned int frameID = mpdc->FrameID();

    // Check if we got a new data set
    this->stateInvalid = ((hash != this->oldHash) || (frameID != this->oldFrameID));

    this->oldHash = hash;
    this->oldFrameID = frameID;

    glPointSize(static_cast<GLfloat>(std::max(this->curVpWidth, this->curVpHeight)));

    this->getClipData(this->curClipDat, this->curClipCol);

    glEnable(GL_LIGHTING);
    glGetLightfv(GL_LIGHT0, GL_POSITION, this->curLightPos);
    glDisable(GL_LIGHTING);

    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> MV(&modelViewMatrix_column[0]);

    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> scaleMat;
    scaleMat.SetAt(0, 0, scaling);
    scaleMat.SetAt(1, 1, scaling);
    scaleMat.SetAt(2, 2, scaling);
    MV = MV * scaleMat;

    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> PM(&projMatrix_column[0]);
    this->curMVinv = MV;
    this->curMVinv.Invert();
    this->curMVP = PM * MV;
    this->curMVPinv = this->curMVP;
    this->curMVPinv.Invert();
    this->curMVPtransp = this->curMVP;
    this->curMVPtransp.Transpose();
    // ------------------------------------------------------------------------

    glDisable(GL_BLEND);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS); /// default - necessary for early depth test in fragment shader to work.

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    switch (currentRenderMode) {
    case (RenderMode::SIMPLE):
        return this->renderSimple(cr3d, mpdc);
    case (RenderMode::SIMPLE_CLUSTERED):
        return this->renderSimple(cr3d, mpdc);
    case (RenderMode::SIMPLE_GEO):
        return this->renderGeo(cr3d, mpdc);
    case (RenderMode::NG):
        return this->renderNG(cr3d, mpdc);
    case (RenderMode::NG_SPLAT):
        return this->renderNGSplat(cr3d, mpdc);
    case (RenderMode::NG_BUFFER_ARRAY):
        return this->renderNGBufferArray(cr3d, mpdc);
    case (RenderMode::AMBIENT_OCCLUSION):
        return this->renderAmbientOcclusion(cr3d, mpdc);
    default:
        break;
    }

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // Save some current data
    this->lastVpHeight = this->curVpHeight;
    this->lastVpWidth = this->curVpWidth;
    for (size_t i = 0; i < 4; ++i) {
        this->oldClipDat[i] = this->curClipDat[i];
    }

    // timer.EndFrame();

#ifdef DEBUG_GL_CALLBACK
    glDisable(GL_DEBUG_OUTPUT);
    glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif

    return false;
}


/*
 * moldyn::SimpleSphereRenderer::renderSimple
 */
bool moldyn::SimpleSphereRenderer::renderSimple(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc) {

    this->sphereShader.Enable();

    GLuint vertAttribLoc = glGetAttribLocationARB(this->sphereShader, "inVertex");
    GLuint colAttribLoc = glGetAttribLocationARB(this->sphereShader, "inColor");
    GLuint colIdxAttribLoc = glGetAttribLocationARB(this->sphereShader, "colIdx");

    glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, this->curViewAttrib);
    glUniform3fv(
        this->sphereShader.ParameterLocation("camIn"), 1, cr3d->GetCameraParameters()->Front().PeekComponents());
    glUniform3fv(
        this->sphereShader.ParameterLocation("camRight"), 1, cr3d->GetCameraParameters()->Right().PeekComponents());
    glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, cr3d->GetCameraParameters()->Up().PeekComponents());
    glUniform1f(
        this->sphereShader.ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphereShader.ParameterLocation("clipDat"), 1, this->curClipDat);
    glUniform4fv(this->sphereShader.ParameterLocation("clipCol"), 1, this->curClipCol);
    glUniform4fv(this->sphereShader.ParameterLocation("lpos"), 1, this->curLightPos);
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVinv"), 1, GL_FALSE, this->curMVinv.PeekComponents());
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVP"), 1, GL_FALSE, this->curMVP.PeekComponents());
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPinv"), 1, GL_FALSE, this->curMVPinv.PeekComponents());
    glUniformMatrix4fv(
        this->sphereShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, this->curMVPtransp.PeekComponents());

    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        GLuint vao, vb, cb;
        if (this->renderMode == RenderMode::SIMPLE_CLUSTERED) {
            parts.GetVAOs(vao, vb, cb);
            if (parts.IsVAO()) {
                glBindVertexArray(vao);
                this->setPointers<GLSLShader>(parts, this->sphereShader, vb, parts.GetVertexData(), vertAttribLoc, cb,
                    parts.GetColourData(), colAttribLoc, colIdxAttribLoc);
            }
        }
        if ((this->renderMode == RenderMode::SIMPLE) || (!parts.IsVAO())) {
            this->setPointers<GLSLShader>(parts, this->sphereShader, 0, parts.GetVertexData(), vertAttribLoc, 0,
                parts.GetColourData(), colAttribLoc, colIdxAttribLoc);
        }

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

        if (this->renderMode == RenderMode::SIMPLE_CLUSTERED) {
            if (parts.IsVAO()) {
                glBindVertexArray(0);             // vao
                glBindBuffer(GL_ARRAY_BUFFER, 0); // enabled in setPointers().
            }
        }
        glDisableVertexAttribArrayARB(vertAttribLoc);
        glDisableVertexAttribArrayARB(colAttribLoc);
        glDisableVertexAttribArrayARB(colIdxAttribLoc);
        glDisable(GL_TEXTURE_1D);
    }

    mpdc->Unlock();

    this->sphereShader.Disable();

    return true;
}


/*
 * moldyn::SimpleSphereRenderer::renderNG
 */
bool moldyn::SimpleSphereRenderer::renderNG(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc) {

#ifdef CHRONOTIMING
    std::vector<std::chrono::steady_clock::time_point> deltas;
    std::chrono::steady_clock::time_point before, after;
#endif

    // currBuf = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (colType != parts.GetColourDataType() || vertType != parts.GetVertexDataType()) {
            this->newShader = this->generateShader(parts);
        }

        this->newShader->Enable();

        glUniform4fv(this->newShader->ParameterLocation("viewAttr"), 1, this->curViewAttrib);
        glUniform3fv(
            this->newShader->ParameterLocation("camIn"), 1, cr3d->GetCameraParameters()->Front().PeekComponents());
        glUniform3fv(
            this->newShader->ParameterLocation("camRight"), 1, cr3d->GetCameraParameters()->Right().PeekComponents());
        glUniform3fv(
            this->newShader->ParameterLocation("camUp"), 1, cr3d->GetCameraParameters()->Up().PeekComponents());
        glUniform1f(this->newShader->ParameterLocation("scaling"),
            this->radiusScalingParam.Param<param::FloatParam>()->Value());
        glUniform4fv(this->newShader->ParameterLocation("clipDat"), 1, this->curClipDat);
        glUniform4fv(this->newShader->ParameterLocation("clipCol"), 1, this->curClipCol);
        glUniform4fv(this->newShader->ParameterLocation("lpos"), 1, this->curLightPos);
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVinv"), 1, GL_FALSE, this->curMVinv.PeekComponents());
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVP"), 1, GL_FALSE, this->curMVP.PeekComponents());
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVPinv"), 1, GL_FALSE, this->curMVPinv.PeekComponents());
        glUniformMatrix4fv(
            this->newShader->ParameterLocation("MVPtransp"), 1, GL_FALSE, this->curMVPtransp.PeekComponents());

        float minC = 0.0f, maxC = 0.0f;
        unsigned int colTabSize = 0;

        // colour
        switch (parts.GetColourDataType()) {
        case MultiParticleDataCall::Particles::COLDATA_NONE: {
            glUniform4f(this->newShader->ParameterLocation("globalCol"),
                static_cast<float>(parts.GetGlobalColour()[0]) / 255.0f,
                static_cast<float>(parts.GetGlobalColour()[1]) / 255.0f,
                static_cast<float>(parts.GetGlobalColour()[2]) / 255.0f, 1.0f);
        } break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
        case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
            glEnable(GL_TEXTURE_1D);
            view::CallGetTransferFunction* cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
            if ((cgtf != nullptr) && ((*cgtf)())) {
                glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                colTabSize = cgtf->TextureSize();
            }
            else {
                glBindTexture(GL_TEXTURE_1D, this->greyTF);
                colTabSize = 2;
            }
            glUniform1i(this->newShader->ParameterLocation("colTab"), 0);
            minC = parts.GetMinColourIndexValue();
            maxC = parts.GetMaxColourIndexValue();
        } break;
        default:
            glUniform4f(this->newShader->ParameterLocation("globalCol"), 0.5f, 0.5f, 0.5f, 1.0f);
            break;
        }

        // radius and position
        switch (parts.GetVertexDataType()) {
        case MultiParticleDataCall::Particles::VERTDATA_NONE:
            continue;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
            glUniform4f(this->newShader->ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC,
                float(colTabSize));
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            glUniform4f(this->newShader->ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
            break;
        case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
            glUniform4f(this->newShader->ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC,
                float(colTabSize));
        default:
            continue;
        }

        unsigned int colBytes, vertBytes, colStride, vertStride;
        bool interleaved;
        const bool staticData = this->useStaticDataParam.Param<param::BoolParam>()->Value();
        this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride, interleaved);

        // does all data reside interleaved in the same memory?
        if (interleaved) {
            if (staticData) {
                if (this->stateInvalid || (this->bufArray.GetNumChunks() == 0)) {
                    this->bufArray.SetDataWithSize(
                        parts.GetVertexData(), vertStride, vertStride, parts.GetCount(), (2 * 1024 * 1024 * 1024)); // 2 GB - khronos: Most implementations will let you allocate a size up to the limit of GPU memory.
                }
                const GLuint numChunks = this->bufArray.GetNumChunks();

                for (GLuint x = 0; x < numChunks; ++x) {
                    glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);
                    auto actualItems = this->bufArray.GetNumItems(x);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->bufArray.GetHandle(x));
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->bufArray.GetHandle(x));
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->bufArray.GetHandle(x), 0,
                        this->bufArray.GetMaxNumItemsPerChunk() * vertStride);
                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(actualItems));
                    this->bufArray.SignalCompletion();
                }
            }
            else {
                const GLuint numChunks = this->streamer.SetDataWithSize(
                    parts.GetVertexData(), vertStride, vertStride, parts.GetCount(), 3, (32 * 1024 * 1024)); // 32 MB
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->streamer.GetHandle());
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->streamer.GetHandle());

                for (GLuint x = 0; x < numChunks; ++x) {
                    GLuint numItems, sync;
                    GLsizeiptr dstOff, dstLen;
                    this->streamer.UploadChunk(x, numItems, sync, dstOff, dstLen);
                    // streamer.UploadChunk<float, float>(x, [](float f) -> float { return f + 100.0; },
                    //    numItems, sync, dstOff, dstLen);
                    // vislib::sys::Log::DefaultLog.WriteInfo("uploading chunk %u at %lu len %lu", x, dstOff, dstLen);
                    glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    glBindBufferRange(
                        GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->streamer.GetHandle(), dstOff, dstLen);
                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(numItems));
                    this->streamer.SignalCompletion(sync);
                }
            }
        }
        else {
            if (staticData) {
                if (this->stateInvalid || (this->bufArray.GetNumChunks() == 0)) {
                    this->bufArray.SetDataWithSize(
                        parts.GetVertexData(), vertStride, vertStride, parts.GetCount(), (2 * 1024 * 1024 * 1024)); // 2 GB - khronos: Most implementations will let you allocate a size up to the limit of GPU memory.
                    this->colBufArray.SetDataWithItems(parts.GetColourData(), colStride, colStride, parts.GetCount(),
                        this->bufArray.GetMaxNumItemsPerChunk());
                }
                const GLuint numChunks = this->bufArray.GetNumChunks();

                for (GLuint x = 0; x < numChunks; ++x) {
                    glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);
                    auto actualItems = this->bufArray.GetNumItems(x);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->bufArray.GetHandle(x));
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->bufArray.GetHandle(x));
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->bufArray.GetHandle(x), 0,
                        this->bufArray.GetMaxNumItemsPerChunk() * vertStride);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->colBufArray.GetHandle(x));
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOcolorBindingPoint, this->colBufArray.GetHandle(x));
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBOcolorBindingPoint, this->colBufArray.GetHandle(x), 0,
                        this->colBufArray.GetMaxNumItemsPerChunk() * colStride);
                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(actualItems));
                    this->bufArray.SignalCompletion();
                    this->colBufArray.SignalCompletion();
                }
            }
            else {
                const GLuint numChunks = this->streamer.SetDataWithSize(
                    parts.GetVertexData(), vertStride, vertStride, parts.GetCount(), 3, (32 * 1024 * 1024)); // 32 MB
                const GLuint colSize = this->colStreamer.SetDataWithItems(parts.GetColourData(), colStride, colStride,
                    parts.GetCount(), 3, this->streamer.GetMaxNumItemsPerChunk());
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->streamer.GetHandle());
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->streamer.GetHandle());
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
                        GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->streamer.GetHandle(), dstOff, dstLen);
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBOcolorBindingPoint, this->colStreamer.GetHandle(),
                        dstOff2, dstLen2);
                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(numItems));
                    this->streamer.SignalCompletion(sync);
                    this->colStreamer.SignalCompletion(sync2);
                }
            }
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glDisable(GL_TEXTURE_1D);

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


/*
 * moldyn::SimpleSphereRenderer::renderNGSplat
 */
bool moldyn::SimpleSphereRenderer::renderNGSplat(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc) {

    glDisable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
#if 1
    // maybe for blending against white, remove pre-mult alpha and use this:
    // @gl.blendFuncSeparate @gl.SRC_ALPHA, @gl.ONE_MINUS_SRC_ALPHA, @gl.ONE, @gl.ONE_MINUS_SRC_ALPHA
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
#else
    // glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
    glBlendFunc(GL_ONE, GL_ONE);
#endif

    glEnable(GL_POINT_SPRITE);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, theSingleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer);

    // currBuf = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (colType != parts.GetColourDataType() || vertType != parts.GetVertexDataType()) {
            this->newShader = this->generateShader(parts);
        }

        this->newShader->Enable();

        glUniform4fv(this->newShader->ParameterLocation("viewAttr"), 1, this->curViewAttrib);
        glUniform3fv(
            this->newShader->ParameterLocation("camIn"), 1, cr3d->GetCameraParameters()->Front().PeekComponents());
        glUniform3fv(
            this->newShader->ParameterLocation("camRight"), 1, cr3d->GetCameraParameters()->Right().PeekComponents());
        glUniform3fv(
            this->newShader->ParameterLocation("camUp"), 1, cr3d->GetCameraParameters()->Up().PeekComponents());
        glUniform1f(this->newShader->ParameterLocation("scaling"),
            this->radiusScalingParam.Param<param::FloatParam>()->Value());
        glUniform4fv(this->newShader->ParameterLocation("clipDat"), 1, this->curClipDat);
        glUniform4fv(this->newShader->ParameterLocation("clipCol"), 1, this->curClipCol);
        glUniform4fv(this->newShader->ParameterLocation("lpos"), 1, this->curLightPos);
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVinv"), 1, GL_FALSE, this->curMVinv.PeekComponents());
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVP"), 1, GL_FALSE, this->curMVP.PeekComponents());
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVPinv"), 1, GL_FALSE, this->curMVPinv.PeekComponents());
        glUniformMatrix4fv(
            this->newShader->ParameterLocation("MVPtransp"), 1, GL_FALSE, this->curMVPtransp.PeekComponents());
        glUniform1f(this->newShader->ParameterLocation("alphaScaling"),
            this->alphaScalingParam.Param<param::FloatParam>()->Value());
        glUniform1i(this->newShader->ParameterLocation("attenuateSubpixel"),
            this->attenuateSubpixelParam.Param<param::BoolParam>()->Value() ? 1 : 0);
        glUniform1f(this->newShader->ParameterLocation("zNear"), cr3d->GetCameraParameters()->NearClip());

        float minC = 0.0f, maxC = 0.0f;
        unsigned int colTabSize = 0;

        // colour
        switch (parts.GetColourDataType()) {
        case MultiParticleDataCall::Particles::COLDATA_NONE: {
            glUniform4f(this->newShader->ParameterLocation("globalCol"),
                static_cast<float>(parts.GetGlobalColour()[0]) / 255.0f,
                static_cast<float>(parts.GetGlobalColour()[1]) / 255.0f,
                static_cast<float>(parts.GetGlobalColour()[2]) / 255.0f, 1.0f);
        } break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
        case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
            glEnable(GL_TEXTURE_1D);
            view::CallGetTransferFunction* cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
            if ((cgtf != nullptr) && ((*cgtf)())) {
                glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                colTabSize = cgtf->TextureSize();
            }
            else {
                glBindTexture(GL_TEXTURE_1D, this->greyTF);
                colTabSize = 2;
            }
            glUniform1i(this->newShader->ParameterLocation("colTab"), 0);
            minC = parts.GetMinColourIndexValue();
            maxC = parts.GetMaxColourIndexValue();
        } break;
        default:
            glUniform4f(this->newShader->ParameterLocation("globalCol"), 0.5f, 0.5f, 0.5f, 1.0f);
            break;
        }

        // radius and position
        switch (parts.GetVertexDataType()) {
        case MultiParticleDataCall::Particles::VERTDATA_NONE:
            continue;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
            glUniform4f(this->newShader->ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC,
                float(colTabSize));
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            glUniform4f(this->newShader->ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
            break;
        default:
            continue;
        }

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
                void* mem = static_cast<char*>(theSingleMappedMem) + bufSize * currBuf;
                currCol = colStride == 0 ? currVert : currCol;
                // currCol = currCol == 0 ? currVert : currCol;
                const char* whence = currVert < currCol ? currVert : currCol;
                UINT64 vertsThisTime = vislib::math::Min(parts.GetCount() - vertCounter, numVerts);
                this->waitSingle(this->fences[currBuf]);
                // vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "memcopying %u bytes from %016"
                // PRIxPTR " to %016" PRIxPTR "\n", vertsThisTime * vertStride, whence, mem);
                memcpy(mem, whence, vertsThisTime * vertStride);
                glFlushMappedNamedBufferRangeEXT(theSingleBuffer, bufSize * currBuf, vertsThisTime * vertStride);
                // glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
                // glUniform1i(this->newShader->ParameterLocation("instanceOffset"), numVerts * currBuf);
                glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);

                // this->setPointers(parts, this->theSingleBuffer, reinterpret_cast<const void *>(currVert - whence),
                // this->theSingleBuffer, reinterpret_cast<const void *>(currCol - whence));
                // glBindBuffer(GL_ARRAY_BUFFER, 0);
                glBindBufferRange(
                    GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer, bufSize * currBuf, bufSize);
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vertsThisTime));
                // glDrawArraysInstanced(GL_POINTS, 0, 1, vertsThisTime);
                this->lockSingle(fences[currBuf]);

                currBuf = (currBuf + 1) % this->numBuffers;
                vertCounter += vertsThisTime;
                currVert += vertsThisTime * vertStride;
                currCol += vertsThisTime * colStride;
                // break;
            }
        }
        else {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "NGSplat mode does not support not interleaved data so far ...");
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glDisable(GL_TEXTURE_1D);

        newShader->Disable();
    }

    mpdc->Unlock();

    glDisable(GL_POINT_SPRITE);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    return true;
}


/*
 * moldyn::SimpleSphereRenderer::renderNGBufferArray
 */
bool moldyn::SimpleSphereRenderer::renderNGBufferArray(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc) {

    this->sphereShader.Enable();

    GLuint vertAttribLoc = glGetAttribLocationARB(this->sphereShader, "inVertex");
    GLuint colAttribLoc = glGetAttribLocationARB(this->sphereShader, "inColor");
    GLuint colIdxAttribLoc = glGetAttribLocationARB(this->sphereShader, "colIdx");

    glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, this->curViewAttrib);
    glUniform3fv(
        this->sphereShader.ParameterLocation("camIn"), 1, cr3d->GetCameraParameters()->Front().PeekComponents());
    glUniform3fv(
        this->sphereShader.ParameterLocation("camRight"), 1, cr3d->GetCameraParameters()->Right().PeekComponents());
    glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, cr3d->GetCameraParameters()->Up().PeekComponents());
    glUniform1f(
        this->sphereShader.ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphereShader.ParameterLocation("clipDat"), 1, this->curClipDat);
    glUniform4fv(this->sphereShader.ParameterLocation("clipCol"), 1, this->curClipCol);
    glUniform4fv(this->sphereShader.ParameterLocation("lpos"), 1, this->curLightPos);
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVinv"), 1, GL_FALSE, this->curMVinv.PeekComponents());
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVP"), 1, GL_FALSE, this->curMVP.PeekComponents());
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPinv"), 1, GL_FALSE, this->curMVPinv.PeekComponents());
    glUniformMatrix4fv(
        this->sphereShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, this->curMVPtransp.PeekComponents());

    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

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
                // vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "memcopying %u bytes from %016"
                // PRIxPTR " to %016" PRIxPTR "\n", vertsThisTime * vertStride, whence, mem);
                memcpy(mem, whence, vertsThisTime * vertStride);
                glFlushMappedNamedBufferRangeEXT(
                    this->theSingleBuffer, numVerts * this->currBuf, vertsThisTime * vertStride);
                // glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
                this->setPointers<GLSLShader>(parts, this->sphereShader, this->theSingleBuffer,
                    reinterpret_cast<const void*>(currVert - whence), vertAttribLoc, this->theSingleBuffer,
                    reinterpret_cast<const void*>(currCol - whence), colAttribLoc, colIdxAttribLoc);
                glDrawArrays(
                    GL_POINTS, static_cast<GLint>(numVerts * this->currBuf), static_cast<GLsizei>(vertsThisTime));
                this->lockSingle(this->fences[this->currBuf]);

                this->currBuf = (this->currBuf + 1) % this->numBuffers;
                vertCounter += vertsThisTime;
                currVert += vertsThisTime * vertStride;
                currCol += vertsThisTime * colStride;
            }
        }
        else {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "NGBufferArray mode does not support not interleaved data so far ...");
        }

        glBindBuffer(GL_ARRAY_BUFFER, 0); // enabled in setPointers()
        glDisableVertexAttribArrayARB(vertAttribLoc);
        glDisableVertexAttribArrayARB(colAttribLoc);
        glDisableVertexAttribArrayARB(colIdxAttribLoc);
        glDisable(GL_TEXTURE_1D);
    }

    mpdc->Unlock();

    this->sphereShader.Disable();

    return true;
}


/*
 * moldyn::SimpleSphereRenderer::renderGeo
 */
bool moldyn::SimpleSphereRenderer::renderGeo(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc) {

    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    // glDepthFunc(GL_LEQUAL); // Default GL_LESS works, too?

    /// If enabled and a vertex shader is active, it specifies that the GL will choose between front and
    /// back colors based on the polygon's face direction of which the vertex being shaded is a part.
    /// It has no effect on points or lines.
    // glEnable(GL_VERTEX_PROGRAM_TWO_SIDE); // ! Has significant negative performance impact ....

    this->sphereGeometryShader.Enable();

    GLuint vertAttribLoc = glGetAttribLocationARB(this->sphereGeometryShader, "inVertex");
    GLuint colAttribLoc = glGetAttribLocationARB(this->sphereGeometryShader, "inColor");
    GLuint colIdxAttribLoc = glGetAttribLocationARB(this->sphereGeometryShader, "colIdx");

    // Set shader variables
    glUniform4fv(this->sphereGeometryShader.ParameterLocation("viewAttr"), 1, this->curViewAttrib);
    glUniform3fv(this->sphereGeometryShader.ParameterLocation("camIn"), 1,
        cr3d->GetCameraParameters()->Front().PeekComponents());
    glUniform3fv(this->sphereGeometryShader.ParameterLocation("camRight"), 1,
        cr3d->GetCameraParameters()->Right().PeekComponents());
    glUniform3fv(
        this->sphereGeometryShader.ParameterLocation("camUp"), 1, cr3d->GetCameraParameters()->Up().PeekComponents());
    glUniform1f(
        this->sphereGeometryShader.ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphereGeometryShader.ParameterLocation("clipDat"), 1, this->curClipDat);
    glUniform4fv(this->sphereGeometryShader.ParameterLocation("clipCol"), 1, this->curClipCol);
    glUniform4fv(this->sphereGeometryShader.ParameterLocation("lpos"), 1, this->curLightPos);
    glUniformMatrix4fv(
        this->sphereGeometryShader.ParameterLocation("MVinv"), 1, GL_FALSE, this->curMVinv.PeekComponents());
    glUniformMatrix4fv(this->sphereGeometryShader.ParameterLocation("MVP"), 1, GL_FALSE, this->curMVP.PeekComponents());
    glUniformMatrix4fv(
        this->sphereGeometryShader.ParameterLocation("MVPinv"), 1, GL_FALSE, this->curMVPinv.PeekComponents());
    glUniformMatrix4fv(
        this->sphereGeometryShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, this->curMVPtransp.PeekComponents());

    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        this->setPointers<GLSLGeometryShader>(parts, this->sphereGeometryShader, 0, parts.GetVertexData(),
            vertAttribLoc, 0, parts.GetColourData(), colAttribLoc, colIdxAttribLoc);

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

        glDisableVertexAttribArray(vertAttribLoc);
        glDisableVertexAttribArray(colAttribLoc);
        glDisableVertexAttribArray(colIdxAttribLoc);
        glDisable(GL_TEXTURE_1D);
    }

    mpdc->Unlock();

    this->sphereGeometryShader.Disable();

    // glDisable(GL_VERTEX_PROGRAM_TWO_SIDE);
    // glDepthFunc(GL_LESS); // default

    return true;
}


/*
 * moldyn::SimpleSphereRenderer::renderAmbientOcclusion
 */
bool moldyn::SimpleSphereRenderer::renderAmbientOcclusion(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc) {

    // We need to regenerate the shader if certain settings are changed
    if (this->enableAOSlot.IsDirty() || this->enableLightingSlot.IsDirty() || this->aoConeApexSlot.IsDirty()) {
        this->aoConeApexSlot.ResetDirty();
        this->enableLightingSlot.ResetDirty();
        this->enableAOSlot.ResetDirty();

        this->rebuildShader();
    }

    // Rebuild the GBuffer if neccessary
    this->rebuildGBuffer();

    // Rebuild and reupload working data if neccessary
    this->rebuildWorkingData(cr3d, mpdc);

    GLint prevFBO;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFBO);

    glBindFramebuffer(GL_FRAMEBUFFER, this->gBuffer.fbo);
    checkGLError;
    GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, bufs);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    checkGLError;

    glBindFragDataLocation(sphereShader.ProgramHandle(), 0, "outColor");
    checkGLError;
    glBindFragDataLocation(sphereShader.ProgramHandle(), 1, "outNormal");
    checkGLError;

    // Render the particles' geometry
    this->renderParticlesGeometry(cr3d, mpdc);

    glBindFramebuffer(GL_FRAMEBUFFER, prevFBO);

    this->renderDeferredPass(cr3d);

    return true;
}


/*
 * moldyn::SimpleSphereRenderer::setPointers
 */
template <typename T>
void moldyn::SimpleSphereRenderer::setPointers(MultiParticleDataCall::Particles& parts, T& shader, GLuint vertBuf,
    const void* vertPtr, GLuint vertAttribLoc, GLuint colBuf, const void* colPtr, GLuint colAttribLoc,
    GLuint colIdxAttribLoc) {

    float minC = 0.0f, maxC = 0.0f;
    unsigned int colTabSize = 0;

    // colour
    glBindBuffer(GL_ARRAY_BUFFER, colBuf);
    switch (parts.GetColourDataType()) {
    case MultiParticleDataCall::Particles::COLDATA_NONE: {
        const unsigned char* gc = parts.GetGlobalColour();
        glVertexAttrib3d(colAttribLoc, static_cast<double>(gc[0]) / 255.0, static_cast<double>(gc[1]) / 255.0,
            static_cast<double>(gc[2]) / 255.0);
    } break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
        glEnableVertexAttribArray(colAttribLoc);
        glVertexAttribPointer(colAttribLoc, 3, GL_UNSIGNED_BYTE, GL_TRUE, parts.GetColourDataStride(), colPtr);
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        glEnableVertexAttribArray(colAttribLoc);
        glVertexAttribPointer(colAttribLoc, 4, GL_UNSIGNED_BYTE, GL_TRUE, parts.GetColourDataStride(), colPtr);
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
        glEnableVertexAttribArray(colAttribLoc);
        glVertexAttribPointer(colAttribLoc, 3, GL_FLOAT, GL_TRUE, parts.GetColourDataStride(), colPtr);
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
        glEnableVertexAttribArray(colAttribLoc);
        glVertexAttribPointer(colAttribLoc, 4, GL_FLOAT, GL_TRUE, parts.GetColourDataStride(), colPtr);
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
    case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
        glEnableVertexAttribArray(colIdxAttribLoc);
        if (parts.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
            glVertexAttribPointer(colIdxAttribLoc, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), colPtr);
        }
        else {
            glVertexAttribPointer(colIdxAttribLoc, 1, GL_DOUBLE, GL_FALSE, parts.GetColourDataStride(), colPtr);
        }

        glEnable(GL_TEXTURE_1D);

        view::CallGetTransferFunction* cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
        if ((cgtf != nullptr) && ((*cgtf)())) {
            glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
            colTabSize = cgtf->TextureSize();
        }
        else {
            glBindTexture(GL_TEXTURE_1D, this->greyTF);
            colTabSize = 2;
        }

        glUniform1i(shader.ParameterLocation("colTab"), 0);
        minC = parts.GetMinColourIndexValue();
        maxC = parts.GetMaxColourIndexValue();
    } break;
    case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA:
        glEnableVertexAttribArray(colAttribLoc);
        glVertexAttribPointer(colAttribLoc, 4, GL_UNSIGNED_SHORT, GL_TRUE, parts.GetColourDataStride(), colPtr);
        break;
    default:
        glVertexAttrib3f(colAttribLoc, 0.5f, 0.5f, 0.5f);
        break;
    }

    // radius and position
    glBindBuffer(GL_ARRAY_BUFFER, vertBuf);
    switch (parts.GetVertexDataType()) {
    case MultiParticleDataCall::Particles::VERTDATA_NONE:
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        glEnableVertexAttribArray(vertAttribLoc);
        glVertexAttribPointer(vertAttribLoc, 3, GL_FLOAT, GL_FALSE, parts.GetVertexDataStride(), vertPtr);
        glUniform4f(shader.ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        glEnableVertexAttribArray(vertAttribLoc);
        glVertexAttribPointer(vertAttribLoc, 4, GL_FLOAT, GL_FALSE, parts.GetVertexDataStride(), vertPtr);
        glUniform4f(shader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
        break;
    case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
        glEnableVertexAttribArray(vertAttribLoc);
        glVertexAttribPointer(vertAttribLoc, 3, GL_DOUBLE, GL_FALSE, parts.GetVertexDataStride(), vertPtr);
        glUniform4f(shader.ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
        break;
    case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
        glEnableVertexAttribArray(vertAttribLoc);
        glVertexAttribPointer(vertAttribLoc, 3, GL_SHORT, GL_FALSE, parts.GetVertexDataStride(), vertPtr);
        glUniform4f(shader.ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
        break;
    default:
        break;
    }
}


/*
 * moldyn::SimpleSphereRenderer::makeColorString
 */
bool moldyn::SimpleSphereRenderer::makeColorString(
    MultiParticleDataCall::Particles& parts, std::string& code, std::string& declaration, bool interleaved) {

    bool ret = true;

    switch (parts.GetColourDataType()) {
    case MultiParticleDataCall::Particles::COLDATA_NONE:
        declaration = "";
        code = "    theColor = globalCol;\n";
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
        vislib::sys::Log::DefaultLog.WriteError("Cannot pack an unaligned RGB color into an SSBO! Giving up.");
        ret = false;
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        declaration = "    uint color;\n";
        if (interleaved) {
            code = "    theColor = unpackUnorm4x8(theBuffer[" NGS_THE_INSTANCE "+ instanceOffset].color);\n";
        }
        else {
            code = "    theColor = unpackUnorm4x8(theColBuffer[" NGS_THE_INSTANCE "+ instanceOffset].color);\n";
        }
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
        declaration = "    float r; float g; float b;\n";
        if (interleaved) {
            code = "    theColor = vec4(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].r,\n"
                "                       theBuffer[" NGS_THE_INSTANCE " + instanceOffset].g,\n"
                "                       theBuffer[" NGS_THE_INSTANCE " + instanceOffset].b, 1.0); \n";
        }
        else {
            code = "    theColor = vec4(theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].r,\n"
                "                       theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].g,\n"
                "                       theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].b, 1.0); \n";
        }
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
        declaration = "    float r; float g; float b; float a;\n";
        if (interleaved) {
            code = "    theColor = vec4(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].r,\n"
                "                       theBuffer[" NGS_THE_INSTANCE " + instanceOffset].g,\n"
                "                       theBuffer[" NGS_THE_INSTANCE " + instanceOffset].b,\n"
                "                       theBuffer[" NGS_THE_INSTANCE " + instanceOffset].a); \n";
        }
        else {
            code = "    theColor = vec4(theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].r,\n"
                "                       theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].g,\n"
                "                       theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].b,\n"
                "                       theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].a); \n";
        }
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
        declaration = "    float colorIndex;\n";
        if (interleaved) {
            code = "    theColIdx = theBuffer[" NGS_THE_INSTANCE " + instanceOffset].colorIndex; \n";
        }
        else {
            code = "    theColIdx = theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].colorIndex; \n";
        }
    } break;
    case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
        declaration = "    double colorIndex;\n";
        if (interleaved) {
            code = "    theColIdx = float(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].colorIndex); \n";
        }
        else {
            code = "    theColIdx = float(theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].colorIndex); \n";
        }
    } break;
    case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA: {
        declaration = "    uint col1; uint col2;\n";
        if (interleaved) {
            code = "    theColor.xy = unpackUnorm2x16(theBuffer[" NGS_THE_INSTANCE "+ instanceOffset].col1);\n"
                "    theColor.zw = unpackUnorm2x16(theBuffer[" NGS_THE_INSTANCE "+ instanceOffset].col2);\n";
        }
        else {
            code = "    theColor.xy = unpackUnorm2x16(theColBuffer[" NGS_THE_INSTANCE "+ instanceOffset].col1);\n"
                "    theColor.zw = unpackUnorm2x16(theColBuffer[" NGS_THE_INSTANCE "+ instanceOffset].col2);\n";
        }
    } break;
    default:
        declaration = "";
        code = "    theColor = globalCol;\n";
        break;
    }
    // code = "    theColor = vec4(0.2, 0.7, 1.0, 1.0);";

    return ret;
}


/*
 * moldyn::SimpleSphereRenderer::makeVertexString
 */
bool moldyn::SimpleSphereRenderer::makeVertexString(
    MultiParticleDataCall::Particles& parts, std::string& code, std::string& declaration, bool interleaved) {

    bool ret = true;

    switch (parts.GetVertexDataType()) {
    case MultiParticleDataCall::Particles::VERTDATA_NONE:
        declaration = "";
        code = "";
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        declaration = "    float posX; float posY; float posZ;\n";
        if (interleaved) {
            code = "    inPos = vec4(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posX,\n"
                "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posY,\n"
                "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posZ, 1.0); \n"
                "    rad = CONSTRAD;";
        }
        else {
            code = "    inPos = vec4(thePosBuffer[" NGS_THE_INSTANCE " + instanceOffset].posX,\n"
                "                 thePosBuffer[" NGS_THE_INSTANCE " + instanceOffset].posY,\n"
                "                 thePosBuffer[" NGS_THE_INSTANCE " + instanceOffset].posZ, 1.0); \n"
                "    rad = CONSTRAD;";
        }
        break;
    case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
        declaration = "    double posX; double posY; double posZ;\n";
        if (interleaved) {
            code = "    inPos = vec4(float(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posX),\n"
                "                 float(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posY),\n"
                "                 float(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posZ), 1.0); \n"
                "    rad = CONSTRAD;";
        }
        else {
            code = "    inPos = vec4(float(thePosBuffer[" NGS_THE_INSTANCE " + instanceOffset].posX),\n"
                "                 float(thePosBuffer[" NGS_THE_INSTANCE " + instanceOffset].posY),\n"
                "                 float(thePosBuffer[" NGS_THE_INSTANCE " + instanceOffset].posZ), 1.0); \n"
                "    rad = CONSTRAD;";
        }
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        declaration = "    float posX; float posY; float posZ; float posR;\n";
        if (interleaved) {
            code = "    inPos = vec4(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posX,\n"
                "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posY,\n"
                "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posZ, 1.0); \n"
                "    rad = theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posR;";
        }
        else {
            code = "    inPos = vec4(thePosBuffer[" NGS_THE_INSTANCE " + instanceOffset].posX,\n"
                "                 thePosBuffer[" NGS_THE_INSTANCE " + instanceOffset].posY,\n"
                "                 thePosBuffer[" NGS_THE_INSTANCE " + instanceOffset].posZ, 1.0); \n"
                "    rad = thePosBuffer[" NGS_THE_INSTANCE " + instanceOffset].posR;";
        }
        break;
    default:
        declaration = "";
        code = "";
        break;
    }

    return ret;
}


/*
 * moldyn::SimpleSphereRenderer::makeShader
 */
std::shared_ptr<GLSLShader> moldyn::SimpleSphereRenderer::makeShader(
    vislib::SmartPtr<ShaderSource> vert, vislib::SmartPtr<ShaderSource> frag) {

    std::shared_ptr<GLSLShader> sh = std::make_shared<GLSLShader>(GLSLShader());
    try {
        if (!sh->Create(vert->Code(), vert->Count(), frag->Code(), frag->Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to compile sphere shader: Unknown error\n");
            return nullptr;
        }

    }
    catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return nullptr;
    }
    catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile sphere shader: %s\n", e.GetMsgA());
        return nullptr;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile sphere shader: Unknown exception\n");
        return nullptr;
    }
    return sh;
}


/*
 * moldyn::SimpleSphereRenderer::generateShader
 */
std::shared_ptr<vislib::graphics::gl::GLSLShader> moldyn::SimpleSphereRenderer::generateShader(
    MultiParticleDataCall::Particles& parts) {

    int c = parts.GetColourDataType();
    int p = parts.GetVertexDataType();

    unsigned int colBytes, vertBytes, colStride, vertStride;
    bool interleaved;
    this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride, interleaved);

    shaderMap::iterator i = this->theShaders.find(std::make_tuple(c, p, interleaved));
    if (i == this->theShaders.end()) {
        // instance()->ShaderSourceFactory().MakeShaderSource()

        vislib::SmartPtr<ShaderSource> v2 = new ShaderSource(*this->vertShader);
        vislib::SmartPtr<ShaderSource::Snippet> codeSnip, declarationSnip;
        std::string vertCode, colCode, vertDecl, colDecl, decl;
        makeVertexString(parts, vertCode, vertDecl, interleaved);
        makeColorString(parts, colCode, colDecl, interleaved);

        if (interleaved) {

            decl = "\nstruct SphereParams {\n";

            if (parts.GetColourData() < parts.GetVertexData()) {
                decl += colDecl;
                decl += vertDecl;
            }
            else {
                decl += vertDecl;
                decl += colDecl;
            }
            decl += "};\n";

            decl += "layout(" NGS_THE_ALIGNMENT ", binding = " + std::to_string(SSBObindingPoint) +
                ") buffer shader_data {\n"
                "    SphereParams theBuffer[];\n"
                // flat float version
                //"    float theBuffer[];\n"
                "};\n";

        }
        else {
            // we seem to have separate buffers for vertex and color data

            decl = "\nstruct SpherePosParams {\n" + vertDecl;
            decl +="};\n";

            decl += "\nstruct SphereColParams {\n" + colDecl;
            decl += "};\n";

            decl += "layout(" NGS_THE_ALIGNMENT ", binding = " + std::to_string(SSBObindingPoint) +
                ") buffer shader_data {\n"
                "    SpherePosParams thePosBuffer[];\n"
                "};\n";
            decl += "layout(" NGS_THE_ALIGNMENT ", binding = " + std::to_string(SSBOcolorBindingPoint) +
                ") buffer shader_data2 {\n"
                "    SphereColParams theColBuffer[];\n"
                "};\n";
        }
        std::string code = "\n";
        code += colCode;
        code += vertCode;
        declarationSnip = new ShaderSource::StringSnippet(decl.c_str());
        codeSnip = new ShaderSource::StringSnippet(code.c_str());

        /// Generated shader declaration snippet is inserted between 2nd and 3rd snippet (after
        /// ngsphere_vert_attributes.glsl)
        v2->Insert(3, declarationSnip);
        /// Generated shader code snippet is inserted between 4th and 5th snippet (after ngsphere_vert_mainstart.glsl)
        /// => consider new index through first Insertion!
        v2->Insert(5, codeSnip);
        // std::string s(v2->WholeCode());

        vislib::SmartPtr<ShaderSource> vss(v2);
        this->theShaders.emplace(std::make_pair(std::make_tuple(c, p, interleaved), makeShader(v2, this->fragShader)));
        i = this->theShaders.find(std::make_tuple(c, p, interleaved));
    }
    return i->second;
}


/*
 * moldyn::SimpleSphereRenderer::getBytesAndStride
 */
void moldyn::SimpleSphereRenderer::getBytesAndStride(MultiParticleDataCall::Particles& parts, unsigned int& colBytes,
    unsigned int& vertBytes, unsigned int& colStride, unsigned int& vertStride, bool& interleaved) {

    vertBytes = MultiParticleDataCall::Particles::VertexDataSize[parts.GetVertexDataType()];
    colBytes = MultiParticleDataCall::Particles::ColorDataSize[parts.GetColourDataType()];

    colStride = parts.GetColourDataStride();
    colStride = colStride < colBytes ? colBytes : colStride;
    vertStride = parts.GetVertexDataStride();
    vertStride = vertStride < vertBytes ? vertBytes : vertStride;

    interleaved = (std::abs(reinterpret_cast<const ptrdiff_t>(parts.GetColourData()) -
        reinterpret_cast<const ptrdiff_t>(parts.GetVertexData())) <= vertStride &&
        vertStride == colStride) ||
        colStride == 0;
}


/*
 * moldyn::SimpleSphereRenderer::lockSingle
 */
void moldyn::SimpleSphereRenderer::lockSingle(GLsync& syncObj) {
    if (syncObj) {
        glDeleteSync(syncObj);
    }
    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}


/*
 * moldyn::SimpleSphereRenderer::waitSingle
 */
void moldyn::SimpleSphereRenderer::waitSingle(GLsync& syncObj) {
    if (syncObj) {
        while (1) {
            GLenum wait = glClientWaitSync(syncObj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                return;
            }
        }
    }
}


// Ambient Occlusion ----------------------------------------------------------


/*
 * moldyn::SimpleSphereRenderer::rebuildShader
 */
bool moldyn::SimpleSphereRenderer::rebuildShader() {
    vislib::graphics::gl::ShaderSource vert, frag;
    core::utility::ShaderSourceFactory& factory = instance()->ShaderSourceFactory();

    // Create the sphere shader if neccessary
    if (!vislib::graphics::gl::GLSLShader::IsValidHandle(this->sphereShader) &&
        !megamol::core::utility::InitializeShader(&factory, this->sphereShader, "mdao2::vertex", "mdao2::fragment")) {
        return false;
    }

    if (!vislib::graphics::gl::GLSLGeometryShader::IsValidHandle(this->sphereGeometryShader) &&
        !megamol::core::utility::InitializeShader(&factory, this->sphereGeometryShader, "mdao2::geometry::vertex",
            "mdao2::fragment", "mdao2::geometry::geometry")) {
        return false;
    }


    // Load the vertex shader
    if (!factory.MakeShaderSource("mdao2::deferred::vertex", vert)) return false;

    bool enableAO = this->enableAOSlot.Param<megamol::core::param::BoolParam>()->Value();
    bool enableLighting = this->enableLightingSlot.Param<megamol::core::param::BoolParam>()->Value();

    frag.Append(factory.MakeShaderSnippet("mdao2::deferred::fragment::Main"));

    if (enableLighting) {
        frag.Append(factory.MakeShaderSnippet("mdao2::deferred::fragment::Lighting"));
    }
    else {
        frag.Append(factory.MakeShaderSnippet("mdao2::deferred::fragment::LightingStub"));
    }

    if (enableAO) {
        float apex = this->aoConeApexSlot.Param<megamol::core::param::FloatParam>()->Value();

        std::vector<vislib::math::Vector<float, 4>> directions;
        this->generate3ConeDirections(directions, apex * static_cast<float>(M_PI) / 180.0f);
        std::string directionsCode = this->generateDirectionShaderArrayString(directions, "coneDirs");

        vislib::graphics::gl::ShaderSource::StringSnippet* dirSnippet =
            new vislib::graphics::gl::ShaderSource::StringSnippet(directionsCode.c_str());
        frag.Append(dirSnippet);

        frag.Append(factory.MakeShaderSnippet("mdao2::deferred::fragment::AmbientOcclusion"));
    }
    else {
        frag.Append(factory.MakeShaderSnippet("mdao2::deferred::fragment::AmbientOcclusionStub"));
    }

    try {
        this->lightingShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count());
    }
    catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile mdao shader: %s", ce.GetMsg());
        return false;
    }

    return true;
}


/*
 * moldyn::SimpleSphereRenderer::rebuildGBuffer
 */
bool moldyn::SimpleSphereRenderer::rebuildGBuffer() {
    if ((this->curVpWidth == this->lastVpWidth) && (this->curVpHeight == this->lastVpHeight) &&
        !this->useHPTexturesSlot.IsDirty()) {
        return true;
    }

    this->useHPTexturesSlot.ResetDirty();

    bool highPrecision = this->useHPTexturesSlot.Param<megamol::core::param::BoolParam>()->Value();

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
    checkGLError;
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->gBuffer.color, 0);
    checkGLError;
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, this->gBuffer.normals, 0);
    checkGLError;
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->gBuffer.depth, 0);
    checkGLError;

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "Framebuffer NOT complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, prevFBO);

    return true;
}


/*
 * moldyn::SimpleSphereRenderer::rebuildWorkingData
 */
void moldyn::SimpleSphereRenderer::rebuildWorkingData(
    megamol::core::view::CallRender3D* cr3d, megamol::core::moldyn::MultiParticleDataCall* dataCall) {

    // Upload new data if neccessary
    if (stateInvalid) {
        unsigned int partsCount = dataCall->GetParticleListCount();

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
            uploadDataToGPU(this->gpuData[i], dataCall->AccessParticles(i));
        }
    }

    // Check if voxelization is even needed
    if (this->enableAOSlot.IsDirty()) {
        bool enableAO = this->enableAOSlot.Param<megamol::core::param::BoolParam>()->Value();

        if (!enableAO && this->volGen != nullptr) {
            delete this->volGen;
            this->volGen = nullptr;
        }
        if (enableAO && this->volGen == nullptr) {
            this->volGen = new megamol::core::utility::MDAO2VolumeGenerator();
            this->volGen->SetShaderSourceFactory(&instance()->ShaderSourceFactory());
            this->volGen->Init();
        }
    }

    // Recreate the volume if neccessary
    bool equalClipData = true;
    for (size_t i = 0; i < 4; ++i) {
        if (this->oldClipDat[i] != this->curClipDat[i]) {
            equalClipData = false;
            break;
        }
    }
    if (volGen != nullptr &&
        (stateInvalid || this->enableAOSlot.IsDirty() || this->aoVolSizeSlot.IsDirty() || !equalClipData)) {
        int volSize = this->aoVolSizeSlot.Param<megamol::core::param::IntParam>()->Value();

        const vislib::math::Cuboid<float>& cube = cr3d->AccessBoundingBoxes().ObjectSpaceClipBox();
        vislib::math::Dimension<float, 3> dims = cube.GetSize();

        // Calculate the extensions of the volume by using the specified number of voxels for the longest edge
        float longestEdge = cube.LongestEdge();
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

        this->volGen->StartInsertion(cube, vislib::math::Vector<float, 4>(this->curClipDat[0], this->curClipDat[1],
            this->curClipDat[2], this->curClipDat[3]));
        for (unsigned int i = 0; i < this->gpuData.size(); ++i) {
            float globalRadius = 0.0f;
            if (dataCall->AccessParticles(i).GetVertexDataType() !=
                megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)
                globalRadius = dataCall->AccessParticles(i).GetGlobalRadius();
            this->volGen->InsertParticles(static_cast<unsigned int>(dataCall->AccessParticles(i).GetCount()),
                globalRadius, this->gpuData[i].vertexArray);
        }
        this->volGen->EndInsertion();

        this->volGen->RecreateMipmap();
    }

    // reset shotter for legacy opengl crap
    glBindVertexArray(0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    this->enableAOSlot.ResetDirty();
    this->aoVolSizeSlot.ResetDirty();
}


/*
 * moldyn::SimpleSphereRenderer::renderParticlesGeometry
 */
void moldyn::SimpleSphereRenderer::renderParticlesGeometry(
    megamol::core::view::CallRender3D* cr3d, megamol::core::moldyn::MultiParticleDataCall* dataCall) {
    bool highPrecision = this->useHPTexturesSlot.Param<megamol::core::param::BoolParam>()->Value();

    bool useGeo = this->enableGeometryShader.Param<core::param::BoolParam>()->Value();
    vislib::graphics::gl::GLSLShader& theShader = useGeo ? this->sphereGeometryShader : this->sphereShader;

    theShader.Enable();

    glUniformMatrix4fv(theShader.ParameterLocation("inMvp"), 1, GL_FALSE, this->curMVP.PeekComponents());
    glUniformMatrix4fv(theShader.ParameterLocation("inMvpInverse"), 1, GL_FALSE, this->curMVPinv.PeekComponents());
    glUniformMatrix4fv(theShader.ParameterLocation("inMvpTrans"), 1, GL_FALSE, this->curMVPtransp.PeekComponents());

    glUniform1f(theShader.ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());

    theShader.SetParameterArray4("inViewAttr", 1, this->curViewAttrib);
    theShader.SetParameterArray3("inCamFront", 1, cr3d->GetCameraParameters()->Front().PeekComponents());
    theShader.SetParameterArray3("inCamRight", 1, cr3d->GetCameraParameters()->Right().PeekComponents());
    theShader.SetParameterArray3("inCamUp", 1, cr3d->GetCameraParameters()->Up().PeekComponents());
    theShader.SetParameterArray4("inCamPos", 1, this->curMVinv.GetColumn(3).PeekComponents());

    theShader.SetParameterArray4("inClipDat", 1, this->curClipDat);
    theShader.SetParameterArray4("inClipCol", 1, this->curClipCol);

    theShader.SetParameter("inUseHighPrecision", highPrecision);

    for (unsigned int i = 0; i < gpuData.size(); ++i) {
        glBindVertexArray(gpuData[i].vertexArray);

        core::moldyn::SimpleSphericalParticles& parts = dataCall->AccessParticles(i);

        float globalRadius = 0.0f;
        if (parts.GetVertexDataType() != megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)
            globalRadius = parts.GetGlobalRadius();

        theShader.SetParameter("inGlobalRadius", globalRadius);

        bool useGlobalColor = false;
        if (parts.GetColourDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE) {
            useGlobalColor = true;
            const unsigned char* globalColor = parts.GetGlobalColour();
            float globalColorFlt[4] = { static_cast<float>(globalColor[0]) / 255.0f,
                static_cast<float>(globalColor[1]) / 255.0f, static_cast<float>(globalColor[2]) / 255.0f, 1.0f };
            theShader.SetParameterArray4("inGlobalColor", 1, globalColorFlt);
        }
        theShader.SetParameter("inUseGlobalColor", useGlobalColor);

        bool useTransferFunction = false;
        if (parts.GetColourDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
            useTransferFunction = true;
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_1D, getTransferFunctionHandle());
            theShader.SetParameter("inTransferFunction", static_cast<int>(0));
            float tfRange[2] = { parts.GetMinColourIndexValue(), parts.GetMaxColourIndexValue() };
            theShader.SetParameterArray2("inIndexRange", 1, tfRange);
        }
        theShader.SetParameter("inUseTransferFunction", useTransferFunction);

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(dataCall->AccessParticles(i).GetCount()));
    }

    glBindVertexArray(0);

    theShader.Disable();
}


/*
 * moldyn::SimpleSphereRenderer::renderDeferredPass
 */
void moldyn::SimpleSphereRenderer::renderDeferredPass(megamol::core::view::CallRender3D* cr3d) {
    bool enableAO = this->enableAOSlot.Param<megamol::core::param::BoolParam>()->Value();
    bool enableLighting = this->enableLightingSlot.Param<megamol::core::param::BoolParam>()->Value();
    bool highPrecision = this->useHPTexturesSlot.Param<megamol::core::param::BoolParam>()->Value();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, this->gBuffer.depth);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->gBuffer.normals);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->gBuffer.color);

    this->lightingShader.Enable();

    this->lightingShader.SetParameter("inWidth", static_cast<float>(this->curVpWidth));
    this->lightingShader.SetParameter("inHeight", static_cast<float>(this->curVpHeight));
    glUniformMatrix4fv(
        this->lightingShader.ParameterLocation("inMvpInverse"), 1, GL_FALSE, this->curMVPinv.PeekComponents());
    this->lightingShader.SetParameter("inColorTex", static_cast<int>(0));
    this->lightingShader.SetParameter("inNormalsTex", static_cast<int>(1));
    this->lightingShader.SetParameter("inDepthTex", static_cast<int>(2));

    this->lightingShader.SetParameter("inUseHighPrecision", highPrecision);

    if (enableLighting) {
        vislib::math::Vector<float, 4> lightDir = this->curMVinv * vislib::math::Vector<float, 4>(this->curLightPos);
        lightDir.Normalise();
        this->lightingShader.SetParameterArray3("inObjLightDir", 1, lightDir.PeekComponents());
        this->lightingShader.SetParameterArray3("inObjCamPos", 1, this->curMVinv.GetColumn(3).PeekComponents());
    }

    if (enableAO) {
        float aoOffset = this->aoOffsetSlot.Param<megamol::core::param::FloatParam>()->Value();
        float aoStrength = this->aoStrengthSlot.Param<megamol::core::param::FloatParam>()->Value();
        float aoConeLength = this->aoConeLengthSlot.Param<megamol::core::param::FloatParam>()->Value();
        if (volGen != nullptr) {
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_3D, this->volGen->GetVolumeTextureHandle());
            glActiveTexture(GL_TEXTURE0);
        }
        this->lightingShader.SetParameter("inAOOffset", aoOffset);
        this->lightingShader.SetParameter("inDensityTex", static_cast<int>(3));
        this->lightingShader.SetParameter("inAOStrength", aoStrength);
        this->lightingShader.SetParameter("inAOConeLength", aoConeLength);
        this->lightingShader.SetParameter("inAmbVolShortestEdge", this->ambConeConstants[0]);
        this->lightingShader.SetParameter("inAmbVolMaxLod", this->ambConeConstants[1]);
        this->lightingShader.SetParameterArray3(
            "inBoundsMin", 1, cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().GetLeftBottomBack().PeekCoordinates());
        this->lightingShader.SetParameterArray3(
            "inBoundsSize", 1, cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().GetSize().PeekDimension());
    }

    glBegin(GL_POINTS);
    glVertex2f(0.0f, 0.0f);
    glEnd();

    this->lightingShader.Disable();
}


/*
 * moldyn::SimpleSphereRenderer::getTransferFunctionHandle
 */
GLuint moldyn::SimpleSphereRenderer::getTransferFunctionHandle() {
    core::view::CallGetTransferFunction* cgtf = this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();
    if ((cgtf != nullptr) && (*cgtf)()) return cgtf->OpenGLTexture();

    return tfFallbackHandle;
}


/*
 * moldyn::SimpleSphereRenderer::uploadDataToGPU
 */
void moldyn::SimpleSphereRenderer::uploadDataToGPU(const moldyn::SimpleSphereRenderer::gpuParticleDataType& gpuData,
    megamol::core::moldyn::MultiParticleDataCall::Particles& particles) {
    glBindVertexArray(gpuData.vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, gpuData.colorVBO);
    unsigned int partCount = static_cast<unsigned int>(particles.GetCount());
    // colour
    switch (particles.GetColourDataType()) {
    case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE:
        break;
    case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
        glBufferData(GL_ARRAY_BUFFER, partCount * (std::max)(particles.GetColourDataStride(), 3u),
            particles.GetColourData(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, particles.GetColourDataStride(), 0);
        break;
    case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        glBufferData(GL_ARRAY_BUFFER, partCount * (std::max)(particles.GetColourDataStride(), 4u),
            particles.GetColourData(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, particles.GetColourDataStride(), 0);
        break;
    case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
        glBufferData(GL_ARRAY_BUFFER,
            partCount * (std::max)(particles.GetColourDataStride(), static_cast<unsigned int>(3 * sizeof(float))),
            particles.GetColourData(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, particles.GetColourDataStride(), 0);
        break;
    case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
        glBufferData(GL_ARRAY_BUFFER,
            partCount * (std::max)(particles.GetColourDataStride(), static_cast<unsigned int>(4 * sizeof(float))),
            particles.GetColourData(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, particles.GetColourDataStride(), 0);
        break;
        // Not supported - fall through to the gay version
        // FIXME: this will probably not work!
    case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
        glBufferData(GL_ARRAY_BUFFER,
            partCount * (std::max)(particles.GetColourDataStride(), static_cast<unsigned int>(1 * sizeof(float))),
            particles.GetColourData(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, particles.GetColourDataStride(), 0);
        // std::cout<<"Transfer function"<<std::endl;
        break;
    default:
        glColor4ub(127, 127, 127, 255);
        glDisableVertexAttribArray(1);
        break;
    }

    // radius and position
    glBindBuffer(GL_ARRAY_BUFFER, gpuData.vertexVBO);
    switch (particles.GetVertexDataType()) {
    case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE:
        return;
    case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        glBufferData(GL_ARRAY_BUFFER,
            partCount * (std::max)(particles.GetVertexDataStride(), static_cast<unsigned int>(3 * sizeof(float))),
            particles.GetVertexData(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, particles.GetVertexDataStride(), 0);
        break;

    case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        glBufferData(GL_ARRAY_BUFFER,
            partCount * (std::max)(particles.GetVertexDataStride(), static_cast<unsigned int>(4 * sizeof(float))),
            particles.GetVertexData(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, particles.GetVertexDataStride(), 0);
        break;
    default:
        glDisableVertexAttribArray(0);
        return;
    }
}


/*
 * moldyn::SimpleSphereRenderer::generate3ConeDirections
 */
void moldyn::SimpleSphereRenderer::generate3ConeDirections(
    std::vector<vislib::math::Vector<float, 4>>& directions, float apex) {
    directions.clear();

    float edge_length = 2.0f * tan(0.5f * apex);
    float height = sqrt(1.0f - edge_length * edge_length / 12.0f);
    float radius = sqrt(3.0f) / 3.0f * edge_length;

    for (int i = 0; i < 3; ++i) {
        float angle = static_cast<float>(i) / 3.0f * 2.0f * static_cast<float>(M_PI);

        vislib::math::Vector<float, 3> center(cos(angle) * radius, height, sin(angle) * radius);
        center.Normalise();
        directions.push_back(vislib::math::Vector<float, 4>(center.X(), center.Y(), center.Z(), edge_length));
    }
}


/*
 * moldyn::SimpleSphereRenderer::generateDirectionShaderArrayString
 */
std::string moldyn::SimpleSphereRenderer::generateDirectionShaderArrayString(
    const std::vector<vislib::math::Vector<float, 4>>& directions, const std::string& directionsName) {
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
