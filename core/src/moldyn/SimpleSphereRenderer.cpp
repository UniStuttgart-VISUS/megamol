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

#define NGS_THE_INSTANCE  "gl_VertexID" // "gl_InstanceID"
#define NGS_THE_ALIGNMENT "packed"

const GLuint SSBObindingPoint      = 2;
const GLuint SSBOcolorBindingPoint = 3;


// typedef void (APIENTRY *GLDEBUGPROC)(GLenum source,GLenum type,GLuint id,GLenum severity,GLsizei length,const GLchar *message,const void *userParam);
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
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[%s %s] (%s %u) %s\n", sourceText, severityText, typeText, id, message);
}


/******************************************************************************
 * Known DebugGLCallback Errors (CAN BE IGNORED):
 *
 * >>> glUnmapNamedBuffe (only if named buffer wasn't mapped before):
 *      [API High] (Error 1282) GL_INVALID_OPERATION error generated. Buffer is unbound or is already unmapped.
 *
 * >>> glMapNamedBufferRange:
 *      [API Notification] (Other 131185) Buffer detailed info: Buffer object 1 (bound to GL_SHADER_STORAGE_BUFFER, usage hint is GL_DYNAMIC_DRAW) will use SYSTEM HEAP memory as the source for buffer object operations.
 *      [API Notification] (Other 131185) Buffer detailed info: Buffer object 1 (bound to GL_SHADER_STORAGE_BUFFER, usage hint is GL_DYNAMIC_DRAW) has been mapped WRITE_ONLY in SYSTEM HEAP memory(fast).
 *
 ******************************************************************************/


/*
 * moldyn::SimpleSphereRenderer::SimpleSphereRenderer
 */
moldyn::SimpleSphereRenderer::SimpleSphereRenderer(void) : AbstractSimpleSphereRenderer(),
    renderMode(RenderMode::NG),
    sphereShader(),
    sphereGeometryShader(),
    vertShader(nullptr),
    fragShader(nullptr),
    geoShader(nullptr),
    vertArray(),
    colType(SimpleSphericalParticles::ColourDataType::COLDATA_NONE),
    vertType(SimpleSphericalParticles::VertexDataType::VERTDATA_NONE),
    newShader(nullptr),
    theShaders(),
    theShaders_splat(),
    streamer(),
    colStreamer(),
    fences(),
    theSingleBuffer(),
    currBuf(0),
    bufSize(32 * 1024 * 1024),
    numBuffers(3),
    theSingleMappedMem(nullptr),
    singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT), 
    singleBufferMappingBits( GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT),
    //timer(),
    renderModeParam(       "renderMode",        "The sphere render mode."),
    toggleModeParam(       "renderModeButton",  "Toggle sphere render modes."),
    radiusScalingParam(    "scaling",           "Scaling factor for particle radii."),
    alphaScalingParam(     "alphaScaling",      "NGSplat: Scaling factor for particle alpha."), 
    attenuateSubpixelParam("attenuateSubpixel", "NGSplat: Attenuate alpha of points that should have subpixel size.")
{
    param::EnumParam* rmp = new param::EnumParam(this->renderMode);
    rmp->SetTypePair(RenderMode::SIMPLE,           "Simple");
    rmp->SetTypePair(RenderMode::SIMPLE_CLUSTERED, "Simple_Clustered");
    rmp->SetTypePair(RenderMode::SIMPLE_GEO,       "Simple_Geometry_Shader");
    rmp->SetTypePair(RenderMode::NG,               "NG");
    rmp->SetTypePair(RenderMode::NG_SPLAT,         "NG_Splat");
    rmp->SetTypePair(RenderMode::NG_BUFFER_ARRAY,  "NG_Buffer_Array");

    this->renderModeParam << rmp;
    this->MakeSlotAvailable(&this->renderModeParam);

    this->toggleModeParam.SetParameter(new param::ButtonParam('r'));
    this->toggleModeParam.SetUpdateCallback(&SimpleSphereRenderer::toggleRenderMode);
    this->MakeSlotAvailable(&this->toggleModeParam);

    this->radiusScalingParam << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->radiusScalingParam);

    this->alphaScalingParam << new core::param::FloatParam(5.0f);
    this->MakeSlotAvailable(&this->alphaScalingParam);

    this->attenuateSubpixelParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->attenuateSubpixelParam);

    //this->resetResources();
}


/*
 * moldyn::SimpleSphereRenderer::~SimpleSphereRenderer
 */
moldyn::SimpleSphereRenderer::~SimpleSphereRenderer(void) {
    this->Release();
}


/*
 * moldyn::SimpleSphereRenderer::create
 */
bool moldyn::SimpleSphereRenderer::create(void) {

    ASSERT(IsAvailable());

#ifdef DEBUG_GL_CALLBACK
    glDebugMessageCallback(DebugGLCallback, nullptr);
#endif

    this->renderMode = static_cast<RenderMode>(this->renderModeParam.Param<param::EnumParam>()->Value());
    if (!this->createShaders()) {
        return false;
    }

    //timer.SetNumRegions(4);
    //const char *regions[4] = {"Upload1", "Upload2", "Upload3", "Rendering"};
    //timer.SetRegionNames(4, regions);
    //timer.SetStatisticsFileName("fullstats.csv");
    //timer.SetSummaryFileName("summary.csv");
    //timer.SetMaximumFrames(20, 100);

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

    glUnmapNamedBufferEXT(this->theSingleBuffer);
    for (auto &x : fences) {
        if (x) {
            glDeleteSync(x);
        }
    }

    glDeleteBuffers(1, &this->theSingleBuffer);
    glDeleteVertexArrays(1, &this->vertArray);

    this->currBuf = 0;
    this->bufSize = (32 * 1024 * 1024);
    this->numBuffers = 3;
    this->fences.clear();
    this->fences.resize(numBuffers);

    this->colType = SimpleSphericalParticles::ColourDataType::COLDATA_NONE;
    this->vertType = SimpleSphericalParticles::VertexDataType::VERTDATA_NONE;

    // this variant should not need the fence
    //singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT);
    //singleBufferMappingBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT); 
    this->singleBufferCreationBits = (GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT);
    this->singleBufferMappingBits = (GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);

    return true;
}


/*
 * moldyn::SimpleSphereRenderer::createShaders
 */
bool moldyn::SimpleSphereRenderer::createShaders() {

    this->resetResources();

    this->vertShader = new ShaderSource();
    this->fragShader = new ShaderSource();

    vislib::StringA vertShaderName;
    vislib::StringA fragShaderName;
    vislib::StringA geoShaderName;

    vislib::StringA mode;
    switch (this->renderMode) {
    case (RenderMode::SIMPLE):           mode = "SIMPLE"; break;
    case (RenderMode::SIMPLE_CLUSTERED): mode = "SIMPLE CLUSTERED"; break;
    case (RenderMode::SIMPLE_GEO):       mode = "SIMPLE GEOMETRY SHADER"; break;
    case (RenderMode::NG):               mode = "NG"; break;
    case (RenderMode::NG_SPLAT):         mode = "NG SPLAT"; break;
    case (RenderMode::NG_BUFFER_ARRAY):  mode = "NG BUFFER ARRAY"; break;
    default: break;
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
                if (!this->sphereShader.Create(this->vertShader->Code(), this->vertShader->Count(), this->fragShader->Code(), this->fragShader->Count())) {
                    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                        "Unable to compile sphere shader: Unknown error\n");
                    return false;
                }
            }
            break;

        case (RenderMode::SIMPLE_GEO):
            this->geoShader = new ShaderSource();
            vertShaderName = "geosphere::vertex";
            fragShaderName = "geosphere::fragment";
            geoShaderName  = "geosphere::geometry";
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
                this->geoShader->Code(), this->geoShader->Count(),
                this->fragShader->Code(), this->fragShader->Count())) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "Unable to compile sphere geometry shader: Unknown error\n");
                return false;
            }
            if (!this->sphereGeometryShader.Link()) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "Unable to link sphere geometry shader: Unknown error\n");
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
            glBufferStorage(GL_SHADER_STORAGE_BUFFER, this->bufSize * this->numBuffers, nullptr, singleBufferCreationBits);
            this->theSingleMappedMem = glMapNamedBufferRangeEXT(this->theSingleBuffer, 0, this->bufSize * this->numBuffers, singleBufferMappingBits);
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
            if (!this->sphereShader.Create(this->vertShader->Code(), this->vertShader->Count(), this->fragShader->Code(), this->fragShader->Count())) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "Unable to compile sphere shader: Unknown error\n");
                return false;
            }
            glGenVertexArrays(1, &this->vertArray);
            glBindVertexArray(this->vertArray);
            glGenBuffers(1, &this->theSingleBuffer);
            glBindBuffer(GL_ARRAY_BUFFER, this->theSingleBuffer);
            glBufferStorage(GL_ARRAY_BUFFER, this->bufSize * this->numBuffers, nullptr, singleBufferCreationBits);
            this->theSingleMappedMem = glMapNamedBufferRangeEXT(this->theSingleBuffer, 0, this->bufSize * this->numBuffers, singleBufferMappingBits);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
            break;

        default:
            return false;
        }

    }
    catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
                ce.FailedAction()), ce.GetMsgA());
        return false;
    }
    catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: %s\n", e.GetMsgA());
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: Unknown exception\n");
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
        if (!this->createShaders()) {
            return false;
        }
    }

    // timer.BeginFrame();

    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>(&call);
    if (cr3d == nullptr) return false;

    float scaling = 1.0f;
    MultiParticleDataCall *mpdc = this->getData(static_cast<unsigned int>(cr3d->Time()), scaling);
    if (mpdc == nullptr) return false;

    float viewport[4];
    glGetFloatv(GL_VIEWPORT, viewport);
    glPointSize(vislib::math::Max(viewport[2], viewport[3]));
    if (viewport[2] < 1.0f) viewport[2] = 1.0f;
    if (viewport[3] < 1.0f) viewport[3] = 1.0f;
    viewport[2] = 2.0f / viewport[2];
    viewport[3] = 2.0f / viewport[3];

    float clipDat[4];
    float clipCol[4];
    this->getClipData(clipDat, clipCol);

    GLfloat lightPos[4];
    glGetLightfv(GL_LIGHT0, GL_POSITION, lightPos);

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

    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> MVinv = MV;
    MVinv.Invert();

    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> MVP = PM * MV;

    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> MVPinv = MVP;
    MVPinv.Invert();

    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> MVPtransp = MVP;
    MVPtransp.Transpose();

    glDisable(GL_BLEND);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS); /// default - necessary for early depth test in fragment shader to work.

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    switch (currentRenderMode) {
        case (RenderMode::SIMPLE):           return this->renderSimple(cr3d, mpdc, viewport, clipDat, clipCol, scaling, lightPos, MVinv, MVP, MVPinv, MVPtransp);
        case (RenderMode::SIMPLE_CLUSTERED): return this->renderSimple(cr3d, mpdc, viewport, clipDat, clipCol, scaling, lightPos, MVinv, MVP, MVPinv, MVPtransp);
        case (RenderMode::SIMPLE_GEO):       return this->renderGeo(cr3d, mpdc, viewport, clipDat, clipCol, scaling, lightPos, MVinv, MVP, MVPinv, MVPtransp);
        case (RenderMode::NG):               return this->renderNG(cr3d, mpdc, viewport, clipDat, clipCol, scaling, lightPos, MVinv, MVP, MVPinv, MVPtransp);
        case (RenderMode::NG_SPLAT):         return this->renderNGSplat(cr3d, mpdc, viewport, clipDat, clipCol, scaling, lightPos, MVinv, MVP, MVPinv, MVPtransp);
        case (RenderMode::NG_BUFFER_ARRAY):  return this->renderNGBufferArray(cr3d, mpdc, viewport, clipDat, clipCol, scaling, lightPos, MVinv, MVP, MVPinv, MVPtransp);
        
        default: break;
    }

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

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
bool moldyn::SimpleSphereRenderer::renderSimple(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc, 
    float vp[4], float clipDat[4], float clipCol[4], float scaling, float lp[4],
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVinv,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVP,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPinv,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPtransp) {

    this->sphereShader.Enable();

    GLuint vertAttribLoc = glGetAttribLocationARB(this->sphereShader, "inVertex");
    GLuint colAttribLoc = glGetAttribLocationARB(this->sphereShader, "inColor");
    GLuint colIdxAttribLoc = glGetAttribLocationARB(this->sphereShader, "colIdx");

    glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, vp);
    glUniform3fv(this->sphereShader.ParameterLocation("camIn"), 1, cr3d->GetCameraParameters()->Front().PeekComponents());
    glUniform3fv(this->sphereShader.ParameterLocation("camRight"), 1, cr3d->GetCameraParameters()->Right().PeekComponents());
    glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, cr3d->GetCameraParameters()->Up().PeekComponents());
    glUniform1f(this->sphereShader.ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphereShader.ParameterLocation("clipDat"), 1, clipDat);
    glUniform4fv(this->sphereShader.ParameterLocation("clipCol"), 1, clipCol);
    glUniform4fv(this->sphereShader.ParameterLocation("lpos"), 1, lp);
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVinv"), 1, GL_FALSE, MVinv.PeekComponents());
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVP"), 1, GL_FALSE, MVP.PeekComponents());
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPinv"), 1, GL_FALSE, MVPinv.PeekComponents());
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, MVPtransp.PeekComponents());

    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles &parts = mpdc->AccessParticles(i);

        GLuint vao, vb, cb;
        if (this->renderMode == RenderMode::SIMPLE_CLUSTERED) {
            parts.GetVAOs(vao, vb, cb);
            if (parts.IsVAO()) {
                glBindVertexArray(vao);
                this->setPointers<GLSLShader>(parts, this->sphereShader, 
                    vb, parts.GetVertexData(), vertAttribLoc, 
                    cb, parts.GetColourData(), colAttribLoc, colIdxAttribLoc);
            }
        }
        if ((this->renderMode == RenderMode::SIMPLE) || (!parts.IsVAO())) {
            this->setPointers<GLSLShader>(parts, this->sphereShader, 
                0, parts.GetVertexData(), vertAttribLoc, 
                0, parts.GetColourData(), colAttribLoc, colIdxAttribLoc);
        }

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

        if (this->renderMode == RenderMode::SIMPLE_CLUSTERED) {
            if (parts.IsVAO()) {
                glBindVertexArray(0); // vao
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
bool moldyn::SimpleSphereRenderer::renderNG(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc,
    float vp[4], float clipDat[4], float clipCol[4], float scaling, float lp[4],
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVinv,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVP,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPinv,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPtransp) {

#ifdef CHRONOTIMING
    std::vector<std::chrono::steady_clock::time_point> deltas;
    std::chrono::steady_clock::time_point before, after;
#endif

    //currBuf = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles &parts = mpdc->AccessParticles(i);

        if (colType != parts.GetColourDataType() || vertType != parts.GetVertexDataType()) {
            this->newShader = this->generateShader(parts);
        }

        this->newShader->Enable();

        //GLuint vertAttribLoc   = glGetAttribLocationARB(this->sphereShader, "inVertex");
        //GLuint colAttribLoc    = glGetAttribLocationARB(this->sphereShader, "inColor");
        //GLuint colIdxAttribLoc = glGetAttribLocationARB(this->sphereShader, "colIdx");

        glUniform4fv(this->newShader->ParameterLocation("viewAttr"), 1, vp);
        glUniform3fv(this->newShader->ParameterLocation("camIn"), 1, cr3d->GetCameraParameters()->Front().PeekComponents());
        glUniform3fv(this->newShader->ParameterLocation("camRight"), 1, cr3d->GetCameraParameters()->Right().PeekComponents());
        glUniform3fv(this->newShader->ParameterLocation("camUp"), 1, cr3d->GetCameraParameters()->Up().PeekComponents());
        glUniform1f(this->newShader->ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
        glUniform4fv(this->newShader->ParameterLocation("clipDat"), 1, clipDat);
        glUniform4fv(this->newShader->ParameterLocation("clipCol"), 1, clipCol);
        glUniform4fv(this->newShader->ParameterLocation("lpos"), 1, lp);
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVinv"), 1, GL_FALSE, MVinv.PeekComponents());
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVP"), 1, GL_FALSE, MVP.PeekComponents());
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVPinv"), 1, GL_FALSE, MVPinv.PeekComponents());
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVPtransp"), 1, GL_FALSE, MVPtransp.PeekComponents());

        float minC = 0.0f, maxC = 0.0f;
        unsigned int colTabSize = 0;

        // colour
        switch (parts.GetColourDataType()) {
        case MultiParticleDataCall::Particles::COLDATA_NONE: {
            glUniform4f(this->newShader->ParameterLocation("globalCol"),
                static_cast<float>(parts.GetGlobalColour()[0]) / 255.0f,
                static_cast<float>(parts.GetGlobalColour()[1]) / 255.0f,
                static_cast<float>(parts.GetGlobalColour()[2]) / 255.0f,
                1.0f);
        } break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
        case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
            glEnable(GL_TEXTURE_1D);
            view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
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
            break;
        }

        // radius and position
        switch (parts.GetVertexDataType()) {
        case MultiParticleDataCall::Particles::VERTDATA_NONE:
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
            glUniform4f(this->newShader->ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            glUniform4f(this->newShader->ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
            break;
        default:
            break;
        }

        unsigned int colBytes, vertBytes, colStride, vertStride;
        bool interleaved;
        this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride, interleaved);

        //currBuf = 0;
        //UINT64 numVerts, vertCounter;
        // does all data reside interleaved in the same memory?
        if (interleaved) {

            const GLuint numChunks = streamer.SetDataWithSize(parts.GetVertexData(), vertStride, vertStride,
                parts.GetCount(), 3, 32 * 1024 * 1024);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, streamer.GetHandle());
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, streamer.GetHandle());

            for (GLuint x = 0; x < numChunks; ++x) {
                GLuint numItems, sync;
                GLsizeiptr dstOff, dstLen;
                streamer.UploadChunk(x, numItems, sync, dstOff, dstLen);
                //streamer.UploadChunk<float, float>(x, [](float f) -> float { return f + 100.0; },
                //    numItems, sync, dstOff, dstLen);
                //vislib::sys::Log::DefaultLog.WriteInfo("uploading chunk %u at %lu len %lu", x, dstOff, dstLen);
                glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint,
                    this->streamer.GetHandle(), dstOff, dstLen);
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(numItems));
                streamer.SignalCompletion(sync);
            }
        }
        else {

            const GLuint numChunks = streamer.SetDataWithSize(parts.GetVertexData(), vertStride, vertStride,
                parts.GetCount(), 3, 32 * 1024 * 1024);
            const GLuint colSize = colStreamer.SetDataWithItems(parts.GetColourData(), colStride, colStride,
                parts.GetCount(), 3, streamer.GetMaxNumItemsPerChunk());
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, streamer.GetHandle());
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, streamer.GetHandle());
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, colStreamer.GetHandle());
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOcolorBindingPoint, colStreamer.GetHandle());

            for (GLuint x = 0; x < numChunks; ++x) {
                GLuint numItems, numItems2, sync, sync2;
                GLsizeiptr dstOff, dstLen, dstOff2, dstLen2;
                streamer.UploadChunk(x, numItems, sync, dstOff, dstLen);
                colStreamer.UploadChunk(x, numItems2, sync2, dstOff2, dstLen2);
                ASSERT(numItems == numItems2);
                glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint,
                    this->streamer.GetHandle(), dstOff, dstLen);
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBOcolorBindingPoint,
                    this->colStreamer.GetHandle(), dstOff2, dstLen2);
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(numItems));
                streamer.SignalCompletion(sync);
                streamer.SignalCompletion(sync2);
            }

        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        //glDisableVertexAttribArrayARB(vertAttribLoc);
        //glDisableVertexAttribArrayARB(colAttribLoc);
        //glDisableVertexAttribArrayARB(colIdxAttribLoc);
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
bool moldyn::SimpleSphereRenderer::renderNGSplat(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc,
    float vp[4], float clipDat[4], float clipCol[4], float scaling, float lp[4],
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVinv,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVP,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPinv,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPtransp) {

    glDisable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
#if 1
    // maybe for blending against white, remove pre-mult alpha and use this:
    // @gl.blendFuncSeparate @gl.SRC_ALPHA, @gl.ONE_MINUS_SRC_ALPHA, @gl.ONE, @gl.ONE_MINUS_SRC_ALPHA
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
#else
    //glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
    glBlendFunc(GL_ONE, GL_ONE);
#endif

    glEnable(GL_POINT_SPRITE);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, theSingleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer);

    //currBuf = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles &parts = mpdc->AccessParticles(i);

        if (colType != parts.GetColourDataType() || vertType != parts.GetVertexDataType()) {
            this->newShader = this->generateShader(parts);
        }

        this->newShader->Enable();

        //GLuint vertAttribLoc   = glGetAttribLocationARB(this->sphereShader, "inVertex");
        //GLuint colAttribLoc    = glGetAttribLocationARB(this->sphereShader, "inColor");
        //GLuint colIdxAttribLoc = glGetAttribLocationARB(this->sphereShader, "colIdx");

        glUniform4fv(this->newShader->ParameterLocation("viewAttr"), 1, vp);
        glUniform3fv(this->newShader->ParameterLocation("camIn"), 1, cr3d->GetCameraParameters()->Front().PeekComponents());
        glUniform3fv(this->newShader->ParameterLocation("camRight"), 1, cr3d->GetCameraParameters()->Right().PeekComponents());
        glUniform3fv(this->newShader->ParameterLocation("camUp"), 1, cr3d->GetCameraParameters()->Up().PeekComponents());
        glUniform1f(this->newShader->ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
        glUniform4fv(this->newShader->ParameterLocation("clipDat"), 1, clipDat);
        glUniform4fv(this->newShader->ParameterLocation("clipCol"), 1, clipCol);
        glUniform4fv(this->newShader->ParameterLocation("lpos"), 1, lp);
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVinv"), 1, GL_FALSE, MVinv.PeekComponents());
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVP"), 1, GL_FALSE, MVP.PeekComponents());
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVPinv"), 1, GL_FALSE, MVPinv.PeekComponents());
        glUniformMatrix4fv(this->newShader->ParameterLocation("MVPtransp"), 1, GL_FALSE, MVPtransp.PeekComponents());
        glUniform1f(this->newShader->ParameterLocation("alphaScaling"), this->alphaScalingParam.Param<param::FloatParam>()->Value());
        glUniform1i(this->newShader->ParameterLocation("attenuateSubpixel"), this->attenuateSubpixelParam.Param<param::BoolParam>()->Value() ? 1 : 0);
        glUniform1f(this->newShader->ParameterLocation("zNear"), cr3d->GetCameraParameters()->NearClip());

        float minC = 0.0f, maxC = 0.0f;
        unsigned int colTabSize = 0;

        // colour
        switch (parts.GetColourDataType()) {
        case MultiParticleDataCall::Particles::COLDATA_NONE: {
            glUniform4f(this->newShader->ParameterLocation("globalCol"),
                static_cast<float>(parts.GetGlobalColour()[0]) / 255.0f,
                static_cast<float>(parts.GetGlobalColour()[1]) / 255.0f,
                static_cast<float>(parts.GetGlobalColour()[2]) / 255.0f,
                1.0f);
        } break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
        case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
            glEnable(GL_TEXTURE_1D);
            view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
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
            break;
        }

        // radius and position
        switch (parts.GetVertexDataType()) {
        case MultiParticleDataCall::Particles::VERTDATA_NONE:
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
            glUniform4f(this->newShader->ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            glUniform4f(this->newShader->ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
            break;
        default:
            break;
        }

        unsigned int colBytes, vertBytes, colStride, vertStride;
        bool interleaved;
        this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride, interleaved);

        //currBuf = 0;
        UINT64 numVerts, vertCounter;
        // does all data reside interleaved in the same memory?
        if (interleaved) {

            numVerts = this->bufSize / vertStride;
            const char *currVert = static_cast<const char *>(parts.GetVertexData());
            const char *currCol = static_cast<const char *>(parts.GetColourData());
            vertCounter = 0;
            while (vertCounter < parts.GetCount()) {
                //GLuint vb = this->theBuffers[currBuf];
                void *mem = static_cast<char*>(theSingleMappedMem) + bufSize * currBuf;
                currCol = colStride == 0 ? currVert : currCol;
                //currCol = currCol == 0 ? currVert : currCol;
                const char *whence = currVert < currCol ? currVert : currCol;
                UINT64 vertsThisTime = vislib::math::Min(parts.GetCount() - vertCounter, numVerts);
                this->waitSingle(this->fences[currBuf]);
                //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "memcopying %u bytes from %016" PRIxPTR " to %016" PRIxPTR "\n", vertsThisTime * vertStride, whence, mem);
                memcpy(mem, whence, vertsThisTime * vertStride);
                glFlushMappedNamedBufferRangeEXT(theSingleBuffer, bufSize * currBuf, vertsThisTime * vertStride);
                //glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
                //glUniform1i(this->newShader->ParameterLocation("instanceOffset"), numVerts * currBuf);
                glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);

                //this->setPointers(parts, this->theSingleBuffer, reinterpret_cast<const void *>(currVert - whence), this->theSingleBuffer, reinterpret_cast<const void *>(currCol - whence));
                //glBindBuffer(GL_ARRAY_BUFFER, 0);
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer, bufSize * currBuf, bufSize);
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vertsThisTime));
                //glDrawArraysInstanced(GL_POINTS, 0, 1, vertsThisTime);
                this->lockSingle(fences[currBuf]);

                currBuf = (currBuf + 1) % this->numBuffers;
                vertCounter += vertsThisTime;
                currVert += vertsThisTime * vertStride;
                currCol += vertsThisTime * colStride;
                //break;
            }
        }
        else {
            // nothing
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        //glDisableVertexAttribArrayARB(vertAttribLoc);
        //glDisableVertexAttribArrayARB(colAttribLoc);
        //glDisableVertexAttribArrayARB(colIdxAttribLoc);
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
bool moldyn::SimpleSphereRenderer::renderNGBufferArray(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc,
    float vp[4], float clipDat[4], float clipCol[4], float scaling, float lp[4],
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVinv,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVP,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPinv,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPtransp) {

    this->sphereShader.Enable();

    GLuint vertAttribLoc   = glGetAttribLocationARB(this->sphereShader, "inVertex");
    GLuint colAttribLoc    = glGetAttribLocationARB(this->sphereShader, "inColor");
    GLuint colIdxAttribLoc = glGetAttribLocationARB(this->sphereShader, "colIdx");

    glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, vp);
    glUniform3fv(this->sphereShader.ParameterLocation("camIn"), 1, cr3d->GetCameraParameters()->Front().PeekComponents());
    glUniform3fv(this->sphereShader.ParameterLocation("camRight"), 1, cr3d->GetCameraParameters()->Right().PeekComponents());
    glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, cr3d->GetCameraParameters()->Up().PeekComponents());
    glUniform1f(this->sphereShader.ParameterLocation("scaling"), this->radiusScalingParam.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphereShader.ParameterLocation("clipDat"), 1, clipDat);
    glUniform4fv(this->sphereShader.ParameterLocation("clipCol"), 1, clipCol);
    glUniform4fv(this->sphereShader.ParameterLocation("lpos"), 1, lp);
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVinv"), 1, GL_FALSE, MVinv.PeekComponents());
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVP"), 1, GL_FALSE, MVP.PeekComponents());
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPinv"), 1, GL_FALSE, MVPinv.PeekComponents());
    glUniformMatrix4fv(this->sphereShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, MVPtransp.PeekComponents());

    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles &parts = mpdc->AccessParticles(i);

        unsigned int colBytes, vertBytes, colStride, vertStride;
        bool interleaved;
        this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride, interleaved);

        UINT64 numVerts, vertCounter;
        // does all data reside interleaved in the same memory?
        if (interleaved) {

            numVerts = this->bufSize / vertStride;
            const char *currVert = static_cast<const char *>(parts.GetVertexData());
            const char *currCol = static_cast<const char *>(parts.GetColourData());
            vertCounter = 0;
            while (vertCounter < parts.GetCount()) {
                //GLuint vb = this->theBuffers[currBuf];
                void *mem = static_cast<char*>(this->theSingleMappedMem) + numVerts * vertStride * this->currBuf;
                currCol = colStride == 0 ? currVert : currCol;
                //currCol = currCol == 0 ? currVert : currCol;
                const char *whence = currVert < currCol ? currVert : currCol;
                UINT64 vertsThisTime = vislib::math::Min(parts.GetCount() - vertCounter, numVerts);
                this->waitSingle(this->fences[this->currBuf]);
                //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "memcopying %u bytes from %016" PRIxPTR " to %016" PRIxPTR "\n", vertsThisTime * vertStride, whence, mem);
                memcpy(mem, whence, vertsThisTime * vertStride);
                glFlushMappedNamedBufferRangeEXT(this->theSingleBuffer, numVerts * this->currBuf, vertsThisTime * vertStride);
                //glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
                this->setPointers<GLSLShader>(parts, this->sphereShader,
                    this->theSingleBuffer, reinterpret_cast<const void *>(currVert - whence), vertAttribLoc, 
                    this->theSingleBuffer, reinterpret_cast<const void *>(currCol - whence),  colAttribLoc, colIdxAttribLoc);
                glDrawArrays(GL_POINTS, static_cast<GLint>(numVerts * this->currBuf), static_cast<GLsizei>(vertsThisTime));
                this->lockSingle(this->fences[this->currBuf]);

                this->currBuf = (this->currBuf + 1) % this->numBuffers;
                vertCounter += vertsThisTime;
                currVert += vertsThisTime * vertStride;
                currCol += vertsThisTime * colStride;
            }
        }
        else {
            // nothing
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
bool moldyn::SimpleSphereRenderer::renderGeo(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc,
    float vp[4], float clipDat[4], float clipCol[4], float scaling, float lp[4],
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVinv,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVP,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPinv,
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPtransp) {

    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    //glDepthFunc(GL_LEQUAL); // Why not use default GL_LESS?

    /// If enabled and a vertex shader is active, it specifies that the GL will choose between front and 
    /// back colors based on the polygon's face direction of which the vertex being shaded is a part. 
    /// It has no effect on points or lines.
    //glEnable(GL_VERTEX_PROGRAM_TWO_SIDE); // ! Has significant negative performance impact ....

    this->sphereGeometryShader.Enable();

    GLuint vertAttribLoc   = glGetAttribLocationARB(this->sphereGeometryShader, "inVertex");
    GLuint colAttribLoc    = glGetAttribLocationARB(this->sphereGeometryShader, "inColor");
    GLuint colIdxAttribLoc = glGetAttribLocationARB(this->sphereGeometryShader, "colIdx");

    // Set shader variables
    glUniform4fv(this->sphereGeometryShader.ParameterLocation("viewAttr"), 1, vp);
    glUniform3fv(this->sphereGeometryShader.ParameterLocation("camIn"), 1, cr3d->GetCameraParameters()->Front().PeekComponents());
    glUniform3fv(this->sphereGeometryShader.ParameterLocation("camRight"), 1, cr3d->GetCameraParameters()->Right().PeekComponents());
    glUniform3fv(this->sphereGeometryShader.ParameterLocation("camUp"), 1, cr3d->GetCameraParameters()->Up().PeekComponents());
    glUniform4fv(this->sphereGeometryShader.ParameterLocation("clipDat"), 1, clipDat);
    glUniform4fv(this->sphereGeometryShader.ParameterLocation("clipCol"), 1, clipCol);
    glUniform4fv(this->sphereGeometryShader.ParameterLocation("lpos"), 1, lp);
    glUniformMatrix4fv(this->sphereGeometryShader.ParameterLocation("MVinv"), 1, GL_FALSE, MVinv.PeekComponents());
    glUniformMatrix4fv(this->sphereGeometryShader.ParameterLocation("MVP"), 1, GL_FALSE, MVP.PeekComponents());
    glUniformMatrix4fv(this->sphereGeometryShader.ParameterLocation("MVPinv"), 1, GL_FALSE, MVPinv.PeekComponents());
    glUniformMatrix4fv(this->sphereGeometryShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, MVPtransp.PeekComponents());

    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles &parts = mpdc->AccessParticles(i);

        this->setPointers<GLSLGeometryShader>(parts, this->sphereGeometryShader,
            0, parts.GetVertexData(), vertAttribLoc, 
            0, parts.GetColourData(), colAttribLoc, colIdxAttribLoc);

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

        glDisableVertexAttribArray(vertAttribLoc);
        glDisableVertexAttribArray(colAttribLoc);
        glDisableVertexAttribArray(colIdxAttribLoc);
        glDisable(GL_TEXTURE_1D);
    }

    mpdc->Unlock();

    this->sphereGeometryShader.Disable();

    //glDisable(GL_VERTEX_PROGRAM_TWO_SIDE);
    //glDepthFunc(GL_LESS); // default

    return true;
}


///////////////////////////////////////////////////////////////////////////////
// NG functions ///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
 * NG - moldyn::SimpleSphereRenderer::setPointers
 */
template <typename T>
void moldyn::SimpleSphereRenderer::setPointers(MultiParticleDataCall::Particles& parts, T& shader,
    GLuint vertBuf, const void *vertPtr, GLuint vertAttribLoc,
    GLuint colBuf, const void *colPtr, GLuint colAttribLoc, GLuint colIdxAttribLoc) {

    float minC = 0.0f, maxC = 0.0f;
    unsigned int colTabSize = 0;

    // colour
    glBindBuffer(GL_ARRAY_BUFFER, colBuf);
    switch (parts.GetColourDataType()) {
    case MultiParticleDataCall::Particles::COLDATA_NONE: {
            const unsigned char* gc = parts.GetGlobalColour();
            glVertexAttrib3d(colAttribLoc,
                static_cast<double>(gc[0]) / 255.0,
                static_cast<double>(gc[1]) / 255.0,
                static_cast<double>(gc[2]) / 255.0);
        } 
        break;
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

        view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
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
        }
        break;
    case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA:
        glEnableVertexAttribArray(colAttribLoc);
        glVertexAttribPointer(colAttribLoc, 4, GL_SHORT, GL_TRUE, parts.GetColourDataStride(), colPtr);
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
 * NG - moldyn::SimpleSphereRenderer::makeColorString
 */
bool moldyn::SimpleSphereRenderer::makeColorString(MultiParticleDataCall::Particles &parts, std::string &code, std::string &declaration, bool interleaved) {

    bool ret = true;

    switch (parts.GetColourDataType()) {
    case MultiParticleDataCall::Particles::COLDATA_NONE:
        declaration = "";
        code = "";
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
                "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].g,\n"
                "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].b, 1.0); \n";
        }
        else {
            code = "    theColor = vec4(theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].r,\n"
                "                 theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].g,\n"
                "                 theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].b, 1.0); \n";
        }
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
        declaration = "    float r; float g; float b; float a;\n";
        if (interleaved) {
            code = "    theColor = vec4(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].r,\n"
                "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].g,\n"
                "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].b,\n"
                "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].a); \n";
        }
        else {
            code = "    theColor = vec4(theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].r,\n"
                "                 theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].g,\n"
                "                 theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].b,\n"
                "                 theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].a); \n";
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
        code = "    theColor = gl_Color;\n"
            "    theColIdx = colIdx;";
        break;
    }
    //code = "    theColor = vec4(0.2, 0.7, 1.0, 1.0);";

    return ret;
}


/*
 * NG - moldyn::SimpleSphereRenderer::makeVertexString
 */
bool moldyn::SimpleSphereRenderer::makeVertexString(MultiParticleDataCall::Particles &parts, std::string &code, std::string &declaration, bool interleaved) {

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
        code = "    inPos = gl_Vertex;\n"
            "    rad = (CONSTRAD < -0.5) ? inPos.w : CONSTRAD;\n"
            "    inPos.w = 1.0; ";
        break;
    }

    return ret;
}


/*
 * NG - moldyn::SimpleSphereRenderer::makeShader
 */
std::shared_ptr<GLSLShader> moldyn::SimpleSphereRenderer::makeShader(vislib::SmartPtr<ShaderSource> vert, vislib::SmartPtr<ShaderSource> frag) {

    std::shared_ptr<GLSLShader> sh = std::make_shared<GLSLShader>(GLSLShader());
    try {
        if (!sh->Create(vert->Code(), vert->Count(), frag->Code(), frag->Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile sphere shader: Unknown error\n");
            return nullptr;
        }

    }
    catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
                ce.FailedAction()), ce.GetMsgA());
        return nullptr;
    }
    catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: %s\n", e.GetMsgA());
        return nullptr;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: Unknown exception\n");
        return nullptr;
    }
    return sh;
}


/*
 * NG - moldyn::SimpleSphereRenderer::generateShader
 */
std::shared_ptr<vislib::graphics::gl::GLSLShader> moldyn::SimpleSphereRenderer::generateShader(MultiParticleDataCall::Particles &parts) {

    int c = parts.GetColourDataType();
    int p = parts.GetVertexDataType();

    unsigned int colBytes, vertBytes, colStride, vertStride;
    bool interleaved;
    this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride, interleaved);

    shaderMap::iterator i = this->theShaders.find(std::make_tuple(c, p, interleaved));
    if (i == this->theShaders.end()) {
        //instance()->ShaderSourceFactory().MakeShaderSource()

        vislib::SmartPtr<ShaderSource> v2 = new ShaderSource(*this->vertShader);
        vislib::SmartPtr<ShaderSource::Snippet> codeSnip, declarationSnip;
        std::string vertCode, colCode, vertDecl, colDecl, decl;
        makeVertexString(parts, vertCode, vertDecl, interleaved);
        makeColorString(parts, colCode, colDecl, interleaved);

        if (interleaved) {

            decl = "\nstruct SphereParams {\n";

            //if (vertStride > vertBytes) {
            //    unsigned int rest = (vertStride - vertBytes);
            //    if (rest % 4 == 0) {
            //        char heinz[128];
            //        while (rest > 0) {
            //            sprintf(heinz, "    float padding%u;\n", rest);
            //            decl += heinz;
            //            rest -= 4;
            //        }
            //    }
            //}

            if (parts.GetColourData() < parts.GetVertexData()) {
                decl += colDecl;
                decl += vertDecl;
            }
            else {
                decl += vertDecl;
                decl += colDecl;
            }
            decl += "};\n";

            decl += "layout(" NGS_THE_ALIGNMENT ", binding = " + std::to_string(SSBObindingPoint) + ") buffer shader_data {\n"
                "    SphereParams theBuffer[];\n"
                // flat float version
                //"    float theBuffer[];\n"
                "};\n";

        }
        else {
            // we seem to have separate buffers for vertex and color data

            decl = "\nstruct SpherePosParams {\n" + vertDecl + "};\n";
            decl += "\nstruct SphereColParams {\n" + colDecl + "};\n";

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
      
        /// Generated shader declaration snippet is inserted between 2nd and 3rd snippet (after ngsphere_vert_attributes.glsl)
        v2->Insert(3, declarationSnip); 
        /// Generated shader code snippet is inserted between 4th and 5th snippet (after ngsphere_vert_mainstart.glsl) => consider new index through first Insertion!
        v2->Insert(5, codeSnip);        
        //std::string s(v2->WholeCode());

        vislib::SmartPtr<ShaderSource> vss(v2);
        this->theShaders.emplace(std::make_pair(std::make_tuple(c, p, interleaved), makeShader(v2, this->fragShader)));
        i = this->theShaders.find(std::make_tuple(c, p, interleaved));
    }
    return i->second;
}


/*
 * NG - moldyn::SimpleSphereRenderer::getBytesAndStride
 */
void moldyn::SimpleSphereRenderer::getBytesAndStride(MultiParticleDataCall::Particles &parts, unsigned int &colBytes, unsigned int &vertBytes,
    unsigned int &colStride, unsigned int &vertStride, bool &interleaved) {

    vertBytes = 0; colBytes = 0;
    switch (parts.GetColourDataType()) {
    case MultiParticleDataCall::Particles::COLDATA_NONE:
        // nothing
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
        colBytes = vislib::math::Max(colBytes, 3U);
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        colBytes = vislib::math::Max(colBytes, 4U);
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
        colBytes = vislib::math::Max(colBytes, 3 * 4U);
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
        colBytes = vislib::math::Max(colBytes, 4 * 4U);
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
        colBytes = vislib::math::Max(colBytes, 1 * 4U);
    } break;
    case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
        colBytes = vislib::math::Max(colBytes, 1 * 8U);
    } break;
    case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA: {
        colBytes = vislib::math::Max(colBytes, 4 * 2U);
    } break;
    default:
        // nothing
        break;
    }

    // radius and position
    switch (parts.GetVertexDataType()) {
    case MultiParticleDataCall::Particles::VERTDATA_NONE:
        //continue;
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        vertBytes = vislib::math::Max(vertBytes, 3 * 4U);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
        vertBytes = vislib::math::Max(vertBytes, 3 * 8U);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        vertBytes = vislib::math::Max(vertBytes, 4 * 4U);
        break;
    default:
        //continue;
        break;
    }

    colStride  = parts.GetColourDataStride();
    colStride  = colStride < colBytes ? colBytes : colStride;
    vertStride = parts.GetVertexDataStride();
    vertStride = vertStride < vertBytes ? vertBytes : vertStride;

    interleaved = (std::abs(reinterpret_cast<const ptrdiff_t>(parts.GetColourData())
        - reinterpret_cast<const ptrdiff_t>(parts.GetVertexData())) <= vertStride
        && vertStride == colStride) || colStride == 0;
}


/*
 * NG - moldyn::SimpleSphereRenderer::lockSingle
 */
void moldyn::SimpleSphereRenderer::lockSingle(GLsync& syncObj) {
    if (syncObj) {
        glDeleteSync(syncObj);
    }
    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}


/*
 * NG - moldyn::SimpleSphereRenderer::waitSingle
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

