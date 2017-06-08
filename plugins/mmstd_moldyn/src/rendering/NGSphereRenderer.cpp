/*
 * NGSphereRenderer.cpp
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "NGSphereRenderer.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/FloatParam.h"
#include "vislib/assert.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"
#include <inttypes.h>
#include <stdint.h>

#include <chrono>
#include <sstream>
#include <iostream>
#include <iterator>

//#define CHRONOTIMING

using namespace megamol::core;
using namespace megamol::stdplugin::moldyn::rendering;
#define MAP_BUFFER_LOCALLY
#define DEBUG_BLAHBLAH

const GLuint SSBObindingPoint = 2;
//#define NGS_THE_INSTANCE "gl_InstanceID"
#define NGS_THE_INSTANCE "gl_VertexID"

//typedef void (APIENTRY *GLDEBUGPROC)(GLenum source,GLenum type,GLuint id,GLenum severity,GLsizei length,const GLchar *message,const void *userParam);
void APIENTRY MyFunkyDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
    const GLchar* message, const GLvoid* userParam) {
        const char *sourceText, *typeText, *severityText;
        switch(source) {
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
        switch(type) {
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
        switch(severity) {
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


/*
 * moldyn::NGSphereRenderer::NGSphereRenderer
 */
NGSphereRenderer::NGSphereRenderer(void) : AbstractSimpleSphereRenderer(),
//sphereShader(),
    fences(), currBuf(0), bufSize(32 * 1024 * 1024), numBuffers(3),
    scalingParam("scaling", "scaling factor for particle radii"),
//	timer(),
    // this variant should not need the fence
    //singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT),
    //singleBufferMappingBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT),
    singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT),
    singleBufferMappingBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT),
    colType(SimpleSphericalParticles::COLDATA_NONE), vertType(SimpleSphericalParticles::VERTDATA_NONE),
    theShaders() {

    this->scalingParam << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->scalingParam);
    fences.resize(numBuffers);
}


/*
 * moldyn::NGSphereRenderer::~NGSphereRenderer
 */
NGSphereRenderer::~NGSphereRenderer(void) {
    this->Release();
}

void NGSphereRenderer::queueSignal(GLsync& syncObj) {
    if (syncObj) {
        glDeleteSync(syncObj);
    }
    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

void NGSphereRenderer::waitSignal(GLsync& syncObj) {
    if (syncObj) {
        while (1) {
            GLenum wait = glClientWaitSync(syncObj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                return;
            }
        }
    }
}


/*
 * moldyn::SimpleSphereRenderer::create
 */
bool NGSphereRenderer::create(void) {
#ifdef DEBUG_BLAHBLAH
    glDebugMessageCallback(MyFunkyDebugCallback, NULL);
#endif
    //vislib::graphics::gl::ShaderSource vert, frag;
    vert = new ShaderSource();
    frag = new ShaderSource();
    if (!instance()->ShaderSourceFactory().MakeShaderSource("NGsphere::vertex", *vert)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("NGsphere::fragment", *frag)) {
        return false;
    }

    //printf("\nVertex Shader:\n%s\n\nFragment Shader:\n%s\n",
    //    vert.WholeCode().PeekBuffer(),
    //    frag.WholeCode().PeekBuffer());

    //bool ret = makeShader(vert, frag);
    //if (!ret) {
    //	return false;
    //}

    glGenVertexArrays(1, &this->vertArray);
    glBindVertexArray(this->vertArray);
    glGenBuffers(1, &this->theSingleBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSingleBuffer);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, this->bufSize * this->numBuffers, nullptr, singleBufferCreationBits);
    this->theSingleMappedMem = glMapNamedBufferRangeEXT(this->theSingleBuffer, 0, this->bufSize * this->numBuffers, singleBufferMappingBits);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindVertexArray(0);

    //timer.SetNumRegions(4);
    //const char *regions[4] = {"Upload1", "Upload2", "Upload3", "Rendering"};
    //timer.SetRegionNames(4, regions);
    //timer.SetStatisticsFileName("fullstats.csv");
    //timer.SetSummaryFileName("summary.csv");
    //timer.SetMaximumFrames(20, 100);
    return AbstractSimpleSphereRenderer::create();
}

bool NGSphereRenderer::makeColorString(MultiParticleDataCall::Particles &parts, std::string &code, std::string &declaration) {
    bool ret = true;
    switch (parts.GetColourDataType()) {
        case MultiParticleDataCall::Particles::COLDATA_NONE:
            declaration = "";
            code = "    theColor = vec4(1.0);\n";
            //glColor3ubv(parts.GetGlobalColour());
            break;
        case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
            //glColorPointer(3, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), colPtr);
            ret = false;
            break;
        case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
            //glEnableClientState(GL_COLOR_ARRAY);
            //glColorPointer(4, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), colPtr);
            declaration = "    uint color;\n";
            code = "    theColor = unpackUnorm4x8(theBuffer[" NGS_THE_INSTANCE "+ instanceOffset].color);\n";
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
            //glEnableClientState(GL_COLOR_ARRAY);
            //glColorPointer(3, GL_FLOAT, parts.GetColourDataStride(), colPtr);
            declaration = "    vec3 color;\n";
            code = "    theColor = vec4(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].color, 1.0); \n";
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
            //glEnableClientState(GL_COLOR_ARRAY);
            //glColorPointer(4, GL_FLOAT, parts.GetColourDataStride(), colPtr);
            declaration = "    vec4 color;\n";
            code = "    theColor = theBuffer[" NGS_THE_INSTANCE " + instanceOffset].color;\n";
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
            //glEnableVertexAttribArrayARB(colIdxAttribLoc);
            //glVertexAttribPointerARB(colIdxAttribLoc, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), colPtr);

            //glEnable(GL_TEXTURE_1D);

            //view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
            //if ((cgtf != NULL) && ((*cgtf)())) {
            //	glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
            //	colTabSize = cgtf->TextureSize();
            //}
            declaration = "    float colorIndex;\n";
            code = "    theColIdx = theBuffer[" NGS_THE_INSTANCE " + instanceOffset].colorIndex; \n";
            // flat float version
            //code = "    theColIdx = theBuffer[5 * " NGS_THE_INSTANCE " + 5 * instanceOffset + 4]; \n";

            //else {
            //	glBindTexture(GL_TEXTURE_1D, this->greyTF);
            //	colTabSize = 2;
            //}

            //glUniform1i(this->sphereShader.ParameterLocation("colTab"), 0);
            //minC = parts.GetMinColourIndexValue();
            //maxC = parts.GetMaxColourIndexValue();
            //glColor3ub(127, 127, 127);
        } break;
        default:
            //glColor3ub(127, 127, 127);
            declaration = "";
            code = "    theColor = gl_Color;\n"
                   "    theColIdx = colIdx;";
            break;
    }
    return ret;
}

bool NGSphereRenderer::makeVertexString(MultiParticleDataCall::Particles &parts, std::string &code, std::string &declaration)  {
    bool ret = true;
    //std::string code;
    //std::string declaration;

    switch (parts.GetVertexDataType()) {
    case MultiParticleDataCall::Particles::VERTDATA_NONE:
        declaration = "";
        code = "";
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        //glEnableClientState(GL_VERTEX_ARRAY);
        //glUniform4f(this->sphereShader.ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
        //glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), vertPtr);
        declaration = "    float posX; float posY; float posZ;\n";
        code = "    inPos = vec4(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posX,\n"
               "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posY,\n"
               "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posZ, 1.0); \n"
               "    rad = CONSTRAD;";
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        //glEnableClientState(GL_VERTEX_ARRAY);
        //glUniform4f(this->sphereShader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
        //glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), vertPtr);
        // broken
        //declaration = "    vec4 posR;\n";
        //code = "    inPos = theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posR; \n"
        //       "    rad = inPos.w;\n"
        //       "    inPos.w = 1.0;";

        declaration = "    float posX; float posY; float posZ; float posR;\n";
        code = "    inPos = vec4(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posX,\n"
            "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posY,\n"
            "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posZ, 1.0); \n"
            "    rad = theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posR;";

        // flat float version
        //code = "    inPos = vec4(theBuffer[ 5 * " NGS_THE_INSTANCE " + 5 * instanceOffset + 0],"
        //    "theBuffer[ 5 * " NGS_THE_INSTANCE " + 5 * instanceOffset + 1],"
        //    "theBuffer[ 5 * " NGS_THE_INSTANCE " + 5 * instanceOffset + 2],"
        //    "theBuffer[ 5 * " NGS_THE_INSTANCE " + 5 * instanceOffset + 3]);\n"
        //    "    rad = inPos.w;\n"
        //    "    inPos.w = 1.0;";
        break;
    default:
        declaration = "";
        code = "    inPos = gl_Vertex;\n"
               "    rad = (CONSTRAD < -0.5) ? inPos.w : CONSTRAD;\n"
               "    inPos.w = 1.0; ";
        break;
    }
    //codeSnippet = new ShaderSource::StringSnippet(code.c_str());
    //declarationSnippet = new ShaderSource::StringSnippet(declaration.c_str());
    return ret;
}

std::shared_ptr<GLSLShader> NGSphereRenderer::makeShader(vislib::SmartPtr<ShaderSource> vert, vislib::SmartPtr<ShaderSource> frag) {
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

std::shared_ptr<vislib::graphics::gl::GLSLShader> NGSphereRenderer::generateShader(MultiParticleDataCall::Particles &parts) {
    int c = parts.GetColourDataType();
    int p = parts.GetVertexDataType();

    shaderMap::iterator i = theShaders.find({ c, p });
    if (i == theShaders.end()) {
        //instance()->ShaderSourceFactory().MakeShaderSource()

        unsigned int colBytes, vertBytes, colStride, vertStride;
        this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride);

        if ((reinterpret_cast<const ptrdiff_t>(parts.GetColourData())
            - reinterpret_cast<const ptrdiff_t>(parts.GetVertexData()) <= vertStride
            && vertStride == colStride) || colStride == 0) {

            vislib::SmartPtr<ShaderSource> v2 = new ShaderSource(*vert);
            vislib::SmartPtr<ShaderSource::Snippet> codeSnip, declarationSnip;
            std::string vertCode, colCode, vertDecl, colDecl;
            makeVertexString(parts, vertCode, vertDecl);
            makeColorString(parts, colCode, colDecl);
            std::string decl = "\nstruct SphereParams {\n";

            if (vertStride > vertBytes) {
                unsigned int rest = (vertStride - vertBytes);
                if (rest % 4 == 0) {
                    char heinz[128];
                    while (rest > 0) {
                        sprintf(heinz, "    float padding%u;\n", rest);
                        decl += heinz;
                        rest -= 4;
                    }
                }
            }

            if (parts.GetColourData() < parts.GetVertexData()) {
                decl += colDecl;
                decl += vertDecl;
            } else {
                decl += vertDecl;
                decl += colDecl;
            }
            decl += "};\n";

            decl += "layout(packed, binding = " + std::to_string(SSBObindingPoint) + ") buffer shader_data {\n"
                "    SphereParams theBuffer[];\n"
                // flat float version
                //"    float theBuffer[];\n"
                "};\n";
            std::string code = "\n";
            code += colCode;
            code += vertCode;
            declarationSnip = new ShaderSource::StringSnippet(decl.c_str());
            codeSnip = new ShaderSource::StringSnippet(code.c_str());
            v2->Insert(3, declarationSnip);
            v2->Insert(5, codeSnip);
            std::string s(v2->WholeCode());

            vislib::SmartPtr<ShaderSource> vss(v2);
            theShaders.emplace(std::make_pair(std::make_pair(c, p), makeShader(v2, frag)));
            i = theShaders.find({ c, p });
        } else {
            // no clue yet
            throw new vislib::Exception("er. uhm. I don't know.", __FILE__, __LINE__);
        }
    }
    return i->second;
}


/*
 * moldyn::SimpleSphereRenderer::release
 */
void NGSphereRenderer::release(void) {
    glUnmapNamedBufferEXT(this->theSingleBuffer);
    for (auto &x : fences) {
        if (x) {
            glDeleteSync(x);
        }
    }
    //this->sphereShader.Release();
    // TODO release all shaders!
    glDeleteVertexArrays(1, &this->vertArray);
    glDeleteBuffers(1, &this->theSingleBuffer);
    AbstractSimpleSphereRenderer::release();
}


void NGSphereRenderer::setPointers(MultiParticleDataCall::Particles &parts, GLuint vertBuf, const void *vertPtr, GLuint colBuf, const void *colPtr) {
    float minC = 0.0f, maxC = 0.0f;
    unsigned int colTabSize = 0;

    // colour
    glBindBuffer(GL_ARRAY_BUFFER, colBuf);
    switch (parts.GetColourDataType()) {
        case MultiParticleDataCall::Particles::COLDATA_NONE:
            glColor3ubv(parts.GetGlobalColour());
            break;
        case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(3, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), colPtr);
            break;
        case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(4, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), colPtr);
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(3, GL_FLOAT, parts.GetColourDataStride(), colPtr);
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(4, GL_FLOAT, parts.GetColourDataStride(), colPtr);
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
            glEnableVertexAttribArrayARB(colIdxAttribLoc);
            glVertexAttribPointerARB(colIdxAttribLoc, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), colPtr);

            glEnable(GL_TEXTURE_1D);

            view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
            if ((cgtf != NULL) && ((*cgtf)())) {
                glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                colTabSize = cgtf->TextureSize();
            } else {
                glBindTexture(GL_TEXTURE_1D, this->greyTF);
                colTabSize = 2;
            }

            glUniform1i(this->newShader->ParameterLocation("colTab"), 0);
            minC = parts.GetMinColourIndexValue();
            maxC = parts.GetMaxColourIndexValue();
            glColor3ub(127, 127, 127);
        } break;
        default:
            glColor3ub(127, 127, 127);
            break;
    }
    glBindBuffer(GL_ARRAY_BUFFER, vertBuf);
    // radius and position
    switch (parts.GetVertexDataType()) {
        case MultiParticleDataCall::Particles::VERTDATA_NONE:
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
            glEnableClientState(GL_VERTEX_ARRAY);
            glUniform4f(this->newShader->ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
            glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), vertPtr);
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            glEnableClientState(GL_VERTEX_ARRAY);
            glUniform4f(this->newShader->ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
            glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), vertPtr);
            break;
        default:
            break;
    }
}

void NGSphereRenderer::getBytesAndStride(MultiParticleDataCall::Particles &parts, unsigned int &colBytes, unsigned int &vertBytes,
    unsigned int &colStride, unsigned int &vertStride) {
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
            // nothing else
        }
            break;
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
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            vertBytes = vislib::math::Max(vertBytes, 4 * 4U);
            break;
        default:
            //continue;
            break;
    }

    colStride = parts.GetColourDataStride();
    colStride = colStride < colBytes ? colBytes : colStride;
    vertStride = parts.GetVertexDataStride();
    vertStride = vertStride < vertBytes ? vertBytes : vertStride;
}


/*
 * moldyn::SimpleSphereRenderer::Render
 */
bool NGSphereRenderer::Render(Call& call) {
#ifdef DEBUG_BLAHBLAH
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    float scaling = 1.0f;
    MultiParticleDataCall *c2 = this->getData(static_cast<unsigned int>(cr->Time()), scaling);
    if (c2 == NULL) return false;

//	timer.BeginFrame();

    float clipDat[4];
    float clipCol[4];
    this->getClipData(clipDat, clipCol);
    
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    float viewportStuff[4];
    ::glGetFloatv(GL_VIEWPORT, viewportStuff);
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    //glScalef(scaling, scaling, scaling);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, theSingleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer);

    // this is the apex of suck and must die
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> scaleMat;
    scaleMat.SetAt(0, 0, scaling);
    scaleMat.SetAt(1, 1, scaling);
    scaleMat.SetAt(2, 2, scaling);
    modelViewMatrix = modelViewMatrix * scaleMat;
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);
    // Compute modelviewprojection matrix
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrixInv = modelViewMatrix;
    modelViewMatrixInv.Invert();
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrix = projMatrix * modelViewMatrix;
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrixInv = modelViewProjMatrix;
    modelViewProjMatrixInv.Invert();
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrixTransp = modelViewProjMatrix;
    modelViewProjMatrixTransp.Transpose();
    // end suck

#ifdef CHRONOTIMING
    std::vector<std::chrono::steady_clock::time_point> deltas;
    std::chrono::steady_clock::time_point before, after;
#endif

    //currBuf = 0;
    for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);

        if (colType != parts.GetColourDataType() || vertType != parts.GetVertexDataType()) {
             newShader = this->generateShader(parts);
        }
        newShader->Enable();
        colIdxAttribLoc = glGetAttribLocation(*this->newShader, "colIdx");
        glUniform4fv(newShader->ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(newShader->ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
        glUniform3fv(newShader->ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
        glUniform3fv(newShader->ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
        glUniform4fv(newShader->ParameterLocation("clipDat"), 1, clipDat);
        glUniform4fv(newShader->ParameterLocation("clipCol"), 1, clipCol);
        glUniformMatrix4fv(newShader->ParameterLocation("MVinv"), 1, GL_FALSE, modelViewMatrixInv.PeekComponents());
        glUniformMatrix4fv(newShader->ParameterLocation("MVP"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());
        glUniformMatrix4fv(newShader->ParameterLocation("MVPinv"), 1, GL_FALSE, modelViewProjMatrixInv.PeekComponents());
        glUniformMatrix4fv(newShader->ParameterLocation("MVPtransp"), 1, GL_FALSE, modelViewProjMatrixTransp.PeekComponents());
        glUniform1f(newShader->ParameterLocation("scaling"), this->scalingParam.Param<param::FloatParam>()->Value());
        float minC = 0.0f, maxC = 0.0f;
        unsigned int colTabSize = 0;
        // colour
        switch (parts.GetColourDataType()) {
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                glEnable(GL_TEXTURE_1D);
                view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
                if ((cgtf != NULL) && ((*cgtf)())) {
                    glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                    colTabSize = cgtf->TextureSize();
                } else {
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
        switch (parts.GetVertexDataType()) {
            case MultiParticleDataCall::Particles::VERTDATA_NONE:
                break;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                glUniform4f(this->newShader->ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
                break;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                glUniform4f(this->newShader->ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
                break;
            default:
                break;
        }


        unsigned int colBytes, vertBytes, colStride, vertStride;
        this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride);

        //currBuf = 0;
        UINT64 numVerts, vertCounter;
        // does all data reside interleaved in the same memory?
        if ((reinterpret_cast<const ptrdiff_t>(parts.GetColourData()) 
                - reinterpret_cast<const ptrdiff_t>(parts.GetVertexData()) <= vertStride
                && vertStride == colStride) || colStride == 0)  {
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

#ifdef CHRONOTIMING
                        before = std::chrono::high_resolution_clock::now();
#endif
                        this->waitSignal(fences[currBuf]);
#ifdef CHRONOTIMING
                        after = std::chrono::high_resolution_clock::now();
                        deltas.push_back(static_cast<std::chrono::steady_clock::time_point>(after - before));
#endif
                        //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "memcopying %u bytes from %016" PRIxPTR " to %016" PRIxPTR "\n", vertsThisTime * vertStride, whence, mem);
                        memcpy(mem, whence, vertsThisTime * vertStride);
                        glFlushMappedNamedBufferRangeEXT(theSingleBuffer, bufSize * currBuf, vertsThisTime * vertStride);
                        //glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
                        //glUniform1i(this->newShader->ParameterLocation("instanceOffset"), numVerts * currBuf);
                        glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);

                        //this->setPointers(parts, this->theSingleBuffer, reinterpret_cast<const void *>(currVert - whence), this->theSingleBuffer, reinterpret_cast<const void *>(currCol - whence));
                        //glBindBuffer(GL_ARRAY_BUFFER, 0);
                        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer, bufSize * currBuf, bufSize);
                        glDrawArrays(GL_POINTS, 0, vertsThisTime);
                        //glDrawArraysInstanced(GL_POINTS, 0, 1, vertsThisTime);
                        this->queueSignal(fences[currBuf]);

                        currBuf = (currBuf + 1) % this->numBuffers;
                        vertCounter += vertsThisTime;
                        currVert += vertsThisTime * vertStride;
                        currCol += vertsThisTime * colStride;
                        //break;
                    }
        } else {

        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        //glDisableClientState(GL_COLOR_ARRAY);
        //glDisableClientState(GL_VERTEX_ARRAY);
        //glDisableVertexAttribArrayARB(colIdxAttribLoc);
        glDisable(GL_TEXTURE_1D);
        newShader->Disable();
#ifdef CHRONOTIMING
        printf("waitSignal times:\n");
        for (auto d : deltas) {
            printf("%u, ", d);
        }
        printf("\n");
#endif
    }

    c2->Unlock();


    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
#ifdef DEBUG_BLAHBLAH
    glDisable(GL_DEBUG_OUTPUT);
    glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif

//	timer.EndFrame();

    return true;
}
