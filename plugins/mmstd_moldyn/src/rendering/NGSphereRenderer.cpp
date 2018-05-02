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
#include "vislib/math/mathfunctions.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"
#include <cmath>
#include <cinttypes>

#include <chrono>
#include <sstream>
#include <iterator>

//#define CHRONOTIMING

using namespace megamol::core;
using namespace megamol::stdplugin::moldyn::rendering;
#define MAP_BUFFER_LOCALLY
#define DEBUG_BLAHBLAH

const GLuint SSBObindingPoint = 2;
const GLuint SSBOcolorBindingPoint = 3;
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
    scalingParam("scaling", "scaling factor for particle radii"),
    colType(SimpleSphericalParticles::COLDATA_NONE), vertType(SimpleSphericalParticles::VERTDATA_NONE) {

    this->scalingParam << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->scalingParam);
}


/*
 * moldyn::NGSphereRenderer::~NGSphereRenderer
 */
NGSphereRenderer::~NGSphereRenderer(void) {
    this->Release();
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

    glGenVertexArrays(1, &this->vertArray);
    glBindVertexArray(this->vertArray);
    glBindVertexArray(0);

    //timer.SetNumRegions(4);
    //const char *regions[4] = {"Upload1", "Upload2", "Upload3", "Rendering"};
    //timer.SetRegionNames(4, regions);
    //timer.SetStatisticsFileName("fullstats.csv");
    //timer.SetSummaryFileName("summary.csv");
    //timer.SetMaximumFrames(20, 100);
    return AbstractSimpleSphereRenderer::create();
}

bool NGSphereRenderer::makeColorString(MultiParticleDataCall::Particles &parts, std::string &code, std::string &declaration, bool interleaved) {
    bool ret = true;
    switch (parts.GetColourDataType()) {
        case MultiParticleDataCall::Particles::COLDATA_NONE:
            declaration = "";
            code = "";
            break;
        case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
            ret = false;
            break;
        case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
            declaration = "    uint color;\n";
            if (interleaved) {
                code = "    theColor = unpackUnorm4x8(theBuffer[" NGS_THE_INSTANCE "+ instanceOffset].color);\n";
            } else {
                code = "    theColor = unpackUnorm4x8(theColBuffer[" NGS_THE_INSTANCE "+ instanceOffset].color);\n";
            }
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
            declaration = "    float r; float g; float b;\n";
            if (interleaved) {
                code = "    theColor = vec4(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].r,\n"
                    "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].g,\n"
                    "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].b, 1.0); \n";
            } else {
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
            } else {
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
            } else {
                code = "    theColIdx = theColBuffer[" NGS_THE_INSTANCE " + instanceOffset].colorIndex; \n";
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

bool NGSphereRenderer::makeVertexString(MultiParticleDataCall::Particles &parts, std::string &code, std::string &declaration, bool interleaved)  {
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
        } else {
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
        } else {
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

    unsigned int colBytes, vertBytes, colStride, vertStride;
    bool interleaved;
    this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride, interleaved);

    shaderMap::iterator i = theShaders.find(std::make_tuple(c, p, interleaved));
    if (i == theShaders.end()) {
        //instance()->ShaderSourceFactory().MakeShaderSource()

        vislib::SmartPtr<ShaderSource> v2 = new ShaderSource(*vert);
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

        } else {
            // we seem to have separate buffers for vertex and color data

            decl = "\nstruct SpherePosParams {\n" + vertDecl + "};\n";
            decl += "\nstruct SphereColParams {\n" + colDecl + "};\n";

            decl += "layout(packed, binding = " + std::to_string(SSBObindingPoint) + ") buffer shader_data {\n"
                "    SpherePosParams thePosBuffer[];\n"
                "};\n";
            decl += "layout(packed, binding = " + std::to_string(SSBOcolorBindingPoint) + ") buffer shader_data2 {\n"
                "    SphereColParams theColBuffer[];\n"
                "};\n";
        }
        std::string code = "\n";
        code += colCode;
        code += vertCode;
        declarationSnip = new ShaderSource::StringSnippet(decl.c_str());
        codeSnip = new ShaderSource::StringSnippet(code.c_str());
        v2->Insert(3, declarationSnip);
        v2->Insert(5, codeSnip);
        std::string s(v2->WholeCode());

        vislib::SmartPtr<ShaderSource> vss(v2);
        theShaders.emplace(std::make_pair(std::make_tuple(c, p, interleaved), makeShader(v2, frag)));
        i = theShaders.find(std::make_tuple(c, p, interleaved));
    }
    return i->second;
}


/*
 * moldyn::SimpleSphereRenderer::release
 */
void NGSphereRenderer::release(void) {
    //this->sphereShader.Release();
    // TODO release all shaders!
    glDeleteVertexArrays(1, &this->vertArray);
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

    interleaved = (std::abs(reinterpret_cast<const ptrdiff_t>(parts.GetColourData())
        - reinterpret_cast<const ptrdiff_t>(parts.GetVertexData())) <= vertStride
        && vertStride == colStride) || colStride == 0;
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
            case MultiParticleDataCall::Particles::COLDATA_NONE: {
                glUniform4f(this->newShader->ParameterLocation("globalCol"), 
                    static_cast<float>(parts.GetGlobalColour()[0]) / 255.0f,
                    static_cast<float>(parts.GetGlobalColour()[1]) / 255.0f,
                    static_cast<float>(parts.GetGlobalColour()[2]) / 255.0f,
                    1.0f);
            } break;
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
        bool interleaved;
        this->getBytesAndStride(parts, colBytes, vertBytes, colStride, vertStride, interleaved);

        //currBuf = 0;
        //UINT64 numVerts, vertCounter;

        // does all data reside interleaved in the same memory?
        if (interleaved)  {

            const GLuint numChunks = streamer.SetDataWithSize(parts.GetVertexData(), vertStride, vertStride,
                parts.GetCount(), 3, 32 * 1024 * 1024);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, streamer.GetHandle());
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, streamer.GetHandle());

            for(GLuint x = 0; x < numChunks; ++x) {
                GLuint numItems, sync;
                GLsizeiptr dstOff, dstLen;
                streamer.UploadChunk(x, numItems, sync, dstOff, dstLen);
                //streamer.UploadChunk<float, float>(x, [](float f) -> float { return f + 100.0; },
                //    numItems, sync, dstOff, dstLen);
                glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint,
                    this->streamer.GetHandle(), dstOff, dstLen);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(numItems));
                streamer.SignalCompletion(sync);
            }
        } else {

            const GLuint numChunks = streamer.SetDataWithSize(parts.GetVertexData(), vertStride, vertStride,
                parts.GetCount(), 3, 32 * 1024 * 1024);
            const GLuint colSize = colStreamer.SetDataWithItems(parts.GetColourData(), colStride, colStride,
                parts.GetCount(), 3, streamer.GetMaxNumItemsPerChunk());
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, streamer.GetHandle());
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, streamer.GetHandle());
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, colStreamer.GetHandle());
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOcolorBindingPoint, colStreamer.GetHandle());

            for (GLuint x = 0; x < numChunks; ++x) {
                GLuint numItems, sync, sync2;
                GLsizeiptr dstOff, dstLen, dstOff2, dstLen2;
                streamer.UploadChunk(x, numItems, sync, dstOff, dstLen);
                colStreamer.UploadChunk(x, numItems, sync2, dstOff2, dstLen2);
                glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint,
                    this->streamer.GetHandle(), dstOff, dstLen);
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBOcolorBindingPoint,
                    this->colStreamer.GetHandle(), dstOff2, dstLen2);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(numItems));
                streamer.SignalCompletion(sync);
                streamer.SignalCompletion(sync2);
            }

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
