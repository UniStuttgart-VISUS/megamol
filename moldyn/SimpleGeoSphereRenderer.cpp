/*
 * SimpleGeoSphereRenderer.cpp
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "SimpleGeoSphereRenderer.h"
#include "view/CallRender3D.h"
#include "vislib/ShaderSource.h"
#include "CoreInstance.h"
#include "view/CallGetTransferFunction.h"
#include "vislib/Matrix.h"
#include "vislib/ShallowMatrix.h"

using namespace megamol::core;


/*
 * moldyn::SimpleGeoSphereRenderer::SimpleGeoSphereRenderer
 */
moldyn::SimpleGeoSphereRenderer::SimpleGeoSphereRenderer(void) : AbstractSimpleSphereRenderer(),
        sphereShader() {
    // intentionally empty
}


/*
 * moldyn::SimpleGeoSphereRenderer::~SimpleGeoSphereRenderer
 */
moldyn::SimpleGeoSphereRenderer::~SimpleGeoSphereRenderer(void) {
    this->Release();
}


/*
 * moldyn::SimpleGeoSphereRenderer::create
 */
bool moldyn::SimpleGeoSphereRenderer::create(void) {
    using vislib::sys::Log;
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        Log::DefaultLog.WriteError("Failed to init OpenGL extensions: GLSLShader");
        return false;
    }
    if (!vislib::graphics::gl::GLSLGeometryShader::InitialiseExtensions()) {
        Log::DefaultLog.WriteError("Failed to init OpenGL extensions: GLSLGeometryShader");
        return false;
    }
    if (glh_init_extensions("GL_EXT_gpu_shader4 GL_EXT_geometry_shader4 GL_EXT_bindable_uniform") != GL_TRUE) {
        Log::DefaultLog.WriteError("Failed to init OpenGL extensions: shader4 and bindable_uniform");
        return false;
    }
    if (glh_init_extensions("GL_VERSION_2_0") != GL_TRUE) {
        Log::DefaultLog.WriteError("Failed to init OpenGL extensions: GL_VERSION_2_0");
        return false;
    }
    if (glh_init_extensions("GL_ARB_vertex_shader GL_ARB_vertex_program GL_ARB_shader_objects") != GL_TRUE) {
        Log::DefaultLog.WriteError("Failed to init OpenGL extensions: arb shader");
        return false;
    }

    vislib::graphics::gl::ShaderSource vert, geom, frag;

    if (!instance()->ShaderSourceFactory().MakeShaderSource("simplegeosphere::vertex", vert)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("simplegeosphere::geometry", geom)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("simplegeosphere::fragment", frag)) {
        return false;
    }

    const char *buildState = "compile";
    try {
        if (!this->sphereShader.Compile(vert.Code(), vert.Count(), geom.Code(), geom.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile sphere shader: Unknown error\n");
            return false;
        }
        // not needed any more because its version 1.5 GLSL
        //buildState = "setup";
        //this->sphereShader.SetProgramParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
        //this->sphereShader.SetProgramParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
        //this->sphereShader.SetProgramParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 4);
        buildState = "link";
        if (!this->sphereShader.Link()) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to link sphere shader: Unknown error\n");
            return false;
        }

    } catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to %s sphere shader (@%s): %s\n", buildState, 
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()) ,ce.GetMsgA());
        return false;
    } catch(vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to %s sphere shader: %s\n", buildState, e.GetMsgA());
        return false;
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to %s sphere shader: Unknown exception\n", buildState);
        return false;
    }

    return AbstractSimpleSphereRenderer::create();
}


/*
 * moldyn::SimpleGeoSphereRenderer::release
 */
void moldyn::SimpleGeoSphereRenderer::release(void) {
    this->sphereShader.Release();
    AbstractSimpleSphereRenderer::release();
}


/*
 * moldyn::SimpleGeoSphereRenderer::Render
 */
bool moldyn::SimpleGeoSphereRenderer::Render(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    float scaling = 1.0f;
    MultiParticleDataCall *c2 = this->getData(static_cast<unsigned int>(cr->Time()), scaling);
    if (c2 == NULL) return false;

    float clipDat[4];
    float clipCol[4];
    this->getClipData(clipDat, clipCol);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

    glScalef(scaling, scaling, scaling);

    float viewportStuff[4];
    ::glGetFloatv(GL_VIEWPORT, viewportStuff);
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // Get GL_MODELVIEW matrix
    GLfloat modelMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelMatrix(&modelMatrix_column[0]);

    // Get GL_PROJECTION matrix
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);

    // Compute modelviewprojection matrix
    vislib::math::Matrix<GLfloat, 4, vislib::math::ROW_MAJOR> modelProjMatrix = projMatrix * modelMatrix;

    // Get light position
    GLfloat lightPos[4];
    glGetLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    this->sphereShader.Enable();

    // Set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());

    glUniformMatrix4fvARB(this->sphereShader.ParameterLocation("modelview"), 1, false, modelMatrix_column);
    glUniformMatrix4fvARB(this->sphereShader.ParameterLocation("proj"), 1, false, projMatrix_column);
    glUniform4fvARB(this->sphereShader.ParameterLocation("lightPos"), 1, lightPos);

    glUniform4fvARB(this->sphereShader.ParameterLocation("clipDat"), 1, clipDat);
    glUniform4fvARB(this->sphereShader.ParameterLocation("clipCol"), 1, clipCol);

    // Vertex attributes
    GLint vertexPos = glGetAttribLocation(this->sphereShader, "vertex");
    GLint vertexColor = glGetAttribLocation(this->sphereShader, "color");

    for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);
        float minC = 0.0f, maxC = 0.0f;
        unsigned int colTabSize = 0;

        // colour
        switch (parts.GetColourDataType()) {
            case MultiParticleDataCall::Particles::COLDATA_NONE: {
                const unsigned char* gc = parts.GetGlobalColour();
                ::glVertexAttrib3d(vertexColor,
                    static_cast<double>(gc[0]) / 255.0,
                    static_cast<double>(gc[1]) / 255.0,
                    static_cast<double>(gc[2]) / 255.0);
            } break;
            case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                ::glEnableVertexAttribArray(vertexColor);
                ::glVertexAttribPointer(vertexColor, 3, GL_UNSIGNED_BYTE, GL_TRUE, parts.GetColourDataStride(), parts.GetColourData());
                break;
            case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                ::glEnableVertexAttribArray(vertexColor);
                ::glVertexAttribPointer(vertexColor, 4, GL_UNSIGNED_BYTE, GL_TRUE, parts.GetColourDataStride(), parts.GetColourData());
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                ::glEnableVertexAttribArray(vertexColor);
                ::glVertexAttribPointer(vertexColor, 3, GL_FLOAT, GL_TRUE, parts.GetColourDataStride(), parts.GetColourData());
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                ::glEnableVertexAttribArray(vertexColor);
                ::glVertexAttribPointer(vertexColor, 4, GL_FLOAT, GL_TRUE, parts.GetColourDataStride(), parts.GetColourData());
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                ::glEnableVertexAttribArray(vertexColor);
                ::glVertexAttribPointer(vertexColor, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), parts.GetColourData());

                glEnable(GL_TEXTURE_1D);

                view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
                if ((cgtf != NULL) && ((*cgtf)())) {
                    glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                    colTabSize = cgtf->TextureSize();
                } else {
                    glBindTexture(GL_TEXTURE_1D, this->greyTF);
                    colTabSize = 2;
                }

                glUniform1iARB(this->sphereShader.ParameterLocation("colTab"), 0);
                minC = parts.GetMinColourIndexValue();
                maxC = parts.GetMaxColourIndexValue();
            } break;
            default:
                ::glVertexAttrib3f(vertexColor, 0.5f, 0.5f, 0.5f);
                break;
        }

        // radius and position
        switch (parts.GetVertexDataType()) {
            case MultiParticleDataCall::Particles::VERTDATA_NONE:
                continue;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                ::glEnableVertexAttribArray(vertexPos);
                ::glVertexAttribPointer(vertexPos, 3, GL_FLOAT, GL_FALSE, parts.GetVertexDataStride(), parts.GetVertexData());
                ::glUniform4fARB(this->sphereShader.ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
                break;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                ::glEnableVertexAttribArray(vertexPos);
                ::glVertexAttribPointer(vertexPos, 4, GL_FLOAT, GL_FALSE, parts.GetVertexDataStride(), parts.GetVertexData());
                ::glUniform4fARB(this->sphereShader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
                break;
            default:
                continue;
        }

        ::glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

        ::glDisableVertexAttribArray(vertexPos);
        ::glDisableVertexAttribArray(vertexColor);

        glDisable(GL_TEXTURE_1D);
    }

    c2->Unlock();

    this->sphereShader.Disable();

    return true;
}
