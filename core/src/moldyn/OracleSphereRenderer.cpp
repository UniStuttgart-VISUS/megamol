/*
 * OracleSphereRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/moldyn/OracleSphereRenderer.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include <GL/glu.h>
#include "vislib/assert.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/Trace.h"
#include "vislib/math/Vector.h"

using namespace megamol::core;


/*
 * moldyn::OracleSphereRenderer::OracleSphereRenderer
 */
moldyn::OracleSphereRenderer::OracleSphereRenderer(void) : Renderer3DModule(),
        sphereShader(), getDataSlot("getdata", "Connects to the data source"),
        getTFSlot("gettransferfunction", "Connects to the transfer function module"),
        getClipPlaneSlot("getclipplane", "Connects to a clipping plane module"),
        greyTF(0), fbo(), mixShader() {

    this->getDataSlot.SetCompatibleCall<moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getClipPlaneSlot.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);
}


/*
 * moldyn::OracleSphereRenderer::~OracleSphereRenderer
 */
moldyn::OracleSphereRenderer::~OracleSphereRenderer(void) {
    this->Release();
}


/*
 * moldyn::OracleSphereRenderer::create
 */
bool moldyn::OracleSphereRenderer::create(void) {
    ASSERT(IsAvailable());

    this->fbo.Create(1, 1);

    vislib::graphics::gl::ShaderSource vert, frag;

    if (!instance()->ShaderSourceFactory().MakeShaderSource("oraclesphere::vertex", vert)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("oraclesphere::fragment", frag)) {
        return false;
    }

    //printf("\nVertex Shader:\n%s\n\nFragment Shader:\n%s\n",
    //    vert.WholeCode().PeekBuffer(),
    //    frag.WholeCode().PeekBuffer());

    try {
        if (!this->sphereShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile sphere shader: Unknown error\n");
            return false;
        }

    } catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (@%s): %s\n", 
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()) ,ce.GetMsgA());
        return false;
    } catch(vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: %s\n", e.GetMsgA());
        return false;
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: Unknown exception\n");
        return false;
    }

    if (!instance()->ShaderSourceFactory().MakeShaderSource("oraclesphere::mixvertex", vert)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("oraclesphere::mixfragment", frag)) {
        return false;
    }

    //printf("\nVertex Shader:\n%s\n\nFragment Shader:\n%s\n",
    //    vert.WholeCode().PeekBuffer(),
    //    frag.WholeCode().PeekBuffer());

    try {
        if (!this->mixShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile sphere shader: Unknown error\n");
            return false;
        }

    } catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (@%s): %s\n", 
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()) ,ce.GetMsgA());
        return false;
    } catch(vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: %s\n", e.GetMsgA());
        return false;
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: Unknown exception\n");
        return false;
    }


    glEnable(GL_TEXTURE_1D);
    glGenTextures(1, &this->greyTF);
    unsigned char tex[6] = {
        0, 0, 0,  255, 255, 255
    };
    glBindTexture(GL_TEXTURE_1D, this->greyTF);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);

    glDisable(GL_TEXTURE_1D);

    return true;
}


/*
 * moldyn::OracleSphereRenderer::GetExtents
 */
bool moldyn::OracleSphereRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if ((c2 != NULL) && ((*c2)(1))) {
        cr->SetTimeFramesCount(c2->FrameCount());
        cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();

        float scaling = c2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }
        cr->AccessBoundingBoxes().MakeScaledWorld(scaling);

    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }

    return true;
}


/*
 * moldyn::OracleSphereRenderer::release
 */
void moldyn::OracleSphereRenderer::release(void) {
    this->fbo.Release();
    this->sphereShader.Release();
    ::glDeleteTextures(1, &this->greyTF);
}


/*
 * moldyn::OracleSphereRenderer::Render
 */
bool moldyn::OracleSphereRenderer::Render(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    float scaling = 1.0f;
    if (c2 != NULL) {
        c2->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*c2)(1)) return false;

        // calculate scaling
        scaling = c2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }

        c2->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*c2)(0)) return false;
    } else {
        return false;
    }

    GLint viewport[4];
    ::glGetIntegerv(GL_VIEWPORT, viewport);
    if ((this->fbo.GetWidth() != static_cast<UINT>(viewport[2]))
            || (this->fbo.GetHeight() != static_cast<UINT>(viewport[3]))) {
        this->fbo.Release();

        vislib::graphics::gl::FramebufferObject::ColourAttachParams cap[2];
        cap[0].format = GL_RGBA;
        cap[0].internalFormat = GL_RGBA32F;
        cap[0].type = GL_FLOAT;
        cap[1].format = GL_RGBA;
        cap[1].internalFormat = GL_RGBA8;
        cap[1].type = GL_UNSIGNED_BYTE;

        vislib::graphics::gl::FramebufferObject::DepthAttachParams dap;
        dap.format = GL_DEPTH_COMPONENT24;
        dap.state = vislib::graphics::gl::FramebufferObject::ATTACHMENT_RENDERBUFFER;

        vislib::graphics::gl::FramebufferObject::StencilAttachParams sap;
        sap.format = GL_STENCIL_INDEX;
        sap.state = vislib::graphics::gl::FramebufferObject::ATTACHMENT_DISABLED;

        if (!this->fbo.Create(viewport[2], viewport[3], 2, cap, dap, sap)) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Could not create framebuffer object");
            throw vislib::Exception("Could not create framebuffer object", __FILE__, __LINE__);
        }
    }

    UINT oldlevel = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_NONE);

    //this->fbo.Enable(0);
    //::glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    //::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //this->fbo.Disable();
    //this->fbo.Enable(1);
    //::glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    //::glClear(GL_COLOR_BUFFER_BIT);
    //this->fbo.Disable();

    VERIFY((this->fbo.EnableMultiple(2, GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT) == GL_NO_ERROR));

    ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<view::CallClipPlane>();
    float clipDat[4];
    float clipCol[3];
    if ((ccp != NULL) && (*ccp)()) {
        clipDat[0] = ccp->GetPlane().Normal().X();
        clipDat[1] = ccp->GetPlane().Normal().Y();
        clipDat[2] = ccp->GetPlane().Normal().Z();
        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
        clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
        clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
        clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;

    } else {
        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
        clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
    }

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    float viewportStuff[4] = {
        cr->GetCameraParameters()->TileRect().Left(),
        cr->GetCameraParameters()->TileRect().Bottom(),
        cr->GetCameraParameters()->TileRect().Width(),
        cr->GetCameraParameters()->TileRect().Height()};
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    this->sphereShader.Enable();

    glScalef(scaling, scaling, scaling);

    glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"),
        1, viewportStuff);
    glUniform3fv(this->sphereShader.ParameterLocation("camIn"),
        1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fv(this->sphereShader.ParameterLocation("camRight"),
        1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fv(this->sphereShader.ParameterLocation("camUp"),
        1, cr->GetCameraParameters()->Up().PeekComponents());
    glUniform1f(this->sphereShader.ParameterLocation("datascale"), scaling); // not nice, but ok for now

    glUniform4fv(this->sphereShader.ParameterLocation("clipDat"), 1, clipDat);
    glUniform3fv(this->sphereShader.ParameterLocation("clipCol"), 1, clipCol);

    if (c2 != NULL) {
        unsigned int cial = glGetAttribLocationARB(this->sphereShader, "colIdx");

        for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
            MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);
            float minC = 0.0f, maxC = 0.0f;
            unsigned int colTabSize = 0;

            // colour
            switch (parts.GetColourDataType()) {
                case MultiParticleDataCall::Particles::COLDATA_NONE:
                    glColor3ubv(parts.GetGlobalColour());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(3, GL_UNSIGNED_BYTE,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(4, GL_UNSIGNED_BYTE,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(3, GL_FLOAT,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(4, GL_FLOAT,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                    glEnableVertexAttribArrayARB(cial);
                    glVertexAttribPointerARB(cial, 1, GL_FLOAT, GL_FALSE,
                        parts.GetColourDataStride(), parts.GetColourData());

                    glEnable(GL_TEXTURE_1D);

                    ::glActiveTextureARB(GL_TEXTURE0);
                    view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
                    if ((cgtf != NULL) && ((*cgtf)())) {
                        glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                        colTabSize = cgtf->TextureSize();
                    } else {
                        glBindTexture(GL_TEXTURE_1D, this->greyTF);
                        colTabSize = 2;
                    }

                    glUniform1i(this->sphereShader.ParameterLocation("colTab"), 0);
                    minC = parts.GetMinColourIndexValue();
                    maxC = parts.GetMaxColourIndexValue();
                    glColor3ub(127, 127, 127);
                } break;
                default:
                    glColor3ub(127, 127, 127);
                    break;
            }

            // radius and position
            switch (parts.GetVertexDataType()) {
                case MultiParticleDataCall::Particles::VERTDATA_NONE:
                    continue;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4f(this->sphereShader.ParameterLocation("inConsts1"),
                        parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
                    glVertexPointer(3, GL_FLOAT,
                        parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4f(this->sphereShader.ParameterLocation("inConsts1"),
                        -1.0f, minC, maxC, float(colTabSize));
                    glVertexPointer(4, GL_FLOAT,
                        parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                default:
                    continue;
            }

            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableVertexAttribArrayARB(cial);
            glDisable(GL_TEXTURE_1D);
        }

        c2->Unlock();

    }

    this->sphereShader.Disable();

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    this->fbo.Disable();

    ::glDisable(GL_BLEND);
    this->mixShader.Enable();

    this->mixShader.SetParameter("paramTex", 0);
    this->mixShader.SetParameter("colourTex", 1);

    glUniform4fv(this->mixShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv(this->mixShader.ParameterLocation("globCamPos"), 1, cr->GetCameraParameters()->EyePosition().PeekCoordinates());

    double mvMat[16];
    double pMat[16];
    int viewPort[4];
    ::glGetDoublev(GL_MODELVIEW_MATRIX, mvMat);
    ::glGetDoublev(GL_PROJECTION_MATRIX, pMat);
    ::glGetIntegerv(GL_VIEWPORT, viewPort);
    double tmpval[6];
    vislib::math::ShallowPoint<double, 3> p1(tmpval);
    vislib::math::ShallowPoint<double, 3> p2(tmpval + 3);

    ::gluUnProject(0.0, 0.0, 0.0, mvMat, pMat, viewPort, &tmpval[0], &tmpval[1], &tmpval[2]);
    ::gluUnProject(0.0, 0.0, 1.0, mvMat, pMat, viewPort, &tmpval[3], &tmpval[4], &tmpval[5]);
    vislib::math::Vector<double, 3> ray00 = p1 - p2;
    ray00.Normalise();

    ::gluUnProject(static_cast<double>(viewPort[2]), 0.0, 0.0, mvMat, pMat, viewPort, &tmpval[0], &tmpval[1], &tmpval[2]);
    ::gluUnProject(static_cast<double>(viewPort[2]), 0.0, 1.0, mvMat, pMat, viewPort, &tmpval[3], &tmpval[4], &tmpval[5]);
    vislib::math::Vector<double, 3> ray10 = p1 - p2;
    ray10.Normalise();

    ::gluUnProject(0.0, static_cast<double>(viewPort[3]), 0.0, mvMat, pMat, viewPort, &tmpval[0], &tmpval[1], &tmpval[2]);
    ::gluUnProject(0.0, static_cast<double>(viewPort[3]), 1.0, mvMat, pMat, viewPort, &tmpval[3], &tmpval[4], &tmpval[5]);
    vislib::math::Vector<double, 3> ray01 = p1 - p2;
    ray01.Normalise();

    ::gluUnProject(static_cast<double>(viewPort[2]), static_cast<double>(viewPort[3]), 0.0, mvMat, pMat, viewPort, &tmpval[0], &tmpval[1], &tmpval[2]);
    ::gluUnProject(static_cast<double>(viewPort[2]), static_cast<double>(viewPort[3]), 1.0, mvMat, pMat, viewPort, &tmpval[3], &tmpval[4], &tmpval[5]);
    vislib::math::Vector<double, 3> ray11 = p1 - p2;
    ray11.Normalise();

    ::glEnable(GL_TEXTURE_2D);
    ::glActiveTextureARB(GL_TEXTURE0);
    this->fbo.BindColourTexture(0);
    ::glActiveTextureARB(GL_TEXTURE1);
    this->fbo.BindColourTexture(1);

    // TODO: This is always a headlight, but should be the configured light
    float lightPos[4];
    ::glGetLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    vislib::math::Vector<float, 3> lightPosV;
    lightPosV += cr->GetCameraParameters()->EyeRightVector() * lightPos[0];
    lightPosV += cr->GetCameraParameters()->EyeUpVector() * lightPos[1];
    lightPosV -= cr->GetCameraParameters()->EyeDirection() * lightPos[2];
    this->mixShader.SetParameterArray3("lightPos", 1, lightPosV.PeekComponents());

    ::glMatrixMode(GL_PROJECTION);
    ::glPushMatrix();
    ::glLoadIdentity();
    ::glMatrixMode(GL_MODELVIEW);
    ::glPushMatrix();
    ::glLoadIdentity();

    ::glDisable(GL_DEPTH_TEST);

    ::glColor3ub(255, 255, 255);
    ::glBegin(GL_QUADS);
    ::glMultiTexCoord3dvARB(GL_TEXTURE1, ray00.PeekComponents());
    ::glTexCoord2f(0.0f, 0.0f);
    ::glVertex2i(-1, -1);
    ::glMultiTexCoord3dvARB(GL_TEXTURE1, ray10.PeekComponents());
    ::glTexCoord2f(1.0f, 0.0f);
    ::glVertex2i( 1, -1);
    ::glMultiTexCoord3dvARB(GL_TEXTURE1, ray11.PeekComponents());
    ::glTexCoord2f(1.0f, 1.0f);
    ::glVertex2i( 1,  1);
    ::glMultiTexCoord3dvARB(GL_TEXTURE1, ray01.PeekComponents());
    ::glTexCoord2f(0.0f, 1.0f);
    ::glVertex2i(-1,  1);
    ::glEnd();

    ::glEnable(GL_DEPTH_TEST);

    ::glMatrixMode(GL_PROJECTION);
    ::glPopMatrix();
    ::glMatrixMode(GL_MODELVIEW);
    ::glPopMatrix();

    ::glActiveTextureARB(GL_TEXTURE1);
    ::glBindTexture(GL_TEXTURE_2D, 0);
    ::glActiveTextureARB(GL_TEXTURE0);
    ::glBindTexture(GL_TEXTURE_2D, 0);

    this->mixShader.Disable();

    vislib::Trace::GetInstance().SetLevel(oldlevel);

    return true;
}
