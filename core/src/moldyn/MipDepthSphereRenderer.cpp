/*
 * MipDepthSphereRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/moldyn/MipDepthSphereRenderer.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/assert.h"
#include "vislib/Trace.h"

using namespace megamol::core;


/*
 * moldyn::MipDepthSphereRenderer::MipDepthSphereRenderer
 */
moldyn::MipDepthSphereRenderer::MipDepthSphereRenderer(void) : Renderer3DModule(),
        sphereShader(), initDepthShader(), fbo(),
        getDataSlot("getdata", "Connects to the data source"),
        getTFSlot("gettransferfunction", "Connects to the transfer function module"),
        getClipPlaneSlot("getclipplane", "Connects to a clipping plane module"),
        greyTF(0) {

    this->getDataSlot.SetCompatibleCall<moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getClipPlaneSlot.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);
}


/*
 * moldyn::MipDepthSphereRenderer::~MipDepthSphereRenderer
 */
moldyn::MipDepthSphereRenderer::~MipDepthSphereRenderer(void) {
    this->Release();
}


/*
 * moldyn::MipDepthSphereRenderer::create
 */
bool moldyn::MipDepthSphereRenderer::create(void) {
    ASSERT(IsAvailable());

    vislib::graphics::gl::ShaderSource vert, frag;

    const char *shaderName = "sphere";
    try {

        shaderName = "sphere";
        if (!instance()->ShaderSourceFactory().MakeShaderSource("simplesphere::vertex", vert)) { return false; }
        if (!instance()->ShaderSourceFactory().MakeShaderSource("simplesphere::fragment", frag)) { return false; }
        //printf("\nVertex Shader:\n%s\n\nFragment Shader:\n%s\n",
        //    vert.WholeCode().PeekBuffer(),
        //    frag.WholeCode().PeekBuffer());
        if (!this->sphereShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile sphere shader: Unknown error\n");
            return false;
        }

        shaderName = "depth-init";
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::init::vertex", vert)) { return false; }
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::init::fragment", frag)) { return false; }
        if (!this->initDepthShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile sphere shader: Unknown error\n");
            return false;
        }

    } catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile %s shader (@%s): %s\n", shaderName,
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()) ,ce.GetMsgA());
        return false;
    } catch(vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile %s shader: %s\n", shaderName, e.GetMsgA());
        return false;
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile %s shader: Unknown exception\n", shaderName);
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

    this->fbo.Create(1, 1); // year, right.

    // textures for mipmap storage/generation
    ::glEnable(GL_TEXTURE_2D);
    float f[1] = { 0.5f };
    ::glGenTextures(1, &this->mipDepth);
    ::glBindTexture(GL_TEXTURE_2D, this->mipDepth);
    ::glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE16F_ARB, 1, 1, 0, GL_LUMINANCE, GL_FLOAT, f);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    ::glGenTextures(1, &this->mipDepthTmp);
    ::glBindTexture(GL_TEXTURE_2D, this->mipDepthTmp);
    ::glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE16F_ARB, 1, 1, 0, GL_LUMINANCE, GL_FLOAT, f);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    ::glBindTexture(GL_TEXTURE_2D, 0);

    return true;
}


/*
 * moldyn::MipDepthSphereRenderer::GetExtents
 */
bool moldyn::MipDepthSphereRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if ((c2 != NULL) && ((*c2)(1))) {
        cr->SetTimeFramesCount(c2->FrameCount());
        cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();

        float scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
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
 * moldyn::MipDepthSphereRenderer::release
 */
void moldyn::MipDepthSphereRenderer::release(void) {
    this->sphereShader.Release();
    ::glDeleteTextures(1, &this->greyTF);
}


/*
 * moldyn::MipDepthSphereRenderer::Render
 */
bool moldyn::MipDepthSphereRenderer::Render(Call& call) {
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

    // update fbo size, if required
    GLint viewport[4];
    ::glGetIntegerv(GL_VIEWPORT, viewport);
    if ((this->fbo.GetWidth() != static_cast<UINT>(viewport[2]))
            || (this->fbo.GetHeight() != static_cast<UINT>(viewport[3]))) {
        this->fbo.Release();
        this->fbo.Create(viewport[2], viewport[3],
                GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, // colour buffer
                vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE,
                GL_DEPTH_COMPONENT24); // depth buffer
    }

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

    // z-buffer-filling
#if defined(DEBUG) || defined(_DEBUG)
    UINT oldlevel = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_NONE);
#endif
    this->fbo.Enable();
    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    this->initDepthShader.Enable();

    ::glScalef(scaling, scaling, scaling);

    glUniform4fv(this->initDepthShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv(this->initDepthShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fv(this->initDepthShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fv(this->initDepthShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
    // no clipping plane for now
    glColor4ub(192, 192, 192, 255);
    glDisableClientState(GL_COLOR_ARRAY);

    for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);

        // radius and position
        switch (parts.GetVertexDataType()) {
            case MultiParticleDataCall::Particles::VERTDATA_NONE:
                continue;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                glEnableClientState(GL_VERTEX_ARRAY);
                glUniform4f(this->initDepthShader.ParameterLocation("inConsts1"), parts.GetGlobalRadius(), 0.0f, 0.0f, 0.0f);
                glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                break;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                glEnableClientState(GL_VERTEX_ARRAY);
                glUniform4f(this->initDepthShader.ParameterLocation("inConsts1"), -1.0f, 0.0f, 0.0f, 0.0f);
                glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                break;
            default:
                continue;
        }

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

        glDisableClientState(GL_VERTEX_ARRAY);
    }

    this->initDepthShader.Disable();
    this->fbo.Disable();
#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(oldlevel);
#endif

    // DEBUG OUTPUT OF FBO
    ::glEnable(GL_TEXTURE_2D);
    ::glDisable(GL_LIGHTING);
    ::glDisable(GL_DEPTH_TEST);
    this->fbo.BindDepthTexture();
    //this->fbo.BindColourTexture();
    ::glMatrixMode(GL_PROJECTION);
    ::glPushMatrix();
    ::glLoadIdentity();
    ::glMatrixMode(GL_MODELVIEW);
    ::glPushMatrix();
    ::glLoadIdentity();
    ::glColor3ub(255, 255, 255);
    ::glBegin(GL_QUADS);
    ::glTexCoord2f(0.0f, 0.0f);
    ::glVertex2i(-1, -1);
    ::glTexCoord2f(1.0f, 0.0f);
    ::glVertex2i(1, -1);
    ::glTexCoord2f(1.0f, 1.0f);
    ::glVertex2i(1, 1);
    ::glTexCoord2f(0.0f, 1.0f);
    ::glVertex2i(-1, 1);
    ::glEnd();
    ::glMatrixMode(GL_PROJECTION);
    ::glPopMatrix();
    ::glMatrixMode(GL_MODELVIEW);
    ::glPopMatrix();
    ::glBindTexture(GL_TEXTURE_2D, 0);

    //this->sphereShader.Enable();

    //glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"),
    //    1, viewportStuff);
    //glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"),
    //    1, cr->GetCameraParameters()->Front().PeekComponents());
    //glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"),
    //    1, cr->GetCameraParameters()->Right().PeekComponents());
    //glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"),
    //    1, cr->GetCameraParameters()->Up().PeekComponents());

    //glUniform4fvARB(this->sphereShader.ParameterLocation("clipDat"), 1, clipDat);
    //glUniform3fvARB(this->sphereShader.ParameterLocation("clipCol"), 1, clipCol);

    //glScalef(scaling, scaling, scaling);

    //if (c2 != NULL) {
    //    unsigned int cial = glGetAttribLocationARB(this->sphereShader, "colIdx");

    //    for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
    //        MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);
    //        float minC = 0.0f, maxC = 0.0f;
    //        unsigned int colTabSize = 0;

    //        // colour
    //        switch (parts.GetColourDataType()) {
    //            case MultiParticleDataCall::Particles::COLDATA_NONE:
    //                glColor3ubv(parts.GetGlobalColour());
    //                break;
    //            case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
    //                glEnableClientState(GL_COLOR_ARRAY);
    //                glColorPointer(3, GL_UNSIGNED_BYTE,
    //                    parts.GetColourDataStride(), parts.GetColourData());
    //                break;
    //            case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
    //                glEnableClientState(GL_COLOR_ARRAY);
    //                glColorPointer(4, GL_UNSIGNED_BYTE,
    //                    parts.GetColourDataStride(), parts.GetColourData());
    //                break;
    //            case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
    //                glEnableClientState(GL_COLOR_ARRAY);
    //                glColorPointer(3, GL_FLOAT,
    //                    parts.GetColourDataStride(), parts.GetColourData());
    //                break;
    //            case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
    //                glEnableClientState(GL_COLOR_ARRAY);
    //                glColorPointer(4, GL_FLOAT,
    //                    parts.GetColourDataStride(), parts.GetColourData());
    //                break;
    //            case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
    //                glEnableVertexAttribArrayARB(cial);
    //                glVertexAttribPointerARB(cial, 1, GL_FLOAT, GL_FALSE,
    //                    parts.GetColourDataStride(), parts.GetColourData());

    //                glEnable(GL_TEXTURE_1D);

    //                view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
    //                if ((cgtf != NULL) && ((*cgtf)())) {
    //                    glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
    //                    colTabSize = cgtf->TextureSize();
    //                } else {
    //                    glBindTexture(GL_TEXTURE_1D, this->greyTF);
    //                    colTabSize = 2;
    //                }

    //                glUniform1iARB(this->sphereShader.ParameterLocation("colTab"), 0);
    //                minC = parts.GetMinColourIndexValue();
    //                maxC = parts.GetMaxColourIndexValue();
    //                glColor3ub(127, 127, 127);
    //            } break;
    //            default:
    //                glColor3ub(127, 127, 127);
    //                break;
    //        }

    //        // radius and position
    //        switch (parts.GetVertexDataType()) {
    //            case MultiParticleDataCall::Particles::VERTDATA_NONE:
    //                continue;
    //            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
    //                glEnableClientState(GL_VERTEX_ARRAY);
    //                glUniform4fARB(this->sphereShader.ParameterLocation("inConsts1"),
    //                    parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
    //                glVertexPointer(3, GL_FLOAT,
    //                    parts.GetVertexDataStride(), parts.GetVertexData());
    //                break;
    //            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
    //                glEnableClientState(GL_VERTEX_ARRAY);
    //                glUniform4fARB(this->sphereShader.ParameterLocation("inConsts1"),
    //                    -1.0f, minC, maxC, float(colTabSize));
    //                glVertexPointer(4, GL_FLOAT,
    //                    parts.GetVertexDataStride(), parts.GetVertexData());
    //                break;
    //            default:
    //                continue;
    //        }

    //        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

    //        glDisableClientState(GL_COLOR_ARRAY);
    //        glDisableClientState(GL_VERTEX_ARRAY);
    //        glDisableVertexAttribArrayARB(cial);
    //        glDisable(GL_TEXTURE_1D);
    //    }

    //    c2->Unlock();

    //}

    //this->sphereShader.Disable();

    //glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    c2->Unlock();

    return true;
}
