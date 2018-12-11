/*
 * BlinnPhongRendererDeferred.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/CallRenderDeferred3D.h"
#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/sys/Log.h"
#include "mmcore/view/BlinnPhongRendererDeferred.h"

using namespace megamol;
using namespace megamol::core;


/*
 * view::BlinnPhongRendererDeferred::BlinnPhongRendererDeferred
 */
view::BlinnPhongRendererDeferred::BlinnPhongRendererDeferred(void)
    : AbstractRendererDeferred3D(),
    renderModeParam("renderMode", "The render mode to be used") {

    param::EnumParam *rm = new param::EnumParam(BLINN_PHONG);
    rm->SetTypePair(BLINN_PHONG, "BlinnPhong");
    rm->SetTypePair(DEPTH, "Depth");
    rm->SetTypePair(NORMAL, "Normal");
    rm->SetTypePair(COLOR, "Color");
    this->renderModeParam << rm;
    this->MakeSlotAvailable(&renderModeParam);
}


/*
 * view::BlinnPhongRendererDeferred::create
 */
bool view::BlinnPhongRendererDeferred::create(void) {

    vislib::graphics::gl::ShaderSource fragSrc;
    vislib::graphics::gl::ShaderSource vertSrc;

    CoreInstance *ci = this->GetCoreInstance();
    if (!ci) {
        return false;
    }

    if (!vislib::graphics::gl::FramebufferObject::InitialiseExtensions()) {
        return false;
    }

    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    // Try to load fragment shader
    if (!ci->ShaderSourceFactory().MakeShaderSource("deferred::blinnPhongFrag", fragSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, 
            "%s: Unable to load fragment shader source", this->ClassName() );
        return false;
    }
    // Try to load vertex shader
    if (!ci->ShaderSourceFactory().MakeShaderSource("deferred::vertex", vertSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, 
            "%s: Unable to load vertex shader source", this->ClassName() );
        return false;
    }
    try {
        if (!this->blinnPhongShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
    catch( vislib::Exception e ){
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, 
            "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    return true;
}


/*
 * view::BlinnPhongRendererDeferred::release
 */
void view::BlinnPhongRendererDeferred::release(void) {
    this->fbo.Release();
    this->blinnPhongShader.Release();
}


/*
 * view::BlinnPhongRendererDeferred::~BlinnPhongRendererDeferred
 */
view::BlinnPhongRendererDeferred::~BlinnPhongRendererDeferred(void) {
    this->Release();
}


/*
 * view::BlinnPhongRendererDeferred::GetExtents
 */
bool view::BlinnPhongRendererDeferred::GetExtents(Call& call) {

    CallRender3D *crIn = dynamic_cast<CallRender3D*>(&call);
    if (crIn == NULL) return false;

    CallRenderDeferred3D *crOut = this->rendererSlot.CallAs<CallRenderDeferred3D>();
    if (crOut == NULL) return false;

    if (!(*crOut)(CallRender3D::FnGetExtents)) return false;

    // Set extends of for incoming render call
    crIn->AccessBoundingBoxes() = crOut->GetBoundingBoxes();
    crIn->SetLastFrameTime(crOut->LastFrameTime());

    return true;
}


/*
 * view::BlinnPhongRendererDeferred::Render
 */
bool view::BlinnPhongRendererDeferred::Render(Call& call) {

    CallRender3D *crIn = dynamic_cast<CallRender3D*>(&call);
    if (crIn == NULL) return false;

    CallRenderDeferred3D *crOut = this->rendererSlot.CallAs<CallRenderDeferred3D>();
    if (crOut == NULL) return false;

    /* First render pass */
    
    crOut->SetCameraParameters(crIn->GetCameraParameters());

    // Get current viewport and recreate fbo if necessary
    float curVP[4];
    glGetFloatv(GL_VIEWPORT, curVP);

    if (!this->fbo.IsValid()) {
        if (!this->createFBO((UINT)curVP[2], (UINT)curVP[3]))
            return false;
    }
    else {
        if ((curVP[2] != this->fbo.GetWidth()) || (curVP[3] != this->fbo.GetHeight())) {
            this->fbo.Release();
            if (!this->createFBO((UINT)curVP[2],(UINT)curVP[3]))
                return false;
        }
    }
    
    crOut->SetOutputBuffer(&this->fbo);
    crOut->EnableOutputBuffer();

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Call for render
    (*crOut)(AbstractCallRender::FnRender);

    crOut->DisableOutputBuffer();

    /* Second render pass */

    this->blinnPhongShader.Enable();

    glUniform3fv(this->blinnPhongShader.ParameterLocation("camWS"), 1,
        crIn->GetCameraParameters()->Position().PeekCoordinates());
    glUniform2f(this->blinnPhongShader.ParameterLocation("clipPlanes"),
        crIn->GetCameraParameters()->NearClip(),
        crIn->GetCameraParameters()->FarClip());
    glUniform2f(this->blinnPhongShader.ParameterLocation("winSize"),
        curVP[2] - curVP[0], curVP[3] - curVP[1]);
    glUniform1i(this->blinnPhongShader.ParameterLocation("depthBuff"), 0);
    glUniform1i(this->blinnPhongShader.ParameterLocation("colorBuff"), 1);
    glUniform1i(this->blinnPhongShader.ParameterLocation("normalBuff"), 2);
    glUniform1i(this->blinnPhongShader.ParameterLocation("renderMode"), 
        this->renderModeParam.Param<core::param::EnumParam>()->Value());

    // Orthogonal projection for rendering the screen quad
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    
    //glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glEnable(GL_TEXTURE_2D);
    //glDisable(GL_DEPTH_TEST);
	glEnable(GL_DEPTH_TEST);

    // Preserve the current framebuffer content (e.g. back of the bounding box)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Bind textures
    glActiveTexture(GL_TEXTURE1);
    this->fbo.BindColourTexture(0);
    glActiveTexture(GL_TEXTURE2);
    this->fbo.BindColourTexture(1);
    glActiveTexture(GL_TEXTURE0);
    this->fbo.BindDepthTexture();
    
    //vislib::math::Vector<float, 3> ray(crIn->GetCameraParameters()->Front());
    //vislib::math::Vector<float, 3> up(crIn->GetCameraParameters()->Up());
    //vislib::math::Vector<float, 3> right(crIn->GetCameraParameters()->Right());
    vislib::math::Vector<float, 3> ray(0, 0,-1);
    vislib::math::Vector<float, 3> up(0, 1, 0);
    vislib::math::Vector<float, 3> right(1, 0, 0);
    
    up *= sinf(crIn->GetCameraParameters()->HalfApertureAngle());
    right *= sinf(crIn->GetCameraParameters()->HalfApertureAngle())
        * static_cast<float>(curVP[2]) / static_cast<float>(curVP[3]);

    // Draw
    glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
    glBegin(GL_QUADS);
    glNormal3fv((ray - right - up).PeekComponents());
    glTexCoord2f(0, 0);
    //glVertex2f(0.0f, 0.0f);
    glVertex2f(-1.0f,-1.0f);
    glNormal3fv((ray + right - up).PeekComponents());
    glTexCoord2f(1, 0); 
    //glVertex2f(1.0f, 0.0f);
    glVertex2f(1.0f,-1.0f);
    glNormal3fv((ray + right + up).PeekComponents());
    glTexCoord2f(1, 1); 
    glVertex2f(1.0f, 1.0f);
    glNormal3fv((ray - right + up).PeekComponents());
    glTexCoord2f(0, 1); 
    //glVertex2f(0.0f, 1.0f);
    glVertex2f(-1.0f, 1.0f);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    this->blinnPhongShader.Disable();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    return true;
}


/*
 * view::BlinnPhongRendererDeferred::createFBO
 */
bool view::BlinnPhongRendererDeferred::createFBO(UINT width, UINT height) {

    vislib::graphics::gl::FramebufferObject::ColourAttachParams cap[2];
    cap[0].internalFormat = GL_RGBA32F;
    cap[0].format =  GL_RGBA;
    cap[0].type = GL_FLOAT;
    cap[1].internalFormat = GL_RGB32F;
    cap[1].format =  GL_RGBA;
    cap[1].type = GL_FLOAT;

    vislib::graphics::gl::FramebufferObject::DepthAttachParams dap;
    dap.state = vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE;
    dap.format = GL_DEPTH_COMPONENT24;

    vislib::graphics::gl::FramebufferObject::StencilAttachParams sap;
    sap.state = vislib::graphics::gl::FramebufferObject::ATTACHMENT_DISABLED;
    sap.format = GL_STENCIL_INDEX;

    // Create the framebuffer object
    return (this->fbo.Create(width, height, 2, cap, dap, sap));
}
