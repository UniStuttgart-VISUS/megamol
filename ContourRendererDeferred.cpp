/*
 * ContourRendererDeferred.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include <param/EnumParam.h>
#include <param/BoolParam.h>
#include <param/IntParam.h>
#include <param/FloatParam.h>
#include <view/CallRender3D.h>
#include <view/CallRenderDeferred3D.h>
#include <CoreInstance.h>

#include <vislib/ShaderSource.h>
#include <vislib/Log.h>

#include "ContourRendererDeferred.h"

using namespace megamol::protein;

/*
 * ContourRendererDeferred::ContourRendererDeferred
 */
ContourRendererDeferred::ContourRendererDeferred(void)
    : megamol::core::view::AbstractRendererDeferred3D(),
    renderModeParam("renderMode", "The render mode to be used"),
    thresholdParam("threshold", "Threshold for the gradient"),
    scaleParam("scale", "Scale for the gradient"),
    conModeParam("conMode", "Choose the contour mode"),
    widthFBO(-1), heightFBO(-1) {

    // Render mode param
    megamol::core::param::EnumParam *rm =
        new megamol::core::param::EnumParam(0);
    rm->SetTypePair(0, "Contour + Color");
    rm->SetTypePair(1, "Contour");
    rm->SetTypePair(2, "Depth");
    rm->SetTypePair(3, "Normal");
    rm->SetTypePair(4, "Color");
    this->renderModeParam << rm;
    this->MakeSlotAvailable(&renderModeParam);

    // Contour mode param
    megamol::core::param::EnumParam *cm =
        new megamol::core::param::EnumParam(0);
    cm->SetTypePair(0, "DepthMap");
    cm->SetTypePair(1, "NormalMap");
    this->conModeParam << cm;
    this->MakeSlotAvailable(&conModeParam);

    // Threshold param
    this->thresholdParam << new megamol::core::param::FloatParam(0.6, 0.0, 1.0);
    this->MakeSlotAvailable(&this->thresholdParam);

    // Scale param
    this->scaleParam << new megamol::core::param::FloatParam(0.6, 0.0, 1.0);
    this->MakeSlotAvailable(&this->scaleParam);
}


/*
 * ContourRendererDeferred::create
 */
bool ContourRendererDeferred::create(void) {

    vislib::graphics::gl::ShaderSource vertSrc;
    vislib::graphics::gl::ShaderSource fragSrc;

    megamol::core::CoreInstance *ci = this->GetCoreInstance();
    if(!ci) {
        return false;
    }

    if(!glh_init_extensions("GL_EXT_framebuffer_object GL_ARB_draw_buffers"))
        return false;

    if(!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    if(!glh_init_extensions("GL_ARB_texture_non_power_of_two")) return false;

    // Try to load the deferred shader
    if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::contour::vertex", vertSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load contour vertex shader source", this->ClassName() );
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::contour::fragment", fragSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load contour fragment shader source", this->ClassName() );
        return false;
    }
    try {
        if(!this->contourShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
    catch(vislib::Exception e){
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    return true;
}


/*
 * ContourRendererDeferred::release
 */
void ContourRendererDeferred::release(void) {
    this->contourShader.Release();
    glDeleteTextures(1, &this->colorBuff);
    glDeleteTextures(1, &this->normalBuff);
    glDeleteFramebuffers(1, &this->fbo);
}


/*
 * ContourRendererDeferred::~ContourRendererDeferred
 */
ContourRendererDeferred::~ContourRendererDeferred(void) {
    this->Release();
}


/*
 * ContourRendererDeferred::GetCapabilities
 */
bool ContourRendererDeferred::GetCapabilities(megamol::core::Call& call) {

    megamol::core::view::CallRender3D *crIn =
        dynamic_cast< megamol::core::view::CallRender3D*>(&call);
    if(crIn == NULL) return false;

     megamol::core::view::CallRenderDeferred3D *crOut =
        this->rendererSlot.CallAs< megamol::core::view::CallRenderDeferred3D>();
    if(crOut == NULL) return false;

    // Call for getCapabilities
    if(!(*crOut)(2)) return false;

    // Set capabilities of for incoming render call
    crIn->SetCapabilities(crOut->GetCapabilities());

    return true;
}


/*
 * ContourRendererDeferred::GetExtents
 */
bool ContourRendererDeferred::GetExtents(megamol::core::Call& call) {

     megamol::core::view::CallRender3D *crIn =
         dynamic_cast< megamol::core::view::CallRender3D*>(&call);
    if(crIn == NULL) return false;

    megamol::core::view:: CallRenderDeferred3D *crOut =
        this->rendererSlot.CallAs< megamol::core::view::CallRenderDeferred3D>();
    if(crOut == NULL) return false;

    // Call for getExtends
    if(!(*crOut)(1)) return false;

    // Set extends of for incoming render call
    crIn->AccessBoundingBoxes() = crOut->GetBoundingBoxes();
    crIn->SetLastFrameTime(crOut->LastFrameTime());

    return true;
}


/*
 * ContourRendererDeferred::Render
 */
bool ContourRendererDeferred::Render(megamol::core::Call& call) {

    if(!updateParams()) return false;

    megamol::core::view::CallRender3D *crIn =
        dynamic_cast< megamol::core::view::CallRender3D*>(&call);
    if(crIn == NULL) return false;

    megamol::core::view::CallRenderDeferred3D *crOut =
        this->rendererSlot.CallAs< megamol::core::view::CallRenderDeferred3D>();
    if(crOut == NULL) return false;

    crOut->SetCameraParameters(crIn->GetCameraParameters());

    // Set call time
    crOut->SetTime(crIn->Time());

    float curVP[4];
    glGetFloatv(GL_VIEWPORT, curVP);

    vislib::math::Vector<float, 3> ray(0, 0,-1);
    vislib::math::Vector<float, 3> up(0, 1, 0);
    vislib::math::Vector<float, 3> right(1, 0, 0);

    up *= sinf(crIn->GetCameraParameters()->HalfApertureAngle());
    right *= sinf(crIn->GetCameraParameters()->HalfApertureAngle())
        * curVP[2] / curVP[3];

    // Recreate FBO if necessary
    if((curVP[2] != this->widthFBO) || (curVP[3] != this->heightFBO)) {
        if(!this->createFBO(static_cast<UINT>(curVP[2]), static_cast<UINT>(curVP[3]))) {
            return false;
        }
        this->widthFBO = curVP[2];
        this->heightFBO = curVP[3];
    }

    /// 1. Offscreen rendering ///

    // Enable rendering to FBO
    glBindFramebuffer(GL_FRAMEBUFFER, this->fbo);
    //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, this->colorBuff, 0);
    //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,  GL_TEXTURE_2D, this->normalBuff, 0);
    //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, this->depthBuff, 0);

    GLenum mrt[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, mrt);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Call for render
    (*crOut)(0);

    // Detach textures that are not needed anymore
    //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, 0, 0);
    //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,  GL_TEXTURE_2D, 0, 0);
    //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, 0, 0);

    // Disable rendering to framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);


    /// 2. Deferred shading ///
    this->contourShader.Enable();

    glUniform2f(this->contourShader.ParameterLocation("clip"),
        crIn->GetCameraParameters()->NearClip(),
        crIn->GetCameraParameters()->FarClip());
    glUniform2f(this->contourShader.ParameterLocation("winSize"),
        curVP[2], curVP[3]);
    glUniform1i(this->contourShader.ParameterLocation("depthBuff"), 0);
    glUniform1i(this->contourShader.ParameterLocation("colorBuff"), 1);
    glUniform1i(this->contourShader.ParameterLocation("normalBuff"), 2);
    glUniform1i(this->contourShader.ParameterLocation("discBuff"), 3);
    glUniform1i(this->contourShader.ParameterLocation("renderMode"),
        this->renderModeParam.Param<core::param::EnumParam>()->Value());
    glUniform1f(this->contourShader.ParameterLocation("threshold"),
        this->thresholdParam.Param<core::param::FloatParam>()->Value());
    glUniform1f(this->contourShader.ParameterLocation("scale"),
        this->scaleParam.Param<core::param::FloatParam>()->Value());
    glUniform1i(this->contourShader.ParameterLocation("conMode"),
        this->conModeParam.Param<core::param::EnumParam>()->Value());


    // Preserve the current framebuffer content (e.g. back of the bounding box)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Bind textures
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->colorBuff); // Color buffer
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, this->normalBuff); // Normal buffer
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->depthBuff); // Depth buffer

    // Draw
    glBegin(GL_QUADS);
        glNormal3fv((ray - right - up).PeekComponents());
        glTexCoord2f(0, 0); glVertex2f(-1.0f, -1.0f);
        glNormal3fv((ray + right - up).PeekComponents());
        glTexCoord2f(1, 0); glVertex2f(1.0f, -1.0f);
        glNormal3fv((ray + right + up).PeekComponents());
        glTexCoord2f(1, 1); glVertex2f(1.0f, 1.0f);
        glNormal3fv((ray - right + up).PeekComponents());
        glTexCoord2f(0, 1); glVertex2f(-1.0f, 1.0f);
    glEnd();

    this->contourShader.Disable();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    return true;
}


/*
 * ContourRendererDeferred::createFBO
 */
bool ContourRendererDeferred::createFBO(UINT width, UINT height) {

    // Delete textures + fbo if necessary
    if(glIsFramebuffer(this->fbo)) {
        glDeleteTextures(1, &this->colorBuff);
        glDeleteTextures(1, &this->normalBuff);
        glDeleteFramebuffers(1, &this->fbo);
    }

    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &this->colorBuff);
    glBindTexture(GL_TEXTURE_2D, this->colorBuff);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &this->normalBuff);
    glBindTexture(GL_TEXTURE_2D, this->normalBuff);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &this->depthBuff);
    glBindTexture(GL_TEXTURE_2D, this->depthBuff);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Generate framebuffer
    glGenFramebuffers(1, &this->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, this->colorBuff, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,  GL_TEXTURE_2D, this->normalBuff, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, this->depthBuff, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if(status != GL_FRAMEBUFFER_COMPLETE) {
      vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Could not create FBO");
      return false;
    }

    // Detach all textures
    /*glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, 0, 0);*/
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return true;
}

/*
 * ContourRendererDeferred::updateParams
 */
bool ContourRendererDeferred::updateParams() {
    return true;
}
