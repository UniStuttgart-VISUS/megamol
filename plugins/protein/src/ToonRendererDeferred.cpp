/*
 * ToonRendererDeferred.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/CallRenderDeferred3D.h"
#include "mmcore/CoreInstance.h"

#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/sys/Log.h"

#include "ToonRendererDeferred.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

using namespace megamol::protein;

/*
 * ToonRendererDeferred::ToonRendererDeferred
 */
ToonRendererDeferred::ToonRendererDeferred(void)
    : megamol::core::view::AbstractRendererDeferred3D(),
    threshFineLinesParam("linesFine", "Threshold for fine silhouette."),
    threshCoarseLinesParam("linesCoarse", "Threshold for coarse silhouette."),
    ssaoParam("toggleSSAO", "Toggle Screen Space Ambient Occlusion."),
    ssaoRadiusParam("ssaoRadius", "Radius for SSAO samples."),
    illuminationParam("illumination", "Change local lighting."),
    colorParam("toggleColor", "Toggle coloring."),
    widthFBO(-1), heightFBO(-1) {

    // Threshold for fine lines
    this->threshFineLinesParam << new megamol::core::param::FloatParam(9.5f, 0.0f, 9.95f);
    this->MakeSlotAvailable(&this->threshFineLinesParam);

    // Threshold for coarse lines
    this->threshCoarseLinesParam << new megamol::core::param::FloatParam(7.9f, 0.0f, 9.95f);
    this->MakeSlotAvailable(&this->threshCoarseLinesParam);

    // Toggle SSAO
    this->ssaoParam << new megamol::core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->ssaoParam);

    // Toggle local lighting
    megamol::core::param::EnumParam *illuParam =
        new megamol::core::param::EnumParam(0);
    illuParam->SetTypePair(0, "None");
    illuParam->SetTypePair(1, "Phong-Shading");
    illuParam->SetTypePair(2, "Toon-Shading");
    this->illuminationParam << illuParam;
    this->MakeSlotAvailable(&illuminationParam);

    // Toggle coloring
    this->colorParam << new megamol::core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->colorParam);

    // SSAO radius param
    this->ssaoRadiusParam << new megamol::core::param::FloatParam(1.0, 0.0);
    this->MakeSlotAvailable(&this->ssaoRadiusParam);
}


/*
 * ToonRendererDeferred::create
 */
bool ToonRendererDeferred::create(void) {

    vislib::graphics::gl::ShaderSource vertSrc;
    vislib::graphics::gl::ShaderSource fragSrc;

    // Init random number generator
    srand((unsigned)time(0));

    // Create 4x4-texture with random rotation vectors
    if(!this->createRandomRotSampler()) {
        return false;
    }

    // Create sampling kernel
    if(!this->createRandomKernel(16)) {
        return false;
    }

    megamol::core::CoreInstance *ci = this->GetCoreInstance();
    if(!ci) {
        return false;
    }

    if(!areExtsAvailable("GL_EXT_framebuffer_object GL_ARB_draw_buffers")) {
        return false;
    }

    if(!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    if(!isExtAvailable("GL_ARB_texture_non_power_of_two")) return false;

    // Try to load the gradient shader
    if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::sobel::vertex", vertSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load gradient vertex shader source", this->ClassName() );
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::sobel::fragment", fragSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load gradient fragment shader source", this->ClassName() );
        return false;
    }
    try {
        if(!this->sobelShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
    catch(vislib::Exception &e){
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Try to load the ssao shader
    if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::ssao::vertex", vertSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load gradient vertex shader source", this->ClassName() );
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::ssao::fragment", fragSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load gradient fragment shader source", this->ClassName() );
        return false;
    }
    try {
        if(!this->ssaoShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
    catch(vislib::Exception &e){
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Try to load the toon shader
    if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::toon::vertex", vertSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load toon vertex shader source", this->ClassName() );
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::toon::fragment", fragSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load toon fragment shader source", this->ClassName() );
        return false;
    }
    try {
        if(!this->toonShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
    catch(vislib::Exception &e){
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    return true;
}


/*
 * ToonRendererDeferred::release
 */
void ToonRendererDeferred::release(void) {
    this->toonShader.Release();
    this->sobelShader.Release();
    this->ssaoShader.Release();
    glDeleteTextures(1, &this->colorBuffer);
    glDeleteTextures(1, &this->normalBuffer);
    glDeleteTextures(1, &this->gradientBuffer);
    glDeleteTextures(1, &this->depthBuffer);
    glDeleteTextures(1, &this->ssaoBuffer);
    glDeleteTextures(1, &this->randomKernel);
    glDeleteTextures(1, &this->rotationSampler);
    glDeleteFramebuffers(1, &this->fbo);
}


/*
 * ToonRendererDeferred::~ToonRendererDeferred
 */
ToonRendererDeferred::~ToonRendererDeferred(void) {
    this->Release();
}


/*
 * ToonRendererDeferred::GetExtents
 */
bool ToonRendererDeferred::GetExtents(megamol::core::Call& call) {

     megamol::core::view::CallRender3D *crIn =
         dynamic_cast< megamol::core::view::CallRender3D*>(&call);
    if(crIn == NULL) return false;

    megamol::core::view:: CallRenderDeferred3D *crOut =
        this->rendererSlot.CallAs< megamol::core::view::CallRenderDeferred3D>();
    if(crOut == NULL) return false;

    // Call for getExtends
    if(!(*crOut)(core::view::AbstractCallRender::FnGetExtents)) return false;

    // Set extends of for incoming render call
    crIn->AccessBoundingBoxes() = crOut->GetBoundingBoxes();
    crIn->SetLastFrameTime(crOut->LastFrameTime());
    crIn->SetTimeFramesCount(crOut->TimeFramesCount());

    return true;
}


/*
 * ToonRendererDeferred::Render
 */
bool ToonRendererDeferred::Render(megamol::core::Call& call) {

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
        this->widthFBO = (int)curVP[2];
        this->heightFBO = (int)curVP[3];
    }

    /// 1. Offscreen rendering ///

    // Enable rendering to FBO
    glBindFramebufferEXT(GL_FRAMEBUFFER, this->fbo);

    // Enable rendering to color attachents 0 and 1
    GLenum mrt[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, mrt);

    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, this->colorBuffer, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,  GL_TEXTURE_2D, this->normalBuffer, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, this->depthBuffer, 0);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Call for render
    (*crOut)(core::view::AbstractCallRender::FnRender);

    // Detach texture that are not needed anymore
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, 0, 0);

    // Prepare rendering screen quad
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);

     /// 2. Calculate gradient using the sobel operator ///

    GLenum mrt2[] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, mrt2);
    glDepthMask(GL_FALSE); // Disable writing to depth buffer

    // Attach gradient texture
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, this->gradientBuffer, 0);

    this->sobelShader.Enable();
    glUniform1i(this->sobelShader.ParameterLocation("depthBuffer"), 0);
    // Bind depth texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->depthBuffer); // Depth buffer
    // Draw
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1, 0); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(1, 1); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0, 1); glVertex2f(-1.0f, 1.0f);
    glEnd();
    this->sobelShader.Disable();
    glBindTexture(GL_TEXTURE_2D, 0);

    /// 3. Calculate ssao value if needed ///
    if(this->ssaoParam.Param<core::param::BoolParam>()->Value()) {
        // Attach ssao buffer
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, this->ssaoBuffer, 0);

        this->ssaoShader.Enable();
        glUniform4f(this->ssaoShader.ParameterLocation("clip"),
                crIn->GetCameraParameters()->NearClip(), // Near
                crIn->GetCameraParameters()->FarClip(),  // Far
                tan(crIn->GetCameraParameters()->HalfApertureAngle()) * crIn->GetCameraParameters()->NearClip(), // Top
                tan(crIn->GetCameraParameters()->HalfApertureAngle()) * crIn->GetCameraParameters()->NearClip() * curVP[2] / curVP[3]); // Right
        glUniform2f(this->ssaoShader.ParameterLocation("winSize"), curVP[2], curVP[3]);
        glUniform1i(this->ssaoShader.ParameterLocation("depthBuff"), 0);
        glUniform1i(this->ssaoShader.ParameterLocation("rotSampler"), 1);
        glUniform1i(this->ssaoShader.ParameterLocation("normalBuff"), 2);
        glUniform1f(this->ssaoShader.ParameterLocation("ssaoRadius"),
                this->ssaoRadiusParam.Param<core::param::FloatParam>()->Value());
        // Bind depth texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, this->rotationSampler);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, this->normalBuffer);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->depthBuffer);
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
        this->ssaoShader.Disable();
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Disable rendering to framebuffer
    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);

    /// 4. Deferred shading ///

    glDepthMask(GL_TRUE); // Enable writing to depth buffer

    // Preserve the current framebuffer content (e.g. back of the bounding box)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    this->toonShader.Enable();
    glUniform2f(this->toonShader.ParameterLocation("clip"),
        crIn->GetCameraParameters()->NearClip(),
        crIn->GetCameraParameters()->FarClip());
    glUniform2f(this->toonShader.ParameterLocation("winSize"),
        curVP[2], curVP[3]);
    glUniform1i(this->toonShader.ParameterLocation("depthBuffer"), 0);
    glUniform1i(this->toonShader.ParameterLocation("colorBuffer"), 1);
    glUniform1i(this->toonShader.ParameterLocation("normalBuffer"), 2);
    glUniform1i(this->toonShader.ParameterLocation("gradientBuffer"), 3);
    glUniform1i(this->toonShader.ParameterLocation("ssaoBuffer"), 4);

    glUniform1f(this->toonShader.ParameterLocation("threshFine"),
        (10.0f - this->threshFineLinesParam.Param<core::param::FloatParam>()->Value())*0.1f);
    glUniform1f(this->toonShader.ParameterLocation("threshCoarse"),
        (10.0f - this->threshCoarseLinesParam.Param<core::param::FloatParam>()->Value())*0.1f);
    glUniform1i(this->toonShader.ParameterLocation("ssao"),
        this->ssaoParam.Param<core::param::BoolParam>()->Value());
    glUniform1i(this->toonShader.ParameterLocation("lighting"),
        this->illuminationParam.Param<core::param::EnumParam>()->Value());
    glUniform1i(this->toonShader.ParameterLocation("withColor"),
        this->colorParam.Param<core::param::BoolParam>()->Value());

    // Bind textures
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->colorBuffer); // Color buffer

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, this->normalBuffer); // Normal buffer

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, this->gradientBuffer); // Gradient buffer

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, this->ssaoBuffer); // SSAO buffer
    glGenerateMipmapEXT(GL_TEXTURE_2D); // Generate mip map levels for ssao texture

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->depthBuffer); // Depth buffer

    // Draw quad
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

    this->toonShader.Disable();
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
 * ToonRendererDeferred::createFBO
 */
bool ToonRendererDeferred::createFBO(UINT width, UINT height) {

    // Delete textures + fbo if necessary
    if(glIsFramebufferEXT(this->fbo)) {
        glDeleteTextures(1, &this->colorBuffer);
        glDeleteTextures(1, &this->normalBuffer);
        glDeleteTextures(1, &this->gradientBuffer);
        glDeleteTextures(1, &this->depthBuffer);
        glDeleteTextures(1, &this->ssaoBuffer);
        glDeleteFramebuffersEXT(1, &this->fbo);
    }

    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &this->colorBuffer);
    glBindTexture(GL_TEXTURE_2D, this->colorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Normal buffer
    glGenTextures(1, &this->normalBuffer);
    glBindTexture(GL_TEXTURE_2D, this->normalBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Gradient buffer
    // TODO Texture format
    glGenTextures(1, &this->gradientBuffer);
    glBindTexture(GL_TEXTURE_2D, this->gradientBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // SSAO buffer
    // TODO Texture format
    glGenTextures(1, &this->ssaoBuffer);
    glBindTexture(GL_TEXTURE_2D, this->ssaoBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Depth buffer
    glGenTextures(1, &this->depthBuffer);
    glBindTexture(GL_TEXTURE_2D, this->depthBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);


    // Generate framebuffer
    glGenFramebuffersEXT(1, &this->fbo);

    GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER);
    if(status != GL_FRAMEBUFFER_COMPLETE) {
      vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Could not create FBO");
      return false;
    }

    // Detach all textures
    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);

    return true;
}

/*
 * ToonRendererDeferred::updateParams
 */
bool ToonRendererDeferred::updateParams() {
    return true;
}

/*
 * ToonRendererDeferred::createRandomRotSampler
 */
bool ToonRendererDeferred::createRandomRotSampler() {

    float data [4][4][3];

    for(unsigned int s = 0; s < 4; s++) {
        for(unsigned int t = 0; t < 4; t++) {
            data[s][t][0] = getRandomFloat(-1.0, 1.0, 3);
            data[s][t][1] = getRandomFloat(-1.0, 1.0, 3);
            data[s][t][2] = 0.0;

            // Compute magnitude
            float mag = sqrt(data[s][t][0]*data[s][t][0] +
                             data[s][t][1]*data[s][t][1] +
                             data[s][t][2]*data[s][t][2]);

            // Normalize
            data[s][t][0] /= mag;
            data[s][t][1] /= mag;
            data[s][t][2] /= mag;

            // Map to range 0 ... 1
            data[s][t][0] += 1.0; data[s][t][0] /= 2.0;
            data[s][t][1] += 1.0; data[s][t][1] /= 2.0;
            data[s][t][2] += 1.0; data[s][t][2] /= 2.0;
        }
    }

    glGenTextures(1, &this->rotationSampler);
    glBindTexture(GL_TEXTURE_2D, this->rotationSampler);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, 4, 0, GL_RGB, GL_FLOAT, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindTexture(GL_TEXTURE_2D, 0);

    if(glGetError() != GL_NO_ERROR)
        return false;

    return true;
}


/*
 * ToonRendererDeferred::getRandomFloat
 */
float ToonRendererDeferred::getRandomFloat(float min, float max, unsigned int prec) {
    // Note: RAND_MAX is only guaranteed to be at least 32767, so prec should not be more than 4
    float base = 10.0;
    float precision = pow(base, (int)prec);
    int range = (int)(max*precision - min*precision + 1);
    return static_cast<float>(rand()%range + min*precision) / precision;
}

/*
* ToonRendererDeferred::::createRandomKernel
*/
// TODO enforce lower boundary to prevent numerical problems leading to artifacts
bool ToonRendererDeferred::createRandomKernel(UINT size) {

   float *kernel;
   kernel = new float[size*3];

   for(unsigned int s = 0; s < size; s++) {

       float scale = float(s) / float(size);
       scale *= scale;
       scale = 0.1f*(1.0f - scale) + scale; // Interpolate between 0.1 ... 1.0

       kernel[s*3 + 0] = getRandomFloat(-1.0, 1.0, 3);
       kernel[s*3 + 1] = getRandomFloat(-1.0, 1.0, 3);
       kernel[s*3 + 2] = getRandomFloat( 0.0, 1.0, 3);

       // Compute magnitude
       float mag = sqrt(kernel[s*3 + 0]*kernel[s*3 + 0] +
                        kernel[s*3 + 1]*kernel[s*3 + 1] +
                        kernel[s*3 + 2]*kernel[s*3 + 2]);

       // Normalize
       kernel[s*3 + 0] /= mag;
       kernel[s*3 + 1] /= mag;
       kernel[s*3 + 2] /= mag;

       // Scale values
       kernel[s*3 + 0] *= scale;
       kernel[s*3 + 1] *= scale;
       kernel[s*3 + 2] *= scale;

       // Map values to range 0 ... 1
       kernel[s*3 + 0] += 1.0; kernel[s*3 + 0] /= 2.0;
       kernel[s*3 + 1] += 1.0; kernel[s*3 + 1] /= 2.0;
       kernel[s*3 + 2] += 1.0; kernel[s*3 + 2] /= 2.0;
   }

   glGenTextures(1, &this->randomKernel);
   glBindTexture(GL_TEXTURE_2D, this->randomKernel);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size, 1, 0, GL_RGB, GL_FLOAT, kernel);
   glBindTexture(GL_TEXTURE_2D, 0);

   delete[] kernel;

   if(glGetError() != GL_NO_ERROR)
       return false;

   return true;
}
