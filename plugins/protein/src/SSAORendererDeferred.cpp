/*
 * SSAORendererDeferred.cpp
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

#include "SSAORendererDeferred.h"

using namespace megamol::protein;

/*
 * SSAORendererDeferred::SSAORendererDeferred
 */
SSAORendererDeferred::SSAORendererDeferred(void)
    : megamol::core::view::AbstractRendererDeferred3D(),
    renderModeParam("renderMode", "The render mode to be used"),
    aoRadiusParam("ssaoRadius", "The radius of the approximated sphere for ao"),
    depthThresholdParam("depthThreshold", "The depth threshold for the discontinuity test"),
    aoSamplesParam("ssaoSamples", "The number of samples for the ssao factor."),
    nFilterSamplesParam("nFilterSamples", "The number of samples for the blur filter."),
    aoScaleParam("ssaoScale", "The scale factor for the ssao factor."),
    depthMipMapLvlParam("depthMipMapLvl", "The mip map level of the depth texture."),
    filterParam("ssaofilter", "Toggle blurring of the ssao value."),
    widthFBO(-1), heightFBO(-1) {

    // Render mode param
    megamol::core::param::EnumParam *rm =
        new megamol::core::param::EnumParam(0);
    rm->SetTypePair(0, "SSAO+Light");
    rm->SetTypePair(1, "SSAO+Color");
    rm->SetTypePair(2, "SSAO");
    rm->SetTypePair(3, "Light");
    rm->SetTypePair(4, "Depth");
    rm->SetTypePair(5, "Normal");
    rm->SetTypePair(6, "Color");
    rm->SetTypePair(7, "Edges");
    this->renderModeParam << rm;
    this->MakeSlotAvailable(&renderModeParam);

    this->aoRadiusParam << new megamol::core::param::FloatParam(5.0f, 0.0f);
    this->MakeSlotAvailable(&this->aoRadiusParam);

    this->depthThresholdParam << new megamol::core::param::FloatParam(0.14f, 0.0f);
    this->MakeSlotAvailable(&this->depthThresholdParam);

    this->aoSamplesParam << new megamol::core::param::IntParam(16, 1);
    this->MakeSlotAvailable(&this->aoSamplesParam);

    this->aoScaleParam << new megamol::core::param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->aoScaleParam);

    this->nFilterSamplesParam << new megamol::core::param::IntParam(4);
    this->MakeSlotAvailable(&this->nFilterSamplesParam);

    this->depthMipMapLvlParam << new megamol::core::param::IntParam(0, 0);
    this->MakeSlotAvailable(&this->depthMipMapLvlParam);

    this->filterParam << new megamol::core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->filterParam);
}


/*
 * SSAORendererDeferred::create
 */
bool SSAORendererDeferred::create(void) {

    vislib::graphics::gl::ShaderSource vertSrc;
    vislib::graphics::gl::ShaderSource fragSrc;

    // Init random number generator
    srand((unsigned)time(0));

    // Create 4x4-texture with random rotation vectors
    if(!this->createRandomRotSampler()) {
        return false;
    }

    // Create sampling kernel
    if(!this->createRandomKernel(this->aoSamplesParam.Param<core::param::IntParam>()->Value())) {
        return false;
    }

    megamol::core::CoreInstance *ci = this->GetCoreInstance();
    if(!ci) {
        return false;
    }

    if(!areExtsAvailable("GL_EXT_framebuffer_object GL_ARB_draw_buffers"))
        return false;

    if(!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    if(!isExtAvailable("GL_ARB_texture_non_power_of_two")) return false;

    // Try to load the ssao shader
    if(!ci->ShaderSourceFactory().MakeShaderSource("SSAOdeferred::ssao::vertex", vertSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load ssao vertex shader source", this->ClassName() );
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("SSAOdeferred::ssao::fragment", fragSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load ssao fragment shader source", this->ClassName() );
        return false;
    }
    try {
        if(!this->ssaoShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
    catch(vislib::Exception e){
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Try to load the deferred shader
    if(!ci->ShaderSourceFactory().MakeShaderSource("SSAOdeferred::deferred::vertex", vertSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load deferred vertex shader source", this->ClassName() );
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("SSAOdeferred::deferred::fragment", fragSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load deferred fragment shader source", this->ClassName() );
        return false;
    }
    try {
        if(!this->deferredShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
    catch(vislib::Exception e){
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Try to load the filter shaders
    if(!ci->ShaderSourceFactory().MakeShaderSource("SSAOdeferred::filter::vertex", vertSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load filter vertex shader source", this->ClassName() );
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("SSAOdeferred::filter::fragmentHor", fragSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load filter fragment shader source", this->ClassName() );
        return false;
    }
    try {
        if(!this->filterShaderX.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
    catch(vislib::Exception e){
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("SSAOdeferred::filter::fragmentVert", fragSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load filter fragment shader source", this->ClassName() );
        return false;
    }
    try {
        if(!this->filterShaderY.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
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
 * SSAORendererDeferred::release
 */
void SSAORendererDeferred::release(void) {

    this->ssaoShader.Release();
    this->deferredShader.Release();
    this->filterShaderX.Release();
    this->filterShaderY.Release();

    glDeleteTextures(1, &this->randKernel);
    glDeleteTextures(1, &this->rotSampler);

    glDeleteTextures(1, &this->colorBuff);
    glDeleteTextures(1, &this->normalBuff);
    glDeleteTextures(1, &this->ssaoBuff);
    glDeleteTextures(1, &this->filterBuff);
    glDeleteTextures(1, &this->discBuff);
    glDeleteFramebuffersEXT(1, &this->deferredFBO);
}


/*
 * SSAORendererDeferred::~SSAORendererDeferred
 */
SSAORendererDeferred::~SSAORendererDeferred(void) {
    this->Release();
}


/*
 * SSAORendererDeferred::GetExtents
 */
bool SSAORendererDeferred::GetExtents(megamol::core::Call& call) {

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
	crIn->SetTimeFramesCount( crOut->TimeFramesCount());

    return true;
}


/*
 * SSAORendererDeferred::Render
 */
// TODO Write 'real' depth to depth buffer
bool SSAORendererDeferred::Render(megamol::core::Call& call) {

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

    /// Render scene ///

    // Enable rendering to FBO
    glBindFramebufferEXT(GL_FRAMEBUFFER, this->deferredFBO);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, this->colorBuff, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,  GL_TEXTURE_2D, this->normalBuff, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, this->depthBuff, 0);

    GLenum mrt[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	glDrawBuffersARB(2, mrt);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //GLfloat projMat[16];
    //glGetFloatv(GL_PROJECTION_MATRIX, projMat);

    // Call for render
    (*crOut)(core::view::AbstractCallRender::FnRender);

    // Detach textures that are not needed anymore
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, 0, 0);

    // Attach SSAO and discontinuity buffer
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, this->ssaoBuff, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,  GL_TEXTURE_2D, this->discBuff, 0);
    //GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    //if(status != GL_FRAMEBUFFER_COMPLETE) {
    //  return false;
    //}

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);

    /// Calculate ssao factor ///
    this->ssaoShader.Enable();
    glUniform4fARB(this->ssaoShader.ParameterLocation("clip"),
        crIn->GetCameraParameters()->NearClip(), // Near
        crIn->GetCameraParameters()->FarClip(),  // Far
        tan(crIn->GetCameraParameters()->HalfApertureAngle()) * crIn->GetCameraParameters()->NearClip(), // Top
        tan(crIn->GetCameraParameters()->HalfApertureAngle()) * crIn->GetCameraParameters()->NearClip() * curVP[2] / curVP[3]); // Right
    glUniform2fARB(this->ssaoShader.ParameterLocation("winSize"), curVP[2], curVP[3]);
    glUniform1iARB(this->ssaoShader.ParameterLocation("depthBuffer"), 0);
    glUniform1iARB(this->ssaoShader.ParameterLocation("rotSampler"), 1);
    glUniform1iARB(this->ssaoShader.ParameterLocation("normalBuffer"), 2);
    glUniform1iARB(this->ssaoShader.ParameterLocation("randKernel"), 3);
    glUniform1fARB(this->ssaoShader.ParameterLocation("aoRadius"),
        this->aoRadiusParam.Param<core::param::FloatParam>()->Value());
    glUniform1fARB(this->ssaoShader.ParameterLocation("depthThreshold"),
        this->depthThresholdParam.Param<core::param::FloatParam>()->Value());
    glUniform1iARB(this->ssaoShader.ParameterLocation("aoSamples"),
        this->aoSamplesParam.Param<core::param::IntParam>()->Value());
    glUniform1iARB(this->ssaoShader.ParameterLocation("depthLodLvl"),
        this->depthMipMapLvlParam.Param<core::param::IntParam>()->Value());
    //glUniformMatrix4fv(this->ssaoShader.ParameterLocation("projMat"), 1, false,
    //    projMat);

	glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->rotSampler);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, this->normalBuff);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, this->randKernel);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->depthBuff);
    glGenerateMipmapEXT(GL_TEXTURE_2D); // Regenerate mip map levels of the depth texture

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
    glBindTexture(GL_TEXTURE_2D,0);

    // Detach textures that are not needed anymore
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,  GL_TEXTURE_2D, 0, 0);


    if(this->filterParam.Param<core::param::BoolParam>()->Value()) {

        //// Horizontal filter pass ///

        // Attach temporary filter buffer
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D,
                this->filterBuff, 0);
        //GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        //if(status != GL_FRAMEBUFFER_COMPLETE) {
        //  return false;
        //}
        this->filterShaderX.Enable();
		glUniform2fARB(this->filterShaderX.ParameterLocation("winSize"), curVP[2], curVP[3]);
        glUniform1iARB(this->filterShaderX.ParameterLocation("ssaoBuff"), 0);
        glUniform1iARB(this->filterShaderX.ParameterLocation("discBuff"), 1);
        glUniform1iARB(this->filterShaderX.ParameterLocation("nFilterSamples"),
                this->nFilterSamplesParam.Param<core::param::IntParam>()->Value());
		glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, this->discBuff);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->ssaoBuff);
        // Draw
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1, 0); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(1, 1); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0, 1); glVertex2f(-1.0f, 1.0f);
        glEnd();
        this->filterShaderX.Disable();
        glBindTexture(GL_TEXTURE_2D, 0);

        glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, 0, 0); // TODO necessary?

        /// Vertical filter pass ///

        glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D,
                this->ssaoBuff, 0);
        this->filterShaderY.Enable();
        glUniform2fARB(this->filterShaderY.ParameterLocation("winSize"),
                curVP[2], curVP[3]);
        glUniform1iARB(this->filterShaderY.ParameterLocation("ssaoBuff"), 0);
        glUniform1iARB(this->filterShaderY.ParameterLocation("discBuff"), 1);
        glUniform1iARB(this->filterShaderY.ParameterLocation("nFilterSamples"),
                this->nFilterSamplesParam.Param<core::param::IntParam>()->Value());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, this->discBuff);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->filterBuff); // Apply vertical filter to horizontally filtered values
        // Draw
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1, 0); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(1, 1); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0, 1); glVertex2f(-1.0f, 1.0f);
        glEnd();
        this->filterShaderY.Disable();
        glBindTexture(GL_TEXTURE_2D,0);

        glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, 0, 0); // TODO necessary?
    }

    // Disable rendering to framebuffer
    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);

    /// Deferred shading ///
    this->deferredShader.Enable();

    glUniform2fARB(this->deferredShader.ParameterLocation("clip"),
        crIn->GetCameraParameters()->NearClip(),
        crIn->GetCameraParameters()->FarClip());
    glUniform2fARB(this->deferredShader.ParameterLocation("winSize"),
        curVP[2], curVP[3]);
    glUniform1iARB(this->deferredShader.ParameterLocation("depthBuff"), 0);
    glUniform1iARB(this->deferredShader.ParameterLocation("colorBuff"), 1);
    glUniform1iARB(this->deferredShader.ParameterLocation("normalBuff"), 2);
    glUniform1iARB(this->deferredShader.ParameterLocation("ssaoBuff"), 3);
    glUniform1iARB(this->deferredShader.ParameterLocation("discBuff"), 4);
    this->deferredShader.SetParameter("scale",
        this->aoScaleParam.Param<megamol::core::param::FloatParam>()->Value());
	glUniform1iARB(this->deferredShader.ParameterLocation("renderMode"),
        this->renderModeParam.Param<core::param::EnumParam>()->Value());
    glUniform1iARB(this->deferredShader.ParameterLocation("depthLodLvl"),
        this->depthMipMapLvlParam.Param<core::param::IntParam>()->Value());

    // Preserve the current framebuffer content (e.g. back of the bounding box)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Bind textures
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->colorBuff); // Color buffer

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, this->normalBuff); // Normal buffer

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, this->ssaoBuff); // Ssao value

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, this->discBuff); // Discontinuity buffer value

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

    this->deferredShader.Disable();
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
 * SSAORendererDeferred::createFBO
 */
bool SSAORendererDeferred::createFBO(UINT width, UINT height) {

    // Delete textures + fbo if necessary
    if(glIsFramebufferEXT(this->deferredFBO)) {
        glDeleteTextures(1, &this->colorBuff);
        glDeleteTextures(1, &this->normalBuff);
        glDeleteTextures(1, &this->ssaoBuff);
        glDeleteTextures(1, &this->filterBuff);
        glDeleteTextures(1, &this->discBuff);
        glDeleteFramebuffersEXT(1, &this->deferredFBO);
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

    glGenTextures(1, &this->ssaoBuff);
    glBindTexture(GL_TEXTURE_2D, this->ssaoBuff);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0); // TODO texture format
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &this->filterBuff);
    glBindTexture(GL_TEXTURE_2D, this->filterBuff);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0); // TODO texture format
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &this->discBuff);
    glBindTexture(GL_TEXTURE_2D, this->discBuff);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0); // TODO texture format
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffersEXT(1, &this->deferredFBO);
    glBindFramebufferEXT(GL_FRAMEBUFFER, deferredFBO);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, this->colorBuff, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,  GL_TEXTURE_2D, this->normalBuff, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2,  GL_TEXTURE_2D, this->ssaoBuff, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3,  GL_TEXTURE_2D, this->filterBuff, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4,  GL_TEXTURE_2D, this->discBuff, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, this->depthBuff, 0);

    GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER);
    if(status != GL_FRAMEBUFFER_COMPLETE) {
      vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Could not create FBO");
      return false;
    }

    // Detach all textures
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4,  GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, 0, 0);
    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);

    return true;
}


/*
 * SSAORendererDeferred::createRandomRotSampler
 */
bool SSAORendererDeferred::createRandomRotSampler() {

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

    glGenTextures(1, &this->rotSampler);
    glBindTexture(GL_TEXTURE_2D, this->rotSampler);
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
 * SSAORendererDeferred::createRandomKernel
 */
// TODO enforce lower boundary to prevent numerical problems leading to artifacts
bool SSAORendererDeferred::createRandomKernel(UINT size) {

    float *kernel;
    kernel = new float[size*3];

    for(unsigned int s = 0; s < size; s++) {

        float scale = float(s) / float(size);
        scale *= scale;
        scale = 0.1f*(1.0f - scale) + scale; // Interpolate between 0.1 ... 1.0

        kernel[s*3 + 0] = getRandomFloat(-1.0f, 1.0f, 3);
        kernel[s*3 + 1] = getRandomFloat(-1.0f, 1.0f, 3);
        kernel[s*3 + 2] = getRandomFloat( 0.0f, 1.0f, 3);

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

    glGenTextures(1, &this->randKernel);
    glBindTexture(GL_TEXTURE_2D, this->randKernel);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size, 1, 0, GL_RGB, GL_FLOAT, kernel);
    glBindTexture(GL_TEXTURE_2D, 0);

    delete[] kernel;

    if(glGetError() != GL_NO_ERROR)
        return false;

    return true;
}


/*
 * SSAORendererDeferred::getRandomFloat
 */
float SSAORendererDeferred::getRandomFloat(float min, float max, unsigned int prec) {
    // Note: RAND_MAX is only guaranteed to be at least 32767, so prec should not be more than 4
    float base = 10.0;
    float precision = pow(base, (int)prec);
    int range = (int)(max*precision - min*precision + 1);
    return static_cast<float>(rand()%range + min*precision) / precision;
}


/*
 * SSAORendererDeferred::updateParams
 */
bool SSAORendererDeferred::updateParams() {

    // Create new kernel if necessary
    if(this->aoSamplesParam.IsDirty()) {
        this->aoSamplesParam.ResetDirty();
        glDeleteTextures(1, &this->randKernel);
        if(!this->createRandomKernel(
                this->aoSamplesParam.Param<core::param::IntParam>()->Value())) {
            return false;
        }
    }
    return true;
}
