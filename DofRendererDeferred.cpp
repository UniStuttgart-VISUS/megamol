/*
 * DofRendererDeferred.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include <GL/glu.h>
#include <GL/freeglut.h>

#include <param/EnumParam.h>
#include <param/BoolParam.h>
#include <param/IntParam.h>
#include <param/FloatParam.h>
#include <view/CallRender3D.h>
#include <view/CallRenderDeferred3D.h>
#include <CoreInstance.h>

#include <vislib/ShaderSource.h>
#include <vislib/Log.h>
#include <vislib/Vector.h>

#include "DofRendererDeferred.h"

using namespace megamol;
using namespace vislib;


/*
 * protein::DofRendererDeferred::protein::DofRendererDeferred
 */
protein::DofRendererDeferred::DofRendererDeferred(void)
	: core::view::AbstractRendererDeferred3D(),
	  rModeParam("rMode", "The render mode"),
	  dofModeParam("dofMode", "The depth of field mode"),
	  toggleGaussianParam("gaussian", "Toggle gaussian filtering"),
	  focalDistParam("focalDist", "Change the focal distance"),
	  apertureParam("aperture", "Change the aperture"),
	  width(-1),
	  height(-1),
	  focalLength(0.035f),
	  filmWidth(0.035f),
	  maxCoC(2.0f),
	  cocRadiusScale(0.4f),
	  originalCoC(false) {

	// Param for depth of field mode
	this->rMode = DOF;
	core::param::EnumParam *rm = new core::param::EnumParam(this->rMode);
	rm->SetTypePair(DOF, "depth of field");
	rm->SetTypePair(FOCAL_DIST, "focal distance");
	this->rModeParam << rm;
	this->MakeSlotAvailable(&this->rModeParam);

	// Param for depth of field mode
	this->dofMode = DOF_SHADERX;
	core::param::EnumParam *dofModeParam = new core::param::EnumParam(this->dofMode);
	dofModeParam->SetTypePair(DOF_SHADERX, "ShaderX");
	dofModeParam->SetTypePair(DOF_MIPMAP,  "MipMap");
	dofModeParam->SetTypePair(DOF_LEE,     "Lee");
	this->dofModeParam << dofModeParam;
	this->MakeSlotAvailable(&this->dofModeParam);

	// Toggle gaussian filter
	this->useGaussian = false;
	this->toggleGaussianParam << new core::param::BoolParam(this->useGaussian);
	this->MakeSlotAvailable(&this->toggleGaussianParam);


	// Param for the focal distance
	this->focalDist = 1.0f;
	this->focalDistParam << new core::param::FloatParam(this->focalDist, 0.0f, 1000.0f);
	this->MakeSlotAvailable(&this->focalDistParam);

	// Param for aperture
	this->aperture = 5.6f;
	this->apertureParam << new core::param::FloatParam(this->aperture, 0.5f, 128.0f);
	this->MakeSlotAvailable(&this->apertureParam);
}


/*
 * protein::DofRendererDeferred::create
 */
bool protein::DofRendererDeferred::create(void) {

	graphics::gl::ShaderSource vertSrc;
	graphics::gl::ShaderSource fragSrc;

	core::CoreInstance *ci = this->GetCoreInstance();
	if(!ci) {
		return false;
	}

	if(!glh_init_extensions("GL_EXT_framebuffer_object GL_ARB_draw_buffers"))
		return false;

	if(!graphics::gl::GLSLShader::InitialiseExtensions()) {
		return false;
	}

	if(!glh_init_extensions("GL_ARB_texture_non_power_of_two")) return false;

	// Try to load depth of field shader (mipmap, reduce)
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::dof::reduceVertex", vertSrc)) {
		sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: depth of field shader (mipmap, reduce)", this->ClassName());
		return false;
	}
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::dof::mipmap::reduceFragment", fragSrc)) {
		sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: depth of field shader (mipmap, reduce)", this->ClassName());
		return false;
	}
	try {
		if(!this->reduceMipMap.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(Exception e){
		sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}

	// Try to load depth of field shader (mipmap, blur)
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::dof::blurVertex", vertSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: depth of field shader (mipmap, blur)", this->ClassName());
		return false;
	}
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::dof::mipmap::blurFragment", fragSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: depth of field shader (mipmap, blur)", this->ClassName());
		return false;
	}
	try {
		if(!this->blurMipMap.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(Exception e){
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}

	// Try to load depth of field shader (shaderx, reduce)
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::dof::reduceVertex", vertSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: depth of field shader shaderx, reduce)", this->ClassName());
		return false;
	}
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::dof::shaderx::reduceFragment", fragSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: depth of field shader (shaderx, reduce)", this->ClassName());
		return false;
	}
	try {
		if(!this->reduceShaderX.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(Exception e){
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}

	// Try to load depth of field shader (shaderx, blur)
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::dof::blurVertex", vertSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: depth of field shader shaderx, blur)", this->ClassName());
		return false;
	}
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::dof::shaderx::blurFragment", fragSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: depth of field shader (shaderx, blur)", this->ClassName());
		return false;
	}
	try {
		if(!this->blurShaderX.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(Exception e){
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}

	// Try to load shader for gaussian filter (horizontal)
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::gaussian::vertex", vertSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: gaussian filter (horizontal)", this->ClassName());
		return false;
	}
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::gaussian::fragmentHoriz", fragSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: gaussian filter (horizontal)", this->ClassName());
		return false;
	}
	try {
		if(!this->gaussianHoriz.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(Exception e){
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}

	// Try to load shader for gaussian filter (vertical)
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::gaussian::vertex", vertSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: gaussian filter (vertical)", this->ClassName());
		return false;
	}
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::gaussian::fragmentVert", fragSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: gaussian filter (vertical)", this->ClassName());
		return false;
	}
	try {
		if(!this->gaussianVert.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(Exception e){
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}

	// Try to load shader for gaussian filter (lee)
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::gaussian::vertex", vertSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: gaussian filter (lee)", this->ClassName());
		return false;
	}
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::gaussian::fragmentLee", fragSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: gaussian filter (lee)", this->ClassName());
		return false;
	}
	try {
		if(!this->gaussianLee.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(Exception e){
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}

	// Try  to load shader for blinn phong illumination
	if(!ci->ShaderSourceFactory().MakeShaderSource("deferred::vertex", vertSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: gaussian filter (lee)", this->ClassName());
		return false;
	}
	if(!ci->ShaderSourceFactory().MakeShaderSource("deferred::blinnPhongFrag", fragSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: gaussian filter (lee)", this->ClassName());
		return false;
	}
	try {
		if(!this->blinnPhongShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(Exception e){
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}

	// Try  to load shader for non linear depth
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::nonLinDepth::vertex", vertSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: non linear depth", this->ClassName());
		return false;
	}
	if(!ci->ShaderSourceFactory().MakeShaderSource("proteinDeferred::nonLinDepth::fragment", fragSrc)) {
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: non linear depth", this->ClassName());
		return false;
	}
	try {
		if(!this->nonLinDepthShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(Exception e){
		sys::Log::DefaultLog.WriteMsg(sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}

	/*this->glutMainWinID = glutGetWindow();
	this->glutParamWinID = glutCreateWindow("DOF Parameters");
	glutSetWindow(this->glutParamWinID);
	glutSetOption(GLUT_RENDERING_CONTEXT, GLUT_CREATE_NEW_CONTEXT);
	glutDisplayFunc(glutParamWinDisplayFunc);
	glutReshapeFunc(glutParamWinReshapeFunc);
	//glutKeyboardFunc(glutKeyboardCallback);
	//glutSpecialFunc(glutSpecialCallback);
	//glutMouseFunc(glutMouseCallback);
	//glutMotionFunc(glutMotionCallback);
	//glutPassiveMotionFunc(glutMotionCallback);
	//glutCloseFunc(glutCloseCallback);
	glutSetWindow(this->glutMainWinID);*/

	return true;
}


/*
 * protein::DofRendererDeferred::release
 */
void protein::DofRendererDeferred::release(void) {
	glDeleteTextures(1, &this->colorBuffer);
	glDeleteTextures(1, &this->normalBuffer);
	glDeleteTextures(1, &this->depthBuffer);
	glDeleteTextures(1, &this->sourceBuffer);
	glDeleteTextures(1, &this->fboMipMapTexId[0]);
	glDeleteTextures(1, &this->fboMipMapTexId[1]);
	glDeleteTextures(1, &this->fboLowResTexId[0]);
	glDeleteTextures(1, &this->fboLowResTexId[1]);
	glDeleteFramebuffers(1, &this->fbo);
	this->nonLinDepthShader.Release();
}


/*
 * protein::DofRendererDeferred::~DofRendererDeferred
 */
protein::DofRendererDeferred::~DofRendererDeferred(void) {
	this->Release();
}


/*
 * protein::DofRendererDeferred::GetCapabilities
 */
bool protein::DofRendererDeferred::GetCapabilities(megamol::core::Call& call) {

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
 * protein::DofRendererDeferred::GetExtents
 */
bool protein::DofRendererDeferred::GetExtents(megamol::core::Call& call) {

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
 * protein::DofRendererDeferred::Render
 */
bool protein::DofRendererDeferred::Render(megamol::core::Call& call) {

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

    // Get camera information
    this->cameraInfo =  crIn->GetCameraParameters();

	int curVP[4];
	glGetIntegerv(GL_VIEWPORT, curVP);

	vislib::math::Vector<float, 3> ray(0, 0,-1);
	vislib::math::Vector<float, 3> up(0, 1, 0);
	vislib::math::Vector<float, 3> right(1, 0, 0);

	up *= sinf(crIn->GetCameraParameters()->HalfApertureAngle());
	right *= sinf(crIn->GetCameraParameters()->HalfApertureAngle())
        		* curVP[2] / curVP[3];

	// Recreate FBO if necessary
	if((curVP[2] != this->width) || (curVP[3] != this->height)) {
		if(!this->createFbo(static_cast<GLuint>(curVP[2]),
				static_cast<GLuint>(curVP[3]))) {
			return false;
		}
		this->width = curVP[2];
		this->height = curVP[3];
		this->widthInv = 1.0/curVP[2];
		this->heightInv = 1.0/curVP[3];
	}


	// 1. Offscreen rendering

	// Enable rendering to FBO
	glBindFramebuffer(GL_FRAMEBUFFER, this->fbo);

	// Enable rendering to color attachents 0 and 1
	GLenum mrt[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	glDrawBuffers(2, mrt);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, this->colorBuffer, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
			GL_TEXTURE_2D, this->normalBuffer, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_TEXTURE_2D, this->depthBuffer, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Call for render
	(*crOut)(0);

	// Detach texture that are not needed anymore
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
			GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_TEXTURE_2D, 0, 0);

	// Prepare rendering screen quad
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);


	// 2. Local lighting

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D,
			this->sourceBuffer, 0);
	GLenum mrt2[] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, mrt2);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, this->normalBuffer);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, this->colorBuffer);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->depthBuffer);

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

	// --> Enable BLinn phong illumination
	glUniform1i(this->blinnPhongShader.ParameterLocation("renderMode"), 0);


	up *= sinf(crIn->GetCameraParameters()->HalfApertureAngle());
	right *= sinf(crIn->GetCameraParameters()->HalfApertureAngle())
        		* static_cast<float>(curVP[2]) / static_cast<float>(curVP[3]);

	// Draw quad
	glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
	glBegin(GL_QUADS);
	glNormal3fv((ray - right - up).PeekComponents());
	glTexCoord2f(0, 0);
	glVertex2f(-1.0f,-1.0f);
	glNormal3fv((ray + right - up).PeekComponents());
	glTexCoord2f(1, 0);
	glVertex2f(1.0f,-1.0f);
	glNormal3fv((ray + right + up).PeekComponents());
	glTexCoord2f(1, 1);
	glVertex2f(1.0f, 1.0f);
	glNormal3fv((ray - right + up).PeekComponents());
	glTexCoord2f(0, 1);
	glVertex2f(-1.0f, 1.0f);
	glEnd();

	this->blinnPhongShader.Disable();

	glBindTexture(GL_TEXTURE_2D, 0);


	// 3. Calc non linear depth

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D,
			this->nonLinDepthBuffer, 0);
	glDrawBuffers(1, mrt2);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->depthBuffer);

	this->nonLinDepthShader.Enable();
	glUniform1i(this->nonLinDepthShader.ParameterLocation("depthBuff"), 0);
	glUniform2f(this->nonLinDepthShader.ParameterLocation("zNearFar"),
			this->cameraInfo->NearClip(), this->cameraInfo->FarClip());


	up *= sinf(crIn->GetCameraParameters()->HalfApertureAngle());
	right *= sinf(crIn->GetCameraParameters()->HalfApertureAngle())
        		* static_cast<float>(curVP[2]) / static_cast<float>(curVP[3]);

	// Draw quad
	glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
	glBegin(GL_QUADS);
	glNormal3fv((ray - right - up).PeekComponents());
	glTexCoord2f(0, 0);
	glVertex2f(-1.0f,-1.0f);
	glNormal3fv((ray + right - up).PeekComponents());
	glTexCoord2f(1, 0);
	glVertex2f(1.0f,-1.0f);
	glNormal3fv((ray + right + up).PeekComponents());
	glTexCoord2f(1, 1);
	glVertex2f(1.0f, 1.0f);
	glNormal3fv((ray - right + up).PeekComponents());
	glTexCoord2f(0, 1);
	glVertex2f(-1.0f, 1.0f);
	glEnd();

	this->nonLinDepthShader.Disable();

	glBindTexture(GL_TEXTURE_2D, 0);

	// 3. Apply depth of field

	switch(this->dofMode) {
	case DOF_SHADERX: this->createReducedTexShaderX(); break;
	case DOF_LEE:     // TODO ?
	case DOF_MIPMAP:  this->createReducedTexMipmap(); break;
	default: break;
	}

	if(this->useGaussian) {
		switch(this->dofMode) {
		case DOF_SHADERX: this->filterShaderX(); break;
		case DOF_MIPMAP:  this->filterMipmap(); break;
		case DOF_LEE:     this->filterLee(); break;
		default: break;
		}
	}

	// Disable rednering to framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Restore viewport
	glViewport(0, 0, this->width, this->height);

	// Preserve the current framebuffer content (e.g. back of the bounding box)
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);

	switch(this->dofMode) {
	case DOF_SHADERX: this->drawBlurShaderX(); break;
	case DOF_LEE: // TODO ?
	case DOF_MIPMAP:  this->drawBlurMipmap(); break;
	default: break;
	}

	glDisable(GL_TEXTURE_2D);
	//glDisable(GL_BLEND);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	//this->renderOrthoView(crOut, crIn);

	GLenum err;
	err = glGetError();
	if(err != GL_NO_ERROR) {
		printf("OpenGL ERROR: %s\n", gluErrorString(err));
	}

	return true;
}


/*
 * protein::DofRendererDeferred::calcCocSlope
 */
float protein::DofRendererDeferred::calcCocSlope(float d_focus, float a, float f) {
	// radius of the circle of confusion
	// according to coc = abs( fd / (d - f) - f d_focus / (d_focus - f) )
	//                    * (d - f) / (a d)
	float d = 2.0f * d_focus;
	float coc = - ((f * d)/(d - f) - (f * d_focus)/(d_focus - f))
        		* (d - f) / (a * d);

	// * _width / 0.035  for resolution independency
	// ATTENTION: disc RADIUS is calculated in the shader, not diameter
	float slope = coc / this->filmWidth*this->width;

	return slope;
}


/*
 * protein::DofRendererDeferred::createFbo
 */
bool protein::DofRendererDeferred::createFbo(GLuint width, GLuint height) {

	// Delete textures + fbo if necessary
	if(glIsFramebuffer(this->fbo)) {
		glDeleteTextures(1, &this->colorBuffer);
		glDeleteTextures(1, &this->normalBuffer);
		glDeleteTextures(1, &this->depthBuffer);
		glDeleteTextures(1, &this->sourceBuffer);
		glDeleteTextures(1, &this->fboMipMapTexId[0]);
		glDeleteTextures(1, &this->fboMipMapTexId[1]);
		glDeleteTextures(1, &this->fboLowResTexId[0]);
		glDeleteTextures(1, &this->fboLowResTexId[1]);
		glDeleteFramebuffers(1, &this->fbo);
	}

	glEnable(GL_TEXTURE_2D);

	glGenTextures(1, &this->colorBuffer);
	glBindTexture(GL_TEXTURE_2D, this->colorBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA,
			GL_UNSIGNED_BYTE, 0);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Normal buffer
	glGenTextures(1, &this->normalBuffer);
	glBindTexture(GL_TEXTURE_2D, this->normalBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA,
			GL_UNSIGNED_BYTE, 0);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Depth buffer
	glGenTextures(1, &this->depthBuffer);
	glBindTexture(GL_TEXTURE_2D, this->depthBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, width, height, 0,
			GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Non linear depth buffer
	glGenTextures(1, &this->nonLinDepthBuffer);
	glBindTexture(GL_TEXTURE_2D, this->nonLinDepthBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, width, height, 0,
			GL_ALPHA, GL_FLOAT, 0);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Source buffer
	glGenTextures(1, &this->sourceBuffer);
	glBindTexture(GL_TEXTURE_2D, this->sourceBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA,
			GL_UNSIGNED_BYTE, 0);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenTextures(2, this->fboMipMapTexId);
	// First mipmap texture
	glBindTexture(GL_TEXTURE_2D, this->fboMipMapTexId[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
			GL_UNSIGNED_BYTE, 0);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
			GL_NEAREST_MIPMAP_NEAREST);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glGenerateMipmap(GL_TEXTURE_2D); // Establish a mipmap chain for the texture
	glBindTexture(GL_TEXTURE_2D, 0);
	// Second mipmap texture
	glBindTexture(GL_TEXTURE_2D, this->fboMipMapTexId[1]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
			GL_UNSIGNED_BYTE, 0);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
			GL_NEAREST_MIPMAP_NEAREST);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glGenerateMipmap(GL_TEXTURE_2D); // Establish a mipmap chain for the texture
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenTextures(2, this->fboLowResTexId);
	// First lowres texture
	glBindTexture(GL_TEXTURE_2D, this->fboLowResTexId[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width/4, height/4, 0, GL_RGBA,
			GL_UNSIGNED_BYTE, 0);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
	// Second lowres texture
	glBindTexture(GL_TEXTURE_2D, this->fboLowResTexId[1]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width/4, height/4, 0, GL_RGBA,
			GL_UNSIGNED_BYTE, 0);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Generate framebuffer
	glGenFramebuffers(1, &this->fbo);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(status != GL_FRAMEBUFFER_COMPLETE) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Could not create fbo");
		return false;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return true;
}


/*
 * protein::DofRendererDeferred::filterShaderX
 */
void protein::DofRendererDeferred::filterShaderX() {

	// Filter horizontal

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->fboLowResTexId[0]);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, this->fboLowResTexId[1], 0);

	this->gaussianHoriz.Enable();
	glUniform1i(this->gaussianHoriz.ParameterLocation("sourceTex"), 0);
	glUniform2f(this->gaussianHoriz.ParameterLocation("screenResInv"),
			4.0f/this->width, 4.0f/this->height);

	glRecti(-1, -1, 1, 1); // Draw screen quad
	this->gaussianHoriz.Disable();
	glBindTexture(GL_TEXTURE_2D, 0);

	// Filter vertical

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->fboLowResTexId[1]);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, this->fboLowResTexId[0], 0);

	this->gaussianVert.Enable();
	glUniform1i(this->gaussianVert.ParameterLocation("sourceTex"), 0);
	glUniform2f(this->gaussianVert.ParameterLocation("screenResInv"),
			4.0f/this->width, 4.0f/this->height);

	glRecti(-1, -1, 1, 1); // Draw screen quad
	this->gaussianVert.Disable();
	glBindTexture(GL_TEXTURE_2D, 0);

}


/*
 * protein::DofRendererDeferred::filterMipmap
 */
void protein::DofRendererDeferred::filterMipmap() {

	int maxMipMaps = (int)floor(log((double) std::max(this->width,
			this->height))/log(2.0)) + 1;

	glActiveTexture(GL_TEXTURE0);

	for(int pass = 0; pass < 2; ++pass) {
		GLuint targetTex;
		GLint screenResInv;

		if(pass == 0) {

			glBindTexture(GL_TEXTURE_2D, this->fboMipMapTexId[0]);
			glGenerateMipmap(GL_TEXTURE_2D); // Generate all mipmaps
			//CHECK_GL_ERROR();

			this->gaussianHoriz.Enable();
			glUniform1i(this->gaussianHoriz.ParameterLocation("sourceTex"), 0);

			screenResInv = this->gaussianHoriz.ParameterLocation("screenResInv");
			targetTex = this->fboMipMapTexId[1];

		}
		else {

			glBindTexture(GL_TEXTURE_2D, this->fboMipMapTexId[1]);
			//CHECK_GL_ERROR();
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
					GL_TEXTURE_2D, 0, 0);

			this->gaussianVert.Enable();
			glUniform1i(this->gaussianVert.ParameterLocation("sourceTex"), 0);

			screenResInv = this->gaussianVert.ParameterLocation("screenResInv");
			targetTex = this->fboMipMapTexId[0];

		}

		// Filter all mipmap levels
		for(int i = 1; i < maxMipMaps; ++i) {
			int resX = std::max(1, this->width / (1<<i));
			int resY = std::max(1, this->height / (1<<i));
			glViewport(0, 0, resX, resY);

			glUniform2f(screenResInv, 1.0f/resX, 1.0f/resY);
			//CHECK_GL_ERROR();

			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
					GL_TEXTURE_2D, targetTex, i);
			//CHECK_FRAMEBUFFER_STATUS();
			//CHECK_GL_ERROR();

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, i);
			//CHECK_GL_ERROR();

			glRecti(-1, -1, 1, 1);
			//CHECK_GL_ERROR();
		}
		// Reset the mipmap levels (needed in 2nd pass for working framebuffer object)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1000);

		// Disable shader
		if(pass == 0) {
			this->gaussianHoriz.Disable();
		}
		else {
			this->gaussianVert.Disable();
		}
	}

	glBindTexture(GL_TEXTURE_2D, 0);
}


/*
 * protein::DofRendererDeferred::filterLee
 */
void protein::DofRendererDeferred::filterLee() {

	int maxMipMaps = (int)floor(log((double) std::max(this->width,
			this->height))/log(2.0)) + 1;

	// Generate all mipmaps
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->fboMipMapTexId[0]);
	glGenerateMipmap(GL_TEXTURE_2D);

	this->gaussianLee.Enable();
	glUniform1i(this->gaussianLee.ParameterLocation("sourceTex"), 0);

	// Filter all mipmap levels
	for(int i=1; i<maxMipMaps; ++i) {
		int resX = std::max(1, this->width / (1<<i));
		int resY = std::max(1, this->height / (1<<i));
		glViewport(0, 0, resX, resY);

		glUniform2f(this->gaussianLee.ParameterLocation("screenResInv"),
				1.0f/resX, 1.0f/resY);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
				GL_TEXTURE_2D, this->fboMipMapTexId[1], i);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, i);

		glRecti(-1, -1, 1, 1);
	}
	// Reset the mipmap levels
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);

	// Swap mipmap textures
	std::swap(this->fboMipMapTexId[0], this->fboMipMapTexId[1]);

	this->gaussianLee.Disable();
}


/*
 * protein::DofRendererDeferred::recalcShaderXParams
 */
void protein::DofRendererDeferred::recalcShaderXParams() {
	const float coc_max = 10.0f; // 10 pixel coc_max (diameter) in shaderX
	const float focalLen = this->focalLength/this->filmWidth*this->width;
	const float d_focus = this->focalDist/this->filmWidth*this->width;
	//const float h = Squared(this->focalLength)/(this->aperture*coc_max);

	const float h = (this->focalLength*this->focalLength)/
			(this->aperture*coc_max); // TODO squared?

	const float coc_inf = (focalLen*focalLen) // TODO Squared?
        		/ (this->aperture * (this->focalDist - focalLen));

	this->dNear  = h * this->focalDist / (h + (this->focalDist - focalLen));

	if(coc_max <= coc_inf) {
		const float d_far90_max = coc_inf * d_focus / (coc_inf - 0.9f*coc_max);
		this->dFar = d_focus + (d_far90_max - d_focus) / 0.9f;
	}
	else {
		const float d_far90_inf = coc_inf * d_focus / (coc_inf - 0.9f*coc_inf);
		this->dFar  = d_focus+(d_far90_inf-d_focus)/0.9f*(coc_max / coc_inf);
	}

	this->clampFar = std::min(coc_inf / coc_max, 1.0f);

	this->dNear *= filmWidth * this->widthInv;
	this->dFar *= filmWidth * this->widthInv;
}


/*
 * protein::DofRendererDeferred::createReducedTexShaderX
 */
void protein::DofRendererDeferred::createReducedTexShaderX() {

    // Create blurred low-res image

    // TODO 2x width?
    //glViewport(0, 0, this->fboWidth/4, this->fboWidth/4);
    glViewport(0, 0, this->width/4, this->height/4);

    // Enable rendering to framebuffer
    //glBindFramebuffer(GL_FRAMEBUFFER, this->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, this->fboLowResTexId[0], 0);
    GLenum mrt[] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, mrt);

    // Bind textures
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->depthBuffer);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->sourceBuffer);

    // Enable shader
    this->reduceShaderX.Enable();
    glUniform2f(this->reduceShaderX.ParameterLocation("screenResInv"),
        this->widthInv, this->heightInv);
    glUniform1i(this->reduceShaderX.ParameterLocation("sourceTex"), 0);
    glUniform1i(this->reduceShaderX.ParameterLocation("depthTex"), 1);
	glUniform2f(this->reduceShaderX.ParameterLocation("zNearFar"),
			this->cameraInfo->NearClip(), this->cameraInfo->FarClip());

    glClear(GL_COLOR_BUFFER_BIT);
    glRecti(-1, -1, 1, 1); // Draw screen quad

    this->reduceShaderX.Disable();

    glBindTexture(GL_TEXTURE_2D, 0);
}


/*
 * protein::DofRendererDeferred::setupReduceMipmap
 */
void protein::DofRendererDeferred::createReducedTexMipmap() {

	//glViewport(0, 0, this->width/4, this->height/4);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, this->fboMipMapTexId[0], 0);
	GLenum mrt[] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, mrt);

	// Bind textures
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, this->depthBuffer);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->sourceBuffer);

	this->reduceMipMap.Enable();

	glUniform2f(this->reduceMipMap.ParameterLocation("screenResInv"),
			this->widthInv, this->heightInv);
	glUniform1i(this->reduceMipMap.ParameterLocation("sourceTex"), 0);
	glUniform1i(this->reduceMipMap.ParameterLocation("depthTex"), 1);
	glUniform1f(this->reduceMipMap.ParameterLocation("d_focus"),
			this->focalDist);
	glUniform1f(this->reduceMipMap.ParameterLocation("d_near"), this->dNear);
	glUniform1f(this->reduceMipMap.ParameterLocation("d_far"), this->dFar);
	glUniform1f(this->reduceMipMap.ParameterLocation("clamp_far"),
			this->clampFar);
	glUniform2f(this->reduceMipMap.ParameterLocation("zNearFar"),
			this->cameraInfo->NearClip(), this->cameraInfo->FarClip());

	glRecti(-1, -1, 1, 1); // Draw screen quad

	this->reduceMipMap.Disable();

	glBindTexture(GL_TEXTURE_2D, 0);
}


/*
 * protein::DofRendererDeferred::drawBlurShaderX
 */
void protein::DofRendererDeferred::drawBlurShaderX() {

	float cocSlope = this->calcCocSlope(this->focalDist, this->aperture,
			this->focalLength);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, this->fboLowResTexId[0]);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, this->depthBuffer);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->sourceBuffer);

	this->blurShaderX.Enable();
	glUniform1i(this->blurShaderX.ParameterLocation("sourceTex"), 0);
	glUniform1i(this->blurShaderX.ParameterLocation("depthTex"), 1);
	glUniform1i(this->blurShaderX.ParameterLocation("sourceTexLow"), 2);
	glUniform2f(this->blurShaderX.ParameterLocation("screenResInv"),
			this->widthInv, this->heightInv);
	glUniform1f(this->blurShaderX.ParameterLocation("maxCoC"), this->maxCoC);
	glUniform1f(this->blurShaderX.ParameterLocation("radiusScale"), this->cocRadiusScale);
	glUniform1f(this->blurShaderX.ParameterLocation("d_focus"), this->focalDist);
	glUniform1f(this->blurShaderX.ParameterLocation("d_near"), this->dNear);
	glUniform1f(this->blurShaderX.ParameterLocation("d_far"), this->dFar);
	glUniform1f(this->blurShaderX.ParameterLocation("clamp_far"), this->clampFar);
	glUniform2f(this->blurShaderX.ParameterLocation("zNearFar"),
			this->cameraInfo->NearClip(), this->cameraInfo->FarClip());
	glUniform1i(this->blurShaderX.ParameterLocation("renderMode"), this->rMode);

	printf("==== near %f, far %f, focalDist %f\n",
			this->cameraInfo->NearClip(),
			this->cameraInfo->FarClip(),
			this->focalDist); // DEBUG

	if(this->originalCoC)
		glUniform1f(this->blurShaderX.ParameterLocation("cocSlope"), -cocSlope);
	else
		glUniform1f(this->blurShaderX.ParameterLocation("cocSlope"), cocSlope);

	// Draw quad
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(1, 0); glVertex2f(1.0f, -1.0f);
	glTexCoord2f(1, 1); glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0, 1); glVertex2f(-1.0f, 1.0f);
	glEnd();

	this->blurShaderX.Disable();

	glBindTexture(GL_TEXTURE_2D, 0);
}


/*
 * protein::DofRendererDeferred::setupBlurMipmap
 */
void protein::DofRendererDeferred::drawBlurMipmap() {

	// compute circle of confusion slope
	float cocSlope = calcCocSlope(this->focalDist, this->aperture,
			this->focalLength);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, this->depthBuffer);

	glActiveTexture(GL_TEXTURE0);
	// rgb channels contain original color tex, alpha contains depth
	glBindTexture(GL_TEXTURE_2D, this->fboMipMapTexId[0]);
	// Generate all mipmaps when not done during gaussian filtering
	if(!this->useGaussian) {
		glGenerateMipmap(GL_TEXTURE_2D);
	}

	this->blurMipMap.Enable();

	glUniform1i(this->blurMipMap.ParameterLocation("sourceTex"), 0);
	glUniform1i(this->blurMipMap.ParameterLocation("depthTex"), 1);
	glUniform2f(this->blurMipMap.ParameterLocation("screenResInv"),
			this->widthInv, this->heightInv);
	glUniform1f(this->blurMipMap.ParameterLocation("maxCoC"), this->maxCoC);
	glUniform1f(this->blurMipMap.ParameterLocation("maxCoCInv"), 1.0f/this->maxCoC);
	glUniform1f(this->blurMipMap.ParameterLocation("d_focus"), this->focalDist);
	glUniform1f(this->blurMipMap.ParameterLocation("cocSlope"), cocSlope);
	glUniform2f(this->blurMipMap.ParameterLocation("zNearFar"),
			this->cameraInfo->NearClip(), this->cameraInfo->FarClip());
	glUniform1i(this->blurMipMap.ParameterLocation("renderMode"), this->rMode);

	printf("==== near %f, far %f, focalDist %f\n",
			this->cameraInfo->NearClip(),
			this->cameraInfo->FarClip(),
			this->focalDist); // DEBUG

	// Draw quad
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(1, 0); glVertex2f(1.0f, -1.0f);
	glTexCoord2f(1, 1); glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0, 1); glVertex2f(-1.0f, 1.0f);
	glEnd();

	this->blurMipMap.Disable();

	glBindTexture(GL_TEXTURE_2D, 0);
}


/*
 * protein::DofRendererDeferred::renderOrthoView
 */
bool protein::DofRendererDeferred::renderOrthoView(megamol::core::view::CallRenderDeferred3D *crOut,
		core::view::CallRender3D *crIn) {


	vislib::math::Vector<float, 3> ray(0, 0,-1);
	vislib::math::Vector<float, 3> up(0, 1, 0);
	vislib::math::Vector<float, 3> right(1, 0, 0);

	glDisable(GL_BLEND);
	glDisable(GL_LIGHTING);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	// Enable rendering to FBO
	glBindFramebuffer(GL_FRAMEBUFFER, this->fbo);

	// Enable rendering to color attachents 0 and 1
	GLenum mrt[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	glDrawBuffers(2, mrt);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->colorBuffer, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, this->normalBuffer, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->depthBuffer, 0);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Call for render
	(*crOut)(0);

	// Detach texture that are not needed anymore
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);

	/// 2. Local lighting ///

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,  GL_TEXTURE_2D, this->sourceBuffer, 0);
	// Enable rendering to color attachment 0
	GLenum mrt2[] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, mrt2);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, this->normalBuffer);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, this->colorBuffer);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->depthBuffer);

	this->blinnPhongShader.Enable();
	glUniform3fv(this->blinnPhongShader.ParameterLocation("camWS"), 1,
			crIn->GetCameraParameters()->Position().PeekCoordinates());
	glUniform2f(this->blinnPhongShader.ParameterLocation("clipPlanes"), crIn->GetCameraParameters()->NearClip(), crIn->GetCameraParameters()->FarClip());
	glUniform2f(this->blinnPhongShader.ParameterLocation("winSize"), this->width, this->height);
	glUniform1i(this->blinnPhongShader.ParameterLocation("depthBuff"), 0);
	glUniform1i(this->blinnPhongShader.ParameterLocation("colorBuff"), 1);
	glUniform1i(this->blinnPhongShader.ParameterLocation("normalBuff"), 2);

	// --> Enable BLinn phong illumination
	glUniform1i(this->blinnPhongShader.ParameterLocation("renderMode"), 0);


	up *= sinf(crIn->GetCameraParameters()->HalfApertureAngle());
	right *= sinf(crIn->GetCameraParameters()->HalfApertureAngle())
        		 * static_cast<float>(this->width) / static_cast<float>(this->height);

	// Draw quad
	glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
	glBegin(GL_QUADS);
	glNormal3fv((ray - right - up).PeekComponents());
	glTexCoord2f(0, 0);
	glVertex2f(-1.0f,-1.0f);
	glNormal3fv((ray + right - up).PeekComponents());
	glTexCoord2f(1, 0);
	glVertex2f(1.0f,-1.0f);
	glNormal3fv((ray + right + up).PeekComponents());
	glTexCoord2f(1, 1);
	glVertex2f(1.0f, 1.0f);
	glNormal3fv((ray - right + up).PeekComponents());
	glTexCoord2f(0, 1);
	glVertex2f(-1.0f, 1.0f);
	glEnd();

	this->blinnPhongShader.Disable();
	// Enable rendering to FBO
	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	// Render to screen

	// Set viewport
	glViewport(this->width-200, 0, this->width, 100);
	//glOrtho(-100.0f, 100.0f, -100.0f, 100.0f, 1.0f, 100.0f);

	glBindTexture(GL_TEXTURE_2D, this->sourceBuffer);

	// Draw quad
	glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex2f(-1.0f,-1.0f);
	glTexCoord2f(1, 0); glVertex2f(1.0f,-1.0f);
	glTexCoord2f(1, 1); glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0, 1); glVertex2f(-1.0f, 1.0f);
	glEnd();


	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	// Restore viewport
	glViewport(0, 0, this->width, this->height);

	return true;
}


/*
 * protein::DofRendererDeferred::updateParams
 */
bool protein::DofRendererDeferred::updateParams() {

	// Render mode
	if (this->rModeParam.IsDirty()) {
		this->rModeParam.ResetDirty();
		this->rMode = static_cast<RenderMode>(this->rModeParam.Param<core::param::EnumParam>()->Value());
	}
	// Depth of field mode
	if (this->dofModeParam.IsDirty()) {
		this->dofModeParam.ResetDirty();
		this->dofMode = static_cast<DepthOfFieldMode> (this->dofModeParam.Param<core::param::EnumParam>()->Value());
	}
	// Toggle gaussian filter
	if (this->toggleGaussianParam.IsDirty()) {
		this->useGaussian = this->toggleGaussianParam.Param<core::param::BoolParam>()->Value();
		this->toggleGaussianParam.ResetDirty();
	}
	// Focal distance (note: scaled x10 for greater accuracy)
	if (this->focalDistParam.IsDirty()) {
		this->focalDist = this->focalDistParam.Param<core::param::FloatParam>()->Value()/10.0f;
		this->recalcShaderXParams();
	}
	// Aperture
	if (this->apertureParam.IsDirty()) {
		this->aperture = this->apertureParam.Param<core::param::FloatParam>()->Value();
		this->apertureParam.ResetDirty();
		this->recalcShaderXParams();
	}
	return true;
}


/*
 * protein::DofRendererDeferred::glutParamWinDisplayFunc
 */
void protein::DofRendererDeferred::glutParamWinDisplayFunc() {
	printf("==== PARAM WIN display callback\n");
	glClearColor(1.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);
	glColor3f(1.0, 0.0, 0.0);
	glRecti(-1, -1, 1, 1);
	glutSwapBuffers();
}


/*
 * protein::DofRendererDeferred::glutParamWinReshapeFunc
 *
 * s. http://www.lighthouse3d.com/tutorials/glut-tutorial/preparing-the-window-for-a-reshape/
 */
void protein::DofRendererDeferred::glutParamWinReshapeFunc(int w, int h) {

	printf("==== PARAM WIN reshape callback\n");
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if(h == 0)
		h = 1;
	float ratio = 1.0* w / h;

	// Use the Projection Matrix
	glMatrixMode(GL_PROJECTION);

	// Reset Matrix
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(45,ratio,1,1000);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);
}



