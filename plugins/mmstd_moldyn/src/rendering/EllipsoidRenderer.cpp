/*
* EllipsoidRenderer.cpp
*
* Copyright (C) 2008-2015 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "EllipsoidRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/math/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "mmcore/utility/log/Log.h"
#include "inttypes.h"
#include <iostream>
#include <cstring>
#include <GL/glu.h>


using namespace megamol;
using namespace megamol::core;
using namespace megamol::stdplugin;
using namespace megamol::stdplugin::moldyn;
using namespace megamol::stdplugin::moldyn::rendering;

EllipsoidRenderer::EllipsoidRenderer(void) : Renderer3DModule(),
getDataSlot("getData", "The slot to fetch the ellipsoidal data"){

	this->getDataSlot.SetCompatibleCall<core::moldyn::EllipsoidalParticleDataCallDescription>();
	this->MakeSlotAvailable(&this->getDataSlot);
}
EllipsoidRenderer::~EllipsoidRenderer(void) {
	this->Release();
}
bool EllipsoidRenderer::create(void){

	if (!ogl_IsVersionGEQ(2, 0))
		return false;

	if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions())
		return false;


	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

	using namespace megamol::core::utility::log;
	using namespace vislib::graphics::gl;

	ShaderSource vertSrc;
	ShaderSource fragSrc;

	if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("ellipsoid::vertex", vertSrc)) {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for ellipsoid shader");
		return false;
	}
	if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("ellipsoid::fragment", fragSrc)) {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for ellipsoid shader");
		return false;
	}
	try{
		if (!ellipsoidShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())){
			megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
				"unable to compile ellipsoid shader: unknown error\n");
			return false;
		}
	}
	catch (AbstractOpenGLShader::CompileException ce){
		megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
			"unable to compile ellipsoid shader (@%s): %s\n",
			vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
			ce.FailedAction()), ce.GetMsgA());
		return false;
	}
	catch (vislib::Exception e) {
		megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
			"unable to compile ellipsoid shader: %s\n", e.GetMsgA());
		return false;
	}
	catch (...) {
		megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
			"unable to compile ellipsoid shader: unknown exception\n");
		return false;
	}

	return true;

}


bool EllipsoidRenderer::GetExtents(Call& call) {
	view::CallRender3D *cr = dynamic_cast<view::CallRender3D *>(&call);
	if (cr == NULL) return false;

	core::moldyn::EllipsoidalParticleDataCall *epdc = this->getDataSlot.CallAs<core::moldyn::EllipsoidalParticleDataCall>();
	if ((epdc != NULL) && ((*epdc)(1))) {
		cr->SetTimeFramesCount(epdc->FrameCount());
		cr->AccessBoundingBoxes() = epdc->AccessBoundingBoxes();

		//float scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
		//if (scaling > 0.0000001) {
		//	scaling = 10.0f / scaling;
		//}
		//else {
		//	scaling = 1.0f;
		//}
		//cr->AccessBoundingBoxes().MakeScaledWorld(scaling);
        cr->AccessBoundingBoxes().MakeScaledWorld(1.0f);

	}
	else {
		cr->SetTimeFramesCount(1);
		cr->AccessBoundingBoxes().Clear();
	}

	return true;
}
void EllipsoidRenderer::release(void) {
	this->ellipsoidShader.Release();

}
bool EllipsoidRenderer::Render(Call& call){
	view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
	if (cr == NULL) return false;

	this->cameraInfo = cr->GetCameraParameters();

	core::moldyn::EllipsoidalParticleDataCall *epdc = this->getDataSlot.CallAs<core::moldyn::EllipsoidalParticleDataCall>();

	if (epdc == NULL) return false;

	epdc->SetFrameID(static_cast<int>(cr->Time()));
	if (!(*epdc)(1)) return false;

	//glPushMatrix();

	//float scale;
	//if (!vislib::math::IsEqual(epdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
	//	scale = 2.0f / epdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
	//}
	//else {
	//	scale = 1.0f;
	//}

	//glScalef(scale, scale, scale);

	epdc->SetFrameID(static_cast<int>(cr->Time()));
	if (!(*epdc)(0)) return false;

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
	glDisable(GL_POINT_SPRITE_ARB);

	//glEnableClientState(GL_VERTEX_ARRAY);
	//glEnableClientState(GL_COLOR_ARRAY);

	glEnable(GL_DEPTH_TEST);

	float viewportStuff[4] = {
		cameraInfo->TileRect().Left(),
		cameraInfo->TileRect().Bottom(),
		cameraInfo->TileRect().Width(),
		cameraInfo->TileRect().Height() };
	if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
	if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];

	this->ellipsoidShader.Enable();

	glUniform4fvARB(this->ellipsoidShader.ParameterLocation("viewAttr"), 1, viewportStuff);
	glUniform3fvARB(this->ellipsoidShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
	glUniform3fvARB(this->ellipsoidShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
	glUniform3fvARB(this->ellipsoidShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());

	unsigned int radiiAttrib = glGetAttribLocationARB(this->ellipsoidShader, "radii");
	unsigned int quatAttrib = glGetAttribLocationARB(this->ellipsoidShader, "quatC");
	glEnableVertexAttribArrayARB(radiiAttrib);
	glEnableVertexAttribArrayARB(quatAttrib);

	for (unsigned int i = 0; i < epdc->GetParticleListCount(); i++){

		auto &elParts = epdc->AccessParticles(i);

        if (elParts.GetCount() == 0 || elParts.GetQuatData() == nullptr || elParts.GetRadiiData() == nullptr) continue;

		glVertexAttribPointerARB(radiiAttrib, 3, GL_FLOAT, false, elParts.GetRadiiDataStride(), elParts.GetRadiiData());
		glVertexAttribPointerARB(quatAttrib, 4, GL_FLOAT, false, elParts.GetQuatDataStride(), elParts.GetQuatData());

		switch (elParts.GetColourDataType()) {
        case core::moldyn::EllipsoidalParticleDataCall::Particles::COLDATA_NONE :
			// Doesn't seem to be accessible in any shader, using values or pointer to values doesn't differ in outcome
			glColor3ubv(elParts.GetGlobalColour());
			break;
        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(3, GL_UNSIGNED_BYTE, elParts.GetColourDataStride(), elParts.GetColourData());
			break;
        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(4, GL_UNSIGNED_BYTE, elParts.GetColourDataStride(), elParts.GetColourData());
			break;
        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(3, GL_FLOAT, elParts.GetColourDataStride(), elParts.GetColourData());
			break;
        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(4, GL_FLOAT, elParts.GetColourDataStride(), elParts.GetColourData());
			break;
		default:
			glColor3ub(127, 127, 127);
			break;
		}

		switch (elParts.GetVertexDataType()) {
        case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE:
			continue;
        case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_FLOAT, elParts.GetVertexDataStride(), elParts.GetVertexData());
			break;
        case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(4, GL_FLOAT, elParts.GetVertexDataStride(), elParts.GetVertexData());
			break;
        case core::moldyn::EllipsoidalParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_SHORT, elParts.GetVertexDataStride(), elParts.GetVertexData());
			break;
		default:
			continue;
		}

		glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(elParts.GetCount()));
		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisable(GL_TEXTURE_1D);

	}
    glDisableVertexAttribArrayARB(radiiAttrib);
    glDisableVertexAttribArrayARB(quatAttrib);
    epdc->Unlock();
	this->ellipsoidShader.Disable();
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	// Check for opengl error
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		printf("Fehler ist: %s", gluErrorString(err));


	}

	return true;
}