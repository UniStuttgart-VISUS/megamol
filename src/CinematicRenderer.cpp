/*
* CinematicRenderer.cpp
*
* Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
//#define _USE_MATH_DEFINES
#include "CinematicRenderer.h"
#include "CallCinematicCamera.h"
#include "CameraParamsOverride.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/view/View3D.h"
#include "vislib/Array.h"
#include "vislib/math/Point.h"
#include "vislib/sys/Log.h"
#include "vislib/String.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/graphics/gl/CameraOpenGL.h"
#include "vislib/StringSerialiser.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/Thread.h"
#include "vislib/Trace.h"
#include <iostream>

//#include <cmath>

using namespace megamol;
using namespace core;
using namespace view;
using namespace cinematiccamera;
using namespace vislib::sys;

/*
* CinematicRenderer::CinematicRenderer
*/
CinematicRenderer::CinematicRenderer(void) : Renderer3DModule(),
slaveRendererSlot("renderer", "outgoing renderer"),
keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper."),
rendererActiveSlot("active", "De-/Activaties outgoing renderer"),
renderModeParam("renderMode", "The rendering mode"),
addKeyframeParam("addKeyframe", "add a new Keyframe at the current position"),
manipulateKeyframe("manipulateKeyframe", "Toggle manipulation of the currently selected Keyframe"),
frameCnt(0), bboxs(), scale(0.5f), steps("steps", "amount of interpolation steps between keyframes") {

	this->slaveRendererSlot.SetCompatibleCall<CallRender3DDescription>();
	this->MakeSlotAvailable(&this->slaveRendererSlot);

	this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
	this->MakeSlotAvailable(&this->keyframeKeeperSlot);

	this->rendererActiveSlot.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->rendererActiveSlot);

	this->currentMode = OVERVIEW;
	param::EnumParam *rm = new param::EnumParam(int(this->currentMode));
	rm->SetTypePair(OVERVIEW, "Overview");
	rm->SetTypePair(PREVIEW, "Preview");
	this->renderModeParam << rm;
	this->MakeSlotAvailable(&this->renderModeParam);

	this->addKeyframeParam.SetParameter(new param::ButtonParam('k'));
	this->MakeSlotAvailable(&this->addKeyframeParam);

	this->steps.SetParameter(new param::IntParam(20));
	this->MakeSlotAvailable(&this->steps);

	this->manipulateKeyframe.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->manipulateKeyframe);


}

/*
* CinematicRenderer::~CinematicRenderer
*/
CinematicRenderer::~CinematicRenderer(void) {
	this->Release();
}

/*
* CinematicRenderer::create
*/

bool CinematicRenderer::create(void) {
	// intentionally empty
	return true;
}


/*
* CinematicRenderer::release
*/
void CinematicRenderer::release(void) {
	// intentionally empty
}


/*
* CinematicRenderer::GetCapabilities
*/
bool CinematicRenderer::GetCapabilities(Call& call) {
	CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
	if (cr3d == NULL) return false;

	CallRender3D *oc = this->slaveRendererSlot.CallAs<CallRender3D>();
	if (!(oc == NULL) || (!(*oc)(2))) {
		cr3d->AddCapability(oc->GetCapabilities());
	}

	return true;
}


/*
* CinematicRenderer::GetExtents
*TODO: union von keyframes & molekül bbx
*/
bool CinematicRenderer::GetExtents(Call& call) {
	auto cr3d = dynamic_cast<CallRender3D*>(&call);
	if (cr3d == NULL) return false;

	auto oc = this->slaveRendererSlot.CallAs<CallRender3D>();
	if (oc == NULL) return false;

	auto kfc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
	if (!(*kfc)(CallCinematicCamera::CallForGetKeyframes)) return false;

	// Get bounding box of renderer.
	if (!(*oc)(1)) return false;

	// Get bounding box of spline.
	if (!(*kfc)(CallCinematicCamera::CallForGetKeyframes)) return false;

	*cr3d = *oc;

	// Compute clip box including spline (in world space) and object (in world space).
	auto bbox = cr3d->AccessBoundingBoxes().WorldSpaceClipBox();
	bbox.Union(*kfc->getBoundingBox());

	cr3d->AccessBoundingBoxes().SetWorldSpaceClipBox(bbox);

	return true;

}


/*
* CinematicRenderer::Render
*/
bool CinematicRenderer::Render(Call& call) {

	UpdateParameters(dynamic_cast<CallRender3D*>(&call));

	switch (currentMode) {
		case CinematicRenderer::RenderingMode::OVERVIEW
			: return CinematicRenderer::RenderOverview(call);
		case CinematicRenderer::RenderingMode::PREVIEW
			: return CinematicRenderer::RenderPreview(call);
		default : return false;	

	}
}


bool CinematicRenderer::RenderOverview(Call& call){

	CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
	auto camParams = new vislib::graphics::CameraParamsStore();
	if (cr3d == NULL) return false;


	auto oc = this->slaveRendererSlot.CallAs<CallRender3D>();
	if (oc == nullptr) return false;
	*oc = *cr3d;
	oc->SetTime(cr3d->Time());

	// Call original renderer.
	::glMatrixMode(GL_MODELVIEW);
	::glPushMatrix();
	(*oc)(0);
	::glMatrixMode(GL_MODELVIEW);
	::glPopMatrix();

	// Draw the splines.
	CallCinematicCamera *kfc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();

	if (!(*kfc)(CallCinematicCamera::CallForGetKeyframes)) return false;

	vislib::Array<Keyframe> *keyframes = kfc->getKeyframes();

	if (!keyframes) return false;

	glPushMatrix();
	//::glScalef(this->scale, this->scale, this->scale);

	if (keyframes->Count() > 1){
		::glLineWidth(2.5);
		::glColor3f(1.0, 0.0, 0.0);
		::glDisable(GL_LIGHTING);
		::glBegin(GL_LINE_STRIP);
		
		if (steps.Param<param::IntParam>()->Value() == 1 || keyframes->Count() < 3){
			for (unsigned int i = 0; i < keyframes->Count(); i++){
				glVertex3f((*keyframes)[i].getCamPosition().GetX(), (*keyframes)[i].getCamPosition().GetY(), (*keyframes)[i].getCamPosition().GetZ());
			}
		}
		else {
			for (float s = 0.0f; s <= (float)keyframes->Count() - 1; s = s + (1.0f / (float)steps.Param<param::IntParam>()->Value())){
				
				kfc->setIndexToInterpolate(s);
				if (!(*kfc)(CallCinematicCamera::CallForInterpolatedKeyframe)){
					vislib::sys::Log::DefaultLog.WriteError("CallForInterpolatedKeyframe failed!");
					return false;
				};
				glVertex3f(kfc->getInterpolatedKeyframe().getCamPosition().GetX(), kfc->getInterpolatedKeyframe().getCamPosition().GetY(), kfc->getInterpolatedKeyframe().getCamPosition().GetZ());
			}
		}


		::glEnd();
	}

	// draw the selection marker
	if ((*kfc)(CallCinematicCamera::CallForGetSelectedKeyframe)) {
		// no exact keyframe is selected
		if (kfc->getSelectedKeyframe().getID() == -1){
			kfc->setIndexToInterpolate(kfc->getSelectedKeyframeIndex());
			if (!(*kfc)(CallCinematicCamera::CallForInterpolatedKeyframe)){
				vislib::sys::Log::DefaultLog.WriteError("CallForInterpolatedKeyframe failed!");
				return false;
			}

			auto selectedPos = kfc->getInterpolatedKeyframe().getCamPosition();
			glLineWidth(1.5);
			glColor3f(1.0, 1.0, 0.0);
			glBegin(GL_LINES);
			glVertex3f(selectedPos.GetX() + 0.05f, selectedPos.GetY() - 0.05f, selectedPos.GetZ() + 0.05f);
			glVertex3f(selectedPos.GetX() - 0.05f, selectedPos.GetY() + 0.05f, selectedPos.GetZ() - 0.05f);

			glVertex3f(selectedPos.GetX() + 0.05f, selectedPos.GetY() + 0.05f, selectedPos.GetZ() - 0.05f);
			glVertex3f(selectedPos.GetX() - 0.05f, selectedPos.GetY() - 0.05f, selectedPos.GetZ() + 0.05f);

			glVertex3f(selectedPos.GetX() - 0.05f, selectedPos.GetY() + 0.05f, selectedPos.GetZ() + 0.05f);
			glVertex3f(selectedPos.GetX() + 0.05f, selectedPos.GetY() - 0.05f, selectedPos.GetZ() - 0.05f);

			glEnd();


		}
		// an exact keyframe is selected, ignore dummy keyframe
		else if (kfc->getSelectedKeyframe().getID() != -2){

			if (!this->manipulateKeyframe.Param<param::BoolParam>()->Value()) {

				auto selectedPos = kfc->getSelectedKeyframe().getCamPosition();

				glLineWidth(1.5);
				glColor3f(1.0, 1.0, 0.0);
				glBegin(GL_LINES);
				glVertex3f(selectedPos.GetX() + 0.1f, selectedPos.GetY() - 0.1f, selectedPos.GetZ() + 0.1f);
				glVertex3f(selectedPos.GetX() - 0.1f, selectedPos.GetY() + 0.1f, selectedPos.GetZ() - 0.1f);

				glVertex3f(selectedPos.GetX() + 0.1f, selectedPos.GetY() + 0.1f, selectedPos.GetZ() - 0.1f);
				glVertex3f(selectedPos.GetX() - 0.1f, selectedPos.GetY() - 0.1f, selectedPos.GetZ() + 0.1f);

				glVertex3f(selectedPos.GetX() - 0.1f, selectedPos.GetY() + 0.1f, selectedPos.GetZ() + 0.1f);
				glVertex3f(selectedPos.GetX() + 0.1f, selectedPos.GetY() - 0.1f, selectedPos.GetZ() - 0.1f);

				glEnd();
			}
			else {
				vislib::math::Point<float, 3> pos = kfc->getSelectedKeyframe().getCamPosition();
				vislib::math::Point<float, 3> lookAt = kfc->getSelectedKeyframe().getCamLookAt();
		/*		
				auto calc = lookAt - pos;
				calc.Normalise();
				calc = calc*0.5;
				calc.SetX(calc.GetX() + pos.GetX());
				calc.SetX(calc.GetY() + pos.GetY());
				calc.SetX(calc.GetZ() + pos.GetZ());
				
			*/		

				glLineWidth(1.0);
				glColor3f(1.0, 1.0, 0.0);
				glBegin(GL_LINES);

				glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ());
				glVertex3f(pos.GetX() + 0.5f, pos.GetY(), pos.GetZ());

				glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ());
				glVertex3f(pos.GetX(), pos.GetY() + 0.5f, pos.GetZ());

				glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ());
				glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ() + 0.5f);
				glEnd();
				

			}
		}
	}

	glPopMatrix();

	return true;
}

bool CinematicRenderer::RenderPreview(Call& call){
	CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
	if (cr3d == NULL) return false;

	auto oc = this->slaveRendererSlot.CallAs<CallRender3D>();
	if (oc == nullptr) return false;

	// Get the camera to be previewed
	auto kfc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
	if (kfc == nullptr) return false;


	vislib::SmartPtr<vislib::graphics::CameraParameters> camParams(
		new CameraParamsOverride(cr3d->GetCameraParameters()));

	if (!(*kfc)(CallCinematicCamera::CallForGetSelectedKeyframe)) return false;

	// ignore dummy keyframe
	if (kfc->getSelectedKeyframe().getID() != -2){

		// no exact keyframe is selected
		if (kfc->getSelectedKeyframe().getID() == -1){
			kfc->setIndexToInterpolate(kfc->getSelectedKeyframeIndex());
			if (!(*kfc)(CallCinematicCamera::CallForInterpolatedKeyframe)){
				vislib::sys::Log::DefaultLog.WriteError("CallForInterpolatedKeyframe failed!");
				return false;
			}
			kfc->getInterpolatedKeyframe().putCamParameters(camParams);
		}
		else {
			kfc->getSelectedKeyframe().putCamParameters(camParams);
		}

		*oc = *cr3d;
		(*oc)(1);
		oc->SetCameraParameters(camParams);

		int octfc = oc->TimeFramesCount();
		// ToDo:
		// Code fuer Screenshot-Mode!! Hier muesste statt cr3d->Time die Zeit des selected keyframe eingetragen werden (normalisiert * Zeit-Skalierung)
		oc->SetTime(vislib::math::Min<float>(cr3d->Time(), static_cast<float>(octfc - 1)));

		vislib::graphics::gl::CameraOpenGL hackCam(camParams);
		//this->hackCam.SetParameters(camParams);

		::glMatrixMode(GL_PROJECTION);
		::glPushMatrix();
		hackCam.glSetProjectionMatrix();

		::glMatrixMode(GL_MODELVIEW);
		::glPushMatrix();
		hackCam.glSetViewMatrix();
		(*oc)(0);
		::glMatrixMode(GL_MODELVIEW);
		::glPopMatrix();
	}
	return true;
}

void CinematicRenderer::UpdateParameters(CallRender3D *cr3d){

	if (this->renderModeParam.IsDirty()) {
		this->currentMode = static_cast<CinematicRenderer::RenderingMode>
			(int(this->renderModeParam.Param<param::EnumParam>()->Value()));
	}

	if (this->addKeyframeParam.IsDirty()){

		CallCinematicCamera *kfc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
		if (kfc != NULL && cr3d != NULL){

			vislib::SmartPtr<vislib::graphics::CameraParameters> paramsOverride = new CameraParamsOverride(cr3d->GetCameraParameters());
	/*		vislib::graphics::Camera *cam;
			cam = new vislib::graphics::Camera();
			cam->Parameters()->CopyFrom(cr3d->GetCameraParameters());
			*/
			// delete dummy keyframe if it is still there
			if (kfc->getKeyframes()->Count() == 1 && kfc->getSelectedKeyframe().getID() == -2){
				kfc->deleteKeyframe(0);
			}

			kfc->setCameraForNewKeyframe(paramsOverride);
			//kfc->setCameraForNewKeyframe(*cam);
			if ((*kfc)(CallCinematicCamera::CallForNewKeyframeAtPosition)){
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
					"Keyframe added");
			}
		}
		this->addKeyframeParam.ResetDirty();
	}
}

// triggered only when "TAB" is pressed
bool CinematicRenderer::MouseEvent(float x, float y, core::view::MouseFlags flags) {
	// show manipulator whenever a keyframe is selected. (implement in renderer)
	// if not needed anymore
	if (this->manipulateKeyframe.Param<param::BoolParam>()->Value()) {
		// do manipulator stuff

		// consume mouse event
		return false;
	}
	else {
		// do not consume mouse event
		return true;
	}
}