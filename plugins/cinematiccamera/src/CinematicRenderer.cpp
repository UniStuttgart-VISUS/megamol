/*
* CinematicRenderer.cpp
*
* Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"

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

#include "CinematicRenderer.h"
#include "CallCinematicCamera.h"

//#define _USE_MATH_DEFINES

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
    editKeyframeParam( "01 Edit Selected Keyframe", "Toggle manipulation of the currently selected Keyframe"),
    showLookatParam(   "02 Show LookAt", "Render the path of the lookat point"),
    stepsParam(        "03 Spline Subdivision", "amount of interpolation steps between keyframes") ,
    loadTimeParam(     "04 Load Time", "load time from slave renderer")
    {


    this->slaveRendererSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->slaveRendererSlot);

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // init variables --- UNUSED so far
    //this->mouseX;
    //this->mouseY;
    //this->startSelect;
    //this->endSelect;
    //this->bboxs;

	this->showLookatParam.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->showLookatParam);

	this->editKeyframeParam.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->editKeyframeParam);

    this->stepsParam.SetParameter(new param::IntParam(20));
    this->MakeSlotAvailable(&this->stepsParam);

    this->loadTimeParam.SetParameter(new param::ButtonParam('t'));
    this->MakeSlotAvailable(&this->loadTimeParam);
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
*/
bool CinematicRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
	if (cr3d == NULL) return false;

	view::CallRender3D *oc = this->slaveRendererSlot.CallAs<CallRender3D>();
	if (oc == NULL) return false;

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
	if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeper)) return false;

	// Get bounding box of renderer.
	if (!(*oc)(1)) return false;
	*cr3d = *oc;

    // Compute bounding box including spline (in world space) and object (in world space).
    vislib::math::Cuboid<float> bboxCR3D = cr3d->AccessBoundingBoxes().WorldSpaceBBox();
    vislib::math::Cuboid<float> cboxCR3D = cr3d->AccessBoundingBoxes().WorldSpaceClipBox();

    if (ccc->getKeyframes()->Count() > 0) {
        // Get bounding box of spline.
        vislib::math::Cuboid<float> *bboxCCC = ccc->getBoundingBox();
        bboxCR3D.Union(*bboxCCC);
        cboxCR3D.Union(*bboxCCC); // use boundingbox to get new clipbox
    }

	//cr3d->AccessBoundingBoxes().SetWorldSpaceBBox(bboxCR3D);
    //cr3d->AccessBoundingBoxes().SetWorldSpaceClipBox(cboxCR3D);

    cr3d->AccessBoundingBoxes().SetWorldSpaceClipBox(bboxCR3D);

	return true;
}


/*
* CinematicRenderer::Render
*/
bool CinematicRenderer::Render(Call& call) {

    CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
    if (cr3d == NULL) return false;

    view::CallRender3D *oc = this->slaveRendererSlot.CallAs<CallRender3D>();
    if (oc == NULL) return false;

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (ccc == NULL) return false;

    // Update parameter
    if (this->loadTimeParam.IsDirty()) {
        float tt = static_cast<float>(oc->TimeFramesCount());
        ccc->setTotalTime(tt);
        ccc->setChangedTotalTime(true);
        vislib::sys::Log::DefaultLog.WriteInfo(" CINEMATIC RENDERER [loadTimeParam] Set frame count time from slave renderer data source.");
        this->loadTimeParam.ResetDirty();
    }

    // Propagate camera parameter to keyframe keeper
    vislib::SmartPtr<vislib::graphics::CameraParameters> camPam = cr3d->GetCameraParameters();
    ccc->setCameraParameter(camPam);
    ccc->setChangedCameraParameter(true);

    // Update data in cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeper)) return false;

    // ...
    *oc = *cr3d;
    oc->SetTime(cr3d->Time());

    // Get pointer to keyframes
    vislib::Array<Keyframe> *keyframes = ccc->getKeyframes();

    // Edit selected keyframe
    if (this->editKeyframeParam.Param<param::BoolParam>()->Value()) {
        // TODO





    }

    // Recalculate spline only when keyframes changed
    if (ccc->changedKeyframes()) {
        ccc->setChangedKeyframes(false);
// TODO


    }

    // Call original renderer.
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    (*oc)(0);
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // Draw cinematic renderer stuff
    if (keyframes->Count() > 0) {

        glPushMatrix();
        glDisable(GL_LIGHTING);

        // draw spline
        glLineWidth(2.0f);
        glColor3f(1.0f, 0.0f, 0.0f);
        glBegin(GL_LINE_STRIP);
        for (unsigned int i = 0; i < keyframes->Count()-1; i++) {
            int steps = this->stepsParam.Param<param::IntParam>()->Value();
            float startTime = (*keyframes)[i].getTime();
            float deltaTimeStep = ((*keyframes)[i + 1].getTime() - startTime) / (float)steps;

            for (int m = 0; m <= steps; m++) {

                ccc->setInterpolatedKeyframeTime(startTime + deltaTimeStep*(float)m);
                ccc->setChangedInterpolatedKeyframeTime(true);
                if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeper)) return false;
                Keyframe *k = ccc->getInterpolatedKeyframe();
                glVertex3f(k->getCamPosition().GetX(), k->getCamPosition().GetY(), k->getCamPosition().GetZ());

            }
        }
        glEnd();

        // draw look-at vector
        if (this->showLookatParam.Param<param::BoolParam>()->Value()) {
            glLineWidth(1.0f);
            glColor3f(0.0f, 1.0f, 1.0f);
            glBegin(GL_LINES);
            if (this->stepsParam.Param<param::IntParam>()->Value() == 1 || keyframes->Count() < 3) {
                for (unsigned int i = 0; i < keyframes->Count(); i++) {
                    glVertex3f((*keyframes)[i].getCamPosition().GetX(), (*keyframes)[i].getCamPosition().GetY(), (*keyframes)[i].getCamPosition().GetZ());
                    glVertex3f((*keyframes)[i].getCamLookAt().GetX(), (*keyframes)[i].getCamLookAt().GetY(), (*keyframes)[i].getCamLookAt().GetZ());
                }
            }
            else {
                for (float s = 0.0f; s <= (float)keyframes->Count() - 1.0f; s = s + (1.0f / (float)stepsParam.Param<param::IntParam>()->Value())) {
                    ccc->setInterpolatedKeyframeTime(s);
                    ccc->setChangedInterpolatedKeyframeTime(true);
                    if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeper)) return false;
                    Keyframe *k = ccc->getInterpolatedKeyframe();
                    glVertex3f(k->getCamPosition().GetX(), k->getCamPosition().GetY(), k->getCamPosition().GetZ());
                    glVertex3f(k->getCamLookAt().GetX(), k->getCamLookAt().GetY(), k->getCamLookAt().GetZ());
                }
            }
            glEnd();
        }


        // draw the selection marker
        Keyframe *s = ccc->getSelectedKeyframe();

        vislib::math::Point<float, 3> pos = s->getCamPosition();
        vislib::math::Point<float, 3> lookAt = s->getCamLookAt();

        glLineWidth(1.0f);
        glColor3f(0.0f, 0.7f, 0.0f);
        glBegin(GL_LINES);

        glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ());
        glVertex3f(pos.GetX() + 0.5f, pos.GetY(), pos.GetZ());

        glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ());
        glVertex3f(pos.GetX(), pos.GetY() + 0.5f, pos.GetZ());

        glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ());
        glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ() + 0.5f);
        glEnd();

        glPopMatrix();
    }

    return true;
}


/*
* CinematicRenderer::MouseEvent
*/
bool CinematicRenderer::MouseEvent(float x, float y, core::view::MouseFlags flags) {

    const bool error = true;

    // Triggered only when "TAB" is pressed

	// Show manipulator whenever a keyframe is selected. (implement in renderer)
	// If not needed anymore
	if (this->editKeyframeParam.Param<param::BoolParam>()->Value()) {
		// Do manipulator stuff

        CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
        if (ccc == NULL) return error;
        if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeper)) return error;



		// Consume mouse event
		return true;
	}
	else {
		// Do not consume mouse event
		return false;
	}
}