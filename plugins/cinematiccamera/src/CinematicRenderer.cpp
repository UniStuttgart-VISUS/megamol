/*
* CinematicRenderer.cpp
*
* Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

// TODO: draw labels to axis


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
    stepsParam(        "01 Spline Subdivision", "amount of interpolation steps between keyframes"),
    loadTimeParam(     "02 Load Time", "load time from slave renderer")
    {

    this->slaveRendererSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->slaveRendererSlot);

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    this->stepsParam.SetParameter(new param::IntParam(20));
    this->MakeSlotAvailable(&this->stepsParam);

    this->loadTimeParam.SetParameter(new param::ButtonParam('t'));
    this->MakeSlotAvailable(&this->loadTimeParam);

    // Load total time from animation at startup
    this->loadTimeParam.ForceSetDirty();
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
	if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeperData)) return false;

	// Get bounding box of renderer.
	if (!(*oc)(1)) return false;
	*cr3d = *oc;

    // Compute bounding box including spline (in world space) and object (in world space).
    vislib::math::Cuboid<float> bboxCR3D = oc->AccessBoundingBoxes().WorldSpaceBBox();
    vislib::math::Cuboid<float> cboxCR3D = oc->AccessBoundingBoxes().WorldSpaceClipBox();

    if (ccc->getKeyframes()->Count() > 0) {
        // Get bounding box of spline.
        vislib::math::Cuboid<float> *bboxCCC = ccc->getBoundingBox();
        bboxCR3D.Union(*bboxCCC);
        cboxCR3D.Union(*bboxCCC); // use boundingbox to get new clipbox
    }

	cr3d->AccessBoundingBoxes().SetWorldSpaceBBox(bboxCR3D);
    cr3d->AccessBoundingBoxes().SetWorldSpaceClipBox(cboxCR3D);

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
	// Update data in cinematic camera call
	if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeperData)) return false;

    // Update parameter
    if (this->loadTimeParam.IsDirty()) {
        float tt = static_cast<float>(oc->TimeFramesCount());
        ccc->setTotalTime(tt);
		if (!(*ccc)(CallCinematicCamera::CallForSetTotalTime)) return false;
        this->loadTimeParam.ResetDirty();
    }

    // ...
    *oc = *cr3d;
    oc->SetTime(cr3d->Time());

    // Get pointer to keyframes
    vislib::Array<Keyframe> *keyframes = ccc->getKeyframes();

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
        glColor3f(0.0f, 0.0f, 1.0f);
        glBegin(GL_LINE_STRIP);
        vislib::math::Point<float, 3> p;
        for (unsigned int i = 0; i < keyframes->Count()-1; i++) {
            int   steps         = this->stepsParam.Param<param::IntParam>()->Value();
            float startTime     = (*keyframes)[i].getTime();
            float deltaTimeStep = ((*keyframes)[i + 1].getTime() - startTime) / (float)steps;

            // Draw fixed keyframe
            p = (*keyframes)[i].getCamPosition();
            glVertex3f(p.GetX(), p.GetY(), p.GetZ());

            // Draw interpolated keyframes
            for (int m = 1; m < steps; m++) {
                ccc->setInterpolatedKeyframeTime(startTime + deltaTimeStep*(float)m);
                if (!(*ccc)(CallCinematicCamera::CallForRequestInterpolatedKeyframe)) return false;
				p = ccc->getInterpolatedKeyframe().getCamPosition();
                glVertex3f(p.GetX(), p.GetY(), p.GetZ());
            }
        }
        // Draw last fixed keyframe
        p = (*keyframes)[keyframes->Count() - 1].getCamPosition();
        glVertex3f(p.GetX(), p.GetY(), p.GetZ());
        glEnd();

        // draw the selected camera marker
		Keyframe s = ccc->getSelectedKeyframe();
        vislib::math::Point<float, 3> pos     = s.getCamPosition();
        vislib::math::Vector<float, 3> up     = s.getCamUp();
        vislib::math::Point<float, 3> lookat  = s.getCamLookAt();
        up.ScaleToLength(0.5f);

        glLineWidth(1.5f);
        glBegin(GL_LINES);

        // Up vector at camera position
		glColor3f(1.0f, 0.0f, 1.0f);
        glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ());
        glVertex3f(pos.GetX()+ up.GetX(), pos.GetY() + up.GetY(), pos.GetZ() + up.GetZ());
        // LookAt vector at camera position
		glColor3f(0.0f, 1.0f, 1.0f);
        glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ());
        glVertex3f(lookat.GetX(), lookat.GetY(), lookat.GetZ());
        // X-axis
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ());
        glVertex3f(pos.GetX()+0.5f, pos.GetY(), pos.GetZ());
        // Y-axis
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ());
        glVertex3f(pos.GetX(), pos.GetY()+0.5f, pos.GetZ());
        // Z-axis
        glColor3f(1.0f, 1.0f, 0.0f);
        glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ());
        glVertex3f(pos.GetX(), pos.GetY(), pos.GetZ()+0.5f);

        glEnd();

        glPopMatrix();
    }

    return true;
}


/*
* CinematicRenderer::MouseEvent
*/
bool CinematicRenderer::MouseEvent(float x, float y, core::view::MouseFlags flags) {

    /*
    const bool error = true;

    // Triggered only when "TAB" is pressed

	// Show manipulator whenever a keyframe is selected. (implement in renderer)
	// If not needed anymore
	if (this->editKeyframeParam.Param<param::BoolParam>()->Value()) {
		// Do manipulator stuff

        CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
        if (ccc == NULL) return error;
        if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeperData)) return error;



		// Consume mouse event
		return true;
	}
	else {
		// Do not consume mouse event
		return false;
	}

    */

    return false;
}