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

    this->interpolSteps = 20;

    this->slaveRendererSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->slaveRendererSlot);

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    this->stepsParam.SetParameter(new param::IntParam((int)this->interpolSteps, 1));
    this->MakeSlotAvailable(&this->stepsParam);

    this->loadTimeParam.SetParameter(new param::ButtonParam());
    this->MakeSlotAvailable(&this->loadTimeParam);

    // init variables
    this->modelViewProjMatrix.SetIdentity();
    //this->viewportStuff = ;

    // Load total time from animation at startup
    this->loadTimeParam.ForceSetDirty();
    // Load spline interpolation keyframes at startup
    this->stepsParam.ForceSetDirty();

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
    vislib::math::Cuboid<float> bboxCR3D = oc->AccessBoundingBoxes().WorldSpaceBBox();
    vislib::math::Cuboid<float> cboxCR3D = oc->AccessBoundingBoxes().WorldSpaceClipBox();

    if (ccc->getKeyframes() == NULL) {
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Mouse Event] Pointer to keyframe array is NULL.");
        return false;
    }
    if (ccc->getKeyframes()->Count() > 0) {
        // Get bounding box of spline.
        vislib::math::Cuboid<float> *bboxCCC = ccc->getBoundingBox();
        if (bboxCCC != NULL) {
            bboxCR3D.Union(*bboxCCC);
            cboxCR3D.Union(*bboxCCC); // use boundingbox to get new clipbox
        }
        else {
            vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Get Extents] Pointer to boundingbox array is NULL.");
            return false;
        }
    }

    // Apply new boundingbox 
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
    if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeper)) return false;

    // Update parameter
    if (this->loadTimeParam.IsDirty()) {
        float tt = static_cast<float>(oc->TimeFramesCount());
        ccc->setTotalTime(tt);
        if (!(*ccc)(CallCinematicCamera::CallForSetTotalTime)) return false;
        this->loadTimeParam.ResetDirty();
    }
    if (this->stepsParam.IsDirty()) {
        this->interpolSteps = this->stepsParam.Param<param::IntParam>()->Value();
        ccc->setInterpolationSteps(this->interpolSteps);
        if (!(*ccc)(CallCinematicCamera::CallForInterpolatedCamPos)) return false;
        this->stepsParam.ResetDirty();
    }

    // ...
    *oc = *cr3d;

    // Set animation time based on selected keyframe ('disables' animation via view3d)
    Keyframe s = ccc->getSelectedKeyframe();
    oc->SetTime(s.getTime());

    // Call original renderer.
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    (*oc)(0);
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // Get current Model-View-Projection matrix  for world space to screen space projection of keyframe camera position for mouse selection
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);
    // Compute modelViewProjMatrix matrix
    this-> modelViewProjMatrix = projMatrix * modelViewMatrix;
    // Get current viewport
    glGetFloatv(GL_VIEWPORT, this->viewportStuff);

    // Get pointer to keyframes array
    vislib::Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == NULL) {
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Render] Pointer to keyframe array is NULL.");
        return false;
    }

    // Draw cinematic renderer stuff
    if (keyframes->Count() > 0) {

        glPushMatrix();
        glDisable(GL_LIGHTING);

        // Get pointer to interpolated keyframes array
        vislib::Array<vislib::math::Point<float, 3> > *interpolKeyframes = ccc->getInterpolatedCamPos();
        if (interpolKeyframes == NULL) {
            vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Render] Pointer to interpolated camera positions array is NULL.");
            return false;
        }

        // draw spline
        vislib::math::Point<float, 3> p;
        glLineWidth(2.0f);
        glColor3f(0.0f, 0.0f, 1.0f);
        glBegin(GL_LINE_STRIP);
        for (unsigned int i = 0; i < interpolKeyframes->Count(); i++) {
            p = (*interpolKeyframes)[i];
            glVertex3f(p.GetX(), p.GetY(), p.GetZ());
        }
        glEnd();

        // Draw point for every fixed keyframe
        glPointSize(15.0f);
        glColor3f(0.3f, 0.3f, 1.0f);
        glBegin(GL_POINTS);
        for (unsigned int i = 0; i < keyframes->Count(); i++) {
            // Draw fixed keyframe
            p = (*keyframes)[i].getCamPosition();
            glVertex3f(p.GetX(), p.GetY(), p.GetZ());
        }
        glEnd();

        // draw the selected camera marker
		Keyframe s = ccc->getSelectedKeyframe();
        vislib::math::Point<float, 3>  pos    = s.getCamPosition();
        vislib::math::Vector<float, 3> up     = s.getCamUp();
        vislib::math::Point<float, 3>  lookat = s.getCamLookAt();
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

        glEnd();

        glPopMatrix();
    }

    return true;
}


/*
* CinematicRenderer::MouseEvent
*/
bool CinematicRenderer::MouseEvent(float x, float y, core::view::MouseFlags flags) {

    // !!! Triggered only when "TAB" is pressed => triggers parameter 'enableMouseSelection' in View3D

    // on leftclick
    if (flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) {

        CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
        if (ccc == NULL) return false;
        // Update data in cinematic camera call
        if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeper)) return false;

        // Get pointer to keyframes
        vislib::Array<Keyframe> *keyframes = ccc->getKeyframes();
        if (keyframes == NULL) {
            vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Mouse Event] Pointer to keyframe array is NULL.");
            return false;
        }

        float selectOffset = 10.0f;

        // Process keyframe selection
        for (unsigned int i = 0; i < keyframes->Count(); i++) {

            // Current camera position
            vislib::math::Point<GLfloat, 3>  cPos = (*keyframes)[i].getCamPosition();
            // World space position
            vislib::math::Vector<GLfloat, 4> wsPos = vislib::math::Vector<GLfloat, 4>(cPos.GetX(), cPos.GetY(), cPos.GetZ(), 1.0f);
            // Screen space position
            vislib::math::Vector<GLfloat, 4> ssPos = modelViewProjMatrix * wsPos;
            // Division by 'w'
            ssPos = ssPos / ssPos.GetW();
            // Viewport coordinates (x,y in [-1,1])
            ssPos.SetX((ssPos.GetX() + 1.0f) / 2.0f * this->viewportStuff[2]);
            ssPos.SetY(std::fabsf(ssPos.GetY() - 1.0f) / 2.0f * this->viewportStuff[3]); // flip y-axis

            //vislib::sys::Log::DefaultLog.WriteInfo("### MOUSE X: %f - Y: %f  | KEYFRAME: X: %f - Y: %f ", x, y, ssPos.GetX(), ssPos.GetY());

            if (((ssPos.GetX() < x+selectOffset) && (ssPos.GetX() > x-selectOffset)) && 
                ((ssPos.GetY() < y+selectOffset) && (ssPos.GetY() > y-selectOffset))) {
                ccc->setSelectedKeyframeTime((*keyframes)[i].getTime());
                if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return false;
            }
        }
    }

    return true;
}