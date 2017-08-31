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
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::cinematiccamera;
using namespace vislib;

#define M_PI 3.1415926535897

/*
* CinematicRenderer::CinematicRenderer
*/
CinematicRenderer::CinematicRenderer(void) : Renderer3DModule(),
    slaveRendererSlot("renderer", "outgoing renderer"),
    keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper."),
    stepsParam(           "01 Spline Subdivision", "Amount of interpolation steps between keyframes"),
    toggleManipulateParam("02 Toggle Manipulator", "Toggle between position manipulation or lookup manipulation."),
    loadTimeParam(        "03 Load Time", "Load time from slave renderer")
    {

    this->interpolSteps     = 20;
    this->toggleManipulator = false;

    this->slaveRendererSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->slaveRendererSlot);

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    this->stepsParam.SetParameter(new param::IntParam((int)this->interpolSteps, 1));
    this->MakeSlotAvailable(&this->stepsParam);

    this->loadTimeParam.SetParameter(new param::ButtonParam(true));
    this->MakeSlotAvailable(&this->loadTimeParam);

    this->toggleManipulateParam.SetParameter(new param::ButtonParam('m'));
    this->MakeSlotAvailable(&this->toggleManipulateParam);


    // init variables
    this->modelViewProjMatrix.SetIdentity();
    this->viewport.SetNull();
    // Setting (constant) colors
    this->colors.Clear();
    this->colors.AssertCapacity(100);
    this->colors.Add(vislib::math::Vector<float, 3>(0.4f, 0.4f, 1.0f)); // COL_SPLINE          = 0,
    this->colors.Add(vislib::math::Vector<float, 3>(0.7f, 0.7f, 1.0f)); // COL_KEYFRAME        = 1,
    this->colors.Add(vislib::math::Vector<float, 3>(0.1f, 0.1f, 1.0f)); // COL_SELECT_KEYFRAME = 2,
    this->colors.Add(vislib::math::Vector<float, 3>(0.3f, 0.8f, 0.8f)); // COL_SELECT_LOOKAT   = 3,
    this->colors.Add(vislib::math::Vector<float, 3>(0.0f, 0.8f, 0.8f)); // COL_SELECT_UP       = 4,
    this->colors.Add(vislib::math::Vector<float, 3>(0.8f, 0.1f, 0.0f)); // COL_SELECT_X_AXIS   = 5,
    this->colors.Add(vislib::math::Vector<float, 3>(0.8f, 0.8f, 0.0f)); // COL_SELECT_Y_AXIS   = 6,
    this->colors.Add(vislib::math::Vector<float, 3>(0.1f, 0.8f, 0.0f)); // COL_SELECT_Z_AXIS   = 7


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
	if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;

	// Get bounding box of renderer.
	if (!(*oc)(1)) return false;
	*cr3d = *oc;

    // Compute bounding box including spline (in world space) and object (in world space).
    math::Cuboid<float> bboxCR3D = oc->AccessBoundingBoxes().WorldSpaceBBox();
    math::Cuboid<float> cboxCR3D = oc->AccessBoundingBoxes().WorldSpaceClipBox();

    if (ccc->getKeyframes() == NULL) {
        sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Mouse Event] Pointer to keyframe array is NULL.");
        return false;
    }
    if (ccc->getKeyframes()->Count() > 0) {
        // Get bounding box of spline.
        math::Cuboid<float> *bboxCCC = ccc->getBoundingBox();
        if (bboxCCC != NULL) {
            bboxCR3D.Union(*bboxCCC);
            cboxCR3D.Union(*bboxCCC); // use boundingbox to get new clipbox
        }
        else {
            sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Get Extents] Pointer to boundingbox array is NULL.");
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
    if (this->toggleManipulateParam.IsDirty()) {
        this->toggleManipulator = !this->toggleManipulator;
        this->toggleManipulateParam.ResetDirty();
    }

    // Updated data from cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;

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
    math::ShallowMatrix<GLfloat, 4, math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    math::ShallowMatrix<GLfloat, 4, math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);
    // Compute modelViewProjMatrix matrix
    this-> modelViewProjMatrix = projMatrix * modelViewMatrix;
    // Get current viewport
    this->viewport = cr3d->GetViewport();

    // Get pointer to keyframes array
    Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == NULL) {
        sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Render] Pointer to keyframe array is NULL.");
        return false;
    }

    // Draw cinematic renderer stuff
    if (keyframes->Count() > 0) {

        glPushMatrix();
        glDisable(GL_LIGHTING);

        glLineWidth(2.0f);
        glPointSize(15.0f);

        float        circleRadius = 0.15f;
        unsigned int circleSubDiv = 20;

        // Get the selected Keyframe
        Keyframe s = ccc->getSelectedKeyframe();

        // Get camera position
        math::Point<float, 3> camPosP = oc->GetCameraParameters()->Position();

        // Get pointer to interpolated keyframes array
        Array<math::Point<float, 3> > *interpolKeyframes = ccc->getInterpolatedCamPos();
        if (interpolKeyframes == NULL) {
            sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Render] Pointer to interpolated camera positions array is NULL.");
            return false;
        }
        // Draw spline
        math::Point<float, 3> tmpP;
        glColor3fv(this->colors[(int)colType::COL_SPLINE].PeekComponents());
        glBegin(GL_LINE_STRIP);
            for (unsigned int i = 0; i < interpolKeyframes->Count(); i++) {
                tmpP = (*interpolKeyframes)[i];
                glVertex3f(tmpP.GetX(), tmpP.GetY(), tmpP.GetZ());
            }
        glEnd();
        // Draw point for every fixed keyframe (but not slected one)
        for (unsigned int i = 0; i < keyframes->Count(); i++) {
            // Draw fixed keyframe
            tmpP = (*keyframes)[i].getCamPosition();
            if (tmpP == s.getCamPosition()) {
                this->renderCircle2D(circleRadius, circleSubDiv, camPosP, tmpP, this->colors[(int)colType::COL_SELECT_KEYFRAME]);
            }
            else {
                this->renderCircle2D(circleRadius, circleSubDiv, camPosP, tmpP, this->colors[(int)colType::COL_KEYFRAME]);
            }
        }

        // Draw manipulators only for selected keyframe whitch exists in keyframe array
        math::Point<float, 3>  sPosP = s.getCamPosition();
        math::Vector<float, 3> sUpV = s.getCamUp();
        math::Point<float, 3>  sLaP = s.getCamLookAt();
        glBegin(GL_LINES);
            // LookAt vector at keyframe position
            glColor3fv(this->colors[(int)colType::COL_SELECT_LOOKAT].PeekComponents());
            glVertex3f(sPosP.GetX(), sPosP.GetY(), sPosP.GetZ());
            glVertex3f(sLaP.GetX(), sLaP.GetY(), sLaP.GetZ());
        glEnd();
        
        int selIndex = static_cast<int>(keyframes->IndexOf(s));
        if (selIndex < 0) {
            glBegin(GL_LINES);
                // Up vector at camera position
                glColor3fv(this->colors[(int)colType::COL_SELECT_UP].PeekComponents());
                glVertex3f(sPosP.GetX(), sPosP.GetY(), sPosP.GetZ());
                glVertex3f(sPosP.GetX() + sUpV.GetX(), sPosP.GetY() + sUpV.GetY(), sPosP.GetZ() + sUpV.GetZ());
            glEnd();
            this->renderCircle2D(circleRadius/2.0f, circleSubDiv, camPosP, sPosP, this->colors[(int)colType::COL_SELECT_KEYFRAME]);
        }
        else { //(selIndex >= 0) 
            // Draw up and lookat manipulator as default
            if (this->toggleManipulator) {
                // LookAt vector at camera position for changing camera position in direction of LookAt
                math::Vector<float, 3> laV = math::Vector<float, 3>(sLaP.GetX(), sLaP.GetY(), sLaP.GetZ());
                math::Vector<float, 3> posV = math::Vector<float, 3>(sPosP.GetX(), sPosP.GetY(), sPosP.GetZ());
                laV = (posV - laV);
                laV.Normalise();
                glBegin(GL_LINES);
                // Up vector at camera position
                glColor3fv(this->colors[(int)colType::COL_SELECT_UP].PeekComponents());
                glVertex3f(sPosP.GetX(), sPosP.GetY(), sPosP.GetZ());
                glVertex3f(sPosP.GetX() + sUpV.GetX(), sPosP.GetY() + sUpV.GetY(), sPosP.GetZ() + sUpV.GetZ());

                // LookAt
                glColor3fv(this->colors[(int)colType::COL_SELECT_UP].PeekComponents());
                glVertex3f(sPosP.GetX(), sPosP.GetY(), sPosP.GetZ());
                glVertex3f(sPosP.GetX() + laV.GetX(), sPosP.GetY() + laV.GetY(), sPosP.GetZ() + laV.GetZ());

                // lookat x-axis
                glColor3fv(this->colors[(int)colType::COL_SELECT_X_AXIS].PeekComponents());
                glVertex3f(sLaP.GetX(), sLaP.GetY(), sLaP.GetZ());
                glVertex3f(sLaP.GetX() + 1.0f, sLaP.GetY(), sLaP.GetZ());

                // lookat y-axis
                glColor3fv(this->colors[(int)colType::COL_SELECT_Y_AXIS].PeekComponents());
                glVertex3f(sLaP.GetX(), sLaP.GetY(), sLaP.GetZ());
                glVertex3f(sLaP.GetX(), sLaP.GetY() + 1.0f, sLaP.GetZ());

                // lookat z-axis
                glColor3fv(this->colors[(int)colType::COL_SELECT_Z_AXIS].PeekComponents());
                glVertex3f(sLaP.GetX(), sLaP.GetY(), sLaP.GetZ());
                glVertex3f(sLaP.GetX(), sLaP.GetY(), sLaP.GetZ() + 1.0f);
                glEnd();
                this->renderCircle2D(circleRadius, circleSubDiv, camPosP, sPosP + sUpV, this->colors[(int)colType::COL_SELECT_UP]);
                this->renderCircle2D(circleRadius, circleSubDiv, camPosP, sPosP + laV, this->colors[(int)colType::COL_SELECT_UP]);
                this->renderCircle2D(circleRadius, circleSubDiv, camPosP, sLaP + math::Vector<float, 3>(1.0f, 0.0f, 0.0f), this->colors[(int)colType::COL_SELECT_X_AXIS]);
                this->renderCircle2D(circleRadius, circleSubDiv, camPosP, sLaP + math::Vector<float, 3>(0.0f, 1.0f, 0.0f), this->colors[(int)colType::COL_SELECT_Y_AXIS]);
                this->renderCircle2D(circleRadius, circleSubDiv, camPosP, sLaP + math::Vector<float, 3>(0.0f, 0.0f, 1.0f), this->colors[(int)colType::COL_SELECT_Z_AXIS]);
            }
            else { // Draw axis for position manipulation
                glBegin(GL_LINES);
                // keyframe pos x-axis
                glColor3fv(this->colors[(int)colType::COL_SELECT_X_AXIS].PeekComponents());
                glVertex3f(sPosP.GetX(), sPosP.GetY(), sPosP.GetZ());
                glVertex3f(sPosP.GetX() + 1.0f, sPosP.GetY(), sPosP.GetZ());
                // keyframe pos y-axis
                glColor3fv(this->colors[(int)colType::COL_SELECT_Y_AXIS].PeekComponents());
                glVertex3f(sPosP.GetX(), sPosP.GetY(), sPosP.GetZ());
                glVertex3f(sPosP.GetX(), sPosP.GetY() + 1.0f, sPosP.GetZ());

                // keyframe  pos z-axis
                glColor3fv(this->colors[(int)colType::COL_SELECT_Z_AXIS].PeekComponents());
                glVertex3f(sPosP.GetX(), sPosP.GetY(), sPosP.GetZ());
                glVertex3f(sPosP.GetX(), sPosP.GetY(), sPosP.GetZ() + 1.0f);
                glEnd();
                this->renderCircle2D(circleRadius, circleSubDiv, camPosP, sPosP + math::Vector<float, 3>(1.0f, 0.0f, 0.0f), this->colors[(int)colType::COL_SELECT_X_AXIS]);
                this->renderCircle2D(circleRadius, circleSubDiv, camPosP, sPosP + math::Vector<float, 3>(0.0f, 1.0f, 0.0f), this->colors[(int)colType::COL_SELECT_Y_AXIS]);
                this->renderCircle2D(circleRadius, circleSubDiv, camPosP, sPosP + math::Vector<float, 3>(0.0f, 0.0f, 1.0f), this->colors[(int)colType::COL_SELECT_Z_AXIS]);
            }
            ///////////////////////////////////////////
            //// ADD CODE FOR NEW MAIPULATOR HERE /////
            ///////////////////////////////////////////
        }

        glPopMatrix();
    }
    return true;
}


/*
* CinematicRenderer::MouseEvent
*/
bool CinematicRenderer::MouseEvent(float x, float y, core::view::MouseFlags flags) {

    // !!! Triggered only when "TAB" is pressed => triggers parameter 'enableMouseSelection' in View3D

    bool consume = false;

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (ccc == NULL) return false;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;


    // on leftclick
    if ((flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) && (flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED)) {

        // Get pointer to keyframes
        Array<Keyframe> *keyframes = ccc->getKeyframes();
        if (keyframes == NULL) {
            sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Mouse Event] Pointer to keyframe array is NULL.");
            return false;
        }

        // Process keyframe selection
        for (unsigned int i = 0; i < keyframes->Count(); i++) {
            // Current camera position
            math::Point<float, 3>  cPosP = (*keyframes)[i].getCamPosition();
            // Check if position of current keyframe is hit by mouse
            if (this->processPointHit(x, y, cPosP, cPosP, manipulatorType::NONE)) {
                ccc->setSelectedKeyframeTime((*keyframes)[i].getTime());
                if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return false;
                return true; // no further checking has to be done (?)
            }
        }

        // Process manipulator selection 
        Keyframe               s    = ccc->getSelectedKeyframe();
        math::Point<float, 3>  posP = s.getCamPosition();
        math::Vector<float, 3> upV  = s.getCamUp();
        math::Point<float, 3>  laP  = s.getCamLookAt();

        if (this->toggleManipulator) {
            // up and lookat
            math::Point<float, 3>  uManiP   = posP + upV;
            math::Point<float, 3>  xLaManiP = math::Point<float, 3>(laP.GetX() + 1.0f, laP.GetY(), laP.GetZ());
            math::Point<float, 3>  yLaManiP = math::Point<float, 3>(laP.GetX(), laP.GetY() + 1.0f, laP.GetZ());
            math::Point<float, 3>  zLaManiP = math::Point<float, 3>(laP.GetX(), laP.GetY(), laP.GetZ() + 1.0f);
            math::Vector<float, 3> laV      = math::Vector<float, 3>(laP.GetX(), laP.GetY(), laP.GetZ());
            math::Vector<float, 3> posV     = math::Vector<float, 3>(posP.GetX(), posP.GetY(), posP.GetZ());
            laV = (posV - laV);
            laV.Normalise();
            math::Point<float, 3> laManiP = posP + laV;

            if (this->processPointHit(x, y, posP, uManiP, manipulatorType::CAM_UP)) { 
                consume = true;
            }
            else if (this->processPointHit(x, y, posP, laManiP, manipulatorType::CAM_POS)) {
                consume = true;
            }
            else if (this->processPointHit(x, y, laP, xLaManiP, manipulatorType::LOOKAT_X_AXIS)) {
                consume = true;
            }
            else if (this->processPointHit(x, y, laP, yLaManiP, manipulatorType::LOOKAT_Y_AXIS)) {
                consume = true;
            }
            else if (this->processPointHit(x, y, laP, zLaManiP, manipulatorType::LOOKAT_Z_AXIS)) {
                consume = true;
            }
        }
        else {
            // keyframe camera pos
            math::Point<float, 3> xCamManiP = math::Point<float, 3>(posP.GetX() + 1.0f, posP.GetY(), posP.GetZ());
            math::Point<float, 3> yCamManiP = math::Point<float, 3>(posP.GetX(), posP.GetY() + 1.0f, posP.GetZ());
            math::Point<float, 3> zCamManiP = math::Point<float, 3>(posP.GetX(), posP.GetY(), posP.GetZ() + 1.0f);
            if (this->processPointHit(x, y, posP, xCamManiP, manipulatorType::CAM_X_AXIS)) {
                consume = true;
            }
            else if (this->processPointHit(x, y, posP, yCamManiP, manipulatorType::CAM_Y_AXIS)) {
                consume = true;
            }
            else if (this->processPointHit(x, y, posP, zCamManiP, manipulatorType::CAM_Z_AXIS)) {
                consume = true;
            }
        }
        ///////////////////////////////////////////
        //// ADD CODE FOR NEW MAIPULATOR HERE /////
        ///////////////////////////////////////////

    }
    else if ((flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) && !(flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED)) {

        if (this->currentManipulator.active) {

            // Relationship between mouse movement and length changes of coordinates
            float sensitivity = 0.01f;

            float lineDiff  = 0.0f;
            // Select manipulator axis with greatest contribution
            if (math::Abs(this->currentManipulator.ssDiffX) > math::Abs(this->currentManipulator.ssDiffY)) {
                lineDiff = (x - this->currentManipulator.lastMouseX) * sensitivity;
                if (this->currentManipulator.ssDiffX < 0.0f) { // Adjust line changes depending on manipulator axis direction 
                    lineDiff *= -1.0f;
                }
            }
            else {
                lineDiff = (y - this->currentManipulator.lastMouseY) * sensitivity;
                if (this->currentManipulator.ssDiffY < 0.0f) { // Adjust line changes depending on manipulator axis direction 
                    lineDiff *= -1.0f;
                }
            }

            // Apply changes
            Keyframe               s    = ccc->getSelectedKeyframe();
            math::Point<float, 3>  posP = s.getCamPosition();
            math::Point<float, 3>  laP  = s.getCamLookAt();
            math::Vector<float, 3> upV  = math::Vector<float, 3>(s.getCamUp().GetX(), s.getCamUp().GetY(), s.getCamUp().GetZ());
            math::Vector<float, 3> posV = math::Vector<float, 3>(posP.GetX(), posP.GetY(), posP.GetZ());
            math::Vector<float, 3> laV  = math::Vector<float, 3>(laP.GetX(), laP.GetY(), laP.GetZ());
            if (this->currentManipulator.type == manipulatorType::CAM_X_AXIS) {
                posP.SetX(posP.GetX() + lineDiff);
                ccc->setSelectedKeyframePosition(posP);
            }
            else if (this->currentManipulator.type == manipulatorType::CAM_Y_AXIS) {
                posP.SetY(posP.GetY() + lineDiff);
                ccc->setSelectedKeyframePosition(posP);
            }
            else if (this->currentManipulator.type == manipulatorType::CAM_Z_AXIS) {
                posP.SetZ(posP.GetZ() + lineDiff);
                ccc->setSelectedKeyframePosition(posP);
            }
            else if (this->currentManipulator.type == manipulatorType::CAM_UP) {

                // change rotation direction depending on whether lookat position is nearer or not
                view::CallRender3D *oc = this->slaveRendererSlot.CallAs<CallRender3D>();
                if (oc == NULL) return false;
                math::Point<float, 3>  camPosP = oc->GetCameraParameters()->Position();
                math::Vector<float, 3> camPosV = math::Vector<float, 3>(camPosP.GetX(), camPosP.GetY(), camPosP.GetZ());
                bool cwRot = ((camPosV - laV).Norm() > (camPosV - posV).Norm()); // true if keyframe position is nearer to camera position than lookat position of keyframe camera
                sensitivity /= 2.0f;

                // Select manipulator axis with smallest contribution
                if (math::Abs(this->currentManipulator.ssDiffX) < math::Abs(this->currentManipulator.ssDiffY)) {
                    if ((cwRot && (this->currentManipulator.ssDiffY < 0.0f)) ||
                        (!cwRot && (this->currentManipulator.ssDiffY > 0.0f))) { 
                        lineDiff = (x - this->currentManipulator.lastMouseX) * sensitivity;
                    }
                    else {
                        lineDiff = (this->currentManipulator.lastMouseX - x) * sensitivity;
                    }
                }
                else {
                    if ((cwRot && (this->currentManipulator.ssDiffX > 0.0f)) ||
                        (!cwRot && (this->currentManipulator.ssDiffX < 0.0f))) {
                        lineDiff = (y - this->currentManipulator.lastMouseY) * sensitivity;
                    }
                    else {
                        lineDiff = (this->currentManipulator.lastMouseY - y) * sensitivity; 
                    }
                }

                /* rotate up vector aroung lookat vector with the "Rodrigues' rotation formula" */
                float                  t = lineDiff;    // => theta angle                  
                math::Vector<float, 3> k = laV;         // => rotation axis = camera lookat
                math::Vector<float, 3> v = upV;         // => vector to rotate       
                upV = v * cos(t) + k.Cross(v) * sin(t) + k * (k.Dot(v)) * (1.0f - cos(t));
                ccc->setSelectedKeyframeUp(upV);
            }
            else if (this->currentManipulator.type == manipulatorType::CAM_POS) {
                laV = (posV - laV);
                laV.ScaleToLength(lineDiff);
                posP = posP + laV;
                ccc->setSelectedKeyframePosition(posP);
            }
            else if (this->currentManipulator.type == manipulatorType::LOOKAT_X_AXIS) {
                laP.SetX(laP.GetX() + lineDiff);
                ccc->setSelectedKeyframeLookAt(laP);
            }
            else if (this->currentManipulator.type == manipulatorType::LOOKAT_Y_AXIS) {
                laP.SetY(laP.GetY() + lineDiff);
                ccc->setSelectedKeyframeLookAt(laP);
            }
            else if (this->currentManipulator.type == manipulatorType::LOOKAT_Z_AXIS) {
                laP.SetZ(laP.GetZ() + lineDiff);
                ccc->setSelectedKeyframeLookAt(laP);
            }
            ///////////////////////////////////////////
            //// ADD CODE FOR NEW MAIPULATOR HERE /////
            ///////////////////////////////////////////


            if (!(*ccc)(CallCinematicCamera::CallForManipulateSelectedKeyframe)) return false;

            this->currentManipulator.lastMouseX = x;
            this->currentManipulator.lastMouseY = y;
            consume = true;
        }
    }

    return consume;
}


/*
* CinematicRenderer::checkPointHit
*/
bool CinematicRenderer::processPointHit(float x, float y, vislib::math::Point<float, 3> camPos, vislib::math::Point<float, 3> manipPos, manipulatorType t) {

    // (Rectangular) Offset around point within point is hit
    float selectOffset = 15.0f;

    // Transform manipulator position from world space to screen space
    // World space position
    math::Vector<float, 4> wsPosV = math::Vector<float, 4>(manipPos.GetX(), manipPos.GetY(), manipPos.GetZ(), 1.0f);
    // Screen space position
    math::Vector<float, 4> manSSPosV = this->modelViewProjMatrix * wsPosV;
    // Division by 'w'
    manSSPosV = manSSPosV / manSSPosV.GetW();
    // Transform to viewport coordinates (x,y in [-1,1] -> viewport size)
    float manX = (manSSPosV.GetX() + 1.0f) / 2.0f * this->viewport.GetSize().GetWidth();
    float manY = math::Abs(manSSPosV.GetY() - 1.0f) / 2.0f * this->viewport.GetSize().GetHeight(); // flip y-axis

    // Clear previous manipulator selection
    this->currentManipulator.active = false;

    // Check if mouse position lies within offset quad around manipulator position
    if (((manX < x + selectOffset) && (manX > x - selectOffset)) &&
        ((manY < y + selectOffset) && (manY > y - selectOffset))) {

        // Transform camera position from world space to screen space
        // World space position
        wsPosV = math::Vector<float, 4>(camPos.GetX(), camPos.GetY(), camPos.GetZ(), 1.0f);
        // Screen space position
        math::Vector<float, 4> camSSPosV = this->modelViewProjMatrix * wsPosV;
        // Division by 'w'
        camSSPosV = camSSPosV / camSSPosV.GetW();
        // Transform to viewport coordinates (x,y in [-1,1] -> viewport size)
        float camX = (camSSPosV.GetX() + 1.0f) / 2.0f * this->viewport.GetSize().GetWidth();
        float camY = math::Abs(camSSPosV.GetY() - 1.0f) / 2.0f * this->viewport.GetSize().GetHeight(); // flip y-axis

        // Set current manipulator values
        if (t != manipulatorType::NONE) { // camera position marker is no manipulator
            this->currentManipulator.active = true;
            this->currentManipulator.type = t;
            this->currentManipulator.lastMouseX = x;
            this->currentManipulator.lastMouseY = y;
            this->currentManipulator.ssDiffX = manX - camX;
            this->currentManipulator.ssDiffY = manY - camY;
        }
        return true;
    }
    return false;
}


/*
* CinematicRenderer::renderCircle2D
*/
void CinematicRenderer::renderCircle2D(float radius, unsigned int subdiv, vislib::math::Point<float, 3> camPos, vislib::math::Point<float, 3> centerPos, vislib::math::Vector<float, 3> col) {

    vislib::math::Vector<float, 3> normalV = vislib::math::Vector<float, 3>(centerPos.GetX() - camPos.GetX(), centerPos.GetY() - camPos.GetY(), centerPos.GetZ() - camPos.GetZ());
    normalV.Normalise();
    // Get arbitary vector vertical to normalVec
    vislib::math::Vector<float, 3> rotV = vislib::math::Vector<float, 3>(normalV.GetZ(), 0.0f, -(normalV.GetX()));
    rotV.ScaleToLength(radius);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_CULL_FACE);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslatef(centerPos.GetX(), centerPos.GetY(), centerPos.GetZ());

    /* rotate up vector aroung lookat vector with the "Rodrigues' rotation formula" */              
    math::Vector<float, 3> v;                             // vector to rotate   
    math::Vector<float, 3> k = normalV;                 // rotation axis = camera lookat
    float                  t = 2.0f*(float)(M_PI) / (float)subdiv; // theta angle for rotation   

    glColor3fv(col.PeekComponents());
    glBegin(GL_TRIANGLE_FAN);
        glVertex3f(0.0f, 0.0f, 0.0f);
        for (unsigned int i = 0; i <= subdiv; i++) {
            glVertex3fv(rotV.PeekComponents());
            v = rotV;    
            rotV = v * cos(t) + k.Cross(v) * sin(t) + k * (k.Dot(v)) * (1.0f - cos(t));
            rotV.ScaleToLength(radius);
        }
    glEnd();

    glPopMatrix();
}