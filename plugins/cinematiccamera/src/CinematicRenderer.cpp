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
    this->toggleManipulator = true;

    this->slaveRendererSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->slaveRendererSlot);

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    this->stepsParam.SetParameter(new param::IntParam((int)this->interpolSteps, 1));
    this->MakeSlotAvailable(&this->stepsParam);

    this->loadTimeParam.SetParameter(new param::ButtonParam());
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
        float        circleRadius = 0.175f;
        unsigned int circleSubDiv = 20;

        // Get the selected Keyframe
        Keyframe s = ccc->getSelectedKeyframe();

        // Get camera position
        math::Point<float, 3> camPos = oc->GetCameraParameters()->Position();

        // Get pointer to interpolated keyframes array
        Array<math::Point<float, 3> > *interpolKeyframes = ccc->getInterpolatedCamPos();
        if (interpolKeyframes == NULL) {
            sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Render] Pointer to interpolated camera positions array is NULL.");
            return false;
        }

        // Draw spline
        math::Point<float, 3> p;
        glColor3fv(this->colors[(int)colType::COL_SPLINE].PeekComponents());
        glBegin(GL_LINE_STRIP);
            for (unsigned int i = 0; i < interpolKeyframes->Count(); i++) {
                p = (*interpolKeyframes)[i];
                glVertex3f(p.GetX(), p.GetY(), p.GetZ());
            }
        glEnd();
        // Draw point for every fixed keyframe (but not slected one)
        for (unsigned int i = 0; i < keyframes->Count(); i++) {
            // Draw fixed keyframe
            p = (*keyframes)[i].getCamPosition();
            glColor3fv(this->colors[(int)colType::COL_KEYFRAME].PeekComponents());
            if (p == s.getCamPosition()) {
                glColor3fv(this->colors[(int)colType::COL_SELECT_KEYFRAME].PeekComponents());
            }
            this->renderCircle2D(circleRadius, circleSubDiv, camPos, p);
        }

        // Draw manipulators for selected keyframe
        math::Point<float, 3>  sPos    = s.getCamPosition();
        math::Vector<float, 3> sUp     = s.getCamUp();
        math::Point<float, 3>  sLookat = s.getCamLookAt();
        StringA tmpStr;
        float strWidth;
        float fontSize = 0.5f;

        // LookAt vector at camera position
        glColor3fv(this->colors[(int)colType::COL_SELECT_LOOKAT].PeekComponents());
        glBegin(GL_LINES);
            glVertex3f(sPos.GetX(), sPos.GetY(), sPos.GetZ());
            glVertex3f(sLookat.GetX(), sLookat.GetY(), sLookat.GetZ());
        glEnd();
        // Draw up manipulator as default
        if (this->toggleManipulator) {
            glColor3fv(this->colors[(int)colType::COL_SELECT_UP].PeekComponents());
            glBegin(GL_LINES);
                // Up vector at camera position
                glVertex3f(sPos.GetX(), sPos.GetY(), sPos.GetZ());
                glVertex3f(sPos.GetX() + sUp.GetX(), sPos.GetY() + sUp.GetY(), sPos.GetZ() + sUp.GetZ());
            glEnd();
            this->renderCircle2D(circleRadius, circleSubDiv, camPos, sPos + sUp);
            
        }
        else { // Draw axis for position manipulation
            // x-axis
            glColor3fv(this->colors[(int)colType::COL_SELECT_X_AXIS].PeekComponents());
            glBegin(GL_LINES);
                glVertex3f(sPos.GetX(), sPos.GetY(), sPos.GetZ());
                glVertex3f(sPos.GetX() + 1.0f, sPos.GetY(), sPos.GetZ());
            glEnd();
            this->renderCircle2D(circleRadius, circleSubDiv, camPos, sPos + math::Vector<float, 3>(1.0f, 0.0f, 0.0f));

            // y-axis
            glColor3fv(this->colors[(int)colType::COL_SELECT_Y_AXIS].PeekComponents());
            glBegin(GL_LINES);
                glVertex3f(sPos.GetX(), sPos.GetY(), sPos.GetZ());
                glVertex3f(sPos.GetX(), sPos.GetY() + 1.0f, sPos.GetZ());
            glEnd();
            this->renderCircle2D(circleRadius, circleSubDiv, camPos, sPos + math::Vector<float, 3>(0.0f, 1.0f, 0.0f));

            // z-axis
            glColor3fv(this->colors[(int)colType::COL_SELECT_Z_AXIS].PeekComponents());
            glBegin(GL_LINES);
                glVertex3f(sPos.GetX(), sPos.GetY(), sPos.GetZ());
                glVertex3f(sPos.GetX(), sPos.GetY(), sPos.GetZ() + 1.0f);
            glEnd();
            this->renderCircle2D(circleRadius, circleSubDiv, camPos, sPos + math::Vector<float, 3>(0.0f, 0.0f, 1.0f));
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
            math::Point<float, 3>  cPos = (*keyframes)[i].getCamPosition();
            // Check if position of current keyframe is hit by mouse
            if (this->processPointHit(x, y, cPos, cPos, manipulatorType::CAMPOS)) {
                ccc->setSelectedKeyframeTime((*keyframes)[i].getTime());
                if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return false;
                return true; // no further checking has to be done (?)
            }
        }

        // Get the selected Keyframe
        Keyframe              s = ccc->getSelectedKeyframe();
        math::Point<float, 3> pos = s.getCamPosition();
        math::Point<float, 3> up = math::Point<float, 3>(s.getCamUp().GetX(), s.getCamUp().GetY(), s.getCamUp().GetZ());
        // Process manipulator selection 
        if (this->toggleManipulator) {
            math::Point<float, 3> uManip = math::Point<float, 3>(pos.GetX() + up.GetX(), pos.GetY() + up.GetY(), pos.GetZ() + up.GetZ());
            consume = this->processPointHit(x, y, pos, uManip, manipulatorType::CAMUP);
        }
        else {
            math::Point<float, 3> xManip = math::Point<float, 3>(pos.GetX() + 1.0f, pos.GetY(), pos.GetZ());
            math::Point<float, 3> yManip = math::Point<float, 3>(pos.GetX(), pos.GetY() + 1.0f, pos.GetZ());
            math::Point<float, 3> zManip = math::Point<float, 3>(pos.GetX(), pos.GetY(), pos.GetZ() + 1.0f);
            if (this->processPointHit(x, y, pos, xManip, manipulatorType::XAXIS)) {
                consume = true;
            }
            else if (this->processPointHit(x, y, pos, yManip, manipulatorType::YAXIS)) {
                consume = true;
            }
            else if (this->processPointHit(x, y, pos, zManip, manipulatorType::ZAXIS)) {
                consume = true;
            }
        }
    }
    else if ((flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) && !(flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED)) {

        if (this->currentManipulator.active) {
            // Get the selected Keyframe
            Keyframe               s      = ccc->getSelectedKeyframe();
            math::Point<float, 3>  pos    = s.getCamPosition();
            math::Vector<float, 3> lookat = math::Vector<float, 3>(s.getCamLookAt().GetX(), s.getCamLookAt().GetY(), s.getCamLookAt().GetZ());
            math::Vector<float, 3> up     = math::Vector<float, 3>(s.getCamUp().GetX(), s.getCamUp().GetY(), s.getCamUp().GetZ());

            // Relationship between mouse movement and length changes of coordinates
            const float sensitivity = 0.01f;

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
            if (this->currentManipulator.type == manipulatorType::XAXIS) {
                pos.SetX(pos.GetX() + lineDiff);
            }
            else if (this->currentManipulator.type == manipulatorType::YAXIS) {
                pos.SetY(pos.GetY() + lineDiff);
            }
            else if (this->currentManipulator.type == manipulatorType::ZAXIS) {
                pos.SetZ(pos.GetZ() + lineDiff);
            }
            else if (this->currentManipulator.type == manipulatorType::CAMUP) {
                /* rotate up vector aroung lookat vector with the "Rodrigues' rotation formula" */
                float                  t = lineDiff;   // => theta angle                  
                math::Vector<float, 3> k = lookat;     // => rotation axis = camera lookat
                math::Vector<float, 3> v = up;         // => vector to rotate       
                up = v * cos(t) + k.Cross(v) * sin(t) + k * (k.Dot(v)) * (1.0f - cos(t));
            }
            ccc->setSelectedKeyframePosition(pos);
            ccc->setSelectedKeyframeUp(up);
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
    math::Vector<float, 4> wsPos = math::Vector<float, 4>(manipPos.GetX(), manipPos.GetY(), manipPos.GetZ(), 1.0f);
    // Screen space position
    math::Vector<float, 4> manSSPos = this->modelViewProjMatrix * wsPos;
    // Division by 'w'
    manSSPos = manSSPos / manSSPos.GetW();
    // Transform to viewport coordinates (x,y in [-1,1])
    float manX = (manSSPos.GetX() + 1.0f) / 2.0f * this->viewport.GetSize().GetWidth();
    float manY = math::Abs(manSSPos.GetY() - 1.0f) / 2.0f * this->viewport.GetSize().GetHeight(); // flip y-axis

    // Clear previous manipulator selection
    this->currentManipulator.active = false;

    // Check if mouse position lies within offset quad around manipulator position
    if (((manX < x + selectOffset) && (manX > x - selectOffset)) &&
        ((manY < y + selectOffset) && (manY > y - selectOffset))) {

        // Transform camera position from world space to screen space
        // World space position
        wsPos = math::Vector<float, 4>(camPos.GetX(), camPos.GetY(), camPos.GetZ(), 1.0f);
        // Screen space position
        math::Vector<float, 4> camSSPos = this->modelViewProjMatrix * wsPos;
        // Division by 'w'
        camSSPos = camSSPos / camSSPos.GetW();
        // Transform to viewport coordinates (x,y in [-1,1])
        float camX = (camSSPos.GetX() + 1.0f) / 2.0f * this->viewport.GetSize().GetWidth();
        float camY = math::Abs(camSSPos.GetY() - 1.0f) / 2.0f * this->viewport.GetSize().GetHeight(); // flip y-axis

        // Set current manipulator values
        if (t != manipulatorType::CAMPOS) { // camera position is no manipulator
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
void CinematicRenderer::renderCircle2D(float radius, unsigned int subdiv, vislib::math::Point<float, 3> camPos, vislib::math::Point<float, 3> centerPos) {

    vislib::math::Vector<float, 3> normalVec = vislib::math::Vector<float, 3>(centerPos.GetX() - camPos.GetX(), centerPos.GetY() - camPos.GetY(), centerPos.GetZ() - camPos.GetZ());
    normalVec.Normalise();
    // Get arbitary vector vertical to normalVec
    vislib::math::Vector<float, 3> rotVec = vislib::math::Vector<float, 3>(normalVec.GetZ(), 0.0f, -(normalVec.GetX()));
    rotVec.ScaleToLength(radius);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_CULL_FACE);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslatef(centerPos.GetX(), centerPos.GetY(), centerPos.GetZ());

    /* rotate up vector aroung lookat vector with the "Rodrigues' rotation formula" */              
    math::Vector<float, 3> v;                             // vector to rotate   
    math::Vector<float, 3> k = normalVec;                 // rotation axis = camera lookat
    float                  t = 2.0f*(float)(M_PI) / (float)subdiv; // theta angle for rotation   
    glBegin(GL_TRIANGLE_FAN);
        glVertex3f(0.0f, 0.0f, 0.0f);
        for (unsigned int i = 0; i <= subdiv; i++) {
            glVertex3fv(rotVec.PeekComponents());
            v = rotVec;    
            rotVec = v * cos(t) + k.Cross(v) * sin(t) + k * (k.Dot(v)) * (1.0f - cos(t));
            rotVec.ScaleToLength(radius);
        }
    glEnd();

    glPopMatrix();
}