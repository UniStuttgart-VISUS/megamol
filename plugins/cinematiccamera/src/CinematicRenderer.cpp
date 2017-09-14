/*
* CinematicRenderer.cpp
*
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

#include "CinematicRenderer.h"
#include "CallCinematicCamera.h"

//#define _USE_MATH_DEFINES

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::cinematiccamera;
using namespace vislib;

/*
* CinematicRenderer::CinematicRenderer
*/
CinematicRenderer::CinematicRenderer(void) : Renderer3DModule(),
    slaveRendererSlot("renderer", "outgoing renderer"),
    keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper."),
#ifndef USE_SIMPLE_FONT
    theFont(vislib::graphics::gl::FontInfo_Verdana),
#endif // USE_SIMPLE_FONT
    stepsParam(           "01 Spline subdivision", "Amount of interpolation steps between keyframes"),
    toggleManipulateParam("02 Toggle manipulator", "Toggle between position manipulation or lookup manipulation."),
    toggleHelpTextParam(  "03 Toggle help text", "Show/hide help text with key assignments."),
    manipulator()
    {

    // init variables
    this->interpolSteps     = 20;
    this->toggleManipulator = false;
    this->maxAnimTime       = 1.0f;
    this->bboxCenter        = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);
    this->showHelpText      = false;

    // init parameters
    this->slaveRendererSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->slaveRendererSlot);

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    this->stepsParam.SetParameter(new param::IntParam((int)this->interpolSteps, 1));
    this->MakeSlotAvailable(&this->stepsParam);

    this->toggleManipulateParam.SetParameter(new param::ButtonParam('m'));
    this->MakeSlotAvailable(&this->toggleManipulateParam);

    this->toggleHelpTextParam.SetParameter(new param::ButtonParam('h'));
    this->MakeSlotAvailable(&this->toggleHelpTextParam);

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

    // Check for bounding box center before extending it by keyframes
    this->bboxCenter = cr3d->AccessBoundingBoxes().WorldSpaceBBox().CalcCenter();
    if (this->bboxCenter != ccc->getBboxCenter()) {
        ccc->setBboxCenter(this->bboxCenter);
        if (!(*ccc)(CallCinematicCamera::CallForSetAnimationData)) return false;
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
    if (this->toggleHelpTextParam.IsDirty()) {
        this->showHelpText = !this->showHelpText;
        this->toggleHelpTextParam.ResetDirty();
    }

    // Check for new max anim time
    this->maxAnimTime = static_cast<float>(oc->TimeFramesCount());
    if (this->maxAnimTime != ccc->getMaxAnimTime()) {
        ccc->setMaxAnimTime(this->maxAnimTime);
        if (!(*ccc)(CallCinematicCamera::CallForSetAnimationData)) return false;
    }

    // Updated data from cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;

    // Set animation time based on selected keyframe ('disables' animation via view3d)
    Keyframe skf = ccc->getSelectedKeyframe();
    // Wrap time
    float selectTime = skf.getTime();
    float frameCnt   = static_cast<float>(oc->TimeFramesCount());
    selectTime       = selectTime - (floorf(selectTime / frameCnt) * frameCnt);

    *oc = *cr3d;
    oc->SetTime(selectTime);

    // Call slave renderer.
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
    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrix = projMatrix * modelViewMatrix;

    // Get current viewport
    vislib::math::Dimension<int, 2> viewportSize = cr3d->GetViewport().GetSize();

    // Get pointer to keyframes array
    Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == NULL) {
        sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Render] Pointer to keyframe array is NULL.");
        return false;
    }

    // Get pointer to interpolated keyframes array
    Array<math::Point<float, 3> > *interpolKeyframes = ccc->getInterpolatedCamPos();
    if (interpolKeyframes == NULL) {
        sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Render] Pointer to interpolated camera positions array is NULL.");
        return false;
    }

    // Manipulators
    vislib::Array<KeyframeManipulator::manipType> availManip;
    availManip.Clear();
    availManip.Add(KeyframeManipulator::manipType::KEYFRAME_POS);
    if (this->toggleManipulator) {
        availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_POS_X);
        availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_POS_Y);
        availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_POS_Z);
    }
    else {
        availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_UP);
        availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_LOOKAT_X);
        availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_LOOKAT_Y);
        availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_LOOKAT_Z);
        availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_POS_LOOKAT);
    }
    // Update manipulator data
    this->manipulator.update(availManip, keyframes, skf, viewportSize, cr3d->GetCameraParameters()->Position(), modelViewProjMatrix);
    // Draw manipulators
    this->manipulator.draw();

    // Get the foreground color (inverse background color)
    float bgColor[4];
    float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
    for (unsigned int i = 0; i < 4; i++) {
        fgColor[i] -= bgColor[i];
    }
    // COLORS
    float sColor[4] = { 0.4f, 0.4f, 1.0f, 1.0f }; // Color for SPLINE
    // Adapt colors depending on  Lightness
    float L = (vislib::math::Max(bgColor[0], vislib::math::Max(bgColor[1], bgColor[2])) + vislib::math::Min(bgColor[0], vislib::math::Min(bgColor[1], bgColor[2]))) / 2.0f;
    if (L < 0.5f) {
        // not used so far
    }

    // Opengl setup
    GLfloat tmpLw;
    glGetFloatv(GL_LINE_WIDTH, &tmpLw);
    GLfloat tmpPs;
    glGetFloatv(GL_POINT_SIZE, &tmpPs);

    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    // Draw spline
    math::Point<float, 3> tmpP;
    glColor4fv(sColor);
    // Adding points at vertex ends for better line anti-aliasing
    glDisable(GL_BLEND);
    glPointSize(1.5f);
    glBegin(GL_POINTS);
    for (unsigned int i = 0; i < interpolKeyframes->Count(); i++) {
        tmpP = (*interpolKeyframes)[i];
        glVertex3f(tmpP.GetX(), tmpP.GetY(), tmpP.GetZ());
    }
    glEnd();
    glEnable(GL_BLEND);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
        for (unsigned int i = 0; i < interpolKeyframes->Count(); i++) {
            tmpP = (*interpolKeyframes)[i];
            glVertex3f(tmpP.GetX(), tmpP.GetY(), tmpP.GetZ());
        }
    glEnd();

    // Draw help text 
    if (!this->theFont.Initialise()) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [Render] Couldn't initialize the font.");
        return false;
    }
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0f, (float)viewportSize.GetWidth(), 0.0f, (float)viewportSize.GetHeight(), -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glEnable(GL_POLYGON_SMOOTH);
    glColor4fv(fgColor);
    float fontSize = viewportSize.GetWidth()*0.025f; // 2.5% of viewport width
    vislib::StringA tmpStr = "";
    float strWidth = this->theFont.LineWidth(fontSize, "-------------------------------------------------------");
    if (this->showHelpText) {
        tmpStr += "[t] Timeline: Move or Select/Drag&Drop mode.\n";
        tmpStr += "[tab] Move or Select mode.\n";
        tmpStr += "[m] Toggle keyframe manipulators.\n";
        tmpStr += "[a] Add new keyframe.\n";
        tmpStr += "[c] Change selected keyframe.\n";
        tmpStr += "[d] Delete selected keyframe.\n";
        tmpStr += "[l] Reset Look-At of selected keyframe.\n";
        tmpStr += "[s] Save keyframes to file.\n";
        tmpStr += "[r] Start/Stop rendering animation.\n";
        tmpStr += "[h] Hide help text.\n";
    }
    else {
        tmpStr += "[h] Show help text.\n";
    }
    this->theFont.DrawString(10.0f, viewportSize.GetHeight() - 10.0f, strWidth, 1.0f, fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    // Reset opengl
    glLineWidth(tmpLw);
    glPointSize(tmpPs);
    glDisable(GL_BLEND);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_POLYGON_SMOOTH);
    glEnable(GL_DEPTH_TEST);

    return true;
}


/*
* CinematicRenderer::MouseEvent
*/
bool CinematicRenderer::MouseEvent(float x, float y, core::view::MouseFlags flags) {

    // !!! ONLY triggered when "TAB" is pressed => parameter 'enableMouseSelection' in View3D

    bool consume = false;

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (ccc == NULL) return false;
    Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == NULL) {
        sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] Pointer to keyframe array is NULL.");
        return false;
    }

    // on leftclick
    if ((flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) && (flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED)) {

        // Check if new keyframe position is selected
        int index = this->manipulator.checkKfPosHit(x, y);
        if (index >= 0) {
            ccc->setSelectedKeyframeTime((*keyframes)[index].getTime());
            if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return false;
            consume = true;
        }
        
        // Check if manipulator is selected
        if (this->manipulator.checkManipHit(x, y)) {
            Keyframe skf = ccc->getSelectedKeyframe();
            skf.setCameraPosition(this->manipulator.getManipulatedPos());
            skf.setCameraLookAt(this->manipulator.getManipulatedLookAt());
            skf.setCameraUp(this->manipulator.getManipulatedUp());
            //ccc->setSelectedKeyframe(skf);
            if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return false;
            consume = true;
        }
        
    }
    else if ((flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) && !(flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED)) {
        
        // Apply changes on selected manipulator
        if (this->manipulator.processManipHit(x, y)) {
            Keyframe skf = ccc->getSelectedKeyframe();
            skf.setCameraPosition(this->manipulator.getManipulatedPos());
            skf.setCameraLookAt(this->manipulator.getManipulatedLookAt());
            skf.setCameraUp(this->manipulator.getManipulatedUp());
            //ccc->setSelectedKeyframe(s);
            if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return false;
            consume = true;
        }
        
    }
    
    return consume;
}
