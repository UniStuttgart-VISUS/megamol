/*
* CinematicRenderer.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
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
#include "mmcore/CoreInstance.h"

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
#include "ReplacementRenderer.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::utility;
using namespace megamol::cinematiccamera;

using namespace vislib;


/*
* CinematicRenderer::CinematicRenderer
*/
CinematicRenderer::CinematicRenderer(void) : Renderer3DModule(),
    theFont(megamol::core::utility::SDFFont::FontName::ROBOTO_SANS), manipulator(), textureShader(),
    slaveRendererSlot("renderer", "outgoing renderer"),
    keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper."),
    stepsParam(                "01_splineSubdivision", "Amount of interpolation steps between keyframes."),
    toggleManipulateParam(     "02_toggleManipulators", "Toggle different manipulators for the selected keyframe."),
    toggleHelpTextParam(       "03_toggleHelpText", "Show/hide help text for key assignments."),
    toggleManipOusideBboxParam("04_manipOutsideModel", "Keep manipulators always outside of model bounding box.")
    {

    // init variables
    this->interpolSteps     = 20;
    this->toggleManipulator = 0;
    this->showHelpText      = false;
    this->manipOutsideModel = false;

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

    this->toggleManipOusideBboxParam.SetParameter(new param::ButtonParam('w'));
    this->MakeSlotAvailable(&this->toggleManipOusideBboxParam);

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

    vislib::graphics::gl::ShaderSource vert, frag;

    const char *shaderName = "textureShader";

    try {
        if (!megamol::core::Module::instance()->ShaderSourceFactory().MakeShaderSource("CinematicRenderer::vertex", vert)) {
            return false;
        }
        if (!megamol::core::Module::instance()->ShaderSourceFactory().MakeShaderSource("CinematicRenderer::fragment", frag)) {
            return false;
        }
        if (!this->textureShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }
    }
    catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile %s shader (@%s): %s\n", shaderName,
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()), ce.GetMsgA());
        return false;
    }
    catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile %s shader: %s\n", shaderName, e.GetMsgA());
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile %s shader: Unknown exception\n", shaderName);
        return false;
    }

    // initialise font
    if (!this->theFont.Initialise(this->GetCoreInstance())) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [Render] Couldn't initialize the font.");
        return false;
    }

	return true;
}


/*
* CinematicRenderer::release
*/
void CinematicRenderer::release(void) {

    this->textureShader.Release();

    if (this->fbo.IsEnabled()) {
        this->fbo.Disable();
    }
    this->fbo.Release();
}


/*
* CinematicRenderer::GetCapabilities
*/
bool CinematicRenderer::GetCapabilities(Call& call) {

	CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
	if (cr3d == nullptr) return false;

	CallRender3D *oc = this->slaveRendererSlot.CallAs<CallRender3D>();
	if (!(oc == nullptr) || (!(*oc)(2))) {
		cr3d->AddCapability(oc->GetCapabilities());
	}

	return true;
}


/*
* CinematicRenderer::GetExtents
*/
bool CinematicRenderer::GetExtents(Call& call) {

    view::CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
	if (cr3d == nullptr) return false;

	view::CallRender3D *oc = this->slaveRendererSlot.CallAs<CallRender3D>();
	if (oc == nullptr) return false;

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (ccc == nullptr) return false;
	if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;

	// Get bounding box of renderer.
	if (!(*oc)(1)) return false;
	*cr3d = *oc;

    // Compute bounding box including spline (in world space) and object (in world space).
    vislib::math::Cuboid<float> bboxCR3D = oc->AccessBoundingBoxes().WorldSpaceBBox();
    // Set bounding box center of model
    ccc->setBboxCenter(cr3d->AccessBoundingBoxes().WorldSpaceBBox().CalcCenter());

    // Grow bounding box to manipulators and get information of bbox of model
    this->manipulator.SetExtents(&bboxCR3D);

    vislib::math::Cuboid<float> cboxCR3D = oc->AccessBoundingBoxes().WorldSpaceClipBox();

    // Get bounding box of spline.
    vislib::math::Cuboid<float> *bboxCCC = ccc->getBoundingBox();
    if (bboxCCC == nullptr)  {
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Get Extents] Pointer to boundingbox array is nullptr.");
        return false;
    }

    bboxCR3D.Union(*bboxCCC);
    cboxCR3D.Union(*bboxCCC); // use boundingbox to get new clipbox

    // Set new bounding box center of slave renderer model (before applying keyframe bounding box)
    ccc->setBboxCenter(oc->AccessBoundingBoxes().WorldSpaceBBox().CalcCenter());
    if (!(*ccc)(CallCinematicCamera::CallForSetSimulationData)) return false;

    // Apply new boundingbox 
	cr3d->AccessBoundingBoxes().SetWorldSpaceBBox(bboxCR3D);
    cr3d->AccessBoundingBoxes().SetWorldSpaceClipBox(cboxCR3D);

	return true;
}


/*
* CinematicRenderer::Render
*/
bool CinematicRenderer::Render(Call& call) {

    view::CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
    if (cr3d == nullptr) return false;

    view::CallRender3D *oc = this->slaveRendererSlot.CallAs<CallRender3D>();
    if (oc == nullptr) return false;

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (ccc == nullptr) return false;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;

    // Update parameter
    if (this->stepsParam.IsDirty()) {
        this->interpolSteps = this->stepsParam.Param<param::IntParam>()->Value();
        ccc->setInterpolationSteps(this->interpolSteps);
        if (!(*ccc)(CallCinematicCamera::CallForGetInterpolCamPositions)) return false;
        this->stepsParam.ResetDirty();
    }
    if (this->toggleManipulateParam.IsDirty()) {
        this->toggleManipulator = (this->toggleManipulator + 1) % 2; // There are currently two different manipulator groups ...
        this->toggleManipulateParam.ResetDirty();
    }
    if (this->toggleHelpTextParam.IsDirty()) {
        this->showHelpText = !this->showHelpText;
        this->toggleHelpTextParam.ResetDirty();
    }
    if (this->toggleManipOusideBboxParam.IsDirty()) {
        this->manipOutsideModel = !this->manipOutsideModel;
        this->toggleManipOusideBboxParam.ResetDirty();
    }

    // Set total simulation time of call
    float totalSimTime = static_cast<float>(oc->TimeFramesCount());
    ccc->setTotalSimTime(totalSimTime);
    if (!(*ccc)(CallCinematicCamera::CallForSetSimulationData)) return false;

    *oc = *cr3d;

    Keyframe skf = ccc->getSelectedKeyframe();

    // Set simulation time based on selected keyframe ('disables' animation via view3d)
    float simTime = skf.GetSimTime();
    oc->SetTime(simTime * totalSimTime);

    // Get the foreground color (inverse background color)
    float bgColor[4];
    float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float white[4]   = { 1.0f, 1.0f, 1.0f, 1.0f };
    float yellow[4]  = { 1.0f, 1.0f, 0.0f, 1.0f };
    float menu[4]    = { 0.0f, 0.0f, 0.3f, 1.0f };
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
    for (unsigned int i = 0; i < 3; i++) {
        fgColor[i] -= bgColor[i];
    }
    // COLORS
    float sColor[4] = { 0.4f, 0.4f, 1.0f, 1.0f }; // Color for SPLINE
    // Adapt colors depending on  Lightness
    float L = (vislib::math::Max(bgColor[0], vislib::math::Max(bgColor[1], bgColor[2])) + vislib::math::Min(bgColor[0], vislib::math::Min(bgColor[1], bgColor[2]))) / 2.0f;
    if (L < 0.5f) {
        // not used so far
    }

    // Get current Model-View-Projection matrix  for world space to screen space projection of keyframe camera position for mouse selection
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);
    // Compute modelViewProjMatrix matrix
    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrix = projMatrix * modelViewMatrix;

    // Get current viewport
    int vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    unsigned int vpWidth  = vp[2] - vp[0];
    unsigned int vpHeight = vp[3] - vp[1];

    // Get pointer to keyframes array
    Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Render] Pointer to keyframe array is nullptr.");
        return false;
    }

    // Get pointer to interpolated keyframes array
    Array<vislib::math::Point<float, 3> > *interpolKeyframes = ccc->getInterpolCamPositions();
    if (interpolKeyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Render] Pointer to interpolated camera positions array is nullptr.");
        return false;
    }

    // Draw slave renderer stuff ----------------------------------------------

    // Suppress TRACE output of fbo.Enable() and fbo.Create()
#if defined(DEBUG) || defined(_DEBUG)
    unsigned int otl = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(0);
#endif // DEBUG || _DEBUG 

    if (this->fbo.IsValid()) {
        if ((this->fbo.GetWidth() != vpWidth) || (this->fbo.GetHeight() != vpHeight)) {
            this->fbo.Release();
        }
    }
    if (!this->fbo.IsValid()) {
        if (!this->fbo.Create(vpWidth, vpHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT24)) {
            throw vislib::Exception("[CINEMATIC RENDERER] [render] Unable to create image framebuffer object.", __FILE__, __LINE__);
            return false;
        }
    }

    if (this->fbo.Enable() != GL_NO_ERROR) {
        throw vislib::Exception("[CINEMATIC RENDERER] [render] Cannot enable Framebuffer object.", __FILE__, __LINE__);
        return false;
    }
    // Reset TRACE output level
#if defined(DEBUG) || defined(_DEBUG)
     vislib::Trace::GetInstance().SetLevel(otl);
#endif // DEBUG || _DEBUG 

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set output buffer for override call (otherwise render call is overwritten in Base::Render(context))
    oc->SetOutputBuffer(&this->fbo);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    // Call render function of slave renderer
    (*oc)(0);

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // Disable fbo
    if (this->fbo.IsEnabled()) {
        this->fbo.Disable();
    }

    // Draw textures ------------------------------------------------------
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // DRAW DEPTH ---------------------------------------------------------
    glEnable(GL_DEPTH_TEST);

    this->textureShader.Enable();

    glUniform1f(this->textureShader.ParameterLocation("vpW"), (float)(vpWidth));
    glUniform1f(this->textureShader.ParameterLocation("vpH"), (float)(vpHeight));
    glUniform1i(this->textureShader.ParameterLocation("depthtex"), 0);

    this->fbo.DrawDepthTexture();

    this->textureShader.Disable();

    // DRAW COLORS --------------------------------------------------------
    glDisable(GL_DEPTH_TEST);

    this->fbo.DrawColourTexture();

    // Draw cinematic renderer stuff -------------------------------------------
    glEnable(GL_DEPTH_TEST);

    GLfloat tmpLw;
    glGetFloatv(GL_LINE_WIDTH, &tmpLw);
    GLfloat tmpPs;
    glGetFloatv(GL_POINT_SIZE, &tmpPs);

    if (keyframes->Count() > 0) {
        // Manipulators
        vislib::Array<KeyframeManipulator::manipType> availManip;
        availManip.Clear();
        availManip.Add(KeyframeManipulator::manipType::KEYFRAME_POS);
        if (this->toggleManipulator == 0) { // Keyframe position (along XYZ) manipulators, spline control point
            availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_POS_X);
            availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_POS_Y);
            availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_POS_Z);
            availManip.Add(KeyframeManipulator::manipType::CTRL_POINT_POS_X);
            availManip.Add(KeyframeManipulator::manipType::CTRL_POINT_POS_Y);
            availManip.Add(KeyframeManipulator::manipType::CTRL_POINT_POS_Z);
        }
        else { //if (this->toggleManipulator == 1) { // Keyframe position (along lookat), lookat and up manipulators
            availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_UP);
            availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_LOOKAT_X);
            availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_LOOKAT_Y);
            availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_LOOKAT_Z);
            availManip.Add(KeyframeManipulator::manipType::SELECTED_KF_POS_LOOKAT);
        }

        // Update manipulator data
        this->manipulator.Update(availManip, keyframes, skf, (float)(vpHeight), (float)(vpWidth), modelViewProjMatrix,
            (cr3d->GetCameraParameters()->Position().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>()) -
            (cr3d->GetCameraParameters()->LookAt().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>()),
            (cr3d->GetCameraParameters()->Position().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>()) -
            (ccc->getBboxCenter().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>()), 
            this->manipOutsideModel, ccc->getFirstControlPointPosition(), ccc->getLastControlPointPosition());
        // Draw manipulators
        this->manipulator.Draw();
    }

    // Draw spline    
    vislib::math::Point<float, 3> tmpP;
    glColor4fv(sColor);
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    for (unsigned int i = 0; i < interpolKeyframes->Count(); i++) {
        tmpP = (*interpolKeyframes)[i];
        glVertex3f(tmpP.GetX(), tmpP.GetY(), tmpP.GetZ());
    }
    glEnd();

    // DRAW MENU --------------------------------------------------------------
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0f, (float)(vpWidth), 0.0f, (float)(vpHeight), -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, 1.0f);

    float vpH = (float)(vpHeight);
    float vpW = (float)(vpWidth);

    vislib::StringA leftLabel  = " TRACKING SHOT VIEW ";

    vislib::StringA midLabel = "";
    if (cr3d->MouseSelection()) {
        if (this->toggleManipulator == 0) { 
            midLabel = "KEYFRAME manipulation (keyframe position, spline control point)";
        } else {// if (this->toggleManipulator == 1) { 
            midLabel = "KEYFRAME manipulation (lookat, up, keyframe position along lookat)";
        }
    } else {
        midLabel = "SCENE manipulation";
    }
    vislib::StringA rightLabel = " [h] show help text ";
    if (this->showHelpText) {
        rightLabel = " [h] hide help text ";
    }

    float lbFontSize        = (CC_MENU_HEIGHT); 
    float leftLabelWidth    = this->theFont.LineWidth(lbFontSize, leftLabel);
    float midleftLabelWidth = this->theFont.LineWidth(lbFontSize, midLabel);
    float rightLabelWidth   = this->theFont.LineWidth(lbFontSize, rightLabel);

    // Adapt font size if height of menu text is greater than menu height
    float vpWhalf = vpW / 2.0f;
    while (((leftLabelWidth + midleftLabelWidth/2.0f) > vpWhalf) || ((rightLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf)) {
        lbFontSize -= 0.5f;
        leftLabelWidth    = this->theFont.LineWidth(lbFontSize, leftLabel);
        midleftLabelWidth = this->theFont.LineWidth(lbFontSize, midLabel);
        rightLabelWidth   = this->theFont.LineWidth(lbFontSize, rightLabel);
    }

    // Draw menu background
    glColor4fv(menu);
    glBegin(GL_QUADS);
        glVertex2f(0.0f, vpH);
        glVertex2f(0.0f, vpH - (CC_MENU_HEIGHT));
        glVertex2f(vpW,  vpH - (CC_MENU_HEIGHT));
        glVertex2f(vpW,  vpH);
    glEnd();

    // Draw menu labels
    float labelPosY = vpH - (CC_MENU_HEIGHT) / 2.0f + lbFontSize / 2.0f;
    this->theFont.DrawString(white, 0.0f, labelPosY, lbFontSize, false, leftLabel, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->theFont.DrawString(yellow, (vpW - midleftLabelWidth) / 2.0f, labelPosY, lbFontSize, false, midLabel, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->theFont.DrawString(white, (vpW - rightLabelWidth), labelPosY, lbFontSize, false, rightLabel, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);

    // Draw help text 
    if (this->showHelpText) {
        vislib::StringA helpText = "";
        helpText += "-----[ GLOBAL ]-----\n";
        helpText += "[a] Apply current settings to selected/new keyframe. \n";
        helpText += "[d] Delete selected keyframe. \n";
        helpText += "[s] Save keyframes to file. \n";
        helpText += "[ctrl+z] Undo keyframe changes. \n";
        helpText += "[ctrl+y] Redo keyframe changes. \n";
        helpText += "-----[ TRACKING SHOT VIEW ]----- \n";
        helpText += "[tab] Toggle keyframe/scene manipulation mode. \n";
        helpText += "[m] Toggle different manipulators for the selected keyframe. \n";
        helpText += "[w] Keep manipulators always outside of model bounding box. \n";
        helpText += "[l] Reset Look-At of selected keyframe. \n";
        helpText += "-----[ CINEMATIC VIEW ]----- \n";
        helpText += "[r] Start/Stop rendering complete animation. \n";
        helpText += "[space] Toggle animation preview. \n";
        helpText += "-----[ TIME LINE VIEW ]----- \n";
        helpText += "[right/left] Move keyframe selection to right/left on animation time axis. \n";
        helpText += "[f] Snap all keyframes to animation frames. \n";
        helpText += "[g] Snap all keyframes to simulation frames. \n";
        helpText += "[t] Linearize simulation time between two keyframes. \n";
        helpText += "[left mouse button] Select keyframe. \n";
        helpText += "[middle mouse button] Axes scaling in mouse direction. \n";
        helpText += "[right mouse button] Pan axes OR drag & drop keyframe. \n";
        //UNUSED helpText += "[v] Set same velocity between all keyframes.\n";    // Calcualation is not correct yet ...
        //UNUSED helpText += "[?] Toggle rendering of model or replacement.\n";   // Key assignment is user defined ... (ReplacementRenderer is no "direct" part of cinematiccamera)

        float htFontSize  = vpW*0.027f; // max % of viewport width
        float htStrHeight = this->theFont.LineHeight(htFontSize);
        float htX         = 5.0f;
        float htY         = htX + htStrHeight;
        float htNumOfRows = 21.0f; // Number of rows the help text has
        // Adapt font size if height of help text is greater than viewport height
        while ((htStrHeight*htNumOfRows + htX + this->theFont.LineHeight(lbFontSize)) >vpH) {
            htFontSize -= 0.5f;
            htStrHeight = this->theFont.LineHeight(htFontSize);
        }

        float htStrWidth = this->theFont.LineWidth(htFontSize, helpText);
        htStrHeight      = this->theFont.LineHeight(htFontSize);
        htY              = htX + htStrHeight*htNumOfRows;
        // Draw background colored quad
        glDisable(GL_BLEND);
        glColor4fv(bgColor);
        glBegin(GL_QUADS);
            glVertex2f(htX,              htY);
            glVertex2f(htX,              htY - (htStrHeight*htNumOfRows));
            glVertex2f(htX + htStrWidth, htY - (htStrHeight*htNumOfRows));
            glVertex2f(htX + htStrWidth, htY);
        glEnd();
        // Draw help text
        glEnable(GL_BLEND);
        this->theFont.DrawString(fgColor, htX, htY, htFontSize, false, helpText, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    }

    // ------------------------------------------------------------------------
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    // Reset opengl
    glDisable(GL_BLEND);
    glLineWidth(tmpLw);
    glPointSize(tmpPs);

    return true;
}


/*
* CinematicRenderer::MouseEvent
*
* !!! ONLY triggered when "TAB" is pressed = > parameter 'enableMouseSelection' in View3D
*
*/
bool CinematicRenderer::MouseEvent(float x, float y, core::view::MouseFlags flags) {

    bool consume = false;

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (ccc == nullptr) return false;
    Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] Pointer to keyframe array is nullptr.");
        return false;
    }

    // on leftclick
    if ((flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) && (flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED)) {

        // Check if new keyframe position is selected
        int index = this->manipulator.CheckKeyframePositionHit(x, y);
        if (index >= 0) {
            ccc->setSelectedKeyframeTime((*keyframes)[index].GetAnimTime());
            if (!(*ccc)(CallCinematicCamera::CallForGetSelectedKeyframeAtTime)) return false;
            consume = true;
            //vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] KEYFRAME SELECT.");
        }
        
        // Check if manipulator is selected
        if (this->manipulator.CheckManipulatorHit(x, y)) {
            consume = true;
            //vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] MANIPULATOR SELECTED.");
        }        
    }
    else if ((flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) && !(flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED)) {
        
        // Apply changes on selected manipulator
        if (this->manipulator.ProcessManipulatorHit(x, y)) {

            ccc->setSelectedKeyframe(this->manipulator.GetManipulatedKeyframe());
            if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return false;

            ccc->setControlPointPosition(this->manipulator.GetFirstControlPointPosition(), this->manipulator.GetLastControlPointPosition());
            if (!(*ccc)(CallCinematicCamera::CallForSetCtrlPoints)) return false;

            consume = true;
            //vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] MANIPULATOR CHANGED.");
        }
    }
    /* Is not recognised ... so manipulated keyframes have to be updated each small step ... ;-(
    else if (!(flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) && (flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED)) {
        
        consume = true;
    }
    */
    
    return consume;
}
