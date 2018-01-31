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
using namespace megamol::cinematiccamera;
using namespace vislib;


// DEFINES
#ifndef CC_MENU_HEIGHT
    #define CC_MENU_HEIGHT (20.0f)
#endif


/*
* CinematicRenderer::CinematicRenderer
*/
CinematicRenderer::CinematicRenderer(void) : Renderer3DModule(),
    slaveRendererSlot("renderer", "outgoing renderer"),
    keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper."),
#ifndef USE_SIMPLE_FONT
    theFont(vislib::graphics::gl::FontInfo_Verdana, vislib::graphics::gl::OutlineFont::RENDERTYPE_FILL),
#endif // USE_SIMPLE_FONT
    stepsParam(                "01_splineSubdivision", "Amount of interpolation steps between keyframes."),
    toggleManipulateParam(     "02_toggleManipulators", "Toggle between position manipulators and lookat/up manipulators of selected keyframe."),
    toggleHelpTextParam(       "03_toggleHelpText", "Show/hide help text for key assignments."),
    toggleManipOusideBboxParam("04_manipOutsideModel", "Keep manipulators always outside of model bounding box."),
    textureShader(),
    manipulator()
    {

    // init variables
    this->interpolSteps     = 20;
    this->toggleManipulator = false;
    this->showHelpText      = false;
    this->manipOutsideModel = false;
    this->mouseManipTime    = std::clock();

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
    if (!this->theFont.Initialise()) {
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
    if (ccc == NULL) return false;
	if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;

	// Get bounding box of renderer.
	if (!(*oc)(1)) return false;
	*cr3d = *oc;

    // Compute bounding box including spline (in world space) and object (in world space).
    vislib::math::Cuboid<float> bboxCR3D = oc->AccessBoundingBoxes().WorldSpaceBBox();
    this->modelBboxCenter = bboxCR3D.CalcCenter();

    // Grow bounding box to manipulators and get information of bbox of model
    this->manipulator.updateExtents(&bboxCR3D);

    vislib::math::Cuboid<float> cboxCR3D = oc->AccessBoundingBoxes().WorldSpaceClipBox();

    // Get bounding box of spline.
    vislib::math::Cuboid<float> *bboxCCC = ccc->getBoundingBox();
    if (bboxCCC == NULL)  {
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Get Extents] Pointer to boundingbox array is NULL.");
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
    if (cr3d == NULL) return false;

    view::CallRender3D *oc = this->slaveRendererSlot.CallAs<CallRender3D>();
    if (oc == NULL) return false;

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (ccc == NULL) return false;
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
        this->toggleManipulator = !this->toggleManipulator;
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
    // Set simulation time based on selected keyframe ('disables' animation via view3d)

    *oc = *cr3d;

    Keyframe skf = ccc->getSelectedKeyframe();
    float simTime = skf.getSimTime();
    oc->SetTime(simTime * totalSimTime);


    // Get the foreground color (inverse background color)
    float bgColor[4];
    float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
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
    int   vpWidth  = vp[2] - vp[0];
    int   vpHeight = vp[3] - vp[1];

    // Get pointer to keyframes array
    Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == NULL) {
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Render] Pointer to keyframe array is NULL.");
        return false;
    }

    // Get pointer to interpolated keyframes array
    Array<vislib::math::Point<float, 3> > *interpolKeyframes = ccc->getInterpolCamPositions();
    if (interpolKeyframes == NULL) {
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Render] Pointer to interpolated camera positions array is NULL.");
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

    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_POLYGON_SMOOTH);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // DRAW DEPTH ---------------------------------------------------------
    glDepthFunc(GL_LEQUAL);
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

    // Opengl setup
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_1D);

    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

    GLfloat tmpLw;
    glGetFloatv(GL_LINE_WIDTH, &tmpLw);
    GLfloat tmpPs;
    glGetFloatv(GL_POINT_SIZE, &tmpPs);

    if (keyframes->Count() > 0) {
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
        this->manipulator.updateRendering(availManip, keyframes, skf, (float)(vpHeight), (float)(vpWidth), modelViewProjMatrix,
            cr3d->GetCameraParameters()->Position().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>() -
            cr3d->GetCameraParameters()->LookAt().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>(),
            cr3d->GetCameraParameters()->Position().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>() -
            this->modelBboxCenter.operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>(), 
            this->manipOutsideModel);
        // Draw manipulators
        this->manipulator.draw();
    }

    vislib::math::Point<float, 3> tmpP;
    glColor4fv(sColor);
    // Adding points at vertex ends for better line anti-aliasing -> no gaps between line segments
    glDisable(GL_BLEND);
    glPointSize(1.5f);
    glBegin(GL_POINTS);
    for (unsigned int i = 0; i < interpolKeyframes->Count(); i++) {
        tmpP = (*interpolKeyframes)[i];
        glVertex3f(tmpP.GetX(), tmpP.GetY(), tmpP.GetZ());
    }
    glEnd();
    glEnable(GL_BLEND);
    // Draw spline
    glEnable(GL_LINE_SMOOTH);
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

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    float vpH = (float)(vpHeight);
    float vpW = (float)(vpWidth);

    vislib::StringA leftLabel  = " [ TRACKING SHOT VIEW ] ";

    vislib::StringA midLabel = "  "; // " Manipulation Mode: Camera ";
    if (float(clock() - this->mouseManipTime) / (float)(CLOCKS_PER_SEC) < 1.0f) {
        midLabel = " keyframe manipulation Mode ";
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
    glDisable(GL_BLEND);
    glDisable(GL_POLYGON_SMOOTH);
    glColor4f(0.0f, 0.0f, 0.3f, 1.0f);
    glBegin(GL_QUADS);
        glVertex2f(0.0f, vpH);
        glVertex2f(0.0f, vpH - (CC_MENU_HEIGHT));
        glVertex2f(vpW,  vpH - (CC_MENU_HEIGHT));
        glVertex2f(vpW,  vpH);
    glEnd();

    // Draw menu labels
    float labelPosY = vpH - (CC_MENU_HEIGHT) / 2.0f + lbFontSize / 2.0f;
    glEnable(GL_BLEND);
    glEnable(GL_POLYGON_SMOOTH);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    this->theFont.DrawString(0.0f, labelPosY, leftLabelWidth, 1.0f, lbFontSize, true, leftLabel, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
    glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
    this->theFont.DrawString((vpW - midleftLabelWidth) / 2.0f, labelPosY, midleftLabelWidth, 1.0f, lbFontSize, true, midLabel, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    this->theFont.DrawString((vpW - rightLabelWidth), labelPosY, rightLabelWidth, 1.0f, lbFontSize, true, rightLabel, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);

    // Draw help text 
    if (this->showHelpText) {
        vislib::StringA helpText = "";
        helpText += "-----[ GLOBAL ]-----\n";
        helpText += "[a] Apply current settings to selected/new keyframe.\n";
        helpText += "[d] Delete selected keyframe.\n";
        helpText += "[l] Reset Look-At of selected keyframe.\n";
        helpText += "[r] Start/Stop rendering complete animation.\n";
        helpText += "[s] Save keyframes to file.\n";
        helpText += "[u] Undo keyframe changes.\n";
        helpText += "[z] Redo keyframe changes.\n";
        helpText += "[space] Toggle animation preview.\n";
        helpText += "-----[ TRACKING SHOT VIEW ]-----\n";
        helpText += "[m] Show different keyframe manipulators.\n";
        helpText += "[w] Keep manipulators always outside of model bounding box.\n";
        helpText += "[tab] Toggle selection mode for manipulators.\n";
        helpText += "-----[ TIME LINE VIEW ]-----\n";
        helpText += "[f] Snap all keyframes to animation frames.\n";
        helpText += "[g] Snap all keyframes to simulation frames.\n";
        helpText += "[t] Linearize simulation time between two keyframes.\n";
        helpText += "[left mouse button] Select keyframe.\n";
        helpText += "[middle mouse button] Time axis scaling at mouse position.\n";
        helpText += "[right mouse button] Drag & drop keyframe.\n";
        helpText += "[right/left] Move to right/left animation time frame.\n";
        //UNUSED helpText += "[v] Set same velocity between all keyframes.\n";    // Calcualation is not correct yet ...
        //UNUSED helpText += "[?] Toggle rendering of model or replacement.\n";   // Key assignment is user defined ... (ReplacementRenderer is no "direct" part of cinematiccamera)

        float htFontSize  = vpW*0.025f; // max % of viewport width
        float htStrHeight = this->theFont.LineHeight(htFontSize);
        float htX         = 5.0f;
        float htY         = htX + htStrHeight;
        float htNumOfRows = 21.0f; // Number of rows the help text has
        // Adapt font size if height of help text is greater than viewport height
        while ((htStrHeight*htNumOfRows + htX + this->theFont.LineHeight(lbFontSize)) >vpH) {
            htFontSize -= 0.001f;
            htStrHeight = this->theFont.LineHeight(htFontSize);
        }
        float htStrWidth = this->theFont.LineWidth(htFontSize, "----------------------------------------------------------------------"); // Length of longest help text line
        htStrHeight      = this->theFont.LineHeight(htFontSize);
        htY              = htX + htStrHeight*htNumOfRows;
        // Draw background colored quad
        glDisable(GL_BLEND);
        glDisable(GL_POLYGON_SMOOTH);
        glColor4fv(bgColor);
        glBegin(GL_QUADS);
            glVertex2f(htX,              htY);
            glVertex2f(htX,              htY - (htStrHeight*htNumOfRows));
            glVertex2f(htX + htStrWidth, htY - (htStrHeight*htNumOfRows));
            glVertex2f(htX + htStrWidth, htY);
        glEnd();
        // Draw help text
        glEnable(GL_BLEND);
        glEnable(GL_POLYGON_SMOOTH);
        glColor4fv(fgColor);
        this->theFont.DrawString(htX, htY, htStrWidth, 1.0f, htFontSize, true, helpText, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    // ------------------------------------------------------------------------

    // Reset opengl
    glLineWidth(tmpLw);
    glPointSize(tmpPs);
    glDisable(GL_BLEND);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_POLYGON_SMOOTH);

    return true;
}


/*
* CinematicRenderer::MouseEvent
*
* !!! ONLY triggered when "TAB" is pressed = > parameter 'enableMouseSelection' in View3D
*
*/
bool CinematicRenderer::MouseEvent(float x, float y, core::view::MouseFlags flags) {

    this->mouseManipTime = std::clock();

    bool consume = false;

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (ccc == NULL) return false;
    Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == NULL) {
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] Pointer to keyframe array is NULL.");
        return false;
    }

    // on leftclick
    if ((flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) && (flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED)) {

        // Check if new keyframe position is selected
        int index = this->manipulator.checkKfPosHit(x, y);
        if (index >= 0) {
            ccc->setSelectedKeyframeTime((*keyframes)[index].getAnimTime());
            if (!(*ccc)(CallCinematicCamera::CallForGetSelectedKeyframeAtTime)) return false;
            consume = true;
            //vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] KEYFRAME SELECT.");
        }
        
        // Check if manipulator is selected
        if (this->manipulator.checkManipHit(x, y)) {
            ccc->setSelectedKeyframe(this->manipulator.getManipulatedKeyframe());
            if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return false;
            consume = true;
            //vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] MANIPULATOR SELECTED.");
        }
        
    }
    else if ((flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) && !(flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED)) {
        
        // Apply changes on selected manipulator
        if (this->manipulator.processManipHit(x, y)) {
            ccc->setSelectedKeyframe(this->manipulator.getManipulatedKeyframe());
            if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return false;
            consume = true;
            //vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] MANIPULATOR CHANGED.");
        }
    }
    
    return consume;
}
