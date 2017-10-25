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
    theFont(vislib::graphics::gl::FontInfo_Verdana, vislib::graphics::gl::OutlineFont::RENDERTYPE_FILL),
#endif // USE_SIMPLE_FONT
    stepsParam(           "01_splineSubdivision", "Amount of interpolation steps between keyframes."),
    toggleManipulateParam("02_toggleManipulators", "Toggle between position manipulators and lookat/up manipulators of selected keyframe."),
    toggleHelpTextParam(  "03_toggleHelpText", "Show/hide help text for key assignments."),
    toggleModelBBoxParam( "04_toggleModelBBox", "Toggle between full rendering of the model and semi-transparent bounding box as placeholder of the model."),
    textureShader(),
    manipulator()
    {

    // init variables
    this->interpolSteps     = 20;
    this->toggleManipulator = false;
    this->showHelpText      = false;
    this->toggleModelBBox   = false;
    this->ocBbox.SetNull();

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

    this->toggleModelBBoxParam.SetParameter(new param::ButtonParam('t'));
    this->MakeSlotAvailable(&this->toggleModelBBoxParam);

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
	if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;

	// Get bounding box of renderer.
	if (!(*oc)(1)) return false;
	*cr3d = *oc;

    // Compute bounding box including spline (in world space) and object (in world space).
    vislib::math::Cuboid<float> bboxCR3D = oc->AccessBoundingBoxes().WorldSpaceBBox();
    this->ocBbox = bboxCR3D;
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

    CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
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
    if (this->toggleModelBBoxParam.IsDirty()) {
        this->toggleModelBBox = !this->toggleModelBBox;
        this->toggleModelBBoxParam.ResetDirty();
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
    int   vpWidth = vp[2] - vp[0];
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
    if (!this->toggleModelBBox) {

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
        GLenum callOutBuffer = oc->OutputBuffer();
        oc->SetOutputBuffer(this->fbo.GetID());

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();

        // Call render function of slave renderer
        (*oc)(0);

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        // Reset output buffer
        oc->SetOutputBuffer(callOutBuffer);

        // Disable fbo
        this->fbo.Disable();


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
    }        


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
        cr3d->GetCameraParameters()->LookAt().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>());
    // Draw manipulators
    this->manipulator.draw();

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

    // Draw semi-transparent bounding box of model
    if (this->toggleModelBBox) { 
        glEnable(GL_CULL_FACE);
        // (Blending has to be enabled ...)

        glCullFace(GL_FRONT);
        this->drawBoundingBox();
        glCullFace(GL_BACK);
        this->drawBoundingBox();

        glDisable(GL_CULL_FACE);
    }


    // Draw help text  --------------------------------------------------------
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0f, (float)(vpWidth), 0.0f, (float)(vpHeight), -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Draw help text in front of bounding box rendered by the view3d
    glTranslatef(0.0f, 0.0f, 1.0f);
    glEnable(GL_POLYGON_SMOOTH);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    glColor4fv(fgColor);
    if (!this->theFont.Initialise()) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [Render] Couldn't initialize the font.");
        return false;
    }
    float fontSize = (float)(vpWidth)*0.03f; // 3% of viewport width
    vislib::StringA tmpStr = "";

    if (this->showHelpText) {
        // Adapt font size if height of help text is greater than viewport height
        float strHeight = 20.0f * this->theFont.LineHeight(fontSize);
        while (strHeight > (float)(vpHeight)) {
            fontSize -= 0.001f;
            strHeight = 20.0f * this->theFont.LineHeight(fontSize);
        }
        tmpStr += "-----[ GLOBAL ]-----\n";
        tmpStr += "[h] Hide help text.\n";
        tmpStr += "[a] Add new keyframe.\n";
        tmpStr += "[d] Delete selected Keyframe.\n";
        tmpStr += "[l] Reset Look-At of selected Keyframe.\n";
        tmpStr += "[s] Save Keyframes to file.\n";
        tmpStr += "[r] Toggle rendering complete animation.\n";
        //tmpStr += "[v] Set same velocity between all Keyframes.\n";
        tmpStr += "[space] Toggle playing animation.\n";
        tmpStr += "-----[ CINEMATIC VIEW ]-----\n";
        tmpStr += "[c] Apply view to selected Keyframe.\n \n";
        tmpStr += "-----[ TRACKING SHOT VIEW ]-----\n";
        tmpStr += "[tab] Move or Selection mode.\n";
        tmpStr += "[m] Toggle different Keyframe manipulators.\n";
        tmpStr += "[t] Show model or bounding box.\n";
        tmpStr += "-----[ TIME LINES ]-----\n";
        tmpStr += "[f] Snap keyframes to animation frames.\n";
        tmpStr += "[left mouse] Selection.\n";
        tmpStr += "[right mouse] Drag & Drop.\n";
        tmpStr += "[middle mouse] Axis scaling.\n";
    }
    else {
        tmpStr += "[h] Show help text.\n";
    }

    float strWidth = this->theFont.LineWidth(fontSize, tmpStr);
    this->theFont.DrawString(10.0f, (float)(vpHeight) - 10.0f, strWidth, 1.0f, fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);

    //this->theFont.Deinitialise();

    // Reset opengl -----------------------------------------------------------
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glLineWidth(tmpLw);
    glPointSize(tmpPs);

    glDisable(GL_BLEND);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_POLYGON_SMOOTH);

    return true;
}


/*
* CinematicRenderer::drawBoundingBox
*/
void CinematicRenderer::drawBoundingBox() {

    float alpha = 0.75f;

    glBegin(GL_QUADS);

    glEdgeFlag(true);

    glColor4f(0.5f, 0.5f, 0.5f, alpha);
    glVertex3f(this->ocBbox.Left(), this->ocBbox.Bottom(), this->ocBbox.Back());
    glVertex3f(this->ocBbox.Left(), this->ocBbox.Top(), this->ocBbox.Back());
    glVertex3f(this->ocBbox.Right(), this->ocBbox.Top(), this->ocBbox.Back());
    glVertex3f(this->ocBbox.Right(), this->ocBbox.Bottom(), this->ocBbox.Back());

    glColor4f(0.5f, 0.5f, 0.5f, alpha);
    glVertex3f(this->ocBbox.Left(), this->ocBbox.Bottom(), this->ocBbox.Front());
    glVertex3f(this->ocBbox.Right(), this->ocBbox.Bottom(), this->ocBbox.Front());
    glVertex3f(this->ocBbox.Right(), this->ocBbox.Top(), this->ocBbox.Front());
    glVertex3f(this->ocBbox.Left(), this->ocBbox.Top(), this->ocBbox.Front());

    glColor4f(0.75f, 0.75f, 0.75f, alpha);
    glVertex3f(this->ocBbox.Left(), this->ocBbox.Top(), this->ocBbox.Back());
    glVertex3f(this->ocBbox.Left(), this->ocBbox.Top(), this->ocBbox.Front());
    glVertex3f(this->ocBbox.Right(), this->ocBbox.Top(), this->ocBbox.Front());
    glVertex3f(this->ocBbox.Right(), this->ocBbox.Top(), this->ocBbox.Back());

    glColor4f(0.75f, 0.75f, 0.75f, alpha);
    glVertex3f(this->ocBbox.Left(), this->ocBbox.Bottom(), this->ocBbox.Back());
    glVertex3f(this->ocBbox.Right(), this->ocBbox.Bottom(), this->ocBbox.Back());
    glVertex3f(this->ocBbox.Right(), this->ocBbox.Bottom(), this->ocBbox.Front());
    glVertex3f(this->ocBbox.Left(), this->ocBbox.Bottom(), this->ocBbox.Front());

    glColor4f(0.25f, 0.25f, 0.25f, alpha);
    glVertex3f(this->ocBbox.Left(), this->ocBbox.Bottom(), this->ocBbox.Back());
    glVertex3f(this->ocBbox.Left(), this->ocBbox.Bottom(), this->ocBbox.Front());
    glVertex3f(this->ocBbox.Left(), this->ocBbox.Top(), this->ocBbox.Front());
    glVertex3f(this->ocBbox.Left(), this->ocBbox.Top(), this->ocBbox.Back());

    glColor4f(0.25f, 0.25f, 0.25f, alpha);
    glVertex3f(this->ocBbox.Right(), this->ocBbox.Bottom(), this->ocBbox.Back());
    glVertex3f(this->ocBbox.Right(), this->ocBbox.Top(), this->ocBbox.Back());
    glVertex3f(this->ocBbox.Right(), this->ocBbox.Top(), this->ocBbox.Front());
    glVertex3f(this->ocBbox.Right(), this->ocBbox.Bottom(), this->ocBbox.Front());

    glEnd();

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
