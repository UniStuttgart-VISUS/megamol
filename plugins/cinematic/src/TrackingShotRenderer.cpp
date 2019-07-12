/*
* TrackingShotRenderer.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"

#include "TrackingShotRenderer.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::utility;
using namespace megamol::cinematic;

using namespace vislib;

TrackingShotRenderer::TrackingShotRenderer(void) : Renderer3DModule(),
    rendererCallerSlot("renderer", "outgoing renderer"),
    keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper."),
    stepsParam(                "splineSubdivision", "Amount of interpolation steps between keyframes."),
    toggleManipulateParam(     "toggleManipulators", "Toggle different manipulators for the selected keyframe."),
    toggleHelpTextParam(       "helpText", "Show/hide help text for key assignments."),
    toggleManipOusideBboxParam("manipulatorsOutsideBBox", "Keep manipulators always outside of model bounding box."),

    theFont(megamol::core::utility::SDFFont::FontName::ROBOTO_SANS), 
    interpolSteps(20),
    toggleManipulator(0),
    manipOutsideModel(false),
    showHelpText(false),
    manipulator(), 
    manipulatorGrabbed(false),
    textureShader(),
    fbo(),
    mouseX(0.0f),
    mouseY(0.0f)
{

    this->rendererCallerSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rendererCallerSlot);

    this->keyframeKeeperSlot.SetCompatibleCall<CallKeyframeKeeperDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // init parameters
    this->stepsParam.SetParameter(new param::IntParam((int)this->interpolSteps, 1));
    this->MakeSlotAvailable(&this->stepsParam);

    this->toggleManipulateParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_Q));
    this->MakeSlotAvailable(&this->toggleManipulateParam);

    this->toggleHelpTextParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_H));
    this->MakeSlotAvailable(&this->toggleHelpTextParam);

    this->toggleManipOusideBboxParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_W));
    this->MakeSlotAvailable(&this->toggleManipOusideBboxParam);

    // Load spline interpolation keyframes at startup
    this->stepsParam.ForceSetDirty();
}


TrackingShotRenderer::~TrackingShotRenderer(void) {
	this->Release();
}


bool TrackingShotRenderer::create(void) {

    vislib::graphics::gl::ShaderSource vert, frag;

    const char *shaderName = "TrackingShotShader_Render2Texturer";

    try {
        if (!megamol::core::Module::instance()->ShaderSourceFactory().MakeShaderSource("TrackingShotShader::vertex", vert)) {
            return false;
        }
        if (!megamol::core::Module::instance()->ShaderSourceFactory().MakeShaderSource("TrackingShotShader::fragment", frag)) {
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


void TrackingShotRenderer::release(void) {

    this->textureShader.Release();

    if (this->fbo.IsEnabled()) {
        this->fbo.Disable();
    }
    this->fbo.Release();
}


bool TrackingShotRenderer::GetExtents(megamol::core::view::CallRender3D& call) {

    view::CallRender3D *cr3d_in = dynamic_cast<CallRender3D*>(&call);
    if (cr3d_in == nullptr) return false;

    // Propagate changes made in GetExtents() from outgoing CallRender3D (cr3d_out) to incoming  CallRender3D (cr3d_in).
    view::CallRender3D *cr3d_out = this->rendererCallerSlot.CallAs<view::CallRender3D>();

    if ((cr3d_out != nullptr) && (*cr3d_out)(core::view::AbstractCallRender::FnGetExtents)) {
        CallKeyframeKeeper *ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
        if (ccc == nullptr) return false;
        if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return false;

        // Compute bounding box including spline (in world space) and object (in world space).
        vislib::math::Cuboid<float> bbox = cr3d_out->AccessBoundingBoxes().WorldSpaceBBox();
        // Set bounding box center of model
        ccc->setBboxCenter(cr3d_out->AccessBoundingBoxes().WorldSpaceBBox().CalcCenter());

        // Grow bounding box to manipulators and get information of bbox of model
        this->manipulator.SetExtents(&bbox);

        vislib::math::Cuboid<float> cbox = cr3d_out->AccessBoundingBoxes().WorldSpaceClipBox();

        // Get bounding box of spline.
        vislib::math::Cuboid<float> *bboxCCC = ccc->getBoundingBox();
        if (bboxCCC == nullptr) {
            vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [Get Extents] Pointer to boundingbox array is nullptr.");
            return false;
        }

        bbox.Union(*bboxCCC);
        cbox.Union(*bboxCCC); // use boundingbox to get new clipbox

        // Set new bounding box center of slave renderer model (before applying keyframe bounding box)
        ccc->setBboxCenter(cr3d_out->AccessBoundingBoxes().WorldSpaceBBox().CalcCenter());
        if (!(*ccc)(CallKeyframeKeeper::CallForSetSimulationData)) return false;

        // Propagate changes made in GetExtents() from outgoing CallRender3D (cr3d_out) to incoming  CallRender3D (cr3d_in).
        // => Bboxes and times.

        unsigned int timeFramesCount = cr3d_out->TimeFramesCount();
        cr3d_in->SetTimeFramesCount((timeFramesCount > 0) ? (timeFramesCount) : (1));
        cr3d_in->SetTime(cr3d_out->Time());
        cr3d_in->AccessBoundingBoxes() = cr3d_out->AccessBoundingBoxes();

        // Apply modified boundingbox 
        cr3d_in->AccessBoundingBoxes().SetWorldSpaceBBox(bbox);
        cr3d_in->AccessBoundingBoxes().SetWorldSpaceClipBox(cbox);
    }

	return true;
}


bool TrackingShotRenderer::Render(megamol::core::view::CallRender3D& call) {

    view::CallRender3D *cr3d_in = dynamic_cast<CallRender3D*>(&call);
    if (cr3d_in == nullptr) return false;

    view::CallRender3D *cr3d_out = this->rendererCallerSlot.CallAs<CallRender3D>();
    if (cr3d_out == nullptr) return false;

    CallKeyframeKeeper *ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return false;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return false;

    // Update parameter
    if (this->stepsParam.IsDirty()) {
        this->interpolSteps = this->stepsParam.Param<param::IntParam>()->Value();
        ccc->setInterpolationSteps(this->interpolSteps);
        if (!(*ccc)(CallKeyframeKeeper::CallForGetInterpolCamPositions)) return false;
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
    float totalSimTime = static_cast<float>(cr3d_out->TimeFramesCount());
    ccc->setTotalSimTime(totalSimTime);
    if (!(*ccc)(CallKeyframeKeeper::CallForSetSimulationData)) return false;

    Keyframe skf = ccc->getSelectedKeyframe();

    // Set simulation time based on selected keyframe ('disables'/ignores animation via view3d)
    float simTime = skf.GetSimTime();
    cr3d_in->SetTime(simTime * totalSimTime);

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

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    // Set data of outgoing cr3d to data of incoming cr3d
    *cr3d_out = *cr3d_in;

    // Set output buffer for override call (otherwise render call is overwritten in Base::Render(context))
    cr3d_out->SetOutputBuffer(&this->fbo);

    // Call render function of slave renderer
    (*cr3d_out)(core::view::AbstractCallRender::FnRender);

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

    // MANIPULATORS
    if (keyframes->Count() > 0) {

        // Update manipulator data only if currently no manipulator is grabbed
        if (!this->manipulatorGrabbed) {

            // Available manipulators
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

            this->manipulator.Update(availManip, keyframes, skf, (float)(vpHeight), (float)(vpWidth), modelViewProjMatrix,
                (cr3d_in->GetCameraParameters()->Position().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>()) -
                (cr3d_in->GetCameraParameters()->LookAt().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>()),
                (cr3d_in->GetCameraParameters()->Position().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>()) -
                (ccc->getBboxCenter().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>()),
                this->manipOutsideModel, ccc->getStartControlPointPosition(), ccc->getEndControlPointPosition());
        }

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
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0f, (float)(vpWidth), 0.0f, (float)(vpHeight), 0.0f, 1.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    float vpH = (float)(vpHeight);
    float vpW = (float)(vpWidth);

    vislib::StringA leftLabel  = " TRACKING SHOT ";
    vislib::StringA midLabel   = "";
    vislib::StringA rightLabel = " [h] Show Help Text ";
    if (this->showHelpText) {
        rightLabel = " [h] Hide Help Text ";
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
        helpText += "[Ctrl+s] Save keyframes to file. \n";
        helpText += "[Ctrl+l] Load keyframes from file. \n";
        helpText += "[Ctrl+z] Undo keyframe changes. \n";
        helpText += "[Ctrl+y] Redo keyframe changes. \n";
        helpText += "-----[ TRACKING SHOT ]----- \n";
        helpText += "[q] Toggle different manipulators for the selected keyframe. \n";
        helpText += "[w] Show manipulators inside/outside of model bounding box. \n";
        helpText += "[l] Reset look-at vector of selected keyframe. \n";
        helpText += "-----[ CINEMATIC ]----- \n";
        helpText += "[Ctrl+r] Start/Stop rendering complete animation. \n";
        helpText += "[Ctrl+Space] Start/Stop animation preview. \n";
        helpText += "-----[ TIMELINE ]----- \n";
        helpText += "[Right/Left Arrow] Move selected keyframe on animation time axis. \n";
        helpText += "[f] Snap all keyframes to animation frames. \n";
        helpText += "[g] Snap all keyframes to simulation frames. \n";
        helpText += "[t] Linearize simulation time between two keyframes. \n";
        helpText += "[v] Set same velocity between all keyframes (Experimental).\n"; // Calcualation is not correct yet ...
        helpText += "[p] Reset shifted and scaled time axes. \n";
        helpText += "[Left Mouse Button] Select keyframe. \n";
        helpText += "[Middle Mouse Button] Axes scaling in mouse direction. \n";
        helpText += "[Right Mouse Button] Drag & drop keyframe / pan axes. \n";

        float htNumOfRows = 24.0f; // Number of rows the help text has

        float htFontSize  = vpW*0.027f; // max % of viewport width
        float htStrHeight = this->theFont.LineHeight(htFontSize);
        float htX         = 5.0f;
        float htY         = htX + htStrHeight;
        // Adapt font size if height of help text is greater than viewport height
        while ((htStrHeight*htNumOfRows + htX + this->theFont.LineHeight(lbFontSize)) >vpH) {
            htFontSize -= 0.5f;
            htStrHeight = this->theFont.LineHeight(htFontSize);
        }

        float htStrWidth = this->theFont.LineWidth(htFontSize, helpText);
        htStrHeight      = this->theFont.LineHeight(htFontSize);
        htY              = htX + htStrHeight*htNumOfRows;
        // Draw background colored quad
        glColor4fv(bgColor);
        glBegin(GL_QUADS);
            glVertex2f(htX,              htY);
            glVertex2f(htX,              htY - (htStrHeight*htNumOfRows));
            glVertex2f(htX + htStrWidth, htY - (htStrHeight*htNumOfRows));
            glVertex2f(htX + htStrWidth, htY);
        glEnd();
        // Draw help text
        this->theFont.DrawString(fgColor, htX, htY, htFontSize, false, helpText, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    }

    // ------------------------------------------------------------------------
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    // Reset opengl
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glLineWidth(tmpLw);
    glPointSize(tmpPs);

    return true;
}


bool TrackingShotRenderer::OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) {

    auto* cr = this->rendererCallerSlot.CallAs<view::CallRender3D>();
    if (cr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(view::CallRender3D::FnOnMouseButton)) return true;
    }

    CallKeyframeKeeper *ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return false;
    Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] Pointer to keyframe array is nullptr.");
        return false;
    }

    bool consumed = false;

    bool down = (action == core::view::MouseButtonAction::PRESS);
    if (button == MouseButton::BUTTON_LEFT) {
        if (down) {
            // Check if manipulator is selected
            if (this->manipulator.CheckManipulatorHit(this->mouseX, this->mouseY)) {
                this->manipulatorGrabbed = true;
                consumed = true;
                //vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] MANIPULATOR SELECTED.");
            }
            else {
                // Check if new keyframe position is selected
                int index = this->manipulator.CheckKeyframePositionHit(this->mouseX, this->mouseY);
                if (index >= 0) {
                    ccc->setSelectedKeyframeTime((*keyframes)[index].GetAnimTime());
                    if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return false;
                    consumed = true;
                    //vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] KEYFRAME SELECT.");
                }
            }
        }
        else {
            // Apply changes of selected manipulator and control points
            if (this->manipulator.ProcessManipulatorHit(this->mouseX, this->mouseY)) {

                ccc->setSelectedKeyframe(this->manipulator.GetManipulatedKeyframe());
                if (!(*ccc)(CallKeyframeKeeper::CallForSetSelectedKeyframe)) return false;

                ccc->setControlPointPosition(this->manipulator.GetFirstControlPointPosition(), this->manipulator.GetLastControlPointPosition());
                if (!(*ccc)(CallKeyframeKeeper::CallForSetCtrlPoints)) return false;

                consumed = true;
                //vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC RENDERER] [MouseEvent] MANIPULATOR CHANGED.");
            }
            // ! Mode MUST alwasy be reset on left button 'up', if MOUSE moves out of viewport during manipulator is grabbed !
            this->manipulatorGrabbed = false;
        }
    }

    return consumed;
}


bool TrackingShotRenderer::OnMouseMove(double x, double y) {

    auto* cr = this->rendererCallerSlot.CallAs<view::CallRender3D>();
    if (cr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;
        cr->SetInputEvent(evt);
        if ((*cr)(view::CallRender3D::FnOnMouseMove))  return true;
    }

    // Just store current mouse position
    this->mouseX = (float)static_cast<int>(x);
    this->mouseY = (float)static_cast<int>(y);

    // Update position of grabbed manipulator
    if (!(this->manipulatorGrabbed && this->manipulator.ProcessManipulatorHit(this->mouseX, this->mouseY))) {
        return false;
    }

    return true;
}
