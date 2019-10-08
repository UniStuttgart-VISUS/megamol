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


TrackingShotRenderer::TrackingShotRenderer(void) : Renderer3DModule_2()
    , keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper.")
    , stepsParam("splineSubdivision", "Amount of interpolation steps between keyframes.")
    , toggleHelpTextParam("helpText", "Show/hide help text for key assignments.")
    , interpolSteps(20)
    , showHelpText(false)
    , manipulators()
    , manipulatorGrabbed(false)
    , utils()
    , fbo()
    , mouseX(0.0f)
    , mouseY(0.0f)
    , texture(0) {

    this->keyframeKeeperSlot.SetCompatibleCall<CallKeyframeKeeperDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // init parameters
    this->stepsParam.SetParameter(new param::IntParam((int)this->interpolSteps, 1));
    this->MakeSlotAvailable(&this->stepsParam);

    this->toggleHelpTextParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_H, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->toggleHelpTextParam);

    for (auto slot : this->manipulators.GetParams()) {
        this->MakeSlotAvailable(&(*slot));
    }

    // Load spline interpolation keyframes at startup
    this->stepsParam.ForceSetDirty();
}


TrackingShotRenderer::~TrackingShotRenderer(void) {

	this->Release();
}


bool TrackingShotRenderer::create(void) {

    // Initialise render utils
    if (!this->utils.Initialise(this->GetCoreInstance())) {
        vislib::sys::Log::DefaultLog.WriteError("[TRACKINGSHOT RENDERER] [create] Couldn't initialize render utils.");
        return false;
    }

	return true;
}


void TrackingShotRenderer::release(void) {

    if (this->fbo.IsEnabled()) {
        this->fbo.Disable();
    }
    this->fbo.Release();
}


bool TrackingShotRenderer::GetExtents(megamol::core::view::CallRender3D_2& call) {

    auto cr3d_in = &call;
    if (cr3d_in == nullptr) return false;

    // Propagate changes made in GetExtents() from outgoing CallRender3D_2 (cr3d_out) to incoming CallRender3D_2 (cr3d_in).
    auto cr3d_out = this->chainRenderSlot.CallAs<view::CallRender3D_2>();

    if ((cr3d_out != nullptr) && (*cr3d_out)(core::view::AbstractCallRender::FnGetExtents)) {
        CallKeyframeKeeper *ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
        if (ccc == nullptr) return false;
        if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return false;

        // Compute bounding box including spline (in world space) and object (in world space).
        vislib::math::Cuboid<float> bbox = cr3d_out->AccessBoundingBoxes().BoundingBox();
        // Set bounding box center of model
        ccc->SetBboxCenter(P2G(cr3d_out->AccessBoundingBoxes().BoundingBox().CalcCenter()));

        // Grow bounding box to manipulators and get information of bbox of model
        this->manipulators.UpdateExtents(bbox);

        vislib::math::Cuboid<float> cbox = cr3d_out->AccessBoundingBoxes().ClipBox();

        // Get bounding box of spline.
        auto bboxCCC = ccc->GetBoundingBox();
        if (bboxCCC == nullptr) {
            vislib::sys::Log::DefaultLog.WriteWarn("[TRACKINGSHOT RENDERER] [GetExtents] Pointer to boundingbox array is nullptr.");
            return false;
        }

        bbox.Union(*bboxCCC);
        cbox.Union(*bboxCCC); // use boundingbox to get new clipbox

        // Set new bounding box center of slave renderer model (before applying keyframe bounding box)
        ccc->SetBboxCenter(P2G(cr3d_out->AccessBoundingBoxes().BoundingBox().CalcCenter()));
        if (!(*ccc)(CallKeyframeKeeper::CallForSetSimulationData)) return false;

        // Propagate changes made in GetExtents() from outgoing CallRender3D_2 (cr3d_out) to incoming  CallRender3D_2 (cr3d_in).
        // => Bboxes and times.

        unsigned int timeFramesCount = cr3d_out->TimeFramesCount();
        cr3d_in->SetTimeFramesCount((timeFramesCount > 0) ? (timeFramesCount) : (1));
        cr3d_in->SetTime(cr3d_out->Time());
        cr3d_in->AccessBoundingBoxes() = cr3d_out->AccessBoundingBoxes();

        // Apply modified boundingbox 
        cr3d_in->AccessBoundingBoxes().SetBoundingBox(bbox);
        cr3d_in->AccessBoundingBoxes().SetClipBox(cbox);
    }

	return true;
}


void TrackingShotRenderer::PreRender(core::view::CallRender3D_2& call) {

    auto cr3d_in = &call;
    if (cr3d_in == nullptr) return;

    auto cr3d_out = this->chainRenderSlot.CallAs<CallRender3D_2>();
    if (cr3d_out == nullptr) return;

    // Get current camera
    view::Camera_2 cam;
    cr3d_in->GetCamera(cam);

    // Get current viewport
    ///? auto viewport = cr3d_in->GetViewport().GetSize();
    glm::vec4 viewport;
    if (!cam.image_tile().empty()) {
        viewport = glm::vec4(
            cam.image_tile().left(), cam.image_tile().bottom(), cam.image_tile().width(), cam.image_tile().height());
    }
    else {
        viewport = glm::vec4(0.0f, 0.0f, cam.resolution_gate().width(), cam.resolution_gate().height());
    }
    UINT vpW_int = static_cast<UINT>(viewport.z);
    UINT vpH_int = static_cast<UINT>(viewport.w);

    // Prepare rendering to FBO of chained output -----------------------------

/// Suppress TRACE output of fbo.Enable() and fbo.Create()
#if defined(DEBUG) || defined(_DEBUG)
    unsigned int otl = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(0);
#endif // DEBUG || _DEBUG 

    if (this->fbo.IsValid()) {
        if ((this->fbo.GetWidth() != vpW_int) || (this->fbo.GetHeight() != vpH_int)) {
            this->fbo.Release();
        }
    }
    if (!this->fbo.IsValid()) {
        if (!this->fbo.Create(vpW_int, vpH_int, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT24)) {
            throw vislib::Exception("[TRACKINGSHOT RENDERER] [render] Unable to create image framebuffer object.", __FILE__, __LINE__);
            return;
        }
    }
    if (this->fbo.Enable() != GL_NO_ERROR) {
        throw vislib::Exception("[TRACKINGSHOT RENDERER] [render] Cannot enable Framebuffer object.", __FILE__, __LINE__);
        return;
    }

/// Reset TRACE output level
#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(otl);
#endif // DEBUG || _DEBUG 

    auto backCol = call.BackgroundColor();
    glClearColor(backCol.x, backCol.y, backCol.z, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set data of outgoing cr3d to data of incoming cr3d
    *cr3d_out = *cr3d_in;
    // Set output buffer for override call (otherwise render call is overwritten in Base::Render(context))
    cr3d_out->SetOutputBuffer(&this->fbo);
}


bool TrackingShotRenderer::Render(megamol::core::view::CallRender3D_2& call) {

    auto cr3d_in = &call;
    if (cr3d_in == nullptr) return false;

    auto cr3d_out = this->chainRenderSlot.CallAs<CallRender3D_2>();
    if (cr3d_out == nullptr) return false;

    // Disable fbo from pre render step
    if (this->fbo.IsEnabled()) {
        this->fbo.Disable();
    }

    // Get update data from keyframe keeper -----------------------------------
    auto ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return false;
    if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return false;

    // Set total simulation time 
    float totalSimTime = static_cast<float>(cr3d_out->TimeFramesCount());
    ccc->SetTotalSimTime(totalSimTime);
    if (!(*ccc)(CallKeyframeKeeper::CallForSetSimulationData)) return false;

    // Get selected keyframe
    Keyframe skf = ccc->GetSelectedKeyframe();

    // Set current simulation time based on selected keyframe ('disables'/ignores animation via view3d)
    float simTime = skf.GetSimTime();
    cr3d_in->SetTime(simTime * totalSimTime);

    // Get pointer to keyframes array
    auto keyframes = ccc->GetKeyframes();
    if (keyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TRACKINGSHOT RENDERER] [Render] Pointer to keyframe array is nullptr.");
        return false;
    }

    // Update parameters ------------------------------------------------------
    if (this->stepsParam.IsDirty()) {
        this->interpolSteps = this->stepsParam.Param<param::IntParam>()->Value();
        ccc->SetInterpolationSteps(this->interpolSteps);
        if (!(*ccc)(CallKeyframeKeeper::CallForGetInterpolCamPositions)) return false;
        this->stepsParam.ResetDirty();
    }

    if (this->toggleHelpTextParam.IsDirty()) {
        this->showHelpText = !this->showHelpText;
        this->toggleHelpTextParam.ResetDirty();
    }

    // Init rendering ---------------------------------------------------------
    glm::vec4 back_color;
    glGetFloatv(GL_COLOR_CLEAR_VALUE, static_cast<GLfloat*>(glm::value_ptr(back_color)));
    this->utils.SetBackgroundColor(back_color);

    // Get current camera
    view::Camera_2 cam;
    cr3d_in->GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;
    cam.calc_matrices(snapshot, viewTemp, projTemp, thecam::snapshot_content::all);
    glm::vec4 cam_pos = snapshot.position;
    glm::vec4 cam_view = snapshot.view_vector;
    glm::vec4 cam_right = snapshot.right_vector;
    glm::vec4 cam_up = snapshot.up_vector;
    glm::mat4 view = viewTemp;
    glm::mat4 proj = projTemp;
    glm::mat4 mvp = proj * view;

    // Get current viewport
    ///? auto viewport = cr3d_in->GetViewport().GetSize();
    glm::vec4 viewport;
    if (!cam.image_tile().empty()) {
        viewport = glm::vec4(
            cam.image_tile().left(), cam.image_tile().bottom(), cam.image_tile().width(), cam.image_tile().height());
    }
    else {
        viewport = glm::vec4(0.0f, 0.0f, cam.resolution_gate().width(), cam.resolution_gate().height());
    }
    const float vp_fw = viewport.z;
    const float vp_fh = viewport.w;

    // Get matrix for orthogonal projection of 2D rendering
    glm::mat4 ortho = glm::ortho(0.0f, vp_fw, 0.0f, vp_fh, -1.0f, 1.0f);



    // Push manipulators ------------------------------------------------------
    //  if (keyframes->size() > 0) {
    // Update manipulator data only if currently no manipulator is grabbed
	//if (!this->manipulatorGrabbed) {
	//	// Available manipulators
	//	std::vector<KeyframeManipulator::manipType> availManip;
	//	availManip.clear();
	//	availManip.emplace_back(KeyframeManipulator::manipType::KEYFRAME_POS);
	//	if (this->toggleManipulator == 0) { // Keyframe position (along XYZ) manipulators, spline control point
	//		availManip.emplace_back(KeyframeManipulator::manipType::SELECTED_KF_POS_X);
	//		availManip.emplace_back(KeyframeManipulator::manipType::SELECTED_KF_POS_Y);
	//		availManip.emplace_back(KeyframeManipulator::manipType::SELECTED_KF_POS_Z);
	//		availManip.emplace_back(KeyframeManipulator::manipType::CTRL_POINT_POS_X);
	//		availManip.emplace_back(KeyframeManipulator::manipType::CTRL_POINT_POS_Y);
	//		availManip.emplace_back(KeyframeManipulator::manipType::CTRL_POINT_POS_Z);
	//	}
	//	else { //if (this->toggleManipulator == 1) { // Keyframe position (along lookat), lookat and up manipulators
	//		availManip.emplace_back(KeyframeManipulator::manipType::SELECTED_KF_UP);
	//		availManip.emplace_back(KeyframeManipulator::manipType::SELECTED_KF_LOOKAT_X);
	//		availManip.emplace_back(KeyframeManipulator::manipType::SELECTED_KF_LOOKAT_Y);
	//		availManip.emplace_back(KeyframeManipulator::manipType::SELECTED_KF_LOOKAT_Z);
	//		availManip.emplace_back(KeyframeManipulator::manipType::SELECTED_KF_POS_LOOKAT);
	//	}
	//	// Get current Model-View-Projection matrix for world space to screen space projection of keyframe camera position for mouse selection
	//	view::Camera_2 cam;
	//	cr3d_in->GetCamera(cam);
	//	cam_type::snapshot_type snapshot;
	//	cam_type::matrix_type viewTemp, projTemp;
	//	// Generate complete snapshot and calculate matrices
	//	cam.calc_matrices(snapshot, viewTemp, projTemp, thecam::snapshot_content::all);
	//	glm::vec4 CamPos = snapshot.position;
	//	glm::vec4 CamView = snapshot.view_vector;
	//	glm::mat4 MVP = projTemp * viewTemp;
	//	glm::vec4 BboxCenter = { ccc->GetBboxCenter().x, ccc->GetBboxCenter().y, ccc->GetBboxCenter().z, 1.0f};
	//
    //  this->manipulator.Update(availManip, keyframes, skf, vpW_flt, vpH_flt, MVP, (CamPos - CamView),(CamPos - BboxCenter),
	//  this->manipOutsideModel, ccc->GetStartControlPointPosition(), ccc->GetEndControlPointPosition());
    //  }

    //  // Draw manipulators
    //  this->manipulator.Draw();
    //}

    // Push spline ------------------------------------------------------------
    auto interpolKeyframes = ccc->GetInterpolCamPositions();
    if (interpolKeyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TRACKINGSHOT RENDERER] [Render] Pointer to interpolated camera positions array is nullptr.");
        return false;
    }
    float line_width = 2.0f;
    auto color = this->utils.Color(CinematicUtils::Colors::KEYFRAME_SPLINE);
    auto keyframeCount = interpolKeyframes->size();
    if (keyframeCount > 1) {
        for (int i = 0; i < (keyframeCount - 1); i++) {
            glm::vec3 start = interpolKeyframes->operator[](i);
            glm::vec3 end = interpolKeyframes->operator[](i + 1);
            this->utils.PushLinePrimitive(start, end, line_width, cam_pos, color);
        }
    }

    // Draw 3D ---------------------------------------------------------------
    this->utils.DrawAll(mvp, glm::vec2(vp_fw, vp_fh));

    // Draw textures ----------------------------------------------------------
    /// Draw color texture after 3D stuff and before other 2D stuff (because depth is disabled for color texture drawing).
    glm::vec3 pos_bottom_left = { 0.0f, 0.0f, 0.0f };
    glm::vec3 pos_upper_left = { 0.0f, vp_fh, 0.0f };
    glm::vec3 pos_upper_right = { vp_fw, vp_fh, 0.0f };
    glm::vec3 pos_bottom_right = { vp_fw, 0.0f, 0.0f };
    this->utils.Push2DColorTexture(this->fbo.GetColourTextureID(), pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right);
    this->utils.Push2DDepthTexture(this->fbo.GetDepthTextureID(), pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right);
    this->utils.DrawTextures(ortho, glm::vec2(vp_fw, vp_fh));

    // Push hotkey list ------------------------------------------------------
    // Draw help text 
    if (this->showHelpText) {
        this->utils.PushHotkeyList(vp_fw, vp_fh);
    }

    // Push menu --------------------------------------------------------------
    std::string leftLabel = " TRACKING SHOT ";
    std::string midLabel = "";
    std::string rightLabel = " [Ctrl+h] Show Help Text ";
    if (this->showHelpText) {
        rightLabel = " [Ctrl+h] Hide Help Text ";
    }
    this->utils.PushMenu(leftLabel, midLabel, rightLabel, vp_fw, vp_fh);

    // Draw 2D ---------------------------------------------------------------
    this->utils.DrawAll(ortho, glm::vec2(vp_fw, vp_fh));

    return true;
}


bool TrackingShotRenderer::OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) {

    auto cr = this->chainRenderSlot.CallAs<view::CallRender3D_2>();
    if (cr != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(view::CallRender3D_2::FnOnMouseButton)) return true;
    }

    auto ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return false;
    auto keyframes = ccc->GetKeyframes();
    if (keyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TRACKINGSHOT RENDERER] [OnMouseButton] Pointer to keyframe array is nullptr.");
        return false;
    }

    bool consumed = false;

    //bool down = (action == core::view::MouseButtonAction::PRESS);
    //if (button == MouseButton::BUTTON_LEFT) {
    //    if (down) {
    //        // Check if manipulator is selected
    //        if (this->manipulator.CheckManipulatorHit(this->mouseX, this->mouseY)) {
    //            this->manipulatorGrabbed = true;
    //            consumed = true;
    //            //vislib::sys::Log::DefaultLog.WriteInfo("[TRACKINGSHOT RENDERER] [OnMouseButton] MANIPULATOR SELECTED.");
    //        }
    //        else {
    //            // Check if new keyframe position is selected
    //            int index = this->manipulator.CheckKeyframePositionHit(this->mouseX, this->mouseY);
    //            if (index >= 0) {
    //                ccc->SetSelectedKeyframeTime((*keyframes)[index].GetAnimTime());
    //                if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return false;
    //                consumed = true;
    //                //vislib::sys::Log::DefaultLog.WriteInfo("[TRACKINGSHOT RENDERER] [OnMouseButton] KEYFRAME SELECT.");
    //            }
    //        }
    //    }
    //    else {
    //        // Apply changes of selected manipulator and control points
    //        if (this->manipulator.ProcessManipulatorHit(this->mouseX, this->mouseY)) {

    //            ccc->SetSelectedKeyframe(this->manipulator.GetManipulatedKeyframe());
    //            if (!(*ccc)(CallKeyframeKeeper::CallForSetSelectedKeyframe)) return false;

    //            ccc->SetControlPointPosition(this->manipulator.GetFirstControlPointPosition(), this->manipulator.GetLastControlPointPosition());
    //            if (!(*ccc)(CallKeyframeKeeper::CallForSetCtrlPoints)) return false;

    //            consumed = true;
    //            //vislib::sys::Log::DefaultLog.WriteInfo("[TRACKINGSHOT RENDERER] [OnMouseButton] MANIPULATOR CHANGED.");
    //        }
    //        // ! Mode MUST alwasy be reset on left button 'up', if MOUSE moves out of viewport during manipulator is grabbed !
    //        this->manipulatorGrabbed = false;
    //    }
    //}

    return consumed;
}


bool TrackingShotRenderer::OnMouseMove(double x, double y) {

    auto cr = this->chainRenderSlot.CallAs<view::CallRender3D_2>();
    if (cr != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;
        cr->SetInputEvent(evt);
        if ((*cr)(view::CallRender3D_2::FnOnMouseMove))  return true;
    }

    // Just store current mouse position
    this->mouseX = (float)static_cast<int>(x);
    this->mouseY = (float)static_cast<int>(y);

    // Check for grabbed or hit manipulator
    //if (this->manipulatorGrabbed && this->manipulator.ProcessManipulatorHit(this->mouseX, this->mouseY)) {
    //    return true;
    //}

    return false;
}
