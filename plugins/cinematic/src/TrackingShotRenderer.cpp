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
    , keyframeKeeperSlot("keyframeData", "Connects to the Keyframe Keeper.")
    , stepsParam("splineSubdivision", "Amount of interpolation steps between keyframes.")
    , toggleHelpTextParam("helpText", "Show/hide help text for key assignments.")
    , manipulators()
    , utils()
    , mouseX(0.0f)
    , mouseY(0.0f)
    , texture(0)
    , manipulatorGrabbed(false)
    , interpolSteps(20)
    , showHelpText(false)
    , lineWidth(1.0f) {

    this->keyframeKeeperSlot.SetCompatibleCall<CallKeyframeKeeperDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // init parameters
    this->stepsParam.SetParameter(new param::IntParam((int)this->interpolSteps, 1));
    this->MakeSlotAvailable(&this->stepsParam);

    this->toggleHelpTextParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_H, core::view::Modifier::SHIFT));
    this->MakeSlotAvailable(&this->toggleHelpTextParam);

    for (auto& slot : this->manipulators.GetParams()) {
        if (slot != nullptr) {
            this->MakeSlotAvailable(&(*slot));
        }
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
        megamol::core::utility::log::Log::DefaultLog.WriteError("[TRACKINGSHOT RENDERER] [create] Couldn't initialize render utils. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

	return true;
}


void TrackingShotRenderer::release(void) {

}


bool TrackingShotRenderer::GetExtents(megamol::core::view::CallRender3D_2& call) {

    // Propagate changes made in GetExtents() from outgoing CallRender3D_2 (cr3d_out) to incoming CallRender3D_2 (cr3d_in).
    auto cr3d_out = this->chainRenderSlot.CallAs<view::CallRender3D_2>();

    if ((cr3d_out != nullptr) && (*cr3d_out)(core::view::AbstractCallRender::FnGetExtents)) {
        CallKeyframeKeeper *ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
        if (ccc == nullptr) return false;
        if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return false;

        vislib::math::Cuboid<float> bbox = cr3d_out->AccessBoundingBoxes().BoundingBox();
        vislib::math::Cuboid<float> cbox = cr3d_out->AccessBoundingBoxes().ClipBox();

        // Empirical line width
        this->lineWidth = bbox.LongestEdge() * 0.01f;

        // Grow bounding box to manipulators and get information of bbox of model
        this->manipulators.UpdateExtents(bbox);

        // Use bounding box as clipbox
        cbox.Union(bbox); 

        // Set new bounding box center of slave renderer model (before applying keyframe bounding box)
        ccc->SetBboxCenter(vislib_point_to_glm(cr3d_out->AccessBoundingBoxes().BoundingBox().CalcCenter()));
        if (!(*ccc)(CallKeyframeKeeper::CallForSetSimulationData)) return false;

        // Propagate changes made in GetExtents() from outgoing CallRender3D_2 (cr3d_out) to incoming  CallRender3D_2 (cr3d_in) => Bboxes and times.
        unsigned int timeFramesCount = cr3d_out->TimeFramesCount();
        call.SetTimeFramesCount((timeFramesCount > 0) ? (timeFramesCount) : (1));
        call.SetTime(cr3d_out->Time());
        call.AccessBoundingBoxes() = cr3d_out->AccessBoundingBoxes();
        // Apply modified boundingbox 
        call.AccessBoundingBoxes().SetBoundingBox(bbox);
        call.AccessBoundingBoxes().SetClipBox(cbox);
    }

	return true;
}


bool TrackingShotRenderer::Render(megamol::core::view::CallRender3D_2& call) {

    auto cr3d_out = this->chainRenderSlot.CallAs<CallRender3D_2>();
    if (cr3d_out == nullptr) return false;

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
    call.SetTime(simTime * totalSimTime);

    // Get pointer to keyframes array
    auto keyframes = ccc->GetKeyframes();
    if (keyframes == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("[TRACKINGSHOT RENDERER] [Render] Pointer to keyframe array is nullptr.");
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
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;
    cam.calc_matrices(snapshot, viewTemp, projTemp, thecam::snapshot_content::all);
    glm::vec4 snap_pos = snapshot.position;
    glm::vec4 snap_view = snapshot.view_vector;
    glm::vec3 cam_pos = static_cast<glm::vec3>(snap_pos);
    glm::vec3 cam_view = static_cast<glm::vec3>(snap_view);
    glm::mat4 view = viewTemp;
    glm::mat4 proj = projTemp;
    glm::mat4 mvp = proj * view;

    // Get current viewport
    auto viewport = call.GetViewport();
    const float vp_fw = static_cast<float>(viewport.Width());
    const float vp_fh = static_cast<float>(viewport.Height());

    // Get matrix for orthogonal projection of 2D rendering
    glm::mat4 ortho = glm::ortho(0.0f, vp_fw, 0.0f, vp_fh, -1.0f, 1.0f);

    // Push manipulators ------------------------------------------------------
    if (keyframes->size() > 0) {
        cam_type::minimal_state_type camera_state;
        cam.get_minimal_state(camera_state);
        this->manipulators.UpdateRendering(keyframes, skf, ccc->GetStartControlPointPosition(), ccc->GetEndControlPointPosition(), camera_state, glm::vec2(vp_fw, vp_fh), mvp);
        this->manipulators.PushRendering(this->utils);
    }

    // Push spline ------------------------------------------------------------
    auto interpolKeyframes = ccc->GetInterpolCamPositions();
    if (interpolKeyframes == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("[TRACKINGSHOT RENDERER] [Render] Pointer to interpolated camera positions array is nullptr.");
        return false;
    }
    auto color = this->utils.Color(CinematicUtils::Colors::KEYFRAME_SPLINE);
    auto keyframeCount = interpolKeyframes->size();
    if (keyframeCount > 1) {
        for (int i = 0; i < (keyframeCount - 1); i++) {
            glm::vec3 start = interpolKeyframes->operator[](i);
            glm::vec3 end = interpolKeyframes->operator[](i + 1);
            this->utils.PushLinePrimitive(start, end, this->lineWidth, cam_view, cam_pos, color);
        }
    }

    // Draw 3D ---------------------------------------------------------------
    this->utils.DrawAll(mvp, glm::vec2(vp_fw, vp_fh));

    // Push hotkey list ------------------------------------------------------
    // Draw help text 
    if (this->showHelpText) {
        this->utils.PushHotkeyList(vp_fw, vp_fh);
    }

    // Push menu --------------------------------------------------------------
    std::string leftLabel = " TRACKING SHOT ";
    std::string midLabel = "";
    std::string rightLabel = " [Shift+h] Show Help Text ";
    if (this->showHelpText) {
        rightLabel = " [Shift+h] Hide Help Text ";
    }
    this->utils.PushMenu(leftLabel, midLabel, rightLabel, vp_fw, vp_fh);

    // Draw 2D ---------------------------------------------------------------
    this->utils.DrawAll(ortho, glm::vec2(vp_fw, vp_fh));

    return true;
}


bool TrackingShotRenderer::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {

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
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("[TRACKINGSHOT RENDERER] [OnMouseButton] Pointer to keyframe array is nullptr.");
        return false;
    }

    bool consumed = false;

    bool down = (action == core::view::MouseButtonAction::PRESS);
    if (button == MouseButton::BUTTON_LEFT) {
        if (down) {
            // Check if manipulator is selected
            if (this->manipulators.CheckForHitManipulator(this->mouseX, this->mouseY)) {
                this->manipulatorGrabbed = true;
                consumed = true;
                //megamol::core::utility::log::Log::DefaultLog.WriteInfo("[TRACKINGSHOT RENDERER] [OnMouseButton] MANIPULATOR SELECTED.");
            }
            else {
                // Check if new keyframe position is selected
                int index = this->manipulators.GetSelectedKeyframePositionIndex(this->mouseX, this->mouseY);
                if (index >= 0) {
                    ccc->SetSelectedKeyframeTime((*keyframes)[index].GetAnimTime());
                    if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return false;
                    consumed = true;
                    //megamol::core::utility::log::Log::DefaultLog.WriteInfo("[TRACKINGSHOT RENDERER] [OnMouseButton] KEYFRAME SELECT.");
                }
            }
        }
        else {
            // Apply changes of selected manipulator and control points
            if (this->manipulatorGrabbed) {

                ccc->SetSelectedKeyframe(this->manipulators.GetManipulatedSelectedKeyframe());
                if (!(*ccc)(CallKeyframeKeeper::CallForSetSelectedKeyframe)) return false;

                ccc->SetControlPointPosition(this->manipulators.GetFirstControlPointPosition(), this->manipulators.GetLastControlPointPosition());
                if (!(*ccc)(CallKeyframeKeeper::CallForSetCtrlPoints)) return false;

                consumed = true;
                this->manipulators.ResetHitManipulator();
                //megamol::core::utility::log::Log::DefaultLog.WriteInfo("[TRACKINGSHOT RENDERER] [OnMouseButton] MANIPULATOR CHANGED.");
            }
            // ! Mode MUST alwasy be reset on left button 'up', if MOUSE moves out of viewport during manipulator is grabbed !
            this->manipulatorGrabbed = false;
        }
    }

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
    if (this->manipulatorGrabbed && this->manipulators.ProcessHitManipulator(this->mouseX, this->mouseY)) {
        return true;
    }

    return false;
}
