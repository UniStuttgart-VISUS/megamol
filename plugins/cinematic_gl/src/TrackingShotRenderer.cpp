/*
 * TrackingShotRenderer.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "TrackingShotRenderer.h"
#include "cinematic/CallKeyframeKeeper.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::utility;
using namespace megamol::cinematic_gl;

using namespace vislib;


TrackingShotRenderer::TrackingShotRenderer(void)
        : megamol::core_gl::view::Renderer3DModuleGL()
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
        , lineWidth(1.0f)
        , skipped_first_mouse_interact(false) {

    this->keyframeKeeperSlot.SetCompatibleCall<cinematic::CallKeyframeKeeperDescription>();
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
}


TrackingShotRenderer::~TrackingShotRenderer(void) {

    this->Release();
}


bool TrackingShotRenderer::create(void) {

    // Initialise render utils
    if (!this->utils.Initialise(this->GetCoreInstance())) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[TRACKINGSHOT RENDERER] [create] Couldn't initialize render utils. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


void TrackingShotRenderer::release(void) {}


bool TrackingShotRenderer::GetExtents(megamol::core_gl::view::CallRender3DGL& call) {

    // Propagate changes made in GetExtents() from outgoing CallRender3DGL (cr3d_out) to incoming CallRender3DGL (cr3d_in).
    auto cr3d_out = this->chainRenderSlot.CallAs<core_gl::view::CallRender3DGL>();

    if ((cr3d_out != nullptr) && (*cr3d_out)(core::view::AbstractCallRender::FnGetExtents)) {
        cinematic::CallKeyframeKeeper* ccc = this->keyframeKeeperSlot.CallAs<cinematic::CallKeyframeKeeper>();
        if (ccc == nullptr)
            return false;
        if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForGetUpdatedKeyframeData))
            return false;

        vislib::math::Cuboid<float> bbox = cr3d_out->AccessBoundingBoxes().BoundingBox();
        vislib::math::Cuboid<float> cbox = cr3d_out->AccessBoundingBoxes().ClipBox();

        // Empirical line width
        this->lineWidth = bbox.LongestEdge() * 0.01f;

        // Grow bounding box to manipulators and get information of bbox of model
        this->manipulators.UpdateExtents(bbox);

        // Use bounding box as clipbox
        cbox.Union(bbox);

        // Set new bounding box center of slave renderer model (before applying keyframe bounding box)
        ccc->SetBboxCenter(
            core_gl::utility::vislib_point_to_glm(cr3d_out->AccessBoundingBoxes().BoundingBox().CalcCenter()));
        if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForSetSimulationData))
            return false;

        // Propagate changes made in GetExtents() from outgoing CallRender3DGL (cr3d_out) to incoming  CallRender3DGL (cr3d_in) => Bboxes and times.
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


bool TrackingShotRenderer::Render(megamol::core_gl::view::CallRender3DGL& call) {

    auto cr3d_out = this->chainRenderSlot.CallAs<core_gl::view::CallRender3DGL>();
    if (cr3d_out == nullptr)
        return false;

    // Get update data from keyframe keeper -----------------------------------
    auto ccc = this->keyframeKeeperSlot.CallAs<cinematic::CallKeyframeKeeper>();
    if (ccc == nullptr)
        return false;
    if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForGetUpdatedKeyframeData))
        return false;

    // Set total simulation time
    float totalSimTime = static_cast<float>(cr3d_out->TimeFramesCount());
    ccc->SetTotalSimTime(totalSimTime);
    if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForSetSimulationData))
        return false;

    // Get selected keyframe
    cinematic::Keyframe skf = ccc->GetSelectedKeyframe();

    // Set current simulation time based on selected keyframe ('disables'/ignores animation via view3d)
    float simTime = skf.GetSimTime();
    call.SetTime(simTime * totalSimTime);

    // Get pointer to keyframes array
    auto keyframes = ccc->GetKeyframes();
    if (keyframes == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[TRACKINGSHOT RENDERER] [Render] Pointer to keyframe array is nullptr.");
        return false;
    }

    // Update parameters ------------------------------------------------------
    if (this->stepsParam.IsDirty()) {
        this->interpolSteps = this->stepsParam.Param<param::IntParam>()->Value();
        ccc->SetInterpolationSteps(this->interpolSteps);
        if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForGetInterpolCamPositions))
            return false;
        this->stepsParam.ResetDirty();
    }

    if (this->toggleHelpTextParam.IsDirty()) {
        this->showHelpText = !this->showHelpText;
        this->toggleHelpTextParam.ResetDirty();
    }

    // Init rendering ---------------------------------------------------------
    auto const lhsFBO = call.GetFramebuffer();

    glm::vec4 back_color;
    glGetFloatv(GL_COLOR_CLEAR_VALUE, static_cast<GLfloat*>(glm::value_ptr(back_color)));
    this->utils.SetBackgroundColor(back_color);

    // Get current camera
    core::view::Camera camera = call.GetCamera();
    auto view = camera.getViewMatrix();
    auto proj = camera.getProjectionMatrix();
    glm::mat4 mvp = proj * view;
    auto cam_pose = camera.get<core::view::Camera::Pose>();

    // Get current viewport
    const float vp_fw = static_cast<float>(lhsFBO->getWidth());
    const float vp_fh = static_cast<float>(lhsFBO->getHeight());
    const glm::vec2 vp_dim = {vp_fw, vp_fh};

    // Get matrix for orthogonal projection of 2D rendering
    glm::mat4 ortho = glm::ortho(0.0f, vp_fw, 0.0f, vp_fh, -1.0f, 1.0f);

    // Push manipulators ------------------------------------------------------
    if (keyframes->size() > 0) {
        this->manipulators.UpdateRendering(keyframes, skf, ccc->GetStartControlPointPosition(),
            ccc->GetEndControlPointPosition(), camera, vp_dim, mvp);
        this->manipulators.PushRendering(this->utils);
    }

    // Push spline ------------------------------------------------------------
    auto interpolKeyframes = ccc->GetInterpolCamPositions();
    if (interpolKeyframes == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[TRACKINGSHOT RENDERER] [Render] Pointer to interpolated camera positions array is nullptr.");
        return false;
    }
    auto color = this->utils.Color(CinematicUtils::Colors::KEYFRAME_SPLINE);
    auto keyframeCount = interpolKeyframes->size();
    if (keyframeCount > 1) {
        for (int i = 0; i < (keyframeCount - 1); i++) {
            glm::vec3 start = interpolKeyframes->operator[](i);
            glm::vec3 end = interpolKeyframes->operator[](i + 1);
            this->utils.PushLinePrimitive(start, end, this->lineWidth, cam_pose.direction, cam_pose.position, color);
        }
    }

    // Draw 3D ---------------------------------------------------------------
    this->utils.DrawAll(mvp, vp_dim);

    // Push hotkey list ------------------------------------------------------
    this->utils.HotkeyWindow(this->showHelpText, ortho, vp_dim);

    // Push menu --------------------------------------------------------------
    std::string leftLabel = " TRACKING SHOT ";
    std::string midLabel = "";
    std::string rightLabel = " [Shift+h] Show Hotkeys ";
    if (this->showHelpText) {
        rightLabel = " [Shift+h] Hide Hotkeys ";
    }
    this->utils.PushMenu(ortho, leftLabel, midLabel, rightLabel, vp_dim, 1.0f);

    // Draw 2D ---------------------------------------------------------------
    this->utils.DrawAll(ortho, vp_dim);

    return true;
}


bool TrackingShotRenderer::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {

    auto cr = this->chainRenderSlot.CallAs<core_gl::view::CallRender3DGL>();
    if (cr != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(core_gl::view::CallRender3DGL::FnOnMouseButton))
            return true;
    }

    auto ccc = this->keyframeKeeperSlot.CallAs<cinematic::CallKeyframeKeeper>();
    if (ccc == nullptr)
        return false;
    auto keyframes = ccc->GetKeyframes();
    if (keyframes == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[TRACKINGSHOT RENDERER] [OnMouseButton] Pointer to keyframe array is nullptr.");
        return false;
    }

    bool consumed = false;

    bool down = (action == core::view::MouseButtonAction::PRESS);
    if (button == MouseButton::BUTTON_LEFT) {
        if (down) {
            if (!this->skipped_first_mouse_interact) {
                this->skipped_first_mouse_interact = true;
                return false;
            }

            // Check if manipulator is selected
            if (this->manipulators.CheckForHitManipulator(this->mouseX, this->mouseY)) {
                this->manipulatorGrabbed = true;
                consumed = true;
                //megamol::core::utility::log::Log::DefaultLog.WriteInfo("[TRACKINGSHOT RENDERER] [OnMouseButton] MANIPULATOR SELECTED.");
            } else {
                // Check if new keyframe position is selected
                int index = this->manipulators.GetSelectedKeyframePositionIndex(this->mouseX, this->mouseY);
                if (index >= 0) {
                    ccc->SetSelectedKeyframeTime((*keyframes)[index].GetAnimTime());
                    if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime))
                        return false;
                    consumed = true;
                    //megamol::core::utility::log::Log::DefaultLog.WriteInfo("[TRACKINGSHOT RENDERER] [OnMouseButton] KEYFRAME SELECT.");
                }
            }
        } else {
            // Apply changes of selected manipulator and control points
            if (this->manipulatorGrabbed) {

                ccc->SetSelectedKeyframe(this->manipulators.GetManipulatedSelectedKeyframe());
                if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForSetSelectedKeyframe))
                    return false;

                ccc->SetControlPointPosition(this->manipulators.GetFirstControlPointPosition(),
                    this->manipulators.GetLastControlPointPosition());
                if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForSetCtrlPoints))
                    return false;

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

    auto cr = this->chainRenderSlot.CallAs<core_gl::view::CallRender3DGL>();
    if (cr != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;
        cr->SetInputEvent(evt);
        if ((*cr)(core_gl::view::CallRender3DGL::FnOnMouseMove))
            return true;
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
