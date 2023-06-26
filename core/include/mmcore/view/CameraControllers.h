/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <glm/gtx/quaternion.hpp>    // glm::rotate(quat, vector)
#include <glm/gtx/rotate_vector.hpp> // glm::rotate(quaternion, vector)

#include "KeyboardMouseInput.h"
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/utility/graphics/CameraUtils.h"
#include "mmcore/view/Camera.h"
#include "mmcore/view/cam_typedefs.h"

namespace megamol::core::view {

using megamol::frontend_resources::Key;
using megamol::frontend_resources::KeyAction;
using megamol::frontend_resources::Modifier;
using megamol::frontend_resources::Modifiers;
using megamol::frontend_resources::MouseButton;
using megamol::frontend_resources::MouseButtonAction;

class Camera3DController {
public:
    Camera3DController(Camera* target_camera, bool arcball_is_default = false)
            : _viewKeyMoveStepSlot("viewKey::MoveStep", "The move step size in world coordinates")
            , _viewKeyRunFactorSlot(
                  "viewKey::RunFactor", "The factor for step size multiplication when running (shift)")
            , _viewKeyAngleStepSlot("viewKey::AngleStep", "The angle rotate step in degrees")
            , _viewKeyFixToWorldUpSlot("viewKey::FixToWorldUp", "Fix rotation manipulator to world up vector")
            , _mouseSensitivitySlot("viewKey::MouseSensitivity", "used for WASD mode")
            , _viewKeyRotPointSlot("viewKey::RotPoint", "The point around which the view will be rotated")
            , _target_camera(target_camera)
            , _cameraControlOverrideActive(false)
            , _arcballDefault(false)
            , _cameraSetViewChooserParam("view::defaultView", "Choose a default view to look from")
            , _cameraSetOrientationChooserParam("view::defaultOrientation", "Choose a default orientation to look from")
            , _cameraViewOrientationParam("view::cubeOrientation", "Current camera orientation used for view cube.")
            , _cameraPositionParam("cam::position", "")
            , _cameraOrientationParam("cam::orientation", "")
            , _cameraProjectionTypeParam("cam::projectiontype", "")
            , _cameraNearPlaneParam("cam::nearplane", "")
            , _cameraFarPlaneParam("cam::farplane", "")
            , _cameraHalfApertureDegreesParam("cam::halfaperturedegrees", "") {

        this->_viewKeyMoveStepSlot.SetParameter(new param::FloatParam(0.5f, 0.001f));

        this->_viewKeyRunFactorSlot.SetParameter(new param::FloatParam(2.0f, 0.1f));

        this->_viewKeyAngleStepSlot.SetParameter(new param::FloatParam(90.0f, 0.1f, 360.0f));

        this->_viewKeyFixToWorldUpSlot.SetParameter(new param::BoolParam(true));

        this->_mouseSensitivitySlot.SetParameter(new param::FloatParam(3.0f, 0.001f, 10.0f));

        // TODO clean up vrpsev memory after use
        param::EnumParam* vrpsev = new param::EnumParam(1);
        vrpsev->SetTypePair(0, "Position");
        vrpsev->SetTypePair(1, "Look-At");
        this->_viewKeyRotPointSlot.SetParameter(vrpsev);

        _translateManipulator.set_target(_target_camera);
        _translateManipulator.enable();

        _rotateManipulator.set_target(_target_camera);
        _rotateManipulator.enable();

        _arcballManipulator.set_target(_target_camera);
        _arcballManipulator.enable();
        _rotCenter = glm::vec3(0.0f, 0.0f, 0.0f);

        _turntableManipulator.set_target(_target_camera);
        _turntableManipulator.enable();

        _orbitAltitudeManipulator.set_target(_target_camera);
        _orbitAltitudeManipulator.enable();

        using namespace utility;
        auto defaultViewParam = new param::EnumParam(0);
        defaultViewParam->SetTypePair(DEFAULTVIEW_FACE_FRONT, "FACE - Front");
        defaultViewParam->SetTypePair(DEFAULTVIEW_FACE_BACK, "FACE - Back");
        defaultViewParam->SetTypePair(DEFAULTVIEW_FACE_RIGHT, "FACE - Right");
        defaultViewParam->SetTypePair(DEFAULTVIEW_FACE_LEFT, "FACE - Left");
        defaultViewParam->SetTypePair(DEFAULTVIEW_FACE_TOP, "FACE - Top");
        defaultViewParam->SetTypePair(DEFAULTVIEW_FACE_BOTTOM, "FACE - Bottom");
        defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_TOP_LEFT_FRONT, "CORNER - Top Left Front");
        defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_TOP_RIGHT_FRONT, "CORNER - Top Right Front");
        defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_TOP_LEFT_BACK, "CORNER - Top Left Back");
        defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_TOP_RIGHT_BACK, "CORNER - Top Right Back");
        defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_BOTTOM_LEFT_FRONT, "CORNER - Bottom Left Front");
        defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_BOTTOM_RIGHT_FRONT, "CORNER - Bottom Right Front");
        defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_BOTTOM_LEFT_BACK, "CORNER - Bottom Left Back");
        defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_BOTTOM_RIGHT_BACK, "CORNER - Bottom Right Back");
        defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_TOP_FRONT, "EDGE - Top Front");
        defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_TOP_LEFT, "EDGE - Top Left");
        defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_TOP_RIGHT, "EDGE - Top Right");
        defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_TOP_BACK, "EDGE - Top Back");
        defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_BOTTOM_FRONT, "EDGE - Bottom Front");
        defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_BOTTOM_LEFT, "EDGE - Bottom Left");
        defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_BOTTOM_RIGHT, "EDGE - Bottom Right");
        defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_BOTTOM_BACK, "EDGE - Bottom Back");
        defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_FRONT_LEFT, "EDGE - Front Left");
        defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_FRONT_RIGHT, "EDGE - Front Right");
        defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_BACK_LEFT, "EDGE - Back Left");
        defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_BACK_RIGHT, "EDGE - Back Right");
        this->_cameraSetViewChooserParam.SetParameter(defaultViewParam);
        // this->_cameraSetViewChooserParam.SetUpdateCallback(&AbstractView::OnResetView);

        auto defaultOrientationParam = new param::EnumParam(0);
        defaultOrientationParam->SetTypePair(DEFAULTORIENTATION_TOP, "Top");
        defaultOrientationParam->SetTypePair(DEFAULTORIENTATION_RIGHT, "Right");
        defaultOrientationParam->SetTypePair(DEFAULTORIENTATION_BOTTOM, "Bottom");
        defaultOrientationParam->SetTypePair(DEFAULTORIENTATION_LEFT, "Left");
        this->_cameraSetOrientationChooserParam.SetParameter(defaultOrientationParam);
        // this->_cameraSetOrientationChooserParam.SetUpdateCallback(&AbstractView::OnResetView);

        this->_cameraViewOrientationParam.SetParameter(
            new param::Vector4fParam(vislib::math::Vector<float, 4>(0.0f, 0.0f, 0.0f, 1.0f)));
        this->_cameraViewOrientationParam.Parameter()->SetGUIReadOnly(true);
        this->_cameraViewOrientationParam.Parameter()->SetGUIVisible(false);


        const bool camparamvisibility = true;

        auto camposparam = new param::Vector3fParam(vislib::math::Vector<float, 3>());
        camposparam->SetGUIVisible(camparamvisibility);
        this->_cameraPositionParam.SetParameter(camposparam);

        auto camorientparam = new param::Vector4fParam(vislib::math::Vector<float, 4>());
        camorientparam->SetGUIVisible(camparamvisibility);
        this->_cameraOrientationParam.SetParameter(camorientparam);

        auto projectionParam = new param::EnumParam(static_cast<int>(Camera::ProjectionType::PERSPECTIVE));
        projectionParam->SetTypePair(static_cast<int>(Camera::ProjectionType::PERSPECTIVE), "Perspective");
        projectionParam->SetTypePair(static_cast<int>(Camera::ProjectionType::ORTHOGRAPHIC), "Orthographic");
        projectionParam->SetGUIVisible(camparamvisibility);
        this->_cameraProjectionTypeParam.SetParameter(projectionParam);

        auto farplaneparam = new param::FloatParam(100.0f, 0.0f);
        farplaneparam->SetGUIVisible(camparamvisibility);
        this->_cameraFarPlaneParam.SetParameter(farplaneparam);

        auto nearplaneparam = new param::FloatParam(0.1f, 0.0f);
        nearplaneparam->SetGUIVisible(camparamvisibility);
        this->_cameraNearPlaneParam.SetParameter(nearplaneparam);

        auto apertureparam = new param::FloatParam(35.0f, 0.0f);
        apertureparam->SetGUIVisible(camparamvisibility);
        this->_cameraHalfApertureDegreesParam.SetParameter(apertureparam);
    }
    ~Camera3DController() {}

    void setTargetCamera(Camera* target_camera) {
        _target_camera = target_camera;

        _translateManipulator.set_target(_target_camera);

        _rotateManipulator.set_target(_target_camera);

        _arcballManipulator.set_target(_target_camera);

        _turntableManipulator.set_target(_target_camera);

        _orbitAltitudeManipulator.set_target(_target_camera);
    }

    void setRotationalCenter(glm::vec3 new_center) {
        _rotCenter = new_center;
    }

    glm::vec3 getRotationalCenter() {
        return _rotCenter;
    }

    void setArcballDefault(bool default_to_arcball) {
        _arcballDefault = default_to_arcball;
    }

    std::vector<AbstractSlot*> getParameterSlots() {
        return {&_viewKeyMoveStepSlot, &this->_viewKeyRunFactorSlot, &this->_viewKeyAngleStepSlot,
            &_viewKeyFixToWorldUpSlot, &this->_mouseSensitivitySlot, &this->_viewKeyRotPointSlot,
            &_cameraViewOrientationParam, &_cameraSetViewChooserParam, &_cameraSetOrientationChooserParam,
            &_cameraPositionParam, &_cameraOrientationParam, &_cameraProjectionTypeParam, &_cameraNearPlaneParam,
            &_cameraFarPlaneParam, &_cameraHalfApertureDegreesParam};
    }

    bool isActive() {
        return _arcballManipulator.manipulating() || _translateManipulator.manipulating() ||
               _rotateManipulator.manipulating() || _turntableManipulator.manipulating() ||
               _orbitAltitudeManipulator.manipulating();
    }

    bool isOverriding() {
        return _cameraControlOverrideActive;
    }

    /*
     * Performs the actual camera movement based on the pressed keys
     */
    void handleCameraMovement(float dt) {
        float step = this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->Value();
        step *= dt;

        const float runFactor = this->_viewKeyRunFactorSlot.Param<param::FloatParam>()->Value();
        if (this->_modkeys.test(Modifier::SHIFT)) {
            step *= runFactor;
        }

        bool anymodpressed = !this->_modkeys.none();
        float rotationStep = this->_viewKeyAngleStepSlot.Param<param::FloatParam>()->Value();
        rotationStep *= dt;

        glm::vec3 currCamPos(this->_target_camera->get<Camera::Pose>().position);
        float orbitalAltitude = glm::length(currCamPos - _rotCenter);

        if (this->_translateManipulator.manipulating() || this->_rotateManipulator.manipulating()) {

            if (this->_pressedKeyMap.count(Key::KEY_W) > 0 && this->_pressedKeyMap[Key::KEY_W]) {
                this->_translateManipulator.move_forward(step);
            }
            if (this->_pressedKeyMap.count(Key::KEY_S) > 0 && this->_pressedKeyMap[Key::KEY_S]) {
                this->_translateManipulator.move_forward(-step);
            }
            if (this->_pressedKeyMap.count(Key::KEY_A) > 0 && this->_pressedKeyMap[Key::KEY_A]) {
                this->_translateManipulator.move_horizontally(-step);
            }
            if (this->_pressedKeyMap.count(Key::KEY_D) > 0 && this->_pressedKeyMap[Key::KEY_D]) {
                this->_translateManipulator.move_horizontally(step);
            }
            if (this->_pressedKeyMap.count(Key::KEY_C) > 0 && this->_pressedKeyMap[Key::KEY_C]) {
                this->_translateManipulator.move_vertically(step);
            }
            if (this->_pressedKeyMap.count(Key::KEY_V) > 0 && this->_pressedKeyMap[Key::KEY_V]) {
                this->_translateManipulator.move_vertically(-step);
            }
            if (this->_pressedKeyMap.count(Key::KEY_Q) > 0 && this->_pressedKeyMap[Key::KEY_Q]) {
                this->_rotateManipulator.roll(-rotationStep);
            }
            if (this->_pressedKeyMap.count(Key::KEY_E) > 0 && this->_pressedKeyMap[Key::KEY_E]) {
                this->_rotateManipulator.roll(rotationStep);
            }
            if (this->_pressedKeyMap.count(Key::KEY_UP) > 0 && this->_pressedKeyMap[Key::KEY_UP]) {
                this->_rotateManipulator.pitch(-rotationStep);
            }
            if (this->_pressedKeyMap.count(Key::KEY_DOWN) > 0 && this->_pressedKeyMap[Key::KEY_DOWN]) {
                this->_rotateManipulator.pitch(rotationStep);
            }
            if (this->_pressedKeyMap.count(Key::KEY_LEFT) > 0 && this->_pressedKeyMap[Key::KEY_LEFT]) {
                this->_rotateManipulator.yaw(rotationStep, true);
            }
            if (this->_pressedKeyMap.count(Key::KEY_RIGHT) > 0 && this->_pressedKeyMap[Key::KEY_RIGHT]) {
                this->_rotateManipulator.yaw(-rotationStep, true);
            }
        }

        glm::vec3 newCamPos(this->_target_camera->get<Camera::Pose>().position);
        glm::vec3 camDir(this->_target_camera->get<Camera::Pose>().direction);
        _rotCenter = newCamPos + orbitalAltitude * glm::normalize(camDir);
    }

    void reset(BoundingBoxes_2 const& bboxs, float window_aspect) {
        Camera::PerspectiveParameters cam_intrinsics;
        cam_intrinsics.near_plane = 0.01f;
        cam_intrinsics.far_plane = 100.0f;
        cam_intrinsics.fovy = 0.5;
        cam_intrinsics.aspect = window_aspect;
        cam_intrinsics.image_plane_tile =
            Camera::ImagePlaneTile(); // view is in control -> no tiling -> use default tile values

        auto dv =
            static_cast<utility::DefaultView>(this->_cameraSetViewChooserParam.Param<param::EnumParam>()->Value());
        auto dor = static_cast<utility::DefaultOrientation>(
            this->_cameraSetOrientationChooserParam.Param<param::EnumParam>()->Value());

        auto cam_orientation = utility::get_default_camera_orientation(dv, dor);
        auto cam_position = utility::get_default_camera_position(bboxs, cam_intrinsics, cam_orientation, dv);
        Camera::Pose cam_pose(glm::vec3(cam_position), cam_orientation);

        auto min_max_dist = utility::get_min_max_dist_to_bbox(bboxs, cam_pose);
        cam_intrinsics.far_plane = std::max(0.0f, min_max_dist.y);
        cam_intrinsics.near_plane = std::max(cam_intrinsics.far_plane / 10000.0f, min_max_dist.x);


        *_target_camera = Camera(cam_pose, cam_intrinsics);

        auto bbc = bboxs.BoundingBox().CalcCenter();
        setRotationalCenter(glm::vec3(bbc.GetX(), bbc.GetY(), bbc.GetZ()));
    }

    void setParameterSlotsFromCamera() {
        auto cam_pose = _target_camera->get<Camera::Pose>();

        glm::vec3 pos = cam_pose.position;
        const bool makeDirty = false;
        this->_cameraPositionParam.Param<param::Vector3fParam>()->SetValue(
            vislib::math::Vector<float, 3>(pos.x, pos.y, pos.z), makeDirty);
        // TODO before here we made a consume on the param, which did nothing internally.
        //  But without dirty flag camera changes are currently not automatically detectable. Is this wanted behavior?

        glm::quat orientation = cam_pose.to_quat();
        this->_cameraOrientationParam.Param<param::Vector4fParam>()->SetValue(
            vislib::math::Vector<float, 4>(orientation.x, orientation.y, orientation.z, orientation.w), makeDirty);

        this->_cameraViewOrientationParam.Param<param::Vector4fParam>()->SetValue(
            vislib::math::Vector<float, 4>(orientation.x, orientation.y, orientation.z, orientation.w), makeDirty);

        auto cam_proj_type = _target_camera->get<Camera::ProjectionType>();
        this->_cameraProjectionTypeParam.Param<param::EnumParam>()->SetValue(
            static_cast<int>(cam_proj_type), makeDirty);

        /*this->cameraNearPlaneParam.Param<param::FloatParam>()->SetValue(cam.near_clipping_plane(), makeDirty);
        this->cameraFarPlaneParam.Param<param::FloatParam>()->SetValue(cam.far_clipping_plane(), makeDirty);*/
        /*this->cameraEyeParam.Param<param::EnumParam>()->SetValue(static_cast<int>(cam.eye()), makeDirty);
        this->cameraGateScalingParam.Param<param::EnumParam>()->SetValue(static_cast<int>(cam.gate_scaling()),
        makeDirty); this->cameraFilmGateParam.Param<param::Vector2fParam>()->SetValue(
        vislib::math::Vector<float, 2>(cam.film_gate().width(), cam.film_gate().height()), makeDirty);*/
        /*this->cameraResolutionXParam.Param<param::IntParam>()->SetValue(cam.resolution_gate().width());
        this->cameraResolutionYParam.Param<param::IntParam>()->SetValue(cam.resolution_gate().height());*/

        if (cam_proj_type == Camera::ProjectionType::PERSPECTIVE) {
            auto cam_intrinsics = _target_camera->get<Camera::PerspectiveParameters>();
            this->_cameraHalfApertureDegreesParam.Param<param::FloatParam>()->SetValue(
                cam_intrinsics.fovy * 180.0f / 3.14159265359 /*TODO*/, makeDirty);

            this->_cameraNearPlaneParam.Param<param::FloatParam>()->SetValue(cam_intrinsics.near_plane, makeDirty);
            this->_cameraFarPlaneParam.Param<param::FloatParam>()->SetValue(cam_intrinsics.far_plane, makeDirty);
        }
    }

    void applyParameterSlotsToCamera(BoundingBoxes_2 const& bboxs) {
        // Set default view and reset camera if params are dirty
        // Note: Has lowest priority, i.e. if other camera parameters are dirty in the same frame, they
        // will overwrite the reseted camera. Important for expected behaviour when loading project files.
        if (_cameraSetOrientationChooserParam.IsDirty() || _cameraSetViewChooserParam.IsDirty()) {
            _cameraSetOrientationChooserParam.ResetDirty();
            _cameraSetViewChooserParam.ResetDirty();
            reset(bboxs, _target_camera->get<Camera::AspectRatio>());
        }

        // Set pose
        auto cam_pose = _target_camera->get<Camera::Pose>();
        if (this->_cameraPositionParam.IsDirty()) {
            auto val = this->_cameraPositionParam.Param<param::Vector3fParam>()->Value();
            cam_pose.position = glm::vec3(val.GetX(), val.GetY(), val.GetZ());
            this->_cameraPositionParam.ResetDirty();
        }
        if (this->_cameraOrientationParam.IsDirty()) {
            auto val = this->_cameraOrientationParam.Param<param::Vector4fParam>()->Value();
            const auto orientation = glm::quat(val.GetW(), val.GetX(), val.GetY(), val.GetZ());
            cam_pose = Camera::Pose(cam_pose.position, orientation);
            this->_cameraOrientationParam.ResetDirty();
        }
        _target_camera->setPose(cam_pose);

        // Set intrinsics
        if (_target_camera->get<Camera::ProjectionType>() != Camera::UNKNOWN) {
            auto curr_proj_type = _target_camera->get<Camera::ProjectionType>();
            auto cam_pose = _target_camera->get<Camera::Pose>();
            float orbitalAltitude = glm::length(cam_pose.position - _rotCenter);

            // Intialize with current camera state
            float fovy;
            float vertical_height;
            float aspect = _target_camera->get<Camera::AspectRatio>();
            float near_plane = _target_camera->get<Camera::NearPlane>();
            float far_plane = _target_camera->get<Camera::FarPlane>();
            Camera::ImagePlaneTile tile = _target_camera->get<Camera::ImagePlaneTile>();

            if (curr_proj_type == Camera::ProjectionType::PERSPECTIVE) {
                fovy = _target_camera->get<Camera::PerspectiveParameters>().fovy;
                vertical_height = std::tan(fovy) * orbitalAltitude;
            } else if (curr_proj_type == Camera::ProjectionType::ORTHOGRAPHIC) {
                vertical_height = _target_camera->get<Camera::OrthographicParameters>().frustrum_height;
                fovy = std::atan(vertical_height / orbitalAltitude);
            }

            if (true) // TODO
            {
                // compute auto-adjusted near far plane
                auto min_max_dist = utility::get_min_max_dist_to_bbox(bboxs, cam_pose);
                far_plane = std::max(0.0f, min_max_dist.y);
                near_plane = std::max(far_plane / 10000.0f, min_max_dist.x);
            } else {
                // set near far based on UI param input
                if (_cameraNearPlaneParam.IsDirty()) {
                    near_plane = _cameraNearPlaneParam.Param<param::FloatParam>()->Value();
                    _cameraNearPlaneParam.ResetDirty();
                }
                if (_cameraFarPlaneParam.IsDirty()) {
                    far_plane = _cameraFarPlaneParam.Param<param::FloatParam>()->Value();
                    _cameraFarPlaneParam.ResetDirty();
                }
            }

            if (_cameraHalfApertureDegreesParam.IsDirty()) {
                fovy =
                    this->_cameraHalfApertureDegreesParam.Param<param::FloatParam>()->Value() * 3.14159265359 / 180.0f;
                _cameraHalfApertureDegreesParam.ResetDirty();
            }

            // projection param is used every frame, but reset if dirty anyway
            if (this->_cameraProjectionTypeParam.IsDirty()) {
                this->_cameraProjectionTypeParam.ResetDirty();
            }

            auto proj_type = static_cast<Camera::ProjectionType>(
                this->_cameraProjectionTypeParam.Param<param::EnumParam>()->Value());
            if (proj_type == Camera::PERSPECTIVE) {
                Camera::PerspectiveParameters cam_intrinsics;
                cam_intrinsics.aspect = aspect;
                cam_intrinsics.fovy = fovy;
                cam_intrinsics.near_plane = near_plane;
                cam_intrinsics.far_plane = far_plane;
                cam_intrinsics.image_plane_tile = tile;

                *_target_camera = Camera(cam_pose, cam_intrinsics);
            } else if (proj_type == Camera::ORTHOGRAPHIC) {
                Camera::OrthographicParameters cam_intrinsics;
                cam_intrinsics.aspect = aspect;
                cam_intrinsics.frustrum_height = vertical_height;
                cam_intrinsics.near_plane = near_plane;
                cam_intrinsics.far_plane = far_plane;
                cam_intrinsics.image_plane_tile = tile;

                *_target_camera = Camera(cam_pose, cam_intrinsics);
            }
        }
    }

    /**
     * This event handler can be reimplemented to receive key code events.
     */
    virtual void OnKey(Key key, KeyAction action, Modifiers mods) {
        if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
            this->_pressedKeyMap[key] = true;
        } else if (action == view::KeyAction::RELEASE) {
            this->_pressedKeyMap[key] = false;
        }

        if (key == view::Key::KEY_LEFT_ALT || key == view::Key::KEY_RIGHT_ALT) {
            if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
                _modkeys.set(view::Modifier::ALT);
                _cameraControlOverrideActive = true;
            } else if (action == view::KeyAction::RELEASE) {
                _modkeys.reset(view::Modifier::ALT);
                _cameraControlOverrideActive = false;
            }
        }
        if (key == view::Key::KEY_LEFT_SHIFT || key == view::Key::KEY_RIGHT_SHIFT) {
            if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
                _modkeys.set(view::Modifier::SHIFT);
            } else if (action == view::KeyAction::RELEASE) {
                _modkeys.reset(view::Modifier::SHIFT);
            }
        }
        if (key == view::Key::KEY_LEFT_CONTROL || key == view::Key::KEY_RIGHT_CONTROL) {
            if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
                _modkeys.set(view::Modifier::CTRL);
                _cameraControlOverrideActive = true;
            } else if (action == view::KeyAction::RELEASE) {
                _modkeys.reset(view::Modifier::CTRL);
                _cameraControlOverrideActive = false;
            }
        }
    }

    /**
     * This event handler can be reimplemented to receive unicode events.
     */
    virtual void OnChar(unsigned int codePoint) {}

    /**
     * This event handler can be reimplemented to receive mouse button events.
     */
    virtual void OnMouseButton(
        MouseButton button, MouseButtonAction action, Modifiers mods, int wndWidth, int wndHeight) {
        if (action == view::MouseButtonAction::PRESS) {
            this->_pressedMouseMap[button] = true;
        } else if (action == view::MouseButtonAction::RELEASE) {
            this->_pressedMouseMap[button] = false;
        }

        // This mouse handling/mapping is so utterly weird and should die!
        auto down = action == view::MouseButtonAction::PRESS;
        bool altPressed = mods.test(view::Modifier::ALT);   // _modkeys.test(view::Modifier::ALT);
        bool ctrlPressed = mods.test(view::Modifier::CTRL); // _modkeys.test(view::Modifier::CTRL);

        switch (button) {
        case megamol::core::view::MouseButton::BUTTON_LEFT:
            if (!isActive()) {
                if (altPressed ^ (this->_arcballDefault && !ctrlPressed)) // Left mouse press + alt/arcDefault+noCtrl ->
                                                                          // activate arcball manipluator
                {
                    this->_arcballManipulator.setActive(
                        wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
                } else if (ctrlPressed) // Left mouse press + Ctrl -> activate orbital manipluator
                {
                    this->_turntableManipulator.setActive(
                        wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
                }
            }

            break;
        case megamol::core::view::MouseButton::BUTTON_RIGHT:
            if (!isActive()) {
                if ((altPressed ^ this->_arcballDefault) || ctrlPressed) {
                    this->_orbitAltitudeManipulator.setActive(
                        wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
                } else {
                    this->_rotateManipulator.setActive();
                    this->_translateManipulator.setActive(
                        wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
                }
            }

            break;
        case megamol::core::view::MouseButton::BUTTON_MIDDLE:
            if (!isActive()) {
                this->_translateManipulator.setActive(
                    wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
            }

            break;
        default:
            break;
        }

        if (action == view::MouseButtonAction::RELEASE) // Mouse release + no other mouse button pressed ->
                                                        // deactivate all mouse manipulators
        {
            if (!(_pressedMouseMap[MouseButton::BUTTON_LEFT] || _pressedMouseMap[MouseButton::BUTTON_MIDDLE] ||
                    _pressedMouseMap[MouseButton::BUTTON_RIGHT])) {
                this->_arcballManipulator.setInactive();
                this->_orbitAltitudeManipulator.setInactive();
                this->_rotateManipulator.setInactive();
                this->_turntableManipulator.setInactive();
                this->_translateManipulator.setInactive();
            }
        }
    }

    /**
     * This event handler can be reimplemented to receive mouse move events.
     */
    virtual void OnMouseMove(double x, double y, int wndWidth, int wndHeight) {
        this->_mouseX = (float)static_cast<int>(x);
        this->_mouseY = (float)static_cast<int>(y);

        glm::vec3 newPos;

        if (this->_turntableManipulator.manipulating()) {
            this->_turntableManipulator.on_drag(wndWidth - static_cast<int>(this->_mouseX),
                static_cast<int>(this->_mouseY), glm::vec4(_rotCenter, 1.0), wndWidth, wndHeight);
        }

        if (this->_arcballManipulator.manipulating()) {
            this->_arcballManipulator.on_drag(wndWidth - static_cast<int>(this->_mouseX),
                static_cast<int>(this->_mouseY), glm::vec4(_rotCenter, 1.0));
        }

        if (this->_orbitAltitudeManipulator.manipulating()) {
            this->_orbitAltitudeManipulator.on_drag(wndWidth - static_cast<int>(this->_mouseX),
                static_cast<int>(this->_mouseY), glm::vec4(_rotCenter, 1.0));
        }

        if (this->_translateManipulator.manipulating() && !this->_rotateManipulator.manipulating()) {

            // compute proper step size by computing pixel world size at distance to rotCenter
            auto projType = _target_camera->get<Camera::ProjectionType>();
            glm::vec3 currCamPos = this->_target_camera->get<Camera::Pose>().position;
            float orbitalAltitude = glm::length(currCamPos - _rotCenter);
            float pixel_world_size;
            if (projType == Camera::ProjectionType::PERSPECTIVE) {
                auto fovy = _target_camera->get<Camera::PerspectiveParameters>().fovy;
                auto vertical_height = std::tan(fovy) * orbitalAltitude;
                pixel_world_size = vertical_height / static_cast<float>(wndHeight);
            } else if (projType == Camera::ProjectionType::ORTHOGRAPHIC) {
                auto vertical_height = _target_camera->get<Camera::OrthographicParameters>().frustrum_height;
                pixel_world_size = vertical_height / static_cast<float>(wndHeight);
            } else {
                // Oh bother...
            }

            this->_translateManipulator.set_step_size(pixel_world_size);

            this->_translateManipulator.move_horizontally(wndWidth - static_cast<int>(this->_mouseX));
            this->_translateManipulator.move_vertically(static_cast<int>(this->_mouseY));
        }
    }

    /**
     * This event handler can be reimplemented to receive mouse scroll events.
     */
    virtual void OnMouseScroll(double dx, double dy) {
        // This mouse handling/mapping is so utterly weird and should die!
        if ((abs(dy) > 0.0)) {
            if (this->_rotateManipulator.manipulating()) {
                this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->SetValue(
                    this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->Value() +
                    (dy * 0.1f * this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->Value()));
            } else {
                auto cam_pose = _target_camera->get<Camera::Pose>();
                auto v = glm::normalize(_rotCenter - cam_pose.position);
                auto altitude = glm::length(_rotCenter - cam_pose.position);
                cam_pose.position = cam_pose.position + (v * static_cast<float>(dy) * (altitude / 50.0f));
                _target_camera->setPose(cam_pose);
            }
        }
    }

private:
    /** The move step size in world coordinates */
    param::ParamSlot _viewKeyMoveStepSlot;

    param::ParamSlot _viewKeyRunFactorSlot;

    /** The angle rotate step in degrees */
    param::ParamSlot _viewKeyAngleStepSlot;

    param::ParamSlot _viewKeyFixToWorldUpSlot;

    /** sensitivity for mouse rotation in WASD mode */
    param::ParamSlot _mouseSensitivitySlot;

    /** The point around which the view will be roateted */
    param::ParamSlot _viewKeyRotPointSlot;

    /** Standard camera views */
    param::ParamSlot _cameraViewOrientationParam;
    param::ParamSlot _cameraSetViewChooserParam;
    param::ParamSlot _cameraSetOrientationChooserParam;

    /** Invisible parameters for lua manipulation */
    param::ParamSlot _cameraPositionParam;
    param::ParamSlot _cameraOrientationParam;
    param::ParamSlot _cameraProjectionTypeParam;
    param::ParamSlot _cameraNearPlaneParam;
    param::ParamSlot _cameraFarPlaneParam;
    param::ParamSlot _cameraHalfApertureDegreesParam;


    //////
    // Individual manipulators
    //////
    /** The orbital arcball manipulator for the camera */
    arcball_type _arcballManipulator;

    /** The translation manipulator for the camera */
    xlate_type _translateManipulator;

    /** The rotation manipulator for the camera */
    rotate_type _rotateManipulator;

    /** The orbital manipulator turntable for the camera */
    turntable_type _turntableManipulator;

    /** The manipulator for changing the orbital altitude */
    orbit_altitude_type _orbitAltitudeManipulator;

    /** Center of rotation for orbital manipulators */
    glm::vec3 _rotCenter;

    /** Flag determining whether the arcball is the default steering method of the camera */
    bool _arcballDefault;


    //////
    // Track controller state...
    //////

    bool _cameraControlOverrideActive;

    /** The camera that is currently controlled */
    Camera* _target_camera; // TODO not liking this...

    //////
    // Mirror mouse and keyboard state...
    //////

    /** The mouse x coordinate */
    float _mouseX;

    /** The mouse y coordinate */
    float _mouseY;

    /** Map storing the pressed state of all keyboard buttons */
    std::map<view::Key, bool> _pressedKeyMap;

    /** the set input modifiers*/
    core::view::Modifiers _modkeys;

    /** Map storing the pressed state of all mouse buttons */
    std::map<view::MouseButton, bool> _pressedMouseMap;
};

class Camera2DController {
public:
    Camera2DController(Camera* target_camera) : _target_camera(target_camera) {}
    ~Camera2DController() {}

    std::vector<AbstractSlot*> getParameterSlots() {
        return {};
    }

    bool isActive() {
        return false; //TODO
    }

    bool isOverriding() {
        return false; //TODO
    }

    void reset(BoundingBoxes_2 bboxs, float window_aspect) {
        Camera::OrthographicParameters cam_intrinsics;
        cam_intrinsics.near_plane = 0.1f;
        cam_intrinsics.far_plane = 100.0f;
        cam_intrinsics.frustrum_height = bboxs.BoundingBox().Height();
        cam_intrinsics.aspect = window_aspect;
        cam_intrinsics.image_plane_tile =
            Camera::ImagePlaneTile(); // view is in control -> no tiling -> use default tile values

        if (window_aspect < (bboxs.BoundingBox().Width()) / bboxs.BoundingBox().Height()) {
            cam_intrinsics.frustrum_height = bboxs.BoundingBox().Width() / cam_intrinsics.aspect;
        }

        Camera::Pose cam_pose;
        cam_pose.position = glm::vec3(0.5f * (bboxs.BoundingBox().Right() + bboxs.BoundingBox().Left()),
            0.5f * (bboxs.BoundingBox().Top() + bboxs.BoundingBox().Bottom()), 1.0f);
        cam_pose.direction = glm::vec3(0.0, 0.0, -1.0);
        cam_pose.up = glm::vec3(0.0, 1.0, 0.0);

        *_target_camera = Camera(cam_pose, cam_intrinsics);
    }

    /**
     * This event handler can be reimplemented to receive key code events.
     */
    virtual void OnKey(Key key, KeyAction action, Modifiers mods) {
        _ctrlDown = mods.test(core::view::Modifier::CTRL);
    }

    /**
     * This event handler can be reimplemented to receive unicode events.
     */
    virtual void OnChar(unsigned int codePoint) {}

    /**
     * This event handler can be reimplemented to receive mouse button events.
     */
    virtual void OnMouseButton(
        MouseButton button, MouseButtonAction action, Modifiers mods, int wndWidth, int wndHeight) {
        this->_mouseMode = MouseMode::Propagate;

        if (_ctrlDown) {
            auto down = action == MouseButtonAction::PRESS;
            if (button == MouseButton::BUTTON_LEFT && down) {
                this->_mouseMode = MouseMode::Pan;
            } else if (button == MouseButton::BUTTON_MIDDLE && down) {
                this->_mouseMode = MouseMode::Zoom;
            }
        } else {
            if (button == MouseButton::BUTTON_MIDDLE && action == MouseButtonAction::PRESS) {
                this->_mouseMode = MouseMode::Pan;
            }
        }
    }

    /**
     * This event handler can be reimplemented to receive mouse move events.
     */
    virtual void OnMouseMove(double x, double y, int wndWidth, int wndHeight) {
        if (this->_mouseMode == MouseMode::Pan) {

            // compute size of a pixel in world space
            float stepSize = _target_camera->get<Camera::OrthographicParameters>().frustrum_height / wndHeight;
            auto dx = (this->_mouseX - x) * stepSize;
            auto dy = (this->_mouseY - y) * stepSize;

            auto cam_pose = _target_camera->get<Camera::Pose>();
            cam_pose.position += glm::vec3(dx, -dy, 0.0f);

            _target_camera->setPose(cam_pose);


        } else if (this->_mouseMode == MouseMode::Zoom) {

            auto dy = (this->_mouseY - y);

            auto cam_pose = _target_camera->get<Camera::Pose>();
            auto cam_intrinsics = _target_camera->get<Camera::OrthographicParameters>();

            cam_intrinsics.frustrum_height -= (dy / wndHeight) * (cam_intrinsics.frustrum_height);

            *_target_camera = Camera(cam_pose, cam_intrinsics);
        }

        this->_mouseX = x;
        this->_mouseY = y;
    }

    /**
     * This event handler can be reimplemented to receive mouse scroll events.
     */
    virtual void OnMouseScroll(double dx, double dy) {
        auto cam_pose = _target_camera->get<Camera::Pose>();
        auto cam_intrinsics = _target_camera->get<Camera::OrthographicParameters>();
        cam_intrinsics.frustrum_height -= (dy / 10.0) * (cam_intrinsics.frustrum_height);

        *_target_camera = Camera(cam_pose, cam_intrinsics);
    }

    enum MouseMode : uint8_t { Propagate, Pan, Zoom };

private:
    /** The camera that is currently controlled */
    Camera* _target_camera; // TODO not liking this...

    /** Track state of ctrl key for camera controls */
    bool _ctrlDown;

    /** The mouse drag mode */
    MouseMode _mouseMode;

    /** The mouse x coordinate */
    float _mouseX;

    /** The mouse y coordinate */
    float _mouseY;
};

} // namespace megamol::core::view
