#pragma once

#include "mmcore/view/Camera.h"

#include <optional>

namespace camera_controllers {

// Input Concept: IAAC
// (Device ->) Inputs -> Axis -> Action -> Controller/Command (-> {Camera, Handler, ...})

// Axes are inputs derived from raw device inputs (e.g., which come from GLFW)
// Axes have a normalized range of values (while device inputs may have arbitrary values)
// and may be derived/constructed from raw device inputs in an application-specific way
// Example: keyboard keys W and S move should move the camera forward/backward
// The Axis1D with range [-1.0, 1.0] to represent the input to move in direction of the
// camera forward vector can be defined as 1.0 * key_pressed(W) - 1.0 * key_pressed(S)
// other input device sources for the same Axis1D semantics may come from controller sticks
// or motion sensors, each with inidivual min/max value ranges coming from the hardware

/// Axis0D range is considered [false, true]
using Axis0D = bool;

/// Axis values are considered to be in the range [-1.0f, 1.0f]
using Axis1D = float;

/// Axis values are considered to be in the range [-1.0f, 1.0f]
using Axis2D = glm::vec2;

/// Axis values are considered to be in the range [-1.0f, 1.0f]
using Axis3D = glm::vec3;

Axis1D to_axis1d(bool const b);
Axis1D to_axis1d(bool const min, bool const max);

// While Axes are considered as abstract inputs in a normalized range,
// Actions are Inputs interpreted in the context of the framework or scene,
// i.e. they have specific semantics in context of the things they are applied to.
// Actions encode data that serves as input to controllers.
// Controllers implement behaviour on scene objects or application functionality
// and are the final step to transform inputs to
// state changes in the program observable by the user.
// Example: Action1D to move the camera forward in the scene can be derived
// form an Axis1D in range [-1.0, 1.0] by means of a default distance the camera
// is supposed to move upon user inputs, i.e. a movement delta that fits the scale
// of the scene and user intentions.
// Action1D move_camera = axis1d_inputs * camera_step_size
// Camera::Pose new_pose = controller::camera_translation(old_pose, move_camera)
// i.e. scaling of Axes derived from device inputs needs to be done in
// context of the scale of the scene and thus is a different concept than Axes
using Action0D = bool;
using Action1D = float;
using Action2D = glm::vec2;
using Action3D = glm::vec3;

Action1D simple_screen_delta_to_rotation_rad(Axis1D const delta);
Action2D simple_screen_delta_to_rotation_rad(Axis2D const delta);
Action3D simple_screen_delta_to_rotation_rad(Axis3D const delta);

Action1D screen_fov_to_rotation_rad(Axis1D const delta, float const fov_rad);
Action2D screen_fov_to_rotation_rad(Axis2D const delta, glm::vec2 const fov_rad);
Action3D screen_fov_to_rotation_rad(Axis3D const delta, glm::vec3 const fov_rad);

using Camera = megamol::core::view::Camera;
using Pose = Camera::Pose;

struct arcball {
    glm::vec3 rotation_center = {0.0f, 0.0f, 0.0f};

    Pose apply(Pose pose, Action2D const rotation_rad);
};

struct orbit_altitude {
    glm::vec3 rotation_center = {0.0f, 0.0f, 0.0f};

    enum Mode { Absolute, Relative_Factor };
    Mode movement = Relative_Factor;

    Pose apply(Pose pose, Action1D const move_distance);
};

struct rotate {
    std::optional<glm::vec3> fixed_world_up = std::nullopt;

    /// Rotates the camera around the right vector
    Pose pitch(Pose pose, Action1D const rotation_rad);

    /// Rotates the camera around the up vector
    Pose yaw(Pose pose, Action1D const rotation_rad);

    /// Rotates the camera around the view vector
    Pose roll(Pose pose, Action1D const rotation_rad);

    Pose apply(Pose pose, Action3D const pitch_yaw_roll_rad);
};

struct translate {
    /// Move the camera in view direction
    Pose move_forward(Pose pose, Action1D const move_distance);

    /// Move the camera along its right vector.
    Pose move_horizontally(Pose pose, Action1D const move_distance);

    /// Move the camera along its up vector.
    Pose move_vertically(Pose pose, Action1D const move_distance);

    Pose apply(Pose pose, Action3D const horizontally_vertically_forward_distance);
};

struct turntable {
    glm::vec3 rotation_center = {0.0f, 0.0f, 0.0f};

    Pose apply(Pose pose, Action2D const rotation_rad);
};

struct fps {
    Pose apply(Pose pose, Action3D const horizontally_vertically_forward_distance, Action2D const pitch_yaw_rad);
};

}; // namespace camera_controllers
