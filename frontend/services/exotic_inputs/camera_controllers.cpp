#include "camera_controllers.h"

#include <glm/ext.hpp>
#include <glm/gtx/rotate_vector.hpp>

using namespace camera_controllers;

/*
 * The algorithms in the following manipulators namespace are derived from the thecam/thelib library.
 * We provide the manipulators code according and subject to the original TheLib License.
 * Authors of the TheLib Library seem to be: Sebastian Grottel, Christoph Müller (VISUS, Uni Stuttgart)
 *
 * The routines themselves may have been altered by the MegaMol team from the inital thecam/thelib code,
 * and have been refactored for this codebase to fit into a data-driven camera pose manipulation paradigm.
 *
 * Other code in this file, outside of the manipulators namespace, stems from the MegaMol codebase
 * or is original work in context of the camera manipulators/controllers refactoring and redesign.
 */

namespace manipulators {
/*
 * Copyright (C) 2016 TheLib Team (http://www.thelib.org/license)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of TheLib, TheLib Team, nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THELIB TEAM AS IS AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THELIB TEAM BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


Pose arcball(Pose pose, Action2D const rotation_rad, glm::vec3 const rotation_center) {
    // split movement into horizontal and vertical (in camera space)
    auto rx = rotation_rad.x;
    auto ry = rotation_rad.y;

    // rotate horizontally
    glm::quat rot_pitch = glm::angleAxis(rx, pose.up);
    pose.right = glm::rotate(rot_pitch, pose.right);
    pose.direction = glm::rotate(rot_pitch, pose.direction);

    // rotate vertically
    glm::quat rot_yaw = glm::angleAxis(ry, -pose.right);
    pose.direction = glm::rotate(rot_yaw, pose.direction);
    pose.up = glm::rotate(rot_yaw, pose.up);

    // transform s.t. rotation center is origin
    auto shifted_pos = pose.position - rotation_center;
    shifted_pos = glm::rotate(rot_pitch, shifted_pos);
    shifted_pos = glm::rotate(rot_yaw, shifted_pos);

    // transform back
    pose.position = shifted_pos + glm::vec3(rotation_center);

    return pose;
}

Pose translate_forward(Pose pose, Action1D const distance) {
    pose.position += distance * pose.direction;
    return pose;
}
Pose translate_horizontally(Pose pose, Action1D const distance) {
    pose.position += distance * pose.right;
    return pose;
}
Pose translate_vertically(Pose pose, Action1D const distance) {
    pose.position += distance * pose.up;
    return pose;
}

} // namespace manipulators

Pose arcball::apply(Pose pose, Action2D const rotation_rad) {
    return manipulators::arcball(pose, rotation_rad, rotation_center);
}

Pose orbit_altitude::apply(Pose pose, Action1D const move_distance) {
    auto position = pose.position;
    auto v = glm::normalize(rotation_center - position);
    auto altitude = glm::length(rotation_center - position);
    switch (movement) {
    case Mode::Absolute:
        pose.position = position - move_distance;
        break;
    case Mode::Relative_Factor: {
        pose.position = position - (v * move_distance);
    } break;
    }

    return pose;
}

Pose rotate::pitch(Pose pose, Action1D const rotation_rad) {
    pose.direction = glm::rotate(pose.direction, rotation_rad, pose.right);
    pose.up = glm::rotate(pose.up, rotation_rad, pose.right);

    return pose;
}
Pose rotate::yaw(Pose pose, Action1D const rotation_rad) {
    auto up = fixed_world_up.has_value() ? fixed_world_up.value() : pose.up;
    pose.direction = glm::rotate(pose.direction, rotation_rad, up);
    pose.right = glm::rotate(pose.right, rotation_rad, up);

    return pose;
}
Pose rotate::roll(Pose pose, Action1D const rotation_rad) {
    pose.up = glm::rotate(pose.up, rotation_rad, pose.direction);
    pose.right = glm::rotate(pose.right, rotation_rad, pose.direction);

    return pose;
}
Pose rotate::apply(Pose pose, Action3D const pitch_yaw_roll_rad) {
    auto first = pitch(pose, pitch_yaw_roll_rad.x);
    auto second = yaw(first, pitch_yaw_roll_rad.y);
    auto third = roll(second, pitch_yaw_roll_rad.z);

    return third;
}

Pose translate::move_forward(Pose pose, Action1D const move_distance) {
    return manipulators::translate_forward(pose, move_distance);
}
Pose translate::move_horizontally(Pose pose, Action1D const move_distance) {
    return manipulators::translate_horizontally(pose, move_distance);
}
Pose translate::move_vertically(Pose pose, Action1D const move_distance) {
    return manipulators::translate_vertically(pose, move_distance);
}
Pose translate::apply(Pose pose, Action3D const horizontally_vertically_forward_distance) {
    auto first = move_horizontally(pose, horizontally_vertically_forward_distance.x);
    auto second = move_vertically(first, horizontally_vertically_forward_distance.y);
    auto third = move_forward(second, horizontally_vertically_forward_distance.z);

    return third;
}

Pose turntable::apply(Pose pose, Action2D const rotation_rad) {
    auto rx = rotation_rad.x;
    auto ry = rotation_rad.y;

    // split movement into horizontal and vertical (in camera space)
    glm::quat rot_lat;
    glm::quat rot_lon;

    // rotate horizontally
    rot_lon = glm::angleAxis(rx, glm::vec3(0.0, 1.0, 0.0));
    pose.right = glm::rotate(rot_lon, pose.right);
    pose.direction = glm::rotate(rot_lon, pose.direction);
    pose.up = glm::rotate(rot_lon, pose.up);

    // rotate vertically
    rot_lat = glm::angleAxis(ry, -pose.right);
    pose.direction = glm::rotate(rot_lat, pose.direction);
    pose.up = glm::rotate(rot_lat, pose.up);

    // transform s.t. rotation center is origin
    auto shifted_pos = pose.position - glm::vec3(rotation_center);
    shifted_pos = glm::rotate(rot_lon, shifted_pos);
    shifted_pos = glm::rotate(rot_lat, shifted_pos);

    // transform back
    pose.position = shifted_pos + glm::vec3(rotation_center);

    return pose;
}

Pose fps::apply(Pose pose, Action3D const horizontally_vertically_forward_distance, Action2D const pitch_yaw_rad) {
    pose = translate{}.apply(pose, horizontally_vertically_forward_distance);
    pose = rotate{}.apply(pose, {pitch_yaw_rad, 0.0f});
    return pose;
}

Axis1D camera_controllers::to_axis1d(bool const b) {
    return Axis1D{(float) b};
}
Axis1D camera_controllers::to_axis1d(bool const min, bool const max) {
    return (float) min * (-1.0f) + (float) max * 1.0f;
}

Action1D camera_controllers::simple_screen_delta_to_rotation_rad(Axis1D const delta) {
    const float rad_factor = glm::pi<float>() / 2.f; // assume ~90 degrees fov
    return delta * rad_factor;
}

Action2D camera_controllers::simple_screen_delta_to_rotation_rad(Axis2D const delta) {
    return {
        simple_screen_delta_to_rotation_rad(delta.x),
        simple_screen_delta_to_rotation_rad(delta.y),
    };
}

Action3D camera_controllers::simple_screen_delta_to_rotation_rad(Axis3D const delta) {
    return {
        simple_screen_delta_to_rotation_rad(delta.x),
        simple_screen_delta_to_rotation_rad(delta.y),
        simple_screen_delta_to_rotation_rad(delta.z),
    };
}

Action1D camera_controllers::screen_fov_to_rotation_rad(Axis1D const delta, float const fov_rad) {
    // delta == 0 => return 0
    // delta == 1.0 => return +fov/2
    // delta == -1.0 => return -fov/2
    return (fov_rad * 0.5f) * delta;
}

Action2D camera_controllers::screen_fov_to_rotation_rad(Axis2D const delta, glm::vec2 const fov_rad) {
    return {
        screen_fov_to_rotation_rad(delta.x, fov_rad.x),
        screen_fov_to_rotation_rad(delta.y, fov_rad.y),
    };
}

Action3D camera_controllers::screen_fov_to_rotation_rad(Axis3D const delta, glm::vec3 const fov_rad) {
    return {
        screen_fov_to_rotation_rad(delta.x, fov_rad.x),
        screen_fov_to_rotation_rad(delta.y, fov_rad.y),
        screen_fov_to_rotation_rad(delta.z, fov_rad.z),
    };
}
