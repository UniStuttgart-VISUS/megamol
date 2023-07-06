#include "arcball_manipulator.h"
/*
 * thecam/arcball_manipulator.h
 *
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


/*
 * megamol::core::thecam::arcball_manipulator<T>::~arcball_manipulator
 */
template<class T>
megamol::core::thecam::arcball_manipulator<T>::~arcball_manipulator() {}


/*
 * megamol::core::thecam::arcball_manipulator<T>::on_drag
 */
template<class T>
void megamol::core::thecam::arcball_manipulator<T>::on_drag(
    const screen_type x, const screen_type y, const point_type& rotCentre) {
    if (this->manipulating() && this->enabled()) {
        auto cam = this->camera();
        assert(cam != nullptr);

        if (this->lastSx != x || this->lastSy != y) {
            screen_type dx = x - lastSx;
            screen_type dy = y - lastSy;

            // get camera pose
            auto cam_pose = cam->template get<view::Camera::Pose>();

            // split movement into horizontal and vertical (in camera space)
            quaternion_type rot_pitch;
            quaternion_type rot_yaw;

            // rotate horizontally
            rot_pitch = glm::angleAxis(dx * (3.14159265f / 180.0f), cam_pose.up);
            cam_pose.right = glm::rotate(rot_pitch, cam_pose.right);
            cam_pose.direction = glm::rotate(rot_pitch, cam_pose.direction);

            // rotate vertically
            rot_yaw = glm::angleAxis(dy * (3.14159265f / 180.0f), -cam_pose.right);
            cam_pose.direction = glm::rotate(rot_yaw, cam_pose.direction);
            cam_pose.up = glm::rotate(rot_yaw, cam_pose.up);

            // transform s.t. rotation center is origin
            auto shifted_pos = cam_pose.position - glm::vec3(rotCentre);
            shifted_pos = glm::rotate(rot_pitch, shifted_pos);
            shifted_pos = glm::rotate(rot_yaw, shifted_pos);

            // transform back
            cam_pose.position = shifted_pos + glm::vec3(rotCentre);
            cam->setPose(cam_pose);

            // update reference values for next call to on_drag (that happens without drag start event)
            this->lastSx = x;
            this->lastSy = y;
        }
    }
}


/*
 * megamol::core::thecam::arcball_manipulator<T>::on_drag_start
 */
template<class T>
void megamol::core::thecam::arcball_manipulator<T>::setActive(const screen_type x, const screen_type y) {
    if (!this->manipulating() && this->enabled()) {
        this->begin_manipulation();
        this->lastSx = x;
        this->lastSy = y;
    }
}
