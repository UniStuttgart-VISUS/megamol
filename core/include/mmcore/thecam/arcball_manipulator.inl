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
template <class T> megamol::core::thecam::arcball_manipulator<T>::~arcball_manipulator(void) {}


/*
 * megamol::core::thecam::arcball_manipulator<T>::on_drag
 */
template <class T>
void megamol::core::thecam::arcball_manipulator<T>::on_drag(
    const screen_type x, const screen_type y, const point_type& rotCentre) {
    if (this->manipulating() && this->enabled()) {
        auto cam = this->camera();
        THE_ASSERT(cam != nullptr);

        if (this->lastSx != x || this->lastSy != y) {
            screen_type dx = x - lastSx;
            screen_type dy = y - lastSy;

            // split movement into horizontal and vertical (in camera space)
            quaternion_type rot_pitch;
            quaternion_type rot_yaw;
            
            // rotate horizontally
            thecam::math::set_from_angle_axis(rot_pitch, dx * (3.14159265f / 180.0f), cam->up_vector());
            cam->orientation(rot_pitch * this->camera()->orientation());
            auto updated_orientation = cam->orientation();
            
            // get cam right vector after horizontal rotation to rotate vertically
            auto cam_right = cam->right_vector();
            thecam::math::set_from_angle_axis(rot_yaw, dy * (3.14159265f / 180.0f), -cam_right);
            cam->orientation(rot_yaw * updated_orientation);
            
            // transform s.t. rotation center is origin
            auto shifted_pos = this->camera()->eye_position() - rotCentre;
            quaternion_type pos_quat(shifted_pos.x(), shifted_pos.y(), shifted_pos.z(), 0.0f);
            // move camera based on applied rotation
            auto rot_pitch_conj = thecam::math::conjugate(rot_pitch);
            auto rot_yaw_conj = thecam::math::conjugate(rot_yaw);
            pos_quat = rot_pitch * pos_quat * rot_pitch_conj;
            pos_quat = rot_yaw * pos_quat * rot_yaw_conj;
            
            // transform back 
            cam->position(point_type(
                pos_quat.x() +  rotCentre.x(),
                pos_quat.y() +  rotCentre.y(),
                pos_quat.z() +  rotCentre.z(),
                1.0f)
            );
            
            // update reference values for next call to on_drag (that happens without drag start event)
            this->lastSx = x;
            this->lastSy = y;
        }
    }
}


/*
 * megamol::core::thecam::arcball_manipulator<T>::on_drag_start
 */
template <class T>
void megamol::core::thecam::arcball_manipulator<T>::setActive(const screen_type x, const screen_type y) {
    if (!this->manipulating() && this->enabled()) {
        this->begin_manipulation();
        this->lastSx = x;
        this->lastSy = y;
    }
}