/*
 * orbital_manipulator.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include <glm/ext.hpp>

#include "mmcore/thecam/manipulator_base.h"

namespace megamol::core::thecam {

/**
 * Implements an orbtial camera maniupulator.
 *
 * @tparam T The type of the camera to be manipulated.
 */
template<class T>
class TurntableManipulator : public manipulator_base<T> {

public:
    /** The type of the camera to be manipulated by the manipulator. */
    typedef typename manipulator_base<T>::camera_type camera_type;

    // Typedef all mathematical types we need in the manipulator.
    typedef typename glm::vec4 point_type;
    typedef typename glm::quat quaternion_type;
    typedef int screen_type;
    typedef typename glm::vec4 vector_type;
    typedef float world_type;

    TurntableManipulator() = default;

    /**
     * Finalises the instance.
     */
    ~TurntableManipulator() = default;

    /**
     * Report that the mouse pointer has been dragged (moved while the
     * designated button was down) to the specified screen coordinates.
     *
     * @param x
     * @param y
     */
    void on_drag(const screen_type x, const screen_type y, const point_type& rotCentre, const screen_type wnd_width,
        const screen_type wnd_height) {
        if (this->manipulating() && this->enabled()) {
            auto cam = this->camera();
            assert(cam != nullptr);

            if (this->m_last_sx != x || this->m_last_sy != y) {

                float factor_x = 720.0f / wnd_width;
                float factor_y = 720.0f / wnd_height;

                screen_type dx = x - m_last_sx;
                screen_type dy = y - m_last_sy;

                // get camera pose
                auto cam_pose = cam->template get<view::Camera::Pose>();

                // split movement into horizontal and vertical (in camera space)
                quaternion_type rot_lat;
                quaternion_type rot_lon;

                // rotate horizontally
                rot_lon = glm::angleAxis(factor_x * dx * (3.14159265f / 180.0f), glm::vec3(0.0, 1.0, 0.0));
                cam_pose.right = glm::rotate(rot_lon, cam_pose.right);
                cam_pose.direction = glm::rotate(rot_lon, cam_pose.direction);
                cam_pose.up = glm::rotate(rot_lon, cam_pose.up);

                // rotate vertically
                rot_lat = glm::angleAxis(factor_y * dy * (3.14159265f / 180.0f), -cam_pose.right);
                cam_pose.direction = glm::rotate(rot_lat, cam_pose.direction);
                cam_pose.up = glm::rotate(rot_lat, cam_pose.up);

                // transform s.t. rotation center is origin
                auto shifted_pos = cam_pose.position - glm::vec3(rotCentre);
                shifted_pos = glm::rotate(rot_lon, shifted_pos);
                shifted_pos = glm::rotate(rot_lat, shifted_pos);

                // transform back
                cam_pose.position = shifted_pos + glm::vec3(rotCentre);
                cam->setPose(cam_pose);

                // update reference values for next call to on_drag (that happens without drag start event)
                this->m_last_sx = x;
                this->m_last_sy = y;
            }
        }
    }

    /**
     * Report that dragging begun (mouse for dragging button is down)
     * at the specified screen coordinates.
     *
     * @param x
     * @param y
     */
    void setActive(const screen_type x, const screen_type y) {
        if (!this->manipulating() && this->enabled()) {
            this->begin_manipulation();
            this->m_last_sx = x;
            this->m_last_sy = y;
        }
    }

    /**
     * Report that dragging ended (mouse button was released).
     */
    inline void setInactive() {
        this->end_manipulation();
    }

private:
    /** The x-coordinate of the last clicked screen position */
    screen_type m_last_sx;

    /** The y-coordinate of the last clicked screen position */
    screen_type m_last_sy;
};

} // namespace megamol::core::thecam
