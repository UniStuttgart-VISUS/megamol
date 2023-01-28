/*
 * zoom_manipulator.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef ORBIT_ALTITUDE_MANIPULATOR_H_INCLUDED
#define ORBIT_ALTITUDE_MANIPULATOR_H_INCLUDED

#include "mmcore/thecam/manipulator_base.h"

namespace megamol::core::thecam {

/**
 * Manipulator
 *
 * @tparam T The type of the camera to be manipulated.
 */
template<class T>
class OrbitAltitudeManipulator : public manipulator_base<T> {

public:
    /** The type of the camera to be manipulated by the manipulator. */
    typedef typename manipulator_base<T>::camera_type camera_type;

    // Typedef all mathematical types we need in the manipulator.
    typedef typename glm::vec4 point_type;
    typedef typename glm::quat quaternion_type;
    typedef int screen_type;
    typedef typename glm::vec4 vector_type;
    typedef float world_type;

    OrbitAltitudeManipulator() = default;

    /**
     * Finalises the instance.
     */
    ~OrbitAltitudeManipulator() = default;

    void on_drag(const screen_type x, const screen_type y, const point_type& rotCentre) {
        if (this->manipulating() && this->enabled()) {
            auto cam = this->camera();
            assert(cam != nullptr);

            if (this->m_last_sy != y) {

                world_type dy = static_cast<world_type>(y - m_last_sy);

                auto cam_pose = cam->template get<view::Camera::Pose>();

                auto cam_pos = cam_pose.position;
                auto rot_cntr = glm::vec3(rotCentre);

                auto v = glm::normalize(rot_cntr - cam_pos);

                auto altitude = glm::length(rot_cntr - cam_pos);

                cam_pose.position = cam_pos - (v * dy * (altitude / 500.0f));
                cam->setPose(cam_pose);
            }

            this->m_last_sx = x;
            this->m_last_sy = y;
        }
    }

    /**
     * Set manipulator active (mouse for dragging button is down)
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
     * Set manipulator to inactive (usually on mouse button release).
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


#endif // !ORBIT_ALTITUDE_MANIPULATOR_H_INCLUDED
