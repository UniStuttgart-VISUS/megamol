/*
 * zoom_manipulator.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef ORBIT_ALTITUDE_MANIPULATOR_H_INCLUDED
#define ORBIT_ALTITUDE_MANIPULATOR_H_INCLUDED


#include "mmcore/thecam/utility/config.h"

#include "mmcore/thecam/manipulator_base.h"


namespace megamol {
namespace core {
namespace thecam {

/**
 * Manipulator
 *
 * @tparam T The type of the camera to be manipulated.
 */
template <class T> class OrbitAltitudeManipulator : public manipulator_base<T> {

public:
    /** The type of the camera to be manipulated by the manipulator. */
    typedef typename manipulator_base<T>::camera_type camera_type;

    /** The mathematical traits of the camera. */
    typedef typename manipulator_base<T>::maths_type maths_type;

    // Typedef all mathematical types we need in the manipulator.
    typedef typename maths_type::point_type point_type;
    typedef typename maths_type::quaternion_type quaternion_type;
    typedef typename maths_type::screen_type screen_type;
    typedef typename maths_type::vector_type vector_type;
    typedef typename maths_type::world_type world_type;

    OrbitAltitudeManipulator() = default;

    /**
     * Finalises the instance.
     */
    ~OrbitAltitudeManipulator() = default;

    void on_drag(const screen_type x, const screen_type y, const point_type& rotCentre) {
        if (this->manipulating() && this->enabled()) {
            auto cam = this->camera();
            THE_ASSERT(cam != nullptr);

            if (this->m_last_sy != y) {

                screen_type dy = y - m_last_sy;

                auto cam_pos = cam->eye_position();
                auto rot_cntr = rotCentre;

                cam_pos.w() = 0.0f;
                rot_cntr.w() = 0.0f;

                auto v = thecam::math::normalise(rot_cntr - cam_pos);

                auto altitude = thecam::math::length(rot_cntr - cam_pos);

                cam->position(cam_pos - (v * dy * (altitude / 500.0f)));
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
    inline void setInactive(void) { this->end_manipulation(); }

private:
    /** The x-coordinate of the last clicked screen position */
    screen_type m_last_sx;

    /** The y-coordinate of the last clicked screen position */
    screen_type m_last_sy;
};

} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */


#endif // !ORBIT_ALTITUDE_MANIPULATOR_H_INCLUDED