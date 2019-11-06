/*
 * thecam/translate_manipulator.h
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

#ifndef THE_GRAPHICS_CAMERA_TRANSLATE_MANIPULATOR_H_INCLUDED
#define THE_GRAPHICS_CAMERA_TRANSLATE_MANIPULATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include "mmcore/thecam/manipulator_base.h"

namespace megamol {
namespace core {
namespace thecam {

/**
 * A manipulator for moving the camera around.
 *
 * @tparam T The type of the camera to be manipulated.
 */
template <class T> class translate_manipulator : public manipulator_base<T> {

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

    translate_manipulator(const world_type stepSize = 1);

    /**
     * Finalises the instance.
     */
    virtual ~translate_manipulator(void);

    /**
     * Move the camera in view direction.
     */
    void move_forward(const world_type dist);

    /**
     * Move the camera in view direction.
     */
    inline void move_forward(void) { this->move_forward(this->stepSize); }

    /**
     * Move the camera along its x-axis.
     */
    void move_horizontally(const world_type dist);

    /**
     * Move the camera along its x-axis.
     */
    inline void move_horizontally(void) { this->move_horizontally(this->stepSize); }

    /**
     * Move the camera along its up vector proportinaly to the mouse y coordinate delta
     */
    inline void move_horizontally(const screen_type x) {
        screen_type dx = x - this->m_last_sx;
        this->m_last_sx = x;
        move_horizontally(this->stepSize * static_cast<world_type>(dx));
    }

    /**
     * Move the camera along its up vector.
     */
    void move_vertically(const world_type dist);

    /**
     * Move the camera along its up vector.
     */
    inline void move_vertically(void) { this->move_horizontally(this->stepSize); }

    /**
     * Move the camera along its up vector proportinaly to the mouse y coordinate delta
     */
    inline void move_vertically(const screen_type y) { 
        screen_type dy = y - this->m_last_sy;
        this->m_last_sy = y;
        move_vertically(this->stepSize * static_cast<world_type>(dy));
    }

    /**
     * Sets a new step size in world units.
     *
     * @param stepSize The new translation step size.
     */
    inline void set_step_size(const world_type stepSize) { this->stepSize = stepSize; }

    /**
     * Gets a new step size in world units.
     *
     * @return The translation step size.
     */
    inline world_type step_size(void) const { return this->stepSize; }

    /**
     * Set manipulator active, i.e. set manipulating flag true and store mouse coords at time of activation
     */
    inline void setActive(const screen_type x, const screen_type y) {
        if (!this->manipulating() && this->enabled()) {
            this->begin_manipulation();
            this->m_last_sx = x;
            this->m_last_sy = y;
        }
    }

    /**
     * Set manipulator inactive, i.e. set manipulating flag false
     */
    inline void setInactive() { this->end_manipulation(); }

private:
    world_type stepSize;

     /** The x-coordinate of the last clicked screen position */
    screen_type m_last_sx;

    /** The y-coordinate of the last clicked screen position */
    screen_type m_last_sy;
};

} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#include "mmcore/thecam/translate_manipulator.inl"

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_GRAPHICS_CAMERA_TRANSLATE_MANIPULATOR_H_INCLUDED */
