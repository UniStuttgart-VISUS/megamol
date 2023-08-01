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

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <glm/glm.hpp>

#include "mmcore/thecam/manipulator_base.h"

namespace megamol::core::thecam {

/**
 * Implements an arcball that rotates a camera.
 *
 * @tparam T The type of the camera to be manipulated.
 */
template<class T>
class arcball_manipulator : public manipulator_base<T> {

public:
    /** The type of the camera to be manipulated by the manipulator. */
    typedef typename manipulator_base<T>::camera_type camera_type;

    // Typedef all mathematical types we need in the manipulator.
    typedef typename glm::vec4 point_type;
    typedef typename glm::quat quaternion_type;
    typedef int screen_type;
    typedef typename glm::vec4 vector_type;
    typedef float world_type;

    arcball_manipulator() = default;

    /**
     * Finalises the instance.
     */
    virtual ~arcball_manipulator();

    /**
     * Report that the mouse pointer has been dragged (moved while the
     * designated button was down) to the specified screen coordinates.
     *
     * @param x
     * @param y
     */
    void on_drag(const screen_type x, const screen_type y, const point_type& rotCentre);

    /**
     * Report that dragging begun (mouse for dragging button is down)
     * at the specified screen coordinates.
     *
     * @param x
     * @param y
     */
    void setActive(const screen_type x, const screen_type y);

    /**
     * Report that dragging ended (mouse button was released).
     */
    inline void setInactive() {
        this->end_manipulation();
    }

private:
    /** The x-coordinate of the last clicked screen position */
    screen_type lastSx;

    /** The y-coordinate of the last clicked screen position */
    screen_type lastSy;
};

} // namespace megamol::core::thecam

#include "mmcore/thecam/arcball_manipulator.inl"

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
