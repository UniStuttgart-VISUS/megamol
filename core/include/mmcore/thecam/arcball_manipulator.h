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

#ifndef THE_GRAPHICS_CAMERA_ARCBALL_MANIPULATOR_H_INCLUDED
#define THE_GRAPHICS_CAMERA_ARCBALL_MANIPULATOR_H_INCLUDED
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
 * Implements an arcball that rotates a camera.
 *
 * @tparam T The type of the camera to be manipulated.
 */
template <class T> class arcball_manipulator : public manipulator_base<T> {

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

    arcball_manipulator(const point_type& rotCentre = point_type(), const world_type radius = 1);

    /**
     * Finalises the instance.
     */
    virtual ~arcball_manipulator(void);

    /**
     * Report that the mouse pointer has been dragged (moved while the
     * designated button was down) to the specified screen coordinates.
     *
     * @param x
     * @param y
     */
    void on_drag(const screen_type x, const screen_type y);

    /**
     * Report that dragging begun (mouse for dragging button is down)
     * at the specified screen coordinates.
     *
     * @param x
     * @param y
     */
    void on_drag_start(const screen_type x, const screen_type y);

    /**
     * Report that dragging ended (mouse button was released).
     */
    inline void on_drag_stop(void) { this->end_manipulation(); }

    /**
     * Answer the radius of the arcball.
     *
     * @return The radius of the arcball.
     */
    inline world_type radius(void) const { return this->ballRadius; }

    /**
     * Gets the centre of rotation (usually the centre of the object to
     * rotate the camera around).
     *
     * @returns The centre of rotation.
     */
    inline const point_type& rotation_centre(void) const { return this->rotCentre; }

    /**
     * Changes the radius of the arcball.
     *
     * This method must not be called while the arcball is begin dragged.
     *
     * @param radius The new radius of the arcball.
     */
    inline void set_radius(const world_type radius) {
        THE_ASSERT(!this->manipulating());
        this->ballRadius = radius;
    }

    /**
     * Changes the centre of rotation.
     *
     * This method must not be called while the arcball is begin dragged.
     *
     * @param rotCentre The new centre of rotation.
     */
    inline void set_rotation_centre(const point_type& rotCentre) {
        THE_ASSERT(!this->manipulating());
        this->rotCentre = rotCentre;
    }

private:
    /**
     * Convert screen point to a point on the arcball.
     *
     * @param sx The absicssa of the current mouse position.
     * @param sy The ordinate of the current mouse position.
     *
     * @return The normalised vector from the centre of the arcball to the
     *         point on the arcball represented by the given screen
     *         coordinates.
     */
    vector_type mapToSphere(const screen_type sx, const screen_type sy) const;

    /** The radius of the arcball .*/
    world_type ballRadius;

    /** The latest point on the arcball. */
    vector_type currentVector;

    /** The centre of rotation (target point of the camera). */
    point_type rotCentre;

    vector_type startPos;

    /** The camera rotation quaterion when the drag interaction started. */
    quaternion_type invStartRot;

    quaternion_type startRot;

    /** The point on the arcball when the drag interaction started. */
    vector_type startVector;

    /** The x-coordinate of the last clicked screen position */
    screen_type lastSx;

    /** The y-coordinate of the last clicked screen position */
    screen_type lastSy;
};

} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#include "mmcore/thecam/arcball_manipulator.inl"

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_GRAPHICS_CAMERA_ARCBALL_MANIPULATOR_H_INCLUDED */
