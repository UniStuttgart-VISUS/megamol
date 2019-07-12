/*
 * thecam/types.h
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

#ifndef THE_GRAPHICS_CAMERA_TYPES_H_INCLUDED
#define THE_GRAPHICS_CAMERA_TYPES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"


namespace megamol {
namespace core {
namespace thecam {

/**
 * Determines the possible stereo eyes that the camera can compute.
 */
enum class Eye {

    /** The left eye. */
    left = -1,

    /**
     * The actual position of the camera between the eyes. Use this for
     * non-stereo scenes.
     */
    centre = 0,

    /** Alias for 'centre'. */
    mono = 0,

    /** The right eye. */
    right = 1
};


/**
 * Determines how the camera handles different aspect rations of the film
 * gate and the resolution.
 */
enum class Gate_scaling {

    /** Ignore the problem and skew the image to the image aspect ratio. */
    none = 0,

    /** Force the resolution gate within the film gate. */
    fill = 1,

    /** Force the film gate within the resolution gate. */
    overscan = 2,

    /**
     * Ensure that the whole resolution gate is filled (this is an alias for
     * 'fill'). Some of the imagery visible in the film gate might to be
     * rendered.
     */
    uniform_to_fill = 1,

    /**
     * Ensure that the whole film gate is visible (this is an alias for
     * 'overscan'). There might be some unfilled areas in the image.
     */
    uniform_to_fit = 2
};


/** The handedness of Cartesian coordinate systems. */
enum class Handedness {

    /**
     * The coordinate system is left-handed.
     *
     * If the thumb of the left hand is pointing along the positive x-axis
     * and the forefinger along the positive y-axis, the long finger is
     * pointing along the positive z-axis.
     */
    left_handed = 0,

    /**
     * An alias for 'left_handed'.
     *
     * When viewed from z-axis, a left-handed system is clockwise.
     */
    clockwise = 0,

    /**
     * The coordinate system is right-handed.
     *
     * If the thumb of the right hand is pointing along the positive x-axis
     * and the forefinger along the positive y-axis, the long finger is
     * pointing along the positive z-axis.
     */
    right_handed = 1,

    /**
     * An alias for 'right_handed'.
     *
     * When viewed from z-axis, a right-handed system is anticlockwise.
     */
    anticlockwise = 1
};


/** Possible methods to compute mono or stereo projections. */
enum class Projection_type {

    /** Enables a mono perspective projection. */
    perspective = 0,

    /** Enables a mono orthographic projection. */
    orthographic = 1,

    /**
     * Enables a stereo mode, which has effectively no convergence plane.
     * This is useful for landscape settings where objects exist at
     * effectively infinity.
     */
    parallel = 2,

    /**
     * Enables a stereo mode, which computes the convergence plane by
     * shifting the frustum using camera film back. This is the safer
     * way to compute stereo image pairs and does not have any keystone
     * artefacts.
     */
    off_axis = 3,

    toe_in = 4,

    converged = 4
    // Computes the zero parallax plane by toeing in the cameras. This effect is equivalent to the human eye where we
    // tend to focus on an object by rotating our pupils inwards at an object. However, this has a very dangerous side
    // effect where you can get a keystone effect on the pairs of render images causing visual confusing in other
    // elements in the scene. In a rendered image, our focus tends to saccade over the entire image and we are not
    // focusing on a single object, which is not true in real life. You should only use 'Converged' in very specific
    // circumstances, i.e. an object is at the center of the screen and there are no scene elements at the render
    // borders on either the left or right camera frustum.
    // converged = 4??
};

} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */


#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_GRAPHICS_CAMERA_TYPES_H_INCLUDED */
