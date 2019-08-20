/*
 * thecam/minimal_camera_state.h
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

#ifndef THE_GRAPHICS_CAMERA_MINIMAL_CAMERA_STATE_H_INCLUDED
#define THE_GRAPHICS_CAMERA_MINIMAL_CAMERA_STATE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include <array>
#include "mmcore/thecam/types.h"


namespace megamol {
namespace core {
namespace thecam {

/**
 * This structure holds the minimum information that is required to fully
 * restore the state of a megamol::core::thecam::graphics::camera::camera.
 *
 * The structure uses only basic types as members such that it can be
 * serialised for network transfer and persistence.
 *
 * @tparam M The configuration of the mathematical types the camera uses.
 */
template <class M> struct minimal_camera_state {

    typedef typename M::fractional_type fractional_type;
    typedef typename M::screen_type screen_type;
    typedef typename M::world_type world_type;

    /** The relative offset (x, y) of the projection centre. */
    std::array<fractional_type, 2> centre_offset;

    /** The distance of the plane of zero parallax.*/
    world_type convergence_plane;

    /** The stereo eye. */
    Eye eye;

    /** The distance of the far clipping plane. */
    world_type far_clipping_plane;

    /** With and height of the film gate. */
    std::array<world_type, 2> film_gate;

    /** The gate scaling method. */
    Gate_scaling gate_scaling;

    /** Half of the aperture angle in radians. */
    fractional_type half_aperture_angle_radians;

    /** Half of the stereo disparity. */
    world_type half_disparity;

    /** Left, top, right, bottom of the image tile. */
    std::array<screen_type, 4> image_tile;

    /** The distance of the near clipping plane. */
    world_type near_clipping_plane;

    /** The components (x, y, z, w) of the orientation quaternion. */
    std::array<world_type, 4> orientation;

    /** The position of the camera. */
    std::array<world_type, 3> position;

    /** The projection type. */
    Projection_type projection_type;

    /** Width and height of the resolution gate. */
    std::array<screen_type, 2> resolution_gate;
};

} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_GRAPHICS_CAMERA_MINIMAL_CAMERA_STATE_H_INCLUDED */
