/*
 * thecam/camera_snapshot.h
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

#ifndef THE_GRAPHICS_CAMERA_CAMERA_SNAPSHOT_H_INCLUDED
#define THE_GRAPHICS_CAMERA_CAMERA_SNAPSHOT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include <cstdint>
#include <type_traits>

#include "mmcore/thecam/view_frustum.h"


namespace megamol {
namespace core {
namespace thecam {

/**
 *
 */
enum class snapshot_content : uint32_t {

    /**
     * No content is set. You must assume that the fields in the snapshot
     * contain only rubbish. Do not use anything of it.
     */
    nothing = 0x00000000,

    /** Includes the view-vector in the snapshot. */
    view_vector = 0x000000001,

    /** Includes the up-vector in the snapshot. */
    up_vector = 0x000000002,

    /**
     * Includes the whole camera coordinate system in the snapshot, which
     * includes the view-vector, the up-vector, the right-vector and the
     * final camera position (considering all stereo effects).
     */
    camera_coordinate_system = 0x000000004,

    /** Includes the view frustum bounds in camera space in the snapshot. */
    camera_space_frustum = 0x000000008,

    /** Include the aspect ratio of the image size. */
    resolution_aspect = 0x000000010,

    /** Includes the aspect ration of the film size. */
    film_aspect = 0x000000020,

    /**
     * Includes the gate scaling factors to match the film gate to the
     * resolution gate.
     */
    gate_scaling = 0x000000040,

    /** Includes all available properties in the snapshot. */
    all = UINT32_MAX
};


/**
 * Combines two snapshot_content flags.
 *
 * @param lhs The left hand side operand.
 * @param rhs The right hand side operand.
 *
 * @return The combination of all bits in 'lhs' and 'rhs'.
 */
inline snapshot_content operator|(const snapshot_content lhs, const snapshot_content rhs) {
    auto l = static_cast<std::underlying_type<snapshot_content>::type>(lhs);
    auto r = static_cast<std::underlying_type<snapshot_content>::type>(rhs);
    return static_cast<snapshot_content>(l | r);
}

/**
 * Combines two snapshot_content flags into 'lhs'
 *
 * @param lhs The left hand side operand.
 * @param rhs The right hand side operand.
 *
 * @return 'lhs'.
 */
inline snapshot_content& operator|=(snapshot_content& lhs, const snapshot_content rhs) {
    auto l = static_cast<std::underlying_type<snapshot_content>::type>(lhs);
    auto r = static_cast<std::underlying_type<snapshot_content>::type>(rhs);
    lhs = static_cast<snapshot_content>(l | r);
    return lhs;
}

/**
 * Answer the snapshot_content flags set in 'lhs' and 'rhs'.
 *
 * @param lhs The left hand side operand.
 * @param rhs The right hand side operand.
 *
 * @return The bits set in 'lhs' and 'rhs'.
 */
inline snapshot_content operator&(const snapshot_content lhs, const snapshot_content rhs) {
    auto l = static_cast<std::underlying_type<snapshot_content>::type>(lhs);
    auto r = static_cast<std::underlying_type<snapshot_content>::type>(rhs);
    return static_cast<snapshot_content>(l & r);
}

/**
 * Answer the snapshot_content flags set in 'lhs' and 'rhs' in 'lhs'.
 *
 * @param lhs The left hand side operand.
 * @param rhs The right hand side operand.
 *
 * @return 'lhs'.
 */
inline snapshot_content& operator&=(snapshot_content& lhs, const snapshot_content rhs) {
    auto l = static_cast<std::underlying_type<snapshot_content>::type>(lhs);
    auto r = static_cast<std::underlying_type<snapshot_content>::type>(rhs);
    lhs = static_cast<snapshot_content>(l & r);
    return lhs;
}


/**
 * Caches a specific set of derived camera properties for reuse.
 *
 * @tparam M The configuration of the mathematical types the camera that the
 *           snapshot is from uses.
 */
template <class M> struct camera_snapshot {

    /**
     * The type used for expressing ratios. This must never be an
     * integral type, but always a rational one.
     */
    typedef typename M::fractional_type fractional_type;

    /** The type of matrices used by the camera. */
    typedef typename M::matrix_type matrix_type;

    /** The type used for world-space positions. */
    typedef typename M::point_type point_type;

    typedef typename M::screen_rectangle_type screen_rectangle_type;

    /** The type used for expressing screen-space dimensions. */
    typedef typename M::screen_size_type screen_size_type;

    /** The type used for expressing screen coordinate values. */
    typedef typename M::screen_type screen_type;

    /** The type of quaternions used by the camera. */
    typedef typename M::quaternion_type quaternion_type;

    /** The camera traits. */
    typedef M maths_type;

    /** The type of vectors used by the camera. */
    typedef typename M::vector_type vector_type;

    typedef view_frustum<typename M::world_type> view_frustum_type;

    /** The type used for expressing world-space dimensions. */
    typedef typename M::world_size_type world_size_type;

    /** The type for expressing world-space values. */
    typedef typename M::world_type world_type;

    /**
     * Holds a bitmask of all camera properties which are valid in this
     * snapshot. All other properties are invalid and must not be used.
     */
    snapshot_content content;

    /** Location of bottom clipping plane on near clipping plane. */
    world_type frustum_bottom;

    /** Distance of far clipping plane from camera position. */
    world_type frustum_far;

    /** Location of left clipping plane on near clipping plane. */
    world_type frustum_left;

    /** Distance of near clipping plane from camera position. */
    world_type frustum_near;

    /** Location of right clipping plane on near clipping plane. */
    world_type frustum_right;

    /** Location of top clipping plane on near clipping plane. */
    world_type frustum_top;

    /** The aspect ratio (width / height) of the image. */
    fractional_type resolution_aspect;

    /** The aspect ratio (width / height) of the film. */
    fractional_type film_aspect;

    /**
     * This scaling accounts for non-matching resolution_aspect and
     * film_aspect (x and y).
     */
    fractional_type gate_scaling[2];

    /** The position of the camera. */
    point_type position;

    /** The normalised right vector. */
    vector_type right_vector;

    /** The normalised view vector. */
    vector_type view_vector;

    /** The normalised up vector. */
    vector_type up_vector;

    /** Initialises a new instance by setting everything invalid. */
    inline camera_snapshot(void) : content(snapshot_content::nothing) {}

    /**
     * Answe whether the properties identified by 'which' are set.
     *
     * @param which A bitmask to be tested against 'content'.
     *
     * @return true if all of 'which's bits are set, false otherwise.
     */
    inline bool contains(snapshot_content which) const { return ((this->content & which) == which); }
};

} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_GRAPHICS_CAMERA_CAMERA_SNAPSHOT_H_INCLUDED */
