/*
 * thecam/camera_maths.h
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

#ifndef THE_GRAPHICS_CAMERA_MATHS_H_INCLUDED
#define THE_GRAPHICS_CAMERA_MATHS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include "mmcore/thecam/types.h"

#include "mmcore/thecam/math/matrix.h"
#include "mmcore/thecam/math/point.h"
#include "mmcore/thecam/math/quaternion.h"
#include "mmcore/thecam/math/rectangle.h"
#include "mmcore/thecam/math/size.h"
#include "mmcore/thecam/math/vector.h"

#ifdef WITH_THE_GLM
#    include <glm/glm.hpp>
#endif

namespace megamol {
namespace core {
namespace thecam {

/**
 * General, configurable specification of maths classes for the camera.
 *
 * Note: The camera_maths type provides the initial coordinate frame of the
 * camera by means of its view (front) and up vector. The camer_maths type
 * must ensure that these vectors are orthonormal. No normalisation is done
 * in the camera. This implementation aligns the camera with the coordinate
 * axes and uses unit vectors, ie the conditions are met.
 *
 * @tparam W
 * @tparam S
 * @tparam F
 * @tparam H
 */
template <class W = float, class S = int, class F = float, Handedness H = Handedness::right_handed>
struct camera_maths {

    typedef F fractional_type;

    typedef S screen_type;

    typedef W world_type;

    // TODO: customisable layout?
    typedef math::matrix<world_type, 4, 4> matrix_type;

    typedef math::point<world_type, 4> point_type;

    typedef math::quaternion<world_type> quaternion_type;

    typedef math::rectangle<screen_type> screen_rectangle_type;

    typedef math::size<screen_type, 2> screen_size_type;

    typedef math::vector<world_type, 4> vector_type;

    typedef math::size<world_type, 2> world_size_type;

    /* The handedness of the coordinate system used. */
    static const Handedness handedness = H;

    /**
     * The up-vector in the initial rest position of the camera; in this
     * case a vector pointing towards the positive y-axis.
     */
    static const vector_type up_vector;

    /**
     * The view-vector in the initial rest position of the camera; if the
     * coordinate system is right-handed, the vector points towards the
     * negative z-axis; otherwise, it points towards the positive z-axis.
     */
    static const vector_type view_vector;
};

#ifdef WITH_THE_GLM

template <Handedness H = Handedness::right_handed> struct glm_camera_maths {
    typedef float fractional_type;

    typedef int screen_type;

    typedef float world_type;

    typedef math::matrix<glm::mat4> matrix_type;

    typedef math::point<glm::vec4> point_type;

    typedef math::quaternion<glm::quat> quaternion_type;

    typedef math::rectangle<screen_type> screen_rectangle_type;

    typedef math::size<glm::ivec2> screen_size_type;

    typedef math::vector<glm::vec4> vector_type;

    typedef math::size<world_type, 2> world_size_type;

    static const Handedness handedness = H;

    static const vector_type up_vector;

    static const vector_type view_vector;
};

/** Left-handed XMATH camera maths types. */
typedef glm_camera_maths<Handedness::left_handed> glm_camera_maths_lh;

/** Right-handed XMATH camera maths types. */
typedef glm_camera_maths<Handedness::right_handed> glm_camera_maths_rh;
#endif /* WITH_THE_GLM */

} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */


/*
 * megamol::core::thecam::camera_maths<W, S, F, H>::up_vector
 */
template <class W, class S, class F, megamol::core::thecam::Handedness H>
const typename megamol::core::thecam::camera_maths<W, S, F, H>::vector_type
    megamol::core::thecam::camera_maths<W, S, F, H>::up_vector(
        static_cast<F>(0), static_cast<F>(1), static_cast<F>(0), static_cast<F>(0));


/*
 * megamol::core::thecam::camera_maths<W, S, F, H>::view_vector
 */
template <class W, class S, class F, megamol::core::thecam::Handedness H>
const typename megamol::core::thecam::camera_maths<W, S, F, H>::vector_type
    megamol::core::thecam::camera_maths<W, S, F, H>::view_vector(static_cast<F>(0), static_cast<F>(0),
        static_cast<F>(H == megamol::core::thecam::Handedness::right_handed ? -1 : 1), static_cast<F>(0));

#ifdef WITH_THE_GLM
/*
 * thecam::xmath_camera_maths<H>::up_vector
 */
template <megamol::core::thecam::Handedness H>
const typename megamol::core::thecam::glm_camera_maths<H>::vector_type
    megamol::core::thecam::glm_camera_maths<H>::up_vector(0.0f, 1.0f, 0.0f, 0.0f);


/*
 * thecam::xmath_camera_maths<H>::view_vector
 */
template <megamol::core::thecam::Handedness H>
const typename megamol::core::thecam::glm_camera_maths<H>::vector_type
    megamol::core::thecam::glm_camera_maths<H>::view_vector(
        0.0f, 0.0f, (H == thecam::Handedness::right_handed) ? -1.0f : 1.0f, 0.0f);
#endif

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_GRAPHICS_CAMERA_MATHS_H_INCLUDED */
