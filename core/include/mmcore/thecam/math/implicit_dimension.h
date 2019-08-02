/*
 * the/math/implicit_dimension.h
 *
 * Copyright (C) 2014 - 2016 TheLib Team (http://www.thelib.org/license)
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

#ifndef THE_MATH_IMPLICIT_DIMENSION_H_INCLUDED
#define THE_MATH_IMPLICIT_DIMENSION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#ifdef WITH_THE_GLM
#    include <glm/glm.hpp>
#    include <glm/gtc/quaternion.hpp>
#endif /* WITH_THE_GLM */

#ifdef THE_WINDOWS
#    include <windows.h>
#endif /* THE_WINDOWS */

namespace megamol {
namespace core {
namespace thecam {
namespace math {
namespace detail {

/**
 * Derives the dimension of a given multi-dimensional type.
 *
 * The default implementation does nothing. Template specialisations must
 * provide the dimension by means of a field
 *
 * static const size_t value;
 *
 * This template is used to provide an automatic deduction of the dimension
 * of given multi-dimensional types like Windows' SIZE structure or external
 * vector classes.
 */
template <class T> struct implicit_dimension {};


#ifdef WITH_THE_GLM
/**
 * Implicit dimension deduction for glm::quat.
 */
template <> struct implicit_dimension<glm::quat> { static const size_t value = 4; };
#endif /* WITH_THE_GLM */


#ifdef WITH_THE_GLM
/**
 * Implicit dimension deduction for glm::vec4.
 */
template <> struct implicit_dimension<glm::vec4> { static const size_t value = 4; };
#endif /* WITH_THE_GLM */


#ifdef WITH_THE_GLM
/**
 * Implicit dimension deduction for glm::vec3.
 */
template <> struct implicit_dimension<glm::vec3> { static const size_t value = 3; };
#endif /* WITH_THE_GLM */


#ifdef WITH_THE_GLM
/**
 * Implicit dimension deduction for glm::vec2.
 */
template <> struct implicit_dimension<glm::vec2> { static const size_t value = 2; };
#endif /* WITH_THE_GLM */


#ifdef WITH_THE_GLM
/**
 * Implicit dimension deduction for glm::vec4.
 */
template <> struct implicit_dimension<glm::ivec4> { static const size_t value = 4; };
#endif /* WITH_THE_GLM */


#ifdef WITH_THE_GLM
/**
 * Implicit dimension deduction for glm::vec3.
 */
template <> struct implicit_dimension<glm::ivec3> { static const size_t value = 3; };
#endif /* WITH_THE_GLM */


#ifdef WITH_THE_GLM
/**
 * Implicit dimension deduction for glm::vec2.
 */
template <> struct implicit_dimension<glm::ivec2> { static const size_t value = 2; };
#endif /* WITH_THE_GLM */

#ifdef THE_WINDOWS
/**
 * Implicit dimension deduction for RECT.
 */
template <> struct implicit_dimension<RECT> { static const size_t value = 4; };
#endif /* THE_WINDOWS */


#ifdef THE_WINDOWS
/**
 * Implicit dimension deduction for SIZE.
 */
template <> struct implicit_dimension<SIZE> { static const size_t value = 2; };
#endif /* THE_WINDOWS */

} /* end namespace detail */
} /* end namespace math */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_MATH_IMPLICIT_DIMENSION_H_INCLUDED */
