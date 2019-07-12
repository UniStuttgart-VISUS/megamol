/*
 * thecam\math\vector_accessors.h
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

#ifndef THE_MATH_VECTOR_ACCESSORS_H_INCLUDED
#define THE_MATH_VECTOR_ACCESSORS_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"


namespace megamol {
namespace core {
namespace thecam {
namespace math {
namespace detail {


/**
 * A utility class that provides named accessors for vector components
 * using the CRTP.
 *
 * @param V The actual vector type, which inherits from this class.
 * @param T The traits type used for V. We need this information, because
 *          V is not yet complete when we need to determine value_type.
 * @param D The dimension of the vector.
 */
template <class V, class T, size_t D> struct vector_accessors {};


/**
 * Specialisation of vector_accessors for 1D vectors, which adds a named
 * accessor for the x-component.
 *
 * @param V The actual vector type, which inherits from this class.
 * @param T The traits type used for V. We need this information, because
 *          V is not yet complete when we need to determine value_type.
 */
template <class V, class T> struct vector_accessors<V, T, 1> {

    typedef typename T::value_type value_type;

    inline value_type x(void) const {
        auto that = static_cast<const V*>(this);
        return (*that)[0];
    }

    inline value_type& x(void) {
        auto that = static_cast<V*>(this);
        return (*that)[0];
    }
};


/**
 * Specialisation of vector_accessors for 2D vectors, which inherits all
 * accessors from 1D vectors and adds a named accessor for the y-component.
 *
 * @param V The actual vector type, which inherits from this class.
 * @param T The traits type used for V. We need this information, because
 *          V is not yet complete when we need to determine value_type.
 */
template <class V, class T> struct vector_accessors<V, T, 2> : public vector_accessors<V, T, 1> {

    typedef typename T::value_type value_type;

    inline value_type y(void) const {
        auto that = static_cast<const V*>(this);
        return (*that)[1];
    }

    inline value_type& y(void) {
        auto that = static_cast<V*>(this);
        return (*that)[1];
    }
};


/**
 * Specialisation of vector_accessors for 3D vectors, which inherits all
 * accessors from 2D vectors and adds a named accessor for the z-component.
 *
 * @param V The actual vector type, which inherits from this class.
 * @param T The traits type used for V. We need this information, because
 *          V is not yet complete when we need to determine value_type.
 */
template <class V, class T> struct vector_accessors<V, T, 3> : public vector_accessors<V, T, 2> {

    typedef typename T::value_type value_type;

    inline value_type z(void) const {
        auto that = static_cast<const V*>(this);
        return (*that)[2];
    }

    inline value_type& z(void) {
        auto that = static_cast<V*>(this);
        return (*that)[2];
    }
};


/**
 * Specialisation of vector_accessors for 4D vectors, which inherits all
 * accessors from 3D vectors and adds a named accessor for the w-component.
 *
 * @param V The actual vector type, which inherits from this class.
 * @param T The traits type used for V. We need this information, because
 *          V is not yet complete when we need to determine value_type.
 */
template <class V, class T> struct vector_accessors<V, T, 4> : public vector_accessors<V, T, 3> {

    typedef typename T::value_type value_type;

    inline value_type w(void) const {
        auto that = static_cast<const V*>(this);
        return (*that)[3];
    }

    inline value_type& w(void) {
        auto that = static_cast<V*>(this);
        return (*that)[3];
    }
};

} /* end namespace detail */
} /* end namespace math */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_MATH_VECTOR_ACCESSORS_H_INCLUDED */
