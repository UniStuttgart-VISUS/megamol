/*
 * the\math\size_accessors.h
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

#ifndef THE_MATH_SIZE_ACCESSORS_H_INCLUDED
#define THE_MATH_SIZE_ACCESSORS_H_INCLUDED
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
 * A utility class that provides named accessors for size components
 * using the CRTP.
 *
 * @param S The actual size type, which inherits from this class.
 * @param T The traits type used for S. We need this information, because
 *          S is not yet complete when we need to determine value_type.
 * @param D The dimension the size is for..
 */
template <class S, class T, size_t D> struct size_accessors {};


/**
 * Specialisation of size_accessors for 2D, which adds width() and height().
 *
 * @param S The actual size type, which inherits from this class.
 * @param T The traits type used for S. We need this information, because
 *          S is not yet complete when we need to determine value_type.
 */
template <class S, class T> struct size_accessors<S, T, 2> {

    typedef typename T::value_type value_type;

    inline value_type height(void) const {
        auto that = static_cast<const S*>(this);
        return (*that)[1];
    }

    inline value_type& height(void) {
        auto that = static_cast<S*>(this);
        return (*that)[1];
    }

    inline value_type width(void) const {
        auto that = static_cast<const S*>(this);
        return (*that)[0];
    }

    inline value_type& width(void) {
        auto that = static_cast<S*>(this);
        return (*that)[0];
    }
};


/**
 * Specialisation of size_accessors for 3D, which inherits from the 2D
 * specialisation and adds depth().
 *
 * @param S The actual size type, which inherits from this class.
 * @param T The traits type used for S. We need this information, because
 *          S is not yet complete when we need to determine value_type.
 */
template <class S, class T> struct size_accessors<S, T, 3> : public size_accessors<S, T, 2> {

    typedef typename T::value_type value_type;

    inline value_type depth(void) const {
        auto that = static_cast<const S*>(this);
        return (*that)[2];
    }

    inline value_type& depth(void) {
        auto that = static_cast<S*>(this);
        return (*that)[2];
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
#endif /* THE_MATH_SIZE_ACCESSORS_H_INCLUDED */
