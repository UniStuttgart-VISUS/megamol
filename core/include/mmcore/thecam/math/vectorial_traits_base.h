/*
 * thecam/math/vectorial_traits_base.h
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

#ifndef THE_MATH_VECTORIAL_TRAITS_BASE_H_INCLUDED
#define THE_MATH_VECTORIAL_TRAITS_BASE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include <array>

#include "mmcore/thecam/utility/aligned_allocator.h"
#include "mmcore/thecam/utility/assert.h"
#include "mmcore/thecam/utility/force_inline.h"


namespace megamol {
namespace core {
namespace thecam {
namespace math {
namespace detail {


/**
 * A basic implementation of a traits type for vectorial types like vectors,
 * n-dimensional sizes and quaternions.
 *
 * The before-mentioned types share a great part of its implementation,
 * mostly because the use the same kind of storage (a fixed size array).
 * Therefore, this structure provides a shared implementation that is
 * customised for the actual types.
 *
 * There are specialisations that allow reinterpreting the type parameter T,
 * which is primarily intended to be the value type of scalars, as the
 * storage type. This way, we can easily provide wrappers around other
 * types, like the ones used by DirectX or GLM.
 *
 * Any specialisation must include the following typedefs:
 *
 * allocator_type: An allocator template that can be used to allocate
 *                 instances on the heap. The reason for providing the
 *                 allocator is that eg DirectX maths instances must be
 *                 16-byte aligned and therefore need a special allocator.
 * native_type: The type to actually store the data.
 * size_type: The type to specify array dimensions and indices.
 * value_type: The type of a scalar.
 *
 * Any specialisation must provide the following static methods:
 *
 * template<class... P> static void assign(native_type& dst, P&&... values);
 * static value_type at(const native_type& data, const size_type i);
 * static value_type& at(native_type& data, const size_type i);
 * static void copy(native_type& dst, const native_type& src)
 * static bool equals(const native_type& lhs, const native_type& rhs);
 *
 * @tparam T The scalar value type or the native storage type in case of
 *           specialisations for existing libraries.
 * @tparam N The dimension of the vector.
 */
template <class T, size_t N> struct vectorial_traits_base {

    /** The allocator for heap allocations of the vector class. */
    template <class C> using allocator_type = std::allocator<C>;

    /** The native storage type. */
    typedef std::array<T, N> native_type;

    /** The type to specify array dimensions and indices. */
    typedef size_t size_type;

    /** The scalar type. */
    typedef T value_type;

    /**
     * Assign all components of the vector to 'dst'.
     *
     * @tparam P The variable argument list, which must comprise N elements
     *           of value_type.
     *
     * @param dst    The native storage of the destination.
     * @param values An argument list of N values to be assigned to the
     *               components of 'dst'.
     */
    template <class... P> static THE_FORCE_INLINE void assign(native_type& dst, P&&... values) {
        static_assert(sizeof...(P) == N, "The parameter list 'value' must "
                                         "contain all components of the vector.");
        dst = {values...};
    }

    /**
     * Get the value of the specified component 'i'.
     *
     * @param data The native data.
     * @param i    The component to retrieve.
     *
     * @return The value of the 'i'th component.
     */
    static THE_FORCE_INLINE value_type at(const native_type& data, const size_type i) {
        THE_ASSERT(i >= 0);
        THE_ASSERT(i < N);
        return data[i];
    }

    /**
     * Get a non-constant reference for component 'i'.
     *
     * @param data The native data.
     * @param i    The component to retrieve.
     *
     * @return A reference to the 'i'th component.
     */
    static THE_FORCE_INLINE value_type& at(native_type& data, const size_type i) {
        THE_ASSERT(i >= 0);
        THE_ASSERT(i < N);
        return data[i];
    }

    /**
     * Copy 'src' to 'dst'.
     *
     * @param dst The native storage of the destination.
     * @param src The native storage of the source.
     */
    static THE_FORCE_INLINE void copy(native_type& dst, const native_type& src) {
        std::copy(src.cbegin(), src.cend(), dst.begin());
    }

    /**
     * Test for equality of two native quaternions.
     *
     * @param lhs The left-hand side operand.
     * @param rhs The right-hand side operand.
     *
     * @return true if 'lhs' and 'rhs' are equal, false otherwise.
     */
    static THE_FORCE_INLINE bool equals(const native_type& lhs, const native_type& rhs) { return (lhs == rhs); }
};
} /* end namespace detail */
} /* end namespace math */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_MATH_VECTORIAL_TRAITS_BASE_H_INCLUDED */
