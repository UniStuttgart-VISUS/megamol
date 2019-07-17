/*
 * the/math/size.h
 *
 * Copyright (C) 2014 TheLib Team (http://www.thelib.org/license)
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
/*
 * size.h  28.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef THE_MATH_SIZE_H_INCLUDED
#define THE_MATH_SIZE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#if defined(THE_WINDOWS)
#    include <windows.h>
#endif /* defined(THE_WINDOWS) */

#include "mmcore/thecam/utility/equatable.h"
#include "mmcore/thecam/utility/types.h"

#include "mmcore/thecam/math/implicit_dimension.h"
#include "mmcore/thecam/math/size_accessors.h"
#include "mmcore/thecam/math/vectorial_traits_base.h"

#ifdef WITH_THE_GLM
#    include <glm/glm.hpp>
#endif /* WITH_THE_GLM */

namespace megamol {
namespace core {
namespace thecam {
namespace math {

/**
 * The default type traits for an D-dimensional size.
 *
 * @tparam T The scalar value type used to specify one dimension or the
 *           native storage type in case of specialisations for existing
 *           libraries.
 * @tparam D The dimension (size) of the size.
 */
template <class T, size_t D> struct size_traits : public detail::vectorial_traits_base<T, D> {

    /** The base traits type. */
    typedef detail::vectorial_traits_base<T, D> base;

    /** The allocator for heap allocations of the size. */
    template <class C>
    using allocator_type = typename detail::template vectorial_traits_base<T, D>::template allocator_type<C>;

    /** The native type used to store the size. */
    typedef typename base::native_type native_type;

    /** The type to specify array dimensions and indices. */
    typedef typename base::size_type size_type;

    /** The scalar type used in the size. */
    typedef typename base::value_type value_type;
};


#if defined(THE_WINDOWS)
/**
 * Specialisation for the native SIZE type of Windows.
 */
template <> struct size_traits<SIZE, 2> {

    /** The allocator for heap allocations of the size. */
    template <class C> using allocator_type = std::allocator<C>;

    /** The native type used to store the size. */
    typedef SIZE native_type;

    /** The type to specify array dimensions and indices. */
    typedef size_t size_type;

    /** The scalar type used in the size. */
    typedef LONG value_type;

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
        static_assert(sizeof...(P) == 2, "The parameter list 'value' must "
                                         "contain all 2 components of the size.");
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
        THE_ASSERT(static_cast<size_t>(i) >= 0);
        THE_ASSERT(static_cast<size_t>(i) < 2);
        return (i == 0) ? data.cx : data.cy;
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
        THE_ASSERT(static_cast<size_t>(i) >= 0);
        THE_ASSERT(static_cast<size_t>(i) < 2);
        return (i == 0) ? data.cx : data.cy;
    }

    /**
     * Copy 'src' to 'dst'.
     *
     * @param dst The native storage of the destination.
     * @param src The native storage of the source.
     */
    static THE_FORCE_INLINE void copy(native_type& dst, const native_type& src) { dst = src; }

    /**
     * Test for equality of two native sizes.
     *
     * @param lhs The left-hand side operand.
     * @param rhs The right-hand side operand.
     *
     * @return true if 'lhs' and 'rhs' are equal, false otherwise.
     */
    static THE_FORCE_INLINE bool equals(const native_type& lhs, const native_type& rhs) {
        return (::memcmp(&lhs, &rhs, sizeof(native_type)) == 0);
    }
};
#endif /* defined(THE_WINDOWS) */


#if defined(WITH_THE_GLM)
/**
 * Specialisation for the native SIZE type of Windows.
 */
template <> struct size_traits<glm::ivec2, 2> {

    /** The allocator for heap allocations of the size. */
    template <class C> using allocator_type = std::allocator<C>;

    /** The native type used to store the size. */
    typedef glm::ivec2 native_type;

    /** The type to specify array dimensions and indices. */
    typedef size_t size_type;

    /** The scalar type used in the size. */
    typedef int value_type;

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
        static_assert(sizeof...(P) == 2, "The parameter list 'value' must "
                                         "contain all 2 components of the size.");
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
        THE_ASSERT(static_cast<size_t>(i) >= 0);
        THE_ASSERT(static_cast<size_t>(i) < 2);
        return (i == 0) ? data.x : data.y;
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
        THE_ASSERT(static_cast<size_t>(i) >= 0);
        THE_ASSERT(static_cast<size_t>(i) < 2);
        return (i == 0) ? data.x : data.y;
    }

    /**
     * Copy 'src' to 'dst'.
     *
     * @param dst The native storage of the destination.
     * @param src The native storage of the source.
     */
    static THE_FORCE_INLINE void copy(native_type& dst, const native_type& src) { dst = src; }

    /**
     * Test for equality of two native sizes.
     *
     * @param lhs The left-hand side operand.
     * @param rhs The right-hand side operand.
     *
     * @return true if 'lhs' and 'rhs' are equal, false otherwise.
     */
    static THE_FORCE_INLINE bool equals(const native_type& lhs, const native_type& rhs) {
        return (::memcmp(&lhs, &rhs, sizeof(native_type)) == 0);
    }
};
#endif /* defined(WITH_THE_GLM) */


/**
 * Stores a size in 'D' dimensions.
 *
 * @tparam V The scalar type used to store the size or the native size type
 *           to be wrapped.
 * @tparam D The number of dimensions. This defaults to
 *           megamol::core::thecam::math::detail::implicit_dimension<V>::value in order to
 *           automatically derive the dimension of built-in types via
 *           template specialisation.
 */
template <class V, size_t D = detail::implicit_dimension<V>::value, class T = size_traits<V, D>>
class size : public megamol::core::thecam::utility::equatable<size<V, D, T>>,
             public detail::size_accessors<size<V, D, T>, T, D> {

public:
    /** The type of the storage class actually holding the components. */
    typedef typename T::native_type native_type;

    /** The type to specify indices and dimensions. */
    typedef typename T::size_type size_type;

    /** The traits that provide manipulators for native_type objects. */
    typedef T traits_type;

    /** The type of a scalar. */
    typedef typename T::value_type value_type;

    /**
     * Initialise all dimensions with zero.
     */
    size(void);

    /**
     * Create a new instance, but do not initialise the components.
     */
    inline size(const utility::do_not_initialise_t) {}

    /**
     * Initialises a new size.
     *
     * Rationale: We require 'dx' to be explicitly specified in order to
     * resolve ambiguities. It is, however, reasonable to assume that a
     * size has at least one dimension.
     *
     * @param dx         The first component of the size.
     * @param components The initial values of the remaining components.
     */
    template <class... P> inline size(const value_type dx, P&&... components) {
        traits_type::assign(this->data, dx, std::forward<P>(components)...);
    }

    /**
     * Initialises a new size.
     *
     * @param il The initialiser list. If not all components are specified,
     *           the rest will be zero.
     */
    size(std::initializer_list<value_type> il);

    /**
     * Clone 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    inline size(const size& rhs) { traits_type::copy(this->data, rhs.data); }

    /**
     * Convert 'rhs'.
     *
     * @tparam Vp The value or native storage type of 'rhs'.
     * @tparam Tp The traits type of 'rhs'.
     *
     * @param rhs The object to be converted.
     */
    template <class Vp, class Tp> inline size(const size<Vp, D, Tp>& rhs) { *this = rhs; }

    /**
     * Initialises the size from its native representation.
     *
     * This constructor enables implicit casts from native_type.
     *
     * @param data The initial data.
     */
    THE_FORCE_INLINE size(const native_type& data) { traits_type::copy(this->data, data); }

    /**
     * Assign all components of the size.
     *
     * @tparam P The variable argument list, which must comprise
     *           dimensions() elements of value_type.
     *
     * @param components An argument list of D values to be assigned.
     */
    template <class... P> inline void assign(P&&... components) {
        traits_type::assign(this->data, std::forward<P>(components)...);
    }

    /**
     * Answer the number of components in the size.
     *
     * @return The dimension of the size.
     */
    inline size_type dimensions(void) const { return D; }

    /**
     * Answer whether the size is empty (all components exactly zero).
     *
     * @return true if the size is a null size, false otherwise.
     */
    inline bool empty(void) const {
        static const size EMPTY;
        return this->equals(EMPTY);
    }

    /**
     * Answer whether the size (approximately) empty.
     *
     * @param epsilon An epsilon value used for comparison. This defaults to
     *                megamol::core::thecam::math::epsilon<value_type>::value.
     *
     * @return true if the size is a null size, false otherwise.
     */
    bool empty(const value_type epsilon) const;

    /**
     * Test for equality.
     *
     * @param rhs The object to be compared.
     *
     * @return true if this object and 'rhs' are equal, false otherwise.
     */
    THE_FORCE_INLINE bool equals(const size& rhs) const { return traits_type::equals(this->data, rhs.data); }

    /**
     * Test for (approximate) equality.
     *
     * @tparam Vp The value or native storage type of 'rhs'.
     * @tparam Tp The traits type of 'rhs'.
     *
     * @param rhs     The object to be compared.
     * @param epsilon The epsilon value used for comparison.
     *
     * @return true if this object and 'rhs' are equal, false otherwise.
     */
    template <class Vp, size_t Dp, class Tp> bool equals(const size<Vp, Dp, Tp>& rhs, const value_type epsilon) const;

    /**
     * Answer the generalised volume (area in the 2D case) of the size.
     *
     * @return The volume of the size.
     */
    value_type volume(void) const;

    /**
     * Assignment.
     *
     * @param rhs The right-hand side operand.
     *
     * @return *this.
     */
    size& operator=(const size& rhs);

    /**
     * Conversion assignment.
     *
     * @tparam Vp The value or native storage type of 'rhs'.
     * @tparam Tp The traits type of 'rhs'.
     *
     * @param rhs The right-hand side operand.
     *
     * @return *this.
     */
    template <class Vp, class Tp> size& operator=(const size<Vp, D, Tp>& rhs);

    /**
     * Answer the given component.
     *
     * @param i The index of the component to retrieve, which must be within
     *          [0, dimensions()[. No range checks are performed by the
     *          callee.
     *
     * @return The 'i'th component.
     */
    THE_FORCE_INLINE value_type operator[](const size_type i) const { return traits_type::at(this->data, i); }
    /**
     * Answer the given component.
     *
     * @param i The index of the component to retrieve, which must be within
     *          [0, dimensions()[. No range checks are performed by the
     *          callee.
     *
     * @return A reference to the 'ith component.
     */
    THE_FORCE_INLINE value_type& operator[](const size_type i) { return traits_type::at(this->data, i); }

    /**
     * Conversion to native_type.
     *
     * @return The native representation of the size.
     */
    THE_FORCE_INLINE operator native_type&(void) { return this->data; }

    /**
     * Conversion to native_type.
     *
     * @return The native representation of the size.
     */
    THE_FORCE_INLINE operator const native_type&(void)const { return this->data; }

private:
    /** Actually stores the components of the size. */
    native_type data;
};

} /* end namespace math */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#include "mmcore/thecam/math/size.inl"

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_MATH_SIZE_H_INCLUDED */
