/*
 * the/math/rectangle.h
 *
 * Copyright (C) 2012 - 2016 TheLib Team (http://www.thelib.org/license)
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
 * Rectangle.h  27.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef THE_MATH_RECTANGLE_H_INCLUDED
#define THE_MATH_RECTANGLE_H_INCLUDED
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

#include "mmcore/thecam/utility/assert.h"
#include "mmcore/thecam/utility/equatable.h"
#include "mmcore/thecam/utility/types.h"

#include "mmcore/thecam/math/functions.h"
#include "mmcore/thecam/math/point.h"
#include "mmcore/thecam/math/size.h"
#include "mmcore/thecam/math/vectorial_traits_base.h"


namespace megamol {
namespace core {
namespace thecam {
namespace math {

/**
 * Index values to access rectangle components.
 */
enum class rectangle_component {
    left = 0,   //!< The index of the left border of the rectangle.
    top = 1,    //!< The index of the top border of the rectangle.
    right = 2,  //!< The index of the right border of the rectangle.
    bottom = 3, //!< The index of the bottom border of the rectangle.
};


/**
 * The default type traits for a rectangle that is being stored as flat
 * array indexed by rectangle_component.
 *
 * @tparam T The scalar value type used in the rectangle or the native
 *           storage type in case of specialisations for existing libraries.
 */
template <class T> struct rectangle_traits : public detail::vectorial_traits_base<T, 4> {

    /** The base traits type. */
    typedef detail::vectorial_traits_base<T, 4> base;

    /** The allocator for heap allocations of the rectangles. */
    template <class C>
    using allocator_type = typename detail::template vectorial_traits_base<T, 4>::template allocator_type<C>;

    /** The native type used to store the rectangle. */
    typedef typename base::native_type native_type;

    /** The type to specify array dimensions and indices. */
    typedef typename base::size_type size_type;

    /** The scalar type used in the rectangle. */
    typedef typename base::value_type value_type;
};


#if defined(THE_WINDOWS)
/**
 * Specialisation for the native RECT type of Windows.
 */
template <> struct rectangle_traits<RECT> {

    /** The allocator for heap allocations of the rectangle. */
    template <class C> using allocator_type = std::allocator<C>;

    /** The native type used to store the rectangle. */
    typedef RECT native_type;

    /** The type to specify array dimensions and indices. */
    typedef size_t size_type;

    /** The scalar type used in the rectangle. */
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
        static_assert(sizeof...(P) == 4, "The parameter list 'value' must "
                                         "contain all components of the rectangle.");
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
        static_assert(sizeof(native_type) == 4 * sizeof(value_type),
            "Uhm, this stunt is only working if there is no padding in "
            "native_type.");
        THE_ASSERT(static_cast<size_t>(i) >= 0);
        THE_ASSERT(static_cast<size_t>(i) < 4);
        return (&data.left)[i];
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
        static_assert(sizeof(native_type) == 4 * sizeof(value_type),
            "Uhm, this stunt is only working if there is no padding in "
            "native_type.");
        THE_ASSERT(static_cast<size_t>(i) >= 0);
        THE_ASSERT(static_cast<size_t>(i) < 4);
        return (&data.left)[i];
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


/**
 * A rectangle.
 *
 * @tparam V The scalar type used to store the rectangle's components or
 *           the native size type to be wrapped.
 */
template <class V, class T = rectangle_traits<V>>
class rectangle : public megamol::core::thecam::utility::equatable<rectangle<V, T>> {

public:
    /** The type to express extents of a rectangle. */
    typedef size<typename T::value_type, 2> extent_type;

    /** The type of the storage class actually holding the components. */
    typedef typename T::native_type native_type;

    /** The type to specify indices and dimensions. */
    typedef typename T::size_type size_type;

    /** The traits that provide manipulators for native_type objects. */
    typedef T traits_type;

    /** The type of a scalar. */
    typedef typename T::value_type value_type;

    /**
     * Create a new rectangle from an origin at the left/bottom corner and
     * and the size.
     *
     * The rectangle is constructed under the assumption that the y-axis is
     * running bottom up, ie the top bound will be computed as
     * 'bottom' + 'height'.
     *
     * @param left   The left bound of the rectangle.
     * @param bottom The bottom bound of the rectangle.
     * @param width  The width of the rectangle.
     * @param height The height of the rectangle.
     *
     * @returns A rectangle with the specified extents.
     */
    static inline rectangle from_bottom_left(
        const value_type left, const value_type bottom, const value_type width, const value_type height) {
        rectangle retval(megamol::core::thecam::utility::do_not_initialise);
        traits_type::assign(retval.data, left, bottom + height, left + width, bottom);
        return retval;
    }

    /**
     * Create a new rectangle from an origin at the left/bottom corner and
     * and the size.
     *
     * The rectangle is constructed under the assumption that the y-axis is
     * running bottom up, ie the top bound will be computed as
     * 'bottom' + 'height'.
     *
     * @param point The left/bottom point of the rectangle.
     * @param size  The size of the rectangle
     *
     * @returns A rectangle with the specified extents.
     */
    template <class P, class S> static rectangle from_bottom_left(const P& point, const S& size);

    /**
     * Create a new rectangle from its bounds.
     *
     * @param left   The left bound of the rectangle.
     * @param top    The top bound of the rectangle.
     * @param right  The right bound of the rectangle.
     * @param bottom The bottom bound of the rectangle.
     *
     * @returns A rectangle with the specified bounds.
     */
    static inline rectangle from_bounds(
        const value_type left, const value_type top, const value_type right, const value_type bottom) {
        rectangle retval(megamol::core::thecam::utility::do_not_initialise);
        traits_type::assign(retval.data, left, top, right, bottom);
        return retval;
    }

    /**
     * Create a new rectangle from an origin at the left/top corner and
     * and the size.
     *
     * The rectangle is constructed under the assumption that the y-axis is
     * running top down, ie the bottom bound will be computed as
     * 'top' + 'height'.
     *
     * @param left   The left bound of the rectangle.
     * @param top    The top bound of the rectangle.
     * @param width  The width of the rectangle.
     * @param height The height of the rectangle.
     *
     * @returns A rectangle with the specified extents.
     */
    static inline rectangle from_top_left(
        const value_type left, const value_type top, const value_type width, const value_type height) {
        rectangle retval(megamol::core::thecam::utility::do_not_initialise);
        traits_type::assign(retval.data, left, top, left + width, top + height);
        return retval;
    }

    /**
     * Create a new rectangle from an origin at the left/top corner and
     * and the size.
     *
     * The rectangle is constructed under the assumption that the y-axis is
     * running top down, ie the bottom bound will be computed as
     * 'top' + 'height'.
     *
     * @param point The left/top point of the rectangle.
     * @param size  The size of the rectangle
     *
     * @returns A rectangle with the specified extents.
     */
    template <class P, class S> static rectangle from_top_left(const P& point, const S& size);

    /**
     * Initialise all bounds with zero.
     */
    rectangle(void);

    /**
     * Create a new instance, but do not initialise the components.
     */
    inline rectangle(const utility::do_not_initialise_t) {}

    /**
     * Initialises a new size.
     *
     * @param il The initialiser list. The expected order is left, top,
     *           right, bottom. If not all components are specified,
     *           the rest will be zero.
     */
    rectangle(std::initializer_list<value_type> il);

    /**
     * Clone 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    inline rectangle(const rectangle& rhs) { traits_type::copy(this->data, rhs.data); }

    /**
     * Convert 'rhs'.
     *
     * @tparam Vp The value or native storage type of 'rhs'.
     * @tparam Tp The traits type of 'rhs'.
     *
     * @param rhs The object to be converted.
     */
    template <class Vp, class Tp> inline rectangle(const rectangle<Vp, Tp>& rhs) { *this = rhs; }

    /**
     * Initialises the rectangle from its native representation.
     *
     * This constructor enables implicit casts from native_type.
     *
     * @param data The initial data.
     */
    THE_FORCE_INLINE rectangle(const native_type& data) { traits_type::copy(this->data, data); }

    /**
     * Answer the area of the rectangle.
     *
     * @return The area covered by the rectangle.
     */
    inline value_type area(void) const { return (this->width() * this->height()); }

    /**
     * Assign all components of the rectangle.
     *
     * @tparam P The variable argument list, which must comprise rectangle()
     *           elements of value_type.
     *
     * @param components An argument list of D values to be assigned.
     */
    template <class... P> inline void assign(P&&... components) {
        traits_type::assign(this->data, std::forward<P>(components)...);
    }

    /**
     * Answer the bottom bound of the rectangle.
     *
     * @return The bottom bound of the rectangle.
     */
    THE_FORCE_INLINE value_type bottom(void) const {
        return traits_type::at(this->data, static_cast<size_type>(rectangle_component::bottom));
    }

    /**
     * Answer the bottom bound of the rectangle.
     *
     * @return The bottom bound of the rectangle.
     */
    THE_FORCE_INLINE value_type& bottom(void) {
        return traits_type::at(this->data, static_cast<size_type>(rectangle_component::bottom));
    }

    // TODO y-direction?
    ///**
    // * Answer whether the given point is within the bounds of the rectangle.
    // *
    // * @param point The point to be tested.
    // */
    // template<class P> bool contains(const P& point) const;

    /**
     * Answer whether the rectangle is all null (all components exactly
     * zero).
     *
     * Note: This method does not check whether the area of the rectangle is
     * zero!
     *
     * @return true if the size is a null size, false otherwise.
     */
    bool empty(void) const;

    /**
     * Answer whether the rectangle is approximately empty.
     *
     * Note: This method does not check whether the area of the rectangle is
     * approximately zero!
     *
     * @param epsilon An epsilon value used for comparison. This defaults to
     *                megamol::core::thecam::math::epsilon<value_type>::value.
     *
     * @return true if the size is a null rectangle, false otherwise.
     */
    bool empty(const value_type epsilon) const;

    /**
     * Test for equality.
     *
     * @param rhs The object to be compared.
     *
     * @return true if this object and 'rhs' are equal, false otherwise.
     */
    THE_FORCE_INLINE bool equals(const rectangle& rhs) const { return traits_type::equals(this->data, rhs.data); }

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
    template <class Vp, class Tp> bool equals(const rectangle<Vp, Tp>& rhs, const value_type epsilon) const;

    /**
     * Answer the size of the rectangle.
     *
     * @return The size (width and height) of the rectangle.
     */
    inline extent_type extent(void) const { return extent_type(this->width(), this->height()); }

    /**
     * Answer the height of the rectangle.
     *
     * @return The height of the rectangle. The height is always positive.
     */
    inline value_type height(void) const { return std::abs(this->bottom() - this->top()); }

    /**
     * Answer the left bound of the rectangle.
     *
     * @return The left bound of the rectangle.
     */
    THE_FORCE_INLINE value_type left(void) const {
        return traits_type::at(this->data, static_cast<size_type>(rectangle_component::left));
    }

    /**
     * Answer the left bound of the rectangle.
     *
     * @return The left bound of the rectangle.
     */
    THE_FORCE_INLINE value_type& left(void) {
        return traits_type::at(this->data, static_cast<size_type>(rectangle_component::left));
    }

    /**
     * Answer the right bound of the rectangle.
     *
     * @return The right bound of the rectangle.
     */
    THE_FORCE_INLINE value_type right(void) const {
        return traits_type::at(this->data, static_cast<size_type>(rectangle_component::right));
    }

    /**
     * Answer the right bound of the rectangle.
     *
     * @return The right bound of the rectangle.
     */
    THE_FORCE_INLINE value_type& right(void) {
        return traits_type::at(this->data, static_cast<size_type>(rectangle_component::right));
    }

    /**
     * Answer the number of components in the rectangle.
     *
     * @return 4.
     */
    inline size_type size_(void) const { return 4; }

    /**
     * Answer the top bound of the rectangle.
     *
     * @return The top bound of the rectangle.
     */
    THE_FORCE_INLINE value_type top(void) const {
        return traits_type::at(this->data, static_cast<size_type>(rectangle_component::top));
    }

    /**
     * Answer the top bound of the rectangle.
     *
     * @return The top bound of the rectangle.
     */
    THE_FORCE_INLINE value_type& top(void) {
        return traits_type::at(this->data, static_cast<size_type>(rectangle_component::top));
    }

    /**
     * Answer the width of the rectangle.
     *
     * @return The width of the rectangle. The width is always positive.
     */
    inline value_type width(void) const { return std::abs(this->right() - this->left()); }

    /**
     * Assignment.
     *
     * @param rhs The right-hand side operand.
     *
     * @return *this.
     */
    rectangle& operator=(const rectangle& rhs);

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
    template <class Vp, class Tp> rectangle& operator=(const rectangle<Vp, Tp>& rhs);

    /**
     * Answer the given component.
     *
     * @param i The index of the component to retrieve.
     *
     * @return The 'i'th component.
     */
    THE_FORCE_INLINE value_type operator[](const rectangle_component i) const {
        return traits_type::at(this->data, static_cast<size_type>(i));
    }
    /**
     * Answer the given component.
     *
     * @param i The index of the component to retrieve.
     *
     * @return A reference to the 'ith component.
     */
    THE_FORCE_INLINE value_type& operator[](const rectangle_component i) {
        return traits_type::at(this->data, static_cast<size_type>(i));
    }

    /**
     * Conversion to native_type.
     *
     * @return The native representation of the rectangle.
     */
    THE_FORCE_INLINE operator native_type&(void) { return this->data; }

    /**
     * Conversion to native_type.
     *
     * @return The native representation of the rectangle.
     */
    THE_FORCE_INLINE operator const native_type&(void)const { return this->data; }

private:
    /** Actually stores the bounds of the rectangle. */
    native_type data;
};

} /* end namespace math */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#include "mmcore/thecam/math/rectangle.inl"

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_MATH_RECTANGLE_H_INCLUDED */
