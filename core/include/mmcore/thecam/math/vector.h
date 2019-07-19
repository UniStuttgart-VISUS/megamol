/*
 * thecam/math/vector.h
 *
 * Copyright (c) 2012 - 2016 TheLib Team (http://www.thelib.org/license)
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
 * vector.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef THE_MATH_VECTOR_H_INCLUDED
#define THE_MATH_VECTOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include <type_traits>
#include <vector>

#ifdef WITH_THE_GLM
#    include <glm/glm.hpp>
#endif /* WITH_THE_GLM */

#include "mmcore/thecam/utility/assert.h"
#include "mmcore/thecam/utility/equatable.h"
#include "mmcore/thecam/utility/is_parameter_set_size.h"
#include "mmcore/thecam/utility/memory.h"

#include "mmcore/thecam/math/functions.h"
#include "mmcore/thecam/math/implicit_dimension.h"
#include "mmcore/thecam/math/mathtypes.h"
#include "mmcore/thecam/math/vector_accessors.h"
#include "mmcore/thecam/math/vectorial_traits_base.h"


namespace megamol {
namespace core {
namespace thecam {
namespace math {

/**
 * The default type traits for an D-dimensional vector.
 *
 * There are full template specialisations which allow reinterpreting T as
 * the actual vector type of DirectX math and GLM, for instance.
 *
 * @tparam T The scalar value type used in the vector or the native
 *           storage type in case of specialisations for existing libraries.
 * @tparam D The dimension (size) of the vector.
 */
template <class T, size_t D> struct vector_traits : public detail::vectorial_traits_base<T, D> {

    /** The base traits type. */
    typedef detail::vectorial_traits_base<T, D> base;

    /** The allocator for heap allocations of the vectors. */
    template <class C>
    using allocator_type = typename detail::template vectorial_traits_base<T, D>::template allocator_type<C>;

    /** The native type used to store the vector. */
    typedef typename base::native_type native_type;

    /** The type to specify array dimensions and indices. */
    typedef typename base::size_type size_type;

    /** The scalar type used in the vector. */
    typedef typename base::value_type value_type;
};


#ifdef WITH_THE_GLM
/**
 * Specialisation of the vector_traits 4-dimensional GLM vectors.
 */
template <> struct vector_traits<glm::vec4, 4> {

    /** The allocator for heap allocations of the vector class. */
    template <class C> using allocator_type = std::allocator<C>;

    /** The native storage type. */
    typedef glm::vec4 native_type;

    /** The type to specify array dimensions and indices. */
    typedef size_t size_type;

    /** The scalar type. */
    typedef float value_type;

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
    template <class... P> static THE_TRY_FORCE_INLINE void assign(native_type& dst, P&&... values) {
        static_assert(sizeof...(P) == 4, "The parameter list 'value' must "
                                         "contain all 4 components of the vector.");
        dst = native_type(values...);
    }

    /**
     * Get the value of the specified component 'i'.
     *
     * @param data The native data.
     * @param i    The component to retrieve.
     *
     * @return The value of the 'i'th component.
     */
    static THE_TRY_FORCE_INLINE value_type at(const native_type& data, const size_type i) {
        THE_ASSERT(static_cast<size_t>(i) >= 0);
        THE_ASSERT(static_cast<size_t>(i) < 4);
        return data[static_cast<native_type::length_type>(i)];
    }

    /**
     * Get a non-constant reference for component 'i'.
     *
     * @param data The native data.
     * @param i    The component to retrieve.
     *
     * @return A reference to the 'i'th component.
     */
    static THE_TRY_FORCE_INLINE value_type& at(native_type& data, const size_type i) {
        THE_ASSERT(static_cast<size_t>(i) >= 0);
        THE_ASSERT(static_cast<size_t>(i) < 4);
        return data[static_cast<native_type::length_type>(i)];
    }

    /**
     * Copy 'src' to 'dst'.
     *
     * @param dst The native storage of the destination.
     * @param src The native storage of the source.
     */
    static THE_TRY_FORCE_INLINE void copy(native_type& dst, const native_type& src) { dst = src; }

    /**
     * Test for equality of two native vectors.
     *
     * @param lhs The left-hand side operand.
     * @param rhs The right-hand side operand.
     *
     * @return true if 'lhs' and 'rhs' are equal, false otherwise.
     */
    static THE_TRY_FORCE_INLINE bool equals(const native_type& lhs, const native_type& rhs) { return (lhs == rhs); }
};
#endif /* WITH_THE_GLM */

#ifdef WITH_THE_GLM
/**
 * Specialisation of the vector_traits 3-dimensional GLM vectors.
 */
template <> struct vector_traits<glm::vec3, 3> {

    /** The allocator for heap allocations of the vector class. */
    template <class C> using allocator_type = std::allocator<C>;

    /** The native storage type. */
    typedef glm::vec3 native_type;

    /** The type to specify array dimensions and indices. */
    typedef size_t size_type;

    /** The scalar type. */
    typedef float value_type;

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
    template <class... P> static THE_TRY_FORCE_INLINE void assign(native_type& dst, P&&... values) {
        static_assert(sizeof...(P) == 3, "The parameter list 'value' must "
                                         "contain all 3 components of the vector.");
        dst = native_type(values...);
    }

    /**
     * Get the value of the specified component 'i'.
     *
     * @param data The native data.
     * @param i    The component to retrieve.
     *
     * @return The value of the 'i'th component.
     */
    static THE_TRY_FORCE_INLINE value_type at(const native_type& data, const size_type i) {
        THE_ASSERT(static_cast<size_t>(i) >= 0);
        THE_ASSERT(static_cast<size_t>(i) < 3);
        return data[static_cast<native_type::length_type>(i)];
    }

    /**
     * Get a non-constant reference for component 'i'.
     *
     * @param data The native data.
     * @param i    The component to retrieve.
     *
     * @return A reference to the 'i'th component.
     */
    static THE_TRY_FORCE_INLINE value_type& at(native_type& data, const size_type i) {
        THE_ASSERT(static_cast<size_t>(i) >= 0);
        THE_ASSERT(static_cast<size_t>(i) < 3);
        return data[static_cast<native_type::length_type>(i)];
    }

    /**
     * Copy 'src' to 'dst'.
     *
     * @param dst The native storage of the destination.
     * @param src The native storage of the source.
     */
    static THE_TRY_FORCE_INLINE void copy(native_type& dst, const native_type& src) { dst = src; }

    /**
     * Test for equality of two native vectors.
     *
     * @param lhs The left-hand side operand.
     * @param rhs The right-hand side operand.
     *
     * @return true if 'lhs' and 'rhs' are equal, false otherwise.
     */
    static THE_TRY_FORCE_INLINE bool equals(const native_type& lhs, const native_type& rhs) { return (lhs == rhs); }
};
#endif /* WITH_THE_GLM */


#ifdef WITH_THE_GLM
/**
 * Specialisation of the vector_traits 3-dimensional GLM vectors.
 */
template <> struct vector_traits<glm::vec2, 2> {

    /** The allocator for heap allocations of the vector class. */
    template <class C> using allocator_type = std::allocator<C>;

    /** The native storage type. */
    typedef glm::vec2 native_type;

    /** The type to specify array dimensions and indices. */
    typedef size_t size_type;

    /** The scalar type. */
    typedef float value_type;

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
    template <class... P> static THE_TRY_FORCE_INLINE void assign(native_type& dst, P&&... values) {
        static_assert(sizeof...(P) == 2, "The parameter list 'value' must "
                                         "contain all 2 components of the vector.");
        dst = native_type(values...);
    }

    /**
     * Get the value of the specified component 'i'.
     *
     * @param data The native data.
     * @param i    The component to retrieve.
     *
     * @return The value of the 'i'th component.
     */
    static THE_TRY_FORCE_INLINE value_type at(const native_type& data, const size_type i) {
        THE_ASSERT(static_cast<size_t>(i) >= 0);
        THE_ASSERT(static_cast<size_t>(i) < 2);
        return data[static_cast<native_type::length_type>(i)];
    }

    /**
     * Get a non-constant reference for component 'i'.
     *
     * @param data The native data.
     * @param i    The component to retrieve.
     *
     * @return A reference to the 'i'th component.
     */
    static THE_TRY_FORCE_INLINE value_type& at(native_type& data, const size_type i) {
        THE_ASSERT(static_cast<size_t>(i) >= 0);
        THE_ASSERT(static_cast<size_t>(i) < 2);
        return data[static_cast<native_type::length_type>(i)];
    }

    /**
     * Copy 'src' to 'dst'.
     *
     * @param dst The native storage of the destination.
     * @param src The native storage of the source.
     */
    static THE_TRY_FORCE_INLINE void copy(native_type& dst, const native_type& src) { dst = src; }

    /**
     * Test for equality of two native vectors.
     *
     * @param lhs The left-hand side operand.
     * @param rhs The right-hand side operand.
     *
     * @return true if 'lhs' and 'rhs' are equal, false otherwise.
     */
    static THE_TRY_FORCE_INLINE bool equals(const native_type& lhs, const native_type& rhs) { return (lhs == rhs); }
};
#endif /* WITH_THE_GLM */


/**
 * Implementation of a D-dimensional vector.
 *
 * The vector can be instantiated for a scalar value type and a dimension or
 * for the the native storage type. The interpretation of the template
 * parameter 'V' depends on the type traits 'T'.
 *
 * The class is only intended to be a container which does not support a lot
 * of operations. Operations are provided as free functions, but only to an
 * extent as they are required for implementing functionality required in
 * TheLib, eg in the camera class. Other geometric functions should be
 * obtained from an external library, which for you provide traits for.
 *
 * @tparam V The scalar value type used in the vector or the native
 *           storage type in case of specialisations for existing libraries.
 * @tparam D The dimension of the vector. If 'V' is a storage type, there
 *           must be a specialisation for detail::implicit_dimension<V> which
 *           derives the dimension of the vector from 'V'. If there is no
 *           implicit dimension, the dimension must be specified.
 * @tparam T The traits type that interprets 'V' and provides the scalar
 *           type, the storage type and basic operations.
 */
template <class V, size_t D = detail::implicit_dimension<V>::value, class T = vector_traits<V, D>>
class vector : public megamol::core::thecam::utility::equatable<vector<V, D, T>>,
               public detail::vector_accessors<vector<V, D, T>, T, D> {

public:
    /** The type of the storage class actually holding the components. */
    typedef typename T::native_type native_type;

    /** The type to specify indices and dimensions. */
    typedef typename T::size_type size_type;

    /** The traits that provide manipulators for native_type objects. */
    typedef T traits_type;

    /** The type of a scalar. */
    typedef typename T::value_type value_type;

#ifdef TODO_ALIGNED_TYPES
    /**
     * Frees heap allocations of the class.
     *
     * @param ptr  A pointer to memory that has been allocated with
     *             vector::operator new.
     * @param size The size that has been passed to the allocator.
     */
    static void operator delete(void* ptr, const std::size_t size) {
        // Note: operator new/new[]/delete/delete[] work on bytes, not on
        // number of objects!
        static typename traits_type::allocator_type<byte> alloc;
        alloc.deallocate(static_cast<byte*>(ptr), size);
    }

    /**
     * Frees heap allocations of the class.
     *
     * @param ptr  A pointer to memory that has been allocated with
     *             vector::operator new[].
     * @param size The size that has been passed to the allocator.
     */
    static void operator delete[](void* ptr, const std::size_t size) {
        // Note: operator new/new[]/delete/delete[] work on bytes, not on
        // number of objects!
        static typename traits_type::allocator_type<byte> alloc;
        alloc.deallocate(static_cast<byte*>(ptr), size);
    }

    /**
     * Allocates a new instance on the heap.
     *
     * @param size The amount of memory to be allocated (in bytes).
     */
    static void* operator new(const std::size_t size) {
        // Note: operator new/new[]/delete/delete[] work on bytes, not on
        // number of objects!
        static typename traits_type::allocator_type<byte> alloc;
        return alloc.allocate(size);
    }

    /**
     * Allocates a new instance array on the heap.
     *
     * @param size The amount of memory to be allocated (in bytes).
     */
    static void* operator new[](const std::size_t size) {
        // Note: operator new/new[]/delete/delete[] work on bytes, not on
        // number of objects!
        static typename traits_type::allocator_type<byte> alloc;
        return alloc.allocate(size);
    }
#endif

    /**
     * Initialises an empty vector, all components zero.
     */
    inline vector(void) { set_empty(*this); }

    /**
     * Create a new vector, but do not initialise the components.
     */
    inline vector(const utility::do_not_initialise_t) {}

    /**
     * Initialises a new vector.
     *
     * Rationale: We require 'x' to be explicitly specified in order to
     * resolve ambiguities. It is, however, reasonable to assume that a
     * vector has at least one dimension.
     *
     * @param x          The first component of the vector.
     * @param components The initial values of the remaining components.
     */
    template <class... P> inline vector(const value_type x, P&&... components) {
        traits_type::assign(this->data, x, std::forward<P>(components)...);
    }

    /**
     * Initialises a new vector.
     *
     * @param il The initialiser list. If not all components are specified,
     *           the rest will be zero.
     */
    vector(std::initializer_list<value_type> il);

    /**
     * Clone 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    inline vector(const vector& rhs) { traits_type::copy(this->data, rhs.data); }

    /**
     * Convert 'rhs'.
     *
     * @tparam Vp The value or native storage type of 'rhs'.
     * @tparam Tp The traits type of 'rhs'.
     *
     * @param rhs The object to be converted.
     */
    template <class Vp, class Tp> inline vector(const vector<Vp, D, Tp>& rhs) { *this = rhs; }

    /**
     * Widening/narrowing conversion of a vector. If the vector 'rhs' is
     * widened, the components in the back will be set to 'value'.
     *
     * @tparam Vp The value or native storage type of 'rhs'.
     * @tparam Tp The traits type of 'rhs'.
     *
     * @param rhs   The object to be converted.
     * @param value The value to fill missing components at the end.
     */
    template <class Vp, size_t Dp, class Tp> explicit vector(const vector<Vp, Dp, Tp>& rhs, const value_type value = 0);

    /**
     * Initialises the vector from its native representation.
     *
     * This constructor enables implicit casts from native_type.
     *
     * @param data The initial data.
     */
    THE_TRY_FORCE_INLINE vector(const native_type& data) { traits_type::copy(this->data, data); }

    /**
     * Assign all components of the vector.
     *
     * @tparam P The variable argument list, which must comprise size()
     *           elements of value_type.
     *
     * @param components An argument list of D values to be assigned.
     */
    template <class... P> inline void assign(P&&... components) {
        traits_type::assign(this->data, std::forward<P>(components)...);
    }

    /**
     * Answer whether the vector is a null vector. (all components are
     * exactly zero).
     *
     * Depending on the underlying storage class, this implementation might
     * be vectorised and therefore more efficient than the epsilon
     * comparison.
     *
     * @return true if the vector is a null vector, false otherwise.
     */
    inline bool empty(void) const {
        static const vector EMPTY;
        return this->equals(EMPTY);
    }

    /**
     * Answer whether the vector is a null vector.
     *
     * @param epsilon An epsilon value used for comparison. This defaults to
     *                megamol::core::thecam::math::epsilon<value_type>::value.
     *
     * @return true if the vector is a null vector, false otherwise.
     */
    bool empty(const value_type epsilon) const;

    /**
     * Test for equality.
     *
     * @param rhs The object to be compared.
     *
     * @return true if this object and 'rhs' are equal, false otherwise.
     */
    THE_TRY_FORCE_INLINE bool equals(const vector& rhs) const { return traits_type::equals(this->data, rhs.data); }

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
    template <class Vp, size_t Dp, class Tp> bool equals(const vector<Vp, Dp, Tp>& rhs, const value_type epsilon) const;

    /**
     * Answer the number of components in the vector.
     *
     * @return The dimension of the vector.
     */
    inline size_type size(void) const { return D; }

    /**
     * Assignment.
     *
     * @param rhs The right-hand side operand.
     *
     * @return *this.
     */
    vector& operator=(const vector& rhs);

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
    template <class Vp, class Tp> vector& operator=(const vector<Vp, D, Tp>& rhs);

    /**
     * Answer the given component.
     *
     * @param i The index of the component to retrieve, which must be within
     *          [0, size()[. No range checks are performed by the callee.
     *
     * @return The 'i'th component.
     */
    THE_TRY_FORCE_INLINE value_type operator[](const size_type i) const { return traits_type::at(this->data, i); }
    /**
     * Answer the given component.
     *
     * @param i The index of the component to retrieve, which must be within
     *          [0, size()[. No range checks are performed by the callee.
     *
     * @return A reference to the 'ith component.
     */
    THE_TRY_FORCE_INLINE value_type& operator[](const size_type i) { return traits_type::at(this->data, i); }

    /**
     * Conversion to native_type.
     *
     * @return The native representation of the vector.
     */
    THE_TRY_FORCE_INLINE operator native_type&(void) { return this->data; }

    /**
     * Conversion to native_type.
     *
     * @return The native representation of the vector.
     */
    THE_TRY_FORCE_INLINE operator const native_type&(void)const { return this->data; }

private:
    /** Actually stores the components of the vector. */
    native_type data;
};

/**
 * Compute the cross product between two vectors.
 *
 * @tparam V The scalar value or storage type of the vectors.
 * @tparam T The traits of the vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return The cross product of 'lhs' and 'rhs'.
 */
template <class V, class T>
inline typename T::value_type cross(const vector<V, 2, T>& lhs, const vector<V, 2, T>& rhs) {
    return lhs.x() * rhs.y() - rhs.x() * lhs.y();
}


/**
 * Compute the cross product between two vectors.
 *
 * @tparam V The scalar value or storage type of the vectors.
 * @tparam T The traits of the vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return The cross product of 'lhs' and 'rhs'.
 */
template <class V, class T> vector<V, 3, T> cross(const vector<V, 3, T>& lhs, const vector<V, 3, T>& rhs);


/**
 * Compute the cross product between two vectors (their first three
 * components).
 *
 * @tparam V The scalar value or storage type of the vectors.
 * @tparam T The traits of the vectors.
 *
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return The cross product of 'lhs' and 'rhs', with the w-component
 *         being 0.
 */
template <class V, class T> vector<V, 4, T> cross(const vector<V, 4, T>& lhs, const vector<V, 4, T>& rhs);


#ifdef WITH_THE_GLM
/**
 * Compute the cross product between two vectors (their first three
 * components).
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return The cross product of 'lhs' and 'rhs', with the w-component
 *         being 0.
 */
THE_TRY_FORCE_INLINE vector<glm::vec3> cross(const vector<glm::vec3>& lhs, const vector<glm::vec3>& rhs) {
    glm::vec3 l(static_cast<const glm::vec3&>(lhs));
    glm::vec3 r(static_cast<const glm::vec3&>(rhs));
    return glm::cross(l, r);
}


/**
 * Compute the cross product between two vectors (their first three
 * components).
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return The cross product of 'lhs' and 'rhs', with the w-component
 *         being 0.
 */
THE_TRY_FORCE_INLINE vector<glm::vec4> cross(const vector<glm::vec4>& lhs, const vector<glm::vec4>& rhs) {
    glm::vec3 l(static_cast<const glm::vec4&>(lhs));
    glm::vec3 r(static_cast<const glm::vec4&>(rhs));
    glm::vec3 v = glm::cross(l, r);
    return vector<glm::vec4>(v.x, v.y, v.z, 0.0f);
}
#endif /* WITH_THE_GLM */

/**
 * Compute the dot product between two vectors.
 *
 * @tparam V The scalar value or storage type of the vectors.
 * @tparam D The dimension of the vectors.
 * @tparam T The traits of the vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return The dot product of 'lhs' and 'rhs'.
 */
template <class V, size_t D, class T>
typename T::value_type dot(const vector<V, D, T>& lhs, const vector<V, D, T>& rhs);


#ifdef WITH_THE_GLM
/**
 * Compute the dot product between two vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return The dot product of 'lhs' and 'rhs'.
 */
THE_TRY_FORCE_INLINE float dot(const vector<glm::vec2>& lhs, const vector<glm::vec2>& rhs) {
    auto& l = static_cast<const glm::vec2&>(lhs);
    auto& r = static_cast<const glm::vec2&>(rhs);
    return glm::dot(l, r);
}


/**
 * Compute the dot product between two vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return The dot product of 'lhs' and 'rhs'.
 */
THE_TRY_FORCE_INLINE float dot(const vector<glm::vec3>& lhs, const vector<glm::vec3>& rhs) {
    auto& l = static_cast<const glm::vec3&>(lhs);
    auto& r = static_cast<const glm::vec3&>(rhs);
    return glm::dot(l, r);
}


/**
 * Compute the dot product between two vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return The dot product of 'lhs' and 'rhs'.
 */
THE_TRY_FORCE_INLINE float dot(const vector<glm::vec4>& lhs, const vector<glm::vec4>& rhs) {
    auto& l = static_cast<const glm::vec4&>(lhs);
    auto& r = static_cast<const glm::vec4&>(rhs);
    return glm::dot(l, r);
}
#endif /* WITH_THE_GLM */

/**
 * Compute the length (Euclidean norm) of 'vec'.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam D The dimension of the vector.
 * @tparam T The traits of the vector.
 *
 * @param vec A vector.
 *
 * @return The length of 'vec'.
 */
template <class V, size_t D, class T> THE_TRY_FORCE_INLINE typename T::value_type length(const vector<V, D, T>& vec) {
    return sqrt(square_length(vec));
}


#ifdef WITH_THE_GLM
/**
 * Compute the length (Euclidean norm) of 'vec'.
 *
 * @param vec A vector.
 *
 * @return The length of 'vec'.
 */
THE_TRY_FORCE_INLINE float length(const vector<glm::vec2>& vec) {
    auto& v = static_cast<const glm::vec2&>(vec);
    return glm::length(v);
}

/**
 * Compute the length (Euclidean norm) of 'vec'.
 *
 * @param vec A vector.
 *
 * @return The length of 'vec'.
 */
THE_TRY_FORCE_INLINE float length(const vector<glm::vec3>& vec) {
    auto& v = static_cast<const glm::vec3&>(vec);
    return glm::length(v);
}

/**
 * Compute the length (Euclidean norm) of 'vec'.
 *
 * @param vec A vector.
 *
 * @return The length of 'vec'.
 */
THE_TRY_FORCE_INLINE float length(const vector<glm::vec4>& vec) {
    auto& v = static_cast<const glm::vec4&>(vec);
    return glm::length(v);
}
#endif /* WITH_THE_GLM */

/**
 * Normalise 'vec'.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam D The dimension of the vector.
 * @tparam T The traits of the vector.
 *
 * @param vec The vector to be normalised.
 *
 * @return A normalised version of 'vec'.
 */
template <class V, size_t D, class T> vector<V, D, T> normalise(const vector<V, D, T>& vec);

#ifdef WITH_THE_GLM
/**
 * Normalise 'vec'.
 *
 * @param vec The vector to be normalised.
 *
 * @return A normalised version of 'vec'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec2> normalise(const vector<glm::vec2>& vec) {
    return glm::normalize(static_cast<glm::vec2>(vec));
}

/**
 * Normalise 'vec'.
 *
 * @param vec The vector to be normalised.
 *
 * @return A normalised version of 'vec'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec3> normalise(const vector<glm::vec3>& vec) {
    return glm::normalize(static_cast<glm::vec3>(vec));
}

/**
 * Normalise 'vec'.
 *
 * @param vec The vector to be normalised.
 *
 * @return A normalised version of 'vec'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec4> normalise(const vector<glm::vec4>& vec) {
    return glm::normalize(static_cast<glm::vec4>(vec));
}
#endif /* WITH_THE_GLM */

/**
 * Ensure that 'vec1' and 'vec1' are orthonormal.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam T The traits of the vector.
 *
 * @param vec1 The first vector of the system.
 * @param vec2 The second vector of the system.
 */
template <class V, class T> inline void orthonormalise(vector<V, 3, T>& vec1, vector<V, 3, T>& vec2) {
    auto v1 = normalise(vec1);
    auto p = dot(vec1, vec2) * v1;
    vec2 = vec2 - p;
    vec2 = normalise(vec2);
}


/**
 * Ensure that 'vec1' and 'vec1' are orthonormal.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam T The traits of the vector.
 *
 * @param vec1 The first vector of the system.
 * @param vec2 The second vector of the system.
 */
template <class V, class T> inline void orthonormalise(vector<V, 4, T>& vec1, vector<V, 4, T>& vec2) {
    auto v1 = normalise(vec1);
    auto p = dot(vec1, vec2) * v1;
    vec2 = vec2 - p;
    vec2 = normalise(vec2);
}

#ifdef WITH_THE_GLM
/**
 * Ensure that 'vec1' and 'vec1' are orthonormal.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam T The traits of the vector.
 *
 * @param vec1 The first vector of the system.
 * @param vec2 The second vector of the system.
 */
inline void orthonormalise(vector<glm::vec4>& vec1, vector<glm::vec4>& vec2) {
    vec1 = glm::normalize(static_cast<glm::vec4>(vec1));
    auto p = glm::dot(static_cast<glm::vec4>(vec1), static_cast<glm::vec4>(vec2)) * static_cast<glm::vec4>(vec1);
    vec2 = static_cast<glm::vec4>(vec2) - p;
    vec2 = glm::normalize(static_cast<glm::vec4>(vec2));
}
#endif /* WITH_THE_GLM */

/**
 * Normalise homogenous coordinates by performing a perspective divide.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam T The traits of the vector.
 *
 * @param vec The vector to perform the perspective divide on.
 *
 * @retun The nomalised version of 'vec'.
 */
template <class V, class T> inline vector<V, 4, T> perspective_divide(const vector<V, 4, T>& vec) {
    auto w = vec.w();
    return vector<V, 4, T>(vec[0] / w, vec[1] / w, vec[2] / w, w / w);
}

/**
 * Clears all components in a vector.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam D The dimension of the vector.
 * @tparam T The traits of the vector.
 *
 * @param vec The vector to modify.
 *
 * @return 'vec'.
 */
template <class V, size_t D, class T> vector<V, D, T>& set_empty(vector<V, D, T>& vec);


/**
 * Compute the square length of 'vec'.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam D The dimension of the vector.
 * @tparam T The traits of the vector.
 *
 * @param vec A vector.
 *
 * @return The squared length of 'vec'.
 */
template <class V, size_t D, class T>
THE_TRY_FORCE_INLINE typename T::value_type square_length(const vector<V, D, T>& vec) {
    return dot(vec, vec);
}


/**
 * Scale a vector.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam D The dimension of the vector.
 * @tparam T The traits of the vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', scaled by 'rhs'.
 */
template <class V, size_t D, class T>
vector<V, D, T>& operator*=(vector<V, D, T>& lhs, const typename vector<V, D, T>::value_type rhs);


#ifdef WITH_THE_GLM
/**
 * Scale a vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', scaled by 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec2>& operator*=(vector<glm::vec2>& lhs, const vector<glm::vec2>::value_type rhs) {
    lhs = static_cast<glm::vec2>(lhs) * rhs;
    return lhs;
}


/**
 * Scale a vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', scaled by 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec3>& operator*=(vector<glm::vec3>& lhs, const vector<glm::vec3>::value_type rhs) {
    lhs = static_cast<glm::vec3>(lhs) * rhs;
    return lhs;
}


/**
 * Scale a vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', scaled by 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec4>& operator*=(vector<glm::vec4>& lhs, const vector<glm::vec4>::value_type rhs) {
    lhs = static_cast<glm::vec4>(lhs) * rhs;
    return lhs;
}
#endif /* WITH_THE_GLM */

/**
 * Scale a vector by division.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam D The dimension of the vector.
 * @tparam T The traits of the vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', divided by 'rhs'.
 */
template <class V, size_t D, class T>
vector<V, D, T>& operator/=(vector<V, D, T>& lhs, const typename vector<V, D, T>::value_type rhs);

#ifdef WITH_THE_GLM
/**
 * Scale a vector by division.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', divided by 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec2>& operator/=(vector<glm::vec2>& lhs, const vector<glm::vec2>::value_type rhs) {
    lhs *= (1.0f / rhs);
    return lhs;
}


/**
 * Scale a vector by division.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', divided by 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec3>& operator/=(vector<glm::vec3>& lhs, const vector<glm::vec3>::value_type rhs) {
    lhs *= (1.0f / rhs);
    return lhs;
}


/**
 * Scale a vector by division.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', divided by 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec4>& operator/=(vector<glm::vec4>& lhs, const vector<glm::vec4>::value_type rhs) {
    lhs *= (1.0f / rhs);
    return lhs;
}
#endif /* WITH_THE_GLM */

/**
 * Compute a scaled vector.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam D The dimension of the vector.
 * @tparam T The traits of the vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
template <class V, size_t D, class T, class S>
THE_TRY_FORCE_INLINE vector<V, D, T> operator*(
    const typename vector<V, D, T>::value_type lhs, const vector<V, D, T>& rhs) {
    auto retval = rhs;
    retval *= lhs;
    return retval;
}

#ifdef WITH_THE_GLM
/**
 * Compute a scaled vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec2> operator*(
    const vector<glm::vec2>::value_type lhs, const vector<glm::vec2>& rhs) {
    return lhs * static_cast<glm::vec2>(rhs);
}


/**
 * Compute a scaled vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec3> operator*(
    const vector<glm::vec3>::value_type lhs, const vector<glm::vec3>& rhs) {
    return lhs * static_cast<glm::vec3>(rhs);
}


/**
 * Compute a scaled vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec4> operator*(
    const vector<glm::vec4>::value_type lhs, const vector<glm::vec4>& rhs) {
    return lhs * static_cast<glm::vec4>(rhs);
}
#endif /* WITH_THE_GLM */

/**
 * Compute a scaled vector.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam D The dimension of the vector.
 * @tparam T The traits of the vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
template <class V, size_t D, class T>
THE_TRY_FORCE_INLINE vector<V, D, T> operator*(
    const vector<V, D, T>& lhs, const typename vector<V, D, T>::value_type rhs) {
    auto retval = lhs;
    retval *= rhs;
    return retval;
}


#ifdef WITH_THE_GLM
/**
 * Compute a scaled vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec2> operator*(
    const vector<glm::vec2>& lhs, const vector<glm::vec2>::value_type rhs) {
    return static_cast<glm::vec2>(lhs) * rhs;
}


/**
 * Compute a scaled vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec3> operator*(
    const vector<glm::vec3>& lhs, const vector<glm::vec3>::value_type rhs) {
    return static_cast<glm::vec3>(lhs) * rhs;
}


/**
 * Compute a scaled vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec4> operator*(
    const vector<glm::vec4>& lhs, const vector<glm::vec4>::value_type rhs) {
    return static_cast<glm::vec4>(lhs) * rhs;
}
#endif /* WITH_THE_GLM */

/**
 * Compute a scaled vector.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam D The dimension of the vector.
 * @tparam T The traits of the vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' / 'rhs'.
 */
template <class V, size_t D, class T, class S>
THE_TRY_FORCE_INLINE vector<V, D, T> operator/(
    const vector<V, D, T>& lhs, const typename vector<V, D, T>::value_type rhs) {
    auto retval = lhs;
    retval /= rhs;
    return retval;
}


#ifdef WITH_THE_GLM
/**
 * Compute a scaled vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec2> operator/(
    const vector<glm::vec2>& lhs, const vector<glm::vec2>::value_type rhs) {
    auto retval = lhs;
    retval /= rhs;
    return retval;
}


/**
 * Compute a scaled vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec3> operator/(
    const vector<glm::vec3>& lhs, const vector<glm::vec3>::value_type rhs) {
    auto retval = lhs;
    retval /= rhs;
    return retval;
}


/**
 * Compute a scaled vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
THE_TRY_FORCE_INLINE vector<glm::vec4> operator/(
    const vector<glm::vec4>& lhs, const vector<glm::vec4>::value_type rhs) {
    auto retval = lhs;
    retval /= rhs;
    return retval;
}
#endif /* WITH_THE_GLM */

/**
 * Add two vectors.
 *
 * @tparam V The scalar value or storage type of the vectors.
 * @tparam D The dimension of the vectors.
 * @tparam T The traits of the vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' + 'rhs'.
 */
template <class V, size_t D, class T>
THE_TRY_FORCE_INLINE vector<V, D, T> operator+(const vector<V, D, T>& lhs, const vector<V, D, T>& rhs) {
    auto retval = lhs;
    retval += rhs;
    return retval;
}


/**
 * Add two vectors.
 *
 * @tparam V1 The scalar value or storage type of 'lhs'.
 * @tparam T1 The traits of 'lhs'.
 * @tparam V2 The scalar value or storage type of 'rhs'.
 * @tparam T2 The traits of 'rhs'.
 * @tparam D  The dimension of the vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', which 'rhs' has been added to.
 */
template <class V1, class T1, class V2, class T2, size_t D>
vector<V1, D, T1> operator+=(vector<V1, D, T1>& lhs, const vector<V2, D, T2>& rhs);


#ifdef WITH_THE_GLM
/**
 * Add two vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', which 'rhs' has been added to.
 */
THE_TRY_FORCE_INLINE vector<glm::vec2>& operator+=(vector<glm::vec2>& lhs, const vector<glm::vec2>& rhs) {
    lhs = static_cast<glm::vec2>(lhs) + static_cast<glm::vec2>(rhs);
    return lhs;
}


/**
 * Add two vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', which 'rhs' has been added to.
 */
THE_TRY_FORCE_INLINE vector<glm::vec3>& operator+=(vector<glm::vec3>& lhs, const vector<glm::vec3>& rhs) {
    lhs = static_cast<glm::vec3>(lhs) + static_cast<glm::vec3>(rhs);
    return lhs;
}


/**
 * Add two vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', which 'rhs' has been added to.
 */
THE_TRY_FORCE_INLINE vector<glm::vec4>& operator+=(vector<glm::vec4>& lhs, const vector<glm::vec4>& rhs) {
    lhs = static_cast<glm::vec4>(lhs) + static_cast<glm::vec4>(rhs);
    return lhs;
}
#endif /* WITH_THE_GLM */

/**
 * Subtract two vectors.
 *
 * @tparam V The scalar value or storage type of the vectors.
 * @tparam D The dimension of the vectors.
 * @tparam T The traits of the vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' - 'rhs'.
 */
template <class V, size_t D, class T>
THE_TRY_FORCE_INLINE vector<V, D, T> operator-(const vector<V, D, T>& lhs, const vector<V, D, T>& rhs) {
    auto retval = lhs;
    retval -= rhs;
    return retval;
}


/**
 * Subtract two vectors.
 *
 * @tparam V1 The scalar value or storage type of 'lhs'.
 * @tparam T1 The traits of 'lhs'.
 * @tparam V2 The scalar value or storage type of 'rhs'.
 * @tparam T2 The traits of 'rhs'.
 * @tparam D  The dimension of the vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', which 'rhs' has been subtracted from.
 */
template <class V1, class T1, class V2, class T2, size_t D>
vector<V1, D, T1> operator-=(vector<V1, D, T1>& lhs, const vector<V2, D, T2>& rhs);

#ifdef WITH_THE_GLM
/**
 * Subtract two vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', which 'rhs' has been subtracted from.
 */
THE_TRY_FORCE_INLINE vector<glm::vec2>& operator-=(vector<glm::vec2>& lhs, const vector<glm::vec2>& rhs) {
    lhs = static_cast<glm::vec2>(lhs) - static_cast<glm::vec2>(rhs);
    return lhs;
}


/**
 * Subtract two vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', which 'rhs' has been subtracted from.
 */
THE_TRY_FORCE_INLINE vector<glm::vec3>& operator-=(vector<glm::vec3>& lhs, const vector<glm::vec3>& rhs) {
    lhs = static_cast<glm::vec3>(lhs) - static_cast<glm::vec3>(rhs);
    return lhs;
}


/**
 * Subtract two vectors.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', which 'rhs' has been subtracted from.
 */
THE_TRY_FORCE_INLINE vector<glm::vec4>& operator-=(vector<glm::vec4>& lhs, const vector<glm::vec4>& rhs) {
    lhs = static_cast<glm::vec4>(lhs) - static_cast<glm::vec4>(rhs);
    return lhs;
}
#endif /* WITH_THE_GLM */

/**
 * Scale a vector by -1.
 *
 * @tparam V The scalar value or storage type of the vector.
 * @tparam D The dimension of the vector.
 * @tparam T The traits of the vector.
 *
 * @param vec The vector to be scaled.
 *
 * @return -'vec'.
 */
template <class V, size_t D, class T> THE_TRY_FORCE_INLINE vector<V, D, T> operator-(const vector<V, D, T>& vec) {
    auto retval = vec;
    retval *= -1;
    return retval;
}

} /* end namespace math */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#include "mmcore/thecam/math/vector.inl"

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_MATH_VECTOR_H_INCLUDED */
