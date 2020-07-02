/*
 * thecam/math/quaternion.h
 *
 * Copyright (C) 2016 - 2017 TheLib Team (http://www.thelib.org/license)
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

#ifndef THE_MATH_QUATERNION_H_INCLUDED
#define THE_MATH_QUATERNION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include <array>

#ifdef WITH_THE_GLM
#    include <glm/glm.hpp>
#    include <glm/gtc/quaternion.hpp>
#endif /* WITH_THE_GLM */

#include "mmcore/thecam/utility/assert.h"
#include "mmcore/thecam/utility/equatable.h"
#include "mmcore/thecam/utility/force_inline.h"

#include "mmcore/thecam/math/functions.h"
#include "mmcore/thecam/math/mathtypes.h"
#include "mmcore/thecam/math/vector.h"
#include "mmcore/thecam/math/vectorial_traits_base.h"


namespace megamol {
namespace core {
namespace thecam {
namespace math {


/**
 * Index values to access quaternion components.
 */
enum class quaternion_component {
    x = 0, //!< The index of the x-component of the vector part.
    i = 0, //!< The index of the x-component of the vector part.
    y = 1, //!< The index of the y-component of the vector part.
    j = 1, //!< The index of the y-component of the vector part.
    z = 2, //!< The index of the z-component of the vector part.
    k = 2, //!< The index of the z-component of the vector part.
    w = 3, //!< The index of the real component.
    r = 3  //!< The index of the real component.
};


/**
 * The default type traits for a quaternion that is being stored as flat
 * array.
 *
 * There are full template specialisations which allow reinterpreting T as
 * the actual quaternion type of DirectX math and GLM, for instance.
 *
 * @tparam T The scalar value type used in the quaternion or the native
 *           storage type in case of specialisations for existing libraries.
 */
template <class T> struct quaternion_traits : public detail::vectorial_traits_base<T, 4> {

    /** The base traits type. */
    typedef detail::vectorial_traits_base<T, 4> base;

    /** The allocator for heap allocations of the quaternions. */
    template <class C>
    using allocator_type = typename detail::template vectorial_traits_base<T, 4>::template allocator_type<C>;

    /** The native type used to store the quaternion. */
    typedef typename base::native_type native_type;

    /** The type to specify array dimensions and indices. */
    typedef typename base::size_type size_type;

    /** The scalar type used in the quaternion. */
    typedef typename base::value_type value_type;
};


#ifdef WITH_THE_GLM
/**
 * Specialisation of the quaternion_traits for GLM float quaternions.
 */
template <> struct quaternion_traits<glm::quat> {

    /** The native storage type. */
    typedef glm::quat native_type;

    /** The type to specify array dimensions and indices. */
    typedef int size_type;

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
        value_type v[] = {values...};
        // Note: glm uses a different parameter order, namely { w, x, y, z }.
        dst = native_type(v[3], v[0], v[1], v[2]);
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
        THE_ASSERT(static_cast<size_type>(i) >= 0);
        THE_ASSERT(static_cast<size_type>(i) < 4);
        return data[static_cast<size_type>(i)];
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
        THE_ASSERT(static_cast<size_type>(i) >= 0);
        THE_ASSERT(static_cast<size_type>(i) < 4);
        return data[static_cast<size_type>(i)];
    }

    /**
     * Copy 'src' to 'dst'.
     *
     * @param dst The native storage of the destination.
     * @param src The native storage of the source.
     */
    static THE_TRY_FORCE_INLINE void copy(native_type& dst, const native_type& src) { dst = src; }

    /**
     * Test for equality of two native quaternions.
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
 * Forward declaration of quaternion
 */
template <class V, class T> class quaternion;

/**
 * Forward declaration
 */
template <class V, class T> inline quaternion<V, T>& set_identity(quaternion<V, T>& quat);

/**
 * Implementation of a quaternion.
 *
 * The quaternion can be instantiated for a scalar value type or for the
 * the native storage type. The interpretation of the template parameter 'V'
 * depends on the type traits 'T'.
 *
 * @tparam V The scalar value type used in the quaternion or the native
 *           storage type in case of specialisations for existing libraries.
 * @tparam T The traits type that interprets 'V' and provides the scalar
 *           type, the storage type and basic operations.
 */
template <class V, class T = quaternion_traits<V>>
class quaternion : public megamol::core::thecam::utility::equatable<quaternion<V, T>> {

public:
    /** The type of the storage class actually holding the data. */
    typedef typename T::native_type native_type;

    /** The type to specify the dimensions. */
    typedef typename T::size_type size_type;

    /** The traits that provide manipulators for native_type objects. */
    typedef T traits_type;

    /** The type of a scalar. */
    typedef typename T::value_type value_type;

    /**
     * Factory method for initialising an empty quaternion.
     *
     * @return An empty quaternion.
     */
    static inline quaternion create_empty(void) { return quaternion::make_empty(); }

    /**
     * Factory method for initialising an identity quaternion.
     *
     * @return An identity quaternion.
     */
    static inline quaternion create_identity(void) { return quaternion::make_identity(); }

    /**
     * Factory method for initialising an empty quaternion.
     *
     * @return An empty quaternion.
     */
    static inline quaternion make_empty(void) {
        static const quaternion retval;
        return retval;
    }

    /**
     * Factory method for initialising an identity quaternion.
     *
     * @return An identity quaternion.
     */
    static inline quaternion make_identity(void) {
        quaternion retval(thecam::utility::do_not_initialise);
        thecam::math::set_identity(retval);
        return retval;
    }

    /**
     * Factory method for creating a rotation quaternion from an angle and
     * an axis.
     *
     * @tparam V The type of the vector specifying the axis. For the
     *           type+components vector types, this must be a 3-component or
     *           a 4-component vector; for DirectX math quaternions, the
     *           corresponding SSE2-compatible vector type.
     *
     * @param angle The angle to rotate (in radians). Depending on the
     *              coordinate system used, the direction of a positive
     *              rotation turns in the directions your fingers curl when
     *              the left/right thumb points along the rotation axis.
     * @param axis  The axis to rotate around. Note that the vector does not
     *              need to be normalised.
     *
     * @return A quaternion representing the specified rotation.
     */
    template <class Y> static inline quaternion from_angle_axis(const value_type angle, const Y& axis) {
        quaternion retval(thecam::utility::do_not_initialise);
        set_from_angle_axis(retval, angle, axis);
        return retval;
    }

    /**
     * Factory method for creating a rotation between two vectors.
     *
     * @tparam V The type of the vectos. For the type+components vector
     *           types, this must be a 3-component or a 4-component vector;
     *           for DirectX math quaternions, the corresponding
     *           SSE2-compatible vector type.
     *
     * @param u The original vector.
     * @param v The target vector.
     *
     * @return 'quat'.
     */
    template <class D> static inline quaternion from_vectors(const D& u, const D& v) {
        quaternion retval(thecam::utility::do_not_initialise);
        set_from_vectors(retval, u, v);
        return retval;
    }

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
     * Initialises a new quaternion, all zero.
     *
     * @param x The first component of the vector part, which defaults to 0.
     * @param y The second component of the vector part, which defaults
     *          to 0.
     * @param z The third component of the vector part, which defaults to 0.
     * @param w The real part, which defaults to 0.
     */
    inline quaternion(const value_type x = static_cast<value_type>(0), const value_type y = static_cast<value_type>(0),
        const value_type z = static_cast<value_type>(0), const value_type w = static_cast<value_type>(0)) {
        this->assign(x, y, z, w);
    }

    /**
     * Create a new quaternion, but do not initialise the components.
     */
    inline quaternion(const utility::do_not_initialise_t) {}

    /**
     * Initialises a new quaternion from an initialiser list { x, y, z, w }.
     *
     * @param il The initialiser list. If not all components are specified,
     *           the rest will be zero.
     */
    quaternion(std::initializer_list<value_type> il);

    /**
     * Clone 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    THE_TRY_FORCE_INLINE quaternion(const quaternion& rhs) { traits_type::copy(this->data, rhs.data); }

    /**
     * Convert 'rhs'.
     *
     * @tparam Vp The value or native storage type of 'rhs'.
     * @tparam Tp The traits type of 'rhs'.
     *
     * @param rhs The object to be converted.
     */
    template <class Vp, class Tp> THE_TRY_FORCE_INLINE quaternion(const quaternion<Vp, Tp>& rhs) { *this = rhs; }

    /**
     * Initialises the quaternion from its native representation.
     *
     * This constructor enables implicit casts from native_type.
     *
     * @param data The initial data.
     */
    THE_TRY_FORCE_INLINE quaternion(const native_type& data) { traits_type::copy(this->data, data); }

    /**
     * Assign new values to all components of the quaternion.
     *
     * @param x The first component of the vector part.
     * @param y The second component of the vector part.
     * @param z The third component of the vector part.
     * @param w The real part.
     */
    THE_TRY_FORCE_INLINE void assign(const value_type x, const value_type y, const value_type z, const value_type w) {
        traits_type::assign(this->data, x, y, z, w);
    }

    /**
     * Answer whether the quaternion is empty (all components are exactly
     * zero).
     *
     * Depending on the underlying storage class, this implementation might
     * be vectorised and therefore more efficient than the epsilon
     * comparison.
     *
     * @return true if the quaternion is empty, false otherwise.
     */
    THE_TRY_FORCE_INLINE bool empty(void) const {
        static const quaternion EMPTY(static_cast<value_type>(0), static_cast<value_type>(0),
            static_cast<value_type>(0), static_cast<value_type>(0));
        return this->equals(EMPTY);
    }

    /**
     * Answer whether the quaternion is empty (has no non-null component).
     *
     * @param epsilon An epsilon value used for comparison. This defaults to
     *                the::math::epsilon<value_type>::value.
     *
     * @return true if the quaternion is empty, false otherwise.
     */
    inline bool empty(const value_type epsilon) const {
        return (is_equal(this->x(), static_cast<value_type>(0), epsilon) &&
                is_equal(this->y(), static_cast<value_type>(0), epsilon) &&
                is_equal(this->z(), static_cast<value_type>(0), epsilon) &&
                is_equal(this->w(), static_cast<value_type>(0), epsilon));
    }

    /**
     * Test for equality.
     *
     * @param rhs The object to be compared.
     *
     * @return true if this object and 'rhs' are equal, false otherwise.
     */
    THE_TRY_FORCE_INLINE bool equals(const quaternion& rhs) const { return traits_type::equals(this->data, rhs.data); }

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
    template <class Vp, class Tp> inline bool equals(const quaternion<Vp, Tp>& rhs, const value_type epsilon) const {
        return (is_equal(this->x(), static_cast<value_type>(rhs.x()), epsilon) &&
                is_equal(this->y(), static_cast<value_type>(rhs.y()), epsilon) &&
                is_equal(this->z(), static_cast<value_type>(rhs.z()), epsilon) &&
                is_equal(this->w(), static_cast<value_type>(rhs.w()), epsilon));
    }

    /**
     * * Answer whether the quaternion is (exactly) an identity quaternion.
     *
     * Depending on the underlying storage class, this implementation might
     * be vectorised and therefore more efficient than the epsilon
     * comparison.
     *
     * @return true if the quaternion is an identity quaternion,
     *         false otherwise.
     */
    THE_TRY_FORCE_INLINE bool identity(void) const {
        static const quaternion IDENTITY(static_cast<value_type>(0), static_cast<value_type>(0),
            static_cast<value_type>(0), static_cast<value_type>(1));
        return this->equals(IDENTITY);
    }

    /**
     * Answer whether the quaternion is an identity quaternion.
     *
     * @param epsilon An epsilon value used for comparison. This defaults to
     *                megamol::core::thecam::math::epsilon<value_type>::value.
     *
     * @return true if the quaternion is an identity quaternion,
     *         false otherwise.
     */
    inline bool identity(const value_type epsilon) const {
        return (is_equal(this->x(), static_cast<value_type>(0), epsilon) &&
                is_equal(this->y(), static_cast<value_type>(0), epsilon) &&
                is_equal(this->z(), static_cast<value_type>(0), epsilon) &&
                is_equal(this->w(), static_cast<value_type>(1), epsilon));
    }

    /**
     * Answer the number of components in the quaternion.
     *
     * @return 4.
     */
    inline size_type size(void) const { return 4; }

    /**
     * Gets the x-component of the vector part.
     *
     * @return The x-component of the vector part.
     */
    THE_TRY_FORCE_INLINE value_type x(void) const {
        return traits_type::at(this->data, static_cast<size_type>(quaternion_component::x));
    }

    /**
     * Gets a reference to the x-component of the vector part.
     *
     * @return A reference to the x-component of the vector part.
     */
    THE_TRY_FORCE_INLINE value_type& x(void) {
        return traits_type::at(this->data, static_cast<size_type>(quaternion_component::x));
    }

    /**
     * Gets the y-component of the vector part.
     *
     * @return The y-component of the vector part.
     */
    THE_TRY_FORCE_INLINE value_type y(void) const {
        return traits_type::at(this->data, static_cast<size_type>(quaternion_component::y));
    }

    /**
     * Gets a reference to the y-component of the vector part.
     *
     * @return A reference to the y-component of the vector part.
     */
    THE_TRY_FORCE_INLINE value_type& y(void) {
        return traits_type::at(this->data, static_cast<size_type>(quaternion_component::y));
    }

    /**
     * Gets the z-component of the vector part.
     *
     * @return The z-component of the vector part.
     */
    THE_TRY_FORCE_INLINE value_type z(void) const {
        return traits_type::at(this->data, static_cast<size_type>(quaternion_component::z));
    }

    /**
     * Gets a reference to the z-component of the vector part.
     *
     * @return A reference to the z-component of the vector part.
     */
    THE_TRY_FORCE_INLINE value_type& z(void) {
        return traits_type::at(this->data, static_cast<size_type>(quaternion_component::z));
    }

    /**
     * Gets the real part.
     *
     * @return The real part.
     */
    THE_TRY_FORCE_INLINE value_type w(void) const {
        return traits_type::at(this->data, static_cast<size_type>(quaternion_component::w));
    }

    /**
     * Gets a reference to the real part.
     *
     * @return A reference to the real part.
     */
    THE_TRY_FORCE_INLINE value_type& w(void) {
        return traits_type::at(this->data, static_cast<size_type>(quaternion_component::w));
    }

    /**
     * Assignment.
     *
     * @param rhs The right-hand side operand.
     *
     * @return *this.
     */
    quaternion& operator=(const quaternion& rhs);

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
    template <class Vp, class Tp> inline quaternion& operator=(const quaternion<Vp, Tp>& rhs) {
        this->assign(static_cast<value_type>(rhs.x()), static_cast<value_type>(rhs.y()),
            static_cast<value_type>(rhs.z()), static_cast<value_type>(rhs.w()));
        return *this;
    }

    /**
     * Answer the given component.
     *
     * @param c The component to retrieve. Use the quaternion_component
     *          enumeration for accessing only valid components as the
     *          operator might not perform bounds checks.
     *
     * @return The 'c'th component.
     */
    THE_TRY_FORCE_INLINE value_type operator[](const quaternion_component c) const {
        return traits_type::at(this->data, static_cast<size_type>(c));
    }
    /**
     * Answer the given component.
     *
     * @param c The component to retrieve. Use the quaternion_component
     *          enumeration for accessing only valid components as the
     *          operator might not perform bounds checks.
     *
     * @return A reference to the 'c'th component.
     */
    THE_TRY_FORCE_INLINE value_type& operator[](const quaternion_component c) {
        return traits_type::at(this->data, static_cast<size_type>(c));
    }

    /**
     * Conversion to native_type.
     *
     * @return The native representation of the quaternion.
     */
    THE_TRY_FORCE_INLINE operator native_type&(void) { return this->data; }

    /**
     * Conversion to native_type.
     *
     * @return The native representation of the quaternion.
     */
    THE_TRY_FORCE_INLINE operator const native_type&(void)const { return this->data; }

private:
    /** Actually stores the components of the quaternion. */
    native_type data;
};

/**
 * Conjugate a quaternion.
 *
 * @param quat The quaternion to be conjugated.
 *
 * @return The conjugated quaternion.
 */
template <class V, class T> inline quaternion<V, T> conjugate(const quaternion<V, T>& quat) {
    typedef typename T::value_type value_type;
    quaternion<V, T> retval(quat);
    retval.x() *= static_cast<value_type>(-1);
    retval.y() *= static_cast<value_type>(-1);
    retval.z() *= static_cast<value_type>(-1);
    return std::move(retval);
}


#ifdef WITH_THE_GLM
/**
 * Conjugate a quaternion.
 *
 * @param quat The quaternion to be conjugated.
 *
 * @return The conjugated quaternion.
 */
THE_TRY_FORCE_INLINE quaternion<glm::quat> conjugate(const quaternion<glm::quat>& quat) {
    return glm::conjugate(static_cast<glm::quat>(quat));
}
#endif /* WITH_THE_GLM */


/**
 * Invert a quaternion.
 *
 * @tparam V The scalar or native storage type of the quaternion.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param quat The quaternion to be inverted.
 *
 * @return The inverse quaternion.
 */
template <class V, class T> quaternion<V, T> invert(const quaternion<V, T>& quat);


#ifdef WITH_THE_GLM
/**
 * Invert a quaternion.
 *
 * @param quat The quaternion to be inverted.
 *
 * @return The inverse quaternion.
 */
THE_TRY_FORCE_INLINE quaternion<glm::quat> invert(const quaternion<glm::quat>& quat) {
    return glm::inverse(static_cast<glm::quat>(quat));
}
#endif /* WITH_THE_GLM */


/**
 * Compute the norm of a quaternion.
 *
 * @tparam V The scalar or native storage type of the quaternion.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param quat The quaternion to compute the norm for.
 *
 * @return The norm of the quaternion.
 */
template <class V, class T> inline typename T::value_type norm(const quaternion<V, T>& quat) {
    return sqrt(square_norm(quat));
}

#ifdef WITH_THE_GLM
/**
 * Compute the norm of a quaternion.
 *
 * @param quat The quaternion to compute the norm for.
 *
 * @return The norm of the quaternion.
 */
THE_TRY_FORCE_INLINE float norm(const quaternion<glm::quat>& quat) { return glm::length(static_cast<glm::quat>(quat)); }
#endif /* WITH_THE_GLM */


/**
 * Compute a normalised version of a quaternion.
 *
 * @tparam V The scalar or native storage type of the quaternion.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param quat The quaternion to be normalised.
 *
 * @return The normalised quaternion.
 */
template <class V, class T> quaternion<V, T> normalise(const quaternion<V, T>& quat);


#ifdef WITH_THE_GLM
/**
 * Compute a normalised version of a quaternion.
 *
 * @param quat The quaternion to be normalised.
 *
 * @return The normalised quaternion.
 */
THE_TRY_FORCE_INLINE quaternion<glm::quat> normalise(const quaternion<glm::quat>& quat) {
    return glm::normalize(static_cast<glm::quat>(quat));
}
#endif /* WITH_THE_GLM */


/**
 * Rotate the vector 'vec' with the given quaternion 'quat'.
 *
 * @tparam V The scalar or native storage type of the quaternion and vector
 * @tparam TQ The traits type interpreting 'V'.
 *
 * @param vec  The vector to be rotated.
 * @param quat The quaternion describing the rotation.
 *
 * @return The rotated vector, having a 0 z-component.
 */
// TODO: this will not work for GLM!!!
template <class V, class TQ, class TV>
inline vector<V, 3, TV> rotate(const vector<V, 3, TV>& vec, const quaternion<V, TQ>& quat);


#ifdef WITH_THE_GLM
/**
 * Rotate the vector 'vec' with the given quaternion 'quat'.
 *
 * @param vec  The vector to be rotated.
 * @param quat The quaternion describing the rotation.
 *
 * @return The rotated vector, having a 0 z-component.
 */
inline vector<glm::vec3> rotate(const vector<glm::vec3>& vec, const quaternion<glm::quat>& quat) {
    std::decay<decltype(vec)>::type retval(thecam::utility::do_not_initialise);
    glm::quat h(0.0f, static_cast<glm::vec3>(vec).x, static_cast<glm::vec3>(vec).y, static_cast<glm::vec3>(vec).z);
    h = static_cast<glm::quat>(quat) * h * glm::conjugate(static_cast<glm::quat>(quat));
    retval = glm::vec3(h.x, h.y, h.z);
    return retval;
}
#endif /* WITH_THE_GLM */


#ifdef WITH_THE_GLM
/**
 * Rotate the vector 'vec' with the given quaternion 'quat'.
 *
 * @param vec  The vector to be rotated.
 * @param quat The quaternion describing the rotation.
 *
 */
inline vector<glm::vec4> rotate(const vector<glm::vec4>& vec, const quaternion<glm::quat>& quat) {
    std::decay<decltype(vec)>::type retval(thecam::utility::do_not_initialise);
    glm::quat h(0.0f, static_cast<glm::vec4>(vec).x, static_cast<glm::vec4>(vec).y, static_cast<glm::vec4>(vec).z);
    h = static_cast<glm::quat>(quat) * h * glm::conjugate(static_cast<glm::quat>(quat));
    retval = glm::vec4(h.x, h.y, h.z, 0.0f);
    return retval;
}
#endif


/**
 * Clears all components in a quaternion.
 *
 * @tparam V The scalar or native storage type of the quaternion.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param quat The quaternion to modify.
 *
 * @return 'quat'.
 */
template <class V, class T> inline quaternion<V, T>& set_empty(quaternion<V, T>& quat) {
    typedef typename T::value_type value_type;
    quat.assign(
        static_cast<value_type>(0), static_cast<value_type>(0), static_cast<value_type>(0), static_cast<value_type>(0));
    return quat;
}


/**
 * Update 'quat' such that it represents a rotation by 'angle' radians
 * around 'axis'.
 *
 * @param Q
 * @param V
 * @param A The type to specify the angle.
 *
 * @param quat  The quaternion to be updated.
 * @param angle The angle to rotate (in radians). Depending on the
 *              coordinate system used, the direction of a positive
 *              rotation turns in the directions your fingers curl when
 *              the left/right thumb points along the rotation axis.
 * @param axis  The axis to rotate around. Note that the vector does not
 *              need to be normalised.
 *
 * @return 'quat'.
 */
// TODO: this will not work for GLM!
template <class Q, class V, class A>
quaternion<Q>& set_from_angle_axis(quaternion<Q>& quat, const A angle, const vector<V, 3>& axis);

template <class Q, class V, class A>
inline quaternion<Q>& set_from_angle_axis(quaternion<Q>& quat, const A angle, const vector<V, 4>& axis) {
    vector<V, 3> tmp(axis);
    return set_from_angle_axis(quat, angle, tmp);
}

#ifdef WITH_THE_GLM
/**
 * Update 'quat' such that it represents a rotation by 'angle' radians
 * around 'axis'.
 *
 * @param A The type to specify the angle.
 *
 * @param quat  The quaternion to be updated.
 * @param angle The angle to rotate (in radians).
 * @param angle The angle to rotate (in radians). Depending on the
 *              coordinate system used, the direction of a positive
 *              rotation turns in the directions your fingers curl when
 *              the left/right thumb points along the rotation axis.
 * @param axis  The axis to rotate around. Note that the vector does not
 *              need to be normalised.
 *
 * @return 'quat'.
 */
template <class A>
inline quaternion<glm::quat>& set_from_angle_axis(
    quaternion<glm::quat>& quat, const A angle, const vector<glm::vec4>& axis) {
    glm::vec3 a = glm::vec3(static_cast<glm::vec4>(axis));
    a = glm::normalize(a);
    glm::quat unit = glm::quat(1, 0, 0, 0);
    quat = glm::rotate(static_cast<glm::quat>(unit), angle, a);
    return quat;
}
#endif


/**
 * Update 'quat' such that it represents a rotation from the angle 'u' to
 * the angle 'v'.
 *
 * @param Q The value type of the quaternion.
 * @param V The value type of the vectors.
 *
 * @param quat The target quaternion to update.
 * @param u    The original vector.
 * @param v    The target vector.
 *
 * @return 'quat'.
 */
template <class Q, class V>
quaternion<Q>& set_from_vectors(quaternion<Q>& quat, const vector<V, 3>& u, const vector<V, 3>& v);

/**
 * Update 'quat' such that it represents a rotation from the angle 'u' to
 * the angle 'v'.
 *
 * @param Q The value type of the quaternion.
 * @param V The value type of the vectors.
 *
 * @param quat The target quaternion to update.
 * @param u    The original vector.
 * @param v    The target vector.
 *
 * @return 'quat'.
 */
template <class Q, class V>
quaternion<Q>& set_from_vectors(quaternion<Q>& quat, const vector<V, 4>& u, const vector<V, 4>& v);

#ifdef WITH_THE_GLM
/**
 * Update 'quat' such that it represents a rotation from the angle 'u' to
 * the angle 'v'.
 *
 * @param Q
 * @param V
 *
 * @param quat
 * @param u    The original vector.
 * @param v    The target vector.
 *
 * @return 'quat'.
 */
inline quaternion<glm::quat>& set_from_vectors(
    quaternion<glm::quat>& quat, const vector<glm::vec3>& u, const vector<glm::vec3>& v);
#endif /* WITH_THE_GLM */

#ifdef WITH_THE_GLM
/**
 * Update 'quat' such that it represents a rotation from the angle 'u' to
 * the angle 'v'.
 *
 * @param Q
 * @param V
 *
 * @param quat
 * @param u    The original vector.
 * @param v    The target vector.
 *
 * @return 'quat'.
 */
inline quaternion<glm::quat>& set_from_vectors(
    quaternion<glm::quat>& quat, const vector<glm::vec4>& u, const vector<glm::vec4>& v);
#endif /* WITH_THE_GLM */


/**
 * Makes a quaternion an identity quaternion.
 *
 * @tparam V The scalar or native storage type of the quaternion.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param quat The quaternion to modify.
 *
 * @return 'quat'.
 */
template <class V, class T> inline quaternion<V, T>& set_identity(quaternion<V, T>& quat) {
    typedef typename T::value_type value_type;
    quat.assign(
        static_cast<value_type>(0), static_cast<value_type>(0), static_cast<value_type>(0), static_cast<value_type>(1));
    return quat;
}

#ifdef WITH_THE_GLM
/**
 * Makes a quaternion an identity quaternion.
 *
 * @param quat The quaternion to modify.
 *
 * @return 'quat'.
 */
inline quaternion<glm::quat>& set_identity(quaternion<glm::quat>& quat) {
    quat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    return quat;
}
#endif /* WITH_THE_GLM */


/**
 * Compute the squared norm of the given quaternion.
 *
 * @tparam V The scalar or native storage type of the quaternion.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param quat a quaternion.
 *
 * @return The square norm of 'quat'.
 */
template <class V, class T> inline typename T::value_type square_norm(const quaternion<V, T>& quat) {
    return (sqr(quat.x()) + sqr(quat.y()) + sqr(quat.z()) + sqr(quat.w()));
}


#ifdef WITH_THE_GLM
/**
 * Compute the squared norm of the given quaternion.
 *
 * @param quat a quaternion.
 *
 * @return The square norm of 'quat'.
 */
THE_TRY_FORCE_INLINE float square_norm(const quaternion<glm::quat>& quat) {
    float l = glm::length(static_cast<glm::quat>(quat));
    return l * l;
}
#endif /* WITH_THE_GLM */


/**
 * Multiply two quaternions.
 *
 * @tparam V The scalar or native storage type of the quaternion.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
template <class V, class T> quaternion<V, T> operator*(const quaternion<V, T>& lhs, const quaternion<V, T>& rhs);


#ifdef WITH_THE_GLM
/**
 * Multiply two quaternions.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
THE_TRY_FORCE_INLINE quaternion<glm::quat> operator*(
    const quaternion<glm::quat>& lhs, const quaternion<glm::quat>& rhs) {
    return static_cast<glm::quat>(lhs) * static_cast<glm::quat>(rhs);
}
#endif /* WITH_THE_GLM */

} /* end namespace math */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#include "mmcore/thecam/math/quaternion.inl"

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_MATH_QUATERNION_H_INCLUDED */
