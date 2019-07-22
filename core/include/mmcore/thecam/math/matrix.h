/*
 * thecam\math\matrix.h
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

#ifndef THE_MATH_MATRIX_H_INCLUDED
#define THE_MATH_MATRIX_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include <array>

#ifdef WITH_THE_GLM
#    include <glm/glm.hpp>
#    include <glm/gtc/type_ptr.hpp>
#endif /* WITH_THE_GLM */

#include "mmcore/thecam/utility/equatable.h"
#include "mmcore/thecam/utility/memory.h"
#include "mmcore/thecam/utility/types.h"

#include "mmcore/thecam/math/matrix_indexer.h"
#include "mmcore/thecam/math/vector.h"


namespace megamol {
namespace core {
namespace thecam {
namespace math {

namespace detail {

/**
 * The implicit_matrix_properties enable us to derive the dimension and
 * layout of a matrix wrapping existing matrix classes.
 *
 * The default implementation does not provide any information except for a
 * default memory layout. This layout is row-major, because C/C++ arrays are
 * natively laid out this way and if the user does not request any specific
 * layout, we want to use the well-known C/C++ way.
 *
 * Specialisations must provide the following static fields which answer
 * the properties of the wrapped type:
 *
 * static const size_t columns;
 * static const matrix_layout matrix_layout;
 * static const size_t rows);
 */
template <class T> struct implicit_matrix_properties { static const matrix_layout layout = matrix_layout::row_major; };

#ifdef WITH_THE_GLM
/**
 * Provides implicit template parameters of the glm matrix class.
 */
template <> struct implicit_matrix_properties<glm::mat4> {
    static const size_t columns = 4;
    static const matrix_layout layout = matrix_layout::column_major;
    static const size_t rows = 4;
};
#endif /* WITH_THE_GLM */

} /* end namespace detail */

/**
 * A basic implementation of a traits type for matrices with R rows and C
 * columns.
 *
 * The basic implementation of the matrix uses a flat std::array with memory
 * layout L to store the matrix components.
 *
 * Any specialisation must include the following typedefs:
 *
 * allocator_type: An allocator template that can be used to allocate
 *                 instances on the heap. The reason for providing the
 *                 allocator is that eg DirectX maths instances must be
 *                 16-byte aligned and therefore need a special allocator.
 * indexer_type: The type of the indexer that determines the position of the
 *               elements.
 * native_type: The type to actually store the data.
 * size_type: The type to specify dimensions and indices.
 * value_type: The type of a scalar.
 *
 * Any specialisation must include the following static members:
 *
 * static const bool is_contiguous;
 *
 * Any specialisation must provide the following static methods:
 *
 * static value_type at(const native_type& data, const size_type row,
 *     const size_type column);
 * static value_type& at(native_type& data, const size_type row,
 *     const size_type column);
 * static void copy(native_type& dst, const native_type& src)
 * static bool equals(const native_type& lhs, const native_type& rhs);
 *
 * If the static member 'is_contiguous' has a value of true, the respective
 * specialisation must additionally implement the following two methods:
 *
 * static const value_type *data(const native_type& data);
 * static value_type *data(native_type& data);
 *
 * @tparam V The scalar type (in this case) or the native storage type for
 *           some specialisations.
 * @tparam R The number of rows in the matrix.
 * @tparam C The number of columns in the matrix.
 * @tparam L The memory layout of the matrix.
 */
template <class V, size_t R, size_t C, matrix_layout L> struct matrix_traits {

    /** The allocator for heap allocations of the matrix class. */
    template <class U> using allocator_type = std::allocator<U>;

    /** The type of matrix indexer for locating a single component. */
    typedef matrix_indexer<R, C, L> indexer_type;

    /** The native storage type. */
    typedef std::array<V, R * C> native_type;

    /** The type to specify dimensions and indices. */
    typedef size_t size_type;

    /** The scalar type. */
    typedef V value_type;

    /** Indicates that the storage layout of native_type is contiguous. */
    static const bool is_contiguous = true;

    /**
     * Get the value of the specified component 'row', 'column'.
     *
     * @param data   The native data.
     * @param row    The row of the element to retrieve.
     * @param column The columns of the element to retrieve.
     *
     * @return The value of the 'i'th component.
     */
    static THE_TRY_FORCE_INLINE value_type at(const native_type& data, const size_type row, const size_type column) {
        THE_ASSERT(indexer_type::valid(row, column));
        return data[indexer_type::index(row, column)];
    }

    /**
     * Get the value of the specified component 'row', 'column'.
     *
     * @param data   The native data.
     * @param row    The row of the element to retrieve.
     * @param column The columns of the element to retrieve.
     *
     * @return The value of the 'i'th component.
     */
    static THE_TRY_FORCE_INLINE value_type& at(native_type& data, const size_type row, const size_type column) {
        THE_ASSERT(indexer_type::valid(row, column));
        return data[indexer_type::index(row, column)];
    }

    /**
     * Copy 'src' to 'dst'.
     *
     * @param dst The native storage of the destination.
     * @param src The native storage of the source.
     */
    static THE_TRY_FORCE_INLINE void copy(native_type& dst, const native_type& src) {
        std::copy(src.cbegin(), src.cend(), dst.begin());
    }

    /**
     * Provides access to the first element of the matrix; with the
     * following rows() * columns() elements being laid out according to
     * indexer_type.
     *
     * @param data The native storage to access the raw data of.
     *
     * @return A pointer to the first element.
     */
    static THE_TRY_FORCE_INLINE const value_type* data(const native_type& data) { return data.data(); }

    /**
     * Provides access to the first element of the matrix; with the
     * following rows() * columns() elements being laid out according to
     * indexer_type.
     *
     * @param data The native storage to access the raw data of.
     *
     * @return A pointer to the first element.
     */
    static THE_TRY_FORCE_INLINE value_type* data(native_type& data) { return data.data(); }

    /**
     * Test for equality of two native matrices.
     *
     * @param lhs The left-hand side operand.
     * @param rhs The right-hand side operand.
     *
     * @return true if 'lhs' and 'rhs' are equal, false otherwise.
     */
    static THE_TRY_FORCE_INLINE bool equals(const native_type& lhs, const native_type& rhs) { return (lhs == rhs); }
};

#ifdef WITH_THE_GLM
/**
 * Specialisation of the matrix traits for glm.
 */
template <> struct matrix_traits<glm::mat4x4, 4, 4, matrix_layout::column_major> {

    /** The allocator for heap allocations of the matrix class. */
    template <class C> using allocator_type = std::allocator<C>;

    /** The type of matrix indexer for locating a single component. */
    typedef matrix_indexer<4, 4, matrix_layout::column_major> indexer_type;

    /** The native storage type. */
    typedef glm::mat4 native_type;

    /** The type to specify dimensions and indices. */
    typedef int size_type;

    /** The scalar type. */
    typedef float value_type;

    /** Indicates that the storage layout of native_type is contiguous. */
    static const bool is_contiguous = true;

    /**
     * Get the value of the specified component 'row', 'column'.
     *
     * @param data   The native data.
     * @param row    The row of the element to retrieve.
     * @param column The columns of the element to retrieve.
     *
     * @return The value of the 'i'th component.
     */
    static THE_TRY_FORCE_INLINE value_type at(const native_type& data, const size_type row, const size_type column) {
        THE_ASSERT(indexer_type::valid(row, column));
        return data[row][column];
    }

    /**
     * Get the value of the specified component 'row', 'column'.
     *
     * @param data   The native data.
     * @param row    The row of the element to retrieve.
     * @param column The columns of the element to retrieve.
     *
     * @return The value of the 'i'th component.
     */
    static THE_TRY_FORCE_INLINE value_type& at(native_type& data, const size_type row, const size_type column) {
        THE_ASSERT(indexer_type::valid(row, column));
        return data[row][column];
    }

    /**
     * Copy 'src' to 'dst'.
     *
     * @param dst The native storage of the destination.
     * @param src The native storage of the source.
     */
    static THE_TRY_FORCE_INLINE void copy(native_type& dst, const native_type& src) { dst = src; }

    /**
     * Test for equality of two native matrices.
     *
     * @param lhs The left-hand side operand.
     * @param rhs The right-hand side operand.
     *
     * @return true if 'lhs' and 'rhs' are equal, false otherwise.
     */
    static THE_TRY_FORCE_INLINE bool equals(const native_type& lhs, const native_type& rhs) {
        return (::memcmp(glm::value_ptr(lhs), glm::value_ptr(rhs), sizeof(lhs)) == 0);
    }
};
#endif /* WITH_THE_GLM */


// http://stackoverflow.com/questions/12250026/function-templates-different-specializations-with-type-traits

/**
 * Implements a matrix.
 *
 * The matrix type is intended to be relatively dumb, ie supporting as few
 * operations as possible. Its intended use is mainly transporting data from
 * and to THElib. Therefore, we also provide traits types for commonly used
 * maths libraries.
 *
 * The matrix (and its specialisations) uses the following conventions,
 * which follow the predominant mathematical notations: The accessors are
 * row-major regardless of the underlying layout L. Therefore, m(i, j)
 * returns the jth column in the ith row even if the matrix layout is
 * column-major. Please note that this order also affects the initialiser
 * list, ie the elements in an initialiser list must be ordered row-major.
 * The ordering, however, does not affect the conversion to the underlying
 * native type. The actual storage alignment of this type is as defined by
 * the layout property of the matrix.
 *
 * For matrix-vector multiplication, we provide operators for
 * pre-multiplication of row vectors (Direct3D-style) and
 * post-multiplication of column vectors (OpenGL-style) for both layouts.
 * Please be aware that eg for row-major matrices, the implementation of
 * the former (pre-multiplication) can be more efficient, for column-major
 * matrices, the opposite holds. When using post-multiplication with
 * DirectX math row-major matrices, every transform requires the matrix
 * to be transposed, for example. Therefore, the recommendation is using
 * the appropriate "native" multiplication order when dealing with the
 * repsective graphics API. See
 * http://seanmiddleditch.com/matrices-handedness-pre-and-post-multiplication-row-vs-column-major-and-notations/
 * for more details on the impact of the storage layout.
 *
 * @tparam V The scalar value type used in the matrix or the native
 *           storage type in case of specialisations for existing libraries.
 * @tparam R The number of rows in the matrix.
 * @tparam C The number of columns in the matrix.
 * @tparam L The memory layout of the matrix.
 * @tparam T The traits type that interprets 'V' and provides the scalar
 *           type, the storage type and basic operations. In contrast to the
 *           vector, the matrix traits are also used for determining
 *           implicit dimensions of the matrix.
 */
template <class V, size_t R = detail::implicit_matrix_properties<V>::rows,
    size_t C = detail::implicit_matrix_properties<V>::columns,
    matrix_layout L = detail::implicit_matrix_properties<V>::layout, class T = matrix_traits<V, R, C, L>>
class matrix : public megamol::core::thecam::utility::equatable<matrix<V, R, C, L, T>> {

public:
    /** The indexer that computes addresses of elements. */
    typedef typename T::indexer_type indexer_type;

    /** The native type used to store the matrix. */
    typedef typename T::native_type native_type;

    /** The type to specify the dimension of the matrix. */
    typedef typename T::size_type size_type;

    /** The traits that provide manipulators for native_type objects. */
    typedef T traits_type;

    /** The type of a scalar of the matrix. */
    typedef typename T::value_type value_type;

    /**
     * Factory method for initialising an empty matrix.
     *
     * @return An empty quaternion.
     */
    static inline matrix create_empty(void) { return matrix::make_empty(); }

    /**
     * Factory method for initialising an identity matrix.
     *
     * @return An identity quaternion.
     */
    static inline matrix create_identity(void) { return matrix::make_identity(); }

    /**
     * Factory method for initialising an empty matrix.
     *
     * @return An empty quaternion.
     */
    static inline matrix make_empty(void) {
        static const matrix retval;
        return retval;
    }

    /**
     * Factory method for initialising an identity matrix.
     *
     * @return An identity quaternion.
     */
    static inline matrix make_identity(void) {
        matrix retval(megamol::core::thecam::utility::do_not_initialise);
        set_identity(retval);
        return retval;
    }

    /**
     * Frees heap allocations of the class.
     *
     * @param ptr  A pointer to memory that has been allocated with
     *             matrix::operator new.
     * @param size The size that has been passed to the allocator.
     */
    static void operator delete(void* ptr, const std::size_t size) {
        // Note: operator new/new[]/delete/delete[] work on bytes, not on
        // number of objects!
        static typename traits_type::template allocator_type<byte> alloc;
        alloc.deallocate(static_cast<unsigned char*>(ptr), size);
    }

    /**
     * Frees heap allocations of the class.
     *
     * @param ptr  A pointer to memory that has been allocated with
     *             matrix::operator new[].
     * @param size The size that has been passed to the allocator.
     */
    static void operator delete[](void* ptr, const std::size_t size) {
        // Note: operator new/new[]/delete/delete[] work on bytes, not on
        // number of objects!
        static typename traits_type::template allocator_type<byte> alloc;
        alloc.deallocate(static_cast<unsigned char*>(ptr), size);
    }

    /**
     * Allocates a new instance on the heap.
     *
     * @param size The amount of memory to be allocated (in bytes).
     */
    static void* operator new(const std::size_t size) {
        // Note: operator new/new[]/delete/delete[] work on bytes, not on
        // number of objects!
        static typename traits_type::template allocator_type<byte> alloc;
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
        static typename traits_type::template allocator_type<byte> alloc;
        return alloc.allocate(size);
    }

    /**
     * Initialise all elements with zero.
     */
    inline matrix(void) { set_empty(*this); }

    /**
     * Create a new matrix, but do not initialise the components.
     */
    inline matrix(const utility::do_not_initialise_t) {}

    /**
     * Initialises a new matrix.
     *
     * @param il The initialiser list. If not all components are specified,
     *           the rest will be zero. The order of elements is assumed to
     *           be row-major (like in C/C++ arrays), regardless of the
     *           layout of the matrix.
     */
    matrix(std::initializer_list<value_type> il);

    /**
     * Clone 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    inline matrix(const matrix& rhs) { traits_type::copy(this->data, rhs.data); }

    /**
     * Convert 'rhs'.
     *
     * @tparam Vp The value or native storage type of 'rhs'.
     * @tparam Lp The storage layout of 'rhs'.
     * @tparam Tp The traits type of 'rhs'.
     *
     * @param rhs The object to be converted.
     */
    template <class Vp, matrix_layout Lp, class Tp> inline matrix(const matrix<Vp, R, C, Lp, Tp>& rhs) { *this = rhs; }

    /**
     * Initialises the matrix from its native representation.
     *
     * This constructor enables implicit casts from native_type.
     *
     * @param data The initial data.
     */
    THE_TRY_FORCE_INLINE matrix(const native_type& data) { traits_type::copy(this->data, data); }

    /**
     * Answer the number of columns of the matrix.
     *
     * @return The number of columns of the matrix.
     */
    inline size_type columns(void) const { return C; }

    /**
     * Answer whether the quaternion is (exactly) all zero.
     *
     * Depending on the underlying storage class, this implementation might
     * be vectorised and therefore more efficient than the epsilon
     * comparison.
     *
     * @return true if the matrix is empty, false otherwise.
     */
    THE_TRY_FORCE_INLINE bool empty(void) const {
        static const matrix EMPTY;
        return this->equals(EMPTY);
    }

    /**
     * Answer whether the matrix is all zero.
     *
     * @param epsilon An epsilon value used for comparison.
     *
     * @return true if the matrix is empty, false otherwise.
     */
    bool empty(const value_type epsilon) const;

    /**
     * Test for equality.
     *
     * @param rhs The object to be compared.
     *
     * @return true if this object and 'rhs' are equal, false otherwise.
     */
    THE_TRY_FORCE_INLINE bool equals(const matrix& rhs) const { return traits_type::equals(this->data, rhs.data); }

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
    template <class Vp, size_t Rp, size_t Cp, matrix_layout Lp, class Tp>
    bool equals(const matrix<Vp, Rp, Cp, Lp, Tp>& rhs, const value_type epsilon) const;

    /**
     * Answer whether the matrix is an identity matrix.
     *
     * @param epsilon An epsilon value used for comparison.
     *
     * @return true if the matrix is an identity matrix, false otherwise.
     */
    bool identity(const value_type epsilon = megamol::core::thecam::math::epsilon<value_type>::value) const;

    /**
     * Answer the memory layout of the matrix.
     *
     * @return The memory layout of the matrix.
     */
    inline matrix_layout layout(void) const { return L; }

    /**
     * Answer the number of rows of the matrix.
     *
     * @return The number of rows of the matrix.
     */
    inline size_type rows(void) const { return R; }

    /**
     * Answer the total number of components in the matrix.
     *
     * @return The number of elements in the matrix.
     */
    inline size_type size(void) const { return (R * C); }

    /**
     * Assignment.
     *
     * @param rhs The right-hand side operand.
     *
     * @return *this.
     */
    matrix& operator=(const matrix& rhs);

    /**
     * Conversion assignment.
     *
     * @tparam Vp The value or native storage type of 'rhs'.
     * @tparam Lp The layout of 'rhs'.
     * @tparam Tp The traits type of 'rhs'.
     *
     * @param rhs The right-hand side operand.
     *
     * @return *this.
     */
    template <class Vp, matrix_layout Lp, class Tp> matrix& operator=(const matrix<Vp, R, C, Lp, Tp>& rhs);

    /**
     * Access the given matrix element.
     *
     * @param row    The row to be accessed, which must be within
     *               [0, class_type::number_of_rows[.
     * @param column The column to be accessed, which must be within
     *               [0, class_type::number_of_columns[.
     *
     * @return The size of the given dimension.
     */
    inline value_type& operator()(const size_type row, const size_type column) {
        return traits_type::at(this->data, row, column);
    }

    /**
     * Access the given matrix element.
     *
     * @param row    The row to be accessed, which must be within
     *               [0, class_type::number_of_rows[.
     * @param column The column to be accessed, which must be within
     *               [0, class_type::number_of_columns[.
     *
     * @return The size of the given dimension.
     */
    inline value_type operator()(const size_type row, const size_type column) const {
        return traits_type::at(this->data, row, column);
    }

    /**
     * Conversion to native_type.
     *
     * @return The native representation of the matrix.
     */
    THE_TRY_FORCE_INLINE operator native_type&(void) { return this->data; }

    /**
     * Conversion to native_type.
     *
     * @return The native representation of the matrix.
     */
    THE_TRY_FORCE_INLINE operator const native_type&(void)const { return this->data; }

    ///**
    // * Exposes the raw, linear data.
    // *
    // * @return A pointer to the matrix elements.
    // */
    // template<class DUMMY = void>
    // inline operator typename std::enable_if<traits_type::is_contiguous, value_type>::type *(void) {
    //    return traits_type::data(this->data);
    //}

    ///**
    // * Exposes the raw, linear data.
    // *
    // * @return A pointer to the matrix elements.
    // */
    // template<class DUMMY = void>
    // inline operator const typename std::enable_if<traits_type::is_contiguous, value_type>::type *(void) const {
    //    return traits_type::data(this->data);
    //}

    // TODO: We could optimise assignment and test for equality for
    // basic_vector because we know it has a contiguous layout in 'data'.

private:
    /** Stores the components. */
    native_type data;
};

/**
 * Computes the determinant of a matrix.
 *
 * @tparam V The scalar value or native storage type of the matrix.
 * @tparam L The memory layout of the matrix.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param matrix The matrix to compute the determinant of.
 *
 * @return The determinant of 'matrix'.
 */
template <class V, matrix_layout L, class T> inline typename T::value_type det(const matrix<V, 2, 2, L, T>& matrix) {
    return (matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0));
}


/**
 * Computes the determinant of a matrix using the rule of Sarrus.
 *
 * @tparam V The scalar value or native storage type of the matrix.
 * @tparam L The memory layout of the matrix.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param matrix The matrix to compute the determinant of.
 *
 * @return The determinant of 'matrix'.
 */
template <class V, matrix_layout L, class T> inline typename T::value_type det(const matrix<V, 3, 3, L, T>& matrix) {
    return (matrix(0, 0) * matrix(1, 1) * matrix(2, 2) + matrix(0, 1) * matrix(1, 2) * matrix(2, 0) +
            matrix(0, 2) * matrix(1, 0) * matrix(2, 1) - matrix(0, 2) * matrix(1, 1) * matrix(2, 0) -
            matrix(0, 1) * matrix(1, 0) * matrix(2, 2) - matrix(0, 0) * matrix(1, 2) * matrix(2, 1));
}


/**
 * Computes the determinant of a matrix using Laplace expansion and the rule
 * of Sarrus.
 *
 * @tparam V The scalar value or native storage type of the matrix.
 * @tparam L The memory layout of the matrix.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param matrix The matrix to compute the determinant of.
 *
 * @return The determinant of 'matrix'.
 */
template <class V, matrix_layout L, class T> typename T::value_type det(const matrix<V, 4, 4, L, T>& matrix);

#ifdef WITH_THE_GLM
/**
 * Computes the determinant of a matrix.
 *
 * @param matrix The matrix to compute the determinant of.
 *
 * @return The determinant of 'matrix'.
 */
inline float det(matrix<glm::mat4>& matrix) { return glm::determinant(static_cast<glm::mat4>(matrix)); }
#endif /* WITH_THE_GLM */


/**
 * Inverts a matrix in-place using Gauss elimination.
 *
 * @param matrix The matrix to be inverted.
 *
 * @return true if the matrix was invertable,
 *         false if the matrix was not invertable.
 */
template <class V, size_t D, matrix_layout L, class T> bool invert(matrix<V, D, D, L, T>& matrix);

#ifdef WITH_THE_GLM
/**
 * Inverts a matrix in-place.
 *
 * @param matrix The matrix to be inverted.
 *
 * @return true if the matrix was invertable,
 *         false if the matrix was not invertable.
 */
bool invert(matrix<glm::mat4>& matrix);
#endif /* WITH_THE_GLM */


/**
 * Creates a new identity matrix.
 *
 * @return An identity matrix.
 */
// template<class V, size_t D, matrix_layout L>
// inline matrix<V, D, D, L> make_identity_matrix(void) {
//    matrix<V, D, D, L> retval;
//    return std::move(megamol::core::thecam::math::set_identity(retval));
//}
// TODO: chain multiplication function?

/**
 * Makes a matrix an empty matrix.
 *
 * @tparam V The scalar value or native storage type of the matrix.
 * @tparam R The number of rows in the matrix.
 * @tparam C The number of columns in the matrix.
 * @tparam L The memory layout of the matrix.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param matrix The matrix to be set.
 *
 * @return 'matrix'.
 */
template <class V, size_t R, size_t C, matrix_layout L, class T>
matrix<V, R, C, L, T>& set_empty(matrix<V, R, C, L, T>& matrix);


/**
 * Makes a matrix an identity matrix.
 *
 * @tparam V The scalar value or native storage type of the matrix.
 * @tparam D The number of rows and columns in the matrix.
 * @tparam L The memory layout of the matrix.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param matrix The matrix to be set.
 *
 * @return 'matrix'
 */
template <class V, size_t D, matrix_layout L, class T>
matrix<V, D, D, L, T>& set_identity(matrix<V, D, D, L, T>& matrix);


/**
 * Computes the trace (sum over the main diagonal) of a matrix.
 *
 * @tparam V The scalar value or native storage type of the matrix.
 * @tparam D The number of rows and columns in the matrix.
 * @tparam L The memory layout of the matrix.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param matrix The matrix to compute the trace of.
 *
 * @return The trace of 'matrix'.
 */
template <class V, size_t D, matrix_layout L, class T>
typename T::value_type trace(const matrix<V, D, D, L, T>& matrix);


/**
 * Transpose a matrix.
 *
 * @tparam V The scalar value or native storage type of the matrix.
 * @tparam R The number of rows in the matrix.
 * @tparam C The number of columns in the matrix.
 * @tparam L The memory layout of the matrix.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param matrix The matrix to be transposed.
 *
 * @return The transposed matrix.
 */
template <class V, size_t R, size_t C, matrix_layout L, class T>
matrix<V, C, R, L, T> transpose(const matrix<V, R, C, L, T>& matrix);

#ifdef WITH_THE_GLM
/**
 * Transpose a matrix.
 *
 * @param matrix The matrix to be transposed.
 *
 * @return The transposed matrix.
 */
inline matrix<glm::mat4> transpose(const matrix<glm::mat4>& matrix) {
    return glm::transpose(static_cast<glm::mat4>(matrix));
}
#endif /* WITH_THE_GLM */

/**
 * Scale a matrix.
 *
 * @tparam V The scalar value or native storage type of the matrix.
 * @tparam R The number of rows in the matrix.
 * @tparam C The number of columns in the matrix.
 * @tparam L The memory layout of the matrix.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', scaled by 'rhs'.
 */
template <class V, size_t R, size_t C, matrix_layout L, class T>
matrix<V, R, C, L, T>& operator*=(matrix<V, R, C, L, T>& lhs, const typename matrix<V, R, C, L, T>::value_type rhs);


/**
 * Compute a scaled matrix.
 *
 * @tparam V The scalar value or native storage type of the matrix.
 * @tparam R The number of rows in the matrix.
 * @tparam C The number of columns in the matrix.
 * @tparam L The memory layout of the matrix.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', scaled by 'rhs'.
 */
template <class V, size_t R, size_t C, matrix_layout L, class T>
matrix<V, R, C, L, T> operator*(
    const matrix<V, R, C, L, T>& lhs, const typename matrix<V, R, C, L, T>::value_type rhs) {
    auto retval = lhs;
    retval *= rhs;
    return std::move(retval);
}


/**
 * Compute a scaled matrix.
 *
 * @tparam V The scalar value or native storage type of the matrix.
 * @tparam R The number of rows in the matrix.
 * @tparam C The number of columns in the matrix.
 * @tparam L The memory layout of the matrix.
 * @tparam T The traits type interpreting 'V'.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs', scaled by 'rhs'.
 */
template <class V, size_t R, size_t C, matrix_layout L, class T>
matrix<V, R, C, L, T> operator*(
    const typename matrix<V, R, C, L, T>::value_type lhs, const matrix<V, R, C, L, T>& rhs) {
    auto retval = rhs;
    retval *= lhs;
    return std::move(retval);
}


/**
 * Multiply two matrices.
 *
 * @tparam V The scalar value or native storage type of the matrix.
 * @tparam R The number of rows in the matrix.
 * @tparam C The number of columns in the matrix.
 * @tparam L The memory layout of the matrix.
 * @tparam T The traits type interpreting 'V'.
 * @tparam S The type of the scalar 'rhs'.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
template <class VL, class VR, size_t RL, size_t CL, size_t CR, matrix_layout LL, matrix_layout LR, class TL, class TR>
matrix<VL, RL, CR, LL, matrix_traits<VL, RL, CR, LL>> operator*(
    const matrix<VL, RL, CL, LL, TL>& lhs, const matrix<VR, CL, CR, LR, TR>& rhs);

#ifdef WITH_THE_GLM
/**
 * Multiply two matrices.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
inline matrix<glm::mat4> operator*(const matrix<glm::mat4>& lhs, const matrix<glm::mat4>& rhs) {
    return static_cast<glm::mat4>(lhs) * static_cast<glm::mat4>(rhs);
}
#endif /* WITH_THE_GLM */


/**
 * Multiply a matrix and a column vector.
 *
 * @tparam VM The scalar value or native storage type of the matrix.
 * @tparam VV The scalar value or native storage type of the matrix.
 * @tparam R  The number of rows in the matrix.
 * @tparam C  The number of columns in the matrix.
 * @tparam L  The memory layout of the matrix.
 * @tparam TM The traits type interpreting 'VM'.
 * @tparam TV The traits type interpreting 'VV'.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
template <class VM, class VV, size_t R, size_t C, matrix_layout L, class TM, class TV>
vector<typename TM::value_type, R> operator*(const matrix<VM, R, C, L, TM>& lhs, const vector<VV, C, TV>& rhs);


#ifdef WITH_THE_GLM
/**
 * Multiply a matrix and a column vector.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
inline thecam::math::vector<glm::vec4> operator*(const matrix<glm::mat4>& lhs, const vector<glm::vec4>& rhs) {
    return static_cast<glm::mat4>(lhs) * static_cast<glm::vec4>(rhs);
}
#endif /* WITH_THE_GLM */


/**
 * Multiply a row vector and a matrix.
 *
 * @tparam VM The scalar value or native storage type of the matrix.
 * @tparam VV The scalar value or native storage type of the matrix.
 * @tparam R  The number of rows in the matrix.
 * @tparam C  The number of columns in the matrix.
 * @tparam L  The memory layout of the matrix.
 * @tparam TM The traits type interpreting 'VM'.
 * @tparam TV The traits type interpreting 'VV'.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
template <class VM, class VV, size_t R, size_t C, matrix_layout L, class TM, class TV>
vector<typename TM::value_type, C> operator*(const vector<VV, R, TV>& lhs, const matrix<VM, R, C, L, TM>& rhs);


#ifdef WITH_THE_GLM
/**
 * Multiply a row vector and a matrix.
 *
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 *
 * @return 'lhs' * 'rhs'.
 */
inline thecam::math::vector<glm::vec4> operator*(const vector<glm::vec4>& lhs, const matrix<glm::mat4>& rhs) {
    return static_cast<glm::vec4>(lhs) * static_cast<glm::mat4>(rhs);
}
#endif /* WITH_THE_GLM */


} /* end namespace math */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#include "mmcore/thecam/math/matrix.inl"

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_MATH_MATRIX_H_INCLUDED */
