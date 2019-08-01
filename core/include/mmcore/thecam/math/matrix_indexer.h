/*
 * the\math\matrix_indexer.h
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

#ifndef THE_MATH_MATRIX_FUNCTIONS_H_INCLUDED
#define THE_MATH_MATRIX_FUNCTIONS_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include "mmcore/thecam/utility/force_inline.h"

#include "mmcore/thecam/math/mathtypes.h"


namespace megamol {
namespace core {
namespace thecam {
namespace math {

/**
 * Provides static methods for getting the index of a matrix entry for a
 * specific memory layout.
 *
 * The default implementation does not provide any implementation, only the
 * partial specialisations for the supported layouts work correctly.
 *
 * @tparam R The total number of rows in the matrix to compute the indices
 *           for.
 * @tparam C The total number of columns in the matrix to compute the
 *           indices for.
 * @tparam L The memory layout of the matrix to compute the indices for.
 */
template <size_t R, size_t C, matrix_layout L> struct matrix_indexer {};


/**
 * Provides static methods for getting the index of a matrix entry for a
 * specific memory layout.
 *
 * The default implementation does not provide any implementation, only the
 * partial specialisations for the supported layouts work correctly.
 *
 * @tparam R The total number of rows in the matrix to compute the indices
 *           for.
 * @tparam C The total number of columns in the matrix to compute the
 *           indices for.
 * @tparam L The memory layout of the matrix to compute the indices for.
 */
template <size_t R, size_t C> struct matrix_indexer<R, C, matrix_layout::column_major> {

    /** The type of the indices. */
    typedef size_t index_type;

    /**
     * Answer the number of columns in the matrix that the indexer is for.
     *
     * @return The number of columns the indexer is for.
     */
    static THE_FORCE_INLINE size_t columns(void) { return C; }

    /**
     * Answer the index of the given element.
     *
     * Please note that the method does not perform any range-checks! Use
     * matrix_indexer::valid if you are unsure.
     *
     * @param row    The zero-based row index.
     * @param column The zero-based column index.
     *
     * @return The position of the given element in the matrix.
     */
    static THE_FORCE_INLINE index_type index(const index_type row, const index_type column) {
        return (column * R + row);
    }

    /**
     * Answer the memory layout that the indexer is for.
     *
     * @return The memory layout the indexer is for.
     */
    static THE_FORCE_INLINE matrix_layout layout(void) { return matrix_layout::column_major; }

    /**
     * Answer the number of rows in the matrix that the indexer is for.
     *
     * @return The number of rows the indexer is for.
     */
    static THE_FORCE_INLINE size_t rows(void) { return R; }

    /**
     * Answer whether (row, column) designates a valid element of the matrix
     * that the indexer is for.
     *
     * @param row    The zero-based row index.
     * @param column The zero-based column index.
     *
     * @return true if the given element is valid, false otherwise.
     */
    static THE_FORCE_INLINE bool valid(const index_type row, const index_type column) {
        return ((row < R) && (column < C));
    }
};


/**
 * Provides static methods for getting the index of a matrix entry for a
 * specific memory layout.
 *
 * The default implementation does not provide any implementation, only the
 * partial specialisations for the supported layouts work correctly.
 *
 * @tparam R The total number of rows in the matrix to compute the indices
 *           for.
 * @tparam C The total number of columns in the matrix to compute the
 *           indices for.
 * @tparam L The memory layout of the matrix to compute the indices for.
 */
template <size_t R, size_t C> struct matrix_indexer<R, C, matrix_layout::row_major> {

    /** The type of the indices. */
    typedef size_t index_type;

    /**
     * Answer the number of columns in the matrix that the indexer is for.
     *
     * @return The number of columns the indexer is for.
     */
    static THE_FORCE_INLINE size_t columns(void) { return C; }

    /**
     * Answer the index of the given element.
     *
     * Please note that the method does not perform any range-checks! Use
     * matrix_indexer::valid if you are unsure.
     *
     * @param row    The zero-based row index.
     * @param column The zero-based column index.
     *
     * @return The position of the given element in the matrix.
     */
    static THE_FORCE_INLINE index_type index(const index_type row, const index_type column) {
        return (row * C + column);
    }

    /**
     * Answer the memory layout that the indexer is for.
     *
     * @return The memory layout the indexer is for.
     */
    static THE_FORCE_INLINE matrix_layout layout(void) { return matrix_layout::row_major; }

    /**
     * Answer the number of rows in the matrix that the indexer is for.
     *
     * @return The number of rows the indexer is for.
     */
    static THE_FORCE_INLINE size_t rows(void) { return R; }

    /**
     * Answer whether (row, column) designates a valid element of the matrix
     * that the indexer is for.
     *
     * @param row    The zero-based row index.
     * @param column The zero-based column index.
     *
     * @return true if the given element is valid, false otherwise.
     */
    static THE_FORCE_INLINE bool valid(const index_type row, const index_type column) {
        return ((row < R) && (column < C));
    }
};

} /* end namespace math */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_MATH_MATRIX_FUNCTIONS_H_INCLUDED */
