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


/*
 * ...matrix_indexer<R, C, megamol::core::thecam::math::matrix_layout::column_major>::columns
 */
#include "matrix.h"
template <size_t R, size_t C>
const size_t megamol::core::thecam::math::matrix_indexer<R, C,
    megamol::core::thecam::math::matrix_layout::column_major>::columns = C;


/*
 * ...matrix_indexer<R, C, the::math::matrix_layout::column_major>::layout
 */
template <size_t R, size_t C>
const megamol::core::thecam::math::matrix_layout megamol::core::thecam::math::matrix_indexer<R, C,
    megamol::core::thecam::math::matrix_layout::column_major>::layout =
    megamol::core::thecam::math::matrix_layout::column_major;


/*
 * ...matrix_indexer<R, C, the::math::matrix_layout::column_major>::columns
 */
template <size_t R, size_t C>
const size_t
    megamol::core::thecam::math::matrix_indexer<R, C, megamol::core::thecam::math::matrix_layout::column_major>::rows =
        R;


/*
 * ...matrix_indexer<R, C, the::math::matrix_layout::row_major>::columns
 */
template <size_t R, size_t C>
const size_t
    megamol::core::thecam::math::matrix_indexer<R, C, megamol::core::thecam::math::matrix_layout::row_major>::columns =
        C;


/*
 * ...matrix_indexer<R, C, the::math::matrix_layout::row_major>::layout
 */
template <size_t R, size_t C>
const megamol::core::thecam::math::matrix_layout
    megamol::core::thecam::math::matrix_indexer<R, C, megamol::core::thecam::math::matrix_layout::row_major>::layout =
        megamol::core::thecam::math::matrix_layout::row_major;


/*
 * ...matrix_indexer<R, C, the::math::matrix_layout::row_major>::columns
 */
template <size_t R, size_t C>
const size_t
    megamol::core::thecam::math::matrix_indexer<R, C, megamol::core::thecam::math::matrix_layout::row_major>::rows = R;
