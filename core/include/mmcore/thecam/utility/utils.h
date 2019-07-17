/*
 * the/utils.h
 *
 * Copyright (c) 2012, TheLib Team (http://www.thelib.org/license)
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
 * utils.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef THE_UTILS_H_INCLUDED
#define THE_UTILS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

//#include "vislib/String.h"
#include "mmcore/thecam/utility/types.h"


namespace megamol {
namespace core {
namespace thecam {
namespace utility {

///**
// * Swaps the values of left and right.
// *
// * @param left  A reference to a variable to be swapped
// * @param right A reference to a variable to be swapped
// */
// template<class T> void swap(T &left, T &right) {
//    T tmp = left;
//    left = right;
//    right = tmp;
//}

///**
// * Swaps the values of left and right.
// * Uses the specified temporary variable.
// *
// * @param left  A reference to a variable to be swapped
// * @param right A reference to a variable to be swapped
// * @param tmp   A reference to a temporary variable.
// */
// template<class T> void swap(T &left, T &right, T &tmp) {
//    tmp = left;
//    left = right;
//    right = tmp;
//}

/**
 * A comparator using the operator '-'. You can use this comparator for
 * sorting collections of basic types.
 *
 * @param lhs The left hand side operand
 * @param rhs The right hand side operand
 *
 * @return (lhs - rhs)
 *          = 0 if lhs == rhs
 *          < 0 if lhs < rhs
 *          > 0 if lhs > rhs
 */
template <class T> int diff_comparator(const T& lhs, const T& rhs) {
    T diff = (lhs - rhs);
    return (diff < 0) ? -1 : ((diff > 0) ? 1 : 0);
}

/**
 * Encodes an uint64_t as RLE-UInt
 *
 * @param dst The bytes to receive the encoded data.
 *            An uint64_t can be encoded in a maximum of 10 bytes.
 * @param len The length of the encoded data. The caller must set the
 *            length of the memory allocated at 'dst'. The function will
 *            set the actual size of the encoded value
 * @param src The value to be encoded
 *
 * @return True on success. False if the value could not be encoded
 *         because the size of 'dst' provided by 'len' was insufficient.
 */
bool uint_rle_encode(unsigned char* dst, unsigned int& len, uint64_t src);

/**
 * Decodes an RLE-UInt to its original uint64_t
 *
 * @param dst The result from decoding
 * @param src Pointer to the bytes holding the encoded value.
 * @param len The length of the encoded value. The caller must set the
 *            number of bytes, which can safely be read from 'src'. The
 *            method will set the number of bytes actually used for
 *            decoding.
 *
 * @return True on success. False if the number of bytes provided in 'src'
 *         were either insufficient or the RLE-bit in the ninethed byte
 *         was set, and thus the value would not fit into uint64_t.
 */
bool uint_rle_decode(uint64_t& dst, unsigned char* src, unsigned int& len);

/**
 * Answer the number of bytes 'val' will require when encoded as RLE-UInt
 *
 * @param val The value
 *
 * @return The number of bytes required to represent 'val' as RLE-UInt
 */
inline unsigned int uint_rle_length(uint64_t val) {
    if (val < 0x80ul) return 1;
    if (val < 0x4000ul) return 2;
    if (val < 0x200000ul) return 3;
    if (val < 0x10000000ul) return 4;
    if (val < 0x0800000000ul) return 5;
    if (val < 0x040000000000ul) return 6;
    if (val < 0x02000000000000ul) return 7;
    if (val < 0x0100000000000000ul) return 8;
    if (val < 0x8000000000000000ul) return 9;
    return 10;
}

} /* end namespace utility */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_UTILS_H_INCLUDED */
