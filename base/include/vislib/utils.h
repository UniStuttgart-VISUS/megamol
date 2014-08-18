/*
 * utils.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_UTILS_H_INCLUDED
#define VISLIB_UTILS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/String.h"
#include "vislib/types.h"


namespace vislib {

    /**
     * Convert an array of bytes to a hex string.
     *
     * @param bytes    The byte array to be converted.
     * @param cntBytes The size of 'bytes' in bytes.
     *
     * @return A string representing the content of 'bytes.
     */
    StringA BytesToHexStringA(const BYTE *bytes, SIZE_T cntBytes);

    /**
     * Convert an array of bytes to a hex string.
     *
     * @param bytes    The byte array to be converted.
     * @param cntBytes The size of 'bytes' in bytes.
     *
     * @return A string representing the content of 'bytes.
     */
    StringW BytesToHexStringW(const BYTE *bytes, SIZE_T cntBytes);

    /**
     * Swaps the values of left and right.
     *
     * @param left  A reference to a variable to be swapped
     * @param right A reference to a variable to be swapped
     */
    template<class T> void Swap(T &left, T &right) {
        T tmp = left;
        left = right;
        right = tmp;
    }

    /**
     * Swaps the values of left and right. 
     * Uses the specified temporary variable.
     *
     * @param left  A reference to a variable to be swapped
     * @param right A reference to a variable to be swapped
     * @param tmp   A reference to a temporary variable.
     */
    template<class T> void Swap(T &left, T &right, T &tmp) {
        tmp = left; 
        left = right; 
        right = tmp;
    }

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
    template<class T> int DiffComparator(const T& lhs, const T& rhs) {
        T diff = (lhs - rhs);
        return (diff < 0) ? -1 : ((diff > 0) ? 1 : 0);
    }

    /**
     * Encodes an UINT64 as RLE-UInt
     *
     * @param dst The bytes to receive the encoded data.
     *            An UINT64 can be encoded in a maximum of 10 bytes.
     * @param len The length of the encoded data. The caller must set the
     *            length of the memory allocated at 'dst'. The function will
     *            set the actual size of the encoded value
     * @param src The value to be encoded
     *
     * @return True on success. False if the value could not be encoded
     *         because the size of 'dst' provided by 'len' was insufficient.
     */
    bool UIntRLEEncode(unsigned char *dst, unsigned int &len, UINT64 src);

    /**
     * Decodes an RLE-UInt to its original UINT64
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
     *         was set, and thus the value would not fit into UINT64.
     */
    bool UIntRLEDecode(UINT64 &dst, unsigned char *src, unsigned int &len);

    /**
     * Answer the number of bytes 'val' will require when encoded as RLE-UInt
     *
     * @param val The value
     *
     * @return The number of bytes required to represent 'val' as RLE-UInt
     */
    inline unsigned int UIntRLELength(UINT64 val) {
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

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_UTILS_H_INCLUDED */
