/*
 * Float16.cpp  21.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/assert.h"
#include "vislib/Float16.h"


// The half data type is a floating-point data type encoded in an unsigned 
// scalar data type.  If the unsigned scalar holding a half has a value of N,
// the corresponding floating point number is
//
//      (-1)^S * 0.0,                        if E == 0 and M == 0,
//      (-1)^S * 2^-14 * (M / 2^10),         if E == 0 and M != 0,
//      (-1)^S * 2^(E-15) * (1 + M/2^10),    if 0 < E < 31,
//      (-1)^S * INF,                        if E == 31 and M == 0, or
//      NaN,                                 if E == 31 and M != 0,
//
//    where
//
//      S = floor((N mod 65536) / 32768),
//      E = floor((N mod 32768) / 1024), and
//      M = N mod 1024.
//
//    INF (Infinity) is a special representation indicating numerical overflow.
//    NaN (Not a Number) is a special representation indicating the result of
//    illegal arithmetic operations, such as computing the square root or
//    logarithm of a negative number.  Note that all normal values, zero, and
//    INF have an associated sign.  -0.0 and +0.0 are considered equivalent for
//    the purposes of comparisons.  Note also that half is not a native type in
//    most CPUs, so some special processing may be required to generate or
//    interpret half data.


/** Absolute of bias of half exponent (-15). */
static const UINT16 FLT16_BIAS = 0x0070;

/** Bitmask for the exponent bits of half. */
static const UINT16 FLT16_EXPONENT_MASK = 0x7C00;

/** Bitmask for the mantissa bits of half. */
static const UINT16 FLT16_MANTISSA_MASK = 0x03FF;

/** Bitmask for the sign bit of half. */
static const UINT16 FLT16_SIGN_MASK = 0x8000;

/** Distance between half and float sign bit. */
static const UINT32 FLT1632_SIGN_OFFSET = 0x00000010;

/** Bias of float exponent. */
static const UINT32 FLT32_BIAS = 127;

/** Bitmask for the exponent bits of float. */
static const UINT32 FLT32_EXPONENT_MASK = 0x7F800000;

/** Offset of the exponent bits of float. */
static const UINT32 FLT32_EXPONENT_OFFSET = 0x00000017;

/** Bitmask for the mantissa bits of half */
static const UINT32 FLT32_MANTISSA_MASK = 0x007FFFFF;

/** Bitmask for the sign bit of float */
static const UINT32 FLT32_SIGN_MASK = 0x80000000;


/*
 * vislib::math::Float16::Float32To16
 */
UINT16 vislib::math::Float16::Float32To16(const float flt) {
    // Bitwise reinterpretation of input.
    const UINT32 input = *(reinterpret_cast<const UINT32 *>(&flt));

    /* Retrieve value of exponent and mantissa of the input number. */
    const register UINT32 exponent = ((input & FLT32_EXPONENT_MASK) 
        >> FLT32_EXPONENT_OFFSET);
    const register UINT32 mantissa = (input & FLT32_MANTISSA_MASK);

    /* Just move the sign bit directly to its final position. */
    UINT32 result = (input & FLT32_SIGN_MASK) >> FLT1632_SIGN_OFFSET;



    //register int s =  (i >> 16) & 0x00008000;
    //register int e = ((i >> 23) & 0x000000ff) - (127 - 15);
    //register int m =   i        & 0x007fffff;
    //     
    // if (e <= 0) {
    //     if (e < -10){
    //         return 0;
    //     }
    //     m = (m | 0x00800000) >> (1 - e);
 
    //     return s | (m >> 13);
    // } else if (e == 0xff - (127 - 15)) {
    //     if (m == 0) { // Inf 
    //         return s | 0x7c00;
    //     } else {   // NAN
    //         m >>= 13;
    //         return s | 0x7c00 | m | (m == 0);
    //     }
    // } else {
    //     if (e > 30) { // Overflow
    //         return s | 0x7c00;
    //     }
 
    //     return s | (e << 10) | (m >> 13);
    // }


    // TODO: Implementation missing.
    ASSERT(false);

    return static_cast<UINT16>(result);
}


/*
 * vislib::math::Float16:Float16To32
 */
float vislib::math::Float16::Float16To32(const UINT16 half) {
    UINT32 result = 0;

    // TODO: Implementation missing.
    ASSERT(false);

             //register int s = (y >> 15) & 0x00000001;
             //register int e = (y >> 10) & 0x0000001f;
             //register int m =  y        & 0x000003ff;
         
             //if (e == 0)
             //{
             //    if (m == 0) // Plus or minus zero
             //    {
             //        return s << 31;
             //    }
             //    else // Denormalized number -- renormalize it
             //    {
             //        while (!(m & 0x00000400))
             //        {
             //            m <<= 1;
             //            e -=  1;
             //        }
         
             //        e += 1;
             //        m &= ~0x00000400;
             //    }
             //}
             //else if (e == 31)
             //{
             //    if (m == 0) // Inf
             //    {
             //        return (s << 31) | 0x7f800000;
             //    }
             //    else // NaN
             //    {
             //        return (s << 31) | 0x7f800000 | (m << 13);
             //    }
             //}
         
             //e = e + (127 - 15);
             //m = m << 13;
         
             //return (s << 31) | (e << 23) | m;


    // Bitwise reinterpret of result as floating point number.
    return *(reinterpret_cast<float *>(&result));
}


/*
 * vislib::math::Float16::DIG
 */
const INT vislib::math::Float16::DIG = 3;


/*
 * vislib::math::Float16::MANT_DIG
 */
const INT vislib::math::Float16::MANT_DIG = 11;


/*
 * vislib::math::Float16::MAX_EXP
 */
const INT vislib::math::Float16::MAX_EXP = 15;


/*
 * vislib::math::Float16::MIN_EXP
 */
const INT vislib::math::Float16::MIN_EXP = -12;


/*
 * vislib::math::Float16::RADIX
 */
const INT vislib::math::Float16::RADIX = 2;


/*
 * vislib::math::Float16::ROUNDS
 */
const INT vislib::math::Float16::ROUNDS = 1;


/*
 * vislib::math::Float16::EPSILON
 */
const float vislib::math::Float16::EPSILON = 4.8875809e-4f;


/*
 * vislib::math::Float16::MAX
 */
const double vislib::math::Float16::MAX = 6.550400e+004;


/*
 * vislib::math::Float16::MIN
 */
const double vislib::math::Float16::MIN = 6.1035156e-5;


/*
 * vislib::math::Float16::IsInfinity
 */
bool vislib::math::Float16::IsInfinity(void) const {
    // Infinity, if all exponent bits set and no mantissa bit set.
    // TODO: Sign bit?
    // TODO: Performance?
    return (((this->value & FLT16_EXPONENT_MASK) == FLT16_EXPONENT_MASK)
        && ((this->value & FLT16_MANTISSA_MASK) == 0));
}


/*
 * vislib::math::Float16::IsNaN
 */
bool vislib::math::Float16::IsNaN(void) const {
    // Not a number, if all exponent bits set and mantissa not 0.
    // TODO: Sign bit?
    // TODO: Performance?
    return (((this->value & FLT16_EXPONENT_MASK) == FLT16_EXPONENT_MASK)
        && ((this->value & FLT16_MANTISSA_MASK) != 0));
}


/*
 * vislib::math::Float16::operator =
 */
 vislib::math::Float16& vislib::math::Float16::operator =(const Float16& rhs) {
    if (this != &rhs) {
        this->value = rhs.value;
    }

    return *this;
}
