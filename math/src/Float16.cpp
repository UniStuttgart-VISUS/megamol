/*
 * Float16.cpp  21.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/Float16.h"

#include <cfloat>

#include "the/assert.h"


/** Absolute of bias of half exponent (-15). */
static const int32_t FLT16_BIAS = 15;

/** Bitmask for the exponent bits of half. */
static const uint16_t FLT16_EXPONENT_MASK = 0x7C00;

/** Bitmask for the mantissa bits of half. */
static const uint16_t FLT16_MANTISSA_MASK = 0x03FF;

/** Bitmask for the sign bit of half. */
static const uint16_t FLT16_SIGN_MASK = 0x8000;

/** Size difference between half and float mantissa. */
static const uint32_t FLT1632_MANTISSA_OFFSET 
    = (FLT_MANT_DIG - vislib::math::Float16::MANT_DIG) - 1;

/** Distance between half and float sign bit. */
static const uint32_t FLT1632_SIGN_OFFSET = (32 - 16);

/** Bias of float exponent. */
static const int32_t FLT32_BIAS = 127;

/** Bitmask for the exponent bits of float. */
static const uint32_t FLT32_EXPONENT_MASK = 0x7F800000;

/** Bitmask for the mantissa bits of half */
static const uint32_t FLT32_MANTISSA_MASK = 0x007FFFFF;

/** Bitmask for the sign bit of float */
static const uint32_t FLT32_SIGN_MASK = 0x80000000;


/*
 * vislib::math::Float16::FromFloat32
 */
void vislib::math::Float16::FromFloat32(uint16_t *outHalf, size_t cnt,
        const float *flt) {
    int32_t exponent = 0;         // Value of exponent of 'flt'.
    uint32_t input = 0;           // Bitwise reinterpretation of 'flt'.
    uint32_t mantissa = 0;        // Value of mantissa, biased for half.
    uint32_t sign = 0;            // The sign bit.

    for (size_t i = 0; i < cnt; i++) {

        /* Bitwise reinterpretation of input */
        input = *(reinterpret_cast<const uint32_t *>(flt + i));

        /* Just move the sign bit directly to its final position. */
        sign = (input & FLT32_SIGN_MASK) >> FLT1632_SIGN_OFFSET;

        /* Retrieve value of exponent. */
        exponent = static_cast<int32_t>((input & FLT32_EXPONENT_MASK) 
            >> (FLT_MANT_DIG - 1)) - FLT32_BIAS + FLT16_BIAS;

        /* Retrieve value of mantissa. */
        mantissa = (input & FLT32_MANTISSA_MASK);

        if (exponent < -Float16::MANT_DIG) {
            /* Underflow, result is zero. */
            outHalf[i] = 0;

        } else if (exponent <= 0) {
            /* Negative exponent. */

            /* Normalise the number. */
            mantissa = (mantissa | (1 << (FLT_MANT_DIG - 1))) >> (1 - exponent);
                
            outHalf[i] = static_cast<uint16_t>(sign 
                | (mantissa >> FLT1632_MANTISSA_OFFSET));

        } else if (exponent == (FLT32_EXPONENT_MASK >> (FLT_MANT_DIG - 1)) 
                - FLT32_BIAS + FLT16_BIAS) {
            /* 
             * If all exponent bits are set, the input float is either infinity 
             * ('mantissa' == 0) or NaN ('mantissa' != 0). If all mantissa bits
             * are set, NaN is quiet, otherwise, it is signaling. Truncating the
             * mantissa should preserve this.
             */
            
            if (mantissa != 0) {
                /* 
                 * Truncate mantissa, if it contains data (i. e. is NaN), but 
                 * make sure that the mantissa does not become zero by shifting
                 * out all relevant bits as this would make the NaN infinity.
                 */
                mantissa >>= FLT1632_MANTISSA_OFFSET;
                mantissa |= (mantissa == 0);
            }

            outHalf[i] = static_cast<uint16_t>(sign | FLT16_EXPONENT_MASK
                | mantissa);

        } else if (exponent > Float16::MAX_EXP + FLT16_BIAS) {
            /* Overflow, result becomes infinity. */
            outHalf[i] = static_cast<uint16_t>(sign | FLT16_EXPONENT_MASK);

        } else {
            /* 
             * Normal, valid number, truncate mantissa and move exponent to its
             * final position.  
             */
            outHalf[i] = sign | (exponent << Float16::MANT_DIG)
                | (mantissa >> FLT1632_MANTISSA_OFFSET);
        } /* end if (exponent <= 0) */
    } /* end for (size_t i = 0; i < cnt; i++) */
}


/*
 * vislib::math::Float16::ToFloat32
 */
void vislib::math::Float16::ToFloat32(float *outFloat, const size_t cnt,
        const uint16_t *half) {
    int32_t exponent = 0;         // Value of exponent of 'half'.
    uint32_t mantissa = 0;        // Value of mantissa.
    uint32_t result = 0;          // The result
    uint32_t sign = 0;            // The sign bit.

    for (size_t i = 0; i < cnt; i++) {

        /* Just move the sign bit directly to its final position. */
        sign = static_cast<uint32_t>(half[i] & FLT16_SIGN_MASK) 
            << FLT1632_SIGN_OFFSET;

        /* Retrieve value of exponent. */
        exponent = static_cast<int32_t>((half[i] & FLT16_EXPONENT_MASK) 
            >> Float16::MANT_DIG);

        /* Retrieve value of mantissa. */
        mantissa = static_cast<uint32_t>(half[i] & FLT16_MANTISSA_MASK);

        if (exponent == 0) {
            if (mantissa == 0) {
                /* Value is zero, preserve sign. */
                result = sign;

            } else {
                /* Denormalised. */

                /* 
                 * Shift left until the first 1 occurs left of the mantissa. 
                 * This is the implicit 1 which must be deleted afterwards.
                 */
                while ((mantissa & (1 << Float16::MANT_DIG)) == 0) {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                exponent += FLT32_BIAS - FLT16_BIAS + 1;
                mantissa &= ~(1 << Float16::MANT_DIG);

                result = sign | (exponent << (FLT_MANT_DIG - 1))
                    | (mantissa << FLT1632_MANTISSA_OFFSET);
            }

        } else if (exponent == (FLT16_EXPONENT_MASK >> Float16::MANT_DIG)) {
            /*
             * All exponent bits are set, the number is infinity, if the 
             * mantissa is zero, or NaN otherwise. If all mantissa bits are
             * set, the number is a quiet NaN, if only some are set, it is
             * a signaling NaN.
             */

            if (mantissa == 0) {
                /* Infinity. */
                result = (sign | FLT32_EXPONENT_MASK);

            } else if (mantissa == FLT16_MANTISSA_MASK) {
                /* Quiet NaN. */
                result = sign | FLT32_EXPONENT_MASK | FLT32_MANTISSA_MASK;

            } else {
                /* Signaling NaN. */
                result = sign | FLT32_EXPONENT_MASK 
                    | (mantissa << FLT1632_MANTISSA_OFFSET);
            }

        } else {
            result = sign
                | ((exponent - FLT16_BIAS + FLT32_BIAS) << (FLT_MANT_DIG - 1)) 
                | (mantissa << FLT1632_MANTISSA_OFFSET);
        }

        /* Bitwise reinterpret the result as float. */
        outFloat[i] = *reinterpret_cast<float *>(&result);

    } /* end for (size_t i = 0; i < cnt; i++) */
}


/*
 * vislib::math::Float16::MANT_DIG
 */
const int vislib::math::Float16::MANT_DIG = 10;


/*
 * vislib::math::Float16::MAX_EXP
 */
const int vislib::math::Float16::MAX_EXP = (16 - 1);


/*
 * vislib::math::Float16::MIN_EXP
 */
const int vislib::math::Float16::MIN_EXP = -12;


/*
 * vislib::math::Float16::RADIX
 */
const int vislib::math::Float16::RADIX = 2;


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
 * vislib::math::Float16::~Float16
 */
vislib::math::Float16::~Float16(void) {
}


/*
 * vislib::math::Float16::IsInfinity
 */
bool vislib::math::Float16::IsInfinity(void) const {
    // All exponents and no mantissa bits set.
    return (((this->value & FLT16_EXPONENT_MASK) == FLT16_EXPONENT_MASK)
        && ((this->value & FLT16_MANTISSA_MASK) == 0));
}


/*
 * vislib::math::Float16::IsNaN
 */
bool vislib::math::Float16::IsNaN(void) const {
    // All exponent and some mantissa bits set.
    return (((this->value & FLT16_EXPONENT_MASK) == FLT16_EXPONENT_MASK)
        && ((this->value & FLT16_MANTISSA_MASK) != 0));
}


/*
 *  vislib::math::Float16::IsQuietNaN
 */
bool vislib::math::Float16::IsQuietNaN(void) const {
    // All exponent and all mantissa bits set.
    return (((this->value & FLT16_EXPONENT_MASK) == FLT16_EXPONENT_MASK)
        && ((this->value & FLT16_MANTISSA_MASK) == FLT16_MANTISSA_MASK));
}


/*
 * vislib::math::Float16::IsSignalingNaN
 */
bool vislib::math::Float16::IsSignalingNaN(void) const {
    // All exponent bits set, some, but not all, mantissa bits set.
    return (((this->value & FLT16_EXPONENT_MASK) == FLT16_EXPONENT_MASK)
        && ((this->value & FLT16_MANTISSA_MASK) != 0)
        && ((this->value & FLT16_MANTISSA_MASK) != FLT16_MANTISSA_MASK));
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
