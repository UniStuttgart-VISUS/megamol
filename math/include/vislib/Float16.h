/*
 * Float16.h  21.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FLOAT16_H_INCLUDED
#define VISLIB_FLOAT16_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/types.h"


namespace vislib {
namespace math {

    /**
     * Objects of this class represent a half precision floating point number
     * that consists of only 16 bits.
     *
     * The floating point number is represented as follows:
     *
     * | 15   | 14 .. 10 | 9 .. 0   |
     * | Sign | Exponent | Mantissa |
     *
     * The exponent has a bias of 15.
     */
    class Float16 {

public:

        /**
         * TODO: Documentation
         *
         */
        static UINT16 Float32To16(const float flt);

        /**
         * TODO: Documentation
         *
         */
        static float Float16To32(const UINT16 half);

        /** Number of decimal digits of precision. */
        static const INT DIG;

        /** Number of bits in mantissa. */
        static const INT MANT_DIG;

        /** Maximum binary exponent. */
        static const INT MAX_EXP;

        /** Minimum binary exponent. */
        static const INT MIN_EXP;

        /** Exponent radix. */
        static const INT RADIX;

        /** Addition rounding: near. */
        static const INT ROUNDS;

        /** Smallest such that 1.0 + epsilon != 1.0 */
        static const float EPSILON;

        /** Maximum value. */
        static const double MAX;

        /** Minimum positive value. */
        static const double MIN;

        /** Ctor. */
        inline Float16(void) : value(0) {}

        /** 
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Float16(const Float16& rhs) : value(rhs.value) {}

        /**
         * Cast ctor. from float.
         *
         * @param value The initial value of the half float.
         */
        inline Float16(const float value) 
            : value(Float16::Float32To16(value)) {}

        /**
         * Cast ctor. from double.
         *
         * @param value The initial value of the half float.
         */
        inline Float16(const double value)
            : value(Float16::Float32To16(static_cast<float>(value))) {}

        /**
         * Cast ctor. from BYTE.
         *
         * @param value The initial value of the half float.
         */
        inline Float16(const BYTE value)
            : value(Float16::Float32To16(static_cast<float>(value))) {}

        /**
         * Cast ctor. from SHORT.
         *
         * @param value The initial value of the half float.
         */
        inline Float16(const SHORT value) 
            : value(Float16::Float32To16(static_cast<float>(value))) {}

        /**
         * Cast ctor. from INT32.
         *
         * @param value The initial value of the half float.
         */
        inline Float16(const INT32 value)
            : value(Float16::Float32To16(static_cast<float>(value))) {}

        /**
         * Cast ctor. from INT64.
         *
         * @param value The initial value of the half float.
         */
        inline Float16(const INT64 value)
            : value(Float16::Float32To16(static_cast<float>(value))) {}

        /** Dtor. */
        ~Float16(void);

        bool IsInfinity(void) const;

        bool IsNaN(void) const;

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        Float16& operator =(const Float16& rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if 'rhs' and this string are equal, false otherwise.
         */
        inline bool operator ==(const Float16& rhs) const {
            return (this->value == rhs.value);
        }

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if 'rhs' and this string are not equal, false 
         *         otherwise.
         */
        inline bool operator !=(const Float16& rhs) const {
            return (this->value != rhs.value);
        }

        /**
         * Cast to float.
         *
         * @return The value of this half.
         */
        inline operator float(void) const {
            return Float16::Float16To32(this->value);
        }

        /**
         * Cast to double.
         *
         * @return The value of this half.
         */
        inline operator double(void) const {
            return static_cast<double>(Float16::Float16To32(this->value));
        }

        /**
         * Cast to BYTE.
         *
         * @return The value of this half.
         */
        inline operator BYTE(void) const {
            return static_cast<BYTE>(Float16::Float16To32(this->value));
        }

        /**
         * Cast to SHORT.
         *
         * @return The value of this half.
         */
        inline operator SHORT(void) const {
            return static_cast<SHORT>(Float16::Float16To32(this->value));
        }

        /**
         * Cast to INT32.
         *
         * @return The value of this half.
         */
        inline operator INT32(void) const {
            return static_cast<INT32>(Float16::Float16To32(this->value));
        }

        /**
         * Cast to INT64.
         *
         * @return The value of this half.
         */
        inline operator INT64(void) const {
            return static_cast<INT64>(Float16::Float16To32(this->value));
        }

private:

        /** The value represented by this object. */
        UINT16 value;
    };

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_FLOAT16_H_INCLUDED */
