/*
 * Float16.h  21.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FLOAT16_H_INCLUDED
#define VISLIB_FLOAT16_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


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
     * The exponent has a bias of -15. The mantissa has an implicit 1 bit.
     *
     * The following representation of special values are assumed for Float16:
     *
     * Sign Exponent Mantissa      Meaning 
     *    0 111 11   ** **** ****  NAN
     *    0 111 11   00 0000 0000  Infinity
     *    0 111 10   11 1111 1111  65504 (Largest finite value)
     *    0 000 01   00 0000 0000  Float16::MIN (Smallest normalized value)
     *    0 000 00   00 0000 0001  6 * 10-8 (Smallest denormalized value)
     *    0 000 00   00 0000 0000  0
     *    0 011 11   00 0000 0000  1  
     *
     */
    class Float16 {
        // Implementation notes: It is crucial that this class does not have
        // any virtual method in order to ensure its size of exactly two bytes.

    public:

        /**
         * Convert an array of 'cnt' floats to half precision floats. The caller
         * is and remains owner of the memory designated by 'outHalf' and 'flt'.
         * 
         * @param outHalf An array for at least 'cnt' half precision floats. The
         *                caller must provide the memory and remains owner.
         * @param cnt     The number of elements in 'flt' and 'outHalf'.
         * @param flt     An array of at least 'cnt' floats. The caller remains
         *                owner of the memory.
         */
        static void FromFloat32(UINT16 *outHalf, SIZE_T cnt,
            const float *flt);

        /**
         * Convert 'flt' to half.
         *
         * @param flt A 32-bit floating point number.
         *
         * @return 'flt' as half.
         */
        inline static UINT16 FromFloat32(const float flt) {
            UINT16 retval;
            Float16::FromFloat32(&retval, 1, &flt);
            return retval;
        }
    
        /**
         * Convert an array of 'cnt' half precision floats to 32 bit floats. The
         * caller is and remains owner of the memory designated by 'outFloat' and 
         * 'half'.
         * 
         * @param outFloat An array for at least 'cnt' floats. The caller must 
         *                 provide the memory and remains owner.
         * @param cnt      The number of elements in 'half' and 'outFloat'.
         * @param half     An array of at least 'cnt' 16 bit floats. The caller 
         *                 remains owner of the memory.
         */
        static void ToFloat32(float *outFloat, const SIZE_T cnt,
            const UINT16 *half);

        /**
         * Convert 'half' to float.
         *
         * @param half A 16-bit half precision floating point number.
         * 
         * @return 'half' as full float.
         */
        inline static float ToFloat32(const UINT16 half) {
            float retval;
            Float16::ToFloat32(&retval, 1, &half);
            return retval;
        }

        /** Number of bits in mantissa. */
        static const INT MANT_DIG;

        /** Maximum binary exponent. */
        static const INT MAX_EXP;

        /** Minimum binary exponent. */
        static const INT MIN_EXP;

        /** Exponent radix. */
        static const INT RADIX;

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
        inline Float16(const float value) {
            Float16::FromFloat32(&this->value, 1, &value);
        }

        /**
         * Cast ctor. from double.
         *
         * @param value The initial value of the half float.
         */
        inline Float16(const double value)
            : value(Float16::FromFloat32(static_cast<float>(value))) {}

        /**
         * Cast ctor. from BYTE.
         *
         * @param value The initial value of the half float.
         */
        inline Float16(const BYTE value)
            : value(Float16::FromFloat32(static_cast<float>(value))) {}

        /**
         * Cast ctor. from SHORT.
         *
         * @param value The initial value of the half float.
         */
        inline Float16(const SHORT value) 
            : value(Float16::FromFloat32(static_cast<float>(value))) {}

        /**
         * Cast ctor. from INT32.
         *
         * @param value The initial value of the half float.
         */
        inline Float16(const INT32 value)
            : value(Float16::FromFloat32(static_cast<float>(value))) {}

        /**
         * Cast ctor. from INT64.
         *
         * @param value The initial value of the half float.
         */
        inline Float16(const INT64 value)
            : value(Float16::FromFloat32(static_cast<float>(value))) {}

        /** Dtor. */
        ~Float16(void);

        /**
         * Answer whether the number represents positive or negative infinity.
         *
         * @return true, if the number represents infinity.
         */
        bool IsInfinity(void) const;

        /**
         * Answer whether this number is a NaN.
         *
         * @return true, if the number is a NaN, false otherwise.
         */
        bool IsNaN(void) const;

        /**
         * Answer whether the number is a quiet NaN. Note that the number
         * might be a signaling NaN, if this method returns false.
         *
         * @return true, if the number is a quiet NaN, false otherwise.
         */
        bool IsQuietNaN(void) const;

        /**
         * Answer whether the number is a signaling NaN. Note that the number
         * might be a quiet NaN, if this method returns false.
         *
         * @return true, if the number is a signaling NaN, false otherwise.
         */
        bool IsSignalingNaN(void) const;

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
            // TODO: Might want to treat all signaling NaN equal.
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
            return Float16::ToFloat32(this->value);
        }

        /**
         * Cast to double.
         *
         * @return The value of this half.
         */
        inline operator double(void) const {
            return static_cast<double>(Float16::ToFloat32(this->value));
        }

        /**
         * Cast to BYTE.
         *
         * @return The value of this half.
         */
        inline operator BYTE(void) const {
            return static_cast<BYTE>(Float16::ToFloat32(this->value));
        }

        /**
         * Cast to SHORT.
         *
         * @return The value of this half.
         */
        inline operator SHORT(void) const {
            return static_cast<SHORT>(Float16::ToFloat32(this->value));
        }

        /**
         * Cast to INT32.
         *
         * @return The value of this half.
         */
        inline operator INT32(void) const {
            return static_cast<INT32>(Float16::ToFloat32(this->value));
        }

        /**
         * Cast to INT64.
         *
         * @return The value of this half.
         */
        inline operator INT64(void) const {
            return static_cast<INT64>(Float16::ToFloat32(this->value));
        }

    private:

        /** The value represented by this object. */
        UINT16 value;
    };

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_FLOAT16_H_INCLUDED */
