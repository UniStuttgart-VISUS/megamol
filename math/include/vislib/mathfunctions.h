/*
 * mathfunctions.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MATHFUNCTIONS_H_INCLUDED
#define VISLIB_MATHFUNCTIONS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <cmath>
#include <cstdlib>
#include <limits>
#include "vislib/assert.h"
#include "vislib/mathtypes.h"
#include "vislib/Pair.h"
#include "vislib/utils.h"


namespace vislib {
namespace math {

    /**
     * Answer the absolute of 'n'.
     *
     * @return The absolute of 'n'.
     */
    template<class T> inline T Abs(const T n) {
        return ::abs(n);
    }

    /**
     * Answer the absolute of 'n'.
     *
     * @return The absolute of 'n'.
     */
    template<> inline double Abs(const double n) {
        return ::fabs(n);
    }

    /**
     * Answer the absolute of 'n'.
     *
     * @return The absolute of 'n'.
     */
    template<> inline float Abs(const float n) {
        return ::fabsf(n);
    }

    /**
     * Answer the next power of two which is greater or equal to 'n'.
     *
     * This function has logarithmic runtime complexity.
     *
     * Notes: DO NOT INSTANTIATE THE TEMPLATE FOR OTHER DATA TYPES THAN INTEGRAL
     *        NUMBERS (Should not compile for floating point types because of
     *        missing bit shift operators)!
     *
     * @param n A number.
     *
     * @return The smallest power of two with ('n' <= result).
     */
    template<class T> T NextPowerOfTwo(const T n) {
        ASSERT(std::numeric_limits<T>::is_integer);
        
        T retval = static_cast<T>(1);

        while (retval < n) {
            retval <<= 1;
        }

        return retval;
    }


    /**
     * Clamp 'n' to ['minVal', 'maxVal'].
     *
     * @param n      A number.
     * @param minVal The minimum valid number.
     * @param maxVal The maximum valid number, must be greater than 'minVal'.
     *
     * @return The clamped value of 'n'.
     */
    template<class T> inline T Clamp(const T n, const T minVal, 
            const T maxVal) {
        return ((n >= minVal) ? ((n <= maxVal) ? n : maxVal) : minVal);
    }


    /**
     * Answer whether 'm' and 'n' are equal.
     *
     * @param m        A number.
     * @param n        A number.
     *
     * @return true, if 'm' and 'n' are equal, false otherwise.
     */
    template<class T> inline bool IsEqual(const T m, const T n) {
        return (m == n);
    }


    /**
     * Answer whether 'm' and 'n' are nearly equal.
     *
     * @param m       A floating point number.
     * @param n       A floating point number.
     * @param epsilon The epsilon value used for comparsion.
     *
     * @return true, if 'm' and 'n' are nearly equal, false otherwise.
     */
    inline bool IsEqual(const float m, const float n, const float epsilon) {
        return (::fabsf(m - n) < epsilon);
    }


    /**
     * Answer whether 'm' and 'n' are nearly equal. This function uses the
     * FLOAT_EPSILON constant.
     *
     * @param m       A floating point number.
     * @param n       A floating point number.
     *
     * @return true, if 'm' and 'n' are nearly equal, false otherwise.
     */
    template<> inline bool IsEqual(const float m, const float n) {
        return IsEqual(m, n, FLOAT_EPSILON);
    }


    /**
     * Answer whether 'm' and 'n' are nearly equal.
     *
     * @param m       A floating point number.
     * @param n       A floating point number.
     * @param epsilon The epsilon value used for comparsion.
     *
     * @return true, if 'm' and 'n' are nearly equal, false otherwise.
     */
    inline bool IsEqual(const double m, const double n, const double epsilon) {
        return (::fabs(m - n) < epsilon);
    }


    /**
     * Answer whether 'm' and 'n' are nearly equal. This function uses
     * the DOUBLE_EPSILON constant.
     *
     * @param m       A floating point number.
     * @param n       A floating point number.
     *
     * @return true, if 'm' and 'n' are nearly equal, false otherwise.
     */
    template<> inline bool IsEqual(const double m, const double n) {
        return IsEqual(m, n, DOUBLE_EPSILON);
    }


    /**
     * Answer whether 'n' is in the interval of 'minVal' to 'maxVal'.
     *
     * @param n          A number.
     * @param minVal     The lower border of the interval. The caller must 
     *                   ensure that 'minVal' <= 'maxVal'.
     * @param maxVal     The upper border of the interval. The caller must 
     *                   ensure that 'minVal' <= 'maxVal'.
     * @param includeMin If true, the lower border is considered part of 
     *                   the interval.
     * @param includeMax If true, the upper border is considered part of 
     *                   the interval.
     *
     * @return true if 'n' is within the interval, false otherwise.
     */
    template<class T> inline bool IsWithin(const T n, const T minVal, 
            const T maxVal, const bool includeMin = false, 
            const bool includeMax = false) {
        ASSERT(minVal <= maxVal);
        return (((minVal < n) || (includeMin && (minVal == n)))
            && ((n < maxVal) || (includeMax && (maxVal == n))));
    }


    /**
     * Answer the maximum of 'n' and 'm'.
     *
     * @param n An number.
     * @param m Another number.
     *
     * @return The maximum of 'n' and 'm'.
     */
    template<class T> inline T Max(const T n, const T m) {
        return (n > m) ? n : m;
    }


    /**
     * Answer the minimum of 'n' and 'm'.
     *
     * @param n An number.
     * @param m Another number.
     *
     * @return The minimum of 'n' and 'm'.
     */
    template<class T> inline T Min(const T n, const T m) {
        return (n < m) ? n : m;
    }


    /**
     * Compute the signum of 'n'.
     *
     * @param n A number.
     *
     * @return The signum of 'n'.
     */
    template<class T> inline T Signum(const T n) {
        return static_cast<T>((n > static_cast<T>(0)) ? 1 
            : ((n < static_cast<T>(0)) ? -1 : 0));
    }


    /**
     * Compute the square of 'n'.
     * 
     * @param n A number.
     *
     * @return The square of 'n'.
     */
    template<class T> inline T Sqr(const T n) {
        return (n * n);
    }


    /**
     * Answer the square root of 'n'.
     *
     * @param n A number.
     *
     * @return The square root of 'n'.
     */
    inline float Sqrt(const float n) {
        return ::sqrtf(n);
    }


    /**
     * Answer the square root of 'n'.
     *
     * @param n A number.
     *
     * @return The square root of 'n'.
     */
    inline double Sqrt(const double n) {
        return ::sqrt(n);
    }


    /**
     * Answer the square root of 'n'.
     *
     * @param n A number.
     *
     * @return The square root of 'n'.
     */
    inline int Sqrt(const int n) {
        return static_cast<int>(::sqrt(static_cast<double>(n)));
    }


    /*
     * There was a second implementation of 'Swap' here.
     * Instead use '::vislib::Swap' from 'vislib/utils' from base!
     * This 'using' is to preserve the name for backward compatibility.
     */
    using ::vislib::Swap;


    /**
     * Calculates the unsigned modulo value.
     * Only signed interger types can be used to instanciate this function.
     * Example:
     *   -3 % 5 = -3
     *   UMod(-3, 5) = 2
     *
     * @param left The signed dividend.
     * @param right The unsigned divisor. must not be negative.
     *
     * @return The unsigned modulo value.
     */
    template<class T> inline T UMod(const T left, const T right) {
        ASSERT(std::numeric_limits<T>::is_integer);
        ASSERT(std::numeric_limits<T>::is_signed);
        ASSERT(right >= 0);
        // return (left >= 0) ? (left % right) : ((left % right) + right);
        return (left >= 0) ? (left % right) : ((1 - ((left + 1) / right)) * right + left);
    }


    /**
     * Converts an angle from degrees to radians.
     *
     * @param angle Angle in degrees to be converted
     *
     * @return Converted angle in radians
     */
    inline AngleRad AngleDeg2Rad(const AngleDeg &angle) {
        return static_cast<AngleRad>(angle * static_cast<AngleDeg>(PI_DOUBLE) 
            / static_cast<AngleDeg>(180.0));
    }


    /**
     * Converts an angle from radians to degrees.
     *
     * @param angle Angle in radians to be converted
     *
     * @return Converted angle in degrees
     */
    inline AngleDeg AngleRad2Deg(const AngleRad &angle) {
        return static_cast<AngleDeg>(angle * static_cast<AngleRad>(180.0)
            / static_cast<AngleRad>(PI_DOUBLE));
    }


    /**
     * Compares two objects using the 'IsEqual' funtion and the '<' operator
     *
     * @param lhs The left hand side operand
     * @param rhs The right hand side operand
     *
     * @return  0 if lhs and rhs are equal,
     *         -1 if lhs < rhs,
     *          1 else.
     */
    template<class T> inline int Compare(const T& lhs, const T& rhs) {
        if (IsEqual(lhs, rhs)) return 0;
        if (lhs < rhs) return -1;
        return 1;
    }


    /**
     * Compares two pair objects based on their 'first' members
     *
     * @param lhs The left hand side operand
     * @param rhs The right hand side operand
     *
     * @return  0 if lhs and rhs are equal,
     *         -1 if lhs < rhs,
     *          1 else.
     */
    template<class T1, class T2> int ComparePairsFirst(
            const Pair<T1, T2>& lhs,
            const Pair<T1, T2>& rhs) {
        return Compare(lhs.First(), rhs.First());
    }


    /**
     * Compares two pair objects based on their 'second' members
     *
     * @param lhs The left hand side operand
     * @param rhs The right hand side operand
     *
     * @return  0 if lhs and rhs are equal,
     *         -1 if lhs < rhs,
     *          1 else.
     */
    template<class T1, class T2> int ComparePairsSecond(
            const Pair<T1, T2>& lhs,
            const Pair<T1, T2>& rhs) {
        return Compare(lhs.Second(), rhs.Second());
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MATHFUNCTIONS_H_INCLUDED */
