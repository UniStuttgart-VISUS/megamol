/*
 * thecam/math/functions.h
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
 * mathfunctions.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef THE_MATH_FUNCTIONS_H_INCLUDED
#define THE_MATH_FUNCTIONS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>

#include "mmcore/thecam/math/mathtypes.h"

#include "mmcore/thecam/utility/assert.h"
#include "mmcore/thecam/utility/force_inline.h"
#include "mmcore/thecam/utility/utils.h"


namespace megamol {
namespace core {
namespace thecam {
namespace math {

/**
 * Answer the absolute of 'n'.
 *
 * @return The absolute of 'n'.
 */
template <class T> THE_FORCE_INLINE T abs(const T n) { return std::abs(n); }


/**
 * Converts an angle from degrees to radians.
 *
 * @param angle Angle in degrees to be converted
 *
 * @return Converted angle in radians
 */
inline angle_rad_t angle_deg2rad(const angle_deg_t& angle) {
    return static_cast<angle_rad_t>(angle * pi<angle_deg_t>::value / static_cast<angle_deg_t>(180));
}


/**
 * Converts an angle from radians to degrees.
 *
 * @param angle Angle in radians to be converted
 *
 * @return Converted angle in degrees
 */
inline angle_deg_t angle_rad2deg(const angle_rad_t& angle) {
    return static_cast<angle_deg_t>(angle * static_cast<angle_rad_t>(180.0) / pi<angle_rad_t>::value);
}


/**
 * Ensure that 'inOutN' is 'minimum' or larger.
 *
 * This function performs one-sided in-place clamping of 'inOutN'.
 *
 * @param inOutN  A reference of a number to be corrected.
 * @param minimum The minimum value that 'inOutN' will assume once the
 *                function returns.
 */
template <class T> inline void at_least(T& inOutN, const T minimum) {
    if (inOutN < minimum) {
        inOutN = minimum;
    }
}


/**
 * Ensure that 'inOutN' is 'maximum' or less.
 *
 * This function performs one-sided in-place clamping of 'inOutN'.
 *
 * @param inOutN  A reference of a number to be corrected.
 * @param maximum The maximum value that 'inOutN' will assume once the
 *                function returns.
 */
template <class T> inline void at_most(T& inOutN, const T maximum) {
    if (inOutN > maximum) {
        inOutN = maximum;
    }
}


/**
 * Decrement 'value' unless it already reached 'minValid'.
 *
 * @param value    A variable to be decremented.
 * @param minValid The minimum value 'value' may decremented to in this
 *                 operation.
 *
 * @return The new value of T.
 */
template <class T> inline T decrement_until(T& value, const T minValid) {
    if (value > minValid) {
        --value;
    }
    return value;
}


/**
 * Answer the next power of two which is greater or equal to 'n'.
 *
 * This function has logarithmic runtime complexity.
 *
 * Notes: std::is_integral<T>::value must be true for the template parameter
 * T, i.e. The template must be instantiated with integral numbers.
 *
 * @param n A number.
 *
 * @return The smallest power of two with ('n' <= result).
 */
template <class T> typename std::enable_if<std::is_integral<T>::value, T>::type next_power_of_two(const T n) {
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
template <class T> inline T clamp(const T n, const T min_val, const T max_val) {
    return ((n >= min_val) ? ((n <= max_val) ? n : max_val) : min_val);
}


/**
 * Compares two objects using the 'is_equal' funtion and the '<' operator.
 *
 * @param lhs The left hand side operand
 * @param rhs The right hand side operand
 *
 * @return  0 if lhs and rhs are equal,
 *         -1 if lhs < rhs,
 *          1 else.
 */
template <class T> inline int compare(const T& lhs, const T& rhs) {
    if (is_equal(lhs, rhs)) return 0;
    if (lhs < rhs) return -1;
    return 1;
}


/**
 * Compares two pair objects based on their 'first' members.
 *
 * @param lhs The left hand side operand
 * @param rhs The right hand side operand
 *
 * @return  0 if lhs and rhs are equal,
 *         -1 if lhs < rhs,
 *          1 else.
 */
template <class T1, class T2> int compare_pairs_first(const std::pair<T1, T2>& lhs, const std::pair<T1, T2>& rhs) {
    return compare(lhs.first, rhs.first);
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
template <class T1, class T2> int compare_pairs_second(const std::pair<T1, T2>& lhs, const std::pair<T1, T2>& rhs) {
    return compare(lhs.second, rhs.second);
}


/**
 * Increment 'value' unless it already reached 'maxValid'.
 *
 * @param value    A variable to be incremented.
 * @param maxValid The maximum value 'value' may incremented to in this
 *                 operation.
 *
 * @return The new value of T.
 */
template <class T> inline T increment_until(T& value, const T maxValid) {
    if (value < maxValid) {
        ++value;
    }
    return value;
}


/**
 * Answer whether 'm' and 'n' are equal.
 *
 * @param m       A number.
 * @param n       A number.
 * @param epsilon The epsilon value used for comparsion.
 *
 * @return true, if 'm' and 'n' are equal, false otherwise.
 */
template <class T> inline bool is_equal(const T m, const T n, const T epsilon) {
    // HAZARD: epsilon is not used
    return (m == n);
}


/**
 * Answer whether 'm' and 'n' are equal.
 *
 * @param m       A number.
 * @param n       A number.
 *
 * @return true, if 'm' and 'n' are equal, false otherwise.
 */
template <class T> inline bool is_equal(const T m, const T n) { return (m == n); }


/**
 * Answer whether 'm' and 'n' are equal.
 *
 * @param m       A number.
 * @param n       A number.
 * @param epsilon The epsilon value used for comparsion.
 *
 * @return true, if 'm' and 'n' are equal, false otherwise.
 */
template <> inline bool is_equal(const float m, const float n, const float epsilon) { return ::abs(m - n) < epsilon; }


/**
 * Answer whether 'm' and 'n' are equal.
 *
 * @param m       A number.
 * @param n       A number.
 *
 * @return true, if 'm' and 'n' are equal, false otherwise.
 */
template <> inline bool is_equal(const float m, const float n) {
    return (::abs(m - n) < megamol::core::thecam::math::epsilon<float>::value);
}


/**
 * Answer whether 'm' and 'n' are equal.
 *
 * @param m       A number.
 * @param n       A number.
 * @param epsilon The epsilon value used for comparsion.
 *
 * @return true, if 'm' and 'n' are equal, false otherwise.
 */
template <> inline bool is_equal(const double m, const double n, const double epsilon) {
    return (::abs(m - n) < epsilon);
}


/**
 * Answer whether 'm' and 'n' are equal.
 *
 * @param m       A number.
 * @param n       A number.
 *
 * @return true, if 'm' and 'n' are equal, false otherwise.
 */
template <> inline bool is_equal(const double m, const double n) {
    return (::abs(m - n) < megamol::core::thecam::math::epsilon<double>::value);
}


/**
 * Answer whether 'n' is a power of two.
 *
 * @param n A number.
 *
 * @return true, if 'm' is a power of two.
 */
template <class T>
typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value, bool>::type is_power_of_two(
    const T n) {
    // Implementation like http://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
    return ((n != 0) && ((n & (~n + 1)) == n));
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
template <class T>
inline bool is_within(
    const T n, const T min_val, const T max_val, const bool include_min = false, const bool include_max = false) {
    THE_ASSERT(min_val <= max_val);
    return (((min_val < n) || (include_min && (min_val == n))) && ((n < max_val) || (include_max && (max_val == n))));
}


/**
 * Answer the possible maximum value of type 'T'.
 *
 * @return std::numeric_limits<T>::max().
 */
template <class T> inline T maximum(void) {
#ifdef _MSC_VER
#    pragma push_macro("max")
#    undef max
#endif /* _MSC_VER */
    return std::numeric_limits<T>::max();
#ifdef _MSC_VER
#    pragma pop_macro("max")
#endif /* _MSC_VER */
}


/**
 * Answer the possible maximum the variable 'n' can assume as defined by its
 * type.
 *
 * @param n A number.
 *
 * @return std::numeric_limits<T>::max().
 */
template <class T> inline T maximum(const T n) { return megamol::core::thecam::math::maximum<T>(); }

/**
 * Answer the maximum of 'n' and 'm'.
 *
 * @param n A number.
 * @param m Another number.
 *
 * @return The maximum of 'n' and 'm'.
 */
template <class T> inline T maximum(const T n, const T m) { return (n > m) ? n : m; }


/**
 * Answer the possible minimum value of type 'T'.
 *
 * @return std::numeric_limits<T>::min().
 */
template <class T> inline T minimum(void) {
#ifdef _MSC_VER
#    pragma push_macro("min")
#    undef min
#endif /* _MSC_VER */
    return std::numeric_limits<T>::min();
#ifdef _MSC_VER
#    pragma pop_macro("min")
#endif /* _MSC_VER */
}

/**
 * Answer the possible minimum the variable 'n' can assume as defined by its
 * type.
 *
 * @param n A number.
 *
 * @return std::numeric_limits<T>::min().
 */
template <class T> inline T minimum(const T n) { return megamol::core::thecam::math::minimum<T>(); }


/**
 * Answer the minimum of 'n' and 'm'.
 *
 * @param n A number.
 * @param m Another number.
 *
 * @return The minimum of 'n' and 'm'.
 */
template <class T> inline T minimum(const T n, const T m) { return (n < m) ? n : m; }


/**
 * Compute the 'E'th power of 'n'.
 *
 * @param n A number
 *
 * @return 'n' ^ 'E'.
 */
template <size_t E, class T> inline T power(const T n) {
    auto retval = static_cast<T>(1);
    for (size_t i = 0; i < E; ++i) {
        retval *= n;
    }
    return retval;
}


/**
 * Answer the minimum and maximum value in an iterator range.
 *
 * @tparam I The type of the iterator over a range of numeric values.
 *
 * @param begin The begin of the enumerated range.
 * @param end   The end of the enumerated range.
 *
 * @return The first element of the returned pair is the minimum value found
 *         in [begin, end[, the second one is the maximum value.
 */
template <class I> inline std::pair<typename I::value_type, typename I::value_type> range(I begin, I end) {
    auto retval =
        std::make_pair(maximum<typename I::value_type>(), std::numeric_limits<typename I::value_type>::lowest());

    for (auto it = begin; it != end; ++it) {
        if (*it < retval.first) {
            retval.first = *it;
        }
        if (*it > retval.second) {
            retval.second = *it;
        }
    }

    return retval;
}


/**
 * Compute the signum of 'n'.
 *
 * @param n A number.
 *
 * @return The signum of 'n'.
 */
template <class T> inline T signum(const T n) {
    return static_cast<T>((n > static_cast<T>(0)) ? 1 : ((n < static_cast<T>(0)) ? -1 : 0));
}


/**
 * Compute the square of 'n'.
 *
 * @param n A number.
 *
 * @return The square of 'n'.
 */
template <class T> inline T sqr(const T n) { return (n * n); }


/**
 * Answer the square root of 'n'.
 *
 * @param n A number.
 *
 * @return The square root of 'n'.
 */
inline float sqrt(const float n) { return ::sqrtf(n); }


/**
 * Answer the square root of 'n'.
 *
 * @param n A number.
 *
 * @return The square root of 'n'.
 */
inline double sqrt(const double n) { return ::sqrt(n); }


/**
 * Answer the square root of 'n'.
 *
 * @param n A number.
 *
 * @return The square root of 'n'.
 */
inline int sqrt(const int n) { return static_cast<int>(::sqrt(static_cast<double>(n))); }


/**
 * Calculates the unsigned modulo value.
 *
 * Only signed integer types can be used to instantiate this function.
 * Example:
 *   -3 % 5 = -3
 *   megamol::core::thecam::math::umod(-3, 5) = 2
 *
 * @param left  The signed dividend.
 * @param right The unsigned divisor. must not be negative.
 *
 * @return The unsigned modulo value.
 */
template <class T>
inline typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type umod(
    const T left, const T right) {
    THE_ASSERT(right >= 0);
    // return (left >= 0) ? (left % right) : ((left % right) + right);
    return (left >= 0) ? (left % right) : ((1 - ((left + 1) / right)) * right + left);
}


/**
 * Computes 'sqrt(a * a + b * b)' without destructive underflow or
 * overflow.
 *
 * @param a The first operand
 * @param b The second operand
 *
 * @return The result
 */
template <class T>
inline T pythag(const T& a, const T& b, const T& epsilon = ::megamol::core::thecam::math::epsilon<T>::value) {
    T absa = ::megamol::core::thecam::math::abs(a);
    T absb = ::megamol::core::thecam::math::abs(b);
    if (absa > absb) {
        absb /= absa;
        absb *= absb;
        return absa * megamol::core::thecam::math::sqrt(static_cast<T>(1) + absb);
    }
    if (megamol::core::thecam::math::is_equal(absb, static_cast<T>(0), epsilon)) {
        return static_cast<T>(0);
    }
    absa /= absb;
    absa *= absa;
    return absb * megamol::core::thecam::math::sqrt(static_cast<T>(1) + absa);
}


} /* end namespace math */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_MATH_FUNCTIONS_H_INCLUDED */
