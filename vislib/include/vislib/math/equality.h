#pragma once

#include <limits>
#include <type_traits>

namespace vislib {
namespace math {

// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template<class T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type almost_equal(T x, T y, int ulp = 2) {
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x - y) <= std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
           // unless the result is subnormal
           || std::abs(x - y) < std::numeric_limits<T>::min();
}

template<class T>
typename std::enable_if<!std::is_floating_point<T>::value, bool>::type almost_equal(T x, T y) {
    return x == y;
}

} // namespace math
} // namespace vislib
