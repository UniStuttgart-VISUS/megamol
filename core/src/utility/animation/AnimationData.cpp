/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/animation/AnimationData.h"

#include <cmath>
#include <sstream>

#include <imgui_internal.h>

using namespace megamol::core::utility::animation;

std::ostream& megamol::core::utility::operator<<(
    std ::ostream& outs, const animation::VectorKey<animation::FloatKey>::ValueType& value) {
    std::stringstream ss;
    ss << value[0];
    for (int i = 1; i < value.size(); ++i) {
        ss << ";" << value[i];
    }
    return outs << ss.str();
}

FloatKey::ValueType FloatKey::InterpolateForTime(FloatKey first, FloatKey second, KeyTimeType time) {
    FloatKey my_first = first;
    FloatKey my_second = second;
    if (my_first.time > my_second.time) {
        my_first = second;
        my_second = first;
    }
    if (time <= my_first.time) {
        return my_first.value;
    }
    if (time >= my_second.time) {
        return my_second.value;
    }

    float t = static_cast<float>(time - my_first.time) / static_cast<float>(my_second.time - my_first.time);

    switch (first.interpolation) {
    case InterpolationType::Step:
        return my_first.value;
    case InterpolationType::Linear:
        return (1.0f - t) * my_first.value + t * my_second.value;
    case InterpolationType::Hermite: {
        auto x = CubicHermiteValue(my_first.time, my_first.out_tangent.x, my_second.in_tangent.x, my_second.time, t);
        auto err = x - time;
        int iter = 0;
        while (std::abs(err) > 0.01f && iter < 100) {
            // TODO: use the tangent to guesstimate the correction step. should converge much faster!
            t = t - 0.5f * err / static_cast<float>(my_second.time - my_first.time);
            x = CubicHermiteValue(my_first.time, my_first.out_tangent.x, my_second.in_tangent.x, my_second.time, t);
            err = x - time;
            iter++;
        }
        return CubicHermiteValue(my_first.value, my_first.out_tangent.y, my_second.in_tangent.y, my_second.value, t);
    }
    case InterpolationType::CubicBezier: {
        auto x = CubicBezierValue(my_first.time, my_first.time + my_first.out_tangent.x,
            my_second.time + my_second.in_tangent.x, my_second.time, t);
        auto err = x - time;
        int iter = 0;
        while (std::abs(err) > 0.01f && iter < 100) {
            // TODO: use the tangent to guesstimate the correction step. should converge much faster!
            t = t - 0.5f * err / static_cast<float>(my_second.time - my_first.time);
            x = CubicBezierValue(my_first.time, my_first.time + my_first.out_tangent.x,
                my_second.time + my_second.in_tangent.x, my_second.time, t);
            err = x - time;
            iter++;
        }
        return CubicBezierValue(my_first.value, my_first.value + my_first.out_tangent.y,
            my_second.value + my_second.in_tangent.y, my_second.value, t);
    }
    }
    return 0.0f;
}

ImVec2 FloatKey::InterpolateForParameter(FloatKey first, FloatKey second, float t) {
    FloatKey my_first = first;
    FloatKey my_second = second;
    if (my_first.time > my_second.time) {
        my_first = second;
        my_second = first;
    }
    if (t <= 0.0f) {
        return {static_cast<float>(my_first.time), my_first.value};
    }
    if (t >= 1.0f) {
        return {static_cast<float>(my_second.time), my_second.value};
    }

    switch (first.interpolation) {
    case InterpolationType::Step:
        return {
            (1.0f - t) * static_cast<float>(my_first.time) + t * static_cast<float>(my_second.time), my_first.value};
    case InterpolationType::Linear:
        return {(1.0f - t) * static_cast<float>(my_first.time) + t * static_cast<float>(my_second.time),
            (1.0f - t) * my_first.value + t * my_second.value};
    case InterpolationType::Hermite: {
        auto x = CubicHermiteValue(my_first.time, my_first.out_tangent.x, my_second.in_tangent.x, my_second.time, t);
        auto y = CubicHermiteValue(my_first.value, my_first.out_tangent.y, my_second.in_tangent.y, my_second.value, t);
        return {x, y};
    }
    case InterpolationType::CubicBezier: {
        auto x = CubicBezierValue(my_first.time, my_first.time + my_first.out_tangent.x,
            my_second.time + my_second.in_tangent.x, my_second.time, t);
        auto y = CubicBezierValue(my_first.value, my_first.value + my_first.out_tangent.y,
            my_second.value + my_second.in_tangent.y, my_second.value, t);
        return {x, y};
    }
    // should never come here
    case InterpolationType::SLERP:
    default:
        return {0.0f, 0.0f};
    }
}


ImVec2 FloatKey::TangentForTime(FloatKey first, FloatKey second, KeyTimeType time) {
    FloatKey my_first = first;
    FloatKey my_second = second;
    if (my_first.time > my_second.time) {
        my_first = second;
        my_second = first;
    }
    if (time <= my_first.time) {
        return my_first.in_tangent;
    }
    if (time >= my_second.time) {
        return my_second.out_tangent;
    }

    float t = static_cast<float>(time - my_first.time) / static_cast<float>(my_second.time - my_first.time);

    switch (first.interpolation) {
    case InterpolationType::Step:
    case InterpolationType::Linear:
        return TangentForParameter(first, second, t);
    case InterpolationType::Hermite: {
        auto x = CubicHermiteValue(my_first.time, my_first.out_tangent.x, my_second.in_tangent.x, my_second.time, t);
        auto err = x - time;
        int iter = 0;
        while (std::abs(err) > 0.01f && iter < 100) {
            // TODO: use the tangent to guesstimate the correction step. should converge much faster!
            t = t - 0.5f * err / static_cast<float>(my_second.time - my_first.time);
            x = CubicHermiteValue(my_first.time, my_first.out_tangent.x, my_second.in_tangent.x, my_second.time, t);
            err = x - time;
            iter++;
        }
        auto rx = CubicHermiteTangent(my_first.time, my_first.out_tangent.x, my_second.in_tangent.x, my_second.time, t);
        auto ry =
            CubicHermiteTangent(my_first.value, my_first.out_tangent.y, my_second.in_tangent.y, my_second.value, t);
        return {rx, ry};
    }
    case InterpolationType::CubicBezier: {
        auto x = CubicBezierValue(my_first.time, my_first.time + my_first.out_tangent.x,
            my_second.time + my_second.in_tangent.x, my_second.time, t);
        auto err = x - time;
        int iter = 0;
        while (std::abs(err) > 0.01f && iter < 100) {
            // TODO: use the tangent to guesstimate the correction step. should converge much faster!
            t = t - 0.5f * err / static_cast<float>(my_second.time - my_first.time);
            x = CubicBezierValue(my_first.time, my_first.time + my_first.out_tangent.x,
                my_second.time + my_second.in_tangent.x, my_second.time, t);
            err = x - time;
            iter++;
        }
        auto rx = CubicBezierTangent(my_first.time, my_first.time + my_first.out_tangent.x,
            my_second.time + my_second.in_tangent.x, my_second.time, t);
        auto ry = CubicBezierTangent(my_first.value, my_first.value + my_first.out_tangent.y,
            my_second.value + my_second.in_tangent.y, my_second.value, t);
        return {rx, ry};
    }
    }
    return ImVec2();
}


ImVec2 FloatKey::TangentForParameter(FloatKey first, FloatKey second, float t) {
    FloatKey my_first = first;
    FloatKey my_second = second;
    if (my_first.time > my_second.time) {
        my_first = second;
        my_second = first;
    }
    if (t <= 0.0f) {
        return my_first.in_tangent;
    }
    if (t >= 1.0f) {
        return my_second.out_tangent;
    }

    switch (first.interpolation) {
    case InterpolationType::Step:
        return {1.0f, 0.0f};
    case InterpolationType::Linear: {
        if (first.value == second.value) {
            // no slope
            return {1.0f, 0.0f};
        } else {
            ImVec2 r = {static_cast<float>(second.time - first.time), second.value - first.value};
            // TODO normalize to what actually
            return r / sqrtf(ImLengthSqr(r));
        }
    }
    case InterpolationType::Hermite: {
        auto x = CubicHermiteTangent(my_first.time, my_first.out_tangent.x, my_second.in_tangent.x, my_second.time, t);
        auto y =
            CubicHermiteTangent(my_first.value, my_first.out_tangent.y, my_second.in_tangent.y, my_second.value, t);
        return {x, y};
    }
    case InterpolationType::CubicBezier: {
        auto x = CubicBezierTangent(my_first.time, my_first.time + my_first.out_tangent.x,
            my_second.time + my_second.in_tangent.x, my_second.time, t);
        auto y = CubicBezierTangent(my_first.value, my_first.value + my_first.out_tangent.y,
            my_second.value + my_second.in_tangent.y, my_second.value, t);
        return {x, y};
    }
    // should never come here
    case InterpolationType::SLERP:
    default:
        return {0.0f, 0.0f};
    }
}

float FloatKey::CubicBezierValue(float value1, float value2, float value3, float value4, float t) {
    const auto invt2 = (1.0f - t) * (1.0f - t);
    const auto invt3 = invt2 * (1.0f - t);
    const auto t2 = t * t;
    const auto t3 = t2 * t;
    return invt3 * value1 + 3.0f * invt2 * t * value2 + 3.0f * (1.0f - t) * t2 * value3 + t3 * value4;
}

float FloatKey::CubicBezierTangent(float value1, float value2, float value3, float value4, float t) {
    const auto invt2 = (1.0f - t) * (1.0f - t);
    const auto t2 = t * t;
    //return 3.0f * invt2 * (value2 - value1) + 6.0f * (1.0f - t) * t * (value3 - value2) + 3.0f * t2 * (value4 - value3);
    // shortening by "degree"
    return invt2 * (value2 - value1) + 2.0f * (1.0f - t) * t * (value3 - value2) + t2 * (value4 - value3);
}

float FloatKey::CubicHermiteValue(float value1, float outTangent1, float inTangent2, float value2, float t) {
    const auto t2 = t * t;
    const auto t3 = t2 * t;
    return value1 * (2.0f * t3 - 3.0f * t2 + 1.0f) + value2 * (-2.0f * t3 + 3.0f * t2) +
           outTangent1 * (t3 - 2.0f * t2 + t) - inTangent2 * (t3 - t2);
}

float FloatKey::CubicHermiteTangent(float value1, float outTangent1, float inTangent2, float value2, float t) {
    const auto t2 = t * t;
    //return value1 * (6.0f * t2 - 6.0f * t) + value2 * (-6.0f * t2 + 6.0f * t) +
    //             outTangent1 * (3.0f * t2 - 4.0f * t + 1) - inTangent2 * (2.0f * t2 - 2.0f * t);
    // shortening by "degree"
    auto out = value1 * (6.0f * t2 - 6.0f * t) + value2 * (-6.0f * t2 + 6.0f * t) +
               outTangent1 * (3.0f * t2 - 4.0f * t + 1.0f) - inTangent2 * (3.0f * t2 - 2.0f * t);
    return out / 3.0f;
}

template<>
GenericAnimation<FloatKey>::ValueType::ValueType GenericAnimation<FloatKey>::GetValue(KeyTimeType time) const {
    if (keys.size() < 2) {
        if (keys.empty()) {
            return ValueType::ValueType();
        }
        return keys.begin()->second.value;
    }
    FloatKey before_key = keys.begin()->second, after_key = keys.begin()->second;
    bool ok = false;
    for (auto it = keys.begin(); it != keys.end(); ++it) {
        if (it->second.time == time) {
            return it->second.value;
        }
        if (it->second.time < time) {
            before_key = it->second;
        }
        if (it->second.time > time) {
            after_key = it->second;
            ok = true;
            break;
        }
    }
    if (ok) {
        return FloatKey::InterpolateForTime(before_key, after_key, time);
    } else {
        return before_key.value;
    }
}


template<>
ImVec2 GenericAnimation<FloatKey>::GetTangent(KeyTimeType time) const {
    if (keys.size() < 2) {
        if (keys.empty()) {
            return ImVec2();
        }
        return keys.begin()->second.out_tangent;
    }
    FloatKey before_key = keys.begin()->second, after_key = keys.begin()->second;
    bool ok = false;
    for (auto it = keys.begin(); it != keys.end(); ++it) {
        if (it->second.time == time) {
            return it->second.out_tangent;
        }
        if (it->second.time < time) {
            before_key = it->second;
        }
        if (it->second.time > time) {
            after_key = it->second;
            ok = true;
            break;
        }
    }
    if (ok) {
        return FloatKey::TangentForTime(before_key, after_key, time);
    } else {
        return before_key.out_tangent;
    }
}


template<>
InterpolationType GenericAnimation<FloatKey>::GetInterpolation(KeyTimeType time) const {
    InterpolationType interp = InterpolationType::Step;
    for (auto it = keys.begin(); it != keys.end(); ++it) {
        if (it->second.time == time) {
            return it->second.interpolation;
        }
        if (it->second.time < time) {
            interp = it->second.interpolation;
        }
        if (it->second.time > time) {
            break;
        }
    }
    return interp;
}


template<>
float GenericAnimation<FloatKey>::GetMinValue() const {
    if (!keys.empty()) {
        auto min = std::numeric_limits<float>::max();
        for (auto& k : keys) {
            min = std::min(min, k.second.value);
        }
        return min;
    } else {
        return 0.0f;
    }
}

template<>
float GenericAnimation<FloatKey>::GetMaxValue() const {
    if (!keys.empty()) {
        auto max = std::numeric_limits<float>::lowest();
        for (auto& k : keys) {
            max = std::max(max, k.second.value);
        }
        return max;
    } else {
        return 1.0f;
    }
}
