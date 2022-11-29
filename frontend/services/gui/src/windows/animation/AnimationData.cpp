#include "AnimationData.h"


using namespace megamol::gui::animation;

float FloatKey::Interpolate(FloatKey first, FloatKey second, KeyTimeType time) {
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
        float t2, t3;
        auto recompute_ts = [&]() {
            t2 = t * t;
            t3 = t2 * t;
        };
        recompute_ts();
        auto x = my_first.time * (2.0f * t3 - 3.0f * t2 + 1.0f) + my_second.time * (-2.0f * t3 + 3.0f * t2) +
                 my_first.out_tangent.x * (t3 - 2.0f * t2 + t) - my_second.in_tangent.x * (t3 - t2);
        auto err = x - time;
        int iter = 0;
        while (std::abs(err) > 0.01f && iter < 100) {
            // TODO: use the tangent to guesstimate the correction step. should converge much faster!
            t = t - 0.5 * err / static_cast<float>(my_second.time - my_first.time);
            recompute_ts();
            x = my_first.time * (2.0f * t3 - 3.0f * t2 + 1.0f) + my_second.time * (-2.0f * t3 + 3.0f * t2) +
                my_first.out_tangent.x * (t3 - 2.0f * t2 + t) - my_second.in_tangent.x * (t3 - t2);
            err = x - time;
            iter++;
        }
        return my_first.value * (2.0f * t3 - 3.0f * t2 + 1.0f) + my_second.value * (-2.0f * t3 + 3.0f * t2) +
               my_first.out_tangent.y * (t3 - 2.0f * t2 + t) - my_second.in_tangent.y * (t3 - t2);
    }
    case InterpolationType::CubicBezier: {
        float t2, t3, invt2, invt3;
        auto recompute_ts = [&]() {
            t2 = t * t;
            t3 = t2 * t;
            invt2 = (1.0f - t) * (1.0f - t);
            invt3 = (1.0f - t) * (1.0f - t) * (1.0f - t);
        };
        recompute_ts();
        auto p1 = ImVec2(my_first.time, my_first.value);
        auto p4 = ImVec2(my_second.time, my_second.value);
        auto p2 = p1 + my_first.out_tangent;
        auto p3 = p4 + my_second.in_tangent;
        auto x = invt3 * p1.x + 3.0f * invt2 * t * p2.x + 3.0f * (1.0f - t) * t2 * p3.x + t3 * p4.x;
        auto err = x - time;
        int iter = 0;
        while (std::abs(err) > 0.01f && iter < 100) {
            // TODO: use the tangent to guesstimate the correction step. should converge much faster!
            t = t - 0.5 * err / static_cast<float>(my_second.time - my_first.time);
            recompute_ts();
            x = invt3 * p1.x + 3.0f * invt2 * t * p2.x + 3.0f * (1.0f - t) * t2 * p3.x + t3 * p4.x;
            err = x - time;
            iter++;
        }
        return invt3 * my_first.value + 3.0f * invt2 * t * p2.y + 3.0f * (1.0f - t) * t2 * p3.y + t3 * my_second.value;
    }
    }
    return 0.0f;
}

ImVec2 FloatKey::Interpolate(FloatKey first, FloatKey second, float t) {
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
        const auto t2 = t * t, t3 = t2 * t;
        auto x = my_first.time * (2.0f * t3 - 3.0f * t2 + 1.0f) + my_second.time * (-2.0f * t3 + 3.0f * t2) +
                 my_first.out_tangent.x * (t3 - 2.0f * t2 + t) - my_second.in_tangent.x * (t3 - t2);
        auto y = my_first.value * (2.0f * t3 - 3.0f * t2 + 1.0f) + my_second.value * (-2.0f * t3 + 3.0f * t2) +
                 my_first.out_tangent.y * (t3 - 2.0f * t2 + t) - my_second.in_tangent.y * (t3 - t2);
        return {x, y};
    }
    case InterpolationType::CubicBezier: {
        const auto t2 = t * t, t3 = t2 * t;
        const auto invt2 = (1.0f - t) * (1.0f - t), invt3 = (1.0f - t) * (1.0f - t) * (1.0f - t);
        auto p1 = ImVec2(my_first.time, my_first.value);
        auto p4 = ImVec2(my_second.time, my_second.value);
        auto p2 = p1 + my_first.out_tangent;
        auto p3 = p4 + my_second.in_tangent;
        auto x = invt3 * p1.x + 3.0f * invt2 * t * p2.x + 3.0f * (1.0f - t) * t2 * p3.x + t3 * p4.x;
        auto y = invt3 * p1.y + 3.0f * invt2 * t * p2.y + 3.0f * (1.0f - t) * t2 * p3.y + t3 * p4.y;
        return {x, y};
    }
    default:
        return {0.0f, 0.0f};
    }
}


template<>
void GenericAnimation<Vec3Key>::AddKey(Vec3Key k) {
    keys[k.nestedData[0].time] = k;
}


glm::vec3 Vec3Key::Interpolate(Vec3Key first, Vec3Key second, KeyTimeType time) {
    glm::vec3 ret;
    ret.x = FloatKey::Interpolate(first.nestedData[0], second.nestedData[0], time);
    ret.y = FloatKey::Interpolate(first.nestedData[1], second.nestedData[1], time);
    ret.z = FloatKey::Interpolate(first.nestedData[2], second.nestedData[2], time);
    return ret;
}


std::array<ImVec2, 3> Vec3Key::Interpolate(Vec3Key first, Vec3Key second, float t) {
    std::array<ImVec2, 3> ret;
    ret[0] = FloatKey::Interpolate(first.nestedData[0], second.nestedData[0], t);
    ret[1] = FloatKey::Interpolate(first.nestedData[1], second.nestedData[1], t);
    ret[2] = FloatKey::Interpolate(first.nestedData[2], second.nestedData[2], t);
    return ret;
}


std::ostream& megamol::gui::operator<<(std::ostream& outs, const Vec3Key::ValueType& value) {
    return outs << value[0] << ";" << value[1] << ";" << value[2];
}

template<>
GenericAnimation<Vec3Key>::ValueType::ValueType GenericAnimation<Vec3Key>::GetValue(KeyTimeType time) const {
    if (keys.size() < 2) {
        if (keys.empty()) {
            return ValueType::ValueType();
        }
        return {keys.begin()->second.nestedData[0].value, keys.begin()->second.nestedData[1].value,
            keys.begin()->second.nestedData[2].value};
    }
    Vec3Key before_key = keys.begin()->second, after_key = keys.begin()->second;
    bool ok = false;
    for (auto it = keys.begin(); it != keys.end(); ++it) {
        if (it->second.nestedData[0].time == time) {
            return {it->second.nestedData[0].value, it->second.nestedData[1].value, it->second.nestedData[2].value};
        }
        if (it->second.nestedData[0].time < time) {
            before_key = it->second;
        }
        if (it->second.nestedData[0].time > time) {
            after_key = it->second;
            ok = true;
            break;
        }
    }
    if (ok) {
        return {FloatKey::Interpolate(before_key.nestedData[0], after_key.nestedData[0], time),
            FloatKey::Interpolate(before_key.nestedData[1], after_key.nestedData[1], time),
            FloatKey::Interpolate(before_key.nestedData[2], after_key.nestedData[2], time)};
    } else {
        return {before_key.nestedData[0].value, before_key.nestedData[1].value, before_key.nestedData[2].value};
    }
}


template<>
InterpolationType GenericAnimation<Vec3Key>::GetInterpolation(KeyTimeType time) const {
    InterpolationType interp = InterpolationType::Step;
    for (auto it = keys.begin(); it != keys.end(); ++it) {
        if (it->second.nestedData[0].time == time) {
            return it->second.nestedData[0].interpolation;
        }
        if (it->second.nestedData[0].time < time) {
            interp = it->second.nestedData[0].interpolation;
        }
        if (it->second.nestedData[0].time > time) {
            break;
        }
    }
    return interp;
}


template<>
KeyTimeType GenericAnimation<Vec3Key>::GetStartTime() const {
    if (!keys.empty()) {
        return keys.begin()->second.nestedData[0].time;
    } else {
        return 0;
    }
}


template<>
KeyTimeType GenericAnimation<Vec3Key>::GetEndTime() const {
    if (!keys.empty()) {
        return keys.rbegin()->second.nestedData[0].time;
    } else {
        return 1;
    }
}


template<>
KeyTimeType GenericAnimation<Vec3Key>::GetLength() const {
    if (!keys.empty()) {
        return keys.rbegin()->second.nestedData[0].time - keys.begin()->second.nestedData[0].time;
    } else {
        return 1;
    }
}


template<>
float GenericAnimation<Vec3Key>::GetMinValue() const {
    if (!keys.empty()) {
        auto min = std::numeric_limits<float>::max();
        for (auto& k : keys) {
            for (int i = 0; i < 3; ++i) {
                min = std::min(min, k.second.nestedData[i].value);
            }
        }
        return min;
    } else {
        return 0.0f;
    }
}


template<>
float GenericAnimation<Vec3Key>::GetMaxValue() const {
    if (!keys.empty()) {
        auto max = std::numeric_limits<float>::lowest();
        for (auto& k : keys) {
            for (int i = 0; i < 3; ++i) {
                max = std::max(max, k.second.nestedData[i].value);
            }
        }
        return max;
    } else {
        return 1.0f;
    }
}


template<>
void GenericAnimation<Vec3Key>::FixSorting() {
    for (auto& k : keys) {
        if (k.first != k.second.nestedData[0].time) {
            auto wrong = keys.extract(k.first);
            wrong.key() = k.second.nestedData[0].time;
            keys.insert(std::move(wrong));
        }
    }
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
        return FloatKey::Interpolate(before_key, after_key, time);
    } else {
        return before_key.value;
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
