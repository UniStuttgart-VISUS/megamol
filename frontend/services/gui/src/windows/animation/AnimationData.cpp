#include "AnimationData.h"



using namespace megamol::gui::animation;

float Key::Interpolate(Key first, Key second, KeyTimeType time) {
    Key my_first = first;
    Key my_second = second;
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

ImVec2 Key::Interpolate(Key first, Key second, float t) {
    Key my_first = first;
    Key my_second = second;
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


void FloatAnimation::AddKey(Key k) {
    keys[k.time] = k;
}


void FloatAnimation::DeleteKey(KeyTimeType time) {
    keys.erase(time);
}


FloatAnimation::KeyMap::iterator FloatAnimation::begin() {
    return keys.begin();
}


FloatAnimation::KeyMap::iterator FloatAnimation::end() {
    return keys.end();
}


FloatAnimation::ValueType FloatAnimation::GetValue(KeyTimeType time) const {
    if (keys.size() < 2)
        return 0.0f;
    Key before_key = keys.begin()->second, after_key = keys.begin()->second;
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
        return Key::Interpolate(before_key, after_key, time);
    } else {
        return 0.0f;
    }
}


const std::string& FloatAnimation::GetName() const {
    return param_name;
}


KeyTimeType FloatAnimation::GetStartTime() const {
    if (!keys.empty()) {
        return keys.begin()->second.time;
    } else {
        return 0;
    }
}


KeyTimeType FloatAnimation::GetEndTime() const {
    if (!keys.empty()) {
        return keys.rbegin()->second.time;
    } else {
        return 1;
    }
}

KeyTimeType FloatAnimation::GetLength() const {
    if (!keys.empty()) {
        return keys.rbegin()->second.time - keys.begin()->second.time;
    } else {
        return 1;
    }
}


FloatAnimation::ValueType FloatAnimation::GetMinValue() const {
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


FloatAnimation::ValueType FloatAnimation::GetMaxValue() const {
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
