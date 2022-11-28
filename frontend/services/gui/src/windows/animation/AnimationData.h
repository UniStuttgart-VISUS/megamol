/*
 * AnimationEditor.h
 *
 * Copyright (C) 2022 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "gui_utils.h"
#include "mmcore/MegaMolGraph.h"
#include <map>


namespace megamol {
namespace gui {
namespace animation {

using KeyTimeType = int32_t;

enum class InterpolationType : int32_t { Step = 0, Linear = 1, Hermite = 2, CubicBezier = 3 };

struct FloatKey {
    using ValueType = float;
    KeyTimeType time;
    ValueType value;
    InterpolationType interpolation = InterpolationType::Linear;
    bool tangents_linked = true;
    ImVec2 in_tangent{-1.0f, 0.0f};
    ImVec2 out_tangent{1.0f, 0.0f};

    // this is expensive (accurately hit time first!)...
    static float Interpolate(FloatKey first, FloatKey second, KeyTimeType time);
    // ... and that is only good for drawing (x will not sit on the time grid)
    static ImVec2 Interpolate(FloatKey first, FloatKey second, float t);
};

struct StringKey {
    using ValueType = std::string;
    KeyTimeType time;
    ValueType value;
};

template<class KeyType>
class GenericAnimation {
public:
    using KeyMap = std::map<KeyTimeType, KeyType>;
    using ValueType = KeyType;

    GenericAnimation(std::string ParamName) : param_name(ParamName) {}
    void AddKey(KeyType k) {
        keys[k.time] = k;
    }
    void DeleteKey(KeyTimeType time) {
        keys.erase(time);
    }
    bool HasKey(KeyTimeType time) {
        return keys.find(time) != keys.end();
    }

    typename KeyMap::iterator begin() {
        return keys.begin();
    }
    typename KeyMap::iterator end() {
        return keys.end();
    }

    const std::string& GetName() const {
        return param_name;
    }
    typename KeyMap::size_type GetSize() const {
        return keys.size();
    }
    ValueType& operator[](KeyTimeType k) {
        return keys[k];
    }
    const ValueType& operator[](KeyTimeType k) const {
        return keys.at(k);
    }
    InterpolationType GetInterpolation(KeyTimeType time) const {
        // standard interpolation is step
        return InterpolationType::Step;
    }

    KeyTimeType GetStartTime() const;
    KeyTimeType GetEndTime() const;
    KeyTimeType GetLength() const;

    // these two are a bit special
    float GetMinValue() const {
        return -1.0f;
    }
    float GetMaxValue() const {
        return 1.0f;
    }


    typename ValueType::ValueType GetValue(KeyTimeType time) const;
    std::vector<KeyTimeType> GetAllKeys() const;

private:
    KeyMap keys;
    std::string param_name;
};

template<class KeyType>
KeyTimeType GenericAnimation<KeyType>::GetStartTime() const {
    if (!keys.empty()) {
        return keys.begin()->second.time;
    } else {
        return 0;
    }
}

template<class KeyType>
KeyTimeType GenericAnimation<KeyType>::GetEndTime() const {
    if (!keys.empty()) {
        return keys.rbegin()->second.time;
    } else {
        return 1;
    }
}

template<class KeyType>
KeyTimeType GenericAnimation<KeyType>::GetLength() const {
    if (!keys.empty()) {
        return keys.rbegin()->second.time - keys.begin()->second.time;
    } else {
        return 1;
    }
}

template<class KeyType>
typename KeyType::ValueType GenericAnimation<KeyType>::GetValue(KeyTimeType time) const {
    if (keys.size() < 2) {
        if (keys.empty()) {
            return ValueType::ValueType();
        }
        return keys.begin()->second.value;
    }
    auto before_key = keys.begin()->second, after_key = keys.begin()->second;
    for (auto it = keys.begin(); it != keys.end(); ++it) {
        if (it->second.time == time) {
            return it->second.value;
        }
        if (it->second.time < time) {
            before_key = it->second;
        }
        if (it->second.time > time) {
            after_key = it->second;
            break;
        }
    }
    // standard interpolation is step.
    return before_key.value;
}

template<class KeyType>
std::vector<KeyTimeType> GenericAnimation<KeyType>::GetAllKeys() const {
    std::vector<KeyTimeType> the_keys;
    the_keys.reserve(keys.size());
    std::transform(
        keys.begin(), keys.end(), std::back_inserter(the_keys), [](const KeyMap::value_type& v) { return v.first; });
    return the_keys;
}

using StringAnimation = GenericAnimation<StringKey>;
using FloatAnimation = GenericAnimation<FloatKey>;

// floats can actually interpolate!
template<>
float GenericAnimation<FloatKey>::GetValue(KeyTimeType time) const;
template<>
InterpolationType GenericAnimation<FloatKey>::GetInterpolation(KeyTimeType time) const;

// and tell about their extents
template<>
float GenericAnimation<FloatKey>::GetMinValue() const;
template<>
float GenericAnimation<FloatKey>::GetMaxValue() const;

} // namespace animation
} // namespace gui
} // namespace megamol
