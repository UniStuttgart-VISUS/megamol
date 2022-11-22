/*
 * AnimationEditor.h
 *
 * Copyright (C) 2022 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/MegaMolGraph.h"
#include "gui_utils.h"
#include <map>


namespace megamol {
namespace gui {
namespace animation {

using KeyTimeType = int32_t;

enum class InterpolationType : int32_t { Step = 0, Linear = 1, Hermite = 2, CubicBezier = 3 };

struct Key {
    KeyTimeType time;
    float value;
    InterpolationType interpolation = InterpolationType::Linear;
    bool tangents_linked = true;
    ImVec2 in_tangent{-1.0f, 0.0f};
    ImVec2 out_tangent{1.0f, 0.0f};

    // this is expensive (accurately hit time first!)...
    static float Interpolate(Key first, Key second, KeyTimeType time);
    // ... and that is only good for drawing (x will not sit on the time grid)
    static ImVec2 Interpolate(Key first, Key second, float t);
};

class FloatAnimation {
public:
    using KeyMap = std::map<KeyTimeType, Key>;
    using ValueType = float;

    FloatAnimation(std::string ParamName) : param_name(ParamName) {}
    void AddKey(Key k);
    void DeleteKey(KeyTimeType time);

    KeyMap::iterator begin();
    KeyMap::iterator end();

    ValueType GetValue(KeyTimeType time) const;
    const std::string& GetName() const;
    KeyTimeType GetStartTime() const;
    KeyTimeType GetEndTime() const;
    KeyTimeType GetLength() const;
    ValueType GetMinValue() const;
    ValueType GetMaxValue() const;
    KeyMap::size_type GetSize() const {
        return keys.size();
    }
    bool HasKey(KeyTimeType k) {
        return keys.find(k) != keys.end();
    }
    Key& operator[](KeyTimeType k) {
        return keys[k];
    }
    const Key& operator[](KeyTimeType k) const {
        return keys.at(k);
    }
    std::vector<KeyTimeType> GetAllKeys() const {
        std::vector<KeyTimeType> the_keys;
        ;
        the_keys.reserve(keys.size());
        std::transform(keys.begin(), keys.end(), std::back_inserter(the_keys),
            [](const KeyMap::value_type& v) { return v.first; });
        return the_keys;
    }

private:
    KeyMap keys;
    std::string param_name;
};

} // namespace animation
} // namespace gui
} // namespace megamol
