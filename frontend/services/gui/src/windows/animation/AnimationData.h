/*
 * AnimationEditor.h
 *
 * Copyright (C) 2022 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "gui_utils.h"
#include "mmcore/MegaMolGraph.h"

#include "glm/glm.hpp"
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
    static ValueType Interpolate(FloatKey first, FloatKey second, KeyTimeType time);
    // ... and that is only good for drawing (x will not sit on the time grid)
    static ImVec2 Interpolate(FloatKey first, FloatKey second, float t);
};

template<class C>
struct VectorKey {
    using ValueType = std::vector<typename C::ValueType>;
    std::vector<C> nestedData;

    // this is expensive (accurately hit time first!)...
    static ValueType Interpolate(VectorKey first, VectorKey second, KeyTimeType time);
    // ... and that is only good for drawing (x will not sit on the time grid)
    static std::vector<ImVec2> Interpolate(VectorKey first, VectorKey second, float t);
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

    void FixSorting() {
        for (auto& k : keys) {
            if (k.first != k.second.time) {
                auto wrong = keys.extract(k.first);
                wrong.key() = k.second.time;
                keys.insert(std::move(wrong));
            }
        }
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

protected:
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
    if (keys.size() > 1) {
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
GenericAnimation<FloatKey>::ValueType::ValueType GenericAnimation<FloatKey>::GetValue(KeyTimeType time) const;
template<>
InterpolationType GenericAnimation<FloatKey>::GetInterpolation(KeyTimeType time) const;

// and tell about their extents
template<>
float GenericAnimation<FloatKey>::GetMinValue() const;
template<>
float GenericAnimation<FloatKey>::GetMaxValue() const;

// same goes for vec keys, plus some more specialization
template<class C>
class VectorAnimation : public GenericAnimation<VectorKey<C>> {
public:
    using KeyType = VectorKey<C>;

    VectorAnimation(std::string ParamName) : GenericAnimation<KeyType>(ParamName) {}

    void AddKey(KeyType k) {
        this->keys[k.nestedData[0].time] = k;
        this->vec_length = k.nestedData.size();
    }

    typename KeyType::ValueType Interpolate(C first, C second, KeyTimeType time) {
        typename KeyType::ValueType ret;
        ret.resize(vec_length);
        for (int i = 0; i < vec_length; ++i) {
            ret[i] = C::Interpolate(first.nestedData[i], second.nestedData[i], time);
        }
        return ret;
    }

    std::vector<ImVec2> Interpolate(C first, C second, float t) {
        std::vector<ImVec2> ret;
        ret.resize(vec_length);
        for (int i = 0; i < vec_length; ++i) {
            ret[i] = C::Interpolate(first.nestedData[i], second.nestedData[i], t);
        }
        return ret;
    }

    typename KeyType::ValueType GetValue(KeyTimeType time) const {
        KeyType::ValueType ret;
        ret.resize(vec_length);
        if (this->keys.size() < 2) {
            if (this->keys.empty()) {
                return ret;
            }
            for (int i = 0; i < vec_length; ++i) {
                ret[i] = this->keys.begin()->second.nestedData[i].value;
            }
            return ret;
        }
        auto before_key = this->keys.begin()->second, after_key = this->keys.begin()->second;
        bool ok = false;
        for (auto it = this->keys.begin(); it != this->keys.end(); ++it) {
            if (it->second.nestedData[0].time == time) {
                for (int i = 0; i < vec_length; ++i) {
                    ret[i] = it->second.nestedData[i].value;
                }
                return ret;
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
            for (int i = 0; i < vec_length; ++i) {
                ret[i] = C::Interpolate(before_key.nestedData[i], after_key.nestedData[i], time);
            }
            return ret;
        } else {
            for (int i = 0; i < vec_length; ++i) {
                ret[i] = before_key.nestedData[i].value;
            }
            return ret;
        }
    }

    KeyTimeType GetStartTime() const {
        if (!this->keys.empty()) {
            return this->keys.begin()->second.nestedData[0].time;
        } else {
            return 0;
        }
    }

    KeyTimeType GetEndTime() const {
        if (!this->keys.empty()) {
            return this->keys.rbegin()->second.nestedData[0].time;
        } else {
            return 1;
        }
    }

    KeyTimeType GetLength() const {
        if (this->keys.size() > 1) {
            return this->keys.rbegin()->second.nestedData[0].time - this->keys.begin()->second.nestedData[0].time;
        } else {
            return 1;
        }
    }

    typename C::ValueType GetMinValue() const {
        if (!this->keys.empty()) {
            auto min = std::numeric_limits<float>::max();
            for (auto& k : this->keys) {
                for (int i = 0; i < vec_length; ++i) {
                    min = std::min(min, k.second.nestedData[i].value);
                }
            }
            return min;
        } else {
            return 0.0f;
        }
    }

    typename C::ValueType GetMaxValue() const {
        if (!this->keys.empty()) {
            auto max = std::numeric_limits<float>::lowest();
            for (auto& k : this->keys) {
                for (int i = 0; i < vec_length; ++i) {
                    max = std::max(max, k.second.nestedData[i].value);
                }
            }
            return max;
        } else {
            return 1.0f;
        }
    }

    void FixSorting() {
        for (auto& k : this->keys) {
            if (k.first != k.second.nestedData[0].time) {
                auto wrong = this->keys.extract(k.first);
                wrong.key() = k.second.nestedData[0].time;
                this->keys.insert(std::move(wrong));
            }
        }
    }

    int32_t VectorLength() const {
        return vec_length;
    }
private:
    int32_t vec_length = 4;
};

using FloatVectorAnimation = VectorAnimation<FloatKey>;

} // namespace animation

std::ostream& operator<<(std::ostream& outs, const animation::VectorKey<animation::FloatKey>::ValueType& value);

} // namespace gui
} // namespace megamol
