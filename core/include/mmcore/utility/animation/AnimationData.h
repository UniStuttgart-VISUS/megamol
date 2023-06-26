/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <map>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>

#include "mmcore/MegaMolGraph.h"

namespace megamol::core::utility::animation {

using KeyTimeType = int32_t;

enum class InterpolationType : int32_t { Step = 0, Linear = 1, Hermite = 2, CubicBezier = 3, SLERP = 4 };

struct FloatKey {
    using ValueType = float;
    KeyTimeType time;
    ValueType value;
    InterpolationType interpolation = InterpolationType::Linear;
    bool tangents_linked = true;
    ImVec2 in_tangent{-1.0f, 0.0f};
    ImVec2 out_tangent{1.0f, 0.0f};

    // this is expensive (accurately hit time first!)...
    static ValueType InterpolateForTime(FloatKey first, FloatKey second, KeyTimeType time);
    static ImVec2 TangentForTime(FloatKey first, FloatKey second, KeyTimeType time);
    // ... and that is only good for drawing (x will not sit on the time grid)
    static ImVec2 InterpolateForParameter(FloatKey first, FloatKey second, float t);
    static ImVec2 TangentForParameter(FloatKey first, FloatKey second, float t);

    static float CubicBezierValue(float value1, float value2, float value3, float value4, float t);
    static float CubicBezierTangent(float value1, float value2, float value3, float value4, float t);

    static float CubicHermiteValue(float value1, float outTangent1, float inTangent2, float value2, float t);
    static float CubicHermiteTangent(float value1, float outTangent1, float inTangent2, float value2, float t);
    //static float FindTForValue()
};

template<class C>
struct VectorKey {
    using ValueType = std::vector<typename C::ValueType>;
    std::vector<C> nestedData;

    // this is expensive (accurately hit time first!)...
    static ValueType InterpolateForTime(VectorKey first, VectorKey second, KeyTimeType time);
    // ... and that is only good for drawing (x will not sit on the time grid)
    static std::vector<ImVec2> InterpolateForParameter(VectorKey first, VectorKey second, float t);
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
    ImVec2 GetTangent(KeyTimeType time) const;
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
            return typename KeyType::ValueType();
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
ImVec2 GenericAnimation<KeyType>::GetTangent(KeyTimeType time) const {
    // generally speaking, useless
    return ImVec2{1.0f, 0.0f};
}

template<class KeyType>
std::vector<KeyTimeType> GenericAnimation<KeyType>::GetAllKeys() const {
    std::vector<KeyTimeType> the_keys;
    the_keys.reserve(keys.size());
    std::transform(keys.begin(), keys.end(), std::back_inserter(the_keys),
        [](const typename KeyMap::value_type& v) { return v.first; });
    return the_keys;
}

using StringAnimation = GenericAnimation<StringKey>;
using FloatAnimation = GenericAnimation<FloatKey>;

// floats can actually interpolate!
template<>
GenericAnimation<FloatKey>::ValueType::ValueType GenericAnimation<FloatKey>::GetValue(KeyTimeType time) const;
template<>
ImVec2 GenericAnimation<FloatKey>::GetTangent(KeyTimeType time) const;
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
    using ValueType = std::vector<typename C::ValueType>;

    VectorAnimation(std::string ParamName) : GenericAnimation<KeyType>(ParamName) {}

    void AddKey(KeyType k) {
        this->keys[k.nestedData[0].time] = k;
        this->vec_length = k.nestedData.size();
    }

    ValueType InterpolateForTime(KeyType first, KeyType second, KeyTimeType time) const {
        ValueType ret;
        ret.resize(vec_length);
        if (first.nestedData[0].interpolation == InterpolationType::SLERP && vec_length == 4) {
            KeyType my_first = first;
            KeyType my_second = second;
            if (my_first.nestedData[0].time > my_second.nestedData[0].time) {
                my_first = second;
                my_second = first;
            }
            if (time <= my_first.nestedData[0].time) {
                return GetValue(my_first.nestedData[0].time);
            }
            if (time >= my_second.nestedData[0].time) {
                return GetValue(my_second.nestedData[0].time);
            }

            float t = static_cast<float>(time - my_first.nestedData[0].time) /
                      static_cast<float>(my_second.nestedData[0].time - my_first.nestedData[0].time);
            glm::quat q1(first.nestedData[3].value, first.nestedData[0].value, first.nestedData[1].value,
                first.nestedData[2].value);
            glm::quat q2(second.nestedData[3].value, second.nestedData[0].value, second.nestedData[1].value,
                second.nestedData[2].value);
            auto q = glm::slerp(q1, q2, t);
            ret[0] = q.x;
            ret[1] = q.y;
            ret[2] = q.z;
            ret[3] = q.w;
            return ret;
        } else {
            for (int i = 0; i < vec_length; ++i) {
                ret[i] = C::InterpolateForTime(first.nestedData[i], second.nestedData[i], time);
            }
        }
        return ret;
    }

    std::vector<ImVec2> InterpolateForParameter(C first, C second, float t) const {
        std::vector<ImVec2> ret;
        ret.resize(vec_length);
        if (first.nestedData[0].interpolation == InterpolationType::SLERP) {
            // TODO
        } else {
            for (int i = 0; i < vec_length; ++i) {
                ret[i] = C::InterpolateForParameter(first.nestedData[i], second.nestedData[i], t);
            }
        }
        return ret;
    }

    template<class R>
    bool TrivialReturn(KeyTimeType time, std::function<R(const C&)> accessor, std::vector<R>& ret, KeyType& before_key,
        KeyType& after_key) const {
        ret.resize(vec_length);
        if (this->keys.size() < 2) {
            if (this->keys.empty()) {
                return true;
            }
            for (int i = 0; i < vec_length; ++i) {
                ret[i] = accessor(this->keys.begin()->second.nestedData[i]);
            }
            return true;
        }
        before_key = this->keys.begin()->second, after_key = this->keys.begin()->second;
        bool ok = false;
        for (auto it = this->keys.begin(); it != this->keys.end(); ++it) {
            if (it->second.nestedData[0].time == time) {
                for (int i = 0; i < vec_length; ++i) {
                    ret[i] = accessor(it->second.nestedData[i]);
                }
                return true;
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
            // read: interpolate yourself
            return false;
        } else {
            for (int i = 0; i < vec_length; ++i) {
                ret[i] = accessor(before_key.nestedData[i]);
            }
            return true;
        }
    }

    std::vector<InterpolationType> GetInterpolation(KeyTimeType time) const {
        std::vector<InterpolationType> ret;
        ret.resize(vec_length);
        KeyType before, after;
        // extract the value from the key
        auto getter = [](const C& k) -> InterpolationType { return k.interpolation; };
        if (TrivialReturn<InterpolationType>(time, getter, ret, before, after)) {
            return ret;
        } else {
            for (int i = 0; i < vec_length; ++i) {
                ret[i] = before.nestedData[i].interpolation;
            }
            return ret;
        }
    }

    ValueType GetValue(KeyTimeType time) const {
        ValueType ret;
        ret.resize(vec_length);
        KeyType before, after;
        // extract the value from the key
        auto getter = [](const C& k) -> typename C::ValueType { return k.value; };
        if (TrivialReturn<typename C::ValueType>(time, getter, ret, before, after)) {
            return ret;
        } else {
            return InterpolateForTime(before, after, time);
        }
    }

    std::vector<ImVec2> GetTangent(KeyTimeType time) const {
        std::vector<ImVec2> ret;
        ret.resize(vec_length);
        KeyType before, after;
        // extract the value from the key
        auto getter = [](const C& k) -> ImVec2 { return k.out_tangent; };
        if (TrivialReturn<ImVec2>(time, getter, ret, before, after)) {
            return ret;
        } else {
            for (int i = 0; i < vec_length; ++i) {
                ret[i] = FloatKey::TangentForTime(before.nestedData[i], after.nestedData[i], time);
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
            return this->keys.rbegin()->second.nestedData[0].time - this->keys.begin()->second.nestedData[0].time + 1;
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

} // namespace megamol::core::utility::animation

namespace megamol::core::utility {
std::ostream& operator<<(std::ostream& outs, const animation::VectorKey<animation::FloatKey>::ValueType& value);
}
