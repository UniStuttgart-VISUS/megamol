/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <array>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <unordered_map>

namespace megamol::core::view::light {

struct BaseLightType {
    std::array<float, 4> colour;
    float intensity;
};

class LightCollection {
public:
    LightCollection() = default;
    ~LightCollection() = default;

    /**
     *
     */
    template<typename LightType>
    std::vector<LightType> get() const;

    /**
     *
     */
    template<typename LightType>
    void add(std::shared_ptr<LightType> const& light);

    /**
     *
     */
    template<typename LightType>
    void add(std::shared_ptr<LightType>&& light);

private:
    // Note to future maintainer: Use of pointer type for individual lights is part of
    // what turns this into a generic solution that requires no knowledge about the actual
    // light types contained later on. At the same time, it will propably not scale with
    // larger amounts of lights
    std::unordered_multimap<std::type_index, std::shared_ptr<BaseLightType>> m_lights;
};

template<typename LightType>
inline std::vector<LightType> LightCollection::get() const {
    std::vector<LightType> retval;

    auto range = m_lights.equal_range(std::type_index(typeid(LightType)));
    for (auto it = range.first; it != range.second; ++it) {
        retval.push_back(*(static_cast<LightType*>(it->second.get())));
    }

    return retval;
}

template<typename LightType>
inline void LightCollection::add(std::shared_ptr<LightType> const& light) {
    m_lights.insert(std::type_index(typeid(LightType)), std::shared_ptr<LightType>(light));
}

template<typename LightType>
inline void LightCollection::add(std::shared_ptr<LightType>&& light) {
    m_lights.emplace(std::type_index(typeid(LightType)), std::forward<std::shared_ptr<LightType>>(light));
}

} // namespace megamol::core::view::light
