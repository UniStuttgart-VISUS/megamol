#pragma once

#include <algorithm>

#include "glm/glm.hpp"

namespace megamol::stdplugin::datatools {
inline float tf_lerp(float a, float b, float inter) {
    return a * (1.0f - inter) + b * inter;
}

inline glm::vec4 sample_tf(float const* tf, unsigned int tf_size, int base, float rest) {
    if (base < 0 || tf_size == 0)
        return glm::vec4(0);
    auto const last_el = tf_size - 1;
    if (base >= last_el)
        return glm::vec4(tf[last_el * 4], tf[last_el * 4 + 1], tf[last_el * 4 + 2], tf[last_el * 4 + 3]);

    auto const a = base;
    auto const b = base + 1;

    return glm::vec4(tf_lerp(tf[a * 4], tf[b * 4], rest), tf_lerp(tf[a * 4 + 1], tf[b * 4 + 1], rest),
        tf_lerp(tf[a * 4 + 2], tf[b * 4 + 2], rest), tf_lerp(tf[a * 4 + 3], tf[b * 4 + 3], rest));
}

inline glm::vec4 get_sample_from_tf(float const* tf, unsigned int tf_size, float val, float min_val, float fac) {
    auto const val_a = (val - min_val) * fac * static_cast<float>(tf_size);
    std::decay_t<decltype(val_a)> main_a = 0;
    auto rest_a = std::modf(val_a, &main_a);
    rest_a = static_cast<int>(main_a) >= 0 && static_cast<int>(main_a) < tf_size ? rest_a : 0.0f;
    main_a = std::clamp(static_cast<int>(main_a), 0, static_cast<int>(tf_size) - 1);
    return stdplugin::datatools::sample_tf(tf, tf_size, static_cast<int>(main_a), rest_a);
}
} // namespace megamol::stdplugin::datatools
