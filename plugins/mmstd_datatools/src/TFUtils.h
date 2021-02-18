#pragma once

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
} // namespace megamol::stdplugin::datatools
