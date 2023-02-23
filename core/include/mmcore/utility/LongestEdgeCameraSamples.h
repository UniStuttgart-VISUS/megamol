#pragma once

#include <tuple>
#include <vector>

#include <glm/glm.hpp>

#include "mmcore/BoundingBoxes_2.h"

namespace megamol::core::utility {
std::tuple<std::vector<glm::vec3>, std::vector<glm::vec3>> longest_edge_camera_samples(
    core::BoundingBoxes_2 const& bboxes, unsigned int num_samples);
} // namespace megamol::core::utility
