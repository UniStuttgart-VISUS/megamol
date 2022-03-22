#pragma once

#include "glm/glm.hpp"

namespace megamol {
namespace optix_hpg {
namespace device {

struct MeshGeoData {
    glm::uvec3* index_buffer;
    glm::vec3* vertex_buffer;
};

} // namespace device
} // namespace optix_hpg
} // namespace megamol
