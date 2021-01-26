#pragma once

#include "glm/glm.hpp"

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {

            struct MeshGeoData {
                glm::uvec3* index_buffer;
                glm::vec3* vertex_buffer;
            };

        }
    } // namespace optix
} // namespace hpg
} // namespace megamol
