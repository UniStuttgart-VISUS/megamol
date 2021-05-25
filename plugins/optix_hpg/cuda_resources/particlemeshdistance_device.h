#pragma once

#include "glm/glm.hpp"

#include "optix.h"

namespace megamol {
namespace optix_hpg {
    namespace device {
        struct ParticleMeshDistanceData {
            float* distances;

            glm::uvec3* index_buffer;
            glm::vec3* vertex_buffer;

            void* ray_buffer;

            unsigned int num_rays;
            unsigned int num_tris;

            OptixTraversableHandle world;
        };
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
