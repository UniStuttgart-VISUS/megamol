#pragma once

#include "glm/glm.hpp"

#include "optix.h"

namespace megamol {
namespace optix_hpg {
    namespace device {
        struct ParticleMeshDistanceData2 {
            float* distances;
            unsigned int* inter_count;

            glm::uvec3* index_buffer;
            glm::vec3* vertex_buffer;

            unsigned int num_tris;
        };

        struct PMDRayGenData {
            void* ray_buffer;

            unsigned int num_rays;

            OptixTraversableHandle world;
        };
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
