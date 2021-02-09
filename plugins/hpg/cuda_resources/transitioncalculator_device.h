#pragma once

#include "glm/glm.hpp"

#include "optix.h"

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {
            struct TransitionCalculatorData {
                unsigned int* mesh_inbound_ctr_ptr;
                unsigned int* mesh_outbound_ctr_ptr;
                unsigned char* ray_state;

                glm::uvec3* index_buffer;
                glm::vec3* vertex_buffer;

                void* ray_buffer;

                unsigned int num_rays;
                unsigned int num_tris;

                OptixTraversableHandle world;
            };
        }
    } // namespace optix
} // namespace hpg
} // namespace megamol
