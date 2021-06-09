#pragma once

#include "glm/glm.hpp"

namespace megamol {
namespace optix_hpg {
    namespace device {

        struct MeshGeoData {
            glm::uvec3* index_buffer;
            glm::vec3* vertex_buffer;
            glm::vec4* color_buffer;

            bool got_color;

            bool test_rad_func = false;
            int time_slice;
            int frame_id;
            glm::vec4* event_buffer;
            int num_events;
            float max_rad;
            float thickness;
        };

    } // namespace device
} // namespace optix_hpg
} // namespace megamol
