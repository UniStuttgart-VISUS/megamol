#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "srtest/rendering_task.h"

#ifdef MEGAMOL_USE_PROFILING
#include "PerformanceManager.h"
#endif

namespace megamol::moldyn_gl::rendering {
class SphereRasterizer {
public:
    struct config {
        glm::uvec2 res;
        glm::vec2 fres;
        glm::vec2 near_far;
        float global_radius;
        glm::mat4 MVP;
        glm::mat4 MVPinv;
        glm::vec3 camDir, camUp, camRight, camPos;
        float fovy;
        float ratio;
        glm::vec3 lightDir;
        glm::vec3 lower;
        glm::vec3 upper;
    };

    using config_t = config;

    std::vector<glm::u8vec4> Compute(config_t& config, data_package_t const& data, std::vector<float> const& radius);

#ifdef MEGAMOL_USE_PROFILING
    megamol::frontend_resources::PerformanceManager::handle_vector* timing_handles_;

    megamol::frontend_resources::PerformanceManager* pm;
#endif
protected:
private:
};
} // namespace megamol::moldyn_gl::rendering
