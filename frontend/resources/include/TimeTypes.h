#pragma once

#include <chrono>
#include <cstdint>

namespace megamol::frontend_resources::performance {

using handle_type = uint32_t;
using timer_index = uint32_t;
using handle_vector = std::vector<handle_type>;
using frame_type = uint32_t;
using user_index_type = uint32_t;
using time_point = std::chrono::steady_clock::time_point;
inline constexpr time_point zero_time = time_point::min();

enum class query_api { CPU, OPENGL }; // TODO: CUDA, OpenCL, Vulkan, whatnot

static constexpr const char* query_api_string(query_api api) {
    switch (api) {
    case query_api::CPU:
        return "CPU";
    case query_api::OPENGL:
        return "OpenGL";
    }
    return "unknown";
}

} // namespace megamol::frontend_resources::performance
