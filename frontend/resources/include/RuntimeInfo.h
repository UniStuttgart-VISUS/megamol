#pragma once

#include <functional>
#include <string>

namespace megamol::frontend_resources {
struct RuntimeInfo {
    std::function<std::string()> get_hardware_info;
    std::function<std::string()> get_os_info;
    std::function<std::string()> get_runtime_libraries;
};
} // namespace megamol::frontend_resources
