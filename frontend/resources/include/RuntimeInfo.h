#pragma once

#include <functional>
#include <string>

namespace megamol::frontend_resources {
struct RuntimeInfo {
    std::function<std::string()> get_hardware_info;
    std::function<std::string()> get_os_info;
    std::function<std::string()> get_runtime_libraries;

    std::function<std::string()> get_smbios_info;
    std::function<std::string()> get_cpu_info;
    std::function<std::string()> get_gpu_info;
    std::function<std::string()> get_OS_info;
};
} // namespace megamol::frontend_resources
