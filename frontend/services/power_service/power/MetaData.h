#pragma once

#include <string>
#include <unordered_map>

namespace megamol::power {
struct MetaData {
    std::string project_file;
    std::unordered_map<std::string, std::string> oszi_configs;
    std::string runtime_libs;
    std::unordered_map<std::string, std::string> hardware_software_info;
    std::unordered_map<std::string, std::string> analysis_recipes;
};
} // namespace megamol::power
