#pragma once

#include <string>
#include <unordered_map>

namespace megamol::power {
struct MetaData {
    std::string project_file;
    std::unordered_map<std::string, std::string> oszi_configs;
};
} // namespace megamol::power
