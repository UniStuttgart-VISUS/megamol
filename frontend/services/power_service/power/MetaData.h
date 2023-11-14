#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "Timestamp.h"

namespace megamol::power {
struct MetaData {
    std::string project_file;
    std::unordered_map<std::string, std::string> oszi_configs;
    std::string runtime_libs;
    std::unordered_map<std::string, std::string> hardware_software_info;
    std::unordered_map<std::string, std::string> analysis_recipes;
    std::vector<filetime_dur_t> trigger_ts;
};
} // namespace megamol::power
