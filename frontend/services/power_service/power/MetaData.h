#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "Timestamp.h"

namespace megamol::power {
/**
 * @brief Struct to hold the meta data of the measurements.
 */
struct MetaData {
    /**
     * @brief MegaMol project file while running the measurement.
     */
    std::string project_file;
    /**
     * @brief Power-overwhelming configurations.
     */
    std::unordered_map<std::string, std::string> oszi_configs;
    /**
     * @brief Loaded libraries.
     */
    std::string runtime_libs;
    /**
     * @brief Hardware and software information.
     */
    std::unordered_map<std::string, std::string> hardware_software_info;
    /**
     * @brief The mapping of the different sensors to power rails for later analysis.
     */
    std::unordered_map<std::string, std::string> analysis_recipes;
    /**
     * @brief The timestamps of the trigger signals.
     */
    std::vector<filetime_dur_t> trigger_ts;
};
} // namespace megamol::power
