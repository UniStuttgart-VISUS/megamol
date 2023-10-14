#pragma once

#ifdef MEGAMOL_USE_POWER

#include <filesystem>
#include <set>

#include "MetaData.h"
#include "ParquetWriter.h"
#include "Utility.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

namespace megamol::power {

inline std::filesystem::path create_full_path(
    std::filesystem::path const& output_folder, std::string const& device_name, std::size_t const s_idx) {
    return output_folder / (device_name + "_s" + std::to_string(s_idx) + ".parquet");
}

inline void wf_parquet(std::filesystem::path const& output_folder, std::string const& device_name,
    power::segments_t const& values_map, power::MetaData const* meta) {
    for (std::size_t s_idx = 0; s_idx < values_map.size(); ++s_idx) {
        auto const fullpath = create_full_path(output_folder, device_name, s_idx);
        ParquetWriter(fullpath, values_map[s_idx], meta);
    }
}

inline void wf_parquet_dataverse(std::filesystem::path const& output_folder, std::string const& device_name,
    power::segments_t const& values_map, power::MetaData const* meta,
    std::function<void(std::string)> const& dataverse_writer) {
    for (std::size_t s_idx = 0; s_idx < values_map.size(); ++s_idx) {
        auto const fullpath = create_full_path(output_folder, device_name, s_idx);
        ParquetWriter(fullpath, values_map[s_idx], meta);
        dataverse_writer(fullpath.string());
    }
}

inline void wf_tracy(std::filesystem::path const& output_folder, [[maybe_unused]] std::string const&,
    power::segments_t const& values_map, power::MetaData const* meta) {
#ifdef MEGAMOL_USE_TRACY
    static std::set<std::string> tpn_library;
    for (std::size_t s_idx = 0; s_idx < values_map.size(); ++s_idx) {
        auto const& vm = values_map[s_idx];
        auto const& timestamps = std::get<power::timeline_t>(vm.at("timestamps"));
        for (auto const& [name, v_values] : vm) {
            if (std::holds_alternative<power::samples_t>(v_values)) {
                auto const c_name = name + "\0";
                tpn_library.insert(c_name);
                auto t_name_it = tpn_library.find(name.c_str());
                if (t_name_it != tpn_library.end()) {
                    auto const& values = std::get<power::samples_t>(v_values);
                    TracyPlotConfig(t_name_it->data(), tracy::PlotFormatType::Number, false, true, 0);
                    for (std::size_t v_idx = 0; v_idx < values.size(); ++v_idx) {
                        tracy::Profiler::PlotData(t_name_it->data(), values[v_idx], timestamps[v_idx]);
                    }
                }
            }
        }
    }
#endif
}

} // namespace megamol::power

#endif
