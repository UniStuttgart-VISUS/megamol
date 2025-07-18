#pragma once

#ifdef MEGAMOL_USE_POWER

#include <filesystem>
#include <set>
#include <unordered_set>

#include "ColumnNames.h"
#include "MetaData.h"
#include "ParquetWriter.h"
#include "Utility.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

namespace megamol::power {

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

struct wf_tracy_wrapper {
    static std::unordered_set<std::string> name_lib;

    static void wf_tracy(std::filesystem::path const& output_folder, std::string const&,
        power::segments_t const& values_map, power::MetaData const* meta) {
#ifdef MEGAMOL_USE_TRACY
#ifdef MEGAMOL_USE_TRACY_TIME_PLOT
        for (auto const& vm : values_map) {
            for (auto const& [name, v] : vm) {
                name_lib.insert(name);
            }
        }

        for (std::size_t s_idx = 0; s_idx < values_map.size(); ++s_idx) {
            auto const& vm = values_map[s_idx];
            auto const& timestamps = std::get<power::timeline_t>(vm.at(global_ts_name));
            for (auto const& [name, v_values] : vm) {
                if (std::holds_alternative<power::samples_t>(v_values)) {
                    auto const t_name_it = name_lib.find(name);
                    if (t_name_it != name_lib.end()) {
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
#endif
    }
};

inline void wf_tracy(std::filesystem::path const& output_folder, [[maybe_unused]] std::string const&,
    power::segments_t const& values_map, power::MetaData const* meta) {
#ifdef MEGAMOL_USE_TRACY
#ifdef MEGAMOL_USE_TRACY_TIME_PLOT
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
#endif
}

} // namespace megamol::power

#endif
