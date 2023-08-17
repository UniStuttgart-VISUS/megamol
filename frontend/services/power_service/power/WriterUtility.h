#pragma once

#ifdef MEGAMOL_USE_POWER

#include <filesystem>
#include <set>

#include "ParquetWriter.h"
#include "Utility.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

namespace megamol::power {

inline void wf_parquet(std::filesystem::path const& output_folder, power::segments_t const& values_map) {
    for (std::size_t s_idx = 0; s_idx < values_map.size(); ++s_idx) {
        auto const fullpath = output_folder / ("rtx_s" + std::to_string(s_idx) + ".parquet");
        ParquetWriter(fullpath, values_map[s_idx]);
    }
}

inline void wf_tracy(std::filesystem::path const& output_folder, power::segments_t const& values_map) {
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
