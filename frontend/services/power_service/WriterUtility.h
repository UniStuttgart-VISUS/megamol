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

inline void wf_parquet(int64_t, std::filesystem::path const& output_folder, power::segments_t const& values_map) {
    for (std::size_t s_idx = 0; s_idx < values_map.size(); ++s_idx) {
        auto const fullpath = output_folder / ("rtx_s" + std::to_string(s_idx) + ".parquet");
        ParquetWriter(fullpath, values_map[s_idx]);
    }
}

inline int64_t get_tracy_time(int64_t base, int64_t tracy_offset) {
    static int64_t const frequency = get_highres_timer_freq();
    auto base_ticks =
        static_cast<int64_t>((static_cast<double>(base) / 1000. / 1000. / 1000.) * static_cast<double>(frequency));
    return base_ticks + tracy_offset;
}

inline void wf_tracy(
    int64_t qpc_last_trigger, std::filesystem::path const& output_folder, power::segments_t const& values_map) {
#ifdef MEGAMOL_USE_TRACY
    static std::set<std::string> tpn_library;
    for (std::size_t s_idx = 0; s_idx < values_map.size(); ++s_idx) {
        auto const& vm = values_map[s_idx];
        auto const& samples_time = std::get<power::timeline_t>(vm.at("abs_times"));
        for (auto const& [name, v_values] : vm) {
            if (std::holds_alternative<power::samples_t>(v_values)) {
                auto const c_name = name + "\0";
                tpn_library.insert(c_name);
                auto t_name_it = tpn_library.find(name.c_str());
                if (t_name_it != tpn_library.end()) {
                    auto const& values = std::get<power::samples_t>(v_values);
                    TracyPlotConfig(t_name_it->data(), tracy::PlotFormatType::Number, false, true, 0);
                    for (std::size_t v_idx = 0; v_idx < values.size(); ++v_idx) {
                        tracy::Profiler::PlotData(
                            t_name_it->data(), values[v_idx], get_tracy_time(samples_time[v_idx], qpc_last_trigger));
                    }
                }
            }
        }
    }
#endif
}

} // namespace megamol::frontend

#endif
