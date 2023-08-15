#pragma once

#ifdef MEGAMOL_USE_POWER

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <power_overwhelming/rtx_instrument.h>
#include <power_overwhelming/rtx_instrument_configuration.h>

#include "ParallelPortTrigger.h"
#include "ParquetWriter.h"

#include <sol/sol.hpp>

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

#ifdef WIN32
#include <Windows.h>
#else
#include <time.h>
#endif

namespace megamol::frontend {

class RTXInstrument {
public:
    using timeline_t = std::vector<int64_t>;
    using samples_t = std::vector<float>;
    using value_map_t = std::unordered_map<std::string, std::variant<samples_t, timeline_t>>;
    using segments_t = std::vector<value_map_t>;

    RTXInstrument();

    void UpdateConfigs(std::filesystem::path const& config_folder, int points, int count,
        std::chrono::milliseconds range, std::chrono::milliseconds timeout);

    void ApplyConfigs();

    void StartMeasurement(std::filesystem::path const& output_folder,
        std::vector<std::function<void(std::filesystem::path const&, segments_t const&)>> const& writer_funcs);

    void SetLPTTrigger(std::string const& address);

    void SetSoftwareTrigger(bool set) {
        enforce_software_trigger_ = set;
    }

private:
    bool waiting_on_trigger() const;

    std::chrono::system_clock::time_point trigger() {
        if (enforce_software_trigger_) {
            std::for_each(rtx_instr_.begin(), rtx_instr_.end(), [](auto& instr) { instr.second.trigger_manually(); });
        } else {
            if (lpt_trigger_) {
                lpt_trigger_->SetBit(6, true);
                lpt_trigger_->SetBit(6, false);
            }
        }
        return std::chrono::system_clock::now();
    }

    timeline_t generate_timestamps_ns(visus::power_overwhelming::oscilloscope_waveform const& waveform) const;

    timeline_t offset_timeline(timeline_t const& timeline, std::chrono::nanoseconds offset) const;

    std::unordered_map<std::string, visus::power_overwhelming::rtx_instrument> rtx_instr_;

    std::unordered_map<std::string, visus::power_overwhelming::rtx_instrument_configuration> rtx_config_;

    sol::state sol_state_;

    std::chrono::milliseconds config_range_;

    std::unique_ptr<ParallelPortTrigger> lpt_trigger_ = nullptr;

    bool enforce_software_trigger_ = false;
};

inline std::string get_name(visus::power_overwhelming::rtx_instrument const& i) {
    auto const name_size = i.name(nullptr, 0);
    std::string name;
    name.resize(name_size);
    i.name(name.data(), name.size());
    return name;
}

inline std::string get_identity(visus::power_overwhelming::rtx_instrument& i) {
    auto const id_size = i.identify(nullptr, 0);
    std::string id;
    id.resize(id_size);
    i.identify(id.data(), id.size());
    return id;
}

inline std::chrono::nanoseconds tp_dur_to_epoch_ns(std::chrono::system_clock::time_point const& tp) {
    static auto epoch = std::chrono::system_clock::from_time_t(0);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp - epoch);
}

inline std::vector<float> transform_waveform(visus::power_overwhelming::oscilloscope_waveform const& wave) {
    std::vector<float> ret(wave.record_length());
    std::copy(wave.begin(), wave.end(), ret.begin());
    return ret;
}

inline void wf_parquet(std::filesystem::path const& output_folder, RTXInstrument::segments_t const& values_map) {
    for (std::size_t s_idx = 0; s_idx < values_map.size(); ++s_idx) {
        auto const fullpath = output_folder / ("rtx_s" + std::to_string(s_idx) + ".parquet");
        ParquetWriter(fullpath, values_map[s_idx]);
    }
}

inline int64_t get_highres_timer_freq() {
#ifdef WIN32
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    return f.QuadPart;
#else
    timespec tp;
    clock_getres(CLOCK_MONOTONIC_RAW, &tp);
    return tp.tv_nsec;
#endif
}

inline int64_t get_tracy_time(int64_t base, int64_t tracy_offset) {
    static int64_t const frequency = get_highres_timer_freq();
    auto base_ticks =
        static_cast<int64_t>((static_cast<double>(base) / 1000. / 1000. / 1000.) * static_cast<double>(frequency));
    return base_ticks + tracy_offset;
}

inline void wf_tracy(
    int64_t qpc_last_trigger, std::filesystem::path const& output_folder, RTXInstrument::segments_t const& values_map) {
#ifdef MEGAMOL_USE_TRACY
    for (std::size_t s_idx = 0; s_idx < values_map.size(); ++s_idx) {
        auto const& vm = values_map[s_idx];
        auto const& samples_time = std::get<RTXInstrument::timeline_t>(vm.at("abs_time"));
        for (auto const& [name, v_values] : vm) {
            if (std::holds_alternative<RTXInstrument::samples_t>(v_values)) {
                auto const c_name = name.c_str();
                auto const& values = std::get<RTXInstrument::samples_t>(v_values);
                TracyPlotConfig(c_name, tracy::PlotFormatType::Number, false, true, 0);
                for (std::size_t v_idx = 0; v_idx < values.size(); ++v_idx) {
                    tracy::Profiler::PlotData(
                        c_name, values[v_idx], get_tracy_time(samples_time[v_idx], qpc_last_trigger));
                }
            }
        }
    }
#endif
}

} // namespace megamol::frontend

#endif
