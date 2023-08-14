#pragma once

#ifdef MEGAMOL_USE_POWER

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <power_overwhelming/rtx_instrument.h>
#include <power_overwhelming/rtx_instrument_configuration.h>

#include <sol/sol.hpp>

namespace megamol::frontend {

class RTXInstrument {
public:
    using timeline_t = std::vector<int64_t>;
    using samples_t = std::vector<float>;

    RTXInstrument();

    void UpdateConfigs(std::filesystem::path const& config_folder, int points, int count,
        std::chrono::milliseconds range, std::chrono::milliseconds timeout);

    void ApplyConfigs();

    void StartMeasurement(std::filesystem::path const& output_folder);

private:
    bool waiting_on_trigger() const;

    std::chrono::system_clock::time_point trigger();

    void write_data(
        std::filesystem::path const& output_folder, visus::power_overwhelming::oscilloscope_sample const& data) const;

    timeline_t generate_timestamps_ns(visus::power_overwhelming::oscilloscope_waveform const& waveform) const;

    timeline_t offset_timeline(timeline_t const& timeline, std::chrono::nanoseconds offset) const;

    std::unordered_map<std::string, visus::power_overwhelming::rtx_instrument> rtx_instr_;

    std::unordered_map<std::string, visus::power_overwhelming::rtx_instrument_configuration> rtx_config_;

    sol::state sol_state_;

    std::chrono::milliseconds config_range_;
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

inline int64_t tp_dur_to_epoch_ns(std::chrono::system_clock::time_point const& tp) {
    static auto epoch = std::chrono::system_clock::from_time_t(0);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp - epoch).count();
}

inline std::vector<float> transform_waveform(visus::power_overwhelming::oscilloscope_waveform const& wave) {
    std::vector<float> ret(wave.record_length());
    std::copy(wave.begin(), wave.end(), ret.begin());
    return ret;
}

} // namespace megamol::frontend

#endif
