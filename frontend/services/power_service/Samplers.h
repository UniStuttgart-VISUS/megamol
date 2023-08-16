#pragma once

#if MEGAMOL_USE_POWER

#include <chrono>
#include <codecvt>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <power_overwhelming/emi_sensor.h>
#include <power_overwhelming/msr_sensor.h>
#include <power_overwhelming/nvml_sensor.h>
#include <power_overwhelming/tinkerforge_sensor.h>

#include "SampleBuffer.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

namespace megamol::frontend {
namespace power {
template<typename T>
using samplers_t = std::unordered_map<std::string, T>;

using buffers_t = std::vector<SampleBuffer>;
} // namespace power

inline void tracy_sample(
    wchar_t const*, visus::power_overwhelming::measurement_data const* m, std::size_t const n, void* usr_ptr) {
    auto usr_data = static_cast<std::pair<std::string, SampleBuffer*>*>(usr_ptr);
    auto const& name = usr_data->first;
    auto buffer = usr_data->second;
#ifdef MEGAMOL_USE_TRACY
    for (std::size_t i = 0; i < n; ++i) {
        TracyPlot(name.data(), m[i].power());
    }
#endif
    for (std::size_t i = 0; i < n; ++i) {
        buffer->Add(m[i].power(), m[i].timestamp());
    }
}

inline std::string unmueller_string(wchar_t const* name) {
    std::string no_mueller =
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t>{}.to_bytes(std::wstring(name));

    /*char* sensor_name = new char[wcslen(name) + 1];
    wcstombs(sensor_name, name, wcslen(name) + 1);
    std::string no_mueller(sensor_name);
    delete[] sensor_name;*/
    return no_mueller;
}

template<typename T>
inline std::tuple<power::samplers_t<T>, power::buffers_t> InitSampler(
    std::chrono::milliseconds const& sample_range, std::chrono::milliseconds const& sample_dis) {
    using namespace visus::power_overwhelming;
    auto sensor_count = T::for_all(nullptr, 0);
    std::vector<T> tmp_sensors(sensor_count);
    T::for_all(tmp_sensors.data(), tmp_sensors.size());

    power::buffers_t buffers;
    buffers.reserve(sensor_count);

    power::samplers_t<T> sensors;
    sensors.reserve(sensor_count);

    for (auto& sensor : tmp_sensors) {
        auto sensor_name = unmueller_string(sensor.name());

#ifdef MEGAMOL_USE_TRACY
        TracyPlotConfig(sensor_name.data(), tracy::PlotFormatType::Number, false, true, 0);
#endif

        buffers.push_back(SampleBuffer(sensor_name, sample_range, sample_dis));

        sensors[sensor_name] = std::move(sensor);
        sensors[sensor_name].sample(
            std::move(async_sampling()
                          .delivers_measurement_data_to(&tracy_sample)
                          .stores_and_passes_context(std::make_pair(sensor_name, &buffers.back()))
                          .samples_every(std::chrono::duration_cast<std::chrono::microseconds>(sample_dis).count())
                          .using_resolution(timestamp_resolution::microseconds)));
    }

    return std::make_tuple(std::move(sensors), std::move(buffers));
}
} // namespace megamol::frontend

#endif
