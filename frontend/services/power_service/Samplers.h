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
#include "Utility.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

namespace megamol::power {

template<typename T>
using samplers_t = std::unordered_map<std::string, T>;

using buffers_t = std::vector<SampleBuffer>;

inline int64_t convert_timestamp(int64_t const& offset, int64_t const& ts) {
    return ts + offset;
}

inline int64_t convert_walltime(int64_t const& ts) {
    constexpr int64_t offset =
        std::chrono::duration<std::int64_t, std::ratio<1, 10000000>>(116444736000000000LL).count();
    return ts - offset;
}

inline void tracy_sample(
    wchar_t const*, visus::power_overwhelming::measurement_data const* m, std::size_t const n, void* usr_ptr) {
    auto usr_data = static_cast<std::tuple<std::string, SampleBuffer*, bool const&, int64_t const&>*>(usr_ptr);
    auto const& name = std::get<0>(*usr_data);
    auto buffer = std::get<1>(*usr_data);
    auto const& do_buffer = std::get<2>(*usr_data);
#ifdef MEGAMOL_USE_TRACY
    for (std::size_t i = 0; i < n; ++i) {
        TracyPlot(name.data(), m[i].power());
    }
#endif
    if (do_buffer) {
        auto const& offset = std::get<3>(*usr_data);
        for (std::size_t i = 0; i < n; ++i) {
            buffer->Add(m[i].power(), convert_timestamp(offset, m[i].timestamp()), convert_walltime(m[i].timestamp()));
        }
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
inline std::tuple<samplers_t<T>, buffers_t> InitSampler(std::chrono::milliseconds const& sample_range,
    std::chrono::milliseconds const& sample_dis, bool const& do_buffer, int64_t const& offset) {
    using namespace visus::power_overwhelming;
    auto sensor_count = T::for_all(nullptr, 0);
    std::vector<T> tmp_sensors(sensor_count);
    T::for_all(tmp_sensors.data(), tmp_sensors.size());

    buffers_t buffers;
    buffers.reserve(sensor_count);

    samplers_t<T> sensors;
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
                          .stores_and_passes_context(
                              std::make_tuple(sensor_name, &buffers.back(), std::cref(do_buffer), std::cref(offset)))
                          .samples_every(std::chrono::duration_cast<std::chrono::microseconds>(sample_dis).count())
                          .using_resolution(timestamp_resolution::hundred_nanoseconds)));
    }

    return std::make_tuple(std::move(sensors), std::move(buffers));
}
} // namespace megamol::power

#endif
