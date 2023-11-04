#pragma once

#if MEGAMOL_USE_POWER

#include <chrono>
#include <codecvt>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <power_overwhelming/adl_sensor.h>
#include <power_overwhelming/emi_sensor.h>
#include <power_overwhelming/msr_sensor.h>
#include <power_overwhelming/nvml_sensor.h>
#include <power_overwhelming/tinkerforge_sensor.h>

#include "DataverseWriter.h"
#include "ParquetWriter.h"
#include "SampleBuffer.h"
#include "StringContainer.h"
#include "Utility.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

using namespace visus::power_overwhelming;

namespace megamol::power {

template<typename T>
using samplers_t = std::unordered_map<std::string, T>;

using buffers_t = std::vector<SampleBuffer>;

using context_t = std::tuple<char const*, SampleBuffer*, bool const&>;


inline void tracy_sample(
    wchar_t const*, visus::power_overwhelming::measurement_data const* m, std::size_t const n, void* usr_ptr) {
    auto usr_data = static_cast<context_t*>(usr_ptr);
    auto name = std::get<0>(*usr_data);
    auto buffer = std::get<1>(*usr_data);
    auto const& do_buffer = std::get<2>(*usr_data);
#ifdef MEGAMOL_USE_TRACY
    for (std::size_t i = 0; i < n; ++i) {
        TracyPlot(name, m[i].power());
    }
#endif
    if (do_buffer) {
        for (std::size_t i = 0; i < n; ++i) {
            buffer->Add(m[i].power(), m[i].timestamp());
        }
    }
}

inline std::string unmueller_string(wchar_t const* name) {
    /*std::string no_mueller =
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t>{}.to_bytes(std::wstring(name));*/

    // https://en.cppreference.com/w/cpp/locale/codecvt/out

    auto const& f = std::use_facet<std::codecvt<wchar_t, char, std::mbstate_t>>(std::locale());

    std::wstring internal(name);
    std::mbstate_t mb = std::mbstate_t();
    std::string external(internal.size() * f.max_length(), '\0');
    const wchar_t* from_next;
    char* to_next;

    auto const res = f.out(
        mb, &internal[0], &internal[internal.size()], from_next, &external[0], &external[external.size()], to_next);
    if (res != std::codecvt_base::ok) {
        throw std::runtime_error("could not convert string");
    }
    external.resize(to_next - &external[0]);

    return external;
}

using discard_func_t = std::function<bool(std::string const&)>;

template<typename T>
using config_func_t = std::function<void(T&)>;

//template<typename T>
//inline std::tuple<samplers_t<T>, buffers_t> InitSampler(std::chrono::milliseconds const& sample_range,
//    std::chrono::milliseconds const& sample_dis, StringContainer* str_cont, bool const& do_buffer,
//    discard_func_t discard = nullptr, config_func_t<T> config = nullptr) {
//    using namespace visus::power_overwhelming;
//    auto sensor_count = T::for_all(nullptr, 0);
//    std::vector<T> tmp_sensors(sensor_count);
//    T::for_all(tmp_sensors.data(), tmp_sensors.size());
//
//    buffers_t buffers;
//    buffers.reserve(sensor_count);
//
//    samplers_t<T> sensors;
//    sensors.reserve(sensor_count);
//
//    for (auto& sensor : tmp_sensors) {
//        auto str_ptr = str_cont->Add(unmueller_string(sensor.name()));
//        if (discard) {
//            if (discard(*str_ptr))
//                continue;
//        }
//
//#ifdef MEGAMOL_USE_TRACY
//        TracyPlotConfig(str_ptr->data(), tracy::PlotFormatType::Number, false, true, 0);
//#endif
//
//        buffers.push_back(SampleBuffer(*str_ptr, sample_range, sample_dis));
//
//        sensors[*str_ptr] = std::move(sensor);
//        if (config) {
//            config(sensors[*str_ptr]);
//        }
//        sensors[*str_ptr].sample(std::move(
//            async_sampling()
//                .delivers_measurement_data_to(&tracy_sample)
//                .stores_and_passes_context(std::make_tuple(str_ptr->data(), &buffers.back(), std::cref(do_buffer)))
//                .samples_every(std::chrono::duration_cast<std::chrono::microseconds>(sample_dis).count())
//                .using_resolution(timestamp_resolution::hundred_nanoseconds)
//                .from_source(tinkerforge_sensor_source::power)));
//    }
//
//    return std::make_tuple(std::move(sensors), std::move(buffers));
//}

} // namespace megamol::power

#endif
