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
#include "StringContainer.h"
#include "Utility.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

namespace megamol::power {

template<typename T>
using samplers_t = std::unordered_map<std::string, T>;

using buffers_t = std::vector<SampleBuffer>;

using context_t = std::tuple<char const*, SampleBuffer*, bool const&>;

inline int64_t convert_timestamp_ns(int64_t const& ts_100ns) {
    //return ts_100ns + offset_100ns;
    return ts_100ns;
}

inline int64_t convert_walltime_ns(int64_t const& ts_100ns) {
    constexpr int64_t offset =
        std::chrono::duration<std::int64_t, std::ratio<1, 10000000>>(116444736000000000LL).count();
    return ts_100ns - offset;
}

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
            buffer->Add(m[i].power(), m[i].timestamp(), convert_walltime_ns(m[i].timestamp()));
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

using discard_func_t = std::function<bool(std::string const&)>;

template<typename T>
using config_func_t = std::function<void(T&)>;

template<typename T>
inline std::tuple<samplers_t<T>, buffers_t> InitSampler(std::chrono::milliseconds const& sample_range,
    std::chrono::milliseconds const& sample_dis, StringContainer* str_cont, bool const& do_buffer,
    discard_func_t discard = nullptr, config_func_t<T> config = nullptr) {
    using namespace visus::power_overwhelming;
    auto sensor_count = T::for_all(nullptr, 0);
    std::vector<T> tmp_sensors(sensor_count);
    T::for_all(tmp_sensors.data(), tmp_sensors.size());

    buffers_t buffers;
    buffers.reserve(sensor_count);

    samplers_t<T> sensors;
    sensors.reserve(sensor_count);

    for (auto& sensor : tmp_sensors) {
        auto str_ptr = str_cont->Add(unmueller_string(sensor.name()));
        if (discard) {
            if (discard(*str_ptr))
                continue;
        }

#ifdef MEGAMOL_USE_TRACY
        TracyPlotConfig(str_ptr->data(), tracy::PlotFormatType::Number, false, true, 0);
#endif

        buffers.push_back(SampleBuffer(*str_ptr, sample_range, sample_dis));

        sensors[*str_ptr] = std::move(sensor);
        if (config) {
            config(sensors[*str_ptr]);
        }
        sensors[*str_ptr].sample(std::move(
            async_sampling()
                .delivers_measurement_data_to(&tracy_sample)
                .stores_and_passes_context(std::make_tuple(str_ptr->data(), &buffers.back(), std::cref(do_buffer)))
                .samples_every(std::chrono::duration_cast<std::chrono::microseconds>(sample_dis).count())
                .using_resolution(timestamp_resolution::hundred_nanoseconds)
                .from_source(tinkerforge_sensor_source::power)));
    }

    return std::make_tuple(std::move(sensors), std::move(buffers));
}

template<typename T>
class SamplerCollection final {
public:
    SamplerCollection(std::chrono::milliseconds const& sample_range, std::chrono::milliseconds const& sample_dis,
        discard_func_t discard = nullptr, config_func_t<T> config = nullptr) {
        using namespace visus::power_overwhelming;

        auto const sensor_count = T::for_all(nullptr, 0);

        std::vector<T> tmp_sensors(sensor_count);
        T::for_all(tmp_sensors.data(), tmp_sensors.size());

        sensors_.reserve(sensor_count);
        buffers_.reserve(sensor_count);
        sensor_names_.reserve(sensor_count);

        for (auto& sensor : tmp_sensors) {
            sensor_names_.push_back(unmueller_string(sensor.name()));
            auto const& name = sensor_names_.back();
            if (discard) {
                if (discard(name))
                    continue;
            }

#ifdef MEGAMOL_USE_TRACY
            TracyPlotConfig(name.data(), tracy::PlotFormatType::Number, false, true, 0);
#endif

            auto& buffer = buffers_.emplace_back(name, sample_range, sample_dis);

            sensors_[name] = std::move(sensor);
            if (config) {
                config(sensors_[name]);
            }
            sensors_[name].sample(
                std::move(async_sampling()
                              .delivers_measurement_data_to(&tracy_sample)
                              .stores_and_passes_context(std::make_tuple(name.data(), &buffer, std::cref(do_buffer_)))
                              .samples_every(std::chrono::duration_cast<std::chrono::microseconds>(sample_dis).count())
                              .using_resolution(timestamp_resolution::hundred_nanoseconds)
                              .from_source(tinkerforge_sensor_source::power)));
        }
    }

    SamplerCollection(SamplerCollection const&) = delete;
    SamplerCollection& operator=(SamplerCollection const&) = delete;

    /*SamplerCollection(SamplerCollection&& rhs) noexcept
            : do_buffer_{std::exchange(rhs.do_buffer_, false)}
            , sensor_names_{std::exchange(rhs.sensor_names_, std::vector<std::string>{})}
            , buffers_{std::exchange(rhs.buffers_, buffers_t{})}
            , sensors_{std::exchange(rhs.sensors_, samplers_t<T>{})} {}

    SamplerCollection& operator=(SamplerCollection&& rhs) noexcept {
        if (this != std::addressof(rhs)) {
            sensors_ = std::exchange(rhs.sensors_, samplers_t<T>{});
            buffers_ = std::exchange(rhs.buffers_, buffers_t{});
        }
        return *this;
    }*/

    void SetSegmentRange(std::chrono::milliseconds const& range) {
        for (auto& b : buffers_) {
            b.SetSampleRange(range);
        }
    }

    void ResetBuffers() {
        for (auto& b : buffers_) {
            b.Clear();
        }
    }

    void WriteBuffers(std::filesystem::path const& path, MetaData const* meta) const {
        ParquetWriter(path, buffers_, meta);
    }

    void WriteBuffers(std::filesystem::path const& path, MetaData const* meta, std::string const& dataverse_path,
        std::string const& dataverse_doi, char const* dataverse_token, char& fin_signal) const {
        ParquetWriter(path, buffers_, meta);
        DataverseWriter(dataverse_path, dataverse_doi, path.string(), dataverse_token, fin_signal);
    }

    void StartRecording() {
        do_buffer_ = true;
    }

    void StopRecording() {
        do_buffer_ = false;
    }

private:
    bool do_buffer_ = false;
    std::vector<std::string> sensor_names_;
    power::buffers_t buffers_;
    power::samplers_t<T> sensors_;
};
} // namespace megamol::power

#endif
