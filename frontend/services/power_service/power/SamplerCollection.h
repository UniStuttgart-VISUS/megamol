#pragma once

#ifdef MEGAMOL_USE_POWER

#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "DataverseWriter.h"
#include "MetaData.h"
#include "ParquetWriter.h"
#include "SamplerUtility.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

using namespace visus::power_overwhelming;

namespace megamol::power {
/// <summary>
/// Interface definition for the <see cref="SamplerCollection"/> class.
/// </summary>
struct ISamplerCollection {
    virtual void SetSegmentRange(std::chrono::milliseconds const& range) = 0;

    virtual void ResetBuffers() = 0;

    virtual void WriteBuffers(std::filesystem::path const& path, MetaData const* meta) const = 0;

    virtual void WriteBuffers(std::filesystem::path const& path, MetaData const* meta,
        std::string const& dataverse_path, std::string const& dataverse_doi, char const* dataverse_token,
        char& fin_signal) const = 0;

    virtual void Reset() = 0;

    virtual void StartRecording() = 0;

    virtual void StopRecording() = 0;

    virtual ~ISamplerCollection() = default;
};

/// <summary>
/// Class holding all samplers from a specific power-overwhelming sampler type.
/// Holds the corresponding sample buffers, as well.
/// </summary>
/// <typeparam name="T">Power-overwhelming sampler type.</typeparam>
template<typename T>
class SamplerCollection final : public ISamplerCollection {
public:
    /// <summary>
    /// Opens and configures all sensors of a specific power-overwhelming sampler type.
    /// </summary>
    /// <param name="sample_range">Time range of a measurement segment in milliseconds.</param>
    /// <param name="sample_dis">Intended time distance between each sample in milliseconds.</param>
    /// <param name="discard">Power-overwhelming name-based discard function for individual sensors.</param>
    /// <param name="config">Config function for each sensor.</param>
    /// <param name="transform">Transform function that converts the power-overwhelming name of a sensor into a MegaMol name.</param>
    SamplerCollection(std::chrono::milliseconds const& sample_range, std::chrono::milliseconds const& sample_dis,
        discard_func_t discard = nullptr, config_func_t<T> config = nullptr, transform_func_t transform = nullptr) {
        using namespace visus::power_overwhelming;

        auto const sensor_count = T::for_all(nullptr, 0);
        if (sensor_count == 0) {
            throw std::runtime_error("no sensors found");
        }

        std::vector<T> tmp_sensors(sensor_count);
        T::for_all(tmp_sensors.data(), tmp_sensors.size());

        sensors_.reserve(sensor_count);
        buffers_.reserve(sensor_count);
        sensor_names_.reserve(sensor_count);

        for (auto& sensor : tmp_sensors) {
            auto tmp_name = unmueller_string(sensor.name());
            if (discard) {
                if (discard(tmp_name))
                    continue;
            }
            try {
                if (transform) {
                    tmp_name = transform(tmp_name);
                }
            } catch (...) {
                continue;
            }
            sensor_names_.push_back(tmp_name);
            auto const& name = sensor_names_.back();

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
                              .delivers_measurement_data_to(&sample_func)
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

    /// <summary>
    /// Sets the time range of a measurement segment for potential buffer realloc.
    /// </summary>
    /// <param name="range">Time range in milliseconds.</param>
    void SetSegmentRange(std::chrono::milliseconds const& range) override {
        for (auto& b : buffers_) {
            b.SetSampleRange(range);
        }
    }

    /// <summary>
    /// Clears content of the sample buffers.
    /// </summary>
    void ResetBuffers() override {
        for (auto& b : buffers_) {
            b.Clear();
        }
    }

    /// <summary>
    /// Writes sample buffers as Parquet file.
    /// </summary>
    /// <param name="path">Path to the output file.</param>
    /// <param name="meta">Meta information that should be embedded into the output file.</param>
    void WriteBuffers(std::filesystem::path const& path, MetaData const* meta) const override {
        ParquetWriter(path, buffers_, meta);
    }

    /// <summary>
    /// Writes sample buffers as Parquet file and, additionally, sends their content to a Dataverse dataset.
    /// </summary>
    /// <param name="path">Path to the output file.</param>
    /// <param name="meta">Meta information that should be embedded into the output file.</param>
    /// <param name="dataverse_path">Path of the Dataverse API endpoint.</param>
    /// <param name="dataverse_doi">DOI of the dataset in the Dataverse.</param>
    /// <param name="dataverse_token">Dataverse API token.</param>
    /// <param name="fin_signal">Inout param that will be set true
    /// once the data transfer operation is finished (<see cref="SignalBroker"/>).</param>
    void WriteBuffers(std::filesystem::path const& path, MetaData const* meta, std::string const& dataverse_path,
        std::string const& dataverse_doi, char const* dataverse_token, char& fin_signal) const override {
        WriteBuffers(path, meta);
        DataverseWriter(dataverse_path, dataverse_doi, path.string(), dataverse_token, fin_signal);
    }

    /// <summary>
    /// Specific reset for tinkerforge sensors.
    /// Resyncs the internal clocks of the sensors with the system clock.
    /// </summary>
    void Reset() override {
        if constexpr (std::is_same_v<T, tinkerforge_sensor>) {
            for (auto& [n, s] : sensors_) {
                s.resync_internal_clock();
            }
        }
    }

    /// <summary>
    /// Starts the recording of samples into the sample buffers.
    /// </summary>
    void StartRecording() override {
        do_buffer_ = true;
    }

    /// <summary>
    /// Stops the recording of samples into the sample buffers.
    /// </summary>
    void StopRecording() override {
        do_buffer_ = false;
    }

private:
    bool do_buffer_ = false;
    std::vector<std::string> sensor_names_;
    power::buffers_t buffers_;
    power::samplers_t<T> sensors_;
};

/// <summary>
/// Wrapper for the <see cref="SamplerCollection"/> class.
/// Abstraction over the template type.
/// Holds collections of all power-overwhelming sensor types.
/// </summary>
class SamplersCollectionWrapper final {
public:
    struct base_path_t {
        explicit base_path_t(std::filesystem::path const& path) : path_(path) {}
        operator std::filesystem::path const&() const {
            return path_;
        }

    protected:
        ~base_path_t() = default;

    private:
        std::filesystem::path const& path_;
    };

    struct nvml_path_t final : public base_path_t {
        explicit nvml_path_t(std::filesystem::path const& path) : base_path_t(path) {}
    };
    struct adl_path_t final : public base_path_t {
        explicit adl_path_t(std::filesystem::path const& path) : base_path_t(path) {}
    };
    struct emi_path_t final : public base_path_t {
        explicit emi_path_t(std::filesystem::path const& path) : base_path_t(path) {}
    };
    struct msr_path_t final : public base_path_t {
        explicit msr_path_t(std::filesystem::path const& path) : base_path_t(path) {}
    };
    struct tinker_path_t final : public base_path_t {
        explicit tinker_path_t(std::filesystem::path const& path) : base_path_t(path) {}
    };

    /// <summary>
    /// Signature for a <see cref="SamplerCollection"/> function visitor.
    /// </summary>
    /// <typeparam name="...Ts">Types of the parameter set to pass to the <see cref="SamplerCollection"/> function.</typeparam>
    template<typename... Ts>
    using to_invoke_f = void (ISamplerCollection::*)(Ts...);
    /// <summary>
    /// Signature for a <see cref="SamplerCollection"/> const function visitor.
    /// </summary>
    /// <typeparam name="...Ts">Types of the parameter set to be pass to the <see cref="SamplerCollection"/> function.</typeparam>
    template<typename... Ts>
    using to_invoke_f_c = void (ISamplerCollection::*)(Ts...) const;

    /// <summary>
    /// Visits all available power-overwhelming sensors with a specified <see cref="SamplerCollection"/> function.
    /// </summary>
    /// <typeparam name="...Ts">Types of the parameter set to be passed to the <see cref="SamplerCollection"/> function.</typeparam>
    /// <param name="to_invoke">The <see cref="SamplerCollection"/> function to be visited.</param>
    /// <param name="...args">Parameter set to be passed to the <see cref="SamplerCollection"/> function.</param>
    template<typename... Ts>
    void visit(to_invoke_f<Ts...> to_invoke, Ts... args) {
        if (nvml_samplers_)
            (*nvml_samplers_.*to_invoke)(std::forward<Ts>(args)...);
        if (adl_samplers_)
            (*adl_samplers_.*to_invoke)(std::forward<Ts>(args)...);
        if (emi_samplers_)
            (*emi_samplers_.*to_invoke)(std::forward<Ts>(args)...);
        if (msr_samplers_)
            (*msr_samplers_.*to_invoke)(std::forward<Ts>(args)...);
        if (tinker_samplers_)
            (*tinker_samplers_.*to_invoke)(std::forward<Ts>(args)...);
    }

    /// <summary>
    /// Visits all available power-overwhelming sensors with a specified <see cref="SamplerCollection"/> const function.
    /// Specialized for the output function with the file path tuple.
    /// </summary>
    /// <typeparam name="...Ts">Types of the parameter set to be passed to the <see cref="SamplerCollection"/> function.</typeparam>
    /// <param name="to_invoke">The <see cref="SamplerCollection"/> function to be visited.</param>
    /// <param name="paths">Set of file paths for the specific power-overwhelming sensor types.</param>
    /// <param name="...args">Parameter set to be passed to the <see cref="SamplerCollection"/> function.</param>
    template<typename... Ts>
    void visit(to_invoke_f_c<std::filesystem::path const&, Ts...> to_invoke,
        std::tuple<nvml_path_t, adl_path_t, emi_path_t, msr_path_t, tinker_path_t> const& paths, Ts... args) {
        if (nvml_samplers_)
            (*nvml_samplers_.*to_invoke)(std::get<0>(paths), std::forward<Ts>(args)...);
        if (adl_samplers_)
            (*adl_samplers_.*to_invoke)(std::get<1>(paths), std::forward<Ts>(args)...);
        if (emi_samplers_)
            (*emi_samplers_.*to_invoke)(std::get<2>(paths), std::forward<Ts>(args)...);
        if (msr_samplers_)
            (*msr_samplers_.*to_invoke)(std::get<3>(paths), std::forward<Ts>(args)...);
        if (tinker_samplers_)
            (*tinker_samplers_.*to_invoke)(std::get<4>(paths), std::forward<Ts>(args)...);
    }

    /// <summary>
    /// Ctor.
    /// </summary>
    SamplersCollectionWrapper() = default;

    /// <summary>
    /// Specialized ctor taking ownership of <see cref="SamplerCollection"/> for all power-overwhelming sensor types.
    /// </summary>
    /// <param name="nvml_samplers"><see cref="SamplerCollection"/> for NVML sensors.</param>
    /// <param name="adl_samplers"><see cref="SamplerCollection"/> for ADL sensors.</param>
    /// <param name="emi_samplers"><see cref="SamplerCollection"/> for EMI sensors.</param>
    /// <param name="msr_samplers"><see cref="SamplerCollection"/> for MSR sensors.</param>
    /// <param name="tinker_samplers"><see cref="SamplerCollection"/> for Tinkerforge sensors.</param>
    SamplersCollectionWrapper(std::unique_ptr<SamplerCollection<nvml_sensor>>&& nvml_samplers,
        std::unique_ptr<SamplerCollection<adl_sensor>>&& adl_samplers,
        std::unique_ptr<SamplerCollection<emi_sensor>>&& emi_samplers,
        std::unique_ptr<SamplerCollection<msr_sensor>>&& msr_samplers,
        std::unique_ptr<SamplerCollection<tinkerforge_sensor>>&& tinker_samplers)
            : nvml_samplers_{std::move(nvml_samplers)}
            , adl_samplers_{std::move(adl_samplers)}
            , emi_samplers_{std::move(emi_samplers)}
            , msr_samplers_{std::move(msr_samplers)}
            , tinker_samplers_{std::move(tinker_samplers)} {}

    /// <summary>
    /// Move ctor.
    /// </summary>
    SamplersCollectionWrapper(SamplersCollectionWrapper&& rhs) noexcept
            : nvml_samplers_{std::exchange(rhs.nvml_samplers_, nullptr)}
            , adl_samplers_{std::exchange(rhs.adl_samplers_, nullptr)}
            , emi_samplers_{std::exchange(rhs.emi_samplers_, nullptr)}
            , msr_samplers_{std::exchange(rhs.msr_samplers_, nullptr)}
            , tinker_samplers_{std::exchange(rhs.tinker_samplers_, nullptr)} {}

    /// <summary>
    /// Move assignment.
    /// </summary>
    SamplersCollectionWrapper& operator=(SamplersCollectionWrapper&& rhs) noexcept {
        nvml_samplers_ = std::exchange(rhs.nvml_samplers_, nullptr);
        adl_samplers_ = std::exchange(rhs.adl_samplers_, nullptr);
        emi_samplers_ = std::exchange(rhs.emi_samplers_, nullptr);
        msr_samplers_ = std::exchange(rhs.msr_samplers_, nullptr);
        tinker_samplers_ = std::exchange(rhs.tinker_samplers_, nullptr);

        return *this;
    }

    /// <summary>
    /// Dtor.
    /// </summary>
    ~SamplersCollectionWrapper() = default;

private:
    std::unique_ptr<SamplerCollection<nvml_sensor>> nvml_samplers_ = nullptr;

    std::unique_ptr<SamplerCollection<adl_sensor>> adl_samplers_ = nullptr;

    std::unique_ptr<SamplerCollection<emi_sensor>> emi_samplers_ = nullptr;

    std::unique_ptr<SamplerCollection<msr_sensor>> msr_samplers_ = nullptr;

    std::unique_ptr<SamplerCollection<tinkerforge_sensor>> tinker_samplers_ = nullptr;
};
} // namespace megamol::power

#endif // MEGAMOL_USE_POWER
