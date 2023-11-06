#pragma once

#include "SamplerUtility.h"

#ifdef MEGAMOL_USE_POWER

using namespace visus::power_overwhelming;

namespace megamol::power {
struct ISamplerCollection {
    virtual void SetSegmentRange(std::chrono::milliseconds const& range) = 0;

    virtual void ResetBuffers() = 0;

    virtual void WriteBuffers(std::filesystem::path const& path, MetaData const* meta) const = 0;

    virtual void WriteBuffers(std::filesystem::path const& path, MetaData const* meta,
        std::string const& dataverse_path, std::string const& dataverse_doi, char const* dataverse_token,
        char& fin_signal) const = 0;

    virtual void StartRecording() = 0;

    virtual void StopRecording() = 0;

    virtual ~ISamplerCollection() = default;
};

template<typename T>
class SamplerCollection final : public ISamplerCollection {
public:
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
            if (transform) {
                tmp_name = transform(tmp_name);
            }
            sensor_names_.push_back(tmp_name);
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

    void SetSegmentRange(std::chrono::milliseconds const& range) override {
        for (auto& b : buffers_) {
            b.SetSampleRange(range);
        }
    }

    void ResetBuffers() override {
        for (auto& b : buffers_) {
            b.Clear();
        }
    }

    void WriteBuffers(std::filesystem::path const& path, MetaData const* meta) const override {
        ParquetWriter(path, buffers_, meta);
    }

    void WriteBuffers(std::filesystem::path const& path, MetaData const* meta, std::string const& dataverse_path,
        std::string const& dataverse_doi, char const* dataverse_token, char& fin_signal) const override {
        ParquetWriter(path, buffers_, meta);
        DataverseWriter(dataverse_path, dataverse_doi, path.string(), dataverse_token, fin_signal);
    }

    void StartRecording() override {
        do_buffer_ = true;
    }

    void StopRecording() override {
        do_buffer_ = false;
    }

private:
    bool do_buffer_ = false;
    std::vector<std::string> sensor_names_;
    power::buffers_t buffers_;
    power::samplers_t<T> sensors_;
};

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

    template<typename... Ts>
    using to_invoke_f = void (ISamplerCollection::*)(Ts...);
    template<typename... Ts>
    using to_invoke_f_c = void (ISamplerCollection::*)(Ts...) const;

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

    SamplersCollectionWrapper() = default;

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

    SamplersCollectionWrapper(SamplersCollectionWrapper&& rhs) noexcept
            : nvml_samplers_{std::exchange(rhs.nvml_samplers_, nullptr)}
            , adl_samplers_{std::exchange(rhs.adl_samplers_, nullptr)}
            , emi_samplers_{std::exchange(rhs.emi_samplers_, nullptr)}
            , msr_samplers_{std::exchange(rhs.msr_samplers_, nullptr)}
            , tinker_samplers_{std::exchange(rhs.tinker_samplers_, nullptr)} {}

    SamplersCollectionWrapper& operator=(SamplersCollectionWrapper&& rhs) noexcept {
        nvml_samplers_ = std::exchange(rhs.nvml_samplers_, nullptr);
        adl_samplers_ = std::exchange(rhs.adl_samplers_, nullptr);
        emi_samplers_ = std::exchange(rhs.emi_samplers_, nullptr);
        msr_samplers_ = std::exchange(rhs.msr_samplers_, nullptr);
        tinker_samplers_ = std::exchange(rhs.tinker_samplers_, nullptr);

        return *this;
    }

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
