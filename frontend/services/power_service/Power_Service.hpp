/*
 * Power_Service.hpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#ifdef MEGAMOL_USE_POWER

#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>

#include "mmcore/MegaMolGraph.h"

#include "AbstractFrontendService.hpp"

#include "RuntimeInfo.h"
#include "PowerCallbacks.h"

#include "power/CryptToken.h"
#include "power/MetaData.h"
#include "power/RTXInstruments.h"
#include "power/SamplerCollection.h"
#include "power/SignalBroker.h"
#include "power/Trigger.h"
#include "power/Utility.h"

#include <power_overwhelming/hmc8015_sensor.h>
#include <power_overwhelming/rtx_instrument.h>


namespace megamol {
namespace frontend {

inline std::string gen_hmc_filename(unsigned int const cnt) {
    return std::string("PWR_") + std::to_string(cnt);
}

inline std::string gen_hmc_filename() {
    return std::string("PWR");
}

inline std::string gen_hmc_filename(std::string const& fix) {
    return fix + std::string(".CSV");
}

class Power_Service final : public AbstractFrontendService {
public:
    struct Config {
        std::string lpt = "lpt1";
        bool write_to_files = false;
        std::string folder = "./";
        std::string tinker_map_filename = "./tinkerforge.json";
    };

    std::string serviceName() const override {
        return "Power_Service";
    }

    Power_Service();

    ~Power_Service();

    bool init(void* configPtr) override;

    void close() override;

    std::vector<FrontendResource>& getProvidedResources() override;

    const std::vector<std::string> getRequestedResourceNames() const override;

    void setRequestedResources(std::vector<FrontendResource> resources) override;

    void updateProvidedResources() override;

    void digestChangedRequestedResources() override;

    void resetProvidedResources() override;

    void preGraphRender() override;

    void postGraphRender() override;

private:
    std::vector<FrontendResource> m_providedResourceReferences;

    std::vector<std::string> m_requestedResourcesNames;

    std::vector<FrontendResource> m_requestedResourceReferences;

    frontend_resources::PowerCallbacks callbacks_;

    std::vector<visus::power_overwhelming::rtx_instrument> rtx_instr_;

    std::unique_ptr<power::SamplersCollectionWrapper> samplers;

    std::unordered_map<std::string, visus::power_overwhelming::hmc8015_sensor> hmc_sensors_;

    void fill_lua_callbacks();

    void reset_segment_range(std::chrono::milliseconds const& range);

    void reset_measurement();

    void write_sample_buffers(std::size_t seg_cnt);

    void seg_post_trg() {
        ++seg_cnt_;
    }

    void sb_pre_trg() {
        do_buffer_ = true;

        samplers->visit(&power::ISamplerCollection::Reset);
        samplers->visit(&power::ISamplerCollection::StartRecording);
    }

    void sb_post_trg() {
        do_buffer_ = false;

        samplers->visit(&power::ISamplerCollection::StopRecording);

        if (write_to_files_) {
            write_sample_buffers(seg_cnt_);
        }
    }

    void hmc_init_trg() {
        for (auto& [n, s] : hmc_sensors_) {
            s.log_file(gen_hmc_filename().c_str(), true, false);
            auto const path_size = s.log_file((char*) nullptr, 0);
            std::string path;
            path.resize(path_size);
            s.log_file(path.data(), path.size());
            if (!path.empty()) {
                path.resize(path.size() - 1);
            }
            std::regex reg(R"(^\"(\w+)\", INT$)");
            std::smatch match;
            if (std::regex_match(path, match, reg)) {
                hmc_paths_[n] = gen_hmc_filename(match[1]);
            }
        }
        // TODO: maybe need sleep here
        for (auto& [n, s] : hmc_sensors_) {
            s.reset_integrator();
            s.start_integrator();
            s.log(true);
        }
    }

    void hmc_fin_trg() {
        for (auto& [n, s] : hmc_sensors_) {
            s.log(false);
            s.stop_integrator();
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (write_to_files_) {
            for (auto& [n, s] : hmc_sensors_) {
                try {
                    //auto blob = s.copy_file_from_instrument(gen_hmc_filename(hmc_m).c_str(), false);
                    auto blob = s.copy_file_from_instrument(hmc_paths_[n].c_str(), false);

                    auto hmc_file = std::string(blob.begin(), blob.end());

                    auto const [meta_str, csv_str, vals] = power::parse_hmc_file(hmc_file);

                    auto const csv_path = power::create_full_path(write_folder_, n, ".csv");
                    std::ofstream file(csv_path.string(), std::ios::binary);
                    file.write(csv_str.data(), csv_str.size());
                    file.close();
                    auto const meta_path = power::create_full_path(write_folder_, n, ".meta.csv");
                    file.open(meta_path.string(), std::ios::binary);
                    file.write(meta_str.data(), meta_str.size());
                    file.close();

                    auto const parquet_path = power::create_full_path(write_folder_, n);
                    power::ParquetWriter(parquet_path, vals, &meta_);
                } catch (...) {
                    core::utility::log::Log::DefaultLog.WriteError(
                        "HMC: failed to fetch data {}", hmc_paths_[n].back());
                }
            }
        }
        for (auto& [n, s] : hmc_sensors_) {
            s.delete_file_from_instrument(hmc_paths_[n].c_str(), false);
            hmc_paths_[n].clear();
        }
    }

    void trigger_ts_signal(power::filetime_dur_t const& ts) {
        meta_.trigger_ts.push_back(ts);
    }

    bool write_to_files_ = false;

    std::string write_folder_ = "./";

    std::unique_ptr<megamol::power::RTXInstruments> rtx_;

    bool do_buffer_ = false;

    std::size_t seg_cnt_ = 0;

    std::shared_ptr<megamol::power::Trigger> main_trigger_;

    frontend_resources::RuntimeInfo const* ri_;

    power::MetaData meta_;

    std::unique_ptr<power::CryptToken> dataverse_key_ = nullptr;

    struct dataverse_config_s {
        std::string base_path;
        std::string doi;
    } dataverse_config_;

    power::SignalBroker sbroker_;

    core::MegaMolGraph* megamolgraph_ptr_;

    std::unordered_map<std::string, std::string> hmc_paths_;
};

} // namespace frontend
} // namespace megamol

#endif
