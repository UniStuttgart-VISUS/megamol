/*
 * Power_Service.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

// search/replace Power_Service with your class name
// you should also delete the FAQ comments in these template files after you read and understood them
#include "Power_Service.hpp"

#ifdef MEGAMOL_USE_POWER

#include <filesystem>
#include <format>

#include "LuaApiResource.h"
#include "mmcore/LuaAPI.h"

#include "ModuleGraphSubscription.h"

#include "power/DataverseWriter.h"
#include "power/Tinkerforge.h"
#include "power/WriterUtility.h"

#include <nlohmann/json.hpp>

// local logging wrapper for your convenience until central MegaMol logger established
#include "mmcore/utility/log/Log.h"

static const std::string service_name = "[Power_Service] ";
static void log(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}

static void log_error(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}

static void log_warning(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(msg.c_str());
}


namespace megamol {
namespace frontend {

//bool Power_Service::init_sol_commands_ = true;

Power_Service::Power_Service() /*: sol_state_(nullptr)*/ {
    // init members to default states
}

Power_Service::~Power_Service() {
    // clean up raw pointers you allocated with new, which is bad practice and nobody does
}


bool Power_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    const auto conf = static_cast<Config*>(configPtr);
    auto const lpt = conf->lpt;
    write_to_files_ = conf->write_to_files;
    write_folder_ = conf->folder;

    main_trigger_ = std::make_shared<megamol::power::Trigger>(lpt);
    main_trigger_->RegisterPreTrigger("Power_Service_sb", std::bind(&Power_Service::sb_pre_trg, this));
    main_trigger_->RegisterInitTrigger("Power_Service_hmc", std::bind(&Power_Service::hmc_init_trg, this));
    main_trigger_->RegisterPostTrigger("Power_Service_sb", std::bind(&Power_Service::sb_post_trg, this));
    main_trigger_->RegisterPostTrigger("Power_Service_seg", std::bind(&Power_Service::seg_post_trg, this));
    main_trigger_->RegisterFinTrigger("Power_Service_hmc", std::bind(&Power_Service::hmc_fin_trg, this));
    main_trigger_->RegisterSignal(
        "Power_Service_trg", std::bind(&Power_Service::trigger_ts_signal, this, std::placeholders::_1));

    try {
        rtx_ = std::make_unique<megamol::power::RTXInstruments>(main_trigger_);
    } catch (std::exception& ex) {
        log_warning(std::format("RTX devices not available: {}", ex.what()));
        rtx_ = nullptr;
    }

    callbacks_.signal_high =
        std::bind(&megamol::power::ParallelPortTrigger::SetBit, main_trigger_->GetHandle(), 7, true);
    callbacks_.signal_low =
        std::bind(&megamol::power::ParallelPortTrigger::SetBit, main_trigger_->GetHandle(), 7, false);
    callbacks_.signal_frame = [&]() -> void {
        main_trigger_->GetHandle()->SetBit(7, true);
        main_trigger_->GetHandle()->SetBit(7, false);
    };

    m_providedResourceReferences = {{frontend_resources::PowerCallbacks_Req_Name, callbacks_}};

    m_requestedResourcesNames = {
        frontend_resources::LuaAPI_Req_Name, "RuntimeInfo", frontend_resources::MegaMolGraph_Req_Name};

    using namespace visus::power_overwhelming;

    auto emi_discard_func = [](std::string const& name) { return name.find("RAPL_Package0_PKG") == std::string::npos; };

    auto msr_discard_func = [](std::string const& name) { return name.find("msr/0/package") == std::string::npos; };

    auto adl_discard_func = [](std::string const& name) {
        static bool first = false;
        if (name.find("Radeon") == std::string::npos || first) {
            return true;
        } else {
            first = true;
            return false;
        }
    };

    auto tinker_config_func = [](tinkerforge_sensor& sensor) {
        sensor.reset();
        //sensor.resync_internal_clock_after(20);
        sensor.configure(
            sample_averaging::average_of_4, conversion_time::milliseconds_1_1, conversion_time::milliseconds_1_1);
    };

    std::function<std::string(std::string const&)> tinker_transform_func = [](std::string const& name) { return name; };
    nlohmann::json json_data;
    try {
        json_data = power::parse_json_file(conf->tinker_map_filename);
        tinker_transform_func = std::bind(&power::transform_tf_name, std::cref(json_data), std::placeholders::_1);
    } catch (...) {
        core::utility::log::Log::DefaultLog.WriteWarn(
            "[Power_Service] Could not parse Tinker json file. Using fallback.");
    }

    auto nvml_transform_func = [](std::string const& name) -> std::string { return "NVML[" + name + "]"; };

    auto adl_transform_func = [](std::string const& name) -> std::string { return "ADL[" + name + "]"; };

    auto msr_transform_func = [](std::string const& name) -> std::string { return "MSR[" + name + "]"; };

    std::unique_ptr<power::SamplerCollection<nvml_sensor>> nvml_samplers = nullptr;
    try {
        nvml_samplers = std::make_unique<power::SamplerCollection<nvml_sensor>>(
            std::chrono::milliseconds(600), std::chrono::milliseconds(10), nullptr, nullptr, nvml_transform_func);
    } catch (...) {
        nvml_samplers = nullptr;
        core::utility::log::Log::DefaultLog.WriteWarn("[Power_Service] No NVML sensors available");
    }
    std::unique_ptr<power::SamplerCollection<adl_sensor>> adl_samplers = nullptr;
    try {
        adl_samplers = std::make_unique<power::SamplerCollection<adl_sensor>>(std::chrono::milliseconds(600),
            std::chrono::milliseconds(10), adl_discard_func, nullptr, adl_transform_func);
    } catch (...) {
        adl_samplers = nullptr;
        core::utility::log::Log::DefaultLog.WriteWarn("[Power_Service] No ADL sensors available");
    }
    std::unique_ptr<power::SamplerCollection<emi_sensor>> emi_samplers = nullptr;
    try {
        emi_samplers = std::make_unique<power::SamplerCollection<emi_sensor>>(
            std::chrono::milliseconds(600), std::chrono::milliseconds(1), emi_discard_func);
    } catch (...) {
        emi_samplers = nullptr;
        core::utility::log::Log::DefaultLog.WriteWarn("[Power_Service] No EMI sensors available");
    }
    std::unique_ptr<power::SamplerCollection<msr_sensor>> msr_samplers = nullptr;
    if (!emi_samplers) {
        try {
            msr_samplers = std::make_unique<power::SamplerCollection<msr_sensor>>(std::chrono::milliseconds(600),
                std::chrono::milliseconds(1), msr_discard_func, nullptr, msr_transform_func);
        } catch (...) {
            msr_samplers = nullptr;
            core::utility::log::Log::DefaultLog.WriteWarn("[Power_Service] No MSR sensors available");
        }
    }
    std::unique_ptr<power::SamplerCollection<tinkerforge_sensor>> tinker_samplers = nullptr;
    try {
        tinker_samplers = std::make_unique<power::SamplerCollection<tinkerforge_sensor>>(std::chrono::milliseconds(600),
            std::chrono::milliseconds(10), nullptr, tinker_config_func, tinker_transform_func);
    } catch (...) {
        tinker_samplers = nullptr;
        core::utility::log::Log::DefaultLog.WriteWarn("[Power_Service] No Tinkerforge sensors available");
    }

    samplers = std::make_unique<power::SamplersCollectionWrapper>(std::move(nvml_samplers), std::move(adl_samplers),
        std::move(emi_samplers), std::move(msr_samplers), std::move(tinker_samplers));

    try {
        std::vector<hmc8015_sensor> hmc_tmp(hmc8015_sensor::for_all(nullptr, 0));
        hmc8015_sensor::for_all(hmc_tmp.data(), hmc_tmp.size());
        hmc_sensors_.reserve(hmc_tmp.size());
        for (auto& s : hmc_tmp) {
            s.synchronise_clock(true);
            s.voltage_range(instrument_range::explicitly, 300);
            s.current_range(instrument_range::explicitly, 5);
            s.log_behaviour(std::numeric_limits<float>::lowest(), log_mode::unlimited);
            auto const name = power::get_pwrowg_str(s, &hmc8015_sensor::instrument_name);
            hmc_sensors_[name] = std::move(s);
        }
    } catch (...) {
        core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: No HMC devices found");
    }

    log("initialized successfully");
    //return init(*static_cast<Config*>(configPtr));
    return true;
}

void Power_Service::close() {
    hmc_sensors_.clear();
    samplers.reset();
}

std::vector<FrontendResource>& Power_Service::getProvidedResources() {
    return m_providedResourceReferences;
}

const std::vector<std::string> Power_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void Power_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    // maybe we want to keep the list of requested resources
    this->m_requestedResourceReferences = resources;

    ri_ = &m_requestedResourceReferences[1].getResource<frontend_resources::RuntimeInfo>();
    megamolgraph_ptr_ =
        const_cast<core::MegaMolGraph*>(&m_requestedResourceReferences[2].getResource<core::MegaMolGraph>());

    meta_.runtime_libs = ri_->get_runtime_libraries();

    meta_.hardware_software_info.clear();
    meta_.hardware_software_info["OS"] = ri_->get_OS_info();
    meta_.hardware_software_info["SMBIOS"] = ri_->get_smbios_info();
    meta_.hardware_software_info["GPU"] = ri_->get_gpu_info();
    meta_.hardware_software_info["CPU"] = ri_->get_cpu_info();

    fill_lua_callbacks();
}

void Power_Service::updateProvidedResources() {}

void Power_Service::digestChangedRequestedResources() {}

void Power_Service::resetProvidedResources() {}

void Power_Service::preGraphRender() {}

void Power_Service::postGraphRender() {}

void Power_Service::fill_lua_callbacks() {
    auto luaApi = m_requestedResourceReferences[0].getResource<core::LuaAPI*>();

    luaApi->RegisterCallback("mmPowerSetup", "()", [&]() -> void {
        //setup_measurement();
        if (rtx_) {
            rtx_->ApplyConfigs(&meta_);
        }
    });

    luaApi->RegisterCallback("mmPowerMeasure", "(string path)", [&](std::string path) -> void {
        //start_measurement();
        reset_measurement();
        write_folder_ = path;
        meta_.project_file = megamolgraph_ptr_->Convenience().SerializeGraph();
        if (rtx_) {
            if (write_to_files_) {
                if (dataverse_key_) {
                    std::function<void(std::string)> dataverse_writer =
                        std::bind(&power::DataverseWriter, dataverse_config_.base_path, dataverse_config_.doi,
                            std::placeholders::_1, dataverse_key_->GetToken(), std::ref(sbroker_.Get(false)));
                    power::writer_func_t parquet_dataverse_writer =
                        std::bind(&power::wf_parquet_dataverse, std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3, std::placeholders::_4, dataverse_writer);
                    rtx_->StartMeasurement(path, {parquet_dataverse_writer, &power::wf_tracy_wrapper::wf_tracy}, &meta_,
                        sbroker_.Get(false));
                } else {
                    rtx_->StartMeasurement(
                        path, {&power::wf_parquet, &power::wf_tracy_wrapper::wf_tracy}, &meta_, sbroker_.Get(false));
                }
            } else {
                rtx_->StartMeasurement(path, {&power::wf_tracy_wrapper::wf_tracy}, &meta_, sbroker_.Get(false));
            }
        }
    });

    luaApi->RegisterCallback(
        "mmPowerWriteToFile", "(bool flag)", [&](bool const flag) -> void { write_to_files_ = flag; });

    luaApi->RegisterCallback("mmPowerSetLPTAddress", "(string address)",
        [&](std::string const address) -> void { main_trigger_->SetLPTAddress(address); });

    luaApi->RegisterCallback("mmPowerConfig", "(string path, int points, int count, int range_ms, int timeout_ms)",
        [&](std::string path, int points, int count, int range, int timeout) -> void {
            if (rtx_) {
                rtx_->UpdateConfigs(
                    path, points, count, std::chrono::milliseconds(range), std::chrono::milliseconds(timeout));
            }
            meta_.trigger_ts.reserve(count);
            reset_segment_range(std::chrono::milliseconds(range));
        });

    luaApi->RegisterCallback("mmPowerIsPending", "()", [&]() -> bool { return sbroker_.GetValue(); });

    luaApi->RegisterCallback("mmPowerSoftwareTrigger", "(bool set)", [&](bool set) -> void {
        if (rtx_) {
            rtx_->SetSoftwareTrigger(set);
        }
    });

    luaApi->RegisterCallback("mmPowerDataverseKey", "(string path_to_key)",
        [&](std::string path_to_key) -> void { dataverse_key_ = std::make_unique<power::CryptToken>(path_to_key); });

    luaApi->RegisterCallback("mmPowerDataverseDataset", "(string base_path, string doi)",
        [&](std::string base_path, std::string doi) -> void {
            dataverse_config_.base_path = base_path;
            dataverse_config_.doi = doi;
        });

    luaApi->RegisterCallback(
        "mmPowerRecipe", "(string name, string path)", [&](std::string name, std::string path) -> void {
            if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) {
                auto const size = std::filesystem::file_size(path);
                std::ifstream f(path);
                std::vector<char> data(size);
                f.read(data.data(), data.size());
                f.close();
                meta_.analysis_recipes[name] = std::string(data.begin(), data.end());
            }
        });
}


void Power_Service::write_sample_buffers(std::size_t seg_cnt) {
    auto const nvml_path = std::filesystem::path(write_folder_) / ("nvml_s" + std::to_string(seg_cnt) + ".parquet");
    auto const adl_path = std::filesystem::path(write_folder_) / ("adl_s" + std::to_string(seg_cnt) + ".parquet");
    auto const emi_path = std::filesystem::path(write_folder_) / ("emi_s" + std::to_string(seg_cnt) + ".parquet");
    auto const msr_path = std::filesystem::path(write_folder_) / ("msr_s" + std::to_string(seg_cnt) + ".parquet");
    auto const tinker_path = std::filesystem::path(write_folder_) / ("tinker_s" + std::to_string(seg_cnt) + ".parquet");

    auto const tpl = std::make_tuple(power::SamplersCollectionWrapper::nvml_path_t{nvml_path},
        power::SamplersCollectionWrapper::adl_path_t{adl_path}, power::SamplersCollectionWrapper::emi_path_t{emi_path},
        power::SamplersCollectionWrapper::msr_path_t{msr_path},
        power::SamplersCollectionWrapper::tinker_path_t{tinker_path});

    if (dataverse_key_) {
        samplers->visit<power::MetaData const*, std::string const&, std::string const&, char const*, char&>(
            &power::ISamplerCollection::WriteBuffers, tpl, &meta_, dataverse_config_.base_path, dataverse_config_.doi,
            dataverse_key_->GetToken(), sbroker_.Get(false));
    } else {
        samplers->visit<power::MetaData const*>(&power::ISamplerCollection::WriteBuffers, tpl, &meta_);
    }

    samplers->visit(&power::ISamplerCollection::ResetBuffers);
}


void Power_Service::reset_segment_range(std::chrono::milliseconds const& range) {
    auto const [trg_prefix, trg_postfix, trg_wait] = power::get_trigger_timings(range);
    samplers->visit<std::chrono::milliseconds const&>(
        &power::ISamplerCollection::SetSegmentRange, trg_prefix + trg_postfix + std::chrono::seconds(1));
}

void Power_Service::reset_measurement() {
    sbroker_.Reset();
    seg_cnt_ = 0;
    meta_.trigger_ts.clear();
}

} // namespace frontend
} // namespace megamol

#endif
