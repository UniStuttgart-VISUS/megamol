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

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

#include "LuaCallbacksCollection.h"

#include "ModuleGraphSubscription.h"

#include "power/DataverseWriter.h"
#include "power/WriterUtility.h"

// local logging wrapper for your convenience until central MegaMol logger established
#include "mmcore/utility/log/Log.h"

static const std::string service_name = "Power_Service: ";
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

    //sol_state_.open_libraries(sol::lib::base);

    //visus::power_overwhelming::sol_expressions(sol_state_, values_map_);

    const auto conf = static_cast<Config*>(configPtr);
    auto const lpt = conf->lpt;
    str_cont_ = conf->str_container;
    write_to_files_ = conf->write_to_files;
    write_folder_ = conf->folder;

    main_trigger_ = std::make_shared<megamol::power::Trigger>(lpt);
    main_trigger_->RegisterPreTrigger("Power_Service", std::bind(&Power_Service::sb_pre_trg, this));
    //main_trigger_->RegisterSignal("Power_Service", std::bind(&Power_Service::sb_sgn_trg, this, std::placeholders::_1));
    main_trigger_->RegisterPostTrigger("Power_Service", std::bind(&Power_Service::sb_post_trg, this));
    main_trigger_->RegisterPostTrigger("Power_Service", std::bind(&Power_Service::seg_post_trg, this));

    try {
        rtx_ = std::make_unique<megamol::power::RTXInstruments>(main_trigger_);
    } catch (std::exception& ex) {
        log_error(std::format("RTX devices not available: {}", ex.what()));
        rtx_ = nullptr;
    }

    callbacks_.signal_high =
        std::bind(&megamol::power::ParallelPortTrigger::SetBit, main_trigger_->GetHandle(), 7, true);
    callbacks_.signal_low =
        std::bind(&megamol::power::ParallelPortTrigger::SetBit, main_trigger_->GetHandle(), 7, false);
    callbacks_.signal_frame = [&]() -> void {
        /*auto m_func = [&]() {
            trigger_->SetBit(7, true);
            if (have_triggered_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                have_triggered_ = false;
            }
            trigger_->SetBit(7, false);
        };
        auto t = std::thread(m_func);
        t.detach();*/
        main_trigger_->GetHandle()->SetBit(7, true);
        main_trigger_->GetHandle()->SetBit(7, false);
    };

    m_providedResourceReferences = {{frontend_resources::PowerCallbacks_Req_Name, callbacks_}};

    m_requestedResourcesNames = {"RegisterLuaCallbacks", "RuntimeInfo", frontend_resources::MegaMolGraph_Req_Name};

    using namespace visus::power_overwhelming;

    auto emi_discard_func = [](std::string const& name) { return name.find("RAPL_Package0_PKG") == std::string::npos; };

    auto msr_discard_func = [](std::string const& name) { return name.find("msr/0/package") == std::string::npos; };

    auto tinker_config_func = [](tinkerforge_sensor& sensor) {
        sensor.reset();
        sensor.configure(
            sample_averaging::average_of_4, conversion_time::microseconds_588, conversion_time::microseconds_588);
    };

    std::unique_ptr<power::SamplerCollection<nvml_sensor>> nvml_samplers = nullptr;
    try {
        nvml_samplers = std::make_unique<power::SamplerCollection<nvml_sensor>>(
            std::chrono::milliseconds(600), std::chrono::milliseconds(1));
    } catch (...) {
        nvml_samplers = nullptr;
    }
    /*std::tie(nvml_sensors_, nvml_buffers_) = megamol::power::InitSampler<nvml_sensor>(
        std::chrono::milliseconds(600), std::chrono::milliseconds(1), str_cont_, do_buffer_);*/
    std::unique_ptr<power::SamplerCollection<adl_sensor>> adl_samplers = nullptr;
    try {
        adl_samplers = std::make_unique<power::SamplerCollection<adl_sensor>>(
            std::chrono::milliseconds(600), std::chrono::milliseconds(1));
    } catch (...) {
        adl_samplers = nullptr;
    }
    /*std::tie(adl_sensors_, adl_buffers_) = megamol::power::InitSampler<adl_sensor>(
        std::chrono::milliseconds(600), std::chrono::milliseconds(1), str_cont_, do_buffer_);*/
    std::unique_ptr<power::SamplerCollection<emi_sensor>> emi_samplers = nullptr;
    try {
        emi_samplers = std::make_unique<power::SamplerCollection<emi_sensor>>(
            std::chrono::milliseconds(600), std::chrono::milliseconds(1), emi_discard_func);
    } catch (...) {
        emi_samplers = nullptr;
    }
    /*std::tie(emi_sensors_, emi_buffers_) = megamol::power::InitSampler<emi_sensor>(
        std::chrono::milliseconds(600), std::chrono::milliseconds(1), str_cont_, do_buffer_, emi_discard_func);*/
    std::unique_ptr<power::SamplerCollection<msr_sensor>> msr_samplers = nullptr;
    if (!emi_samplers) {
        try {
            msr_samplers = std::make_unique<power::SamplerCollection<msr_sensor>>(
                std::chrono::milliseconds(600), std::chrono::milliseconds(1), msr_discard_func);
        } catch (...) {
            msr_samplers = nullptr;
        }
        /*std::tie(msr_sensors_, msr_buffers_) = megamol::power::InitSampler<msr_sensor>(
            std::chrono::milliseconds(600), std::chrono::milliseconds(1), str_cont_, do_buffer_, msr_discard_func);*/
    }
    std::unique_ptr<power::SamplerCollection<tinkerforge_sensor>> tinker_samplers = nullptr;
    try {
        tinker_samplers = std::make_unique<power::SamplerCollection<tinkerforge_sensor>>(
            std::chrono::milliseconds(600), std::chrono::milliseconds(5), nullptr, tinker_config_func);
    } catch (...) {
        tinker_samplers = nullptr;
    }
    /*std::tie(tinker_sensors_, tinker_buffers_) =
        megamol::power::InitSampler<tinkerforge_sensor>(std::chrono::milliseconds(600), std::chrono::milliseconds(5),
            str_cont_, do_buffer_, nullptr, tinker_config_func);*/

    samplers = power::SamplersCollectionWrapper(std::move(nvml_samplers), std::move(adl_samplers),
        std::move(emi_samplers), std::move(msr_samplers), std::move(tinker_samplers));

    try {
        std::vector<hmc8015_sensor> hmc_tmp(hmc8015_sensor::for_all(nullptr, 0));
        hmc8015_sensor::for_all(hmc_tmp.data(), hmc_tmp.size());
        hmc_sensors_.reserve(hmc_tmp.size());
        for (auto& s : hmc_tmp) {
            s.synchronise_clock();
            s.voltage_range(instrument_range::explicitly, 300);
            s.current_range(instrument_range::explicitly, 5);
            s.log_behaviour(std::numeric_limits<float>::lowest(), log_mode::unlimited);
            auto const name = power::unmueller_string(s.name());
            hmc_sensors_[name] = std::move(s);
        }
    } catch (...) {
        core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: No HMC devices found");
    }

    //return init(*static_cast<Config*>(configPtr));
    return true;
}

bool Power_Service::init(const Config& config) {
    // initialize your service and its provided resources using config parameters
    // for now, you dont need to worry about your service beeing initialized or closed multiple times
    // init() and close() only get called once in the lifetime of each service object
    // but maybe more instances of your service will get created? this may be relevant for central resources you manage (like libraries, network connections).

    /*if (init_failed) {
        log_error("failed initialization because");
        return false;
    }*/
    log("initialized successfully");
    return true;
}

void Power_Service::close() {
    // close libraries or APIs you manage
    // wrap up resources your service provides, but don not depend on outside resources to be available here
    // after this, at some point only the destructor of your service gets called
    /*nvml_sensors_.clear();
    emi_sensors_.clear();
    msr_sensors_.clear();
    tinker_sensors_.clear();
    adl_sensors_.clear();*/
    hmc_sensors_.clear();
}

std::vector<FrontendResource>& Power_Service::getProvidedResources() {
    //this->m_providedResource1 = MyProvidedResource_1{...};
    //this->m_providedResource2 = MyProvidedResource_2{...};
    //this->m_providedResource3 = MyProvidedResource_3{...};

    //this->m_providedResourceReferences = {// construct std::vector
    //    {"MyProvidedResource_1",
    //        m_providedResource1}, // constructor FrontendResource{"MyProvidedResource_1", m_providedResource1}
    //    {"MyProvidedResource_2", m_providedResource2 /*reference to resource gets passed around*/},
    //    {"MyProvidedResource_3" /*resources are identified using unique names in the system*/, m_providedResource3}};

    return m_providedResourceReferences;
}

const std::vector<std::string> Power_Service::getRequestedResourceNames() const {
    // since this function should not change the state of the service
    // you should assign your requested resource names in init()
    /*this->m_requestedResourcesNames = {"ExternalResource_1", "ExternalResource_2"};*/

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

void Power_Service::updateProvidedResources() {
    // update resources we provide to others with new available data

    //this->m_providedResource1.update();
    //this->m_providedResource2 = MyProvidedResource_2{new_data()};

    //// deleting resources others may be using is not good
    //// you need to guarantee that your resource objects are alive and usable until your close() gets called
    //delete this->m_providedResource3; // DONT DO THIS
}

void Power_Service::digestChangedRequestedResources() {
    //digest_changes(*this->m_externalResource_1_ptr); // not that the pointer should never become invalid by design
    //digest_changes(*this->m_externalResource_2_ptr); // not that the pointer should never become invalid by design

    //// FrontendResource::getResource<>() returns CONST references. if you know what you are doing you may modify resources that are not yours.
    //modify_resource(const_cast<ExternalResource_1&>(resources[0].getResource<ExternalResource_1>()));

    //if (need_to_shutdown)
    //    this->setShutdown();
}

void Power_Service::resetProvidedResources() {
    // this gets called at the end of the main loop iteration
    // since the current resources state should have been handled in this frame already
    // you may clean up resources whose state is not needed for the next iteration
    // e.g. m_keyboardEvents.clear();
    // network_traffic_buffer.reset_to_empty();
}

void Power_Service::preGraphRender() {
    // this gets called right before the graph is told to render something
    // e.g. you can start a start frame timer here

    // rendering via MegaMol View is called after this function finishes
    // in the end this calls the equivalent of ::mmcRenderView(hView, &renderContext)
    // which leads to view.Render()
}

void Power_Service::postGraphRender() {
    // the graph finished rendering and you may more stuff here
    // e.g. end frame timer
    // update window name
    // swap buffers, glClear
}

//std::vector<float> Power_Service::examine_expression(std::string const& name, std::string const& exp_path, int s_idx) {
//    sol_state_["s_idx"] = s_idx;
//    sol_state_.script_file(exp_path);
//    return sol_state_[name];
//}

void Power_Service::fill_lua_callbacks() {
    frontend_resources::LuaCallbacksCollection callbacks;

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult>(
        "mmPowerSetup", "()", {[&]() -> frontend_resources::LuaCallbacksCollection::VoidResult {
            //setup_measurement();
            if (rtx_) {
                rtx_->ApplyConfigs();
            }
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult, std::string>("mmPowerMeasure",
        "(string path)", {[&](std::string path) -> frontend_resources::LuaCallbacksCollection::VoidResult {
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
                        rtx_->StartMeasurement(
                            path, {parquet_dataverse_writer, &megamol::power::wf_tracy}, &meta_, sbroker_.Get(false));
                    } else {
                        rtx_->StartMeasurement(path, {&megamol::power::wf_parquet, &megamol::power::wf_tracy}, &meta_,
                            sbroker_.Get(false));
                    }
                } else {
                    rtx_->StartMeasurement(path, {&megamol::power::wf_tracy}, &meta_, sbroker_.Get(false));
                }
            }
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    /*callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult>(
        "mmPowerSignalHalt", "()", {[&]() -> frontend_resources::LuaCallbacksCollection::VoidResult {
            sbroker_.Reset();
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});*/

    /*callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult>(
        "mmPowerTrigger", "()", {[&]() -> frontend_resources::LuaCallbacksCollection::VoidResult {
            trigger();
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});*/

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult, std::string, int, int, int, int>(
        "mmPowerConfig", "(string path, int points, int count, int range_ms, int timeout_ms)",
        {[&](std::string path, int points, int count, int range,
             int timeout) -> frontend_resources::LuaCallbacksCollection::VoidResult {
            /*sol_state_["points"] = points;
            sol_state_["count"] = count;
            sol_state_["range"] = range;
            sol_state_["timeout"] = timeout;

            sol_state_.script_file(path);

            visus::power_overwhelming::rtx_instrument_configuration config = sol_state_[name];

            config_map_[name] = config;*/
            if (rtx_) {
                rtx_->UpdateConfigs(
                    path, points, count, std::chrono::milliseconds(range), std::chrono::milliseconds(timeout));
            }
            reset_segment_range(std::chrono::milliseconds(range));

            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::BoolResult>(
        "mmPowerIsPending", "()", {[&]() -> frontend_resources::LuaCallbacksCollection::BoolResult {
            return frontend_resources::LuaCallbacksCollection::BoolResult{sbroker_.GetValue()};
        }});

    /*callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult>(
        "mmPowerForceTrigger", "()", {[&]() -> frontend_resources::LuaCallbacksCollection::VoidResult {
            for (auto& i : rtx_instr_) {
                i.trigger_manually();
            }
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});*/

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult, bool>("mmPowerSoftwareTrigger", "(bool set)",
        {[&](bool set) -> frontend_resources::LuaCallbacksCollection::VoidResult {
            if (rtx_) {
                rtx_->SetSoftwareTrigger(set);
            }
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    /*callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult, std::string, std::string>(
        "mmPowerRegisterTracyExp", "(string name, string path)",
        {[&](std::string name, std::string path) -> frontend_resources::LuaCallbacksCollection::VoidResult {
            exp_map_[name] = path;
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});*/

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult, std::string>("mmPowerDataverseKey",
        "(string path_to_key)",
        {[&](std::string path_to_key) -> frontend_resources::LuaCallbacksCollection::VoidResult {
            dataverse_key_ = std::make_unique<power::CryptToken>(path_to_key);
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult, std::string, std::string>(
        "mmPowerDataverseDataset", "(string base_path, string doi)",
        {[&](std::string base_path, std::string doi) -> frontend_resources::LuaCallbacksCollection::VoidResult {
            dataverse_config_.base_path = base_path;
            dataverse_config_.doi = doi;
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    auto& register_callbacks =
        m_requestedResourceReferences[0]
            .getResource<std::function<void(frontend_resources::LuaCallbacksCollection const&)>>();

    register_callbacks(callbacks);
}

//void clear_sb(power::buffers_t& buffers) {
//    for (auto& b : buffers) {
//        b.Clear();
//    }
//}

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

    /*ParquetWriter(nvml_path, nvml_buffers_);
    ParquetWriter(adl_path, adl_buffers_);
    ParquetWriter(emi_path, emi_buffers_);
    ParquetWriter(msr_path, msr_buffers_);
    ParquetWriter(tinker_path, tinker_buffers_);*/

    if (dataverse_key_) {
        /*if (!nvml_buffers_.empty())
            power::DataverseWriter(dataverse_config_.base_path, dataverse_config_.doi, nvml_path.string(),
                dataverse_key_->GetToken(), sbroker_.Get(false));
        if (!adl_buffers_.empty())
            power::DataverseWriter(dataverse_config_.base_path, dataverse_config_.doi, adl_path.string(),
                dataverse_key_->GetToken(), sbroker_.Get(false));
        if (!emi_buffers_.empty())
            power::DataverseWriter(dataverse_config_.base_path, dataverse_config_.doi, emi_path.string(),
                dataverse_key_->GetToken(), sbroker_.Get(false));
        if (!msr_buffers_.empty())
            power::DataverseWriter(dataverse_config_.base_path, dataverse_config_.doi, msr_path.string(),
                dataverse_key_->GetToken(), sbroker_.Get(false));
        if (!tinker_buffers_.empty())
            power::DataverseWriter(dataverse_config_.base_path, dataverse_config_.doi, tinker_path.string(),
                dataverse_key_->GetToken(), sbroker_.Get(false));*/

        /*if (samplers.nvml_samplers_)
            samplers.nvml_samplers_->WriteBuffers(nvml_path, &meta_, dataverse_config_.base_path, dataverse_config_.doi,
                dataverse_key_->GetToken(), sbroker_.Get(false));
        if (samplers.adl_samplers_)
            samplers.adl_samplers_->WriteBuffers(adl_path, &meta_, dataverse_config_.base_path, dataverse_config_.doi,
                dataverse_key_->GetToken(), sbroker_.Get(false));
        if (samplers.emi_samplers_)
            samplers.emi_samplers_->WriteBuffers(emi_path, &meta_, dataverse_config_.base_path, dataverse_config_.doi,
                dataverse_key_->GetToken(), sbroker_.Get(false));
        if (samplers.msr_samplers_)
            samplers.msr_samplers_->WriteBuffers(msr_path, &meta_, dataverse_config_.base_path, dataverse_config_.doi,
                dataverse_key_->GetToken(), sbroker_.Get(false));
        if (samplers.tinker_samplers_)
            samplers.tinker_samplers_->WriteBuffers(tinker_path, &meta_, dataverse_config_.base_path,
                dataverse_config_.doi,
                dataverse_key_->GetToken(), sbroker_.Get(false));*/

        samplers.visit<power::MetaData const*, std::string const&, std::string const&, char const*, char&>(
            &power::ISamplerCollection::WriteBuffers, tpl, &meta_, dataverse_config_.base_path, dataverse_config_.doi,
            dataverse_key_->GetToken(), sbroker_.Get(false));
    } else {
        /*if (samplers.nvml_samplers_)
            samplers.nvml_samplers_->WriteBuffers(nvml_path, &meta_);
        if (samplers.adl_samplers_)
            samplers.adl_samplers_->WriteBuffers(adl_path, &meta_);
        if (samplers.emi_samplers_)
            samplers.emi_samplers_->WriteBuffers(emi_path, &meta_);
        if (samplers.msr_samplers_)
            samplers.msr_samplers_->WriteBuffers(msr_path, &meta_);
        if (samplers.tinker_samplers_)
            samplers.tinker_samplers_->WriteBuffers(tinker_path, &meta_);*/

        samplers.visit<power::MetaData const*>(&power::ISamplerCollection::WriteBuffers, tpl, &meta_);
    }

    //#if defined(DEBUG) && defined(MEGAMOL_USE_TRACY)
    //    static std::string name = "nvml_debug";
    //    for (auto const& b : nvml_buffers_) {
    //        TracyPlotConfig(name.c_str(), tracy::PlotFormatType::Number, false, true, 0);
    //        auto const& values = b.ReadSamples();
    //        auto const& ts = b.ReadTimestamps();
    //        for (std::size_t v_idx = 0; v_idx < values.size(); ++v_idx) {
    //            tracy::Profiler::PlotData(name.c_str(), values[v_idx], ts[v_idx]);
    //        }
    //    }
    //#endif

    samplers.visit(&power::ISamplerCollection::ResetBuffers);

    /*if (nvml_samplers_)
        nvml_samplers_->ResetBuffers();
    if (adl_samplers_)
        adl_samplers_->ResetBuffers();
    if (emi_samplers_)
        emi_samplers_->ResetBuffers();
    if (msr_samplers_)
        msr_samplers_->ResetBuffers();
    if (tinker_samplers_)
        tinker_samplers_->ResetBuffers();*/


    /*clear_sb(nvml_buffers_);
    clear_sb(adl_buffers_);
    clear_sb(emi_buffers_);
    clear_sb(msr_buffers_);
    clear_sb(tinker_buffers_);*/
}

//void set_sb_range(power::buffers_t& buffers, std::chrono::milliseconds const& range) {
//    for (auto& b : buffers) {
//        b.SetSampleRange(range);
//    }
//}

void Power_Service::reset_segment_range(std::chrono::milliseconds const& range) {
    samplers.visit<std::chrono::milliseconds const&>(&power::ISamplerCollection::SetSegmentRange, range);

    /*if (nvml_samplers_)
        nvml_samplers_->SetSegmentRange(range);
    if (adl_samplers_)
        adl_samplers_->SetSegmentRange(range);
    if (emi_samplers_)
        emi_samplers_->SetSegmentRange(range);
    if (msr_samplers_)
        msr_samplers_->SetSegmentRange(range);
    if (tinker_samplers_)
        tinker_samplers_->SetSegmentRange(range);*/

    /*set_sb_range(nvml_buffers_, range);
    set_sb_range(adl_buffers_, range);
    set_sb_range(emi_buffers_, range);
    set_sb_range(msr_buffers_, range);
    set_sb_range(tinker_buffers_, range);*/
}

void Power_Service::reset_measurement() {
    sbroker_.Reset();
    seg_cnt_ = 0;
}

} // namespace frontend
} // namespace megamol

#endif