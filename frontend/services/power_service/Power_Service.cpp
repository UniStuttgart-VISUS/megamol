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

#include <codecvt>
#include <fstream>
#include <numeric>
#include <regex>
#include <stdexcept>

#ifdef WIN32
#include <Windows.h>
#endif

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

#include "LuaCallbacksCollection.h"

#include "sol_rtx_instrument.h"

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

bool Power_Service::init_sol_commands_ = true;

Power_Service::Power_Service() : sol_state_(nullptr) {
    // init members to default states
}

Power_Service::~Power_Service() {
    // clean up raw pointers you allocated with new, which is bad practice and nobody does
}

std::string unmueller_string(wchar_t const* name) {
    std::string no_mueller =
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t>{}.to_bytes(std::wstring(name));

    /*char* sensor_name = new char[wcslen(name) + 1];
    wcstombs(sensor_name, name, wcslen(name) + 1);
    std::string no_mueller(sensor_name);
    delete[] sensor_name;*/
    return no_mueller;
}

std::string get_device_name(visus::power_overwhelming::rtx_instrument const& i) {
    auto name_size = i.name(nullptr, 0);
    std::string name;
    name.resize(name_size);
    i.name(name.data(), name_size);
    if (name_size != 0) {
        name.resize(name_size - 1);
    }
    return name;
}

static int measure_time_in_ms = 50;
static int sample_count = 50000;

void test_func(const visus::power_overwhelming::measurement& m, void*, std::string const&) {}

bool Power_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    sol_state_.open_libraries(sol::lib::base);

    visus::power_overwhelming::sol_register_all(sol_state_);

    const auto conf = static_cast<Config*>(configPtr);
    auto const lpt = conf->lpt;

    std::regex p("^(lpt|LPT)(\\d)$");
    std::smatch m;
    if (!std::regex_search(lpt, m, p)) {
        log_error("LPT parameter malformed");
        return false;
    }

    try {
        trigger_ = std::make_unique<ParallelPortTrigger>(("\\\\.\\" + lpt).c_str());
    } catch (...) {
        trigger_ = nullptr;
    }

    callbacks_.signal_high = std::bind(&ParallelPortTrigger::SetBit, trigger_.get(), 7, true);
    callbacks_.signal_low = std::bind(&ParallelPortTrigger::SetBit, trigger_.get(), 7, false);

    m_providedResourceReferences = {{frontend_resources::PowerCallbacks_Req_Name, callbacks_}};

    m_requestedResourcesNames = {"RegisterLuaCallbacks"};


    int64_t incr = std::chrono::nanoseconds(std::chrono::milliseconds(measure_time_in_ms)).count() /
                   static_cast<int64_t>(sample_count);
    int64_t start = (measure_time_in_ms / 10) * 1000 * 1000 * (-1);
    sample_times_.resize(sample_count);
    std::generate(sample_times_.begin(), sample_times_.end(), [&]() {
        static int64_t i;
        auto ret = start + i * incr;
        ++i;
        return ret;
    });

    // begin tracy::Profiler::CalibrateTimer
    std::atomic_signal_fence(std::memory_order_acq_rel);
    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto r0 = __rdtsc();
    std::atomic_signal_fence(std::memory_order_acq_rel);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::atomic_signal_fence(std::memory_order_acq_rel);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto r1 = __rdtsc();
    std::atomic_signal_fence(std::memory_order_acq_rel);

    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const auto dr = r1 - r0;

    timer_mul_ = double(dt) / double(dr);
    // end tracy::Profiler::CalibrateTimer


    using namespace visus::power_overwhelming;

    //setup_measurement();

    auto sensor_count = nvml_sensor::for_all(nullptr, 0);
    std::vector<visus::power_overwhelming::nvml_sensor> tmp_sensors(sensor_count);
    nvml_sensor::for_all(tmp_sensors.data(), tmp_sensors.size());

    sensor_count = msr_sensor::for_all(nullptr, 0);
    std::vector<msr_sensor> tmp_msr_sensors(sensor_count);
    msr_sensor::for_all(tmp_msr_sensors.data(), tmp_msr_sensors.size());

    //#define TINKER

#ifdef TINKER
    sensor_count = tinkerforge_sensor::for_all(nullptr, 0);
    std::vector<tinkerforge_sensor> tmp_tinker_sensors(sensor_count);
    tinkerforge_sensor::for_all(tmp_tinker_sensors.data(), tmp_tinker_sensors.size());
#endif

#ifdef MEGAMOL_USE_TRACY
    for (auto& sensor : tmp_sensors) {
        auto sensor_name = unmueller_string(sensor.name());

        TracyPlotConfig(sensor_name.data(), tracy::PlotFormatType::Number, false, true, 0);

        //sensor.sample([](const visus::power_overwhelming::measurement& m, void*) {
        //    auto name = unmueller_string(m.sensor());
        //    TracyPlot(name.data(), m.power());
        //});

        nvml_sensors_[sensor_name] = std::move(sensor);
        sensor_names_.push_back(sensor_name);
        nvml_sensors_[sensor_name].sample(
            [](const visus::power_overwhelming::measurement& m, void* sensor_name) {
                //auto name = unmueller_string(sensor->name());
                TracyPlot(reinterpret_cast<char const*>(sensor_name), m.power());
            },
            10Ui64, static_cast<void*>(sensor_names_.back().data()));
    }
    for (auto& sensor : tmp_msr_sensors) {
        auto sensor_name = unmueller_string(sensor.name());
        if (sensor_name.find("package") != std::string::npos) {
            if (sensor_name.find("msr/0/") == std::string::npos)
                continue;
        }

        TracyPlotConfig(sensor_name.c_str(), tracy::PlotFormatType::Number, false, true, 0);

        /*sensor.sample([](const visus::power_overwhelming::measurement& m, void*) {
            auto name = unmueller_string(m.sensor());
            TracyPlot(name.c_str(), m.power());
        });*/

        msr_sensors_[std::string(sensor_name)] = std::move(sensor);
        //sensor_names_.push_back(sensor_name);
        //msr_sensors_[sensor_name].sample(
        //    [](const visus::power_overwhelming::measurement& m, void* sensor_name) {
        //        //auto name = unmueller_string(sensor->name());
        //        TracyPlot(reinterpret_cast<char const*>(sensor_name), m.power());
        //    },
        //    10Ui64, timestamp_resolution::microseconds, static_cast<void*>(sensor_names_.back().data()));
    }
#ifdef TINKER
    for (auto& sensor : tmp_tinker_sensors) {
        auto sensor_name = unmueller_string(sensor.name());

        TracyPlotConfig(sensor_name.data(), tracy::PlotFormatType::Number, false, true, 0);

        //sensor.sample([](const visus::power_overwhelming::measurement& m, void*) {
        //    auto name = unmueller_string(m.sensor());
        //    TracyPlot(name.data(), m.power());
        //});

        tinker_sensors_[sensor_name] = std::move(sensor);
        sensor_names_.push_back(sensor_name);
        tinker_sensors_[sensor_name].sample(
            [](const visus::power_overwhelming::measurement& m, void* sensor_name) {
                //auto name = unmueller_string(sensor->name());
                TracyPlot(reinterpret_cast<char const*>(sensor_name), m.power());
            },
            tinkerforge_sensor_source::power, 1000Ui64, static_cast<void*>(sensor_names_.back().data()));
    }
#endif
    TracyPlotConfig("V", tracy::PlotFormatType::Number, false, true, 0);
    TracyPlotConfig("A", tracy::PlotFormatType::Number, false, true, 0);
    TracyPlotConfig("W", tracy::PlotFormatType::Number, false, true, 0);
    TracyPlotConfig("Frame", tracy::PlotFormatType::Number, false, true, 0);
#endif

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

    // alternative
    return {"ExternalResource_1", "ExternalResource_2"};
}

void Power_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    // maybe we want to keep the list of requested resources
    this->m_requestedResourceReferences = resources;
    //    if (init_sol_commands_) {
    //        //sol_state_ = get_sol_state_view();
    //        sol_state_.script("print('got sol instance')");
    //
    //        visus::power_overwhelming::sol_register_all(sol_state_);
    //
    //        sol_state_.script("local s = simple:new()\ns:do_stuff()\n");
    //
    //        /*sol_state_->script(R"(
    //local instr = find_resources("0x0AAD", "0x01D6")
    //instr[1]:reference_position(oscilloscope_reference_point.left)
    //local chan_1 = oscilloscope_channel:new(1)
    //chan_1:state(true):attenuation(oscilloscope_quantity:new(1, "V")):range(oscilloscope_quantity:new(26, "V"))
    //instr[1]:channel(chan_1)
    //)");*/
    //
    //        sol_state_.script(R"(acq = oscilloscope_single_acquisition:new():points(50000):count(2):segmented(true)
    //trigger = oscilloscope_edge_trigger:new("EXT")
    //trigger:level(5, oscilloscope_quantity:new(2000.0, "mV")):slope(oscilloscope_trigger_slope.rising):mode(oscilloscope_trigger_mode.normal)
    //--trigger = get_trigger("EXT")
    //--trigger:level(5, oscilloscope_quantity:new(2000.0, "mV")):slope(oscilloscope_trigger_slope.rising):mode(oscilloscope_trigger_mode.normal)
    //quant = oscilloscope_quantity:new(50, "ms")
    //config = rtx_instrument_configuration:new(quant, acq, trigger, 10000);
    //--config = get_config(quant, acq)
    //chan_1 = oscilloscope_channel:new(1)
    //chan_1:state(true):attenuation(oscilloscope_quantity:new(1, "V")):range(oscilloscope_quantity:new(26, "V")):label(oscilloscope_label:new("voltage", true))
    //chan_2 = oscilloscope_channel:new(2)
    //chan_2:state(true):attenuation(oscilloscope_quantity:new(10, "A")):range(oscilloscope_quantity:new(40, "V")):label(oscilloscope_label:new("current", true))
    //config:channel(chan_1)
    //config:channel(chan_2)
    //config = as_slave(config)
    //)");
    //
    //        auto devices = visus::power_overwhelming::visa_instrument::find_resources("0x0AAD", "0x01D6");
    //
    //        for (auto d = devices.as<char>(); (d != nullptr) && (*d != 0); d += strlen(d) + 1) {
    //            core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Found device %s", d);
    //
    //            rtx_instr_.emplace_back(d);
    //        }
    //        visus::power_overwhelming::rtx_instrument_configuration config = sol_state_["config"];
    //        config.beep_on_apply(1);
    //
    //        config.apply(rtx_instr_[0]);
    //
    //        /*i.channel(oscilloscope_channel(1)
    //                      .label(oscilloscope_label("voltage"))
    //                      .state(true)
    //                      .attenuation(oscilloscope_quantity(1, "V"))
    //                      .range(oscilloscope_quantity(26)));*/
    //
    //        init_sol_commands_ = false;
    //    }

    // prepare usage of requested resources
    //this->m_externalResource_1_ptr =
    //    &resources[0].getResource<ExternalResource_1>(); // resources are in requested order and not null
    //this->m_externalResource_2_ptr =
    //    &resources[1]
    //         .getResource<
    //             namspace::to::resource::ExternalResource_2>(); // ptr will be not null or program terminates by design

    try {
        auto devices = visus::power_overwhelming::visa_instrument::find_resources("0x0AAD", "0x01D6");

        for (auto d = devices.as<char>(); (d != nullptr) && (*d != 0); d += strlen(d) + 1) {
            core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Found device %s", d);

            rtx_instr_.emplace_back(d);
        }
    } catch (std::exception& ex) {
        core::utility::log::Log::DefaultLog.WriteError(
            "[Power_Service]: Error during instrument discovery: %s", ex.what());
    }

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
    /*if (trigger_)
        trigger_->WriteHigh();*/
}

void Power_Service::postGraphRender() {
// the graph finished rendering and you may more stuff here
// e.g. end frame timer
// update window name
// swap buffers, glClear
/*if (trigger_)
    trigger_->WriteLow();*/
#ifdef MEGAMOL_USE_TRACY
    /*for (auto& [name, sensor] : nvml_sensors_) {
        auto val = sensor.sample_data();
        TracyPlot(name.data(), val.power());
    }*/
    for (auto& [name, sensor] : msr_sensors_) {
        auto val = sensor.sample_data();
        TracyPlot(name.data(), val.power());
    }
#endif
}

void Power_Service::setup_measurement() {
    using namespace visus::power_overwhelming;
    core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Starting setup");
    //auto m_func = [&]() -> void {
    try {
        for (auto& i : rtx_instr_) {
            i.synchronise_clock();
            i.reset(true, true);

            auto name = get_device_name(i);

            auto fit = config_map_.find(name);
            if (fit == config_map_.end()) {
                core::utility::log::Log::DefaultLog.WriteError(
                    "[Power_Service]: Could not find config for device %s", name.c_str());
                continue;
            }

            auto config = fit->second;
            config.apply(i);
            //i.timeout(timeout);
            i.reference_position(oscilloscope_reference_point::left);
            i.trigger_position(oscilloscope_quantity(0, "ms"));
            i.operation_complete();
            core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Configured device %s", name.c_str());
        }

        //auto devices = visa_instrument::find_resources("0x0AAD", "0x01D6");

        //for (auto d = devices.as<char>(); (d != nullptr) && (*d != 0); d += strlen(d) + 1) {
        //    core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Found device %s", d);

        //    rtx_instr_.emplace_back(d);
        //}

        //oscilloscope_edge_trigger trigger = oscilloscope_edge_trigger("EXT");
        //trigger.level(5, oscilloscope_quantity(2000.0f, "mV"))
        //    .slope(oscilloscope_trigger_slope::rising)
        //    .mode(oscilloscope_trigger_mode::normal);

        //auto rtx_conf = rtx_instrument_configuration(oscilloscope_quantity(measure_time_in_ms, "ms"),
        //    oscilloscope_single_acquisition().points(sample_count).count(2).segmented(true), trigger,
        //    visa_instrument::timeout_type(10000));

        //std::vector<rtx_instrument_configuration> rtx_cfg(rtx_instr_.size(), rtx_conf);

        ////auto devices = visa_instrument::find_resources("0x0AAD", "0x01D6");

        ////for (auto d = devices.as<char>(); (d != nullptr) && (*d != 0); d += strlen(d) + 1) {
        ////for (auto& i : rtx_instr_) {
        //for (int idx = 0; idx < rtx_instr_.size(); ++idx) {
        //    auto& i = rtx_instr_[idx];
        //    //core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Found device %s", d);

        //    //rtx_instrument i(d);

        //    i.synchronise_clock();
        //    i.reset(true, true);

        //    auto& cfg = rtx_cfg[idx];
        //    if (idx != 0) {
        //        cfg.as_slave();
        //    }

        //    //i.timeout(10000);

        //    i.reference_position(oscilloscope_reference_point::left);
        //    //i.time_range(oscilloscope_quantity(measure_time_in_ms, "ms"));

        //    if (idx == 0) {
        //        i.channel(oscilloscope_channel(1)
        //                      .label(oscilloscope_label("voltage"))
        //                      .state(true)
        //                      .attenuation(oscilloscope_quantity(1, "V"))
        //                      .range(oscilloscope_quantity(26)));

        //        i.channel(oscilloscope_channel(2)
        //                      .label(oscilloscope_label("current"))
        //                      .state(true)
        //                      .attenuation(oscilloscope_quantity(10, "A"))
        //                      .range(oscilloscope_quantity(40)));

        //        i.channel(oscilloscope_channel(3)
        //                      .label(oscilloscope_label("frame"))
        //                      .state(true)
        //                      .attenuation(oscilloscope_quantity(1, "V"))
        //                      .range(oscilloscope_quantity(7)));
        //    } else {
        //        i.channel(oscilloscope_channel(1)
        //                      .label(oscilloscope_label("current#1"))
        //                      .state(true)
        //                      .attenuation(oscilloscope_quantity(10, "A"))
        //                      .range(oscilloscope_quantity(40)));
        //        i.channel(oscilloscope_channel(2)
        //                      .label(oscilloscope_label("current#2"))
        //                      .state(true)
        //                      .attenuation(oscilloscope_quantity(10, "A"))
        //                      .range(oscilloscope_quantity(40)));
        //    }


        //    i.trigger_position(oscilloscope_quantity(0.f, "ms"));
        //    /*i.trigger(oscilloscope_edge_trigger("EXT")
        //                  .level(5, oscilloscope_quantity(2000.0f, "mV"))
        //                  .slope(oscilloscope_trigger_slope::rising)
        //                  .mode(oscilloscope_trigger_mode::normal));*/

        //    //i.acquisition(oscilloscope_single_acquisition().points(sample_count).count(2).segmented(true));

        //    /*std::cout << "RTX interface type: " << i.interface_type() << std::endl
        //              << "RTX status before acquire: " << i.status() << std::endl;*/

        //    cfg.apply(i);
        //    i.operation_complete();

        //    core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Completed setup");

        //    //i.acquisition(oscilloscope_acquisition_state::single);

        //    //trigger_->SetBit(6, true);
        //    //trigger_->SetBit(6, false);

        //    //i.operation_complete();

        //    //auto segment0_1 = i.data(1, oscilloscope_waveform_points::maximum);
        //    ////i.clear();
        //    //auto segment0_2 = i.data(2, oscilloscope_waveform_points::maximum);

        //    //auto segment0_3 = i.data(3, oscilloscope_waveform_points::maximum);

        //    //core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service] Started writing");
        //    //std::ofstream out_file("channel_data_0.csv");
        //    //for (size_t i = 0; i < segment0_1.record_length(); ++i) {
        //    //    out_file << segment0_1.begin()[i] << "," << segment0_2.begin()[i] << "," << segment0_3.begin()[i]
        //    //             << std::endl;
        //    //}
        //    //out_file.close();
        //}
    } catch (std::exception& ex) {
        core::utility::log::Log::DefaultLog.WriteError("[Power_Service]: %s", ex.what());
    }
    //};
    //auto m_thread = std::thread(m_func);
    //m_thread.detach();
}

void Power_Service::start_measurement() {
    using namespace visus::power_overwhelming;
    core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Starting measurement");
    auto m_func = [&]() -> void {
        try {
            //auto devices = visa_instrument::find_resources("0x0AAD", "0x01D6");

            //for (auto d = devices.as<char>(); (d != nullptr) && (*d != 0); d += strlen(d) + 1) {
            pending_measurement_ = true;
            for (auto& i : rtx_instr_) {
                //core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Found device %s", d);

                //rtx_instrument i(d);
                i.acquisition(oscilloscope_acquisition_state::single);
            }

            /*auto trigger = [&]() {
                for (int i = 0; i < 100; ++i) {
                    trigger_->SetBit(6, true);
                    trigger_->SetBit(6, false);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            };
            auto t_thread = std::thread(trigger);
            t_thread.detach();*/


            int counter = 0;
            for (auto& i : rtx_instr_) {
                i.operation_complete();

                auto name = get_device_name(i);

                auto fit = config_map_.find(name);
                if (fit == config_map_.end()) {
                    core::utility::log::Log::DefaultLog.WriteError(
                        "[Power_Service]: Could not find config for device %s", name);
                    continue;
                }

                auto config = fit->second;
                auto num_channels = config.channels();

                std::vector<oscilloscope_waveform> waveforms;
                waveforms.reserve(num_channels);

                for (auto w_idx = 0; w_idx < num_channels; ++w_idx) {
                    waveforms.push_back(i.data(w_idx + 1, oscilloscope_waveform_points::maximum));
                }

                if (!waveforms.empty()) {
                    auto t_begin = waveforms[0].time_begin();
                    auto t_end = waveforms[0].time_end();
                    core::utility::log::Log::DefaultLog.WriteInfo(
                        "[Power_Service] Segment begin %f end %f", t_begin, t_end);
                    auto range = t_end - t_begin;
                    auto incr = range / static_cast<float>(waveforms[0].record_length());
                    auto t_b_s = std::chrono::duration<float>(t_begin);
                    auto t_b_ns = std::chrono::round<std::chrono::nanoseconds>(t_b_s);
                    auto incr_s = std::chrono::duration<float>(incr);
                    auto incr_ns = std::chrono::round<std::chrono::nanoseconds>(incr_s);

                    sample_times_.resize(waveforms[0].record_length());
                    std::generate(sample_times_.begin(), sample_times_.end(), [&]() {
                        static int64_t i;
                        auto ret = t_b_ns.count() + i * incr_ns.count();
                        ++i;
                        return ret;
                    });

                    std::ofstream out_file("channel_data_" + name + ".csv");
                    for (size_t i = 0; i < waveforms[0].record_length(); ++i) {
                        out_file << sample_times_[i];
                        for (auto const& wave : waveforms) {
                            out_file << "," << wave.begin()[i];
                        }
                        out_file << std::endl;
                    }
                    out_file.close();
                }

                //if (counter == 0) {
                //    auto segment0_1 = i.data(1, oscilloscope_waveform_points::maximum);
                //    //i.clear();
                //    auto segment0_2 = i.data(2, oscilloscope_waveform_points::maximum);

                //    auto segment0_3 = i.data(3, oscilloscope_waveform_points::maximum);

                //    auto t_begin = segment0_1.time_begin();
                //    auto t_end = segment0_1.time_end();
                //    core::utility::log::Log::DefaultLog.WriteInfo(
                //        "[Power_Service] Segment begin %f end %f", t_begin, t_end);
                //    auto range = t_end - t_begin;
                //    auto incr = range / static_cast<float>(segment0_1.record_length());
                //    auto t_b_s = std::chrono::duration<float>(t_begin);
                //    auto t_b_ns = std::chrono::round<std::chrono::nanoseconds>(t_b_s);
                //    auto incr_s = std::chrono::duration<float>(incr);
                //    auto incr_ns = std::chrono::round<std::chrono::nanoseconds>(incr_s);

                //    sample_times_.resize(segment0_1.record_length());
                //    std::generate(sample_times_.begin(), sample_times_.end(), [&]() {
                //        static int64_t i;
                //        auto ret = t_b_ns.count() + i * incr_ns.count();
                //        ++i;
                //        return ret;
                //    });

                //    core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service] Started writing");
                //    std::ofstream out_file("channel_data_" + std::to_string(counter++) + ".csv");
                //    for (size_t i = 0; i < segment0_1.record_length(); ++i) {
                //        out_file << sample_times_[i] << "," << segment0_1.begin()[i] << "," << segment0_2.begin()[i]
                //                 << "," << segment0_3.begin()[i] << std::endl;
                //        tracy::Profiler::PlotData("V", segment0_1.begin()[i],
                //            static_cast<double>(sample_times_[i] + trigger_offset_.count()) / timer_mul_);
                //        tracy::Profiler::PlotData("A", segment0_2.begin()[i],
                //            static_cast<double>(sample_times_[i] + trigger_offset_.count()) / timer_mul_);
                //        tracy::Profiler::PlotData("W", segment0_1.begin()[i] * segment0_2.begin()[i],
                //            static_cast<double>(sample_times_[i] + trigger_offset_.count()) / timer_mul_);
                //        tracy::Profiler::PlotData("Frame", segment0_3.begin()[i],
                //            static_cast<double>(sample_times_[i] + trigger_offset_.count()) / timer_mul_);
                //    }
                //    out_file.close();
                //} else {
                //    auto segment0_1 = i.data(1, oscilloscope_waveform_points::maximum);
                //    //i.clear();
                //    auto segment0_2 = i.data(2, oscilloscope_waveform_points::maximum);

                //    auto t_begin = segment0_1.time_begin();
                //    auto t_end = segment0_1.time_end();
                //    core::utility::log::Log::DefaultLog.WriteInfo(
                //        "[Power_Service] Segment begin %f end %f", t_begin, t_end);
                //    auto range = t_end - t_begin;
                //    auto incr = range / static_cast<float>(segment0_1.record_length());
                //    auto t_b_s = std::chrono::duration<float>(t_begin);
                //    auto t_b_ns = std::chrono::round<std::chrono::nanoseconds>(t_b_s);
                //    auto incr_s = std::chrono::duration<float>(incr);
                //    auto incr_ns = std::chrono::round<std::chrono::nanoseconds>(incr_s);

                //    sample_times_.resize(segment0_1.record_length());
                //    std::generate(sample_times_.begin(), sample_times_.end(), [&]() {
                //        static int64_t i;
                //        auto ret = t_b_ns.count() + i * incr_ns.count();
                //        ++i;
                //        return ret;
                //    });

                //    core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service] Started writing");
                //    std::ofstream out_file("channel_data_" + std::to_string(counter++) + ".csv");
                //    for (size_t i = 0; i < segment0_1.record_length(); ++i) {
                //        out_file << sample_times_[i] << "," << segment0_1.begin()[i] << "," << segment0_2.begin()[i]
                //                 << std::endl;
                //    }
                //    out_file.close();
                //}


                core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Completed measurement");
            }
            pending_measurement_ = false;
        } catch (std::exception& ex) {
            core::utility::log::Log::DefaultLog.WriteError("[Power_Service]: %s", ex.what());
        }
    };
    auto m_thread = std::thread(m_func);
    m_thread.detach();
}

void Power_Service::trigger() {
#ifdef WIN32
    trigger_offset_ = std::chrono::nanoseconds(static_cast<int64_t>(static_cast<double>(__rdtsc()) * timer_mul_));
    /*core::utility::log::Log::DefaultLog.WriteInfo("RDTSC: %f", __rdtsc()*m_timerMul);
    core::utility::log::Log::DefaultLog.WriteInfo("Tracy: %lld", tracy::Profiler::GetTime());
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    auto cnt_ms = counter.QuadPart * 1000000;
    cnt_ms /= frequency.QuadPart;
    core::utility::log::Log::DefaultLog.WriteInfo("QPC Counter: %lld", counter.QuadPart);
    core::utility::log::Log::DefaultLog.WriteInfo("QPC: %lld", cnt_ms*1000);*/
#endif
    trigger_->SetBit(6, true);
    trigger_->SetBit(6, false);
}

void Power_Service::fill_lua_callbacks() {
    frontend_resources::LuaCallbacksCollection callbacks;

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult>(
        "mmPowerSetup", "()", {[&]() -> frontend_resources::LuaCallbacksCollection::VoidResult {
            setup_measurement();
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult>(
        "mmPowerMeasure", "()", {[&]() -> frontend_resources::LuaCallbacksCollection::VoidResult {
            start_measurement();
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult>(
        "mmPowerTrigger", "()", {[&]() -> frontend_resources::LuaCallbacksCollection::VoidResult {
            trigger();
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult, std::string, std::string, int, int, int, int>(
        "mmPowerConfig", "(string name, string path, int points, int count, int range, int timeout)",
        {[&](std::string name, std::string path, int points, int count, int range,
             int timeout) -> frontend_resources::LuaCallbacksCollection::VoidResult {
            sol_state_["points"] = points;
            sol_state_["count"] = count;
            sol_state_["range"] = range;
            sol_state_["timeout"] = timeout;

            sol_state_.script_file(path);

            visus::power_overwhelming::rtx_instrument_configuration config = sol_state_[name];

            config_map_[name] = config;

            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::BoolResult>(
        "mmPowerIsPending", "()", {[&]() -> frontend_resources::LuaCallbacksCollection::BoolResult {
            return frontend_resources::LuaCallbacksCollection::BoolResult{is_measurement_pending()};
        }});

    auto& register_callbacks =
        m_requestedResourceReferences[0]
            .getResource<std::function<void(frontend_resources::LuaCallbacksCollection const&)>>();

    register_callbacks(callbacks);
}

} // namespace frontend
} // namespace megamol

#endif
