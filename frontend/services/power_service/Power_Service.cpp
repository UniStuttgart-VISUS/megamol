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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <regex>
#include <stdexcept>

#ifdef WIN32
#include <Windows.h>
#else
#include <time.h>
#endif

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

#include "LuaCallbacksCollection.h"
#include <power_overwhelming/timestamp_resolution.h>

#include "ParquetWriter.h"
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

#ifdef MEGAMOL_USE_TRACY
void tracy_sample(
    wchar_t const*, visus::power_overwhelming::measurement_data const* m, std::size_t const n, void* name_ptr) {
    auto name = static_cast<std::string*>(name_ptr);
    for (std::size_t i = 0; i < n; ++i) {
        TracyPlot(name->data(), m[i].power());
    }
}
#endif

bool Power_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    sol_state_.open_libraries(sol::lib::base);

    visus::power_overwhelming::sol_register_all(sol_state_);

    visus::power_overwhelming::sol_expressions(sol_state_, values_map_);

#ifdef WIN32
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    qpc_frequency_ = f.QuadPart;
#else
    timespec tp;
    clock_getres(CLOCK_MONOTONIC_RAW, &tp);
    qpc_frequency_ = tp.tv_nsec;
#endif

    const auto conf = static_cast<Config*>(configPtr);
    auto const lpt = conf->lpt;
    write_to_files_ = conf->write_to_files;
    write_folder_ = conf->folder;

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
        trigger_->SetBit(7, true);
        trigger_->SetBit(7, false);
    };

    m_providedResourceReferences = {{frontend_resources::PowerCallbacks_Req_Name, callbacks_}};

    m_requestedResourcesNames = {"RegisterLuaCallbacks"};


    /*int64_t incr = std::chrono::nanoseconds(std::chrono::milliseconds(measure_time_in_ms)).count() /
                   static_cast<int64_t>(sample_count);
    int64_t start = (measure_time_in_ms / 10) * 1000 * 1000 * (-1);
    sample_times_.resize(sample_count);
    std::generate(sample_times_.begin(), sample_times_.end(), [&]() {
        static int64_t i;
        auto ret = start + i * incr;
        ++i;
        return ret;
    });*/

    // begin tracy::Profiler::CalibrateTimer
    /*std::atomic_signal_fence(std::memory_order_acq_rel);
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

    timer_mul_ = double(dt) / double(dr);*/
    // end tracy::Profiler::CalibrateTimer


    using namespace visus::power_overwhelming;

    //setup_measurement();

    auto sensor_count = nvml_sensor::for_all(nullptr, 0);
    std::vector<nvml_sensor> tmp_sensors(sensor_count);
    nvml_sensor::for_all(tmp_sensors.data(), tmp_sensors.size());

    sensor_count = msr_sensor::for_all(nullptr, 0);
    std::vector<msr_sensor> tmp_msr_sensors(sensor_count);
    msr_sensor::for_all(tmp_msr_sensors.data(), tmp_msr_sensors.size());

    sensor_count = tinkerforge_sensor::for_all(nullptr, 0);
    std::vector<tinkerforge_sensor> tmp_tinker_sensors(sensor_count);
    tinkerforge_sensor::for_all(tmp_tinker_sensors.data(), tmp_tinker_sensors.size());

#ifdef MEGAMOL_USE_TRACY
    for (auto& sensor : tmp_sensors) {
        auto sensor_name = unmueller_string(sensor.name());

        TracyPlotConfig(sensor_name.data(), tracy::PlotFormatType::Number, false, true, 0);

        nvml_sensors_[sensor_name] = std::move(sensor);
        nvml_sensors_[sensor_name].sample(std::move(async_sampling()
                                                        .delivers_measurement_data_to(&tracy_sample)
                                                        .stores_and_passes_context(sensor_name)
                                                        .samples_every(1000Ui64)
                                                        .using_resolution(timestamp_resolution::microseconds)));
    }
    for (auto& sensor : tmp_msr_sensors) {
        auto sensor_name = unmueller_string(sensor.name());
        if (sensor_name.find("package") == std::string::npos || sensor_name.find("msr/0/") == std::string::npos) {
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
        //    1000Ui64, timestamp_resolution::microseconds, static_cast<void*>(sensor_names_.back().data()));
        msr_sensors_[sensor_name].sample(std::move(async_sampling()
                                                       .delivers_measurement_data_to(&tracy_sample)
                                                       .stores_and_passes_context(sensor_name)
                                                       .samples_every(1000Ui64)
                                                       .using_resolution(timestamp_resolution::microseconds)));
    }

    for (auto& sensor : tmp_tinker_sensors) {
        auto sensor_name = unmueller_string(sensor.name());

        TracyPlotConfig(sensor_name.data(), tracy::PlotFormatType::Number, false, true, 0);

        //sensor.sample([](const visus::power_overwhelming::measurement& m, void*) {
        //    auto name = unmueller_string(m.sensor());
        //    TracyPlot(name.data(), m.power());
        //});

        tinker_sensors_[sensor_name] = std::move(sensor);
        //sensor_names_.push_back(sensor_name);
        tinker_sensors_[sensor_name].reset();
        tinker_sensors_[sensor_name].configure(
            sample_averaging::average_of_4, conversion_time::microseconds_588, conversion_time::microseconds_588);
        //tinker_sensors_[sensor_name].sample(
        //    [](const visus::power_overwhelming::measurement& m, void* sensor_name) {
        //        //auto name = unmueller_string(sensor->name());
        //        TracyPlot(reinterpret_cast<char const*>(sensor_name), m.power());
        //    },
        //    tinkerforge_sensor_source::power, 1000Ui64, static_cast<void*>(sensor_names_.back().data()));
        tinker_sensors_[sensor_name].sample(std::move(async_sampling()
                                                          .delivers_measurement_data_to(&tracy_sample)
                                                          .stores_and_passes_context(sensor_name)
                                                          .samples_every(5000Ui64)
                                                          .using_resolution(timestamp_resolution::microseconds)
                                                          .from_source(tinkerforge_sensor_source::power)));
    }

    /*TracyPlotConfig("V", tracy::PlotFormatType::Number, false, true, 0);
    TracyPlotConfig("A", tracy::PlotFormatType::Number, false, true, 0);
    TracyPlotConfig("W", tracy::PlotFormatType::Number, false, true, 0);
    TracyPlotConfig("Frame", tracy::PlotFormatType::Number, false, true, 0);*/
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
    //#ifdef MEGAMOL_USE_TRACY
    //    for (auto& [name, sensor] : nvml_sensors_) {
    //        auto val = sensor.sample_data();
    //        TracyPlot(name.data(), val.power());
    //    }
    //    for (auto& [name, sensor] : msr_sensors_) {
    //        auto val = sensor.sample_data();
    //        TracyPlot(name.data(), val.power());
    //    }
    //    for (auto& [name, sensor] : tinker_sensors_) {
    //        auto val = sensor.sample_data();
    //        TracyPlot(name.data(), val.power());
    //    }
    //#endif
}

void Power_Service::setup_measurement() {
    using namespace visus::power_overwhelming;
    core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Starting setup");
    //auto m_func = [&]() -> void {
    try {
        for (auto& i : rtx_instr_) {
            i.synchronise_clock();
            i.reset(true, true);
            //i.operation_complete();

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
    } catch (std::exception& ex) {
        core::utility::log::Log::DefaultLog.WriteError("[Power_Service]: %s", ex.what());
    }
    //};
    //auto m_thread = std::thread(m_func);
    //m_thread.detach();
}

bool Power_Service::waiting_on_trigger() const {
    for (auto& i : rtx_instr_) {
        auto res = i.query("STAT:OPER:COND?\n");
        *strchr(res.as<char>(), '\n') = 0;
        auto val = std::atoi(res.as<char>());
        //core::utility::log::Log::DefaultLog.WriteInfo("WTR is %d", val);
        if ((val & 8)) {
            return true;
        }
    }
    return false;
}

std::vector<int64_t> generate_timestamps_ns(float begin, float end, float dis, size_t length) {
    std::vector<int64_t> ret(length);

    /*auto range = end - begin;
    auto incr = range / static_cast<float>(length);*/
    auto t_b_s = std::chrono::duration<float>(begin);
    auto t_b_ns = std::chrono::round<std::chrono::nanoseconds>(t_b_s);
    auto incr_s = std::chrono::duration<float>(dis);
    auto incr_ns = std::chrono::round<std::chrono::nanoseconds>(incr_s);

    int64_t iter = 0;
    std::generate(ret.begin(), ret.end(), [&iter, &t_b_ns, &incr_ns]() {
        auto ret = t_b_ns.count() + iter * incr_ns.count();
        ++iter;
        return ret;
    });

    return ret;
}

int64_t Power_Service::get_tracy_time(int64_t base, int64_t tracy_offset, float seg_off) const {
    auto seg_off_ticks = static_cast<int64_t>(static_cast<double>(seg_off) * static_cast<double>(qpc_frequency_));
    auto base_ticks =
        static_cast<int64_t>((static_cast<double>(base) / 1000. / 1000. / 1000.) * static_cast<double>(qpc_frequency_));
    return base_ticks + tracy_offset + seg_off_ticks;
}

int64_t Power_Service::get_tracy_time(int64_t base, int64_t tracy_offset) const {
    auto base_ticks =
        static_cast<int64_t>((static_cast<double>(base) / 1000. / 1000. / 1000.) * static_cast<double>(qpc_frequency_));
    return base_ticks + tracy_offset;
}


void Power_Service::write_to_files(std::string const& folder_path, file_type ft) const {
    std::filesystem::path path_to_folder(folder_path);
    if (!std::filesystem::exists(path_to_folder)) {
        core::utility::log::Log::DefaultLog.WriteError("[Power_Service]: Target folder does not exist");
        return;
    }

    if (!std::filesystem::is_directory(path_to_folder)) {
        core::utility::log::Log::DefaultLog.WriteError("[Power_Service]: Target path is not a folder");
        return;
    }

    for (size_t s_idx = 0; s_idx < values_map_.size(); ++s_idx) {
        auto const& values_map = values_map_[s_idx];

        auto full_path =
            path_to_folder / (std::string("pwr_") + std::string("s") + std::to_string(s_idx) + std::string(".parquet"));

        ParquetWriter(full_path, values_map);

        /*for (auto const& [name, values] : values_map) {

            std::stringstream stream;
            for (auto const& v : values) {

            }
            auto file = std::ofstream(full_path);
            file.close();
        }*/
    }
}


std::vector<float> Power_Service::examine_expression(std::string const& name, std::string const& exp_path, int s_idx) {
    sol_state_["s_idx"] = s_idx;
    sol_state_.script_file(exp_path);
    return sol_state_[name];
}


std::vector<float> transform_waveform(visus::power_overwhelming::oscilloscope_waveform const& wave) {
    std::vector<float> ret(wave.record_length());
    std::copy(wave.begin(), wave.end(), ret.begin());
    return ret;
}


void Power_Service::start_measurement() {
    using namespace visus::power_overwhelming;
    core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Starting measurement");
    auto m_func = [&]() -> void {
        try {
            pending_measurement_ = true;
            int counter = 0;
            for (auto& i : rtx_instr_) {
                i.on_operation_complete_ex([&counter](visa_instrument&) { ++counter; });
                i.acquisition(oscilloscope_acquisition_state::single);
                i.operation_complete_async();
            }

            std::chrono::high_resolution_clock::time_point last;
            bool first = true;
            while (counter < rtx_instr_.size()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                core::utility::log::Log::DefaultLog.WriteInfo("We are still waiting");
                if (waiting_on_trigger()) {
                    core::utility::log::Log::DefaultLog.WriteWarn("Do trigger");
                    auto now = std::chrono::high_resolution_clock::now();
                    trigger();
                    if (!first) {
                        auto d = now - last;
                        core::utility::log::Log::DefaultLog.WriteWarn(
                            "Trigger offset: %d", std::chrono::duration_cast<std::chrono::milliseconds>(d).count());
                    } else {
                        last = now;
                        first = false;
                    }
                }
            }
            //pending_read_ = true;
            core::utility::log::Log::DefaultLog.WriteInfo("Not waiting anymore");
#ifdef MEGAMOL_USE_TRACY
            for (auto const& [name, path] : exp_map_) {
                TracyPlotConfig(name.c_str(), tracy::PlotFormatType::Number, false, true, 0);
            }
#endif

            values_map_.clear();
            //std::vector<std::vector<int64_t>> sample_times;
            //std::vector<float> seg_off;

            for (auto& i : rtx_instr_) {
                //i.operation_complete();

                auto name = get_device_name(i);

                auto fit = config_map_.find(name);
                if (fit == config_map_.end()) {
                    core::utility::log::Log::DefaultLog.WriteError(
                        "[Power_Service]: Could not find config for device %s", name);
                    continue;
                }

                auto config = fit->second;
                auto num_channels = config.channels();
                std::vector<oscilloscope_channel> channels(num_channels);
                config.channels(channels.data(), num_channels);

                auto const num_segments = i.history_segments();
                core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service] Number of segments %d", num_segments);

                values_map_.resize(num_segments);
                //sample_times.resize(num_segments);
                //seg_off.resize(num_segments);

                // collecting waveforms
                auto all_waveforms = i.data(oscilloscope_waveform_points::maximum);
                for (size_t s_idx = 0; s_idx < num_segments; ++s_idx) {
                    //i.history_segment(s_idx + 1);
                    for (unsigned int c_idx = 0; c_idx < num_channels; ++c_idx) {
                        auto tpn = name + "_" + channels[c_idx].label().text();
                        auto const& waveform = all_waveforms[c_idx + s_idx * num_channels].waveform();
                        //auto waveform = i.data(c_idx + 1, oscilloscope_waveform_points::maximum);
                        values_map_[s_idx][tpn] = transform_waveform(waveform);
                        if (c_idx == 0) {
                            auto t_begin = waveform.time_begin();
                            auto t_end = waveform.time_end();
                            auto t_dis = waveform.sample_distance();
                            auto t_off = waveform.segment_offset();
                            auto r_length = waveform.record_length();

                            auto sample_times = generate_timestamps_ns(t_begin, t_end, t_dis, r_length);
                            values_map_[s_idx]["rel_time"] = sample_times;
                            std::vector<int64_t> abs_times(sample_times.size());
                            auto const t_off_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::duration<float>(t_off))
                                                      .count();
                            std::transform(sample_times.begin(), sample_times.end(), abs_times.begin(),
                                [&t_off_ns](int64_t s_time) { return s_time + t_off_ns; });
                            values_map_[s_idx]["abs_time"] = abs_times;
                            std::vector<int64_t> wall_times(abs_times.size());
                            auto const ltrg_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::time_point_cast<std::chrono::system_clock::duration>(last_trigger_) -
                                std::chrono::system_clock::from_time_t(0))
                                                     .count();
                            std::transform(abs_times.begin(), abs_times.end(), wall_times.begin(),
                                [&ltrg_ns](int64_t a_time) { return a_time + ltrg_ns; });
                            values_map_[s_idx]["wall_time"] = wall_times;

                            //seg_off[s_idx] = t_off;
                        }
                    }
                }


#if 0

                for (size_t s_idx = 0; s_idx < num_segments; ++s_idx) {
                    i.history_segment(s_idx + 1);

                    std::vector<oscilloscope_waveform> waveforms;
                    waveforms.reserve(num_channels);

                    for (auto w_idx = 0; w_idx < num_channels; ++w_idx) {
                        waveforms.push_back(i.data(w_idx + 1, oscilloscope_waveform_points::maximum));
                    }

                    if (!waveforms.empty()) {
                        auto t_begin = waveforms[0].time_begin();
                        auto t_end = waveforms[0].time_end();
                        auto t_dis = waveforms[0].sample_distance();

                        auto t_off = waveforms[0].segment_offset();
                        seg_off[s_idx] = t_off;

                        core::utility::log::Log::DefaultLog.WriteInfo(
                            "[Power_Service] Segment %d begin %f end %f dis %e with offset %e",
                            static_cast<int64_t>(s_idx + 1) - num_segments, t_begin, t_end, t_dis, t_off);

                        sample_times[s_idx] =
                            generate_timestamps_ns(t_begin, t_end, t_dis, waveforms[0].record_length());

                        std::vector<std::string> tpns;
                        tpns.reserve(num_channels);

                        std::ofstream out_file("channel_data_" + name + "_s" + std::to_string(s_idx + 1) + ".csv");
                        out_file << "time,";
                        for (auto const& chan : channels) {
                            core::utility::log::Log::DefaultLog.WriteInfo(
                                "[Power_Service] Channel label: %s", chan.label().text());
                            out_file << chan.label().text() << ",";
#ifdef MEGAMOL_USE_TRACY
                            auto tpn = name + "_" + chan.label().text();
                            TracyPlotConfig(tpn.c_str(), tracy::PlotFormatType::Number, false, true, 0);
                            tpns.push_back(tpn);
#endif
                        }

                        for (int vm = 0; vm < num_channels; ++vm) {
                            values_map_[s_idx][tpns[vm]] = transform_waveform(waveforms[vm]);
                        }

                        out_file.seekp(-1, std::ios_base::end);
                        out_file << "\n";
                        for (size_t i = 0; i < waveforms[0].record_length(); ++i) {
#ifdef MEGAMOL_USE_TRACY
                            std::vector<std::pair<std::string, float>> t_vals;
                            t_vals.reserve(waveforms.size());
#endif
                            out_file << sample_times[s_idx][i];
                            for (int w_idx = 0; w_idx < waveforms.size(); ++w_idx) {
                                out_file << "," << waveforms[w_idx].begin()[i];
#ifdef MEGAMOL_USE_TRACY
                                tracy::Profiler::PlotData(tpns[w_idx].c_str(), waveforms[w_idx].begin()[i],
                                    get_tracy_time(sample_times[s_idx][i], tracy_last_trigger_, t_off));

                                t_vals.emplace_back(tpns[w_idx], waveforms[w_idx].begin()[i]);
#endif
                            }
                            out_file << std::endl;
                        }
                        out_file.close();
                    }
                }
#endif
                core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Completed measurement");
            }

            core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Start writing");

            // writing data
#ifdef MEGAMOL_USE_TRACY
            for (size_t s_idx = 0; s_idx < values_map_.size(); ++s_idx) {
                auto const& values_map = values_map_[s_idx];
                auto const& samples_time = std::get<std::vector<int64_t>>(values_map.at("abs_time"));
                for (auto const& [name, v_values] : values_map) {
                    if (std::holds_alternative<std::vector<float>>(v_values)) {
                        auto c_name = name.c_str();
                        auto const& values = std::get<std::vector<float>>(v_values);
                        TracyPlotConfig(c_name, tracy::PlotFormatType::Number, false, true, 0);
                        for (std::size_t v_idx = 0; v_idx < values.size(); ++v_idx) {
                            tracy::Profiler::PlotData(
                                c_name, values[v_idx], get_tracy_time(samples_time[v_idx], tracy_last_trigger_));
                        }
                    }
                }
            }
#endif

            // evaluate expressions
#ifdef MEGAMOL_USE_TRACY
            for (int s_idx = 0; s_idx < values_map_.size(); ++s_idx) {
                auto const& samples_time = std::get<std::vector<int64_t>>(values_map_[s_idx].at("abs_time"));
                for (auto const& [name, exp_path] : exp_map_) {
                    auto val = examine_expression(name, exp_path, s_idx);
                    for (size_t i = 0; i < val.size(); ++i) {
                        tracy::Profiler::PlotData(
                            name.c_str(), val[i], get_tracy_time(samples_time[i], tracy_last_trigger_));
                    }
                }
            }
#endif

            if (write_to_files_) {
                write_to_files(write_folder_, file_type::RAW);
            }

            core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Completed writing");

            //pending_read_ = false;
            pending_measurement_ = false;
        } catch (std::exception& ex) {
            core::utility::log::Log::DefaultLog.WriteError("[Power_Service]: %s", ex.what());
        }
    };
    auto m_thread = std::thread(m_func);
    m_thread.detach();
}

void Power_Service::trigger() {
#ifdef MEGAMOL_USE_TRACY
    ZoneScopedNC("Power_Service::trigger", 0xDB0ABF);
#endif
    have_triggered_ = true;
    last_trigger_ = std::chrono::system_clock::now();
#ifdef WIN32
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    tracy_last_trigger_ = t.QuadPart;
#else
    timespec tp;
    clock_gettime(CLOCK_MONOTONIC_RAW, &tp);
    tracy_last_trigger_ = tp.tv_nsec;
#endif
    if (enforce_software_trigger_) {
        for (auto& i : rtx_instr_) {
            i.trigger_manually();
        }
    } else {
        trigger_->SetBit(6, true);
        trigger_->SetBit(6, false);
    }
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
            return frontend_resources::LuaCallbacksCollection::BoolResult{pending_measurement_};
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult>(
        "mmPowerForceTrigger", "()", {[&]() -> frontend_resources::LuaCallbacksCollection::VoidResult {
            for (auto& i : rtx_instr_) {
                i.trigger_manually();
            }
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult, bool>("mmPowerSoftwareTrigger", "(bool set)",
        {[&](bool set) -> frontend_resources::LuaCallbacksCollection::VoidResult {
            enforce_software_trigger_ = set;
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult, std::string, std::string>(
        "mmPowerRegisterTracyExp", "(string name, string path)",
        {[&](std::string name, std::string path) -> frontend_resources::LuaCallbacksCollection::VoidResult {
            exp_map_[name] = path;
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    auto& register_callbacks =
        m_requestedResourceReferences[0]
            .getResource<std::function<void(frontend_resources::LuaCallbacksCollection const&)>>();

    register_callbacks(callbacks);
}

} // namespace frontend
} // namespace megamol

#endif
