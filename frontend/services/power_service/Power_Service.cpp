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

#include <fstream>
#include <regex>
#include <stdexcept>

#include "LuaCallbacksCollection.h"

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

Power_Service::Power_Service() {
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
    using namespace visus::power_overwhelming;

    auto devices = visa_instrument::find_resources("0x0AAD", "0x01D6");

    for (auto d = devices.as<char>(); (d != nullptr) && (*d != 0); d += strlen(d) + 1) {
        core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Found device %s", d);

        rtx_instr_.emplace_back(d);
    }

    setup_measurement();

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
    fill_lua_callbacks();

    // prepare usage of requested resources
    //this->m_externalResource_1_ptr =
    //    &resources[0].getResource<ExternalResource_1>(); // resources are in requested order and not null
    //this->m_externalResource_2_ptr =
    //    &resources[1]
    //         .getResource<
    //             namspace::to::resource::ExternalResource_2>(); // ptr will be not null or program terminates by design
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
}

void Power_Service::setup_measurement() {
    using namespace visus::power_overwhelming;
    core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Starting setup");
    //auto m_func = [&]() -> void {
    try {
        //auto devices = visa_instrument::find_resources("0x0AAD", "0x01D6");

        //for (auto d = devices.as<char>(); (d != nullptr) && (*d != 0); d += strlen(d) + 1) {
        for (auto& i : rtx_instr_) {
            //core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Found device %s", d);

            //rtx_instrument i(d);

            i.synchronise_clock();
            i.reset(true, true);
            i.timeout(5000);

            i.reference_position(oscilloscope_reference_point::left);
            i.time_range(oscilloscope_quantity(50, "ms"));

            i.channel(oscilloscope_channel(1)
                          .label(oscilloscope_label("voltage"))
                          .state(true)
                          .attenuation(oscilloscope_quantity(10, "V"))
                          .range(oscilloscope_quantity(26)));

            i.channel(oscilloscope_channel(2)
                          .label(oscilloscope_label("current"))
                          .state(true)
                          .attenuation(oscilloscope_quantity(10, "A"))
                          .range(oscilloscope_quantity(40)));

            i.channel(oscilloscope_channel(3)
                          .label(oscilloscope_label("frame"))
                          .state(true)
                          .attenuation(oscilloscope_quantity(1, "V"))
                          .range(oscilloscope_quantity(7)));


            i.trigger_position(oscilloscope_quantity(0.f, "ms"));
            i.trigger(oscilloscope_edge_trigger("EXT")
                          .level(5, oscilloscope_quantity(2000.0f, "mV"))
                          .slope(oscilloscope_trigger_slope::rising)
                          .mode(oscilloscope_trigger_mode::normal));

            i.acquisition(oscilloscope_single_acquisition().points(50000).count(2).segmented(true));

            /*std::cout << "RTX interface type: " << i.interface_type() << std::endl
                      << "RTX status before acquire: " << i.status() << std::endl;*/

            i.operation_complete();

            core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Completed setup");

            //i.acquisition(oscilloscope_acquisition_state::single);

            //trigger_->SetBit(6, true);
            //trigger_->SetBit(6, false);

            //i.operation_complete();

            //auto segment0_1 = i.data(1, oscilloscope_waveform_points::maximum);
            ////i.clear();
            //auto segment0_2 = i.data(2, oscilloscope_waveform_points::maximum);

            //auto segment0_3 = i.data(3, oscilloscope_waveform_points::maximum);

            //core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service] Started writing");
            //std::ofstream out_file("channel_data_0.csv");
            //for (size_t i = 0; i < segment0_1.record_length(); ++i) {
            //    out_file << segment0_1.begin()[i] << "," << segment0_2.begin()[i] << "," << segment0_3.begin()[i]
            //             << std::endl;
            //}
            //out_file.close();
        }
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
            trigger_->SetBit(6, true);
            trigger_->SetBit(6, false);

            for (auto& i : rtx_instr_) {

                i.operation_complete();

                auto segment0_1 = i.data(1, oscilloscope_waveform_points::maximum);
                //i.clear();
                auto segment0_2 = i.data(2, oscilloscope_waveform_points::maximum);

                auto segment0_3 = i.data(3, oscilloscope_waveform_points::maximum);

                core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service] Started writing");
                std::ofstream out_file("channel_data_0.csv");
                for (size_t i = 0; i < segment0_1.record_length(); ++i) {
                    out_file << segment0_1.begin()[i] << "," << segment0_2.begin()[i] << "," << segment0_3.begin()[i]
                             << std::endl;
                }
                out_file.close();

                core::utility::log::Log::DefaultLog.WriteInfo("[Power_Service]: Completed measurement");
            }
        } catch (std::exception& ex) {
            core::utility::log::Log::DefaultLog.WriteError("[Power_Service]: %s", ex.what());
        }
    };
    auto m_thread = std::thread(m_func);
    m_thread.detach();
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

    auto& register_callbacks =
        m_requestedResourceReferences[0]
            .getResource<std::function<void(frontend_resources::LuaCallbacksCollection const&)>>();

    register_callbacks(callbacks);
}

} // namespace frontend
} // namespace megamol

#endif
