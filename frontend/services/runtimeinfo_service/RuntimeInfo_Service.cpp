/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

// search/replace Template_Service with your class name
// you should also delete the FAQ comments in these template files after you read and understood them
#include "RuntimeInfo_Service.hpp"


// local logging wrapper for your convenience until central MegaMol logger established
#include "mmcore/utility/log/Log.h"

static const std::string service_name = "RuntimeInfo_Service: ";
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

RuntimeInfo_Service::RuntimeInfo_Service() {
    // init members to default states
}

RuntimeInfo_Service::~RuntimeInfo_Service() {
    // clean up raw pointers you allocated with new, which is bad practice and nobody does
}

bool RuntimeInfo_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    return init(*static_cast<Config*>(configPtr));
}

bool RuntimeInfo_Service::init(const Config& config) {
    // initialize your service and its provided resources using config parameters
    // for now, you dont need to worry about your service beeing initialized or closed multiple times
    // init() and close() only get called once in the lifetime of each service object
    // but maybe more instances of your service will get created? this may be relevant for central resources you manage (like libraries, network connections).

    ri_resource_.get_hardware_info = [&]() { return get_hardware_info(); };
    ri_resource_.get_os_info = [&]() { return get_os_info(); };
    ri_resource_.get_runtime_libraries = [&]() { return get_runtime_libraries(); };

    m_providedResourceReferences = {{"RuntimeInfo", ri_resource_}};

    log("initialized successfully");
    return true;
}

void RuntimeInfo_Service::close() {
    // close libraries or APIs you manage
    // wrap up resources your service provides, but don not depend on outside resources to be available here
    // after this, at some point only the destructor of your service gets called
}

std::vector<FrontendResource>& RuntimeInfo_Service::getProvidedResources() {
    return m_providedResourceReferences;
}

const std::vector<std::string> RuntimeInfo_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void RuntimeInfo_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    // maybe we want to keep the list of requested resources
    this->m_requestedResourceReferences = resources;
}

void RuntimeInfo_Service::updateProvidedResources() {}

void RuntimeInfo_Service::digestChangedRequestedResources() {}

void RuntimeInfo_Service::resetProvidedResources() {
    // this gets called at the end of the main loop iteration
    // since the current resources state should have been handled in this frame already
    // you may clean up resources whose state is not needed for the next iteration
    // e.g. m_keyboardEvents.clear();
    // network_traffic_buffer.reset_to_empty();
}

void RuntimeInfo_Service::preGraphRender() {
    // this gets called right before the graph is told to render something
    // e.g. you can start a start frame timer here

    // rendering via MegaMol View is called after this function finishes
    // in the end this calls the equivalent of ::mmcRenderView(hView, &renderContext)
    // which leads to view.Render()
}

void RuntimeInfo_Service::postGraphRender() {
    // the graph finished rendering and you may more stuff here
    // e.g. end frame timer
    // update window name
    // swap buffers, glClear
}

std::string RuntimeInfo_Service::get_hardware_info() {
    return ri_.GetHardwareInfo();
}

std::string RuntimeInfo_Service::get_os_info() {
    return ri_.GetOsInfo();
}

std::string RuntimeInfo_Service::get_runtime_libraries() {
    return ri_.GetRuntimeLibraries();
}


} // namespace frontend
} // namespace megamol
