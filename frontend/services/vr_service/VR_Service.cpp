/*
 * VR_Service.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "VR_Service.hpp"

// local logging wrapper for your convenience until central MegaMol logger established
#include "mmcore/utility/log/Log.h"

static const std::string service_name = "VR_Service: ";
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

VR_Service::VR_Service() {}

VR_Service::~VR_Service() {}

bool VR_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    return init(*static_cast<Config*>(configPtr));
}

bool VR_Service::init(const Config& config) {

    m_requestedResourcesNames = {};
    log("initialized successfully");
    return true;
}

void VR_Service::close() {}

std::vector<FrontendResource>& VR_Service::getProvidedResources() {
    m_providedResourceReferences = {};

    return m_providedResourceReferences;
}

const std::vector<std::string> VR_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void VR_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    this->m_requestedResourceReferences = resources;
}

void VR_Service::updateProvidedResources() {}

void VR_Service::digestChangedRequestedResources() {}

void VR_Service::resetProvidedResources() {}

void VR_Service::preGraphRender() {}

void VR_Service::postGraphRender() {}

} // namespace frontend
} // namespace megamol
