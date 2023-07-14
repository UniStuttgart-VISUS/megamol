/*
 * OpenGL_GLFW_Service.cpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "FrontendServiceCollection.hpp"

#include "FrontendResourcesLookup.h"

#include <algorithm>
#include <iostream>
#include <numeric>

#include "mmcore/utility/log/Log.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

static void log(std::string const& text) {
    const std::string msg = "FrontendServiceCollection: " + text;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}

static void log_error(std::string const& text) {
    const std::string msg = "FrontendServiceCollection: " + text;
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}

static void log_warning(std::string const& text) {
    const std::string msg = "FrontendServiceCollection: " + text;
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(msg.c_str());
}

namespace megamol::frontend {

void FrontendServiceCollection::sortServices() {
    std::sort(m_services.begin(), m_services.end(),
        [](auto& lhs, auto& rhs) { return lhs.get().getPriority() < rhs.get().getPriority(); });
}

void FrontendServiceCollection::add(AbstractFrontendService& service, void* service_config) {
    m_services.push_back({&service, service_config});
    this->sortServices();
}

#define for_each_service for (auto& service : m_services)

bool FrontendServiceCollection::init() {
    for_each_service {
        if (!service.service->init(service.service_config)) {
            log_error(service.service->serviceName() + " failed init");
            return false;
        }
    }

    return true;
}

void FrontendServiceCollection::close() {
    // close services in reverse order,
    // hopeully avoiding dangling references to closed resources
    for (auto it = m_services.rbegin(); it != m_services.rend(); it++) {
        (*it).get().close();
    }
}

std::vector<FrontendResource>& FrontendServiceCollection::getProvidedResources() {
    return m_serviceResources;
}

// TODO: using the requested resource dependencies we can derive correct update order of services
bool FrontendServiceCollection::assignRequestedResources() {
    // gather resources from all services
    for_each_service {
        auto& resources = service.get().getProvidedResources();

        // attention: ModuleResources are copyable because of the reference wrapper used internally
        // copying stuff should not break anything, but this frontend manager should be the only one
        // copying ModuleResources because he knows what he is doing
        for (auto& r : resources)
            m_serviceResources.push_back(r);
    }

    megamol::frontend_resources::FrontendResourcesLookup m_resource_lookup{m_serviceResources};

    bool overall_success = true;

    // for each servie, provide him with requested resources
    for_each_service {
        auto request_names = service.get().getRequestedResourceNames();
        auto [success, resources] = m_resource_lookup.get_requested_resources(request_names);

        overall_success &= success;

        if (success) {
            service.get().setRequestedResources(resources);
        } else {
            // if a requested resource can not be found we fail and should stop program execution
            log_error("could not find all resources: for service: " + service.get().serviceName() + "\nRequests: " +
                      std::accumulate(request_names.begin(), request_names.end(), std::string{},
                          [](std::string const& lhs, std::string const& rhs) { return lhs + ", " + rhs; }) +
                      "\nFound Resources: " +
                      std::accumulate(resources.begin(), resources.end(), std::string{},
                          [](std::string const& lhs, megamol::frontend::FrontendResource const& rhs) {
                              return lhs + ", " + rhs.getIdentifier();
                          }));
        }
    }

    return overall_success;
}

static auto const t_service_color = 0x4DAE59;

void FrontendServiceCollection::updateProvidedResources() {
    for_each_service {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedC(t_service_color);
        auto const name = service.get().serviceName() + "::updateProvidedResources";
        ZoneName(name.c_str(), name.size());
#endif
        service.get().updateProvidedResources();
    }
}

void FrontendServiceCollection::digestChangedRequestedResources() {
    for_each_service {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedC(t_service_color);
        auto const name = service.get().serviceName() + "::digestChangedRequestedResources";
        ZoneName(name.c_str(), name.size());
#endif
        service.get().digestChangedRequestedResources();
    }
}

void FrontendServiceCollection::resetProvidedResources() {
    for_each_service {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedC(t_service_color);
        auto const name = service.get().serviceName() + "::resetProvidedResources";
        ZoneName(name.c_str(), name.size());
#endif
        service.get().resetProvidedResources();
    }
}

void FrontendServiceCollection::preGraphRender() {
    for_each_service {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedC(t_service_color);
        auto const name = service.get().serviceName() + "::preGraphRender";
        ZoneName(name.c_str(), name.size());
#endif
        service.get().preGraphRender();
    }
}

void FrontendServiceCollection::postGraphRender() {
    // traverse post update in reverse order
    for (auto it = m_services.rbegin(); it != m_services.rend(); it++) {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedC(t_service_color);
        auto const name = (*it).get().serviceName() + "::postGraphRender";
        ZoneName(name.c_str(), name.size());
#endif
        (*it).get().postGraphRender();
    }
}

bool FrontendServiceCollection::shouldShutdown() const {
    bool shutdown = false;

    for (auto const& service : m_services) {
        bool shut_this = service.get().shouldShutdown();
        shutdown |= shut_this;

        if (shut_this) {
            log(service.get().serviceName() + " requests shutdown");
        }
    }

    return shutdown;
}

} // namespace megamol::frontend
