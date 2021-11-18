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

    m_requestedResourcesNames = {
        "ImagePresentationEntryPoints",
    };

    switch (config.mode) {
    case Config::Mode::Off:
        break;
    default:
        log_error("Unknown VR Service Mode: " + std::to_string(static_cast<int>(config.mode)));
        return false;
        break;
    }

    log("initialized successfully");
    return true;
}

void VR_Service::close() {
    m_vr_device_ptr.reset();
}

std::vector<FrontendResource>& VR_Service::getProvidedResources() {
    m_providedResourceReferences = {};

    return m_providedResourceReferences;
}

const std::vector<std::string> VR_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

std::string VR_Service::vr_service_marker = "#vr_service";

auto mark = [](auto const& name) { return name + VR_Service::vr_service_marker; };

void VR_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    this->m_requestedResourceReferences = resources;

    auto& entry_points =
        m_requestedResourceReferences[0].getResource<frontend_resources::ImagePresentationEntryPoints>();

    using SubscriptionEvent = frontend_resources::ImagePresentationEntryPoints::SubscriptionEvent;

    entry_points.subscribe_to_entry_point_changes(
        [&](SubscriptionEvent const& event, std::vector<std::any> const& arguments) {
            std::string entry_point_name;

            if (arguments.size() > 0) {
                entry_point_name = std::any_cast<std::string>(arguments[0]);

                if (entry_point_name.find(vr_service_marker) != std::string::npos) {
                    return;
                }

                if (entry_point_name.find("GUI Service") != std::string::npos) {
                    return;
                }
            }

            switch (event) {
            case SubscriptionEvent::Add:
                m_entry_points_registry.add_entry_point(
                    entry_point_name, std::any_cast<frontend_resources::EntryPointRenderFunctions const>(arguments[1]));
                break;
            case SubscriptionEvent::Remove:
                m_entry_points_registry.remove_entry_point(entry_point_name);
                break;
            case SubscriptionEvent::Rename:
                m_entry_points_registry.rename_entry_point(
                    entry_point_name, std::any_cast<std::string const>(arguments[1]));
                break;
            case SubscriptionEvent::Clear:
                m_entry_points_registry.clear_entry_points();
                break;
            default:
                log_error("unknown ImagePresentationEntryPoints::SubscriptionEvent type");
                break;
            }
        });

    // VR stereo rendering: clone each view entry point in order to have two images rendered: left + right
    m_entry_points_registry.add_entry_point = [&](auto const& name, auto const& callbacks) -> bool {
        vr_device(add_entry_point(
            name, callbacks, const_cast<frontend_resources::ImagePresentationEntryPoints&>(entry_points)));

        return true;
    };
    m_entry_points_registry.remove_entry_point = [&](auto const& name) -> bool {
        vr_device(
            remove_entry_point(name, const_cast<frontend_resources::ImagePresentationEntryPoints&>(entry_points)));

        return true;
    };
    m_entry_points_registry.rename_entry_point = [&](auto const& name, auto const& newname) -> bool { return true; };
    m_entry_points_registry.clear_entry_points = [&]() -> void { vr_device(clear_entry_points()); };
    m_entry_points_registry.subscribe_to_entry_point_changes = [&](auto const& func) -> void {};
    m_entry_points_registry.get_entry_point = [&](std::string const& name) -> auto {
        return std::nullopt;
    };
}

void VR_Service::updateProvidedResources() {}

void VR_Service::digestChangedRequestedResources() {}

void VR_Service::resetProvidedResources() {}

void VR_Service::preGraphRender() {
    vr_device(preGraphRender());
}

void VR_Service::postGraphRender() {
    vr_device(postGraphRender());
}

} // namespace frontend
} // namespace megamol
