/*
 * ExoticInputs_Service.cpp
 *
 * Copyright (C) 2022 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ExoticInputs_Service.hpp"

#include "GamepadState.h"

#include "GUIRegisterWindow.h" // register UI window for remote control
#include "imgui_stdlib.h"

// local logging wrapper for your convenience until central MegaMol logger established
#include "mmcore/utility/log/Log.h"

#include <algorithm>

static const std::string service_name = "ExoticInputs_Service: ";
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

ExoticInputs_Service::ExoticInputs_Service() {
    // init members to default states
}

ExoticInputs_Service::~ExoticInputs_Service() {
    // clean up raw pointers you allocated with new, which is bad practice and nobody does
}

bool ExoticInputs_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    return init(*static_cast<Config*>(configPtr));
}

bool ExoticInputs_Service::init(const Config& config) {
    // initialize your service and its provided resources using config parameters
    // for now, you dont need to worry about your service beeing initialized or closed multiple times
    // init() and close() only get called once in the lifetime of each service object
    // but maybe more instances of your service will get created? this may be relevant for central resources you manage (like libraries, network connections).

    m_providedResourceReferences = {};

    m_requestedResourcesNames = {
        "optional<Connected_Gamepads>",
        "optional<GUIRegisterWindow>",
    };

    log("initialized successfully");
    return true;
}

void ExoticInputs_Service::close() {}

std::vector<FrontendResource>& ExoticInputs_Service::getProvidedResources() {
    return m_providedResourceReferences;
}

const std::vector<std::string> ExoticInputs_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void ExoticInputs_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    m_requestedResourceReferences = resources;

    gamepad_window();
}

void ExoticInputs_Service::updateProvidedResources() {}

void ExoticInputs_Service::digestChangedRequestedResources() {}

void ExoticInputs_Service::resetProvidedResources() {}

void ExoticInputs_Service::preGraphRender() {}

void ExoticInputs_Service::postGraphRender() {}

void ExoticInputs_Service::gamepad_window() const {
    auto maybe_gamepad_resource =
        m_requestedResourceReferences[0].getOptionalResource<megamol::frontend_resources::Connected_Gamepads>();
    auto maybe_window_resource =
        m_requestedResourceReferences[1].getOptionalResource<megamol::frontend_resources::GUIRegisterWindow>();

    if (maybe_gamepad_resource.has_value() && maybe_window_resource.has_value()) {
        // draw window showing gamepad stats
        auto& gui_window = maybe_window_resource.value().get();
        auto& connected_gamepads = maybe_gamepad_resource.value().get();

        gui_window.register_window("GLFW Gamepads ", [&](megamol::gui::AbstractWindow::BasicConfig& window_config) {
            for (auto& gamepad : connected_gamepads.gamepads) {
                window_config.flags = ImGuiWindowFlags_AlwaysAutoResize;

                ImGui::Text(gamepad.get().name.c_str());

                int i = 0;
                for (auto axis : gamepad.get().axes) {
                    ImGui::Text(("Axis " + std::to_string(i) + ": " + std::to_string(axis)).c_str());
                    ImGui::SameLine();
                    i++;
                }
                ImGui::NewLine();

                i = 0;
                for (auto button : gamepad.get().buttons) {
                    ImGui::Text(("Button " + std::to_string(i) + ": " + std::to_string(button)).c_str());
                    ImGui::SameLine();
                    i++;
                }
                ImGui::NewLine();
                ImGui::Separator();
            }
        });
    }
}


} // namespace frontend
} // namespace megamol
