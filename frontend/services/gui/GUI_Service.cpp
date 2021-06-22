/*
 * GUI_Service.cpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */


#include "GUI_Service.hpp"
#include "Window_Events.h"
#include "Framebuffer_Events.h"
#include "KeyboardMouse_Events.h"
#include "ScriptPaths.h"
#include "ProjectLoader.h"
#include "FrameStatistics.h"
#include "RuntimeConfig.h"
#include "WindowManipulation.h"
#include "mmcore/utility/log/Log.h"


namespace megamol {
namespace frontend {

#define VALIDATE_GUI_PTR(RETVAL)  if (static_cast<bool>((this->m_gui == nullptr) || (this->m_gui->Get() == nullptr))) return RETVAL; auto gui = this->m_gui->Get();

bool GUI_Service::init(void* configPtr) {

    if (configPtr == nullptr) return false;
    return init(*static_cast<Config*>(configPtr));
}


bool GUI_Service::init(const Config& config) {

    this->m_time = 0.0;
    this->m_framebuffer_size = glm::vec2(1.0f, 1.0f);
    this->m_window_size = glm::vec2(1.0f, 1.0f);
    this->m_megamol_graph = nullptr;
    this->m_gui = nullptr;

    this->m_queuedProjectFiles.clear();
    this->m_requestedResourceReferences.clear();
    this->m_providedResourceReferences.clear();
    this->m_requestedResourcesNames = {
        "MegaMolGraph",                          // 0 - sync graph
        "WindowEvents",                          // 1 - time, size, clipboard
        "KeyboardEvents",                        // 2 - key press
        "MouseEvents",                           // 3 - mouse click
        "IOpenGL_Context",                       // 4 - graphics api for imgui context
        "FramebufferEvents",                     // 5 - viewport size
        "GLFrontbufferToPNG_ScreenshotTrigger",  // 6 - trigger screenshot
        "LuaScriptPaths",                        // 7 - current project path
        "ProjectLoader",                         // 8 - trigger loading of new running project
        "FrameStatistics",                       // 9 - current fps and ms value
        "RuntimeConfig",                         // 10 - resource paths
        "WindowManipulation"                     // 11 - GLFW window pointer
    };

    // init gui
    if (config.imgui_api == GUI_Service::ImGuiAPI::OPEN_GL) { 
        if (this->m_gui == nullptr) {
            this->m_gui = std::make_shared<megamol::gui::GUIWrapper>();
            if (this->m_gui != nullptr) {
                auto gui = this->m_gui->Get();

                // Create context
                if (gui->CreateContext(megamol::gui::GUIImGuiAPI::OPEN_GL, config.core_instance)) {

                    // Set function pointer in state resource once
                    this->m_providedStateResource.request_gui_state = [&](bool as_lua) -> std::string {return this->resource_request_gui_state(as_lua);};
                    this->m_providedStateResource.provide_gui_state = [&](const std::string& json_state) -> void {
                        return this->resource_provide_gui_state(json_state);
                    };
                    this->m_providedStateResource.request_gui_visibility = [&]() -> bool {return this->resource_request_gui_visibility();};
                    this->m_providedStateResource.provide_gui_visibility = [&](bool show) -> void {
                        return this->resource_provide_gui_visibility(show);
                    };
                    this->m_providedStateResource.request_gui_scale = [&]() -> float {return this->resource_request_gui_scale();};
                    this->m_providedStateResource.provide_gui_scale = [&](float scale) -> void {
                        return this->resource_provide_gui_scale(scale);
                    };
                    this->resource_provide_gui_visibility(config.gui_show);
                    this->resource_provide_gui_scale(config.gui_scale);

                    this->m_providedRegisterWindowResource.register_window = [&](const std::string& name, std::function<void(megamol::gui::WindowConfiguration::Basic&)> func) -> void {
                        this->resource_register_window(name, func);
                    };
                    this->m_providedRegisterWindowResource.register_popup = [&](const std::string& name, bool& open, std::function<void(void)> func) -> void {
                        this->resource_register_popup(name, open, func);
                    };
                    this->m_providedRegisterWindowResource.register_notification = [&](const std::string& name, bool& open, const std::string& message) -> void {
                        this->resource_register_notification(name, open, message);
                    };

                    gui->SetVisibility(config.gui_show);
                    gui->SetScale(config.gui_scale);

                    megamol::core::utility::log::Log::DefaultLog.WriteInfo("GUI_Service: initialized successfully.");
                    return true;
                }
            }
        }
    }


    return false;
}


void GUI_Service::close() {

}
    

void GUI_Service::updateProvidedResources() {

}


void GUI_Service::digestChangedRequestedResources() {

    VALIDATE_GUI_PTR()

    // Trigger shutdown
    this->setShutdown(gui->GetTriggeredShutdown());

    // Check for updates in requested resources --------------------------------

    /// MegaMolGraph = resource index 0
    auto graph_resource_ptr = &this->m_requestedResourceReferences[0].getResource<megamol::core::MegaMolGraph>();
    // Synchronise changes between core graph and gui graph
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    this->m_megamol_graph = const_cast<megamol::core::MegaMolGraph*>(graph_resource_ptr);

    /// WindowEvents = resource index 1
    auto window_events =
        &this->m_requestedResourceReferences[1].getResource<megamol::frontend_resources::WindowEvents>();
    this->m_time = window_events->time;
    for (auto& size_event : window_events->size_events) {
        this->m_window_size.x = static_cast<float>(std::get<0>(size_event));
        this->m_window_size.y = static_cast<float>(std::get<1>(size_event));
    }
    gui->SetClipboardFunc(window_events->_getClipboardString_Func, window_events->_setClipboardString_Func,
        window_events->_clipboard_user_data);

    /// KeyboardEvents = resource index 2
    auto keyboard_events =
        &this->m_requestedResourceReferences[2].getResource<megamol::frontend_resources::KeyboardEvents>();
    std::vector<std::tuple<megamol::frontend_resources::Key, megamol::frontend_resources::KeyAction,
        megamol::frontend_resources::Modifiers>>
        pass_key_events;
    for (auto& key_event : keyboard_events->key_events) {
        auto key = std::get<0>(key_event);
        auto action = std::get<1>(key_event);
        auto modifiers = std::get<2>(key_event);
        if (!gui->OnKey(key, action, modifiers)) {
            pass_key_events.emplace_back(key_event);
        }
    }
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    const_cast<megamol::frontend_resources::KeyboardEvents*>(keyboard_events)->key_events = pass_key_events;

    std::vector<unsigned int> pass_codepoint_events;
    for (auto& codepoint_event : keyboard_events->codepoint_events) {
        if (!gui->OnChar(codepoint_event)) {
            pass_codepoint_events.emplace_back(codepoint_event);
        }
    }
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    const_cast<megamol::frontend_resources::KeyboardEvents*>(keyboard_events)->codepoint_events = pass_codepoint_events;

    /// MouseEvents = resource index 3
    auto mouse_events = &this->m_requestedResourceReferences[3].getResource<megamol::frontend_resources::MouseEvents>();
    std::vector<std::tuple<double, double>> pass_mouse_pos_events;
    for (auto& position_event : mouse_events->position_events) {
        auto x_pos = std::get<0>(position_event);
        auto y_pos = std::get<1>(position_event);
        if (!gui->OnMouseMove(x_pos, y_pos)) {
            pass_mouse_pos_events.emplace_back(position_event);
        }
    }
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    const_cast<megamol::frontend_resources::MouseEvents*>(mouse_events)->position_events = pass_mouse_pos_events;

    std::vector<std::tuple<double, double>> pass_mouse_scroll_events;
    for (auto& scroll_event : mouse_events->scroll_events) {
        auto x_scroll = std::get<0>(scroll_event);
        auto y_scroll = std::get<1>(scroll_event);
        if (!gui->OnMouseScroll(x_scroll, y_scroll)) {
            pass_mouse_scroll_events.emplace_back(scroll_event);
        }
    }
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    const_cast<megamol::frontend_resources::MouseEvents*>(mouse_events)->scroll_events = pass_mouse_scroll_events;

    std::vector<std::tuple<megamol::frontend_resources::MouseButton, megamol::frontend_resources::MouseButtonAction,
        megamol::frontend_resources::Modifiers>>
        pass_mouse_btn_events;
    for (auto& button_event : mouse_events->buttons_events) {
        auto button = std::get<0>(button_event);
        auto action = std::get<1>(button_event);
        auto modifiers = std::get<2>(button_event);
        if (!gui->OnMouseButton(button, action, modifiers)) {
            pass_mouse_btn_events.emplace_back(button_event);
        }
    }
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    const_cast<megamol::frontend_resources::MouseEvents*>(mouse_events)->buttons_events = pass_mouse_btn_events;

    /// FramebufferEvents = resource index 5
    auto framebuffer_events =
        &this->m_requestedResourceReferences[5].getResource<megamol::frontend_resources::FramebufferEvents>();
    for (auto& size_event : framebuffer_events->size_events) {
        this->m_framebuffer_size.x = static_cast<float>(size_event.width);
        this->m_framebuffer_size.y = static_cast<float>(size_event.height);
    }

    /// Trigger Screenshot = resource index 6
    if (gui->GetTriggeredScreenshot()) {
        auto& screenshot_to_file_trigger =
            this->m_requestedResourceReferences[6].getResource<std::function<bool(std::string const&)>>();
        screenshot_to_file_trigger(gui->GetScreenshotFileName());
    }

    /// Pipe lua script paths to gui = resource index 7
    auto& script_paths = this->m_requestedResourceReferences[7].getResource<megamol::frontend_resources::ScriptPaths>();
    gui->SetProjectScriptPaths(script_paths.lua_script_paths);

    /// Pipe project loading request from GUI to project loader = resource index 8
    auto requested_project_file = gui->GetProjectLoadRequest();
    if (!requested_project_file.empty()) {
        auto& project_loader =
            this->m_requestedResourceReferences[8].getResource<megamol::frontend_resources::ProjectLoader>();
        project_loader.load_filename(requested_project_file);
    }

    /// Get current FPS and MS frame statistic = resource index 9
    auto& frame_statistics =  this->m_requestedResourceReferences[9].getResource<megamol::frontend_resources::FrameStatistics>();
    gui->SetFrameStatistics(frame_statistics.last_averaged_fps, frame_statistics.last_averaged_mspf, frame_statistics.rendered_frames_count);

    /// Get window manipulation resource = resource index 11
    auto& window_manipulation = this->m_requestedResourceReferences[11].getResource<megamol::frontend_resources::WindowManipulation>();
    window_manipulation.set_mouse_cursor(gui->GetMouseCursor());
}


void GUI_Service::resetProvidedResources() {

}


void GUI_Service::preGraphRender() {

    VALIDATE_GUI_PTR()
    if (this->m_megamol_graph != nullptr) {
        // Requires enabled OpenGL context, e.g. for textures used in parameters
        gui->SynchronizeGraphs(this->m_megamol_graph);
    }
    gui->PreDraw(this->m_framebuffer_size, this->m_window_size, this->m_time);
}


void GUI_Service::postGraphRender() {

    VALIDATE_GUI_PTR()
    gui->PostDraw();
}


std::vector<FrontendResource>& GUI_Service::getProvidedResources() {

    this->m_providedResourceReferences = {
        {"GUIState", this->m_providedStateResource},
        {"GUIRegisterWindow", this->m_providedRegisterWindowResource},
    };
    return this->m_providedResourceReferences;
}


const std::vector<std::string> GUI_Service::getRequestedResourceNames() const {

    return this->m_requestedResourcesNames;
}


void GUI_Service::setRequestedResources(std::vector<FrontendResource> resources) {

    this->m_requestedResourceReferences = resources;

    /// Get resource directories = resource index 10
    // (Required to set only once)
    VALIDATE_GUI_PTR()
    auto& runtime_config =
            this->m_requestedResourceReferences[10].getResource<megamol::frontend_resources::RuntimeConfig>();
    if (!runtime_config.resource_directories.empty()) {
        gui->SetResourceDirectories(runtime_config.resource_directories);
    }
}


std::string GUI_Service::resource_request_gui_state(bool as_lua) {

    VALIDATE_GUI_PTR("")
    return gui->GetState(as_lua);
}


bool GUI_Service::resource_request_gui_visibility() {

    VALIDATE_GUI_PTR(false)
    return gui->GetVisibility();
}


float GUI_Service::resource_request_gui_scale() {

    VALIDATE_GUI_PTR(1.0f)
    return gui->GetScale();
}


void GUI_Service::resource_provide_gui_state(const std::string& json_state) {

    VALIDATE_GUI_PTR()
    gui->SetState(json_state);
}


void GUI_Service::resource_provide_gui_visibility(bool show) {

    VALIDATE_GUI_PTR()
    gui->SetVisibility(show);
}


void GUI_Service::resource_provide_gui_scale(float scale) {

    VALIDATE_GUI_PTR()
    gui->SetScale(scale);
}


void GUI_Service::resource_register_window(const std::string& name, std::function<void(megamol::gui::WindowConfiguration::Basic&)>& func) {

    VALIDATE_GUI_PTR()
    gui->RegisterWindow(name, func);
}


void GUI_Service::resource_register_popup(const std::string& name, bool& open, std::function<void(void)>& func) {

    VALIDATE_GUI_PTR()
    gui->RegisterPopUp(name, open, func);
}


void GUI_Service::resource_register_notification(const std::string& name, bool& open, const std::string& message) {

    VALIDATE_GUI_PTR()
    gui->RegisterNotification(name, open, message);
}

} // namespace frontend
} // namespace megamol
