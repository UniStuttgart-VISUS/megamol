/*
 * GUI_Service.cpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */


#include "GUI_Service.hpp"

#include "CommandRegistry.h"
#include "FrameStatistics.h"
#include "Framebuffer_Events.h"
#include "GUIManager.h"
#include "ImagePresentationEntryPoints.h"
#include "KeyboardMouse_Events.h"
#include "ModuleGraphSubscription.h"
#include "OpenGL_Context.h"
#include "PluginsResource.h"
#include "ProjectLoader.h"
#include "RuntimeConfig.h"
#include "ScriptPaths.h"
#include "WindowManipulation.h"
#include "Window_Events.h"

#include "mmcore/utility/log/Log.h"


namespace megamol::frontend {


bool GUI_Service::init(void* configPtr) {

    if (configPtr == nullptr)
        return false;
    return init(*static_cast<Config*>(configPtr));
}


bool GUI_Service::init(const Config& config) {

    this->m_time = 0.0;
    this->m_framebuffer_size = glm::vec2(1.0f, 1.0f);
    this->m_window_size = glm::vec2(1.0f, 1.0f);
    this->m_megamol_graph = nullptr;
    this->m_config = config;

    this->m_queuedProjectFiles.clear();
    this->m_providedResourceReferences.clear();
    this->m_requestedResourcesNames = {
        "MegaMolGraph",                                                 // 0 - sync graphs
        "optional<WindowEvents>",                                       // 1 - time, size, clipboard
        "optional<KeyboardEvents>",                                     // 2 - key press
        "optional<MouseEvents>",                                        // 3 - mouse click
        "optional<OpenGL_Context>",                                     // 4 - graphics api for imgui context
        "FramebufferEvents",                                            // 5 - viewport size
        "GLFrontbufferToPNG_ScreenshotTrigger",                         // 6 - trigger screenshot
        "LuaScriptPaths",                                               // 7 - current project path
        "ProjectLoader",                                                // 8 - trigger loading of new running project
        "FrameStatistics",                                              // 9 - current fps and ms value
        "RuntimeConfig",                                                // 10 - resource paths
        "optional<WindowManipulation>",                                 // 11 - GLFW window pointer
        frontend_resources::CommandRegistry_Req_Name,                   // 12 - Command registry
        "ImagePresentationEntryPoints",                                 // 13 - Entry point
        "ExecuteLuaScript",                                             // 14 - Execute Lua Scripts (from Console)
        frontend_resources::MegaMolGraph_SubscriptionRegistry_Req_Name, // 15 MegaMol Graph subscription
        "PluginsResource",                                              // 16 - Plugins
#ifdef MEGAMOL_USE_PROFILING
        frontend_resources::PerformanceManager_Req_Name // 17 - Performance Manager
#endif
    };

    this->m_gui = std::make_shared<megamol::gui::GUIManager>();
    auto gui_resources = m_gui->requested_lifetime_resources();
    m_requestedResourcesNames.insert(m_requestedResourcesNames.end(), gui_resources.begin(), gui_resources.end());
    m_requestedResourcesNames.erase(std::unique(m_requestedResourcesNames.begin(), m_requestedResourcesNames.end()),
        m_requestedResourcesNames.end());

    // Set function pointer in state resource once
    this->m_providedStateResource.request_gui_state = [&](bool as_lua) -> std::string {
        return this->resource_request_gui_state(as_lua);
    };
    this->m_providedStateResource.provide_gui_state = [&](const std::string& json_state) -> void {
        return this->resource_provide_gui_state(json_state);
    };
    this->m_providedStateResource.request_gui_visibility = [&]() -> bool {
        return this->resource_request_gui_visibility();
    };
    this->m_providedStateResource.provide_gui_visibility = [&](bool show) -> void {
        return this->resource_provide_gui_visibility(show);
    };
    this->m_providedStateResource.request_gui_scale = [&]() -> float { return this->resource_request_gui_scale(); };
    this->m_providedStateResource.provide_gui_scale = [&](float scale) -> void {
        return this->resource_provide_gui_scale(scale);
    };
    this->m_providedRegisterWindowResource.register_window =
        [&](const std::string& name, std::function<void(megamol::gui::AbstractWindow::BasicConfig&)> func) -> void {
        this->resource_register_window(name, func);
    };
    this->m_providedRegisterWindowResource.register_popup = [&](const std::string& name, std::weak_ptr<bool> open,
                                                                std::function<void(void)> func) -> void {
        this->resource_register_popup(name, open, func);
    };
    this->m_providedRegisterWindowResource.register_notification =
        [&](const std::string& name, std::weak_ptr<bool> open, const std::string& message) -> void {
        this->resource_register_notification(name, open, message);
    };

    // NB: Config values are applied before project file values and therefore overwritten by project settings
    this->resource_provide_gui_visibility(m_config.gui_show);
    this->resource_provide_gui_scale(m_config.gui_scale);

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("GUI_Service: initialized successfully");

    return true;
}


void GUI_Service::close() {}


void GUI_Service::updateProvidedResources() {}


void GUI_Service::digestChangedRequestedResources() {

    if (this->m_gui == nullptr) {
        return;
    }

    // Trigger shutdown
    this->setShutdown(this->m_gui->GetTriggeredShutdown());

    // Check for updates in requested resources --------------------------------

    /// KeyboardEvents = resource index 2
    auto maybe_keyboard_events = frontend_resources->getOptional<megamol::frontend_resources::KeyboardEvents>();
    if (maybe_keyboard_events.has_value()) {
        megamol::frontend_resources::KeyboardEvents const& keyboard_events = maybe_keyboard_events.value().get();

        std::vector<std::tuple<megamol::frontend_resources::Key, megamol::frontend_resources::KeyAction,
            megamol::frontend_resources::Modifiers>>
            pass_key_events;
        for (auto& key_event : keyboard_events.key_events) {
            auto key = std::get<0>(key_event);
            auto action = std::get<1>(key_event);
            auto modifiers = std::get<2>(key_event);
            if (!this->m_gui->OnKey(key, action, modifiers)) {
                pass_key_events.emplace_back(key_event);
            }
        }
        /// WARNING: Changing a constant type will lead to an undefined behavior!
        const_cast<megamol::frontend_resources::KeyboardEvents&>(keyboard_events).key_events = pass_key_events;

        std::vector<unsigned int> pass_codepoint_events;
        for (auto& codepoint_event : keyboard_events.codepoint_events) {
            if (!this->m_gui->OnChar(codepoint_event)) {
                pass_codepoint_events.emplace_back(codepoint_event);
            }
        }
        /// WARNING: Changing a constant type will lead to an undefined behavior!
        const_cast<megamol::frontend_resources::KeyboardEvents&>(keyboard_events).codepoint_events =
            pass_codepoint_events;
    }

    /// MouseEvents = resource index 3
    auto maybe_mouse_events = frontend_resources->getOptional<megamol::frontend_resources::MouseEvents>();
    if (maybe_mouse_events.has_value()) {
        megamol::frontend_resources::MouseEvents const& mouse_events = maybe_mouse_events.value().get();

        std::vector<std::tuple<double, double>> pass_mouse_pos_events;
        for (auto& position_event : mouse_events.position_events) {
            auto x_pos = std::get<0>(position_event);
            auto y_pos = std::get<1>(position_event);
            if (!this->m_gui->OnMouseMove(x_pos, y_pos)) {
                pass_mouse_pos_events.emplace_back(position_event);
            }
        }
        /// WARNING: Changing a constant type will lead to an undefined behavior!
        const_cast<megamol::frontend_resources::MouseEvents&>(mouse_events).position_events = pass_mouse_pos_events;

        std::vector<std::tuple<double, double>> pass_mouse_scroll_events;
        for (auto& scroll_event : mouse_events.scroll_events) {
            auto x_scroll = std::get<0>(scroll_event);
            auto y_scroll = std::get<1>(scroll_event);
            if (!this->m_gui->OnMouseScroll(x_scroll, y_scroll)) {
                pass_mouse_scroll_events.emplace_back(scroll_event);
            }
        }
        /// WARNING: Changing a constant type will lead to an undefined behavior!
        const_cast<megamol::frontend_resources::MouseEvents&>(mouse_events).scroll_events = pass_mouse_scroll_events;

        std::vector<std::tuple<megamol::frontend_resources::MouseButton, megamol::frontend_resources::MouseButtonAction,
            megamol::frontend_resources::Modifiers>>
            pass_mouse_btn_events;
        for (auto& button_event : mouse_events.buttons_events) {
            auto button = std::get<0>(button_event);
            auto action = std::get<1>(button_event);
            auto modifiers = std::get<2>(button_event);
            if (!this->m_gui->OnMouseButton(button, action, modifiers)) {
                pass_mouse_btn_events.emplace_back(button_event);
            }
        }
        /// WARNING: Changing a constant type will lead to an undefined behavior!
        const_cast<megamol::frontend_resources::MouseEvents&>(mouse_events).buttons_events = pass_mouse_btn_events;
    }

    /// FramebufferEvents = resource index 5
    auto framebuffer_events = &frontend_resources->get<megamol::frontend_resources::FramebufferEvents>();
    for (auto& size_event : framebuffer_events->size_events) {
        this->m_framebuffer_size.x = static_cast<float>(size_event.width);
        this->m_framebuffer_size.y = static_cast<float>(size_event.height);
    }

    /// WindowEvents = resource index 1
    auto maybe_window_events = frontend_resources->getOptional<megamol::frontend_resources::WindowEvents>();
    if (maybe_window_events.has_value()) {
        megamol::frontend_resources::WindowEvents const& window_events = maybe_window_events.value().get();

        this->m_time = window_events.time;
        for (auto& size_event : window_events.size_events) {
            this->m_window_size.x = static_cast<float>(std::get<0>(size_event));
            this->m_window_size.y = static_cast<float>(std::get<1>(size_event));
        }
        this->m_gui->SetClipboardFunc(window_events._getClipboardString_Func, window_events._setClipboardString_Func,
            window_events._clipboard_user_data);
    } else {
        // no GL
        this->m_window_size = m_framebuffer_size;
    }

    /// Trigger Screenshot = resource index 6
    if (this->m_gui->GetTriggeredScreenshot()) {
        auto& screenshot_to_file_trigger = frontend_resources->get<std::function<bool(std::filesystem::path const&)>>();
        screenshot_to_file_trigger(this->m_gui->GetScreenshotFileName());
    }

    /// Pipe lua script paths to gui = resource index 7
    auto& script_paths = frontend_resources->get<megamol::frontend_resources::ScriptPaths>();
    this->m_gui->SetProjectScriptPaths(script_paths.lua_script_paths);

    /// Pipe project loading request from GUI to project loader = resource index 8
    auto requested_project_file = this->m_gui->GetProjectLoadRequest();
    if (!requested_project_file.empty()) {
        auto& project_loader = frontend_resources->get<megamol::frontend_resources::ProjectLoader>();
        project_loader.load_filename(requested_project_file);
    }

    /// Get current FPS and MS frame statistic = resource index 9
    auto& frame_statistics = frontend_resources->get<megamol::frontend_resources::FrameStatistics>();
    this->m_gui->SetFrameStatistics(frame_statistics.last_averaged_fps, frame_statistics.last_averaged_mspf,
        frame_statistics.rendered_frames_count);

    /// Get window manipulation resource = resource index 11
    auto maybe_window_manipulation = frontend_resources->getOptional<megamol::frontend_resources::WindowManipulation>();
    if (maybe_window_manipulation.has_value()) {
        megamol::frontend_resources::WindowManipulation const& window_manipulation =
            maybe_window_manipulation.value().get();

        const_cast<megamol::frontend_resources::WindowManipulation&>(window_manipulation)
            .set_mouse_cursor(this->m_gui->GetMouseCursor());
    }
}


void GUI_Service::resetProvidedResources() {}


void GUI_Service::preGraphRender() {

    if (this->m_gui != nullptr) {
        // Propagate changes from the GUI graph to the MegaMol graph
        if ((this->m_megamol_graph != nullptr)) {
            // Requires enabled OpenGL context, e.g. for textures used in parameters
            this->m_gui->SynchronizeGraphs((*this->m_megamol_graph));
        }
        this->m_gui->PreDraw(this->m_framebuffer_size, this->m_window_size, this->m_time);
    }
}


void GUI_Service::postGraphRender() {

    if (this->m_gui != nullptr) {
        this->m_gui->PostDraw();
    }
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

    /// (Called only once)
    frontend_resources = std::make_shared<frontend_resources::FrontendResourcesMap>(resources);

    if (this->m_gui == nullptr) {
        return;
    }

    auto const& pluginsRes = frontend_resources->get<megamol::frontend_resources::PluginsResource>();

    this->m_gui->InitializeGraphSynchronisation(pluginsRes);

    // Check render backend prerequisites
    auto maybe_opengl_context = frontend_resources->getOptional<frontend_resources::OpenGL_Context>();
    if (!maybe_opengl_context.has_value() && (m_config.backend == megamol::gui::GUIRenderBackend::OPEN_GL)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "GUI_Service: no OpenGL_Context available ... switching to CPU backend.");
        m_config.backend = megamol::gui::GUIRenderBackend::CPU;
    }
    // Create GUI context
    if (!this->m_gui->CreateContext(m_config.backend)) {
        this->setShutdown();
    }

    /// MegaMolGraph = resource index 0
    auto graph_resource_ptr = &frontend_resources->get<megamol::core::MegaMolGraph>();
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    this->m_megamol_graph = const_cast<megamol::core::MegaMolGraph*>(graph_resource_ptr);
    assert(this->m_megamol_graph != nullptr);

    /// Resource Directories = resource index 10
    auto& runtime_config = frontend_resources->get<megamol::frontend_resources::RuntimeConfig>();
    if (!runtime_config.resource_directories.empty()) {
        this->m_gui->SetResourceDirectories(runtime_config.resource_directories);
    }

    /// Command Registry = resource index 12
    auto& command_registry = const_cast<megamol::frontend_resources::CommandRegistry&>(
        frontend_resources->get<megamol::frontend_resources::CommandRegistry>());
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    this->m_gui->RegisterHotkeys(command_registry, *this->m_megamol_graph);

    /// Image Presentation = resource index 13
    // the image presentation will issue the rendering and provide the view with resources for rendering
    // probably we dont care or dont check wheter the same view is added as entry point multiple times
    auto& image_presentation = const_cast<megamol::frontend_resources::ImagePresentationEntryPoints&>(
        frontend_resources->get<megamol::frontend_resources::ImagePresentationEntryPoints>());
    const std::string gui_entry_point_name = "GUI_Service";
    bool view_presentation_ok = image_presentation.add_entry_point(
        gui_entry_point_name, {static_cast<void*>(this->m_gui.get()), std::function{gui_rendering_execution},
                                  std::function{get_gui_runtime_resources_requests}});
    view_presentation_ok &= image_presentation.set_entry_point_priority(
        gui_entry_point_name, 100); // render after views (default priority is 0)
    if (!view_presentation_ok) {
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "GUI_Service: error adding graph entry point ... image presentation service rejected GUI Service.");
    }

    m_exec_lua = const_cast<megamol::frontend_resources::common_types::lua_func_type*>(
        &frontend_resources->get<frontend_resources::common_types::lua_func_type>());
    m_gui->SetLuaFunc(m_exec_lua);

    // MegaMol Graph Subscription
    auto& megamolgraph_subscription = const_cast<frontend_resources::MegaMolGraph_SubscriptionRegistry&>(
        frontend_resources->get<frontend_resources::MegaMolGraph_SubscriptionRegistry>());

    frontend_resources::ModuleGraphSubscription gui_subscription("GUI");

    gui_subscription.AddModule = [&](core::ModuleInstance_t const& module_inst) {
        return m_gui->NotifyRunningGraph_AddModule(module_inst);
    };
    gui_subscription.DeleteModule = [&](core::ModuleInstance_t const& module_inst) {
        return m_gui->NotifyRunningGraph_DeleteModule(module_inst);
    };
    gui_subscription.RenameModule = [&](std::string const& old_name, std::string const& new_name,
                                        core::ModuleInstance_t const& module_inst) {
        return m_gui->NotifyRunningGraph_RenameModule(old_name, new_name, module_inst);
    };
    gui_subscription.AddParameters =
        [&](std::vector<megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr> const& param_slots) {
            return m_gui->NotifyRunningGraph_AddParameters(param_slots);
        };
    gui_subscription.RemoveParameters =
        [&](std::vector<megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr> const& param_slots) {
            return m_gui->NotifyRunningGraph_RemoveParameters(param_slots);
        };
    gui_subscription.ParameterChanged =
        [&](megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr const& param_slot,
            std::string const& new_value) { return m_gui->NotifyRunningGraph_ParameterChanged(param_slot, new_value); };

    gui_subscription.AddCall = [&](core::CallInstance_t const& call_inst) {
        return m_gui->NotifyRunningGraph_AddCall(call_inst);
    };
    gui_subscription.DeleteCall = [&](core::CallInstance_t const& call_inst) {
        return m_gui->NotifyRunningGraph_DeleteCall(call_inst);
    };
    gui_subscription.EnableEntryPoint = [&](core::ModuleInstance_t const& module_inst) {
        return m_gui->NotifyRunningGraph_EnableEntryPoint(module_inst);
    };
    gui_subscription.DisableEntryPoint = [&](core::ModuleInstance_t const& module_inst) {
        return m_gui->NotifyRunningGraph_DisableEntryPoint(module_inst);
    };

    megamolgraph_subscription.subscribe(gui_subscription);

#ifdef MEGAMOL_USE_PROFILING
    // PerformanceManager
    perf_manager = const_cast<frontend_resources::PerformanceManager*>(
        &frontend_resources->get<frontend_resources::PerformanceManager>());
    perf_logging = const_cast<frontend_resources::ProfilingLoggingStatus*>(
        &frontend_resources->get<frontend_resources::ProfilingLoggingStatus>());
    m_gui->SetProfilingLoggingStatus(perf_logging);
    // this needs to happen before the first (gui) module is spawned to help it look up the timers
    m_gui->SetPerformanceManager(perf_manager);
    perf_manager->subscribe_to_updates(
        [&](const frontend_resources::PerformanceManager::frame_info& fi) { m_gui->AppendPerformanceData(fi); });
#endif

    // now come the resources for the gui windows
    m_gui->setRequestedResources(frontend_resources);
}


std::vector<std::string> GUI_Service::get_gui_runtime_resources_requests() {

    /// Already provided via getRequestedResourceNames()
    return {};
}


bool GUI_Service::gui_rendering_execution(void* void_ptr,
    std::vector<megamol::frontend::FrontendResource> const& resources,
    megamol::frontend_resources::ImageWrapper& result_image) {

    auto gui_ptr = static_cast<megamol::gui::GUIManager*>(void_ptr);
    if (gui_ptr == nullptr) {
        return false;
    }
    result_image = gui_ptr->GetImage();

    return true;
}


std::string GUI_Service::resource_request_gui_state(bool as_lua) {

    if (this->m_gui == nullptr) {
        return std::string();
    }
    return this->m_gui->GetState(as_lua);
}


bool GUI_Service::resource_request_gui_visibility() {

    if (this->m_gui == nullptr) {
        return false;
    }
    return this->m_gui->GetVisibility();
}


float GUI_Service::resource_request_gui_scale() {

    if (this->m_gui == nullptr) {
        return float();
    }
    return this->m_gui->GetScale();
}


void GUI_Service::resource_provide_gui_state(const std::string& json_state) {

    if (this->m_gui == nullptr) {
        return;
    }
    this->m_gui->SetState(json_state);
}


void GUI_Service::resource_provide_gui_visibility(bool show) {

    if (this->m_gui == nullptr) {
        return;
    }
    this->m_gui->SetVisibility(show);
}


void GUI_Service::resource_provide_gui_scale(float scale) {

    if (this->m_gui != nullptr) {
        this->m_gui->SetScale(scale);
    }
}


void GUI_Service::resource_register_window(
    const std::string& name, std::function<void(megamol::gui::AbstractWindow::BasicConfig&)>& func) {

    if (this->m_gui != nullptr) {
        this->m_gui->RegisterWindow(name, func);
    }
}


void GUI_Service::resource_register_popup(
    const std::string& name, std::weak_ptr<bool> open, std::function<void(void)>& func) {

    if (this->m_gui != nullptr) {
        this->m_gui->RegisterPopUp(name, open, func);
    }
}


void GUI_Service::resource_register_notification(
    const std::string& name, std::weak_ptr<bool> open, const std::string& message) {

    if (this->m_gui != nullptr) {
        this->m_gui->RegisterNotification(name, open, message);
    }
}

} // namespace megamol::frontend
