/*
 * GUI_Service.cpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "GUI_Service.hpp"


namespace megamol {
namespace frontend {


#define check_gui_not_nullptr (static_cast<bool>((this->m_gui != nullptr) && (this->m_gui->Get() != nullptr)))


GUI_Service::~GUI_Service() {

    // nothing to do here so far ...
}


bool GUI_Service::init(void* configPtr) {

    if (configPtr == nullptr) return false;
    return init(*static_cast<Config*>(configPtr));
}


bool GUI_Service::init(const Config& config) {

    // init resource state
    this->m_resource_state.time = 0.0;
    this->m_resource_state.framebuffer_size = glm::vec2(1.0f, 1.0f);
    this->m_resource_state.window_size = glm::vec2(1.0f, 1.0f);
    this->m_resource_state.opengl_context_ptr = nullptr;

    this->m_requestedResourcesNames = {
        {"MegaMolGraph"},                          // resource index 0
        {"WindowEvents"},                          // resource index 1
        {"KeyboardEvents"},                        // resource index 2
        {"MouseEvents"},                           // resource index 3
        {"IOpenGL_Context"},                       // resource index 4
        {"FramebufferEvents"},                     // resource index 5
        {"GLFrontbufferToPNG_ScreenshotTrigger"},  // resource index 6
        {"LuaScriptPaths"}                         // resource index 7
    };

    // init gui
    if (config.imgui_api == GUI_Service::ImGuiAPI::OPEN_GL) { 
        if (this->m_gui == nullptr) {
            this->m_gui = std::make_shared<megamol::gui::GUIWrapper>();

            if (check_gui_not_nullptr) {
                if (this->m_gui->Get()->CreateContext(megamol::gui::GUIImGuiAPI::OPEN_GL, config.core_instance)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo("GUI_Service: initialized successfully.");
                    return true;
                }
            }
        }
    }

    return false;
}


void GUI_Service::close() {

    // nothing to do here so far ...
}
    

void GUI_Service::updateProvidedResources() {

    // nothing to do here.
}


void GUI_Service::digestChangedRequestedResources() {

    if (!check_gui_not_nullptr) return;
    auto gui = this->m_gui->Get();

    // Trigger shutdown
    this->setShutdown(gui->ConsumeTriggeredShutdown());

    // Check for updates in requested resources --------------------------------

    /// MegaMolGraph = resource index 0
    auto graph_resource_ptr = &this->m_requestedResourceReferences[0].getResource<megamol::core::MegaMolGraph>(); 

    // Synchronise changes between core graph and gui graph
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    this->m_resource_state.megamol_graph = const_cast<megamol::core::MegaMolGraph*>(graph_resource_ptr);

    /// WindowEvents = resource index 1
    auto window_events = &this->m_requestedResourceReferences[1].getResource<megamol::frontend_resources::WindowEvents>();
    this->m_resource_state.time = window_events->time;
    for (auto& size_event : window_events->size_events) {
        m_resource_state.window_size.x = static_cast<float>(std::get<0>(size_event));
        m_resource_state.window_size.y = static_cast<float>(std::get<1>(size_event));
    }
    gui->SetClipboardFunc(window_events->_getClipboardString_Func, window_events->_setClipboardString_Func, window_events->_clipboard_user_data);

    /// KeyboardEvents = resource index 2
    auto keyboard_events = &this->m_requestedResourceReferences[2].getResource<megamol::frontend_resources::KeyboardEvents>();

    std::vector<std::tuple<megamol::frontend_resources::Key, megamol::frontend_resources::KeyAction, megamol::frontend_resources::Modifiers>> pass_key_events;
    for (auto it = keyboard_events->key_events.begin(); it != keyboard_events->key_events.end(); it++) {
        auto key = std::get<0>((*it));
        auto action = std::get<1>((*it));
        auto modifiers = std::get<2>((*it));
        if (!gui->OnKey(key, action, modifiers)) {
            pass_key_events.emplace_back((*it));
        }
    }
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    const_cast<megamol::frontend_resources::KeyboardEvents*>(keyboard_events)->key_events = pass_key_events;

    std::vector<unsigned int> pass_codepoint_events;
    for (auto it = keyboard_events->codepoint_events.begin(); it != keyboard_events->codepoint_events.end(); it++) {
        if (!gui->OnChar((*it))) {
            pass_codepoint_events.emplace_back((*it));
        }
    }
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    const_cast<megamol::frontend_resources::KeyboardEvents*>(keyboard_events)->codepoint_events = pass_codepoint_events;

    /// MouseEvents = resource index 3
    auto mouse_events = &this->m_requestedResourceReferences[3].getResource<megamol::frontend_resources::MouseEvents>();

    std::vector<std::tuple<double, double>>  pass_mouse_pos_events;
    for (auto it = mouse_events->position_events.begin(); it != mouse_events->position_events.end(); it++) {
        auto x_pos = std::get<0>((*it));
        auto y_pos = std::get<1>((*it));
        if (!gui->OnMouseMove(x_pos, y_pos)) {
            pass_mouse_pos_events.emplace_back((*it));
        }
    }
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    const_cast<megamol::frontend_resources::MouseEvents*>(mouse_events)->position_events = pass_mouse_pos_events;

    std::vector<std::tuple<double, double>> pass_mouse_scroll_events;
    for (auto it = mouse_events->scroll_events.begin(); it != mouse_events->scroll_events.end(); it++) {
        auto x_scroll = std::get<0>((*it));
        auto y_scroll = std::get<1>((*it));
        if (!gui->OnMouseScroll(x_scroll, y_scroll)) {
            pass_mouse_scroll_events.emplace_back((*it));
        }
    }
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    const_cast<megamol::frontend_resources::MouseEvents*>(mouse_events)->scroll_events = pass_mouse_scroll_events;

    std::vector<std::tuple<megamol::frontend_resources::MouseButton, megamol::frontend_resources::MouseButtonAction, megamol::frontend_resources::Modifiers>>  pass_mouse_btn_events;
    for (auto it = mouse_events->buttons_events.begin(); it != mouse_events->buttons_events.end(); it++) {
        auto button = std::get<0>((*it));
        auto action = std::get<1>((*it));
        auto modifiers = std::get<2>((*it));
        if (!gui->OnMouseButton(button, action, modifiers)) {
            pass_mouse_btn_events.emplace_back((*it));
        }
    }
    /// WARNING: Changing a constant type will lead to an undefined behavior!
    const_cast<megamol::frontend_resources::MouseEvents*>(mouse_events)->buttons_events = pass_mouse_btn_events;

    /// IOpenGL_Context = resource index 4
    this->m_resource_state.opengl_context_ptr = &this->m_requestedResourceReferences[4].getResource<megamol::frontend_resources::IOpenGL_Context>();
         
    /// FramebufferEvents = resource index 5
    auto framebuffer_events = &this->m_requestedResourceReferences[5].getResource<megamol::frontend_resources::FramebufferEvents>();
    for (auto& size_event : framebuffer_events->size_events) {
        m_resource_state.framebuffer_size.x = static_cast<float>(size_event.width);
        m_resource_state.framebuffer_size.y = static_cast<float>(size_event.height);
    }

    /// Trigger Screenshot = resource index 6
    if (gui->ConsumeTriggeredScreenshot()) {
        auto& screenshot_to_file_trigger = this->m_requestedResourceReferences[6].getResource< std::function<bool(std::string const&)> >();
        screenshot_to_file_trigger(gui->ConsumeScreenshotFileName());
    }

    /// Pipe lua script paths to gui = resource index 7
   auto& script_paths = this->m_requestedResourceReferences[7].getResource< megamol::frontend_resources::ScriptPaths>();
   gui->SetProjectScriptPaths(script_paths.lua_script_paths);
}


void GUI_Service::resetProvidedResources() {

    // nothing to do here.
}


void GUI_Service::preGraphRender() {

    if (!check_gui_not_nullptr) return;
    auto gui = this->m_gui->Get();

    if (this->m_resource_state.opengl_context_ptr) {
        this->m_resource_state.opengl_context_ptr->activate();

        if (this->m_resource_state.megamol_graph != nullptr) {
            // Requires enabled OpenGL context, e.g. for textures used in parameters
            gui->SynchronizeGraphs(this->m_resource_state.megamol_graph);
        }

        gui->PreDraw(this->m_resource_state.framebuffer_size, this->m_resource_state.window_size, this->m_resource_state.time);
        this->m_resource_state.opengl_context_ptr->close();
    }
}


void GUI_Service::postGraphRender() {

    if (!check_gui_not_nullptr) return;
    auto gui = this->m_gui->Get();

    if (this->m_resource_state.opengl_context_ptr) {
        this->m_resource_state.opengl_context_ptr->activate();
        gui->PostDraw();
        this->m_resource_state.opengl_context_ptr->close();
    }
}


std::vector<FrontendResource>& GUI_Service::getProvidedResources() {

    return this->m_providedResourceReferences;
}


const std::vector<std::string> GUI_Service::getRequestedResourceNames() const {

    return this->m_requestedResourcesNames;
}


void GUI_Service::setRequestedResources(std::vector<FrontendResource> resources) {

    this->m_requestedResourceReferences = resources;
}


} // namespace frontend
} // namespace megamol
