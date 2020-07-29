/*
 * GUI_Service.cpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "GUI_Service.hpp"

#include "GUI_Wrapper.h"


namespace megamol {
namespace frontend {


#define check_gui_not_nullptr (static_cast<bool>((this->m_gui.ptr != nullptr) && (this->m_gui.ptr->Get() != nullptr)))


GUI_Service::~GUI_Service() {

    // nothing to do here so far ...
}


bool GUI_Service::init(void* configPtr) {

    if (configPtr == nullptr) return false;
    return init(*static_cast<Config*>(configPtr));
}


bool GUI_Service::init(const Config& config) {

    // init resource state
    this->m_resource_state.viewport_size = glm::vec2(1.0f, 1.0f);
    this->m_resource_state.opengl_context_ptr = nullptr;

    // init gui
    if (config.imgui_api == GUI_Service::ImGuiAPI::OPEN_GL) {
        if (this->m_gui.ptr == nullptr) {
            this->m_gui.ptr = std::make_shared<megamol::gui::GUI_Wrapper>();

            if (check_gui_not_nullptr) {
                if (this->m_gui.ptr->Get()->CreateContext_GL(config.core_instance)) {
                    this->m_providedResourceReferences = { {"GUIResource", this->m_gui} };
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Successfully initialized GUI service.\n");
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
    auto gui = this->m_gui.ptr->Get();

    // check for updates in required resources

    /// MegaMolGraph = resource index 0
    auto megamol_graph = &this->m_requestedResourceReferences[0].getResource<megamol::core::MegaMolGraph>(); 
    /// TODO

    /// KeyboardEvents = resource index 1
    auto keyboard_events = &this->m_requestedResourceReferences[1].getResource<megamol::input_events::KeyboardEvents>();
    for (auto& key_event : keyboard_events->key_events) {
        auto key = std::get<0>(key_event);
        auto action = std::get<1>(key_event);
        auto modifiers = std::get<2>(key_event);
        gui->OnKey(key, action, modifiers);
    }
    for (auto& codepoint : keyboard_events->codepoint_events) {
        gui->OnChar(codepoint);
    }

    /// MouseEvents = resource index 2
    auto mouse_events = &this->m_requestedResourceReferences[2].getResource<megamol::input_events::MouseEvents>();
    for (auto& pos_event  : mouse_events->position_events) {
        auto x_pos = std::get<0>(pos_event);
        auto y_pos = std::get<1>(pos_event);
        gui->OnMouseMove(x_pos, y_pos);
    }
    for (auto& scroll_event  : mouse_events->scroll_events) {
        auto x_scroll = std::get<0>(scroll_event);
        auto y_scroll = std::get<1>(scroll_event);
        gui->OnMouseScroll(x_scroll, y_scroll);
    }
    for (auto& btn_event  : mouse_events->buttons_events) {
        auto button = std::get<0>(btn_event);
        auto action = std::get<1>(btn_event);
        auto modifiers = std::get<2>(btn_event);
        gui->OnMouseButton(button, action, modifiers);
    }

    /// IOpenGL_Context = resource index 3
    this->m_resource_state.opengl_context_ptr = &this->m_requestedResourceReferences[3].getResource<megamol::input_events::IOpenGL_Context>();
         
    /// FramebufferEvents = resource index 4
    auto framebuffer_events = &this->m_requestedResourceReferences[4].getResource<megamol::input_events::FramebufferEvents>();
    for (auto& size_event : framebuffer_events->size_events) {
        m_resource_state.viewport_size.x = size_event.width;
        m_resource_state.viewport_size.y = size_event.height;
    }
}


void GUI_Service::resetProvidedResources() {

    // nothing to do here.
}


void GUI_Service::preGraphRender() {

    if (!check_gui_not_nullptr) return;
    auto gui = this->m_gui.ptr->Get();

    if (this->m_resource_state.opengl_context_ptr) {
        this->m_resource_state.opengl_context_ptr->activate();
        gui->PreDraw(m_resource_state.viewport_size, 0.0);
        this->m_resource_state.opengl_context_ptr->close();
    }
}


void GUI_Service::postGraphRender() {

    if (!check_gui_not_nullptr) return;
    auto gui = this->m_gui.ptr->Get();

    if (this->m_resource_state.opengl_context_ptr) {
        this->m_resource_state.opengl_context_ptr->activate();
        gui->PostDraw();
        this->m_resource_state.opengl_context_ptr->close();
    }
}


std::vector<ModuleResource>& GUI_Service::getProvidedResources() {

    // unused - returning empty list
	return this->m_providedResourceReferences;
}


const std::vector<std::string> GUI_Service::getRequestedResourceNames() const {

	return {
        {"MegaMolGraph"},      // resource index 0
        {"KeyboardEvents"},    // resource index 1
        {"MouseEvents"},       // resource index 2
        {"IOpenGL_Context"},   // resource index 3
        {"FramebufferEvents"}  // resource index 4
    };
}


void GUI_Service::setRequestedResources(std::vector<ModuleResource> resources) {

    this->m_requestedResourceReferences = resources;
}


} // namespace frontend
} // namespace megamol
