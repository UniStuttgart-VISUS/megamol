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
    for (auto& res : this->m_requestedResourceReferences) {
         if (res.getIdentifier() == "MegaMolGraph") {
             auto resource = &res.getResource<megamol::core::MegaMolGraph>();
             if (resource != nullptr) {
                 ///megamol::core::utility::log::Log::DefaultLog.WriteWarn("MegaMolGraph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                 
                 /// TODO
             }
        }
        else if (res.getIdentifier() == "KeyboardEvents") {
             auto resource = &res.getResource<megamol::module_resources::KeyboardEvents>();
             if (resource != nullptr) {
                 if (resource->key_events.size() > 0) {
                     auto key_event = resource->key_events.back();
                     auto key = std::get<0>(key_event);
                     auto action = std::get<1>(key_event);
                     auto modifiers = std::get<2>(key_event);
                     gui->OnKey(key, action, modifiers);
                 }
                 if (resource->codepoint_events.size() > 0) {
                     gui->OnChar(resource->codepoint_events.back());
                 }
             }
        }
        else if (res.getIdentifier() == "MouseEvents") {
             auto resource = &res.getResource<megamol::module_resources::MouseEvents>();
             if (resource != nullptr) {
                if (resource->position_events.size() > 0) {
                    auto pos_event = resource->position_events.back();
                    auto x_pos = std::get<0>(pos_event);
                    auto y_pos = std::get<1>(pos_event);
                    gui->OnMouseMove(x_pos, y_pos);
                }
                if (resource->scroll_events.size() > 0) {
                    auto scroll_event = resource->scroll_events.back();
                    auto x_scroll = std::get<0>(scroll_event);
                    auto y_scroll = std::get<1>(scroll_event);
                    gui->OnMouseScroll(x_scroll, y_scroll);
                }
                if (resource->buttons_events.size() > 0) {
                    auto btn_event = resource->buttons_events.back();
                    auto button = std::get<0>(btn_event);
                    auto action = std::get<1>(btn_event);
                    auto modifiers = std::get<2>(btn_event);
                    gui->OnMouseButton(button, action, modifiers);
                }
             }
        }
        else if (res.getIdentifier() == "IOpenGL_Context") {
             auto resource = &res.getResource<megamol::module_resources::IOpenGL_Context>();
             if (resource != nullptr) {
                 this->m_resource_state.opengl_context_ptr = resource;
             }
         }
        else if (res.getIdentifier() == "FramebufferEvents") {
            auto resource = &res.getResource<megamol::module_resources::FramebufferEvents>();
            if (resource != nullptr) {
                if (resource->size_events.size() > 0) {
                    auto size = resource->size_events.back();
                    m_resource_state.viewport_size.x = size.width;
                    m_resource_state.viewport_size.y = size.height;
                }
            }
        }
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
        {"MegaMolGraph"},
        {"KeyboardEvents"},
        {"MouseEvents"},
        {"IOpenGL_Context"},
        {"FramebufferEvents"}   
    };
}


void GUI_Service::setRequestedResources(std::vector<ModuleResource> resources) {

    this->m_requestedResourceReferences = resources;
}


} // namespace frontend
} // namespace megamol
