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
    // init gui
    if (config.imgui_api == GUI_Service::ImGuiAPI::OPEN_GL) {
        if (this->m_gui.ptr == nullptr) {
            this->m_gui.ptr = std::make_shared<megamol::gui::GUI_Wrapper>();
            if (check_gui_not_nullptr) {
                if (this->m_gui.ptr->Get()->CreateContext_GL(config.core_instance)) {
                    this->m_providedResourceReferences = { {"GUIResource", this->m_gui} };
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Created ImGui context.\n"); // [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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
    // check for updates in required resources
    for (auto& res : this->m_requestedResourceReferences) {
         if (res.getIdentifier() == "MegaMolGraph") {
             auto resource = &res.getResource<megamol::core::MegaMolGraph>();
             if (resource != nullptr) {
                 
             }
        }
        else if (res.getIdentifier() == "KeyboardEvents") {
             auto resource = &res.getResource<megamol::input_events::KeyboardEvents>();
             if (resource != nullptr) {
                 
             }
        }
        else if (res.getIdentifier() == "MouseEvents") {
             auto resource = &res.getResource<megamol::input_events::MouseEvents>();
             if (resource != nullptr) {
                 
             }
        }
        else if (res.getIdentifier() == "IOpenGL_Context") {
             auto resource = &res.getResource<megamol::input_events::IOpenGL_Context>();
             if (resource != nullptr) {
                 this->m_resource_state.opengl_context_ptr = resource;
             }
         }
        else if (res.getIdentifier() == "FramebufferEvents") {
            auto resource = &res.getResource<megamol::input_events::FramebufferEvents>();
            if (resource != nullptr) {
                auto framebuffer_state = resource->previous_state;
                m_resource_state.viewport_size.x = framebuffer_state.width;
                m_resource_state.viewport_size.y = framebuffer_state.height;
            }
        }
    }
}


void GUI_Service::resetProvidedResources() {
    // nothing to do here.
}


void GUI_Service::preGraphRender() {
    if (!check_gui_not_nullptr) return;

    if (this->m_resource_state.opengl_context_ptr) {
        this->m_resource_state.opengl_context_ptr->activate(); // makes GL context current

        // pre render gui
        this->m_gui.ptr->Get()->PreDraw(m_resource_state.viewport_size, 0.0);

        this->m_resource_state.opengl_context_ptr->close();
    }
}


void GUI_Service::postGraphRender() {
    if (!check_gui_not_nullptr) return;

    if (this->m_resource_state.opengl_context_ptr) {
        this->m_resource_state.opengl_context_ptr->activate(); // makes GL context current

        // post render gui
        this->m_gui.ptr->Get()->PostDraw();

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


void GUI_Service::setRequestedResources(std::vector<ModuleResource>& resources) {
    this->m_requestedResourceReferences = resources;
}


} // namespace frontend
} // namespace megamol
