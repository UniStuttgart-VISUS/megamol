/*
 * GUI_Service.cpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "GUI_Service.hpp"

#include "GUI_Wrapper.h"
#include "Framebuffer_Events.h"


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

    if (config.imgui_api == GUI_Service::ImGuiAPI::OPEN_GL) {
        if (this->m_gui.ptr == nullptr) {
            this->m_gui.ptr = std::make_shared<megamol::gui::GUI_Wrapper>();
            if (check_gui_not_nullptr) {
                if (this->m_gui.ptr->Get()->CreateContext_GL(config.core_instance)) {
                    this->m_providedResourceReferences = {
                    {"GUIResource", this->m_gui}
                    };
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
 
    // nothing to do here so far ...
}


void GUI_Service::digestChangedRequestedResources() {
 
    // nothing to do here so far ...
}


void GUI_Service::resetProvidedResources() {

    // nothing to do here so far ...
}


void GUI_Service::preGraphRender() {

    if (check_gui_not_nullptr) {
        glm::vec2 viewport_size(1, 1);
        for (auto& res : this->m_requestedResourceReferences) {
            if (res.getIdentifier() == "FramebufferEvents") {
                auto framebuffer_resource = res.getResource<megamol::input_events::FramebufferEvents>();
                auto framebuffer_state = framebuffer_resource.previous_state;
                viewport_size.x = framebuffer_state.width;
                viewport_size.y = framebuffer_state.height;
            }

        }
        this->m_gui.ptr->Get()->PreDraw(viewport_size, 0.0);
    }
}


void GUI_Service::postGraphRender() {

    if (check_gui_not_nullptr) {
        this->m_gui.ptr->Get()->PostDraw();
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
        {"FramebufferEvents"}   
    };
}


void GUI_Service::setRequestedResources(std::vector<ModuleResource>& resources) {
    this->m_requestedResourceReferences = resources;
}


} // namespace frontend
} // namespace megamol
