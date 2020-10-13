/*
 * GUI_Service.hpp
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_SERVICE_HPP_INCLUDED
#define MEGAMOL_GUI_SERVICE_HPP_INCLUDED
#pragma once

#include "AbstractFrontendService.hpp"

#include "Window_Events.h"
#include "Framebuffer_Events.h"
#include "KeyboardMouse_Events.h"
#include "IOpenGL_Context.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/utility/log/Log.h"

#include <glm/glm.hpp>


 // Forward declaration
namespace megamol {
namespace gui {
class GUI_Wrapper;
}
}

namespace megamol {
namespace frontend {


class GUI_Service final : public AbstractFrontendService {

public:

    struct GUIResource {
        std::shared_ptr<megamol::gui::GUI_Wrapper> ptr;
    };

    enum ImGuiAPI {
        OPEN_GL
    };

    struct Config {
        ImGuiAPI imgui_api = GUI_Service::ImGuiAPI::OPEN_GL;
        megamol::core::CoreInstance* core_instance = nullptr;
    };

	std::string serviceName() const override { return "GUI_Service"; }

    GUI_Service() = default;
    ~GUI_Service() override;
    // TODO: delete copy/move/assign?

    // init API, e.g. init GLFW with OpenGL and open window with certain decorations/hints
    bool init(const Config& config);
    bool init(void* configPtr) override;
    void close() override;
	
    void updateProvidedResources() override;
    void digestChangedRequestedResources() override;
    void resetProvidedResources() override;

    void preGraphRender() override;  // prepare rendering with API, e.g. set OpenGL context, frame-timers, etc
    void postGraphRender() override; // clean up after rendering, e.g. stop and show frame-timers in GLFW window

    // expose the resources and input events this RAPI provides: Keyboard inputs, Mouse inputs, GLFW Window events, Framebuffer resize events
    std::vector<ModuleResource>& getProvidedResources() override;
    const std::vector<std::string> getRequestedResourceNames() const override;
    void setRequestedResources(std::vector<ModuleResource> resources) override;

    // from AbstractFrontendService:
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

private:

    struct ResourceState {
        double time;
        glm::vec2 framebuffer_size;
        glm::vec2 window_size;
        megamol::core::MegaMolGraph* megamol_graph;
        megamol::module_resources::IOpenGL_Context const* opengl_context_ptr;
    };

    std::vector<ModuleResource> m_providedResourceReferences;
    std::vector<ModuleResource> m_requestedResourceReferences;

    GUIResource m_gui;
    ResourceState m_resource_state;
};

} // namespace frontend
} // namespace megamol

#endif // MEGAMOL_GUI_SERVICE_HPP_INCLUDED