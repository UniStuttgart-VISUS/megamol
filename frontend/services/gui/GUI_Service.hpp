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

#include "gui-wrapper.h"

#include "IOpenGL_Context.h"
#include "GUI_Resource.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"

#include <glm/glm.hpp>

namespace megamol {
namespace frontend {


class GUI_Service final : public AbstractFrontendService {

public:

    enum ImGuiAPI {
        OPEN_GL
    };

    struct Config {
        ImGuiAPI imgui_api = GUI_Service::ImGuiAPI::OPEN_GL;
        megamol::core::CoreInstance* core_instance = nullptr;
        bool gui_show = true;
        float gui_scale = 1.0f;
        bool show_fbos_test = false;
        bool show_headnode_remote_control = false;
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
    std::vector<FrontendResource>& getProvidedResources() override;
    const std::vector<std::string> getRequestedResourceNames() const override;
    void setRequestedResources(std::vector<FrontendResource> resources) override;

    /// from AbstractFrontendService:
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

private:

    double m_time;
    glm::vec2 m_framebuffer_size;
    glm::vec2 m_window_size;
    megamol::core::MegaMolGraph* m_megamol_graph;
    std::shared_ptr<megamol::gui::GUIWrapper> m_gui = nullptr;
    std::vector<std::string> m_queuedProjectFiles;

    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<FrontendResource> m_requestedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    megamol::frontend_resources::GUIResource m_providedResource;

    std::string resource_request_gui_state(void);
    void resource_provide_gui_state(const std::string& json_state);
    void resource_provide_gui_visibility(bool show);
    void resource_provide_gui_scale(float scale);

    Config m_config_frontend_fbos_test;
};

} // namespace frontend
} // namespace megamol

#endif // MEGAMOL_GUI_SERVICE_HPP_INCLUDED
