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
#include "CommonTypes.h"
#include "GUIRegisterWindow.h"
#include "GUIState.h"
#include "PerformanceManager.h"
#include "gui_render_backend.h"
#include "mmcore/MegaMolGraph.h"


namespace megamol {
namespace frontend {


class GUIManager;


class GUI_Service final : public AbstractFrontendService {

public:
    struct Config {
        megamol::gui::GUIRenderBackend backend = megamol::gui::GUIRenderBackend::CPU;
        bool gui_show = true;
        float gui_scale = 1.0f;
    };

    std::string serviceName() const override {
        return "GUI_Service";
    }

    GUI_Service() = default;
    ~GUI_Service() override = default;
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

    // expose the resources and input events this RAPI provides: Keyboard inputs, Mouse inputs, GLFW Window events,
    // Framebuffer resize events
    std::vector<FrontendResource>& getProvidedResources() override;
    const std::vector<std::string> getRequestedResourceNames() const override;
    void setRequestedResources(std::vector<FrontendResource> resources) override;

    /// from AbstractFrontendService:
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

    static std::vector<std::string> get_gui_runtime_resources_requests();

    static bool gui_rendering_execution(void* void_ptr,
        std::vector<megamol::frontend::FrontendResource> const& resources,
        megamol::frontend_resources::ImageWrapper& result_image);

private:
    double m_time;
    glm::vec2 m_framebuffer_size;
    glm::vec2 m_window_size;
    Config m_config;
    megamol::core::MegaMolGraph* m_megamol_graph;
    megamol::frontend_resources::PerformanceManager* perf_manager = nullptr;
    std::shared_ptr<megamol::gui::GUIManager> m_gui = nullptr;
    std::vector<std::string> m_queuedProjectFiles;

    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::shared_ptr<frontend_resources::FrontendResourcesMap> frontend_resources;

    megamol::frontend_resources::GUIState m_providedStateResource;
    megamol::frontend_resources::GUIRegisterWindow m_providedRegisterWindowResource;
    megamol::frontend_resources::common_types::lua_func_type* m_exec_lua;

    std::string resource_request_gui_state(bool as_lua);
    bool resource_request_gui_visibility(void);
    float resource_request_gui_scale(void);
    void resource_provide_gui_state(const std::string& json_state);
    void resource_provide_gui_visibility(bool show);
    void resource_provide_gui_scale(float scale);

    void resource_register_window(
        const std::string& name, std::function<void(megamol::gui::AbstractWindow::BasicConfig&)>& func);
    void resource_register_popup(const std::string& name, std::weak_ptr<bool> open, std::function<void(void)>& func);
    void resource_register_notification(const std::string& name, std::weak_ptr<bool> open, const std::string& message);
};

} // namespace frontend
} // namespace megamol

#endif // MEGAMOL_GUI_SERVICE_HPP_INCLUDED
