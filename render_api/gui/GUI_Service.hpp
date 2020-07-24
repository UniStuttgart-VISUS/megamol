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


namespace megamol {
namespace frontend {

class GUI_Service final : public AbstractFrontendService {

public:

    struct Config {
        // imgui api: open_gl 
    };

	std::string serviceName() override { return "GUI_Service"; }

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
    void setRequestedResources(std::vector<ModuleResource>& resources) override;

    // from AbstractFrontendService:
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

private:

    // this holds references to the event structs we fill. the events are passed to the renderers/views using
    // const std::vector<ModuleResource>& getModuleResources() override
    std::vector<ModuleResource> m_providedResourceReferences;

    // this holds references to the event structs we use. 
    std::vector<ModuleResource> m_requestedResourceReferences;
};

} // namespace frontend
} // namespace megamol

#endif MEGAMOL_GUI_SERVICE_HPP_INCLUDED