/*
 * AbstractFrontendService.hpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include <string>

#include "FrontendResource.h"
#include <vector>

namespace megamol::frontend {

class AbstractFrontendService {
private:
    // creation/deletion of render APIs may depend on other RAPI instances, since they may share data or context state.
    // using individual priorities, one can schedule RAPI instances of lower priorities
    // to be initialized after and deleted earlier than their dependencies
    // alternative approach: explicit dependency managment between RAPIs in a graph-like structure, like the MegaMol
    // graph
    int m_processingPriority = 0;
    bool m_shouldShutdown = false;

public:
    int setPriority(const int p) {
        const auto old = m_processingPriority;
        m_processingPriority = p;
        return old;
    }
    int getPriority() const {
        return m_processingPriority;
    }

    bool shouldShutdown() const {
        return m_shouldShutdown;
    }
    void setShutdown(const bool s = true) {
        m_shouldShutdown = s;
    }

public:
    virtual std::string serviceName() const = 0;
    virtual ~AbstractFrontendService() = default;

    virtual bool init(
        void* configPtr) = 0; // init API, e.g. init GLFW with OpenGL and open window with certain decorations/hints
    virtual void close() = 0;

    virtual void updateProvidedResources() = 0;
    virtual void digestChangedRequestedResources() = 0;
    virtual void resetProvidedResources() = 0;

    virtual void preGraphRender() = 0;  // prepare rendering with API, e.g. set OpenGL context, frame-timers, etc
    virtual void postGraphRender() = 0; // clean up after rendering, e.g. stop and show frame-timers in GLFW window

    // we assume that render apis have different kinds of resources they need/want to share with the MegaMol View or Renderers.
    // different kinds of resources are: Keyboard Input Events, Mouse Input Events, Window Resize/Input
    // Events, Framebuffer Resize Events (or more, e.g. incoming configuration requests via network). the resources are
    // looked up via this function and given to the Views that request them. the matching of correct resource to the
    // Views that request them is done by the identifier of the resource (i.e. by a std::string) followed by a cast to
    // the expected type
    virtual std::vector<FrontendResource>& getProvidedResources() = 0;

    virtual const std::vector<std::string> getRequestedResourceNames() const = 0;
    virtual void setRequestedResources(std::vector<FrontendResource> resources) = 0;
};


} // namespace megamol::frontend
