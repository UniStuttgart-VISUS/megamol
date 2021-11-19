/*
 * OpenGL_GLFW_Service.hpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractFrontendService.hpp"

#include "KeyboardMouse_Events.h"
#include "Framebuffer_Events.h"
#include "Window_Events.h"
#include "WindowManipulation.h"
#include "OpenGL_Context.h"

#include <memory>

namespace megamol {
namespace frontend {

struct WindowPlacement {
    int x = 100, y = 100, w = 800, h = 600, mon = 0;
    bool pos = false;
    bool size = false;
    bool noDec = false;
    bool fullScreen = false;
    bool topMost = false;
    bool noCursor = false;
};

class OpenGL_GLFW_Service final : public AbstractFrontendService {
    using KeyboardEvents = megamol::frontend_resources::KeyboardEvents;
    using MouseEvents = megamol::frontend_resources::MouseEvents;
    using WindowEvents = megamol::frontend_resources::WindowEvents;
    using FramebufferEvents = megamol::frontend_resources::FramebufferEvents;
    using WindowManipulation = megamol::frontend_resources::WindowManipulation;

public:

    struct Config {
        int versionMajor = 4;
        int versionMinor = 6;
        std::string windowTitlePrefix = "MegaMol";
        WindowPlacement windowPlacement{}; // window position, glfw creation hints // TODO: sane defaults??
        bool enableKHRDebug = true;        // max error reporting
        bool enableVsync = false;          // max frame rate
        bool glContextCoreProfile = false;
    };

    std::string serviceName() const override { return "OpenGL_GLFW_Service"; }

    OpenGL_GLFW_Service() = default;
    ~OpenGL_GLFW_Service() override {}
    // TODO: delete copy/move/assign?

    void resetProvidedResources() override {
        m_keyboardEvents.clear();
        m_mouseEvents.clear();
        m_windowEvents.clear();
        m_framebufferEvents.clear();
    }

    void preGraphRender() override {}  // prepare rendering with API, e.g. set OpenGL context, frame-timers, etc
    void postGraphRender() override {
        this->do_every_second();
    } // clean up after rendering, e.g. stop and show frame-timers in GLFW window

    // expose the resources and input events this service provides: Keyboard inputs, Mouse inputs, GLFW Window events, Framebuffer resize events
    std::vector<FrontendResource>& getProvidedResources() override {
        return m_renderResourceReferences;
    }
    const std::vector<std::string> getRequestedResourceNames() const override {
        return m_requestedResourcesNames;
        }

#ifdef WITH_GL
    // init API, e.g. init GLFW with OpenGL and open window with certain decorations/hints
    bool init(const Config& config);
    bool init(void* configPtr) override;
    void close() override;

    void updateProvidedResources() override;
    void digestChangedRequestedResources() override;
    void setRequestedResources(std::vector<FrontendResource> resources) override;

    // GLFW event callbacks need to be public for technical reasons.
    // keyboard events
    void glfw_onKey_func(const int key, const int scancode, const int action, const int mods);
    void glfw_onChar_func(const unsigned int codepoint);

    // mouse events
    void glfw_onMouseButton_func(const int button, const int action, const int mods);
    void glfw_onMouseCursorPosition_func(const double xpos, const double ypos);
    void glfw_onMouseCursorEnter_func(const bool entered);
    void glfw_onMouseScroll_func(const double xoffset, const double yoffset);

    // window events
    void glfw_onWindowSize_func(const int width /* in screen coordinates, of the window */, const int height);
    void glfw_onWindowFocus_func(const bool focused);
    void glfw_onWindowShouldClose_func(const bool shouldclose);
    void glfw_onWindowIconified_func(const bool iconified);
    void glfw_onWindowContentScale_func(const float xscale, const float yscale);
    void glfw_onPathDrop_func(const int path_count, const char* paths[]);

    // framebuffer events
    void glfw_onFramebufferSize_func(const int widthpx, const int heightpx);

private:
    void register_glfw_callbacks();
    void do_every_second();

    void create_glfw_mouse_cursors();
    void update_glfw_mouse_cursors(const int cursor_id);
#else
        // init API, e.g. init GLFW with OpenGL and open window with certain decorations/hints
        bool init(const Config& config) {return 1;}
        bool init(void* configPtr) override {return 1;}
        void close() override {}

        void updateProvidedResources() override {}
        void digestChangedRequestedResources() override {}
        void setRequestedResources(std::vector<FrontendResource> resources) override {}

        // GLFW event callbacks need to be public for technical reasons.
        // keyboard events
        void glfw_onKey_func(const int key, const int scancode, const int action, const int mods) {}
        void glfw_onChar_func(const unsigned int codepoint) {}

        // mouse events
        void glfw_onMouseButton_func(const int button, const int action, const int mods) {}
        void glfw_onMouseCursorPosition_func(const double xpos, const double ypos) {}
        void glfw_onMouseCursorEnter_func(const bool entered) {}
        void glfw_onMouseScroll_func(const double xoffset, const double yoffset) {}

        // window events
        void glfw_onWindowSize_func(const int width, const int height) {}
        void glfw_onWindowFocus_func(const bool focused) {}
        void glfw_onWindowShouldClose_func(const bool shouldclose) {}
        void glfw_onWindowIconified_func(const bool iconified) {}
        void glfw_onWindowContentScale_func(const float xscale, const float yscale) {}
        void glfw_onPathDrop_func(const int path_count, const char* paths[]) {}

        // framebuffer events
        void glfw_onFramebufferSize_func(const int widthpx, const int heightpx) {}

    private:
        void register_glfw_callbacks() {}
        void do_every_second() {}

        void create_glfw_mouse_cursors() {}
        void update_glfw_mouse_cursors(const int cursor_id) {}
#endif

    // abstract away GLFW library details behind pointer-to-implementation. only use GLFW header in .cpp
    struct PimplData;
    std::unique_ptr<PimplData, std::function<void(PimplData*)>> m_pimpl;

    // GLFW fills those events and we propagate them to the View3D/the MegaMol graph
    KeyboardEvents m_keyboardEvents;
    MouseEvents m_mouseEvents;
    WindowEvents m_windowEvents;
    FramebufferEvents m_framebufferEvents;
    frontend_resources::OpenGL_Context m_opengl_context;
    WindowManipulation m_windowManipulation;

    // this holds references to the event structs we fill. the events are passed to the renderers/views using
    // const std::vector<FrontendResource>& getModuleResources() override
    std::vector<FrontendResource> m_renderResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::vector<FrontendResource> m_requestedResourceReferences;
};

} // namespace frontend
} // namespace megamol
