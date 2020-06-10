/*
 * OpenGL_GLFW_RAPI.hpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_OPENGL_GLFW_RAPI_HPP_INCLUDED
#define MEGAMOL_OPENGL_GLFW_RAPI_HPP_INCLUDED
#pragma once

#include "AbstractRenderAPI.hpp"

#include "KeyboardMouse_Events.h"
#include "Framebuffer_Events.h"
#include "Window_Events.h"

#include <memory>

namespace megamol {
namespace render_api {

struct WindowPlacement {
    int x = 100, y = 100, w = 800, h = 600, mon = 0;
    bool pos = false;
    bool size = false;
    bool noDec = false;
    bool fullScreen = false;
    bool topMost = false;
};

class OpenGL_GLFW_RAPI final : public AbstractRenderAPI {
    using KeyboardEvents = megamol::input_events::KeyboardEvents;
    using MouseEvents = megamol::input_events::MouseEvents;
    using WindowEvents = megamol::input_events::WindowEvents;
    using FramebufferEvents = megamol::input_events::FramebufferEvents;

public:
    // make capabilities of OpenGL/GLFW RAPI statically query-able (we would like to force this from the abstract
    // class, but this is not possible?)
    std::string getAPIName() const override { return std::string{"OpenGL GLFW"}; };
    RenderAPIVersion getAPIVersion() const override { return RenderAPIVersion{0, 0}; };

    // TODO: how to force RAPI subclasses to implement a Config struct which should be passed to constructor??
    // set sane defaults for all options here, so usage is as simple as possible
    struct Config {
        int versionMajor = 4;
        int versionMinor = 6;
        std::string windowTitlePrefix = "MegaMol";
        void* sharedContextPtr = nullptr;
        std::string viewInstanceName = "";
        WindowPlacement windowPlacement{}; // window position, glfw creation hints // TODO: sane defaults??
        bool enableKHRDebug = true;        // max error reporting
        bool enableVsync = false;          // max frame rate
                                           // TODO: request OpenGL context version, extensions?
    };

    OpenGL_GLFW_RAPI() = default;
    ~OpenGL_GLFW_RAPI() override;
    // TODO: delete copy/move/assign?

    // init API, e.g. init GLFW with OpenGL and open window with certain decorations/hints
    bool initAPI(const Config& config);
    bool initAPI(void* configPtr) override;
    void closeAPI() override;

    void preViewRender() override;  // prepare rendering with API, e.g. set OpenGL context, frame-timers, etc
    void postViewRender() override; // clean up after rendering, e.g. stop and show frame-timers in GLFW window

    // expose the resources and input events this RAPI provides: Keyboard inputs, Mouse inputs, GLFW Window events, Framebuffer resize events
    const std::vector<RenderResource>& getRenderResources() const override;

    const void* getAPISharedDataPtr() const override; // ptr non-owning, share data should be only borrowed

    // from AbstractRenderAPI:
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

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

    // abstract away GLFW library details behind pointer-to-implementation. only use GLFW header in .cpp
    struct PimplData;
    std::unique_ptr<PimplData, std::function<void(PimplData*)>> m_pimpl;
    void updateWindowTitle();

    // GLFW fills those events and we propagate them to the View3D/the MegaMol graph
    KeyboardEvents m_keyboardEvents;
    MouseEvents m_mouseEvents;
    WindowEvents m_windowEvents;
    FramebufferEvents m_framebufferEvents;

    // this holds references to the event structs we fill. the events are passed to the renderers/views using
    // const std::vector<RenderResource>& getRenderResources() override
    std::vector<RenderResource> m_renderResourceReferences;
};

} // namespace render_api
} // namespace megamol

#endif MEGAMOL_OPENGL_GLFW_RAPI_HPP_INCLUDED