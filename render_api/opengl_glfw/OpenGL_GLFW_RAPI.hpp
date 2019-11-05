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
#include "AbstractUILayer.h"

#include <string>
#include <tuple>
#include <vector>
#include <functional>

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

class OpenGL_GLFW_RAPI : public AbstractRenderAPI {
public:
    struct UIEvents {
        std::vector<std::tuple<render_api::Key, render_api::KeyAction, render_api::Modifiers>> onKey_list;
        std::vector<unsigned int> onChar_list;
        std::vector<std::tuple<double, double>> onMouseMove_list;
        std::vector<std::tuple<render_api::MouseButton, render_api::MouseButtonAction, render_api::Modifiers>> onMouseButton_list;
        std::vector<std::tuple<double, double>> onMouseWheel_list;
        std::vector<std::tuple<int, int>> resize_list;
        bool is_window_resized = false;

        void clear() {
            this->onKey_list.clear();
            this->onChar_list.clear();
            this->onMouseMove_list.clear();
            this->onMouseButton_list.clear();
            this->onMouseWheel_list.clear();
            this->resize_list.clear();
            this->is_window_resized = false;
        }
    };

public:
    // make capabilities of OpenGL/GLFW RAPI statically query-able (we would like to force this from the abstract
    // class, but this is not possible?)
    std::string getAPIName() const override { return std::string{"OpenGL GLFW"}; };
    RenderAPIVersion getAPIVersion() const override { return RenderAPIVersion{0, 0}; };

    // how to force RAPI subclasses to implement a Config struct which should be passed to constructor??
    // set sane defaults for all options here, so usage is as simple as possible
    struct Config {
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

    const void* getAPISharedDataPtr() const override; // ptr non-owning, share data should be only borrowed

    void AddUILayer(std::shared_ptr<AbstractUILayer> uiLayer);
    void RemoveUILayer(std::shared_ptr<AbstractUILayer> uiLayer);
    UIEvents& getUIEvents();

    // TODO: register passing user inputs? via existing callbacks?
    void glfw_onKey_func(int k, int s, int a, int m);
    void glfw_onChar_func(unsigned int charcode);
    void glfw_onMouseMove_func(double x, double y);
    void glfw_onMouseButton_func(int b, int a, int m);
    void glfw_onMouseWheel_func(double x, double y);

    void on_resize(int w, int h);
    bool checkWindowResize();

    // from AbstractRenderAPI:
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

private:
    struct PimplData;
    std::unique_ptr<PimplData, std::function<void(PimplData*)>>
        m_pimpl; // abstract away GLFW library details behind pointer-to-implementation
    void updateWindowTitle();

    UIEvents ui_events;
};

} // namespace render_api
} // namespace megamol

#endif MEGAMOL_OPENGL_GLFW_RAPI_HPP_INCLUDED