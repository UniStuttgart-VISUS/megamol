/*
 * Window.h
 *
 * Copyright (C) 2008, 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_GL_WINDOW_H_INCLUDED
#define MEGAMOLCON_GL_WINDOW_H_INCLUDED
#pragma once

#include "CoreHandle.h"
#include <memory>
#include <vector>
#include "vislib/graphics/gl/IncludeAllGL.h"
#ifndef USE_EGL
#include "GLFW/glfw3.h"
#include "gl/glfwInst.h"
#else
    #include <EGL/egl.h>
#endif
#include "utility/ConfigHelper.h"
#include "mmcore/api/MegaMolCore.h"
#include "AbstractUILayer.h"
#include "vislib/graphics/FpsCounter.h"
#include "UILayersCollection.hpp"
#include <chrono>
#include <array>
#include <cstring>

namespace megamol {
namespace console {
namespace gl {

    /**
     * Class of viewing windows.
     */
    class Window {
    public:
        static void MEGAMOLCORE_CALLBACK RequestCloseCallback(void *w) {
            Window* win = static_cast<Window*>(w);
            if (win != nullptr) win->RequestClose();
        }

#ifndef USE_EGL
        Window(const char* title, const utility::WindowPlacement & placement, GLFWwindow* share);
#else
        Window(const char* title, const utility::WindowPlacement & placement, EGLContext* share);
#endif
        virtual ~Window();

        void EnableVSync();
        void AddUILayer(std::shared_ptr<AbstractUILayer> uiLayer);
        void RemoveUILayer(std::shared_ptr<AbstractUILayer> uiLayer);
        inline void ForceIssueResizeEvent() {
            on_resize(width, height);
        }

        inline void * Handle() {
            return hView;
        }
        
        inline void const * Handle() const {
            return hView;
        }
        
#ifndef USE_EGL
        inline GLFWwindow * WindowHandle() const {
            return hWnd;
        }
#else
        inline EGLContext * WindowHandle() const {
            return hWnd;
        }
#endif
        
        inline const char* Name() const {
            return name.c_str();
        }
        
        inline const float& FPS() const {
            return fps;
        }
        
        inline float LiveFPS() const {
            return fpsCntr.FPS();
        }
        
        inline const std::array<float, 20>& LiveFPSList() const {
            return fpsList;
        }
        
        inline bool ShowFPSinTitle() const {
            return showFpsInTitle;
        }
        void SetShowFPSinTitle(bool show);
        inline bool ShowSamplesinTitle() const {
            return showFragmentsInTitle;
        }
        void SetShowSamplesinTitle(bool show);
        inline bool ShowPrimsinTitle() const {
            return showPrimsInTitle;
        }
        void SetShowPrimsinTitle(bool show);

        inline bool IsAlive() const {
            return hWnd != nullptr;
        }
        void RequestClose();

        void Update();

    private:
#ifdef USE_EGL
        void captureFramebufferPPM(GLuint framebuffer, uint32_t width, uint32_t height, const std::string& path);
#endif

#ifndef USE_EGL
        static void glfw_onKey_func(GLFWwindow* wnd, int k, int s, int a, int m);
        static void glfw_onChar_func(GLFWwindow* wnd, unsigned int charcode);
        static void glfw_onMouseMove_func(GLFWwindow* wnd, double x, double y);
        static void glfw_onMouseButton_func(GLFWwindow* wnd, int b, int a, int m);
        static void glfw_onMouseWheel_func(GLFWwindow* wnd, double x, double y);
#endif

        void on_resize(int w, int h);
        void on_fps_value(float fps_val);

        CoreHandle hView; // Handle to the core view instance
        
#ifndef USE_EGL
        std::shared_ptr<glfwInst> glfw;
        GLFWwindow *hWnd; // glfw window handle
#else
        EGLContext *hWnd; // EGL context
        EGLDisplay eglDisplay; // EGL display
        EGLSurface eglSurface; // EGL surface
#endif
        int width, height;
        mmcRenderViewContext renderContext;
        UILayersCollection uiLayers;
        std::shared_ptr<AbstractUILayer> mouseCapture;
        std::string name;
        vislib::graphics::FpsCounter fpsCntr;
        float fps;
        std::array<float, 20> fpsList;
        bool showFpsInTitle;
        std::chrono::system_clock::time_point fpsSyncTime;
        bool topMost;
        GLuint fragmentQuery;
        GLuint primsQuery;
        bool showFragmentsInTitle;
        bool showPrimsInTitle;
    };

} /* end namespace gl */
} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_GL_WINDOW_H_INCLUDED */
