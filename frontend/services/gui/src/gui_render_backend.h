/*
 * gui_render_backend.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIRENDERBACKEND_H_INCLUDED
#define MEGAMOL_GUI_GUIRENDERBACKEND_H_INCLUDED
#pragma once


#include "imgui.h"
#include "imgui_impl_generic.h"
#include "imgui_sw.h"
#include "mmcore/view/CPUFramebuffer.h"
#include <glm/glm.hpp>
#include <memory>

#ifdef WITH_GL
#include <glowl/FramebufferObject.hpp>
#endif


namespace megamol {
namespace gui {


enum class GUIRenderBackend { NONE, CPU, OPEN_GL };


/** ************************************************************************
 * Managing available ImGui backends
 */
class gui_render_backend {
public:
    /**
     * CTOR.
     */
    gui_render_backend();

    /**
     * DTOR.
     */
    ~gui_render_backend();

    bool IsBackendInitialized() {
        return (this->initialized_backend != GUIRenderBackend::NONE);
    }

    bool CheckPrerequisites(GUIRenderBackend backend);

    bool Init(GUIRenderBackend backend);

    void NewFrame(glm::vec2 framebuffer_size, glm::vec2 window_size);

    bool EnableRendering(unsigned int width, unsigned int height);

    bool Render(ImDrawData* draw_data);

    bool ShutdownBackend();

    bool CreateFontsTexture();

#ifdef WITH_GL

    inline void GetFBOData_GL(
        unsigned int& out_fbo_color_buffer_gl_handle, size_t& out_fbo_width, size_t& out_fbo_height) const {
        if (this->ogl_fbo == nullptr) {
            out_fbo_color_buffer_gl_handle = 0;
            out_fbo_width = 0;
            out_fbo_height = 0;
        } else {
            // IS THIS SAFE?? IS THIS THE COLOR BUFFER??
            out_fbo_color_buffer_gl_handle = this->ogl_fbo->getColorAttachment(0)->getName();
            out_fbo_width = static_cast<size_t>(this->ogl_fbo->getWidth());
            out_fbo_height = static_cast<size_t>(this->ogl_fbo->getHeight());
        }
    }

#else

    inline std::shared_ptr<megamol::core::view::CPUFramebuffer>& GetFBOData_CPU() {
        return this->cpu_fbo;
    }

#endif // WITH_GL

private:
    // VARIABLES --------------------------------------------------------------

    GUIRenderBackend initialized_backend;

    GenericWindow sw_window;
    GenericMonitor sw_monitor;

#ifdef WITH_GL
    std::shared_ptr<glowl::FramebufferObject> ogl_fbo = nullptr;
#endif // WITH_GL
    std::shared_ptr<megamol::core::view::CPUFramebuffer> cpu_fbo = nullptr;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIRENDERBACKEND_H_INCLUDED
