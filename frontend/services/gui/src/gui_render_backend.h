/*
 * gui_render_backend.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIRENDERBACKEND_H_INCLUDED
#define MEGAMOL_GUI_GUIRENDERBACKEND_H_INCLUDED
#pragma once



#ifdef WITH_GL
#include "imgui_impl_opengl3.h"
#include <glowl/FramebufferObject.hpp>
#else
#include "imgui_sw.h"
#include "backends/imgui_impl_generic.h"
#include "mmcore/view/CPUFramebuffer.h"
#endif


namespace megamol {
namespace gui {


enum class GUIRenderBackend { NONE, OPEN_GL, CPU };


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

    bool InitializeBackend(GUIRenderBackend backend);

    void NewFrame();

    bool EnableRendering(size_t width, size_t height);

    bool Render();

    bool ShutdownBackend();

    bool CreateFont();

#ifdef WITH_GL

    inline void GetFBOData_GL(
        unsigned int& out_fbo_color_buffer_gl_handle, size_t& out_fbo_width, size_t& out_fbo_height) const {
        if (this->fbo == nullptr) {
            out_fbo_color_buffer_gl_handle = 0;
            out_fbo_width = 0;
            out_fbo_height = 0;
        } else {
            // IS THIS SAFE?? IS THIS THE COLOR BUFFER??
            out_fbo_color_buffer_gl_handle = this->fbo->getColorAttachment(0)->getName();
            out_fbo_width = this->fbo->getWidth();
            out_fbo_height = this->fbo->getHeight();
        }
    }

#else

    inline std::shared_ptr<core::view::CPUFramebuffer> GetFBOData_CPU() {
        return this->fbo;
    }

#endif // WITH_GL

private:

    // VARIABLES --------------------------------------------------------------

    GUIRenderBackend initialized_backend;

// FBO
#ifdef WITH_GL
    std::shared_ptr<glowl::FramebufferObject> fbo;
#else
    std::shared_ptr<core::view::CPUFramebuffer> fbo;
#endif // WITH_GL

    // FUNCTIONS --------------------------------------------------------------


};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIRENDERBACKEND_H_INCLUDED
