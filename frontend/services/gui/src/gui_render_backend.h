/*
 * gui_render_backend.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <memory>

#include <glm/glm.hpp>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <imgui_sw.hpp>

#include "ImageWrapper.h"
#include "imgui_backends/imgui_impl_generic.h"
#include "mmcore/view/CPUFramebuffer.h"

#ifdef MEGAMOL_USE_OPENGL
#include <glowl/FramebufferObject.hpp>
#endif // MEGAMOL_USE_OPENGL


namespace megamol::gui {


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

    void ClearFrame();

    void NewFrame(glm::vec2 framebuffer_size, glm::vec2 window_size);

    bool EnableRendering(unsigned int framebuffer_width, unsigned int framebuffer_height);

    bool Render(ImDrawData* draw_data);

    bool ShutdownBackend();

    bool CreateFontsTexture();

    bool SupportsCustomFonts() const;

    megamol::frontend_resources::ImageWrapper GetImage();

private:
    // VARIABLES --------------------------------------------------------------

    GUIRenderBackend initialized_backend;

    GenericWindow cpu_window;
    GenericMonitor cpu_monitor;

#ifdef MEGAMOL_USE_OPENGL
    std::shared_ptr<glowl::FramebufferObject> ogl_fbo = nullptr;
#endif // MEGAMOL_USE_OPENGL
    std::shared_ptr<megamol::core::view::CPUFramebuffer> cpu_fbo = nullptr;

    // FUNCTIONS --------------------------------------------------------------

    bool createCPUFramebuffer(unsigned int width, unsigned int height);
    bool createOGLFramebuffer(unsigned int width, unsigned int height);
};


} // namespace megamol::gui
