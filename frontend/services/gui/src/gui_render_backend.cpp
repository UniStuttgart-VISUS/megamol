/*
 * gui_render_backend.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "gui_render_backend.h"


using namespace megamol::gui;


gui_render_backend::gui_render_backend() : initialized_backend(GUIRenderBackend::NONE) {

}


gui_render_backend::~gui_render_backend() {

    this->fbo.reset();
}


bool megamol::gui::gui_render_backend::CheckPrerequisites(GUIRenderBackend backend) {

    // Check prerequisites for given render backend
    switch (backend) {

    case (GUIRenderBackend::OPEN_GL): {
#ifdef WITH_GL
        bool prerequisities_given = true;

#ifdef _WIN32 // Windows
        HDC ogl_current_display = ::wglGetCurrentDC();
        HGLRC ogl_current_context = ::wglGetCurrentContext();
        if (ogl_current_display == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] There is no OpenGL rendering context available.");
            prerequisities_given = false;
        }
        if (ogl_current_context == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] There is no current OpenGL rendering context available from the calling thread.");
            prerequisities_given = false;
        }
#else
        // LINUX
        /// XXX The following throws segfault if OpenGL is not loaded yet:
        // Display* gl_current_display = ::glXGetCurrentDisplay();
        // GLXContext ogl_current_context = ::glXGetCurrentContext();
        /// XXX Is there a better way to check existing OpenGL context?
        if (glXGetCurrentDisplay == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] There is no OpenGL rendering context available.");
            prerequisities_given = false;
        }
        if (glXGetCurrentContext == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] There is no current OpenGL rendering context available from the calling thread.");
            prerequisities_given = false;
        }
#endif // _WIN32
        if (!prerequisities_given) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Missing prerequisities to initialize render backend OpenGL. [%s, %s, line %d]", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
#else
        megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Render backend OpenGL is not available.");
        return false;
#endif // WITH_GL
    } break;

    case (ImGuiRenderBackend::CPU): {



        /// TODO



    } break;

    default: {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown render backen... [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    } break;
    }

    return true;
}


bool gui_render_backend::InitializeBackend(GUIRenderBackend backend) {

    if (this->IsBackendInitialized()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Render backend is already initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }

    switch (backend) {
    case (ImGuiRenderBackend::OPEN_GL): {
#ifdef WITH_GL
        // Init OpenGL for ImGui
        if (ImGui_ImplOpenGL3_Init(nullptr)) {
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Created ImGui render backend for Open GL.");
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unable to initialize OpenGL render backend for ImGui. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
            return false;
        }
#endif
    } break;
    case (ImGuiRenderBackend::CPU): {
        /*

         /// TODO

        // Init CPU (=software) renderer for ImGui
        if (/// TODO(nullptr)) {
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Created ImGui render Backend for CPU.");
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unable to initialize CPU render backend for ImGui. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
            return false;
        }
        */
    } break;
    default: {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown render backend. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    } break;
    }

    this->initialized_backend = backend;
    return true;
}


void gui_render_backend::NewFrame() {

    switch (this->gui_render_backend) {
    case (ImGuiRenderBackend::OPEN_GL): {
#ifdef WITH_GL
        ImGui_ImplOpenGL3_NewFrame();
#endif
    } break;
    case (ImGuiRenderBackend::CPU): {
        /// TODO
    } break;
    default: {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown render backend. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    } break;
    }


}


bool gui_render_backend::EnableBackendRendering(size_t width, size_t height) {

    switch (this->initialized_backend) {
    case (ImGuiRenderBackend::OPEN_GL): {
#ifdef WITH_GL
        bool create_fbo = false;
        if (this->fbo == nullptr) {
            create_fbo = true;
        } else if (((this->fbo->getWidth() != width) || (this->fbo->getHeight() != height)) && (width != 0) &&
                   (height != 0)) {
            create_fbo = true;
        }
        if (create_fbo) {
            try {
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                this->fbo.reset();
                this->fbo = std::make_shared<glowl::FramebufferObject>(width, height);
                this->fbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
                // TODO: check completness and throw if not?
            } catch (glowl::FramebufferObjectException const& exc) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Unable to create framebuffer object: %s [%s, %s, line %d]\n", exc.what(), __FILE__,
                    __FUNCTION__, __LINE__);
            }
        }
        if (this->fbo == nullptr)
            break;
        this->fbo->bind();
        // glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        // glClearDepth(1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, width, height);
        glEnable(GL_DEPTH_TEST);
#endif
    } break;
    case (ImGuiRenderBackend::CPU): {
        /// TODO
    } break;
    default: {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown render backend. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    } break;
    }
}


bool gui_render_backend::Render() {

    switch (this->imgui_initialized_rbnd) {
    case (ImGuiRenderBackend::OPEN_GL): {
#ifdef WITH_GL
        ImGui_ImplOpenGL3_RenderDrawData(draw_data);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif
    } break;
    case (ImGuiRenderBackend::CPU): {
        /// TODO
    } break;
    default: {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown render backend. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    } break;
    }
}


bool megamol::gui::gui_render_backend::ShutdownBackend() {

    switch (this->initialized_backend) {
    case (GUIRenderBackend::OPEN_GL): {
#ifdef WITH_GL
        ImGui_ImplOpenGL3_Shutdown();
#endif
    } break;
    case (GUIRenderBackend::CPU): {


        /// TODO


    } break;
    default:
        break;
    }
    this->initialized_backend = GUIRenderBackend::NONE;
    return true;
}


bool gui_render_backend::CreateFont() {

    switch (this->imgui_initialized_rbnd) {
    case (ImGuiRenderBackend::OPEN_GL): {
#ifdef WITH_GL
        font_api_load_success = ImGui_ImplOpenGL3_CreateFontsTexture();
#endif
    } break;
    case (ImGuiRenderBackend::CPU): {
        /// TODO
    } break;
    default: {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] ImGui API is not supported. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    } break;
    }

    /*
     * switch (this->imgui_initialized_rbnd) {
case (ImGuiRenderBackend::NONE): {
    megamol::core::utility::log::Log::DefaultLog.WriteError(
        "[GUI] Fonts can only be loaded after API was initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
        __LINE__);
} break;
case (ImGuiRenderBackend::OPEN_GL): {
#ifdef WITH_GL
    ImGui_ImplOpenGL3_CreateFontsTexture();
#endif
} break;
case (ImGuiRenderBackend::CPU): {
    /// TODO
} break;
default: {
    megamol::core::utility::log::Log::DefaultLog.WriteError(
        "[GUI] ImGui API is not supported. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
} break;
}
     * */
}

