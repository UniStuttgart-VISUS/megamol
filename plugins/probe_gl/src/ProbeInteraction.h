/*
 * ProbeInteraction.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef PROBE_INTERACTION_H_INCLUDED
#define PROBE_INTERACTION_H_INCLUDED

#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

#include "ProbeInteractionCollection.h"

namespace megamol {
namespace probe_gl {

class ProbeInteraction : public megamol::mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ProbeInteraction";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module for handling viewport interaction with probes.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
#ifdef _WIN32
#if defined(DEBUG) || defined(_DEBUG)
        HDC dc = ::wglGetCurrentDC();
        HGLRC rc = ::wglGetCurrentContext();
        ASSERT(dc != NULL);
        ASSERT(rc != NULL);
#endif // DEBUG || _DEBUG
#endif // _WIN32
        return true;
    }

    bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    bool OnMouseMove(double x, double y) override;

    ProbeInteraction();
    ~ProbeInteraction();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create();

    /**
     * Implementation of 'Release'.
     */
    void release();

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call);

    bool getInteractionMetaData(core::Call& call);

private:
    uint32_t m_version;

    double m_cursor_x, m_cursor_y;

    double m_cursor_x_lastRightClick, m_cursor_y_lastRightClick;

    bool m_open_context_menu;

    bool m_open_showMenu_dropdown;
    bool m_open_probeMenu_dropdown;
    bool m_open_dataMenu_dropdown;

    bool m_open_dataFilterByDepth_popup;

    bool m_show_probes;
    bool m_show_hull;
    bool m_show_glyphs;

    /** Map storing the pressed state of all mouse buttons */
    std::map<core::view::MouseButton, bool> m_mouse_button_states = {{core::view::MouseButton::BUTTON_1, false},
        {core::view::MouseButton::BUTTON_2, false}, {core::view::MouseButton::BUTTON_3, false},
        {core::view::MouseButton::BUTTON_4, false}, {core::view::MouseButton::BUTTON_5, false},
        {core::view::MouseButton::BUTTON_6, false}, {core::view::MouseButton::BUTTON_7, false},
        {core::view::MouseButton::BUTTON_8, false}, {core::view::MouseButton::BUTTON_LEFT, false},
        {core::view::MouseButton::BUTTON_MIDDLE, false}, {core::view::MouseButton::BUTTON_RIGHT, false}};

    int32_t last_active_probe_id;

    std::vector<int32_t> m_selected_probes;

    megamol::core::CallerSlot m_probe_fbo_slot;
    megamol::core::CallerSlot m_hull_fbo_slot;
    megamol::core::CallerSlot m_glyph_fbo_slot;

    megamol::core::CallerSlot m_event_write_slot;

    // local storage of projection and view matrix (for 3D space interaction computations)
    glm::mat4 m_view_mx_cpy;
    glm::mat4 m_proj_mx_cpy;
};

} // namespace probe_gl
} // namespace megamol


#endif // !PROBE_INTERACTION_H_INCLUDED
