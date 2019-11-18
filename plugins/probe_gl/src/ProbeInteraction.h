/*
 * ProbeInteraction.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef PROBE_INTERACTION_H_INCLUDED
#define PROBE_INTERACTION_H_INCLUDED

#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "ProbeInteractionCollection.h"

namespace megamol {
namespace probe_gl {

class ProbeInteraction : public megamol::core::view::Renderer3DModule_2
{
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ProbeInteraction"; }

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
#    if defined(DEBUG) || defined(_DEBUG)
        HDC dc = ::wglGetCurrentDC();
        HGLRC rc = ::wglGetCurrentContext();
        ASSERT(dc != NULL);
        ASSERT(rc != NULL);
#    endif // DEBUG || _DEBUG
#endif     // _WIN32
        return true;
    }

    bool OnMouseButton(
        core::view::MouseButton button, 
        core::view::MouseButtonAction action, 
        core::view::Modifiers mods) override;

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
    bool GetExtents(core::view::CallRender3D_2& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(core::view::CallRender3D_2& call);

private:

    double m_cursor_x, m_cursor_y;

    megamol::core::CallerSlot m_probe_fbo_slot;
    megamol::core::CallerSlot m_hull_fbo_slot;

    megamol::core::CalleeSlot m_interaction_collection_slot;
};

}
}



#endif // !PROBE_INTERACTION_H_INCLUDED
