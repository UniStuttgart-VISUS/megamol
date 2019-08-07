/*
 * Render3DUI.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef RENDER_3D_UI_H_INCLUDED
#define RENDER_3D_UI_H_INCLUDED

#include "glowl/FramebufferObject.hpp"
#include "RenderMDIMesh.h"

namespace megamol {
namespace mesh {

    class Render3DUI : public RenderMDIMesh
    {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char* ClassName(void) { return "Render3DUI"; }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char* Description(void) {
            return "Renderer for mesh-based 3D UI elements. Build on top of RenderMDIMesh.";
        }

        /**
         *
         */
        bool OnMouseButton(
            core::view::MouseButton       button,
            core::view::MouseButtonAction action,
            core::view::Modifiers         mods) override;

        bool OnMouseMove(double x, double y) override;

        /** Ctor. */
        Render3DUI();

        /** Dtor. */
        ~Render3DUI();

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
        bool GetExtents(core::Call& call);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        bool Render(core::Call& call);
    
    private:

        double m_cursor_x, m_cursor_y;

        std::unique_ptr<glowl::FramebufferObject> m_fbo;

        megamol::core::CallerSlot m_3DInteraction_callerSlot;

    };

}
}

#endif // !RENDER_3D_UI_H_INCLUDED
