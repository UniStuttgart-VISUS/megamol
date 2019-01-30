/*
 * ScreenSpaceEdgeRenderer.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_TRISOUP_SCREENSPACEEDGERENDERER_H_INCLUDED
#define MEGAMOL_TRISOUP_SCREENSPACEEDGERENDERER_H_INCLUDED
#pragma once

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/FramebufferObject.h"


namespace megamol {
namespace trisoup {


    /**
     * Renderer for tri-mesh data
     */
    class ScreenSpaceEdgeRenderer : public core::view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ScreenSpaceEdgeRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Screen-Space Renderer to only show edges";
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
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
                && vislib::graphics::gl::FramebufferObject::AreExtensionsAvailable();
        }

        /** Ctor. */
        ScreenSpaceEdgeRenderer(void);

        /** Dtor. */
        virtual ~ScreenSpaceEdgeRenderer(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(core::Call& call);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(core::Call& call);

    private:

        /** The slot to fetch the data */
        core::CallerSlot rendererSlot;
        core::param::ParamSlot colorSlot;

        vislib::graphics::gl::FramebufferObject fbo;
        vislib::graphics::gl::GLSLShader shader;

    };


} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOL_TRISOUP_SCREENSPACEEDGERENDERER_H_INCLUDED */
