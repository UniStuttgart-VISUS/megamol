/*
 * QuartzPlaneRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractMultiShaderQuartzRenderer.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/view/CallRender2DGL.h"
#include "vislib/graphics/gl/GLSLShader.h"


namespace megamol {
namespace demos {

    /**
     * QuartzPlaneRenderer
     */
    class QuartzPlaneRenderer : public core::view::Renderer2DModule, public AbstractMultiShaderQuartzRenderer {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "QuartzPlaneRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module rendering gridded quartz particles onto a clipping plane";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return AbstractMultiShaderQuartzRenderer::IsAvailable();
        }

        /**
         * Ctor
         */
        QuartzPlaneRenderer(void);

        /**
         * Dtor
         */
        virtual ~QuartzPlaneRenderer(void);

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
        virtual bool GetExtents(core::view::CallRender2DGL& call);

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
        virtual bool Render(core::view::CallRender2DGL& call);

        /**
         * Creates a raycasting shader for the specified crystalite
         *
         * @param c The crystalite
         *
         * @return The shader
         */
        virtual vislib::graphics::gl::GLSLShader* makeShader(const CrystalDataCall::Crystal& c);

    private:

        /** Use clipping plane or grain colour for grains */
        core::param::ParamSlot useClipColSlot;

    };

} /* end namespace demos */
} /* end namespace megamol */
