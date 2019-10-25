/*
 * QuartzRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/view/Renderer3DModule_2.h"
#include "AbstractMultiShaderQuartzRenderer.h"
#include "mmcore/CallerSlot.h"
#include "QuartzCrystalDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "QuartzParticleGridDataCall.h"
#include "mmcore/view/CallRender3D_2.h"
#include "vislib/graphics/gl/GLSLShader.h"


namespace megamol {
namespace demos {

    /**
     * Module rendering gridded quarts particle data
     */
    class QuartzRenderer : public core::view::Renderer3DModule_2, public AbstractMultiShaderQuartzRenderer {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "QuartzRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module rendering gridded quartz particles using GLSL ray casting shader";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return AbstractMultiShaderQuartzRenderer::IsAvailable();
        }

        /** Ctor */
        QuartzRenderer(void);

        /** Dtor */
        virtual ~QuartzRenderer(void);

    protected:

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(core::view::CallRender3D_2& call);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(core::view::CallRender3D_2& call);

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        /**
         * Creates a raycasting shader for the specified crystalite
         *
         * @param c The crystalite
         *
         * @return The shader
         */
        virtual vislib::graphics::gl::GLSLShader* makeShader(const CrystalDataCall::Crystal& c);

        /** Shows/Hides the axes (x and y) of the clipping plane */
        core::param::ParamSlot showClipAxesSlot;

    };

} /* end namespace demos */
} /* end namespace megamol */

