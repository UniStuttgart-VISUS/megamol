/*
 * AbstractBezierRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/GLSLShader.h"


namespace megamol {
namespace demos {

    /**
     * Raycasting-based renderer for bézier curve tubes
     */
    class AbstractBezierRenderer : public core::view::Renderer3DModule {
    public:

    protected:

        /** Ctor. */
        AbstractBezierRenderer(void);

        /** Dtor. */
        virtual ~AbstractBezierRenderer(void);

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

        /**
         * The implementation of the render callback
         *
         * @param call The calling rendering call
         *
         * @return The return value of the function
         */
        virtual bool render(core::view::CallRender3D& call) = 0;

        /**
         * Informs the class if the shader is required
         *
         * @return True if the shader is required
         */
        virtual bool shader_required(void) const {
            // TODO: This is not cool at all
            return true;
        }

        /** The call for data */
        core::CallerSlot getDataSlot;

        /** The data hash of the objects stored in the list */
        SIZE_T objsHash;

        /** The selected shader */
        vislib::graphics::gl::GLSLShader *shader;

        /** The scale used when rendering */
        float scaling;

    private:

    };

} /* end namespace demos */
} /* end namespace megamol */
