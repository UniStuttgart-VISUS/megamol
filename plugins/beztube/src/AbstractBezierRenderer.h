/*
 * AbstractBezierRenderer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_BEZTUBE_ABSTRACTBEZIERRAYCASTRENDERER_H_INCLUDED
#define MEGAMOL_BEZTUBE_ABSTRACTBEZIERRAYCASTRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/GLSLShader.h"


namespace megamol {
namespace beztube {

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
         * The get capabilities callback. The module should set the members
         * of 'call' to tell the caller its capabilities.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetCapabilities(core::Call& call);

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

} /* end namespace beztube */
} /* end namespace megamol */

#endif /* MEGAMOL_BEZTUBE_ABSTRACTBEZIERRAYCASTRENDERER_H_INCLUDED */
