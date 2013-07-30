/*
 * AbstractBezierRaycastRenderer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTBEZIERRAYCASTRENDERER_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTBEZIERRAYCASTRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "view/Renderer3DModule.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
#include "view/CallRender3D.h"
#include "vislib/GLSLShader.h"


namespace megamol {
namespace core {
namespace misc {

    /**
     * Raycasting-based renderer for bézier curve tubes
     */
    class MEGAMOLCORE_API AbstractBezierRaycastRenderer : public view::Renderer3DModule {
    public:

    protected:

        /** Ctor. */
        AbstractBezierRaycastRenderer(void);

        /** Dtor. */
        virtual ~AbstractBezierRaycastRenderer(void);

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
        virtual bool GetCapabilities(Call& call);

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(Call& call);

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
        virtual bool Render(Call& call);

        /**
         * The implementation of the render callback
         *
         * @param call The calling rendering call
         *
         * @return The return value of the function
         */
        virtual bool render(view::CallRender3D& call) = 0;

        /** The call for data */
        CallerSlot getDataSlot;

        /** The data hash of the objects stored in the list */
        SIZE_T objsHash;

        /** The selected shader */
        vislib::graphics::gl::GLSLShader *shader;

        /** The scale used when rendering */
        float scaling;

    private:

    };

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTBEZIERRAYCASTRENDERER_H_INCLUDED */
