/*
 * BezierRaycastRenderer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BEZIERRAYCASTRENDERER_H_INCLUDED
#define MEGAMOLCORE_BEZIERRAYCASTRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "misc/AbstractBezierRaycastRenderer.h"
//#include "CallerSlot.h"
//#include "param/ParamSlot.h"
#include "vislib/GLSLShader.h"


namespace megamol {
namespace core {
namespace misc {

    /**
     * Raycasting-based renderer for bézier curve tubes
     */
    class BezierRaycastRenderer : public AbstractBezierRaycastRenderer {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "BezierRaycastRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Ray caster for bézier curve";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
        }

        /** Ctor. */
        BezierRaycastRenderer(void);

        /** Dtor. */
        virtual ~BezierRaycastRenderer(void);

    protected:

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

        /**
         * The implementation of the render callback
         *
         * @param call The calling rendering call
         *
         * @return The return value of the function
         */
        virtual bool render(view::CallRender3D& call);

        ///**
        // * The render callback.
        // *
        // * @param call The calling call.
        // *
        // * @return The return value of the function.
        // */
        //virtual bool Render(Call& call);

        ///** The call for data */
        //CallerSlot getDataSlot;

        ///** The data hash of the objects stored in the list */
        //SIZE_T objsHash;

        /** The point-based shader */
        vislib::graphics::gl::GLSLShader pbShader;

        int pbShaderPos2Pos;
        int pbShaderPos3Pos;
        int pbShaderPos4Pos;
        int pbShaderCol2Pos;
        int pbShaderCol3Pos;
        int pbShaderCol4Pos;

        ///** The scale used when rendering */
        //float scaling;

    private:

    };

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_BEZIERRAYCASTRENDERER_H_INCLUDED */
