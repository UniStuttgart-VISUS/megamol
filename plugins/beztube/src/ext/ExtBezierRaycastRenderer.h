/*
 * ExtBezierRaycastRenderer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef BEZTUBE_EXTBEZIERRAYCASTRENDERER_H_INCLUDED
#define BEZTUBE_EXTBEZIERRAYCASTRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "beztube/beztube.h"
#include "AbstractBezierRenderer.h"
#include "vislib/graphics/gl/GLSLShader.h"


namespace megamol {
namespace beztube {
namespace ext {

    /**
     * Raycasting-based renderer for bézier curve tubes
     */
    class ExtBezierRaycastRenderer : public AbstractBezierRenderer {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ExtBezierRaycastRenderer";
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

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false; // TODO: Change as soon as it works
        }

        /** Ctor. */
        ExtBezierRaycastRenderer(void);

        /** Dtor. */
        virtual ~ExtBezierRaycastRenderer(void);

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
        virtual bool render(core::view::CallRender3D& call);

        /** The point-based shader with elliptic profile */
        vislib::graphics::gl::GLSLShader pbEllShader;

        /** vertex attribute for y-axis and z-radius of point 0 */
        int pbEllShaderYAx1Pos;

        /** vertex attribute for position and y-radius of point 1 */
        int pbEllShaderPos2Pos;

        /** vertex attribute for y-axis and z-radius of point 1 */
        int pbEllShaderYAx2Pos;

        /** vertex attribute for position and y-radius of point 2 */
        int pbEllShaderPos3Pos;

        /** vertex attribute for y-axis and z-radius of point 2 */
        int pbEllShaderYAx3Pos;

        /** vertex attribute for position and y-radius of point 3 */
        int pbEllShaderPos4Pos;

        /** vertex attribute for y-axis and z-radius of point 3 */
        int pbEllShaderYAx4Pos;

        /** vertex attribute for colours of all points */
        int pbEllShaderColours;

    private:

    };

} /* end namespace ext */
} /* end namespace beztube */
} /* end namespace megamol */

#endif /* BEZTUBE_EXTBEZIERRAYCASTRENDERER_H_INCLUDED */
