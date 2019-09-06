/*
 * BezierCPUMeshRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractBezierRenderer.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace demos {

    /**
     * Mesh-based renderer for bézier curve tubes
     */
    class BezierCPUMeshRenderer : public AbstractBezierRenderer {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "BezierCPUMeshRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for bézier curve";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        BezierCPUMeshRenderer(void);

        /** Dtor. */
        virtual ~BezierCPUMeshRenderer(void);

    protected:

        /**
         * The implementation of the render callback
         *
         * @param call The calling rendering call
         *
         * @return The return value of the function
         */
        virtual bool render(core::view::CallRender3D_2& call);

        /**
         * Informs the class if the shader is required
         *
         * @return True if the shader is required
         */
        virtual bool shader_required(void) const {
            return false;
        }

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

        /** Powerfull brute force tube rendering with many parameters */
        void drawTube(
            float const *p1, float const *r1, unsigned char const *c1,
            float const *p2, float const *r2, unsigned char const *c2,
            float const *p3, float const *r3, unsigned char const *c3,
            float const *p4, float const *r4, unsigned char const *c4,
            bool hasRad, bool hasCol, int curSeg, int proSeg, int capSeg);

        /** The number of linear sections along the curve */
        core::param::ParamSlot curveSectionsSlot;

        /** The number of section along the profile */
        core::param::ParamSlot profileSectionsSlot;

        /** The number of linear sections along the curve */
        core::param::ParamSlot capSectionsSlot;

        /** The display list storing the objects */
        unsigned int geo;

        /** The incoming data hash */
        size_t dataHash;

    };

} /* end namespace demos */
} /* end namespace megamol */
