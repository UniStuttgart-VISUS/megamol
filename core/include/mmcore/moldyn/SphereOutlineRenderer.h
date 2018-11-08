/*
 * SphereOutlineRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SPHEREOUTLINERENDERER_H_INCLUDED
#define MEGAMOLCORE_SPHEREOUTLINERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace core {
namespace moldyn {

    /**
     * Renderer for simple sphere glyphs
     */
    class SphereOutlineRenderer : public view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SphereOutlineRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for outlines of sphere glyphs";
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
        SphereOutlineRenderer(void);

        /** Dtor. */
        virtual ~SphereOutlineRenderer(void);

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

    private:

        /** The call for data */
        CallerSlot getDataSlot;

        /** The base colour for the sphere outline */
        param::ParamSlot colourSlot;

        /** The representation type */
        param::ParamSlot repSlot;

        /** The number of line segments to construct the circle/sphere */
        param::ParamSlot circleSegSlot;

        /** The (half) number of additional outlines */
        param::ParamSlot multiOutlineCntSlot;

        /** The distance of the additional outlines as angles in radians */
        param::ParamSlot multiOutLineDistSlot;

        /** The sphere quadric */
        void *sphereQuadric;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SPHEREOUTLINERENDERER_H_INCLUDED */
