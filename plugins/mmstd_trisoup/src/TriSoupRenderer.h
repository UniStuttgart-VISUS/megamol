/*
 * TriSoupRenderer.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2008-2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_TRISOUPRENDERER_H_INCLUDED
#define MEGAMOLCORE_TRISOUPRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer3DModule_2.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Cuboid.h"
#include "vislib/memutils.h"


namespace megamol {
namespace trisoup {


    /**
     * Renderer for tri-mesh data
     */
    class TriSoupRenderer : public core::view::Renderer3DModule_2 {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "TriSoupRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for tri-mesh data";
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
        TriSoupRenderer(void);

        /** Dtor. */
        virtual ~TriSoupRenderer(void);

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
        virtual bool GetExtents(core::view::CallRender3D_2& call);

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
        virtual bool Render(core::view::CallRender3D_2& call);

    private:

        /** The slot to fetch the data */
        core::CallerSlot getDataSlot;

        /** The slot to fetch the volume data */
        core::CallerSlot getVolDataSlot;

        /** Flag whether or not to show vertices */
        core::param::ParamSlot showVertices;

        /** Flag whether or not use lighting for the surface */
        core::param::ParamSlot lighting;

        /** The rendering style for the front surface */
        core::param::ParamSlot surFrontStyle;

        /** The rendering style for the back surface */
        core::param::ParamSlot surBackStyle;
        
        /** The Triangle winding rule */
        core::param::ParamSlot windRule;
        
        /** The Triangle color */
        core::param::ParamSlot colorSlot;

    };


} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TRISOUPRENDERER_H_INCLUDED */
