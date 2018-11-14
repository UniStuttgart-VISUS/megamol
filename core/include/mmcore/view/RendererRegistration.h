/*
 * MuxRenderer3D.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERERREGISTRATIO_H_INCLUDED
#define MEGAMOLCORE_RENDERERREGISTRATIO_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/BoundingBoxes.h"
#include "mmcore/CallerSlot.h"
#include "CallRender3D.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ParamSlot.h"
#include "Renderer3DModule.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * A module that modifies the ModelView matrix of subsequent
     * renderers for manual "registration".
     */
    class RendererRegistration : public Renderer3DModule {
    public:

        /**
         * Gets the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "RendererRegistration";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "A module that modifies the ModelView matrix of subsequent renderers for manual 'registration'";
        }

        /**
         * Gets whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /** Ctor. */
        RendererRegistration(void);

        /** Dtor. */
        virtual ~RendererRegistration(void);

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
        virtual bool GetExtents(Call& call);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

    private:

        /** The renderer caller slots */
        CallerSlot rendererSlot;

        param::ParamSlot scaleXSlot;
        param::ParamSlot scaleYSlot;
        param::ParamSlot scaleZSlot;

        param::ParamSlot translateXSlot;
        param::ParamSlot translateYSlot;
        param::ParamSlot translateZSlot;

        param::ParamSlot rotateXSlot;
        param::ParamSlot rotateYSlot;
        param::ParamSlot rotateZSlot;

        /** The frame count */
        unsigned int frameCnt;

        /** The bounding boxes */
        BoundingBoxes bboxs;

        float scale;
    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDERERREGISTRATIO_H_INCLUDED */
