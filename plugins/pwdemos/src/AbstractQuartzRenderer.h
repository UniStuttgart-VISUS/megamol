/*
 * AbstractQuartzRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractQuartzModule.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallClipPlane.h"


namespace megamol {
namespace demos {

    /**
     * AbstractQuartzRenderer
     */
    class AbstractQuartzRenderer : public AbstractQuartzModule{
    public:

        /**
         * Ctor
         */
        AbstractQuartzRenderer(void);

        /**
         * Dtor
         */
        virtual ~AbstractQuartzRenderer(void);

    protected:

        /**
         * Updates the grain colour if necessary
         */
        void assertGrainColour(void);

        /**
         * Answer the clipping plane from the connected module
         *
         * @return The clipping plane from the connected module or NULL if no
         *         data could be received
         */
        core::view::CallClipPlane *getClipPlaneData(void);

        /** Slot connecting to the clipping plane provider */
        core::CallerSlot clipPlaneSlot;

        /** The call for light sources */
        core::CallerSlot lightsSlot;

        /** Shows/Hides the bounding box polygon of the clipping plane */
        core::param::ParamSlot showClipPlanePolySlot;

        /** The colour to be used for the crystalites */
        core::param::ParamSlot grainColSlot;

        /** The grain colour */
        float grainCol[3];

        /** Activate correct handling of periodic boundary conditions */
        core::param::ParamSlot correctPBCSlot;

    };

} /* end namespace demos */
} /* end namespace megamol */

