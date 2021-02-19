/*
 * AbstractStereoView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTSTEREOVIEW_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTSTEREOVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractOverrideView.h"


namespace megamol {
namespace core {
namespace view {
namespace special {


    /**
     * Abstract base class of override rendering views
     */
    class AbstractStereoView : public AbstractOverrideView {
    public:

        /** Ctor. */
        AbstractStereoView(void);

        /** Dtor. */
        virtual ~AbstractStereoView(void);

    protected:

        /**
         * Packs the mouse coordinates, which are relative to the virtual
         * viewport size.
         *
         * @param x The x coordinate of the mouse position
         * @param y The y coordinate of the mouse position
         */
        virtual void packMouseCoordinates(float &x, float &y);

        /**
         * Answer the projection type to be used
         *
         * @return The selected stereo projection type
         */
        thecam::Projection_type getProjectionType(void) const;

        /**
         * Answer the flag if the eyes should be switched
         *
         * @return The switch-eyes flag
         */
        bool getSwitchEyes(void) const;

    private:

        /** The projection type slot */
        param::ParamSlot projTypeSlot;

        /** Flag to switch eyes */
        param::ParamSlot switchEyesSlot;

    };

} /* end namespace special */
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTSTEREOVIEW_H_INCLUDED */
