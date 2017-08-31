/*
 * DataSourceModule.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DATASOURCEMODULE_H_INCLUDED
#define MEGAMOLCORE_DATASOURCEMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/Module.h"
#include "vislib/graphics/graphicstypes.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Base class of rendering graph data source modules.
     *
     * The data returned (including the bounding box) will be scaled and moved
     * while rendering to gain optimal conditions regarding z-buffer
     * resolution. The values for scaling and offset will be set by the
     * framework and should not be altered by the data source. Note that the
     * data should first be scaled and then be offsetted.
     */
    class DataSourceModule : public Module {
    public:

        /** Ctor. */
        DataSourceModule(void);

        /** Dtor. */
        virtual ~DataSourceModule(void);

        /**
         * Gets the scene space bounding box of the dataset. This bounding box
         * must enclose all elements of the dataset for all time frames.
         *
         * @return The scene space bounding box of the dataset.
         */
        virtual const vislib::graphics::SceneSpaceCuboid& BoundingBox(void)
            const = 0;

        /**
         * Gets the number of time frames of the dataset. This default
         * implementation returns '1' meaning that the dataset is not time-
         * dependent. A return value of zero will result in undefined
         * behaviour.
         *
         * @return The number of time frames of the dataset.
         */
        virtual unsigned int NumberOfTimeFrames(void) const;

        /**
         * Gets the data offset vector.
         *
         * @return The data offset vector.
         */
        inline const vislib::graphics::SceneSpaceVector3D&
        OffsetVector(void) const {
            return this->offsetVec;
        }

        /**
         * Gets the data scale factor.
         *
         * @return The data scale factor.
         */
        inline vislib::graphics::SceneSpaceType ScaleFactor(void) const {
            return this->scale;
        }

        /**
         * Sets the data offset vector.
         *
         * @param off The new data offset vector.
         */
        inline void SetOffsetVector(
                const vislib::graphics::SceneSpaceVector3D& off) {
            this->offsetVec = off;
        }

        /**
         * Sets the data scale factor.
         *
         * @param s The data scale factor.
         */
        inline void SetScaleFactor(vislib::graphics::SceneSpaceType s) {
            this->scale = s;
        }

    protected:

        /** The data offset vector. */
        vislib::graphics::SceneSpaceVector3D offsetVec;

        /** The data scale factor */
        vislib::graphics::SceneSpaceType scale;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DATASOURCEMODULE_H_INCLUDED */
