/*
 * GridBalls.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_GRIDBALLS_H_INCLUDED
#define MEGAMOLCORE_GRIDBALLS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
#include "vislib/Cuboid.h"


namespace megamol {
namespace core {


    /**
     * Class generation buck ball informations
     */
    class GridBalls : public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "GridBalls";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Maps volume data to a ball grid.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor */
        GridBalls(void);

        /** Dtor */
        virtual ~GridBalls(void);

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

    private:

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool outDataCallback(Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool outExtentCallback(Call& caller);

        /** The slot for requesting input data */
        CallerSlot inDataSlot;

        /** The slot for requesting output data */
        CalleeSlot outDataSlot;

        /** The attribute to show */
        param::ParamSlot attributeSlot;

        /** The spheres' radius */
        param::ParamSlot radiusSlot;

        /** minimum value */
        param::ParamSlot lowValSlot;

        /** maximum value */
        param::ParamSlot highValSlot;

        /** The grid positions */
        float* grid;

        /** The grid sizes */
        unsigned int sx, sy, sz;

        /** The object space bounding box */
        vislib::math::Cuboid<float> osbb;

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_GRIDBALLS_H_INCLUDED */
