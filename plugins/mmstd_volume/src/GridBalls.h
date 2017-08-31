/*
 * GridBalls.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_VOLUME_GRIDBALLS_H_INCLUDED
#define MMSTD_VOLUME_GRIDBALLS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Cuboid.h"


namespace megamol {
namespace stdplugin {
namespace volume {


    /**
     * Class generation buck ball informations
     */
    class GridBalls : public core::Module {
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
        bool outDataCallback(core::Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool outExtentCallback(core::Call& caller);

        /** The slot for requesting input data */
        core::CallerSlot inDataSlot;

        /** The slot for requesting output data */
        core::CalleeSlot outDataSlot;

        /** The attribute to show */
        core::param::ParamSlot attributeSlot;

        /** The spheres' radius */
        core::param::ParamSlot radiusSlot;

        /** minimum value */
        core::param::ParamSlot lowValSlot;

        /** maximum value */
        core::param::ParamSlot highValSlot;

        /** The grid positions */
        float* grid;

        /** The grid sizes */
        unsigned int sx, sy, sz;

        /** The object space bounding box */
        vislib::math::Cuboid<float> osbb;

    };


} /* end namespace volume */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_VOLUME_GRIDBALLS_H_INCLUDED */
