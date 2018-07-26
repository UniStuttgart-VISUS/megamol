/*
 * QuartzParticleFortLoader.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "vislib/math/Cuboid.h"


namespace megamol {
namespace demos {

    /**
     * Module for loading quartz particle data from binary-fortran files
     */
    class ParticleFortLoader : public megamol::core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "QuartzParticleFortLoader";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module for loading quartz particle data from binary-fortran files";
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
        ParticleFortLoader(void);

        /** Dtor */
        virtual ~ParticleFortLoader(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Call callback to get the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getData(core::Call& c);

        /**
         * Call callback to get the extent
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getExtent(core::Call& c);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        /**
         * Ensures the correct data is loaded
         */
        void assertData(void);

        /** The data callee slot */
        core::CalleeSlot dataOutSlot;

        /** The path to the position file */
        core::param::ParamSlot positionFileNameSlot;

        /** The path to the attribute file (radius + orientation) */
        core::param::ParamSlot attributeFileNameSlot;

        /** The data hash */
        SIZE_T datahash;

        /** The number of particle types */
        unsigned int typeCnt;

        /** The particle crystal type for each type */
        unsigned int *partTypes;

        /** The particle count for each type */
        unsigned int *partCnts;

        /** The particel data for each type */
        float **partDatas;

        /** The data bounding box */
        vislib::math::Cuboid<float> bbox;

        /** The data clip box */
        vislib::math::Cuboid<float> cbox;

        /** Flag whether or not to use the calculated bouning box */
        core::param::ParamSlot autoBBoxSlot;

        /** The minimum values for the manual bounding box */
        core::param::ParamSlot bboxMinSlot;

        /** The maximum values for the manual bounding box */
        core::param::ParamSlot bboxMaxSlot;

    };

} /* end namespace demos */
} /* end namespace megamol */
