/*
 * DirPartFilter.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DirPartFilter_H_INCLUDED
#define MEGAMOLCORE_DirPartFilter_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "mmcore/CallVolumeData.h"
#include "vislib/RawStorage.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace moldyn {


    /**
     * Module to modulate the colour (saturation) of directional particles based on a scalar volume.
     */
    class DirPartFilter : public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "DirPartFilter";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module to filter partices based on the scalar value at their position";
        }

        /**
         * Answers whether this module is available on the current system.
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
        DirPartFilter(void);

        /** Dtor. */
        virtual ~DirPartFilter(void);

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
         * Callback publishing the gridded data
         *
         * @param call The call requesting the gridded data
         *
         * @return 'true' on success, 'false' on failure
         */
        bool getData(Call& call);

        /**
         * Callback publishing the extend of the data
         *
         * @param call The call requesting the extend of the data
         *
         * @return 'true' on success, 'false' on failure
         */
        bool getExtend(Call& call);

        CallerSlot inParticlesDataSlot;

        CallerSlot inVolumeDataSlot;

        CalleeSlot outDataSlot;

        param::ParamSlot attributeSlot;

        param::ParamSlot attrMinValSlot;

        param::ParamSlot attrMaxValSlot;

        SIZE_T datahashOut;

        SIZE_T datahashParticlesIn;

        SIZE_T datahashVolumeIn;

        unsigned int frameID;

        vislib::RawStorage vertData;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DirPartFilter_H_INCLUDED */
