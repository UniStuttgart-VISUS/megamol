/*
 * ParticleListFilter.h
 *
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLELISTFILTER_H_INCLUDED
#define MEGAMOLCORE_PARTICLELISTFILTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "Module.h"
#include "param/ParamSlot.h"
#include "moldyn/DirectionalParticleDataCall.h"
#include "moldyn/MultiParticleDataCall.h"
#include "vislib/RawStorage.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace moldyn {


    /**
     * Module to filter calls with multiple particle lists (currently directional and spherical) by list index
     */
    class ParticleListFilter : public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ParticleListFilter";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module to filter calls with multiple particle lists (currently directional and spherical) by particle type";
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
        ParticleListFilter(void);

        /** Dtor. */
        virtual ~ParticleListFilter(void);

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
        bool getDataCallback(Call& call);

        /**
         * Callback publishing the extend of the data
         *
         * @param call The call requesting the extend of the data
         *
         * @return 'true' on success, 'false' on failure
         */
        bool getExtentCallback(Call& call);

        /**
         * Tokenize includedListsSlot->GetValue into an array of type IDs
         *
         * @return the array of type IDs
         */
        vislib::Array<unsigned int> getSelectedLists();

        CallerSlot inParticlesDataSlot;

        CalleeSlot outParticlesDataSlot;

        param::ParamSlot includedListsSlot;

        param::ParamSlot includeAllSlot;

        param::ParamSlot globalColorMapComputationSlot;

        param::ParamSlot includeHiddenInColorMapSlot;

        SIZE_T datahashParticlesOut;

        SIZE_T datahashParticlesIn;

        unsigned int frameID;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLELISTFILTER_H_INCLUDED */
