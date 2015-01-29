/*
 * AbstractSimpleParticleDataSource.h
 *
 * Copyright (C) 2012 by TU Dresden (CGV)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTSIMPLEPARTICLEDATASOURCE_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTSIMPLEPARTICLEDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "Module.h"
#include "mmcore/param/ParamSlot.h"
#include "CalleeSlot.h"
#include "MultiParticleDataCall.h"


namespace megamol {
namespace core {
namespace moldyn {


    /**
     * Abstract base class for simple particle loaders (single time step = no animation)
     */
    class MEGAMOLCORE_API AbstractSimpleParticleDataSource : public Module {
    public:

    protected:

        /** Ctor. */
        AbstractSimpleParticleDataSource(void);

        /** Dtor. */
        virtual ~AbstractSimpleParticleDataSource(void);

        /**
         * Gets the data from the source.
         *
         * @param call The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        virtual bool getData(MultiParticleDataCall& call) = 0;

        /**
         * Gets the data from the source.
         *
         * @param call The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        virtual bool getExtent(MultiParticleDataCall& call) = 0;

        /** The file name */
        param::ParamSlot filenameSlot;

    private:

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getDataCallback(Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getExtentCallback(Call& caller);

        /** The slot for requesting data */
        CalleeSlot getDataSlot;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTSIMPLEPARTICLEDATASOURCE_H_INCLUDED */
