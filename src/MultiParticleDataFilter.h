/*
 * MultiParticleDataFilter.h
 *
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * Author: Michael Krone
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_MPDFILTER_H_INCLUDED
#define MMPROTEINPLUGIN_MPDFILTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace protein {

    /**
     * Filters particle data by a given attribute.
     */
    class MultiParticleDataFilter : public megamol::core::Module
    {
    public:
        
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "MultiParticleDataFilter";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Filters particle data.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        MultiParticleDataFilter(void);
        ~MultiParticleDataFilter(void);

    protected:
        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'release'.
         */
        virtual void release(void);

        /**
         * Call for get data.
         */
        bool getData(megamol::core::Call& call);

        /**
         * Call for get extent.
         */
        bool getExtent(megamol::core::Call& call);

    private:
        
        /** data caller slot */
        megamol::core::CallerSlot getDataSlot;
        
        /** data caller slot */
        megamol::core::CalleeSlot dataOutSlot;

        /** The parameter slot for the filter threshold */
        megamol::core::param::ParamSlot thresholdParam;
        
        vislib::Array<vislib::Array<float> > vertexData;
        vislib::Array<vislib::Array<float> > colorData;
    };

}
}

#endif // MMPROTEINPLUGIN_MPDFILTER_H_INCLUDED
