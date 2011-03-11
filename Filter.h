/*
 * Filter.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_FILTER_H_INCLUDED
#define MMPROTEINPLUGIN_FILTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "MolecularDataCall.h"


namespace megamol {
namespace protein {

    /*
     * Filter class
     */
     
    class Filter : public megamol::core::Module {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)
        {
            return "Filter";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void)
        {
            return "Offers data filtering.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void)
        {
            return true;
        }

        /** Ctor. */
        Filter(void);

        /** Dtor. */
        virtual ~Filter(void);

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
         * Call callback to get the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getData(core::Call& call);

        /**
         * Call callback to get the extent of the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getExtent(core::Call& call);
        
        /**
         * Update all parameter slots.
         *
         * @param mol   Pointer to the data call.
         */
        void updateParams(const MolecularDataCall *mol);

    private:

        /**********************************************************************
         * variables
         **********************************************************************/

        /** caller slot */
        megamol::core::CallerSlot molDataCallerSlot;
        
        /** The data callee slot */
        megamol::core::CalleeSlot dataOutSlot;
        
        /** The calltime of the last call */
        float calltimeOld;

    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_FILTER_H_INCLUDED
