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

#include "param/ParamSlot.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "MolecularDataCall.h"
#include "vislib/Array.h"
//#include "vislib/Cuboid.h"

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
            return "Offers molecular data filtering.";
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
        //void UpdateParameters(const MolecularDataCall *mol);

    private:
    
        /**
         * Helper class to unlock frame data.
         */
        class Unlocker : public MolecularDataCall::Unlocker {
        public:

            /**
             * Ctor.
             *
             * @param mol The molecular data call whos 'Unlock'-method is to be 
             *            called.
             */
            Unlocker(MolecularDataCall& mol) : MolecularDataCall::Unlocker(),
                mol(&mol){
                // intentionally empty
            }

            /** Dtor. */
            virtual ~Unlocker(void) {
                this->Unlock();
            }

            /** Unlocks the data */
            virtual void Unlock(void) {
                this->mol->Unlock();            
            }
            
        private:
        
            MolecularDataCall *mol;

        };

        /**********************************************************************
         * variables
         **********************************************************************/

        /** caller slot */
        megamol::core::CallerSlot molDataCallerSlot;
        
        /** The data callee slot */
        megamol::core::CalleeSlot dataOutSlot;
        
        /** parameter slot for positional interpolation */
        megamol::core::param::ParamSlot interpolParam;
        
        /** The data hash */
        SIZE_T datahash;
        
        /** Array of interpolated atom positions */
        vislib::Array<float> atomPosInter;

    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_FILTER_H_INCLUDED
