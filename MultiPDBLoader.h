/*
 * MultiPDBLoader.h
 *
 * Copyright (C) 2013 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_MULTIPDBLOADER_H_INCLUDED
#define MMPROTEINPLUGIN_MULTIPDBLOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "param/ParamSlot.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "vislib/Array.h"
#include "vislib/math/Cuboid.h"
#include "MolecularDataCall.h"
#include "view/AnimDataModule.h"


namespace megamol {
namespace protein {

    class PDBLoader;

    /**
     * Data source for multiple PDB files
     */
    class MultiPDBLoader : public megamol::core::Module
    {
    public:
        MultiPDBLoader(void);
        virtual ~MultiPDBLoader(void);
        
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)  {
            return "MultiPDBLoader";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers multiple PDB data sets.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

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

        /**
         * Call callback to get the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getData( core::Call& call);

        /**
         * Call callback to get the extent of the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getExtent( core::Call& call);
        
        /** Ensures that the data is loaded */
        void assertData(void);

    private:
        
        /** The file name */
        core::param::ParamSlot filenameSlot;
    
        /** The data callee slot */
        core::CalleeSlot molecularDataOutSlot;
        
        /** The data hash for pdb data */
        SIZE_T dataHash;
    
        /* the structure data */            
        vislib::Array<MolecularDataCall*> datacall;

        /* a pointer to the real data, required to change atom positions! */
        vislib::Array<PDBLoader*> pdb;

    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_MULTIPDBLOADER_H_INCLUDED
