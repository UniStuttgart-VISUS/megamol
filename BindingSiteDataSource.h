/*
 * BindingSiteData.h
 *
 * Author: Michael Krone
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MEGAMOLPROTEIN_BSITEDATA_H_INCLUDED
#define MEGAMOLPROTEIN_BSITEDATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "param/ParamSlot.h"
#include "CalleeSlot.h"
#include "Module.h"
#include "vislib/String.h"
		

namespace megamol {
namespace protein {

    class BindingSiteDataSource : public megamol::core::Module {
    public:
        
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "BindingSiteDataSource";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers binding site information for biomolecules.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** ctor */
        BindingSiteDataSource( void );
        
        /** dtor */
        ~BindingSiteDataSource( void );
        
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
        bool getData( megamol::core::Call& call);

    private:
        /**
         * Load information about amino acids and residues from a PDB file.
         *
         * @param filename The PDB file name.
         */
        void loadPDBFile( vislib::StringA filename);

        /** The data callee slot */
        core::CalleeSlot dataOutSlot;

        /** the parameter slot for the binding site file (PDB) */
        core::param::ParamSlot pdbFilenameSlot;

        /** The binding site information */
        vislib::Array<vislib::Array<unsigned int> > bindingSites;

    };

} /* end namespace protein */
} /* end namespace megamol */

#endif // MEGAMOLPROTEIN_BSITEDATA_H_INCLUDED
