/*
 * ResidueSelection.h
 *
 * Author: Daniel Kauker & Michael Krone
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MEGAMOL_PROTEIN_RESIDUE_SELECTION_H_INCLUDED
#define MEGAMOL_PROTEIN_RESIDUE_SELECTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "vislib/math/Cuboid.h"
#include "vislib/RawStorage.h"
#include "vislib/Array.h"
#include "protein_calls/ResidueSelectionCall.h"


namespace megamol {
namespace protein {

    /**
     * zeug
     */
    class ResidueSelection : public core::Module {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ResidueSelection";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module holding a list of selected residues (IDs, ...)";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        ResidueSelection(void);

        /** Dtor. */
        virtual ~ResidueSelection(void);

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
        bool getSelectionCallback(core::Call& caller);

        /**
         * Sets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool setSelectionCallback(core::Call& caller);

        /** The slot for requesting data */
        core::CalleeSlot getSelectionSlot;

        /** The data */
        vislib::Array<protein_calls::ResidueSelectionCall::Residue> selection;

        /** The data hash */
        SIZE_T datahash;

    };

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MEGAMOL_PROTEIN_SELECTION_H_INCLUDED */
