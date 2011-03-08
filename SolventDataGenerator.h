/*
 * SolventDataGenerator.h
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_SolventDataGenerator_H_INCLUDED
#define MMPROTEINPLUGIN_SolventDataGenerator_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "param/ParamSlot.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "vislib/Array.h"
#include "vislib/Vector.h"
#include "vislib/Cuboid.h"
#include "CallProteinData.h"
#include "MolecularDataCall.h"
#include "Stride.h"
#include "view/AnimDataModule.h"
#include <fstream>



namespace megamol {
namespace protein {

    /**
     * generator for hydrogent bounds etc ...
	 * this class can be put in place between PDBLoader and a molecule renderer (SolventVolumeRenderer for example)...
     */

    class SolventDataGenerator : public megamol::core::/*view::AnimData*/Module
    {
    public:

        /** Ctor */
        SolventDataGenerator(void);

        /** Dtor */
        virtual ~SolventDataGenerator(void);

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)  {
            return "SolventDataGenerator";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Special molecule data preprocessing stepts (from PDB data).";
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
         * Call callback to get the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getData( core::Call& call);

        bool getExtent( core::Call& call);

		/**
         * Implementation of 'Release'.
         */
        virtual void release(void);

	private:

        /**********************************************************************
         * variables
         **********************************************************************/

        /** caller slot */
        megamol::core::CallerSlot molDataInputCallerSlot;

        /** The data callee slot */
        megamol::core::CalleeSlot dataOutSlot;
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_SolventDataGenerator_H_INCLUDED
