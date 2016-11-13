/*
 * UncertaintyDataLoader.h
 *
 * Author: Matthias Braun
 * Copyright (C) 2016 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * This module is based on the source code of "BindingSiteData" in megamol protein plugin (svn revision 1500).
 *
 */


/**
* TODO:
* - param pdb-id -> filename ...
*
*/


#ifndef MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYDATALOADER_H_INCLUDED
#define MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYDATALOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/String.h"
#include "vislib/math/Vector.h"


namespace megamol {
	namespace protein_uncertainty {


		class UncertaintyDataLoader : public megamol::core::Module {
		public:

			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "UncertaintyDataLoader";
			}

			/**
			* Answer a human readable description of this module.
			*
			* @return A human readable description of this module.
			*/
			static const char *Description(void) {
				return "Offers protein secondary structure uncertainty data.";
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
			UncertaintyDataLoader(void);

			/** dtor */
			~UncertaintyDataLoader(void);

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
			* @param call The calling call
			*
			* @return True on success
			*/
			bool getData(megamol::core::Call& call);

		private:

			/**
			* Read the input file containing secondary structure data collected by the the python script.
			*
			* @param filename The filename of the uncertainty input data file.
			*/
			void readInputFile(const vislib::TString& filename);


			// ------------------ variables ------------------- 

			/** The data callee slot */
			core::CalleeSlot dataOutSlot;

			/** the parameter slot for the pdb id */
			core::param::ParamSlot filenameSlot;


			/** The DSSP secondary structure information */
            vislib::Array<char> dsspSecStructure;
			
			/** The STRIDE secondary structure information */
            vislib::Array<char> strideSecStructure;

			/** The PDB secondary structure information */
            vislib::Array<char> pdbSecStructure;

			/** The pdb index with amino-acid three letter code and chain ID*/
			vislib::Array<vislib::Pair<int, vislib::Pair<vislib::StringA, char> > > indexAminoAcidchainID;

		};
	} /* end namespace protein_uncertainty */
} /* end namespace megamol */

#endif // MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYDATALOADER_H_INCLUDED
