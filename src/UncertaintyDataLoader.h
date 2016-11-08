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
			* @param c The calling call
			*
			* @return True on success
			*/
			bool getData(megamol::core::Call& call);

		private:
			/**
			* Load information about amino acids and residues from a PDB file.
			*
			* @param filename The PDB file name.
			*/
			void loadPDBFile(const vislib::TString &filename);

			/** The data callee slot */
			core::CalleeSlot dataOutSlot;

			/** the parameter slot for the binding site file (PDB) */
			core::param::ParamSlot pdbFilenameSlot;
			// the file name for the color table
			megamol::core::param::ParamSlot colorTableFileParam;

			/** The binding site information */
			vislib::Array<vislib::Array<vislib::Pair<char, unsigned int> > > bindingSites;
			/** Pointer to binding site residue name array */
			vislib::Array<vislib::Array<vislib::StringA> > bindingSiteResNames;
			/** The binding site name */
			vislib::Array<vislib::StringA> bindingSiteNames;
			/** The binding site description */
			vislib::Array<vislib::StringA> bindingSiteDescription;

			// color table
			vislib::Array<vislib::math::Vector<float, 3> > colorLookupTable;
			// color table
			vislib::Array<vislib::math::Vector<float, 3> > bindingSiteColors;

		};
	} /* end namespace protein_uncertainty */
} /* end namespace megamol */

#endif // MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYDATALOADER_H_INCLUDED
