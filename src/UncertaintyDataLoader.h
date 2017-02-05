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

#include "UncertaintyDataCall.h"


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
            *
            *@return True on success
			*/
			bool ReadInputFile(const vislib::TString& filename);



            /**
            * Quick sorting of the corresponding types of the secondary structure uncertainties.
            *
            * @param valueArr  The pointer to the vector keeping the uncertainty values
            * @param structArr The pointer to the vector keeping the structural indices
            * @param left      The left index of the array
            * @param right     The right index of the array
            */
            void QuickSortUncertainties(vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> *valueArr,  
                                        vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> *structArr,  
                                        int left, int right);

            /**
             * enumeration of available uncertainty calculation methods. 
             */
             enum calculationMethod {
                 AVERAGE    = 0,
				 EXTENDED   = 1
             };
             
            /**
            * Compute uncertainty on current secondary structure data with method AVERAGE.
            *
            *@return True on success
            */
            bool CalculateUncertaintyAverage(void);    

			/**
			* Compute uncertainty on current secondary structure data with method EXTENDED.
			*
			*@return True on success
			*/
			bool CalculateUncertaintyExtended(void);

			// ------------------ variables ------------------- 

			/** The data callee slot */
			core::CalleeSlot dataOutSlot;

			/** The parameter slot for the uid filename */
			core::param::ParamSlot filenameSlot;

			/** The parameter slot for choosing uncertainty calculation method */
			core::param::ParamSlot methodSlot;
            
            /** The currently used uncertainty calculation method */
            calculationMethod currentMethod;
            

            /** The PDB index */
			vislib::Array<vislib::StringA> pdbIndex;

            /** The chain ID */
            vislib::Array<char> chainID;

             /** The flag giving additional information */
            vislib::Array<UncertaintyDataCall::addFlags> residueFlag;

            /** The amino-acid name */
            vislib::Array<vislib::StringA> aminoAcidName;
            


            /** The uncertainty of the assigned secondary structure types for each assignment method and for each amino-acid */
            vislib::Array<vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> > > secStructUncertainty;
            
            /** The sorted assigned secondary structure types (sorted by descending uncertainty values) for each assignment method and for each amino-acid */
            vislib::Array<vislib::Array<vislib::math::Vector<static_cast<int>(UncertaintyDataCall::secStructure), static_cast<int>(UncertaintyDataCall::secStructure::NOE)> > > sortedSecStructAssignment;
            // sortedSecStructUncertainty
            // secStructAssignment
            
            
			/** The secondary structure assignment length for each assignment method */
			vislib::Array<vislib::Array<unsigned int> > secStructLength;
                        
			/** The uncertainty of secondary structure assignment for each amino-acid */
			vislib::Array<float> uncertainty;


            /** The 5 STRIDE threshold values per amino-acid */
			vislib::Array<vislib::math::Vector<float, 5> > strideStructThreshold;
            /** The 2 STRIDE energy values per amino-acid */
			vislib::Array<vislib::math::Vector<float, 2> > strideStructEnergy;
            /** The 4 DSSP energy values per amino-acid */
            vislib::Array<vislib::math::Vector<float, 4> > dsspStructEnergy;


            /** The pdb id */
            vislib::StringA pdbID;
            
            /** The pdb assignment method for helix */
            UncertaintyDataCall::pdbAssMethod pdbAssignmentHelix;            
            
            /** The pdb assignment method for helix */
            UncertaintyDataCall::pdbAssMethod pdbAssignmentSheet;              

		};
	} /* end namespace protein_uncertainty */
} /* end namespace megamol */

#endif // MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYDATALOADER_H_INCLUDED
