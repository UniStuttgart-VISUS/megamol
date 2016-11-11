/*
 * UncertaintyDataCall.h
 *
 * Author: Matthias Braun
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 *
 * This module is based on the source code of "BindingSiteCall" in megamol protein_calls plugin (svn revision 17).
 *
 */


#ifndef MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYDATACALL_H_INCLUDED
#define MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/String.h"
#include "vislib/math/Vector.h"

#include "protein_uncertainty/Protein_Uncertainty.h"


namespace megamol {
	namespace protein_uncertainty {


	/**
	* Class for binding site calls and data interfaces.
	*/

		class PROTEIN_UNCERTAINTY_API UncertaintyDataCall : public megamol::core::Call {
	public:
		/**
		* Answer the name of the objects of this description.
		*
		* @return The name of the objects of this description.
		*/
		static const char *ClassName(void) {
			return "UncertaintyDataCall";
		}

		/**
		* Gets a human readable description of the module.
		*
		* @return A human readable description of the module.
		*/
		static const char *Description(void) {
			return "Call to get uncertaintay data.";
		}

		/** Index of the 'GetData' function */
		static const unsigned int CallForGetData;

		/**
		* Answer the number of functions used for this call.
		*
		* @return The number of functions used for this call.
		*/
		static unsigned int FunctionCount(void) {
			return 1;
		}

		/**
		* Answer the name of the function used for this call.
		*
		* @param idx The index of the function to return it's name.
		*
		* @return The name of the requested function.
		*/
		static const char* FunctionName(unsigned int idx) {
			switch (idx) {
			case 0:
				return "dataOut";
			}
			return "";
		}

        UncertaintyDataCall(void);

        ~UncertaintyDataCall(void);


        // ------------------ GET functions ------------------- 

		/**
		* Get the number of amino-acids.
		*
		* @return The amino-acid count.
		*/
		inline unsigned int GetAminoAcidCount(void) const {
            if (!this->indexAminoAcidchainID)
				return 0;
			else
                return static_cast<unsigned int>(this->indexAminoAcidchainID->Count());
		}

        /**
        * Get the pdb amino-acid index.
        *
        * @return The pdb amino-acid index.
        */
        inline int GetPDBAminoAcidIndex(unsigned int i) const {
            if (!this->indexAminoAcidchainID)
                return 0;
            else if (this->indexAminoAcidchainID->Count() <= i)
                return 0;
            else
                return (this->indexAminoAcidchainID->operator[](i).First());
        }

        /**
        * Get the amino-acid chain id.
        *
        * @return The amino-acid index.
        */
        inline char GetAminoAcidChainID(unsigned int i) const {
            if (!this->indexAminoAcidchainID)
                return 0;
            else if (this->indexAminoAcidchainID->Count() <= i)
                return 0;
            else
                return (this->indexAminoAcidchainID->operator[](i).Second().Second());
        }

        /**
        * Get the amino-acid in three letter code.
        *
        * @return The amino-acid index.
        */
        inline vislib::StringA GetAminoAcid(unsigned int i) const {
            if (!this->indexAminoAcidchainID)
                return 0;
            else if (this->indexAminoAcidchainID->Count() <= i)
                return 0;
            else
                return (this->indexAminoAcidchainID->operator[](i).Second().First());
        }

        /**
        * Get the information of the DSSP secondary structure.
        *
        * @param i The index of the amino-acid.
        * @return Pointer to the array of DSSP secondary structure information.
        */
        inline char GetDsspSecStructure(unsigned int i) const {
            if (!this->dsspSecStructure)
                return static_cast<char>(' ');
            else if (this->dsspSecStructure->Count() <= i)
                return static_cast<char>(' ');
            else
                return (this->dsspSecStructure->operator[](i));
        }

        /**
        * Get the information of the STRIDE secondary structure.
        *
        * @param i The index of the amino-acid.
        * @return Pointer to the array of STRIDE secondary structure information.
        */
        inline char GetStrideSecStructure(unsigned int i) const {
            if (!this->strideSecStructure)
                return static_cast<char>(' ');
            else if (this->strideSecStructure->Count() <= i)
                return static_cast<char>(' ');
            else
                return (this->strideSecStructure->operator[](i));
        }

        /**
        * Get the information of the PDB secondary structure.
        *
        * @param i The index of the amino-acid.
        * @return Pointer to the array of PDB secondary structure information.
        */
        inline char GetPDBSecStructure(unsigned int i) const {
            if (!this->pdbSecStructure)
                return static_cast<char>(' ');
            else if (this->pdbSecStructure->Count() <= i)
                return static_cast<char>(' ');
            else
                return (this->pdbSecStructure->operator[](i));
        }


        // ------------------ SET functions ------------------- 

        /**
        * Set the pointer to the DSSP secondary structure information.
        *
        * @param rnPtr The pointer.
        */
        inline void SetDsspSecStructure(vislib::Array<char> *rnPtr) {
            this->dsspSecStructure = rnPtr;
        }

        /**
        * Set the pointer to the STRIDE secondary structure information.
        *
        * @param rnPtr The pointer.
        */
        inline void SetStrideSecStructure(vislib::Array<char> *rnPtr) {
            this->strideSecStructure = rnPtr;
        }

        /**
        * Set the pointer to the PDB secondary structure information.
        *
        * @param rnPtr The pointer.
        */
        inline void SetPdbSecStructure(vislib::Array<char> *rnPtr) {
            this->pdbSecStructure = rnPtr;
        }

        /**
        * Set the pointer to the DSSP secondary structure information.
        *
        * @param rnPtr The pointer.
        */
        inline void SetIndexAminoAcidchainID(vislib::Array<vislib::Pair<int, vislib::Pair<vislib::StringA, char> > > *rnPtr) {
            this->indexAminoAcidchainID = rnPtr;
        }


	private:

        // ------------------ variables ------------------- 

        /** Pointer to the DSSP secondary structure information */
        vislib::Array<char> *dsspSecStructure;

        /** Pointer to the STRIDE secondary structure information */
        vislib::Array<char> *strideSecStructure;

        /** Pointer to the PDB secondary structure information */
        vislib::Array<char> *pdbSecStructure;

        /** Pointer to the pdb index with amino-acid three letter code and chain ID*/
        vislib::Array<vislib::Pair<int, vislib::Pair<vislib::StringA, char> > > *indexAminoAcidchainID;

    };

    /** Description class typedef */
    typedef megamol::core::factories::CallAutoDescription<UncertaintyDataCall> UncertaintyDataCallDescription;
    
	} /* end namespace protein_uncertainty */
} /* end namespace megamol */

#endif /* MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYDATACALL_H_INCLUDED */
