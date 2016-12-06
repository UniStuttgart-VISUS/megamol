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


	class PROTEIN_UNCERTAINTY_API UncertaintyDataCall : public megamol::core::Call {

	public:

        /**
        * Enumeration of secondary structure types.
        * 
        * (!) Indices must be in the range from 0 to NOE-1.
        */
        enum secStructure {
            H_ALPHA_HELIX = 0,
            G_310_HELIX   = 1,
            I_PI_HELIX    = 2,
            E_EXT_STRAND  = 3,
            T_H_TURN      = 4,
            B_BRIDGE      = 5,
            S_BEND        = 6,
            C_COIL        = 7,
            NOTDEFINED    = 8,
            NOE           = 9   // Number of Elements -> must always be the last index!
        };


        /**
        * Enumeration of assignment methods.
        * 
        * (!) Indices must be in the range from 0 to NOM-1.
        */
        enum assMethod {
            PDB    = 0,
            STRIDE = 1,
            DSSP   = 2,
            NOM    = 3   // Number of Methods -> must always be the last index!
        };
        
        /**
        * Enumeration of additional residue flags.
        */
        enum addFlags {
            NOTHING   = 0,
            MISSING   = 1,
            HETEROGEN = 2
        };
        
        /**
        * Enumeration of pdb secondary structure assignment methods.
        */
        enum pdbAssMethod {
            PDB_PROMOTIF = 0,
			PDB_AUTHOR   = 1,
			PDB_DSSP     = 2,
			PDB_UNKNOWN  = 3
        };        
                
        // ------------------ class functions ------------------- 

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
            if (!this->pdbIndex)
                return static_cast<unsigned int>(0);
			else
                return static_cast<unsigned int>(this->pdbIndex->Count());
		}

        /**
        * Get the pdb amino-acid index.
        *
        * @return The pdb amino-acid index.
        */
        inline vislib::StringA GetPDBAminoAcidIndex(unsigned int i) const {
            if (!this->pdbIndex)
                return static_cast<vislib::StringA>("");
            else if (this->pdbIndex->Count() <= i)
                return static_cast<vislib::StringA>("");
            else
                return (this->pdbIndex->operator[](i));
        }

        /**
        * Get the amino-acid chain id.
        *
        * @return The amino-acid chain id.
        */
        inline char GetChainID(unsigned int i) const {
            if (!this->chainID)
                return static_cast<char>(' ');
            else if (this->chainID->Count() <= i)
                return static_cast<char>(' ');
            else
                return (this->chainID->operator[](i));
        }

        /**
        * Get the amino-acid in three letter code.
        *
        * @return The amino-acid three letter code.
        */
        inline vislib::StringA GetAminoAcid(unsigned int i) const {
            if (!this->aminoAcidName)
                return static_cast<vislib::StringA>("");
            else if (this->aminoAcidName->Count() <= i)
                return static_cast<vislib::StringA>("");
            else
                return (this->aminoAcidName->operator[](i));
        }

        /**
        * Get the missing amino-acid flag.
        *
        * @return The missing amino-acid flag.
        */
        inline addFlags GetResidueFlag(unsigned int i) const {
            if (!this->residueFlag)
                return addFlags::NOTHING;
            else if (this->residueFlag->Count() <= i)
                return addFlags::NOTHING;
            else
                return (this->residueFlag->operator[](i));
        }

        /**
        * Get the type of the secondary structure for an given method.
        *
        * @param m The secondary structure assignment method.
        * @param i The index of the amino-acid.
        * @return The DSSP secondary structure type.
        */
        inline secStructure GetSecStructure(assMethod m, unsigned int i) const {
            if (!this->secStructAssignment)
                return NOTDEFINED;
            else if (this->secStructAssignment->Count() <= m)
                return NOTDEFINED;
            else if (this->secStructAssignment->operator[](m).Count() <= i)
                return NOTDEFINED;                
            else
                return (this->secStructAssignment->operator[](m)[i]);
        }

        /**
        * Get the information of the secondary structure uncertainty.
        *
        * @param i The index of the amino-acid.
        * @return The array of secondary structure uncertainty.
        */
        inline vislib::math::Vector<float, static_cast<unsigned int>(secStructure::NOE)> GetSecStructUncertainty(unsigned int i) const {
            vislib::math::Vector<float, static_cast<unsigned int>(secStructure::NOE)> default;
            for (int x = 0; x < static_cast<int>(secStructure::NOE); x++) {
                default[x] = 0.0f;
            }
            if (!this->secStructUncertainty)
                return default;
            else if (this->secStructUncertainty->Count() <= i)
                return default;
            else
                return (this->secStructUncertainty->operator[](i));
        }

        /**
        * Get the sorted secondary structure types.
        *
        * @param i The index of the amino-acid.
        * @return The sorted secondary strucutre types.
        */
        inline vislib::math::Vector<secStructure, static_cast<int>(secStructure::NOE)> GetSortedSecStructureIndices(unsigned int i) const {
            vislib::math::Vector<secStructure, static_cast<unsigned int>(secStructure::NOE)> default;
            for (int x = 0; x < static_cast<int>(secStructure::NOE); x++) {
                default[x] = static_cast<secStructure>(x);
            }
            if (!this->sortedSecStructUncertainty)
                return default;
            else if (this->sortedSecStructUncertainty->Count() <= i)
                return default;
            else
                return (this->sortedSecStructUncertainty->operator[](i));
        }

        /**
        * Get the PDB ID.
        *
        * @return The pdb id.
        */
        inline vislib::StringA GetPdbID(void) {
            return *this->pdbID;
        }  

        /**
        * Get the flag indicating if uncertainty has been recalculated.
        *
        * @return The unceratinty recalculation flag.
        */
        inline bool GetRecalcFlag(void) {
            return this->recalcUncertainty;
        }
        
        /**
         * Get the color to the corresponding secondary structure type.
         *
         * @param s The secondary structure type.
         * @return The color for the given secondary structure type.
         */
        vislib::StringA GetSecStructDesc(UncertaintyDataCall::secStructure s);
            
        /**
        * Get the description to the corresponding secondary structure type.
        *
        * @param s The secondary structure type.
        * @return The description for the given secondary structure type.
        * 
		* 
		* Source: https://wiki.selfhtml.org/wiki/Grafik/Farbpaletten
        */
        vislib::math::Vector<float, 4> GetSecStructColor(UncertaintyDataCall::secStructure s);            
        
        /**
        * Get the pdb assignment method for helix.
        *
        * @return The pdb assignment method for helix.
        */
        inline UncertaintyDataCall::pdbAssMethod GetPdbAssMethodHelix(void) {
            return *this->pdbAssignmentHelix;
        }
        
        /**
        * Get the pdb assignment method for sheet.
        *
        * @return The pdb assignment method for sheet.
        */
        inline UncertaintyDataCall::pdbAssMethod GetPdbAssMethodSheet(void) {
            return *this->pdbAssignmentSheet;
        }        
                
        // ------------------ SET functions ------------------- 

        /**
        * Set the pointer to the secondary structure type.
        *
        * @param rnPtr The pointer.
        */
        inline void SetSecStructure(vislib::Array<vislib::Array<secStructure> > *rnPtr) {
            this->secStructAssignment = rnPtr;
        }

        /**
        * Set the pointer to the pdb index.
        *
        * @param rnPtr The pointer.
        */
        inline void SetPdbIndex(vislib::Array<vislib::StringA> *rnPtr) {
            this->pdbIndex = rnPtr;
        }

        /**
        * Set the pointer to the 3-letter amino-acid name.
        *
        * @param rnPtr The pointer.
        */
        inline void SetAminoAcidName(vislib::Array<vislib::StringA> *rnPtr) {
            this->aminoAcidName = rnPtr;
        }

        /**
        * Set the pointer to the chain ID.
        *
        * @param rnPtr The pointer.
        */
        inline void SetChainID(vislib::Array<char> *rnPtr) {
            this->chainID = rnPtr;
        }

        /**
        * Set the pointer to the residue flag.
        *
        * @param rnPtr The pointer.
        */
        inline void SetResidueFlag(vislib::Array<addFlags> *rnPtr) {
            this->residueFlag = rnPtr;
        }

        /**
        * Set the pointer to the secondary structure uncertainty.
        *
        * @param rnPtr The pointer.
        */
        inline void SetSecStructUncertainty(vislib::Array<vislib::math::Vector<float, static_cast<int>(secStructure::NOE)> > *rnPtr) {
            this->secStructUncertainty = rnPtr;
        }

        /**
        * Set the pointer to the sorted secondary structure types.
        *
        * @param rnPtr The pointer.
        */
        inline void SetSortedSecStructTypes(vislib::Array<vislib::math::Vector<secStructure, static_cast<int>(secStructure::NOE)> > *rnPtr) {
            this->sortedSecStructUncertainty = rnPtr;
        }
        
        /**
        * Set the PDB ID.
        *
        * @param rnPtr The pointer to the pdb id.
        */
		inline void SetPdbID(vislib::StringA *rnPtr) {
			this->pdbID = rnPtr;
        }
        
        /**
        * Set the flag indicating if uncertainty has been recalculated.
        *
        * @param flag The unceratinty recalculation flag.
        */
		inline void SetRecalcFlag(bool rnData) {
			this->recalcUncertainty = rnData;
        }

        /**
        * Set the pdb assignment method for helix.
        *
        * @param rnPtr The pointer to the pdb assignment method for helix.
        */
		inline void SetPdbAssMethodHelix(UncertaintyDataCall::pdbAssMethod *rnPtr) {
			this->pdbAssignmentHelix = rnPtr;
        }
        
       /**
        * Set the pdb assignment method for sheet.
        *
        * @param rnPtr The pointer to the pdb assignment method for sheet.
        */
		inline void SetPdbAssMethodSheet(UncertaintyDataCall::pdbAssMethod *rnPtr) {
			this->pdbAssignmentSheet = rnPtr;
        }  
        

	private:

        // ------------------ variables ------------------- 

        /** Pointer to the secondary structure assignment methods and their secondary structure type assignments */
        vislib::Array<vislib::Array<secStructure> > *secStructAssignment;
        
        /** Pointer to the PDB index */
        vislib::Array<vislib::StringA> *pdbIndex;

        /** Pointer to the chain ID */
        vislib::Array<char> *chainID;

        /** Pointer to the flag giving additional information */
        vislib::Array<addFlags> *residueFlag;

        /** Pointer to the amino-acid name */
        vislib::Array<vislib::StringA> *aminoAcidName;

        /** Pointer to the values of the secondary structure uncertainty for each amino-acid */
        vislib::Array<vislib::math::Vector<float, static_cast<int>(secStructure::NOE)> > *secStructUncertainty;

        /** Pointer to the sorted structure types of the uncertainty values */
        vislib::Array<vislib::math::Vector<secStructure, static_cast<int>(secStructure::NOE)> > *sortedSecStructUncertainty;
        
        /** Flag indicating that uncertainty was recalculated */
        bool recalcUncertainty;
                
        /** The PDB ID */
        vislib::StringA *pdbID;
        
        /** The pdb assignment method for helix */
        UncertaintyDataCall::pdbAssMethod *pdbAssignmentHelix;            
        
        /** The pdb assignment method for helix */
        UncertaintyDataCall::pdbAssMethod *pdbAssignmentSheet; 
                    
    };

    /** Description class typedef */
    typedef megamol::core::factories::CallAutoDescription<UncertaintyDataCall> UncertaintyDataCallDescription;
    
	} /* end namespace protein_uncertainty */
} /* end namespace megamol */

#endif /* MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYDATACALL_H_INCLUDED */
