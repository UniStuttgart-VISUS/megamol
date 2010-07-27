/*
 * ProteinData.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MMPROTEINPLUGIN_PROTEINDATA_H_INCLUDED
#define MMPROTEINPLUGIN_PROTEINDATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallProteinData.h"
#include "Module.h"
#include "CalleeSlot.h"
#include "param/ParamSlot.h"
#include "vislib/IllegalParamException.h"
#include "vislib/Vector.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/memutils.h"
#include <vector>
#include <map>
#include "Stride.h"

namespace megamol {
namespace protein {

    /**
	 * Data source for PDB files
	 *
	 * TODO:
	 * - load and connect solvent atoms / molecules
	 * - gaps in amino acid chains are not handled yet
	 *   --> insert amino acids of type 'unknown' with zero atoms as placeholders
	 */

	class ProteinData : public megamol::core::Module
	{
	public:
		
		/** Ctor */
		ProteinData(void);

		/** Dtor */
		virtual ~ProteinData(void);

		/**
		 * Answer the name of this module.
		 *
		 * @return The name of this module.
		 */
		static const char *ClassName(void) 
		{
			return "ProteinData";
		}

		/**
		 * Answer a human readable description of this module.
		 *
		 * @return A human readable description of this module.
		 */
		static const char *Description(void) 
		{
			return "Offers protein data.";
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
		virtual void release(void) {};

	private:

		/**
		 * ProtData callback.
		 *
		 * @param call The calling call.
		 *
		 * @return 'true' on success, 'false' otherwise.
		 */
		bool ProtDataCallback( megamol::core::Call& call);

        /**
         * Call callback to get the extent of the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getExtent( megamol::core::Call& call);

		/**********************************************************************
		 * 'protein'-functions
		 **********************************************************************/

		/**
		 * Parse a string containing an ATOM entry of a PDB file.
		 * The obtained data is stored in the appropriate containers.
		 * 
		 * @param line The string containing the ATOM entry.
		 *
		 * @return 'true' if the string could be parsed correctly, 'false' otherwise
		 */
		bool ParsePDBAtom( const vislib::StringA &line );

		/**
		 * Parse a string containing an HETATM entry of a PDB file.
		 * The obtained data is stored in the appropriate containers.
		 * 
		 * @param line The string containing the HETATM entry.
		 *
		 * @return 'true' if the string could be parsed correctly, 'false' otherwise
		 */
		bool ParsePDBHetatm( const vislib::StringA &line );
		
		/**
		 * Parse a string containing an CONECT entry of a PDB file.
		 * The obtained data is stored in the appropriate containers.
		 * 
		 * @param line The string containing the CONECT entry.
		 *
		 * @return 'true' if the string could be parsed correctly, 'false' otherwise
		 */
		bool ParsePDBConect( const vislib::StringA &line );

		/**
		 * Make and store the connections for the atoms of the amino acid with the 
		 * residue sequence number 'resSeq' in the chain 'chainId'.
		 *
		 * @param chainId The ID string of the chain containing the amino acid.
		 * @param resSeq The residue sequence number string of the amino acid.
		 *
		 * @return 'true' if connections could be made for the desired amino acid, 'false' otherwise
		 */
		bool MakeConnections( unsigned int chainIdIdx, unsigned int resSeqIdx );

		/**
		 * Try to read the secondary structure information from STRIDE file.
		 * 
		 * @return 'true' if STRIDE file could be read, 'false' otherwise
		 */
		bool ReadSecondaryStructure();

		/**
		 * Compute bounding box from atomic positions
		 */
		void ComputeBoundingBox();

		/**
		 * Check for possible disulfide bonds.
			*
		 * Disulfide bonds can be assumed to form between sulfur atoms of 
		 * cysteines which are nearer than 3.0 Anstrom to each other.
		 * This check is highly unexact and does not necessarily produce
		 * correct results. For a better estimation, the angles between
		 * the sulfurs should be taken account of, too.
		 */
		void EstimateDisulfideBonds();

		/**
		 * Returns the atomic number of an element from the periodic table.
		 *
		 * @param symbol The symbol of an element.
		 *
		 * @return The atomic number of the symbol or 0, if no valid symbol was given.
		 */
		unsigned int GetAtomicNumber( const char* symbol) const;

		/**
		 * Returns the index of an amino acid in the amino acid name array.
		 *
		 * @param name The name of the amino acid (in 3 character code, e.g. "CYS" for cystein).
		 *
		 * @return The index of the amino acid name or 0, if no valid name was given.
		 */
		unsigned int GetAminoAcidNameIdx( const char* name) const;

		/** Clear and reset all data structures */
		void ClearData();

		/** Add all elements from the periodic table of elements to atomTypes-array */
		void FillAtomTypesWithPeriodicTable();

		/** Add all amino acids to the aminoAcidNames array */
		void FillAminoAcidNames();

		/** compare two strings */
		struct strCmp
		{
			bool operator()( const char* s1, const char* s2 ) const
			{
				return strcmp( s1, s2 ) < 0;
			}
		};

		/**
		 * Tries to load the file 'm_filename' into memory
         * 
		 * @ param rturn 'true' if file(s) could be loaded, 'false' otherwise 
		 */
		bool tryLoadFile(void);

		/**********************************************************************
		 * variables
		 **********************************************************************/

		/** Callee slot */
		megamol::core::CalleeSlot m_protDataCalleeSlot;

        /** The filename parameter */
        megamol::core::param::ParamSlot  m_filename;

        /** The data hash */
        SIZE_T datahash;

		////////////////////////////////////////////////////////////
		// variables for storing all data needed by the interface //
		////////////////////////////////////////////////////////////

		vislib::Array<vislib::StringA>                               m_aminoAcidNames;
        vislib::Array<CallProteinData::AtomType>                     m_atomTypes;
		vislib::Array<vislib::Array<CallProteinData::AminoAcid> >    m_aminoAcidChains;
		vislib::Array<vislib::Array<CallProteinData::SecStructure> > m_secStruct;
		vislib::Array<CallProteinData::IndexPair>                    m_dsBonds;
		vislib::Array<CallProteinData::AtomData>                     m_protAtomData;
		vislib::Array<float>                                         m_protAtomPos;

		// bounding box
		float m_minX, m_minY, m_minZ, m_maxX, m_maxY, m_maxZ, m_maxDimension;

		// minimmum and maximum temperature factor value
		float m_minTempFactor, m_maxTempFactor;
		// minumum and maximum occupancy
		float m_minOccupancy, m_maxOccupancy;
		// minumum and maximum charge
		float m_minCharge, m_maxCharge;

		/////////////////////////////////////////////////////////////////
		// temporary variables for storing and processing the PDB data //
		/////////////////////////////////////////////////////////////////

		// current atom index
		unsigned int tmp_currentAtomIdx;

		// all ATOM-entries from the PDB-file:
		// -> chain
		// --> amino acid
		// ---> atom
		std::vector<std::vector<std::vector<vislib::StringA> > > tmp_atomEntries;

		// get the array-index of the chainId
		std::map<char, unsigned int> tmp_chainIdMap;

		// get the array-index of the chainId
		vislib::Array<std::map<const char*, unsigned int, strCmp> > tmp_resSeqMap;

		std::map<const char*, unsigned int, strCmp> tmp_atomicNumbers;
		std::map<const char*, unsigned int, strCmp> tmp_aminoAcidNameIdx;
		vislib::Array<CallProteinData::AtomType> tmp_atomTypes;

		vislib::Array<unsigned int> tmp_cysteineSulfurAtoms;
		
		/** Stride secondary structure */
		Stride *m_stride;
		/** Stride sec struct computed */
		bool m_secondaryStructureComputed;
	};


} /* end namespace protein */
} /* end namespace megamol */

#endif //MMPROTEINPLUGIN_PROTEINDATA_H_INCLUDED
