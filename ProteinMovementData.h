/*
 * ProteinMovementData.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PROTEINMOVEMENTDATA_H_INCLUDED
#define MEGAMOLCORE_PROTEINMOVEMENTDATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallProteinMovementData.h"
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
     * Data source for PDB files containing atomic movement.
     * Reads two PDB files with potentially different atomic positions.
     */

    class ProteinMovementData : public megamol::core::Module
    {
    public:

        /** Ctor */
        ProteinMovementData ( void );

        /** Dtor */
        virtual ~ProteinMovementData ( void );

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName ( void )
        {
            return "ProteinMovementData";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description ( void )
        {
            return "Offers protein data.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable ( void )
        {
            return true;
        }

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create ( void );

        /**
         * Implementation of 'Release'.
         */
        virtual void release ( void ) {};

    private:

        /**
         * ProtData callback.
         *
         * @param call The calling call.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        bool ProtDataCallback ( megamol::core::Call& call );

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
        bool ParsePDBAtom ( const vislib::StringA &line );

        /**
         * Make and store the connections for the atoms of the amino acid with the
         * residue sequence number 'resSeq' in the chain 'chainId'.
         *
         * @param chainId The ID string of the chain containing the amino acid.
         * @param resSeq The residue sequence number string of the amino acid.
         *
         * @return 'true' if connections could be made for the desired amino acid, 'false' otherwise
         */
        bool MakeConnections ( unsigned int chainIdIdx, unsigned int resSeqIdx );

        /**
         * Compute bounding box from atomic positions
         */
        void ComputeBoundingBox();

        /**
         * Returns the atomic number of an element from the periodic table.
         *
         * @param symbol The symbol of an element.
         *
         * @return The atomic number of the symbol or 0, if no valid symbol was given.
         */
        unsigned int GetAtomicNumber ( const char* symbol ) const;

        /**
         * Returns the index of an amino acid in the amino acid name array.
         *
         * @param name The name of the amino acid (in 3 character code, e.g. "CYS" for cystein).
         *
         * @return The index of the amino acid name or 0, if no valid name was given.
         */
        unsigned int GetAminoAcidNameIdx ( const char* name ) const;

        /** Clear and reset all data structures */
        void ClearData();

        /** Add all elements from the periodic table of elements to atomTypes-array */
        void FillAtomTypesWithPeriodicTable();

        /** Add all amino acids to the aminoAcidNames array */
        void FillAminoAcidNames();

        /** compare two strings */
        struct strCmp
        {
            bool operator() ( const char* s1, const char* s2 ) const
            {
                return strcmp ( s1, s2 ) < 0;
            }
        };

        /**
         * Tries to load the file 'filename' into memory
         *
         * @ param rturn 'true' if file(s) could be loaded, 'false' otherwise
         */
        bool tryLoadFile ( void );

        //**********************************************************************
        //    variables
        //**********************************************************************

        /** Callee slot */
        megamol::core::CalleeSlot protDataCalleeSlot;

        /** The filename parameter for the main file */
        megamol::core::param::ParamSlot  mainFilename;

        /** The filename parameter for the differences */
        megamol::core::param::ParamSlot  diffFilename;

        ////////////////////////////////////////////////////////////
        // variables for storing all data needed by the interface //
        ////////////////////////////////////////////////////////////

        vislib::Array<vislib::StringA>                                       aminoAcidNames;
        vislib::Array<CallProteinMovementData::AtomType>                     atomTypes;
        vislib::Array<vislib::Array<CallProteinMovementData::AminoAcid> >    aminoAcidChains;
        vislib::Array<vislib::Array<CallProteinMovementData::SecStructure> > secStruct;
        vislib::Array<CallProteinMovementData::AtomData>                     protAtomData;
        vislib::Array<float>                                                 protAtomPos;
        vislib::Array<float>                                                 protAtomMovedPos;

        // bounding box
        float minX, minY, minZ, maxX, maxY, maxZ, maxDimension;

        // maximum movement length
        float maxMovementDist;

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
        vislib::Array<CallProteinMovementData::AtomType> tmp_atomTypes;
        
        /** Stride secondary structure */
        Stride *stride;
        /** Stride sec struct computed */
        bool secondaryStructureComputed;
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif //MEGAMOLCORE_PROTEINMOVEMENTDATA_H_INCLUDED
