/*
 * CallProteinData.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLPROTEINDATA_H_INCLUDED
#define MEGAMOLCORE_CALLPROTEINDATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "AbstractGetData3DCall.h"
#include "CallAutoDescription.h"
#include "vislib/IllegalParamException.h"
#include "vislib/Pair.h"
#include "vislib/Cuboid.h"
#include "vislib/Vector.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include <vector>

#ifdef _WIN32
#define DEPRECATED __declspec(deprecated)
#else /* _WIN32 */
#define DEPRECATED
#endif /* _WIN32 */

namespace megamol {
namespace protein {

    /**
     * Base class of rendering graph calls and
     * of data interfaces for protein and protein-solvent data.
     *
     * Note that all data has to be sorted! The atoms of chains of the protein
     * are sorted by their amino acids in the order they appear along the
     * chains. Solvent atoms are sorted as molecules sorted by their types in
     * the same order as the types are present in the solvent molecule type
     * table. There are no IDs anymore, always use the index values into the
     * right tables.
     *
     * All data has to be stored in the corresponding data source object. This
     * interface object will only pass pointers to the renderer objects. A
     * compatible data source should therefore use the public nested class of
     * this interface to store the data, so that easy transfer is guanateed.
     * An exception is given for the structural data. The 'Chain' data is
     * considdered to be static (except for the secondary structure data) so
     * it is possible to store this data in the data interface. However, it is
     * not possible to store atom positions inside the interface and it is not
     * recommended to store the secondary structure inside the interface if
     * this structure might change over time!
     */

	class CallProteinData : public megamol::core::AbstractGetData3DCall {
    public:

        /** Index of the 'GetData' function */
        static const unsigned int CallForGetData;

        /** Index of the 'GetExtent' function */
        static const unsigned int CallForGetExtent;

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallProteinData";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get protein data";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 2;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch( idx) {
                case 0:
            return "GetData";
                case 1:
                    return "GetExtend";
            }
        }

        /** name alias for index pairs */
        typedef vislib::Pair<unsigned int, unsigned int> IndexPair;

        /* forward declaration */
        class SecStructure;

        /**
         * Nested class describing one amino acid
         */
        class AminoAcid {
        public:

            /** Default ctor initialising all fields to zero. */
            AminoAcid(void);

            /**
             * Copy ctor performin a deep copy from 'src'.
             *
             * @param src The object to clone from.
             */
            AminoAcid(const AminoAcid& src);

            /**
             * Ctor.
             *
             * @param firstAtomIdx The index of the first atom of this amino
             *                     acid in this chain.
             * @param atomCnt The size of the protein in number of atoms.
             * @param cAlphaIdx The index to be set as index for the c alpha
             *                  atom.
             * @param cCarbIdx The index to be set as index for the c atom of
             *                 the carboxyl.
             * @param oIdx The index to be set as index for the o atom.
             * @param nIdx The index to be set as index for the n of the amino
             *             acid.
             * @param nameIdx The index of the name of the amino acid.
             */
            AminoAcid(unsigned int firstAtomIdx, unsigned int atomCnt,
                unsigned int cAlphaIdx, unsigned int cCarbIdx,
                unsigned int nIdx, unsigned int oIdx, unsigned int nameIdx);

            /** Dtor. */
            ~AminoAcid(void);

            /**
             * Returns a writeable reference to the array of connections
             * forming the connectivity information of the amino acid. This
             * method is used to access this array for filling it with data.
             */
            inline vislib::Array<IndexPair>& AccessConnectivity(void) {
                return this->connectivity;
            }

            /**
             * Returns the number of atoms.
             *
             * @return The number of atoms.
             */
            inline unsigned int AtomCount(void) const {
                return this->atomCnt;
            }

            /**
             * Returns the index of the c alpha atom.
             *
             * @return The index of the c alpha atom.
             */
            inline unsigned int CAlphaIndex(void) const {
                return this->cAlphaIdx;
            }

            /**
             * Returns the index of the c atom of the carboxyl.
             *
             * @return The index of the c atom of the carboxyl.
             */
            inline unsigned int CCarbIndex(void) const {
                return this->cCarbIdx;
            }

            /**
             * Returns the connectivity information of the amino acid. Atom
             * indices are relative to the amino acid, starting from zero.
             *
             * @return The connectivity information of the amino acid.
             */
            inline const vislib::Array<IndexPair>& Connectivity(void) const {
                return this->connectivity;
            }

            /**
             * Returns the index of the first atom in the chain of the protein.
             *
             * @return The index of the first atom in the chain of the protein.
             */
            inline unsigned int FirstAtomIndex(void) const {
                return this->firstAtomIdx;
            }

            /**
             * Returns the index of the name in the amino acid name table.
             *
             * @return The index of the name in the amino acid name table.
             */
            inline unsigned int NameIndex(void) const {
                return this->nameIdx;
            }

            /**
             * Returns the index of the n atom.
             *
             * @return The index of the n atom.
             */
            inline unsigned int NIndex(void) const {
                return this->nIdx;
            }

            /**
             * Returns the index of the o atom.
             *
             * @return The index of the o atom.
             */
            inline unsigned int OIndex(void) const {
                return this->oIdx;
            }

            /**
             * Sets the index of the c alpha atom.
             *
             * @param idx The index to be set as index for the c alpha atom.
             */
            void SetCAlphaIndex(unsigned int idx);

            /**
             * Sets the index of the c atom of the carboxyl.
             *
             * @param idx The index to be set as index for the c atom of the
             *            carboxyl.
             */
            void SetCCarbIndex(unsigned int idx);

            /**
             * Sets the index of the n atom.
             *
             * @param idx The index to be set as index for the n atom.
             */
            void SetNIndex(unsigned int idx);

            /**
             * Sets the index of the o atom.
             *
             * @param idx The index to be set as index for the o atom.
             */
            void SetOIndex(unsigned int idx);

            /**
             * Sets the position of the amino acid inside the chain by
             * specifying the first atom index and the size in number of
             * atoms.
             *
             * @param firstAtom The index of the first atom of this amino acid
             *                  in this chain.
             * @param atomCnt The size of the protein in number of atoms.
             */
            void SetPosition(unsigned int firstAtom, unsigned int atomCnt);

            /**
             * Sets the index of the name of the amino acid in the amino acid
             * name table.
             *
             * @param idx The index of the name of the amino acid.
             */
            void SetNameIndex(unsigned int idx);

            /**
             * Copy operator performs a deep copy of the amino acid object.
             *
             * @param rhs The right hand side operand to clone from.
             *
             * @return A reference to this object.
             */
            AminoAcid& operator=(const AminoAcid& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'this' and 'rhs' represent the same amino
             *         acid, 'false' otherwise.
             */
            bool operator==(const AminoAcid& rhs) const;

            /**
             * Test for inequality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'false' if 'this' and 'rhs' represent the same amino
             *         acid, 'true' otherwise.
             */
            inline bool operator!=(const AminoAcid& rhs) const {
                return !(*this == rhs);
            }

        private:

            /** The size of this amino acid in atoms */
            unsigned int atomCnt;

            /** The index of the c alpha atom in the chain of the protein */
            unsigned int cAlphaIdx;

            /**
             * The index of the c atom of the carboxyl in the chain of the
             * protein.
             */
            unsigned int cCarbIdx;

            /** 
             * The connectivity of the atoms in this amino acid. The index of
             * the atoms is relative to the amino acid, starting from zero.
             */
            vislib::Array<IndexPair> connectivity;

            /**
             * The index of the first atom of this amino acid in the chain of
             * the protein.
             */
            unsigned int firstAtomIdx;

            /** Index of the amino acids name in the name index table. */
            unsigned int nameIdx;

            /** The index of the n atom in the chain of the protein */
            unsigned int nIdx;

            /** The index of the o atom in the chain of the protein */
            unsigned int oIdx;

        };

        /**
         * Nested class holding all information about one atom.
         */
        class AtomData {
        public:

            /**
             * Ctor.
             *
             * @param typeIdx    The index of the atom type.
             * @param charge     The charge of the atom.
             * @param tempFactor The temperature factor of the atom.
             * @param occupancy  The occupancy of the atom.
             */
            AtomData(unsigned int typeIdx = 0, float charge = 0.0f, 
                float tempFactor = 0.0f, float occupancy = 0.0f);

            /**
             * Copy ctor.
             *
             * @param src The source object to clone from
             */
            AtomData(const AtomData& src);

            /** Dtor. */
            ~AtomData(void);

            /**
             * Returns the charge of the atom.
             *
             * @return The charge of the atom.
             */
            inline float Charge(void) const {
                return this->charge;
            }

            /**
             * Returns the occupancy of the atom.
             *
             * @return The occupancy of the atom.
             */
            inline float Occupancy(void) const {
                return this->occupancy;
            }

            /**
             * Sets the charge of the atom.
             *
             * @param charge The new charge of the atom.
             */
            inline void SetCharge(float charge) {
                this->charge = charge;
            }

            /**
             * Sets the occupancy of the atom.
             *
             * @param occupancy The new occupancy of the atom.
             */
            inline void SetOccupancy(float occ) {
                this->occupancy = occ;
            }

            /**
             * Sets the temperature factor of the atom.
             *
             * @param tempFactor The new temperature factor of the atom.
             */
            inline void SetTempFactor(float tempFactor) {
                this->tempFactor = tempFactor;
            }

            /**
             * Sets the index of the atom type.
             *
             * @param idx The index of the new atom type.
             */
            inline void SetTypeIndex(unsigned int idx) {
                this->typeIdx = idx;
            }

            /**
             * Returns the temperature factor of the atom.
             *
             * @return The temperature factor of the atom.
             */
            inline float TempFactor(void) const {
                return this->tempFactor;
            }

            /**
             * Returns the index of the atom type.
             *
             * @return The index of the atom type.
             */
            inline unsigned int TypeIndex(void) const {
                return this->typeIdx;
            }

            /**
             * Assignment operator.
             *
             * @param rhs The right hand side operand.
             *
             * @return The reference to 'this'
             */
            AtomData& operator=(const AtomData& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'this' and 'rhs' are equal, 'false' otherwise.
             */
            bool operator==(const AtomData& rhs) const;

        private:

            /** The charge of the atom */
            float charge;

            /** The occupancy of the atom */
            float occupancy;

            /** 
             * The temperature factor of the atom (also called B-factor or 
             * Debye-Waller factor DWF).
             */
            float tempFactor;

            /** Index of the atom type */
            unsigned int typeIdx;

        };

        /**
         * Nested class holding all information about one atom type.
         */
        class AtomType {
        public:

            /**
             * Default ctor
             * Produces an atom type with empty name, rad = 0.5 and 75% gray
             * colour.
             */
            AtomType(void);

            /**
             * Ctor.
             *
             * @param name The name of the atom type.
             * @param rad  The radius of the atoms representing sphere.
             * @param colR The red colour component for this atom type.
             * @param colG The green colour component for this atom type.
             * @param colB The blue colour component for this atom type.
             */
            AtomType(const vislib::StringA& name, float rad = 0.5f,
                unsigned char colR = 191, unsigned char colG = 191,
                unsigned char colB = 191);

            /**
             * Copy ctor.
             *
             * @param src The object to clone from
             */
            AtomType(const AtomType& src);

            /** Dtor. */
            ~AtomType(void);

            /**
             * Sets the colour for atoms of this type.
             *
             * @param red   The red colour component.
             * @param green The green colour component.
             * @param blue  The blue colour component.
             */
            inline void SetColour(unsigned char red, unsigned char green,
                    unsigned char blue) {
                this->col[0] = red;
                this->col[1] = green;
                this->col[2] = blue;
            }

            /**
             * Sets the name of the atom type.
             *
             * @param name The name for this atom type.
             */
            inline void SetName(const vislib::StringA& name) {
                this->name = name;
            }

            /**
             * Sets the radius of the atoms of this type.
             *
             * @param rad The radius for atoms of this type.
             */
            inline void SetRadius(float rad) {
                this->rad = rad;
            }

            /**
             * Returns the colour for atoms of this type.
             *
             * @return The colour for atoms of this type.
             */
            inline const unsigned char * Colour(void) const {
                return this->col;
            }

            /**
             * Returns the name of the atom type.
             *
             * @return The name of the atom type.
             */
            inline const vislib::StringA& Name(void) const {
                return this->name;
            }

            /**
             * Returns the radius for atoms of this type.
             *
             * @return The radius for atoms of this type.
             */
            inline float Radius(void) const {
                return this->rad;
            }

            /**
             * Assignment operator.
             *
             * @param rhs The right hand side operand.
             *
             * @return The reference to 'this'
             */
            AtomType& operator=(const AtomType& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'this' and 'rhs' are equal, 'false'
             *         otherwise.
             */
            bool operator==(const AtomType& rhs) const;

        private:

            /** The colour for atoms of this type. */
            unsigned char col[3];

            /** The name of the atom type. */
            vislib::StringA name;

            /** The radius of atoms of this type. */
            float rad;

        };

        /**
         * Nested class of one protein chain.
         */
        class Chain {
        public:

            /** Default ctor. */
            Chain(void);

            /**
             * Copy ctor.
             * If 'src' owns the memory of the data arrays, these arrays will
             * be deep copied. If not, only the pointers will be copied.
             *
             * @param src The source object to clone from.
             */
            Chain(const Chain& rhs);

            /** Dtor. */
            ~Chain(void);

            /**
             * Returns a non-const pointer to the 'idx'-th amino acids of this
             * chain. This method is ment for writing the data if this object
             * is used to store the chain data (after calling
             * 'SetAminoAcidCount').
             *
             * @param idx The index if the amino acid object to be returned.
             *
             * @return A pointer to the amino acids requested.
             */
            CallProteinData::AminoAcid& AccessAminoAcid(
                unsigned int idx);

            /**
             * Returns a non-const pointer to the 'idx'-th secondary structure
             * object of this chain. This method is ment for writing the data
             * if this object is used to store the chain data (after calling
             * 'SetSecondaryStructureCount').
             *
             * @param idx The index if the secondary structure object to be
             *            returned.
             *
             * @return A pointer to the secondary structure requested.
             */
            CallProteinData::SecStructure& AccessSecondaryStructure(
                unsigned int idx);

            /**
             * Returns a pointer to the amino acids of this chain.
             *
             * @return A pointer to the amino acids of this chain.
             */
            inline const CallProteinData::AminoAcid * AminoAcid(void) 
                    const {
                return this->aminoAcid;
            }

            /**
             * Returns the number of amino acids in this chain.
             *
             * @return The number of amino acids in this chain.
             */
            inline unsigned int AminoAcidCount(void) const {
                return this->aminoAcidCnt;
            }

            /**
             * Returns a pointer to the seconardy structure of this chain.
             *
             * @return A pointer to the seconardy structure of this chain.
             */
            inline const SecStructure * SecondaryStructure(void) const {
                return this->secStruct;
            }

            /**
             * Returns the number of secondary structure elements in this
             * chain.
             *
             * @return The number of secondary structure elements in this
             *         chain.
             */
            inline unsigned int SecondaryStructureCount(void) const {
                return this->secStructCnt;
            }

            /**
             * Sets the amino acid data of this chain. Does not copy the data.
             * The caller remains owner of the memory of the pointer
             * 'aminoAcids' and must ensure that it remains valid as long as
             * it used by this interface.
             *
             * @param cnt The number of amino acids in the array 'aminoAcids'
             *            points to.
             * @param aminoAcids Pointer to an array of amino acid objects to
             *                   be set as amino acids of this chain.
             */
            void SetAminoAcid(unsigned int cnt,
                const CallProteinData::AminoAcid* aminoAcids);

            /**
             * Allocates memory to place 'cnt' amino acid object inside an
             * sets this as memory for the amino acids removing any previously
             * set memory.
             *
             * @param cnt The number of the amino acids in this chain.
             */
            void SetAminoAcidCount(unsigned int cnt);

            /**
             * Sets the secondary structure data of this chain. Does not copy
             * the data. The caller remains owner of the memory of the pointer
             * 'structs' and must ensure that it remains valid as long as it
             * used by this interface.
             *
             * @param cnt The number of secondary structure elements in the
             *            array 'structs' points to.
             * @param structs Pointer to an array of secondary structure
             *                objects to be set as secondary structure of this
             *                chain.
             */
            void SetSecondaryStructure(unsigned int cnt,
                const CallProteinData::SecStructure* structs);

            /**
             * Allocates memory to place 'cnt' secondary structure object
             * inside an sets this as memory for the secondary structure
             * removing any previously set memory.
             *
             * @param cnt The number of the secondary structure elements
             *            in this chain.
             */
            void SetSecondaryStructureCount(unsigned int cnt);

            /**
             * Assignment operator.
             * If 'rhs' owns the memory of the data arrays, these arrays will
             * be deep copied. If not, only the pointers will be copied.
             *
             * @param rhs The right hand side operand.
             *
             * @return A reference to 'this'.
             */
            Chain& operator=(const Chain& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if both chains (deep!) are equal,
             *         'false' otherwise.
             */
            bool operator==(const Chain& rhs) const;

        private:

            /**
             * Pointer to the amino acid objects of the amino acids of this
             * chain.
             * Must point to 'aminoAcidCnt' times objects.
             */
            CallProteinData::AminoAcid *aminoAcid;

            /** The number of amino acids in this chain */
            unsigned int aminoAcidCnt;

            /**
             * Flag indicating whether or not the memory of the amino acid
             * data is owned by this object.
             **/
            bool aminoAcidMemory;

            /**
             * Pointer to the secondary structure elements of this chain.
             * Must point to 'secStructCnt' times objects.
             */
            CallProteinData::SecStructure *secStruct;

            /** The number of secondary structure elements */
            unsigned int secStructCnt;

            /**
             * Flag indicating whether or not the memory of the secondary
             * structure data is owned by this object.
             **/
            bool secStructMemory;

        };

        /**
         * Nested class holding all information about on segment of a proteins
         * secondary structure.
         */
        class SecStructure {
        public:

            /** possible types of secondary structure elements */
            enum ElementType {
                TYPE_COIL  = 0,
                TYPE_SHEET = 1,
                TYPE_HELIX = 2,
                TYPE_TURN  = 3
            };

            /**
             * Default ctor initialising all elements to zero (or equivalent
             * values).
             */
            SecStructure(void);

            /**
             * Copy ctor performing a deep copy.
             *
             * @param src The object to clone from.
             */
            SecStructure(const SecStructure& src);

            /** Dtor. */
            ~SecStructure(void);

            /**
             * Returns the size of the element in (partial) amino acids.
             *
             * @return The size of the element in (partial) amino acids.
             */
            inline unsigned int AminoAcidCount(void) const {
                return this->aminoAcidCnt;
            }

            /**
             * Returns the size of the element in atoms.
             *
             * @return The size of the element in atoms.
             */
            inline unsigned int AtomCount(void) const {
                return this->atomCnt;
            }

            /**
             * Returns the index of the amino acid in which this element
             * starts.
             *
             * @return The index of the amino acid in which this element
             *         starts.
             */
            inline unsigned int FirstAminoAcidIndex(void) const {
                return this->firstAminoAcidIdx;
            }

            /**
             * Returns the index of the atom where this element starts.
             *
             * @return The index of the atom where this element starts.
             */
            inline unsigned int FirstAtomIndex(void) const {
                return this->firstAtomIdx;
            }

            /**
             * Sets the position of the secondary structure element by the
             * indices of the atom and amino acid where this element starts
             * and the size of the element in atoms and (partial) amino acids.
             *
             * @param firstAtomIdx The index of the atom where this element
             *                     starts.
             * @param atomCnt The size of the element in atoms.
             * @param firstAminoAcidIdx The index of the amino acid where this
             *                          element starts.
             * @param aminoAcidCnt The size of the element in (partial) amino
             *                     acids.
             */
            void SetPosition(unsigned int firstAtomIdx, unsigned int atomCnt,
                unsigned int firstAminoAcidIdx, unsigned int aminoAcidCnt);

            /**
             * Sets the type of the element.
             *
             * @param type The new type for this element.
             */
            void SetType(ElementType type);

            /**
             * Returns the type of this element.
             *
             * @return The type of this element.
             */
            inline ElementType Type(void) const {
                return this->type;
            }

            /**
             * Assignment operator.
             *
             * @param rhs The right hand side to clone from.
             *
             * @return The reference to 'this'.
             */
            SecStructure& operator=(const SecStructure& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'this' and 'rhs' are equal, 'false' if not.
             */
            bool operator==(const SecStructure& rhs) const;

            /**
             * Test for inequality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'this' and 'rhs' are not equal, 'false' if
             *         they are equal.
             */
            inline bool operator!=(const SecStructure& rhs) const {
                return !(*this == rhs);
            }

        private:

            /** The size of the element in (partial) amino acids */
            unsigned int aminoAcidCnt;

            /** The size of the element in number of atoms */
            unsigned int atomCnt;

            /** The index of the amino acid in which this element starts */
            unsigned int firstAminoAcidIdx;

            /** The index of the atom where this element starts */
            unsigned int firstAtomIdx;

            /** The type of this element */
            ElementType type;

        };


        /**
         * Nested class storing information about the molecules of the
         * solvent, such as name, atom count, and connectivity.
         */
        class SolventMoleculeData {
        public:

            /**
             * Ctor.
             */
            SolventMoleculeData(void);

            /**
             * Copy ctor. Performs a deep copy of 'src'.
             *
             * @param src The object to clone from.
             */
            SolventMoleculeData(const SolventMoleculeData& src);

            /**
             * Ctor.
             *
             * @param name The name of this solvent molecule type.
             * @param atomCnt The number of atoms in molecules of this type.
             */
            SolventMoleculeData(const vislib::StringA& name,
                unsigned int atomCnt);

            /**
             * Dtor.
             */
            ~SolventMoleculeData(void);

            /**
             * Adds a connection of two atoms to the connectivity table of
             * the molecule type.
             *
             * @param connection The pair of atom indices to connect.
             */
            void AddConnection(const IndexPair& connection);

            /**
             * Adds a connection of two atoms the the connectivity table of
             * the molecule type.
             *
             * @param idx1 The index of the first atom to connect.
             * @param idx2 The index of the second atom to connect.
             */
            inline void AddConnection(unsigned int idx1, unsigned int idx2) {
                this->AddConnection(IndexPair(idx1, idx2));
            }

            /**
             * Allocates a connectivity list of the 'cnt' elements. Allocating
             * a list and then setting the entries with 'SetConnection' should
             * be faster then using 'AddConnection'. Be aware that allocating
             * a connectivity list will delete all connectivity information
             * currently stored in the object.
             *
             * @param cnt The size of the connectivity list to be allocated.
             */
            void AllocateConnectivityList(unsigned int cnt);

            /**
             * Answers the number of atoms in each molecule of this type.
             *
             * @return The number of atoms in each molecule of this type.
             */
            inline unsigned int AtomCount(void) const {
                return this->atomCnt;
            }

            /**
             * Clears the connectivity list of the molecule type.
             */
            void ClearConnectivity(void);

            /**
             * Answers a pointer to the array of IndexPairs storing the
             * connectivity information of the molecules of this type. The
             * array has 'ConnectivityCount' entries.
             *
             * @return A pointer to the array of IndexPairs storing the
             *         connectivity information of the molecules of this type.
             */
            inline const IndexPair * Connectivity(void) const {
                return this->connectivity.PeekElements();
            }

            /**
             * Answers the number of connections in the connectivity data
             * returned by 'Connectivity'.
             *
             * @return The number of connections in the connectivity data.
             */
            unsigned int ConnectivityCount(void) const {
                return static_cast<unsigned int>(this->connectivity.Count());
            }

            /**
             * Answers the name of the molecules type.
             *
             * @return The name of the molecules type.
             */
            inline const vislib::StringA& Name(void) const {
                return this->name;
            }

            /**
             * Sets the number of atoms for the molecules of this type.
             *
             * @param cnt The number of atoms for the molecules of this type.
             */
            void SetAtomCount(unsigned int cnt);

            /**
             * Sets the 'idx'-th connection of the connectivity.
             *
             * @param idx The index of the connection to be set.
             * @param connection The pair of atom indices to connect.
             */
            void SetConnection(unsigned int idx, const IndexPair& connection);

            /**
             * Sets the 'idx'-th connection of the connectivity.
             *
             * @param idx The index of the connection to be set.
             * @param idx1 The index of the first atom to connect.
             * @param idx2 The index of the second atom to connect.
             */
            inline void SetConnection(unsigned int idx, unsigned int idx1,
                    unsigned int idx2) {
                this->SetConnection(idx, IndexPair(idx1, idx2));
            }

            /**
             * Sets the whole connectivity data by performing a deep copy from
             * an array of IndexPair objects.
             *
             * @param cnt The number of IndexPair object to be copied from
             *            'connections'.
             * @param connections The source array of IndexPair object to copy
             *                    from. Must point to 'cnt' IndexPair objects.
             */
            void SetConnections(unsigned int cnt,
                const IndexPair *connections);

            /**
             * Sets the name of the molecule type.
             *
             * @param name The new name of the molecule type.
             */
            void SetName(const vislib::StringA& name);

            /**
             * Assignment operator. Performs a deep copy from 'rhs'.
             *
             * @param rhs The right hand side operand.
             *
             * @return A reference to this.
             */
            SolventMoleculeData& operator=(const SolventMoleculeData& rhs);

            /**
             * Test for equality. This will compare all elements of the
             * connectivity data and this method is sensitive to the order the
             * connections are stored in this data.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'this' and 'rhs' are equal, 'false' otherwise.
             */
            bool operator==(const SolventMoleculeData& rhs) const;

        private:

            /** The number of atoms for this molecule type */
            unsigned int atomCnt;

            /** The name of the molecule type */
            vislib::StringA name;

            /** The connectivity of the atoms in this molecule type */
            vislib::Array<IndexPair> connectivity;

        };

        /** Ctor. */
        CallProteinData(void);

        /** Dtor. */
        virtual ~CallProteinData(void);

        /**
         * Sets the bounding box.
         *
         * @param minX minimal X value.
         * @param minY minimal Y value.
         * @param minZ minimal Z value.
         * @param maxX maximal X value.
         * @param maxY maximal Y value.
         * @param maxZ maximal Z value.
         */
        inline void SetBoundingBox(float minX, float minY, float minZ, float maxX, float maxY, float maxZ) {
            this->bbox.Set(minX, minY, minZ, maxX, maxY, maxZ);
            if (this->bbox.IsEmpty()) {
                this->bbox.Set(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
            }
        }

        /**
         * Sets the bounding box.
         *
         * @param box The bounding box cuboid.
         */
        inline void SetBoundingBox(const vislib::math::Cuboid<FLOAT> &box) {
            this->bbox = box;
            if (this->bbox.IsEmpty()) {
                this->bbox.Set(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
            }
        }

        /**
         * Answer the bounding box.
         *
         * @return the bounding box.
         */
        inline const vislib::math::Cuboid<FLOAT>& BoundingBox(void) const {
            return this->bbox;
        }

        /**
         * Sets the scaling factor of the data set.
         *
         * @param s The new scaling factor of the data set.
         */
        inline void SetScaling(float s) {
            this->scaling = s;
        }

        /**
         * Answer the scaling factor of the data set.
         *
         * @return The scaling factor of the data set.
         */
        inline float Scaling(void) const {
            return this->scaling;
        }

        /**
         * Accesses the 'idx'-th chain. This method must not be called if the
         * interface does not own the memory of the chains.
         *
         * @param idx The index of the chain to be returned.
         *
         * @return The chain object of the requested index.
         */
        Chain& AccessChain(unsigned int idx);

        /**
         * Allocates the memory to store 'cnt' AtomType object in the
         * interface object.
         *
         * @param cnt The number of AtomType to be allocated.
         */
        void AllocateAtomTypes(unsigned int cnt);

        /**
         * Allocates the memory to store 'cnt' Chain objects in the interface
         * object.
         *
         * @param cnt The number of chains to be allocated.
         */
        void AllocateChains(unsigned int cnt);

        /**
         * Allocates space for the atom data objects for the protein atoms.
         * The number of protein atoms must first be set using
         * 'SetProteinAtomCount'.
         */
        void AllocateProteinAtomData(void);

        /**
         * Allocates space for the atom positions for the protein atoms. The
         * number of protein atoms must first be set using
         * 'SetProteinAtomCount'.
         */
        void AllocateProteinAtomPositions(void);

        /**
         * Allocates space for the atom data objects for the solvent atoms.
         * The number of solvent atoms must first be set using
         * 'SetSolventAtomCount'.
         */
        void AllocateSolventAtomData(void);

        /**
         * Allocates space for the atom positions for the solvent atoms. The
         * number of solvent atoms must first be set using
         * 'SetSolventAtomCount'.
         */
        void AllocateSolventAtomPositions(void);

        /**
         * Returns the number of entries in the amino acid name table.
         *
         * @return The number of entries in the amino acid name table.
         */
        inline unsigned int AminoAcidNameCount(void) const {
            return this->aminoAcidNameCnt;
        }

        /**
         * Returns the name of the idx-th amino acid from the amino acid name
         * table.
         *
         * @param idx The indes of the amino acid which name is requested.
         *
         * @return The name of the requested amino acid.
         */
        inline const vislib::StringA& AminoAcidName(unsigned int idx) const {
            if (idx >= this->aminoAcidNameCnt) {
                throw vislib::IllegalParamException("idx", __FILE__, __LINE__);
            }
            return this->aminoAcidNames[idx];
        }

        /**
         * Returns the number of entries in the atom type table.
         *
         * @return The number of entries in the atom type table.
         */
        inline unsigned int AtomTypeCount(void) const {
            return this->atomTypeCnt;
        }

        /**
         * Returns a pointer to the array of atom type objects. The pointer
         * points to internal memory of the object. The caller must not free
         * or alter the memory returned.
         *
         * @return A pointer to the array of atom type objects.
         */
        inline const AtomType * AtomTypes(void) const {
            return this->atomTypes;
        }

        /**
         * This methods checks the integrity of the solvent molecule
         * information, by calculating the number of atoms required by the
         * solvent molecule description set using 'SetSolventMoleculeInfo' and
         * comparing this value to the number of solvent molecules set using
         * 'SetSolventAtomCount'.
         *
         * @return 'true' if the solvent molecule information seams to be
         *         correct, 'false' otherwise.
         */
        bool CheckSolventMoleculeInformation(void);

        /**
         * Returns a pointer to the array of index pairs storing the atoms
         * building up disulfid bonds. There will be 'DisulfidBondsCount'
         * elements in the array.
         *
         * @return A pointer to the array of index pairs of the disulfid bonds
         */
        inline const IndexPair * DisulfidBonds(void) const {
            return this->dsBonds;
        }

        /**
         * Returns the number of disulfid bonds.
         *
         * @return The number of disulfid bonds.
         */
        inline unsigned int DisulfidBondsCount(void) const {
            return this->dsBondsCnt;
        }

        /**
         * Returns the number of atoms in all chains of all proteins all
         * together.
         *
         * @return The number of atoms in all proteins.
         */
        inline unsigned int ProteinAtomCount(void) const {
            return this->protAtomCnt;
        }

        /**
         * Returns a pointer to the atom data of the protein atoms. Points to
         * 'ProteinAtomCount' times 'AtomData' objects holding the data. The
         * pointer points to the internal memory structure. The caller must
         * not free or alter the memory returned.
         *
         * @return A pointer to the AtomData of the protein atoms.
         */
        inline const AtomData * ProteinAtomData(void) const {
            return this->protAtomData;
        }

        /**
         * Returns a pointer to the positions of the protein atoms. Points to
         * 'ProteinAtomCount' times 3 floats holding the data layouted
         * xyzxyz... The pointer points to the internal memory structure. The
         * caller must not free or alter the memory returned.
         *
         * @return A pointer to the positions of the protein atoms.
         */
        inline const float * ProteinAtomPositions(void) const {
            return this->protAtomPos;
        }

        /**
         * Returns the number of chains of the protein.
         *
         * @return The number of chains of the protein.
         */
        inline unsigned int ProteinChainCount(void) const {
            return this->chainCnt;
        }

        /**
         * Returns the idx-th chain of the protein.
         *
         * @param idx The index of the chain requested.
         *
         * @return The requested chain.
         */
        inline const Chain& ProteinChain(unsigned int idx) const {
            if (idx >= this->chainCnt) {
                throw vislib::IllegalParamException("idx", __FILE__, __LINE__);
            }
            return this->chains[idx];
        }

        /**
         * Sets the size of the amino acid name table. This will delete ALL
         * entries of the previous amino acid name table.
         *
         * @param cnt The number of entries for the new amino acid name table.
         */
        void SetAminoAcidNameCount(unsigned int cnt);

        /**
         * Sets the name of the 'idx'-th amino acid in the amino acid name
         * table. This is only possible if the name table is not set to an
         * external pointer using 'SetAminoAcidNameTable'.
         *
         * @param idx  The index of the amino acid to set.
         * @param name The new name for the 'idx'-th amino acid.
         *
         * @throw vislib::IllegalStateException if the amino acid name table
         *        is set to an external pointer.
         */
        void SetAminoAcidName(unsigned int idx, const vislib::StringA& name);

        /**
         * Sets the amino acid name table to an external table. The data is
         * not copied, the caller remains owner of the memory, and must ensure
         * that the memory is not freed or altered while it is used by this
         * interface.
         *
         * @param cnt   The number of entries in the amino acid name table.
         * @param names Pointer to an array of 'cnt' 'vislib::StringA' objects
         *              holding the entries of the amino acid name table.
         */
        void SetAminoAcidNameTable(unsigned int cnt,
            const vislib::StringA *names);

        /**
         * Sets the atom type object for the 'idx'-th entry in the atom type
         * table. You must not call this method if the interface is not owner
         * of the atom type memory.
         *
         * @param idx The index of the atom type entry to be set.
         * @param type The type object holding the new values.
         */
        void SetAtomType(unsigned int idx, const AtomType& type);

        /**
         * Sets the atom type table to an external table. The data is not
         * copied, the caller remains owner of the memory, and must ensure
         * that the memory is not freed or altered while it is used by this
         * interface.
         *
         * @param cnt   The number of entries in the atom type table.
         * @param types Pointer to an array of 'cnt' 'AtomType' objects
         *              holding the entries of the atom type table.
         */
        void SetAtomTypeTable(unsigned int cnt, const AtomType *types);

        /**
         * Sets a pointer to an array of 'IndexPair' objects to be used
         * through this interface to access the disulfid bonds. The memory
         * pointed to will remain owned by the caller and the caller must
         * ensure it remains valid as long as it is used through this
         * interface.
         *
         * @param cnt The number of entries in the array 'bonds'.
         * @param bonds Pointer to an array of 'IndexPair' objects
         *              representing the disulfid bonds.
         */
        void SetDisulfidBondsPointer(unsigned int cnt, IndexPair *bonds);

        /**
         * Sets the number of atoms in the protein. This reverts all previous
         * calls to 'AllocateProteinAtomData' and
         * 'AllocateProteinAtomPositions', so if you plan to use memory stored
         * inside the interface, you must call these methods again.
         *
         * @param cnt The number of atoms to be in the protein.
         */
        void SetProteinAtomCount(unsigned int cnt);

        /**
         * Sets the atom data for one atom of the protein. You must not call
         * this method if the interface is not owner of the atom data objects.
         *
         * @param idx  The index of the atom to set it's data
         * @param data The data to be set for the atom.
         */
        void SetProteinAtomData(unsigned int idx, const AtomData& data);

        /**
         * Sets a pointer to an array of atom data object to be used through
         * this interface. The memory pointed to will remain owned by the
         * caller and the caller must ensure it remains valid as long as it is
         * used through this interface. The array must be of sufficient size
         * compare to the number of atoms which should be set earlier using
         * 'SetProteinAtomCount'.
         *
         * @param data Pointer to an array of atom data objects.
         */
        void SetProteinAtomDataPointer(AtomData *data);

        /**
         * Sets the position for one atom of the protein. You mut not call this
         * method if the interface is not owner of the position memory.
         *
         * @param idx The index of the atom to be set.
         * @param x   The x component of the atoms position.
         * @param y   The y component of the atoms position.
         * @param z   The z component of the atoms position.
         */
        void SetProteinAtomPosition(unsigned int idx, float x, float y,
            float z);

        /**
         * Sets a pointer to an array of floats to be used through this
         * interface of the positions of the protein atoms. The memory pointed
         * to will remain owned by the caller and the caller must ensure it
         * remains valid as long as it is* used through this interface. The
         * array must be of sufficient size compare to the number of atoms
         * which should be set earlier using 'SetProteinAtomCount'. The float
         * array must be layouted 'xyzxyz...'
         *
         * @param pos Pointer to an array of float to be used as protein atom
         *            positions.
         */
        void SetProteinAtomPositionPointer(float *pos);

        /**
         * Sets the number of atoms in the solvent. This reverts all previous
         * calls to 'AllocateSolventAtomData' and
         * 'AllocateSolventAtomPositions', so if you plan to use memory stored
         * inside the interface, you must call these methods again.
         *
         * @param cnt The number of atoms to be in the solvent.
         */
        void SetSolventAtomCount(unsigned int cnt);

        /**
         * Sets the atom data for one atom of the solvent. You must not call
         * this method if the interface is not owner of the atom data objects.
         *
         * @param idx  The index of the atom to set it's data
         * @param data The data to be set for the atom.
         */
        void SetSolventAtomData(unsigned int idx, const AtomData& data);

        /**
         * Sets a pointer to an array of atom data object to be used through
         * this interface. The memory pointed to will remain owned by the
         * caller and the caller must ensure it remains valid as long as it is
         * used through this interface. The array must be of sufficient size
         * compare to the number of atoms which should be set earlier using
         * 'SetSolventAtomCount'.
         *
         * @param data Pointer to an array of atom data objects.
         */
        void SetSolventAtomDataPointer(AtomData *data);

        /**
         * Sets the position for one atom of the solvent. You mut not call this
         * method if the interface is not owner of the position memory.
         *
         * @param idx The index of the atom to be set.
         * @param x   The x component of the atoms position.
         * @param y   The y component of the atoms position.
         * @param z   The z component of the atoms position.
         */
        void SetSolventAtomPosition(unsigned int idx, float x, float y,
            float z);

        /**
         * Sets a pointer to an array of floats to be used through this
         * interface of the positions of the solvent atoms. The memory pointed
         * to will remain owned by the caller and the caller must ensure it
         * remains valid as long as it is* used through this interface. The
         * array must be of sufficient size compare to the number of atoms
         * which should be set earlier using 'SetSolventAtomCount'. The float
         * array must be layouted 'xyzxyz...'
         *
         * @param pos Pointer to an array of float to be used as solvent atom
         *            positions.
         */
        void SetSolventAtomPositionPointer(float *pos);

        /**
         * Sets the number of molecule types present in the solvent. This
         * reverts any previous call to 'SetSolventMoleculeInfo'.
         *
         * @param cnt The number of molecule types in the solvent.
         */
        void SetSolventMoleculeTypeCount(unsigned int cnt);

        /**
         * Sets the number of molecules of the 'idx'-th type in the solvent
         * atom data.
         *
         * @param idx The index of the solvent molecule type to set the count
         *            information.
         * @param cnt The number of molecules of this type in the solvent atom
         *            data.
         */
        void SetSolventMoleculeCount(unsigned int idx, unsigned int cnt);

        /**
         * Sets the information about one solvent molecule type. These are how
         * many atoms are there in each molecule of this type, how many
         * molecules of this type are present in the solvent, and the name of
         * this molecule type.
         *
         * The solvent atoms must be sorted accorrdingly to this information.
         * That means that first all atoms for the first molecule of the first
         * molecule type are stored, then the ones for the second molecule of
         * the first type, and so on. Then the molecules for the second type
         * follow. The atoms of each molecule must be sorted that the molecule
         * connectivity structure, specified for the molecule remains correct.
         *
         * @param idx  The index of the molecule type to be specified.
         * @param data The 'SolventMoleculeData' object that holds all
         *             information about the solvent molecule type.
         */
        void SetSolventMoleculeData(unsigned int idx,
            const SolventMoleculeData& data);

        /**
         * Returns the overall number of atoms in the solvent.
         *
         * @return The overall number of atoms in the solvent.
         */
        inline unsigned int SolventAtomCount(void) const {
            return this->solAtomCnt;
        }

        /**
         * Returns a pointer to the atom data of the solvent atoms. Points to
         * 'SolventAtomCount' times 'AtomData' objects holding the data. The
         * pointer points to the internal memory structure. The caller must
         * not free or alter the memory returned.
         *
         * @return A pointer to the AtomData of the solvent atoms.
         */
        inline const AtomData * SolventAtomData(void) const {
            return this->solAtomData;
        }

        /**
         * Returns a pointer to the positions of the solvent atoms. Points to
         * 'SolventAtomCount' times 3 floats holding the data layouted
         * xyzxyz... The pointer points to the internal memory structure. The
         * caller must not free or alter the memory returned.
         *
         * @return A pointer to the positions of the solvent atoms.
         */
        inline const float * SolventAtomPositions(void) const {
            return this->solAtomPos;
        }

        /**
         * Returns the number of solvent molecules of a given type.
         *
         * @param type The type index of the solvent molecule type to be
         *             returned.
         *
         * @return The number of solvent molecules of the specified type.
         */
        inline unsigned int SolventMoleculeCount(unsigned int type) const {
            if (type >= this->solMolTypeCnt) {
                throw vislib::IllegalParamException("type", __FILE__, __LINE__);
            }
            return this->solMolCnt[type];
        }

        /**
         * Returns the number of solvent molecule types.
         *
         * @return The number of solvent molecule types.
         */
        inline unsigned int SolventMoleculeTypeCount(void) const {
            return this->solMolTypeCnt;
        }

        /**
         * Returns the name of a solvent molecule type.
         *
         * @param type The solvent molecule type requested.
         *
         * @return The name of the requested solvent molecule type.
         */
        inline const SolventMoleculeData& SolventMoleculeTypeData(
                unsigned int type) const {
            if (type >= this->solMolTypeCnt) {
                throw vislib::IllegalParamException("type", __FILE__, __LINE__);
            }
            return this->solMolTypeData[type];
        }
        /**
         * Sets the minimum temperature factor.
         *
         * @param minTF  The minimum temperature factor.
         */
        inline void SetMinimumTemperatureFactor(float minTF) {
            this->minTempFactor = minTF;
        }

        /**
         * Sets the minimum temperature factor.
         * [DEPRECATED]
         *
         * @param minTF  Pointer to the minimum temperature factor.
         */
        DEPRECATED void SetMinimumTemperatureFactor(float* minTF) {
            if (minTF == NULL) {
                this->minTempFactor = 0.0f;
                return;
            }
            this->minTempFactor = *minTF;
        }

        /**
         * Sets the maximum temperature factor.
         *
         * @param maxTF  The maximum temperature factor.
         */
        inline void SetMaximumTemperatureFactor(float maxTF) {
            this->maxTempFactor = maxTF;
        }

        /**
         * Sets the maximum temperature factor.
         * [DEPRECATED]
         *
         * @param maxTF  Pointer to the maximum temperature factor.
         */
        DEPRECATED void SetMaximumTemperatureFactor(float* maxTF) {
            if (maxTF == NULL) {
                this->maxTempFactor = 0.0f;
                return;
            }
            this->maxTempFactor = *maxTF;
        }

        /**
         * Returns the minimum temperature factor occuring thoughout the protein.
         *
         * @return The minimum temperature factor.
         */
        inline float MinimumTemperatureFactor(void) const {
            return this->minTempFactor;
        }

        /**
         * Returns the maximum temperature factor occuring thoughout the protein.
         *
         * @return The maximum temperature factor.
         */
        inline float MaximumTemperatureFactor(void) const {
            return this->maxTempFactor;
        }

        /**
         * Sets the minimum occupancy.
         *
         * @param minO  The minimum occupancy.
         */
        inline void SetMinimumOccupancy(float minO) {
            this->minOccupancy = minO;
        }

        /**
         * Sets the minimum occupancy.
         * [DEPRECATED]
         *
         * @param minO  Pointer to the minimum occupancy.
         */
        DEPRECATED void SetMinimumOccupancy(float* minO) {
            if (minO == NULL) {
                this->minOccupancy = 0.0f;
                return;
            }
            this->minOccupancy = *minO;
        }

        /**
         * Sets the maximum occupancy.
         *
         * @param maxO  The maximum occupancy.
         */
        inline void SetMaximumOccupancy(float maxO) {
            this->maxOccupancy = maxO;
        }

        /**
         * Sets the maximum occupancy.
         * [DEPRECATED]
         *
         * @param maxO  Pointer to the maximum occupancy.
         */
        DEPRECATED void SetMaximumOccupancy(float* maxO) {
            if (maxO == NULL) {
                this->maxOccupancy = 0.0f;
                return;
            }
            this->maxOccupancy = *maxO;
        }

        /**
         * Returns the minimum occupancy occuring thoughout the protein.
         *
         * @return The minimum occupancy.
         */
        inline float MinimumOccupancy(void) const {
            return this->minOccupancy;
        }

        /**
         * Returns the maximum occupancy occuring thoughout the protein.
         *
         * @return The maximum occupancy.
         */
        inline float MaximumOccupancy(void) const {
            return this->maxOccupancy;
        }

        /**
         * Sets the minimum charge.
         *
         * @param minC  The minimum charge.
         */
        inline void SetMinimumCharge(float minC) {
            this->minCharge = minC;
        }

        /**
         * Sets the maximum charge.
         *
         * @param maxC  The maximum charge.
         */
        inline void SetMaximumCharge(float maxC) {
            this->maxCharge = maxC;
        }

        /**
         * Returns the minimum charge occuring thoughout the protein.
         *
         * @return The minimum charge.
         */
        inline float MinimumCharge(void) const {
            return this->minCharge;
        }

        /**
         * Returns the maximum charge occuring thoughout the protein.
         *
         * @return The maximum charge.
         */
        inline float MaximumCharge(void) const {
            return this->maxCharge;
        }

        /**
         * Returns whether RMS specific frame handling should be used or not.
         * (Should only called by NetCDFData!)
         *
         * @return 'True' if frames should be loaded for RMS diagram, 'false' otherwise.
         */
        inline bool GetRMSUse(void) const {
            return this->useRMS;
        }

        /**
         * Sets whether RMS specific frame handling should be used or not.
         * (Should only called by Renderer!)
         * 
         * @param 'True' if RMS diagram is used, 'false' otherwise.
         */
        inline void SetRMSUse(bool userms) {
            this->useRMS = userms;
        }

        /**
         * Set ID of requested RMS frame.
         * (Should only called by Renderer!)
         * 
         * @param ID of requested frame
         */
        void SetRequestedRMSFrame(unsigned int newfrmID) {
            this->currentRMSFrameID = newfrmID;
        }

        /**
         * Get ID of requested RMS frame.
         * (Should only called by NetCDFData!)
         *
         * @return ID of requested frame
         */
        unsigned int GetRequestedRMSFrame(void) {
            return this->currentRMSFrameID;
        }

			/**
          * Sets the Id of the current frame of the data set.
          *
          * @param fId The new frame ID of the data set.
          */
			inline void SetCurrentFrameId(unsigned int fId) {
				this->currentFrameId = fId;
			}
			
			/**
			 * Sets the Id of the current frame of the data set.
			 *
			 * @param fId The new frame ID of the data set.
			 */
			inline unsigned int GetCurrentFrameId(void) {
				return this->currentFrameId;
			}
		  
    private:

        /** 'true' if data is used for RMS diagram */
        bool useRMS;
        /** ID of requested frame */
        unsigned int currentRMSFrameID;

        /** Number of amino acids in the amino acid name table */
        unsigned int aminoAcidNameCnt;
    
        /**
         * Flag whether or not the interface owns the memory of the amino acid
         * name table.
         */
        bool aminoAcidNameMemory;
    
        /** The amino acid name table */
        vislib::StringA *aminoAcidNames;

        /** The number of atom types in the atom type table */
        unsigned int atomTypeCnt;

        /**
         * Flag whether of not the interface owns the memory of the atom type
         * table.
         */
        bool atomTypeMemory;

        /** The atom type table */
        AtomType *atomTypes;

        /** Number of chains of the protein */
        unsigned int chainCnt;

        /**
         * The chains of the protein. Must point to 'chainCnt' times 'Chain'
         * objects.
         */
        Chain *chains;

        /**
         * Flag whether or not the interface owns the memory of the protein
         * chain objects.
         */
        bool chainsMemory;

        /** The number of disfulid bonds in the interface */
        unsigned int dsBondsCnt;

        /**
         * An array pointing to the index pairs each representing a disulfid
         * bond.
         */
        IndexPair *dsBonds;

        /**
         * Flag whether of not the interface owns the memory of the disulfid
         * bonds.
         */
        bool dsBondsMemory;

        /** Number of atoms in the proteins chains */
        unsigned int protAtomCnt;

        /**
         * Points to data objects for the atoms of the protein.
         * Must point to 'protAtomCnt' times 'AtomData' objects.
         */
        AtomData* protAtomData;

        /**
         * Pointer to the position information of the atoms of the proteins.
         * Must point to '3 * protAtomCnt' floats layouted x,y,z,x,y,z, ...
         */
        float *protAtomPos;

        /**
         * Flag whether or not the interface owns the memory of the atom data
         * array of the protein.
         */
        bool protDataMemory;
    
        /**
         * Flag whether or not the interface owns the memory of the position
         * array of the protein.
         */
        bool protPosMemory;

        /** Number of atoms in the solvent */
        unsigned int solAtomCnt;

        /**
         * Points to data objects for the atoms of the solvent.
         * Must point to 'solAtomCnt' times 'AtomData' objects.
         */
        AtomData* solAtomData;

        /**
         * Pointer to the position information of the atoms of the solvent.
         * Must point to '3 * solAtomCnt' floats layouted x,y,z,x,y,z, ...
         */
        float *solAtomPos;
    
        /**
         * Flag whether or not the interface owns the memory of the atom data
         * array of the solvent.
         */
        bool solDataMemory;

        /** The number of solvent molecules of each type */
        unsigned int *solMolCnt;
    
        /**
         * Flag whether or not the interface owns the memory of the solvent
         * molecule count array.
         */
        bool solMolCntMemory;

        /** Number of solvent molecule types */
        unsigned int solMolTypeCnt;

        /** The data of solvent molecule types */
        SolventMoleculeData *solMolTypeData;
        
        /**
         * Flag whether or not the interface owns the memory of the solvent
         * molecule type data array.
         */
        bool solMolTypeDataMemory;

        /**
         * Flag whether or not the interface owns the memory of the position
         * array of the solvent.
         */
        bool solPosMemory;

        /** Minimum occumancy */
        float minOccupancy;

        /** Maximum occupancy */
        float maxOccupancy;

        /** Minimum temperature factor */
        float minTempFactor;

        /** Maximum temperature factor */
        float maxTempFactor;

        /** Minimum charge */
        float minCharge;

        /** Maximum charge */
        float maxCharge;

        /** bounding box of the represented data */
        vislib::math::Cuboid<FLOAT> bbox;

        /** the scaling factor of this data */
        float scaling;

		  /** the ID of the current frame */
		  unsigned int currentFrameId;
    };

    /** Description class typedef */
	typedef megamol::core::CallAutoDescription<CallProteinData> CallProteinDataDescription;


} /* end namespace protein */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLPROTEINDATA_H_INCLUDED */
