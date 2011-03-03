/*
 * MolecularDataCall.h
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_MOLECULARDATACALL_H_INCLUDED
#define MMPROTEINPLUGIN_MOLECULARDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "AbstractGetData3DCall.h"
#include "CallAutoDescription.h"
#include "vislib/IllegalParamException.h"
#include "vislib/Vector.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include <vector>

namespace megamol {
namespace protein {

    /**
     * Base class of rendering graph calls and of data interfaces for 
     * molecular data (e.g. protein-solvent-systems).
     *
     * Note that all data has to be sorted!
     * There are no IDs anymore, always use the index values into the right 
     * tables.
     *
     * All data has to be stored in the corresponding data source object. This
     * interface object will only pass pointers to the renderer objects. A
     * compatible data source should therefore use the public nested class of
     */

    class MolecularDataCall : public megamol::core::AbstractGetData3DCall {
    public:

        /**
         * Nested class describing one unspecific residue.
         *
         * Base class for all specific residues like amino acids, nucleic 
         * acids, lipids etc.
         */
        class Residue {
        public:
            /** Residue types */
            enum ResidueType {
                UNSPECIFIC  = 0,
                AMINOACID   = 1
            };

            /** Default ctor initialising all fields to zero. */
            Residue(void);

            /**
             * Copy ctor performin a deep copy from 'src'.
             *
             * @param src The object to clone from.
             */
            Residue( const Residue& src);

            /**
             * Ctor.
             *
             * @param firstAtomIdx The index of the first atom of this residue.
             * @param atomCnt The size of the residue in number of atoms.
             * @param bbox The bounding box of this residue.
             */
            Residue( unsigned int firstAtomIdx, unsigned int atomCnt,
                vislib::math::Cuboid<float> bbox, unsigned int typeIdx);

            /** Dtor. */
            ~Residue(void);

            /**
             * Get the type of the residue
             */
            virtual ResidueType Identifier() { return UNSPECIFIC; }

            /**
             * Returns the number of atoms.
             *
             * @return The number of atoms.
             */
            inline unsigned int AtomCount(void) const {
                return this->atomCnt;
            }

            /**
             * Returns the index of the first atom in the residue.
             *
             * @return The index of the first atom in the residue.
             */
            inline unsigned int FirstAtomIndex(void) const {
                return this->firstAtomIdx;
            }

            /**
             * Returns the type index of the residue.
             *
             * @return The type index of the residue.
             */
            unsigned int Type(void) const { return type; };

            /**
             * The bounding box of the residue.
             *
             * @return The bounding box of the residue.
             */
            inline vislib::math::Cuboid<float> BoundingBox(void) const { 
                return this->boundingBox; 
            }
            
            /**
             * Sets the position of the residue by specifying the first atom 
             * index and the size in number of atoms.
             *
             * @param firstAtom The index of the first atom of this residue.
             * @param atomCnt The size of the residue in number of atoms.
             */
            void SetPosition( unsigned int firstAtom, unsigned int atomCnt);

            /**
             * Sets the type index of the residue.
             *
             * @param name The type index of the residue.
             */
            inline void SetType( unsigned int typeIdx) { this->type = typeIdx; };

            /**
             * Set the bounding box of the residue.
             *
             * @param bbox The bounding box of the residue.
             */
            void SetBoundingBox( vislib::math::Cuboid<float> bbox) {
                this->boundingBox = bbox;
            }

            /**
             * Copy operator performs a deep copy of the residue object.
             *
             * @param rhs The right hand side operand to clone from.
             *
             * @return A reference to this object.
             */
            Residue& operator=(const Residue& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'this' and 'rhs' represent the same residue,
             *         'false' otherwise.
             */
            bool operator==(const Residue& rhs) const;

            /**
             * Test for inequality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'false' if 'this' and 'rhs' represent the same amino,
             *         'true' otherwise.
             */
            inline bool operator!=(const Residue& rhs) const {
                return !(*this == rhs);
            }

            /**
             * Sets the filter value of the residue.
             *
             * @param val The filter value.
             */
            inline void SetFilter( int val) { this->filter = val; };

            /**
             * Get the filter value of the residue.
             *
             * @return The filter value.
             */
            int Filter(void) const { return this->filter; };
        

        protected:

            /** The size of this residue in atoms */
            unsigned int atomCnt;

            /** The index of the first atom of this residue */
            unsigned int firstAtomIdx;

            /** The bouding box surrounding all the atoms in this residue */
            vislib::math::Cuboid<float> boundingBox;

            /** The index of the type of the residue */
            unsigned int type;

            /** The filter value */
            int filter;

        };

        /**
         * Nested class describing one specific amino acid.
         */
        class AminoAcid : public Residue {
        public:

            /** Default ctor initialising all fields to zero. */
            AminoAcid(void);

            /**
             * Copy ctor performin a deep copy from 'src'.
             *
             * @param src The object to clone from.
             */
            AminoAcid( const AminoAcid& src);

            /**
             * Ctor.
             *
             * @param firstAtomIdx The index of the first atom of this amino
             *                     acid.
             * @param atomCnt The size of the amino acid in number of atoms.
             * @param cAlphaIdx The index to be set as index for the C alpha
             *                  atom.
             * @param cCarbIdx The index to be set as index for the C atom of
             *                 the carboxyl group.
             * @param oIdx The index to be set as index for the O atom.
             * @param nIdx The index to be set as index for the N atom of the 
             *             amino acid.
             * @param nameIdx The index of the name of the amino acid.
             * @param bbox The bounding box of this residue.
             */
            AminoAcid( unsigned int firstAtomIdx, unsigned int atomCnt,
                unsigned int cAlphaIdx, unsigned int cCarbIdx,
                unsigned int nIdx, unsigned int oIdx, 
                vislib::math::Cuboid<float> bbox, unsigned int typeIdx);

            /** Dtor. */
            ~AminoAcid(void);

            /**
             * Get the type of the residue
             */
            virtual ResidueType Identifier() { return AMINOACID; }

            /**
             * Returns the index of the C alpha atom.
             *
             * @return The index of the C alpha atom.
             */
            inline unsigned int CAlphaIndex(void) const {
                return this->cAlphaIdx;
            }

            /**
             * Returns the index of the C atom of the carboxyl group.
             *
             * @return The index of the C atom of the carboxyl group.
             */
            inline unsigned int CCarbIndex(void) const {
                return this->cCarbIdx;
            }

            /**
             * Returns the index of the N atom.
             *
             * @return The index of the N atom.
             */
            inline unsigned int NIndex(void) const {
                return this->nIdx;
            }

            /**
             * Returns the index of the O atom.
             *
             * @return The index of the O atom.
             */
            inline unsigned int OIndex(void) const {
                return this->oIdx;
            }

            /**
             * Sets the index of the C alpha atom.
             *
             * @param idx The index to be set as index for the C alpha atom.
             */
            void SetCAlphaIndex(unsigned int idx);

            /**
             * Sets the index of the C atom of the carboxyl group.
             *
             * @param idx The index to be set as index for the C atom of the
             *            carboxyl group.
             */
            void SetCCarbIndex(unsigned int idx);

            /**
             * Sets the index of the N atom.
             *
             * @param idx The index to be set as index for the N atom.
             */
            void SetNIndex(unsigned int idx);

            /**
             * Sets the index of the O atom.
             *
             * @param idx The index to be set as index for the O atom.
             */
            void SetOIndex(unsigned int idx);

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

        protected:

            /** The index of the C alpha atom */
            unsigned int cAlphaIdx;

            /** The index of the C atom of the carboxyl group */
            unsigned int cCarbIdx;

            /** The index of the N atom */
            unsigned int nIdx;

            /** The index of the O atom */
            unsigned int oIdx;

        };

        /**
         * Nested class describing a molecule
         */
        class Molecule {
        public:
            /** ctor */
            Molecule();
            
            /**
             * Copy ctor performin a deep copy from 'src'.
             *
             * @param src The object to clone from.
             */
            Molecule( const Molecule& src);

            /**
             * Ctor.
             *
             * @param firstResIdx The index of the first residue of this molecule.
             * @param resCnt The size of the molecule in number of residues.
             */
            Molecule( unsigned int firstResIdx, unsigned int resCnt);

            /** dtor */
            ~Molecule();

            /**
             * Sets the position of the molecule by specifying the first 
             * residue index and the size in number of residues.
             *
             * @param firstRes The index of the first residue of this molecule.
             * @param resCnt   The size of the molecule in number of residues.
             */
            void SetPosition( unsigned int resIdx, unsigned int resCnt) {
                this->firstResidueIndex = resIdx;
                this->residueCount = resCnt;
            }

            /**
             * Sets the range of the connections by specifying the first 
             * connection index and the number of connections.
             *
             * @param firstCon  The index of the first connection.
             * @param lastCon   The connection count.
             */
            void SetConnectionRange( unsigned int firstCon, unsigned int conCnt) {
                this->firstConIdx = firstCon;
                this->conCount = conCnt;
            }

            /**
             * Sets the position of the secondary structure of the molecule by 
             * specifying the first secondary structure element index and the 
             * size in number of secondary structure elements.
             *
             * @param firstSecS The index of the first sec struct elem.
             * @param secSCnt   The number of sec struct elems.
             */
            void SetSecondaryStructure( unsigned int firstSecS, unsigned int secSCnt) {
                this->firstSecStructIdx = firstSecS;
                this->secStructCount = secSCnt;
            }

            /**
             * Returns the number of residues.
             *
             * @return The number of residues.
             */
            inline unsigned int ResidueCount(void) const {
                return this->residueCount;
            }

            /**
             * Returns the index of the first residue in the molecule.
             *
             * @return The index of the first residue in the molecule.
             */
            inline unsigned int FirstResidueIndex(void) const {
                return this->firstResidueIndex;
            }

            /**
             * Returns the first connection index.
             *
             * @return The first connection index.
             */
            inline unsigned int FirstConnectionIndex(void) const {
                return firstConIdx;
            }

            /**
             * Returns the connection count.
             *
             * @return The connection count.
             */
            inline unsigned int ConnectionCount(void) const {
                return conCount;
            }

            /**
             * Returns the number of sec struct elems.
             *
             * @return The number of sec struct elems.
             */
            inline unsigned int SecStructCount(void) const {
                return this->secStructCount;
            }

            /**
             * Returns the index of the first sec struct elem.
             *
             * @return The index of the first sec struct elem.
             */
            inline unsigned int FirstSecStructIndex(void) const {
                return this->firstSecStructIdx;
            }

            /**
             * Copy operator performs a deep copy of the molecule object.
             *
             * @param rhs The right hand side operand to clone from.
             *
             * @return A reference to this object.
             */
            Molecule& operator=(const Molecule& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'this' and 'rhs' represent the same molecule,
             *         'false' otherwise.
             */
            bool operator==(const Molecule& rhs) const;

            /**
             * Test for inequality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'false' if 'this' and 'rhs' represent the same molecule,
             *         'true' otherwise.
             */
            inline bool operator!=(const Molecule& rhs) const {
                return !(*this == rhs);
            }

            /**
             * Sets the filter value of the residue.
             *
             * @param val The filter value.
             */
            inline void SetFilter( int val) { this->filter = val; };

            /**
             * Get the filter value of the residue.
             *
             * @return The filter value.
             */
            int Filter(void) const { return this->filter; };

        private:
            /** the index of the first residue in the molecule */
            unsigned int firstResidueIndex;
            /** the number of residues in the molecule */
            unsigned int residueCount;

            /** the index of the first secondary structure element in the molecule */
            unsigned int firstSecStructIdx;
            /** the number of secondary structure elements in the molecule */
            unsigned int secStructCount;

            /** The first connection index */
            unsigned int firstConIdx;
            /** The connection count */
            unsigned int conCount;

            /** The filter value */
            int filter;
        };

        /**
         * Nested class holding all information about one segment of a proteins
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
             * Sets the position of the secondary structure element by the
             * indices of the amino acid where this element starts and the 
             * size of the element in (partial) amino acids.
             *
             * @param firstAminoAcidIdx The index of the amino acid where this
             *                          element starts.
             * @param aminoAcidCnt The size of the element in (partial) amino
             *                     acids.
             */
            void SetPosition( unsigned int firstAminoAcidIdx, 
                unsigned int aminoAcidCnt);

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

            /** The index of the amino acid in which this element starts */
            unsigned int firstAminoAcidIdx;

            /** The type of this element */
            ElementType type;

        };

        /**
         * Nested class describing a chain
         */
        class Chain {
        public:
            /** Residue types */
            enum ChainType {
                UNSPECIFIC  = 0,
                SOLVENT     = 1
            };

            /** ctor */
            Chain();
            
            /**
             * Copy ctor performin a deep copy from 'src'.
             *
             * @param src The object to clone from.
             */
            Chain( const Chain& src);

            /**
             * Ctor.
             *
             * @param firstMolIdx The index of the first molecule of this chain.
             * @param molCnt The size of the chain in number of molecules.
             * @param chainType The type of the chain.
             */
            Chain( unsigned int firstMolIdx, unsigned int molCnt, ChainType chainType = UNSPECIFIC);

            /** dtor */
            ~Chain();

            /**
             * Sets the type of the chain.
             *
             * @param t The type of the chain.
             */
            void SetType( ChainType t) {
                this->type = t;
            }

            /**
             * Returns the number of molecules.
             *
             * @return The number of molecules.
             */
            inline ChainType Type(void) const {
                return this->type;
            }

            /**
             * Sets the position of the chain by specifying the first 
             * molecule index and the size in number of molecules.
             *
             * @param firstMol The index of the first molecule of this chain.
             * @param molCnt The size of the chain in number of molecules.
             */
            void SetPosition( unsigned int molIdx, unsigned int molCnt) {
                this->firstMoleculeIndex = molIdx;
                this->moleculeCount = molCnt;
            }

            /**
             * Returns the number of molecules.
             *
             * @return The number of molecules.
             */
            inline unsigned int MoleculeCount(void) const {
                return this->moleculeCount;
            }

            /**
             * Returns the index of the first molecule in the chain.
             *
             * @return The index of the first molecule in the chain.
             */
            inline unsigned int FirstMoleculeIndex(void) const {
                return this->firstMoleculeIndex;
            }

            /**
             * Copy operator performs a deep copy of the molecule object.
             *
             * @param rhs The right hand side operand to clone from.
             *
             * @return A reference to this object.
             */
            Chain& operator=(const Chain& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'this' and 'rhs' represent the same molecule,
             *         'false' otherwise.
             */
            bool operator==(const Chain& rhs) const;

            /**
             * Test for inequality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'false' if 'this' and 'rhs' represent the same molecule,
             *         'true' otherwise.
             */
            inline bool operator!=(const Chain& rhs) const {
                return !(*this == rhs);
            }

            /**
             * Sets the filter value of the residue.
             *
             * @param val The filter value.
             */
            inline void SetFilter( int val) { this->filter = val; };

            /**
             * Get the filter value of the residue.
             *
             * @return The filter value.
             */
            int Filter(void) const { return this->filter; };

        private:
            /** the chain type */
            ChainType type;
            /** the index of the first molecule in the chain */
            unsigned int firstMoleculeIndex;
            /** the number of molecules in the chain */
            unsigned int moleculeCount;

            /** The filter value */
            int filter;
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
            return "MolecularDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get molecular data";
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

        /** Ctor. */
        MolecularDataCall(void);

        /** Dtor. */
        virtual ~MolecularDataCall(void);

        // -------------------- get and set routines --------------------

        /**
         * Get the total number of atoms.
         *
         * @return The atom count.
         */
        unsigned int AtomCount(void) const { return atomCount; }

        /**
         * Get the atom positions.
         *
         * @return The atom position array.
         */
        const float* AtomPositions(void) const { return atomPos; }

        /**
         * Get the atom b-factors.
         *
         * @return The atom b-factor array.
         */
        const float* AtomBFactors(void) const { return atomBFactors; }

        /**
         * Get the atom charges.
         *
         * @return The atom charge array.
         */
        const float* AtomCharges(void) const { return atomCharges; }

        /**
         * Get the atom occupancy.
         *
         * @return The atom occupancy array.
         */
        const float* AtomOccupancies(void) const { return atomOccupancies; }

        /**
         * Get the indices of atom type.
         *
         * @return The atom type index array.
         */
        const unsigned int* AtomTypeIndices(void) const { return atomTypeIdx; }

        /**
         * Get the residue count.
         *
         * @return The residue count.
         */
        unsigned int ResidueCount(void) const { return resCount; }

        /**
         * Get the residues.
         *
         * @return The residue array.
         */
        Residue** Residues(void) const { return residues; }

        /**
         * Get the residue type names.
         *
         * @return The residue type name array.
         */
        vislib::StringA* ResidueTypeNames(void) const { return resTypeNames; }

        /**
         * Get the residue type name count.
         *
         * @return The residue type name count.
         */
        unsigned int ResidueTypeNameCount(void) const { return resTypeNameCnt; }

        /**
         * Get the atom count.
         *
         * @return The atom count.
         */
        unsigned int AtomTypeCount(void) const { return atomTypeCount; }

        /**
         * Get the atom types.
         *
         * @return The atom type array.
         */
        const AtomType* AtomTypes(void) const { return atomType; }

        /**
         * Get the number of connections (bonds).
         *
         * @return The connection count.
         */
        unsigned int ConnectionCount(void) const { return connectionCount; }

        /**
         * Get the connections (bonds).
         *
         * @return The connections.
         */
        const unsigned int* Connection(void) const { return connections; }

        /**
         * Get the number of molecules.
         *
         * @return The molecule count.
         */
        unsigned int MoleculeCount(void) const { return molCount; }

        /**
         * Get the molecules.
         *
         * @return The molecules.
         */
        const Molecule* Molecules(void) const { return molecules; }

        /**
         * Get the number of chains.
         *
         * @return The chain count.
         */
        unsigned int ChainCount(void) const { return chainCount; }

        /**
         * Get the chains.
         *
         * @return The chains.
         */
        const Chain* Chains(void) const { return chains; }

        /**
         * Set the atom types and positions.
         *
         * @param atomCnt       The atom count.
         * @param atomTypeCnt   The atom type count.
         * @param typeIdx       The atom type indices.
         * @param pos           The atom positions.
         * @param types         The atom types.
         * @param bfactor       The atom b-factors.
         * @param charge        The atom charges.
         * @param occupancies   The atom occupancies.
         */
        void SetAtoms( unsigned int atomCnt, unsigned int atomTypeCnt, 
            unsigned int* typeIdx, float* pos, AtomType* types,
            float* bfactor, float* charge, float* occupancy);

        /**
         * Set the residues.
         *
         * @param resCnt    The residue count.
         * @param res       The residues.
         */
        void SetResidues( unsigned int resCnt, Residue** res);

        /**
         * Set the residue type names.
         *
         * @param namesCnt  The residue type name count.
         * @param names     The residue type names.
         */
        void SetResidueTypeNames( unsigned int namesCnt, vislib::StringA* names);

        /**
         * Set the connections (bonds).
         *
         * @param conCnt    The connection count.
         * @param con       The connections.
         */
        void SetConnections( unsigned int conCnt, unsigned int* con);

        /**
         * Set the molecules.
         *
         * @param molCnt    The molecule count.
         * @param mol       The molecules.
         */
        void SetMolecules( unsigned int molCnt, Molecule* mol);

        /**
         * Set the number of secondary structure elements.
         *
         * @param secStructCnt The secondary structure element count.
         */
        void SetSecondaryStructureCount( unsigned int cnt);

        /**
         * Set a secondary stucture element to the array.
         *
         * @param idx   The index of the element.
         * @param secS  The secondary structure element.
         *
         * @return 'true' if successful, 'false' otherwise.
         */
        bool SetSecondaryStructure( unsigned int idx, SecStructure secS);

        /**
         * Sets the position of the secondary structure to the molecule by 
         * specifying the first secondary structure element index and the 
         * size in number of secondary structure elements.
         *
         * @param molIdx    The index of the molecule.
         * @param firstSecS The index of the first sec struct elem.
         * @param secSCnt   The number of sec struct elems.
         */
        void SetMoleculeSecondaryStructure( unsigned int molIdx, 
            unsigned int firstSecS, unsigned int secSCnt);

        /**
         * Get the secondary structure.
         *
         * @return The secondary structure array.
         */
        const SecStructure* SecondaryStructures() const;

        /**
         * Get the number of secondary structure elements.
         *
         * @return The secondary structure element count.
         */
        unsigned int SecondaryStructureCount() const;

        /**
         * Set the chains.
         *
         * @param chainCnt  The chain count.
         * @param chain     The chains.
         */
        void SetChains( unsigned int chainCnt, Chain* chain);

        /**
         * Set the bfactor range.
         *
         * @param min    The minimum bfactor.
         * @param max    The minimum bfactor.
         */
        void SetBFactorRange( float min, float max) {
            this->minBFactor = min; this->maxBFactor = max; }

        /**
         * Get the minimum bfactor.
         *
         * @return The minimum bfactor value.
         */
        float MinimumBFactor() const { return this->minBFactor; }

        /**
         * Get the maximum bfactor.
         *
         * @return The maximum bfactor value.
         */
        float MaximumBFactor() const { return this->maxBFactor; }

        /**
         * Set the charge range.
         *
         * @param min    The minimum charge.
         * @param max    The minimum charge.
         */
        void SetChargeRange( float min, float max) {
            this->minCharge = min; this->maxCharge = max; }

        /**
         * Get the minimum charge.
         *
         * @return The minimum charge value.
         */
        float MinimumCharge() const { return this->minCharge; }

        /**
         * Get the maximum charge.
         *
         * @return The maximum charge value.
         */
        float MaximumCharge() const { return this->maxCharge; }

        /**
         * Set the occupancy range.
         *
         * @param min    The minimum occupancy.
         * @param max    The minimum occupancy.
         */
        void SetOccupancyRange( float min, float max) {
            this->minOccupancy = min; this->maxOccupancy = max; }

        /**
         * Get the minimum occupancy.
         *
         * @return The minimum occupancy value.
         */
        float MinimumOccupancy() const { return this->minOccupancy; }

        /**
         * Get the maximum occupancy.
         *
         * @return The maximum occupancy value.
         */
        float MaximumOccupancy() const { return this->maxOccupancy; }
        
        /**
        * Answer the callTime
        *
        * @return the calltime
        */
        inline float CallTime(void) const {
            return this->callTime;
        }
        
         /**
         * Sets the calltime to request data for.
         *
         * @param callTime The calltime to request data for.
         *
         */
        inline void SetCallTime(float callTime) {
            this->callTime = callTime;
        }

    private:
        // -------------------- variables --------------------

        /** The number of atoms. */
        unsigned int atomCount;
        /** The array of atom positions. */
        float* atomPos;
        /** The array of atom type indices. */
        unsigned int* atomTypeIdx;

        /** The array of residues. */
        Residue** residues;
        /** The number of residues. */
        unsigned int resCount;

        /** The array pf residue type names */
        vislib::StringA* resTypeNames;
        /** The number of residue type names */
        unsigned int resTypeNameCnt;

        /** The array of molecules. */
        Molecule* molecules;
        /** The number of molecules. */
        unsigned int molCount;

        /** The array of secondary structures */
        vislib::Array<SecStructure> secStruct;

        /** The array of chains. */
        Chain* chains;
        /** The number of chains. */
        unsigned int chainCount;

        /** The number of atom types. */
        unsigned int atomTypeCount;
        /** The array of atom types. */
        AtomType* atomType;

        /** The total number of connections (bonds) */
        unsigned int connectionCount;
        /** The array of connections (bonds) of the atoms */
        unsigned int* connections;

        /** The array of b-factors */
        float* atomBFactors;
        /** The minimum bfactor */
        float minBFactor;
        /** The maximum bfactor */
        float maxBFactor;
        
        /** The array of charges */
        float* atomCharges;
        /** The minimum charges */
        float minCharge;
        /** The maximum charges */
        float maxCharge;
        
        /** The array of occupancies */
        float* atomOccupancies;
        /** The minimum occupancies */
        float minOccupancy;
        /** The maximum occupancies */
        float maxOccupancy;
        
        /** The exact requested/stored calltime. */
        float callTime;

    };

    /** Description class typedef */
    typedef megamol::core::CallAutoDescription<MolecularDataCall> MolecularDataCallDescription;
    


} /* end namespace protein */
} /* end namespace megamol */

#endif /* MMPROTEINPLUGIN_MOLECULARDATACALL_H_INCLUDED */
