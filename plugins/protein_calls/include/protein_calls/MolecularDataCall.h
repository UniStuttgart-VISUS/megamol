/*
 * MolecularDataCall.h
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MEGAMOL_PROTEIN_CALL_MOLECULARDATACALL_H_INCLUDED
#define MEGAMOL_PROTEIN_CALL_MOLECULARDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetData3DCall.h"
#include "vislib/Array.h"
#include "vislib/IllegalParamException.h"
#include "vislib/String.h"
#include "vislib/macro_utils.h"
#include "vislib/math/Vector.h"
#include <vector>

namespace megamol {
namespace protein_calls {

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
        enum ResidueType { UNSPECIFIC = 0, AMINOACID = 1 };

        /** Default ctor initialising all fields to zero. */
        Residue(void);

        /**
         * Copy ctor performin a deep copy from 'src'.
         *
         * @param src The object to clone from.
         */
        Residue(const Residue& src);

        /**
         * Ctor.
         *
         * @param firstAtomIdx The index of the first atom of this residue.
         * @param atomCnt The size of the residue in number of atoms.
         * @param bbox The bounding box of this residue.
         * @param typeIdx The bounding box of this residue.
         * @param moleculeIdx The index of the type of this residue.
         * @param origResIdx The original index of this residue.
         */
        Residue(unsigned int firstAtomIdx, unsigned int atomCnt, vislib::math::Cuboid<float> bbox, unsigned int typeIdx,
            int moleculeIdx, unsigned int origResIdx);

        /** Dtor. */
        ~Residue(void);

        /**
         * Get the type of the residue
         */
        virtual ResidueType Identifier() const {
            return UNSPECIFIC;
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
        unsigned int Type(void) const {
            return type;
        };

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
        void SetPosition(unsigned int firstAtom, unsigned int atomCnt);

        /**
         * Sets the type index of the residue.
         *
         * @param name The type index of the residue.
         */
        inline void SetType(unsigned int typeIdx) {
            this->type = typeIdx;
        };

        /**
         * Sets the molecule index of the residue.
         *
         * @param name The molecule index of the residue.
         */
        inline void SetMoleculeIndex(int moleculeIndex) {
            this->moleculeIndex = moleculeIndex;
        }
        inline int MoleculeIndex(void) const {
            return moleculeIndex;
        }

        /**
         * Set the bounding box of the residue.
         *
         * @param bbox The bounding box of the residue.
         */
        void SetBoundingBox(vislib::math::Cuboid<float> bbox) {
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
         * Sets the original index of the residue.
         *
         * @param val The index.
         */
        inline void SetOriginalResIndex(unsigned int val) {
            this->origResIndex = val;
        };

        /**
         * Get the original index of the residue.
         *
         * @return The index.
         */
        unsigned int OriginalResIndex(void) const {
            return this->origResIndex;
        };

        /**
         * Sets the filter value of the residue.
         *
         * @param val The filter value.
         */
        inline void SetFilter(int val) {
            this->filter = val;
        };

        /**
         * Get the filter value of the residue.
         *
         * @return The filter value.
         */
        int Filter(void) const {
            return this->filter;
        };


    protected:
        /** The size of this residue in atoms */
        unsigned int atomCnt;

        /** The index of the first atom of this residue */
        unsigned int firstAtomIdx;

        /** The bouding box surrounding all the atoms in this residue */
        VISLIB_MSVC_SUPPRESS_WARNING(4251)
        vislib::math::Cuboid<float> boundingBox;

        /** The index of the type of the residue */
        unsigned int type;

        /** molecule index of the residue (may be undefined -> -1) */
        int moleculeIndex;

        /** the original index of this resiude (from the PDB file) */
        unsigned int origResIndex;

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
        AminoAcid(const AminoAcid& src);

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
         * @param typeIdx The index of the type of this residue.
         * @param moleculeIdx The molecule index of this residue.
         * @param origResIdx The original residue index
         */
        AminoAcid(unsigned int firstAtomIdx, unsigned int atomCnt, unsigned int cAlphaIdx, unsigned int cCarbIdx,
            unsigned int nIdx, unsigned int oIdx, vislib::math::Cuboid<float> bbox, unsigned int typeIdx,
            int moleculeIdx, unsigned int origResIdx);

        /** Dtor. */
        ~AminoAcid(void);

        /**
         * Get the type of the residue
         */
        virtual ResidueType Identifier() const {
            return AMINOACID;
        }

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
        Molecule(const Molecule& src);

        /**
         * Ctor.
         *
         * @param firstResIdx The index of the first residue of this molecule.
         * @param resCnt The size of the molecule in number of residues.
         */
        Molecule(unsigned int firstResIdx, unsigned int resCnt, int chainIdx);

        /** dtor */
        ~Molecule();

        /**
         * Sets the position of the molecule by specifying the first
         * residue index and the size in number of residues.
         *
         * @param firstRes The index of the first residue of this molecule.
         * @param resCnt   The size of the molecule in number of residues.
         */
        void SetPosition(unsigned int resIdx, unsigned int resCnt) {
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
        void SetConnectionRange(unsigned int firstCon, unsigned int conCnt) {
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
        void SetSecondaryStructure(unsigned int firstSecS, unsigned int secSCnt) {
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
        inline void SetFilter(int val) {
            this->filter = val;
        };

        /**
         * Get the filter value of the residue.
         *
         * @return The filter value.
         */
        inline int Filter(void) const {
            return this->filter;
        };

        /**
         * Get the chain index of the residue.
         *
         * @return The filter value.
         */
        inline int ChainIndex(void) const {
            return this->chainIndex;
        }

    private:
        /** the index of the first residue in the molecule */
        unsigned int firstResidueIndex;
        /** the number of residues in the molecule */
        unsigned int residueCount;

        /** chain index of this molecule (may be undefined -> -1) */
        int chainIndex;

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
        enum ElementType { TYPE_COIL = 0, TYPE_SHEET = 1, TYPE_HELIX = 2, TYPE_TURN = 3 };

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
        void SetPosition(unsigned int firstAminoAcidIdx, unsigned int aminoAcidCnt);

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
        enum ChainType { UNSPECIFIC = 0, SOLVENT = 1 };

        /** ctor */
        Chain();

        /**
         * Copy ctor performin a deep copy from 'src'.
         *
         * @param src The object to clone from.
         */
        Chain(const Chain& src);

        /**
         * Ctor.
         *
         * @param firstMolIdx The index of the first molecule of this chain.
         * @param molCnt The size of the chain in number of molecules.
         * @param chainName The name of the chain.
         * @param chainType The type of the chain.
         */
        Chain(unsigned int firstMolIdx, unsigned int molCnt, char name = ' ', ChainType chainType = UNSPECIFIC);

        /** dtor */
        ~Chain();

        /**
         * Sets the type of the chain.
         *
         * @param t The type of the chain.
         */
        void SetType(ChainType t) {
            this->type = t;
        }

        /**
         * Returns the type of the chain.
         *
         * @return The type of the chain.
         */
        inline ChainType Type(void) const {
            return this->type;
        }

        /**
         * Sets the name of the chain.
         *
         * @param n The name of the chain.
         */
        void SetName(char n) {
            this->name = n;
        }

        /**
         * Returns the name of the chain.
         *
         * @return The name of the chain.
         */
        inline char Name(void) const {
            return this->name;
        }

        /**
         * Sets the position of the chain by specifying the first
         * molecule index and the size in number of molecules.
         *
         * @param firstMol The index of the first molecule of this chain.
         * @param molCnt The size of the chain in number of molecules.
         */
        void SetPosition(unsigned int molIdx, unsigned int molCnt) {
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
        inline void SetFilter(int val) {
            this->filter = val;
        };

        /**
         * Get the filter value of the residue.
         *
         * @return The filter value.
         */
        int Filter(void) const {
            return this->filter;
        };

    private:
        /** the chain type */
        ChainType type;
        /** the chain name */
        char name;
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
        AtomType(const vislib::StringA& name, float rad = 0.5f, unsigned char colR = 191, unsigned char colG = 191,
            unsigned char colB = 191, const vislib::StringA& element = "");

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
        inline void SetColour(unsigned char red, unsigned char green, unsigned char blue) {
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
         * Sets the element symbol for the atom type.
         *
         * @param element The element symbol for this atom type.
         */
        inline void SetElement(const vislib::StringA& element) {
            this->element = element;
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
        inline const unsigned char* Colour(void) const {
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
         * Returns the string representing the element symbol of this atom.
         *
         * @return The element symbol of the atom type.
         */
        inline const vislib::StringA& Element(void) const {
            return this->element;
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
        VISLIB_MSVC_SUPPRESS_WARNING(4251)
        vislib::StringA name;

        /** The name of the atom type. */
        VISLIB_MSVC_SUPPRESS_WARNING(4251)
        vislib::StringA element;

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
    static const char* ClassName(void) {
        return "MolecularDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
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
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "GetData";
        case 1:
            return "GetExtend";
        }
        return "";
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
    unsigned int AtomCount(void) const {
        return atomCount;
    }

    /**
     * Get the atom positions.
     *
     * @return The atom position array.
     */
    const float* AtomPositions(void) const {
        return atomPos;
    }

    /**
     * Get the atom b-factors.
     *
     * @return The atom b-factor array.
     */
    float* AtomBFactors(void) const {
        return atomBFactors;
    }

    /**
     * Get the atom charges.
     *
     * @return The atom charge array.
     */
    const float* AtomCharges(void) const {
        return atomCharges;
    }

    /**
     * Get the atom occupancy.
     *
     * @return The atom occupancy array.
     */
    const float* AtomOccupancies(void) const {
        return atomOccupancies;
    }

    /**
     * Get the indices of atom type.
     *
     * @return The atom type index array.
     */
    const unsigned int* AtomTypeIndices(void) const {
        return atomTypeIdx;
    }

    /**
     * Get the indices of atom residue.
     *
     * @return The atom residue index array.
     */
    const int* AtomResidueIndices(void) const {
        return atomResidueIdx;
    }
    // void SetAtomResidueIndices(const int*indices) { atomResidueIdx = indices; }

    /**
     * Get the indices of atom hydrogen bonds.
     * DEPRECATED. ONLY LEFT IN BECAUSE OF BACKWARDS COMPATIBILITY
     *
     * @return The atom hydrogen bonds index array.
     */
    const int* AtomHydrogenBondIndices(void) const {
        return atomHydrogenBondIdx;
    }

    const int* AtomFormerIndices(void) const {
        return atomFormerIdx;
    }

    /**
     * Set the indices of atom hydrogen bonds.
     * DEPRECATED. ONLY LEFT IN BECAUSE OF BACKWARDS COMPATIBILITY
     *
     * @param indices The atom hydrogen bonds index array.
     */
    void SetAtomHydrogenBondIndices(const int* indices) {
        atomHydrogenBondIdx = indices;
    }

    /**
     * Get the maximal distance of atom hydrogen bonds.
     * DEPRECATED. ONLY LEFT IN BECAUSE OF BACKWARDS COMPATIBILITY
     *
     * @return The maximal distance of a atom hydrogen bond.
     */
    float AtomHydrogenBondDistance(void) const {
        return atomHydrogenBondDistance;
    }

    /**
     * Set the maximal distance of atom hydrogen bonds.
     * DEPRECATED. ONLY LEFT IN BECAUSE OF BACKWARDS COMPATIBILITY
     *
     * @param dist The maximal distance of a atom hydrogen bond.
     */
    void SetAtomHydrogenBondDistance(float dist) {
        atomHydrogenBondDistance = dist;
    }

    /**
     * Get the hydrogen bond statistics array (the number of hydrogen bonds per atom)
     * DEPRECATED. ONLY LEFT IN BECAUSE OF BACKWARDS COMPATIBILITY
     *
     * @return The statistics array
     */
    const unsigned int* AtomHydrogenBondStatistics(void) const {
        return atomHydrogenBondStatistics;
    }

    /**
     * Get the number of solvent residues.
     *
     * @return The number of solvent residues.
     */
    unsigned int AtomSolventResidueCount(void) const {
        return atomSolventResCount;
    }

    /**
     * Set the atom hydrogen bond statistics (the number of hydrogen bonds per atom)
     * DEPRECATED. ONLY LEFT IN BECAUSE OF BACKWARDS COMPATIBILITY
     *
     * @param statistics The statistics array.
     */
    void SetAtomHydrogenBondStatistics(const unsigned int* statistics /*, int solventResCount*/) {
        atomHydrogenBondStatistics = statistics; /*atomSolventResCount=solventResCount;*/
    }

    /**
     * Returns whether the set hydrogen bonds are fake ones, e. g. the point from c alpha atom to c alpha atom
     *
     * @return True, if the hydrogen bonds are fake. False otherwise.
     */
    bool AtomHydrogenBondsFake(void) {
        return this->atomHydrogenBondsFake;
    }

    /**
     * Sets whether the set hydrogen bonds are fake.
     *
     * @param fake True, if the hydrogen bonds are fake. False otherwise
     */
    void SetAtomHydrogenBondsFake(const bool fake) {
        this->atomHydrogenBondsFake = fake;
    }

    /**
     * Set the index array of the solvent residues.
     *
     * @param count The number of solvent residues.
     * @param residueIndex The index array itself
     */
    void SetSolventResidueIndices(unsigned int count, const unsigned int* residueIndices) {
        this->solventResidueIdx = residueIndices;
        atomSolventResCount = count;
    }

    /**
     * Returns the solvent residue index array.
     *
     * @return The solvent residue index array.
     */
    const unsigned int* SolventResidueIndices(void) const {
        return solventResidueIdx;
    }

    /**
     * Sets the hydrogen bonds.
     *
     * @param hydroBonds The hydrogen bonds array. Two atom indices per hydrogen bond (donor, acceptor, donor,
     *acceptor, ...).
     * @param count The number of hydrogen bonds (array size / 2).
     */
    void SetHydrogenBonds(const unsigned int* hydroBonds, unsigned int count) {
        this->hydrogenBonds = hydroBonds;
        this->numHydrogenBonds = count;
    }

    /**
     * Returns the hydrogen bond array of the molecule.
     * Every hydrogen bond consists of two unsigned int atom indices.
     * First Index: Donor. Second Index: Acceptor
     *
     * @return The hydrogen bond array.
     */
    const unsigned int* GetHydrogenBonds(void) const {
        return this->hydrogenBonds;
    }

    /**
     * Returns the number of available hydrogen bonds.
     *
     * @param The number of hydrogen bonds.
     */
    unsigned int HydrogenBondCount(void) const {
        return this->numHydrogenBonds;
    }

    /**
     * Get the residue count.
     *
     * @return The residue count.
     */
    unsigned int ResidueCount(void) const {
        return resCount;
    }

    /**
     * Get the residues.
     *
     * @return The residue array.
     */
    const Residue** Residues(void) const {
        return residues;
    }

    /**
     * Get the residue type names.
     *
     * @return The residue type name array.
     */
    const vislib::StringA* ResidueTypeNames(void) const {
        return resTypeNames;
    }

    /**
     * Get the residue type name count.
     *
     * @return The residue type name count.
     */
    unsigned int ResidueTypeNameCount(void) const {
        return resTypeNameCnt;
    }

    /**
     * Get the atom count.
     *
     * @return The atom count.
     */
    unsigned int AtomTypeCount(void) const {
        return atomTypeCount;
    }

    /**
     * Get the atom types.
     *
     * @return The atom type array.
     */
    const AtomType* AtomTypes(void) const {
        return atomType;
    }

    /**
     * Get the number of connections (bonds).
     *
     * @return The connection count.
     */
    unsigned int ConnectionCount(void) const {
        return connectionCount;
    }

    /**
     * Get the connections (bonds).
     *
     * @return The connections.
     */
    const unsigned int* Connection(void) const {
        return connections;
    }

    /**
     * Get the number of molecules.
     *
     * @return The molecule count.
     */
    unsigned int MoleculeCount(void) const {
        return molCount;
    }

    /**
     * Get the molecules.
     *
     * @return The molecules.
     */
    const Molecule* Molecules(void) const {
        return molecules;
    }

    /**
     * Get the number of chains.
     *
     * @return The chain count.
     */
    unsigned int ChainCount(void) const {
        return chainCount;
    }

    /**
     * Get the chains.
     *
     * @return The chains.
     */
    const Chain* Chains(void) const {
        return chains;
    }

    /**
     * Get the filter information
     *
     * @return The filter information
     */
    const int* Filter(void) const {
        return atomFilter;
    }

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
    void SetAtoms(unsigned int atomCnt, unsigned int atomTypeCnt, const unsigned int* typeIdx, const float* pos,
        const AtomType* types, const int* residueIdx, float* bfactor, const float* charge, const float* occupancy);

    void SetAtomPositions(const float* atomPositions) {
        atomPos = atomPositions;
    }

    void SetAtomBFactors(float* bfac, bool ownsBFacMem = false) {
        if (this->ownsBFactorMemory)
            delete[] this->atomBFactors;
        this->atomBFactors = bfac;
        this->ownsBFactorMemory = ownsBFacMem;
    }

    void SetFormerAtomIndices(const int* formerAtomIndices) {
        this->atomFormerIdx = formerAtomIndices;
    }

    /**
     * Set the residues.
     *
     * @param resCnt    The residue count.
     * @param res       The residues.
     */
    void SetResidues(unsigned int resCnt, const Residue** res);

    /**
     * Set the residue type names.
     *
     * @param namesCnt  The residue type name count.
     * @param names     The residue type names.
     */
    void SetResidueTypeNames(unsigned int namesCnt, const vislib::StringA* names);

    /**
     * Set the connections (bonds).
     *
     * @param conCnt    The connection count.
     * @param con       The connections.
     */
    void SetConnections(unsigned int conCnt, const unsigned int* con);

    /**
     * Set the molecules.
     *
     * @param molCnt    The molecule count.
     * @param mol       The molecules.
     */
    void SetMolecules(unsigned int molCnt, const Molecule* mol);

    /**
     * Set the number of secondary structure elements.
     *
     * @param secStructCnt The secondary structure element count.
     */
    void SetSecondaryStructureCount(unsigned int cnt);

    /**
     * Set a secondary stucture element to the array.
     *
     * @param idx   The index of the element.
     * @param secS  The secondary structure element.
     *
     * @return 'true' if successful, 'false' otherwise.
     */
    bool SetSecondaryStructure(unsigned int idx, SecStructure secS);

    /**
     * Sets the position of the secondary structure to the molecule by
     * specifying the first secondary structure element index and the
     * size in number of secondary structure elements.
     *
     * @param molIdx    The index of the molecule.
     * @param firstSecS The index of the first sec struct elem.
     * @param secSCnt   The number of sec struct elems.
     */
    void SetMoleculeSecondaryStructure(unsigned int molIdx, unsigned int firstSecS, unsigned int secSCnt);

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
    void SetChains(unsigned int chainCnt, const Chain* chain);

    /**
     * Set the bfactor range.
     *
     * @param min    The minimum bfactor.
     * @param max    The minimum bfactor.
     */
    void SetBFactorRange(float min, float max) {
        this->minBFactor = min;
        this->maxBFactor = max;
    }

    /**
     * Get the minimum bfactor.
     *
     * @return The minimum bfactor value.
     */
    float MinimumBFactor() const {
        return this->minBFactor;
    }

    /**
     * Get the maximum bfactor.
     *
     * @return The maximum bfactor value.
     */
    float MaximumBFactor() const {
        return this->maxBFactor;
    }

    /**
     * Set the charge range.
     *
     * @param min    The minimum charge.
     * @param max    The minimum charge.
     */
    void SetChargeRange(float min, float max) {
        this->minCharge = min;
        this->maxCharge = max;
    }

    /**
     * Get the minimum charge.
     *
     * @return The minimum charge value.
     */
    float MinimumCharge() const {
        return this->minCharge;
    }

    /**
     * Get the maximum charge.
     *
     * @return The maximum charge value.
     */
    float MaximumCharge() const {
        return this->maxCharge;
    }

    /**
     * Set the occupancy range.
     *
     * @param min    The minimum occupancy.
     * @param max    The minimum occupancy.
     */
    void SetOccupancyRange(float min, float max) {
        this->minOccupancy = min;
        this->maxOccupancy = max;
    }

    /**
     * Get the minimum occupancy.
     *
     * @return The minimum occupancy value.
     */
    float MinimumOccupancy() const {
        return this->minOccupancy;
    }

    /**
     * Get the maximum occupancy.
     *
     * @return The maximum occupancy value.
     */
    float MaximumOccupancy() const {
        return this->maxOccupancy;
    }

    /**
     * Answer the calltime
     *
     * @return the calltime
     */
    float Calltime(void) const {
        return this->calltime;
    }

    /**
     * Sets the calltime to request data for.
     *
     * @param calltime The calltime to request data for.
     *
     */
    void SetCalltime(float calltime) {
        this->calltime = calltime;
    }

    /**
     * Get the neighborhood sizes.
     *
     * @return The neighborhood sizes.
     *         May be null if no neighboring information is available
     */
    const unsigned int* NeighborhoodSizes(void) const {
        return this->neighborhoodSizes;
    }

    /**
     * Sets the neighborhood sizes.
     *
     * @param neighborhoodSizes The list containing the neighborhoodSizes
     */
    void SetNeighborhoodSizes(const unsigned int* neighborhoodSizes) {
        this->neighborhoodSizes = neighborhoodSizes;
    }

    /**
     * Get the atom neighborhood indices.
     *
     * @return The neighborhood indices per atom.
     *         May be null if no neighboring information is available
     */
    const unsigned int** Neighborhoods(void) const {
        return this->neighborhoods;
    }

    /**
     * Sets the neighborhood indices.
     *
     * @param neighborhoods The neighborhood indices per atom.
     */
    void SetNeighborhoods(const unsigned int** neighborhoods) {
        this->neighborhoods = neighborhoods;
    }

    /**
     * Set the filter information
     *
     * @param atomFilter The filter information
     */
    void SetFilter(const int* atomFilter);

    inline MolecularDataCall& operator=(const MolecularDataCall& s) {
        AbstractGetData3DCall::operator=(s);
        this->SetFrameID(s.FrameID());
        this->SetDataHash(s.DataHash());
        this->atomCount = s.atomCount;
        this->atomFilter = s.atomFilter;
        this->atomPos = s.atomPos;
        this->atomTypeCount = s.atomTypeCount;
        this->atomTypeIdx = s.atomTypeIdx;
        this->atomResidueIdx = s.atomResidueIdx;
        this->atomHydrogenBondIdx = s.atomHydrogenBondIdx;
        this->atomHydrogenBondDistance = s.atomHydrogenBondDistance;
        this->atomHydrogenBondStatistics = s.atomHydrogenBondStatistics;
        this->atomHydrogenBondsFake = s.atomHydrogenBondsFake;
        this->solventResidueIdx = s.solventResidueIdx;
        this->atomSolventResCount = s.atomSolventResCount;
        this->atomType = s.atomType;
        this->ownsBFactorMemory = s.ownsBFactorMemory;
        if (!this->ownsBFactorMemory) {
            this->atomBFactors = s.atomBFactors;
        } else {
            this->atomBFactors = new float[this->atomCount];
            memcpy(this->atomBFactors, s.atomBFactors, sizeof(float) * this->atomCount);
        }
        this->atomCharges = s.atomCharges;
        this->atomOccupancies = s.atomOccupancies;
        this->minBFactor = s.minBFactor;
        this->maxBFactor = s.maxBFactor;
        this->minCharge = s.minCharge;
        this->maxCharge = s.maxCharge;
        this->minOccupancy = s.minOccupancy;
        this->maxOccupancy = s.maxOccupancy;
        this->connectionCount = s.connectionCount;
        this->connections = s.connections;
        this->resCount = s.resCount;
        this->residues = s.residues;
        this->resTypeNameCnt = s.resTypeNameCnt;
        this->resTypeNames = s.resTypeNames;
        this->molCount = s.molCount;
        this->molecules = s.molecules;
        this->chainCount = s.chainCount;
        this->chains = s.chains;
        this->secStruct = s.secStruct; // TODO: besser zeiger und anzahl ?!
        this->calltime = s.calltime;
        this->neighborhoods = s.neighborhoods;
        this->neighborhoodSizes = s.neighborhoodSizes;
        this->hydrogenBonds = s.hydrogenBonds;
        this->numHydrogenBonds = s.numHydrogenBonds;
        this->pdbFilename = s.pdbFilename;
        return *this;
    }

    /*
     */
    inline bool IsSolvent(const Residue* res) {
        // TODO: index tests here?
        int idx = res->MoleculeIndex();
        const Molecule& molecule = this->Molecules()[idx];
        const Chain& chain = this->Chains()[molecule.ChainIndex()];
        return chain.Type() == MolecularDataCall::Chain::SOLVENT;
    }

    /*
     * Sets the filename of the pdb file
     *
     * @param pdbFilename The filename of the pdb file
     */
    inline void SetPDBFilename(vislib::TString pdbFilename) {
        this->pdbFilename = pdbFilename;
    }

    /*
     * Returns the filename of the containing pdb file.
     */
    inline vislib::TString GetPDBFilename(void) const {
        return this->pdbFilename;
    }

private:
    // -------------------- variables --------------------

    /** The number of atoms. */
    unsigned int atomCount;
    /** The array of atom positions. */
    const float* atomPos;
    /** The array of atom type indices. */
    const unsigned int* atomTypeIdx;
    /** array of atom residue indices (may be undefined -> -1)*/
    const int* atomResidueIdx;
    /** marks wether an atom has a hydrogen bond to another H-atom (-1 marks "no connection") */
    const int* atomHydrogenBondIdx;
    float atomHydrogenBondDistance;
    const unsigned int* atomHydrogenBondStatistics;
    /** marks whether we use fake hydrogen bonds (calpha -> calpha) */
    bool atomHydrogenBondsFake;
    /** The former atom indices from the pdb file */
    const int* atomFormerIdx;

    const unsigned int* solventResidueIdx;
    unsigned int atomSolventResCount;

    /** The array of residues. */
    const Residue** residues;
    /** The number of residues. */
    unsigned int resCount;

    /** The array pf residue type names */
    const vislib::StringA* resTypeNames;
    /** The number of residue type names */
    unsigned int resTypeNameCnt;

    /** The array of molecules. */
    const Molecule* molecules;
    /** The number of molecules. */
    unsigned int molCount;

    /** The array of secondary structures */
    VISLIB_MSVC_SUPPRESS_WARNING(4251)
    vislib::Array<SecStructure> secStruct;

    /** The array of chains. */
    const Chain* chains;
    /** The number of chains. */
    unsigned int chainCount;

    /** The number of atom types. */
    unsigned int atomTypeCount;
    /** The array of atom types. */
    const AtomType* atomType;

    /** The total number of connections (bonds) */
    unsigned int connectionCount;
    /** The array of connections (bonds) of the atoms */
    const unsigned int* connections;

    /** The array of b-factors */
    float* atomBFactors;
    /** Flag whether the call owns the B-factor memory */
    bool ownsBFactorMemory;
    /** The minimum bfactor */
    float minBFactor;
    /** The maximum bfactor */
    float maxBFactor;

    /** The array of charges */
    const float* atomCharges;
    /** The minimum charges */
    float minCharge;
    /** The maximum charges */
    float maxCharge;

    /** The array of occupancies */
    const float* atomOccupancies;
    /** The minimum occupancies */
    float minOccupancy;
    /** The maximum occupancies */
    float maxOccupancy;

    /** The exact requested/stored calltime. */
    float calltime;

    /** Filter information for all atoms */
    const int* atomFilter;

    /** List of all sizes of the atom neighborhoods */
    const unsigned int* neighborhoodSizes;

    /** The atom neighborhoods as 2D-array. first dim: neighborhood per atom, second dim: atom index of the j-th
     * neighbor*/
    const unsigned int** neighborhoods;

    /** The hydrogen bond array. */
    const unsigned int* hydrogenBonds;

    /** The number of hydrogen bonds in the hydrogen bonds array */
    unsigned int numHydrogenBonds;

    /** The name of the pdb file the data is from */
    VISLIB_MSVC_SUPPRESS_WARNING(4251)
    vislib::TString pdbFilename;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<MolecularDataCall> MolecularDataCallDescription;


} /* end namespace protein_calls */
} /* end namespace megamol */

#endif /* MEGAMOL_PROTEIN_CALL_MOLECULARDATACALL_H_INCLUDED */
