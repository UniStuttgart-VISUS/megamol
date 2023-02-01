/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/String.h"
#include "vislib/math/Vector.h"

// Thresholds values for STRIDE and DSSP:
// [STRIDE - stride.c line: 311ff]
#define STRIDE_THRESHOLDH1 -230.00f
#define STRIDE_THRESHOLDH3 0.12f
#define STRIDE_THRESHOLDH4 0.06f
#define STRIDE_THRESHOLDE1 -240.00f // anti-parallel
#define STRIDE_THRESHOLDE2 -310.00f // parallel

#define DSSP_HBENERGY -0.5f

#define PROSIGN_ALPHA 0.3f
#define PROSIGN_BETA 0.75f

namespace megamol::protein_calls {
class UncertaintyDataCall : public megamol::core::Call {
public:
    /**
     * Enumeration of secondary structure types.
     *
     * (!) Indices must be in the range from 0 to NOE-1.
     */
    enum secStructure {
        G_310_HELIX = 0,
        T_H_TURN = 1,
        H_ALPHA_HELIX = 2,
        I_PI_HELIX = 3,
        S_BEND = 4,
        C_COIL = 5,
        B_BRIDGE = 6,
        E_EXT_STRAND = 7,
        NOTDEFINED = 8,
        NOE = 9 // Number of Elements -> must always be the last index!
    };

    /**
     * Enumeration of assignment methods.
     *
     * (!) Indices must be in the range from 0 to NOM-1.
     */
    enum assMethod {
        PDB = 0,
        STRIDE = 1,
        DSSP = 2,
        PROSIGN = 3,
        // Add new methods here (before UNCERTAINTY index !!!)
        UNCERTAINTY = 4,
        NOM = 5 // Number of Methods -> must always be the last index!
    };

    /**
     * Enumeration of additional residue flags.
     */
    enum addFlags { NOTHING = 0, MISSING = 1, HETEROGEN = 2 };

    /**
     * Enumeration of pdb secondary structure assignment methods.
     */
    enum pdbAssMethod { PDB_PROMOTIF = 0, PDB_AUTHOR = 1, PDB_DSSP = 2, PDB_UNKNOWN = 3 };

    // ------------------ class functions -------------------

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "UncertaintyDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call to get uncertaintay data.";
    }

    /** Index of the 'GetData' function */
    static const unsigned int CallForGetData;

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
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

    /** CTOR */
    UncertaintyDataCall();

    /** DTOR */
    ~UncertaintyDataCall() override;


    // ------------------ GET functions -------------------

    /**
     * Get the number of amino-acids.
     *
     * @return The amino-acid count.
     */
    inline unsigned int GetAminoAcidCount() const {
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
    inline vislib::StringA GetAminoAcidThreeLetterCode(unsigned int i) const {
        if (!this->aminoAcidName)
            return static_cast<vislib::StringA>("");
        else if (this->aminoAcidName->Count() <= i)
            return static_cast<vislib::StringA>("");
        else
            return (this->aminoAcidName->operator[](i));
    }

    /**
     * Get the amino-acid in one letter code.
     *
     * @return The amino-acid one letter code.
     */
    inline char GetAminoAcidOneLetterCode(unsigned int i) {
        if (!this->aminoAcidName)
            return static_cast<char>(' ');
        else if (this->aminoAcidName->Count() <= i)
            return static_cast<char>(' ');
        else
            return (this->AminoacidThreeToOneLetterCode(this->aminoAcidName->operator[](i)));
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
     * Get the the secondary structure uncertainty.
     *
     * @param m The assignment method.
     * @param i The index of the amino-acid.
     * @return The array of secondary structure uncertainty.
     */
    inline vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)>
    GetSecStructUncertainty(assMethod m, unsigned int i) const {
        vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> ret;
        int index = static_cast<int>(m);
        for (int x = 0; x < static_cast<int>(secStructure::NOE); x++) {
            ret[x] = 0.0f;
        }
        if (!this->secStructUncertainty)
            return ret;
        else if (static_cast<int>(this->secStructUncertainty->Count()) <= index)
            return ret;
        else if (this->secStructUncertainty->operator[](index).Count() <= i)
            return ret;
        else
            return (this->secStructUncertainty->operator[](index)[i]);
    }

    /**
     * Get the sorted secondary structure types.
     *
     * @param m The assignment method.
     * @param i The index of the amino-acid.
     * @return The sorted secondary structure types.
     */
    inline vislib::math::Vector<UncertaintyDataCall::secStructure,
        static_cast<int>(UncertaintyDataCall::secStructure::NOE)>
    GetSortedSecStructAssignment(assMethod m, unsigned int i) const {
        vislib::math::Vector<UncertaintyDataCall::secStructure,
            static_cast<int>(UncertaintyDataCall::secStructure::NOE)>
            ret;
        int index = static_cast<int>(m);
        for (int x = 0; x < static_cast<int>(secStructure::NOE); x++) {
            ret[x] = UncertaintyDataCall::secStructure::NOTDEFINED;
        }
        if (!this->sortedSecStructAssignment)
            return ret;
        else if (static_cast<int>(this->sortedSecStructAssignment->Count()) <= index)
            return ret;
        else if (this->sortedSecStructAssignment->operator[](index).Count() <= i)
            return ret;
        else
            return (this->sortedSecStructAssignment->operator[](index)[i]);
    }

    /**
     * Get the PDB ID.
     *
     * @return The pdb id.
     */
    inline vislib::StringA GetPdbID() {
        return *this->pdbID;
    }

    /**
     * Get the flag indicating if uncertainty has been recalculated.
     *
     * @return The unceratinty recalculation flag.
     */
    inline bool GetRecalcFlag() {
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
     */
    vislib::math::Vector<float, 4> GetSecStructColor(UncertaintyDataCall::secStructure s);

    /**
     * Get the pdb assignment method for helix.
     *
     * @return The pdb assignment method for helix.
     */
    inline UncertaintyDataCall::pdbAssMethod GetPdbAssMethodHelix() {
        return *this->pdbAssignmentHelix;
    }

    /**
     * Get the pdb assignment method for sheet.
     *
     * @return The pdb assignment method for sheet.
     */
    inline UncertaintyDataCall::pdbAssMethod GetPdbAssMethodSheet() {
        return *this->pdbAssignmentSheet;
    }

    /**
     * Get the uncertainty.
     *
     * @param i The index of the amino-acid.
     * @return The uncertainty value.
     */
    inline float GetUncertainty(unsigned int i) const {
        if (!this->uncertainty)
            return 0.0f;
        else if (this->uncertainty->Count() <= i)
            return 0.0f;
        else
            return (this->uncertainty->operator[](i));
    }


    /**
     * Get the STRIDE threshold values.
     *
     * @param i The index of the amino-acid.
     * @return The threshold values.
     */
    inline vislib::math::Vector<float, 7> GetStrideThreshold(unsigned int i) {
        vislib::math::Vector<float, 7> ret;
        for (unsigned int j = 0; j < 7; j++) {
            ret[j] = 0.0f;
        }
        if (!this->strideStructThreshold)
            return ret;
        else if (this->strideStructThreshold->Count() <= i)
            return ret;
        else
            return (this->strideStructThreshold->operator[](i));
    }

    /**
     * Get the DSSP energy values.
     *
     * @param i The index of the amino-acid.
     * @return The energy values.
     */
    inline vislib::math::Vector<float, 4> GetDsspEnergy(unsigned int i) {
        vislib::math::Vector<float, 4> ret = {0.0f, 0.0f, 0.0f, 0.0f};
        if (!this->dsspStructEnergy)
            return ret;
        else if (this->dsspStructEnergy->Count() <= i)
            return ret;
        else
            return (this->dsspStructEnergy->operator[](i));
    }

    /**
     * Get the PROSIGN threshold values.
     *
     * @param i The index of the amino-acid.
     * @return The energy values.
     */
    inline vislib::math::Vector<float, 6> GetProsignThreshold(unsigned int i) {
        vislib::math::Vector<float, 6> ret;
        for (unsigned int j = 0; j < 6; j++) {
            ret[j] = 1.0f;
        }
        ret[4] = 0.3f;
        ret[5] = 0.75f;
        if (!this->prosignStructThreshold)
            return ret;
        else if (this->prosignStructThreshold->Count() <= i)
            return ret;
        else
            return (this->prosignStructThreshold->operator[](i));
    }


    // ------------------ SET functions -------------------

    /**
     * Set the pointer to the pdb index.
     *
     * @param rnPtr The pointer.
     */
    inline void SetPdbIndex(vislib::Array<vislib::StringA>* rnPtr) {
        this->pdbIndex = rnPtr;
    }

    /**
     * Set the pointer to the 3-letter amino-acid name.
     *
     * @param rnPtr The pointer.
     */
    inline void SetAminoAcidName(vislib::Array<vislib::StringA>* rnPtr) {
        this->aminoAcidName = rnPtr;
    }

    /**
     * Set the pointer to the chain ID.
     *
     * @param rnPtr The pointer.
     */
    inline void SetChainID(vislib::Array<char>* rnPtr) {
        this->chainID = rnPtr;
    }

    /**
     * Set the pointer to the residue flag.
     *
     * @param rnPtr The pointer.
     */
    inline void SetResidueFlag(vislib::Array<addFlags>* rnPtr) {
        this->residueFlag = rnPtr;
    }

    /**
     * Set the pointer to the secondary structure uncertainty.
     *
     * @param rnPtr The pointer.
     */
    inline void SetSecStructUncertainty(vislib::Array<
        vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)>>>* rnPtr) {
        this->secStructUncertainty = rnPtr;
    }

    /**
     * Set the pointer to the sorted secondary structure types.
     *
     * @param rnPtr The pointer.
     */
    inline void SetSortedSecStructAssignment(
        vislib::Array<vislib::Array<vislib::math::Vector<UncertaintyDataCall::secStructure,
            static_cast<int>(UncertaintyDataCall::secStructure::NOE)>>>* rnPtr) {
        this->sortedSecStructAssignment = rnPtr;
    }

    /**
     * Set the PDB ID.
     *
     * @param rnPtr The pointer to the pdb id.
     */
    inline void SetPdbID(vislib::StringA* rnPtr) {
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
    inline void SetPdbAssMethodHelix(UncertaintyDataCall::pdbAssMethod* rnPtr) {
        this->pdbAssignmentHelix = rnPtr;
    }

    /**
     * Set the pdb assignment method for sheet.
     *
     * @param rnPtr The pointer to the pdb assignment method for sheet.
     */
    inline void SetPdbAssMethodSheet(UncertaintyDataCall::pdbAssMethod* rnPtr) {
        this->pdbAssignmentSheet = rnPtr;
    }

    /**
     * Set the pointer to the uncertainty.
     *
     * @param rnPtr The pointer.
     */
    inline void SetUncertainty(vislib::Array<float>* rnPtr) {
        this->uncertainty = rnPtr;
    }

    /**
     * Set the pointer to the STRIDE threshold values.
     *
     * @param rnPtr The pointer.
     */
    inline void SetStrideThreshold(vislib::Array<vislib::math::Vector<float, 7>>* rnPtr) {
        this->strideStructThreshold = rnPtr;
    }

    /**
     * Set the pointer to the DSSP energy values.
     *
     * @param rnPtr The pointer.
     */
    inline void SetDsspEnergy(vislib::Array<vislib::math::Vector<float, 4>>* rnPtr) {
        this->dsspStructEnergy = rnPtr;
    }

    /**
     * Set the pointer to the PROSIGN threshold values.
     *
     * @param rnPtr The pointer.
     */
    inline void SetProsignThreshold(vislib::Array<vislib::math::Vector<float, 6>>* rnPtr) {
        this->prosignStructThreshold = rnPtr;
    }


private:
    /**
     * Returns the single letter code for an amino acid given the three letter code.
     *
     * @param resName The name of the residue as three letter code.
     * @return The single letter code for the amino acid.
     */
    char AminoacidThreeToOneLetterCode(vislib::StringA resName);

    // ------------------ variables -------------------


    /** Pointer to the PDB index */
    vislib::Array<vislib::StringA>* pdbIndex;

    /** Pointer to the chain ID */
    vislib::Array<char>* chainID;

    /** Pointer to the flag giving additional information */
    vislib::Array<addFlags>* residueFlag;

    /** Pointer to the amino-acid name */
    vislib::Array<vislib::StringA>* aminoAcidName;


    /** Pointer to the uncertainty of the assigned secondary structure types for each assignment method and for each
     * amino-acid */
    vislib::Array<vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)>>>*
        secStructUncertainty;
    /** Pointer to the sorted assigned secondary structure types (sorted by descending uncertainty values) for each
     * assignment method and for each amino-acid */
    vislib::Array<vislib::Array<vislib::math::Vector<UncertaintyDataCall::secStructure,
        static_cast<int>(UncertaintyDataCall::secStructure::NOE)>>>* sortedSecStructAssignment;


    /** Pointer to the uncertainty of secondary structure types */
    vislib::Array<float>* uncertainty;

    /** Flag indicating that uncertainty was recalculated */
    bool recalcUncertainty;

    /** Flag indicatin whether this is a time accumulation */
    bool isTimeAccumulation;

    /** Number of timesteps */
    int timestepNumber;

    /** The PDB ID */
    vislib::StringA* pdbID;

    /** The pdb assignment method for helix */
    UncertaintyDataCall::pdbAssMethod* pdbAssignmentHelix;

    /** The pdb assignment method for helix */
    UncertaintyDataCall::pdbAssMethod* pdbAssignmentSheet;

    /** The pointer to the 5 STRIDE threshold values per amino-acid */
    vislib::Array<vislib::math::Vector<float, 7>>* strideStructThreshold;
    /** The pointer to the 4 DSSP energy values per amino-acid */
    vislib::Array<vislib::math::Vector<float, 4>>* dsspStructEnergy;
    /** The pointer to the 4 PROSIGN values per amino acid*/
    vislib::Array<vislib::math::Vector<float, 6>>* prosignStructThreshold;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<UncertaintyDataCall> UncertaintyDataCallDescription;

} // namespace megamol::protein_calls
