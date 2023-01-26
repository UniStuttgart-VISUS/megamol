/*
 * CrystalStructureDataCall.h
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id: CrystalStructureDataCall.h 1443 2015-07-08 12:18:12Z reina $
 */

#ifndef MMPROTEINCALLPLUGIN_CRYSTALSTRCUTUREDATACALL_H_INCLUDED
#define MMPROTEINCALLPLUGIN_CRYSTALSTRCUTUREDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetData3DCall.h"

namespace megamol {
namespace protein_calls {

/**
 * Class providing data call for CrystalStructureDataSource.
 * Note: Arrays can be NULL at any time.
 */
class CrystalStructureDataCall : public megamol::core::AbstractGetData3DCall {
public:
    /** Possible atom types */
    enum AtomType { GENERIC, BA, TI, O };

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
        return "CrystalStructureDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call to get dynamic particle data in a crystal structure.";
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
     * @param[in] idx The index of the function to return it's name.
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
    CrystalStructureDataCall(void);

    /** Dtor. */
    ~CrystalStructureDataCall(void) override;

    /**
     * Answers the atom count.
     *
     * @return The atom count.
     */
    unsigned int GetAtomCnt() const {
        return this->atomCnt;
    }

    /**
     * Answers the dipole count.
     *
     * @return The dipole count.
     */
    unsigned int GetDipoleCnt() const {
        return this->dipoleCnt;
    }

    /**
     * Answers the number of edges.
     *
     * @return The number of edges
     */
    unsigned int GetConCnt() const {
        return this->conCnt;
    }

    /**
     * Getter for atom type array.
     *
     * @return Array with atom types
     */
    const AtomType* GetAtomType() const {
        return this->atomType;
    }

    /**
     * Get the atom positions.
     *
     * @return The array with the atom positions
     */
    const float* GetAtomPos() const {
        return this->atomPos;
    }

    /**
     * Get the dipole positions.
     *
     * @return The dipole positions
     */
    const float* GetDipolePos() const {
        return this->dipolePos;
    }

    /**
     * Get the dipole array.
     *
     * @return The dipole array
     */
    const float* GetDipole() const {
        return this->dipole;
    }

    /**
     * Getter for atom connections.
     *
     * @return The atom connectivity array.
     */
    const int* GetAtomCon() const {
        return this->atomCon;
    }

    /**
     * Getter for the cell array.
     *
     * @return The cell array.
     */
    const int* GetCells() const {
        return this->cells;
    }

    /**
     * Answers the cell count.
     *
     * @return The cell count.
     */
    unsigned int GetCellCnt() const {
        return this->cellCnt;
    }

    /**
     * Answer the calltime
     *
     * @return the calltime
     */
    float GetCalltime(void) const {
        return this->calltime;
    }

    /**
     * Sets the pointer to the current frame.
     *
     * @param[in] atomPos  Pointer to the atom positions
     * @param[in] atomType The array with the atom types
     * @param[in] atomCnt  The number of atoms
     */
    void SetAtoms(const float* atomPos, const AtomType* atomType, unsigned int atomCnt) {
        this->atomPos = atomPos;
        this->atomCnt = atomCnt;
        this->atomType = atomType;
    }

    /**
     * Sets the atom connection array.
     *
     * @param[in] atomCon The array with the atom connections
     * @param[in] conCnt  The number of connections
     */
    void SetAtomCon(const int* atomCon, unsigned int conCnt) {
        this->atomCon = atomCon;
        this->conCnt = conCnt;
    }

    /**
     * Sets the cell array and the number of cells.
     *
     * @param[in] cells The array with the celldata.
     * @param[in] cellCnt The number of cells.
     */
    void SetCells(const int* cells, unsigned int cellCnt) {
        this->cells = cells;
        this->cellCnt = cellCnt;
    }

    /**
     * Sets the calltime to request data for.
     *
     * @param[in] calltime The calltime to request data for.
     */
    void SetCalltime(float calltime) {
        this->calltime = calltime;
    }

    /**
     * Set the dipoles.
     *
     * @param[in] dipolePos The Array with the dipole positions
     * @param[in] dipole    The array with the dipoles
     * @param[in] dipoleCnt The number of dipoles
     */
    void SetDipoles(const float* dipolePos, const float* dipole, unsigned int dipoleCnt) {
        this->dipoleCnt = dipoleCnt;
        this->dipolePos = dipolePos;
        this->dipole = dipole;
    }

private:
    /// The numbr of atoms
    unsigned int atomCnt;

    /// The dipole count
    unsigned int dipoleCnt;

    /// The number of edges
    unsigned int conCnt;

    /// The number of cells
    unsigned int cellCnt;

    /// The exact requested/stored calltime
    float calltime;


    // Arrays

    /// The array of atom positions
    const float* atomPos;

    /// The array of atom types
    const CrystalStructureDataCall::AtomType* atomType;

    /// Atom connections
    const int* atomCon;

    /// The array with the cell definition
    const int* cells;

    /// The dipole positions
    const float* dipolePos;

    /// The dipoles
    const float* dipole;
};

/** Description class typedef */
typedef core::factories::CallAutoDescription<CrystalStructureDataCall> CrystalStructureDataCallDescription;

} /* end namespace protein_calls */
} /* end namespace megamol */

#endif /* MMPROTEINCALLPLUGIN_CRYSTALSTRCUTUREDATACALL_H_INCLUDED */
