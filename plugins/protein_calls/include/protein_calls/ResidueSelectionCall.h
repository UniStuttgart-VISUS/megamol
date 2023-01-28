/*
 * ResidueSelectionCall.h
 *
 * Author: Daniel Kauker & Michael Krone
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MEGAMOL_PROTEIN_CALLS_RESIDUE_SELECTIONCALL_H_INCLUDED
#define MEGAMOL_PROTEIN_CALLS_RESIDUE_SELECTIONCALL_H_INCLUDED
#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/Array.h"

namespace megamol::protein_calls {

/**
 * Base class for graph calls and data interfaces.
 *
 * Graphs based on coordinates can contain holes where the respective
 * getters return false for the abscissae. For categorical graphs this
 * seems useless as the abscissa is sparse anyway, but the interface
 * allows for that as well.
 */

class ResidueSelectionCall : public megamol::core::Call {
public:
    struct Residue {
        int id;
        int resNum;
        char chainID;

        bool operator==(Residue const& rhs) {
            return this->id == rhs.id && this->resNum == rhs.resNum && this->chainID == rhs.chainID;
        }
    };

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "ResidueSelectionCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call to get selection Residues";
    }

    /** Index of the 'GetData' function */
    static const unsigned int CallForGetSelection;

    static const unsigned int CallForSetSelection;

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
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
            return "getSelection";
        case 1:
            return "setSelection";
        }
        return "";
    }

    inline vislib::Array<Residue>* GetSelectionPointer() const {
        return this->selection;
    }

    inline void SetSelectionPointer(vislib::Array<Residue>* selection) {
        this->selection = selection;
    }

    ResidueSelectionCall();
    ~ResidueSelectionCall() override;

private:
    vislib::Array<Residue>* selection;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<ResidueSelectionCall> ResidueSelectionCallDescription;

} // namespace megamol::protein_calls

#endif /* MEGAMOL_PROTEIN_CALLS_RESIDUE_SELECTIONCALL_H_INCLUDED */
