//
// VariantMatchDataCall.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 15, 2013
//     Author: scharnkn
//

#ifndef MEGAMOL_PROTEIN_CALL_VARIANTMATCHDATACALL_H_INCLUDED
#define MEGAMOL_PROTEIN_CALL_VARIANTMATCHDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/macro_utils.h"
#include "vislib/math/Cuboid.h"

namespace megamol {
namespace protein_calls {

class VariantMatchDataCall : public core::Call {

public:
    /// Index of the 'CallForGetData' function
    static const unsigned int CallForGetData;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "VariantMatchDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call to transmit variant match data";
    }

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
            return "getData";
        }
        return "";
    }

    /** Ctor. */
    VariantMatchDataCall(void);

    /** Dtor. */
    virtual ~VariantMatchDataCall(void);

    /**
     * TODO
     */
    inline const vislib::StringA* GetLabels() {
        return this->labels;
    }

    /**
     * TODO
     */
    inline const float* GetMatch() {
        return this->match;
    }

    /**
     * TODO
     */
    inline float GetMax() {
        return this->max;
    }

    /**
     * TODO
     */
    inline float GetMin() {
        return this->min;
    }

    /**
     * TODO
     */
    inline unsigned int GetVariantCnt() const {
        return this->variantCnt;
    }

    /**
     * TODO
     */
    void SetLabels(const vislib::StringA* labels) {
        this->labels = labels;
    }

    /**
     * TODO
     */
    void SetMatch(const float* match) {
        this->match = match;
    }

    /**
     * TODO
     */
    void SetMatchRange(float min, float max) {
        this->min = min;
        this->max = max;
    }

    /**
     * TODO
     */
    void SetVariantCnt(unsigned int variantCnt) {
        this->variantCnt = variantCnt;
    }

protected:
private:
    /// The number of variants
    unsigned int variantCnt;

    /// The labels of all variants
    const vislib::StringA* labels;

    /// The matching value matrix
    const float* match;

    /// The maximum match value
    float max;

    /// The minimum match value
    float min;
};

/// Description class typedef
typedef core::factories::CallAutoDescription<VariantMatchDataCall> VariantMatchDataCallDescription;


} // end namespace protein_calls
} // end namespace megamol

#endif // MMPROTEINCALLSPLUGIN_VARIANTMATCHDATACALL_H_INCLUDED
