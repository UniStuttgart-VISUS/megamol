//
// PerAtomFloatCall.h
//
// Copyright (C) 2014 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MMMOLMAPPLG_PERATOMFLOATCALL_H_INCLUDED
#define MMMOLMAPPLG_PERATOMFLOATCALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "MolecularDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol {
namespace protein_calls {

/**
 * Call that transports one float per atom
 */
class PerAtomFloatCall : public core::Call {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "PerAtomFloatCall";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Call for sending an array with one float value per atom";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return 1;
    }

    /** Index of the 'GetFloat' function */
    static const unsigned int CallForGetFloat = 0;

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
            return "GetFloat";
        default:
            return NULL;
        }
    }

    /** Ctor */
    PerAtomFloatCall(void) : data(0), frameID(0) {}

    /** Dtor */
    virtual ~PerAtomFloatCall(void) {}

    const float* GetFloat() const {
        return data.PeekElements();
    }

    unsigned int Count(void) const {
        return static_cast<unsigned int>(data.Count());
    }

    void SetData(vislib::Array<float> dat) {
        this->data = dat;
    }

    unsigned int FrameID(void) const {
        return this->frameID;
    }

    void SetFrameID(unsigned int fID) {
        this->frameID = fID;
    }

    float MinValue(void) const {
        return this->minValue;
    }

    void SetMinValue(float val) {
        this->minValue = val;
    }

    float MidValue(void) const {
        return this->midValue;
    }

    void SetMidValue(float val) {
        this->midValue = val;
    }

    float MaxValue(void) const {
        return this->maxValue;
    }

    void SetMaxValue(float val) {
        this->maxValue = val;
    }

private:
    /** The float array for the given molecule. */
    vislib::Array<float> data;
    /** The index of the frame */
    unsigned int frameID;
    /** minimum value */
    float minValue;
    /** middle value */
    float midValue;
    /** maximum value */
    float maxValue;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<PerAtomFloatCall> PerAtomFloatCallDescription;

} // namespace protein_calls
} // end namespace megamol

#endif // MMMOLMAPPLG_PERATOMFLOATCALL_H_INCLUDED
