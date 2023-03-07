//
// PerAtomFloatCall.h
//
// Copyright (C) 2014 by University of Stuttgart (VISUS).
// All rights reserved.
//

#pragma once

#include "MolecularDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::protein_calls {

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
    static const char* ClassName() {
        return "PerAtomFloatCall";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Call for sending an array with one float value per atom";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
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
    PerAtomFloatCall() : data(0), frameID(0) {}

    /** Dtor */
    ~PerAtomFloatCall() override {}

    const float* GetFloat() const {
        return data.PeekElements();
    }

    unsigned int Count() const {
        return static_cast<unsigned int>(data.Count());
    }

    void SetData(vislib::Array<float> dat) {
        this->data = dat;
    }

    unsigned int FrameID() const {
        return this->frameID;
    }

    void SetFrameID(unsigned int fID) {
        this->frameID = fID;
    }

    float MinValue() const {
        return this->minValue;
    }

    void SetMinValue(float val) {
        this->minValue = val;
    }

    float MidValue() const {
        return this->midValue;
    }

    void SetMidValue(float val) {
        this->midValue = val;
    }

    float MaxValue() const {
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

} // namespace megamol::protein_calls
