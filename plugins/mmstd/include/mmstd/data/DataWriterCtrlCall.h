/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"

namespace megamol::core {

/**
 * Call to control data writer modules
 */
class DataWriterCtrlCall : public Call {
public:
    /** Call to run the writing process */
    static const unsigned int CALL_RUN = 0;

    /** Call to query the writers capabilities */
    static const unsigned int CALL_GETCAPABILITIES = 1;

    /** Call to request an abort */
    static const unsigned int CALL_ABORT = 2;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "DataWriterCtrlCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call to control a data writer";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return 3;
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
        case CALL_RUN:
            return "run";
        case CALL_GETCAPABILITIES:
            return "getCapabilities";
        case CALL_ABORT:
            return "abort";
        default:
            return NULL;
        }
    }

    /** Ctor. */
    DataWriterCtrlCall();

    /** Dtor. */
    ~DataWriterCtrlCall() override;

    /**
     * Answer the abortable capability flag
     *
     * @return The abortable capability flag
     */
    inline bool IsAbortable() const {
        return this->abortable;
    }

    /**
     * Sets the abortable capability flag
     *
     * @param abortable The new value for the abortable capability flag
     */
    inline void SetAbortable(bool abortable) {
        this->abortable = abortable;
    }

    /**
     * Assignment operator.
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    DataWriterCtrlCall& operator=(const DataWriterCtrlCall& rhs);

private:
    /** Flag indicate the capability of being abortable */
    bool abortable;
};

} // namespace megamol::core
