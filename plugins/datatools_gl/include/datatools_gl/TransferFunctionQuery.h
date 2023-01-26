/*
 * TransferFunctionQuery.h
 *
 * Copyright (C) 2014 by CGV TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_TRANSFERFUNCTIONQUERY_H_INCLUDED
#define MEGAMOLCORE_TRANSFERFUNCTIONQUERY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "vislib/RawStorage.h"


namespace megamol {
namespace datatools_gl {


/**
 * In-Between management module to change time codes of a data set
 */
class TransferFunctionQuery {
public:
    /** Ctor. */
    TransferFunctionQuery();

    /** Dtor. */
    ~TransferFunctionQuery();

    /**
     * Answer the slot for the transfer function call
     *
     * @return The slot for the transfer function
     */
    inline core::CallerSlot* GetSlot() {
        return &this->getTFSlot;
    }

    /**
     * Clears the transfer function data
     */
    inline void Clear() {
        this->texDat.EnforceSize(0);
        this->texDatSize = 0;
    }

    /**
     * Queries the transfer function
     *
     * @param col Points to four floats receiving the RGBA value
     * @param val The value to query
     */
    void Query(float* col, float val);

private:
    /** The call for Transfer function */
    core::CallerSlot getTFSlot;

    /** The transfer function raw data */
    vislib::RawStorage texDat;

    /** The size of the transfer function */
    unsigned int texDatSize;
};

} // namespace datatools_gl
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TRANSFERFUNCTIONQUERY_H_INCLUDED */
