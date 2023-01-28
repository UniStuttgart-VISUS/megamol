/*
 * TableProcessorBase.h
 *
 * Copyright (C) 2019 Visualisierungsinstitut der Universität Stuttgart
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "datatools/table/TableDataCall.h"


namespace megamol::datatools::table {

/**
 * A base class for modules processing table data.
 */
class TableProcessorBase : public core::Module {

public:
    /**
     * Finalises an instance.
     */
    ~TableProcessorBase() override = default;

protected:
    typedef megamol::datatools::table::TableDataCall::ColumnInfo ColumnInfo;

    /**
     * Initialises a new instance.
     */
    TableProcessorBase();

    /**
     * Computes the combined hash from the hash of the local state and the
     * hash of the input
     *
     * @return The hash of the data currently stored in the module.
     */
    inline std::size_t getHash() {
        auto retval = this->inputHash;
        retval ^= this->localHash + 0x9e3779b9 + (retval << 6) + (retval >> 2);
        return retval;
    }

    /**
     * Prepares the data requested by 'call'.
     *
     * @param src     The call providing the data.
     * @param frameID The ID of the frame requested by the caller.
     *
     * @return true in case of suceess, false otherwise.
     */
    virtual bool prepareData(TableDataCall& src, const unsigned int frameID) = 0;

    /** Holds the columns of the (filtered) table. */
    std::vector<ColumnInfo> columns;

    /** Holds the ID of the current frame. */
    unsigned int frameID;

    /** Holds the hash of the data as reported by the input module. */
    std::size_t inputHash;

    /** Holds a hash representing the current state of the processor. */
    std::size_t localHash;

    /** The slot providing the input data. */
    core::CallerSlot slotInput;

    /** The slot allowing for retrieval of the output data. */
    core::CalleeSlot slotOutput;

    /** The actual values. */
    std::vector<float> values;

private:
    bool getData(core::Call& call);

    bool getHash(core::Call& call);
};

} // namespace megamol::datatools::table
