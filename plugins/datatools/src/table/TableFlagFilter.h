/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "datatools/table/TableDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol::datatools::table {

/*
 * Module to filter rows from a table based on a flag storage.
 */
class TableFlagFilter : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "TableFlagFilter";
    }

    /** Return module class description */
    static const char* Description() {
        return "Filters rows from a table based on a flag storage.";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    TableFlagFilter();

    /** Dtor */
    ~TableFlagFilter() override;

protected:
    bool create() override;

    void release() override;

    bool getData(core::Call& call);

    bool getHash(core::Call& call);

    bool handleCall(core::Call& call);

private:
    enum FilterMode { FILTERED = 0, SELECTED = 1 };

    core::CallerSlot tableInSlot;
    core::CallerSlot flagStorageInSlot;
    core::CalleeSlot tableOutSlot;

    core::param::ParamSlot filterModeParam;

    // input table properties
    unsigned int tableInFrameCount;
    size_t tableInDataHash;
    size_t tableInColCount;

    // filtered table
    size_t dataHash;
    size_t rowCount;
    std::vector<datatools::table::TableDataCall::ColumnInfo> colInfos;
    std::vector<float> data;
};

} // namespace megamol::datatools::table
