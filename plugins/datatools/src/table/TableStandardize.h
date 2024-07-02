/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "datatools/table/TableDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol::datatools::table {

/*
 * Module to manipulate table (copy) via a LUA script.
 */
class TableStandardize : public core::Module {
public:
    static std::string ModuleName;
    static std::string defaultScript;

    /** Return module class name */
    static const char* ClassName() {
        return ModuleName.c_str();
    }

    /** Return module class description */
    static const char* Description() {
        return "Standardize table data (copy). Helps dimensionality reduction and clustering by reducing bias towards "
               "large values.";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    TableStandardize();

    /** Dtor */
    ~TableStandardize() override;

protected:
    /** Lazy initialization of the module */
    bool create() override;

    /** Resource release */
    void release() override;

private:
    /** Data callback */
    bool processData(core::Call& c);

    bool getExtent(core::Call& c);

    /** Data output slot */
    core::CalleeSlot dataOutSlot;

    /** Data output slot */
    core::CallerSlot dataInSlot;

    /** Parameter slot for strategy selection */
    core::param::ParamSlot stratSlot;

    // inspired by SciKit learn
    enum Strategies { OFF, STANDARD, MINMAX, MAXABS, ROBUST };

    /** ID of the current frame */
    int frameID;

    /** Hash of the current data */
    size_t in_datahash, out_datahash;

    /** Vector storing the actual float data */
    std::vector<float> data;

    /** Vector storing information about columns */
    std::vector<TableDataCall::ColumnInfo> info;

    /** number of columns coming in */
    int column_count = 0;

    /** info of about columns coming in */
    const TableDataCall::ColumnInfo* column_infos = nullptr;

    /** number of rows coming in */
    int row_count = 0;

    /** the data coming in */
    const float* in_data = nullptr;

}; /* end class TableStandardize */

} // namespace megamol::datatools::table
