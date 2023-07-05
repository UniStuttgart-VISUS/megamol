/*
 * TableJoin.h
 *
 * Copyright (C) 2016-2016 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "datatools/table/TableDataCall.h"

namespace megamol::datatools::table {

/**
 * This module joins two tables by copying the values together into one matrix
 */
class TableJoin : public core::Module {
public:
    static std::string ModuleName;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return ModuleName.c_str();
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Joins two tables (union of columns)";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    TableJoin();

    /**
     * Finalises an instance.
     */
    ~TableJoin() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    /** data callback */
    bool processData(core::Call& c);

    /** extent callback */
    bool getExtent(core::Call& c);

    /** concatenates two tables */
    static void concatenate(float* const out, const size_t rowCount, const size_t columnCount, const float* const first,
        const size_t firstRowCount, const size_t firstColumnCount, const float* const second,
        const size_t secondRowCount, const size_t secondColumnCount);

    /** input slot of first table */
    core::CallerSlot firstTableInSlot;

    /** output slot of second table */
    core::CallerSlot secondTableInSlot;

    /** data output */
    core::CalleeSlot dataOutSlot;

    /** frameID */
    int frameID;

    /** datahash */
    size_t firstDataHash;
    size_t secondDataHash;

    /** number of rows of the table */
    size_t rows_count;

    /** number of columns of the table */
    size_t column_count;

    /** vector storing the meta information of each column */
    std::vector<TableDataCall::ColumnInfo> column_info;

    /** vector storing the data values of the table */
    std::vector<float> data;
}; /* end class TableJoin */

} // namespace megamol::datatools::table
