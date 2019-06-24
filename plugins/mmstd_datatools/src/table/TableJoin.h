/*
 * TableJoin.h
 *
 * Copyright (C) 2016-2016 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLEJOIN_H_INCLUDED
#define MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLEJOIN_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

#include "mmcore/param/ParamSlot.h"

#include "mmstd_datatools/table/TableDataCall.h"

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace table {

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
    static inline const char *ClassName(void) {
        return ModuleName.c_str();
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char *Description(void) {
        return "Joins two tables (union of columns)";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable(void) {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    TableJoin(void);

    /**
     * Finalises an instance.
     */
    virtual ~TableJoin(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

private:
    /** data callback */
    bool processData(core::Call &c);

    /** extent callback */
    bool getExtent(core::Call &c);

    /** concatenates two tables */
    static void concatenate(float* const out, const size_t rowCount, const size_t columnCount,
        const float* const first, const size_t firstRowCount, const size_t firstColumnCount, const float* const second,
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

} /* end namespace table */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif // end ifndef MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLEJOIN_H_INCLUDED