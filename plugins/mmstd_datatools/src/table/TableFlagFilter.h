/*
 * TableFlagFilter.h
 *
 * Copyright (C) 2020 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLEFLAGFILTER_H_INCLUDED
#define MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLEFLAGFILTER_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/FlagCall_GL.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_datatools/table/TableDataCall.h"

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace table {

/*
 * Module to filter rows from a table based on a flag storage.
 */
class TableFlagFilter : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName() { return "TableFlagFilter"; }

    /** Return module class description */
    static const char* Description() { return "Filters rows from a table based on a flag storage."; }

    /** Module is always available */
    static bool IsAvailable() { return true; }

    /** Ctor */
    TableFlagFilter();

    /** Dtor */
    ~TableFlagFilter() override;

protected:
    bool create() override;

    void release() override;

    bool getData(core::Call &call);

    bool getHash(core::Call &call);

    bool handleCall(core::Call &call);

private:
    enum FilterMode {
        FILTERED = 0,
        SELECTED = 1
    };

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
    std::vector<TableDataCall::ColumnInfo> colInfos;
    std::vector<float> data;
};

} /* end namespace table */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLEFLAGFILTER_H_INCLUDED */
