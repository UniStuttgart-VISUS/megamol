/*
 * TableColumnFilter.h
 *
 * Copyright (C) 2016-2016 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLECOLUMNFILTER_H_INCLUDED
#define MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLECOLUMNFILTER_H_INCLUDED

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "datatools/table/TableDataCall.h"

namespace megamol::datatools::table {

/*
 * Module to filter columns from a table.
 */
class TableColumnFilter : public core::Module {
public:
    static std::string ModuleName;

    /** Return module class name */
    static const char* ClassName() {
        return ModuleName.c_str();
    }

    /** Return module class description */
    static const char* Description() {
        return "Filters columns from a table";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    TableColumnFilter();

    /** Dtor */
    ~TableColumnFilter() override;

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

    /** Parameter slot for column selection */
    core::param::ParamSlot selectionStringSlot;

    /** ID of the current frame */
    int frameID;

    /** Hash of the current data */
    size_t datahash;

    /** Vector storing information about columns */
    std::vector<TableDataCall::ColumnInfo> columnInfos;

    /** Vector stroing the actual float data */
    std::vector<float> data;
}; /* end class TableColumnFilter */

} // namespace megamol::datatools::table

#endif /* end ifndef MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLECOLUMNFILTER_H_INCLUDED */
