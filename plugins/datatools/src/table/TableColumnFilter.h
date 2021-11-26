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

namespace megamol {
namespace datatools {
namespace table {

/*
 * Module to filter columns from a table.
 */
class TableColumnFilter : public core::Module {
public:
    static std::string ModuleName;

    /** Return module class name */
    static const char* ClassName(void) {
        return ModuleName.c_str();
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Filters columns from a table";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    TableColumnFilter(void);

    /** Dtor */
    virtual ~TableColumnFilter(void);

protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);

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

} /* end namespace table */
} /* end namespace datatools */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLECOLUMNFILTER_H_INCLUDED */
