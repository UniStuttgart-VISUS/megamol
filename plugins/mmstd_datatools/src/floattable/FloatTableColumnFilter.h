/*
 * FloatTableColumnFilter.h
 *
 * Copyright (C) 2016-2016 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLECOLUMNFILTER_H_INCLUDED
#define MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLECOLUMNFILTER_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

#include "mmcore/param/ParamSlot.h"

#include "mmstd_datatools/floattable/CallFloatTableData.h"

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace floattable {

/*
 * Module to filter columns from a float table.
 */
class FloatTableColumnFilter : public core::Module {
public:
    static std::string ModuleName;

    /** Return module class name */
    static const char *ClassName(void) {
        return ModuleName.c_str();
    }

    /** Return module class description */
    static const char *Description(void) {
        return "Filters columns from a float table";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    FloatTableColumnFilter(void);

    /** Dtor */
    virtual ~FloatTableColumnFilter(void);
protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);
private:
    /** Data callback */
    bool processData(core::Call &c);

    bool getExtent(core::Call &c);

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
    std::vector<CallFloatTableData::ColumnInfo> columnInfos;

    /** Vector stroing the actual float data */
    std::vector<float> data;
}; /* end class FloatTableColumnFilter */

} /* end namespace floattable */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLECOLUMNFILTER_H_INCLUDED */