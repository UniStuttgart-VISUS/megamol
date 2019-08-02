/*
 * TableColumnScaler.h
 *
 * Copyright (C) 2016-2016 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLECOLUMNSCALER_H_INCLUDED
#define MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLECOLUMNSCALER_H_INCLUDED

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
 * Module to scale selected columns with a given value.
 */
class TableColumnScaler : public core::Module {
public:
    /** Name of this Module. */
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
        return "Scales specified table columns";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable(void) {
        return true;
    }

    /** ctor */
    TableColumnScaler(void);

    /** dtor */
    virtual ~TableColumnScaler(void);
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
    bool processData(core::Call &c);

    bool getExtent(core::Call &c);

    core::CalleeSlot dataOutSlot;

    core::CallerSlot dataInSlot;

    core::param::ParamSlot scalingFactorSlot;

    core::param::ParamSlot columnSelectorSlot;

    int frameID;

    size_t datahash;

    std::vector<TableDataCall::ColumnInfo> columnInfos;

    std::vector<float> data;
}; /* end class TableColumnScaler */

} /* end namespace table */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif