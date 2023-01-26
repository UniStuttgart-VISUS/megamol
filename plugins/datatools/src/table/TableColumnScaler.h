/*
 * TableColumnScaler.h
 *
 * Copyright (C) 2016-2016 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLECOLUMNSCALER_H_INCLUDED
#define MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLECOLUMNSCALER_H_INCLUDED

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "datatools/table/TableDataCall.h"

namespace megamol {
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
    static inline const char* ClassName() {
        return ModuleName.c_str();
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Scales specified table columns";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /** ctor */
    TableColumnScaler();

    /** dtor */
    ~TableColumnScaler() override;

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
    bool processData(core::Call& c);

    bool getExtent(core::Call& c);

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
} /* end namespace megamol */

#endif
