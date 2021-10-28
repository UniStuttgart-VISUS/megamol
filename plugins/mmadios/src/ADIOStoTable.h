/*
 * ADIOStoTable.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_datatools/table/TableDataCall.h"


namespace megamol {
namespace adios {

class ADIOStoTable : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ADIOStoTable"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Converts ADIOS data to table data"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    ADIOStoTable(void);

    /** Dtor. */
    virtual ~ADIOStoTable(void);

protected:
    virtual bool create();
    virtual void release();

    core::CallerSlot _getDataSlot;
    core::CalleeSlot _deployTableSlot;

private:
    bool getMetaData(core::Call& call);
    bool getData(core::Call& call);

    bool InterfaceIsDirty();
    size_t _currentFrame;
    std::vector<float> _floatBlob;
    size_t _cols = 0;
    size_t _rows = 0;
    std::vector<stdplugin::datatools::table::TableDataCall::ColumnInfo> _colinfo;
};

} // namespace adios
} // namespace megamol
