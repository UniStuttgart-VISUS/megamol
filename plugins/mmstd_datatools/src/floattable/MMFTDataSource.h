/*
 * MMFTDataSource.h
 *
 * Copyright (C) 2016 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_MMFTDATASOURCE_H_INCLUDED
#define MEGAMOL_DATATOOLS_MMFTDATASOURCE_H_INCLUDED
#pragma once

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmstd_datatools/floattable/CallFloatTableData.h"
#include <vector>

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace floattable {

    class MMFTDataSource : public core::Module {
    public:

        static const char *ClassName(void) { return "MMFTDataSource"; }
        static const char *Description(void) { return "Binary float table data source"; }
        static bool IsAvailable(void) { return true; }

        MMFTDataSource(void);
        virtual ~MMFTDataSource(void);

    protected:

        virtual bool create(void);
        virtual void release(void);

    private:

        inline void assertData(void);
        bool getDataCallback(core::Call& caller);
        bool getHashCallback(core::Call& caller);

        core::param::ParamSlot filenameSlot;

        core::CalleeSlot getDataSlot;

        SIZE_T dataHash;

        std::vector<CallFloatTableData::ColumnInfo> columns;
        std::vector<float> values;

    };

} /* end namespace floattable */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_MMFTDATASOURCE_H_INCLUDED */
