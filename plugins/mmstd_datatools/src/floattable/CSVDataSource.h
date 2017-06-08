/*
 * CSVDataSource.h
 *
 * Copyright (C) 2015-2015 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_CSVDATASOURCE_H_INCLUDED
#define MEGAMOL_DATATOOLS_CSVDATASOURCE_H_INCLUDED
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

    class CSVDataSource : public core::Module {
    public:

        static const char *ClassName(void) { return "CSVDataSource"; }
        static const char *Description(void) { return "Comma-separated data source"; }
        static bool IsAvailable(void) { return true; }

        CSVDataSource(void);
        virtual ~CSVDataSource(void);

    protected:

        virtual bool create(void);
        virtual void release(void);

    private:

        inline void assertData(void);
        bool getDataCallback(core::Call& caller);
        bool getHashCallback(core::Call& caller);

        bool clearData(core::param::ParamSlot& caller);
        void shuffleData();

        core::param::ParamSlot filenameSlot;
        core::param::ParamSlot readNameLineSlot;
        core::param::ParamSlot readTypeLineSlot;
        core::param::ParamSlot clearSlot;
        core::param::ParamSlot colSepSlot;
        core::param::ParamSlot decTypeSlot;
        core::param::ParamSlot shuffleSlot;

        core::CalleeSlot getDataSlot;

        SIZE_T dataHash;

        std::vector<CallFloatTableData::ColumnInfo> columns;
        std::vector<float> values;

    };

} /* end namespace floattable */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_LSP_CSVDATASOURCE_H_INCLUDED */
