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
#include "mmstd_datatools/table/TableDataCall.h"
#include <vector>

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace table {

    class CSVDataSource : public core::Module {
    public:

        static const char *ClassName(void) { return "CSVDataSource"; }
        static const char *Description(void) { return "Data source for parsing tabular comma-separated values"; }
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
		core::param::ParamSlot skipPrefaceSlot;
		core::param::ParamSlot headerNamesSlot;
        core::param::ParamSlot headerTypesSlot;
		core::param::ParamSlot commentPrefixSlot;
        core::param::ParamSlot clearSlot;
        core::param::ParamSlot colSepSlot;
        core::param::ParamSlot decSepSlot;
        core::param::ParamSlot shuffleSlot;

        core::CalleeSlot getDataSlot;

        SIZE_T dataHash;

        std::vector<TableDataCall::ColumnInfo> columns;
        std::vector<float> values;

    };

} /* end namespace table */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_LSP_CSVDATASOURCE_H_INCLUDED */
