/*
 * CSVDataSource.h
 *
 * Copyright (C) 2015-2015 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "datatools/table/TableDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include <vector>

namespace megamol::datatools::table {

class CSVDataSource : public core::Module {
public:
    static const char* ClassName() {
        return "CSVDataSource";
    }
    static const char* Description() {
        return "Data source for parsing tabular comma-separated values";
    }
    static bool IsAvailable() {
        return true;
    }

    CSVDataSource();
    ~CSVDataSource() override;

protected:
    bool create() override;
    void release() override;

private:
    inline void assertData();
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

} // namespace megamol::datatools::table
