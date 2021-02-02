/*
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOL_DATATOOLS_MMFTDATASOURCE_H_INCLUDED
#define MEGAMOL_DATATOOLS_MMFTDATASOURCE_H_INCLUDED

#include <vector>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_datatools/table/TableDataCall.h"

namespace megamol::stdplugin::datatools::table {

class MMFTDataSource : public core::Module {
public:
    static const char* ClassName() {
        return "MMFTDataSource";
    }
    static const char* Description() {
        return "Binary float table data source";
    }
    static bool IsAvailable() {
        return true;
    }

    MMFTDataSource();
    ~MMFTDataSource() override;

protected:
    bool create() override;
    void release() override;

    bool reloadCallback(core::param::ParamSlot& caller);

private:
    inline void assertData();
    bool getDataCallback(core::Call& caller);
    bool getHashCallback(core::Call& caller);

    core::CalleeSlot getDataSlot_;

    core::param::ParamSlot filenameSlot_;
    core::param::ParamSlot reloadSlot_;

    std::size_t dataHash_;
    bool reload_;

    std::vector<TableDataCall::ColumnInfo> columns_;
    std::vector<float> values_;
};

} // namespace megamol::stdplugin::datatools::table

#endif // MEGAMOL_DATATOOLS_MMFTDATASOURCE_H_INCLUDED
