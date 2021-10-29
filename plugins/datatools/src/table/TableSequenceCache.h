/*
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOL_DATATOOLS_TABLESEQUENCECACHE_H_INCLUDED
#define MEGAMOL_DATATOOLS_TABLESEQUENCECACHE_H_INCLUDED

#include <vector>

#include "datatools/table/TableDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

namespace megamol::datatools::table {

class TableSequenceCache : public core::Module {
public:
    static const char* ClassName() {
        return "TableSequenceCache";
    }
    static const char* Description() {
        return "Caches and replays table data for all frames passing through this module";
    }
    static bool IsAvailable() {
        return true;
    }

    TableSequenceCache();
    ~TableSequenceCache() override;

protected:
    bool create() override;
    void release() override;

private:
    inline void assertData();
    bool getDataCallback(core::Call& caller);
    bool getHashCallback(core::Call& caller);

    /** The slot for publishing data to the left*/
    core::CalleeSlot outDataSlot;

    /** The slot for requesting data from the source */
    core::CallerSlot inDataSlot;
    void* lastInDataCall = nullptr;

    struct CallData {
        size_t hash;
        TableDataCall call;
        std::vector<TableDataCall::ColumnInfo> column_infos; // referenced by call
        std::vector<float> values;                           // referenced by call
    };

    std::vector<CallData> data_cache;
};

} // namespace megamol::datatools::table

#endif // MEGAMOL_DATATOOLS_TABLESEQUENCECACHE_H_INCLUDED
