/*
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "TableSequenceCache.h"

using namespace megamol::datatools::table;
using namespace megamol;

TableSequenceCache::TableSequenceCache()
        : core::Module()
        , outDataSlot("outDataSlot", "access to table data frames cached by module")
        , lastInDataCall(nullptr)
        , inDataSlot("inDataSlot", "slot that gathers table data frames from data source or file sequence") {

    outDataSlot.SetCallback(TableDataCall::ClassName(), "GetData", &TableSequenceCache::getDataCallback);
    outDataSlot.SetCallback(TableDataCall::ClassName(), "GetHash", &TableSequenceCache::getHashCallback);
    MakeSlotAvailable(&outDataSlot);

    inDataSlot.SetCompatibleCall<table::TableDataCallDescription>();
    MakeSlotAvailable(&inDataSlot);
}

TableSequenceCache::~TableSequenceCache() {
    Release();
}

bool TableSequenceCache::create() {
    // nothing to do
    return true;
}

void TableSequenceCache::release() {
    data_cache.clear();
}

/*
 * Pull data from data source call if neccessary
 */
void TableSequenceCache::assertData() {
    TableDataCall* src = this->inDataSlot.CallAs<table::TableDataCall>();

    if (src == nullptr)
        return;

    if (src == lastInDataCall)
        return;

    // get framecount
    src->SetFrameID(0);
    if (!(*src)(1)) {
        return; // unable to get hash
    }

    auto frame_count = src->GetFrameCount();

    data_cache.resize(frame_count);
    for (size_t i = 0; i < frame_count; i++) {
        auto& frame = data_cache[i];

        // get hash/extent
        src->SetFrameID(i);
        if (!(*src)(1)) {
            return; // unable to get hash
        }

        // get data
        if (!(*src)(0)) {
            return; // unable to get data
        }

        src->GetFrameID(); // unused
        auto hash = src->DataHash();
        auto num_rows = src->GetRowsCount();
        auto num_columns = src->GetColumnsCount();

        frame.column_infos.resize(num_columns);
        auto* column_infos = src->GetColumnsInfos();
        std::copy(column_infos, column_infos + num_columns, frame.column_infos.begin());

        auto num_values = num_rows * num_columns;
        frame.values.resize(num_values);
        auto* values = src->GetData();
        std::copy(values, values + num_values, frame.values.begin());

        frame.hash = i + frame_count * hash * num_values; // or what?

        //frame.call.SetClassName(src->ClassName()); // broken as balls?
        frame.call.Set(num_columns, num_rows, frame.column_infos.data(), frame.values.data());
        frame.call.SetDataHash(frame.hash);
        frame.call.SetFrameCount(frame_count);
        frame.call.SetFrameID(i);
        frame.call.SetUnlocker(nullptr);
    }

    lastInDataCall = src;
}

bool TableSequenceCache::getDataCallback(core::Call& caller) {
    TableDataCall* tfd = dynamic_cast<TableDataCall*>(&caller);

    if (tfd == nullptr)
        return false;

    assertData();

    if (data_cache.empty()) {
        tfd->Set(0, 0, nullptr, nullptr);
        tfd->SetDataHash(0);
        return true;
    }

    auto frame_id = tfd->GetFrameID();

    if (frame_id + 1 > data_cache.size())
        return false;

    auto& frame = data_cache[frame_id];

    tfd->Set(
        frame.call.GetColumnsCount(), frame.call.GetRowsCount(), frame.call.GetColumnsInfos(), frame.call.GetData());
    tfd->SetDataHash(frame.hash);
    tfd->SetFrameCount(data_cache.size());
    tfd->SetUnlocker(nullptr);

    return true;
}

bool TableSequenceCache::getHashCallback(core::Call& caller) {
    // both callbacks do the same thing anyway
    return getDataCallback(caller);
}
