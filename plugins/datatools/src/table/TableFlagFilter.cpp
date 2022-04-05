/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#include "TableFlagFilter.h"
#include "mmcore/flags/FlagCalls.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::datatools;
using namespace megamol::datatools::table;
using namespace megamol;

TableFlagFilter::TableFlagFilter()
        : core::Module()
        , tableInSlot("getDataIn", "Float table input")
        , flagStorageInSlot("readFlagStorage", "Flag storage read input")
        , tableOutSlot("getDataOut", "Float table output")
        , filterModeParam("filterMode", "filter mode")
        , tableInFrameCount(0)
        , tableInDataHash(0)
        , tableInColCount(0)
        , dataHash(0)
        , rowCount(0) {

    this->tableInSlot.SetCompatibleCall<datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->tableInSlot);

    this->flagStorageInSlot.SetCompatibleCall<core::FlagCallRead_CPUDescription>();
    this->MakeSlotAvailable(&this->flagStorageInSlot);

    this->tableOutSlot.SetCallback(datatools::table::TableDataCall::ClassName(),
        datatools::table::TableDataCall::FunctionName(0), &TableFlagFilter::getData);
    this->tableOutSlot.SetCallback(datatools::table::TableDataCall::ClassName(),
        datatools::table::TableDataCall::FunctionName(1), &TableFlagFilter::getHash);
    this->MakeSlotAvailable(&this->tableOutSlot);

    auto* fmp = new core::param::EnumParam(FilterMode::FILTERED);
    fmp->SetTypePair(FilterMode::FILTERED, "Filtered");
    fmp->SetTypePair(FilterMode::SELECTED, "Selected");
    this->filterModeParam << fmp;
    this->MakeSlotAvailable(&this->filterModeParam);
}

TableFlagFilter::~TableFlagFilter() {
    this->Release();
}

bool TableFlagFilter::create() {
    return true;
}

void TableFlagFilter::release() {}

bool TableFlagFilter::getData(core::Call& call) {
    if (!this->handleCall(call)) {
        return false;
    }

    auto* tableOutCall = dynamic_cast<datatools::table::TableDataCall*>(&call);
    tableOutCall->SetFrameCount(this->tableInFrameCount);
    tableOutCall->SetDataHash(this->dataHash);
    tableOutCall->Set(this->tableInColCount, this->rowCount, this->colInfos.data(), this->data.data());

    return true;
}

bool TableFlagFilter::getHash(core::Call& call) {
    if (!this->handleCall(call)) {
        return false;
    }

    auto* tableOutCall = dynamic_cast<datatools::table::TableDataCall*>(&call);
    tableOutCall->SetFrameCount(this->tableInFrameCount);
    tableOutCall->SetDataHash(this->dataHash);

    return true;
}

bool TableFlagFilter::handleCall(core::Call& call) {
    auto* tableOutCall = dynamic_cast<datatools::table::TableDataCall*>(&call);
    auto* tableInCall = this->tableInSlot.CallAs<datatools::table::TableDataCall>();
    auto* flagsInCall = this->flagStorageInSlot.CallAs<core::FlagCallRead_CPU>();

    if (tableOutCall == nullptr) {
        return false;
    }

    if (tableInCall == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "TableFlagFilter requires a table!");
        return false;
    }

    if (flagsInCall == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "TableFlagFilter requires a flag storage!");
        return false;
    }

    tableInCall->SetFrameID(tableOutCall->GetFrameID());
    (*tableInCall)(1);
    (*tableInCall)(0);
    (*flagsInCall)(core::FlagCallRead_CPU::CallGetData);

    if (this->tableInFrameCount != tableInCall->GetFrameCount() || this->tableInDataHash != tableInCall->DataHash() ||
        flagsInCall->hasUpdate()) {
        // megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO, "TableFlagFilter: Filter table.");

        this->dataHash++;

        this->tableInFrameCount = tableInCall->GetFrameCount();
        this->tableInDataHash = tableInCall->DataHash();
        this->tableInColCount = tableInCall->GetColumnsCount();
        size_t tableInRowCount = tableInCall->GetRowsCount();

        // download flags
        flagsInCall->getData()->validateFlagCount(tableInRowCount);
        const auto& flagsData = *flagsInCall->getData()->flags;

        // copy column infos
        this->colInfos.resize(this->tableInColCount);
        for (size_t i = 0; i < this->tableInColCount; ++i) {
            this->colInfos[i] = tableInCall->GetColumnsInfos()[i];
            this->colInfos[i].SetMinimumValue(std::numeric_limits<float>::max());
            this->colInfos[i].SetMaximumValue(std::numeric_limits<float>::lowest());
        }

        core::FlagStorageTypes::flag_item_type testMask = core::FlagStorageTypes::to_integral(
            core::FlagStorageTypes::flag_bits::ENABLED | core::FlagStorageTypes::flag_bits::FILTERED);
        core::FlagStorageTypes::flag_item_type passMask =
            core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::ENABLED);

        if (static_cast<FilterMode>(this->filterModeParam.Param<core::param::EnumParam>()->Value()) ==
            FilterMode::SELECTED) {
            testMask |= core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::SELECTED);
            passMask |= core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::SELECTED);
        }

        // Resize data to size of input table. With this we only need to allocate memory once.
        this->data.resize(this->tableInColCount * tableInRowCount);
        this->rowCount = 0;

        const float* tableInData = tableInCall->GetData();
        for (size_t r = 0; r < tableInRowCount; ++r) {
            if ((flagsData[r] & testMask) == passMask) {
                for (size_t c = 0; c < this->tableInColCount; ++c) {
                    float val = tableInData[this->tableInColCount * r + c];
                    this->data[this->tableInColCount * this->rowCount + c] = val;
                    if (val < this->colInfos[c].MinimumValue()) {
                        this->colInfos[c].SetMinimumValue(val);
                    }
                    if (val > this->colInfos[c].MaximumValue()) {
                        this->colInfos[c].SetMaximumValue(val);
                    }
                }
                this->rowCount++;
            }
        }

        // delete memory of filtered rows
        this->data.resize(this->tableInColCount * this->rowCount);

        // nicer output
        if (this->rowCount == 0) {
            for (size_t i = 0; i < this->tableInColCount; ++i) {
                this->colInfos[i].SetMinimumValue(0.0);
                this->colInfos[i].SetMaximumValue(0.0);
            }
        }
    }

    return true;
}
